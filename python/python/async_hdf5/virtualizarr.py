"""
VirtualiZarr integration for async-hdf5.

Provides ``open_virtual_hdf5``, an async function that uses async-hdf5 for
HDF5 metadata extraction and returns an xarray Dataset backed by
VirtualiZarr's ManifestStore.  This lets you open remote HDF5 files with
targeted byte-range reads (metadata only) and then lazily load array data
through zarr — no libhdf5 or h5netcdf required.

VirtualiZarr is an **optional** dependency.  Importing this module without it
installed will raise ``ImportError``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

try:
    import numpy as np
    import xarray as xr
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.manifests import (
        ChunkManifest,
        ManifestArray,
        ManifestGroup,
        ManifestStore,
    )
    from virtualizarr.manifests.utils import create_v3_array_metadata
except ImportError as e:
    raise ImportError(
        "virtualizarr and its dependencies are required for async_hdf5.virtualizarr. "
        "Install with: pip install virtualizarr"
    ) from e

from async_hdf5 import HDF5File

__all__ = ["open_virtual_hdf5"]


async def open_virtual_hdf5(
    path: str,
    *,
    store: Any,
    group: str | None = None,
    url: str | None = None,
    registry: ObjectStoreRegistry | None = None,
    drop_variables: Iterable[str] | None = None,
    block_size: int = 8 * 1024 * 1024,
) -> xr.Dataset:
    """Open an HDF5 file as a virtual xarray Dataset.

    Uses async-hdf5 (a Rust HDF5 binary parser) for metadata extraction and
    VirtualiZarr's ManifestStore for lazy chunk reads via obstore.

    Parameters
    ----------
    path
        Path to the HDF5 file within the store (e.g. the filename portion of
        an S3 URL).
    store
        An obstore ``ObjectStore`` instance or obspec-compatible backend.
    group
        HDF5 group to open (e.g. ``"science/LSAR/GCOV/grids/frequencyA"``).
        If *None*, the root group is used.
    url
        Full URL of the HDF5 file (e.g. ``"s3://bucket/path/file.h5"``).
        Stored in chunk manifests so ManifestStore can resolve the correct
        store via the registry.  If *None*, *path* is used as-is.
    registry
        An :class:`ObjectStoreRegistry` mapping URL prefixes to store
        instances.  If *None*, one is created automatically and the provided
        *store* is registered under the scheme/netloc of *url*.
    drop_variables
        Variable names to exclude from the virtual dataset.
    block_size
        Block cache size in bytes.  Each unique region of the file accessed
        during metadata parsing triggers a fetch of the aligned block
        containing that region.  Default 8 MiB.

    Returns
    -------
    xr.Dataset
        An xarray Dataset backed by a ManifestStore (zarr v3).  Variables are
        lazily loaded — indexing or calling ``.load()`` triggers byte-range
        reads from the object store.
    """
    f = await HDF5File.open(path, store=store, block_size=block_size)
    root = await f.root_group()

    target = (await root.navigate(group)) if group else root

    file_url = url or path
    manifest_group = await _build_manifest_group(file_url, target, drop_variables)

    if registry is None:
        registry = ObjectStoreRegistry()
    _ensure_store_registered(registry, file_url, store)

    manifest_store = ManifestStore(group=manifest_group, registry=registry)
    return xr.open_dataset(manifest_store, engine="zarr", consolidated=False, zarr_format=3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _build_manifest_group(
    file_url: str,
    group: Any,
    drop_variables: Iterable[str] | None,
) -> ManifestGroup:
    """Recursively build a ManifestGroup from an async-hdf5 HDF5Group."""
    drop = set(drop_variables or ())

    # First pass: collect datasets and their shapes for dimension assignment.
    datasets: list[tuple[str, Any, Any]] = []
    for name in await group.dataset_names():
        if name in drop:
            continue
        ds = await group.dataset(name)
        chunk_idx = await ds.chunk_index()
        datasets.append((name, ds, chunk_idx))

    # Assign phony dimension names grouped by size (like h5netcdf phony_dims="sort").
    dim_names_map = _assign_phony_dims([(name, tuple(int(s) for s in ds.shape)) for name, ds, _ in datasets])

    arrays: dict[str, ManifestArray] = {}
    for name, ds, chunk_idx in datasets:
        arrays[name] = _build_manifest_array(file_url, ds, chunk_idx, dimension_names=dim_names_map[name])

    groups: dict[str, ManifestGroup] = {}
    for name in await group.group_names():
        if name in drop:
            continue
        child = await group.group(name)
        groups[name] = await _build_manifest_group(file_url, child, drop)

    attrs = await group.attributes()

    return ManifestGroup(arrays=arrays, groups=groups, attributes=attrs)


def _assign_phony_dims(
    variables: list[tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[str, ...]]:
    """Assign phony_dim names to variables, grouping dimensions by size.

    Dimensions with the same size share the same phony_dim name (unless a
    variable has multiple axes of the same size, in which case additional
    unique names are created).  This mimics h5netcdf's ``phony_dims="sort"``
    behaviour without requiring HDF5 dimension scale resolution.
    """
    dim_counter = 0
    # size -> list of phony_dim names already created for that size
    size_to_dims: dict[int, list[str]] = {}
    result: dict[str, tuple[str, ...]] = {}

    for varname, shape in variables:
        dims: list[str] = []
        for size in shape:
            candidates = size_to_dims.get(size, [])
            chosen = None
            for c in candidates:
                if c not in dims:  # avoid reusing within same variable
                    chosen = c
                    break
            if chosen is None:
                chosen = f"phony_dim_{dim_counter}"
                dim_counter += 1
                size_to_dims.setdefault(size, []).append(chosen)
            dims.append(chosen)
        result[varname] = tuple(dims)

    return result


def _build_manifest_array(
    file_url: str,
    dataset: Any,
    chunk_index: Any,
    dimension_names: tuple[str, ...] | None = None,
) -> ManifestArray:
    """Build a ManifestArray from an async-hdf5 dataset and its chunk index."""
    grid_shape = tuple(chunk_index.grid_shape)

    paths = np.full(grid_shape, file_url, dtype=np.dtypes.StringDType())
    offsets = np.empty(grid_shape, dtype=np.uint64)
    lengths = np.empty(grid_shape, dtype=np.uint64)

    for chunk in chunk_index:
        idx = tuple(chunk.indices)
        offsets[idx] = chunk.byte_offset
        lengths[idx] = chunk.byte_length

    manifest = ChunkManifest.from_arrays(
        paths=paths, offsets=offsets, lengths=lengths
    )

    codecs = _hdf5_filters_to_zarr_codecs(dataset.filters, dataset.element_size)

    shape = tuple(int(s) for s in dataset.shape)
    if dimension_names is None:
        dimension_names = tuple(f"phony_dim_{i}" for i in range(len(shape)))

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=np.dtype(dataset.numpy_dtype),
        chunk_shape=tuple(int(s) for s in (dataset.chunk_shape or dataset.shape)),
        codecs=codecs,
        dimension_names=dimension_names,
    )

    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _hdf5_filters_to_zarr_codecs(
    filters: list[dict[str, Any]],
    element_size: int,
) -> list[dict[str, Any]]:
    """Map async-hdf5 filter dicts to zarr v3 codec configs.

    Handles the most common HDF5 filters: shuffle, deflate/zlib, fletcher32,
    and zstd.
    """
    codecs: list[dict[str, Any]] = []
    for f in filters:
        fid = f["id"]
        cd = f.get("client_data", [])
        if fid == 2:  # SHUFFLE
            codecs.append(
                {
                    "name": "numcodecs.shuffle",
                    "configuration": {"elementsize": element_size},
                }
            )
        elif fid == 1:  # DEFLATE
            codecs.append(
                {
                    "name": "numcodecs.zlib",
                    "configuration": {"level": cd[0] if cd else 6},
                }
            )
        elif fid == 3:  # FLETCHER32
            pass  # checksum only, not needed for decompression
        elif fid == 32015:  # ZSTD
            codecs.append(
                {
                    "name": "numcodecs.zstd",
                    "configuration": {"level": cd[0] if cd else 3},
                }
            )
    return codecs


def _ensure_store_registered(
    registry: ObjectStoreRegistry,
    file_url: str,
    store: Any,
) -> None:
    """Register *store* in *registry* for the scheme://netloc prefix of *file_url*."""
    parsed = urlparse(file_url)
    if parsed.scheme and parsed.netloc:
        prefix = f"{parsed.scheme}://{parsed.netloc}"
        try:
            registry.resolve(file_url)
        except Exception:
            registry.register(prefix, store)
