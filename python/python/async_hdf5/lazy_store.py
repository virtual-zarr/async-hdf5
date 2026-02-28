"""
Lazy HDF5-to-Zarr store for async-hdf5.

Provides ``LazyHDF5Store``, a read-only Zarr v3 store that wraps async-hdf5
objects and defers chunk index parsing until chunks are actually requested.
Dataset metadata (shape, dtype, filters, chunk_shape) is parsed eagerly at
open time since it's cheap, but the expensive B-tree / FixedArray traversal
for chunk locations only happens on first access to each variable's data.

This module also provides ``open_lazy_hdf5``, an async convenience function
that creates a ``LazyHDF5Store`` and returns an xarray Dataset backed by it.

Both ``virtualizarr`` and ``xarray`` are **optional** dependencies.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import numpy as np
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.buffer.core import default_buffer_prototype as _default_prototype
from zarr.core.common import BytesLike
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata

from async_hdf5 import HDF5Dataset, HDF5File, HDF5Group

if TYPE_CHECKING:
    from async_hdf5 import ChunkIndex
    from obspec_utils.registry import ObjectStoreRegistry

__all__ = ["LazyHDF5Store", "open_lazy_hdf5"]


# ---------------------------------------------------------------------------
# Chunk key parsing (adapted from VirtualiZarr manifests/utils.py)
# ---------------------------------------------------------------------------

_CHUNK_KEY_RE = re.compile(r"c\.(.+)$")


def _parse_chunk_key(suffix: str, separator: str = ".") -> tuple[int, ...]:
    """Parse a chunk key suffix like ``'c.3.5'`` into index tuple ``(3, 5)``."""
    if suffix == "c":
        return ()  # scalar array
    prefix = f"c{separator}"
    if not suffix.startswith(prefix):
        raise ValueError(f"Unexpected chunk key format: {suffix!r}")
    indices_str = suffix[len(prefix) :]
    return tuple(int(x) for x in indices_str.split(separator))


# ---------------------------------------------------------------------------
# Fill value decoding
# ---------------------------------------------------------------------------


def _decode_fill_value(
    raw: list[int] | None, numpy_dtype: str
) -> int | float | None:
    """Convert raw fill value bytes from the HDF5 parser to a numpy scalar.

    Parameters
    ----------
    raw
        Raw fill value as a list of byte values (from Rust ``Vec<u8>`` via PyO3),
        or *None* if no fill value is defined.
    numpy_dtype
        Numpy dtype string for the dataset (e.g. ``"<f4"``).

    Returns
    -------
    int | float | None
        A Python scalar suitable for Zarr's ``ArrayV3Metadata(fill_value=...)``.

    Raises
    ------
    ValueError
        If *raw* cannot be interpreted as *numpy_dtype* (e.g. byte count mismatch).
    """
    if raw is None:
        return None
    raw_bytes = bytes(raw)
    dt = np.dtype(numpy_dtype)
    if len(raw_bytes) != dt.itemsize:
        raise ValueError(
            f"Fill value has {len(raw_bytes)} byte(s) but dtype {numpy_dtype!r} "
            f"expects {dt.itemsize}. Raw bytes: {raw_bytes!r}"
        )
    return np.frombuffer(raw_bytes, dtype=dt).flat[0].item()


# ---------------------------------------------------------------------------
# Attribute sanitization
# ---------------------------------------------------------------------------


def _sanitize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert HDF5 attributes to JSON-serializable Python types.

    Zarr's ``GroupMetadata`` serializes attributes to JSON.  HDF5 attributes
    can include ``bytes`` (from ``AttributeValue::Raw``), numpy arrays, and
    numpy scalars that aren't directly JSON-compatible.

    Binary attributes that can't be decoded as UTF-8 are dropped, and a
    :func:`warnings.warn` lists every dropped key so the user can investigate.
    """
    clean: dict[str, Any] = {}
    non_serializable: list[tuple[str, str, str]] = []

    for k, v in attrs.items():
        if isinstance(v, bytes):
            try:
                clean[k] = v.decode("utf-8")
            except UnicodeDecodeError:
                non_serializable.append((k, type(v).__name__, repr(v[:32])))
                continue
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            clean[k] = v.item()
        else:
            clean[k] = v

    if non_serializable:
        details = "; ".join(f"{k} ({t}): {r}" for k, t, r in non_serializable)
        warnings.warn(
            f"Dropped {len(non_serializable)} non-serializable attribute(s): "
            f"{details}. These contain raw bytes that cannot be represented "
            f"in Zarr metadata.",
            stacklevel=2,
        )
    return clean


# ---------------------------------------------------------------------------
# Dataset info: lightweight metadata cache (no chunk index)
# ---------------------------------------------------------------------------


class _DatasetInfo:
    """Cached metadata for one HDF5 dataset, without its chunk index."""

    __slots__ = (
        "name",
        "shape",
        "numpy_dtype",
        "element_size",
        "chunk_shape",
        "fill_value",
        "filters",
        "dimension_names",
        "hdf5_dataset",
        "_array_metadata",
        "_zarr_json_bytes",
    )

    def __init__(
        self,
        ds: HDF5Dataset,
        dimension_names: tuple[str, ...],
    ) -> None:
        self.name: str = ds.name
        self.shape: tuple[int, ...] = tuple(int(s) for s in ds.shape)
        self.numpy_dtype: str = ds.numpy_dtype
        self.element_size: int = ds.element_size
        self.chunk_shape: tuple[int, ...] = tuple(
            int(s) for s in (ds.chunk_shape or ds.shape)
        )
        self.fill_value: int | float | None = _decode_fill_value(
            ds.fill_value, ds.numpy_dtype
        )
        self.filters: list[dict[str, Any]] = ds.filters
        self.dimension_names = dimension_names
        self.hdf5_dataset: HDF5Dataset = ds
        self._array_metadata: ArrayV3Metadata | None = None
        self._zarr_json_bytes: bytes | None = None

    @property
    def array_metadata(self) -> ArrayV3Metadata:
        if self._array_metadata is None:
            self._array_metadata = _create_array_metadata(self)
        return self._array_metadata

    @property
    def zarr_json_bytes(self) -> bytes:
        if self._zarr_json_bytes is None:
            buf_dict = self.array_metadata.to_buffer_dict(
                prototype=_default_prototype()
            )
            self._zarr_json_bytes = buf_dict["zarr.json"].to_bytes()
        return self._zarr_json_bytes

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(
            -(-s // c) for s, c in zip(self.shape, self.chunk_shape)
        )


def _create_array_metadata(info: _DatasetInfo) -> ArrayV3Metadata:
    """Build Zarr v3 array metadata from cached dataset info."""
    from virtualizarr.codecs import convert_to_codec_pipeline
    from zarr.dtype import parse_data_type

    codecs_config = _hdf5_filters_to_zarr_codecs(info.filters, info.element_size)
    dt = np.dtype(info.numpy_dtype)
    zdtype = parse_data_type(dt, zarr_format=3)

    return ArrayV3Metadata(
        shape=info.shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": info.chunk_shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar() if info.fill_value is None else info.fill_value,
        codecs=convert_to_codec_pipeline(codecs=codecs_config, dtype=dt),
        attributes={},
        dimension_names=info.dimension_names,
        storage_transformers=None,
    )


# ---------------------------------------------------------------------------
# Filter-to-codec mapping (same as virtualizarr.py)
# ---------------------------------------------------------------------------


def _hdf5_filters_to_zarr_codecs(
    filters: list[dict[str, Any]],
    element_size: int,
) -> list[dict[str, Any]]:
    """Map async-hdf5 filter dicts to zarr v3 codec configs."""
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
            pass  # checksum only
        elif fid == 32015:  # ZSTD
            codecs.append(
                {
                    "name": "numcodecs.zstd",
                    "configuration": {"level": cd[0] if cd else 3},
                }
            )
    return codecs


# ---------------------------------------------------------------------------
# Phony dimension assignment (same as virtualizarr.py)
# ---------------------------------------------------------------------------


def _assign_phony_dims(
    variables: list[tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[str, ...]]:
    """Assign phony_dim names grouped by size, matching h5netcdf phony_dims='sort'."""
    dim_counter = 0
    size_to_dims: dict[int, list[str]] = {}
    result: dict[str, tuple[str, ...]] = {}

    for varname, shape in variables:
        dims: list[str] = []
        for size in shape:
            candidates = size_to_dims.get(size, [])
            chosen = None
            for c in candidates:
                if c not in dims:
                    chosen = c
                    break
            if chosen is None:
                chosen = f"phony_dim_{dim_counter}"
                dim_counter += 1
                size_to_dims.setdefault(size, []).append(chosen)
            dims.append(chosen)
        result[varname] = tuple(dims)

    return result


# ---------------------------------------------------------------------------
# LazyHDF5Store
# ---------------------------------------------------------------------------


class LazyHDF5Store(Store):
    """
    Read-only Zarr v3 store backed by async-hdf5 with lazy chunk index resolution.

    Dataset metadata (shape, dtype, codecs, etc.) is parsed eagerly at
    construction — this is fast because the data is already in async-hdf5's
    BlockCache from the superblock / group parse.  Chunk indices (B-tree /
    FixedArray traversal) are parsed lazily on first chunk access per variable.

    Parameters
    ----------
    dataset_infos
        Mapping from variable name to ``_DatasetInfo`` (pre-parsed metadata).
    group_attrs
        Root group attributes.
    file_url
        Full URL of the HDF5 file for byte-range routing.
    registry
        ObjectStoreRegistry mapping URL prefixes to ObjectStore instances.
    """

    _dataset_infos: dict[str, _DatasetInfo]
    _group_attrs: dict[str, Any]
    _file_url: str
    _registry: ObjectStoreRegistry
    _chunk_indices: dict[str, ChunkIndex]
    _group_metadata: GroupMetadata

    def __init__(
        self,
        *,
        dataset_infos: dict[str, _DatasetInfo],
        group_attrs: dict[str, Any],
        file_url: str,
        registry: ObjectStoreRegistry,
    ) -> None:
        super().__init__(read_only=True)
        self._dataset_infos = dataset_infos
        self._group_attrs = group_attrs
        self._file_url = file_url
        self._registry = registry
        self._chunk_indices = {}
        self._group_metadata = GroupMetadata(attributes=_sanitize_attrs(group_attrs))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LazyHDF5Store) and self._file_url == other._file_url

    # ------------------------------------------------------------------
    # Core read
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        await self._ensure_open()

        # Root group metadata
        if key == "zarr.json":
            return self._get_group_metadata(prototype)

        # Split into variable name and remainder
        if "/" not in key:
            return None
        var_name, suffix = key.split("/", 1)

        # Check for v2 metadata keys → return None (Zarr v3 only)
        if suffix.endswith((".zattrs", ".zgroup", ".zarray", ".zmetadata")):
            return None

        info = self._dataset_infos.get(var_name)
        if info is None:
            return None

        # Array metadata
        if suffix == "zarr.json":
            return prototype.buffer.from_bytes(info.zarr_json_bytes)

        # Chunk data — lazily resolve chunk index
        return await self._get_chunk_data(var_name, info, suffix, prototype, byte_range)

    async def _get_chunk_data(
        self,
        var_name: str,
        info: _DatasetInfo,
        suffix: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None,
    ) -> Buffer | None:
        """Fetch chunk data, lazily parsing the chunk index on first access."""
        # Lazily parse chunk index
        if var_name not in self._chunk_indices:
            self._chunk_indices[var_name] = await info.hdf5_dataset.chunk_index()

        chunk_index = self._chunk_indices[var_name]
        separator: str = getattr(
            info.array_metadata.chunk_key_encoding, "separator", "/"
        )
        indices = _parse_chunk_key(suffix, separator)
        location = chunk_index.get(list(indices))
        if location is None:
            return None

        offset = location.byte_offset
        length = location.byte_length

        # Route to the correct object store
        store, _path_after_prefix = self._registry.resolve(self._file_url)
        if not store:
            raise ValueError(
                f"Could not find a store for {self._file_url} in the registry"
            )

        path_in_store = urlparse(self._file_url).path
        if hasattr(store, "prefix") and store.prefix:
            prefix = str(store.prefix).lstrip("/")
        elif hasattr(store, "url"):
            prefix = urlparse(store.url).path.lstrip("/")
        else:
            prefix = ""
        path_in_store = path_in_store.lstrip("/").removeprefix(prefix).lstrip("/")

        # Transform byte range to account for chunk location in file
        chunk_end = offset + length
        resolved_range = _transform_byte_range(
            byte_range, chunk_start=offset, chunk_end_exclusive=chunk_end
        )

        data = await store.get_range_async(
            path_in_store,
            start=resolved_range.start,
            end=resolved_range.end,
        )
        return prototype.buffer.from_bytes(data)

    def _get_group_metadata(self, prototype: BufferPrototype) -> Buffer:
        buf_dict = self._group_metadata.to_buffer_dict(prototype=prototype)
        return buf_dict["zarr.json"]

    # ------------------------------------------------------------------
    # Batch read
    # ------------------------------------------------------------------

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        import asyncio

        return list(
            await asyncio.gather(
                *(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)
            )
        )

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    async def exists(self, key: str) -> bool:
        if key == "zarr.json":
            return True
        if "/" not in key:
            return False
        var_name, suffix = key.split("/", 1)
        if var_name not in self._dataset_infos:
            return False
        if suffix == "zarr.json":
            return True
        # For chunk keys, we'd need to check the chunk index — return True
        # optimistically since zarr-python typically only requests valid keys.
        return True

    # ------------------------------------------------------------------
    # Write stubs (read-only store)
    # ------------------------------------------------------------------

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> Literal[False]:
        return False

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("LazyHDF5Store is read-only")

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("LazyHDF5Store is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("LazyHDF5Store is read-only")

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError("LazyHDF5Store is read-only")

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncGenerator[str, None]:
        yield "zarr.json"
        for name, info in self._dataset_infos.items():
            separator: str = getattr(
                info.array_metadata.chunk_key_encoding, "separator", "/"
            )
            yield f"{name}/zarr.json"
            if info.shape == ():
                yield f"{name}/c"
            else:
                for key in _iter_chunk_keys(info.grid_shape, separator):
                    yield f"{name}/c{separator}{key}"

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        prefix = prefix.rstrip("/")
        if prefix == "":
            # Root: group metadata + variable names
            yield "zarr.json"
            for name in self._dataset_infos:
                yield name
        else:
            # Variable directory: metadata + chunk keys
            info = self._dataset_infos.get(prefix)
            if info is not None:
                separator = getattr(
                    info.array_metadata.chunk_key_encoding, "separator", "/"
                )
                yield "zarr.json"
                if info.shape == ():
                    yield "c"
                else:
                    for key in _iter_chunk_keys(info.grid_shape, separator):
                        yield f"c{separator}{key}"

    @property
    def supports_consolidated_metadata(self) -> bool:
        return False


def _iter_chunk_keys(grid_shape: tuple[int, ...], separator: str = "/") -> Iterable[str]:
    """Yield chunk keys like '0/0', '0/1', ... for a given grid shape."""
    if len(grid_shape) == 0:
        return
    import itertools

    ranges = [range(s) for s in grid_shape]
    for idx in itertools.product(*ranges):
        yield separator.join(str(i) for i in idx)


# ---------------------------------------------------------------------------
# Byte range transformation (same as VirtualiZarr ManifestStore)
# ---------------------------------------------------------------------------


def _transform_byte_range(
    byte_range: ByteRequest | None,
    *,
    chunk_start: int,
    chunk_end_exclusive: int,
) -> RangeByteRequest:
    """Convert a byte_range relative to a chunk into absolute file offsets."""
    if byte_range is None:
        return RangeByteRequest(chunk_start, chunk_end_exclusive)
    elif isinstance(byte_range, RangeByteRequest):
        return RangeByteRequest(
            chunk_start + byte_range.start, chunk_start + byte_range.end
        )
    elif isinstance(byte_range, OffsetByteRequest):
        return RangeByteRequest(chunk_start + byte_range.offset, chunk_end_exclusive)
    elif isinstance(byte_range, SuffixByteRequest):
        return RangeByteRequest(
            chunk_end_exclusive - byte_range.suffix, chunk_end_exclusive
        )
    else:
        raise ValueError(f"Unexpected byte_range type: {type(byte_range)}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def open_lazy_hdf5(
    path: str,
    *,
    store: Any,
    group: str | None = None,
    url: str | None = None,
    registry: ObjectStoreRegistry | None = None,
    drop_variables: Iterable[str] | None = None,
    block_size: int = 8 * 1024 * 1024,
) -> Any:
    """Open an HDF5 file as a lazy xarray Dataset.

    Like ``open_virtual_hdf5`` but uses ``LazyHDF5Store`` to defer chunk
    index parsing until data is actually accessed.  This makes the initial
    ``open_dataset`` call significantly faster for files with many variables.

    Parameters
    ----------
    path
        Path within the store (e.g. the object key for S3).
    store
        An obstore ObjectStore or obspec-compatible backend.
    group
        HDF5 group to open.  If *None*, the root group is used.
    url
        Full URL of the HDF5 file.  Stored in chunk references so the store
        can route reads to the correct ObjectStore.  If *None*, *path* is used.
    registry
        ObjectStoreRegistry mapping URL prefixes to store instances.
    drop_variables
        Variable names to exclude.
    block_size
        BlockCache size in bytes (default 8 MiB).

    Returns
    -------
    xr.Dataset
        An xarray Dataset backed by a LazyHDF5Store.  Variables are lazily
        loaded — chunk indices are only parsed when data is actually read.
    """
    import xarray as xr
    from obspec_utils.registry import ObjectStoreRegistry as Registry

    # Phase 1: parse superblock + group structure
    f = await HDF5File.open(path, store=store, block_size=block_size)
    root = await f.root_group()
    target = (await root.navigate(group)) if group else root

    file_url = url or path
    drop = set(drop_variables or ())

    # Phase 2: parse all dataset headers (fast — data in BlockCache)
    dataset_names = await target.dataset_names()
    datasets: dict[str, HDF5Dataset] = {}
    for name in dataset_names:
        if name in drop:
            continue
        datasets[name] = await target.dataset(name)

    # Compute phony dimension names from shapes
    dim_names_map = _assign_phony_dims(
        [(name, tuple(int(s) for s in ds.shape)) for name, ds in datasets.items()]
    )

    # Build lightweight metadata cache (no chunk indices!)
    dataset_infos: dict[str, _DatasetInfo] = {}
    skipped: list[tuple[str, str]] = []
    for name, ds in datasets.items():
        try:
            dataset_infos[name] = _DatasetInfo(ds, dimension_names=dim_names_map[name])
        except (ValueError, TypeError) as exc:
            skipped.append((name, str(exc)))
    if skipped:
        details = "; ".join(f"{n}: {e}" for n, e in skipped)
        warnings.warn(
            f"Skipped {len(skipped)} dataset(s) with unsupported types: {details}. "
            f"Use drop_variables to suppress this warning.",
            stacklevel=2,
        )

    # Get group attributes
    group_attrs = await target.attributes()

    # Set up registry
    if registry is None:
        registry = Registry()
    _ensure_store_registered(registry, file_url, store)

    # Create lazy store
    lazy_store = LazyHDF5Store(
        dataset_infos=dataset_infos,
        group_attrs=group_attrs,
        file_url=file_url,
        registry=registry,
    )

    return xr.open_dataset(
        lazy_store, engine="zarr", consolidated=False, zarr_format=3
    )


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
