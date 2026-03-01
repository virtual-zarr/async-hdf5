"""Shared utilities for async-hdf5's Zarr and VirtualiZarr integrations."""

from __future__ import annotations

from typing import Any


def hdf5_filters_to_zarr_codecs(
    filters: list[dict[str, Any]],
    element_size: int,
) -> list[dict[str, Any]]:
    """Map async-hdf5 filter dicts to zarr v3 codec configs.

    Args:
        filters: HDF5 filter pipeline entries, each a dict with keys
            ``"id"`` (int), ``"name"`` (str), and ``"client_data"``
            (list of int).
        element_size: Size of a single array element in bytes.

    Returns:
        A list of zarr v3 codec configuration dicts.
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
            pass  # checksum only
        elif fid == 32015:  # ZSTD
            codecs.append(
                {
                    "name": "numcodecs.zstd",
                    "configuration": {"level": cd[0] if cd else 3},
                }
            )
        else:
            name = f.get("name", f"unknown({fid})")
            raise ValueError(
                f"Unsupported HDF5 filter: {name} (id={fid}). "
                "Use drop_variables to skip this dataset."
            )
    return codecs


def assign_phony_dims(
    variables: list[tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[str, ...]]:
    """Assign phony_dim names grouped by size.

    Dimensions with the same size share the same phony_dim name (unless a
    variable has multiple axes of the same size, in which case additional
    unique names are created).  This mimics h5netcdf's ``phony_dims="sort"``
    behaviour without requiring HDF5 dimension scale resolution.

    Args:
        variables: A list of ``(name, shape)`` tuples.

    Returns:
        A dict mapping variable names to tuples of dimension names.
    """
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
