"""Test configuration for async-hdf5 comparison tests.

Provides:
- h5py_comparison(): ground-truth comparison between async-hdf5 and h5py
- metadata_only_check(): lighter check for shape/dtype without loading data
- File discovery helpers for each data source
- pytest configuration (network/slow markers, async setup)
"""

import asyncio
import pathlib

import h5py
import numpy as np
import pytest
from async_hdf5.store import LocalStore

EXTENSIONS = (".h5", ".hdf5", ".nc", ".he5")


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="Run tests requiring a network connection",
    )
    parser.addoption(
        "--run-slow-tests",
        action="store_true",
        help="Run slow tests (large files)",
    )


def pytest_runtest_setup(item):
    """Skip network/slow tests unless explicitly enabled."""
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip("Set --run-network-tests to run tests requiring network")
    if "slow" in item.keywords and not item.config.getoption("--run-slow-tests"):
        pytest.skip("Set --run-slow-tests to run slow tests")


# ---------------------------------------------------------------------------
# Path resolution and file discovery
# ---------------------------------------------------------------------------


def resolve_folder(folder: str) -> pathlib.Path:
    """Resolve a path relative to the python/ directory."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    return repo_root / folder


def discover_files(subfolder: str) -> list[str]:
    """Recursively find all HDF5/NetCDF4 files under tests/data/{subfolder}/."""
    data_dir = resolve_folder(f"tests/data/{subfolder}")
    if not data_dir.exists():
        return []
    files = []
    for ext in EXTENSIONS:
        files.extend(
            str(f.relative_to(data_dir)) for f in sorted(data_dir.rglob(f"*{ext}"))
        )
    return sorted(set(files))


def hdf5_group_examples():
    return discover_files("hdf5-group")


def gdal_examples():
    return discover_files("gdal")


def h5py_examples():
    return discover_files("h5py")


def external_examples():
    return discover_files("external")


def netcdf_test_data_examples():
    return discover_files("netcdf-test-data")


def generated_examples():
    """Generate small HDF5 fixture files and return their filenames."""
    _generate_fixtures()
    return discover_files("generated")


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

_GENERATED_DIR = resolve_folder("tests/data/generated")


_EXPECTED_FIXTURES = [
    "contiguous_float32.h5",
    "chunked_float64.h5",
    "gzip_compressed.h5",
    "multi_dtype.h5",
    "nested_groups.h5",
    "cf_style.h5",
    "simple_1d.h5",
    "multi_chunk.h5",
    "unsigned_int.h5",
    "superblock_v0.h5",
    "fixed_array_paged.h5",
]


def _generate_fixtures():
    """Create small HDF5 files covering key parsing variations.

    Files are only regenerated when one or more expected fixtures are missing.
    """
    if _GENERATED_DIR.exists() and all(
        (_GENERATED_DIR / f).exists() for f in _EXPECTED_FIXTURES
    ):
        return

    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    import h5py
    import numpy as np

    rng = np.random.default_rng(42)

    # 1. Contiguous float32 — simplest case
    with h5py.File(_GENERATED_DIR / "contiguous_float32.h5", "w") as f:
        f.create_dataset("temperature", data=rng.random((10, 20), dtype=np.float32))
        f["temperature"].attrs["units"] = "K"

    # 2. Chunked float64
    with h5py.File(_GENERATED_DIR / "chunked_float64.h5", "w") as f:
        f.create_dataset(
            "values", data=rng.random((50, 100), dtype=np.float64), chunks=(10, 20)
        )

    # 3. Gzip-compressed chunked 3D
    with h5py.File(_GENERATED_DIR / "gzip_compressed.h5", "w") as f:
        f.create_dataset(
            "data",
            data=rng.random((5, 10, 10), dtype=np.float32),
            chunks=(1, 10, 10),
            compression="gzip",
            compression_opts=4,
        )

    # 4. Multiple datasets with different dtypes
    with h5py.File(_GENERATED_DIR / "multi_dtype.h5", "w") as f:
        f.create_dataset(
            "int16_data", data=np.arange(100, dtype=np.int16).reshape(10, 10)
        )
        f.create_dataset(
            "int32_data", data=np.arange(100, dtype=np.int32).reshape(10, 10)
        )
        f.create_dataset("float32_data", data=rng.random((10, 10), dtype=np.float32))
        f.create_dataset("float64_data", data=rng.random((10, 10), dtype=np.float64))

    # 5. Nested groups with datasets at different levels
    with h5py.File(_GENERATED_DIR / "nested_groups.h5", "w") as f:
        g1 = f.create_group("level1")
        g2 = g1.create_group("level2")
        g1.create_dataset("data_a", data=np.ones((5, 5), dtype=np.float32))
        g2.create_dataset("data_b", data=np.zeros((3, 3), dtype=np.float64))

    # 6. CF-style with coordinate dimension scales and attributes
    with h5py.File(_GENERATED_DIR / "cf_style.h5", "w") as f:
        f.attrs["Conventions"] = "CF-1.8"
        f.attrs["title"] = "Test CF dataset"

        time_data = np.arange(10, dtype=np.float64)
        lat_data = np.linspace(-90, 90, 20, dtype=np.float64)
        lon_data = np.linspace(-180, 180, 30, dtype=np.float64)

        f.create_dataset("time", data=time_data)
        f["time"].attrs["units"] = "days since 2000-01-01"
        f["time"].attrs["calendar"] = "standard"
        f["time"].make_scale("time")

        f.create_dataset("lat", data=lat_data)
        f["lat"].attrs["units"] = "degrees_north"
        f["lat"].make_scale("lat")

        f.create_dataset("lon", data=lon_data)
        f["lon"].attrs["units"] = "degrees_east"
        f["lon"].make_scale("lon")

        temp = rng.random((10, 20, 30), dtype=np.float32)
        ds = f.create_dataset(
            "temperature", data=temp, chunks=(2, 10, 15), compression="gzip"
        )
        ds.attrs["units"] = "K"
        ds.attrs["long_name"] = "Temperature"
        ds.dims[0].attach_scale(f["time"])
        ds.dims[1].attach_scale(f["lat"])
        ds.dims[2].attach_scale(f["lon"])

    # 7. Simple 1D dataset
    with h5py.File(_GENERATED_DIR / "simple_1d.h5", "w") as f:
        f.create_dataset("values", data=np.arange(1000, dtype=np.float32))

    # 8. Multi-chunk grid (many chunks along each axis)
    with h5py.File(_GENERATED_DIR / "multi_chunk.h5", "w") as f:
        f.create_dataset(
            "grid",
            data=rng.random((100, 200), dtype=np.float32),
            chunks=(10, 20),
        )

    # 9. Unsigned integer types
    with h5py.File(_GENERATED_DIR / "unsigned_int.h5", "w") as f:
        f.create_dataset(
            "uint8_data", data=np.arange(256, dtype=np.uint8).reshape(16, 16)
        )
        f.create_dataset(
            "uint16_data",
            data=np.arange(100, dtype=np.uint16).reshape(10, 10),
        )

    # 10. Superblock v0 (libver='earliest')
    with h5py.File(_GENERATED_DIR / "superblock_v0.h5", "w", libver="earliest") as f:
        f.create_dataset("data", data=rng.random((10, 10), dtype=np.float32))

    # 11. Fixed Array with paged data block (libver='latest' + many chunks)
    # 2048 chunks with page_bits=10 → 2 pages of 1024 entries each.
    # This tests the paged Fixed Array code path where the page init bitmap
    # must be handled correctly.
    with h5py.File(_GENERATED_DIR / "fixed_array_paged.h5", "w", libver="latest") as f:
        data = np.arange(2048, dtype=np.float32)
        f.create_dataset(
            "data", data=data, chunks=(1,), compression="gzip", compression_opts=1
        )


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------


async def h5py_comparison(filepath: str, group: str | None = None):
    """Compare async-hdf5 virtual dataset against h5py direct reads.

    Opens the file with both async-hdf5 (via open_hdf5) and h5py,
    then compares dataset shapes, dtypes, and array values.
    """
    import xarray as xr
    from async_hdf5.zarr import open_hdf5

    store = LocalStore()

    hdf5_store = await open_hdf5(
        path=filepath,
        store=store,
        group=group,
    )
    ds = xr.open_dataset(
        hdf5_store, engine="zarr", consolidated=False, zarr_format=3,
        mask_and_scale=False, decode_times=False,
    )

    with h5py.File(filepath, "r") as f:
        target = f[group] if group else f

        for varname in ds.data_vars:
            if varname not in target:
                continue
            h5_ds = target[varname]
            if not isinstance(h5_ds, h5py.Dataset):
                continue

            # Skip variable-length datasets
            if h5py.check_vlen_dtype(h5_ds.dtype):
                continue

            # Skip very large datasets (>100M elements) to avoid OOM
            if h5_ds.size > 100_000_000:
                continue

            # Shape comparison
            assert ds[varname].shape == h5_ds.shape, (
                f"{varname}: shape {ds[varname].shape} != {h5_ds.shape}"
            )

            # Dtype kind and size comparison
            async_dtype = ds[varname].dtype
            h5py_dtype = h5_ds.dtype
            assert async_dtype.kind == h5py_dtype.kind, (
                f"{varname}: dtype kind {async_dtype.kind} != {h5py_dtype.kind}"
            )
            assert async_dtype.itemsize == h5py_dtype.itemsize, (
                f"{varname}: itemsize {async_dtype.itemsize} != {h5py_dtype.itemsize}"
            )

            # Value comparison (with timeout to prevent hangs on broken chunks)
            try:
                actual = await asyncio.wait_for(
                    asyncio.to_thread(lambda v=varname: ds[v].values), timeout=10
                )
            except TimeoutError as err:
                msg = (
                    f"{varname}: data read timed out after 10s"
                    " (likely broken chunk index or byte ranges)"
                )
                raise AssertionError(msg) from err
            expected = h5_ds[()]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-6,
                err_msg=f"Data mismatch for variable '{varname}'",
            )


async def metadata_only_check(filepath: str, group: str | None = None):
    """Lighter check: verify metadata (shape, dtype) without loading data.

    Useful for files where data loading fails due to unsupported codecs but
    metadata parsing should still work.
    """
    import xarray as xr
    from async_hdf5.zarr import open_hdf5

    store = LocalStore()

    hdf5_store = await open_hdf5(
        path=filepath,
        store=store,
        group=group,
    )
    ds = xr.open_dataset(
        hdf5_store, engine="zarr", consolidated=False, zarr_format=3,
        mask_and_scale=False, decode_times=False,
    )

    with h5py.File(filepath, "r") as f:
        target = f[group] if group else f
        for varname in ds.data_vars:
            if varname in target and isinstance(target[varname], h5py.Dataset):
                assert ds[varname].shape == target[varname].shape, (
                    f"{varname}: shape {ds[varname].shape} != {target[varname].shape}"
                )
