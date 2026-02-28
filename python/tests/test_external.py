"""Tests against external real-world HDF5/NetCDF4 files.

These files are curated from public scientific data providers (Unidata, NASA,
NOAA, etc.) and exercise real-world data structures, compression, and CF
conventions.
"""

import pytest

from async_hdf5 import HDF5File

from .conftest import external_examples, h5py_comparison, resolve_folder

# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

# Files that are too large for CI without --run-slow-tests
large_files: list[str] = []

# Files with unsupported HDF5 filters (szip, scaleoffset, etc.)
unsupported_filter: list[str] = []

# Files with unsupported data types (bitfield, opaque, reference, etc.)
unsupported_datatype: list[str] = []

# Files that trigger parse errors in async-hdf5
parse_error: list[str] = []

# Files with variable-length datasets that can't be virtualized
vlen_datasets: list[str] = []

# Files that are not HDF5 (e.g. NetCDF3 classic) — kept as negative tests
not_hdf5 = [
    "ECMWF_ERA-40_subset.nc",
    "tos_O1_2001-2002.nc",
    "madis-maritime.nc",
]

skip = unsupported_filter + unsupported_datatype + parse_error + vlen_datasets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", external_examples())
async def test_external_file(filename):
    if filename in large_files:
        pytest.skip("Large file -- use --run-slow-tests")
    if filename in skip:
        pytest.xfail("Known failure")
    filepath = str(resolve_folder("tests/data/external") / filename)

    if filename in not_hdf5:
        # Verify that non-HDF5 files produce a clear error message
        from obstore.store import LocalStore

        store = LocalStore()
        with pytest.raises(Exception, match="Not an HDF5 file"):
            await HDF5File.open(filepath, store=store)
        return

    await h5py_comparison(filepath)
