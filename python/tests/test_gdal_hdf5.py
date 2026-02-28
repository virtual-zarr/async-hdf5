"""Tests against GDAL autotest HDF5/NetCDF4 files.

GDAL's autotest suite includes HDF5 and NetCDF4 files covering various
driver-specific edge cases, geospatial metadata, and data type combinations.
"""

import pytest

from .conftest import gdal_examples, h5py_comparison, resolve_folder

# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

# Fill value returned as raw bytes instead of a numeric scalar — causes
# Zarr's cast_scalar() to fail with "could not convert string to float"
# or "invalid literal for int()".
fill_value_as_bytes: list[str] = [
    "gdrivers/data/hdf5/deflate.h5",
    "gdrivers/data/hdf5/scale_offset.h5",
    "gdrivers/data/netcdf/byte_chunked_multiple.nc",
    "gdrivers/data/netcdf/byte_chunked_not_multiple.nc",
    "gdrivers/data/netcdf/complex.nc",
    "gdrivers/data/netcdf/fake_EMIT_L2A_with_good_wavelengths.nc",
    "gdrivers/data/netcdf/fake_EMIT_L2B_MIN.nc",
    "gdrivers/data/netcdf/fake_ISO_METADATA.nc",
    "gdrivers/data/netcdf/int64.nc",
    "gdrivers/data/netcdf/nc_mixed_raster_vector.nc",
    "gdrivers/data/netcdf/sen3_sral_mwr_fake_standard_measurement.nc",
    "gdrivers/data/netcdf/test_gridded.nc",
    "gdrivers/data/netcdf/uint16_netcdf4_without_fill.nc",
    "gdrivers/data/netcdf/uint64.nc",
    "gdrivers/data/tiledb_input/DeepBlue-SeaWiFS-1.0_L3_20100101_v004-20130604T131317Z.h5",
]

# Unsupported datatype class (compound = class 2, etc.)
unsupported_datatype: list[str] = [
    "gdrivers/data/hdf5/complex.h5",
    "gdrivers/data/netcdf/alldatatypes.nc",
]

# Attributes containing raw bytes that can't be JSON-serialized
unsupported_attrs: list[str] = [
    "gdrivers/data/hdf5/attr_all_datatypes.h5",
]

# Truncated or split files — I/O error: failed to fill whole buffer
truncated_or_split: list[str] = [
    "gdrivers/data/hdf5/test_family_0.h5",
    "gdrivers/data/hdf5/u8be.h5",
    "gdrivers/data/netcdf/byte_truncated.nc",
    "gdrivers/data/netcdf/trmm-nc4.nc",
]

# Dtype or value comparison mismatches
dtype_mismatch: list[str] = [
    "gdrivers/data/hdf5/single_char_varname.h5",
    "gdrivers/data/netcdf/bug5291.nc",
    "gdrivers/data/netcdf/int64dim.nc",
    "gdrivers/data/netcdf/test_ogr_nc4.nc",
]

xfail_files = (
    fill_value_as_bytes
    + unsupported_datatype
    + unsupported_attrs
    + truncated_or_split
    + dtype_mismatch
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", gdal_examples())
async def test_gdal_hdf5_file(rel_path):
    if rel_path in xfail_files:
        pytest.xfail("Known failure")
    filepath = str(resolve_folder("tests/data/gdal") / rel_path)
    await h5py_comparison(filepath)
