"""Tests against h5py test suite files.

h5py's test files exercise edge cases in the HDF5 format including compound
types, enum types, opaque types, external links, virtual datasets, dimensional
scales, and SWMR.
"""

from pathlib import Path

import pytest

from .conftest import h5py_comparison, h5py_examples, resolve_folder

# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

# Files with unsupported filters
unsupported_filter: list[str] = []

# Files with unsupported data types (compound, enum, opaque, reference, etc.)
unsupported_datatype: list[str] = []

# Files that trigger parse errors
parse_error: list[str] = []

# Files with variable-length datasets
vlen_datasets: list[str] = []

# Files with external links
external_link: list[str] = []

# Files using HDF5 Virtual Dataset feature
virtual_dataset: list[str] = []

skip = (
    unsupported_filter
    + unsupported_datatype
    + parse_error
    + vlen_datasets
    + external_link
    + virtual_dataset
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", h5py_examples())
async def test_h5py_file(rel_path):
    filename = Path(rel_path).name
    if filename in skip:
        pytest.xfail("Known failure")
    filepath = str(resolve_folder("tests/data/h5py") / rel_path)
    await h5py_comparison(filepath)
