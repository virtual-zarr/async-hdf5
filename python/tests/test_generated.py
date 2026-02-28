"""Tests against programmatically generated HDF5 fixtures.

These small files are created by h5py at test-collection time and exercise
key parsing variations: storage layout (contiguous vs chunked), compression,
data types, groups, dimension scales, and superblock versions.
"""

import pytest

from .conftest import generated_examples, h5py_comparison, resolve_folder

# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

# Files with features async-hdf5 doesn't yet support
unsupported: list[str] = []

# Files that trigger parse errors
parse_error: list[str] = []

skip = unsupported + parse_error


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", generated_examples())
async def test_generated_file(filename):
    if filename in skip:
        pytest.xfail("Known failure")
    filepath = str(resolve_folder("tests/data/generated") / filename)
    await h5py_comparison(filepath)
