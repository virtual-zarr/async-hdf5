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


async def test_fixed_array_paged_chunk_index():
    """Verify that paged Fixed Array chunk indices are parsed correctly.

    Regression test: the page init bitmap was preventing pages from being read,
    resulting in an empty chunk index and all-zero data.
    """
    from async_hdf5 import HDF5File
    from async_hdf5.store import LocalStore

    generated_examples()  # ensure fixtures exist
    filepath = str(resolve_folder("tests/data/generated") / "fixed_array_paged.h5")

    store = LocalStore()
    f = await HDF5File.open(filepath, store=store)
    root = await f.root_group()
    ds = await root.dataset("data")

    index = await ds.chunk_index()
    assert len(index) == 2048, f"Expected 2048 chunks, got {len(index)}"

    # Verify chunk locations are valid
    for chunk in index:
        assert chunk.byte_offset > 0
        assert chunk.byte_length > 0
