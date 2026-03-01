"""Verify that xfail files produce informative error messages.

For each file marked as xfail in test_hdf5_group.py and test_gdal_hdf5.py,
this test verifies that:

1. If open_hdf5() raises an exception, the error message is non-empty and
   matches at least one known informative pattern from the Rust error types.
2. If open_hdf5() succeeds (data-comparison-only failures), the test is
   skipped with a descriptive message.

This ensures that error messages propagate correctly from the Rust library
through PyO3 to Python, and that users see actionable diagnostics.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from .conftest import resolve_folder
from .test_gdal_hdf5 import xfail_files as gdal_xfail_files
from .test_hdf5_group import xfail_files as hdf5_group_xfail_files

# ---------------------------------------------------------------------------
# Informative error message patterns
# ---------------------------------------------------------------------------
#
# Each pattern corresponds to one or more error variants in src/error.rs
# or python/src/dataset.rs.  They are compiled once for efficiency.

INFORMATIVE_PATTERNS: list[re.Pattern[str]] = [
    # --- src/error.rs: HDF5Error variants ---
    # InvalidSignature
    re.compile(r"Not an HDF5 file: .+"),
    # UnsupportedSuperblockVersion
    re.compile(r"Unsupported superblock version: \d+"),
    # UnsupportedObjectHeaderVersion
    re.compile(r"Unsupported object header version: \d+"),
    # UnsupportedDataLayoutVersion
    re.compile(r"Unsupported data layout version: \d+"),
    # UnsupportedDatatypeClass
    re.compile(r"Unsupported datatype class: \d+"),
    # UnsupportedFilterPipelineVersion
    re.compile(r"Unsupported filter pipeline version: \d+"),
    # UnsupportedChunkIndexingType
    re.compile(r"Unsupported chunk indexing type: \d+"),
    # UnsupportedBTreeVersion
    re.compile(r"Unsupported B-tree version: \d+"),
    # InvalidBTreeSignature
    re.compile(r"Invalid B-tree signature: expected .+, got .+"),
    # UnsupportedHeapVersion
    re.compile(r"Unsupported heap version: \d+"),
    # InvalidHeapSignature
    re.compile(r"Invalid heap signature: expected .+, got .+"),
    # NotFound
    re.compile(r"Group member not found: .+"),
    # NotAGroup
    re.compile(r"Expected group at path: .+"),
    # NotADataset
    re.compile(r"Expected dataset at path: .+"),
    # UnexpectedEof
    re.compile(r"Unexpected end of data: needed \d+ bytes, had \d+"),
    # UndefinedAddress
    re.compile(r"Undefined address encountered \(unallocated storage\)"),
    # InvalidObjectHeaderSignature
    re.compile(r"Invalid object header signature: expected OHDR"),
    # UnsupportedLinkType
    re.compile(r"Unsupported link type: \d+"),
    # UnsupportedMessageType
    re.compile(r"Unsupported message type: 0x[0-9a-fA-F]+"),
    # Io
    re.compile(r"I/O error: .+"),
    # ObjectStore
    re.compile(r"Object store error: .+"),
    # General: overflow protection
    re.compile(r"object header address .+ overflows"),
    re.compile(r"continuation address .+ overflows"),
    # General: "failed to fill whole buffer" (from std::io)
    re.compile(r"failed to fill whole buffer"),
    # General: data too short
    re.compile(r"data too short"),
    # --- python/src/dataset.rs: ValueError messages ---
    re.compile(r"Variable-length string datatype cannot be represented"),
    re.compile(r"Variable-length sequence datatype cannot be represented"),
    re.compile(r"HDF5 reference datatype \(ref_type=\d+\) is not supported"),
    # --- python/python/async_hdf5/zarr.py: _decode_fill_value ---
    re.compile(r"Fill value has \d+ byte\(s\) but dtype .+ expects \d+"),
    # --- Edge cases seen in practice ---
    # Virtual dataset layout (class 3)
    re.compile(r"unknown layout class: \d+"),
    re.compile(r"length must be >= 1"),
    re.compile(r"No such file or directory"),
]


def _is_informative(message: str) -> bool:
    """Check whether an error message matches at least one known pattern."""
    return any(pattern.search(message) for pattern in INFORMATIVE_PATTERNS)


# ---------------------------------------------------------------------------
# Parametrized file list
# ---------------------------------------------------------------------------


def _xfail_params() -> list[pytest.param]:
    """Build parametrized test cases from both xfail sources."""
    params: list[pytest.param] = []

    for rel_path in sorted(hdf5_group_xfail_files):
        filepath = str(resolve_folder("tests/data/hdf5-group") / rel_path)
        params.append(pytest.param(filepath, id=f"hdf5-group/{rel_path}"))

    for rel_path in sorted(gdal_xfail_files):
        filepath = str(resolve_folder("tests/data/gdal") / rel_path)
        params.append(pytest.param(filepath, id=f"gdal/{rel_path}"))

    return params


# ---------------------------------------------------------------------------
# Timeout for open_hdf5() calls (seconds)
# ---------------------------------------------------------------------------

OPEN_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filepath", _xfail_params())
async def test_xfail_error_message(filepath: str):
    """Verify that xfail files produce informative error messages.

    For files that raise during open_hdf5():
      - The error message must be non-empty.
      - The error message must match at least one known informative pattern.

    For files that open successfully:
      - The test is skipped (these fail during data comparison, not parsing).
    """
    from async_hdf5.store import LocalStore
    from async_hdf5.zarr import open_hdf5

    store = LocalStore()

    try:
        _hdf5_store = await asyncio.wait_for(
            open_hdf5(path=filepath, store=store),
            timeout=OPEN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"open_hdf5() timed out after {OPEN_TIMEOUT_SECONDS}s — "
            f"possible hang on {filepath}"
        )
    except FileNotFoundError as exc:
        msg = str(exc)
        assert msg, f"FileNotFoundError with empty message for {filepath}"
        return
    except Exception as exc:
        msg = str(exc)
        assert msg, f"Empty error message from {type(exc).__name__} for {filepath}"
        assert _is_informative(msg), (
            f"Non-informative error message from {type(exc).__name__} "
            f"for {filepath}: {msg!r}"
        )
        return

    pytest.skip("File opened successfully — xfail is for data comparison, not error messaging")
