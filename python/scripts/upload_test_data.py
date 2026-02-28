#!/usr/bin/env python3
"""Upload test HDF5/NetCDF4 files from tests/data/ to S3 bucket.

Uploads files and a generated README.md (with per-file source URLs
and license information) to s3://us-west-2.opendata.source.coop/pangeo/example-hdf5/.
Only new files are uploaded unless --force is given.

Usage:
    # Upload new files and README
    uv run python scripts/upload_test_data.py

    # Dry-run: show files that would be uploaded and preview README
    uv run python scripts/upload_test_data.py --dry-run

    # Force re-upload all files
    uv run python scripts/upload_test_data.py --force

    # Upload without regenerating README
    uv run python scripts/upload_test_data.py --skip-readme
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import obstore as obs
from obstore.store import S3Store

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "pangeo/example-hdf5"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
SOURCES_FILE = REPO_ROOT / "scripts" / "github_sources.json"
EXTENSIONS = (".h5", ".hdf5", ".nc", ".he5")


def get_store():
    return S3Store(
        bucket=BUCKET,
        config={"AWS_REGION": "us-west-2"},
    )


def list_local_files():
    """List all HDF5/NetCDF4 files under the local data directory."""
    files = []
    for ext in EXTENSIONS:
        for f in DATA_DIR.rglob(f"*{ext}"):
            rel = f.relative_to(DATA_DIR)
            files.append(str(rel))
    return sorted(files)


def list_remote_files(store):
    """List all HDF5/NetCDF4 files under the prefix in S3."""
    result = obs.list(store, prefix=PREFIX)
    files = set()
    for batch in result:
        for meta in batch:
            path = meta["path"]
            if any(path.endswith(ext) for ext in EXTENSIONS):
                rel = path[len(PREFIX) + 1 :]
                if rel:
                    files.add(rel)
    return files


def upload_file(store, rel_path):
    """Upload a single file to S3."""
    local_path = DATA_DIR / rel_path
    remote_key = f"{PREFIX}/{rel_path}"
    data = local_path.read_bytes()
    obs.put(store, remote_key, data)


def build_readme(all_files):
    """Build a README.md with metadata and license information."""
    external_sources = {}
    if SOURCES_FILE.exists():
        external_sources = json.loads(SOURCES_FILE.read_text())

    hdf5_group_files = sorted(f for f in all_files if f.startswith("hdf5-group/"))
    gdal_files = sorted(f for f in all_files if f.startswith("gdal/"))
    h5py_files = sorted(f for f in all_files if f.startswith("h5py/"))
    external_files = sorted(f for f in all_files if f.startswith("external/"))
    netcdf_files = sorted(f for f in all_files if f.startswith("netcdf-test-data/"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# HDF5/NetCDF4 test data",
        "",
        "A public collection of HDF5 and NetCDF4 files covering a wide range of "
        "data types, compression filters, chunking schemes, and group structures. "
        "The files are useful for testing any HDF5 reading or parsing library and "
        "are maintained as part of the "
        "[async-hdf5](https://github.com/developmentseed/async-hdf5) project.",
        "",
        f"Last updated: {now}",
        "",
        "## Sources and licenses",
        "",
    ]

    if hdf5_group_files:
        lines += [
            "### HDF5 Group autotest (`hdf5-group/`)",
            "",
            f"{len(hdf5_group_files)} files extracted from the HDF5 library "
            "[test](https://github.com/HDFGroup/hdf5/tree/develop/test) directory.",
            "",
            "- **Source:** <https://github.com/HDFGroup/hdf5>",
            "- **License:** BSD-3-Clause",
            "- **Copyright:** The HDF Group",
            "",
        ]

    if gdal_files:
        lines += [
            "### GDAL autotest (`gdal/`)",
            "",
            f"{len(gdal_files)} HDF5/NetCDF4 files extracted from the GDAL "
            "[autotest](https://github.com/OSGeo/gdal/tree/master/autotest) directory.",
            "",
            "- **Source:** <https://github.com/OSGeo/gdal>",
            "- **License:** MIT/X -- "
            "see [GDAL LICENSE.TXT]"
            "(https://github.com/OSGeo/gdal/blob/master/LICENSE.TXT)",
            "- **Copyright:** Frank Warmerdam, Even Rouault, and contributors",
            "",
        ]

    if h5py_files:
        lines += [
            "### h5py test suite (`h5py/`)",
            "",
            f"{len(h5py_files)} files from the "
            "[h5py](https://github.com/h5py/h5py) test suite.",
            "",
            "- **Source:** <https://github.com/h5py/h5py>",
            "- **License:** BSD-3-Clause",
            "- **Copyright:** h5py contributors",
            "",
        ]

    if external_files:
        lines += [
            "### External files (`external/`)",
            "",
            "Files downloaded from third-party sources for testing a "
            "variety of HDF5/NetCDF4 encodings.",
            "",
            "| File | Source URL | License |",
            "|------|-----------|---------|",
        ]

        license_map = {
            "ECMWF_ERA-40_subset.nc": "CC-BY-4.0 (ECMWF)",
            "tos_O1_2001-2002.nc": "Freely available (PCMDI/CMIP via Unidata)",
            "madis-maritime.nc": "Public domain (NOAA / U.S. Government)",
        }

        for f in external_files:
            filename = Path(f).name
            entry = external_sources.get(filename, {})
            url = entry.get("url", "") if isinstance(entry, dict) else entry
            license_info = license_map.get(filename, "See source")
            lines.append(f"| `{filename}` | {url} | {license_info} |")

        lines.append("")

    if netcdf_files:
        lines += [
            "### netcdf-test-data (`netcdf-test-data/`)",
            "",
            f"{len(netcdf_files)} programmatically generated NetCDF4 files from the "
            "netcdf-test-data project.",
            "",
            "- **Source:** netcdf-test-data",
            "- **License:** MIT",
            "",
        ]

    lines += [
        "## Usage",
        "",
        "These files are publicly hosted and can be used by any project. "
        "To download them with the async-hdf5 helper script:",
        "",
        "```bash",
        "uv run python scripts/download_test_data.py",
        "```",
        "",
        "## Disclaimer",
        "",
        "These files are redistributed solely for automated testing. "
        "Refer to each source for authoritative license terms.",
        "",
    ]

    return "\n".join(lines)


def upload_readme(store, all_files, dry_run=False):
    """Generate and upload README.md to the S3 prefix."""
    readme_content = build_readme(all_files)
    remote_key = f"{PREFIX}/README.md"

    if dry_run:
        print(f"\nWould upload README.md to s3://{BUCKET}/{remote_key}")
        print("--- README.md preview ---")
        print(readme_content)
        print("--- end preview ---")
        return

    obs.put(store, remote_key, readme_content.encode("utf-8"))
    print(f"  Uploaded README.md to s3://{BUCKET}/{remote_key}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-upload files that already exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip generating and uploading README.md",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Run the sync scripts first or populate tests/data/ manually.")
        return

    store = get_store()
    local_files = list_local_files()
    print(f"Found {len(local_files)} local files")

    if args.force:
        to_upload = local_files
    else:
        print(f"Listing existing files in s3://{BUCKET}/{PREFIX}/ ...")
        remote_files = list_remote_files(store)
        to_upload = [f for f in local_files if f not in remote_files]
        print(f"  {len(remote_files)} already in S3, {len(to_upload)} new")

    if args.dry_run:
        if to_upload:
            print(f"\nWould upload {len(to_upload)} files:")
            for f in to_upload:
                print(f"  {f}")
        else:
            print("No new files to upload.")
        if not args.skip_readme:
            upload_readme(store, local_files, dry_run=True)
        return

    for i, rel_path in enumerate(to_upload, 1):
        upload_file(store, rel_path)
        print(f"  [{i}/{len(to_upload)}] Uploaded {rel_path}")

    if not args.skip_readme:
        upload_readme(store, local_files)

    print(f"\nDone: {len(to_upload)} files uploaded")


if __name__ == "__main__":
    main()
