#!/usr/bin/env python3
"""Sync external test HDF5/NetCDF4 files into tests/data/external/.

Downloads files listed in scripts/github_sources.json from their original
URLs using pooch for caching and hash verification. This populates the local
test data directory so it can then be uploaded to S3 with upload_test_data.py.

Usage:
    # Download new files (skips cached files with matching hash)
    uv run --group scripts python scripts/sync_external_hdf5.py

    # Dry-run: show files that would be downloaded
    uv run --group scripts python scripts/sync_external_hdf5.py --dry-run

    # Force re-download all files
    uv run --group scripts python scripts/sync_external_hdf5.py --force

    # Download and update github_sources.json with computed hashes
    uv run --group scripts python scripts/sync_external_hdf5.py --compute-hashes
"""

import argparse
import hashlib
import json
from pathlib import Path

import pooch

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = REPO_ROOT / "tests" / "data" / "external"
SOURCES_FILE = REPO_ROOT / "scripts" / "github_sources.json"


def load_sources() -> dict[str, dict]:
    return json.loads(SOURCES_FILE.read_text())


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download all files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--compute-hashes",
        action="store_true",
        help="Update github_sources.json with SHA-256 hashes after downloading",
    )
    args = parser.parse_args()

    sources = load_sources()
    print(f"Found {len(sources)} entries in {SOURCES_FILE.name}")

    if args.dry_run:
        for filename, entry in sources.items():
            local_path = EXTERNAL_DIR / filename
            status = "cached" if local_path.exists() and not args.force else "to download"
            print(f"  [{status}] {filename}")
        return

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    for i, (filename, entry) in enumerate(sources.items(), 1):
        url = entry["url"]
        known_hash = entry.get("hash")

        try:
            path = pooch.retrieve(
                url=url,
                known_hash=known_hash,
                fname=filename,
                path=EXTERNAL_DIR,
            )
            print(f"  [{i}/{len(sources)}] {filename} -> {path}")
            downloaded += 1

            if args.compute_hashes and not known_hash:
                entry["hash"] = compute_sha256(Path(path))
                print(f"    hash: {entry['hash']}")

        except Exception as e:
            print(f"  [{i}/{len(sources)}] {filename} FAILED: {e}")
            failed += 1

    if args.compute_hashes:
        SOURCES_FILE.write_text(json.dumps(sources, indent=2) + "\n")
        print(f"\nUpdated {SOURCES_FILE.name} with hashes")

    print(f"\nDone: {downloaded} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
