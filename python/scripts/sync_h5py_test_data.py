#!/usr/bin/env python3
"""Sync HDF5 test files from the h5py repository into tests/data/h5py/.

Downloads the h5py source tarball from GitHub using pooch for caching, then
extracts .h5 files from the test directory. Tracks the last-synced commit SHA
in tests/data/h5py/.revision to avoid redundant downloads.

Usage:
    # Sync new files
    uv run --group scripts python scripts/sync_h5py_test_data.py

    # Dry-run: show new files that would be extracted
    uv run --group scripts python scripts/sync_h5py_test_data.py --dry-run

    # Force re-download
    uv run --group scripts python scripts/sync_h5py_test_data.py --force
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pooch

GITHUB_API_COMMITS = "https://api.github.com/repos/h5py/h5py/commits/master"
GITHUB_TARBALL = "https://api.github.com/repos/h5py/h5py/tarball/master"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
H5PY_DIR = DATA_DIR / "h5py"
REVISION_FILE = H5PY_DIR / ".revision"
CACHE_DIR = DATA_DIR / ".cache"
EXTENSIONS = {".h5", ".hdf5"}


def get_latest_commit() -> str:
    """Return the SHA of the latest commit on master."""
    req = Request(GITHUB_API_COMMITS, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["sha"]


def get_local_revision() -> str | None:
    if REVISION_FILE.exists():
        return REVISION_FILE.read_text().strip()
    return None


def save_local_revision(sha: str):
    REVISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVISION_FILE.write_text(sha + "\n")


def extract_test_hdf5(tarball_path: str, dest: Path) -> list[str]:
    """Extract .h5/.hdf5 files from h5py's test directories."""
    hdf5_paths = []
    with tarfile.open(tarball_path, "r:gz") as tar:
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if "/" in m.name}
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected 1 top-level dir in tarball, got: {top_dirs}")
        top_dir = top_dirs.pop()

        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel_in_tarball = member.name[len(top_dir) + 1 :]
            if not rel_in_tarball:
                continue
            if not any(rel_in_tarball.endswith(ext) for ext in EXTENSIONS):
                continue

            target = dest / rel_in_tarball
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                target.write_bytes(src.read())
            hdf5_paths.append(rel_in_tarball)

    return sorted(hdf5_paths)


def find_local_files() -> set[str]:
    if not H5PY_DIR.exists():
        return set()
    files = set()
    for ext in EXTENSIONS:
        files.update(
            str(f.relative_to(H5PY_DIR)) for f in H5PY_DIR.rglob(f"*{ext}")
        )
    return files


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if on latest revision"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be extracted"
    )
    args = parser.parse_args()

    print("Checking latest h5py commit...")
    sha = get_latest_commit()
    local_sha = get_local_revision()
    print(f"  Latest commit: {sha[:12]}")
    print(f"  Local commit:  {(local_sha[:12]) if local_sha else '(none)'}")

    if sha == local_sha and not args.force:
        print(f"\nAlready synced to {sha[:12]}. Use --force to re-download.")
        return

    local_files = find_local_files()

    # Use commit SHA in filename so pooch caches per-revision
    print(f"\nDownloading h5py tarball ({sha[:12]})...")
    tarball_path = pooch.retrieve(
        url=GITHUB_TARBALL,
        known_hash=None,
        fname=f"h5py-{sha[:12]}.tar.gz",
        path=CACHE_DIR,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting HDF5 test files...")
        hdf5_files = extract_test_hdf5(tarball_path, Path(tmpdir))
        print(f"Found {len(hdf5_files)} HDF5 files")

        new_files = [f for f in hdf5_files if f not in local_files]
        print(f"  {len(local_files)} already in tests/data/h5py/")
        print(f"  {len(new_files)} new files")

        if not new_files:
            print("\nNo new files to sync.")
            save_local_revision(sha)
            return

        if args.dry_run:
            print(f"\nWould copy {len(new_files)} files:")
            for f in new_files:
                print(f"  {f}")
            return

        print(f"\nCopying {len(new_files)} new files...")
        H5PY_DIR.mkdir(parents=True, exist_ok=True)
        for rel in new_files:
            src = Path(tmpdir) / rel
            dst = H5PY_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        save_local_revision(sha)
        print(f"Done. Revision marker updated to {sha[:12]}")


if __name__ == "__main__":
    main()
