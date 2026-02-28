#!/usr/bin/env python3
"""Sync HDF5/NetCDF4 test files from the latest GDAL release into tests/data/gdal/.

Downloads a tarball of the latest GDAL release via the GitHub API using pooch
for caching, then extracts .h5, .hdf5, .nc, and .he5 files from autotest/ into
tests/data/gdal/ preserving the directory structure. Files are validated with
HDF5 magic bytes to filter out NetCDF3 classic and HDF4 files.

Usage:
    # Sync new files
    uv run --group scripts python scripts/sync_gdal_hdf5.py

    # Dry-run: show new files that would be extracted
    uv run --group scripts python scripts/sync_gdal_hdf5.py --dry-run

    # Force re-download even if already on latest release
    uv run --group scripts python scripts/sync_gdal_hdf5.py --force
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pooch

GITHUB_API_LATEST = "https://api.github.com/repos/OSGeo/gdal/releases/latest"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
GDAL_DIR = DATA_DIR / "gdal"
RELEASE_FILE = GDAL_DIR / ".release"
CACHE_DIR = DATA_DIR / ".cache"
EXTENSIONS = {".h5", ".hdf5", ".nc", ".he5"}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def get_latest_release() -> tuple[str, str]:
    """Return (tag_name, tarball_url) for the latest GDAL release."""
    req = Request(GITHUB_API_LATEST, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["tag_name"], data["tarball_url"]


def get_local_release() -> str | None:
    if RELEASE_FILE.exists():
        return RELEASE_FILE.read_text().strip()
    return None


def save_local_release(tag: str):
    RELEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RELEASE_FILE.write_text(tag + "\n")


def is_hdf5_file(data: bytes) -> bool:
    """Check if file data starts with HDF5 magic bytes."""
    return data[:8] == HDF5_MAGIC


def extract_autotest_hdf5(tarball_path: str, dest: Path) -> list[str]:
    """Extract HDF5/NetCDF4 files from autotest/ in the GDAL tarball.

    Only extracts files with valid HDF5 magic bytes (filters out NetCDF3
    classic .nc files and HDF4 files).
    """
    hdf5_paths = []
    with tarfile.open(tarball_path, "r:gz") as tar:
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if "/" in m.name}
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected 1 top-level dir in tarball, got: {top_dirs}")
        top_dir = top_dirs.pop()
        autotest_prefix = f"{top_dir}/autotest/"

        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.startswith(autotest_prefix):
                continue
            if not any(member.name.endswith(ext) for ext in EXTENSIONS):
                continue

            rel = member.name[len(autotest_prefix) :]
            if not rel:
                continue

            with tar.extractfile(member) as src:
                content = src.read()

            if not is_hdf5_file(content):
                continue

            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hdf5_paths.append(rel)

    return sorted(hdf5_paths)


def find_local_files() -> set[str]:
    if not GDAL_DIR.exists():
        return set()
    files = set()
    for ext in EXTENSIONS:
        files.update(
            str(f.relative_to(GDAL_DIR)) for f in GDAL_DIR.rglob(f"*{ext}")
        )
    return files


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if on latest release"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be extracted"
    )
    args = parser.parse_args()

    print("Checking latest GDAL release...")
    tag, tarball_url = get_latest_release()
    local_tag = get_local_release()
    print(f"  Latest release: {tag}")
    print(f"  Local release:  {local_tag or '(none)'}")

    if tag == local_tag and not args.force:
        print(f"\nAlready synced to {tag}. Use --force to re-download.")
        return

    local_files = find_local_files()

    print(f"\nDownloading {tag} tarball...")
    tarball_path = pooch.retrieve(
        url=tarball_url,
        known_hash=None,
        fname=f"gdal-{tag}.tar.gz",
        path=CACHE_DIR,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting HDF5/NetCDF4 files from autotest/...")
        hdf5_files = extract_autotest_hdf5(tarball_path, Path(tmpdir))
        print(f"Found {len(hdf5_files)} HDF5/NetCDF4 files in GDAL autotest/")

        new_files = [f for f in hdf5_files if f not in local_files]
        print(f"  {len(local_files)} already in tests/data/gdal/")
        print(f"  {len(new_files)} new files")

        if not new_files:
            print("\nNo new files to sync.")
            save_local_release(tag)
            return

        if args.dry_run:
            print(f"\nWould copy {len(new_files)} files:")
            for f in new_files:
                print(f"  {f}")
            return

        print(f"\nCopying {len(new_files)} new files...")
        GDAL_DIR.mkdir(parents=True, exist_ok=True)
        for rel in new_files:
            src = Path(tmpdir) / rel
            dst = GDAL_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        save_local_release(tag)
        print(f"Done. Release marker updated to {tag}")


if __name__ == "__main__":
    main()
