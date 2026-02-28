#!/usr/bin/env python3
"""Sync HDF5 test files from the latest HDF5 library release into tests/data/hdf5-group/.

Downloads a tarball of the latest HDF5 release from the HDFGroup GitHub repo
using pooch for caching, then extracts .h5 files from test/ directories into
tests/data/hdf5-group/ preserving the directory structure.

Usage:
    # Sync new files
    uv run --group scripts python scripts/sync_hdf5_group.py

    # Dry-run: show new files that would be extracted
    uv run --group scripts python scripts/sync_hdf5_group.py --dry-run

    # Force re-download even if already on latest release
    uv run --group scripts python scripts/sync_hdf5_group.py --force
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pooch

GITHUB_API_LATEST = "https://api.github.com/repos/HDFGroup/hdf5/releases/latest"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
HDF5_GROUP_DIR = DATA_DIR / "hdf5-group"
RELEASE_FILE = HDF5_GROUP_DIR / ".release"
CACHE_DIR = DATA_DIR / ".cache"
EXTENSIONS = {".h5", ".hdf5"}
TEST_PREFIXES = ("test/", "testfiles/", "tools/test/", "hl/test/")


def get_latest_release() -> tuple[str, str]:
    """Return (tag_name, tarball_url) for the latest HDF5 release."""
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


def extract_test_hdf5(tarball_path: str, dest: Path) -> list[str]:
    """Extract .h5/.hdf5 files from test/ directories in the tarball."""
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
            if not any(rel_in_tarball.startswith(p) for p in TEST_PREFIXES):
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
    if not HDF5_GROUP_DIR.exists():
        return set()
    files = set()
    for ext in EXTENSIONS:
        files.update(
            str(f.relative_to(HDF5_GROUP_DIR))
            for f in HDF5_GROUP_DIR.rglob(f"*{ext}")
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

    print("Checking latest HDF5 release...")
    tag, tarball_url = get_latest_release()
    local_tag = get_local_release()
    print(f"  Latest release: {tag}")
    print(f"  Local release:  {local_tag or '(none)'}")

    if tag == local_tag and not args.force:
        print(f"\nAlready synced to {tag}. Use --force to re-download.")
        return

    local_files = find_local_files()

    # Download tarball via pooch (cached by tag name)
    print(f"\nDownloading {tag} tarball...")
    tarball_path = pooch.retrieve(
        url=tarball_url,
        known_hash=None,
        fname=f"hdf5-{tag}.tar.gz",
        path=CACHE_DIR,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting HDF5 test files...")
        hdf5_files = extract_test_hdf5(tarball_path, Path(tmpdir))
        print(f"Found {len(hdf5_files)} HDF5 files in test directories")

        new_files = [f for f in hdf5_files if f not in local_files]
        print(f"  {len(local_files)} already in tests/data/hdf5-group/")
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
        HDF5_GROUP_DIR.mkdir(parents=True, exist_ok=True)
        for rel in new_files:
            src = Path(tmpdir) / rel
            dst = HDF5_GROUP_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        save_local_release(tag)
        print(f"Done. Release marker updated to {tag}")


if __name__ == "__main__":
    main()
