"""Safely extract dataset ZIP archives in place without deleting existing files."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.config import DATASETS, DATA_DIR, ensure_directories


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe ZIP member in {archive}: {member.filename}")
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    args = parser.parse_args()
    ensure_directories()
    for dataset in args.datasets:
        directory = DATA_DIR / dataset
        directory.mkdir(parents=True, exist_ok=True)
        archives = sorted(directory.glob("*.zip"))
        if not archives:
            print(f"{dataset}: no ZIP files found")
            continue
        for archive in archives:
            print(f"Extracting {archive}")
            safe_extract(archive, directory)


if __name__ == "__main__":
    main()
