"""Extract downloaded dataset archives and build the processed CSV files."""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.config import DATASETS, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    args = parser.parse_args()

    dataset_args = ["--datasets", *args.datasets]
    stages = [
        [sys.executable, "-m", "src.unzipper", *dataset_args],
        [sys.executable, "-m", "src.preprocess", *dataset_args],
    ]
    for command in stages:
        print("\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
