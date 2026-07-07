"""Optionally precompute token caches for every paper dataset/backbone pair."""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.config import DATASETS, QUANT_MODELS, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for dataset in DATASETS:
        for model in QUANT_MODELS:
            command = [
                sys.executable,
                "-m",
                "src.collect_tokens_embeddings",
                "--dataset", dataset,
                "--model", model,
                "--device", args.device,
            ]
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
