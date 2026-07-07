"""Schedule all 90 paper training runs across the configured CUDA devices."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

from src.config import DATASETS, DATASET_SETTINGS, DEVICES, QUANT_MODELS, ROOT, SEEDS
from src.functions import checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--devices", nargs="*", default=DEVICES)
    args = parser.parse_args()

    jobs = []
    for dataset in DATASETS:
        for model in QUANT_MODELS:
            for seed in SEEDS:
                output = checkpoint_path(model, dataset, seed)
                if output.exists() and not args.force:
                    print(f"Skipping existing checkpoint: {output.name}")
                    continue
                jobs.append((dataset, model, seed))

    if not jobs:
        print("All training checkpoints already exist.")
        return
    if not args.devices:
        raise RuntimeError("At least one device is required")

    device_cycle = itertools.cycle(args.devices)
    running: list[tuple[subprocess.Popen, str, tuple[str, str, int]]] = []

    while jobs or running:
        while jobs and len(running) < len(args.devices):
            dataset, model, seed = jobs.pop(0)
            device = next(device_cycle)
            settings = DATASET_SETTINGS[dataset]
            command = [
                sys.executable,
                "-m",
                "src.train_prototype_models",
                "--dataset", dataset,
                "--model", model,
                "--seed", str(seed),
                "--device", device,
                "--epochs", str(settings["epochs"]),
                "--prototypes-per-class", str(settings["prototypes_per_class"]),
            ]
            print("Launching:", " ".join(command))
            running.append((subprocess.Popen(command, cwd=ROOT), device, (dataset, model, seed)))

        for process, device, job in running[:]:
            return_code = process.poll()
            if return_code is None:
                continue
            running.remove((process, device, job))
            if return_code != 0:
                raise RuntimeError(f"Training failed for {job} on {device} with code {return_code}")
            print(f"Completed {job} on {device}")
        time.sleep(2)


if __name__ == "__main__":
    main()
