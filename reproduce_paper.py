"""Run the complete paper reproduction pipeline sequentially.

The default command runs every stage from static validation through the final
qualitative analysis. Existing checkpoints/results are reused by resumable
stages unless their own ``--force`` option is used separately.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES: list[tuple[str, list[str]]] = [
    ("preflight", ["-m", "src.preflight"]),
    ("prepare_data", ["run0_prepare_data.py"]),
    ("cache", ["run1_collect_tokens_and_embeddings.py"]),
    ("train", ["run2_train_models.py"]),
    ("optimize", ["run3_optimize_prototypes.py"]),
    ("quantitative_analysis", ["run4_analyze_quantitative.py"]),
    ("qualitative_prompts", ["run5_build_qualitative_prompts.py"]),
    ("judges", ["run6_llm_as_judge.py"]),
    ("qualitative_analysis", ["run7_analyze_qualitative.py"]),
]


def main() -> None:
    stage_names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser(
        description="Run all paper-reproduction stages in order."
    )
    parser.add_argument("--from-stage", choices=stage_names, default=stage_names[0])
    parser.add_argument("--to-stage", choices=stage_names, default=stage_names[-1])
    parser.add_argument(
        "--toy",
        action="store_true",
        help=("Run a fast self-contained smoke test: synthetic data, one real "
              "training batch, deterministic prototype candidates, and mock judges."),
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip the optional cache-warming stage.",
    )
    parser.add_argument(
        "--skip-judges",
        action="store_true",
        help="Skip paid remote judge calls; useful when raw outputs already exist.",
    )
    args = parser.parse_args()

    start = stage_names.index(args.from_stage)
    end = stage_names.index(args.to_stage)
    if start > end:
        parser.error("--from-stage must not come after --to-stage")

    environment = os.environ.copy()
    if args.toy:
        environment["TMLR_TOY"] = "1"
        environment.setdefault("PYTHONHASHSEED", "0")
        os.environ.update({"TMLR_TOY": "1", "PYTHONHASHSEED": environment["PYTHONHASHSEED"]})
        print("Running the self-contained toy smoke test; these are not paper results.", flush=True)

        # Run toy stages in-process. This avoids repeatedly importing PyTorch,
        # pandas, SciPy, and Matplotlib in nine short-lived subprocesses and
        # makes the smoke test complete in a few seconds on a CPU-only machine.
        from src.preflight import main as preflight_main
        from src.toy_pipeline import run_stage

        for stage_name, _ in STAGES[start : end + 1]:
            if stage_name == "cache" and args.skip_cache:
                print("\n=== cache (skipped) ===", flush=True)
                continue
            if stage_name == "judges" and args.skip_judges:
                print("\n=== judges (skipped) ===", flush=True)
                continue
            print(f"\n=== {stage_name} ===", flush=True)
            if stage_name == "preflight":
                preflight_main()
            else:
                run_stage(stage_name)

        print("\nToy smoke-test pipeline finished successfully.", flush=True)
        print(f"Outputs: {ROOT / 'results' / 'toy'}", flush=True)
        return

    for stage_name, command_tail in STAGES[start : end + 1]:
        if stage_name == "cache" and args.skip_cache:
            print("\n=== cache (skipped) ===", flush=True)
            continue
        if stage_name == "judges" and args.skip_judges:
            print("\n=== judges (skipped) ===", flush=True)
            continue

        command = [sys.executable, *command_tail]
        print(f"\n=== {stage_name} ===", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True, env=environment)

    print("\nComplete reproduction pipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
