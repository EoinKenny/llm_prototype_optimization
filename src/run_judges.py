"""Run all five judge models with resumable JSONL output."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import JUDGES, RESULTS_DIR, ensure_directories
from src.judge_clients import call_judge_api


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                completed.add(str(record["prompt_id"]))
    return completed


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_judge(
    judge_name: str,
    prompts: pd.DataFrame,
    batch_size: int,
    retries: int,
    sleep_seconds: float,
) -> None:
    spec = JUDGES[judge_name]
    output = RESULTS_DIR / "qualitative" / "judgments" / f"{judge_name}.jsonl"
    done = completed_ids(output)
    pending = prompts[~prompts["prompt_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"{judge_name}: {len(pending):,} prompts pending, {len(done):,} already complete")

    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        batch_prompts = batch["prompt"].astype(str).tolist()
        last_error = ""

        for attempt in range(1, retries + 1):
            try:
                responses = call_judge_api(
                    prompts=batch_prompts,
                    provider=spec["provider"],
                    model_id=spec["model_id"],
                )
                if len(responses) != len(batch_prompts):
                    raise ValueError(
                        "call_judge_api() must return exactly one response per prompt "
                        f"({len(batch_prompts)} expected, {len(responses)} returned)."
                    )

                for row, response in zip(batch.itertuples(index=False), responses):
                    append_jsonl(output, {
                        "prompt_id": row.prompt_id,
                        "judge": judge_name,
                        "model_id": spec["model_id"],
                        "status": "ok",
                        "response": str(response),
                    })
                break
            except Exception as error:
                last_error = repr(error)
                if attempt == retries:
                    for row in batch.itertuples(index=False):
                        append_jsonl(output, {
                            "prompt_id": row.prompt_id,
                            "judge": judge_name,
                            "model_id": spec["model_id"],
                            "status": "error",
                            "error": last_error,
                        })
                    raise RuntimeError(
                        f"{judge_name} failed for batch starting at row {start}: {last_error}"
                    ) from error
                time.sleep(sleep_seconds * (2 ** (attempt - 1)))

        print(f"{judge_name}: completed {min(start + batch_size, len(pending))}/{len(pending)}")
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(RESULTS_DIR / "qualitative" / "prompts.csv"))
    parser.add_argument("--judges", nargs="*", default=list(JUDGES), choices=sorted(JUDGES))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    args = parser.parse_args()
    ensure_directories()

    prompts = pd.read_csv(args.prompts)
    required = {"prompt_id", "prompt"}
    if not required.issubset(prompts.columns):
        raise ValueError(f"Prompt CSV must contain {sorted(required)}")

    for judge_name in args.judges:
        run_judge(
            judge_name=judge_name,
            prompts=prompts,
            batch_size=args.batch_size,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
