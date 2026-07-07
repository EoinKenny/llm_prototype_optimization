"""Parse judge outputs and reproduce the qualitative metrics and tests."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel

from src.config import JUDGES, RESULTS_DIR


REQUIRED_KEYS = {
    "stage_a_concepts_count",
    "stage_b_concepts_count",
    "stage_a_concepts_in_test",
    "stage_b_concepts_in_test",
    "stage_a_irrelevant_features",
    "stage_b_irrelevant_features",
    "most_similar_prototype",
    "confidence",
}


def extract_json_object(text: str) -> dict | None:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + [text]
    for candidate in candidates:
        starts = [index for index, char in enumerate(candidate) if char == "{"]
        for start in reversed(starts):
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(candidate)):
                char = candidate[index]
                if escaped:
                    escaped = False
                    continue
                if char == "\\" and in_string:
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(candidate[start : index + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(value, dict):
                            return value
                        break
    return None


def validate(value: dict | None) -> dict | None:
    if value is None or not REQUIRED_KEYS.issubset(value):
        return None
    integer_keys = [
        "stage_a_concepts_count", "stage_b_concepts_count",
        "stage_a_concepts_in_test", "stage_b_concepts_in_test",
        "stage_a_irrelevant_features", "stage_b_irrelevant_features",
    ]
    try:
        for key in integer_keys:
            value[key] = int(value[key])
        value["confidence"] = float(value["confidence"])
    except (TypeError, ValueError):
        return None
    if any(value[key] < 0 for key in integer_keys):
        return None
    if value["stage_a_concepts_in_test"] > value["stage_a_concepts_count"]:
        return None
    if value["stage_b_concepts_in_test"] > value["stage_b_concepts_count"]:
        return None
    if value["stage_a_irrelevant_features"] > value["stage_a_concepts_count"]:
        return None
    if value["stage_b_irrelevant_features"] > value["stage_b_concepts_count"]:
        return None
    selected = str(value["most_similar_prototype"]).lower().strip().replace(" ", "_")
    if selected not in {"stage_a", "stage_b"}:
        return None
    value["most_similar_prototype"] = selected
    return value


def load_judgments(directory: Path) -> pd.DataFrame:
    records: list[dict] = []
    for judge in JUDGES:
        path = directory / f"{judge}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing judge output: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "ok":
                    records.append(row)
    return pd.DataFrame(records)


def sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("nan")


def cohen_dz(a: np.ndarray, b: np.ndarray) -> float:
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(differences.mean() / differences.std(ddof=1))


def parse_all(prompts: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    merged = judgments.merge(prompts, on="prompt_id", how="left", validate="many_to_one")
    rows: list[dict] = []
    for record in merged.itertuples(index=False):
        parsed = validate(extract_json_object(record.response))
        if parsed is None:
            continue
        base = {
            "prompt_id": record.prompt_id,
            "judge": record.judge,
            "dataset": record.dataset,
            "model": record.model,
            "seed": int(record.seed),
        }
        stage_data = {
            "stage_a": {
                "method": record.stage_a_method,
                "c": parsed["stage_a_concepts_count"],
                "t": parsed["stage_a_concepts_in_test"],
                "r": parsed["stage_a_irrelevant_features"],
            },
            "stage_b": {
                "method": record.stage_b_method,
                "c": parsed["stage_b_concepts_count"],
                "t": parsed["stage_b_concepts_in_test"],
                "r": parsed["stage_b_irrelevant_features"],
            },
        }
        preferred_method = stage_data[parsed["most_similar_prototype"]]["method"]
        for stage, values in stage_data.items():
            c, t, r = values["c"], values["t"], values["r"]
            rows.append(base | {
                "stage": stage,
                "method": values["method"],
                "c": c,
                "t": t,
                "r": r,
                "size": float(c),
                "precision": float(t / c) if c else np.nan,
                "irrelevant_rate": float(r / c) if c else np.nan,
                "normalized_purity": float((t - r) / c) if c else np.nan,
                "ideal": float(t == c and r == 0),
                "holistic_preferred": float(values["method"] == preferred_method),
            })
    return pd.DataFrame(rows)


def judge_level_means(parsed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["size", "precision", "irrelevant_rate", "normalized_purity", "ideal"]
    # Equal weight to every dataset/seed/backbone cell, then average cells within each judge.
    cell = parsed.groupby(["judge", "method", "dataset", "seed", "model"], as_index=False)[metrics].mean()
    return cell.groupby(["judge", "method"], as_index=False)[metrics].mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(RESULTS_DIR / "qualitative" / "prompts.csv"))
    parser.add_argument("--judgments", default=str(RESULTS_DIR / "qualitative" / "judgments"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "analysis"))
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    prompts = pd.read_csv(args.prompts)
    judgments = load_judgments(Path(args.judgments))
    parsed = parse_all(prompts, judgments)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(output / "qualitative_parsed_long.csv", index=False)

    total_attempts = len(prompts) * len(JUDGES)
    successful_attempts = parsed["prompt_id"].count() // 2
    judge_means = judge_level_means(parsed)
    judge_means.to_csv(output / "qualitative_judge_means.csv", index=False)

    metrics = ["size", "precision", "normalized_purity", "irrelevant_rate", "ideal"]
    rows: list[dict] = []
    tests: list[dict] = []
    for metric in metrics:
        pivot = judge_means.pivot(index="judge", columns="method", values=metric).loc[list(JUDGES)]
        llm_values = pivot["llm_generated"].to_numpy()
        nn_values = pivot["nearest_neighbor"].to_numpy()
        statistic, p_value = ttest_rel(llm_values, nn_values)
        rows.append({
            "metric": metric,
            "llm_mean": float(llm_values.mean() * (100 if metric == "ideal" else 1)),
            "llm_sem": sem(llm_values) * (100 if metric == "ideal" else 1),
            "nearest_mean": float(nn_values.mean() * (100 if metric == "ideal" else 1)),
            "nearest_sem": sem(nn_values) * (100 if metric == "ideal" else 1),
        })
        tests.append({
            "metric": metric,
            "t": float(statistic),
            "p": float(p_value),
            "cohen_dz": cohen_dz(llm_values, nn_values),
        })

    pd.DataFrame(rows).to_csv(output / "qualitative_table.csv", index=False)
    pd.DataFrame(tests).to_csv(output / "qualitative_paired_tests.csv", index=False)

    wide = parsed.pivot_table(
        index=["prompt_id", "judge", "dataset", "model", "seed"],
        columns="method",
        values=["c", "t", "r", "normalized_purity", "holistic_preferred"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["matching_llm"] = wide["t_llm_generated"] - wide["r_llm_generated"]
    wide["matching_nn"] = wide["t_nearest_neighbor"] - wide["r_nearest_neighbor"]
    maximum_c = wide[["c_llm_generated", "c_nearest_neighbor"]].max(axis=1)
    wide["pp_llm"] = wide["normalized_purity_llm_generated"] - args.beta * wide["c_llm_generated"] / maximum_c
    wide["pp_nn"] = wide["normalized_purity_nearest_neighbor"] - args.beta * wide["c_nearest_neighbor"] / maximum_c

    def preference(left: pd.Series, right: pd.Series) -> pd.Series:
        return np.where(left > right, 1.0, np.where(left < right, 0.0, np.nan))

    wide["matching_prefers_llm"] = preference(wide["matching_llm"], wide["matching_nn"])
    wide["pp_prefers_llm"] = preference(wide["pp_llm"], wide["pp_nn"])
    wide["holistic_prefers_llm"] = wide["holistic_preferred_llm_generated"]
    wide.to_csv(output / "qualitative_pairwise_rules.csv", index=False)

    judge_preferences = wide.groupby("judge")[[
        "matching_prefers_llm", "pp_prefers_llm", "holistic_prefers_llm"
    ]].mean()
    preference_tests = []
    for column in judge_preferences.columns:
        statistic, p_value = ttest_1samp(judge_preferences[column].dropna(), popmean=0.5)
        preference_tests.append({
            "comparison": column,
            "mean_percent": float(judge_preferences[column].mean() * 100),
            "sem_percent": sem(judge_preferences[column]) * 100,
            "t": float(statistic),
            "p": float(p_value),
        })
    pd.DataFrame(preference_tests).to_csv(output / "qualitative_preference_tests.csv", index=False)

    dataset_metrics = parsed.groupby(["judge", "dataset", "method"], as_index=False)[
        ["normalized_purity", "ideal"]
    ].mean().groupby(["dataset", "method"], as_index=False).agg(
        normalized_purity_mean=("normalized_purity", "mean"),
        normalized_purity_sem=("normalized_purity", sem),
        ideal_mean=("ideal", "mean"),
        ideal_sem=("ideal", sem),
    )
    dataset_metrics.to_csv(output / "qualitative_by_dataset.csv", index=False)

    summary = {
        "attempts": total_attempts,
        "parsed_successfully": int(successful_attempts),
        "excluded": int(total_attempts - successful_attempts),
        "parse_rate": float(successful_attempts / total_attempts),
    }
    with (output / "qualitative_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
