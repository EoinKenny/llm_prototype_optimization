"""Aggregate the optimization runs into the quantitative tables and curves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DATASETS, DATASET_SETTINGS, QUANT_MODELS, RESULTS_DIR, SEEDS,
    optimization_result_path,
)


def sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if len(array) <= 1:
        return float("nan")
    return float(array.std(ddof=1) / math.sqrt(len(array)))


def load_runs() -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    payloads: list[dict] = []
    for dataset in DATASETS:
        for model in QUANT_MODELS:
            for seed in SEEDS:
                path = optimization_result_path(dataset, model, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Missing optimization result: {path}")
                with path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                payloads.append(result)
                learned = result["learned_metrics"]["accuracy"] * 100.0
                llm = result["llm_generated_metrics"]["accuracy"] * 100.0
                nearest = result["nearest_neighbor_metrics"]["accuracy"] * 100.0
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "experiment": result["experiment"],
                    "learned_accuracy": learned,
                    "llm_accuracy": llm,
                    "nearest_accuracy": nearest,
                    "llm_delta": llm - learned,
                    "nearest_delta": nearest - learned,
                    "llm_similarity": result["avg_llm_similarity"],
                    "nearest_similarity": result["avg_nearest_similarity"],
                    "llm_length": result["avg_llm_length_characters"],
                    "nearest_length": result["avg_nearest_length_characters"],
                })
    return pd.DataFrame(rows), payloads


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "learned_accuracy", "llm_accuracy", "nearest_accuracy", "llm_delta",
        "nearest_delta", "llm_similarity", "nearest_similarity", "llm_length",
        "nearest_length",
    ]
    model_rows: list[dict] = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=False):
        row = {"dataset": dataset, "model": model, "experiment": group["experiment"].iloc[0]}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sem"] = sem(group[metric])
        model_rows.append(row)
    by_model = pd.DataFrame(model_rows)

    dataset_rows: list[dict] = []
    for dataset, group in by_model.groupby("dataset", sort=False):
        row = {"dataset": dataset, "experiment": group["experiment"].iloc[0]}
        for metric in metrics:
            values = group[f"{metric}_mean"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = sem(values)
        dataset_rows.append(row)
    by_dataset = pd.DataFrame(dataset_rows)
    return by_model, by_dataset


def plot_curves(payloads: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in ("compression", "accuracy"):
        datasets = [dataset for dataset, settings in DATASET_SETTINGS.items() if settings["experiment"] == experiment]
        figure, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
        for axis, dataset in zip(axes[0], datasets):
            seed_curves = []
            seed_baselines = []
            for seed in SEEDS:
                relevant = [
                    payload for payload in payloads
                    if payload["dataset"] == dataset and int(payload["seed"]) == seed
                ]
                per_prototype = []
                nearest = []
                for payload in relevant:
                    for prototype in payload["prototype_results"]:
                        per_prototype.append([
                            point["best_similarity"] for point in prototype["history"]
                        ])
                        nearest.append(prototype["nearest_neighbor_similarity"])
                seed_curves.append(np.asarray(per_prototype, dtype=float).mean(axis=0))
                seed_baselines.append(float(np.mean(nearest)))

            matrix = np.asarray(seed_curves, dtype=float)
            x = np.arange(matrix.shape[1])
            mean = matrix.mean(axis=0)
            error = matrix.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
            axis.plot(x, mean, label="LLM-generated")
            axis.fill_between(x, mean - error, mean + error, alpha=0.2)
            axis.axhline(float(np.mean(seed_baselines)), linestyle="--", label="Nearest neighbor")
            axis.set_title(dataset)
            axis.set_xlabel("Optimization iteration")
            axis.set_ylabel("Cosine similarity")
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"optimization_curves_{experiment}.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "analysis"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw, payloads = load_runs()
    by_model, by_dataset = aggregate(raw)
    raw.to_csv(output / "quantitative_raw.csv", index=False)
    by_model.to_csv(output / "quantitative_by_model.csv", index=False)
    by_dataset.to_csv(output / "quantitative_by_dataset.csv", index=False)
    plot_curves(payloads, output)

    compression = by_dataset[by_dataset["experiment"] == "compression"]
    accuracy = by_dataset[by_dataset["experiment"] == "accuracy"]
    summary = {
        "compression": {
            "llm_mean_length": float(compression["llm_length_mean"].mean()),
            "nearest_mean_length": float(compression["nearest_length_mean"].mean()),
            "llm_mean_delta": float(compression["llm_delta_mean"].mean()),
            "nearest_mean_delta": float(compression["nearest_delta_mean"].mean()),
        },
        "accuracy": {
            "llm_mean_delta": float(accuracy["llm_delta_mean"].mean()),
            "nearest_mean_delta": float(accuracy["nearest_delta_mean"].mean()),
            "llm_mean_similarity": float(accuracy["llm_similarity_mean"].mean()),
            "nearest_mean_similarity": float(accuracy["nearest_similarity_mean"].mean()),
        },
    }
    with (output / "quantitative_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
