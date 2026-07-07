"""Construct the 2,700 randomized qualitative-evaluation prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import (
    DATASETS,
    EVAL_BATCH_SIZE,
    INPUT_LENGTH,
    PROTOTYPE_DIM,
    QUALITATIVE_EXAMPLES_PER_CONFIG,
    QUALITATIVE_MAX_WORDS,
    QUAL_MODELS,
    RESULTS_DIR,
    SEEDS,
    ensure_directories,
    optimization_result_path,
    optimization_tensor_path,
)
from src.functions import (
    checkpoint_path,
    encode_texts,
    instantiate_from_checkpoint,
    load_checkpoint,
    load_domain,
)
from src.prompts import qualitative_prompt


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def truncate_words(text: str, maximum: int) -> str:
    words = str(text).split()
    if len(words) <= maximum:
        return " ".join(words)
    return " ".join(words[:maximum]) + "..."


def load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_configuration(
    dataset: str,
    model_name: str,
    seed: int,
    device: str,
    examples_per_config: int,
) -> list[dict]:
    with optimization_result_path(dataset, model_name, seed).open(encoding="utf-8") as handle:
        optimization = json.load(handle)
    tensors = load_tensor_file(optimization_tensor_path(dataset, model_name, seed))
    llm_prototypes = F.normalize(tensors["llm_generated"], p=2, dim=1)
    nn_prototypes = F.normalize(tensors["nearest_neighbor"], p=2, dim=1)

    checkpoint = load_checkpoint(checkpoint_path(model_name, dataset, seed), map_location="cpu")
    domain = load_domain(
        dataset=dataset,
        model_name=model_name,
        device=device,
        prototype_dim=int(checkpoint["metadata"]["prototype_dim"]),
        input_length=int(checkpoint["metadata"]["input_length"]),
    )
    model = instantiate_from_checkpoint(checkpoint, device=device, backbone=domain.backbone)

    rng = np.random.default_rng(stable_seed("qualitative", dataset, model_name, seed))
    if len(domain.test_df) < examples_per_config:
        raise ValueError(f"{dataset} test split has fewer than {examples_per_config} examples")
    test_indices = rng.choice(len(domain.test_df), size=examples_per_config, replace=False).tolist()
    test_texts = domain.test_df.iloc[test_indices]["text"].astype(str).tolist()
    test_embeddings = F.normalize(encode_texts(model, test_texts, INPUT_LENGTH), p=2, dim=1)

    llm_indices = torch.argmax(test_embeddings @ llm_prototypes.T, dim=1).tolist()
    nn_indices = torch.argmax(test_embeddings @ nn_prototypes.T, dim=1).tolist()
    llm_texts = optimization["llm_texts"]
    nn_texts = optimization["nearest_neighbor_texts"]

    order_rng = random.Random(stable_seed("order", dataset, model_name, seed))
    records: list[dict] = []
    for local_index, test_index in enumerate(test_indices):
        llm_text = llm_texts[llm_indices[local_index]]
        nn_text = nn_texts[nn_indices[local_index]]
        if order_rng.random() < 0.5:
            stage_a_method, stage_a_text = "llm_generated", llm_text
            stage_b_method, stage_b_text = "nearest_neighbor", nn_text
        else:
            stage_a_method, stage_a_text = "nearest_neighbor", nn_text
            stage_b_method, stage_b_text = "llm_generated", llm_text

        test_text = truncate_words(test_texts[local_index], QUALITATIVE_MAX_WORDS)
        stage_a_text = truncate_words(stage_a_text, QUALITATIVE_MAX_WORDS)
        stage_b_text = truncate_words(stage_b_text, QUALITATIVE_MAX_WORDS)
        prompt_id = f"{dataset}__{model_name}__seed{seed}__test{test_index}"
        records.append({
            "prompt_id": prompt_id,
            "dataset": dataset,
            "model": model_name,
            "seed": seed,
            "test_index": int(test_index),
            "test_label": int(domain.test_df.iloc[test_index]["label"]),
            "llm_prototype_index": int(llm_indices[local_index]),
            "nearest_neighbor_prototype_index": int(nn_indices[local_index]),
            "stage_a_method": stage_a_method,
            "stage_b_method": stage_b_method,
            "test_text": test_text,
            "stage_a_text": stage_a_text,
            "stage_b_text": stage_b_text,
            "prompt": qualitative_prompt(dataset, test_text, stage_a_text, stage_b_text),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples-per-config", type=int, default=QUALITATIVE_EXAMPLES_PER_CONFIG)
    parser.add_argument("--output", default=str(RESULTS_DIR / "qualitative" / "prompts.csv"))
    args = parser.parse_args()
    ensure_directories()

    records: list[dict] = []
    for dataset in DATASETS:
        for model_name in QUAL_MODELS:
            for seed in SEEDS:
                print(f"Building prompts for {dataset}/{model_name}/seed{seed}")
                records.extend(
                    build_configuration(
                        dataset=dataset,
                        model_name=model_name,
                        seed=seed,
                        device=args.device,
                        examples_per_config=args.examples_per_config,
                    )
                )

    frame = pd.DataFrame(records)
    expected = len(DATASETS) * len(QUAL_MODELS) * len(SEEDS) * args.examples_per_config
    if len(frame) != expected:
        raise RuntimeError(f"Generated {len(frame)} prompts; expected {expected}")
    if frame["prompt_id"].duplicated().any():
        raise RuntimeError("Duplicate prompt IDs were generated")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} prompts to {output}")


if __name__ == "__main__":
    main()
