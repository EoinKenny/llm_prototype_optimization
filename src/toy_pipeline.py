"""Fast, self-contained smoke test for the full reproduction pipeline.

Toy mode deliberately avoids dataset downloads, Hugging Face model downloads, local
Llama inference, and paid judge APIs. It still exercises the repository's stage
boundaries and data flow: synthetic preprocessing, token caching, one real PyTorch
optimization step, prototype projection, quantitative analysis, prompt generation,
mock judge output, and qualitative parsing/statistics.

Enable it with ``python reproduce_paper.py --toy`` or run any numbered stage with
``--toy``.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel

from src.config import JUDGES, ROOT
from src.prompts import qualitative_prompt

TOY_ROOT = ROOT / "results" / "toy"
TOY_DATA = ROOT / "datasets" / "preprocess" / "toy_sentiment"
TOY_CACHE = ROOT / "cache" / "toy"
TOY_WEIGHTS = ROOT / "weights" / "toy"
TOY_ANALYSIS = TOY_ROOT / "analysis"
TOY_QUAL = TOY_ROOT / "qualitative"
TOY_JUDGMENTS = TOY_QUAL / "judgments"


def activate_from_argv() -> bool:
    """Enable toy mode when either the environment or ``--toy`` requests it."""
    if "--toy" in sys.argv:
        sys.argv.remove("--toy")
        os.environ["TMLR_TOY"] = "1"
    return os.getenv("TMLR_TOY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _build_vocab(texts: list[str]) -> dict[str, int]:
    tokens = sorted({token for text in texts for token in _tokenize(text)})
    return {"<pad>": 0, "<unk>": 1, **{token: index + 2 for index, token in enumerate(tokens)}}


def _encode(text: str, vocab: dict[str, int], length: int = 16) -> list[int]:
    values = [vocab.get(token, 1) for token in _tokenize(text)][:length]
    return values + [0] * (length - len(values))


class TinyPrototypeNet(nn.Module):
    """Small CPU model used only by the smoke test."""

    def __init__(self, vocab_size: int, embedding_dim: int = 12, latent_dim: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, latent_dim)
        self.prototypes = nn.Parameter(torch.empty(2, latent_dim))
        self.classifier = nn.Linear(2, 2, bias=False)
        nn.init.xavier_uniform_(self.prototypes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def encode_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(input_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(torch.tanh(self.projection(pooled)), p=2, dim=1)

    def forward(self, input_ids: torch.Tensor, prototypes: torch.Tensor | None = None) -> dict:
        representations = self.encode_ids(input_ids)
        active_prototypes = self.prototypes if prototypes is None else prototypes
        normalized_prototypes = F.normalize(active_prototypes, p=2, dim=1)
        activations = representations @ normalized_prototypes.T
        return {
            "representations": representations,
            "activations": activations,
            "logits": self.classifier(activations),
        }


def _load_cache() -> tuple[dict, dict[str, int]]:
    try:
        payload = torch.load(TOY_CACHE / "tokenized.pt", map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(TOY_CACHE / "tokenized.pt", map_location="cpu")
    vocab = _read_json(TOY_CACHE / "vocab.json")
    return payload, {str(key): int(value) for key, value in vocab.items()}


def _load_model() -> tuple[TinyPrototypeNet, dict]:
    try:
        checkpoint = torch.load(TOY_WEIGHTS / "tiny_prototype_model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(TOY_WEIGHTS / "tiny_prototype_model.pt", map_location="cpu")
    model = TinyPrototypeNet(
        vocab_size=int(checkpoint["metadata"]["vocab_size"]),
        embedding_dim=int(checkpoint["metadata"]["embedding_dim"]),
        latent_dim=int(checkpoint["metadata"]["latent_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _encode_texts(model: TinyPrototypeNet, texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    ids = torch.tensor([_encode(text, vocab) for text in texts], dtype=torch.long)
    with torch.no_grad():
        return model.encode_ids(ids)


def _accuracy(model: TinyPrototypeNet, ids: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor | None = None) -> float:
    with torch.no_grad():
        predictions = model(ids, prototypes=prototypes)["logits"].argmax(dim=1)
    return float(predictions.eq(labels).float().mean().item())


def prepare_data() -> None:
    """Create a deterministic two-class text dataset and reset toy outputs."""
    for path in (TOY_ROOT, TOY_CACHE, TOY_WEIGHTS, TOY_DATA):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame([
        ("A delightful movie with warm acting and a joyful ending", 1),
        ("An enjoyable film with clever humor and strong performances", 1),
        ("A charming story that was uplifting and beautifully made", 1),
        ("Excellent acting and an entertaining plot made this memorable", 1),
        ("A tedious movie with weak acting and a confusing ending", 0),
        ("A dull film with poor humor and lifeless performances", 0),
        ("A frustrating story that was bleak and badly made", 0),
        ("Terrible acting and a boring plot made this forgettable", 0),
    ], columns=["text", "label"])
    test = pd.DataFrame([
        ("Warm performances make this an enjoyable movie", 1),
        ("The film is charming, funny, and uplifting", 1),
        ("Weak acting makes this a tedious and boring movie", 0),
        ("A confusing plot and lifeless performances ruin the film", 0),
    ], columns=["text", "label"])
    train.to_csv(TOY_DATA / "train.csv", index=False)
    test.to_csv(TOY_DATA / "test.csv", index=False)
    _write_json(TOY_ROOT / "manifest.json", {
        "mode": "toy_smoke_test",
        "dataset": "toy_sentiment",
        "train_rows": len(train),
        "test_rows": len(test),
        "warning": "Toy outputs are for software validation only and are not paper results.",
    })
    print(f"Toy data prepared: {len(train)} train / {len(test)} test rows")


def cache_data() -> None:
    train = pd.read_csv(TOY_DATA / "train.csv")
    test = pd.read_csv(TOY_DATA / "test.csv")
    vocab = _build_vocab(train["text"].astype(str).tolist())
    payload = {
        "train_ids": torch.tensor([_encode(text, vocab) for text in train["text"]], dtype=torch.long),
        "train_labels": torch.tensor(train["label"].to_numpy(), dtype=torch.long),
        "test_ids": torch.tensor([_encode(text, vocab) for text in test["text"]], dtype=torch.long),
        "test_labels": torch.tensor(test["label"].to_numpy(), dtype=torch.long),
    }
    TOY_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(payload, TOY_CACHE / "tokenized.pt")
    _write_json(TOY_CACHE / "vocab.json", vocab)
    print(f"Toy cache written with vocabulary size {len(vocab)}")


def train_one_batch() -> None:
    """Perform exactly one AdamW update on one four-example batch."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    payload, vocab = _load_cache()
    model = TinyPrototypeNet(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    input_ids = payload["train_ids"][:4]
    labels = payload["train_labels"][:4]
    model.train()
    output = model(input_ids)
    ce = F.cross_entropy(output["logits"], labels)
    similarities = output["representations"] @ F.normalize(model.prototypes, p=2, dim=1).T
    interpretability = (1 - similarities).min(dim=0).values.mean()
    clustering = (1 - similarities).min(dim=1).values.mean()
    proto_similarity = F.normalize(model.prototypes, p=2, dim=1)
    separation = 1 + (proto_similarity @ proto_similarity.T)[0, 1]
    loss = ce + 0.1 * interpretability + 0.01 * clustering + 0.01 * separation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()

    train_accuracy = _accuracy(model, payload["train_ids"], payload["train_labels"])
    test_accuracy = _accuracy(model, payload["test_ids"], payload["test_labels"])
    TOY_WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            "vocab_size": len(vocab),
            "embedding_dim": 12,
            "latent_dim": 8,
            "optimizer_steps": 1,
            "batch_size": 4,
        },
        "metrics": {
            "loss": float(loss.item()),
            "cross_entropy": float(ce.item()),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        },
    }, TOY_WEIGHTS / "tiny_prototype_model.pt")
    print(
        "Toy model trained for one batch: "
        f"loss={loss.item():.4f}, train_acc={train_accuracy:.3f}, test_acc={test_accuracy:.3f}"
    )


def optimize_prototypes() -> None:
    model, checkpoint = _load_model()
    payload, vocab = _load_cache()
    train = pd.read_csv(TOY_DATA / "train.csv")
    train_embeddings = _encode_texts(model, train["text"].astype(str).tolist(), vocab)
    learned = F.normalize(model.prototypes.detach(), p=2, dim=1)

    candidate_groups = [
        [
            "boring film with weak acting",
            "tedious movie with a confusing plot",
            "poor acting and a dull story",
            "lifeless performances ruin the film",
        ],
        [
            "enjoyable film with warm acting",
            "charming uplifting movie",
            "clever humor and strong performances",
            "entertaining story with a joyful ending",
        ],
    ]

    llm_texts: list[str] = []
    nn_texts: list[str] = []
    llm_vectors: list[torch.Tensor] = []
    nn_vectors: list[torch.Tensor] = []
    prototype_results: list[dict] = []

    for index, target in enumerate(learned):
        train_scores = train_embeddings @ target
        nn_index = int(torch.argmax(train_scores).item())
        nn_text = str(train.iloc[nn_index]["text"])
        nn_vector = train_embeddings[nn_index]

        candidates = candidate_groups[index]
        candidate_vectors = _encode_texts(model, candidates, vocab)
        candidate_scores = candidate_vectors @ target
        order = torch.argsort(candidate_scores, descending=True)
        best_index = int(order[0].item())
        llm_text = candidates[best_index]
        llm_vector = candidate_vectors[best_index]
        initial_score = float(candidate_scores.mean().item())
        best_score = float(candidate_scores[best_index].item())

        llm_texts.append(llm_text)
        nn_texts.append(nn_text)
        llm_vectors.append(llm_vector)
        nn_vectors.append(nn_vector)
        prototype_results.append({
            "prototype_index": index,
            "nearest_neighbor_text": nn_text,
            "nearest_neighbor_similarity": float(train_scores[nn_index].item()),
            "optimized_text": llm_text,
            "optimized_similarity": best_score,
            "history": [
                {"iteration": 0, "best_similarity": initial_score, "mean_similarity": initial_score, "best_text": candidates[0]},
                {"iteration": 1, "best_similarity": best_score, "mean_similarity": float(candidate_scores.mean().item()), "best_text": llm_text},
            ],
        })

    llm_tensor = torch.stack(llm_vectors)
    nn_tensor = torch.stack(nn_vectors)
    learned_accuracy = _accuracy(model, payload["test_ids"], payload["test_labels"])
    llm_accuracy = _accuracy(model, payload["test_ids"], payload["test_labels"], llm_tensor)
    nn_accuracy = _accuracy(model, payload["test_ids"], payload["test_labels"], nn_tensor)

    result = {
        "mode": "toy_smoke_test",
        "dataset": "toy_sentiment",
        "model": "tiny_bow",
        "seed": 0,
        "experiment": "compression",
        "learned_metrics": {"accuracy": learned_accuracy},
        "llm_generated_metrics": {"accuracy": llm_accuracy},
        "nearest_neighbor_metrics": {"accuracy": nn_accuracy},
        "avg_llm_similarity": float(np.mean([row["optimized_similarity"] for row in prototype_results])),
        "avg_nearest_similarity": float(np.mean([row["nearest_neighbor_similarity"] for row in prototype_results])),
        "avg_llm_length_characters": float(np.mean([len(text) for text in llm_texts])),
        "avg_nearest_length_characters": float(np.mean([len(text) for text in nn_texts])),
        "llm_texts": llm_texts,
        "nearest_neighbor_texts": nn_texts,
        "prototype_results": prototype_results,
        "training_metrics": checkpoint["metrics"],
    }
    TOY_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(TOY_ROOT / "optimization_result.json", result)
    torch.save({"learned": learned, "llm_generated": llm_tensor, "nearest_neighbor": nn_tensor}, TOY_ROOT / "prototype_tensors.pt")
    print(
        "Toy prototype optimization complete: "
        f"learned={learned_accuracy:.3f}, generated={llm_accuracy:.3f}, nearest={nn_accuracy:.3f}"
    )


def analyze_quantitative() -> None:
    result = _read_json(TOY_ROOT / "optimization_result.json")
    learned = float(result["learned_metrics"]["accuracy"]) * 100
    generated = float(result["llm_generated_metrics"]["accuracy"]) * 100
    nearest = float(result["nearest_neighbor_metrics"]["accuracy"]) * 100
    frame = pd.DataFrame([{
        "dataset": result["dataset"],
        "model": result["model"],
        "seed": result["seed"],
        "learned_accuracy": learned,
        "llm_accuracy": generated,
        "nearest_accuracy": nearest,
        "llm_delta": generated - learned,
        "nearest_delta": nearest - learned,
        "llm_similarity": result["avg_llm_similarity"],
        "nearest_similarity": result["avg_nearest_similarity"],
        "llm_length": result["avg_llm_length_characters"],
        "nearest_length": result["avg_nearest_length_characters"],
    }])
    TOY_ANALYSIS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TOY_ANALYSIS / "quantitative_raw.csv", index=False)

    figure = plt.figure(figsize=(5, 3.5))
    for prototype in result["prototype_results"]:
        values = [point["best_similarity"] for point in prototype["history"]]
        plt.plot(range(len(values)), values, marker="o", label=f"prototype {prototype['prototype_index']}")
    plt.xlabel("Toy optimization iteration")
    plt.ylabel("Cosine similarity")
    plt.legend()
    plt.tight_layout()
    figure.savefig(TOY_ANALYSIS / "optimization_curves_toy.pdf", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "mode": "toy_smoke_test",
        "learned_accuracy": learned,
        "llm_generated_accuracy": generated,
        "nearest_neighbor_accuracy": nearest,
        "llm_mean_length": result["avg_llm_length_characters"],
        "nearest_mean_length": result["avg_nearest_length_characters"],
    }
    _write_json(TOY_ANALYSIS / "quantitative_summary.json", summary)
    print(json.dumps(summary, indent=2))


def build_qualitative_prompts() -> None:
    model, _ = _load_model()
    _, vocab = _load_cache()
    test = pd.read_csv(TOY_DATA / "test.csv")
    result = _read_json(TOY_ROOT / "optimization_result.json")
    try:
        tensors = torch.load(TOY_ROOT / "prototype_tensors.pt", map_location="cpu", weights_only=True)
    except TypeError:
        tensors = torch.load(TOY_ROOT / "prototype_tensors.pt", map_location="cpu")
    test_vectors = _encode_texts(model, test["text"].astype(str).tolist(), vocab)
    llm_indices = torch.argmax(test_vectors @ F.normalize(tensors["llm_generated"], p=2, dim=1).T, dim=1)
    nn_indices = torch.argmax(test_vectors @ F.normalize(tensors["nearest_neighbor"], p=2, dim=1).T, dim=1)

    rng = random.Random(0)
    records = []
    for index, row in test.iterrows():
        llm_text = result["llm_texts"][int(llm_indices[index].item())]
        nn_text = result["nearest_neighbor_texts"][int(nn_indices[index].item())]
        if rng.random() < 0.5:
            stage_a_method, stage_a_text = "llm_generated", llm_text
            stage_b_method, stage_b_text = "nearest_neighbor", nn_text
        else:
            stage_a_method, stage_a_text = "nearest_neighbor", nn_text
            stage_b_method, stage_b_text = "llm_generated", llm_text
        records.append({
            "prompt_id": f"toy__test{index}",
            "dataset": "toy_sentiment",
            "model": "tiny_bow",
            "seed": 0,
            "test_index": int(index),
            "test_label": int(row["label"]),
            "stage_a_method": stage_a_method,
            "stage_b_method": stage_b_method,
            "test_text": row["text"],
            "stage_a_text": stage_a_text,
            "stage_b_text": stage_b_text,
            "prompt": qualitative_prompt("toy_sentiment", row["text"], stage_a_text, stage_b_text),
        })
    TOY_QUAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(TOY_QUAL / "prompts.csv", index=False)
    print(f"Wrote {len(records)} toy qualitative prompts")


def _mock_counts(method: str, judge_index: int, prompt_index: int) -> tuple[int, int, int]:
    # Deliberately vary judge-level means so the smoke-test paired statistics
    # remain finite rather than collapsing to identical differences.
    if method == "llm_generated":
        c = 3 + ((judge_index + prompt_index) % 3)
        ideal = (judge_index + 2 * prompt_index) % 5 == 0
        t = c if ideal else max(1, c - 1 - (judge_index % 2))
        r = 0 if ideal else (judge_index + prompt_index) % 2
    else:
        c = 5 + ((2 * judge_index + prompt_index) % 3)
        t = min(c, 2 + ((judge_index + prompt_index) % 2))
        r = min(c, 2 + ((judge_index + 2 * prompt_index) % 2))
    return c, t, r


def run_mock_judges() -> None:
    prompts = pd.read_csv(TOY_QUAL / "prompts.csv")
    TOY_JUDGMENTS.mkdir(parents=True, exist_ok=True)
    for judge_index, judge_name in enumerate(JUDGES):
        path = TOY_JUDGMENTS / f"{judge_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for prompt_index, row in prompts.iterrows():
                a = _mock_counts(str(row["stage_a_method"]), judge_index, prompt_index)
                b = _mock_counts(str(row["stage_b_method"]), judge_index, prompt_index)
                preferred = "stage_a" if row["stage_a_method"] == "llm_generated" else "stage_b"
                response_payload = {
                    "stage_a_concepts_count": a[0],
                    "stage_b_concepts_count": b[0],
                    "stage_a_concepts_in_test": a[1],
                    "stage_b_concepts_in_test": b[1],
                    "stage_a_irrelevant_features": a[2],
                    "stage_b_irrelevant_features": b[2],
                    "most_similar_prototype": preferred,
                    "confidence": 0.75 + 0.03 * (judge_index % 3),
                }
                record = {
                    "prompt_id": row["prompt_id"],
                    "judge": judge_name,
                    "model_id": f"mock/{judge_name}",
                    "status": "ok",
                    "response": "Toy reasoning omitted.\n```json\n" + json.dumps(response_payload) + "\n```",
                }
                handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(prompts) * len(JUDGES)} mock judge responses across {len(JUDGES)} judges")


def analyze_qualitative() -> None:
    from src.analyze_qualitative import cohen_dz, judge_level_means, parse_all, sem

    prompts = pd.read_csv(TOY_QUAL / "prompts.csv")
    records = []
    for judge_name in JUDGES:
        with (TOY_JUDGMENTS / f"{judge_name}.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    judgments = pd.DataFrame(records)
    parsed = parse_all(prompts, judgments)
    if len(parsed) != len(prompts) * len(JUDGES) * 2:
        raise RuntimeError("Toy qualitative parser did not recover every mock response")

    TOY_ANALYSIS.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(TOY_ANALYSIS / "qualitative_parsed_long.csv", index=False)
    judge_means = judge_level_means(parsed)
    judge_means.to_csv(TOY_ANALYSIS / "qualitative_judge_means.csv", index=False)

    rows = []
    tests = []
    for metric in ["size", "precision", "normalized_purity", "irrelevant_rate", "ideal"]:
        pivot = judge_means.pivot(index="judge", columns="method", values=metric).loc[list(JUDGES)]
        llm_values = pivot["llm_generated"].to_numpy(dtype=float)
        nn_values = pivot["nearest_neighbor"].to_numpy(dtype=float)
        statistic, p_value = ttest_rel(llm_values, nn_values)
        multiplier = 100 if metric == "ideal" else 1
        rows.append({
            "metric": metric,
            "llm_mean": float(llm_values.mean() * multiplier),
            "llm_sem": float(sem(llm_values) * multiplier),
            "nearest_mean": float(nn_values.mean() * multiplier),
            "nearest_sem": float(sem(nn_values) * multiplier),
        })
        difference_std = np.std(llm_values - nn_values, ddof=1)
        effect = float(cohen_dz(llm_values, nn_values)) if difference_std > 0 else float("inf")
        tests.append({"metric": metric, "t": float(statistic), "p": float(p_value), "cohen_dz": effect})
    pd.DataFrame(rows).to_csv(TOY_ANALYSIS / "qualitative_table.csv", index=False)
    pd.DataFrame(tests).to_csv(TOY_ANALYSIS / "qualitative_paired_tests.csv", index=False)

    attempts = len(prompts) * len(JUDGES)
    summary = {
        "mode": "toy_smoke_test",
        "attempts": attempts,
        "parsed_successfully": attempts,
        "excluded": 0,
        "parse_rate": 1.0,
    }
    _write_json(TOY_ANALYSIS / "qualitative_summary.json", summary)
    print(json.dumps(summary, indent=2))


STAGES: dict[str, Callable[[], None]] = {
    "prepare_data": prepare_data,
    "cache": cache_data,
    "train": train_one_batch,
    "optimize": optimize_prototypes,
    "quantitative_analysis": analyze_quantitative,
    "qualitative_prompts": build_qualitative_prompts,
    "judges": run_mock_judges,
    "qualitative_analysis": analyze_qualitative,
}


def run_stage(name: str) -> None:
    try:
        stage = STAGES[name]
    except KeyError as error:
        raise ValueError(f"Unknown toy stage: {name}") from error
    print(f"[toy mode] {name}")
    stage()
