"""Shared data, evaluation, checkpoint, and embedding utilities."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset
from tqdm import tqdm

from src.config import CACHE_DIR, INPUT_LENGTH, PREPROCESS_DIR, TRAIN_FRACTION
from src.models import LMProtoNet, ModelWrapper


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TokenizedDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: Sequence[int]) -> None:
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if len(self.labels) != len(next(iter(encodings.values()))):
            raise ValueError("Token encodings and labels have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: tensor[index] for name, tensor in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


@dataclass
class DomainData:
    backbone: ModelWrapper
    tokenizer: object
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_dataset: TokenizedDataset
    test_dataset: TokenizedDataset
    num_labels: int


def _load_token_cache(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tokenize_and_cache(
    texts: list[str], tokenizer: object, path: Path, input_length: int
) -> dict[str, torch.Tensor]:
    cached = _load_token_cache(path)
    if cached is not None:
        return cached
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=input_length,
        return_tensors="pt",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(encodings), path)
    return dict(encodings)


def load_domain(
    dataset: str,
    model_name: str,
    device: str,
    prototype_dim: int,
    input_length: int = INPUT_LENGTH,
    hf_token: str | None = None,
) -> DomainData:
    train_path = PREPROCESS_DIR / dataset / "train.csv"
    test_path = PREPROCESS_DIR / dataset / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path} or {test_path}. Run `python run0_prepare_data.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for frame_name, frame in (("train", train_df), ("test", test_df)):
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{frame_name} data for {dataset} must contain text and label columns")
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["label"] = frame["label"].astype(int)

    labels = sorted(train_df["label"].unique().tolist())
    if labels != list(range(len(labels))):
        raise ValueError(f"Training labels for {dataset} are not contiguous from zero: {labels}")

    backbone = ModelWrapper(
        model_name=model_name,
        device=device,
        prototype_dim=prototype_dim,
        hf_token=hf_token,
    )
    cache_prefix = CACHE_DIR / "tokens" / dataset
    train_tokens = _tokenize_and_cache(
        train_df["text"].tolist(),
        backbone.tokenizer,
        cache_prefix / f"train_{model_name}_len{input_length}.pt",
        input_length,
    )
    test_tokens = _tokenize_and_cache(
        test_df["text"].tolist(),
        backbone.tokenizer,
        cache_prefix / f"test_{model_name}_len{input_length}.pt",
        input_length,
    )

    return DomainData(
        backbone=backbone,
        tokenizer=backbone.tokenizer,
        train_df=train_df,
        test_df=test_df,
        train_dataset=TokenizedDataset(train_tokens, train_df["label"].to_numpy()),
        test_dataset=TokenizedDataset(test_tokens, test_df["label"].to_numpy()),
        num_labels=len(labels),
    )


def stratified_indices(labels: Sequence[int], seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=TRAIN_FRACTION,
        random_state=seed,
        stratify=np.asarray(labels),
    )
    return train_indices.tolist(), val_indices.tolist()


def subset_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def full_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    model_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = batch["labels"].to(device)
    return model_inputs, labels


@torch.no_grad()
def evaluate(model: LMProtoNet, loader: DataLoader) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels_all: list[int] = []
    for batch in loader:
        model_inputs, labels = _move_batch(batch, model.backbone.device)
        logits = model(**model_inputs)["logits"]
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels_all, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


@torch.no_grad()
def encode_loader(model: LMProtoNet, loader: DataLoader) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Encoding", leave=False):
        model_inputs, _ = _move_batch(batch, model.backbone.device)
        outputs.append(model(**model_inputs)["representation"].cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_texts(
    model: LMProtoNet,
    texts: Sequence[str],
    input_length: int,
    batch_size: int = 32,
) -> torch.Tensor:
    model.eval()
    all_embeddings: list[torch.Tensor] = []
    texts = [str(text) for text in texts]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = model.backbone.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=input_length,
            return_tensors="pt",
        )
        representations = model.backbone(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        all_embeddings.append(F.normalize(representations, p=2, dim=1).cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def initialize_prototypes(
    backbone: ModelWrapper,
    dataset: Dataset,
    indices: Sequence[int],
    labels: Sequence[int],
    num_labels: int,
    prototypes_per_class: int,
    batch_size: int,
    seed: int,
    max_examples: int = 6400,
) -> torch.Tensor:
    selected_indices = list(indices)[:max_examples]
    loader = subset_loader(dataset, selected_indices, batch_size, shuffle=False, seed=seed)
    embeddings: list[torch.Tensor] = []
    backbone.eval()
    for batch in tqdm(loader, desc="Initializing prototypes", leave=False):
        embeddings.append(
            F.normalize(
                backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ),
                p=2,
                dim=1,
            ).cpu()
        )
    matrix = torch.cat(embeddings, dim=0)
    selected_labels = np.asarray(labels)[selected_indices]

    centroids: list[torch.Tensor] = []
    for class_index in range(num_labels):
        class_matrix = matrix[selected_labels == class_index]
        if len(class_matrix) < prototypes_per_class:
            raise ValueError(
                f"Class {class_index} has only {len(class_matrix)} initialization examples, "
                f"fewer than {prototypes_per_class} prototypes"
            )
        kmeans = KMeans(n_clusters=prototypes_per_class, n_init=10, random_state=seed)
        kmeans.fit(class_matrix.numpy())
        centroids.append(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
    return F.normalize(torch.cat(centroids, dim=0), p=2, dim=1)


def checkpoint_path(model: str, dataset: str, seed: int) -> Path:
    from src.config import WEIGHTS_DIR

    return WEIGHTS_DIR / f"learned_{model}_{dataset}_seed{seed}.pt"


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def instantiate_from_checkpoint(
    checkpoint: dict,
    device: str,
    hf_token: str | None = None,
    backbone: ModelWrapper | None = None,
) -> LMProtoNet:
    metadata = checkpoint["metadata"]
    if backbone is None:
        backbone = ModelWrapper(
            model_name=metadata["model"],
            device=device,
            prototype_dim=int(metadata["prototype_dim"]),
            hf_token=hf_token,
        )
    model = LMProtoNet(
        backbone=backbone,
        num_labels=int(metadata["num_labels"]),
        num_protos_per_class=int(metadata["prototypes_per_class"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model
