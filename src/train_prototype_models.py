"""Train one learned-prototype classifier using the manuscript configuration."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import (
    CLASSIFIER_DEVICE,
    DATASET_SETTINGS,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    INPUT_LENGTH,
    LAMBDA_CLUSTERING,
    LAMBDA_INTERPRETABILITY,
    LAMBDA_SEPARATION,
    LEARNING_RATE,
    PROTOTYPE_DIM,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
    RESULTS_DIR,
    ensure_directories,
)
from src.functions import (
    checkpoint_path,
    evaluate,
    full_loader,
    initialize_prototypes,
    load_domain,
    save_json,
    seed_everything,
    stratified_indices,
    subset_loader,
)
from src.models import LMProtoNet


def train(args: argparse.Namespace) -> dict:
    ensure_directories()
    seed_everything(args.seed)
    settings = DATASET_SETTINGS[args.dataset]

    domain = load_domain(
        dataset=args.dataset,
        model_name=args.model,
        device=args.device,
        prototype_dim=args.prototype_dim,
        input_length=args.input_length,
        hf_token=args.hf_token,
    )
    train_indices, val_indices = stratified_indices(domain.train_df["label"].to_numpy(), args.seed)
    train_loader = subset_loader(
        domain.train_dataset, train_indices, args.train_batch_size, shuffle=True, seed=args.seed
    )
    val_loader = subset_loader(
        domain.train_dataset, val_indices, args.eval_batch_size, shuffle=False, seed=args.seed
    )
    test_loader = full_loader(domain.test_dataset, args.eval_batch_size)

    initial_prototypes = initialize_prototypes(
        backbone=domain.backbone,
        dataset=domain.train_dataset,
        indices=train_indices,
        labels=domain.train_df["label"].to_numpy(),
        num_labels=domain.num_labels,
        prototypes_per_class=args.prototypes_per_class,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    model = LMProtoNet(
        backbone=domain.backbone,
        num_labels=domain.num_labels,
        num_protos_per_class=args.prototypes_per_class,
        init_prototypes=initial_prototypes,
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        running = {"total": 0.0, "ce": 0.0, "lint": 0.0, "lclst": 0.0, "lsep": 0.0}
        seen = 0
        progress = tqdm(train_loader, desc=f"{args.dataset}/{args.model}/seed{args.seed} epoch {epoch + 1}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(args.device)
            outputs = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
            )
            classification_loss = F.cross_entropy(outputs["logits"], labels)
            total_loss = (
                classification_loss
                + args.lambda_interpretability * outputs["l_interpretability"]
                + args.lambda_clustering * outputs["l_clustering"]
                + args.lambda_separation * outputs["l_separation"]
            )
            total_loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            seen += batch_size
            running["total"] += float(total_loss.item()) * batch_size
            running["ce"] += float(classification_loss.item()) * batch_size
            running["lint"] += float(outputs["l_interpretability"].item()) * batch_size
            running["lclst"] += float(outputs["l_clustering"].item()) * batch_size
            running["lsep"] += float(outputs["l_separation"].item()) * batch_size
            progress.set_postfix(loss=f"{running['total'] / seen:.4f}")

        val_metrics = evaluate(model, val_loader)
        epoch_record = {
            "epoch": epoch + 1,
            **{f"train_{key}": value / seen for key, value in running.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch + 1}; best epoch was {best_epoch}.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "num_labels": domain.num_labels,
            "prototypes_per_class": args.prototypes_per_class,
            "prototype_dim": args.prototype_dim,
            "input_length": args.input_length,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_metrics": test_metrics,
        },
    }
    path = checkpoint_path(args.model, args.dataset, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

    result = {
        "checkpoint": str(path),
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "learned_test_metrics": test_metrics,
        "history": history,
    }
    save_json(RESULTS_DIR / "training" / f"{args.model}_{args.dataset}_seed{args.seed}.json", result)
    print(json.dumps(result | {"history": f"{len(history)} epochs"}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SETTINGS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=CLASSIFIER_DEVICE)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prototypes-per-class", type=int, default=None)
    parser.add_argument("--input-length", type=int, default=INPUT_LENGTH)
    parser.add_argument("--prototype-dim", type=int, default=PROTOTYPE_DIM)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-interpretability", type=float, default=LAMBDA_INTERPRETABILITY)
    parser.add_argument("--lambda-clustering", type=float, default=LAMBDA_CLUSTERING)
    parser.add_argument("--lambda-separation", type=float, default=LAMBDA_SEPARATION)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    settings = DATASET_SETTINGS[parsed.dataset]
    parsed.epochs = settings["epochs"] if parsed.epochs is None else parsed.epochs
    parsed.prototypes_per_class = (
        settings["prototypes_per_class"]
        if parsed.prototypes_per_class is None
        else parsed.prototypes_per_class
    )
    train(parsed)
