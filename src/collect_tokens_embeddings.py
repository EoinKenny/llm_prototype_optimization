"""Warm the tokenizer cache for one dataset/backbone pair."""
from __future__ import annotations

import argparse

from src.config import INPUT_LENGTH, PROTOTYPE_DIM
from src.functions import load_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    domain = load_domain(
        dataset=args.dataset,
        model_name=args.model,
        device=args.device,
        prototype_dim=PROTOTYPE_DIM,
        input_length=INPUT_LENGTH,
    )
    print(f"Cached {len(domain.train_dataset)} train and {len(domain.test_dataset)} test examples")


if __name__ == "__main__":
    main()
