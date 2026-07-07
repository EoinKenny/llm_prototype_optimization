"""Paper-aligned experimental configuration.

The values in this file mirror the settings reported in the manuscript.
Environment variables may override hardware-specific settings only.
"""
from __future__ import annotations

from pathlib import Path

from src.devices import build_hardware_plan

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
PREPROCESS_DIR = DATA_DIR / "preprocess"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

DATASETS = [
    "imdb",
    "amazon_reviews",
    "agnews",
    "20newsgroups",
    "trec",
    "dbpedia",
]
QUANT_MODELS = ["bert", "roberta", "electra", "modern_bert", "mpnet"]
QUAL_MODELS = ["bert", "roberta", "electra"]
SEEDS = [0, 1, 2]

MODEL_IDS = {
    "bert": "bert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "electra": "google/electra-base-discriminator",
    "modern_bert": "answerdotai/ModernBERT-base",
    "mpnet": "microsoft/mpnet-base",
}

DATASET_SETTINGS = {
    "imdb": {"epochs": 5, "prototypes_per_class": 3, "iterations": 20, "experiment": "compression"},
    "amazon_reviews": {"epochs": 5, "prototypes_per_class": 3, "iterations": 20, "experiment": "compression"},
    "agnews": {"epochs": 10, "prototypes_per_class": 3, "iterations": 20, "experiment": "compression"},
    "20newsgroups": {"epochs": 20, "prototypes_per_class": 1, "iterations": 15, "experiment": "accuracy"},
    "trec": {"epochs": 100, "prototypes_per_class": 1, "iterations": 15, "experiment": "accuracy"},
    "dbpedia": {"epochs": 15, "prototypes_per_class": 1, "iterations": 15, "experiment": "accuracy"},
}

INPUT_LENGTH = 256
PROTOTYPE_DIM = 256
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_FRACTION = 0.95
EARLY_STOPPING_PATIENCE = 5

LAMBDA_INTERPRETABILITY = 0.1
LAMBDA_CLUSTERING = 0.01
LAMBDA_SEPARATION = 0.01

OPTIMIZER_LLM_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
OPTIMIZER_TEMPERATURE = 1.0
OPTIMIZER_TOP_P = 1.0
NUM_PARALLEL_LLMS = 3
CANDIDATES_RETAINED = 10
CANDIDATES_PER_LLM = 10
NEAREST_NEIGHBOR_POOL = 20
NEIGHBORS_PER_PROMPT = 2

QUALITATIVE_EXAMPLES_PER_CONFIG = 50
QUALITATIVE_MAX_WORDS = 197

DBPEDIA_TARGET_CLASSES = [
    185, 166, 159, 57, 160, 168, 146, 198, 123, 38,
    1, 73, 36, 56, 54, 215, 39, 128, 90, 171,
]

# Hardware is detected automatically. Environment variables remain available
# as optional overrides: PAPER_DEVICES, CLASSIFIER_DEVICE, OPTIMIZER_DEVICES.
HARDWARE = build_hardware_plan(logical_optimizer_copies=NUM_PARALLEL_LLMS)
DEVICES = list(HARDWARE.training_devices)
CLASSIFIER_DEVICE = HARDWARE.classifier_device
OPTIMIZER_DEVICES = list(HARDWARE.optimizer_devices)
# Backward-compatible name retained for older command lines/imports.
OPTIMIZER_GPU_IDS = OPTIMIZER_DEVICES

JUDGES = {
    "claude_sonnet_4_5": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    },
    "gpt_5": {"provider": "openai", "model_id": "gpt-5-2025-08-07"},
    "gpt_5_mini": {"provider": "openai", "model_id": "gpt-5-mini-2025-08-07"},
    "gpt_4o": {"provider": "openai", "model_id": "gpt-4o-2024-11-20"},
    "o4_mini": {"provider": "openai", "model_id": "o4-mini-2025-04-16"},
}


def optimization_result_path(dataset: str, model: str, seed: int) -> Path:
    return RESULTS_DIR / "optimization" / f"{model}_{dataset}_seed{seed}.json"


def optimization_tensor_path(dataset: str, model: str, seed: int) -> Path:
    return RESULTS_DIR / "optimization" / f"{model}_{dataset}_seed{seed}_prototypes.pt"


def ensure_directories() -> None:
    for path in (DATA_DIR, PREPROCESS_DIR, WEIGHTS_DIR, RESULTS_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
