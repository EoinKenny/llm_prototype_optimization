"""Fast static checks that do not train models or call external services."""
from __future__ import annotations

import ast
from pathlib import Path

from src.config import (
    CANDIDATES_PER_LLM,
    CANDIDATES_RETAINED,
    DATASETS,
    DATASET_SETTINGS,
    NEAREST_NEIGHBOR_POOL,
    NEIGHBORS_PER_PROMPT,
    NUM_PARALLEL_LLMS,
    QUALITATIVE_EXAMPLES_PER_CONFIG,
    QUAL_MODELS,
    QUANT_MODELS,
    SEEDS,
    HARDWARE,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPECTED_ROOT_PYTHON = {
    "reproduce_paper.py",
    "run0_prepare_data.py",
    "run1_collect_tokens_and_embeddings.py",
    "run2_train_models.py",
    "run3_optimize_prototypes.py",
    "run4_analyze_quantitative.py",
    "run5_build_qualitative_prompts.py",
    "run6_llm_as_judge.py",
    "run7_analyze_qualitative.py",
}


def main() -> None:
    root_python = {path.name for path in ROOT.glob("*.py")}
    assert root_python == EXPECTED_ROOT_PYTHON, (
        "The repository root should contain only the numbered run scripts and "
        f"reproduce_paper.py. Found: {sorted(root_python)}"
    )

    python_files = sorted(ROOT.glob("*.py")) + sorted(SRC.glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert SEEDS == [0, 1, 2]
    assert len(QUANT_MODELS) == 5
    assert len(QUAL_MODELS) == 3
    assert len(DATASETS) == 6
    assert NUM_PARALLEL_LLMS == 3
    assert CANDIDATES_RETAINED == 10
    assert CANDIDATES_PER_LLM == 10
    assert NEAREST_NEIGHBOR_POOL == 20
    assert NEIGHBORS_PER_PROMPT == 2
    assert all(
        DATASET_SETTINGS[dataset]["iterations"] == 20
        for dataset in ["imdb", "amazon_reviews", "agnews"]
    )
    assert all(
        DATASET_SETTINGS[dataset]["iterations"] == 15
        for dataset in ["trec", "dbpedia", "20newsgroups"]
    )
    expected_prompts = (
        len(DATASETS)
        * len(QUAL_MODELS)
        * len(SEEDS)
        * QUALITATIVE_EXAMPLES_PER_CONFIG
    )
    assert expected_prompts == 2700

    forbidden = [
        "Fill in",
        "DATASETST",
        "claude3.7-sonnet",
        "Llama-3.2-3B-Instruct",
        "from config import",
        "from functions import",
        "from models import",
        "from prompts import",
        "from judge_clients import",
    ]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in python_files
        if path.resolve() != Path(__file__).resolve()
    )
    for token in forbidden:
        assert token not in source, f"Found stale or non-package token: {token}"

    print(
        f"Static preflight passed for {len(python_files)} Python files; "
        "root layout and paper configuration are consistent."
    )
    print(HARDWARE.describe())


if __name__ == "__main__":
    main()
