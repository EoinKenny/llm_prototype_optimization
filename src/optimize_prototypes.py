"""Optimize textual preimages for every learned prototype.

This implementation follows Section 3.2 and Appendix H:
- Meta-Llama-3-8B-Instruct
- three parallel LLM copies
- 10 retained candidates
- 10 proposals per LLM and iteration
- nearest-neighbor pool q=20
- two sampled neighbors per prompt
- 20 iterations for compression datasets and 15 for accuracy datasets
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import transformers

from src.config import (
    CANDIDATES_PER_LLM,
    CANDIDATES_RETAINED,
    DATASETS,
    DATASET_SETTINGS,
    EVAL_BATCH_SIZE,
    INPUT_LENGTH,
    NEAREST_NEIGHBOR_POOL,
    NEIGHBORS_PER_PROMPT,
    NUM_PARALLEL_LLMS,
    OPTIMIZER_GPU_IDS,
    OPTIMIZER_LLM_ID,
    OPTIMIZER_TEMPERATURE,
    OPTIMIZER_TOP_P,
    PROTOTYPE_DIM,
    QUANT_MODELS,
    RESULTS_DIR,
    optimization_result_path,
    optimization_tensor_path,
    SEEDS,
    ensure_directories,
)
from src.functions import (
    checkpoint_path,
    encode_loader,
    encode_texts,
    evaluate,
    instantiate_from_checkpoint,
    load_checkpoint,
    load_domain,
    save_json,
    seed_everything,
    subset_loader,
    full_loader,
)
from src.prompts import initialization_prompt, optimization_prompt



def result_path(dataset: str, model: str, seed: int) -> Path:
    return optimization_result_path(dataset, model, seed)


def prototype_tensor_path(dataset: str, model: str, seed: int) -> Path:
    return optimization_tensor_path(dataset, model, seed)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def extract_python_list(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [index for index, character in enumerate(text) if character == "["]
    for start in reversed(starts):
        end = text.find("]", start)
        while end != -1:
            try:
                value = ast.literal_eval(text[start : end + 1])
            except (ValueError, SyntaxError):
                end = text.find("]", end + 1)
                continue
            if isinstance(value, list):
                for item in value:
                    item = str(item).strip()
                    if item:
                        candidates.append(item)
                return candidates
            end = text.find("]", end + 1)
    return candidates


def unique_texts(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


class ParallelOptimizerLLMs:
    def __init__(
        self,
        model_id: str,
        gpu_ids: Sequence[str],
        copies: int,
        hf_token: str | None,
    ) -> None:
        if copies < 1:
            raise ValueError("At least one optimizer LLM copy is required")
        if len(gpu_ids) < copies:
            raise ValueError(f"Need {copies} GPU IDs, received {gpu_ids}")
        self.pipelines = []
        self.tokenizers = []
        token = hf_token or os.getenv("HF_TOKEN")
        for index in range(copies):
            gpu_id = int(str(gpu_ids[index]).replace("cuda:", ""))
            print(f"Loading optimizer LLM copy {index + 1}/{copies} on cuda:{gpu_id}")
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=gpu_id,
                torch_dtype=torch.bfloat16,
                token=token,
            )
            self.pipelines.append(pipeline)
            self.tokenizers.append(pipeline.tokenizer)

    @staticmethod
    def _chat_prompt(tokenizer: object, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"

    def _query_one(self, index: int, prompt: str, max_new_tokens: int) -> str:
        formatted = self._chat_prompt(self.tokenizers[index], prompt)
        result = self.pipelines[index](
            formatted,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=OPTIMIZER_TEMPERATURE,
            top_p=OPTIMIZER_TOP_P,
            return_full_text=False,
        )
        return str(result[0].get("generated_text", ""))

    def query(self, prompts: Sequence[str], max_new_tokens: int = 768) -> list[str]:
        if len(prompts) != len(self.pipelines):
            raise ValueError("The number of prompts must equal the number of LLM copies")
        responses: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_index = {
                executor.submit(self._query_one, index, prompt, max_new_tokens): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                responses[index] = future.result()
        return responses


def score_candidates(model, texts: Sequence[str], target: torch.Tensor) -> torch.Tensor:
    embeddings = encode_texts(model, texts, input_length=INPUT_LENGTH, batch_size=32)
    target = F.normalize(target.cpu(), p=2, dim=1)
    return embeddings @ target.T.squeeze(1)


def top_candidates(
    model,
    texts: Sequence[str],
    target: torch.Tensor,
    k: int,
) -> tuple[list[str], torch.Tensor]:
    texts = unique_texts(texts)
    if not texts:
        return [], torch.empty(0)
    scores = score_candidates(model, texts, target)
    indices = torch.argsort(scores, descending=True)[:k]
    return [texts[index] for index in indices.tolist()], scores[indices]


def optimize_one_prototype(
    model,
    target: torch.Tensor,
    nearest_pool: list[str],
    llms: ParallelOptimizerLLMs,
    dataset: str,
    iterations: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    initialization_prompts = []
    for _ in range(len(llms.pipelines)):
        examples = rng.sample(nearest_pool, k=min(NEIGHBORS_PER_PROMPT, len(nearest_pool)))
        initialization_prompts.append(
            initialization_prompt(examples, CANDIDATES_PER_LLM, dataset)
        )
    responses = llms.query(initialization_prompts, max_new_tokens=512)
    initial_texts = unique_texts(
        [item for response in responses for item in extract_python_list(response)]
    )
    if len(initial_texts) < CANDIDATES_RETAINED:
        initial_texts = unique_texts(initial_texts + nearest_pool)
    population, scores = top_candidates(model, initial_texts, target, CANDIDATES_RETAINED)
    if len(population) < CANDIDATES_RETAINED:
        raise RuntimeError("Could not construct the initial candidate population")

    history = [{
        "iteration": 0,
        "best_similarity": float(scores.max().item()),
        "mean_similarity": float(scores.mean().item()),
        "best_text": population[int(torch.argmax(scores).item())],
    }]

    for iteration in range(1, iterations + 1):
        prompts = []
        for _ in range(len(llms.pipelines)):
            examples = rng.sample(nearest_pool, k=min(NEIGHBORS_PER_PROMPT, len(nearest_pool)))
            prompts.append(
                optimization_prompt(
                    population=population,
                    distances=scores.tolist(),
                    num_neighbors=CANDIDATES_RETAINED,
                    training_examples=examples,
                    dataset=dataset,
                )
            )
        responses = llms.query(prompts, max_new_tokens=768)
        new_texts = unique_texts(
            [item for response in responses for item in extract_python_list(response)]
        )
        population, scores = top_candidates(
            model,
            list(population) + new_texts,
            target,
            CANDIDATES_RETAINED,
        )
        best_index = int(torch.argmax(scores).item())
        history.append({
            "iteration": iteration,
            "best_similarity": float(scores[best_index].item()),
            "mean_similarity": float(scores.mean().item()),
            "best_text": population[best_index],
        })
        print(
            f"  iteration {iteration:02d}/{iterations}: "
            f"best={scores[best_index].item():.4f}, mean={scores.mean().item():.4f}"
        )

    best_index = int(torch.argmax(scores).item())
    return {
        "final_text": population[best_index],
        "final_similarity": float(scores[best_index].item()),
        "final_population": population,
        "final_scores": [float(value) for value in scores.tolist()],
        "history": history,
    }


def run_configuration(
    dataset: str,
    model_name: str,
    seed: int,
    device: str,
    llms: ParallelOptimizerLLMs,
    force: bool,
    hf_token: str | None,
) -> dict:
    output_path = result_path(dataset, model_name, seed)
    if output_path.exists() and not force:
        print(f"Skipping existing result: {output_path.name}")
        with output_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    seed_everything(seed)
    checkpoint = load_checkpoint(checkpoint_path(model_name, dataset, seed), map_location="cpu")
    metadata = checkpoint["metadata"]
    domain = load_domain(
        dataset=dataset,
        model_name=model_name,
        device=device,
        prototype_dim=int(metadata["prototype_dim"]),
        input_length=int(metadata["input_length"]),
        hf_token=hf_token,
    )
    model = instantiate_from_checkpoint(
        checkpoint, device=device, hf_token=hf_token, backbone=domain.backbone
    )
    train_indices = metadata["train_indices"]
    train_loader = subset_loader(
        domain.train_dataset,
        train_indices,
        EVAL_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )
    test_loader = full_loader(domain.test_dataset, EVAL_BATCH_SIZE)
    train_embeddings = F.normalize(encode_loader(model, train_loader), p=2, dim=1)
    train_texts = domain.train_df.iloc[train_indices]["text"].astype(str).tolist()
    learned_prototypes = F.normalize(model.prototypes.detach().cpu(), p=2, dim=1)

    similarity_matrix = learned_prototypes @ train_embeddings.T
    nearest_indices = torch.argmax(similarity_matrix, dim=1)
    nearest_texts = [train_texts[index] for index in nearest_indices.tolist()]
    nearest_embeddings = train_embeddings[nearest_indices]
    nearest_similarities = similarity_matrix.max(dim=1).values

    settings = DATASET_SETTINGS[dataset]
    optimized: list[dict] = []
    for prototype_index in range(model.num_total_prototypes):
        print(
            f"\nOptimizing {dataset}/{model_name}/seed{seed} "
            f"prototype {prototype_index + 1}/{model.num_total_prototypes}"
        )
        pool_indices = torch.argsort(similarity_matrix[prototype_index], descending=True)[
            :NEAREST_NEIGHBOR_POOL
        ]
        pool_texts = [train_texts[index] for index in pool_indices.tolist()]
        prototype_result = optimize_one_prototype(
            model=model,
            target=learned_prototypes[prototype_index : prototype_index + 1],
            nearest_pool=pool_texts,
            llms=llms,
            dataset=dataset,
            iterations=int(settings["iterations"]),
            seed=_stable_seed(dataset, model_name, seed, prototype_index),
        )
        prototype_result["prototype_index"] = prototype_index
        prototype_result["nearest_neighbor_text"] = nearest_texts[prototype_index]
        prototype_result["nearest_neighbor_similarity"] = float(nearest_similarities[prototype_index].item())
        optimized.append(prototype_result)
        save_json(
            output_path.with_suffix(".progress.json"),
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "completed_prototypes": optimized,
            },
        )

    llm_texts = [item["final_text"] for item in optimized]
    llm_embeddings = F.normalize(encode_texts(model, llm_texts, INPUT_LENGTH), p=2, dim=1)
    llm_similarities = (llm_embeddings * learned_prototypes).sum(dim=1)

    with torch.no_grad():
        model.prototypes.copy_(learned_prototypes.to(device))
    learned_metrics = evaluate(model, test_loader)

    with torch.no_grad():
        model.prototypes.copy_(llm_embeddings.to(device))
    llm_metrics = evaluate(model, test_loader)

    with torch.no_grad():
        model.prototypes.copy_(nearest_embeddings.to(device))
    nearest_metrics = evaluate(model, test_loader)

    tensors_path = prototype_tensor_path(dataset, model_name, seed)
    tensors_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "learned": learned_prototypes,
            "llm_generated": llm_embeddings,
            "nearest_neighbor": nearest_embeddings,
        },
        tensors_path,
    )

    result = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "experiment": settings["experiment"],
        "iterations": settings["iterations"],
        "optimizer_llm": OPTIMIZER_LLM_ID,
        "num_parallel_llms": len(llms.pipelines),
        "candidate_population_size": CANDIDATES_RETAINED,
        "nearest_neighbor_pool_size": NEAREST_NEIGHBOR_POOL,
        "neighbors_sampled_per_prompt": NEIGHBORS_PER_PROMPT,
        "learned_metrics": learned_metrics,
        "llm_generated_metrics": llm_metrics,
        "nearest_neighbor_metrics": nearest_metrics,
        "avg_llm_similarity": float(llm_similarities.mean().item()),
        "avg_nearest_similarity": float(nearest_similarities.mean().item()),
        "avg_llm_length_characters": float(sum(map(len, llm_texts)) / len(llm_texts)),
        "avg_nearest_length_characters": float(sum(map(len, nearest_texts)) / len(nearest_texts)),
        "llm_texts": llm_texts,
        "nearest_neighbor_texts": nearest_texts,
        "prototype_results": optimized,
        "prototype_tensor_path": str(tensors_path),
    }
    save_json(output_path, result)
    progress = output_path.with_suffix(".progress.json")
    if progress.exists():
        progress.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="*", default=QUANT_MODELS, choices=QUANT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0", help="Classifier/encoder device")
    parser.add_argument("--llm-gpu-ids", nargs="*", default=OPTIMIZER_GPU_IDS)
    parser.add_argument("--llm-copies", type=int, default=NUM_PARALLEL_LLMS)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    llms = ParallelOptimizerLLMs(
        model_id=OPTIMIZER_LLM_ID,
        gpu_ids=args.llm_gpu_ids,
        copies=args.llm_copies,
        hf_token=args.hf_token,
    )
    for seed in args.seeds:
        for dataset in args.datasets:
            for model_name in args.models:
                run_configuration(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
                    device=args.device,
                    llms=llms,
                    force=args.force,
                    hf_token=args.hf_token,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
