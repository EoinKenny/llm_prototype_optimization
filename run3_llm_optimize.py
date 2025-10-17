"""
Multi-GPU variant with 3 parallel LLMs for 3x throughput:
- BERT-like model on cuda:0
- 3x Llama LLMs on cuda:1,2,3
- Queries all 3 LLMs in parallel with different random samples
- Concatenates results for better optimization
- Iterates over multiple datasets and models

FIXES:
- distance_function now uses .squeeze(0) instead of .squeeze() to prevent 0-d tensor errors
- nn_reviews_pool texts are truncated to 256 tokens to prevent OOM
- early_example is also truncated
"""

import os
import gc
import ast
import time
import pickle
import argparse
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import transformers

# Local modules expected by the original script
from src.prompts import make_prompt
from src.functions import *          # load_domain, compute_embeddings, etc.
from src.models import LMProtoNet
from src.functions import make_initial_prompt_str
from config import DATASETS, MODELS, SEEDS


def clean_gpus():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_python_list(text: str):
    start_index = text.rfind('[')
    end_index = text.rfind(']')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None
    python_list_str = text[start_index:end_index+1]
    try:
        return ast.literal_eval(python_list_str)
    except (SyntaxError, ValueError):
        return None


# ------------------ Safety helpers ------------------
def coerce_to_string(x, default="placeholder"):
    """
    Convert any object to a non-empty string; if empty after strip, return default.
    """
    try:
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default


def sanitize_text_list(lst, default="placeholder"):
    """
    Produce a list[str] of non-empty strings, deduped while preserving order.
    If lst isn't a list/tuple, return [].
    """
    if not isinstance(lst, (list, tuple)):
        return []
    seen = set()
    out = []
    for x in lst:
        s = coerce_to_string(x, default=default)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def pad_to_length(lst, target_len, filler="placeholder"):
    """
    If lst is shorter than target_len, pad with filler to reach target_len.
    """
    if len(lst) < target_len:
        lst = list(lst) + [filler] * (target_len - len(lst))
    return lst


def truncate_text_to_tokens(text, tokenizer, max_tokens=256):
    """
    Truncate text to max_tokens using the provided tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"System: You are a helpful chatbot.\nUser: {user_prompt}\nAssistant:"


def query_single_llm(prompt: str, llm_pipeline, tokenizer, max_new_tokens: int = 1024) -> str:
    """Query a single LLM instance."""
    input_str = build_chat_prompt(tokenizer, prompt)
    outputs = llm_pipeline(
        input_str,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        return_full_text=False,
    )
    return outputs[0].get("generated_text", "")


def query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=1024):
    """
    Query multiple LLMs in parallel with different prompts.
    Returns a list of responses, one per LLM.
    """
    responses = []
    with ThreadPoolExecutor(max_workers=len(llm_pipelines)) as executor:
        futures = []
        for prompt, pipeline, tokenizer in zip(prompts, llm_pipelines, llm_tokenizers):
            future = executor.submit(query_single_llm, prompt, pipeline, tokenizer, max_new_tokens)
            futures.append(future)
        
        for future in as_completed(futures):
            responses.append(future.result())
    
    return responses


def load_multiple_llms(args):
    """Load multiple LLM instances on different GPUs."""
    model_id = 'meta-llama/Llama-3.2-3B-Instruct'
    hf_token = args.hf_token or os.getenv("HF_TOKEN", None)
    
    # Parse GPU IDs
    gpu_ids = args.llm_gpu_ids.split(',')
    num_llms = args.llm_parallel_copies
    
    if len(gpu_ids) < num_llms:
        # If not enough GPU IDs specified, cycle through the ones we have
        gpu_ids = gpu_ids * (num_llms // len(gpu_ids) + 1)
    gpu_ids = gpu_ids[:num_llms]
    
    pipelines = []
    tokenizers = []
    
    for i, gpu_id in enumerate(gpu_ids):
        device_map = f"cuda:{gpu_id.strip()}"
        print(f"Loading LLM {i+1}/{num_llms} on {device_map}")
        
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device_map,
            token=hf_token,
            temperature=1.0,
            do_sample=True,
        )
        tok = pipe.tokenizer if hasattr(pipe, "tokenizer") else transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        pipelines.append(pipe)
        tokenizers.append(tok)
    
    return pipelines, tokenizers


def distance_function(prototype_hidden_state, candidate_embeddings, distance_func_type='cosine'):
    """
    Returns similarity scores using consistent method with training script.
    For cosine: uses matrix multiplication on normalized vectors, returns values in [-1, 1]
    For L2: returns exp(-distance) in (0, 1]
    
    FIXED: Uses .squeeze(0) instead of .squeeze() to prevent 0-d tensor when only one candidate.
    """
    if distance_func_type == 'l2':
        distances = torch.cdist(prototype_hidden_state, candidate_embeddings, p=2).squeeze(0)
        distances = torch.exp(-distances)
    elif distance_func_type == 'cosine':
        # Use matrix multiplication like in training script
        # prototype_hidden_state: [1, H], candidate_embeddings: [N, H]
        similarities = torch.matmul(prototype_hidden_state, candidate_embeddings.T).squeeze(0)
        return similarities
    else:
        raise ValueError("distance_func_type must be either 'l2' or 'cosine'")
    return distances


# ---- Evaluation helpers ----
def _compute_metrics(preds, labels, num_classes):
    acc = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro": {"precision": float(prec_macro), "recall": float(rec_macro), "f1": float(f1_macro)},
        "micro": {"precision": float(prec_micro), "recall": float(rec_micro), "f1": float(f1_micro)},
    }


@torch.no_grad()
def eval_split(model, loader, device, model_type):
    all_preds, all_labels = [], []
    for batch in loader:
        if model_type == 'bert':
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                forward_type='train'
            )
        elif model_type == 'llm':
            labels = batch[1].to(device)
            outputs = model(llm_encodings=batch[0].to(device), forward_type='train')
        else:
            raise NameError('Unknown backbone type.')
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    num_classes = model.num_labels
    return _compute_metrics(all_preds, all_labels, num_classes)


def run_single_experiment(dataset, model, args):
    """Run experiment for a single dataset-model combination."""
    print(f'\n{"="*60}')
    print(f'RUNNING EXPERIMENT: Dataset={dataset}, Model={model}')
    print(f'{"="*60}')

    
    # Update args for this experiment
    args.dataset = dataset
    args.model = model
    
    random.seed(args.seed)
    clean_gpus()

    # Container matching the original pickle usage
    all_path = f'data/optimization_exp_all_data_seed{args.seed}.pickle'
    if os.path.exists(all_path):
        try:
            with open(all_path, 'rb') as handle:
                all_exp_data = pickle.load(handle)
            if not isinstance(all_exp_data, dict):
                all_exp_data = {}
        except Exception:
            all_exp_data = {}
    else:
        all_exp_data = {}

    print(f'\n=== Loading domain for {dataset} / {model} ===')
    try:
        data_utils = load_domain(args)
    except Exception as e:
        print(f"ERROR: Failed to load domain for {dataset}/{model}: {e}")
        return None
        
    tokenizer = data_utils['tokenizer']
    train_df = data_utils['train_df']

    print('Instantiating Model')
    proto_model = LMProtoNet(
        data_utils['model'],
        num_labels=data_utils['num_labels'],
        num_protos_per_class=args.num_protos,
    )
    model_type = getattr(data_utils['model'], "model_type", getattr(getattr(data_utils['model'], "config", None), "model_type", None))

    if model_type == 'llm':
        train_loader_non_random = DataLoader(data_utils['train_dataset_enc'], batch_size=128, shuffle=False)
        test_loader = DataLoader(data_utils['test_dataset_enc'], batch_size=128, shuffle=False)
    else:
        train_loader_non_random = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)
        test_loader = DataLoader(data_utils['test_dataset'], batch_size=128, shuffle=False)

    weight_dir = f'weights/pre_projection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}.pt'
        
    print(f'Loading weights from {weight_dir}')
    if not os.path.exists(weight_dir):
        print(f"ERROR: Weight file not found: {weight_dir}")
        return None
        
    maploc = torch.device(args.device)
    try:
        state_dict = torch.load(weight_dir, map_location=maploc, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_dir, map_location=maploc)
    proto_model.load_state_dict(state_dict, strict=False)
    proto_model.to(args.device)
    proto_model.eval()

    os.makedirs(f'datasets/preprocess/{args.dataset}', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('weights', exist_ok=True)

    # Load train encodings
    enc_path = f'datasets/preprocess/{dataset}/train_encodings_{model}_{args.seed}.pt'
    print(f"Loading train encodings from {enc_path}")
    if not os.path.exists(enc_path):
        print(f"ERROR: Encodings file not found: {enc_path}")
        return None
    train_encodings = torch.load(enc_path, map_location='cpu')

    print('Evaluating base model on test set...')
    base_metrics = eval_split(proto_model, test_loader, args.device, model_type)
    print(f"Base metrics ⇒ Acc: {base_metrics['accuracy']:.4f}, MacroF1: {base_metrics['macro']['f1']:.4f}")

    # Use learned latent prototypes as targets
    with torch.no_grad():
        learned_latents = proto_model.prototypes.detach().cpu()
        learned_latents_norm = F.normalize(learned_latents, p=2, dim=1)
    prototype_list = list(range(learned_latents_norm.shape[0]))

    normalized_train_encodings = F.normalize(train_encodings, p=2, dim=1)

    # Per-prototype results
    experiment_data = []

    # Extra metric (1): learned prototypes vs nearest training embeddings
    with torch.no_grad():
        # [P, H] x [H, N] = [P, N]
        cosine_sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)
        max_sims_per_proto = cosine_sims.max(dim=1).values
        avg_cosine_proto_to_nearest_train = float(max_sims_per_proto.mean().item())

    # Find and save nearest neighbor texts for each prototype
    nearest_neighbor_texts = []
    with torch.no_grad():
        cosine_sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)  # [P, N]
        nn_indices = torch.argmax(cosine_sims, dim=1)  # [P]
        
        for idx in nn_indices:
            nn_text = train_df['text'].iloc[idx.item()]
            # Truncate to 256 tokens
            truncated_text = truncate_text_to_tokens(nn_text, tokenizer, max_tokens=256)
            nearest_neighbor_texts.append(truncated_text)

    # Pre-sanitize the training text
    train_texts = [coerce_to_string(t) for t in train_df['text'].values.tolist()]

    for prototype_idx in range(len(prototype_list)):
        target_proto = learned_latents_norm[prototype_idx].unsqueeze(0)  # [1, H]
        
        distances = distance_function(target_proto, normalized_train_encodings, distance_func_type=args.distance_func)

        # For cosine, higher is better (1 is perfect similarity)
        nn_indices_pool = torch.argsort(distances, descending=True)[:args.nn_pool_size]
        
        # FIXED: Truncate texts when building pool to prevent OOM
        nn_reviews_pool = [
            truncate_text_to_tokens(train_texts[i], tokenizer, max_tokens=256) 
            for i in nn_indices_pool.cpu().numpy()
        ]

        # Get early example (also truncated)
        early_distance = distances.max().item()
        early_example_raw = train_texts[torch.argmax(distances).item()]
        early_example = truncate_text_to_tokens(early_example_raw, tokenizer, max_tokens=256)

        # Initial guesses phase - now with parallel LLMs
        closest_distances = [None]
        closest_reviews = []
        attempts = 0
        max_attempts = 8

        while len(closest_distances) != args.num_neighbors and attempts < max_attempts:
            attempts += 1
            
            # Create different prompts for each LLM with different random samples
            prompts = []
            for llm_idx in range(args.llm_parallel_copies):
                # Sample different examples for each LLM
                if len(nn_reviews_pool) > 2:
                    sampled_indices = random.sample(range(len(nn_reviews_pool)), min(2, len(nn_reviews_pool)))
                    sampled_examples = [nn_reviews_pool[i] for i in sampled_indices]
                else:
                    sampled_examples = nn_reviews_pool[:2]
                
                prompt = make_initial_prompt_str(
                    sampled_examples, args.num_neighbors, args.dataset
                )
                prompts.append(prompt)
            
            # Query all LLMs in parallel
            llm_responses = query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=512)
            
            # Combine guesses from all LLMs
            all_guesses = []
            for response in llm_responses:
                guessed_list = extract_python_list(response)
                guessed_list = sanitize_text_list(guessed_list)
                if guessed_list:
                    all_guesses.extend(guessed_list)
            
            # Remove duplicates while preserving order
            unique_guesses = []
            seen = set()
            for g in all_guesses:
                if g not in seen:
                    seen.add(g)
                    unique_guesses.append(g)
            
            if not unique_guesses:
                continue

            # Limit to reasonable batch size for BERT processing
            unique_guesses = unique_guesses[:args.num_neighbors * 3]  # 3x the target

            inputs = tokenizer(
                unique_guesses,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.input_size,
            ).to(args.device)

            with torch.no_grad():
                outputs = proto_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                latents = outputs['cls_rep_normalized'].cpu().detach()

            all_distances = distance_function(
                target_proto, latents, distance_func_type=args.distance_func
            )
            
            # Select top-k best guesses (highest similarities)
            top_k_indices = torch.argsort(all_distances, descending=True)[:args.num_neighbors]
            closest_distances = all_distances[top_k_indices]
            # Truncate all guesses before storing
            closest_reviews = [truncate_text_to_tokens(unique_guesses[i], tokenizer, max_tokens=256) 
                             for i in top_k_indices.tolist()]

        # Fallback: fill with nearest training examples
        if len(closest_distances) != args.num_neighbors:
            fallback = nn_reviews_pool[:args.num_neighbors]
            fallback = sanitize_text_list(fallback)
            if not fallback:
                fallback = sanitize_text_list(nn_reviews_pool[:max(1, args.num_neighbors)]) or [early_example]
            fallback = pad_to_length(fallback, args.num_neighbors, filler=early_example)

            inputs = tokenizer(
                fallback,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.input_size,
            ).to(args.device)
            with torch.no_grad():
                outputs = proto_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                latents = outputs['cls_rep_normalized'].cpu().detach()
            closest_distances = distance_function(
                target_proto, latents, distance_func_type=args.distance_func
            )
            closest_reviews = list(fallback)

        max_similarity_history = [closest_distances.max().item()]
        avg_similarity_history = [closest_distances.mean().item()]
        best_guess_history = [closest_reviews[torch.argmax(closest_distances).item()]]

        # Iterative optimization with parallel LLMs
        for current_iter in range(args.num_iters):
            start_iter_time = time.time()
            print(f'=============> Current guesses: Prototype {prototype_idx+1}/{len(prototype_list)} -- Iteration {current_iter}/{args.num_iters}')
            prototype_sequence = "[latent prototype vector]"
            print(f'Ground truth:\n{prototype_sequence}\n')

            pairs = list(zip(closest_reviews, closest_distances.tolist()))
            pairs.sort(key=lambda x: x[1], reverse=True)
            print(f"Top guesses with {args.distance_func} similarity (-1 to 1):")
            print(f"Original NN similarity: {early_distance:.4f}: {early_example}")
            for text, sim in pairs[:min(5, len(pairs))]:
                preview = (text[:160] + '...') if len(text) > 160 else text
                print(f"{sim:.4f}  |  {preview}")

            # Create different prompts for each LLM with different random samples
            prompts = []
            for llm_idx in range(args.llm_parallel_copies):
                # Sample different training examples for each LLM
                k = min(15, len(nn_reviews_pool))
                sampled_indices = torch.randperm(len(nn_reviews_pool))[:k]
                sampled_nn_reviews = [nn_reviews_pool[i] for i in sampled_indices]
                
                prompt = make_prompt(
                    closest_reviews,
                    closest_distances.tolist(),
                    args.num_neighbors,
                    training_examples=sampled_nn_reviews,
                    dataset=args.dataset,
                )
                prompts.append(prompt)
            
            # Query all LLMs in parallel
            llm_responses = query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=1024)
            
            # Combine new guesses from all LLMs
            all_new_guesses = []
            for response in llm_responses:
                new_guesses = extract_python_list(response) or []
                all_new_guesses.extend(new_guesses)
            
            # Build unique, sanitized new guesses
            unique_new = []
            for g in all_new_guesses:
                s = coerce_to_string(g)
                if s and (s not in closest_reviews) and (s not in unique_new):
                    unique_new.append(s)

            if unique_new:
                # Process all new guesses through BERT
                inputs = tokenizer(
                    text=unique_new,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.input_size,
                ).to(args.device)
                with torch.no_grad():
                    new_states = proto_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        forward_type='train',
                    )['cls_rep_normalized'].cpu()

                new_dists = distance_function(
                    target_proto,
                    new_states,
                    distance_func_type=args.distance_func,
                )

                # Update closest_reviews with better guesses
                for g, d in zip(unique_new, new_dists):
                    threshold = closest_distances.min().item()
                    if d.item() > threshold:  # Higher is better for cosine
                        worst_idx = torch.argmin(closest_distances).item()
                        # Truncate before storing
                        closest_reviews[worst_idx] = truncate_text_to_tokens(g, tokenizer, max_tokens=256)
                        closest_distances[worst_idx] = d

            # Log iteration summary
            best_score = closest_distances.max().item()
            max_similarity_history.append(best_score)
            avg_similarity_history.append(closest_distances.mean().item())
            best_guess_history.append(closest_reviews[torch.argmax(closest_distances).item()])

            print(f"Iter took: {round(time.time() - start_iter_time)} sec (with {args.llm_parallel_copies} parallel LLMs)")

        # Finalize best guess (truncate before storing)
        best_idx = torch.argmax(closest_distances).item()
        final_guess_raw = coerce_to_string(closest_reviews[best_idx], default=early_example)
        final_guess = truncate_text_to_tokens(final_guess_raw, tokenizer, max_tokens=256)

        experiment_data.append({
            "prototype_index": prototype_idx,
            "distance_func": args.distance_func,
            "early_distance": early_distance,
            "early_example": early_example,
            "max_similarity": max_similarity_history,
            "avg_similarity": avg_similarity_history,
            "best_guess_history": best_guess_history,
            "final_guess": final_guess,
            "nearest_neighbor_text": nearest_neighbor_texts[prototype_idx],
        })

    # Save per-prototype data
    key = f'{args.dataset}_{args.model}_latent_optim_{args.llm_parallel_copies}llms'
    all_exp_data[key] = experiment_data
    with open(all_path, 'wb') as handle:
        pickle.dump(all_exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate with optimized prototype texts (Stage A)
    print("Updating prototypes to optimized text embeddings and evaluating (Stage A)...")

    P = len(prototype_list)
    final_texts = []
    for d in experiment_data:
        s = coerce_to_string(d.get("final_guess"), default=d.get("early_example", "placeholder"))
        final_texts.append(s)
    final_texts = pad_to_length(final_texts, P, filler="placeholder")

    inputs = tokenizer(
        final_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.input_size,
    ).to(args.device)
    with torch.no_grad():
        outs = proto_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            forward_type='train',
        )
        optimized_proto_latents = outs['cls_rep_normalized']

    # Extra metric (2): optimized latents vs learned prototypes
    with torch.no_grad():
        opt_norm_for_metric = F.normalize(optimized_proto_latents.detach().cpu(), p=2, dim=1)
        
        # Element-wise similarity
        cosine_sims_diag = (opt_norm_for_metric * learned_latents_norm).sum(dim=1)  # [P]
        avg_cosine_optimized_to_proto = float(cosine_sims_diag.mean().item())
        
        # Sanity check
        print(f"Cosine similarity sanity check:")
        print(f"  Metric 1 (proto to nearest train): {avg_cosine_proto_to_nearest_train:.6f}")
        print(f"  Metric 2 (optimized to proto): {avg_cosine_optimized_to_proto:.6f}")
        print(f"  Raw cosine range for metric 2: [{cosine_sims_diag.min():.4f}, {cosine_sims_diag.max():.4f}]")

    with torch.no_grad():
        proto_model.prototypes.copy_(optimized_proto_latents)

    stageA_metrics = eval_split(proto_model, test_loader, args.device, model_type)
    print(f"Stage-A (optimized text latents) ⇒ Acc: {stageA_metrics['accuracy']:.4f}, MacroF1: {stageA_metrics['macro']['f1']:.4f}")
    weight_stageA = f'weights/latent_optim_preprojection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}_{args.llm_parallel_copies}llms.pt'
    torch.save(proto_model.state_dict(), weight_stageA)

    
    # Project prototypes to nearest training embeddings (Stage B)
    print("Projecting ORIGINAL prototypes onto nearest training examples and evaluating (Stage B)...")
    with torch.no_grad():
        # [P, H] x [H, N] = [P, N]
        sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)
        nn_idx = torch.argmax(sims, dim=1)
        projected = normalized_train_encodings[nn_idx]
    
    with torch.no_grad():
        proto_model.prototypes.copy_(projected.to(args.device))

        
    stageB_metrics = eval_split(proto_model, test_loader, args.device, model_type)
    print(f"Stage-B (projected) ⇒ Acc: {stageB_metrics['accuracy']:.4f}, MacroF1: {stageB_metrics['macro']['f1']:.4f}")
    weight_stageB = f'weights/latent_optim_postprojection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}_{args.llm_parallel_copies}llms.pt'
    torch.save(proto_model.state_dict(), weight_stageB)

    # Append summary
    run_summary = {
        "dataset": dataset,
        "model": model,
        "base_metrics": base_metrics,
        "stageA_metrics_optimized_text_latents": stageA_metrics,
        "stageB_metrics_projected": stageB_metrics,
        "optimized_prototype_texts": final_texts,
        "nearest_neighbor_texts": nearest_neighbor_texts,
        "weights_preprojection_path": weight_stageA,
        "weights_postprojection_path": weight_stageB,
        "avg_cosine_proto_to_nearest_train": avg_cosine_proto_to_nearest_train,
        "avg_cosine_optimized_to_proto": avg_cosine_optimized_to_proto,
        "num_llms_used": args.llm_parallel_copies,
    }
    all_exp_data[key + "_summary"] = run_summary
    with open(all_path, 'wb') as handle:
        pickle.dump(all_exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Run complete for", key)
    print(json.dumps(run_summary, indent=2))
    
    return run_summary


def main(args):

    datasets = DATASETS
    models = MODELS    
    
    print(f"Running experiments for:")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Total combinations: {len(datasets) * len(models)}")
    
    # Load LLMs once (they'll be reused across all experiments)
    print(f'\nLoading {args.llm_parallel_copies} LLM pipelines on GPUs: {args.llm_gpu_ids}')
    global llm_pipelines, llm_tokenizers
    llm_pipelines, llm_tokenizers = load_multiple_llms(args)
    
    # Store all results
    all_results = {}
    successful_runs = 0
    failed_runs = 0
    
    # Run experiments for all combinations
    for dataset in DATASETS:

        if dataset in ['imdb', 'amazon_reviews', 'agnews']:
            args.num_protos = 3
        elif dataset in ['trec', 'dbpedia', '20newsgroups']:
            args.num_protos = 1
        else:
            raise NameError('Wrong dataset')
        
        for model in models:
            try:
                print(f'\n{"#"*80}')
                print(f'STARTING: Dataset={dataset}, Model={model}')
                print(f'{"#"*80}')
                
                result = run_single_experiment(dataset, model, args)
                if result is not None:
                    all_results[f"{dataset}_{model}"] = result
                    successful_runs += 1
                    print(f"✅ SUCCESS: {dataset}/{model}")
                else:
                    failed_runs += 1
                    print(f"❌ FAILED: {dataset}/{model}")
                    
            except Exception as e:
                failed_runs += 1
                print(f"❌ ERROR in {dataset}/{model}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Final summary
    print(f'\n{"="*80}')
    print(f'EXPERIMENT BATCH COMPLETE')
    print(f'{"="*80}')
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total attempted: {successful_runs + failed_runs}")
    
    if all_results:
        print(f"\nSUMMARY OF ALL RESULTS:")
        print("-" * 60)
        for key, result in all_results.items():
            print(f"{key}:")
            print(f"  Base Acc: {result['base_metrics']['accuracy']:.4f}")
            print(f"  Stage A Acc: {result['stageA_metrics_optimized_text_latents']['accuracy']:.4f}")
            print(f"  Stage B Acc: {result['stageB_metrics_projected']['accuracy']:.4f}")
            print()
    
    # Save final combined results
    if all_results:
        final_results_path = f'data/all_experiments_summary_seed{args.seed}.json'
        with open(final_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to: {final_results_path}")

    print("All experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_protos', type=int, default=1)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=30)
    parser.add_argument('--models', type=str, nargs='+', default='bert',
                        help='List of models to iterate over (e.g., --models bert mpnet)')
    parser.add_argument('--datasets', type=str, nargs='+', default='20newsgroups',
                        help='List of datasets to iterate over (e.g., --datasets trec dbpedia 20newsgroups)')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for BERT-like model (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prototype_dim', type=int, default=256)
    parser.add_argument('--nn_pool_size', type=int, default=10)
    parser.add_argument('--baseline', action='store_true', default=True)
    parser.add_argument('--no_llm_head', action='store_true', default=False)
    parser.add_argument('--distance_func', type=str, choices=['cosine','l2'], default='cosine')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--llm_parallel_copies', type=int, default=3,
                        help='Number of parallel LLMs to load (default: 3)')
    parser.add_argument('--llm_gpu_ids', type=str, default="1,2,3",
                        help="Comma-separated CUDA device indices for LLMs (default: '1,2,3')")
        
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for seed in SEEDS:
        args.seed = seed
        print(f"\n=== Running experiments with seed {args.seed} ===\n")
        main(args)
        
        
        
        
# """
# Multi-GPU variant with 3 parallel LLMs for 3x throughput:
# - BERT-like model on cuda:0
# - 3x Llama LLMs on cuda:1,2,3
# - Queries all 3 LLMs in parallel with different random samples
# - Concatenates results for better optimization
# - Iterates over multiple datasets and models
# """

# import os
# import gc
# import ast
# import time
# import pickle
# import argparse
# import random
# import json
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# import transformers

# # Local modules expected by the original script
# from src.prompts import make_prompt
# from src.functions import *          # load_domain, compute_embeddings, etc.
# from src.models import LMProtoNet
# from src.functions import make_initial_prompt_str
# from config import DATASETS, MODELS, SEEDS


# def clean_gpus():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def extract_python_list(text: str):
#     start_index = text.rfind('[')
#     end_index = text.rfind(']')
#     if start_index == -1 or end_index == -1 or end_index < start_index:
#         return None
#     python_list_str = text[start_index:end_index+1]
#     try:
#         return ast.literal_eval(python_list_str)
#     except (SyntaxError, ValueError):
#         return None


# # ------------------ Safety helpers ------------------
# def coerce_to_string(x, default="placeholder"):
#     """
#     Convert any object to a non-empty string; if empty after strip, return default.
#     """
#     try:
#         s = str(x).strip()
#         return s if s else default
#     except Exception:
#         return default


# def sanitize_text_list(lst, default="placeholder"):
#     """
#     Produce a list[str] of non-empty strings, deduped while preserving order.
#     If lst isn't a list/tuple, return [].
#     """
#     if not isinstance(lst, (list, tuple)):
#         return []
#     seen = set()
#     out = []
#     for x in lst:
#         s = coerce_to_string(x, default=default)
#         if s and s not in seen:
#             seen.add(s)
#             out.append(s)
#     return out


# def pad_to_length(lst, target_len, filler="placeholder"):
#     """
#     If lst is shorter than target_len, pad with filler to reach target_len.
#     """
#     if len(lst) < target_len:
#         lst = list(lst) + [filler] * (target_len - len(lst))
#     return lst


# def truncate_text_to_tokens(text, tokenizer, max_tokens=256):
#     """
#     Truncate text to max_tokens using the provided tokenizer.
#     """
#     tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
#     return tokenizer.decode(tokens, skip_special_tokens=True)


# # ---------------------------------------------------------


# def build_chat_prompt(tokenizer, user_prompt: str) -> str:
#     messages = [
#         {"role": "system", "content": "You are a helpful chatbot."},
#         {"role": "user", "content": user_prompt},
#     ]
#     if hasattr(tokenizer, "apply_chat_template"):
#         try:
#             return tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#         except Exception:
#             pass
#     return f"System: You are a helpful chatbot.\nUser: {user_prompt}\nAssistant:"


# def query_single_llm(prompt: str, llm_pipeline, tokenizer, max_new_tokens: int = 1024) -> str:
#     """Query a single LLM instance."""
#     input_str = build_chat_prompt(tokenizer, prompt)
#     outputs = llm_pipeline(
#         input_str,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=1.0,
#         return_full_text=False,
#     )
#     return outputs[0].get("generated_text", "")


# def query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=1024):
#     """
#     Query multiple LLMs in parallel with different prompts.
#     Returns a list of responses, one per LLM.
#     """
#     responses = []
#     with ThreadPoolExecutor(max_workers=len(llm_pipelines)) as executor:
#         futures = []
#         for prompt, pipeline, tokenizer in zip(prompts, llm_pipelines, llm_tokenizers):
#             future = executor.submit(query_single_llm, prompt, pipeline, tokenizer, max_new_tokens)
#             futures.append(future)
        
#         for future in as_completed(futures):
#             responses.append(future.result())
    
#     return responses


# def load_multiple_llms(args):
#     """Load multiple LLM instances on different GPUs."""
#     model_id = 'meta-llama/Llama-3.2-3B-Instruct'
#     hf_token = args.hf_token or os.getenv("HF_TOKEN", None)
    
#     # Parse GPU IDs
#     gpu_ids = args.llm_gpu_ids.split(',')
#     num_llms = args.llm_parallel_copies
    
#     if len(gpu_ids) < num_llms:
#         # If not enough GPU IDs specified, cycle through the ones we have
#         gpu_ids = gpu_ids * (num_llms // len(gpu_ids) + 1)
#     gpu_ids = gpu_ids[:num_llms]
    
#     pipelines = []
#     tokenizers = []
    
#     for i, gpu_id in enumerate(gpu_ids):
#         device_map = f"cuda:{gpu_id.strip()}"
#         print(f"Loading LLM {i+1}/{num_llms} on {device_map}")
        
#         pipe = transformers.pipeline(
#             "text-generation",
#             model=model_id,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device_map=device_map,
#             token=hf_token,
#             temperature=1.0,
#             do_sample=True,
#         )
#         tok = pipe.tokenizer if hasattr(pipe, "tokenizer") else transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
#         pipelines.append(pipe)
#         tokenizers.append(tok)
    
#     return pipelines, tokenizers


# def distance_function(prototype_hidden_state, candidate_embeddings, distance_func_type='cosine'):
#     """
#     Returns similarity scores using consistent method with training script.
#     For cosine: uses matrix multiplication on normalized vectors, returns values in [-1, 1]
#     For L2: returns exp(-distance) in (0, 1]
#     """
#     if distance_func_type == 'l2':
#         distances = torch.cdist(prototype_hidden_state, candidate_embeddings, p=2).squeeze()
#         distances = torch.exp(-distances)
#     elif distance_func_type == 'cosine':
#         # Use matrix multiplication like in training script
#         # prototype_hidden_state: [1, H], candidate_embeddings: [N, H]
#         similarities = torch.matmul(prototype_hidden_state, candidate_embeddings.T).squeeze()
#         return similarities
#     else:
#         raise ValueError("distance_func_type must be either 'l2' or 'cosine'")
#     return distances


# # ---- Evaluation helpers ----
# def _compute_metrics(preds, labels, num_classes):
#     acc = accuracy_score(labels, preds)
#     prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
#         labels, preds, average="macro", zero_division=0
#     )
#     prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
#         labels, preds, average="micro", zero_division=0
#     )
#     return {
#         "accuracy": float(acc),
#         "macro": {"precision": float(prec_macro), "recall": float(rec_macro), "f1": float(f1_macro)},
#         "micro": {"precision": float(prec_micro), "recall": float(rec_micro), "f1": float(f1_micro)},
#     }


# @torch.no_grad()
# def eval_split(model, loader, device, model_type):
#     all_preds, all_labels = [], []
#     for batch in loader:
#         if model_type == 'bert':
#             labels = batch['labels'].to(device)
#             outputs = model(
#                 input_ids=batch['input_ids'].to(device),
#                 attention_mask=batch['attention_mask'].to(device),
#                 forward_type='train'
#             )
#         elif model_type == 'llm':
#             labels = batch[1].to(device)
#             outputs = model(llm_encodings=batch[0].to(device), forward_type='train')
#         else:
#             raise NameError('Unknown backbone type.')
#         logits = outputs['logits']
#         preds = torch.argmax(logits, dim=1)
#         all_preds.extend(preds.detach().cpu().tolist())
#         all_labels.extend(labels.detach().cpu().tolist())
#     num_classes = model.num_labels
#     return _compute_metrics(all_preds, all_labels, num_classes)


# def run_single_experiment(dataset, model, args):
#     """Run experiment for a single dataset-model combination."""
#     print(f'\n{"="*60}')
#     print(f'RUNNING EXPERIMENT: Dataset={dataset}, Model={model}')
#     print(f'{"="*60}')

    
#     # Update args for this experiment
#     args.dataset = dataset
#     args.model = model
    
#     random.seed(args.seed)
#     clean_gpus()

#     # Container matching the original pickle usage
#     all_path = f'data/optimization_exp_all_data_seed{args.seed}.pickle'
#     if os.path.exists(all_path):
#         try:
#             with open(all_path, 'rb') as handle:
#                 all_exp_data = pickle.load(handle)
#             if not isinstance(all_exp_data, dict):
#                 all_exp_data = {}
#         except Exception:
#             all_exp_data = {}
#     else:
#         all_exp_data = {}

#     print(f'\n=== Loading domain for {dataset} / {model} ===')
#     try:
#         data_utils = load_domain(args)
#     except Exception as e:
#         print(f"ERROR: Failed to load domain for {dataset}/{model}: {e}")
#         return None
        
#     tokenizer = data_utils['tokenizer']
#     train_df = data_utils['train_df']

#     print('Instantiating Model')
#     proto_model = LMProtoNet(
#         data_utils['model'],
#         num_labels=data_utils['num_labels'],
#         num_protos_per_class=args.num_protos,
#     )
#     model_type = getattr(data_utils['model'], "model_type", getattr(getattr(data_utils['model'], "config", None), "model_type", None))

#     if model_type == 'llm':
#         train_loader_non_random = DataLoader(data_utils['train_dataset_enc'], batch_size=128, shuffle=False)
#         test_loader = DataLoader(data_utils['test_dataset_enc'], batch_size=128, shuffle=False)
#     else:
#         train_loader_non_random = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)
#         test_loader = DataLoader(data_utils['test_dataset'], batch_size=128, shuffle=False)

#     # FIXED: Removed _baselineTrue and _no_llm_head from filename
#     weight_dir = f'weights/pre_projection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}.pt'
        
#     print(f'Loading weights from {weight_dir}')
#     if not os.path.exists(weight_dir):
#         print(f"ERROR: Weight file not found: {weight_dir}")
#         return None
        
#     maploc = torch.device(args.device)
#     try:
#         state_dict = torch.load(weight_dir, map_location=maploc, weights_only=True)
#     except TypeError:
#         state_dict = torch.load(weight_dir, map_location=maploc)
#     proto_model.load_state_dict(state_dict, strict=False)
#     proto_model.to(args.device)
#     proto_model.eval()

#     os.makedirs(f'datasets/preprocess/{args.dataset}', exist_ok=True)
#     os.makedirs('data', exist_ok=True)
#     os.makedirs('weights', exist_ok=True)

#     # FIXED: Load train encodings instead of computing them
#     enc_path = f'datasets/preprocess/{dataset}/train_encodings_{model}_{args.seed}.pt'
#     print(f"Loading train encodings from {enc_path}")
#     if not os.path.exists(enc_path):
#         print(f"ERROR: Encodings file not found: {enc_path}")
#         return None
#     train_encodings = torch.load(enc_path, map_location='cpu')

#     print('Evaluating base model on test set...')
#     base_metrics = eval_split(proto_model, test_loader, args.device, model_type)
#     print(f"Base metrics ⇒ Acc: {base_metrics['accuracy']:.4f}, MacroF1: {base_metrics['macro']['f1']:.4f}")

#     # Use learned latent prototypes as targets
#     with torch.no_grad():
#         learned_latents = proto_model.prototypes.detach().cpu()
#         learned_latents_norm = F.normalize(learned_latents, p=2, dim=1)
#     prototype_list = list(range(learned_latents_norm.shape[0]))

#     normalized_train_encodings = F.normalize(train_encodings, p=2, dim=1)

#     # Per-prototype results
#     experiment_data = []

#     # Extra metric (1): learned prototypes vs nearest training embeddings
#     # FIXED: Use consistent matrix multiplication method
#     with torch.no_grad():
#         # [P, H] x [H, N] = [P, N]
#         cosine_sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)
#         max_sims_per_proto = cosine_sims.max(dim=1).values
#         avg_cosine_proto_to_nearest_train = float(max_sims_per_proto.mean().item())

#     # Find and save nearest neighbor texts for each prototype
#     nearest_neighbor_texts = []
#     with torch.no_grad():
#         cosine_sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)  # [P, N]
#         nn_indices = torch.argmax(cosine_sims, dim=1)  # [P]
        
#         for idx in nn_indices:
#             nn_text = train_df['text'].iloc[idx.item()]
#             # Truncate to 256 tokens
#             truncated_text = truncate_text_to_tokens(nn_text, tokenizer, max_tokens=256)
#             nearest_neighbor_texts.append(truncated_text)

#     # Pre-sanitize the training text
#     train_texts = [coerce_to_string(t) for t in train_df['text'].values.tolist()]

#     for prototype_idx in range(len(prototype_list)):
#         target_proto = learned_latents_norm[prototype_idx].unsqueeze(0)  # [1, H]
        
#         # FIXED: Use consistent distance function with matrix multiplication
#         distances = distance_function(target_proto, normalized_train_encodings, distance_func_type=args.distance_func)

#         # For cosine, higher is better (1 is perfect similarity)
#         nn_indices_pool = torch.argsort(distances, descending=True)[:args.nn_pool_size]
#         nn_reviews_pool = [train_texts[i] for i in nn_indices_pool.cpu().numpy()]

#         early_distance = distances.max().item()
#         early_example = train_texts[torch.argmax(distances).item()]

#         # Initial guesses phase - now with parallel LLMs
#         closest_distances = [None]
#         closest_reviews = []
#         attempts = 0
#         max_attempts = 8

#         while len(closest_distances) != args.num_neighbors and attempts < max_attempts:
#             attempts += 1
            
#             # Create different prompts for each LLM with different random samples
#             prompts = []
#             for llm_idx in range(args.llm_parallel_copies):
#                 # Sample different examples for each LLM
#                 if len(nn_reviews_pool) > 2:
#                     sampled_indices = random.sample(range(len(nn_reviews_pool)), min(2, len(nn_reviews_pool)))
#                     sampled_examples = [nn_reviews_pool[i] for i in sampled_indices]
#                 else:
#                     sampled_examples = nn_reviews_pool[:2]
                
#                 prompt = make_initial_prompt_str(
#                     sampled_examples, args.num_neighbors, args.dataset
#                 )
#                 prompts.append(prompt)
            
#             # Query all LLMs in parallel
#             llm_responses = query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=512)
            
#             # Combine guesses from all LLMs
#             all_guesses = []
#             for response in llm_responses:
#                 guessed_list = extract_python_list(response)
#                 guessed_list = sanitize_text_list(guessed_list)
#                 if guessed_list:
#                     all_guesses.extend(guessed_list)
            
#             # Remove duplicates while preserving order
#             unique_guesses = []
#             seen = set()
#             for g in all_guesses:
#                 if g not in seen:
#                     seen.add(g)
#                     unique_guesses.append(g)
            
#             if not unique_guesses:
#                 continue

#             # Limit to reasonable batch size for BERT processing
#             unique_guesses = unique_guesses[:args.num_neighbors * 3]  # 3x the target

#             inputs = tokenizer(
#                 unique_guesses,
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True,
#                 max_length=args.input_size,
#             ).to(args.device)

#             with torch.no_grad():
#                 outputs = proto_model(
#                     input_ids=inputs['input_ids'],
#                     attention_mask=inputs['attention_mask']
#                 )
#                 latents = outputs['cls_rep_normalized'].cpu().detach()

#             # FIXED: Use consistent distance calculation
#             all_distances = distance_function(
#                 target_proto, latents, distance_func_type=args.distance_func
#             )
            
#             # Select top-k best guesses (highest similarities)
#             top_k_indices = torch.argsort(all_distances, descending=True)[:args.num_neighbors]
#             closest_distances = all_distances[top_k_indices]
#             closest_reviews = [unique_guesses[i] for i in top_k_indices.tolist()]

#         # Fallback: fill with nearest training examples
#         if len(closest_distances) != args.num_neighbors:
#             fallback = nn_reviews_pool[:args.num_neighbors]
#             fallback = sanitize_text_list(fallback)
#             if not fallback:
#                 fallback = sanitize_text_list(nn_reviews_pool[:max(1, args.num_neighbors)]) or [early_example]
#             fallback = pad_to_length(fallback, args.num_neighbors, filler=early_example)

#             inputs = tokenizer(
#                 fallback,
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True,
#                 max_length=args.input_size,
#             ).to(args.device)
#             with torch.no_grad():
#                 outputs = proto_model(
#                     input_ids=inputs['input_ids'],
#                     attention_mask=inputs['attention_mask']
#                 )
#                 latents = outputs['cls_rep_normalized'].cpu().detach()
#             closest_distances = distance_function(
#                 target_proto, latents, distance_func_type=args.distance_func
#             )
#             closest_reviews = list(fallback)

#         max_similarity_history = [closest_distances.max().item()]
#         avg_similarity_history = [closest_distances.mean().item()]
#         best_guess_history = [closest_reviews[torch.argmax(closest_distances).item()]]

#         # Iterative optimization with parallel LLMs
#         for current_iter in range(args.num_iters):
#             start_iter_time = time.time()
#             print(f'=============> Current guesses: Prototype {prototype_idx+1}/{len(prototype_list)} -- Iteration {current_iter}/{args.num_iters}')
#             prototype_sequence = "[latent prototype vector]"
#             print(f'Ground truth:\n{prototype_sequence}\n')

#             pairs = list(zip(closest_reviews, closest_distances.tolist()))
#             pairs.sort(key=lambda x: x[1], reverse=True)
#             print(f"Top guesses with {args.distance_func} similarity (-1 to 1):")
#             print(f"Original NN similarity: {early_distance:.4f}: {early_example}")
#             for text, sim in pairs[:min(5, len(pairs))]:
#                 preview = (text[:160] + '...') if len(text) > 160 else text
#                 print(f"{sim:.4f}  |  {preview}")

#             # Create different prompts for each LLM with different random samples
#             prompts = []
#             for llm_idx in range(args.llm_parallel_copies):
#                 # Sample different training examples for each LLM
#                 k = min(15, len(nn_reviews_pool))
#                 sampled_indices = torch.randperm(len(nn_reviews_pool))[:k]
#                 sampled_nn_reviews = [nn_reviews_pool[i] for i in sampled_indices]
                
#                 prompt = make_prompt(
#                     closest_reviews,
#                     closest_distances.tolist(),
#                     args.num_neighbors,
#                     training_examples=sampled_nn_reviews,
#                     dataset=args.dataset,
#                 )
#                 prompts.append(prompt)
            
#             # Query all LLMs in parallel
#             llm_responses = query_llms_parallel(prompts, llm_pipelines, llm_tokenizers, max_new_tokens=1024)
            
#             # Combine new guesses from all LLMs
#             all_new_guesses = []
#             for response in llm_responses:
#                 new_guesses = extract_python_list(response) or []
#                 all_new_guesses.extend(new_guesses)
            
#             # Build unique, sanitized new guesses
#             unique_new = []
#             for g in all_new_guesses:
#                 s = coerce_to_string(g)
#                 if s and (s not in closest_reviews) and (s not in unique_new):
#                     unique_new.append(s)

#             if unique_new:
#                 # Process all new guesses through BERT
#                 inputs = tokenizer(
#                     text=unique_new,
#                     return_tensors='pt',
#                     padding=True,
#                     truncation=True,
#                     max_length=args.input_size,
#                 ).to(args.device)
#                 with torch.no_grad():
#                     new_states = proto_model(
#                         input_ids=inputs['input_ids'],
#                         attention_mask=inputs['attention_mask'],
#                         forward_type='train',
#                     )['cls_rep_normalized'].cpu()

#                 # FIXED: Use consistent distance calculation
#                 new_dists = distance_function(
#                     target_proto,
#                     new_states,
#                     distance_func_type=args.distance_func,
#                 )

#                 # Update closest_reviews with better guesses
#                 for g, d in zip(unique_new, new_dists):
#                     threshold = closest_distances.min().item()
#                     if d.item() > threshold:  # Higher is better for cosine
#                         worst_idx = torch.argmin(closest_distances).item()
#                         closest_reviews[worst_idx] = g
#                         closest_distances[worst_idx] = d

#             # Log iteration summary
#             best_score = closest_distances.max().item()
#             max_similarity_history.append(best_score)
#             avg_similarity_history.append(closest_distances.mean().item())
#             best_guess_history.append(closest_reviews[torch.argmax(closest_distances).item()])

#             print(f"Iter took: {round(time.time() - start_iter_time)} sec (with {args.llm_parallel_copies} parallel LLMs)")

#         # Finalize best guess
#         best_idx = torch.argmax(closest_distances).item()
#         final_guess = coerce_to_string(closest_reviews[best_idx], default=early_example)

#         experiment_data.append({
#             "prototype_index": prototype_idx,
#             "distance_func": args.distance_func,
#             "early_distance": early_distance,
#             "early_example": early_example,
#             "max_similarity": max_similarity_history,
#             "avg_similarity": avg_similarity_history,
#             "best_guess_history": best_guess_history,
#             "final_guess": final_guess,
#             "nearest_neighbor_text": nearest_neighbor_texts[prototype_idx],  # NEW: Save NN text
#         })

#     # Save per-prototype data
#     key = f'{args.dataset}_{args.model}_latent_optim_{args.llm_parallel_copies}llms'
#     all_exp_data[key] = experiment_data
#     with open(all_path, 'wb') as handle:
#         pickle.dump(all_exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     # Evaluate with optimized prototype texts (Stage A)
#     print("Updating prototypes to optimized text embeddings and evaluating (Stage A)...")

#     P = len(prototype_list)
#     final_texts = []
#     for d in experiment_data:
#         s = coerce_to_string(d.get("final_guess"), default=d.get("early_example", "placeholder"))
#         final_texts.append(s)
#     final_texts = pad_to_length(final_texts, P, filler="placeholder")

#     inputs = tokenizer(
#         final_texts,
#         return_tensors='pt',
#         padding=True,
#         truncation=True,
#         max_length=args.input_size,
#     ).to(args.device)
#     with torch.no_grad():
#         outs = proto_model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             forward_type='train',
#         )
#         optimized_proto_latents = outs['cls_rep_normalized']

#     # Extra metric (2): optimized latents vs learned prototypes
#     # FIXED: Use consistent matrix multiplication method
#     with torch.no_grad():
#         opt_norm_for_metric = F.normalize(optimized_proto_latents.detach().cpu(), p=2, dim=1)
        
#         # Element-wise similarity using diagonal of matrix multiplication
#         # This is equivalent to (opt_norm * learned_latents_norm).sum(dim=1)
#         # but more explicit: we're comparing prototype i with optimized i
#         cosine_sims_diag = (opt_norm_for_metric * learned_latents_norm).sum(dim=1)  # [P]
#         avg_cosine_optimized_to_proto = float(cosine_sims_diag.mean().item())
        
#         # Sanity check
#         print(f"Cosine similarity sanity check:")
#         print(f"  Metric 1 (proto to nearest train): {avg_cosine_proto_to_nearest_train:.6f}")
#         print(f"  Metric 2 (optimized to proto): {avg_cosine_optimized_to_proto:.6f}")
#         print(f"  Raw cosine range for metric 2: [{cosine_sims_diag.min():.4f}, {cosine_sims_diag.max():.4f}]")

#     with torch.no_grad():
#         proto_model.prototypes.copy_(optimized_proto_latents)

#     stageA_metrics = eval_split(proto_model, test_loader, args.device, model_type)
#     print(f"Stage-A (optimized text latents) ⇒ Acc: {stageA_metrics['accuracy']:.4f}, MacroF1: {stageA_metrics['macro']['f1']:.4f}")
#     weight_stageA = f'weights/latent_optim_preprojection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}_{args.llm_parallel_copies}llms.pt'
#     torch.save(proto_model.state_dict(), weight_stageA)

    
#     # Project prototypes to nearest training embeddings (Stage B)
#     print("Projecting ORIGINAL prototypes onto nearest training examples and evaluating (Stage B)...")
#     with torch.no_grad():
#         # FIXED: Use consistent matrix multiplication
#         # [P, H] x [H, N] = [P, N]
#         sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)
#         nn_idx = torch.argmax(sims, dim=1)
#         projected = normalized_train_encodings[nn_idx]
    
#     with torch.no_grad():
#         proto_model.prototypes.copy_(projected.to(args.device))

        
#     stageB_metrics = eval_split(proto_model, test_loader, args.device, model_type)
#     print(f"Stage-B (projected) ⇒ Acc: {stageB_metrics['accuracy']:.4f}, MacroF1: {stageB_metrics['macro']['f1']:.4f}")
#     weight_stageB = f'weights/latent_optim_postprojection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}_{args.llm_parallel_copies}llms.pt'
#     torch.save(proto_model.state_dict(), weight_stageB)

#     # Append summary
#     run_summary = {
#         "dataset": dataset,
#         "model": model,
#         "base_metrics": base_metrics,
#         "stageA_metrics_optimized_text_latents": stageA_metrics,
#         "stageB_metrics_projected": stageB_metrics,
#         "optimized_prototype_texts": final_texts,
#         "nearest_neighbor_texts": nearest_neighbor_texts,  # NEW: Include NN texts in summary
#         "weights_preprojection_path": weight_stageA,
#         "weights_postprojection_path": weight_stageB,
#         "avg_cosine_proto_to_nearest_train": avg_cosine_proto_to_nearest_train,
#         "avg_cosine_optimized_to_proto": avg_cosine_optimized_to_proto,
#         "num_llms_used": args.llm_parallel_copies,
#     }
#     all_exp_data[key + "_summary"] = run_summary
#     with open(all_path, 'wb') as handle:
#         pickle.dump(all_exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     print("Run complete for", key)
#     print(json.dumps(run_summary, indent=2))
    
#     return run_summary


# def main(args):

#     datasets = DATASETS
#     models = MODELS    
    
#     print(f"Running experiments for:")
#     print(f"Datasets: {datasets}")
#     print(f"Models: {models}")
#     print(f"Total combinations: {len(datasets) * len(models)}")
    
#     # Load LLMs once (they'll be reused across all experiments)
#     print(f'\nLoading {args.llm_parallel_copies} LLM pipelines on GPUs: {args.llm_gpu_ids}')
#     global llm_pipelines, llm_tokenizers
#     llm_pipelines, llm_tokenizers = load_multiple_llms(args)
    
#     # Store all results
#     all_results = {}
#     successful_runs = 0
#     failed_runs = 0
    
#     # Run experiments for all combinations
#     for dataset in DATASETS:

#         if dataset in ['imdb', 'amazon_reviews', 'agnews']:
#             args.num_protos = 3
#         elif dataset in ['trec', 'dbpedia', '20newsgroups']:
#             args.num_protos = 1
#         else:
#             raise NameError('Wrong dataset')
        
#         for model in models:
#             try:
#                 print(f'\n{"#"*80}')
#                 print(f'STARTING: Dataset={dataset}, Model={model}')
#                 print(f'{"#"*80}')
                
#                 result = run_single_experiment(dataset, model, args)
#                 if result is not None:
#                     all_results[f"{dataset}_{model}"] = result
#                     successful_runs += 1
#                     print(f"✅ SUCCESS: {dataset}/{model}")
#                 else:
#                     failed_runs += 1
#                     print(f"❌ FAILED: {dataset}/{model}")
                    
#             except Exception as e:
#                 failed_runs += 1
#                 print(f"❌ ERROR in {dataset}/{model}: {str(e)}")
#                 continue
    
#     # Final summary
#     print(f'\n{"="*80}')
#     print(f'EXPERIMENT BATCH COMPLETE')
#     print(f'{"="*80}')
#     print(f"Successful runs: {successful_runs}")
#     print(f"Failed runs: {failed_runs}")
#     print(f"Total attempted: {successful_runs + failed_runs}")
    
#     if all_results:
#         print(f"\nSUMMARY OF ALL RESULTS:")
#         print("-" * 60)
#         for key, result in all_results.items():
#             print(f"{key}:")
#             print(f"  Base Acc: {result['base_metrics']['accuracy']:.4f}")
#             print(f"  Stage A Acc: {result['stageA_metrics_optimized_text_latents']['accuracy']:.4f}")
#             print(f"  Stage B Acc: {result['stageB_metrics_projected']['accuracy']:.4f}")
#             print()
    
#     # Save final combined results
#     if all_results:
#         final_results_path = f'data/all_experiments_summary_seed{args.seed}.json'
#         with open(final_results_path, 'w') as f:
#             json.dump(all_results, f, indent=2)
#         print(f"All results saved to: {final_results_path}")

#     print("All experiments complete.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_protos', type=int, default=1)
#     parser.add_argument('--num_neighbors', type=int, default=10)  # the number of guesses from LLM
#     parser.add_argument('--num_iters', type=int, default=30)  # 20
#     parser.add_argument('--models', type=str, nargs='+', default='bert',
#                         help='List of models to iterate over (e.g., --models bert mpnet)')
#     parser.add_argument('--datasets', type=str, nargs='+', default='20newsgroups',
#                         help='List of datasets to iterate over (e.g., --datasets trec dbpedia 20newsgroups)')
#     parser.add_argument('--input_size', type=int, default=256)
#     parser.add_argument('--device', type=str, default='cuda:0',
#                         help='Device for BERT-like model (default: cuda:0)')
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--prototype_dim', type=int, default=256)
#     parser.add_argument('--nn_pool_size', type=int, default=10)
#     parser.add_argument('--baseline', action='store_true', default=True)
#     parser.add_argument('--no_llm_head', action='store_true', default=False)
#     parser.add_argument('--distance_func', type=str, choices=['cosine','l2'], default='cosine')
#     parser.add_argument('--hf_token', type=str, default=None)
#     parser.add_argument('--llm_parallel_copies', type=int, default=3,
#                         help='Number of parallel LLMs to load (default: 3)')
#     parser.add_argument('--llm_gpu_ids', type=str, default="1,2,3",
#                         help="Comma-separated CUDA device indices for LLMs (default: '1,2,3')")
        
#     args = parser.parse_args()

#     if args.device == 'auto':
#         args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#     for seed in SEEDS:
#         args.seed = seed
#         print(f"\n=== Running experiments with seed {args.seed} ===\n")
#         main(args)









