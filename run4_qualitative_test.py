import os
import pickle
import random
import pandas as pd
import torch
import json
from typing import Dict, List, Any
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Assuming these imports work based on the original script
from src.functions import load_domain, compute_embeddings
from src.models import LMProtoNet


def truncate_text(text: str, max_words: int = 197) -> str:
    """Truncate text to maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'


def get_data_path(dataset: str, seed: int) -> str:
    """Get the correct data path based on dataset."""
    if dataset == 'agnews':
        return f'data/optimization_exp_all_data_seed{seed}_agnews.pickle'
    elif dataset in ['imdb', 'amazon_reviews']:
        return f'data/len_run/optimization_exp_all_data_seed{seed}.pickle'
    elif dataset in ['trec', 'dbpedia', '20newsgroups']:
        return f'data/acc_run/optimization_exp_all_data_seed{seed}.pickle'
    else:
        # Fallback to default path
        return f'data/optimization_exp_all_data_seed{seed}.pickle'


def load_experiment_data(dataset: str, seed: int) -> Dict:
    """Load the saved experiment data for a given seed and dataset."""
    data_path = get_data_path(dataset, seed)
    if os.path.exists(data_path):
        with open(data_path, 'rb') as handle:
            return pickle.load(handle)
    return {}


def get_stage_a_prototypes(exp_data: Dict, dataset: str, model: str, num_llms: int = 3) -> List[str]:
    """Extract optimized prototype texts (Stage A) from experiment data."""
    key = f'{dataset}_{model}_latent_optim_{num_llms}llms_summary'
    if key in exp_data:
        return exp_data[key].get('optimized_prototype_texts', [])
    return []


def get_stage_b_prototypes(dataset: str, model: str, seed: int, args: Any) -> List[str]:
    """Compute Stage B prototypes (training texts that learned prototypes project to)."""
    
    # This is the same logic as in the third script for computing Stage B
    args.dataset = dataset
    args.model = model
    args.seed = seed
    
    # Load domain data
    data_utils = load_domain(args)
    train_df = data_utils['train_df']
    
    # Load the model
    proto_model = LMProtoNet(
        data_utils['model'],
        num_labels=data_utils['num_labels'],
        num_protos_per_class=args.num_protos,
    )
    
    # Load pre-projection weights (the original learned prototypes)
    weight_path = f'weights/pre_projection_{model}_{dataset}_protos{args.num_protos}_baselineTrue_seed{seed}_no_llm_head{args.no_llm_head}.pt'
    
    if not os.path.exists(weight_path):
        print(f"Warning: Weight file not found: {weight_path}")
        return []
    
    try:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location='cpu')
    
    proto_model.load_state_dict(state_dict, strict=False)
    proto_model.eval()
    
    # Get or compute training encodings
    enc_path = f'datasets/preprocess/{dataset}/train_encodings_toy_{model}_{seed}_{args.no_llm_head}_{args.baseline}.pt'
    if os.path.exists(enc_path):
        train_encodings = torch.load(enc_path, map_location="cpu")
    else:
        print(f"Warning: Training encodings not found at {enc_path}")
        return []
    
    # Get learned prototypes and project to nearest training examples (Stage B logic from script 3)
    with torch.no_grad():
        learned_latents = proto_model.prototypes.detach().cpu()
        learned_latents_norm = F.normalize(learned_latents, p=2, dim=1)
        
        normalized_train_encodings = F.normalize(train_encodings, p=2, dim=1)
        
        # Project to nearest training examples (same as script 3)
        sims = torch.matmul(learned_latents_norm, normalized_train_encodings.T)
        nn_idx = torch.argmax(sims, dim=1)
        
        # Get the corresponding training texts
        train_texts = train_df['text'].values.tolist()
        stage_b_texts = [train_texts[idx] for idx in nn_idx.cpu().numpy()]
    
    return stage_b_texts


def load_test_data(dataset: str, model: str, seed: int, args: Any) -> tuple:
    """Load test dataset and model for a given configuration."""
    args.dataset = dataset
    args.model = model
    args.seed = seed
    
    # Load domain data
    data_utils = load_domain(args)
    
    # Load the model
    proto_model = LMProtoNet(
        data_utils['model'],
        num_labels=data_utils['num_labels'],
        num_protos_per_class=args.num_protos,
    )
    
    # Determine model type
    model_type = getattr(data_utils['model'], "model_type", 
                        getattr(getattr(data_utils['model'], "config", None), "model_type", None))
    
    # Create test loader
    if model_type == 'llm':
        test_dataset = data_utils['test_dataset_enc']
    else:
        test_dataset = data_utils['test_dataset']
    
    test_df = data_utils['test_df']
    
    return test_dataset, test_df, proto_model, data_utils, model_type


def load_model_weights(proto_model: Any, dataset: str, model: str, seed: int, 
                       stage: str, num_protos: int, baseline: bool = True, 
                       no_llm_head: bool = False, num_llms: int = 3, device: str = 'cuda:0'):
    """Load model weights for Stage A or Stage B (using actual file names from script 3)."""
    
    if stage == 'A':
        # Stage A uses pre-projection weights (optimized text latents)
        weight_path = f'weights/latent_optim_preprojection_{model}_{dataset}_protos{num_protos}_baseline{baseline}_seed{seed}_no_llm_head{no_llm_head}_{num_llms}llms.pt'
    else:  # Stage B
        # Stage B uses post-projection weights (projected to nearest training)
        weight_path = f'weights/latent_optim_postprojection_{model}_{dataset}_protos{num_protos}_baseline{baseline}_seed{seed}_no_llm_head{no_llm_head}_{num_llms}llms.pt'
    
    if os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weight_path, map_location=device)
        
        proto_model.load_state_dict(state_dict, strict=False)
        proto_model.to(device)
        proto_model.eval()
        return True
    else:
        print(f"Weight file not found: {weight_path}")
        return False


def get_nearest_prototype(proto_model: Any, test_instance: Any, tokenizer: Any, 
                         model_type: str, device: str = 'cuda:0') -> int:
    """Find the nearest prototype for a test instance."""
    proto_model.eval()
    
    with torch.no_grad():
        if model_type == 'llm':
            # For LLM models, test_instance is already encoded
            test_encoding = test_instance.unsqueeze(0).to(device)
            outputs = proto_model(llm_encodings=test_encoding, forward_type='train')
        else:
            # For BERT-like models, need to tokenize
            text = test_instance['text'] if isinstance(test_instance, dict) else str(test_instance)
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            
            outputs = proto_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                forward_type='train'
            )
        
        # Get the embedding
        cls_rep = outputs['cls_rep_normalized']
        
        # Compute distances to all prototypes
        prototypes = proto_model.prototypes
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        # Cosine similarity
        similarities = F.cosine_similarity(cls_rep, prototypes_norm)
        
        # Get nearest prototype index
        nearest_idx = torch.argmax(similarities).item()
        
    return nearest_idx


def create_prompt(test_text: str, stage_a_proto: str, stage_b_proto: str, dataset: str) -> str:
    """Create a single evaluation prompt without class label information."""
    
    # Truncate texts to 197 words
    test_text = truncate_text(test_text, 197)
    stage_a_proto = truncate_text(stage_a_proto, 197)
    stage_b_proto = truncate_text(stage_b_proto, 197)
    
    prompt = f"""You are analyzing prototypes used by a neural network classifier that uses cosine similarity for classification on the {dataset} dataset. The prototypes are being used to classify the test instance based on their cosine similarity to it, you job is to help us analyze if the prototypes have meaningful similarity to the test instance.

## Test Instance to Classify:
{test_text}

## Stage A Prototype:
{stage_a_proto}

## Stage B Prototype:
{stage_b_proto}

Please analyze these prototypes and the test instance:

1. First, identify ALL high-level concepts in the Stage A prototype that could be used by a classifier.

2. Do the same analysis for the Stage B prototype.

3. Based on cosine similarity principles, determine which prototype would be most similar to the test instance.

You should be comprehensive, you don't need to have the same number of concepts for both prototypes, it is ok for one to have many more concepts and/or irrelevant features.

Provide detailed reasoning for your analysis, then output a JSON object with the following structure:
```json
{{
  "stage_a_concepts_count": <integer>,
  "stage_b_concepts_count": <integer>,
  "most_similar_prototype": "<'stage_a' or 'stage_b'>",
}}
```

The length and detail level of the prototypes do NOT matter for classification purposes, do not consider them in your analysis, only focus on high-level concepts for the classification.
"""
    return prompt


def generate_prompts_for_configuration(dataset: str, model: str, seed: int, 
                                      num_samples: int = 100, device: str = 'cuda:0'):
    """Generate prompts for a specific dataset/model/seed configuration."""
    
    print(f"Generating prompts for {dataset}/{model}/seed{seed}")
    
    # Create args object with necessary attributes
    class Args:
        pass
    
    args = Args()
    args.dataset = dataset
    args.model = model
    args.seed = seed
    args.input_size = 256
    args.device = device
    args.baseline = True
    args.no_llm_head = False
    args.prototype_dim = 256
    
    # Set num_protos based on dataset (updated to include agnews)
    if dataset in ['imdb', 'amazon_reviews', 'agnews']:
        args.num_protos = 3
    elif dataset in ['20newsgroups', 'trec', 'dbpedia']:
        args.num_protos = 1
    else:
        args.num_protos = 1
    
    prompts_data = []
    
    try:
        # Load experiment data
        exp_data = load_experiment_data(dataset, seed)
        
        # Get Stage A prototypes (optimized text descriptions)
        stage_a_texts = get_stage_a_prototypes(exp_data, dataset, model)
        
        if not stage_a_texts:
            print(f"Warning: No Stage A texts found for {dataset}/{model}/seed{seed}")
            return prompts_data
        
        # Get Stage B prototypes (projected training texts)
        stage_b_texts = get_stage_b_prototypes(dataset, model, seed, args)
        
        if not stage_b_texts:
            print(f"Warning: No Stage B texts found for {dataset}/{model}/seed{seed}")
            return prompts_data
        
        print(f"Found {len(stage_a_texts)} Stage A and {len(stage_b_texts)} Stage B prototypes")
        
        # Load test data
        test_dataset, test_df, proto_model, data_utils, model_type = load_test_data(
            dataset, model, seed, args
        )
        
        tokenizer = data_utils['tokenizer']
        
        # Sample test instances
        test_size = len(test_df)
        sample_size = min(num_samples, test_size)
        sampled_indices = random.sample(range(test_size), sample_size)
        
        for idx in sampled_indices:
            try:
                # Get test instance
                test_text = test_df.iloc[idx]['text']
                test_label = test_df.iloc[idx]['label'] if 'label' in test_df.columns else 'unknown'
                
                # For Stage A: Load Stage A weights and find nearest prototype
                if load_model_weights(proto_model, dataset, model, seed, 'A', 
                                    args.num_protos, args.baseline, args.no_llm_head, 3, device):
                    
                    if model_type == 'llm':
                        test_instance = test_dataset[idx][0]
                    else:
                        test_instance = {'text': test_text}
                    
                    stage_a_proto_idx = get_nearest_prototype(
                        proto_model, test_instance, tokenizer, model_type, device
                    )
                    stage_a_proto_text = stage_a_texts[stage_a_proto_idx] if stage_a_proto_idx < len(stage_a_texts) else ""
                else:
                    print(f"Warning: Could not load Stage A weights")
                    continue
                
                # For Stage B: Load Stage B weights and find nearest prototype
                if load_model_weights(proto_model, dataset, model, seed, 'B',
                                    args.num_protos, args.baseline, args.no_llm_head, 3, device):
                    
                    stage_b_proto_idx = get_nearest_prototype(
                        proto_model, test_instance, tokenizer, model_type, device
                    )
                    stage_b_proto_text = stage_b_texts[stage_b_proto_idx] if stage_b_proto_idx < len(stage_b_texts) else ""
                else:
                    print(f"Warning: Could not load Stage B weights")
                    continue
                
                # Create prompt (without class label)
                prompt = create_prompt(test_text, stage_a_proto_text, stage_b_proto_text, dataset)
                
                prompts_data.append({
                    'dataset': dataset,
                    'model': model,
                    'seed': seed,
                    'test_idx': idx,
                    'test_label': str(test_label),  # Still store it for analysis but don't use in prompt
                    'stage_a_proto_idx': stage_a_proto_idx,
                    'stage_b_proto_idx': stage_b_proto_idx,
                    'prompt': prompt
                })
                
            except Exception as e:
                print(f"Error processing test instance {idx}: {e}")
                continue
    
    except Exception as e:
        print(f"Error in configuration {dataset}/{model}/seed{seed}: {e}")
    
    return prompts_data


def main():
    """Generate all prompts and save to CSV."""
    
    models = ['bert', 'electra', 'roberta']
    # Updated dataset list to include agnews and match your requirements
    datasets = ['trec', 'dbpedia', '20newsgroups', 'imdb', 'amazon_reviews', 'agnews']
    seeds = [0, 1, 2]
    num_prompts_per_config = 100
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    
    all_prompts = []
    
    total_configs = len(models) * len(datasets) * len(seeds)
    config_count = 0
    
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                config_count += 1
                print(f"\nProcessing configuration {config_count}/{total_configs}: {dataset}/{model}/seed{seed}")
                
                prompts = generate_prompts_for_configuration(
                    dataset, model, seed, num_prompts_per_config, device
                )
                
                all_prompts.extend(prompts)
                print(f"Generated {len(prompts)} prompts for this configuration")
    
    # Save to CSV
    df = pd.DataFrame(all_prompts)
    output_path = 'prototype_evaluation_prompts_corrected.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nTotal prompts generated: {len(all_prompts)}")
    print(f"Expected: {len(models) * len(datasets) * len(seeds) * num_prompts_per_config}")
    print(f"Saved to: {output_path}")
    
    # Save summary statistics
    summary = df.groupby(['dataset', 'model', 'seed']).size().reset_index(name='count')
    print("\nPrompts per configuration:")
    print(summary.to_string())
    
    # Also show which configurations might be missing
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                count = len(df[(df['dataset'] == dataset) & (df['model'] == model) & (df['seed'] == seed)])
                if count < num_prompts_per_config:
                    print(f"Warning: {dataset}/{model}/seed{seed} only has {count} prompts")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()