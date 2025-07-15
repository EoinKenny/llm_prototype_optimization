import zipfile
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random

from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import ModelWrapper


def make_initial_prompt_str(ground_truth_examples, num_guesses_to_generate, dataset_name):
    
    examples_to_show = ground_truth_examples[:min(5, len(ground_truth_examples))]
    examples_str = "\n".join([f'- "{ex}"' for ex in examples_to_show])
    
    if dataset_name == 'trec':
        description = 'question'
    elif dataset_name == '20newsgroups':
        description = 'news article'
    elif dataset_name == 'dbpedia':
        description = 'factual wikipedia page'
    else:
        raise NameError('wrong dataset name')    
    prompt = f"""I am trying to identify a prototypical example from the '{dataset_name}' dataset.
    The prototype should represent a typical example of a '{description}'.
The following examples are very similar to the real prototype:
{examples_str}

Based *only* on these examples, please generate a Python list containing exactly {num_guesses_to_generate} distinct, concise, and relevant phrases or sentences that you believe also capture the core concepts in these examples in a prototypical sentence.
Each phrase should be a potential textual description of the prototype and its core concepts.
Your output must be ONLY a single Python list of strings. For example: ["first candidate phrase", "second candidate phrase", ..., "tenth candidate phrase"]

Generated Python list:
"""    
    return prompt



def compute_embeddings(model, train_loader_non_random):

    model.eval()
    all_reps = []
    device = model.backbone.device

    print('getting embeddings...')
    with torch.no_grad():
        for batch in tqdm(train_loader_non_random):

            # Get logits & labels
            if model.backbone.model_type=='bert':
                reps = model(input_ids=batch['input_ids'].to(device) , attention_mask=batch['attention_mask'].to(device), forward_type='train')
            elif model.backbone.model_type=='llm':
                reps = model(llm_encodings=batch[0].to(device), forward_type='train')
            else:
                raise NameError('wrong')
    
            all_reps.append(reps['cls_rep_normalized'].detach().cpu())   # stay on original device

    return torch.cat(all_reps, dim=0)


class EmbeddingDataset(Dataset):
    """
    Holds a matrix of pre-computed sentence/document embeddings
    and a matching 1-D array/tensor of integer labels.
    """
    def __init__(self, embeddings: torch.Tensor, labels):
        # Store embeddings as float32 and labels as int64/long
        self.embeddings = embeddings.float()          # (N, D)
        self.labels = torch.as_tensor(labels,
                                      dtype=torch.long)  # (N,)

        assert len(self.embeddings) == len(self.labels), \
            "Embeddings and labels must have the same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns a single (embedding, label) tuple
        return self.embeddings[idx], self.labels[idx]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        encodings: a dict returned by tokenizer(..., return_tensors="pt")
        labels: a numpy array or list of integer labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # We clone and detach each tensor so that PyTorch doesn’t complain
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_data(data_df, tokenizer, max_length=128):
    """
    Simply tokenizes the list of texts in data_df['text'], returns a dict with keys
    'input_ids', 'attention_mask', (and possibly 'token_type_ids'—depending on the tokenizer).
    """
    return tokenizer(
        data_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def extract_layer_encodings(
    backbone: ModelWrapper,
    encodings: Dict[str, torch.Tensor],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Runs inputs through the frozen LLM encoder (up to 2/3 depth)
    and returns the last‐token residual hidden state for each example.

    Args:
        backbone:     your ModelWrapper with .encoder set to a ResidualWrapper
        encodings:    dict of tensors from tokenizer(..., return_tensors="pt")
                      must contain at least 'input_ids' and (optionally) 'attention_mask'
        batch_size:   how many examples per forward pass

    Returns:
        latent_encodings: Tensor, shape (N, hidden_size)
    """
    device = backbone.device
    # build a simple dataset of input_ids (+ attention_mask if present)
    input_ids = encodings['input_ids']
    attn_mask = encodings.get('attention_mask', None)

    if attn_mask is not None:
        ds = TensorDataset(input_ids, attn_mask)
    else:
        ds = TensorDataset(input_ids)

    loader = DataLoader(ds, batch_size=batch_size)
    backbone.eval()

    all_feats = []
    with torch.no_grad():
        for batch in loader:
            input_batch = batch[0].to(device)
            if attn_mask is not None:
                mask_batch = batch[1].to(device)
            else:
                mask_batch = None

            # forward_type='enc' grabs the residual from the hook
            feats = backbone(
                input_ids=input_batch,
                attention_mask=mask_batch,
                forward_type='collect_llm_encodings'
            )

            # if you get a full sequence (B, S, H), pick the last token
            if feats.ndim == 3:
                feats = feats[:, -1, :]

            all_feats.append(feats.cpu())

    latent_encodings = torch.cat(all_feats, dim=0)
    return latent_encodings


def load_domain(args):
    """
    1) Reads train.csv / test.csv for the given dataset (args.dataset).
    2) Instantiates a ModelWrapper(args.model).
    3) Loads (or tokenizes & caches) input tokens for train/test.
    4) Loads (or extracts & caches) intermediate layer encodings for train/test.
    5) Wraps everything into DataLoaders and returns a dictionary of utilities.
    """

    # ──────────────── STEP 1: Read CSVs ────────────────
    train_path = f'datasets/preprocess/{args.dataset}/train.csv'
    test_path  = f'datasets/preprocess/{args.dataset}/test.csv'
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find CSV files at {train_path} or {test_path}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Number of labels (assumes a column named 'label' in each CSV)
    num_labels = len(train_df['label'].value_counts())

    # ──────────────── STEP 2: Instantiate BackBone ────────────────
    model = ModelWrapper(model_name=args.model, device=args.device, no_llm_head=args.no_llm_head, prototype_dim=args.prototype_dim)

    # We can grab the config directly from the model object:
    config = model.config

    # ──────────────── STEP 3: Tokenize (or load cached) ────────────────
    # We'll cache under filenames that include both dataset and model name.
    train_tokens_path = f'datasets/preprocess/{args.dataset}/train_tokens_{args.model}.pt'
    test_tokens_path  = f'datasets/preprocess/{args.dataset}/test_tokens_{args.model}.pt'

    if os.path.exists(train_tokens_path) and os.path.exists(test_tokens_path):
        print("Loading tokenized inputs from cache...")
        train_tokens = torch.load(train_tokens_path, weights_only=False)
        test_tokens  = torch.load(test_tokens_path, weights_only=False)
    else:
        print("Tokenizing data (this may take a while)...")
        train_tokens = tokenize_data(train_df, model.tokenizer, max_length=args.input_size)
        test_tokens  = tokenize_data(test_df, model.tokenizer, max_length=args.input_size)
        os.makedirs(os.path.dirname(train_tokens_path), exist_ok=True)
        torch.save(train_tokens, train_tokens_path)
        torch.save(test_tokens, test_tokens_path)

    
    # ──────────────── STEP 4: Extract Layer Encodings (or load cached) for LLM models only
    if model.model_type=='llm':
        train_encodings_path = f'datasets/preprocess/{args.dataset}/train_encodings_{args.model}.pt'
        test_encodings_path  = f'datasets/preprocess/{args.dataset}/test_encodings_{args.model}.pt'

        if os.path.exists(train_encodings_path) and os.path.exists(test_encodings_path):
            print("Loading layer‐wise encodings from cache...")
            train_encodings = torch.load(train_encodings_path)
            test_encodings  = torch.load(test_encodings_path)
        else:
        
            print("Extracting layer‐wise encodings (this may take a while)...")
            train_encodings = extract_layer_encodings(model, train_tokens, batch_size=32)
            test_encodings  = extract_layer_encodings(model, test_tokens,  batch_size=32)

            # Save for next time:
            os.makedirs(os.path.dirname(train_encodings_path), exist_ok=True)
            torch.save(train_encodings, train_encodings_path)
            torch.save(test_encodings, test_encodings_path)

        train_dataset_enc = EmbeddingDataset(train_encodings, train_df['label'].values)
        test_dataset_enc  = EmbeddingDataset(test_encodings,  test_df['label'].values)
    else:
        train_dataset_enc = None
        test_dataset_enc  = None

    # ──────────────── STEP 5: Wrap Everything in Datasets & DataLoaders ────────────────
    train_dataset = CustomDataset(train_tokens, train_df['label'].values)
    test_dataset  = CustomDataset(test_tokens,  test_df['label'].values)

    
    data_utils = {
        "model": model,
        "tokenizer": model.tokenizer,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        # "train_labels": train_df['label'].values,
        # "test_labels": test_df['label'].values,
        "train_df": train_df,
        "test_df": test_df,
        "num_labels": num_labels,
        "config": config,
        "train_dataset_enc": train_dataset_enc,
        "test_dataset_enc": test_dataset_enc
    }

    return data_utils


def beam_search_subsequence(text, prototype, model, tokenizer, device, window_sizes=range(3, 11), batch_size=32):
    """
    Finds the best matching contiguous subsequence (of length 3-10) of 'text' whose BERT [CLS] embedding best matches 'prototype'.
    
    It computes the candidate scores in batches on the GPU. If none of the candidates has a lower distance (or higher similarity)
    than the full text, then it returns the full text's embedding and text.
    
    Args:
        text (str): Input text.
        prototype (torch.Tensor): Prototype embedding (1D tensor).
        bert_model: Pretrained BERT model.
        tokenizer: Corresponding tokenizer.
        device (str): Device string (e.g., 'cuda').
        window_sizes (iterable): Sequence lengths to consider (default: 3 to 10).
        batch_size (int): Batch size for encoding candidates.
        distance_func (str): Either 'l2' or 'cosine'.
    
    Returns:
        best_embedding (torch.Tensor): Embedding of the best candidate.
        best_candidate_text (str): The candidate text.
    """
    import torch
    import torch.nn.functional as F
    
    words = text.split()
    
    # Compute the full text embedding and normalize it.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        full_embedding = outputs['cls_rep_normalized']
    
    # full_embedding = F.normalize(full_embedding, p=2, dim=0)
    prototype = F.normalize(prototype, p=2, dim=0)
    
    # If text is too short, return full text.
    if len(words) < 3:
        return full_embedding[0], text
        
    original_score = torch.norm(full_embedding - prototype).item()
    print(f"Distance to original text: {original_score}, text = {text}")
    
    # Generate all candidate contiguous subsequences for window sizes 3 to 10.
    candidates = []
    for win_size in window_sizes:
        for i in range(len(words) - win_size + 1):
            candidates.append(" ".join(words[i:i+win_size]))
    
    # Batch encode all candidate texts.
    all_embeddings = []
    for i in range(0, len(candidates), batch_size):
        batch_texts = candidates[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            batch_embeddings = outputs['cls_rep_normalized']
        all_embeddings.append(batch_embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    distances = torch.norm(all_embeddings - prototype.unsqueeze(0), dim=1)
    best_idx = torch.argmin(distances).item()
    best_score = distances[best_idx].item()
    
    print(f"Best distance found in subsequences: {best_score}, text={candidates[best_idx]}")
        
    best_embedding = all_embeddings[best_idx]
    best_candidate_text = candidates[best_idx]
        
    return best_embedding, best_candidate_text

    


def evaluate_loaders(train_loader_non_random, val_loader, model, device, just_eval=False):
    
    model.eval()
    # Initialize counters for prototype activations per class
    num_total_prototypes = model.num_total_prototypes
    num_classes = model.num_labels
    
    # Create counters for each prototype x class combination
    # Shape: [num_total_prototypes, num_classes]
    train_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)
    val_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)
    
    if not just_eval:
        total_train_correct = 0
        for batch in tqdm(train_loader_non_random, desc="Evaluating training acc"):
            with torch.no_grad():
    
                # Get logits and losses from the model.
                if model.backbone.model_type=='bert':
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=batch['input_ids'].to(device) , attention_mask=batch['attention_mask'].to(device), forward_type='train')
                    
                # Get logits and losses from the model.
                elif model.backbone.model_type=='llm':
                    labels = batch[1].to(device)
                    outputs = model(llm_encodings=batch[0].to(device), forward_type='train')
                    
                all_similarities = outputs['acts']
                logits = outputs['logits']
                
                # Track which prototypes are maximally activated for each example
                max_proto_indices = torch.argmax(all_similarities, dim=1)  # [batch_size]
                
                # For each example, increment the counter for the max prototype and actual class
                for i, (proto_idx, class_idx) in enumerate(zip(max_proto_indices, labels)):
                    train_proto_class_activations[proto_idx, class_idx] += 1
                    
                preds = torch.argmax(logits, dim=1)
                total_train_correct += (preds == labels).sum().item()
        orig_train_acc = total_train_correct / len(train_loader_non_random.dataset)
        
        # Print activations per prototype and class
        print("\nTraining Set Prototype Activations by Class:")
        print("-" * 50)
        for p in range(num_total_prototypes):
            activation_str = f"Prototype {p}: "
            for c in range(num_classes):
                count = train_proto_class_activations[p, c].item()
                if count > 0:
                    activation_str += f"{count:.0f} times class {c}, "
            # Remove trailing comma and space
            activation_str = activation_str.rstrip(", ")
            print(activation_str)
    else:
        orig_train_acc = 0.0
    
    total_val_correct = 0
    for batch in tqdm(val_loader, desc="Evaluating validation acc"):
        with torch.no_grad():
            # Get logits and losses from the model.
            if model.backbone.model_type=='bert':
                labels = batch['labels'].to(device)
                outputs = model(input_ids=batch['input_ids'].to(device) , attention_mask=batch['attention_mask'].to(device), forward_type='train')
                
            # Get logits and losses from the model.
            elif model.backbone.model_type=='llm':
                labels = batch[1].to(device)
                outputs = model(llm_encodings=batch[0].to(device), forward_type='train')
                
            all_similarities = outputs['acts']
            logits = outputs['logits']
            
            # Track which prototypes are maximally activated for each example
            max_proto_indices = torch.argmax(all_similarities, dim=1)
            
            # For each example, increment the counter for the max prototype and actual class
            for i, (proto_idx, class_idx) in enumerate(zip(max_proto_indices, labels)):
                val_proto_class_activations[proto_idx, class_idx] += 1
                
            preds = torch.argmax(logits, dim=1)
            total_val_correct += (preds == labels).sum().item()
    orig_val_acc = total_val_correct / len(val_loader.dataset)
    
    # # Print activations per prototype and class for validation set
    # print("\nValidation Set Prototype Activations by Class:")
    # print("-" * 50)
    # for p in range(num_total_prototypes):
    #     activation_str = f"Prototype {p}: "
    #     for c in range(num_classes):
    #         count = val_proto_class_activations[p, c].item()
    #         if count > 0:
    #             activation_str += f"{count:.0f} times class {c}, "
    #     # Remove trailing comma and space
    #     activation_str = activation_str.rstrip(", ")
    #     print(activation_str)
    
    # # Check for completely inactive prototypes
    # train_inactive = torch.sum(torch.sum(train_proto_class_activations, dim=1) == 0).item() if not just_eval else 0
    # val_inactive = torch.sum(torch.sum(val_proto_class_activations, dim=1) == 0).item()
    
    # if train_inactive > 0 and not just_eval:
    #     print(f"\nWARNING: {train_inactive} prototypes never maximally activated in training set!")
    # if val_inactive > 0:
    #     print(f"\nWARNING: {val_inactive} prototypes never maximally activated in validation set!")
    
    # # Calculate diversity metrics
    # import numpy as np
    
    # # 1. Calculate entropy-based diversity for training set
    # if not just_eval:
    #     train_total_activations = torch.sum(train_proto_class_activations).item()
    #     if train_total_activations > 0:
    #         # Calculate probability of each prototype being activated
    #         train_proto_probs = torch.sum(train_proto_class_activations, dim=1) / train_total_activations
    #         # Filter out zeros to avoid log(0)
    #         train_proto_probs = train_proto_probs[train_proto_probs > 0].cpu().numpy()
    #         # Calculate entropy
    #         train_entropy = -np.sum(train_proto_probs * np.log(train_proto_probs))
    #         # Normalize by maximum possible entropy
    #         max_entropy = np.log(num_total_prototypes)
    #         train_diversity_score = train_entropy / max_entropy if max_entropy > 0 else 0.0
    #         print(f"\nTraining Set Diversity Score: {train_diversity_score:.4f} (0-1 scale, higher is better)")
    
    # # 2. Calculate entropy-based diversity for validation set
    # val_total_activations = torch.sum(val_proto_class_activations).item()
    # if val_total_activations > 0:
    #     # Calculate probability of each prototype being activated
    #     val_proto_probs = torch.sum(val_proto_class_activations, dim=1) / val_total_activations
    #     # Filter out zeros to avoid log(0)
    #     val_proto_probs = val_proto_probs[val_proto_probs > 0].cpu().numpy()
    #     # Calculate entropy
    #     val_entropy = -np.sum(val_proto_probs * np.log(val_proto_probs))
    #     # Normalize by maximum possible entropy
    #     max_entropy = np.log(num_total_prototypes)
    #     val_diversity_score = val_entropy / max_entropy if max_entropy > 0 else 0.0
    #     print(f"Validation Set Diversity Score: {val_diversity_score:.4f} (0-1 scale, higher is better)")
    
    # # 3. Calculate a Gini coefficient as an alternate metric (lower is more equal/diverse)
    # if not just_eval:
    #     train_counts = torch.sum(train_proto_class_activations, dim=1).cpu().numpy()
    #     if np.sum(train_counts) > 0:
    #         train_counts = np.sort(train_counts)
    #         n = len(train_counts)
    #         cumsum = np.cumsum(train_counts)
    #         train_gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    #         print(f"Training Set Gini Coefficient: {train_gini:.4f} (0-1 scale, lower is better)")
    
    # val_counts = torch.sum(val_proto_class_activations, dim=1).cpu().numpy()
    # if np.sum(val_counts) > 0:
    #     val_counts = np.sort(val_counts)
    #     n = len(val_counts)
    #     cumsum = np.cumsum(val_counts)
    #     val_gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    #     print(f"Validation Set Gini Coefficient: {val_gini:.4f} (0-1 scale, lower is better)")
    
    model.train()
    return orig_train_acc, orig_val_acc


# def evaluate_loaders(train_loader_non_random, val_loader, model, device, just_eval=False):
#     import torch
#     from tqdm import tqdm

#     model.eval()
#     # Initialize counters for prototype activations per class
#     num_total_prototypes = model.num_total_prototypes
#     num_classes = model.num_labels

#     # Create counters for each prototype x class combination
#     # Shape: [num_total_prototypes, num_classes]
#     train_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)
#     val_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)

#     # Initialize per-class accuracy counters
#     val_class_correct = torch.zeros(num_classes, device=device)
#     val_class_total = torch.zeros(num_classes, device=device)

#     if not just_eval:
#         total_train_correct = 0
#         for batch in tqdm(train_loader_non_random, desc="Evaluating training acc"):
#             with torch.no_grad():
#                 # Get logits and activations
#                 if model.backbone.model_type == 'bert':
#                     labels = batch['labels'].to(device)
#                     outputs = model(
#                         input_ids=batch['input_ids'].to(device),
#                         attention_mask=batch['attention_mask'].to(device),
#                         forward_type='train'
#                     )
#                 elif model.backbone.model_type == 'llm':
#                     labels = batch[1].to(device)
#                     outputs = model(
#                         llm_encodings=batch[0].to(device),
#                         forward_type='train'
#                     )
#                 all_similarities = outputs['acts']
#                 logits = outputs['logits']

#                 # Track which prototypes are maximally activated for each example
#                 max_proto_indices = torch.argmax(all_similarities, dim=1)
#                 for proto_idx, class_idx in zip(max_proto_indices, labels):
#                     train_proto_class_activations[proto_idx, class_idx] += 1

#                 # Track overall train accuracy
#                 preds = torch.argmax(logits, dim=1)
#                 total_train_correct += (preds == labels).sum().item()
#         orig_train_acc = total_train_correct / len(train_loader_non_random.dataset)

#         # Print activations per prototype and class
#         print("\nTraining Set Prototype Activations by Class:")
#         print("-" * 50)
#         for p in range(num_total_prototypes):
#             activation_str = f"Prototype {p}: "
#             for c in range(num_classes):
#                 count = train_proto_class_activations[p, c].item()
#                 if count > 0:
#                     activation_str += f"{int(count)} times class {c}, "
#             print(activation_str.rstrip(", "))
#     else:
#         orig_train_acc = 0.0

#     total_val_correct = 0
#     for batch in tqdm(val_loader, desc="Evaluating validation acc"):
#         with torch.no_grad():
#             # Get logits and activations
#             if model.backbone.model_type == 'bert':
#                 labels = batch['labels'].to(device)
#                 outputs = model(
#                     input_ids=batch['input_ids'].to(device),
#                     attention_mask=batch['attention_mask'].to(device),
#                     forward_type='train'
#                 )
#             elif model.backbone.model_type == 'llm':
#                 labels = batch[1].to(device)
#                 outputs = model(
#                     llm_encodings=batch[0].to(device),
#                     forward_type='train'
#                 )
#             all_similarities = outputs['acts']
#             logits = outputs['logits']

#             # Track which prototypes are maximally activated for each example
#             max_proto_indices = torch.argmax(all_similarities, dim=1)
#             for proto_idx, class_idx in zip(max_proto_indices, labels):
#                 val_proto_class_activations[proto_idx, class_idx] += 1

#             # Track per-class validation accuracy
#             preds = torch.argmax(logits, dim=1)
#             for pred, label in zip(preds, labels):
#                 val_class_total[label] += 1
#                 if pred == label:
#                     val_class_correct[label] += 1

#             total_val_correct += (preds == labels).sum().item()
#     orig_val_acc = total_val_correct / len(val_loader.dataset)

#     # Print overall validation accuracy
#     print(f"\nValidation Accuracy: {orig_val_acc:.4f}")

#     # Compute per-class accuracies
#     per_class_acc = val_class_correct / val_class_total.clamp(min=1)
#     # Sort classes by accuracy ascending
#     sorted_acc, sorted_indices = torch.sort(per_class_acc)
#     worst_k = min(20, num_classes)
#     worst_indices = sorted_indices[:worst_k]

#     # Print worst classes
#     print(f"\nWorst {worst_k} Classes by Accuracy:")
#     print("-" * 40)
#     for idx in worst_indices:
#         acc = per_class_acc[idx].item()
#         total = int(val_class_total[idx].item())
#         correct = int(val_class_correct[idx].item())
#         print(f"Class {int(idx)}: {acc:.4f} ({correct}/{total})")

#     model.train()
#     return orig_train_acc, orig_val_acc



from sklearn.cluster import KMeans
import torch
from tqdm import tqdm

from sklearn.cluster import KMeans
import torch
from tqdm import tqdm

def get_unsupervised_prototypes(backbone, dataloader, num_labels, num_prototypes, device, max_batches=30, noise_scale=1e-3):
    """
    Collect representations, then for each class:
      - If M ≥ P: run K-means with P clusters.
      - If 0 < M < P: run K-means with M clusters, then duplicate the first prototype
        with a bit of Gaussian noise to pad up to P.
      - If M = 0: sample P reps at random from the full dataset.
    Returns a tensor of shape (num_labels * num_prototypes, D) where the first P rows
    are for class 0, the next P for class 1, etc.
    
    noise_scale controls the standard deviation of the padding‐noise.
    """
    backbone.eval()
    reps_list, labels_list = [], []

    if backbone.model_type == 'llm':
        max_batches = 200

    # 1) Collect all reps + labels
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting reps")):
            if batch_idx >= max_batches:
                break

            if backbone.model_type == 'bert':
                reps = backbone(
                    input_ids      = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device),
                    forward_type   = 'train'
                )
            elif backbone.model_type == 'llm':
                reps = backbone(
                    llm_encodings = batch[0].to(device),
                    forward_type  = 'train'
                )
            else:
                raise TypeError(f"Unsupported model_type: {backbone.model_type}")

            reps_list.append(reps.cpu())
            labels_list.append(batch['labels'].cpu())

    all_reps   = torch.cat(reps_list,   dim=0)  # [N, D]
    all_labels = torch.cat(labels_list, dim=0)  # [N]

    # 2) Per-class prototypes
    centroids_per_class = []
    for class_idx in range(num_labels):
        mask       = (all_labels == class_idx)
        class_reps = all_reps[mask]            # [M, D]
        M          = class_reps.size(0)
        P          = num_prototypes

        if M >= P:
            # enough data → straight K-means(P)
            kmeans = KMeans(n_clusters=P, n_init='auto', random_state=0)
            kmeans.fit(class_reps.numpy())
            cents = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        elif M > 0:
            # underpopulated → kmeans(M) + padded duplicates with noise
            kmeans = KMeans(n_clusters=M, n_init='auto', random_state=0)
            kmeans.fit(class_reps.numpy())
            base_cents = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [M, D]

            # Duplicate the first centroid with jitter
            first = base_cents[0:1]                             # [1, D]
            pad   = first.repeat(P - M, 1)                      # [P-M, D]
            noise = torch.randn_like(pad) * noise_scale         # small Gaussian noise
            cents = torch.cat([base_cents, pad + noise], dim=0) # [P, D]

        else:
            # no examples → random samples from the full set
            N = all_reps.size(0)
            if N == 0:
                raise ValueError("No data available to initialize any prototypes.")
            idxs = torch.randint(0, N, (P,))
            cents = all_reps[idxs]  # [P, D]

        centroids_per_class.append(cents)

    # 3) Stack so rows [0:P] → class 0, [P:2P] → class 1, etc.
    centroids = torch.cat(centroids_per_class, dim=0)  # [num_labels * P, D]
    return centroids



# def get_unsupervised_prototypes(backbone, dataloader, num_labels, num_prototypes, device, max_batches=30):
#     backbone.eval()
#     all_reps = []

#     if backbone.model_type=='llm':
#         max_batches = 200

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting data for unsupervised prototype init")):
#             if batch_idx >= max_batches:
#                 break

#             label = batch['label']

#             if backbone.model_type == 'bert':
#                 reps = backbone(input_ids=batch['input_ids'].to(device),
#                                 attention_mask=batch['attention_mask'].to(device),
#                                 forward_type='train')
#             elif backbone.model_type == 'llm':
#                 reps = backbone(llm_encodings=batch[0].to(device), forward_type='train')
#             else:
#                 raise TypeError('Unsupported model_type in prototype initialization')

#             all_reps.append(reps.cpu())

#     all_reps_tensor = torch.cat(all_reps, dim=0)  # [N, D]
    
#     # Run k-means on the latent space to get diverse centers
#     kmeans = KMeans(n_clusters=num_prototypes, n_init='auto', random_state=0)
#     kmeans.fit(all_reps_tensor.numpy())
    
#     centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [num_prototypes, D]
#     return centroids





    