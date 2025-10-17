import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
import pickle
import itertools
import json
from typing import List, Tuple

import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.decomposition import PCA  # for 2D scatter
from sklearn.model_selection import train_test_split

from functions import *
from models import *


def clean_gpus() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clean_gpus()


# -------------------------- FOCAL LOSS -------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("alpha_tensor", None)
        if isinstance(alpha, torch.Tensor):
            self.alpha_tensor = alpha
            self.alpha = None
        else:
            self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.alpha_tensor is not None:
            fl = self.alpha_tensor[targets] * fl
        elif self.alpha is not None and self.alpha != 1.0:
            fl = self.alpha * fl
        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl


# ---------------------- METRICS / EVALUATION -------------------- #
def _compute_metrics(preds, labels, num_classes):
    acc = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    per_class = []
    for c in range(num_classes):
        per_class.append({
            "class": int(c),
            "precision": float(per_class_prec[c]),
            "recall": float(per_class_rec[c]),
            "f1": float(per_class_f1[c]),
            "support": int(per_class_support[c])
        })
    return {
        "accuracy": float(acc),
        "macro": {"precision": float(prec_macro), "recall": float(rec_macro), "f1": float(f1_macro)},
        "micro": {"precision": float(prec_micro), "recall": float(rec_micro), "f1": float(f1_micro)},
        "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)},
        "per_class": per_class
    }

def _summarize_metrics(tag, m):
    return (f"{tag} | "
            f"Acc: {m['accuracy']:.4f} | "
            f"Macro P/R/F1: {m['macro']['precision']:.4f}/"
            f"{m['macro']['recall']:.4f}/"
            f"{m['macro']['f1']:.4f} | "
            f"Micro F1: {m['micro']['f1']:.4f}")

def evaluate_loaders(train_loader_non_random, val_loader, model, device, num_classes,
                     track_prototype_activations=False):
    model.eval()
    train_proto_class_activations = None
    val_proto_class_activations = None
    if track_prototype_activations:
        num_total_prototypes = model.num_total_prototypes
        train_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)
        val_proto_class_activations = torch.zeros((num_total_prototypes, num_classes), device=device)

    def run_eval(loader, split_name):
        all_preds, all_labels = [], []
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            with torch.no_grad():
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    forward_type='train'
                )

                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)

                if track_prototype_activations:
                    all_similarities = outputs['acts']
                    max_proto_indices = torch.argmax(all_similarities, dim=1)
                    if split_name == "train":
                        for proto_idx, class_idx in zip(max_proto_indices, labels):
                            train_proto_class_activations[proto_idx, class_idx] += 1
                    else:
                        for proto_idx, class_idx in zip(max_proto_indices, labels):
                            val_proto_class_activations[proto_idx, class_idx] += 1

                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
        return _compute_metrics(all_preds, all_labels, num_classes)

    train_metrics = None
    if train_loader_non_random is not None:
        train_metrics = run_eval(train_loader_non_random, "train")
    val_metrics = run_eval(val_loader, "val")

    if train_metrics is not None:
        print(_summarize_metrics("Train", train_metrics))
    print(_summarize_metrics("Val", val_metrics))

    if track_prototype_activations:
        if train_proto_class_activations is not None:
            train_proto_class_activations = train_proto_class_activations.detach().cpu().tolist()
        if val_proto_class_activations is not None:
            val_proto_class_activations = val_proto_class_activations.detach().cpu().tolist()

    model.train()
    return {
        "train": train_metrics,
        "val": val_metrics,
        "train_proto_class_activations": train_proto_class_activations,
        "val_proto_class_activations": val_proto_class_activations
    }


# ----------------------- CLASS WEIGHTS (optional) ------------------ #
def compute_class_weights(train_loader_non_random, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in tqdm(train_loader_non_random, desc="Computing class weights"):
        if isinstance(batch, dict) and 'labels' in batch:
            labels = batch['labels']
        else:
            labels = batch[1]
        labels = labels.detach().cpu()
        binc = torch.bincount(labels, minlength=num_classes)
        counts += binc
    counts = counts.float()
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights.to(device)


# ----------------------- EMBEDDINGS FOR BATCH MODE -------------- #
def compute_embeddings(model, train_loader_non_random, device=None):
    """
    Compute embeddings for full documents in batch mode.
    Returns [N_docs, H] tensor on CPU.
    """
    model.eval()
    dev = device or (model.backbone.device if hasattr(model.backbone, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    all_reps = []
    print('getting embeddings (batch mode)...')
    with torch.no_grad():
        for batch in tqdm(train_loader_non_random):
            out = model(
                input_ids=batch['input_ids'].to(dev),
                attention_mask=batch['attention_mask'].to(dev),
                forward_type='train'
            )
            all_reps.append(out['cls_rep_normalized'].detach().cpu())
    return torch.cat(all_reps, dim=0)


# ---------------------------- MAIN EXPERIMENT ------------------- #
def run_experiment(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data/domain/backbone
    data_utils = load_domain(args)
    backbone = data_utils['model']
    num_labels = data_utils['num_labels']
    total_num_prototypes = args.num_protos * num_labels
    print('num labels:', num_labels)

    # Split training data into train and validation (95/5 split)
    train_dataset = data_utils['train_dataset']
    
    # Create indices for train/val split
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    
    # Stratified split to maintain class distribution
    if backbone.model_type == 'bert':
        labels = [train_dataset[i]['labels'].item() if isinstance(train_dataset[i]['labels'], torch.Tensor) 
                 else train_dataset[i]['labels'] for i in range(dataset_size)]
    else:
        labels = [train_dataset[i][1].item() if isinstance(train_dataset[i][1], torch.Tensor) 
                 else train_dataset[i][1] for i in range(dataset_size)]
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.05, stratify=labels, random_state=seed
    )
    
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    # Create samplers for DataLoaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=128, sampler=val_sampler)
    test_loader = DataLoader(data_utils['test_dataset'], batch_size=128, shuffle=False)
    
    # For non-random access to full train data (needed for prototype initialization and projection)

    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    train_loader_non_random = DataLoader(train_subset, batch_size=128, shuffle=False)  

    # Initialize prototypes (unsupervised init on subset)
    sample_size_for_protos = 200
    proto_init = get_unsupervised_prototypes(
        backbone, train_loader, num_labels, args.num_protos, args.device, max_batches=sample_size_for_protos
    )  # [P, H]

    model = LMProtoNet(
        data_utils['model'],
        num_labels=num_labels,
        num_protos_per_class=args.num_protos,
        init_prototypes=proto_init,
    )

    device = args.device if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Use class weights for focal loss (now default)
    alpha_for_focal = args.focal_alpha
    if args.use_class_weights:
        class_weights = compute_class_weights(train_loader_non_random, num_labels, device)
        alpha_for_focal = class_weights
        print(f"Using class weights for focal loss: {class_weights}")
    classif_loss_fn = FocalLoss(alpha=alpha_for_focal, gamma=args.focal_gamma, reduction="mean")

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, eps=1e-8)

    # Track best validation accuracy
    best_val_acc = 0.0
    best_val_f1 = 0.0
    train_losses = []
    l_p1_losses = []
    l_p2_losses = []
    l_p3_losses = []
    val_acc_history = []
    val_f1_history = []

    # Path for saving best weights
    weight_dir_pre = f'weights/pre_projection_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}.pt'
    os.makedirs(os.path.dirname(weight_dir_pre), exist_ok=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_l_p1 = 0.0
        total_l_p2 = 0.0
        total_l_p3 = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            # -------- Main model forward on full batch (for task loss) -------- #
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                forward_type='train'
            )

            l_p1, l_p2, l_p3 = outputs['l_p1'], outputs['l_p2'], outputs['l_p3']
            logits = outputs['logits']
            clf_loss = classif_loss_fn(logits, labels)

            # Final loss (removed l_p4 completely)
            loss = clf_loss + (l_p1 * args.l_p1_weight) + (l_p2 * args.l_p2_weight) + (l_p3 * args.l_p3_weight)

            loss.backward()
            optimizer.step()

            global_step += 1

            total_loss += clf_loss.item()
            total_l_p1 += l_p1.item()
            total_l_p2 += l_p2.item()
            total_l_p3 += l_p3.item()

            pb_postfix = {
                'l_clf': clf_loss.item(),
                'l_p1': l_p1.item(),
                'l_p2': l_p2.item(),
                'l_p3': l_p3.item(),
            }
            progress_bar.set_postfix(pb_postfix)

        # ---- epoch logs ----
        avg_train_loss = total_loss / len(train_loader)
        avg_l_p1 = total_l_p1 / len(train_loader)
        avg_l_p2 = total_l_p2 / len(train_loader)
        avg_l_p3 = total_l_p3 / len(train_loader)
        train_losses.append(avg_train_loss)
        l_p1_losses.append(avg_l_p1)
        l_p2_losses.append(avg_l_p2)
        l_p3_losses.append(avg_l_p3)

        # Evaluate on validation set
        metrics_epoch = evaluate_loaders(
            train_loader_non_random=None,
            val_loader=val_loader,
            model=model,
            device=device,
            num_classes=num_labels,
            track_prototype_activations=False
        )
        val_acc = metrics_epoch['val']['accuracy']
        val_macro_f1 = metrics_epoch['val']['macro']['f1']
        val_acc_history.append(val_acc)
        val_f1_history.append(val_macro_f1)
        
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), weight_dir_pre)
            print(f"  → New best validation accuracy: {val_acc:.4f} - Model saved!")
        else:
            print(f"  → Val accuracy: {val_acc:.4f} (best: {best_val_acc:.4f})")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")




    # ---------------- Plot training/validation metrics ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    
    # Plot training losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Classification Loss', linewidth=2)
    ax1.plot(epochs, l_p1_losses, 'r--', label='L_p1 Loss', linewidth=1.5, alpha=0.7)
    ax1.plot(epochs, l_p2_losses, 'g--', label='L_p2 Loss', linewidth=1.5, alpha=0.7)
    ax1.plot(epochs, l_p3_losses, 'm--', label='L_p3 Loss', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax2.plot(epochs, val_acc_history, 'g-', label='Val Accuracy', linewidth=2)
    ax2.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = f'plots/training_metrics_seed{args.seed}_{args.dataset}_{args.model}.png'
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training metrics plot saved to {loss_plot_path}")



    
    # Load best model weights
    print(f"Loading best model from {weight_dir_pre}")
    model.load_state_dict(torch.load(weight_dir_pre, map_location=device))
    model.to(device)

    # ---------------- Evaluate original prototypes on test set ----------------
    original_prototypes = model.prototypes.clone().detach().cpu()
    original_prototypes_normed = F.normalize(original_prototypes, p=2, dim=1)

    orig_metrics = evaluate_loaders(
        train_loader_non_random=None,
        val_loader=test_loader,
        model=model,
        device=device,
        num_classes=num_labels,
        track_prototype_activations=False
    )
    orig_val = orig_metrics['val']
    print("\n=== Original Prototypes (Test Set) ===")
    print(_summarize_metrics("Test", orig_val))

    # ---------------- Prepare full-document projection ----------------
    model.eval()
    # Use full training dataset for embeddings
    full_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    train_embeddings = compute_embeddings(
        model,
        train_loader_non_random=full_train_loader,
        device=device
    ).to('cpu')
    

    train_embeddings_normalized = F.normalize(train_embeddings, p=2, dim=1)
    
    
    
    # Save embeddings
    os.makedirs(f'datasets/preprocess/{args.dataset}', exist_ok=True)
    torch.save(train_embeddings, f'datasets/preprocess/{args.dataset}/train_encodings_{args.model}_{args.seed}.pt')

    

    # Get train_df subset for prototype texts
    # train_df_subset = data_utils['train_df'].iloc[train_indices].reset_index(drop=True)
    train_df_full = data_utils['train_df']


    projected_embeddings_list_full = []
    full_proto_texts_full = []
    original_prototypes_normed_local = F.normalize(model.prototypes.clone().detach().cpu(), p=2, dim=1)
    for i in tqdm(range(total_num_prototypes), desc="Projecting prototypes with full examples"):
        proto_norm = original_prototypes_normed_local[i].unsqueeze(0)
        similarities = torch.matmul(train_embeddings_normalized, proto_norm.T).squeeze(1)
        best_instance_idx = torch.argmax(similarities).item()
        projected_embeddings_list_full.append(train_embeddings_normalized[best_instance_idx])
        full_proto_texts_full.append(train_df_full['text'].iloc[best_instance_idx])

    new_prototypes_full_normalized = torch.stack(projected_embeddings_list_full)

    # Evaluate full-document projection
    with torch.no_grad():
        old = model.prototypes.clone()
        model.prototypes.copy_(new_prototypes_full_normalized.to(device))

    proj_full_metrics = evaluate_loaders(
        train_loader_non_random=None,
        val_loader=test_loader,
        model=model,
        device=device,
        num_classes=num_labels,
        track_prototype_activations=False
    )
    proj_full_val = proj_full_metrics['val']
    print("\n=== Projected Prototypes (Full Document, Test Set) ===")
    print(_summarize_metrics("Test", proj_full_val))

    # Drift wrt original
    proj_cpu = model.prototypes.clone().detach().cpu()
    proj_normed = F.normalize(proj_cpu, p=2, dim=1)
    proj_l2_full = torch.norm(original_prototypes_normed - proj_normed, dim=1).mean().item()
    proj_cossim_full = torch.sum(original_prototypes_normed * proj_normed, dim=1).mean().item()
    print(f"Projection Drift [full]: Avg L2={proj_l2_full:.4f}, Avg CosSim={proj_cossim_full:.4f}")

    # Save weights after projection
    weight_dir_post = f'weights/post_projection_full_{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}.pt'
    os.makedirs(os.path.dirname(weight_dir_post), exist_ok=True)
    torch.save(model.state_dict(), weight_dir_post)
    print(f"Projected model saved to {weight_dir_post}")

    # Uniqueness check
    projected = model.prototypes.detach().cpu()
    unique = torch.unique(projected, dim=0)
    total = projected.size(0)
    num_unique = unique.size(0)
    print(f"Projected prototypes unique count [full]: {num_unique}/{total} are unique")

    # Restore original prototypes
    with torch.no_grad():
        model.prototypes.copy_(old.to(device))

    # ---------------- Modified scatter plot: original vs projected prototypes with arrows ----------------
    model.train()  # to use forward_type='train'
    normal_feats = []

    batches_to_sample = 100
    sampled = 0
    for batch in tqdm(train_loader_non_random, desc="Collecting embeddings for scatter"):
        with torch.no_grad():
            # Normal full-seq embeddings
            out_full = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                forward_type='train'
            )
            
            normal_feats.append(out_full['cls_rep_normalized'].detach().cpu())

        sampled += 1
        if sampled >= batches_to_sample:
            break

    normal_feats = torch.cat(normal_feats, dim=0).numpy()
    
    # Get original and projected prototype features (both normalized)
    original_proto_feats = original_prototypes_normed.numpy()
    projected_proto_feats = new_prototypes_full_normalized.numpy()

    # PCA to 2D on the union
    union = np.concatenate([normal_feats, original_proto_feats, projected_proto_feats], axis=0)
    pca = PCA(n_components=2, random_state=0)
    union_2d = pca.fit_transform(union)

    n_normal = normal_feats.shape[0]
    n_original_proto = original_proto_feats.shape[0]
    n_projected_proto = projected_proto_feats.shape[0]

    Xn = union_2d[:n_normal]
    Xo = union_2d[n_normal:n_normal + n_original_proto]  # original prototypes
    Xp = union_2d[n_normal + n_original_proto:]  # projected prototypes

    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(Xn[:, 0], Xn[:, 1], s=8, alpha=0.4, label='Training data', c='lightgray')
    plt.scatter(Xo[:, 0], Xo[:, 1], s=50, alpha=0.9, label='Original prototypes', c='red', marker='o', edgecolors='darkred', linewidth=1)
    plt.scatter(Xp[:, 0], Xp[:, 1], s=50, alpha=0.9, label='Projected prototypes (full data)', c='blue', marker='s', edgecolors='darkblue', linewidth=1)
    
    # Add arrows from original to projected prototypes
    for i in range(len(Xo)):
        plt.annotate('', xy=(Xp[i, 0], Xp[i, 1]), xytext=(Xo[i, 0], Xo[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1.5))

    plt.title(f'Prototype projection visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = f'plots/prototype_projection_seed{args.seed}_{args.dataset}_{args.model}.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Prototype projection plot saved to {out_path}")

    # ---------------- Package results ----------------
    results = {
        'seed': seed,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'orig': orig_val,
        'proj_full': proj_full_val,
        'proj_l2_full': proj_l2_full,
        'proj_cossim_full': proj_cossim_full,
        'full_proto_texts_full': full_proto_texts_full,
        'num_prototypes': int(total_num_prototypes)
    }
    return results


def save_results_csv_and_json(args, all_results):
    records = []
    per_class_payload = { "orig": {}, "proj_full": {} }

    for r in all_results:
        seed = r['seed']
        orig = r['orig']
        proj_full = r['proj_full']

        rec = {
            'seed': seed,
            'best_val_acc': r['best_val_acc'],
            'best_val_f1': r['best_val_f1'],
            'orig_test_acc': orig['accuracy'],
            'orig_test_precision_macro': orig['macro']['precision'],
            'orig_test_recall_macro': orig['macro']['recall'],
            'orig_test_f1_macro': orig['macro']['f1'],
            'proj_full_test_acc': proj_full['accuracy'],
            'proj_full_test_precision_macro': proj_full['macro']['precision'],
            'proj_full_test_recall_macro': proj_full['macro']['recall'],
            'proj_full_test_f1_macro': proj_full['macro']['f1'],
            'proj_l2_full': r['proj_l2_full'],
            'proj_cossim_full': r['proj_cossim_full'],
            'full_proto_texts_full': json.dumps(r['full_proto_texts_full']),
        }

        records.append(rec)
        per_class_payload["orig"][str(seed)] = r['orig']['per_class']
        per_class_payload["proj_full"][str(seed)] = proj_full['per_class']

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    json_path = f"data/{args.model}_{args.dataset}_protos{args.num_protos}_seed{args.seed}_per_class.json"
    with open(json_path, "w") as f:
        json.dump(per_class_payload, f, indent=2)
    print(f"Per-class metrics saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_protos', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imdb')
    parser.add_argument('--model', type=str, default='bert')  # electra

    parser.add_argument('--prototype_dim', type=int, default=256)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--l_p1_weight', type=float, default=0.1)
    parser.add_argument('--l_p2_weight', type=float, default=0.01)
    parser.add_argument('--l_p3_weight', type=float, default=0.01)

    parser.add_argument('--focal_alpha', type=float, default=1.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_class_weights', action='store_true', default=True)  # Changed to True by default

    args = parser.parse_args()

    all_results = []
    print(f"\nRunning experiment with seed {args.seed}")
    res = run_experiment(args, args.seed)
    all_results.append(res)

    save_results_csv_and_json(args, all_results)
    
    
