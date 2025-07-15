"""
TODO List:
* Find minimal amount of prototypes needed to match this accuracy.
* Do you need the polarized layer to get this accuarcy? -- yes -- it's quicker and more accurate
    * Check 1 prototype for 20newsgruops and dbpedia

* Find best hyperparameters for the baseline (and our method)
* Run synthetic tests using actual test instances for the prototypes? (or just show chatgpt a few instances to make sure they're reasonable prototypes)

python src/train_prototype_models.py --dataset='trec' --num_protos=1 --model='bert' --device='cuda:0' --num_epochs=30

===> trec (needs 3-5-10 prototypes -- needs 25 epochs on modern bert)
bert: 0.90         (quick with 3 prototypes)
electra: 0.88      (quick with 3 prototypes)
modern_bert: 0.81  (needs 25 epochs with 10 prototypes)
roberta: 0.92      (gets here at 30 epochs and 3 prototypes)
mpnet: 0.91        (quick with 3 prototypes)

===> 20newsgroups
bert: 0.83 (3 prototypes, 12 epochs) *(77%)
electra: 0.78 (3 prototypes, 12 epochs) 
modern_bert: 0.78 (3 prototypes, 12 epochs)
roberta: 0.822 (3 prototypes, 12 epochs)
mpnet: 0.84 (3 prototypes, 12 epochs) ***7

===> dbpedia
bert: 0.89 (3 epochs, 3 prototypes) ***
electra: 0.86 (3 epochs, 3 prototypes) ***8
modern_bert: 0.86 (3 epochs, 3 prototypes) ***10
roberta: 0.89 (3 epochs, 3 prototypes) ***10
mpnet: 0.89 (3 epochs, 3 prototypes) ***5

*** 1 prototype works
"""

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
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
# from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW

# from src.llm_proto_optim import optimize_prototypes
from functions import *
from models import *


import gc
import torch
def clean_gpus() -> None:
    gc.collect()
    torch.cuda.empty_cache() 
    
clean_gpus()



# --- run_experiment Function ---
def run_experiment(args, seed):
    
    # Set random seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset=='trec':
        # Add loss weight arguments
        args.l_p1_weight = 1.0
        args.l_p2_weight = 0.1
        args.l_p3_weight = 0.1
    elif args.dataset=='20newsgroups':
        # Add loss weight arguments
        args.l_p1_weight = 1.0
        args.l_p2_weight = 0.1
        args.l_p3_weight = 0.1
    elif args.dataset=='dbpedia':
        # Add loss weight arguments
        args.l_p1_weight = 1.0
        args.l_p2_weight = 0.1
        args.l_p3_weight = 0.1
    else:
        raise NameError('wrong dataset name')
        
    # Load domain and data (including text_column).
    data_utils = load_domain(args)
    backbone = data_utils['model']
    total_num_prototypes = args.num_protos * data_utils['num_labels']

    print('num labels:', data_utils['num_labels'])

    if backbone.model_type=='bert':
        train_loader = DataLoader(data_utils['train_dataset'], batch_size=32, shuffle=True)
        test_loader  = DataLoader(data_utils['test_dataset'],  batch_size=128, shuffle=False)
        train_loader_non_random = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)
        train_loader_ids = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)
        test_loader_ids  = DataLoader(data_utils['test_dataset'],  batch_size=128, shuffle=False)
        
    elif backbone.model_type=='llm':
        train_loader = DataLoader(data_utils['train_dataset_enc'], batch_size=32, shuffle=True)
        test_loader  = DataLoader(data_utils['test_dataset_enc'],  batch_size=128, shuffle=False)
        train_loader_non_random = DataLoader(data_utils['train_dataset_enc'], batch_size=128, shuffle=False)
        train_loader_ids = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)
        test_loader_ids  = DataLoader(data_utils['test_dataset'],  batch_size=128, shuffle=False)
    else:
        raise TypeError('wrong model type')


        backbone, dataloader, num_labels, num_prototypes, device, max_batches


    sample_size_for_protos = 1e100
    proto_init = get_unsupervised_prototypes(backbone, train_loader, data_utils['num_labels'], args.num_protos, args.device, max_batches=sample_size_for_protos)
    
    model = LMProtoNet(data_utils['model'],
                       num_labels=data_utils['num_labels'],
                       num_protos_per_class=args.num_protos,
                       init_prototypes=proto_init,
                      )    
        
    device = args.device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    optimizer = AdamW(model.parameters(),lr=3e-4,weight_decay=0.01,eps=1e-8)
    classif_loss_fn = nn.CrossEntropyLoss()
    
    best_acc = 0.
    train_losses = []
    l_p1_losses = []
    l_p2_losses = []
    l_p3_losses = []
    val_accuracies = []
        
    # Compute total training steps across all epochs
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    
    
    # Training loop.
    global_step = 0  # Tracks total number of batches seen
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_l_p1 = 0
        total_l_p2 = 0
        total_l_p3 = 0
        
        # Use tqdm for progress bar and loss display
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in progress_bar:

            optimizer.zero_grad() 

            # Get logits & labels
            if model.backbone.model_type=='bert':
                labels = batch['labels'].to(device)
                outputs = model(input_ids=batch['input_ids'].to(device) , attention_mask=batch['attention_mask'].to(device), forward_type='train')
            elif model.backbone.model_type=='llm':
                labels = batch[1].to(device)
                outputs = model(llm_encodings=batch[0].to(device), forward_type='train')
            else:
                raise NameError('wrong')
                
            l_p1, l_p2, l_p3 = outputs['l_p1'], outputs['l_p2'], outputs['l_p3']
            clf_loss = classif_loss_fn(outputs['logits'], labels)

            # Our version does not use these losses
            if not args.baseline:
                l_p1 *= 0.
                l_p2 *= 0.
            
            # Final loss (using hyperparams from command line arguments)
            loss = clf_loss + (l_p1 * args.l_p1_weight) + (l_p2 * args.l_p2_weight) + (l_p3 * args.l_p3_weight)                
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Update running totals
            total_loss += clf_loss.item()
            total_l_p1 += l_p1.item()
            total_l_p2 += l_p2.item()
            total_l_p3 += l_p3.item()
            
            # Update progress bar with current batch losses
            progress_bar.set_postfix({
                'l_clf': clf_loss.item(),
                'l_p1': l_p1.item(), 
                'l_p2': l_p2.item(),
                'l_p3': l_p3.item(),
            })
            
                                                
        avg_train_loss = total_loss / len(train_loader)
        avg_l_p1 = total_l_p1 / len(train_loader)
        avg_l_p2 = total_l_p2 / len(train_loader)
        avg_l_p3 = total_l_p3 / len(train_loader)
        
        train_losses.append(avg_train_loss)
        l_p1_losses.append(avg_l_p1)
        l_p2_losses.append(avg_l_p2)
        l_p3_losses.append(avg_l_p3)

        ### Evaluate per epoch if desired
        orig_train_acc, orig_val_acc = evaluate_loaders(train_loader_non_random, test_loader, model, args.device, just_eval=True)
        print(f"Seed {seed} - Current Training Accuracy: {orig_train_acc}")
        print(f"Seed {seed} - Current Validation Accuracy: {orig_val_acc}")    
        
        
    # save model
    weight_dir = f'weights/pre_projection_{args.model}_{args.dataset}_protos{args.num_protos}_baseline{args.baseline}_seed{args.seed}_no_llm_head{args.no_llm_head}.pt'
    torch.save(model.state_dict(), weight_dir)
        
    # del model
    # torch.cuda.empty_cache()   # free cached blocks
    # gc.collect()               # clear Python refs as well
        
    ########################################################
    # Normal train test
    ########################################################   

    # model = LMProtoNet(data_utils['model'],
    #                    num_labels=data_utils['num_labels'],
    #                    num_protos_per_class=args.num_protos,
    #                    init_prototypes=proto_init,
    #                   )    
    # torch.load(weight_dir, map_location=torch.device(args.device), weights_only=True)
    # model.to(args.device)
    
    original_prototypes = model.prototypes.clone().detach().cpu()
    original_prototypes_normed = F.normalize(original_prototypes, p=2, dim=1)
    orig_train_acc, orig_val_acc = evaluate_loaders(train_loader_non_random, test_loader, model, args.device, just_eval=True)
    print(f"Seed {seed} - Original Training Accuracy: {orig_train_acc}")
    print(f"Seed {seed} - Original Validation Accuracy: {orig_val_acc}") 

            
    # --- Extract originals ---
    orig_protos = model.prototypes.detach().cpu()
    orig_protos_norm = F.normalize(orig_protos, p=2, dim=1)
    
    # --- 1) Compute “projection” prototypes & its val acc ---
    # (same as in your run_experiment)
    if backbone.model_type=='bert':
        train_emb = compute_embeddings(model, train_loader_non_random).cpu()
    else:
        train_emb = compute_embeddings(model, train_loader_non_random).cpu()
    train_emb_norm = F.normalize(train_emb, p=2, dim=1)
    
    projected_list = []
    for i in range(orig_protos_norm.size(0)):
        sims = F.cosine_similarity(train_emb_norm, orig_protos_norm[i].unsqueeze(0))
        best_idx = torch.argmax(sims).item()
        projected_list.append(train_emb_norm[best_idx])
    proj_norm = torch.stack(projected_list)
    
    # inject projected prototypes and eval
    with torch.no_grad():
        model.prototypes.copy_(proj_norm.to(device))
    _, proj_val_acc = evaluate_loaders(train_loader_non_random, test_loader, model, device, just_eval=True)
    print(f"Projection-based Val Accuracy: {proj_val_acc:.3f}")
    
    # --- 2) Restore originals before noise sweep ---
    with torch.no_grad():
        model.prototypes.copy_(orig_protos_norm.to(device))
    
    # --- 3) Noise sweep ---
    num_steps  = 20
    max_sigma  = 0.5
    noise_lvls = np.linspace(0.0, max_sigma, num_steps)
    
    cos_sims = []
    val_accs = []
    
    for sigma in noise_lvls:
        noise    = torch.randn_like(orig_protos) * sigma
        pert     = orig_protos + noise
        pert_nm  = F.normalize(pert, p=2, dim=1)
    
        # avg cosine similarity
        with torch.no_grad():
            cos_vals = (orig_protos_norm * pert_nm).sum(dim=1)
        mean_cos = cos_vals.mean().item()
        cos_sims.append(mean_cos)
    
        # eval with perturbed
        with torch.no_grad():
            model.prototypes.copy_(pert_nm.to(device))
        _, val_acc = evaluate_loaders(train_loader_non_random, test_loader, model, device, just_eval=True)
        val_accs.append(val_acc)
        print(f"σ={sigma:.3f} → cos={mean_cos:.3f}, val_acc={val_acc:.3f}")
    
    # --- 4) Plot + save ---
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(6,4))
    plt.plot(cos_sims, val_accs, marker='o', label='Noise sweep')
    plt.axhline(proj_val_acc, color='r', linestyle='--', label=f'Projection acc = {proj_val_acc:.3f}')
    plt.xlabel('Mean Cosine Similarity to Original Prototypes')
    plt.ylabel('Validation Accuracy')
    plt.title('Val Accuracy vs. Prototype Similarity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/val_acc_vs_cosine.png')
    plt.close()
    
    print("Saved plot to plots/val_acc_vs_cosine.png")

    
    
    # ########################################################
    # # Prototype projection with full training examples
    # ########################################################
    # model.eval()
    
    # original_prototypes = model.prototypes.clone().detach().cpu()
    # original_prototypes_normed = F.normalize(original_prototypes, p=2, dim=1)

    # if backbone.model_type=='bert':
    #     train_embeddings = compute_embeddings(model, train_loader_non_random)
    #     train_embeddings = train_embeddings.to('cpu')
    # elif backbone.model_type=='llm' and backbone.no_llm_head:
    #     train_embeddings = data_utils['train_dataset_enc'].embeddings
    # elif backbone.model_type=='llm' and not backbone.no_llm_head:
    #     train_embeddings = compute_embeddings(model, train_loader_non_random)
    #     train_embeddings = train_embeddings.to('cpu')
    # else:
    #     raise NameError('Wrong model name when getting train embeddings.')
        
    # train_embeddings_normalized = F.normalize(train_embeddings, p=2, dim=1)

    # projected_embeddings_list = []
    # full_proto_texts = []
    
    # for i in tqdm(range(total_num_prototypes), desc="Projecting prototypes with full examples"):
    #     proto_norm = original_prototypes_normed[i].unsqueeze(0) # Shape: [1, dim]
    #     similarities = F.cosine_similarity(train_embeddings_normalized, proto_norm) # Shape: [num_train_samples]
    #     best_instance_idx = torch.argmax(similarities).item()
    #     projected_embeddings_list.append(train_embeddings_normalized[best_instance_idx]) # Append the normalized embedding
    #     full_proto_texts.append(data_utils['train_df']['text'].iloc[best_instance_idx]) # Get corresponding text

    #     #### Print prototypes for inspection
    #     # print(f"Full projection prototype {i}: {data_utils['train_df']['text'].iloc[best_instance_idx]}")

    # new_prototypes_projected_normalized = torch.stack(projected_embeddings_list).to(args.device) # Shape: [total_num_protos, dim]

    # # Update the model with the new prototypes
    # with torch.no_grad():
    #     model.prototypes.copy_(new_prototypes_projected_normalized)

    # # Evaluate performance with projected prototypes
    # proj_train_acc, proj_val_acc = evaluate_loaders(train_loader_non_random, test_loader, model, args.device, just_eval=True)
    # print(f"Seed {seed} - Full Projection Accuracy: Train={proj_train_acc:.4f}, Val={proj_val_acc:.4f}")

    # # Calculate distance/similarity between original and projected prototypes
    # full_proj_prototypes = model.prototypes.clone().detach().cpu()
    # full_proj_prototypes_normed = F.normalize(full_proj_prototypes, p=2, dim=1)
    # full_l2_distances = torch.norm(original_prototypes_normed - full_proj_prototypes_normed, dim=1).mean().item()
    # full_cosine_similarities = torch.sum(original_prototypes_normed * full_proj_prototypes_normed, dim=1).mean().item()
    
    # print(f"Seed {seed} - Full Projection: Avg L2={full_l2_distances:.4f}, Avg CosSim={full_cosine_similarities:.4f}")
    
    # # save model
    # weight_dir = f'weights/post_projection_{args.model}_{args.dataset}_protos{args.num_protos}_baseline{args.baseline}_seed{args.seed}_no_llm_head{args.no_llm_head}.pt'
    # torch.save(model.state_dict(), weight_dir)

    
    # ########################################################
    # # Final logging
    # ########################################################
    # # Combine all results
    # results = {
    #     'seed': seed,
    #     'orig_val_acc': orig_val_acc,
    #     'proj_val_acc': proj_val_acc,
    #     'full_proto_texts': full_proto_texts,
    #     }
    # return results


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_protos', type=int, default=10, help='Number of prototypes per class')
    parser.add_argument('--prototype_dim', type=int, default=256, help='dimension of prototypes')
    parser.add_argument('--dataset', type=str, default='20newsgroups', help='Dataset used')
    parser.add_argument('--model', type=str, default='bert', help='Backbone model')
    parser.add_argument('--input_size', type=int, default=256, help='Size of input token length')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument(
        '--baseline',
        action='store_true',          # ← sets baseline = True when the flag is present
        default=False,                # (optional) defaults to False when omitted
        help='Use baseline losses'
    )
    parser.add_argument(
        '--no_llm_head',
        action='store_true',          # ← sets baseline = True when the flag is present
        default=False,                # (optional) defaults to False when omitted
        help='Use mlp on top of llm'
    )
    
    # Add loss weight arguments
    parser.add_argument('--l_p1_weight', type=float, default= 1.0, help='Weight for ensuring each prototype is close to at least one instance')
    parser.add_argument('--l_p2_weight', type=float, default=0.1, help='Weight for ensuring each data instance is close to at least one prototype')
    parser.add_argument('--l_p3_weight', type=float, default=0.1, help='Weight for l_p3 loss')
    
    # LLM prototype optimization parameters
    parser.add_argument('--lm_iters', type=int, default=10, help='Number of iterations for LLM prototype optimization')
    parser.add_argument('--lm_parallel', type=int, default=8, help='Number of parallel optimizations for LLM prototypes')
    
    args = parser.parse_args()
    
    all_results = []
    print(f"\nRunning experiment with seed {args.seed}")

    res = run_experiment(args, args.seed)
    # all_results.append(res)

    # records = []
    # for r in all_results:
    #     record = {
    #         'seed': r['seed'],
    #         'orig_val_acc': r['orig_val_acc'],
    #         'proj_val_acc': r['proj_val_acc'],
    #         'full_proto_texts': r['full_proto_texts'],
    #                     }
    #     records.append(record)

    
    # results_df = pd.DataFrame(records)
    # csv_path = f"data/{args.model}_{args.dataset}_protos{args.num_protos}_baseline{args.baseline}_seed{args.seed}_no_llm_head{args.no_llm_head}.csv"
    # results_df.to_csv(csv_path, index=False)
    # print(f"\nResults saved to {csv_path}")
        
















    # #######################################################
    # # Prototype projection with beam search.
    # #######################################################
    # print('\n--- Projecting prototypes with beam search ---')
    # model.eval()
    # print('Getting new embeddings')
    # train_embeddings = compute_embeddings(model, tokenizer, data_utils['train_loader_non_random'], device, 'pwnet')
    # train_embeddings = train_embeddings.to('cpu')
    # normalized_train_embeddings = F.normalize(train_embeddings, p=2, dim=1)
        
    # proj_indices = []
    # new_prototypes = []
    # new_proto_texts = []

    # for i in range(total_num_prototypes):
    #     with torch.no_grad():
    #         # Check if the prototype exists
    #         if i >= model.prototypes.size(0):
    #             print(f"Warning: Prototype index {i} out of bounds. Total prototypes: {model.prototypes.size(0)}")
    #             continue
                
    #         proto_norm = F.normalize(model.prototypes[i].unsqueeze(0), p=2, dim=1)
    #         # Replace L2 distance with cosine similarity
    #         similarities = F.cosine_similarity(normalized_train_embeddings, proto_norm.to(normalized_train_embeddings.device))
    #         # Find maximum similarity (instead of minimum distance)
    #         cand_index = torch.argmax(similarities)

    #     candidate_text = train_df[text_column].iloc[cand_index.item()]
    #     print(f'Training data max similarity for prototype {i}: {similarities[cand_index]:.4f}')
    #     new_proto_embedding, best_subseq = beam_search_subsequence(candidate_text, proto_norm.squeeze(0), model, tokenizer, device)

    #     if new_proto_embedding is None:
    #         new_proto_embedding = proto_norm.squeeze(0)
    #         best_subseq = candidate_text

    #     new_prototypes.append(new_proto_embedding)
    #     proj_indices.append(cand_index.item())
    #     new_proto_texts.append(best_subseq)

    # new_prototypes = torch.stack(new_prototypes)
    # with torch.no_grad():
    #     model.prototypes.copy_(new_prototypes)

    # beam_train_acc, beam_val_acc = evaluate_loaders(data_utils['train_loader_non_random'], val_loader, model, device, just_eval=True)

    # for i, idx in enumerate(proj_indices):
    #     print(f"Seed {seed} - Beam Search Prototype {i}: {new_proto_texts[i]}")

    # print(f"Seed {seed} - Beam Search Training Accuracy: {beam_train_acc}")
    # print(f"Seed {seed} - Beam Search Validation Accuracy: {beam_val_acc}")

    # projected_prototypes = model.prototypes.clone().detach().cpu()
    # projected_prototypes_normed = F.normalize(projected_prototypes, p=2, dim=1)

    # # Calculate distances between original and projected prototypes
    # beam_l2_distances = torch.norm(original_prototypes_normed - projected_prototypes_normed, dim=1).mean().item()
    # beam_cosine_similarities = torch.sum(original_prototypes_normed * projected_prototypes_normed, dim=1).mean().item()

    # torch.save(model.state_dict(), f'weights/prosenet_beam_{args.model}_{args.dataset}_{args.num_protos}.pt')
    
        
    


    
    


    
    
    
    
    # ########################################################
    # # Our Algorithm - LLM Proto Optimization
    # ########################################################    
    # print('\n--- Running LLM prototype optimization ---')
    # # Reload the original model
    # model.load_state_dict(torch.load(f'weights/prosenet_pre_projection_{args.model}_{args.dataset}_{args.num_protos}.pt', map_location=device))
    # model.to(device).eval()    
    
    # textual_protos = []
    # with torch.no_grad():
    #     textual_protos, distances = optimize_prototypes(
    #         tokenizer,
    #         train_labels,
    #         data_utils['text_column'],
    #         normalized_train_embeddings, 
    #         data_utils['train_df'],  
    #         model, 
    #         model_name=args.model,
    #         dataset_name=args.dataset,
    #         dist_func='cosine',
    #         device=args.device,
    #         num_iters=args.lm_iters,
    #         num_parallel=args.lm_parallel,
    #     )

    #     print('Avg Distance to black box prototypes:', distances)
    #     for i, proto in enumerate(textual_protos):
    #         print(f"LLM Prototype {i}: {proto}")

                
    # # If textual_protos is successful, convert to embeddings and evaluate
    # if textual_protos and not isinstance(textual_protos[0], str):
    #     print("Warning: textual_protos is not a list of strings. Skipping LLM prototype evaluation.")
    #     lm_proto_train_acc, lm_proto_val_acc = -1, -1
    #     lm_proto_l2_distances, lm_proto_cosine_similarities = -1, -1
    # else:
    #     normalized_lm_prototypes = []
    #     try:
    #         with torch.no_grad():
    #             for textual_proto in textual_protos:
    #                 # Process one prototype at a time without padding
    #                 inputs = tokenizer(textual_proto, return_tensors='pt', truncation=True, max_length=args.input_size).to(device)
    #                 outputs = model.backbone(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    #                 prototype_hidden_state = outputs.last_hidden_state[:, 0, :]  # Get CLS token representation
    #                 # Normalize the prototype
    #                 normalized_prototype = F.normalize(prototype_hidden_state, p=2, dim=1)
    #                 normalized_lm_prototypes.append(normalized_prototype)
    #             # Stack all normalized prototypes
    #             all_prototypes = torch.cat(normalized_lm_prototypes, dim=0)
    #             # Copy to model's prototype layer all at once
    #             model.prototypes.copy_(all_prototypes.detach().cpu())
                
    #         torch.save(model.state_dict(), f'weights/prosenet_lm_prototypes_{args.model}_{args.dataset}_{args.num_protos}.pt')
    #         lm_proto_train_acc, lm_proto_val_acc = evaluate_loaders(data_utils['train_loader_non_random'], val_loader, model, device, just_eval=True)
    #         print(f"Seed {seed} - ProtoLM Training Accuracy: {lm_proto_train_acc}")
    #         print(f"Seed {seed} - ProtoLM Validation Accuracy: {lm_proto_val_acc}")
            
    #         # Calculate distance metrics
    #         lm_prototypes = model.prototypes.clone().detach().cpu()
    #         lm_prototypes_normed = F.normalize(lm_prototypes, p=2, dim=1)
    #         lm_proto_l2_distances = torch.norm(original_prototypes_normed - lm_prototypes_normed, dim=1).mean().item()
    #         lm_proto_cosine_similarities = torch.sum(original_prototypes_normed * lm_prototypes_normed, dim=1).mean().item()
    #         print(f"Seed {seed} - LLM Proto: Avg L2={lm_proto_l2_distances:.4f}, Avg CosSim={lm_proto_cosine_similarities:.4f}")
    #     except Exception as e:
    #         print(f"Error during LLM prototype embedding or evaluation: {e}")
    #         lm_proto_train_acc, lm_proto_val_acc = -1, -1
    #         lm_proto_l2_distances, lm_proto_cosine_similarities = -1, -1











    # # Combine all results
    # results = {
    #     'seed': seed,
    #     'orig_train_acc': orig_train_acc,
    #     'orig_val_acc': orig_val_acc,
        
    #     # Beam search results
    #     'beam_train_acc': beam_train_acc,
    #     'beam_val_acc': beam_val_acc,
    #     'beam_proto_texts': new_proto_texts,
    #     'beam_l2_distances': beam_l2_distances,
    #     'beam_cosine_similarities': beam_cosine_similarities,
        
    #     # Full projection results
    #     'full_train_acc': full_train_acc,
    #     'full_val_acc': full_val_acc,
    #     'full_proto_texts': full_proto_texts,
    #     'full_l2_distances': full_l2_distances,
    #     'full_cosine_similarities': full_cosine_similarities,
        
    #     # LLM optimization results
    #     'lm_proto_train_acc': lm_proto_train_acc,
    #     'lm_proto_val_acc': lm_proto_val_acc,
    #     'lm_proto_texts': textual_protos,
    #     'lm_proto_l2_distances': lm_proto_l2_distances,
    #     'lm_proto_cosine_similarities': lm_proto_cosine_similarities,
    # }
    
    # return results

        

"""

===amazon reviews
bert: 0.8507
electra: 0.8746
modern_bert: 0.844
llama: 0.8481
qwen: 0.8564
llama no head: 0.8109
qwen no head: 0.8477

===imdb
bert: 0.905
electra: Val=0.9331
modern_bert: 0.8734
llama: 0.8619
qwen: 0.8944
llama no head: 0.843
qwen no head: 0.8817

===agnews
bert: 0.9292105263157895
electra: 0.9167105263157894
modern_bert: 0.9231578947368421
llama: 0.9071052631578947 (this was highly unstable)
qwen:  0.9019736842105263
llama no head: 0.8875
qwen no head: 0.9032894736842105

===trec
bert: 0.914
electra: 0.8840
modern_bert: 0.862
roberta: 0.868
mpnet: 0.8940
llama: 0.326
qwen: 0.3
llama no head: 0.45
qwen no head: 0.3

===20newsgroups
bert: 0.7955390334572491
electra: 0.7176048858204992
modern_bert: 0.7770844397238449
roberta: 0.8158523632501328
mpnet: 0.8339
llama: 0.4397238449283059
qwen: 0.5007966011683483
llama no head: 43%
qwen no head: 49%

===dbpedia
bert: 0.89
electra: 0.
modern_bert: 0.
roberta: 0.
mpnet: 0.
llama: 0.
qwen: 0.
llama no head: 0.
qwen no head: 0.
"""