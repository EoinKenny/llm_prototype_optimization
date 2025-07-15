"""
This script runs a loop over all models, dataset, and distance metrics and 40 prototypes per dataset which are pre-defined
The goal is to qualitatively (and quantitatively) discover which is the best distance metric
Also however, we want to compare across datasets and models to see how promising this overall method is.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import ast
import argparse
import os
import torch.nn.functional as F
import transformers

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

from src.synthetic_prototypes import prototypes_dict
from src.prompts import make_prompt
from src.functions import *
from src.models import LMProtoNet
from src.llm_proto_optim import make_initial_prompt_str



import gc
import torch
def clean_gpus() -> None:
    gc.collect()
    torch.cuda.empty_cache() 
clean_gpus()


def main(args):

    
    all_exp_data = dict()
    
    for model in ['mpnet']:#, 'electra', 'modern_bert', 'roberta', 'mpnet']:
        args.model = model

        for dataset in ['trec']:#, 'dbpedia', '20newsgroups', ]:   
            args.dataset = dataset
            prototype_list = prototypes_dict[dataset]
            data_utils = load_domain(args)
            tokenizer = data_utils['tokenizer']
            train_df = data_utils['train_df']

            model = LMProtoNet(data_utils['model'],
                               num_labels=data_utils['num_labels'],
                               num_protos_per_class=args.num_protos,
                              )  


            if data_utils['model'].model_type=='bert':
                train_loader_non_random = DataLoader(data_utils['train_dataset'], batch_size=128, shuffle=False)                    
            elif data_utils['model'].model_type=='llm':
                train_loader_non_random = DataLoader(data_utils['train_dataset_enc'], batch_size=128, shuffle=False)
            else:
                raise TypeError('wrong model type')

            weight_dir = f'weights/post_projection_{args.model}_{args.dataset}_protos{args.num_protos}_baseline{args.baseline}_seed{args.seed}_no_llm_head{args.no_llm_head}.pt'
            torch.load(weight_dir, map_location=torch.device(args.device), weights_only=True)
            model.to(args.device)

            llm_pipeline = load_llm(args)

            
            
            print('Getting embeddings')
            # Check if the encodings are already saved
            train_enc_dir = f'datasets/preprocess/{args.dataset}/train_encodings_toy_{args.model}_{args.no_llm_head}.pt'
            if os.path.exists(train_enc_dir):
                print('loading encodings...')
                train_encodings = torch.load(train_enc_dir, weights_only=True)
            else:
                train_encodings = compute_embeddings(model, train_loader_non_random)
                torch.save(train_encodings, train_enc_dir)
            

            # save embeddings if they're not there already
            experiment_data = list()    

            for prototype_idx, prototype_sequence_meta in enumerate(prototype_list[:15]):

                prototype_sequence, _, _ = prototype_sequence_meta

                # Get the hidden state of the prototype
                inputs = tokenizer(text=prototype_sequence, 
                                   return_tensors='pt', 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=args.input_size,
                                  ).to(args.device)
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                prototype_hidden_state_normalized = outputs['cls_rep_normalized'].cpu().detach()

                # Normalize training embeddings on the fly
                normalized_train_encodings = F.normalize(train_encodings, p=2, dim=1)
                distances = distance_function(prototype_hidden_state_normalized, normalized_train_encodings)
                early_distance = distances.max().item()
                early_example  = train_df.iloc[torch.argmax(distances).item()]['text']


                

                
                # Define the nearest neighbors pool
                nn_indices_pool = torch.argsort(distances, descending=True)[:args.nn_pool_size]
                nn_reviews_pool = train_df.iloc[nn_indices_pool.cpu().numpy()]['text'].values.tolist()

                # Define intitial guesses for llm
                closest_distances = [None]
                while len(closest_distances) != args.num_neighbors:
                    initial_guesses_prompt = make_initial_prompt_str(nn_reviews_pool[:3], args.num_neighbors, args.dataset)
                    llm_response = query_llm(initial_guesses_prompt, llm_pipeline)
                    closest_reviews = extract_python_list(llm_response)    
                    if not closest_reviews:
                        continue
                    inputs = tokenizer(closest_reviews, return_tensors='pt', padding=True, truncation=True, max_length=args.input_size).to(args.device)
                    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    latents = outputs['cls_rep_normalized'].cpu().detach()
                    closest_distances = distance_function(prototype_hidden_state_normalized, latents)

                # breakpoint()


                # # ------------------------------------------------------------
                # # 1️⃣  Initialise with nearest-neighbour examples from training
                # # ------------------------------------------------------------
                # # -- Normalise embeddings once (needed for cosine and exp(-L2))
                # normalized_train_encodings = F.normalize(train_encodings, p=2, dim=1)
                
                # # -- Similarity of every training point to the prototype
                # distances = distance_function(
                #     prototype_hidden_state_normalized,          # shape [1, D]
                #     normalized_train_encodings,                 # shape [N, D]
                # )
                
                # # -- Record “early” best match for logging
                # early_idx       = torch.argmax(distances).item()       # highest similarity
                # early_distance  = distances[early_idx].item()
                # early_example   = train_df.iloc[early_idx]['text']
                
                # # ------------------------------------------------------------
                # # 2️⃣  Build candidate pools
                # # ------------------------------------------------------------
                # # Pool we’ll sample from when generating prompts for the LLM
                # pool_idx        = torch.argsort(distances, descending=True)[:args.nn_pool_size]
                # nn_reviews_pool = train_df.iloc[pool_idx.cpu().numpy()]['text'].tolist()
                
                # # Fixed-size working set that we’ll try to improve in later iterations
                # closest_idx        = torch.argsort(distances, descending=True)[:args.num_neighbors]
                # closest_reviews    = train_df.iloc[closest_idx.cpu().numpy()]['text'].tolist()
                # closest_distances  = distances[closest_idx].clone()    # Tensor of shape [k]
                
                # # ------------------------------------------------------------
                # # Everything below this point (LLM loop, replacement logic,
                # # history tracking, etc.) can remain unchanged.
                # # ------------------------------------------------------------
                



                # Initialize history if not already
                closest_distances_history = [closest_distances.max().item()]

                for current_iter in range(args.num_iters):
                    start_time = time.time()

                    # --- 1) Sample a couple of candidates to show the LLM ---
                    sampled_indices = torch.randperm(len(nn_reviews_pool))[:15]
                    sampled_nn_reviews = [nn_reviews_pool[i] for i in sampled_indices]

                    prompt = make_prompt(
                        closest_reviews,
                        closest_distances.tolist(),
                        args.num_neighbors,
                        training_examples=sampled_nn_reviews,
                        dataset=args.dataset,
                    )
                    llm_response = query_llm(prompt, llm_pipeline)
                    print(llm_response)
                    new_guesses = extract_python_list(llm_response)

                    try:
                        print(f'=============> Current guesses: Prototype {prototype_idx}/{len(prototype_list)} -- Iteration {current_iter}/{args.num_iters}' )
                        print(f'Ground truth: \n{prototype_sequence}\n')
                        for g in closest_reviews:
                            print(g)
    
                        # --- 2) Keep only truly new guesses ---
                        unique_new = []
                        for g in new_guesses:
                            if g not in closest_reviews and g not in unique_new:
                                unique_new.append(g)
    
                        # If nothing new, record history and continue
                        if not unique_new:
                            closest_distances_history.append(
                                closest_distances.max().item() if args.distance_func=='cosine'
                                else closest_distances.min().item()
                            )
                            print(f"[Iter {current_iter}] no new guesses")
                            continue
    
                        # --- 3) Embed the new guesses ---
                        inputs = tokenizer(
                            text=unique_new,
                            return_tensors='pt', 
                            padding=True,
                            truncation=True,
                            max_length=args.input_size,
                        ).to(args.device)
                        
                        new_states = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            forward_type = 'train',
                        )['cls_rep_normalized']
    
                        # --- 4) Compute their distances to the prototype ---
                        new_dists = distance_function(
                            prototype_hidden_state_normalized.to('cpu'),
                            new_states.to('cpu'),
                        )
    
                        # --- 5) For each new candidate, if it's better than the current worst, swap it in ---
                        for g, d in zip(unique_new, new_dists):
                            threshold = closest_distances.min().item()
                            if d.item() > threshold:
                                worst_idx = torch.argmin(closest_distances).item()
                                closest_reviews[worst_idx] = g
                                closest_distances[worst_idx] = d
    
                        # --- 6) Record the best score so far for monitoring ---
                        best_score = closest_distances.max().item()
                        closest_distances_history.append(best_score)
                        print(f"best cosine: {best_score:.4f} -- Starting was {early_distance}")
                        print(f"Iter time: {time.time() - start_time:.2f}s\n\n\n")
    
                    except:
                        print('\n\nFailed iteration')# LLM response is:', llm_response)

                # after iterations, pick the single best match
                best_idx = torch.argmax(closest_distances).item()
                
                # —— summary statistics for this prototype —— 
                optimized_distance = closest_distances_history[-1]
                final_guess        = closest_reviews[best_idx]
                ground_truth       = prototype_sequence
                
                # print or log them
                print('==========================')
                print(f"[Prototype {prototype_idx}]")
                print(f"  • Early distance:     {early_distance:.4f}")
                # print(f"  • Early example:      {early_example}")
                # print('==========================')
                print(f"  • Optimized distance: {optimized_distance:.4f}")
                print(f"  • Final guess:        {final_guess}")
                print(f"  • Ground truth:       {ground_truth}\n")
                print('==========================')
                # — end summary —
                
                # store experiment data
                experiment_data.append([
                    closest_distances_history,
                    closest_reviews[best_idx],
                    prototype_sequence
                ])

            all_exp_data[f'{args.dataset}_{args.model}'] = experiment_data

            print('==============')
            print('Results')
            print(f"{args.dataset}, {args.model}")
            print('Prediction:', closest_reviews[best_idx])
            print('Ground Truth:', prototype_sequence)
            print('Distance:', closest_distances_history[-1])
            print(" ")

            with open(f'data/ground_truth_exp_all_data.pickle', 'wb') as handle:
                pickle.dump(all_exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def extract_python_list(text):

    # Start from the end of the string and find the last occurrence of '['
    start_index = text.rfind('[')
    end_index = text.rfind(']')
    
    if start_index != -1:
        # Extract the substring from the start_index to the end
        python_list_str = text[start_index:end_index+1]
        
        try:
            # Use ast.literal_eval to safely convert the string representation of the list to an actual Python list
            python_list = ast.literal_eval(python_list_str)
            return python_list
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing list: {e}")
            return None
    else:
        print("No Python list found in the string.")
        return None


def query_llm(prompt, llm_pipeline):
    """
    Query a huggingface LLM which can fit on 2 GPUs
    |   0  NVIDIA L4                      On  | 00000000:38:00.0 Off |                    0 |
    | N/A   49C    P0              28W /  72W |    196MiB / 23034MiB |      0%      Default |
    |                                         |                      |                  N/A |    
    """

    messages = [
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "user", "content": prompt},
    ]
    
    outputs = llm_pipeline(
        text_inputs=messages,
        max_new_tokens=2056,
    )

    generated_text = outputs[0]["generated_text"][-1]['content']

    return generated_text



def load_llm(args):
    # model_id = 'meta-llama/Llama-3.2-3B-Instruct'  # 16 sec
    # model_id = 'braindao/Qwen2.5-14B'  # 174 sec
    model_id = 'Qwen/Qwen2.5-32B'  # ???
    # model_id = 'Qwen/Qwen2.5-72B'  # OoM
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.llm_device,
        token=hugging_token,
    )
    
    return pipeline
        
    
def distance_function(prototype_hidden_state, train_embeddings, distance_func_type='cosine'):
    """
    Normalizes L2 and cosine to output 'similarities' between 0-1, where 1 is a perfect match
    """
    if distance_func_type == 'l2':
        # Compute L2 distance using cdist
        distances = torch.cdist(prototype_hidden_state, train_embeddings, p=2).squeeze()
        distances = torch.exp(-distances)
    elif distance_func_type == 'cosine':
        # Compute cosine similarity
        distances = F.cosine_similarity(train_embeddings, prototype_hidden_state, dim=1)
        distances += 1
        distances /= 2
    else:
        raise ValueError("distance_func_type must be either 'l2' or 'cosine'")
    
    return distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_protos', type=int, default=10, help='Number of prototypes to test')
    parser.add_argument('--num_neighbors', type=int, default=10, help='Number of random possible examples to help llm')
    parser.add_argument('--num_iters', type=int, default=30, help='Number of iterations per prototype')
    # parser.add_argument('--dataset', type=str, default='20newsgroups', help='dataset')
    parser.add_argument('--model', type=str, default='bert', help='backbone model')
    parser.add_argument('--input_size', type=int, default=256, help='size of input token length')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--llm_device', type=str, default='cuda:1', help='device')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prototype_dim', type=int, default=256, help='dimension of prototypes')
    parser.add_argument('--nn_pool_size', type=int, default=5, help='number of nns to sample from')
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
    args = parser.parse_args()    
    
    with torch.no_grad():
        main(args)    

        
        
        
        
        
        
        
        
        
