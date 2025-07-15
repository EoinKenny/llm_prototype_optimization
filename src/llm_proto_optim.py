import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import ast
import os
import random
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

# Domain and prototype specific imports
from src.functions import *
from src.prompts import make_prompt

    
def optimize_prototypes(tokenizer,
                        train_labels,
                        text_column,
                        normalized_train_embeddings,
                        train_df,
                        model,
                        model_name: str,
                        dataset_name: str,
                        dist_func: str,
                        num_neighbors: int = 15,
                        num_ground_truth_neighbors: int = 30,
                        num_iters: int = 30,
                        num_parallel: int = 8,
                        input_size: int = 128,
                        method=None, # This parameter is defined but not used in the provided snippet
                        device='cpu',
                        ):
    """
    Optimize textual prototypes using an LM, starting with LM-generated initial guesses with retries,
    ensuring the target number of unique initial guesses.
    """
    textual_prototypes = []
    distances_list = []
    num_training_examples = 10

    all_prototypes_max_metric_history = []
    all_prototypes_mean_metric_history = []

    # Load dataset-specific metadata
    if dataset_name == 'imdb':
        column_label = 'sentiment'
        class_names = ['negative review', 'positive review']
    elif dataset_name == 'ag_news':
        column_label = 'Class Index'
        class_names = ['World News', 'Sports News', 'Business News', 'Science/Technology News']
    elif dataset_name == 'db_pedia':
        column_label = 'label'
        class_names = ['Company', 'Educational Institution', 'Artist', 'Athlete', 'Office Holder',
                       'Mean of Transportation', 'Building', 'Natural Place', 'Village', 'Animal',
                       'Plant', 'Album', 'Film', 'Written Work']
    else:
        print(f"Warning: Dataset {dataset_name} metadata not explicitly defined. Using provided text_column.")
        if 'label' in train_df.columns: column_label = 'label'
        elif 'Class Index' in train_df.columns: column_label = 'Class Index'
        elif 'sentiment' in train_df.columns: column_label = 'sentiment'
        else: raise ValueError(f"Cannot determine label column for dataset {dataset_name}")
        class_names = [f"Class {i}" for i in range(len(np.unique(train_labels)))]


    # Calculate class indices for each prototype
    total_num_prototypes = len(model.prototypes)
    total_num_classes = len(class_names)
    prototypes_per_class = total_num_prototypes // total_num_classes
    if total_num_prototypes % total_num_classes != 0:
        print(f"Warning: Total prototypes ({total_num_prototypes}) not evenly divisible by num classes ({total_num_classes})")

    class_indexes = []
    for i in range(total_num_classes):
        for j in range(prototypes_per_class):
            if len(class_indexes) < total_num_prototypes:
                class_indexes.append(i)


    total_num_prototypes = len(model.prototypes)
    NUM_INITIAL_LM_GUESSES = num_neighbors  # Target number of unique initial guesses
    MAX_INITIAL_LM_RETRIES = num_neighbors  # Max attempts to get initial guesses

    for proto_idx, prototype_vector in enumerate(model.prototypes.detach().clone()):
        prototype_class_idx = class_indexes[proto_idx]
        prototype_class_desc = class_names[prototype_class_idx]

        print(f"\nOptimizing prototype {proto_idx} for class: {prototype_class_desc} (label: {prototype_class_idx})")
        
        
        

        current_prototype_max_metric_iter_history = []
        current_prototype_mean_metric_iter_history = []

        prototype_hidden_state = prototype_vector.unsqueeze(0).to(device)
        prototype_hidden_state = prototype_hidden_state / prototype_hidden_state.norm(dim=1, keepdim=True)

        # Assuming distance_function is defined elsewhere
        distances_to_all_train = distance_function(prototype_hidden_state.cpu(), normalized_train_embeddings.cpu(), dist_func)

        if dist_func == 'l2':
            sorted_train_indices = torch.argsort(distances_to_all_train, descending=False)
        else:  # cosine
            sorted_train_indices = torch.argsort(distances_to_all_train, descending=True)

        valid_train_indices = sorted_train_indices[sorted_train_indices < len(train_df)].cpu().numpy()
        ground_truth_nn_reviews = train_df.iloc[valid_train_indices[:num_training_examples]][text_column].values.tolist()
        
        closest_reviews = []
        closest_distances = torch.tensor([], dtype=torch.float32).cpu()
        
        print('Getting intitial nns')
        while True:
            try:
                client = refresh_token()
                prompt = make_initial_prompt_str(ground_truth_nn_reviews, NUM_INITIAL_LM_GUESSES, dataset_name, prototype_class_desc)
                response = test_create_completion_oia_1plus(client, prompt, TEMP=1.0)

                # Assuming extract_python_list is defined
                closest_reviews = extract_python_list(response)
                if len(closest_reviews) == num_neighbors:
                    break
            except:
                print('failed to find initial nns...')

                
        inputs_initial = tokenizer(closest_reviews, return_tensors='pt', padding=True,truncation=True, max_length=input_size).to(device)
        
        with torch.no_grad():
            outputs_initial = model.backbone(input_ids=inputs_initial['input_ids'], attention_mask=inputs_initial['attention_mask'])
            initial_candidate_hidden_states = outputs_initial.last_hidden_state[:, 0, :]
            initial_candidate_hidden_states = initial_candidate_hidden_states / initial_candidate_hidden_states.norm(dim=1, keepdim=True)

        closest_distances = distance_function(prototype_hidden_state.cpu(), initial_candidate_hidden_states.cpu(), dist_func)
        closest_distances = closest_distances.cpu()  # distance of initial guesses to prototype

        if dist_func == 'l2':
            current_prototype_max_metric_iter_history.append(closest_distances.min().item())
        else: # cosine
            current_prototype_max_metric_iter_history.append(closest_distances.max().item())
        current_prototype_mean_metric_iter_history.append(closest_distances.mean().item())
        print(f"Initial best score from {len(closest_reviews)} LM guesses: {current_prototype_max_metric_iter_history[0]:.4f}")

        # Optimization loop
        for iteration in range(num_iters):
            try:
                client = refresh_token() # Assuming refresh_token() is defined

                if dist_func == 'l2':
                    parsed_distances_for_prompt = closest_distances
                else: # cosine
                    parsed_distances_for_prompt = closest_distances

                with ThreadPoolExecutor(max_workers=num_parallel) as executor:
                    training_examples_for_prompt_size = min(2, len(ground_truth_nn_reviews))
                    if training_examples_for_prompt_size > 0 :
                        training_examples_for_prompt = np.random.choice(ground_truth_nn_reviews, size=training_examples_for_prompt_size, replace=False).tolist()
                    else:
                        training_examples_for_prompt = ["No examples available"]

                    futures = []
                    for _ in range(num_parallel):
                        # Assuming make_prompt is defined
                        current_prompt = make_prompt(
                            closest_reviews if closest_reviews else ["No current solutions"],
                            parsed_distances_for_prompt if parsed_distances_for_prompt.numel() > 0 else [],
                            len(closest_reviews) if closest_reviews else 0,
                            training_examples=training_examples_for_prompt,
                            dataset=dataset_name,
                            class_desc=prototype_class_desc,
                        )
                        futures.append(executor.submit(test_create_completion_oia_1plus, # Assuming test_create_completion_oia_1plus is defined
                                                        client,
                                                        current_prompt,
                                                        TEMP=1.0))
                    responses = [future.result() for future in as_completed(futures)]
                    
                new_guesses_from_llm_iteration = []
                for resp_text in responses:
                    guesses_from_one_response = extract_python_list(resp_text) # Assuming extract_python_list is defined
                    if guesses_from_one_response:
                        for guess in guesses_from_one_response:
                            if isinstance(guess, str) and guess.strip():
                                new_guesses_from_llm_iteration.append(guess.strip())

                if new_guesses_from_llm_iteration:
                    unique_new_guesses = []
                    for guess in new_guesses_from_llm_iteration:
                        if guess not in closest_reviews and guess not in unique_new_guesses:
                            unique_new_guesses.append(guess)

                    if unique_new_guesses:
                        inputs_new = tokenizer(unique_new_guesses, return_tensors='pt', padding=True,
                                               truncation=True, max_length=input_size).to(device)
                        with torch.no_grad():
                            outputs_new = model.backbone(input_ids=inputs_new['input_ids'],
                                                         attention_mask=inputs_new['attention_mask'])
                            new_hidden_states = outputs_new.last_hidden_state[:, 0, :]
                            new_hidden_states = new_hidden_states / new_hidden_states.norm(dim=1, keepdim=True)

                        unique_new_distances = distance_function(prototype_hidden_state.cpu(), new_hidden_states.cpu(), dist_func)
                        unique_new_distances = unique_new_distances.cpu()

                        current_closest_distances_tensor = closest_distances if isinstance(closest_distances, torch.Tensor) and closest_distances.numel() > 0 else torch.tensor([], dtype=torch.float32).cpu()

                        combined_reviews = (closest_reviews if closest_reviews else []) + unique_new_guesses
                        combined_distances = torch.cat((current_closest_distances_tensor, unique_new_distances), dim=0)

                        current_k = min(num_neighbors, len(combined_distances))
                        if current_k > 0:
                            if dist_func == 'l2':
                                top_values, top_indices = torch.topk(combined_distances, k=current_k, largest=False)
                            else: # cosine
                                top_values, top_indices = torch.topk(combined_distances, k=current_k, largest=True)

                            closest_reviews = [combined_reviews[i] for i in top_indices.tolist()]
                            closest_distances = top_values.cpu()
                        else:
                            closest_reviews = []
                            closest_distances = torch.tensor([], dtype=torch.float32).cpu()

                        print(f"Prototype {proto_idx + 1}, Iteration {iteration + 1}:")
                        if closest_distances.numel() > 0:
                            for k_idx in range(len(closest_distances)):
                                print(f"  Dist/Sim: {closest_distances[k_idx].item():.4f}, Text: '{closest_reviews[k_idx]}'")
                            print("  ")

                            if dist_func == 'l2':
                                current_prototype_max_metric_iter_history.append(closest_distances.min().item())
                            else: # cosine
                                current_prototype_max_metric_iter_history.append(closest_distances.max().item())
                            current_prototype_mean_metric_iter_history.append(closest_distances.mean().item())
                        else:
                            prev_max = current_prototype_max_metric_iter_history[-1] if current_prototype_max_metric_iter_history else (float('inf') if dist_func == 'l2' else float('-inf'))
                            prev_mean = current_prototype_mean_metric_iter_history[-1] if current_prototype_mean_metric_iter_history else (float('inf') if dist_func == 'l2' else float('-inf'))
                            current_prototype_max_metric_iter_history.append(prev_max)
                            current_prototype_mean_metric_iter_history.append(prev_mean)
                            print("  (No valid candidates to score this iteration)")
                    else:
                        print(f"Prototype {proto_idx + 1}, Iteration {iteration + 1}: No new *unique* guesses found.")
                        if current_prototype_max_metric_iter_history:
                            current_prototype_max_metric_iter_history.append(current_prototype_max_metric_iter_history[-1])
                            current_prototype_mean_metric_iter_history.append(current_prototype_mean_metric_iter_history[-1])
                        else:
                            current_prototype_max_metric_iter_history.append(float('inf') if dist_func == 'l2' else float('-inf'))
                            current_prototype_mean_metric_iter_history.append(float('inf') if dist_func == 'l2' else float('-inf'))
                else:
                    print(f"Prototype {proto_idx + 1}, Iteration {iteration + 1}: LLM returned no new guess strings.")
                    if current_prototype_max_metric_iter_history:
                        current_prototype_max_metric_iter_history.append(current_prototype_max_metric_iter_history[-1])
                        current_prototype_mean_metric_iter_history.append(current_prototype_mean_metric_iter_history[-1])
                    else:
                        current_prototype_max_metric_iter_history.append(float('inf') if dist_func == 'l2' else float('-inf'))
                        current_prototype_mean_metric_iter_history.append(float('inf') if dist_func == 'l2' else float('-inf'))

            except Exception as e:
                print(f"Iteration {iteration + 1} for prototype {proto_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                if current_prototype_max_metric_iter_history:
                    current_prototype_max_metric_iter_history.append(current_prototype_max_metric_iter_history[-1])
                    current_prototype_mean_metric_iter_history.append(current_prototype_mean_metric_iter_history[-1])
                else:
                    placeholder_val = float('inf') if dist_func == 'l2' else float('-inf')
                    current_prototype_max_metric_iter_history.append(placeholder_val)
                    current_prototype_mean_metric_iter_history.append(placeholder_val)

        # Final selection
        if not isinstance(closest_distances, torch.Tensor) or not closest_distances.numel() or not closest_reviews:
            print(f"Warning: No candidates remain for prototype {proto_idx + 1} after optimization.")
            textual_prototypes.append("N/A - Optimization resulted in no candidates")
            distances_list.append(float('inf') if dist_func == 'l2' else float('-inf'))
        else:
            final_candidate_reviews = []
            final_candidate_distances = []
            for i, review_text in enumerate(closest_reviews):
                if not review_text.startswith("placeholder solution "): # Safeguard
                    final_candidate_reviews.append(review_text)
                    final_candidate_distances.append(closest_distances[i].item())

            if final_candidate_reviews:
                final_candidate_distances_tensor = torch.tensor(final_candidate_distances, dtype=torch.float32)
                if dist_func == 'l2':
                    best_dist_val = torch.min(final_candidate_distances_tensor)
                    best_idx_in_filtered = torch.argmin(final_candidate_distances_tensor).item()
                else: # cosine
                    best_dist_val = torch.max(final_candidate_distances_tensor)
                    best_idx_in_filtered = torch.argmax(final_candidate_distances_tensor).item()

                textual_prototypes.append(final_candidate_reviews[best_idx_in_filtered])
                distances_list.append(best_dist_val.item())
                print(f"Selected for P{proto_idx+1}: '{final_candidate_reviews[best_idx_in_filtered][:80]}' with score {best_dist_val.item():.4f}")
            else:
                print(f"Warning: No valid non-placeholder solutions found for prototype {proto_idx + 1} after filtering final candidates.")
                textual_prototypes.append("N/A - No valid non-placeholder solution found")
                distances_list.append(float('inf') if dist_func == 'l2' else float('-inf'))

        all_prototypes_max_metric_history.append(current_prototype_max_metric_iter_history)
        all_prototypes_mean_metric_history.append(current_prototype_mean_metric_iter_history)

    # Assuming plot_convergence_graphs is defined elsewhere
    plot_convergence_graphs(all_prototypes_max_metric_history,
                            all_prototypes_mean_metric_history,
                            dist_func,
                            total_num_prototypes,
                            save_dir="plots")

    return textual_prototypes, distances_list




def extract_python_list(text):
	"""
	Extracts a Python list from a string output.
	Looks for the last occurrence of '[' and ']' and uses ast.literal_eval for safety.
	"""
	start_index = text.rfind('[')
	end_index = text.rfind(']')
	if start_index != -1 and end_index != -1 and start_index < end_index: # ensure valid order
		python_list_str = text[start_index:end_index + 1]
		try:
			python_list = ast.literal_eval(python_list_str)
			if isinstance(python_list, list):
				return python_list
			else:
				print(f"Parsed data is not a list: {python_list_str}")
				return None
		except (SyntaxError, ValueError) as e:
			print(f"Error parsing list from string '{python_list_str}': {e}")
			return None
	else:
		# print("No Python list structure (e.g., [...]) found in the string.") # Can be noisy
		return None


def process_in_batches(model, encodings, dataset_name, data_split='train'):
	"""
	Process model embeddings in batches and save to disk.
	If the file already exists, processing is skipped.
	"""
	batch_size = 1024
	save_path = f"data/{dataset_name}/{data_split}_embeddings.pt"
	
	os.makedirs(os.path.dirname(save_path), exist_ok=True) 
	
	if os.path.exists(save_path):
		print(f"File {save_path} already exists. Skipping processing.")
		return
	
	num_samples = encodings['input_ids'].size(0)
	all_hidden_states = []
	
	model_device = next(model.parameters()).device 
	
	for start_idx in tqdm(range(0, num_samples, batch_size), desc=f"Processing {data_split} embeddings"):
		end_idx = min(start_idx + batch_size, num_samples)
		input_ids_batch = encodings['input_ids'][start_idx:end_idx].to(model_device)
		attention_mask_batch = encodings['attention_mask'][start_idx:end_idx].to(model_device)
		
		with torch.no_grad():
			outputs = model.backbone(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
		hidden_states_batch = outputs.last_hidden_state[:, 0, :].cpu() 
		all_hidden_states.append(hidden_states_batch)
	
	all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)
	torch.save(all_hidden_states_tensor, save_path)
	print(f"Embeddings saved to {save_path}")


def distance_function(prototype_hidden_state, embeddings, distance_func_type):
	"""
	Compute the distance or similarity between the prototype hidden state and embeddings.
	
	Parameters:
		prototype_hidden_state (Tensor): Normalized prototype vector (1, embedding_dim).
		embeddings (Tensor): Normalized embeddings (N, embedding_dim).
		distance_func_type (str): Either "l2" for Euclidean distance or "cosine" for cosine similarity.
	
	Returns:
		Tensor: A 1D tensor containing distances (or similarity scores) of size N.
	"""
	if distance_func_type == 'l2':
		distances = torch.cdist(prototype_hidden_state, embeddings, p=2).squeeze(0) 
	elif distance_func_type == 'cosine':
		distances = F.cosine_similarity(embeddings, prototype_hidden_state.expand_as(embeddings), dim=1)
	else:
		raise ValueError("distance_func_type must be either 'l2' or 'cosine'")
	return distances




# --- [plot_convergence_graphs, extract_python_list, process_in_batches, distance_function, _make_initial_prompt_str remain the same] ---
# (Assuming these functions are defined as in the previous correct answer)
def plot_convergence_graphs(all_prototypes_max_metric_history,
							all_prototypes_mean_metric_history,
							dist_func: str,
							total_num_prototypes: int,
							save_dir: str = "plots"):
	"""
	Plots the convergence of max and mean similarity/distance for each prototype.
	Saves plots to the specified directory.
	"""
	os.makedirs(save_dir, exist_ok=True)

	# --- Plot 1: Max Metric Convergence ---
	plt.figure(figsize=(12, 7))
	for i in range(total_num_prototypes):
		if i < len(all_prototypes_max_metric_history) and all_prototypes_max_metric_history[i]:
			iterations = range(len(all_prototypes_max_metric_history[i]))
			plt.plot(iterations, all_prototypes_max_metric_history[i], label=f'Prototype {i + 1}')

	plt.xlabel("Iteration")
	if dist_func == 'cosine':
		plt.ylabel("Max Cosine Similarity (Higher is Better)")
		plt.title("Max Cosine Similarity Convergence per Prototype")
	elif dist_func == 'l2':
		plt.ylabel("Min L2 Distance (Lower is Better)")
		plt.title("Min L2 Distance Convergence per Prototype")
	else:
		plt.ylabel("Best Metric Value")
		plt.title("Best Metric Convergence per Prototype")

	plt.legend(loc='best', fontsize='small')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, f"max_metric_convergence_{dist_func}.png"))
	plt.close()
	print(f"Saved max metric convergence plot to {os.path.join(save_dir, f'max_metric_convergence_{dist_func}.png')}")

	# --- Plot 2: Mean Metric Convergence ---
	plt.figure(figsize=(12, 7))
	for i in range(total_num_prototypes):
		if i < len(all_prototypes_mean_metric_history) and all_prototypes_mean_metric_history[i]:
			iterations = range(len(all_prototypes_mean_metric_history[i]))
			plt.plot(iterations, all_prototypes_mean_metric_history[i], label=f'Prototype {i + 1}')

	plt.xlabel("Iteration")
	if dist_func == 'cosine':
		plt.ylabel("Mean Cosine Similarity (Higher is Better)")
		plt.title("Mean Cosine Similarity Convergence of Top N Solutions per Prototype")
	elif dist_func == 'l2':
		plt.ylabel("Mean L2 Distance (Lower is Better)")
		plt.title("Mean L2 Distance Convergence of Top N Solutions per Prototype")
	else:
		plt.ylabel("Mean Metric Value")
		plt.title("Mean Metric Convergence of Top N Solutions per Prototype")

	plt.legend(loc='best', fontsize='small')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, f"mean_metric_convergence_{dist_func}.png"))
	plt.close()
	print(f"Saved mean metric convergence plot to {os.path.join(save_dir, f'mean_metric_convergence_{dist_func}.png')}")

	if dist_func == 'cosine':
		plt.figure(figsize=(12, 7))
		for i in range(total_num_prototypes):
			if i < len(all_prototypes_max_metric_history) and all_prototypes_max_metric_history[i]:
				iterations = range(len(all_prototypes_max_metric_history[i]))
				plt.plot(iterations, all_prototypes_max_metric_history[i], label=f'Prototype {i + 1}')
		plt.xlabel("Iteration")
		plt.ylabel("Cosine Similarity")
		plt.title("Max Cosine Similarity Convergence per Prototype")
		plt.legend(loc='best', fontsize='small')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, "cosine_similarity_max_convergence.png"))
		plt.close()
		print(f"Saved specific max cosine similarity plot to {os.path.join(save_dir, 'cosine_similarity_max_convergence.png')}")

		plt.figure(figsize=(12, 7))
		for i in range(total_num_prototypes):
			if i < len(all_prototypes_mean_metric_history) and all_prototypes_mean_metric_history[i]:
				iterations = range(len(all_prototypes_mean_metric_history[i]))
				plt.plot(iterations, all_prototypes_mean_metric_history[i], label=f'Prototype {i + 1}')
		plt.xlabel("Iteration")
		plt.ylabel("Cosine Similarity")
		plt.title("Mean Cosine Similarity Convergence of Top N Solutions per Prototype")
		plt.legend(loc='best', fontsize='small')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, "cosine_similarity_mean_convergence.png"))
		plt.close()
		print(f"Saved specific mean cosine similarity plot to {os.path.join(save_dir, 'cosine_similarity_mean_convergence.png')}")
	else:
		print(f"Skipping specific 'y-axis cosine similarity' plots as dist_func is '{dist_func}'. The generic metric plots were generated.")

    




