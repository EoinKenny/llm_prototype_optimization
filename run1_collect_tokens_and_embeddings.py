"""
This script will collect the tokens for the models on the datasets for future loading
"""

import subprocess

from config import DATASETS, MODELS, SEEDS


def main():
    
    # Loop through each dataset and run the train_bert.py script
    for dataset in DATASETS:
        for model in MODELS:
            command = f"python src/collect_tokens_embeddings.py --dataset='{dataset}' --model='{model}'"
            try:
                print(f"Running command: {command}")
                subprocess.run(command, check=True, shell=True)
                print(f"Successfully trained BERT on {dataset} {model}.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while training BERT on {dataset} {model} dataset: {e}")
        

if __name__=='__main__':
    main()