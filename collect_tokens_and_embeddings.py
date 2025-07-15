import subprocess


def main():
    # List of datasets
    datasets = ['trec', 'dbpedia', '20newsgroups']   #, 'imdb', 'agnews', 'amazon_reviews']

    models = ['bert', 'electra', 'modern_bert', 'llama', 'qwen']

    # Loop through each dataset and run the train_bert.py script
    for dataset in datasets:
        for model in models:
            
            command = f"python src/collect_tokens_embeddings.py --dataset='{dataset}' --model='{model}'"
            
            try:
                print(f"Running command: {command}")
                subprocess.run(command, check=True, shell=True)
                print(f"Successfully trained BERT on {dataset} {model}.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while training BERT on {dataset} {model} dataset: {e}")
        
        
if __name__=='__main__':
    main()