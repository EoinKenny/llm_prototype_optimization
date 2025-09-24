To reproduce the paper results, run the following in the terminal

pip install -r requirements.txt
mkdir datasets
cd datasets
mkdir 20newsgroups
mkdir agnews
mkdir amazon_reviews
mkdir dbpedia
mkdir imdb
mkdir trec

#### Download these dataset zip files to datasets/{dataset_name}
https://www.kaggle.com/datasets/ducanger/imdb-dataset
https://www.kaggle.com/datasets/crawford/20-newsgroups
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews
https://www.kaggle.com/datasets/danofer/dbpedia-classes

python reproduce_paper.py
