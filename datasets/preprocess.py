import os
import shutil
import pandas as pd

# Define the directory paths
imdb_dir = 'trec'
preprocess_dir = 'preprocess/trec'

# Delete the 'preprocess' directory if it exists
if os.path.exists('preprocess'):
    shutil.rmtree('preprocess')

# Create the 'preprocess/imdb' directory
os.makedirs(preprocess_dir, exist_ok=True)

# List of CSV files to process
csv_files = ['train.csv', 'test.csv']

# Iterate over the CSV files in the 'imdb' directory
for csv_file in csv_files:
    file_path = os.path.join(imdb_dir, csv_file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Rename the columns
    df.rename(columns={'label-fine': 'label'}, inplace=True)
    
    # Save the modified DataFrame to the 'preprocess/imdb' directory
    new_file_path = os.path.join(preprocess_dir, csv_file)
    df.to_csv(new_file_path, index=False)

print("TREC Processing complete. Modified files are saved in 'preprocess/trec'.")







import os
import pandas as pd

# Define the directory paths
dbpedia_dir = 'dbpedia'
preprocess_dbpedia_dir = 'preprocess/dbpedia'

# Mapping of original file names to new file names
file_mapping = {
    'DBPEDIA_train.csv': 'train.csv',
    'DBPEDIA_test.csv': 'test.csv',
    'DBPEDIA_val.csv': 'val.csv'
}

# The exact label codes you printed out
target_classes = [185, 166, 159, 57, 160, 168, 146, 198,
                  123, 38, 1, 73, 36, 56, 54, 215,
                  39, 128, 90, 171]

# Initialize an empty DataFrame to accumulate samples
df_accumulated = pd.DataFrame()

# Iterate over the CSV files in the 'dbpedia' directory
for original_file in file_mapping.keys():
    file_path = os.path.join(dbpedia_dir, original_file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Create a new 'label' column with integer values for each unique class in 'l3'
    df['label'] = df['l3'].astype('category').cat.codes
    
    # Filter to only the target classes
    df = df[df['label'].isin(target_classes)]
    
    # Accumulate samples from each file
    df_accumulated = pd.concat([df_accumulated, df], ignore_index=True)

# (Optional) Remap labels to 0..19 so your model sees contiguous classes
df_accumulated['label'] = df_accumulated['label'].astype('category').cat.codes

# Sample 100,000 examples from the accumulated DataFrame
df_sampled = df_accumulated

# Split into train and test sets (90:10)
train_size = int(0.9 * len(df_sampled))
train_df = df_sampled.iloc[:train_size].reset_index(drop=True)
test_df  = df_sampled.iloc[train_size:].reset_index(drop=True)

# Save the datasets
os.makedirs(preprocess_dbpedia_dir, exist_ok=True)
train_df.to_csv(os.path.join(preprocess_dbpedia_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(preprocess_dbpedia_dir, 'test.csv'), index=False)

print("Processing complete. Train and test datasets (only your 20 classes) are saved in 'preprocess/dbpedia'.")

# import os
# import pandas as pd

# # Define the directory paths
# dbpedia_dir = 'dbpedia'
# preprocess_dbpedia_dir = 'preprocess/dbpedia'

# # Create the 'preprocess/dbpedia' directory
# os.makedirs(preprocess_dbpedia_dir, exist_ok=True)

# # Mapping of original file names to new file names
# file_mapping = {
#     'DBPEDIA_train.csv': 'train.csv',
#     'DBPEDIA_test.csv': 'test.csv',
#     'DBPEDIA_val.csv': 'val.csv'
# }

# # Initialize an empty DataFrame to accumulate samples
# df_accumulated = pd.DataFrame()

# # Iterate over the CSV files in the 'dbpedia' directory
# for original_file in file_mapping.keys():
#     file_path = os.path.join(dbpedia_dir, original_file)
    
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
    
#     # Create a new 'label' column with integer values for each unique class in 'l2'
#     df['label'] = df['l2'].astype('category').cat.codes
    
#     # Accumulate samples from each file
#     df_accumulated = pd.concat([df_accumulated, df], ignore_index=True)

# # Sample 100,000 examples from the accumulated DataFrame
# df_sampled = df_accumulated.sample(n=100000, random_state=42)

# # Split into train and test sets (90:10)
# train_size = int(0.9 * len(df_sampled))

# train_df = df_sampled[:train_size]
# test_df = df_sampled[train_size:]

# # Save the datasets
# train_df.to_csv(os.path.join(preprocess_dbpedia_dir, 'train.csv'), index=False)
# test_df.to_csv(os.path.join(preprocess_dbpedia_dir, 'test.csv'), index=False)

# print("Processing complete. Train and test datasets are saved in 'preprocess/dbpedia'.")










# # Read the JSON file into a DataFrame
# df = pd.read_json('amazon_reviews/Cell_Phones_and_Accessories_5.json', lines=True)

# # Select relevant columns
# df = df[['reviewText', 'overall', 'summary']]

# # Define sentiment based on the 'overall' rating
# def defineSentiment(rating):
#     if rating >= 4:
#         return 'Positive'
#     elif rating == 3:
#         return 'Neutral'
#     else:
#         return 'Negative'

# # Apply sentiment definition
# df['sentiment'] = df['overall'].apply(defineSentiment)

# # Create a new 'label' column with integer values for each sentiment
# sentiment_to_label = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
# df['label'] = df['sentiment'].map(sentiment_to_label)

# # Rename 'reviewText' to 'text'
# df.rename(columns={'reviewText': 'text'}, inplace=True)

# # Sample 100,000 rows


# def is_valid_text(x):
#     # Reject NaN, None, non-string types, empty/whitespace-only strings
#     if pd.isna(x):
#         return False
#     if isinstance(x, str):
#         return x.strip() != ""
#     return False
# print('Before filter', df.shape)

# # Apply the filter
# df = df[df["text"].apply(is_valid_text)].copy()
# df["text"] = df["text"].apply(str).str.strip()
# df.reset_index(drop=True, inplace=True)
# print('After filter', df.shape)
# df_sampled = df.sample(n=100000, random_state=0)


# # Split into train, validation, and test sets (80:10:10)
# train_size = int(0.8 * len(df_sampled))
# val_size = int(0.1 * len(df_sampled))

# train_df = df_sampled[:train_size]
# val_df = df_sampled[train_size:train_size + val_size]
# test_df = df_sampled[train_size + val_size:]

# # Create directory for saving the datasets
# preprocess_amazon_dir = 'preprocess/amazon_reviews'
# os.makedirs(preprocess_amazon_dir, exist_ok=True)

# # Save the datasets
# train_df.to_csv(os.path.join(preprocess_amazon_dir, 'train.csv'), index=False)
# val_df.to_csv(os.path.join(preprocess_amazon_dir, 'val.csv'), index=False)
# test_df.to_csv(os.path.join(preprocess_amazon_dir, 'test.csv'), index=False)
# print("Processing complete. Train, validation, and test datasets are saved in 'preprocess/amazon_reviews'.")











from sklearn.datasets import fetch_20newsgroups


newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

# Create DataFrames for train and test sets
train_df = pd.DataFrame({'text': newsgroups_train.data, 'label': newsgroups_train.target})
test_df = pd.DataFrame({'text': newsgroups_test.data, 'label': newsgroups_test.target})

# Create directory for saving the datasets
preprocess_newsgroups_dir = 'preprocess/20newsgroups'
os.makedirs(preprocess_newsgroups_dir, exist_ok=True)

# Save the DataFrames as CSV files
train_df.to_csv(os.path.join(preprocess_newsgroups_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(preprocess_newsgroups_dir, 'test.csv'), index=False)

print("Processing complete. Train and test datasets are saved in 'preprocess/20newsgroups'.")





# import os
# import shutil
# import pandas as pd

# # Define the directory paths
# imdb_dir = 'imdb'
# preprocess_dir = 'preprocess/imdb'

# # Create the 'preprocess/imdb' directory
# os.makedirs(preprocess_dir, exist_ok=True)

# # List of CSV files to process
# csv_files = ['train.csv', 'test.csv', 'val.csv']

# # Iterate over the CSV files in the 'imdb' directory
# for csv_file in csv_files:
#     file_path = os.path.join(imdb_dir, csv_file)
    
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
    
#     # Rename the columns
#     df.rename(columns={'sentiment': 'label', 'review': 'text'}, inplace=True)
    
#     # Save the modified DataFrame to the 'preprocess/imdb' directory
#     new_file_path = os.path.join(preprocess_dir, csv_file)
#     df.to_csv(new_file_path, index=False)

# print("IMDB Processing complete. Modified files are saved in 'preprocess/imdb'.")





# # Define the directory paths
# agnews_dir = 'agnews'
# preprocess_agnews_dir = 'preprocess/agnews'

# # Create the 'preprocess/agnews' directory
# os.makedirs(preprocess_agnews_dir, exist_ok=True)

# # List of CSV files to process
# csv_files = ['train.csv', 'test.csv']

# # Iterate over the CSV files in the 'agnews' directory
# for csv_file in csv_files:
#     file_path = os.path.join(agnews_dir, csv_file)
    
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
    
#     # Rename the columns
#     df.rename(columns={'Description': 'text', 'Class Index': 'label'}, inplace=True)
    
#     df.label = df.label-1
    
#     # Save the modified DataFrame to the 'preprocess/agnews' directory
#     new_file_path = os.path.join(preprocess_agnews_dir, csv_file)
#     df.to_csv(new_file_path, index=False)

# print("AGNEWS Processing complete. Modified files are saved in 'preprocess/agnews'.")








