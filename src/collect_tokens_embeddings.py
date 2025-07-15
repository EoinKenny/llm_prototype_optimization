"""
This script trains a BERT model to classify IMDB, it also saves the input encodings for
use in training the prototype classifiers later on.
Only the last layer of BERT is finetuned. This is necessary for fair comparison to prototype methods.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW

from functions import load_domain


def main(args):
    data_utils = load_domain(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,  default='llama',  help='backbone model')
    parser.add_argument('--input_size', type=int,  default=256,      help='size of input token length')
    parser.add_argument('--num_epochs', type=int,  default=2,        help='Number of epochs')
    parser.add_argument('--dataset',    type=str,  default='imdb',   help='dataset')
    parser.add_argument('--train_full', type=int,  default=1,        help='Should we finetune the full BERT model?')
    parser.add_argument('--device',     type=str,  default='cuda:0', help='device num')
    parser.add_argument('--prototype_dim', type=int, default=256, help='dimension of prototypes')
    parser.add_argument(
        '--baseline',
        action='store_true',          # ← sets baseline = True when the flag is present
        default=False,                # (optional) defaults to False when omitted
        help='Use baseline losses'
    )
    parser.add_argument(
        '--no_llm_head',
        action='store_true',          # ← sets baseline = True when the flag is present
        default=True,                # (optional) defaults to False when omitted
        help='Use mlp on top of llm'
    )
    args = parser.parse_args()    
    main(args)
    

    