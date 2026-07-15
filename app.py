import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import kagglehub
import json
import string
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(WordPredictor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.3)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.rnn_dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim*2)
        
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        
        x, _ = self.rnn(x)
        x = self.rnn_dropout(x)
        x = self.norm(x)

        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class WordDataset(Dataset):
    def __init__(self, data_file, split=1, offset=0):
        self.data = []
        dataset_file = open(data_file, "r").read().replace("\n", "")
        dataset_length = len(dataset_file)
        
        dataset = tokenise(dataset_file[int(dataset_length * offset): int(dataset_length * split)])
        used_length = 0
        print(len(dataset))
        while used_length + 26 <= len(dataset):
            rand_length = int(random.random() * 15) + 10
            self.data.append((dataset[used_length:used_length+rand_length], dataset[used_length+rand_length+1]))
            used_length = used_length + rand_length + 1
        print(self.data)

    def __len__(self):
        self.data.len()

    def __getitem__(self, idx):
        word, label = self.data[idx]
        return word, label

WordDataset("training_data/input.txt", split=0.000015)