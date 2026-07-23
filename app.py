import numpy as np
import pandas as pd

import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"
INPUT_FOLDER = "training_data"

DATAFILE = INPUT_FOLDER + "/tokenized_input.json"
MODEL_INFO_FILE = INPUT_FOLDER + "/model.json"
MODEL_SAVE_PATH = "model.pt"

EMBED_DIM = 384
HIDDEN_DIM = 768

TRAIN_NEW = True
EPOCH_NUM = 10

WARMUP_STEPS = 500

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")

class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(WordPredictor, self).__init__()

        self.encoding_embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoding_dropout = nn.Dropout(0.1)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.rnn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.encoding_embedding(x)
        x = self.encoding_dropout(x)
        
        x, _ = self.rnn(x)
        x = self.norm(x)
        x = self.rnn_dropout(x)

        x = self.fc(x)
        return x

class WordDataset(Dataset):
    def __init__(self, data_file, split=1, offset=0):
        f = json.load(open(data_file))
        self.data = torch.tensor(
            f[int(len(f) * offset): int(len(f) * offset) + int(len(f) * split)],
            dtype=torch.long
        )

    def __len__(self):
        return math.floor(len(self.data) - 64) // 8

    def __getitem__(self, idx):
        idx *= 8

        word = self.data[idx:idx+64]
        label = self.data[idx+1:idx+65]

        return word, label

def build_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,   # start at 1% of base LR
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]  # switch from warmup -> cosine at this step
    )

def get_model_info():
    vocab_size = 0
    with open(MODEL_INFO_FILE) as f:
        vocab_size = json.load(f)["vocab_size"]
    return vocab_size

def epoch_func(epoch, model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, vocab_size, train_losses, valid_losses, writer):    
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{EPOCH_NUM}(training)",
        leave=False
    )

    for i, (words, labels) in enumerate(pbar):
        words = words.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            output = model(words)

            loss = criterion(output.view(-1, vocab_size), labels.view(-1)) # check this

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        global_step = epoch * len(train_dataloader) + i
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            
        running_loss += loss.item() * words.size(0)
            
    train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(train_loss)
        
    model.eval()
    running_loss = 0.0

    pbar = tqdm(
        valid_dataloader,
        desc=f"Epoch {epoch+1}/{EPOCH_NUM}(validating)",
        leave=False
    )

    with torch.no_grad():
        for i, (words, labels) in enumerate(pbar):
            words = words.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(words)
            
            loss = criterion(output.reshape(-1, vocab_size), labels.reshape(-1))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step = epoch * len(valid_dataloader) + i
            writer.add_scalar("Loss/valid_batch", loss.item(), global_step)

            running_loss += loss.item() * words.size(0)
        
    valid_loss = running_loss / len(valid_dataloader.dataset)
    valid_losses.append(valid_loss)

    writer.add_scaler("Valid->Train Distance", valid_loss - train_loss, epoch);

    scripted = torch.jit.script(model)
    scripted.save("model.pt")

def main():
    print("Loading dataset, this may take a while.")

    device = torch.device(DEVICE)
    train_dataloader = DataLoader(WordDataset(DATAFILE, split=0.8), 
        batch_size=512, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)
    
    valid_dataloader = DataLoader(WordDataset(DATAFILE, split=0.16, offset=0.8), 
        batch_size=512, 
        shuffle=False,        
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)
    test_dataloader = DataLoader(WordDataset(DATAFILE, split=0.04, offset=0.96), 
        batch_size=512, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)

    
    writer = SummaryWriter()

    vocab_size = get_model_info()
    model = WordPredictor(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    if (not TRAIN_NEW):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = EPOCH_NUM * len(train_dataloader)
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, total_steps)

    train_losses = []
    valid_losses = []

    model.to(DEVICE)
    pbar = tqdm(total=EPOCH_NUM)
    for epoch in range(EPOCH_NUM):
        epoch_func(        
            epoch,
            model,
            train_dataloader,
            valid_dataloader,
            criterion,
            optimizer,
            scheduler,
            vocab_size,
            train_losses,
            valid_losses,
            writer)
        pbar.update(1);

    print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Valid Loss: {valid_losses[-1]:.4f}")
        
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()