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

CHECKPOINT_PATH = "checkpoint.pt"
MODEL_SAVE_PATH = "model.pt"

EMBED_DIM = 384
NUM_HEADS = 6      # 384 / 6 = 64 per head, standard
NUM_LAYERS = 6
MAX_SEQ_LEN = 64

TRAIN_NEW = True
EPOCH_NUM = 10

WARMUP_STEPS = 500

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")

class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=64, dropout=0.1):
        super(WordPredictor, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.fc.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Scriptable causal mask — plain tensor ops, no static method call
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)

        x = self.norm(x)
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

def save_checkpoint(path, model, optimizer, scheduler, epoch, train_losses, valid_losses):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
    }, path)

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
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

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

            global_step = epoch * len(valid_dataloader) + i
            writer.add_scalar("Loss/valid_batch", loss.item(), global_step)

            running_loss += loss.item() * words.size(0)

    valid_loss = running_loss / len(valid_dataloader.dataset)
    valid_losses.append(valid_loss)

    writer.add_scalar("Valid->Train Distance", valid_loss - train_loss, epoch)
    save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, epoch, train_losses, valid_losses)

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
    model = WordPredictor(vocab_size=vocab_size,
                          embed_dim=EMBED_DIM,
                          num_heads=NUM_HEADS,
                          num_layers=NUM_LAYERS,
                          max_seq_len=MAX_SEQ_LEN)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = EPOCH_NUM * len(train_dataloader)
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, total_steps)

    train_losses = []
    valid_losses = []
    start_epoch = 0

    if not TRAIN_NEW:
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        print(f"Resuming at epoch {start_epoch}/{EPOCH_NUM}")

    model.to(DEVICE)

    if start_epoch >= EPOCH_NUM:
        print(f"Checkpoint epoch ({start_epoch}) already reached EPOCH_NUM ({EPOCH_NUM}). "
              f"Increase EPOCH_NUM to keep training.")

    pbar = tqdm(total=EPOCH_NUM, initial=start_epoch)
    for epoch in range(start_epoch, EPOCH_NUM):
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
        pbar.update(1)

    if train_losses and valid_losses:
        print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Valid Loss: {valid_losses[-1]:.4f}")

    print(f"Exporting TorchScript model to {MODEL_SAVE_PATH}")
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(MODEL_SAVE_PATH)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()