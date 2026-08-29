"""
Training entrypoint. Run with: python -m src.train

Loads data, extracts/caches CNN features, builds the tokenizer, trains
with early stopping + LR scheduling, saves the best checkpoint.
"""
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.data.dataset import load_captions, split_dataset
from src.data.caption_dataset import CaptionDataset
from src.features.extract_features import extract_and_cache
from src.text.tokenizer import Tokenizer
from src.models.caption_model import CaptioningModel
from src.utils import set_seed, get_device, EarlyStopper


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_tokens = 0.0, 0

    for features, captions in loader:
        features, captions = features.to(device), captions.to(device)
        targets = captions[:, 1:]  # shift left (predict next token)

        with torch.set_grad_enabled(train):
            logits = model(features, captions)  # (B, T-1, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
                optimizer.step()

        n_tokens = (targets != 0).sum().item()  # pad_id == 0
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    set_seed(CONFIG.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data
    mapping = load_captions()
    train_ids, val_ids, test_ids = split_dataset(mapping)
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # 2. Image features (cached CNN forward pass)
    all_ids = train_ids + val_ids + test_ids
    features = extract_and_cache(all_ids)
    feature_dim = next(iter(features.values())).shape[-1]  # channel count, last axis

    # 3. Tokenizer (fit on TRAIN captions only, to avoid leakage)
    train_caps = [c for i in train_ids for c in mapping.get(i, [])]
    tokenizer = Tokenizer().fit(train_caps)
    tokenizer.save()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 4. Datasets / loaders
    train_ds = CaptionDataset(train_ids, mapping, features, tokenizer)
    val_ds = CaptionDataset(val_ids, mapping, features, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True,
                               num_workers=CONFIG.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False,
                             num_workers=CONFIG.num_workers)

    # 5. Model / optimiser / scheduler
    model = CaptioningModel(feature_dim, tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[CONFIG.pad_token])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=CONFIG.lr_patience
    )
    early_stopper = EarlyStopper(patience=CONFIG.early_stopping_patience)

    os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    # 6. Training loop
    for epoch in range(1, CONFIG.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": feature_dim,
                "vocab_size": tokenizer.vocab_size,
                "config": CONFIG.__dict__,
            }, CONFIG.best_model_path)
            print(f"  -> saved new best model (val_loss={val_loss:.4f})")

        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("Training complete. Best val_loss:", best_val_loss)


if __name__ == "__main__":
    main()
