"""
Training Script for STC-CapsNet.

Usage:
    python scripts/train.py \
        --config configs/mel_spectrogram.yaml \
        --data_dir data/FoR \
        --feature_type mel \
        --epochs 100 --batch_size 32 --lr 0.001
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import STCCapsNet
from models.losses import MarginLoss
from datasets.for_dataset import FoRDatasetBuilder
from utils.metrics import compute_metrics
from utils.logger import TrainingLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for specs, labels in tqdm(loader, desc="Training", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()

        v = model(specs)
        loss = criterion(v, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        norms = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(norms.argmax(dim=1).cpu().detach().numpy())

    avg_loss = total_loss / len(loader.dataset)
    from utils.metrics import compute_accuracy
    acc = compute_accuracy(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_scores = [], [], []

    for specs, labels in tqdm(loader, desc="Evaluating", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        v = model(specs)
        loss = criterion(v, labels)

        total_loss += loss.item() * specs.size(0)
        norms = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(norms.argmax(dim=1).cpu().numpy())
        all_scores.extend(norms[:, 1].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train STC-CapsNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/default")
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "grayscale"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config) if os.path.exists(args.config) else {}
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  STC-CapsNet Training")
    print(f"  Feature: {args.feature_type} | Device: {device}")
    print(f"  Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    # Build model
    in_channels = 1  # Both mel and grayscale are single-channel
    model = STCCapsNet(in_channels=in_channels, num_classes=2).to(device)
    total, trainable = model.get_num_params()
    print(f"Parameters: {total:,} total, {trainable:,} trainable\n")

    # Loss, optimizer, scheduler (Section III-B)
    criterion = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_val=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Build datasets (70/15/15 split)
    builder = FoRDatasetBuilder(args.data_dir)
    datasets = builder.build(feature_type=args.feature_type, augment_train=True)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    logger = TrainingLogger(args.output_dir, experiment_name="stc_capsnet")
    logger.log(f"Train: {len(datasets['train'])} | Val: {len(datasets['val'])} | Test: {len(datasets['test'])}")

    # Training loop with early stopping
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_eer = val_metrics.get("eer")

        scheduler.step(val_loss)
        logger.log_epoch(epoch, train_loss, val_loss, val_acc, val_eer)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        logger.save_checkpoint(model, optimizer, epoch, val_acc, is_best)

        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.log(f"Early stopping at epoch {epoch}")
            break

    logger.log(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    logger.close()


if __name__ == "__main__":
    main()
