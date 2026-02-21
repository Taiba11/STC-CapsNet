"""
Evaluation Script for STC-CapsNet (including cross-dataset on ASVspoof 2019).

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/for_mel/best_model.pth \
        --data_dir data/FoR --feature_type mel

    # Cross-dataset
    python scripts/evaluate.py \
        --checkpoint experiments/for_mel/best_model.pth \
        --data_dir data/ASVspoof2019/LA --dataset asvspoof2019 --feature_type mel
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import STCCapsNet
from models.losses import MarginLoss
from datasets.for_dataset import FoRDatasetBuilder
from datasets.asvspoof2019 import ASVspoof2019Builder
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


@torch.no_grad()
def evaluate_dataset(model, loader, device):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    for specs, labels in tqdm(loader, desc="Evaluating"):
        specs = specs.to(device)
        v = model(specs)
        norms = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)

        all_labels.extend(labels.numpy())
        all_preds.extend(norms.argmax(dim=1).cpu().numpy())
        all_scores.extend(norms[:, 1].cpu().numpy())

    return compute_metrics(all_labels, all_preds, all_scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate STC-CapsNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="for", choices=["for", "asvspoof2019"])
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "grayscale"])
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = STCCapsNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Build dataset
    if args.dataset == "for":
        builder = FoRDatasetBuilder(args.data_dir)
        datasets = builder.build(feature_type=args.feature_type)
        test_dataset = datasets["test"]
    else:
        builder = ASVspoof2019Builder(args.data_dir)
        test_dataset = builder.get_eval_dataset(feature_type=args.feature_type)

    loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    metrics = evaluate_dataset(model, loader, device)

    print(f"\n{'='*60}")
    print(f"  STC-CapsNet Evaluation | {args.dataset} | {args.feature_type}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    print(f"  EER:       {metrics.get('eer', 0):.4f}%")
    print(f"{'='*60}\n")

    # Save
    save_path = os.path.join(args.output_dir, f"eval_{args.dataset}_{args.feature_type}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved to {save_path}")

    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
