"""Evaluation entry point: runs a trained checkpoint on the held-out test
split (the official RWF-2000 "val" folder) and reports final metrics.

Example usage:
    python evaluate.py --checkpoint checkpoints/lightweight_tsm/best.pt --output logs/lightweight_tsm/test_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from data.dataloaders import build_dataloaders
from models.build import build_model
from training.metrics import compute_binary_metrics, compute_confusion_matrix
from utils.checkpoint import load_checkpoint
from utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write metrics as JSON.")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model: torch.nn.Module, data_loader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_probabilities = [], []

    for clips, labels in tqdm(data_loader, desc="Evaluating"):
        clips = clips.to(device, non_blocking=True)
        logits = model(clips)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
        all_probabilities.append(probabilities)
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probabilities)

    metrics = compute_binary_metrics(y_true, y_prob)
    confusion = compute_confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def main() -> None:
    args = parse_args()

    # weights_only=False: this is a trusted checkpoint produced by this
    # project's own training run (see utils.checkpoint for details).
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config.from_dict(checkpoint_data["config"])

    device = get_device()
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    _, _, test_loader = build_dataloaders(config)
    metrics = run_evaluation(model, test_loader, device)

    print(json.dumps(metrics, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()
