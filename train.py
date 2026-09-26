"""Main training entry point.

Example usage:
    python train.py --architecture lightweight_tsm --data-root ./data/RWF-2000
    python train.py --architecture backbone_transformer --epochs 30 --batch-size 8
    python train.py --config checkpoints/lightweight_tsm/config.json --resume checkpoints/lightweight_tsm/last.pt
"""

import argparse
from pathlib import Path

from config.config import Config, get_default_config
from data.dataloaders import build_dataloaders
from models.build import build_model
from training.trainer import Trainer
from utils.device import get_device
from utils.model_utils import count_parameters
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a violence-detection video classifier.")
    parser.add_argument(
        "--architecture",
        choices=["backbone_transformer", "lightweight_tsm"],
        default="lightweight_tsm",
        help="Which model design to train.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a saved JSON config (overrides --architecture).")
    parser.add_argument("--data-root", type=str, default=None, help="Override the dataset root directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override the experiment name.")
    parser.add_argument(
        "--monitor-metric", type=str, default=None,
        help="Metric to track for checkpointing/early stopping, e.g. 'f1', 'f2', 'recall', 'roc_auc'.",
    )
    parser.add_argument(
        "--unfreeze-num-stages", type=int, default=None, choices=[1, 2],
        help="backbone_transformer only: 1 unfreezes only layer4, 2 unfreezes layer3+layer4 "
        "(more trainable params, more VRAM). Lower this if you hit CUDA OOM right after unfreezing.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.experiment_name is not None:
        config.paths.experiment_name = args.experiment_name
    if args.monitor_metric is not None:
        config.training.monitor_metric = args.monitor_metric
    if args.unfreeze_num_stages is not None:
        config.model.unfreeze_num_stages = args.unfreeze_num_stages
    return config


def main() -> None:
    args = parse_args()
    config = Config.load(args.config) if args.config else get_default_config(args.architecture)
    config = apply_overrides(config, args)

    set_seed(config.training.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(
        f"Dataset sizes -> train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )

    model = build_model(config)
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters -> total: {total_params:,}, trainable: {trainable_params:,}")

    trainer = Trainer(model, train_loader, val_loader, config, device, resume_from=args.resume)

    config_save_path = Path(config.paths.checkpoint_dir) / config.paths.experiment_name / "config.json"
    config.save(config_save_path)

    trainer.fit()


if __name__ == "__main__":
    main()
