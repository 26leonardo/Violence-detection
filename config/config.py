"""Central configuration for the violence-detection project.

All hyperparameters live here as plain dataclasses so that experiments can be
changed (via CLI overrides, a saved JSON file, or `get_default_config`)
without touching any training code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple, Union


@dataclass
class DataConfig:
    """Dataset location, sampling, and DataLoader settings."""

    data_root: str = "./data/RWF-2000"
    train_dir: str = "train"
    val_dir: str = "val"
    class_names: Tuple[str, str] = ("NonFight", "Fight")
    num_frames: int = 16
    frame_size: int = 224
    val_ratio: float = 0.1
    num_workers: int = 1
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Architecture selection and its hyperparameters.

    Fields only used by one architecture are simply ignored by the other
    (e.g. `d_model` is unused by `lightweight_tsm`).
    """

    architecture: str = "lightweight_tsm"  # "lightweight_tsm" or "backbone_transformer"

    # backbone_transformer only
    pretrained: bool = True
    freeze_backbone_epochs: int = 5
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    transformer_dropout: float = 0.1

    # lightweight_tsm only
    width_mult: float = 1.0

    # shared
    classifier_dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Optimisation, regularisation, and experiment-control settings."""

    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 40
    optimizer: str = "adamw"  # "adamw" or "sgd"
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    loss: str = "bce"  # "bce" or "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    mixed_precision: bool = True
    early_stopping_patience: int = 8
    monitor_metric: str = "f1"


@dataclass
class PathsConfig:
    """Where to write checkpoints and logs."""

    experiment_name: str = "violence_detection_exp"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class Config:
    """Top-level configuration bundling all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Config":
        return cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            paths=PathsConfig(**payload.get("paths", {})),
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as config_file:
            json.dump(self.to_dict(), config_file, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        with open(path) as config_file:
            return cls.from_dict(json.load(config_file))


def get_default_config(architecture: str) -> Config:
    """Returns a ready-to-train config for one of the two project architectures.

    These defaults encode the design choices discussed for each model: the
    fine-tuned backbone uses higher resolution, a small batch size, and AdamW
    with a low learning rate; the from-scratch lightweight model uses lower
    resolution, a larger batch size, and SGD with Nesterov momentum.
    """

    config = Config()
    config.model.architecture = architecture
    config.paths.experiment_name = architecture

    if architecture == "backbone_transformer":
        config.data.frame_size = 224
        config.training.batch_size = 8
        config.training.num_epochs = 30
        config.training.optimizer = "adamw"
        config.training.learning_rate = 1e-4
        config.training.backbone_learning_rate = 1e-5
        config.training.weight_decay = 1e-4
        config.model.pretrained = True
        config.model.freeze_backbone_epochs = 5
    elif architecture == "lightweight_tsm":
        config.data.frame_size = 112
        config.training.batch_size = 32
        config.training.num_epochs = 100
        config.training.optimizer = "sgd"
        config.training.learning_rate = 0.05
        config.training.weight_decay = 5e-5
        config.model.pretrained = False
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    return config
