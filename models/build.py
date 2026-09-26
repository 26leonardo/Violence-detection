"""Factory that builds the configured model architecture."""

import torch.nn as nn

from config.config import Config
from models.backbone_transformer import BackboneTransformerNet
from models.lightweight_tsm import LightweightTSMNet


def build_model(config: Config) -> nn.Module:
    model_cfg = config.model
    num_frames = config.data.num_frames

    if model_cfg.architecture == "backbone_transformer":
        return BackboneTransformerNet(
            num_frames=num_frames,
            pretrained=model_cfg.pretrained,
            freeze_backbone=model_cfg.freeze_backbone_epochs > 0,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            ff_dim=model_cfg.ff_dim,
            dropout=model_cfg.transformer_dropout,
            classifier_dropout=model_cfg.classifier_dropout,
        )

    if model_cfg.architecture == "lightweight_tsm":
        return LightweightTSMNet(
            num_frames=num_frames,
            width_mult=model_cfg.width_mult,
            classifier_dropout=model_cfg.classifier_dropout,
        )

    raise ValueError(f"Unknown architecture: {model_cfg.architecture!r}")
