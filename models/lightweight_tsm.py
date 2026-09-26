"""Architecture B - a compact, trained-from-scratch video classifier.

Combines MobileNetV2-style inverted residual blocks (depth-wise separable
convolutions) with a zero-parameter Temporal Shift Module for motion
modelling and Squeeze-and-Excitation gating for channel attention. Designed
to train quickly, from random initialisation, on a single consumer GPU.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from models.layers import InvertedResidualBlock, make_divisible

# (expand_ratio, out_channels, repeats, stride, use_tsm, use_se)
STAGE_CONFIG: List[Tuple[int, int, int, int, bool, bool]] = [
    (1, 24, 2, 1, True, False),
    (4, 48, 2, 2, True, True),
    (4, 96, 3, 2, True, True),
    (4, 160, 3, 2, True, True),
    (4, 224, 2, 1, False, True),
]


class LightweightTSMNet(nn.Module):
    def __init__(
        self,
        num_frames: int,
        width_mult: float = 1.0,
        classifier_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        stem_channels = make_divisible(24 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True),
        )

        stage_blocks = []
        in_channels = stem_channels
        for expand_ratio, base_out_channels, repeats, stride, use_tsm, use_se in STAGE_CONFIG:
            out_channels = make_divisible(base_out_channels * width_mult)
            for block_index in range(repeats):
                stage_blocks.append(
                    InvertedResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if block_index == 0 else 1,
                        num_frames=num_frames,
                        use_tsm=use_tsm,
                        use_se=use_se,
                    )
                )
                in_channels = out_channels
        self.stages = nn.Sequential(*stage_blocks)

        head_channels = make_divisible(512 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(head_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.num_frames:
            raise ValueError(f"Expected clips with {self.num_frames} frames, got {num_frames}.")

        x = x.view(batch_size * num_frames, channels, height, width)
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        x = x.mean(dim=[1, 3, 4])  # spatio-temporal global average pooling
        logits = self.classifier(x).squeeze(-1)
        return logits
