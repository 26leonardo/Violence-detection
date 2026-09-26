"""Reusable building blocks shared by the lightweight architecture."""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Channel-attention gate: global-average-pool -> compress -> excite -> sigmoid."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.pool(x))
        return x * scale


class TemporalShift(nn.Module):
    """Shifts a fraction of channels forward/backward along the time axis.

    Zero-parameter, zero-FLOP mechanism that lets a per-frame (2D) block
    exchange information with neighbouring frames before convolving.
    Input/output shape: (batch * num_frames, channels, height, width).
    """

    def __init__(self, num_frames: int, shift_ratio: float = 1 / 8) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.shift_ratio = shift_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_times_frames, channels, height, width = x.shape
        num_frames = self.num_frames
        if batch_times_frames % num_frames != 0:
            raise ValueError(
                f"Batch*time dimension ({batch_times_frames}) is not divisible "
                f"by num_frames ({num_frames})."
            )
        batch_size = batch_times_frames // num_frames
        x = x.view(batch_size, num_frames, channels, height, width)

        fold = max(1, int(channels * self.shift_ratio))
        shifted = torch.zeros_like(x)
        shifted[:, :-1, :fold] = x[:, 1:, :fold]  # bring the future into the present
        shifted[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # bring the past forward
        shifted[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # untouched channels

        return shifted.view(batch_times_frames, channels, height, width)


def make_divisible(value: float, divisor: int = 8) -> int:
    """Rounds `value` to the nearest multiple of `divisor` (standard
    MobileNet-style channel rounding so width scaling stays hardware-friendly)."""

    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual block: expand -> depth-wise conv ->
    (optional SE gate) -> project, with an optional temporal shift applied
    first and a residual connection when shapes allow it.

    No activation follows the final projection, which keeps the compressed
    bottleneck representation linear and avoids losing information to ReLU
    clipping in a low-dimensional space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        num_frames: int,
        use_tsm: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        self.use_residual = stride == 1 and in_channels == out_channels
        self.temporal_shift = TemporalShift(num_frames) if use_tsm else nn.Identity()

        hidden_dim = int(round(in_channels * expand_ratio))
        expand_layers = []
        if expand_ratio != 1:
            expand_layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        expand_layers += [
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ]
        self.expand_and_depthwise = nn.Sequential(*expand_layers)
        self.se = SqueezeExcitation(hidden_dim) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.temporal_shift(x)
        out = self.expand_and_depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out
