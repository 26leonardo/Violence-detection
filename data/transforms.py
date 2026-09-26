"""Clip-level transforms.

Frames within a clip must be augmented *identically* (same crop region, same
flip decision, same colour jitter factors) - independently randomising each
frame would scramble the temporal signal the model relies on. Both
transforms below therefore sample their random parameters once per clip and
re-use them across every frame.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ClipTrainTransform:
    def __init__(self, frame_size: int, resize_short_side: int = 0):
        self.frame_size = frame_size
        # Resize a bit larger than the crop so RandomCrop has room to move.
        self.resize_short_side = resize_short_side or int(frame_size * 1.15)

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        do_flip = random.random() < 0.5
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        crop_top, crop_left = None, None

        frames = []
        for frame in clip:
            tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W) uint8
            # A single int preserves aspect ratio (resizes the shorter side).
            tensor = TF.resize(tensor, self.resize_short_side, antialias=True)

            if crop_top is None:
                _, height, width = tensor.shape
                max_top = max(0, height - self.frame_size)
                max_left = max(0, width - self.frame_size)
                crop_top = random.randint(0, max_top)
                crop_left = random.randint(0, max_left)

            tensor = TF.crop(tensor, crop_top, crop_left, self.frame_size, self.frame_size)
            if do_flip:
                tensor = TF.hflip(tensor)

            tensor = tensor.float() / 255.0
            tensor = TF.adjust_brightness(tensor, brightness)
            tensor = TF.adjust_contrast(tensor, contrast)
            tensor = TF.adjust_saturation(tensor, saturation)
            tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            frames.append(tensor)

        return torch.stack(frames, dim=0)  # (T, C, H, W)


class ClipEvalTransform:
    def __init__(self, frame_size: int, resize_short_side: int = 0):
        self.frame_size = frame_size
        self.resize_short_side = resize_short_side or int(frame_size * 1.15)

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        frames = []
        for frame in clip:
            tensor = torch.from_numpy(frame).permute(2, 0, 1)
            tensor = TF.resize(tensor, self.resize_short_side, antialias=True)
            tensor = TF.center_crop(tensor, [self.frame_size, self.frame_size])
            tensor = tensor.float() / 255.0
            tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            frames.append(tensor)
        return torch.stack(frames, dim=0)
