"""PyTorch Dataset for short violence/non-violence video clips."""

from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

Sample = Tuple[str, int]


class RWFVideoDataset(Dataset):
    """Loads a video file and returns a fixed-length, transformed clip.

    Frame sampling follows the TSN "segment" strategy: the video is divided
    into `num_frames` equal-length segments and one frame is picked from each
    segment - a random frame during training (temporal jitter augmentation),
    the segment's center frame during evaluation (deterministic).
    """

    def __init__(
        self,
        samples: List[Sample],
        num_frames: int,
        transform: Callable[[np.ndarray], torch.Tensor],
        train: bool,
    ) -> None:
        if not samples:
            raise ValueError("Cannot build a dataset from an empty sample list.")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total_frames: int) -> List[int]:
        n = self.num_frames
        if total_frames <= n:
            # Short video: use every frame, then repeat the last one to pad.
            return list(range(total_frames)) + [total_frames - 1] * (n - total_frames)

        segment_size = total_frames / n
        indices = []
        for i in range(n):
            start = int(i * segment_size)
            end = max(start + 1, min(total_frames, int((i + 1) * segment_size)))
            if self.train:
                indices.append(int(np.random.randint(start, end)))
            else:
                indices.append((start + end - 1) // 2)
        return indices

    def _load_clip(self, path: str) -> np.ndarray:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise IOError(f"Video reports zero frames: {path}")

        indices = self._sample_indices(total_frames)
        needed_indices = set(indices)
        last_needed_index = max(needed_indices)

        # A single forward, sequential pass (with caching) is far more
        # reliable across codecs than seeking with CAP_PROP_POS_FRAMES,
        # which some AVI/H.264 streams handle inaccurately.
        decoded_frames = {}
        current_index = 0
        while current_index <= last_needed_index:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            if current_index in needed_indices:
                decoded_frames[current_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_index += 1
        capture.release()

        if not decoded_frames:
            raise IOError(f"Could not decode any frame from: {path}")

        fallback_frame = next(iter(decoded_frames.values()))
        clip_frames = [decoded_frames.get(index, fallback_frame) for index in indices]
        return np.stack(clip_frames, axis=0)  # (T, H, W, C) uint8, RGB

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        clip = self._load_clip(path)
        clip_tensor = self.transform(clip)
        return clip_tensor, torch.tensor(label, dtype=torch.float32)
