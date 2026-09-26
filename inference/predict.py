"""Inference pipeline: load a trained checkpoint, preprocess a new video the
exact same way it was preprocessed at evaluation time, and return a
prediction with its confidence score."""

from pathlib import Path
from typing import Dict, Optional

import torch

from config.config import Config
from data.dataset import RWFVideoDataset
from data.transforms import ClipEvalTransform
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.device import get_device


class ViolenceDetector:
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None) -> None:
        # weights_only=False: this is a trusted checkpoint produced by this
        # project's own training run (see utils.checkpoint for details).
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = Config.from_dict(checkpoint_data["config"])
        self.device = device or get_device()

        self.model = build_model(self.config).to(self.device)
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

        self.transform = ClipEvalTransform(self.config.data.frame_size)
        self.class_names = self.config.data.class_names

    @torch.no_grad()
    def predict(self, video_path: str, threshold: float = 0.5) -> Dict[str, float]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Reuses the dataset's clip-loading logic so inference sees exactly
        # the same frame-sampling behaviour used during evaluation.
        single_video_dataset = RWFVideoDataset(
            samples=[(video_path, 0)],
            num_frames=self.config.data.num_frames,
            transform=self.transform,
            train=False,
        )
        clip, _ = single_video_dataset[0]
        clip = clip.unsqueeze(0).to(self.device)

        logits = self.model(clip)
        probability = torch.sigmoid(logits).item()
        predicted_index = int(probability >= threshold)

        return {
            "label": self.class_names[predicted_index],
            "probability": probability,
            "threshold": threshold,
        }
