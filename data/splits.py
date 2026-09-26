"""Scans the on-disk RWF-2000 layout and builds train/val/test sample lists.

Expected directory layout (the layout RWF-2000 ships with):

    <data_root>/train/<class_names[0]>/*.avi
    <data_root>/train/<class_names[1]>/*.avi
    <data_root>/val/<class_names[0]>/*.avi
    <data_root>/val/<class_names[1]>/*.avi

The official "val" folder is treated as the held-out **test** set, and a
validation set is carved out of "train" via a stratified split. This avoids
ever tuning on the official test videos, and the split happens at the
video level so no clip can leak between splits.
"""

import random
from pathlib import Path
from typing import List, Tuple

Sample = Tuple[str, int]

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def scan_directory(root: str, split_dir: str, class_names: Tuple[str, str]) -> List[Sample]:
    base = Path(root) / split_dir
    if not base.exists():
        raise FileNotFoundError(f"Expected split directory not found: {base}")

    samples: List[Sample] = []
    for label, class_name in enumerate(class_names):
        class_dir = base / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
        for video_path in sorted(class_dir.iterdir()):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((str(video_path), label))

    if not samples:
        raise RuntimeError(f"No video files found under {base}")
    return samples


def stratified_split(
    samples: List[Sample], val_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample]]:
    """Splits `samples` into (train, val), preserving the class ratio."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = random.Random(seed)
    samples_by_label = {}
    for sample in samples:
        samples_by_label.setdefault(sample[1], []).append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    for label_samples in samples_by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        num_val = max(1, int(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:num_val])
        train_samples.extend(shuffled[num_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_sample_lists(data_cfg, seed: int) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Returns (train_samples, val_samples, test_samples)."""

    train_full = scan_directory(data_cfg.data_root, data_cfg.train_dir, data_cfg.class_names)
    test_samples = scan_directory(data_cfg.data_root, data_cfg.val_dir, data_cfg.class_names)
    train_samples, val_samples = stratified_split(train_full, data_cfg.val_ratio, seed)
    return train_samples, val_samples, test_samples
