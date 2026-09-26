"""Builds train/val/test DataLoaders from a Config object."""

from typing import Tuple

from torch.utils.data import DataLoader

from config.config import Config
from data.dataset import RWFVideoDataset
from data.splits import build_sample_lists
from data.transforms import ClipEvalTransform, ClipTrainTransform


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_samples, val_samples, test_samples = build_sample_lists(config.data, config.training.seed)

    train_transform = ClipTrainTransform(config.data.frame_size)
    eval_transform = ClipEvalTransform(config.data.frame_size)

    train_dataset = RWFVideoDataset(train_samples, config.data.num_frames, train_transform, train=True)
    val_dataset = RWFVideoDataset(val_samples, config.data.num_frames, eval_transform, train=False)
    test_dataset = RWFVideoDataset(test_samples, config.data.num_frames, eval_transform, train=False)

    loader_kwargs = dict(
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.num_workers > 0,
    )
    # drop_last=True keeps BatchNorm statistics stable across batches of equal
    # size, but if the training set is smaller than one batch (e.g. a small
    # debugging subset) it would silently yield zero batches - guard against
    # that rather than let the DataLoader raise StopIteration on first use.
    can_drop_last = len(train_dataset) >= config.training.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=can_drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader
