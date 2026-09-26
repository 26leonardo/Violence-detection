"""Lightweight experiment tracking: TensorBoard scalars plus a JSONL history
file that is trivial to re-parse for plotting in the notebook."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Union

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, log_dir: Union[str, Path], experiment_name: str, append: bool = False) -> None:
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.history_path = self.log_dir / "history.jsonl"
        # A *fresh* run (append=False) truncates any stale history left over
        # from a previous, unrelated run under the same experiment name -
        # otherwise epoch numbers restart from 0 and get appended after the
        # previous run's epochs, which plots as a nonsensical sawtooth line.
        # Only an actual `Trainer(..., resume_from=...)` run should append.
        mode = "a" if append else "w"
        self._history_file = open(self.history_path, mode)

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        with open(self.log_dir / "config.json", "w") as config_file:
            json.dump(config_dict, config_file, indent=2)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)
        record = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self._history_file.write(json.dumps(record) + "\n")
        self._history_file.flush()

    def close(self) -> None:
        self.writer.close()
        self._history_file.close()
