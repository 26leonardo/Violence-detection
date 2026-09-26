# Violence Detection in Video - PyTorch Project

Binary video classification (violent vs. non-violent) on the RWF-2000 dataset,
with two interchangeable architectures. Runs locally on a single consumer GPU
(developed against an RTX 4060 Ti, 16 GB VRAM) and on Google Colab.

## 1. The problem

- **Input:** a short video clip (a few seconds), any common container/codec
  OpenCV can decode (`.avi`, `.mp4`, `.mov`, `.mkv`).
- **Output:** a binary label (`Fight` / `NonFight`) with a probability score
  in `[0, 1]`.
- **Data layout expected** (this is exactly how RWF-2000 ships):

  ```
  data/RWF-2000/
    train/Fight/*.avi
    train/NonFight/*.avi
    val/Fight/*.avi
    val/NonFight/*.avi
  ```

  The official `val/` folder is treated as the held-out **test** set; a
  **validation** set is carved out of `train/` via a stratified, video-level
  split (see `data/splits.py`), so no video ever leaks between splits.

## 2. The two architectures

Both take a clip of `num_frames` uniformly-sampled frames as input and output
a single logit.

### Architecture A - `backbone_transformer` (fine-tuned pretrained backbone)

`models/backbone_transformer.py`

- **Backbone:** ImageNet-pretrained ResNet-50, applied per-frame, truncated
  before its final FC layer (2048-d global-average-pooled features).
- **Temporal aggregation:** the 2048-d features are projected to `d_model`,
  prefixed with a learnable `[CLS]` token, given a learned positional
  encoding, and passed through a **pre-norm** Transformer encoder
  (`nn.TransformerEncoderLayer(norm_first=True)` - pre-norm trains more
  stably than the original post-norm formulation).
- **Head:** the `[CLS]` output goes through a small MLP to a single logit.
- **Fine-tuning strategy:** the backbone starts **frozen** (with its
  BatchNorm layers pinned to eval mode so small batches don't corrupt the
  ImageNet running statistics). After `freeze_backbone_epochs` epochs, the
  last two ResNet stages (`layer3`, `layer4`) are unfrozen and the optimizer
  is rebuilt with a second, lower-learning-rate parameter group for them
  (discriminative fine-tuning).
- **Optimizer:** AdamW - weight decay is decoupled from the gradient moments,
  applied directly to the weights.
- **Best for:** highest accuracy ceiling, given enough VRAM/time (default
  config: 224×224 frames, batch size 8).

### Architecture B - `lightweight_tsm` (from-scratch, efficient)

`models/lightweight_tsm.py`, building blocks in `models/layers.py`

- Trained **from random initialisation** - no pretrained weights.
- **Core block:** a MobileNetV2-style *inverted residual block* (expand →
  depth-wise 3×3 conv → project, no activation after the final projection to
  keep the compressed bottleneck linear).
- **Temporal modelling:** a zero-parameter **Temporal Shift Module** shifts a
  fraction of channels forward/backward across frames before most blocks -
  free motion modelling for a purely 2D-convolutional network.
- **Channel attention:** a **Squeeze-and-Excitation** gate follows most
  blocks.
- **Optimizer:** SGD with Nesterov momentum (empirically strong for CNN
  image/video classification when tuned).
- **Best for:** fast iteration and a much smaller memory/compute footprint
  (default config: 112×112 frames, batch size 32, ~1.9M parameters).

Both are selected and configured entirely through `config/config.py` -
`get_default_config("backbone_transformer")` or
`get_default_config("lightweight_tsm")` returns a ready-to-train `Config`.

## 3. Training strategy, loss, and metrics

- **Loss:** `nn.BCEWithLogitsLoss` by default (RWF-2000 is balanced). A
  **binary focal loss** (`training/losses.py`) is available via
  `config.training.loss = "focal"` for imbalanced generalisation experiments.
- **Metrics** (`training/metrics.py`): accuracy, precision, recall, F1, and
  ROC-AUC - the standard set for a (near-)balanced binary classification
  task. Regression metrics (MAE/MSE/R²) are intentionally not computed; the
  target is a label, not a continuous quantity.
- **Schedule:** linear warmup + cosine decay, stepped every training batch.
- **Mixed precision:** `torch.amp.autocast` + `torch.amp.GradScaler`, enabled
  automatically on CUDA and skipped on CPU.
- **Early stopping:** on the monitored validation metric (F1 by default),
  patience configurable.
- **Reproducibility:** `utils/seed.py` seeds Python, NumPy, and PyTorch
  (CPU + all CUDA devices) and sets cuDNN to deterministic mode. See that
  file's docstring for the (small) set of operations that remain
  non-deterministic on GPU regardless.

## 4. Project structure

```
violence_detection/
├── config/config.py          # Dataclass-based configuration + presets
├── data/
│   ├── splits.py              # Directory scanning + stratified train/val/test split
│   ├── dataset.py              # RWFVideoDataset: frame sampling + decoding
│   ├── transforms.py           # Clip-consistent augmentation / eval preprocessing
│   └── dataloaders.py          # DataLoader factory
├── models/
│   ├── layers.py                # SE block, Temporal Shift, Inverted Residual block
│   ├── lightweight_tsm.py       # Architecture B
│   ├── backbone_transformer.py  # Architecture A
│   └── build.py                 # Architecture selector
├── training/
│   ├── losses.py, metrics.py, optim.py, scheduler.py
│   └── trainer.py               # Train/val loop, AMP, checkpointing, early stopping
├── utils/                      # seed, device, checkpoint I/O, logging, param counting
├── inference/predict.py        # ViolenceDetector: checkpoint -> prediction
├── notebooks/experimentation.ipynb
├── train.py / evaluate.py / inference.py   # CLI entry points
├── requirements.txt
└── checkpoints/, logs/         # created at runtime
```

Every Python file in this project has been syntax-checked and smoke-tested
(forward/backward pass, a full multi-epoch training run, checkpoint
save/resume, evaluation, and inference) against synthetic video files.

## 5. Installation

**Local:**
```bash
git clone <this-repo> violence_detection
cd violence_detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Google Colab:**
```python
!git clone <this-repo> /content/violence_detection
%cd /content/violence_detection
!pip install -q -r requirements.txt
```
Then either `!unzip` a copy of RWF-2000 you've uploaded to Drive into
`data/RWF-2000/`, or mount Drive and point `--data-root` at it directly:
```python
from google.colab import drive
drive.mount("/content/drive")
```
```bash
!python train.py --architecture lightweight_tsm \
    --data-root "/content/drive/MyDrive/RWF-2000" \
    --experiment-name colab_run
```
Colab's free-tier GPUs (T4/L4) comfortably fit `lightweight_tsm`; for
`backbone_transformer` you may need to lower `--batch-size` on a T4.

## 6. Training

```bash
# Architecture B (fast, from scratch)
python train.py --architecture lightweight_tsm --data-root ./data/RWF-2000

# Architecture A (fine-tuned backbone)
python train.py --architecture backbone_transformer --data-root ./data/RWF-2000 --batch-size 8

# Resume a run
python train.py --config checkpoints/lightweight_tsm/config.json \
    --resume checkpoints/lightweight_tsm/last.pt

# Common overrides
python train.py --architecture lightweight_tsm --epochs 50 --batch-size 16 --experiment-name my_run
```
Checkpoints go to `checkpoints/<experiment_name>/{best,last}.pt`; TensorBoard
scalars and a `history.jsonl` go to `logs/<experiment_name>/`.
```bash
tensorboard --logdir logs
```

## 7. Evaluation

```bash
python evaluate.py --checkpoint checkpoints/lightweight_tsm/best.pt \
    --output logs/lightweight_tsm/test_metrics.json
```
Runs the checkpoint on the held-out test split (`val/` folder) and prints/
saves accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.

## 8. Inference

```bash
python inference.py --checkpoint checkpoints/lightweight_tsm/best.pt --video /path/to/clip.avi
```
Or from Python:
```python
from inference.predict import ViolenceDetector
detector = ViolenceDetector("checkpoints/lightweight_tsm/best.pt")
print(detector.predict("/path/to/clip.avi"))
```

## 9. Notebook

`notebooks/experimentation.ipynb` covers setup, EDA, sample visualisation,
training, curve plotting, error analysis, final evaluation, checkpoint
reloading, and inference examples - all by importing the `.py` modules above,
never duplicating their logic. Open it locally with the venv's kernel, or
upload it to Colab after cloning the repo there (see §5).

## 10. Common failure cases and debugging

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError: Expected split directory not found` | `--data-root` doesn't point at the folder containing `train/` and `val/`. Check the path. |
| `RuntimeError: No video files found under ...` | RWF-2000 archive extracted with a different folder/casing (e.g. `Fight` vs `fight`). Rename to match `config.data.class_names`, or edit that field. |
| `IOError: Cannot open video file` | Corrupted download or unsupported codec. Re-download the file, or verify `opencv-python-headless` (not the GUI `opencv-python`) is installed. |
| `CUDA out of memory` | Lower `--batch-size`, or for `backbone_transformer` keep the backbone frozen longer (raise `freeze_backbone_epochs`), or lower `config.data.frame_size`. |
| Training loss stuck / NaN | Check `grad_clip_norm > 0` is active; for `backbone_transformer`, confirm frozen BatchNorm layers stayed in eval mode (this is handled automatically by the model's `train()` override - don't call `model.backbone.train()` directly). |
| `roc_auc: NaN` in a metrics dict | The evaluated set contained only one class (e.g. a tiny custom test set) - ROC-AUC is undefined in that case; accuracy/F1 are still valid. |
| Notebook `ModuleNotFoundError` for project packages | The "Project setup" cell didn't find the repo root - edit `PROJECT_ROOT` in that cell to the actual clone location. |
| Colab session disconnects mid-training | Re-launch and pass `--resume checkpoints/<exp>/last.pt` to continue from the last completed epoch. |
| `UserWarning: enable_nested_tensor is True, but ...norm_first was True` | Harmless PyTorch informational warning from the pre-norm Transformer encoder; safe to ignore. |
| Very slow data loading | Video decoding is CPU-bound; raise `config.data.num_workers` up to your CPU's core count (e.g. 8 for a Ryzen 7800X3D), and make sure `pin_memory=True` when training on GPU. |
