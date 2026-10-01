# Violence Detection in Video

Binary video classification (violent vs. non-violent) on the RWF-2000 dataset,
with two interchangeable architectures. Runs locally on a single consumer GPU
(developed against an RTX 4060 Ti, 16 GB VRAM) and on Google Colab.

Read `notebooks/experimentation_final.ipynb` for the full experiment and
results; `notebooks/overfit_experiments.ipynb` has the full overnight sweep
that fixed `backbone_transformer`'s overfitting (see Results below).

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
  last `unfreeze_num_stages` ResNet stages (`layer4` only, or `layer3` and
  `layer4`) are unfrozen and the optimizer is rebuilt with a second,
  lower-learning-rate parameter group for them (discriminative fine-tuning).
  The final reported run unfreezes `layer4` only (`unfreeze_num_stages = 1`),
  delayed to epoch 13, and keeps its BatchNorm running statistics frozen at
  their pretrained values for the rest of training
  (`freeze_bn_stats_after_unfreeze = True`), a setup reached only after a
  dedicated overnight sweep to fix overfitting in the original setup (see
  Results below and the notebook's ablation section).
- **Optimizer:** AdamW - weight decay is decoupled from the gradient moments,
  applied directly to the weights.
- **Best for:** highest accuracy ceiling, given enough VRAM/time (the final
  reported run uses 224x224 frames, 8 per clip, batch size 15, and a reduced
  Transformer: 1 layer, `d_model = 64`).

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
  (the final reported run used 112×112 frames, 24 frames per clip, batch
  size 24, ~1.9M parameters).

Both are selected and configured entirely through `config/config.py` -
`get_default_config("backbone_transformer")` or
`get_default_config("lightweight_tsm")` returns a ready-to-train `Config`.

## 3. Training strategy, loss, and metrics

- **Loss:** `BCE` by default (RWF-2000 is balanced). 
- **Metrics** (`training/metrics.py`): accuracy, precision, recall, F1, F2 and
  ROC-AUC - the standard set for a (near-)balanced binary classification
  task. 
- **Schedule:** linear warmup + cosine decay, stepped every training batch.
- **Early stopping:** on the monitored validation metric, patience
  configurable; both final reported runs monitor ROC-AUC rather than F1/F2,
  since F1/F2 selection on the small validation split proved unreliable (see
  the notebook's ablation section).

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
git clone https://github.com/26leonardo/Violence-detection.git violence_detection
cd violence_detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Google Colab:**
```python
!git clone https://github.com/26leonardo/Violence-detection.git /content/violence_detection
%cd /content/violence_detection
!pip install -q -r requirements.txt
```
Colab's free-tier GPUs (T4/L4) comfortably fit `lightweight_tsm`; for
`backbone_transformer` you may need to lower `--batch-size` on a T4.

## 6. Results (final models, test set)

At the default 0.5 threshold, seed 42:

| model | accuracy | precision | recall | F1 | F2 | ROC-AUC |
|---|---|---|---|---|---|---|
| `lightweight_tsm` (run1c) | 0.8225 | 0.8342 | 0.8050 | 0.8193 | 0.8107 | 0.894 |
| `backbone_transformer` (C18) | 0.8250 | 0.8155 | 0.8400 | 0.8276 | 0.8350 | 0.903 |

Recall is the metric this project weighs most (a missed fight costs more
than a false alarm): `backbone_transformer` leads on it here, with
`lightweight_tsm` close behind; the two are within a point of each other on
F1 and ROC-AUC.

Threshold tuned on validation for 0.95 target recall, applied to test:

| model | threshold | precision | recall | F1 |
|---|---|---|---|---|
| `lightweight_tsm` | 0.519 | 0.8360 | 0.7900 | 0.8123 |
| `backbone_transformer` | 0.281 | 0.7699 | 0.9200 | 0.8383 |

**Caveat:** `lightweight_tsm`'s numbers above are confirmed stable across two
random seeds (within about a point on every metric). `backbone_transformer`'s
current config (`C18`) is the result of a dedicated overnight sweep that fixed
severe overfitting in its original setup, but has only been run with one
seed so far; its original setup looked just as solid on one seed and then
lost 6 points of recall on a second. Read its recall lead above as
provisional until it is re-seeded, full discussion in the notebook's
Conclusions.