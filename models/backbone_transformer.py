"""Architecture A - a fine-tuned pretrained backbone with a Transformer
temporal head.

Per-frame features come from an ImageNet-pretrained ResNet-50. They are
projected down, prefixed with a learnable [CLS] token (as in ViT), given a
learned positional encoding, and passed through a pre-norm Transformer
encoder. The [CLS] output is the pooled clip representation used for
classification.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision


class BackboneTransformerNet(nn.Module):
    def __init__(
        self,
        num_frames: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        # Sequential indices after dropping the final fc layer:
        # 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2,
        # 6 layer3, 7 layer4, 8 avgpool.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_out_dim = 2048

        self._frozen_bn_modules: List[nn.Module] = []
        self._backbone_unfrozen = False
        if freeze_backbone:
            self.freeze_backbone()

        self.proj = nn.Linear(self.backbone_out_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # pre-norm: trains more stably than post-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 1),
        )
        self._init_new_weights()

    def _init_new_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def freeze_backbone(self) -> None:
        """Freezes every backbone parameter and pins its BatchNorm layers to
        eval mode (so small fine-tuning batches never corrupt the running
        statistics learned during ImageNet pretraining)."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self._frozen_bn_modules = [m for m in self.backbone.modules() if isinstance(m, nn.BatchNorm2d)]

    def unfreeze_last_stages(self, num_stages: int = 2, freeze_bn_stats: bool = False) -> None:
       """Unfreezes `layer4` (num_stages=1) or `layer3` + `layer4` (num_stages=2)
       for a second fine-tuning phase with a lower learning rate.

       `freeze_bn_stats`: when True, the BatchNorm layers inside the newly
       unfrozen stages stay pinned in eval mode (like the still-frozen stages),
       so their running mean/var keep the ImageNet statistics instead of being
       updated from small fine-tuning batches. Their affine weight/bias still
       get gradients and are still trained, since only `requires_grad` controls
       that, not train/eval mode. When False (default), behaviour is unchanged
       from before this option existed.
       """

       if self._backbone_unfrozen:
           return
       stage_indices = {1: [7], 2: [6, 7]}.get(num_stages)
       if stage_indices is None:
           raise ValueError("num_stages must be 1 or 2.")

       for stage_index in stage_indices:
           stage_module = self.backbone[stage_index]
           for parameter in stage_module.parameters():
               parameter.requires_grad = True
           if not freeze_bn_stats:
               stage_bn_modules = [m for m in stage_module.modules() if isinstance(m, nn.BatchNorm2d)]
               self._frozen_bn_modules = [m for m in self._frozen_bn_modules if m not in stage_bn_modules]
       self._backbone_unfrozen = True

    def train(self, mode: bool = True) -> "BackboneTransformerNet":
        super().train(mode)
        if mode:
            for module in self._frozen_bn_modules:
                module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.num_frames:
            raise ValueError(f"Expected clips with {self.num_frames} frames, got {num_frames}.")

        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.backbone(x)  # (B*T, 2048, 1, 1)
        features = torch.flatten(features, 1)  # (B*T, 2048)
        features = self.proj(features)  # (B*T, d_model)
        features = features.view(batch_size, num_frames, -1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, features], dim=1)  # (B, T+1, d_model)
        tokens = tokens + self.pos_embed[:, : num_frames + 1, :]

        encoded = self.encoder(tokens)
        cls_output = encoded[:, 0]
        logits = self.classifier(cls_output).squeeze(-1)
        return logits
