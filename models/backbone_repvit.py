"""RepViT-M2.3 backbone with FPN channel adapter.

The teacher uses a ViT-32L (1024-dim) → FPN with 4 scales (4x, 2x, 1x, 0.5x) → 256-dim.
The student uses RepViT-M2.3 → 4 stage outputs [80, 160, 320, 640] → Conv adapters → 256-dim.

Key: teacher discards the 4th FPN level (0.5x) in its forward pass (line 2285: fpn_hidden_states[:-1]).
We mirror this: produce 4 FPN levels, pass [:-1] to mask decoder, [-1] to DETR encoder.
"""

import torch
import torch.nn as nn
import timm

from .configuration import EfficientSAM3Config
from .utils import SinePositionEmbedding


class FPNChannelAdapter(nn.Module):
    """Adapt each RepViT stage output from its native channel count to fpn_hidden_size (256)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class RepViTBackbone(nn.Module):
    """RepViT-M2.3 + FPN adapters + SinePositionEmbedding.

    Outputs 4 FPN levels, each at 256 channels, with corresponding position encodings.
    The caller (EfficientSAM3) slices [:-1] for mask decoder and [-1] for DETR encoder,
    exactly matching the teacher's data flow.
    """

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        self.config = config

        # RepViT with multi-scale feature extraction
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=config.backbone_pretrained,
            features_only=True,
        )

        # FPN channel adapters: one per stage
        self.fpn_adapters = nn.ModuleList([
            FPNChannelAdapter(in_ch, config.fpn_hidden_size)
            for in_ch in config.backbone_out_channels
        ])

        # Shared position embedding generator
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=config.fpn_hidden_size // 2,
            normalize=False,
        )

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: [batch, 3, H, W]

        Returns:
            fpn_features: list of [batch, 256, H_i, W_i] (4 levels)
            fpn_pos_encodings: list of [batch, 256, H_i, W_i] (4 levels)
        """
        # Extract multi-scale features from RepViT
        stage_outputs = self.backbone(pixel_values)

        fpn_features = []
        fpn_pos_encodings = []

        for stage_feat, adapter in zip(stage_outputs, self.fpn_adapters):
            adapted = adapter(stage_feat)
            pos_enc = self.position_encoding(adapted.shape, adapted.device, adapted.dtype)
            fpn_features.append(adapted)
            fpn_pos_encodings.append(pos_enc)

        return fpn_features, fpn_pos_encodings
