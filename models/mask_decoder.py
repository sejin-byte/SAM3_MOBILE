"""Mask Decoder — PixelDecoder FPN + MaskEmbedder + semantic seg head.

Mirrors Sam3MaskDecoder (teacher lines 1974-2101) with identical structure.
The mask decoder is already lightweight in the teacher (~1.5M params),
so we keep it as-is at hidden_size=256.

Key data flow (teacher lines 2051-2064):
  1. backbone_features → PixelDecoder (FPN upsampling) → pixel_embed
  2. decoder_queries → MaskEmbedder (MLP) → mask_embeddings
  3. pred_masks = einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)
  4. semantic_seg = 1x1 Conv(pixel_embed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import EfficientSAM3Config


class PixelDecoder(nn.Module):
    """FPN decoder: coarse-to-fine upsampling with skip connections.
    Mirrors Sam3PixelDecoder (teacher lines 1924-1971)."""

    def __init__(self, hidden_size: int, num_upsampling_stages: int):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
            for _ in range(num_upsampling_stages)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, hidden_size)
            for _ in range(num_upsampling_stages)
        ])

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            backbone_features: list of [batch, hidden_size, H_i, W_i] low→high resolution

        Returns:
            pixel_embed: [batch, hidden_size, H_finest, W_finest]
        """
        # Start from coarsest feature (teacher line 1957)
        prev_fpn = backbone_features[-1]
        for idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            prev_fpn = F.interpolate(prev_fpn, size=backbone_feat.shape[-2:], mode="nearest")
            prev_fpn = prev_fpn + backbone_feat
            prev_fpn = self.conv_layers[idx](prev_fpn)
            prev_fpn = self.norms[idx](prev_fpn)
            prev_fpn = F.relu(prev_fpn)
        return prev_fpn


class MaskEmbedder(nn.Module):
    """3-layer MLP that embeds object queries for mask prediction.
    Mirrors Sam3MaskEmbedder (teacher lines 1888-1921)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        ])
        self.activation = nn.ReLU()

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """[batch, num_queries, hidden_size] → [batch, num_queries, hidden_size]"""
        h = queries
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


class MaskDecoder(nn.Module):
    """Combines object queries with pixel features to predict masks.
    Mirrors Sam3MaskDecoder (teacher lines 1974-2101).

    Simplified: we omit the prompt cross-attention that the teacher uses
    for geometry prompts (input_boxes), since the student model focuses
    on text-prompted segmentation for distillation.
    """

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        d = config.hidden_size

        self.pixel_decoder = PixelDecoder(d, config.mask_decoder_num_upsampling_stages)
        self.mask_embedder = MaskEmbedder(d)

        # Project pixel decoder output → mask embedding space
        self.instance_projection = nn.Conv2d(d, d, kernel_size=1)

        # Semantic segmentation head (teacher line 1999)
        self.semantic_projection = nn.Conv2d(d, 1, kernel_size=1)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ):
        """
        Args:
            decoder_queries: [batch, num_queries, hidden_size]
            backbone_features: list of [batch, 256, H_i, W_i] (3 levels: 4x, 2x, 1x)
            encoder_hidden_states: [batch, HW, hidden_size] — DETR encoder output

        Returns:
            dict with pred_masks [batch, Q, H, W] and semantic_seg [batch, 1, H, W]
        """
        # Replace finest backbone feature with encoder output (teacher lines 2086-2096)
        backbone_feats = [f.clone() for f in backbone_features]
        finest = backbone_feats[-1]
        spatial_dim = finest.shape[-2] * finest.shape[-1]
        enc_visual = encoder_hidden_states[:, :spatial_dim, :]
        batch_size, _, d = enc_visual.shape
        h, w = finest.shape[-2:]
        enc_visual = enc_visual.transpose(1, 2).reshape(batch_size, d, h, w)
        backbone_feats[-1] = enc_visual

        # FPN upsampling
        pixel_embed = self.pixel_decoder(backbone_feats)

        # Instance masks via dot product
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)

        # Semantic segmentation
        semantic_seg = self.semantic_projection(pixel_embed)

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
        }
