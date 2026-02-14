"""Lightweight DETR — Encoder (3L) + Decoder (3L, 100 queries) + DotProductScoring.

Faithfully mirrors the teacher's Sam3DetrEncoder, Sam3DetrDecoder, and Sam3DotProductScoring
but with reduced layers/queries/FFN dim for mobile deployment.

Teacher reference:
  - Encoder: modeling_sam3.py lines 1262-1458 (6 layers, FFN=2048)
  - Decoder: modeling_sam3.py lines 1487-1791 (6 layers, 200 queries, FFN=2048)
  - Scoring: modeling_sam3.py lines 1794-1885

Student changes:
  - Encoder: 3 layers, FFN=1024
  - Decoder: 3 layers, 100 queries, FFN=1024
  - Uses nn.MultiheadAttention for static graph compatibility
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .configuration import EfficientSAM3Config
from .utils import SinePositionEmbedding, DecoderMLP, inverse_sigmoid, box_cxcywh_to_xyxy


# ── Encoder ──────────────────────────────────────────────


class LightweightDetrEncoderLayer(nn.Module):
    """DETR encoder layer: self-attn(vision) + cross-attn(vision→text) + FFN.
    Mirrors Sam3DetrEncoderLayer (teacher lines 1262-1329)."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        # Self-attention on vision features
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention: vision → text
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_feats: torch.Tensor,
        text_feats: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_feats: [batch, HW, hidden_size]
            text_feats: [batch, text_len, hidden_size]
            vision_pos_encoding: [batch, HW, hidden_size]
        """
        # Self-attention with position encoding (pre-norm)
        residual = vision_feats
        h = self.layer_norm1(vision_feats)
        h_with_pos = h + vision_pos_encoding
        h, _ = self.self_attn(query=h_with_pos, key=h_with_pos, value=h)
        vision_feats = self.dropout1(h) + residual

        # Cross-attention: vision queries → text keys/values (pre-norm)
        residual = vision_feats
        h = self.layer_norm2(vision_feats)
        h, _ = self.cross_attn(query=h, key=text_feats, value=text_feats)
        vision_feats = self.dropout2(h) + residual

        # FFN (pre-norm)
        residual = vision_feats
        h = self.layer_norm3(vision_feats)
        vision_feats = self.ffn(h) + residual

        return vision_feats


class LightweightDetrEncoder(nn.Module):
    """DETR encoder: multi-level feature flatten + N encoder layers.
    Mirrors Sam3DetrEncoder (teacher lines 1332-1458)."""

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([
            LightweightDetrEncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.detr_encoder_num_heads,
                ffn_dim=config.detr_encoder_ffn_dim,
                dropout=config.detr_encoder_dropout,
            )
            for _ in range(config.detr_encoder_num_layers)
        ])

    def _prepare_multilevel_features(
        self,
        vision_features: list[torch.Tensor],
        vision_pos_embeds: list[torch.Tensor],
    ):
        """Flatten multi-level features for encoder processing.
        Mirrors teacher lines 1369-1394."""
        features_flat = []
        pos_flat = []
        spatial_shapes = []

        for feat, pos in zip(vision_features, vision_pos_embeds):
            h, w = feat.shape[-2:]
            spatial_shapes.append((h, w))
            features_flat.append(feat.flatten(2).transpose(1, 2))
            pos_flat.append(pos.flatten(2).transpose(1, 2))

        features_flat = torch.cat(features_flat, dim=1)
        pos_flat = torch.cat(pos_flat, dim=1)
        # Keep shape metadata as python/SymInt tuples (not tensor scalars) so
        # export paths do not introduce data-dependent local_scalar symbols.
        return features_flat, pos_flat, tuple(spatial_shapes)

    def forward(
        self,
        vision_features: list[torch.Tensor],
        text_features: torch.Tensor,
        vision_pos_embeds: list[torch.Tensor],
    ):
        """
        Args:
            vision_features: list of [batch, 256, H_i, W_i] (typically just 1 level)
            text_features: [batch, text_len, 256]
            vision_pos_embeds: list of [batch, 256, H_i, W_i]

        Returns:
            dict with last_hidden_state, pos_embeds_flattened, text_features, spatial_shapes
        """
        features_flat, pos_flat, spatial_shapes = self._prepare_multilevel_features(
            vision_features, vision_pos_embeds
        )

        hidden_states = features_flat
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = grad_checkpoint(
                    layer, hidden_states, text_features, pos_flat,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, text_features, pos_flat)

        return {
            "last_hidden_state": hidden_states,
            "pos_embeds_flattened": pos_flat,
            "text_features": text_features,
            "spatial_shapes": spatial_shapes,
        }


# ── Decoder ──────────────────────────────────────────────


class LightweightDetrDecoderLayer(nn.Module):
    """DETR decoder layer: self-attn + text-cross-attn + vision-cross-attn + FFN.
    Mirrors Sam3DetrDecoderLayer (teacher lines 1487-1585)."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # Text cross-attention
        self.text_cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.text_cross_attn_dropout = nn.Dropout(dropout)
        self.text_cross_attn_norm = nn.LayerNorm(hidden_size)

        # Vision cross-attention
        self.vision_cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.vision_cross_attn_dropout = nn.Dropout(dropout)
        self.vision_cross_attn_norm = nn.LayerNorm(hidden_size)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        vision_cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, num_queries+1, hidden_size] (presence token at pos 0)
            query_pos: [batch, num_queries, hidden_size]
            text_features: [batch, text_len, hidden_size]
            vision_features: [batch, HW, hidden_size]
            vision_pos_encoding: [batch, HW, hidden_size]
            vision_cross_attn_mask: [batch*num_heads, num_queries+1, HW] or None
        """
        # Prepend zero for presence token's position (teacher line 1536)
        query_pos = F.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)

        # Self-attention (post-norm, matching teacher)
        residual = hidden_states
        q = k = hidden_states + query_pos
        attn_out, _ = self.self_attn(query=q, key=k, value=hidden_states)
        hidden_states = residual + self.self_attn_dropout(attn_out)
        hidden_states = self.self_attn_norm(hidden_states)

        # Text cross-attention (post-norm)
        residual = hidden_states
        q = hidden_states + query_pos
        attn_out, _ = self.text_cross_attn(query=q, key=text_features, value=text_features)
        hidden_states = residual + self.text_cross_attn_dropout(attn_out)
        hidden_states = self.text_cross_attn_norm(hidden_states)

        # Vision cross-attention with RPB (post-norm)
        residual = hidden_states
        q = hidden_states + query_pos
        k = vision_features + vision_pos_encoding
        attn_out, _ = self.vision_cross_attn(
            query=q, key=k, value=vision_features,
            attn_mask=vision_cross_attn_mask,
        )
        hidden_states = residual + self.vision_cross_attn_dropout(attn_out)
        hidden_states = self.vision_cross_attn_norm(hidden_states)

        # FFN (post-norm)
        residual = hidden_states
        hidden_states = residual + self.ffn_dropout(self.ffn(hidden_states))
        hidden_states = self.ffn_norm(hidden_states)

        return hidden_states


class LightweightDetrDecoder(nn.Module):
    """DETR decoder with box refinement and presence token.
    Mirrors Sam3DetrDecoder (teacher lines 1588-1791)."""

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        self.gradient_checkpointing = False
        d = config.hidden_size
        num_queries = config.detr_decoder_num_queries
        num_heads = config.detr_decoder_num_heads

        self.layers = nn.ModuleList([
            LightweightDetrDecoderLayer(
                hidden_size=d,
                num_heads=num_heads,
                ffn_dim=config.detr_decoder_ffn_dim,
                dropout=config.detr_decoder_dropout,
            )
            for _ in range(config.detr_decoder_num_layers)
        ])

        self.output_layer_norm = nn.LayerNorm(d)

        # Box head: predicts 4D box delta (3-layer MLP)
        self.box_head = DecoderMLP(d, d, 4, num_layers=3)

        # Learnable queries and reference points
        self.query_embed = nn.Embedding(num_queries, d)
        self.reference_points = nn.Embedding(num_queries, 4)

        # Presence token mechanism (teacher lines 1719-1780)
        self.presence_token = nn.Embedding(1, d)
        self.presence_head = DecoderMLP(d, d, 1, num_layers=3)
        self.presence_layer_norm = nn.LayerNorm(d)
        self.clamp_presence_logit_max_val = 10.0

        # Reference point head for conditional queries
        self.ref_point_head = DecoderMLP(2 * d, d, d, num_layers=2)

        # Box RPB (Relative Position Bias) embeddings
        self.box_rpb_embed_x = DecoderMLP(2, d, num_heads, num_layers=2)
        self.box_rpb_embed_y = DecoderMLP(2, d, num_heads, num_layers=2)

        self.position_encoding = SinePositionEmbedding(num_pos_feats=d // 2, normalize=False)

    def _get_rpb_matrix(
        self,
        reference_boxes: torch.Tensor,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Compute Box RPB matrix. Mirrors teacher lines 1644-1689.
        Computed in float32 to avoid float16 precision issues with log2/sign."""
        height, width = spatial_shape
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        # Generate coordinate grids
        coords_h = torch.arange(0, height, device=reference_boxes.device, dtype=torch.float32) / height
        coords_w = torch.arange(0, width, device=reference_boxes.device, dtype=torch.float32) / width

        # Work in float32 for numerical stability
        boxes_f32 = boxes_xyxy.float()

        # Compute deltas
        deltas_y = coords_h.view(1, -1, 1) - boxes_f32.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_f32.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)

        # Log-scale encoding
        deltas_x_log = deltas_x * 8
        deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / math.log2(8)
        deltas_y_log = deltas_y * 8
        deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / math.log2(8)

        # Cast back to model dtype for MLP
        deltas_x_log = deltas_x_log.to(reference_boxes.dtype)
        deltas_y_log = deltas_y_log.to(reference_boxes.dtype)

        # Embed deltas
        deltas_x = self.box_rpb_embed_x(deltas_x_log)
        deltas_y = self.box_rpb_embed_y(deltas_y_log)

        # Combine into 2D bias matrix
        rpb_matrix = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        rpb_matrix = rpb_matrix.flatten(2, 3)  # [batch, num_queries, H*W, num_heads]
        rpb_matrix = rpb_matrix.permute(0, 3, 1, 2).contiguous()  # [batch, num_heads, num_queries, H*W]
        return rpb_matrix

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        spatial_shapes,
    ):
        """
        Args:
            vision_features: [batch, HW, hidden_size]
            text_features: [batch, text_len, hidden_size]
            vision_pos_encoding: [batch, HW, hidden_size]
            spatial_shapes: sequence of (H, W) shape pairs

        Returns:
            dict with intermediate_hidden_states, reference_boxes, presence_logits
        """
        batch_size = vision_features.shape[0]
        num_heads = self.layers[0].self_attn.num_heads

        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate presence token with query embeddings (teacher line 1722)
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence = []

        for layer in self.layers:
            # Sine embedding for conditional queries (teacher line 1740)
            ref_input = reference_boxes.unsqueeze(2)
            query_sine = self.position_encoding.encode_boxes(ref_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine)

            # Compute RPB for vision cross-attention
            vision_cross_attn_mask = None
            if spatial_shapes is not None and len(spatial_shapes) == 1:
                # Keep shape metadata as SymInt tuples (no .item()) to avoid
                # data-dependent local_scalar symbols during export.
                h, w = spatial_shapes[0]
                rpb = self._get_rpb_matrix(reference_boxes, (h, w))
                # Prepend zeros for presence token (teacher line 1749)
                rpb = F.pad(rpb, (0, 0, 1, 0), mode="constant", value=0)
                # Reshape for nn.MultiheadAttention: [batch*num_heads, tgt_len, src_len]
                vision_cross_attn_mask = rpb.reshape(batch_size * num_heads, -1, rpb.shape[-1])

            if self.gradient_checkpointing and self.training:
                hidden_states = grad_checkpoint(
                    layer,
                    hidden_states, query_pos, text_features,
                    vision_features, vision_pos_encoding, vision_cross_attn_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    query_pos=query_pos,
                    text_features=text_features,
                    vision_features=vision_features,
                    vision_pos_encoding=vision_pos_encoding,
                    vision_cross_attn_mask=vision_cross_attn_mask,
                )

            # Box refinement (teacher lines 1762-1769)
            query_hs = hidden_states[:, 1:]
            ref_before_sigmoid = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(self.output_layer_norm(query_hs))
            new_ref_boxes = (delta_boxes + ref_before_sigmoid).sigmoid()
            reference_boxes = new_ref_boxes.detach()

            intermediate_outputs.append(self.output_layer_norm(query_hs))
            intermediate_boxes.append(new_ref_boxes)

            # Presence token (teacher lines 1774-1780)
            presence_hs = hidden_states[:, :1]
            presence_logits = self.presence_head(self.presence_layer_norm(presence_hs)).squeeze(-1)
            presence_logits = presence_logits.clamp(
                min=-self.clamp_presence_logit_max_val,
                max=self.clamp_presence_logit_max_val,
            )
            intermediate_presence.append(presence_logits)

        intermediate_outputs = torch.stack(intermediate_outputs)
        intermediate_boxes = torch.stack(intermediate_boxes[:-1])
        intermediate_presence = torch.stack(intermediate_presence)

        return {
            "intermediate_hidden_states": intermediate_outputs,
            "reference_boxes": intermediate_boxes,
            "presence_logits": intermediate_presence,
        }


# ── Scoring ──────────────────────────────────────────────


class DotProductScoring(nn.Module):
    """Dot-product scoring between projected queries and pooled text.
    Mirrors Sam3DotProductScoring (teacher lines 1794-1885)."""

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        d = config.hidden_size

        self.text_mlp = DecoderMLP(d, config.detr_decoder_ffn_dim, d, num_layers=2)
        self.text_mlp_dropout = nn.Dropout(config.detr_decoder_dropout)
        self.text_mlp_out_norm = nn.LayerNorm(d)

        self.text_proj = nn.Linear(d, d)
        self.query_proj = nn.Linear(d, d)

        self.scale = float(1.0 / np.sqrt(d))
        self.clamp_max_val = 12.0

    def _pool_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        """Mean pool text features. Teacher lines 1826-1849 (simplified, no mask)."""
        return text_features.mean(dim=1)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden_states: [num_layers, batch, num_queries, hidden_size]
            text_features: [batch, text_len, hidden_size]

        Returns:
            scores: [num_layers, batch, num_queries, 1]
        """
        # Text MLP with residual (teacher lines 1868-1872)
        orig = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features = text_features + orig
        text_features = self.text_mlp_out_norm(text_features)

        pooled = self._pool_text_features(text_features)  # [batch, d]

        proj_text = self.text_proj(pooled)  # [batch, d]
        proj_queries = self.query_proj(decoder_hidden_states)  # [num_layers, batch, Q, d]

        proj_text = proj_text.unsqueeze(-1)  # [batch, d, 1]
        scores = torch.matmul(proj_queries, proj_text.unsqueeze(0))  # [num_layers, batch, Q, 1]
        scores = scores * self.scale
        scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores
