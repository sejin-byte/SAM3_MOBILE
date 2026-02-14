"""Shared utilities ported from the teacher model (Sam3)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class SinePositionEmbedding(nn.Module):
    """Ported from Sam3SinePositionEmbedding (teacher lines 839-938)."""

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float = None):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode 4D box coordinates (x, y, w, h) → [batch, num_queries, num_pos_feats*4]."""
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).to(boxes.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        components = []
        for i in range(4):
            embed = boxes[:, :, i] * self.scale
            pos = embed[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            components.append(pos)

        # Teacher order: y, x, w, h
        return torch.cat((components[1], components[0], components[2], components[3]), dim=2)

    def forward(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate 2D sine positional embedding for feature maps.

        Args:
            shape: (batch, channels, height, width)
            device: target device
            dtype: target dtype

        Returns:
            Position encoding [batch, channels, height, width]
        """
        _, _, h, w = shape
        mask = torch.zeros((1, h, w), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DecoderMLP(nn.Module):
    """Ported from Sam3DecoderMLP (teacher lines 1461-1484)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers == 2:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2 or 3 layers supported, got {num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        if self.layer3 is not None:
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x
