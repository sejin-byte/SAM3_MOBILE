"""Geometric Prompt Encoder — Point/Box coordinates → 256-dim embeddings.

Encodes spatial prompts (point centroids, bounding boxes) into embeddings
compatible with the DETR decoder query space, enabling the student model
to learn from geometric supervision beyond text-only prompts.
"""

import math

import torch
import torch.nn as nn


class GeometricPromptEncoder(nn.Module):
    """Encode point and box prompts into 256-dim query embeddings.

    Point: [x, y] → sine encoding → MLP → 256-dim
    Box: [x1, y1, x2, y2] → sine encoding → MLP → 256-dim
    """

    def __init__(self, hidden_size: int = 256, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

        # Point encoder: 2 coords × num_pos_feats = 256 → hidden_size
        self.point_proj = nn.Sequential(
            nn.Linear(num_pos_feats * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

        # Box encoder: 4 coords × num_pos_feats = 512 → hidden_size
        self.box_proj = nn.Sequential(
            nn.Linear(num_pos_feats * 4, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

        # Learnable type embeddings
        self.point_type_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.box_type_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def _sine_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with sine positional encoding.

        Args:
            coords: [..., D] normalized coordinates in [0, 1]

        Returns:
            [..., D * num_pos_feats] sine-encoded features
        """
        dim_t = torch.arange(self.num_pos_feats, device=coords.device, dtype=coords.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # coords: [..., D] → [..., D, 1] / [num_pos_feats]
        pos = coords.unsqueeze(-1) * math.pi * 2.0 / dim_t  # [..., D, num_pos_feats]
        # Interleave sin/cos
        pos = torch.stack([pos[..., 0::2].sin(), pos[..., 1::2].cos()], dim=-1)
        pos = pos.flatten(-2)  # [..., D, num_pos_feats]
        pos = pos.flatten(-2)  # [..., D * num_pos_feats]
        return pos

    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point prompts.

        Args:
            points: [batch, N, 2] normalized (x, y) coordinates

        Returns:
            [batch, N, hidden_size] point embeddings
        """
        sine_feats = self._sine_encode(points)  # [batch, N, 2*num_pos_feats]
        return self.point_proj(sine_feats) + self.point_type_embed

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode box prompts.

        Args:
            boxes: [batch, N, 4] normalized (x1, y1, x2, y2) coordinates

        Returns:
            [batch, N, hidden_size] box embeddings
        """
        sine_feats = self._sine_encode(boxes)  # [batch, N, 4*num_pos_feats]
        return self.box_proj(sine_feats) + self.box_type_embed

    def forward(
        self,
        prompt_types: list[str],
        prompt_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of prompts (mixed types).

        Args:
            prompt_types: list of "point", "box", or "text" per batch item
            prompt_coords: [batch, N, 4] — for points, only [:, :, :2] used

        Returns:
            [batch, N, hidden_size] prompt embeddings (zeros for text prompts)
        """
        batch_size, num_prompts, _ = prompt_coords.shape
        device = prompt_coords.device
        embeddings = torch.zeros(batch_size, num_prompts, self.hidden_size, device=device, dtype=prompt_coords.dtype)

        for i, ptype in enumerate(prompt_types):
            if ptype == "point":
                embeddings[i] = self.encode_points(prompt_coords[i:i+1, :, :2]).squeeze(0)
            elif ptype == "box":
                embeddings[i] = self.encode_boxes(prompt_coords[i:i+1]).squeeze(0)
            # "text" prompts get zero embeddings (no geometric injection)

        return embeddings
