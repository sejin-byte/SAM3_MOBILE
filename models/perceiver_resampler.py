"""Perceiver Resampler — compresses variable-length tokens into fixed K=64 latents.

Used for video memory: past frame tokens (variable N) → K=64 fixed latent tokens.
This prevents OOM on mobile by bounding memory regardless of video length.

Design: 2-layer cross-attention, pre-norm, using nn.MultiheadAttention for
ExecuTorch/static graph compatibility.
"""

import torch
import torch.nn as nn

from .configuration import EfficientSAM3Config


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver layer: cross-attention (latents query input) + FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        # Pre-norm cross-attention: latents attend to input tokens
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm FFN
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [batch, K, dim]
            input_tokens: [batch, N, dim]

        Returns:
            updated_latents: [batch, K, dim]
        """
        # Cross-attention
        residual = latents
        latents_normed = self.norm_latents(latents)
        input_normed = self.norm_input(input_tokens)
        attn_out, _ = self.cross_attn(
            query=latents_normed,
            key=input_normed,
            value=input_normed,
        )
        latents = residual + attn_out

        # FFN
        residual = latents
        latents = residual + self.ffn(self.norm_ffn(latents))

        return latents


class PerceiverResampler(nn.Module):
    """Compresses variable-length input to fixed K latent tokens.

    Args:
        config: EfficientSAM3Config with perceiver_* fields
    """

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        dim = config.hidden_size
        self.latents = nn.Parameter(torch.randn(config.perceiver_num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(
                dim=dim,
                num_heads=config.perceiver_num_heads,
                ffn_dim=dim * 4,
            )
            for _ in range(config.perceiver_num_layers)
        ])

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: [batch, N, dim] — variable-length input (e.g., past frame features)

        Returns:
            latent_tokens: [batch, K, dim] — fixed-size compressed representation
        """
        batch_size = input_tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            latents = layer(latents, input_tokens)

        return self.output_norm(latents)
