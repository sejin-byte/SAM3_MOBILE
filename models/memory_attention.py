"""MemoryCrossAttention — enriches encoder features with temporal memory.

Sits between the frozen DETR encoder and decoder in the video pipeline:
  encoder_out [batch, HW, 256] (queries) × memory_tokens [batch, 64, 256] (keys/values)
  → enriched features [batch, HW, 256] for the decoder

Pre-norm cross-attention with residual connection, ~198K params.
batch_first=True for ExecuTorch compatibility.
"""

import torch
import torch.nn as nn


class MemoryCrossAttention(nn.Module):
    """Cross-attention: vision features attend to compressed memory tokens.

    Args:
        hidden_size: feature dimension (default 256)
        num_heads: attention heads (default 8)
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm_query = nn.LayerNorm(hidden_size)
        self.norm_memory = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        encoder_features: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [batch, seq_len, hidden_size] from DETR encoder
            memory_tokens: [batch, K, hidden_size] from Perceiver Resampler

        Returns:
            enriched_features: [batch, seq_len, hidden_size]
        """
        residual = encoder_features
        q = self.norm_query(encoder_features)
        kv = self.norm_memory(memory_tokens)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        return residual + self.gate.tanh() * attn_out
