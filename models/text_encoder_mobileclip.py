"""MobileCLIP-S1 text encoder with 512 to 256 projection.

The teacher uses CLIP-24L (1024-dim) and projects the full sequence via
nn.Linear(1024, 256) -- NOT just the pooled output (teacher line 2182).

MobileCLIP-S1 (open_clip) structure:
  - CustomTextCLIP.text = TextTransformer
    - text.token_embedding: Embedding(49408, 512)
    - text.positional_embedding: Parameter(77, 512)
    - text.transformer: Transformer with 12 resblocks
    - text.ln_final: LayerNorm(512)
    - text.attn_mask: buffer (77, 77) causal mask

We manually forward through the text tower to extract full sequence
hidden states (not just the pooled EOS token).
"""

import torch
import torch.nn as nn
import open_clip

from .configuration import EfficientSAM3Config


class MobileCLIPTextEncoder(nn.Module):
    """MobileCLIP-S1 text tower + linear projection to hidden_size (256)."""

    def __init__(self, config: EfficientSAM3Config):
        super().__init__()
        self.config = config

        # Load full CLIP model to extract text components
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.text_model_name,
            pretrained=config.text_pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(config.text_model_name)

        # Extract text tower components (under clip_model.text)
        text_tower = clip_model.text
        self.token_embedding = text_tower.token_embedding
        # positional_embedding is a Parameter, register it properly
        self.positional_embedding = text_tower.positional_embedding
        self.transformer = text_tower.transformer
        self.ln_final = text_tower.ln_final
        # Causal attention mask (buffer)
        self.register_buffer(
            "attn_mask",
            text_tower.attn_mask.clone() if hasattr(text_tower, "attn_mask") and text_tower.attn_mask is not None else None,
        )

        # Our projection: 512 -> 256
        self._text_dim = self.token_embedding.embedding_dim  # 512
        self.projection = nn.Linear(self._text_dim, config.text_projection_dim)

        # Free the vision tower
        del clip_model

    def forward(self, input_ids: torch.LongTensor):
        """
        Args:
            input_ids: [batch, seq_len] tokenized text

        Returns:
            text_features: [batch, seq_len, 256] projected full sequence
        """
        x = self.token_embedding(input_ids)  # [batch, seq_len, 512]
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]

        # Build causal mask
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:seq_len, :seq_len].to(x.device, x.dtype)

        # open_clip Transformer resblocks expect (seq_len, batch, dim)
        x = x.permute(1, 0, 2)
        for block in self.transformer.resblocks:
            x = block(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # back to [batch, seq_len, 512]

        x = self.ln_final(x)

        # Project full sequence to 256-dim
        text_features = self.projection(x)

        return text_features
