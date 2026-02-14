"""EfficientSAM3 configuration — all hyperparameters in one dataclass."""

from dataclasses import dataclass, field


@dataclass
class EfficientSAM3Config:
    # ── Image ──
    image_size: int = 504
    num_channels: int = 3

    # ── RepViT backbone (timm) ──
    backbone_name: str = "repvit_m2_3"
    backbone_pretrained: bool = True
    backbone_out_channels: list = field(default_factory=lambda: [80, 160, 320, 640])

    # ── FPN adapter ──
    fpn_hidden_size: int = 256

    # ── Text encoder (MobileCLIP-S1 via open_clip) ──
    text_model_name: str = "MobileCLIP-S1"
    text_pretrained: str = "datacompdr"
    text_hidden_size: int = 512
    text_projection_dim: int = 256  # project text_hidden_size → fpn_hidden_size

    # ── Perceiver Resampler ──
    perceiver_num_latents: int = 64
    perceiver_num_layers: int = 2
    perceiver_num_heads: int = 8

    # ── DETR Encoder (lightweight) ──
    detr_encoder_num_layers: int = 3
    detr_encoder_num_heads: int = 8
    detr_encoder_ffn_dim: int = 1024
    detr_encoder_dropout: float = 0.1

    # ── DETR Decoder (lightweight) ──
    detr_decoder_num_layers: int = 3
    detr_decoder_num_queries: int = 100
    detr_decoder_num_heads: int = 8
    detr_decoder_ffn_dim: int = 1024
    detr_decoder_dropout: float = 0.1

    # ── Mask Decoder ──
    mask_decoder_num_upsampling_stages: int = 3

    # ── Shared ──
    hidden_size: int = 256  # all post-FPN modules operate at this dimension
