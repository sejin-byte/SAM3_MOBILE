"""EfficientSAM3 — lightweight student model for mobile deployment.

Mirrors the teacher's Sam3Model.forward() data flow (lines 2280-2412):
  Image → Vision Encoder → FPN → [:-1] to mask decoder, [-1] to DETR encoder
  Text  → Text Encoder → projection → DETR encoder
  DETR Encoder → DETR Decoder → boxes, logits, presence
  DETR output + FPN features → Mask Decoder → masks, semantic_seg

Output dict keys match the teacher: pred_masks, pred_boxes, pred_logits,
presence_logits, semantic_seg — enabling direct distillation loss computation.
"""

import torch
import torch.nn as nn

from .configuration import EfficientSAM3Config
from .backbone_repvit import RepViTBackbone
from .text_encoder_mobileclip import MobileCLIPTextEncoder
from .perceiver_resampler import PerceiverResampler
from .memory_attention import MemoryCrossAttention
from .lightweight_detr import LightweightDetrEncoder, LightweightDetrDecoder, DotProductScoring
from .mask_decoder import MaskDecoder
from .utils import inverse_sigmoid, box_cxcywh_to_xyxy, DecoderMLP


class EfficientSAM3(nn.Module):
    """EfficientSAM3 student model (~75M params).

    Drop-in replacement for Sam3Model with matching output format.
    """

    def __init__(self, config: EfficientSAM3Config = None):
        super().__init__()
        if config is None:
            config = EfficientSAM3Config()
        self.config = config

        # Vision encoder: RepViT-M2.3 + FPN adapters
        self.vision_encoder = RepViTBackbone(config)

        # Text encoder: MobileCLIP-S1 + 512→256 projection
        self.text_encoder = MobileCLIPTextEncoder(config)

        # Perceiver Resampler (for video memory, not used in image-only forward)
        self.perceiver_resampler = PerceiverResampler(config)

        # DETR encoder (3 layers)
        self.detr_encoder = LightweightDetrEncoder(config)

        # DETR decoder (3 layers, 100 queries)
        self.detr_decoder = LightweightDetrDecoder(config)

        # Dot product scoring
        self.dot_product_scoring = DotProductScoring(config)

        # Mask decoder
        self.mask_decoder = MaskDecoder(config)

        # Memory cross-attention (for video: enriches encoder features with temporal memory)
        self.memory_cross_attn = MemoryCrossAttention(config.hidden_size)

        # IoU prediction head: per-query mask quality score
        self.iou_head = DecoderMLP(config.hidden_size, config.hidden_size, 1, num_layers=3)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on DETR encoder/decoder to save memory."""
        self.detr_encoder.gradient_checkpointing = True
        self.detr_decoder.gradient_checkpointing = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> dict:
        """
        Forward pass mirroring teacher's Sam3Model.forward().

        Args:
            pixel_values: [batch, 3, H, W] — preprocessed image
            input_ids: [batch, seq_len] — tokenized text prompt

        Returns:
            dict with keys: pred_masks, pred_boxes, pred_logits, presence_logits, semantic_seg
        """
        # ── 1. Vision Encoder ──
        # Produces 4 FPN levels, each [batch, 256, H_i, W_i]
        fpn_features, fpn_pos_encodings = self.vision_encoder(pixel_values)

        # Teacher discards 4th level (line 2285: [:-1])
        fpn_for_mask = fpn_features[:-1]       # levels 0,1,2 (4x, 2x, 1x)
        fpn_pos_for_mask = fpn_pos_encodings[:-1]

        # ── 2. Text Encoder ──
        # Full sequence [batch, seq_len, 256]
        text_features = self.text_encoder(input_ids)

        # ── 3. DETR Encoder ──
        # Only the finest backbone feature (level 2 = 1x) enters encoder
        # (teacher line 2356: vision_features=[fpn_hidden_states[-1]])
        encoder_out = self.detr_encoder(
            vision_features=[fpn_for_mask[-1]],
            text_features=text_features,
            vision_pos_embeds=[fpn_pos_for_mask[-1]],
        )

        # ── 4. DETR Decoder ──
        decoder_out = self.detr_decoder(
            vision_features=encoder_out["last_hidden_state"],
            text_features=encoder_out["text_features"],
            vision_pos_encoding=encoder_out["pos_embeds_flattened"],
            spatial_shapes=encoder_out["spatial_shapes"],
        )

        # ── 5. Box Refinement (teacher lines 2373-2376) ──
        all_box_offsets = self.detr_decoder.box_head(decoder_out["intermediate_hidden_states"])
        ref_boxes_inv = inverse_sigmoid(decoder_out["reference_boxes"])
        all_pred_boxes_cxcywh = (ref_boxes_inv + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        # ── 6. Dot Product Scoring (teacher lines 2378-2382) ──
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_out["intermediate_hidden_states"],
            text_features=encoder_out["text_features"],
        ).squeeze(-1)

        # ── 7. Take last layer outputs (teacher lines 2384-2387) ──
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hs = decoder_out["intermediate_hidden_states"][-1]
        presence_logits = decoder_out["presence_logits"][-1]

        # ── 8. Mask Decoder (teacher lines 2389-2396) ──
        mask_out = self.mask_decoder(
            decoder_queries=decoder_hs,
            backbone_features=list(fpn_for_mask),
            encoder_hidden_states=encoder_out["last_hidden_state"],
        )

        # ── 9. IoU Prediction ──
        iou_scores = self.iou_head(decoder_hs).squeeze(-1)  # [batch, num_queries]

        return {
            "pred_masks": mask_out["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_out["semantic_seg"],
            "iou_scores": iou_scores,
        }

    def forward_with_intermediates(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        prompt_embeddings: torch.Tensor = None,
    ) -> dict:
        """Forward pass that returns intermediate features for distillation.

        Args:
            pixel_values: [batch, 3, H, W]
            input_ids: [batch, seq_len]
            prompt_embeddings: [batch, N, 256] optional geometric prompt embeddings
                              to inject into DETR decoder queries

        Returns:
            dict with all outputs + fpn_features, encoder_output, decoder_hidden_states
        """
        fpn_features, fpn_pos_encodings = self.vision_encoder(pixel_values)
        fpn_for_mask = fpn_features[:-1]
        fpn_pos_for_mask = fpn_pos_encodings[:-1]

        text_features = self.text_encoder(input_ids)

        encoder_out = self.detr_encoder(
            vision_features=[fpn_for_mask[-1]],
            text_features=text_features,
            vision_pos_embeds=[fpn_pos_for_mask[-1]],
        )

        decoder_out = self.detr_decoder(
            vision_features=encoder_out["last_hidden_state"],
            text_features=encoder_out["text_features"],
            vision_pos_encoding=encoder_out["pos_embeds_flattened"],
            spatial_shapes=encoder_out["spatial_shapes"],
        )

        all_box_offsets = self.detr_decoder.box_head(decoder_out["intermediate_hidden_states"])
        ref_boxes_inv = inverse_sigmoid(decoder_out["reference_boxes"])
        all_pred_boxes_cxcywh = (ref_boxes_inv + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_out["intermediate_hidden_states"],
            text_features=encoder_out["text_features"],
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hs = decoder_out["intermediate_hidden_states"][-1]
        presence_logits = decoder_out["presence_logits"][-1]

        # Inject geometric prompt embeddings into decoder queries before mask decode
        if prompt_embeddings is not None:
            num_prompts = prompt_embeddings.shape[1]
            decoder_hs_prompted = decoder_hs.clone()
            decoder_hs_prompted[:, :num_prompts] = decoder_hs_prompted[:, :num_prompts] + prompt_embeddings
        else:
            decoder_hs_prompted = decoder_hs

        mask_out = self.mask_decoder(
            decoder_queries=decoder_hs_prompted,
            backbone_features=list(fpn_for_mask),
            encoder_hidden_states=encoder_out["last_hidden_state"],
        )

        iou_scores = self.iou_head(decoder_hs_prompted).squeeze(-1)

        return {
            "pred_masks": mask_out["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_out["semantic_seg"],
            "iou_scores": iou_scores,
            # Intermediates for distillation
            "fpn_features": fpn_for_mask,
            "encoder_output": encoder_out["last_hidden_state"],
            "decoder_hidden_states": decoder_hs,
        }

    def forward_video(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        memory_features: torch.Tensor,
    ) -> dict:
        """Video forward: compress context memory and fuse with query frame features.

        Args:
            pixel_values: [batch, 3, H, W] — query frame
            input_ids: [batch, seq_len] — tokenized text prompt
            memory_features: [batch, T*HW, hidden_size] — flattened context FPN features

        Returns:
            dict with pred_masks, pred_boxes, pred_logits, presence_logits,
            semantic_seg, iou_scores
        """
        # ── 1. Compress context memory via Perceiver Resampler ──
        memory_tokens = self.perceiver_resampler(memory_features)  # [batch, 64, 256]

        # ── 2. Standard vision + text encoding on query frame ──
        fpn_features, fpn_pos_encodings = self.vision_encoder(pixel_values)
        fpn_for_mask = fpn_features[:-1]
        fpn_pos_for_mask = fpn_pos_encodings[:-1]

        text_features = self.text_encoder(input_ids)

        # ── 3. DETR encoder (frozen) ──
        encoder_out = self.detr_encoder(
            vision_features=[fpn_for_mask[-1]],
            text_features=text_features,
            vision_pos_embeds=[fpn_pos_for_mask[-1]],
        )

        # ── 4. Memory cross-attention: enrich encoder output with temporal memory ──
        enriched = self.memory_cross_attn(
            encoder_features=encoder_out["last_hidden_state"],
            memory_tokens=memory_tokens,
        )

        # ── 5. DETR decoder (frozen) with enriched features ──
        decoder_out = self.detr_decoder(
            vision_features=enriched,
            text_features=encoder_out["text_features"],
            vision_pos_encoding=encoder_out["pos_embeds_flattened"],
            spatial_shapes=encoder_out["spatial_shapes"],
        )

        # ── 6. Box refinement ──
        all_box_offsets = self.detr_decoder.box_head(decoder_out["intermediate_hidden_states"])
        ref_boxes_inv = inverse_sigmoid(decoder_out["reference_boxes"])
        all_pred_boxes_cxcywh = (ref_boxes_inv + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        # ── 7. Scoring ──
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_out["intermediate_hidden_states"],
            text_features=encoder_out["text_features"],
        ).squeeze(-1)

        # ── 8. Last layer outputs ──
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hs = decoder_out["intermediate_hidden_states"][-1]
        presence_logits = decoder_out["presence_logits"][-1]

        # ── 9. Mask decoder ──
        mask_out = self.mask_decoder(
            decoder_queries=decoder_hs,
            backbone_features=list(fpn_for_mask),
            encoder_hidden_states=enriched,
        )

        # ── 10. IoU prediction ──
        iou_scores = self.iou_head(decoder_hs).squeeze(-1)

        return {
            "pred_masks": mask_out["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_out["semantic_seg"],
            "iou_scores": iou_scores,
        }
