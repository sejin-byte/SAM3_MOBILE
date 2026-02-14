"""Distillation Loss — 9 loss terms for EfficientSAM3 knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistillationConfig


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice loss for mask predictions.

    Args:
        pred: [N, H, W] logits
        target: [N, H, W] probabilities or logits (will be sigmoided)
    """
    pred = pred.sigmoid().flatten(1)
    target = target.sigmoid().flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)


def compute_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between paired boxes.

    Args:
        boxes_a, boxes_b: [N, 4] in (x1, y1, x2, y2) format

    Returns:
        [N] GIoU values in [-1, 1]
    """
    x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter

    iou = inter / (union + 1e-6)

    # Enclosing box
    ex1 = torch.min(boxes_a[:, 0], boxes_b[:, 0])
    ey1 = torch.min(boxes_a[:, 1], boxes_b[:, 1])
    ex2 = torch.max(boxes_a[:, 2], boxes_b[:, 2])
    ey2 = torch.max(boxes_a[:, 3], boxes_b[:, 3])
    enclose_area = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
    return giou


class DistillationLoss(nn.Module):
    """9-term distillation loss for EfficientSAM3.

    Output losses (always active):
        1. mask_loss: Dice + BCE on matched pred_masks
        2. box_l1_loss: L1 on matched pred_boxes
        3. box_giou_loss: GIoU on matched pred_boxes
        4. logit_loss: BCE on matched pred_logits
        5. iou_token_loss: MSE(student IoU prediction, actual mask IoU)
        6. presence_loss: MSE on presence_logits
        7. semantic_seg_loss: Dice + BCE on semantic_seg

    Feature losses (Phase 1 only):
        8. fpn_feature_loss: MSE on FPN features (3 levels)
        9. encoder_feature_loss: MSE on DETR encoder output
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

    def _mask_loss(self, student_masks: torch.Tensor, teacher_masks: torch.Tensor) -> torch.Tensor:
        """Dice + BCE on matched masks."""
        # Resize student to teacher spatial size if needed
        if student_masks.shape[-2:] != teacher_masks.shape[-2:]:
            student_masks = F.interpolate(
                student_masks.unsqueeze(0),
                size=teacher_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        d_loss = dice_loss(student_masks, teacher_masks).mean()
        bce_loss = F.binary_cross_entropy_with_logits(
            student_masks, teacher_masks.sigmoid(), reduction="mean"
        )
        return d_loss + bce_loss

    def _iou_token_loss(
        self,
        student_iou_scores: torch.Tensor,
        student_masks: torch.Tensor,
        teacher_masks: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted IoU scores and actual mask IoU with teacher.

        Args:
            student_iou_scores: [N] predicted quality scores
            student_masks: [N, H, W] student mask logits
            teacher_masks: [N, H, W] teacher mask logits
        """
        if student_masks.shape[-2:] != teacher_masks.shape[-2:]:
            student_masks = F.interpolate(
                student_masks.unsqueeze(0),
                size=teacher_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Compute actual Dice IoU per mask
        s = student_masks.sigmoid().flatten(1)
        t = teacher_masks.sigmoid().flatten(1)
        intersection = (s * t).sum(dim=1)
        union = s.sum(dim=1) + t.sum(dim=1)
        actual_iou = (2.0 * intersection + 1e-6) / (union + 1e-6)

        return F.mse_loss(student_iou_scores.sigmoid(), actual_iou.detach())

    def forward(
        self,
        student_out: dict,
        teacher_out: dict,
        student_indices: list,
        teacher_indices: list,
        phase: int = 1,
    ) -> dict:
        """Compute all loss terms.

        Args:
            student_out: dict with pred_masks, pred_boxes, pred_logits, presence_logits,
                        semantic_seg, iou_scores, fpn_features, encoder_output
            teacher_out: dict with same keys (except iou_scores, fpn_features, encoder_output
                        are under different names from teacher forward)
            student_indices: list of [M] tensors, one per batch sample
            teacher_indices: list of [M] tensors, one per batch sample
            phase: 1 or 2

        Returns:
            dict with individual losses and total_loss
        """
        losses = {}
        batch_size = student_out["pred_masks"].shape[0]
        device = student_out["pred_masks"].device

        # ── Per-sample matched losses (accumulated, averaged over batch) ──
        mask_loss_sum = torch.tensor(0.0, device=device)
        box_l1_sum = torch.tensor(0.0, device=device)
        box_giou_sum = torch.tensor(0.0, device=device)
        logit_sum = torch.tensor(0.0, device=device)
        iou_token_sum = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            s_idx = student_indices[b]
            t_idx = teacher_indices[b]

            s_masks = student_out["pred_masks"][b, s_idx].float()
            t_masks = teacher_out["pred_masks"][b, t_idx].float()
            s_boxes = student_out["pred_boxes"][b, s_idx].float()
            t_boxes = teacher_out["pred_boxes"][b, t_idx].float()
            s_logits = student_out["pred_logits"][b, s_idx].float()
            t_logits = teacher_out["pred_logits"][b, t_idx].float()
            s_iou = student_out["iou_scores"][b, s_idx].float()

            mask_loss_sum = mask_loss_sum + self._mask_loss(s_masks, t_masks)
            box_l1_sum = box_l1_sum + F.l1_loss(s_boxes, t_boxes)
            giou = compute_giou(s_boxes, t_boxes)
            box_giou_sum = box_giou_sum + (1.0 - giou).mean()
            logit_sum = logit_sum + F.binary_cross_entropy_with_logits(
                s_logits, t_logits.sigmoid(), reduction="mean"
            )
            iou_token_sum = iou_token_sum + self._iou_token_loss(s_iou, s_masks, t_masks)

        losses["mask_loss"] = (mask_loss_sum / batch_size) * self.config.mask_loss_weight
        losses["box_l1_loss"] = (box_l1_sum / batch_size) * self.config.box_l1_loss_weight
        losses["box_giou_loss"] = (box_giou_sum / batch_size) * self.config.box_giou_loss_weight
        losses["logit_loss"] = (logit_sum / batch_size) * self.config.logit_loss_weight
        losses["iou_token_loss"] = (iou_token_sum / batch_size) * self.config.iou_token_loss_weight

        # ── Batch-level losses (already handle batch dim naturally) ──

        # 6. Presence loss
        s_pres = student_out["presence_logits"].float()
        t_pres = teacher_out["presence_logits"].float()
        losses["presence_loss"] = F.mse_loss(s_pres, t_pres) * self.config.presence_loss_weight

        # 7. Semantic seg loss (squeeze channel dim, not batch)
        s_seg = student_out["semantic_seg"].float()
        t_seg = teacher_out["semantic_seg"].float()
        if s_seg.shape[-2:] != t_seg.shape[-2:]:
            s_seg = F.interpolate(s_seg, size=t_seg.shape[-2:], mode="bilinear", align_corners=False)
        d_seg = dice_loss(s_seg.squeeze(1), t_seg.squeeze(1)).mean()
        bce_seg = F.binary_cross_entropy_with_logits(s_seg, t_seg.sigmoid(), reduction="mean")
        losses["semantic_seg_loss"] = (d_seg + bce_seg) * self.config.semantic_seg_loss_weight

        # ── Feature alignment losses (Phase 1 only) ──
        if phase == 1 and "fpn_features" in student_out and "fpn_features" in teacher_out:
            fpn_loss = torch.tensor(0.0, device=device)
            for s_fpn, t_fpn in zip(student_out["fpn_features"], teacher_out["fpn_features"]):
                s_f = s_fpn.float()
                t_f = t_fpn.float()
                if s_f.shape[-2:] != t_f.shape[-2:]:
                    s_f = F.interpolate(s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False)
                fpn_loss = fpn_loss + F.mse_loss(s_f, t_f)
            losses["fpn_feature_loss"] = fpn_loss / 3.0 * self.config.fpn_feature_loss_weight

            if (student_out.get("encoder_output") is not None
                    and teacher_out.get("encoder_output") is not None):
                s_enc = student_out["encoder_output"].float()
                t_enc = teacher_out["encoder_output"].float()
                if s_enc.shape[1] != t_enc.shape[1]:
                    s_enc = F.interpolate(
                        s_enc.transpose(1, 2).unsqueeze(-1),
                        size=(t_enc.shape[1], 1),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(-1).transpose(1, 2)
                losses["encoder_feature_loss"] = F.mse_loss(s_enc, t_enc) * self.config.encoder_feature_loss_weight

        # ── Total ──
        losses["total_loss"] = sum(losses.values())

        return losses
