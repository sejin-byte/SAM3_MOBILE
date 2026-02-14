"""Video Distillation Loss — GT-based loss for temporal memory training.

Unlike image distillation which uses teacher predictions as targets,
video distillation supervises against SA-V ground truth masks directly.

Reuses dice_loss and compute_giou from distillation/losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import dice_loss, compute_giou
from .video_config import VideoDistillationConfig


class VideoDistillationLoss(nn.Module):
    """Loss for video distillation against GT masks.

    Terms:
    1. mask_loss: Dice + BCE on matched pred_masks vs GT masks
    2. box_l1_loss: L1 on matched pred_boxes vs GT boxes
    3. box_giou_loss: GIoU on matched pred_boxes vs GT boxes
    4. iou_token_loss: MSE(predicted IoU, actual mask IoU with GT)
    5. presence_loss: BCE — matched predictions should be "present"
    """

    def __init__(self, config: VideoDistillationConfig):
        super().__init__()
        self.config = config

    def _mask_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """Dice + BCE between predicted and GT masks.

        Args:
            pred_masks: [N, H_pred, W_pred] logits
            gt_masks: [N, H_gt, W_gt] binary {0, 1}
        """
        # Resize pred to GT spatial size
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0),
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        gt_float = gt_masks.float()

        # Dice loss (expects sigmoid targets, but GT is binary — use directly)
        pred_flat = pred_masks.sigmoid().flatten(1)
        gt_flat = gt_float.flatten(1)
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
        d_loss = (1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)).mean()

        # BCE loss (GT is binary, so use it directly as target)
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_float, reduction="mean")

        return d_loss + bce

    def _iou_token_loss(
        self,
        iou_scores: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted IoU and actual mask IoU with GT.

        Args:
            iou_scores: [N] predicted quality scores
            pred_masks: [N, H, W] logits
            gt_masks: [N, H, W] binary
        """
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0),
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        pred_flat = pred_masks.sigmoid().flatten(1)
        gt_flat = gt_masks.float().flatten(1)
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
        actual_iou = (2.0 * intersection + 1e-6) / (union + 1e-6)

        return F.mse_loss(iou_scores.sigmoid(), actual_iou.detach())

    def forward(
        self,
        student_out: dict,
        gt_masks_list: list,
        gt_boxes_list: list,
        student_indices: list,
        gt_indices: list,
    ) -> dict:
        """Compute all video distillation loss terms.

        Args:
            student_out: dict with pred_masks, pred_boxes, pred_logits,
                        presence_logits, iou_scores (batched)
            gt_masks_list: list of [N_i, H, W] binary masks per sample
            gt_boxes_list: list of [N_i, 4] boxes per sample
            student_indices: list of [M] tensors — matched student prediction indices
            gt_indices: list of [M] tensors — matched GT object indices

        Returns:
            dict with individual losses and total_loss
        """
        losses = {}
        batch_size = student_out["pred_masks"].shape[0]
        device = student_out["pred_masks"].device

        mask_loss_sum = torch.tensor(0.0, device=device)
        box_l1_sum = torch.tensor(0.0, device=device)
        box_giou_sum = torch.tensor(0.0, device=device)
        iou_token_sum = torch.tensor(0.0, device=device)
        presence_sum = torch.tensor(0.0, device=device)

        valid_samples = 0

        for b in range(batch_size):
            s_idx = student_indices[b]
            g_idx = gt_indices[b]

            if len(s_idx) == 0:
                continue

            valid_samples += 1

            s_idx_dev = s_idx.to(device)
            # Extract matched predictions
            s_masks = student_out["pred_masks"][b, s_idx_dev].float()
            s_boxes = student_out["pred_boxes"][b, s_idx_dev].float()
            s_logits = student_out["pred_logits"][b, s_idx_dev].float()
            s_iou = student_out["iou_scores"][b, s_idx_dev].float()

            # Extract matched GT
            # GT tensors come from dataloader on CPU; index them with CPU indices first.
            g_idx_cpu = g_idx.detach().to("cpu", dtype=torch.long)
            g_masks = gt_masks_list[b][g_idx_cpu].to(device).float()
            g_boxes = gt_boxes_list[b][g_idx_cpu].to(device).float()

            # Mask loss
            mask_loss_sum = mask_loss_sum + self._mask_loss(s_masks, g_masks)

            # Box losses
            box_l1_sum = box_l1_sum + F.l1_loss(s_boxes, g_boxes)
            giou = compute_giou(s_boxes, g_boxes)
            box_giou_sum = box_giou_sum + (1.0 - giou).mean()

            # IoU token loss
            iou_token_sum = iou_token_sum + self._iou_token_loss(s_iou, s_masks, g_masks)

            # Presence loss: matched predictions should have high logits
            target_presence = torch.ones_like(s_logits)
            presence_sum = presence_sum + F.binary_cross_entropy_with_logits(
                s_logits, target_presence, reduction="mean"
            )

        # Average over valid samples
        denom = max(valid_samples, 1)
        losses["mask_loss"] = (mask_loss_sum / denom) * self.config.mask_loss_weight
        losses["box_l1_loss"] = (box_l1_sum / denom) * self.config.box_l1_loss_weight
        losses["box_giou_loss"] = (box_giou_sum / denom) * self.config.box_giou_loss_weight
        losses["iou_token_loss"] = (iou_token_sum / denom) * self.config.iou_token_loss_weight
        losses["presence_loss"] = (presence_sum / denom) * self.config.presence_loss_weight

        losses["total_loss"] = sum(losses.values())
        return losses
