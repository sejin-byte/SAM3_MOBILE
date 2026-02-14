"""Video Distillation Trainer — trains Perceiver Resampler + MemoryCrossAttention.

All other model components (backbone, text encoder, DETR encoder/decoder, mask decoder)
are frozen. No teacher model is loaded — context features are pre-cached.

Training pipeline per step:
1. Load batch: cached context FPN L3 [T, 256, 18, 18] + query frame pixels + GT masks
2. Flatten context → [batch, T*324, 256] → Perceiver compress → [batch, 64, 256]
3. Student forward_video: vision+text → encoder → memory_cross_attn → decoder → masks
4. Match 100 predictions to N GT objects via GreedyMatcher
5. Compute 5-term GT-based loss
6. Backward + gradient accumulation + optimizer step
"""

import os
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .video_config import VideoDistillationConfig
from .video_dataset import SAVVideoDataset, video_collate_fn
from .video_losses import VideoDistillationLoss
from .greedy_matcher import GreedyMatcher


class VideoDistillationTrainer:
    """Trains temporal memory modules via video distillation."""

    def __init__(
        self,
        student_model,
        student_tokenizer,
        config: VideoDistillationConfig = None,
        device: str = "mps",
    ):
        self.config = config or VideoDistillationConfig()
        self.device = torch.device(device)

        # Student model — load as float32, then freeze/unfreeze
        self.student = student_model.to(self.device).float()
        self._freeze_and_unfreeze()

        self.student_tokenizer = student_tokenizer

        # Loss and matcher
        self.matcher = GreedyMatcher(
            mask_iou_weight=self.config.matcher_mask_iou_weight,
            box_l1_weight=self.config.matcher_box_l1_weight,
            logit_sim_weight=self.config.matcher_logit_sim_weight,
        )
        self.loss_fn = VideoDistillationLoss(self.config)

        # Optimizer — only trainable parameters
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Logging
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.config.log_dir) if SummaryWriter is not None else None

        self.global_step = 0

    def _freeze_and_unfreeze(self):
        """Freeze everything, then unfreeze only perceiver_resampler + memory_cross_attn."""
        # Freeze all
        self.student.requires_grad_(False)
        self.student.train(False)

        # Unfreeze trainable modules
        for name in ("perceiver_resampler", "memory_cross_attn"):
            module = getattr(self.student, name, None)
            if module is not None:
                module.requires_grad_(True)
                module.train(True)

        # Report
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.student.parameters() if not p.requires_grad)
        print("  Trainable: {:.3f}M | Frozen: {:.1f}M".format(trainable / 1e6, frozen / 1e6))

    def _prepare_memory_features(self, context_features: torch.Tensor) -> torch.Tensor:
        """Flatten cached FPN L3 context frames for Perceiver input.

        Args:
            context_features: [batch, T, 256, 18, 18]

        Returns:
            [batch, T*324, 256] — flattened spatial tokens from all context frames
        """
        batch, T, C, H, W = context_features.shape
        # [batch, T, 256, 18, 18] → [batch, T, 256, 324] → [batch, T, 324, 256]
        x = context_features.reshape(batch, T, C, H * W).permute(0, 1, 3, 2)
        # [batch, T, 324, 256] → [batch, T*324, 256]
        return x.reshape(batch, T * H * W, C)

    def _log_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _build_dataloader(self) -> DataLoader:
        dataset = SAVVideoDataset(
            config=self.config,
            student_tokenizer=self.student_tokenizer,
        )
        print("  Dataset: {} video clips".format(len(dataset)))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=video_collate_fn,
            drop_last=True,
        )

    def _match_predictions_to_gt(
        self,
        student_out: dict,
        gt_masks_list: list,
        gt_boxes_list: list,
    ):
        """Match student predictions to GT objects per sample.

        Uses GreedyMatcher with student masks vs GT masks.
        For GT matching, we create pseudo-logits (all 1.0) since GT is binary.
        """
        batch_size = student_out["pred_masks"].shape[0]
        all_student_idx = []
        all_gt_idx = []

        with torch.no_grad():
            for b in range(batch_size):
                n_gt = gt_masks_list[b].shape[0]
                if n_gt == 0:
                    all_student_idx.append(torch.tensor([], dtype=torch.long, device=self.device))
                    all_gt_idx.append(torch.tensor([], dtype=torch.long, device=self.device))
                    continue

                s_masks = student_out["pred_masks"][b].float()
                s_boxes = student_out["pred_boxes"][b].float()
                s_logits = student_out["pred_logits"][b].float()

                # Resize GT masks to student mask spatial size for matching
                g_masks = gt_masks_list[b].to(self.device).float()
                if g_masks.shape[-2:] != s_masks.shape[-2:]:
                    g_masks = F.interpolate(
                        g_masks.unsqueeze(0),
                        size=s_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                g_boxes = gt_boxes_list[b].to(self.device).float()
                # Pseudo-logits for GT: high confidence (5.0 pre-sigmoid ~ 0.993)
                g_logits = torch.full((n_gt,), 5.0, device=self.device)

                s_idx, g_idx = self.matcher.match(
                    s_masks, g_masks, s_boxes, g_boxes, s_logits, g_logits,
                )
                all_student_idx.append(s_idx)
                all_gt_idx.append(g_idx)

        return all_student_idx, all_gt_idx

    def _train_step(self, batch: dict) -> dict:
        """Single training step."""
        # Move to device
        context_features = batch["context_features"].to(self.device).float()
        student_pixels = batch["student_pixel_values"].to(self.device).float()
        input_ids = batch["student_input_ids"].to(self.device)
        gt_masks_list = batch["gt_masks"]  # list of tensors (variable N)
        gt_boxes_list = batch["gt_boxes"]  # list of tensors (variable N)

        # Prepare memory: [batch, T, 256, 18, 18] → [batch, T*324, 256]
        memory_features = self._prepare_memory_features(context_features)

        # Student forward (video mode)
        student_out = self.student.forward_video(
            pixel_values=student_pixels,
            input_ids=input_ids,
            memory_features=memory_features,
        )

        # Match predictions to GT
        student_indices, gt_indices = self._match_predictions_to_gt(
            student_out, gt_masks_list, gt_boxes_list,
        )

        # Compute loss
        losses = self.loss_fn(
            student_out, gt_masks_list, gt_boxes_list,
            student_indices, gt_indices,
        )

        # Backward with gradient accumulation scaling
        scaled_loss = losses["total_loss"] / self.config.grad_accum_steps
        scaled_loss.backward()

        # Convert to Python floats
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

        del student_out, losses, scaled_loss, context_features, memory_features
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return loss_dict

    def _get_cosine_lr(self, step: int, total_steps: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.lr * step / max(self.config.warmup_steps, 1)
        progress = (step - self.config.warmup_steps) / max(total_steps - self.config.warmup_steps, 1)
        return self.config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train(self, resume_step: int = 0, max_steps: int = None):
        """Run training loop."""
        # max_steps is interpreted as "additional steps to run" from resume_step.
        target_global_step = (resume_step + max_steps) if max_steps is not None else None
        train_loader = self._build_dataloader()
        total_steps = len(train_loader) * self.config.epochs

        print("Video distillation: {} epochs, {} steps/epoch, total {} steps".format(
            self.config.epochs, len(train_loader), total_steps))
        print("LR: {}, warmup: {}, grad_accum: {}".format(
            self.config.lr, self.config.warmup_steps, self.config.grad_accum_steps))

        self.global_step = resume_step

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step_in_epoch, batch in enumerate(train_loader):
                # Cosine LR with warmup
                lr = self._get_cosine_lr(self.global_step, total_steps)
                self._set_lr(lr)

                # Training step
                losses = self._train_step(batch)

                # Gradient accumulation step
                if (step_in_epoch + 1) % self.config.grad_accum_steps == 0:
                    trainable_params = [p for p in self.student.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += losses["total_loss"]
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    self._log_scalar("train/total_loss", losses["total_loss"], self.global_step)
                    self._log_scalar("train/lr", lr, self.global_step)
                    for k, v in losses.items():
                        if k != "total_loss":
                            self._log_scalar("train/{}".format(k), v, self.global_step)

                    print("  [{:6d}] loss={:.4f} mask={:.3f} box_l1={:.3f} "
                          "giou={:.3f} iou={:.3f} lr={:.2e}".format(
                              self.global_step, losses["total_loss"],
                              losses.get("mask_loss", 0),
                              losses.get("box_l1_loss", 0),
                              losses.get("box_giou_loss", 0),
                              losses.get("iou_token_loss", 0),
                              lr))

                # Checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(epoch)

                # Early stop for debug
                if target_global_step is not None and self.global_step >= target_global_step:
                    break

            avg_loss = epoch_loss / max(step_in_epoch + 1, 1)
            elapsed = time.time() - epoch_start
            print("Epoch {}/{} done — avg_loss={:.4f} ({:.1f}s)".format(
                epoch + 1, self.config.epochs, avg_loss, elapsed))

            # End-of-epoch checkpoint
            self._save_checkpoint(epoch)

            if target_global_step is not None and self.global_step >= target_global_step:
                break

        if self.writer is not None:
            self.writer.close()
        print("Video distillation training complete.")

    def _save_checkpoint(self, epoch: int):
        """Save only trainable module state dicts + optimizer."""
        path = os.path.join(
            self.config.checkpoint_dir,
            "video_epoch{}_step{}.pt".format(epoch, self.global_step),
        )
        save_dict = {
            "perceiver_resampler_state_dict": self.student.perceiver_resampler.state_dict(),
            "memory_cross_attn_state_dict": self.student.memory_cross_attn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
        }
        torch.save(save_dict, path)
        print("  Checkpoint saved: {}".format(path))

    def load_checkpoint(self, path: str):
        """Load trainable module weights + optimizer."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.student.perceiver_resampler.load_state_dict(ckpt["perceiver_resampler_state_dict"])
        self.student.memory_cross_attn.load_state_dict(ckpt["memory_cross_attn_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        print("Resumed from {} (step {})".format(path, self.global_step))
        return ckpt
