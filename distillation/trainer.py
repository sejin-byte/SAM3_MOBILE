"""Distillation Trainer — orchestrates teacher/student forward, matching, and training."""

import os
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .config import DistillationConfig
from .dataset import SA1BDistillDataset, collate_fn
from .prompt_encoder import GeometricPromptEncoder
from .greedy_matcher import GreedyMatcher
from .losses import DistillationLoss


def resize_teacher_rope(teacher, new_image_size: int):
    """Recompute SAM3 teacher's global-layer RoPE buffers for a new resolution.

    SAM3's Hiera backbone pre-computes 2D RoPE cos/sin for a fixed grid.
    Windowed layers (24x24) are unaffected; only global attention layers
    (indexes 7, 15, 23, 31) need new buffers sized (new_grid*new_grid, head_dim).

    Args:
        teacher: Sam3Model instance (already on device)
        new_image_size: target resolution (must be divisible by patch_size=14)
    """
    vit_config = teacher.config.vision_config.backbone_config
    patch_size = vit_config.patch_size
    new_grid = new_image_size // patch_size
    window_size = vit_config.window_size
    dim = vit_config.hidden_size // vit_config.num_attention_heads
    rope_theta = vit_config.rope_theta

    for layer_idx in vit_config.global_attn_indexes:
        rotary_emb = teacher.vision_encoder.backbone.layers[layer_idx].rotary_emb

        # Scale factor matches teacher's init: window_size / grid_size
        scale = window_size / new_grid
        end_x, end_y = new_grid, new_grid

        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
        flat_idx = torch.arange(end_x * end_y, dtype=torch.long)
        x_pos = (flat_idx % end_x).float() * scale
        y_pos = torch.div(flat_idx, end_x, rounding_mode="floor").float() * scale
        inv_freq = torch.cat([
            torch.outer(x_pos, freqs),
            torch.outer(y_pos, freqs),
        ], dim=-1).repeat_interleave(2, dim=-1)

        device = rotary_emb.rope_embeddings_cos.device
        dtype = rotary_emb.rope_embeddings_cos.dtype
        rotary_emb.register_buffer(
            "rope_embeddings_cos", inv_freq.cos().to(device=device, dtype=dtype), persistent=False,
        )
        rotary_emb.register_buffer(
            "rope_embeddings_sin", inv_freq.sin().to(device=device, dtype=dtype), persistent=False,
        )
        rotary_emb.end_x = end_x
        rotary_emb.end_y = end_y

    print(f"  Teacher RoPE resized: {vit_config.image_size}px ({vit_config.image_size // patch_size}x"
          f"{vit_config.image_size // patch_size}) -> {new_image_size}px ({new_grid}x{new_grid})")


class DistillationTrainer:
    """Trains EfficientSAM3 student via knowledge distillation from Sam3 teacher.

    Training pipeline per step:
    1. Load batch with dual-preprocessed images + dynamic prompts
    2. Teacher forward (no_grad, FP16) -> outputs + intermediates
    3. Student forward (FP16 + grad) with prompt injection -> outputs + intermediates
    4. Greedy matching between student and teacher predictions
    5. Compute 9-term loss (FP32)
    6. Backward + gradient accumulation + optimizer step
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        teacher_tokenizer,
        student_tokenizer,
        config: DistillationConfig = None,
        phase: int = 1,
        device: str = "mps",
        freeze_vision_epochs: int = 0,
    ):
        self.config = config or DistillationConfig()
        self.phase = phase
        self.device = torch.device(device)
        self.teacher_dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32
        self.freeze_vision_epochs = freeze_vision_epochs

        # Models — teacher FP16 (frozen), student FP32 (trainable, needs gradient precision)
        self.teacher = teacher_model.to(self.device).to(self.teacher_dtype)
        self.teacher.requires_grad_(False)
        self.teacher.train(False)

        # Resize teacher RoPE if training at non-native resolution
        native_size = self.teacher.config.vision_config.backbone_config.image_size
        if self.config.image_size != native_size:
            resize_teacher_rope(self.teacher, self.config.image_size)
        self.student = student_model.to(self.device).float().train()
        if self.config.gradient_checkpointing:
            self.student.enable_gradient_checkpointing()
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

        # Distillation components
        self.prompt_encoder = GeometricPromptEncoder(hidden_size=256).to(self.device).float()
        self.matcher = GreedyMatcher(
            mask_iou_weight=self.config.matcher_mask_iou_weight,
            box_l1_weight=self.config.matcher_box_l1_weight,
            logit_sim_weight=self.config.matcher_logit_sim_weight,
        )
        self.loss_fn = DistillationLoss(self.config)

        # Optimizer — use differential LR when freezing vision
        lr = self.config.phase1_lr if phase == 1 else self.config.phase2_lr
        if freeze_vision_epochs > 0:
            # Group params: vision (low LR) vs text/other (normal LR)
            vision_params = []
            other_params = []
            vision_module_prefixes = ("vision_encoder",)
            for name, param in self.student.named_parameters():
                if any(name.startswith(p) for p in vision_module_prefixes):
                    vision_params.append(param)
                else:
                    other_params.append(param)
            self.optimizer = torch.optim.AdamW([
                {"params": vision_params, "lr": lr * 0.1, "name": "vision"},
                {"params": other_params, "lr": lr, "name": "other"},
                {"params": list(self.prompt_encoder.parameters()), "lr": lr, "name": "prompt_enc"},
            ], weight_decay=self.config.weight_decay)
            print(f"  Differential LR: vision={lr*0.1:.1e}, text/other={lr:.1e}")
            print(f"  Vision frozen for first {freeze_vision_epochs} epoch(s)")
        else:
            self.optimizer = torch.optim.AdamW(
                list(self.student.parameters()) + list(self.prompt_encoder.parameters()),
                lr=lr,
                weight_decay=self.config.weight_decay,
            )

        # LR scheduler
        num_epochs = self.config.phase1_epochs if phase == 1 else self.config.phase2_epochs
        self.num_epochs = num_epochs

        # Logging
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_subdir = os.path.join(self.config.log_dir, f"phase{phase}")
        self.writer = SummaryWriter(log_subdir) if SummaryWriter is not None else None

        # Visualization
        self.vis_dir = self.config.vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self._last_vis_data = None
        self._max_steps = None

        self.global_step = 0

    def _log_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    @torch.no_grad()
    def _visualize_predictions(
        self,
        student_out: dict,
        teacher_out: dict,
        student_indices: torch.Tensor,
        teacher_indices: torch.Tensor,
        student_pixels: torch.Tensor,
        epoch: int,
    ):
        """Save side-by-side comparison of teacher vs student masks as PNG.

        Layout: original image | teacher top-K masks | student top-K masks
        """
        top_k = min(self.config.vis_top_k, len(student_indices))
        if top_k == 0:
            return

        # Denormalize student image for display
        mean = torch.tensor(self.config.student_mean, device=student_pixels.device).view(3, 1, 1)
        std = torch.tensor(self.config.student_std, device=student_pixels.device).view(3, 1, 1)
        img = (student_pixels[0].float() * std + mean).clamp(0, 1)
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_h, img_w = img_np.shape[:2]

        # Select top-K matched pairs (by student logit confidence)
        s_logits = student_out["pred_logits"][0, student_indices].float()
        topk_vals, topk_idx = s_logits.topk(top_k)
        sel_s = student_indices[topk_idx]
        sel_t = teacher_indices[topk_idx]

        # Generate mask overlays
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255),
        ]

        def make_overlay(masks, indices, target_h, target_w):
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                mask = masks[0, idx].float().sigmoid()
                if mask.shape[-2:] != (target_h, target_w):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                color = colors[i % len(colors)]
                for c in range(3):
                    canvas[:, :, c] = np.where(mask_np, color[c], canvas[:, :, c])
            return canvas

        # Resize to a manageable display size
        disp_h, disp_w = 504, 504
        img_disp = np.array(Image.fromarray(img_np).resize((disp_w, disp_h)))

        s_overlay = make_overlay(student_out["pred_masks"], sel_s, disp_h, disp_w)
        t_overlay = make_overlay(teacher_out["pred_masks"], sel_t, disp_h, disp_w)

        # Blend overlays with image
        alpha = 0.5
        s_blend = (img_disp * (1 - alpha) + s_overlay * alpha).astype(np.uint8)
        t_blend = (img_disp * (1 - alpha) + t_overlay * alpha).astype(np.uint8)

        # Concatenate: image | teacher | student
        combined = np.concatenate([img_disp, t_blend, s_blend], axis=1)
        vis_img = Image.fromarray(combined)

        path = os.path.join(self.vis_dir, f"phase{self.phase}_epoch{epoch}_step{self.global_step}.png")
        vis_img.save(path)
        print(f"  Visualization saved: {path}")

    def _build_dataloader(self, split: str = "train") -> DataLoader:
        dataset = SA1BDistillDataset(
            config=self.config,
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer,
            split=split,
            phase=self.phase,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,  # MPS: must be False
            collate_fn=collate_fn,
            drop_last=(split == "train"),
        )

    @torch.no_grad()
    def _teacher_forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> dict:
        """Run teacher model and extract outputs + intermediates."""
        # Get vision features first (for FPN intermediates)
        vision_outputs = self.teacher.get_vision_features(pixel_values)
        fpn_features = list(vision_outputs.fpn_hidden_states[:-1])  # 3 levels

        # Full forward with pre-computed vision features + hidden states for distillation
        outputs = self.teacher(
            vision_embeds=vision_outputs,
            input_ids=input_ids,
            output_hidden_states=True,
        )

        # Extract encoder output (last encoder layer)
        encoder_output = None
        if outputs.encoder_hidden_states is not None and len(outputs.encoder_hidden_states) > 0:
            encoder_output = outputs.encoder_hidden_states[-1]

        return {
            "pred_masks": outputs.pred_masks,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.pred_logits,
            "presence_logits": outputs.presence_logits,
            "semantic_seg": outputs.semantic_seg,
            "fpn_features": fpn_features,
            "encoder_output": encoder_output,
        }

    def _student_forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_types: list,
        prompt_coords: torch.Tensor,
    ) -> dict:
        """Run student model with geometric prompt injection."""
        # Encode geometric prompts
        prompt_embeddings = self.prompt_encoder(prompt_types, prompt_coords)

        # Check if any geometric prompts exist
        has_geometric = any(p in ("point", "box") for p in prompt_types)
        if not has_geometric:
            prompt_embeddings = None

        return self.student.forward_with_intermediates(
            pixel_values=pixel_values,
            input_ids=input_ids,
            prompt_embeddings=prompt_embeddings,
        )

    def _train_step(self, batch: dict) -> dict:
        """Single training step with gradient accumulation."""
        # Move to device — teacher FP16, student FP32
        teacher_pixels = batch["teacher_pixel_values"].to(self.device, dtype=self.teacher_dtype)
        student_pixels = batch["student_pixel_values"].to(self.device).float()
        teacher_input_ids = batch["teacher_input_ids"].to(self.device)
        student_input_ids = batch["student_input_ids"].to(self.device)
        prompt_types = batch["prompt_types"]
        prompt_coords = batch["prompt_coords"].to(self.device).float()

        # Teacher forward (no grad)
        teacher_out = self._teacher_forward(teacher_pixels, teacher_input_ids)

        # Free teacher image tensor before student forward to reduce peak memory
        del teacher_pixels
        if self.device.type == "mps":
            torch.mps.empty_cache()

        # Student forward (with grad)
        student_out = self._student_forward(
            student_pixels,
            student_input_ids,
            prompt_types,
            prompt_coords,
        )

        # Greedy matching (per sample in batch, no grad)
        batch_size = student_out["pred_masks"].shape[0]
        all_student_idx = []
        all_teacher_idx = []
        with torch.no_grad():
            for b in range(batch_size):
                s_masks = student_out["pred_masks"][b].float()
                t_masks = teacher_out["pred_masks"][b].float()
                s_boxes = student_out["pred_boxes"][b].float()
                t_boxes = teacher_out["pred_boxes"][b].float()
                s_logits = student_out["pred_logits"][b].float()
                t_logits = teacher_out["pred_logits"][b].float()

                # Optionally drop very low-confidence teacher queries.
                # Keep at least one teacher query to avoid empty matching.
                t_index_map = None
                if self.config.matcher_teacher_score_threshold > 0:
                    keep = t_logits.sigmoid() >= self.config.matcher_teacher_score_threshold
                    if not keep.any():
                        keep[t_logits.argmax()] = True
                    t_index_map = torch.nonzero(keep, as_tuple=False).squeeze(1)
                    t_masks = t_masks[t_index_map]
                    t_boxes = t_boxes[t_index_map]
                    t_logits = t_logits[t_index_map]

                s_idx, t_idx = self.matcher.match(
                    s_masks, t_masks, s_boxes, t_boxes, s_logits, t_logits,
                    max_matches=self.config.matcher_max_matches,
                )
                if t_index_map is not None:
                    t_idx = t_index_map[t_idx]
                all_student_idx.append(s_idx)
                all_teacher_idx.append(t_idx)

        # Compute loss (FP32 internally)
        losses = self.loss_fn(
            student_out, teacher_out,
            all_student_idx, all_teacher_idx,
            phase=self.phase,
        )

        # Cache data for visualization (first sample only to save memory)
        self._last_vis_data = {
            "student_out": {
                "pred_masks": student_out["pred_masks"][:1].detach(),
                "pred_logits": student_out["pred_logits"][:1].detach(),
            },
            "teacher_out": {
                "pred_masks": teacher_out["pred_masks"][:1].detach(),
                "pred_logits": teacher_out["pred_logits"][:1].detach(),
            },
            "student_indices": all_student_idx[0],
            "teacher_indices": all_teacher_idx[0],
            "student_pixels": student_pixels[:1].detach(),
        }

        # Scale loss for gradient accumulation
        scaled_loss = losses["total_loss"] / self.config.grad_accum_steps
        scaled_loss.backward()

        # Convert to Python floats and free computation graph
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        del student_out, teacher_out, losses, scaled_loss
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return loss_dict

    def _get_cosine_lr(self, step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < warmup_steps:
            return base_lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train(self, resume_step: int = 0, max_steps: int = None):
        """Run training loop."""
        # max_steps is interpreted as "additional steps to run" from resume_step.
        self._max_steps = (resume_step + max_steps) if max_steps is not None else None
        train_loader = self._build_dataloader("train")
        total_steps = len(train_loader) * self.num_epochs
        base_lr = self.config.phase1_lr if self.phase == 1 else self.config.phase2_lr
        warmup_steps = self.config.phase1_warmup_steps if self.phase == 1 else 0

        print(f"Phase {self.phase}: {self.num_epochs} epochs, {len(train_loader)} steps/epoch, "
              f"total {total_steps} steps")
        print(f"LR: {base_lr}, warmup: {warmup_steps}, grad_accum: {self.config.grad_accum_steps}")

        self.global_step = resume_step

        for epoch in range(self.num_epochs):
            # Freeze/unfreeze vision encoder based on epoch
            if self.freeze_vision_epochs > 0:
                if epoch < self.freeze_vision_epochs:
                    self.student.vision_encoder.requires_grad_(False)
                    print(f"  Epoch {epoch}: vision_encoder FROZEN (text alignment phase)")
                elif epoch == self.freeze_vision_epochs:
                    self.student.vision_encoder.requires_grad_(True)
                    # Restore vision LR to full
                    for pg in self.optimizer.param_groups:
                        if pg.get("name") == "vision":
                            pg["lr"] = base_lr
                    print(f"  Epoch {epoch}: vision_encoder UNFROZEN (full fine-tuning)")

            epoch_loss = 0.0
            epoch_start = time.time()

            for step_in_epoch, batch in enumerate(train_loader):
                # Cosine LR with warmup
                lr = self._get_cosine_lr(self.global_step, total_steps, base_lr, warmup_steps)
                self._set_lr(lr)

                # Training step
                losses = self._train_step(batch)

                # Gradient accumulation step
                if (step_in_epoch + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.student.parameters()) + list(self.prompt_encoder.parameters()),
                        self.config.max_grad_norm,
                    )
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
                            self._log_scalar(f"train/{k}", v, self.global_step)

                    print(f"  [{self.global_step:6d}] loss={losses['total_loss']:.4f} "
                          f"mask={losses.get('mask_loss', 0):.3f} "
                          f"box_l1={losses.get('box_l1_loss', 0):.3f} "
                          f"iou={losses.get('iou_token_loss', 0):.3f} "
                          f"lr={lr:.2e}")

                # Checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(epoch, step_in_epoch)

                # Periodic visualization
                if self.global_step % self.config.vis_every_n_steps == 0 and self._last_vis_data is not None:
                    self._visualize_predictions(
                        self._last_vis_data["student_out"],
                        self._last_vis_data["teacher_out"],
                        self._last_vis_data["student_indices"],
                        self._last_vis_data["teacher_indices"],
                        self._last_vis_data["student_pixels"],
                        epoch,
                    )

                # Early stop for debug mode
                if self._max_steps is not None and self.global_step >= self._max_steps:
                    break

            avg_loss = epoch_loss / max(step_in_epoch + 1, 1)
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.num_epochs} done - avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

            # End-of-epoch visualization
            if self._last_vis_data is not None:
                self._visualize_predictions(
                    self._last_vis_data["student_out"],
                    self._last_vis_data["teacher_out"],
                    self._last_vis_data["student_indices"],
                    self._last_vis_data["teacher_indices"],
                    self._last_vis_data["student_pixels"],
                    epoch,
                )

            # End-of-epoch checkpoint
            self._save_checkpoint(epoch, len(train_loader) - 1)

            if self._max_steps is not None and self.global_step >= self._max_steps:
                break

        if self.writer is not None:
            self.writer.close()
        print("Training complete.")

    def _save_checkpoint(self, epoch: int, step: int):
        path = os.path.join(
            self.config.checkpoint_dir,
            f"phase{self.phase}_epoch{epoch}_step{self.global_step}.pt",
        )
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "prompt_encoder_state_dict": self.prompt_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "phase": self.phase,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.prompt_encoder.load_state_dict(ckpt["prompt_encoder_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {path} (step {self.global_step})")
        return ckpt

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and return average losses."""
        val_loader = self._build_dataloader("val")
        self.student.eval()

        total_losses = {}
        num_batches = 0

        for batch in val_loader:
            teacher_pixels = batch["teacher_pixel_values"].to(self.device, dtype=self.teacher_dtype)
            student_pixels = batch["student_pixel_values"].to(self.device).float()
            teacher_input_ids = batch["teacher_input_ids"].to(self.device)
            student_input_ids = batch["student_input_ids"].to(self.device)
            prompt_types = batch["prompt_types"]
            prompt_coords = batch["prompt_coords"].to(self.device).float()

            teacher_out = self._teacher_forward(teacher_pixels, teacher_input_ids)
            student_out = self._student_forward(
                student_pixels,
                student_input_ids,
                prompt_types,
                prompt_coords,
            )

            batch_size = student_out["pred_masks"].shape[0]
            all_student_idx = []
            all_teacher_idx = []
            for b in range(batch_size):
                s_masks = student_out["pred_masks"][b].float()
                t_masks = teacher_out["pred_masks"][b].float()
                s_boxes = student_out["pred_boxes"][b].float()
                t_boxes = teacher_out["pred_boxes"][b].float()
                s_logits = student_out["pred_logits"][b].float()
                t_logits = teacher_out["pred_logits"][b].float()

                t_index_map = None
                if self.config.matcher_teacher_score_threshold > 0:
                    keep = t_logits.sigmoid() >= self.config.matcher_teacher_score_threshold
                    if not keep.any():
                        keep[t_logits.argmax()] = True
                    t_index_map = torch.nonzero(keep, as_tuple=False).squeeze(1)
                    t_masks = t_masks[t_index_map]
                    t_boxes = t_boxes[t_index_map]
                    t_logits = t_logits[t_index_map]

                s_idx, t_idx = self.matcher.match(
                    s_masks, t_masks, s_boxes, t_boxes, s_logits, t_logits,
                    max_matches=self.config.matcher_max_matches,
                )
                if t_index_map is not None:
                    t_idx = t_index_map[t_idx]
                all_student_idx.append(s_idx)
                all_teacher_idx.append(t_idx)

            losses = self.loss_fn(
                student_out, teacher_out, all_student_idx, all_teacher_idx, phase=self.phase,
            )

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val
            num_batches += 1

        self.student.train()

        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        for k, v in avg_losses.items():
            self._log_scalar(f"val/{k}", v, self.global_step)

        print(f"Validation: {' | '.join(f'{k}={v:.4f}' for k, v in avg_losses.items())}")
        return avg_losses
