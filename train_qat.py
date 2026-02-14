"""EfficientSAM3 Quantization-Aware Training (QAT) — Stage 4 Conditional.

Use this script ONLY if PTQ (quantize_model.py) mIoU drop exceeds 2%.
Inserts fake quantization nodes and fine-tunes for 1-2 epochs on SA-1B
to recover accuracy lost from quantization noise.

Usage:
    python train_qat.py                                    # Default: Int4, 2 epochs
    python train_qat.py --mode int8_int4 --epochs 1        # Int8+Int4, 1 epoch
    python train_qat.py --checkpoint path/to/ckpt.pt       # Specific checkpoint
    python train_qat.py --debug                            # Debug: 1 epoch, 10 steps
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import EfficientSAM3, EfficientSAM3Config
from distillation.config import DistillationConfig
from distillation.dataset import SA1BDistillDataset, collate_fn
from distillation.greedy_matcher import GreedyMatcher
from distillation.losses import DistillationLoss

# Same sensitive modules as PTQ — keep them in FP16 during QAT
SKIP_MODULES = {
    "iou_head",
    "dot_product_scoring",
    "perceiver_resampler",
    "memory_cross_attn",
}


def should_quantize(module: torch.nn.Module, fqn: str) -> bool:
    """Filter function for QAT: returns True for modules to insert fake quant."""
    if not isinstance(module, torch.nn.Linear):
        return False
    for skip in SKIP_MODULES:
        if skip in fqn:
            return False
    return True


def find_latest_checkpoint(ckpt_dir: str = "checkpoints/distillation") -> str:
    """Find the latest distillation checkpoint (prefer Phase 2 over Phase 1)."""
    from pathlib import Path
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    for prefix in ["phase2_", "phase1_"]:
        ckpts = sorted([f for f in ckpt_dir.iterdir() if f.name.startswith(prefix)])
        if ckpts:
            return str(ckpts[-1])

    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def load_teacher_with_fallback(model_name: str):
    """Load teacher processor/model with online-first, local-cache fallback."""
    from transformers import Sam3Model, Sam3Processor

    try:
        processor = Sam3Processor.from_pretrained(model_name)
        teacher = Sam3Model.from_pretrained(model_name, torch_dtype=torch.float16)
        return processor, teacher
    except Exception as e:
        print(f"  WARN: teacher online load failed: {e}")
        print("  Retrying with local_files_only=True ...")
        processor = Sam3Processor.from_pretrained(model_name, local_files_only=True)
        teacher = Sam3Model.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True)
        return processor, teacher


def main():
    parser = argparse.ArgumentParser(description="EfficientSAM3 QAT Fine-tuning (Stage 4)")
    parser.add_argument("--mode", type=str, default="int4",
                        choices=["int4", "int8_int4"],
                        help="QAT base quantization config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to distillation checkpoint (auto-detects latest)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cpu)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps for LR scheduler")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--sa1b-dir", type=str, default="data/sa1b",
                        help="SA-1B dataset directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints/quantized",
                        help="Output directory for QAT checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Debug run: 1 epoch, 10 steps")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")

    args = parser.parse_args()

    print("=" * 60)
    print("  EfficientSAM3 Quantization-Aware Training")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    debug_max_steps = None
    if args.debug:
        print("  ** DEBUG MODE: 1 epoch, 10 steps **")
        args.epochs = 1
        args.log_every = 1
        debug_max_steps = 10

    # ── Find checkpoint ──
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_checkpoint()
    print(f"  Checkpoint: {ckpt_path}")

    # ── Load student model ──
    print("\n[1/5] Loading student model...")
    config = EfficientSAM3Config()
    student = EfficientSAM3(config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("student_state_dict", ckpt)
    student.load_state_dict(state_dict, strict=False)
    print(f"  Student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params")

    # ── Insert fake quantization nodes (QAT prepare step) ──
    print("\n[2/5] Inserting fake quantization nodes...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig

    if args.mode == "int4":
        base_config = Int4WeightOnlyConfig(group_size=128)
    elif args.mode == "int8_int4":
        from torchao.quantization import Int8DynamicActivationInt4WeightConfig
        base_config = Int8DynamicActivationInt4WeightConfig()

    qat_config = QATConfig(base_config, step="prepare")
    quantize_(student, qat_config, filter_fn=should_quantize)
    print(f"  Fake quant nodes inserted for mode: {args.mode}")

    # Move to device in float32 (fake quant operates in float)
    student = student.to(args.device).float()
    student.train()

    # ── Load teacher model ──
    print("\n[3/5] Loading teacher model...")

    dist_config = DistillationConfig()
    dist_config.sa1b_dir = args.sa1b_dir

    processor, teacher = load_teacher_with_fallback(dist_config.teacher_model_name)
    teacher = teacher.to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params")

    tokenizer = processor.tokenizer

    # ── Dataset ──
    print("\n[4/5] Setting up dataset...")
    dataset = SA1BDistillDataset(
        config=dist_config,
        tokenizer=tokenizer,
        split="train",
        phase=2,  # Use Phase 2 prompt ratios
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # batch_size=1 for 24GB UMA
        shuffle=True,
        num_workers=dist_config.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"  Training samples: {len(dataset)}")

    # ── Optimizer & Scheduler ──
    for p in student.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    if debug_max_steps:
        total_steps = min(total_steps, debug_max_steps)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss & Matcher ──
    loss_fn = DistillationLoss(dist_config)
    matcher = GreedyMatcher(
        mask_iou_weight=dist_config.matcher_mask_iou_weight,
        box_l1_weight=dist_config.matcher_box_l1_weight,
        logit_sim_weight=dist_config.matcher_logit_sim_weight,
    )

    # ── Training loop ──
    print("\n[5/5] Starting QAT training...")
    print(f"  Total steps: {total_steps}")

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        step_in_epoch = 0

        for batch_idx, batch in enumerate(dataloader):
            if debug_max_steps and global_step >= debug_max_steps:
                break

            teacher_pixels = batch["teacher_pixel_values"].to(args.device, dtype=torch.float16)
            student_pixels = batch["student_pixel_values"].to(args.device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(args.device)

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                teacher_vis = teacher.get_vision_features(teacher_pixels)
                teacher_out = teacher(
                    pixel_values=teacher_pixels,
                    input_ids=input_ids,
                    vision_embeds=teacher_vis,
                )
                teacher_dict = {
                    "pred_masks": teacher_out.pred_masks.float(),
                    "pred_boxes": teacher_out.pred_boxes.float(),
                    "pred_logits": teacher_out.pred_logits.float(),
                    "presence_logits": teacher_out.presence_logits.float(),
                    "semantic_seg": teacher_out.semantic_seg.float(),
                }

            # Student forward with intermediates (includes fake quant noise)
            student_out = student.forward_with_intermediates(student_pixels, input_ids)

            # Greedy matching per sample
            batch_size = student_pixels.shape[0]
            all_s_idx, all_t_idx = [], []
            for b in range(batch_size):
                s_idx, t_idx = matcher.match(
                    student_out["pred_masks"][b].detach(),
                    teacher_dict["pred_masks"][b],
                    student_out["pred_boxes"][b].detach(),
                    teacher_dict["pred_boxes"][b],
                    student_out["pred_logits"][b].detach(),
                    teacher_dict["pred_logits"][b],
                )
                all_s_idx.append(s_idx)
                all_t_idx.append(t_idx)

            # Compute loss (Phase 2 = output-only, no feature alignment losses)
            losses = loss_fn(student_out, teacher_dict, all_s_idx, all_t_idx, phase=2)
            loss = losses["total_loss"] / args.grad_accum

            loss.backward()
            epoch_loss += losses["total_loss"].item()
            step_in_epoch += 1

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / step_in_epoch
                    lr = scheduler.get_last_lr()[0]
                    print(f"  [Epoch {epoch+1}/{args.epochs}] "
                          f"Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e}")

        if debug_max_steps and global_step >= debug_max_steps:
            break

        print(f"  Epoch {epoch+1} complete. Avg loss: {epoch_loss / max(step_in_epoch, 1):.4f}")

    # ── Convert: remove fake quant, apply real quantization ──
    print("\n  Converting fake quant to real quantization...")
    student = student.cpu()
    convert_config = QATConfig(base_config, step="convert")
    quantize_(student, convert_config, filter_fn=should_quantize)

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"qat_{args.mode}.pt")
    torch.save({
        "model_state_dict": student.state_dict(),
        "quantization_mode": f"qat_{args.mode}",
        "skip_modules": list(SKIP_MODULES),
        "epochs": args.epochs,
        "lr": args.lr,
    }, output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    # ── Final sanity check ──
    print("\n  Verifying output integrity...")
    student = student.to(args.device)
    student.eval()
    with torch.no_grad():
        dummy_pixels = torch.randn(1, 3, 504, 504, device=args.device, dtype=torch.float16)
        dummy_tokens = tokenizer("segment everything", return_tensors="pt",
                                 padding="max_length", max_length=16, truncation=True)
        dummy_ids = dummy_tokens["input_ids"].to(args.device)
        out = student(dummy_pixels, dummy_ids)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                finite = torch.isfinite(val).all().item()
                print(f"    {key}: shape={list(val.shape)}, finite={finite}")

    print("\nQAT training complete.")
    print(f"To assess the QAT model, run:")
    print(f"  python quantize_model.py --checkpoint {output_path} --mode {args.mode}")
    print("\nDone.")


if __name__ == "__main__":
    main()
