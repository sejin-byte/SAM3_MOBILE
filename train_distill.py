"""EfficientSAM3 Knowledge Distillation — CLI Entry Point.

Usage:
    python train_distill.py --phase 1               # Feature alignment (5 epochs)
    python train_distill.py --phase 2               # Output refinement (10 epochs)
    python train_distill.py --phase 1 --resume PATH # Resume from checkpoint
    python train_distill.py --phase 1 --validate    # Run validation only
    python train_distill.py --phase 1 --debug       # Debug run (1 epoch, 10 steps)

    # Selective reset: load vision weights from old checkpoint, reset text pathway
    python train_distill.py --phase 1 --selective-reset checkpoints/distillation/phase2_epoch2_step24417.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_device(requested: str) -> str:
    """Resolve runtime device with safe fallback."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  WARN: MPS is not available in this runtime. Falling back to CPU.")
        return "cpu"
    return requested


def selective_reset_text_pathway(student, checkpoint_path: str):
    """Load vision/spatial weights from checkpoint, reset text-dependent modules.

    Keeps: vision_encoder, detr_encoder (vision weights), detr_decoder, mask_decoder, iou_head
    Resets: text_encoder (to MobileCLIP-S1 pretrained), text projection, dot_product_scoring
    """
    print(f"  Loading checkpoint for selective reset: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_state = ckpt["student_state_dict"]

    # 1. Load ALL weights from checkpoint first
    student.load_state_dict(old_state, strict=False)
    print(f"  Loaded {len(old_state)} parameters from checkpoint")

    # 2. Reset text encoder to fresh MobileCLIP-S1 pretrained weights
    from models.text_encoder_mobileclip import MobileCLIPTextEncoder
    from models.configuration import EfficientSAM3Config
    fresh_text_enc = MobileCLIPTextEncoder(EfficientSAM3Config())
    student.text_encoder.load_state_dict(fresh_text_enc.state_dict())
    del fresh_text_enc
    print("  Reset text_encoder -> MobileCLIP-S1 pretrained")

    # 3. Reinitialize text projection (512->256) with Xavier
    nn.init.xavier_uniform_(student.text_encoder.projection.weight)
    nn.init.zeros_(student.text_encoder.projection.bias)
    print("  Reset text_encoder.projection -> Xavier init")

    # 4. Reinitialize DotProductScoring (text_mlp, text_proj, query_proj)
    for name, module in student.dot_product_scoring.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    for name, module in student.dot_product_scoring.named_modules():
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    print("  Reset dot_product_scoring -> Xavier init")

    # Summary: count reset vs kept params
    reset_modules = ["text_encoder", "dot_product_scoring"]
    reset_params = sum(
        p.numel() for n, p in student.named_parameters()
        if any(n.startswith(m) for m in reset_modules)
    )
    kept_params = sum(p.numel() for p in student.parameters()) - reset_params
    print(f"  Kept: {kept_params/1e6:.1f}M params (vision/spatial)")
    print(f"  Reset: {reset_params/1e6:.1f}M params (text pathway)")

    return ckpt.get("global_step", 0)


def main():
    parser = argparse.ArgumentParser(description="EfficientSAM3 Knowledge Distillation")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase: 1=feature alignment, 2=output refinement")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation only")
    parser.add_argument("--debug", action="store_true",
                        help="Debug run: 1 epoch, 10 steps, vis every 5 steps")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cpu)")

    # Config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--sa1b-dir", type=str, default=None)
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--num-train", type=int, default=None,
                        help="Override number of SA-1B train samples")
    parser.add_argument("--num-val", type=int, default=None,
                        help="Override number of SA-1B val samples")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Force stop after this many global steps")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Dataloader workers (0 recommended in restricted environments)")
    parser.add_argument("--selective-reset", type=str, default=None,
                        help="Path to checkpoint for selective reset: keep vision weights, "
                             "reset text pathway to pretrained. Use with --phase 1.")
    parser.add_argument("--freeze-vision-epochs", type=int, default=1,
                        help="Freeze vision encoder for first N epochs during selective reset (default: 1)")

    args = parser.parse_args()
    args.device = resolve_device(args.device)

    # ── Setup ──
    print("=" * 60)
    print("  EfficientSAM3 Knowledge Distillation")
    print("=" * 60)
    print(f"  Phase: {args.phase}")
    print(f"  Device: {args.device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print()

    # ── Config ──
    from distillation.config import DistillationConfig
    config = DistillationConfig()

    if args.sa1b_dir:
        config.sa1b_dir = args.sa1b_dir
    if args.teacher_model:
        config.teacher_model_name = args.teacher_model
    if args.num_train is not None:
        config.num_train = args.num_train
    if args.num_val is not None:
        config.num_val = args.num_val
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.grad_accum:
        config.grad_accum_steps = args.grad_accum
    if args.lr:
        if args.phase == 1:
            config.phase1_lr = args.lr
        else:
            config.phase2_lr = args.lr
    if args.epochs:
        if args.phase == 1:
            config.phase1_epochs = args.epochs
        else:
            config.phase2_epochs = args.epochs

    # ── Debug mode overrides ──
    debug_max_steps = None
    if args.debug:
        print("  ** DEBUG MODE: 1 epoch, 10 steps, vis every 5 steps **")
        if args.phase == 1:
            config.phase1_epochs = 1
        else:
            config.phase2_epochs = 1
        config.grad_accum_steps = 2
        config.log_every_n_steps = 1
        config.vis_every_n_steps = 5
        config.save_every_n_steps = 999999  # skip checkpoints in debug
        debug_max_steps = 10
    max_steps = args.max_steps if args.max_steps is not None else debug_max_steps

    # ── Load Teacher ──
    print("[1/3] Loading teacher model...")
    from transformers import Sam3Model, Sam3Processor

    teacher_dtype = torch.float16 if args.device == "mps" else torch.float32
    try:
        processor = Sam3Processor.from_pretrained(config.teacher_model_name)
    except Exception as e:
        print(f"  WARN: processor online load failed: {e}")
        print("  Retrying with local_files_only=True ...")
        processor = Sam3Processor.from_pretrained(config.teacher_model_name, local_files_only=True)
    try:
        teacher = Sam3Model.from_pretrained(
            config.teacher_model_name,
            torch_dtype=teacher_dtype,
        )
    except Exception as e:
        print(f"  WARN: teacher online load failed: {e}")
        print("  Retrying with local_files_only=True ...")
        teacher = Sam3Model.from_pretrained(
            config.teacher_model_name,
            torch_dtype=teacher_dtype,
            local_files_only=True,
        )
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params")

    # ── Load Student ──
    print("[2/3] Loading student model...")
    from models import EfficientSAM3, EfficientSAM3Config

    student_config = EfficientSAM3Config()
    student = EfficientSAM3(student_config)

    # Selective reset: load vision weights, reset text pathway
    if args.selective_reset:
        if args.phase != 1:
            print("  WARN: --selective-reset is designed for --phase 1. Proceeding anyway.")
        selective_reset_text_pathway(student, args.selective_reset)
    # Load Phase 1 checkpoint for Phase 2
    elif args.phase == 2 and args.resume is None:
        ckpt_dir = config.checkpoint_dir
        if os.path.exists(ckpt_dir):
            phase1_ckpts = sorted([
                f for f in os.listdir(ckpt_dir) if f.startswith("phase1_")
            ])
            if phase1_ckpts:
                latest = os.path.join(ckpt_dir, phase1_ckpts[-1])
                print(f"  Auto-loading Phase 1 checkpoint: {latest}")
                ckpt = torch.load(latest, map_location="cpu", weights_only=False)
                student.load_state_dict(ckpt["student_state_dict"], strict=False)

    print(f"  Student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params")

    # ── Tokenizers (teacher HF + student open_clip) ──
    teacher_tokenizer = processor.tokenizer
    student_tokenizer = student.text_encoder.tokenizer

    # ── Trainer ──
    print("[3/3] Initializing trainer...")
    from distillation.trainer import DistillationTrainer

    freeze_vision_epochs = args.freeze_vision_epochs if args.selective_reset else 0
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        config=config,
        phase=args.phase,
        device=args.device,
        freeze_vision_epochs=freeze_vision_epochs,
    )

    # Resume from checkpoint (not used with --selective-reset)
    resume_step = 0
    if args.resume and not args.selective_reset:
        ckpt = trainer.load_checkpoint(args.resume)
        resume_step = ckpt.get("global_step", 0)

    # ── Run ──
    if args.validate:
        trainer.validate()
    else:
        trainer.train(resume_step=resume_step, max_steps=max_steps)

    print("Done.")


if __name__ == "__main__":
    main()
