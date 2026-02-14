"""EfficientSAM3 Video Distillation — CLI Entry Point.

Trains Perceiver Resampler + MemoryCrossAttention on SA-V videos.
Requires: (1) image distillation checkpoint, (2) cached teacher features.

Usage:
    python train_video_distill.py                                   # Auto-find checkpoint
    python train_video_distill.py --student-ckpt path/to/ckpt.pt   # Specific checkpoint
    python train_video_distill.py --resume path/to/video_ckpt.pt   # Resume video training
    python train_video_distill.py --debug                           # Debug run (5 steps)
"""

import argparse
import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_device(requested: str) -> str:
    """Resolve runtime device with safe fallback."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  WARN: MPS is not available in this runtime. Falling back to CPU.")
        return "cpu"
    return requested


def find_latest_image_checkpoint(ckpt_dir: str = "checkpoints/distillation") -> str:
    """Find the latest image distillation checkpoint (prefer phase2, then phase1)."""
    if not os.path.exists(ckpt_dir):
        return None

    # Prefer phase2 checkpoints
    for prefix in ("phase2_", "phase1_"):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith(prefix)])
        if ckpts:
            return os.path.join(ckpt_dir, ckpts[-1])

    return None


def main():
    parser = argparse.ArgumentParser(description="EfficientSAM3 Video Distillation")
    parser.add_argument("--student-ckpt", type=str, default=None,
                        help="Path to image distillation checkpoint (auto-detected if not given)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to video distillation checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug run: 5 steps, frequent logging")
    parser.add_argument("--device", type=str, default="mps")

    # Config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--sa-v-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Force stop after this many global steps")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Dataloader workers (0 recommended in restricted environments)")

    args = parser.parse_args()
    args.device = resolve_device(args.device)

    # -- Setup --
    print("=" * 60)
    print("  EfficientSAM3 Video Distillation")
    print("=" * 60)
    print("  Device: {}".format(args.device))
    print("  PyTorch: {}".format(torch.__version__))
    print("  MPS available: {}".format(torch.backends.mps.is_available()))
    print()

    # -- Config --
    from distillation.video_config import VideoDistillationConfig
    config = VideoDistillationConfig()

    if args.lr:
        config.lr = args.lr
    if args.epochs:
        config.epochs = args.epochs
    if args.grad_accum:
        config.grad_accum_steps = args.grad_accum
    if args.context_frames:
        config.context_frames = args.context_frames
    if args.sa_v_dir:
        config.sa_v_dir = args.sa_v_dir
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Debug mode overrides
    debug_max_steps = None
    if args.debug:
        print("  ** DEBUG MODE: 5 steps, frequent logging **")
        config.epochs = 1
        config.grad_accum_steps = 1
        config.log_every_n_steps = 1
        config.save_every_n_steps = 999999
        debug_max_steps = 5
    max_steps = args.max_steps if args.max_steps is not None else debug_max_steps

    # -- Load Student --
    print("[1/2] Loading student model...")
    from models import EfficientSAM3, EfficientSAM3Config

    student_config = EfficientSAM3Config()
    student = EfficientSAM3(student_config)

    # Load image distillation weights
    ckpt_path = args.student_ckpt or find_latest_image_checkpoint()
    if ckpt_path and os.path.exists(ckpt_path):
        print("  Loading image distillation checkpoint: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # strict=False: memory_cross_attn is new and won't exist in image checkpoints
        missing, unexpected = student.load_state_dict(
            ckpt["student_state_dict"], strict=False,
        )
        if missing:
            print("  New modules (expected): {}".format(
                [k.split(".")[0] for k in missing[:5]]))
        if unexpected:
            print("  Unexpected keys: {}".format(unexpected[:5]))
    else:
        print("  WARNING: No image distillation checkpoint found!")
        print("  Training from scratch (not recommended)")

    total_params = sum(p.numel() for p in student.parameters()) / 1e6
    print("  Student: {:.1f}M params".format(total_params))

    # -- Tokenizer --
    print("[2/2] Loading tokenizer...")
    tokenizer = student.text_encoder.tokenizer

    # -- Trainer --
    print("Initializing video trainer...")
    from distillation.video_trainer import VideoDistillationTrainer

    trainer = VideoDistillationTrainer(
        student_model=student,
        student_tokenizer=tokenizer,
        config=config,
        device=args.device,
    )

    # Resume from video checkpoint
    resume_step = 0
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        resume_step = ckpt.get("global_step", 0)

    # -- Run --
    trainer.train(resume_step=resume_step, max_steps=max_steps)

    print("Done.")


if __name__ == "__main__":
    main()
