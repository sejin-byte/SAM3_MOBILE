#!/usr/bin/env python3
"""Merge Phase2 student checkpoint with video-distilled memory modules (CPU-only).

Inputs:
- image ckpt: checkpoints/distillation/phase2_*.pt (contains student_state_dict)
- video ckpt: checkpoints/video_distillation/video_*.pt

Output:
- merged checkpoint with full `student_state_dict` suitable for quantize_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Ensure repo root is importable when this script is executed as a file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge image ckpt + video modules")
    parser.add_argument("--image-ckpt", required=True, help="Phase2 image checkpoint path")
    parser.add_argument("--video-ckpt", required=True, help="Video distillation checkpoint path")
    parser.add_argument("--output", required=True, help="Output merged checkpoint path")
    args = parser.parse_args()

    from models import EfficientSAM3, EfficientSAM3Config

    image_ckpt = torch.load(args.image_ckpt, map_location="cpu", weights_only=False)
    video_ckpt = torch.load(args.video_ckpt, map_location="cpu", weights_only=False)

    if "student_state_dict" not in image_ckpt:
        raise SystemExit("image ckpt must contain student_state_dict")
    if "perceiver_resampler_state_dict" not in video_ckpt:
        raise SystemExit("video ckpt missing perceiver_resampler_state_dict")
    if "memory_cross_attn_state_dict" not in video_ckpt:
        raise SystemExit("video ckpt missing memory_cross_attn_state_dict")

    model = EfficientSAM3(EfficientSAM3Config())
    model.load_state_dict(image_ckpt["student_state_dict"], strict=False)
    model.perceiver_resampler.load_state_dict(video_ckpt["perceiver_resampler_state_dict"], strict=True)
    model.memory_cross_attn.load_state_dict(video_ckpt["memory_cross_attn_state_dict"], strict=True)

    merged = dict(image_ckpt)
    merged["student_state_dict"] = model.state_dict()
    merged["merged_from"] = {
        "image_ckpt": str(Path(args.image_ckpt)),
        "video_ckpt": str(Path(args.video_ckpt)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)

    print(f"saved={out}")
    print(f"size_mb={out.stat().st_size / (1024 * 1024):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
