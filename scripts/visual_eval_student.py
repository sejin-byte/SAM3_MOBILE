#!/usr/bin/env python3
"""Visual QA for EfficientSAM3 student checkpoints.

Generates side-by-side PNGs (original | student) and an `index.html` gallery
to quickly sanity-check segmentation quality from a user perspective.

Optional: `--compare-teacher` adds a teacher column (original | teacher | student).

Notes:
- Uses the same 504px preprocessing as distillation (see `distillation/config.py`).
- When `--compare-teacher` is enabled, uses HF `Sam3Processor` tokenizer for BOTH
  teacher and student (matching distillation training).
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure repo root is importable when this script is executed as a file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def _dtype_for_device(device: torch.device) -> torch.dtype:
    # Keep CPU in fp32 to avoid slow fp16 kernels.
    return torch.float16 if device.type == "mps" else torch.float32


def _iter_images(image_paths: List[str], image_dir: str | None) -> List[Path]:
    imgs: List[Path] = []
    for p in image_paths:
        imgs.append(Path(p))
    if image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        for p in sorted(Path(image_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(p)
    # Dedup while preserving order.
    seen = set()
    out: List[Path] = []
    for p in imgs:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _preprocess_image(
    img_path: Path,
    *,
    image_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> tuple[Image.Image, torch.Tensor]:
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0  # HWC in [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    x = (x - m) / s
    return img_resized, x


def _select_topk_indices(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """logits: [Q]"""
    q = logits.shape[0]
    k = min(max(int(top_k), 1), q)
    _, idx = logits.topk(k)
    return idx


def _make_overlay(
    img_hw: Tuple[int, int],
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    *,
    top_k: int,
    threshold: float,
) -> np.ndarray:
    """Return an RGB overlay image (H,W,3) with colored masks."""
    h, w = img_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    # Pick top-k by confidence.
    idx = _select_topk_indices(pred_logits.float(), top_k)

    for i, qidx in enumerate(idx.tolist()):
        mask = pred_masks[qidx].float().sigmoid()  # [h',w']
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask[None, None, ...], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        mask_np = (mask.detach().cpu().numpy() > float(threshold))
        color = colors[i % len(colors)]
        for c in range(3):
            canvas[:, :, c] = np.where(mask_np, color[c], canvas[:, :, c])
    return canvas


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    a = float(alpha)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return (img.astype(np.float32) * (1.0 - a) + overlay.astype(np.float32) * a).astype(np.uint8)


def _concat_h(images: Iterable[np.ndarray]) -> np.ndarray:
    images = list(images)
    if not images:
        raise ValueError("No images to concatenate")
    h = images[0].shape[0]
    for im in images:
        if im.shape[0] != h:
            raise ValueError("All images must have the same height for concatenation")
    return np.concatenate(images, axis=1)


def _load_student(student_ckpt: str, device: torch.device, dtype: torch.dtype):
    from models import EfficientSAM3, EfficientSAM3Config

    model = EfficientSAM3(EfficientSAM3Config())
    ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["student_state_dict"] if isinstance(ckpt, dict) and "student_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("WARN: missing keys in student load (showing up to 20):")
        for k in missing[:20]:
            print("  -", k)
        if len(missing) > 20:
            print("  ...")
    if unexpected:
        print("WARN: unexpected keys in student load (showing up to 20):")
        for k in unexpected[:20]:
            print("  -", k)
        if len(unexpected) > 20:
            print("  ...")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _apply_video_ckpt(student, video_ckpt: str):
    ckpt = torch.load(video_ckpt, map_location="cpu", weights_only=False)
    student.perceiver_resampler.load_state_dict(ckpt["perceiver_resampler_state_dict"])
    student.memory_cross_attn.load_state_dict(ckpt["memory_cross_attn_state_dict"])


def _load_teacher(teacher_model: str, device: torch.device):
    from transformers import Sam3Model
    from distillation.config import DistillationConfig
    from distillation.trainer import resize_teacher_rope

    cfg = DistillationConfig()
    teacher_dtype = torch.float16 if device.type == "mps" else torch.float32
    try:
        teacher = Sam3Model.from_pretrained(teacher_model, torch_dtype=teacher_dtype)
    except Exception as e:
        print(f"WARN: teacher online load failed: {e}")
        print("Retrying with local_files_only=True ...")
        teacher = Sam3Model.from_pretrained(
            teacher_model,
            torch_dtype=teacher_dtype,
            local_files_only=True,
        )
    teacher = teacher.to(device=device, dtype=teacher_dtype)
    teacher.requires_grad_(False)
    teacher.eval()

    native_size = teacher.config.vision_config.backbone_config.image_size
    if cfg.image_size != native_size:
        resize_teacher_rope(teacher, cfg.image_size)
    return teacher


def _tokenize_prompt_hf(teacher_model: str, prompt: str) -> torch.Tensor:
    from transformers import Sam3Processor
    from distillation.config import DistillationConfig

    cfg = DistillationConfig()
    try:
        processor = Sam3Processor.from_pretrained(teacher_model)
    except Exception as e:
        print(f"WARN: processor online load failed: {e}")
        print("Retrying with local_files_only=True ...")
        processor = Sam3Processor.from_pretrained(teacher_model, local_files_only=True)
    tok = processor.tokenizer
    tokens = tok(prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
    # Keep `max_length` aligned with training.
    return tokens["input_ids"]


@torch.no_grad()
def _teacher_forward(teacher, teacher_pixels: torch.Tensor, input_ids: torch.Tensor) -> dict:
    # Match distillation trainer flow (vision_embeds precompute to also enable fpn extraction).
    vision_outputs = teacher.get_vision_features(teacher_pixels)
    outputs = teacher(
        vision_embeds=vision_outputs,
        input_ids=input_ids,
        output_hidden_states=False,
    )
    return {
        "pred_masks": outputs.pred_masks,
        "pred_logits": outputs.pred_logits,
    }


@torch.no_grad()
def _student_forward(student, student_pixels: torch.Tensor, input_ids: torch.Tensor) -> dict:
    out = student(student_pixels, input_ids)
    return {
        "pred_masks": out["pred_masks"],
        "pred_logits": out["pred_logits"],
    }


def _write_index_html(out_dir: Path, rows: List[Tuple[str, str]]) -> None:
    # rows: (basename, relpath_png)
    body = []
    body.append("<!doctype html>")
    body.append("<html><head><meta charset='utf-8'/>")
    body.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    body.append("<title>EfficientSAM3 Visual QA</title>")
    body.append("<style>")
    body.append("body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; padding:16px;}")
    body.append(".grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:14px;}")
    body.append(".card{border:1px solid #ddd; border-radius:10px; padding:10px;}")
    body.append(".name{font-size:13px; color:#333; margin:0 0 8px 0; word-break:break-all;}")
    body.append("img{max-width:100%; height:auto; border-radius:6px; background:#f7f7f7;}")
    body.append("</style></head><body>")
    body.append("<h2 style='margin:0 0 10px 0'>EfficientSAM3 Visual QA</h2>")
    body.append("<div class='grid'>")
    for name, rel in rows:
        body.append("<div class='card'>")
        body.append("<div class='name'>{}</div>".format(html.escape(name)))
        body.append("<a href='{0}' target='_blank'><img src='{0}' loading='lazy'/></a>".format(
            html.escape(rel)
        ))
        body.append("</div>")
    body.append("</div></body></html>")
    (out_dir / "index.html").write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="EfficientSAM3 visual QA")
    parser.add_argument("--student-ckpt", required=True, help="Path to Phase2 checkpoint (.pt)")
    parser.add_argument("--video-ckpt", default=None, help="Optional video distillation ckpt to apply")
    parser.add_argument("--prompt", default="objects in the image", help="Text prompt")
    parser.add_argument("--image", action="append", default=[], help="Image path (repeatable)")
    parser.add_argument("--image-dir", default=None, help="Directory with images")
    parser.add_argument("--out-dir", default="outputs/visual_eval/run1", help="Output directory")
    parser.add_argument("--device", default="auto", help="auto|mps|cpu")
    parser.add_argument("--top-k", type=int, default=5, help="How many queries to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha")
    parser.add_argument("--compare-teacher", action="store_true", help="Add teacher column for reference")
    parser.add_argument("--teacher-model", default="jetjodh/sam3", help="HF teacher id/path for compare-teacher/tokenizer")
    args = parser.parse_args()

    from distillation.config import DistillationConfig
    cfg = DistillationConfig()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = _iter_images(args.image, args.image_dir)
    if not imgs:
        raise SystemExit("No images provided. Use --image or --image-dir.")

    device = _device_from_arg(args.device)
    dtype = _dtype_for_device(device)

    # Load student.
    print("Loading student...")
    student = _load_student(args.student_ckpt, device=device, dtype=dtype)
    if args.video_ckpt:
        print("Applying video ckpt...")
        _apply_video_ckpt(student, args.video_ckpt)
        print("  Note: video ckpt does NOT affect image-only forward(). Use it for forward_video().")

    # Tokenize prompt.
    if args.compare_teacher:
        input_ids = _tokenize_prompt_hf(args.teacher_model, args.prompt)
    else:
        # Student-only mode: use open_clip tokenizer (no HF downloads required).
        input_ids = student.text_encoder.tokenizer([args.prompt])

    input_ids = input_ids.to(device)

    teacher = None
    if args.compare_teacher:
        print("Loading teacher (compare mode)...")
        teacher = _load_teacher(args.teacher_model, device=device)

    rows: List[Tuple[str, str]] = []

    for img_path in imgs:
        print("Processing:", img_path)

        img_disp, student_x = _preprocess_image(
            img_path,
            image_size=cfg.image_size,
            mean=cfg.student_mean,
            std=cfg.student_std,
        )
        img_np = np.asarray(img_disp).astype(np.uint8)

        student_pixels = student_x[None, ...].to(device=device, dtype=dtype)
        s_out = _student_forward(student, student_pixels, input_ids)

        s_overlay = _make_overlay(
            (cfg.image_size, cfg.image_size),
            s_out["pred_masks"][0],
            s_out["pred_logits"][0],
            top_k=args.top_k,
            threshold=args.threshold,
        )
        s_blend = _blend(img_np, s_overlay, args.alpha)

        cols = [img_np]

        if teacher is not None:
            img_disp_t, teacher_x = _preprocess_image(
                img_path,
                image_size=cfg.image_size,
                mean=cfg.teacher_mean,
                std=cfg.teacher_std,
            )
            teacher_dtype = torch.float16 if device.type == "mps" else torch.float32
            teacher_pixels = teacher_x[None, ...].to(device=device, dtype=teacher_dtype)
            t_out = _teacher_forward(teacher, teacher_pixels, input_ids)
            t_overlay = _make_overlay(
                (cfg.image_size, cfg.image_size),
                t_out["pred_masks"][0],
                t_out["pred_logits"][0],
                top_k=args.top_k,
                threshold=args.threshold,
            )
            t_blend = _blend(img_np, t_overlay, args.alpha)
            cols.append(t_blend)

        cols.append(s_blend)

        combined = _concat_h(cols)

        out_name = img_path.stem + ".png"
        out_path = out_dir / out_name
        Image.fromarray(combined).save(out_path)
        rows.append((img_path.name, out_path.name))

    _write_index_html(out_dir, rows)
    print("\nWrote:", out_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
