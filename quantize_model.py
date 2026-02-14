"""EfficientSAM3 Post-Training Quantization (PTQ) — Stage 4.

Applies TorchAO quantization to a distilled EfficientSAM3 checkpoint,
assesses accuracy on SA-1B validation split, and saves quantized weights.

Usage:
    python quantize_model.py --mode int4                        # Int4 weight-only
    python quantize_model.py --mode int8_int4                   # Int8 dynamic act + Int4 weight
    python quantize_model.py --mode compare                     # Compare FP16 / Int4 / Int8+Int4
    python quantize_model.py --checkpoint path/to/ckpt.pt       # Specific checkpoint
    python quantize_model.py --mode int4 --skip-assessment      # Quantize without assessment
"""

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from models import EfficientSAM3, EfficientSAM3Config
from distillation.greedy_matcher import GreedyMatcher

# ── Sensitive modules to keep in FP16 ──
# These are small but precision-critical; quantizing them degrades quality
# disproportionately relative to the minimal size savings.
SKIP_MODULES = {
    "iou_head",               # mask quality scoring — small MLP, quantization-sensitive
    "dot_product_scoring",    # text-vision matching precision
    "perceiver_resampler",    # video memory (small, FP16 cost is negligible)
    "memory_cross_attn",      # video gated attention (small, precision matters)
}


def should_quantize(module: torch.nn.Module, fqn: str) -> bool:
    """Filter function for TorchAO quantize_(): returns True for modules to quantize."""
    if not isinstance(module, torch.nn.Linear):
        return False
    for skip in SKIP_MODULES:
        if skip in fqn:
            return False
    return True


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Compute model size in MB from parameter storage."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nbytes
    for b in model.buffers():
        total_bytes += b.nbytes
    return total_bytes / (1024 * 1024)


def get_state_dict_size_mb(path: str) -> float:
    """Get file size of saved state dict in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def find_latest_checkpoint(ckpt_dir: str = "checkpoints/distillation") -> str:
    """Find the latest distillation checkpoint (prefer Phase 2 over Phase 1)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    for prefix in ["phase2_", "phase1_"]:
        ckpts = sorted([f for f in ckpt_dir.iterdir() if f.name.startswith(prefix)])
        if ckpts:
            return str(ckpts[-1])

    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def load_sam3_processor(model_name: str):
    """Load Sam3Processor with online-first, local-cache fallback."""
    from transformers import Sam3Processor

    try:
        return Sam3Processor.from_pretrained(model_name)
    except Exception as e:
        print(f"  WARN: processor online load failed: {e}")
        print("  Retrying with local_files_only=True ...")
        return Sam3Processor.from_pretrained(model_name, local_files_only=True)


def _coerce_square_image_size(size_obj, fallback: int) -> int:
    """Extract a square image size from common HF image_processor size formats."""
    if isinstance(size_obj, int):
        return int(size_obj)
    if isinstance(size_obj, dict):
        for key in ("height", "width", "longest_edge", "shortest_edge"):
            if key in size_obj and size_obj[key]:
                return int(size_obj[key])
    return int(fallback)


def resolve_teacher_image_size(
    teacher_model_name: str,
    processor=None,
    fallback: int = 1008,
) -> int:
    """Resolve SAM3 teacher input size from processor/config with safe fallback."""
    if processor is not None and hasattr(processor, "image_processor"):
        size = getattr(processor.image_processor, "size", None)
        resolved = _coerce_square_image_size(size, fallback)
        if resolved > 0:
            return resolved

    try:
        from transformers import Sam3Config
        cfg = Sam3Config.from_pretrained(teacher_model_name, local_files_only=True)
        for obj in (cfg, getattr(cfg, "vision_config", None)):
            if obj is None:
                continue
            image_size = getattr(obj, "image_size", None)
            if image_size:
                return int(image_size)
    except Exception:
        pass

    return int(fallback)


def infer_dtype_for_device(device: str) -> torch.dtype:
    """Use fp16 on MPS, fp32 elsewhere for better compatibility."""
    return torch.float16 if device == "mps" else torch.float32


def infer_model_float_dtype(model: torch.nn.Module, device: str) -> torch.dtype:
    """Infer an input dtype compatible with model floating-point weights/buffers."""
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    for b in model.buffers():
        if b.is_floating_point():
            return b.dtype
    return infer_dtype_for_device(device)


def load_student(checkpoint_path: str, device: str = "cpu") -> EfficientSAM3:
    """Load student model from distillation checkpoint."""
    config = EfficientSAM3Config()
    model = EfficientSAM3(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("student_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    dtype = infer_dtype_for_device(device)
    model = model.to(device).to(dtype)
    model.set_default_dtype = dtype
    return model


def apply_quantization(model: EfficientSAM3, mode: str) -> EfficientSAM3:
    """Apply TorchAO PTQ quantization in-place.

    Args:
        model: EfficientSAM3 in FP16
        mode: "int4" or "int8_int4"

    Returns:
        Quantized model (same object, modified in-place)
    """
    from torchao.quantization import quantize_, Int4WeightOnlyConfig

    # Move to CPU float32 for quantization (TorchAO requirement)
    model = model.cpu().float()

    if mode == "int4":
        config = Int4WeightOnlyConfig(group_size=128)
        print(f"  Applying Int4 weight-only quantization (group_size=128)...")
    elif mode == "int8_int4":
        from torchao.quantization import Int8DynamicActivationInt4WeightConfig
        config = Int8DynamicActivationInt4WeightConfig()
        print(f"  Applying Int8 dynamic activation + Int4 weight quantization...")
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    quantize_(model, config, filter_fn=should_quantize)
    return model


# ── Assessment Dataset ──

class SA1BAssessmentDataset:
    """Lightweight SA-1B dataset for quantization validation.

    Uses the last `num_samples` images from SA-1B as a held-out validation set.
    Returns student-preprocessed images and GT annotations for mIoU computation.
    """

    def __init__(
        self,
        sa1b_dir: str = "data/sa1b",
        num_samples: int = 1000,
        image_size: int = 504,
        teacher_image_size: int | None = None,
    ):
        sa1b_dir = Path(sa1b_dir)
        all_jsons = sorted(sa1b_dir.glob("sa_*.json"))

        # Take last num_samples as validation
        self.json_files = all_jsons[-num_samples:]
        self.image_size = image_size
        self.teacher_image_size = teacher_image_size or image_size

        # Student preprocessing (ImageNet normalization)
        self.student_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # Teacher preprocessing (SAM3 style)
        self.teacher_transform = transforms.Compose([
            transforms.Resize((self.teacher_image_size, self.teacher_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        img_path = json_path.with_suffix(".jpg")

        with open(json_path, "r") as f:
            data = json.load(f)

        image = Image.open(img_path).convert("RGB")
        student_pixel_values = self.student_transform(image)
        teacher_pixel_values = self.teacher_transform(image)

        img_w = data["image"]["width"]
        img_h = data["image"]["height"]
        annotations = data.get("annotations", [])

        # GT boxes (normalized to [0, 1])
        gt_boxes = []
        for ann in annotations:
            if "bbox" in ann and ann["bbox"]:
                x, y, w, h = ann["bbox"]
                gt_boxes.append([
                    x / img_w, y / img_h,
                    (x + w) / img_w, (y + h) / img_h,
                ])

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 4)

        # Decode RLE masks
        gt_masks = []
        for ann in annotations:
            seg = ann.get("segmentation")
            if seg and "counts" in seg and "size" in seg:
                mask = rle_decode(seg["counts"], seg["size"])
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                mask_t = F.interpolate(mask_t, size=(self.image_size, self.image_size),
                                       mode="nearest").squeeze()
                gt_masks.append(mask_t)

        if gt_masks:
            gt_masks = torch.stack(gt_masks)
        else:
            gt_masks = torch.zeros(0, self.image_size, self.image_size)

        return {
            # Keep pixel_values alias for backward compatibility (student input).
            "pixel_values": student_pixel_values,
            "student_pixel_values": student_pixel_values,
            "teacher_pixel_values": teacher_pixel_values,
            "gt_masks": gt_masks,
            "gt_boxes": gt_boxes,
            "num_annotations": len(annotations),
        }


def rle_decode(counts, size):
    """Decode COCO-style RLE (uncompressed counts list) to binary mask.

    Args:
        counts: list of int run-length counts, or a string (COCO compressed RLE)
        size: [height, width]

    Returns:
        numpy array [height, width] binary mask
    """
    import numpy as np

    h, w = size
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
            rle = {"counts": counts, "size": size}
            return mask_utils.decode(rle)
        except ImportError:
            return np.zeros((h, w), dtype=np.uint8)

    # Uncompressed RLE: alternating 0/1 run lengths
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        mask[pos:pos + c] = val
        pos += c
        val = 1 - val
    return mask.reshape((h, w), order="F")  # column-major (COCO format)


def assessment_collate_fn(batch):
    """Collate for assessment — handles variable-size GT masks/boxes."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    student_pixel_values = torch.stack([b["student_pixel_values"] for b in batch])
    teacher_pixel_values = torch.stack([b["teacher_pixel_values"] for b in batch])
    gt_masks = [b["gt_masks"] for b in batch]
    gt_boxes = [b["gt_boxes"] for b in batch]
    num_annotations = [b["num_annotations"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "student_pixel_values": student_pixel_values,
        "teacher_pixel_values": teacher_pixel_values,
        "gt_masks": gt_masks,
        "gt_boxes": gt_boxes,
        "num_annotations": num_annotations,
    }


def load_teacher_model(model_name: str, device: str):
    """Load SAM3 teacher model with online-first, local fallback."""
    from transformers import Sam3Model

    dtype = infer_dtype_for_device(device)
    try:
        model = Sam3Model.from_pretrained(model_name, torch_dtype=dtype)
    except Exception as e:
        print(f"  WARN: teacher online load failed: {e}")
        print("  Retrying teacher load with local_files_only=True ...")
        model = Sam3Model.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_teacher_input_ids(teacher_tokenizer, text: str, batch_size: int, device: str,
                             max_length: int = 16) -> torch.Tensor:
    tokens = teacher_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    if isinstance(tokens, dict):
        input_ids = tokens["input_ids"]
    else:
        input_ids = tokens.input_ids
    return input_ids.to(device).expand(batch_size, -1)


def _build_student_input_ids(student_tokenizer, text: str, batch_size: int,
                             device: str, context_length: int = 77) -> torch.Tensor:
    try:
        tokens = student_tokenizer([text], context_length=context_length)
    except TypeError:
        tokens = student_tokenizer([text])

    if isinstance(tokens, torch.Tensor):
        input_ids = tokens
    elif isinstance(tokens, dict):
        input_ids = tokens["input_ids"]
    elif hasattr(tokens, "input_ids"):
        input_ids = tokens.input_ids
    else:
        input_ids = torch.as_tensor(tokens)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device).expand(batch_size, -1)


def _build_input_ids_for_model(
    teacher_tokenizer,
    student_tokenizer,
    text: str,
    batch_size: int,
    device: str,
    model_kind: str,
) -> torch.Tensor:
    if model_kind == "teacher":
        return _build_teacher_input_ids(teacher_tokenizer, text, batch_size, device)
    return _build_student_input_ids(student_tokenizer, text, batch_size, device)


@torch.no_grad()
def _forward_outputs(model, pixel_values: torch.Tensor, input_ids: torch.Tensor, model_kind: str) -> dict:
    """Run a forward pass and return unified dict keys for assessment."""
    if model_kind == "teacher":
        vision_out = model.get_vision_features(pixel_values)
        out = model(vision_embeds=vision_out, input_ids=input_ids)
        return {
            "pred_masks": out.pred_masks,
            "pred_boxes": out.pred_boxes,
            "pred_logits": out.pred_logits,
            "presence_logits": out.presence_logits,
            "semantic_seg": out.semantic_seg,
        }
    return model(pixel_values, input_ids)


@torch.no_grad()
def assess_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    teacher_tokenizer,
    student_tokenizer,
    device: str = "mps",
    max_batches: int = None,
    model_kind: str = "student",
    eval_prompt: str = "segment everything",
    prompt_sensitivity_prompts: list[str] | None = None,
    prompt_sensitivity_batches: int = 0,
) -> dict:
    """Assess model on SA-1B validation split.

    Metrics:
        - mIoU: mean Intersection-over-Union between matched pred/GT masks
        - presence_f1: F1 score for presence detection
        - avg_inference_ms: average forward pass time per image

    Returns:
        dict with metric values
    """
    model_was_training = model.training
    model.eval()
    matcher = GreedyMatcher()

    total_iou = 0.0
    total_matched = 0
    total_presence_tp = 0
    total_presence_fp = 0
    total_presence_fn = 0
    total_presence_tn = 0
    total_presence_samples = 0
    sensitivity_sum = 0.0
    sensitivity_count = 0
    inference_times = []
    prompts = prompt_sensitivity_prompts or []
    if prompt_sensitivity_batches > 0 and len(prompts) >= 2:
        prompts = list(dict.fromkeys(prompts))

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        model_dtype = infer_model_float_dtype(model, device)
        pixel_key = "teacher_pixel_values" if model_kind == "teacher" else "student_pixel_values"
        pixel_values = batch[pixel_key].to(device, dtype=model_dtype)
        gt_masks_list = batch["gt_masks"]
        gt_boxes_list = batch["gt_boxes"]
        batch_input_ids = _build_input_ids_for_model(
            teacher_tokenizer,
            student_tokenizer,
            eval_prompt,
            batch_size=pixel_values.shape[0],
            device=device,
            model_kind=model_kind,
        )

        start = time.perf_counter()
        outputs = _forward_outputs(model, pixel_values, batch_input_ids, model_kind=model_kind)
        if device == "mps":
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        inference_times.append(elapsed_ms / pixel_values.shape[0])

        # Prompt sensitivity: mean pairwise mask-map delta across prompts.
        if prompt_sensitivity_batches > 0 and batch_idx < prompt_sensitivity_batches and len(prompts) >= 2:
            prompt_segs = []
            for p in prompts:
                p_ids = _build_input_ids_for_model(
                    teacher_tokenizer,
                    student_tokenizer,
                    p,
                    batch_size=pixel_values.shape[0],
                    device=device,
                    model_kind=model_kind,
                )
                p_out = _forward_outputs(model, pixel_values, p_ids, model_kind=model_kind)
                prompt_segs.append(p_out["semantic_seg"].float().sigmoid())
            for i, j in combinations(range(len(prompt_segs)), 2):
                sensitivity_sum += (prompt_segs[i] - prompt_segs[j]).abs().mean().item()
                sensitivity_count += 1

        for b in range(pixel_values.shape[0]):
            pred_masks = outputs["pred_masks"][b].float()
            pred_boxes = outputs["pred_boxes"][b].float()
            pred_logits = outputs["pred_logits"][b].float()
            presence = outputs["presence_logits"][b].float()

            gt_masks = gt_masks_list[b].to(device).float()
            gt_boxes = gt_boxes_list[b].to(device).float()
            gt_present = gt_masks.shape[0] > 0
            pred_present = bool((presence.sigmoid() > 0.5).item())

            total_presence_samples += 1
            if pred_present and gt_present:
                total_presence_tp += 1
            elif pred_present and not gt_present:
                total_presence_fp += 1
            elif not pred_present and gt_present:
                total_presence_fn += 1
            else:
                total_presence_tn += 1

            if gt_masks.shape[0] == 0:
                continue

            num_pred = pred_masks.shape[0]
            num_gt = gt_masks.shape[0]
            gt_logits = torch.ones(num_gt, device=device)

            cost = matcher.compute_cost_matrix(
                pred_masks, gt_masks,
                pred_boxes, gt_boxes,
                pred_logits, gt_logits,
            )
            s_idx, t_idx = matcher.greedy_assign(cost, num_matches=min(num_pred, num_gt))

            matched_pred = pred_masks[s_idx].sigmoid()
            matched_gt = gt_masks[t_idx]

            if matched_pred.shape[-2:] != matched_gt.shape[-2:]:
                matched_pred = F.interpolate(
                    matched_pred.unsqueeze(0),
                    size=matched_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            pred_binary = (matched_pred > 0.5).float()
            gt_binary = (matched_gt > 0.5).float()
            intersection = (pred_binary * gt_binary).flatten(1).sum(dim=1)
            union = ((pred_binary + gt_binary) > 0).float().flatten(1).sum(dim=1)
            iou = intersection / (union + 1e-6)

            total_iou += iou.sum().item()
            total_matched += iou.shape[0]

        if (batch_idx + 1) % 20 == 0:
            print(f"    Assessed {batch_idx + 1}/{len(dataloader)} batches...")

    if model_was_training:
        model.train()

    miou = total_iou / max(total_matched, 1)
    presence_precision = total_presence_tp / max(total_presence_tp + total_presence_fp, 1)
    presence_recall = total_presence_tp / max(total_presence_tp + total_presence_fn, 1)
    presence_f1 = (2 * presence_precision * presence_recall /
                   max(presence_precision + presence_recall, 1e-6))
    avg_ms = sum(inference_times) / max(len(inference_times), 1)
    prompt_sensitivity = sensitivity_sum / max(sensitivity_count, 1)

    return {
        "mIoU": miou,
        "presence_f1": presence_f1,
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "presence_tp": total_presence_tp,
        "presence_fp": total_presence_fp,
        "presence_fn": total_presence_fn,
        "presence_tn": total_presence_tn,
        "presence_samples": total_presence_samples,
        "prompt_sensitivity": prompt_sensitivity,
        "avg_inference_ms": avg_ms,
        "num_samples_assessed": total_matched,
    }


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    teacher_tokenizer,
    student_tokenizer,
    device: str,
    num_runs: int = 100,
    model_kind: str = "student",
    eval_prompt: str = "segment everything",
    student_image_size: int = 504,
    teacher_image_size: int = 1008,
) -> float:
    """Benchmark single-image inference time (ms), averaged over num_runs."""
    model_was_training = model.training
    model.eval()

    model_dtype = infer_model_float_dtype(model, device)
    image_size = teacher_image_size if model_kind == "teacher" else student_image_size
    pixel_values = torch.randn(1, 3, image_size, image_size, device=device, dtype=model_dtype)
    input_ids = _build_input_ids_for_model(
        teacher_tokenizer,
        student_tokenizer,
        eval_prompt,
        batch_size=1,
        device=device,
        model_kind=model_kind,
    )

    # Warmup
    for _ in range(5):
        _forward_outputs(model, pixel_values, input_ids, model_kind=model_kind)
    if device == "mps":
        torch.mps.synchronize()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _forward_outputs(model, pixel_values, input_ids, model_kind=model_kind)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    if model_was_training:
        model.train()

    return sum(times) / len(times)


def check_outputs_finite(model: EfficientSAM3, student_tokenizer, device: str) -> bool:
    """Verify quantized model produces finite outputs."""
    model_was_training = model.training
    model.eval()
    model_dtype = infer_model_float_dtype(model, device)
    pixel_values = torch.randn(1, 3, 504, 504, device=device, dtype=model_dtype)
    input_ids = _build_student_input_ids(
        student_tokenizer,
        "segment everything",
        batch_size=1,
        device=device,
    )

    with torch.no_grad():
        out = model(pixel_values, input_ids)

    all_finite = True
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            if not torch.isfinite(val).all():
                print(f"  WARNING: {key} contains non-finite values!")
                all_finite = False
            else:
                print(f"  {key}: shape={list(val.shape)}, finite=True")

    if model_was_training:
        model.train()
    return all_finite


def save_quantized(model: EfficientSAM3, output_dir: str, mode: str, metrics: dict = None):
    """Save quantized model state dict and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"quantized_{mode}.pt"
    output_path = os.path.join(output_dir, filename)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "quantization_mode": mode,
        "skip_modules": list(SKIP_MODULES),
    }
    if metrics:
        save_dict["metrics"] = metrics

    torch.save(save_dict, output_path)
    file_size = get_state_dict_size_mb(output_path)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")
    return output_path


def run_single_mode(
    mode: str,
    checkpoint_path: str,
    teacher_tokenizer,
    student_tokenizer,
    device: str,
    assessment_dataset,
    skip_assessment: bool = False,
    output_dir: str = "checkpoints/quantized",
    eval_prompt: str = "segment everything",
    prompt_sensitivity_prompts: list[str] | None = None,
    prompt_sensitivity_batches: int = 0,
    student_image_size: int = 504,
    teacher_image_size: int = 1008,
):
    """Run quantization + assessment for a single mode."""
    print(f"\n{'='*60}")
    print(f"  Quantization Mode: {mode.upper()}")
    print(f"{'='*60}")

    # Load fresh model
    print("\n[1/5] Loading model...")
    model = load_student(checkpoint_path, device="cpu")
    fp16_size = get_model_size_mb(model)
    print(f"  FP16 model size: {fp16_size:.1f} MB")

    # Apply quantization
    print("\n[2/5] Quantizing...")
    try:
        model = apply_quantization(model, mode)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"  ERROR: quantization failed for mode={mode}: {err}")
        return {
            "mode": mode,
            "error": err,
            "fp16_size_mb": fp16_size,
        }
    quant_size = get_model_size_mb(model)
    compression = fp16_size / max(quant_size, 0.1)
    print(f"  Quantized model size: {quant_size:.1f} MB ({compression:.1f}x compression)")

    # Move to device
    model = model.to(device)
    model_dtype = infer_model_float_dtype(model, device)
    print(f"  Runtime dtype on {device}: {model_dtype}")

    # Sanity check
    print("\n[3/5] Verifying output integrity...")
    finite_ok = check_outputs_finite(model, student_tokenizer, device)
    if not finite_ok:
        print("  ERROR: Quantized model produces non-finite outputs!")
        return {"mode": mode, "error": "non-finite outputs"}

    metrics = {
        "mode": mode,
        "fp16_size_mb": fp16_size,
        "quantized_size_mb": quant_size,
        "compression_ratio": compression,
    }

    if not skip_assessment:
        # Benchmark inference speed
        print("\n[4/5] Benchmarking inference speed...")
        avg_ms = benchmark_inference(
            model,
            teacher_tokenizer,
            student_tokenizer,
            device,
            num_runs=50,
            model_kind="student",
            eval_prompt=eval_prompt,
            student_image_size=student_image_size,
            teacher_image_size=teacher_image_size,
        )
        print(f"  Average inference: {avg_ms:.1f} ms/image")
        metrics["avg_inference_ms"] = avg_ms

        # Assess accuracy
        print("\n[5/5] Assessing on SA-1B validation split...")
        loader = DataLoader(
            assessment_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=assessment_collate_fn,
        )
        assess_results = assess_model(
            model,
            loader,
            teacher_tokenizer,
            student_tokenizer,
            device,
            model_kind="student",
            eval_prompt=eval_prompt,
            prompt_sensitivity_prompts=prompt_sensitivity_prompts,
            prompt_sensitivity_batches=prompt_sensitivity_batches,
        )
        metrics.update(assess_results)

        print(f"\n  Results:")
        print(f"    mIoU:           {assess_results['mIoU']:.4f}")
        print(f"    Presence F1:    {assess_results['presence_f1']:.4f}")
        print(f"    Prompt Sens.:   {assess_results['prompt_sensitivity']:.4f}")
        print(f"    Inference:      {assess_results['avg_inference_ms']:.1f} ms")
    else:
        print("\n[4/5] Skipping assessment (--skip-assessment)")
        print("[5/5] Skipping assessment")

    # Save quantized model
    save_quantized(model, output_dir, mode, metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EfficientSAM3 PTQ Quantization (Stage 4)")
    parser.add_argument("--mode", type=str, default="int4",
                        choices=["int4", "int8_int4", "compare"],
                        help="Quantization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to distillation checkpoint (auto-detects latest)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device for inference (mps/cpu)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/quantized",
                        help="Output directory for quantized models")
    parser.add_argument("--sa1b-dir", type=str, default="data/sa1b",
                        help="SA-1B dataset directory")
    parser.add_argument("--num-val", type=int, default=200,
                        help="Number of validation images (from end of SA-1B)")
    parser.add_argument("--skip-assessment", action="store_true",
                        help="Skip assessment, only quantize and save")
    parser.add_argument("--include-teacher-baseline", action="store_true",
                        help="Include SAM3 teacher baseline in compare mode")
    parser.add_argument("--teacher-model-name", type=str, default="jetjodh/sam3",
                        help="Teacher model id for baseline evaluation")
    parser.add_argument("--eval-prompt", type=str, default="segment everything",
                        help="Prompt text for main mIoU/F1 assessment")
    parser.add_argument("--prompt-sensitivity-prompts", type=str,
                        default="segment everything,person,car,building",
                        help="Comma-separated prompts used for prompt sensitivity metric")
    parser.add_argument("--prompt-sensitivity-batches", type=int, default=10,
                        help="How many assessment batches to use for prompt sensitivity (0 disables)")

    args = parser.parse_args()

    print("=" * 60)
    print("  EfficientSAM3 Post-Training Quantization")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    sensitivity_prompts = [p.strip() for p in args.prompt_sensitivity_prompts.split(",") if p.strip()]
    if len(sensitivity_prompts) < 2:
        sensitivity_prompts = ["segment everything", "person"]

    # ── Find checkpoint ──
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_checkpoint()
    print(f"  Checkpoint: {ckpt_path}")

    # ── Tokenizers ──
    print("  Loading tokenizers...")
    processor = load_sam3_processor("jetjodh/sam3")
    teacher_tokenizer = processor.tokenizer
    import open_clip
    student_cfg = EfficientSAM3Config()
    student_tokenizer = open_clip.get_tokenizer(student_cfg.text_model_name)
    student_image_size = int(student_cfg.image_size)
    teacher_image_size = resolve_teacher_image_size(
        args.teacher_model_name,
        processor=processor,
        fallback=1008,
    )
    print(f"  Student image size: {student_image_size}")
    print(f"  Teacher image size: {teacher_image_size}")

    # ── Assessment dataset ──
    assessment_dataset = None
    if not args.skip_assessment:
        assessment_dataset = SA1BAssessmentDataset(
            sa1b_dir=args.sa1b_dir,
            num_samples=args.num_val,
            image_size=student_image_size,
            teacher_image_size=teacher_image_size,
        )
        print(f"  Validation samples: {len(assessment_dataset)}")

    # ── Run ──
    if args.mode == "compare":
        all_results = []

        # FP16 baseline
        print(f"\n{'='*60}")
        print(f"  FP16 Baseline")
        print(f"{'='*60}")
        model = load_student(ckpt_path, device=args.device)
        fp16_size = get_model_size_mb(model)

        baseline_metrics = {"mode": "fp16", "fp16_size_mb": fp16_size, "quantized_size_mb": fp16_size}

        if not args.skip_assessment:
            print("  Benchmarking inference...")
            avg_ms = benchmark_inference(
                model,
                teacher_tokenizer,
                student_tokenizer,
                args.device,
                num_runs=50,
                model_kind="student",
                eval_prompt=args.eval_prompt,
                student_image_size=student_image_size,
                teacher_image_size=teacher_image_size,
            )
            baseline_metrics["avg_inference_ms"] = avg_ms

            print("  Assessing...")
            loader = DataLoader(
                assessment_dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False, collate_fn=assessment_collate_fn,
            )
            assess_results = assess_model(
                model,
                loader,
                teacher_tokenizer,
                student_tokenizer,
                args.device,
                model_kind="student",
                eval_prompt=args.eval_prompt,
                prompt_sensitivity_prompts=sensitivity_prompts,
                prompt_sensitivity_batches=args.prompt_sensitivity_batches,
            )
            baseline_metrics.update(assess_results)

        all_results.append(baseline_metrics)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()

        # Teacher baseline (optional)
        if args.include_teacher_baseline:
            print(f"\n{'='*60}")
            print("  Teacher FP16 Baseline")
            print(f"{'='*60}")
            teacher = load_teacher_model(args.teacher_model_name, args.device)
            teacher_size = get_model_size_mb(teacher)
            teacher_metrics = {
                "mode": "teacher_fp16",
                "fp16_size_mb": teacher_size,
                "quantized_size_mb": teacher_size,
            }
            if not args.skip_assessment:
                print("  Benchmarking inference...")
                t_ms = benchmark_inference(
                    teacher,
                    teacher_tokenizer,
                    student_tokenizer,
                    args.device,
                    num_runs=20,
                    model_kind="teacher",
                    eval_prompt=args.eval_prompt,
                    student_image_size=student_image_size,
                    teacher_image_size=teacher_image_size,
                )
                teacher_metrics["avg_inference_ms"] = t_ms
                print("  Assessing...")
                loader = DataLoader(
                    assessment_dataset, batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=False, collate_fn=assessment_collate_fn,
                )
                t_assess = assess_model(
                    teacher,
                    loader,
                    teacher_tokenizer,
                    student_tokenizer,
                    args.device,
                    model_kind="teacher",
                    eval_prompt=args.eval_prompt,
                    prompt_sensitivity_prompts=sensitivity_prompts,
                    prompt_sensitivity_batches=args.prompt_sensitivity_batches,
                )
                teacher_metrics.update(t_assess)
            all_results.append(teacher_metrics)
            del teacher
            if args.device == "mps":
                torch.mps.empty_cache()

        # Int4
        int4_metrics = run_single_mode(
            "int4", ckpt_path, teacher_tokenizer, student_tokenizer, args.device,
            assessment_dataset, args.skip_assessment, args.output_dir,
            args.eval_prompt, sensitivity_prompts, args.prompt_sensitivity_batches,
            student_image_size, teacher_image_size,
        )
        all_results.append(int4_metrics)
        if args.device == "mps":
            torch.mps.empty_cache()

        # Int8+Int4
        int8_int4_metrics = run_single_mode(
            "int8_int4", ckpt_path, teacher_tokenizer, student_tokenizer, args.device,
            assessment_dataset, args.skip_assessment, args.output_dir,
            args.eval_prompt, sensitivity_prompts, args.prompt_sensitivity_batches,
            student_image_size, teacher_image_size,
        )
        all_results.append(int8_int4_metrics)

        # ── Comparison table ──
        print(f"\n{'='*60}")
        print(f"  Comparison Summary")
        print(f"{'='*60}")
        header = f"  {'Mode':<12} {'Size(MB)':>10} {'Compress':>10} {'mIoU':>8} {'F1':>8} {'P-Sens':>8} {'ms':>8}"
        print(header)
        print(f"  {'-'*74}")
        for r in all_results:
            mode = r.get("mode", "?")
            if "error" in r:
                print(f"  {mode:<12} {'-':>10} {'-':>10} {'-':>8} {'-':>8} {'-':>8} {'-':>8}  ERROR")
                print(f"    reason: {r['error']}")
                continue
            size = r.get("quantized_size_mb", 0)
            comp = r.get("compression_ratio", 1.0) if mode != "fp16" else 1.0
            miou = r.get("mIoU", 0)
            f1 = r.get("presence_f1", 0)
            ps = r.get("prompt_sensitivity", 0)
            ms = r.get("avg_inference_ms", 0)
            print(f"  {mode:<12} {size:>10.1f} {comp:>9.1f}x {miou:>8.4f} {f1:>8.4f} {ps:>8.4f} {ms:>7.1f}")

        # Check mIoU degradation
        if not args.skip_assessment and "error" not in all_results[0]:
            fp16_miou = all_results[0].get("mIoU", 0)
            for r in all_results[1:]:
                mode = r.get("mode", "?")
                if mode not in ("int4", "int8_int4"):
                    continue
                if "error" in r:
                    continue
                delta = fp16_miou - r.get("mIoU", 0)
                if delta > 0.02:
                    print(f"\n  WARNING: {mode} mIoU drop = {delta:.4f} (>{0.02})")
                    print(f"  -> Consider QAT fine-tuning (train_qat.py)")
                else:
                    print(f"\n  {mode} mIoU drop = {delta:.4f} (within 2% threshold)")

    else:
        run_single_mode(
            args.mode, ckpt_path, teacher_tokenizer, student_tokenizer, args.device,
            assessment_dataset, args.skip_assessment, args.output_dir,
            args.eval_prompt, sensitivity_prompts, args.prompt_sensitivity_batches,
            student_image_size, teacher_image_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
