#!/usr/bin/env python3
"""Export EfficientSAM3 checkpoint to ExecuTorch .pte (Stage 5).

Supports:
- plain student checkpoints (`student_state_dict`)
- quantized checkpoints (`model_state_dict` + `quantization_mode`)

Backends:
- `none`   : generic ExecuTorch export
- `coreml` : CoreML partitioning (falls back to `none` unless --strict-backend)
- `qnn`    : QNN partitioning (falls back to `none` unless --strict-backend)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Ensure repo root imports work when script is executed via file path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import EfficientSAM3, EfficientSAM3Config
from quantize_model import should_quantize


class ExportWrapper(nn.Module):
    """Convert dict outputs to tuple for stable export signatures."""

    def __init__(self, model: EfficientSAM3):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor):
        out = self.model(pixel_values, input_ids)
        return (
            out["pred_masks"],
            out["pred_boxes"],
            out["pred_logits"],
            out["presence_logits"],
            out["semantic_seg"],
            out["iou_scores"],
        )


def normalize_quant_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    mode = str(mode).strip().lower()
    if not mode:
        return None
    if mode == "none":
        return None
    if mode.startswith("qat_"):
        mode = mode[len("qat_") :]
    return mode


def apply_quantized_structure(model: EfficientSAM3, mode: str) -> None:
    """Prepare model module structure before loading quantized state dict."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    mode = normalize_quant_mode(mode)
    if mode == "int4":
        config = Int4WeightOnlyConfig(group_size=128)
    elif mode == "int8_int4":
        from torchao.quantization import Int8DynamicActivationInt4WeightConfig

        config = Int8DynamicActivationInt4WeightConfig()
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    quantize_(model, config, filter_fn=should_quantize)


def load_checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], str | None, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict: {path}")

    quant_mode = ckpt.get("quantization_mode")

    if "student_state_dict" in ckpt:
        return ckpt["student_state_dict"], quant_mode, "student_state_dict"
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], quant_mode, "model_state_dict"
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, quant_mode, "raw_state_dict"

    keys_preview = ", ".join(list(ckpt.keys())[:8])
    raise ValueError(f"Unsupported checkpoint format: {path} (keys: {keys_preview})")


def find_latest_merged_checkpoint(exclude: Path | None = None) -> Path | None:
    merged_ckpts = sorted(Path("checkpoints/final").glob("student_phase2_video_merged_*.pt"))
    if not merged_ckpts:
        return None
    merged_ckpts = list(reversed(merged_ckpts))
    if exclude is None:
        return merged_ckpts[0]
    exclude_resolved = exclude.resolve()
    for ckpt in merged_ckpts:
        if ckpt.resolve() != exclude_resolved:
            return ckpt
    return None


def build_model_from_checkpoint(path: Path, quant_mode_override: str | None) -> Tuple[ExportWrapper, dict]:
    model = EfficientSAM3(EfficientSAM3Config()).eval().cpu().float()
    state_dict, quant_mode_ckpt, state_source = load_checkpoint_state(path)

    quant_mode = normalize_quant_mode(quant_mode_override or quant_mode_ckpt)
    if quant_mode:
        print(f"Applying quantized module structure for mode={quant_mode}")
        apply_quantized_structure(model, quant_mode)
    else:
        print("Loading as non-quantized (FP) checkpoint")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARN: missing keys={len(missing)} (first: {missing[:5]})")
    if unexpected:
        print(f"WARN: unexpected keys={len(unexpected)} (first: {unexpected[:5]})")

    wrapper = ExportWrapper(model).eval().cpu()
    meta = {
        "checkpoint": str(path),
        "state_source": state_source,
        "quant_mode_used": quant_mode,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return wrapper, meta


def make_example_inputs(batch_size: int, image_size: int, seq_len: int):
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return pixel_values, input_ids


@torch.no_grad()
def check_forward_finite(model: nn.Module, example_inputs) -> None:
    outputs = model(*example_inputs)
    for idx, tensor in enumerate(outputs):
        if not torch.is_tensor(tensor):
            continue
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite output detected at tuple index {idx}")
        print(f"  out[{idx}] shape={list(tensor.shape)} finite=True")


def export_program(model: nn.Module, example_inputs, strict_export: bool):
    if strict_export:
        ep = torch.export.export(model, example_inputs)
        return ep, "export"

    try:
        ep = torch.export.export(model, example_inputs)
        return ep, "export"
    except Exception as e:
        err_summary = str(e).splitlines()[0]
        print(f"WARN: torch.export.export failed ({type(e).__name__}: {err_summary})")
        print("Falling back to torch.export.draft_export (fixed-shape deployment only).")
        if not hasattr(torch.export, "draft_export"):
            raise RuntimeError("torch.export.draft_export is unavailable in current PyTorch build") from e
        ep = torch.export.draft_export(model, example_inputs, strict=False)
        return ep, "draft_export"


def maybe_partition_backend(
    edge_program,
    backend: str,
    strict_backend: bool,
    coreml_compute_unit: str,
):
    if backend == "none":
        return edge_program, "none", None

    if backend == "coreml":
        try:
            import coremltools as ct
            from executorch.backends.apple.coreml.compiler.coreml_preprocess import CoreMLBackend
            from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner

            cu_map = {
                "all": ct.ComputeUnit.ALL,
                "cpu_only": ct.ComputeUnit.CPU_ONLY,
                "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
                "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            }
            compile_specs = CoreMLBackend.generate_compile_specs(
                compute_unit=cu_map[coreml_compute_unit],
            )
            partitioner = CoreMLPartitioner(compile_specs=compile_specs)
            return edge_program.to_backend(partitioner), "coreml", None
        except Exception as e:
            msg = f"CoreML partition failed: {type(e).__name__}: {e}"
            if strict_backend:
                raise RuntimeError(msg) from e
            print(f"WARN: {msg}")
            print("Falling back to backend=none")
            return edge_program, "none", msg

    if backend == "qnn":
        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

            partitioner = QnnPartitioner()
            return edge_program.to_backend(partitioner), "qnn", None
        except Exception as e:
            msg = f"QNN partition failed: {type(e).__name__}: {e}"
            if strict_backend:
                raise RuntimeError(msg) from e
            print(f"WARN: {msg}")
            print("Falling back to backend=none")
            return edge_program, "none", msg

    raise ValueError(f"Unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EfficientSAM3 to ExecuTorch .pte")
    parser.add_argument("--checkpoint", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output .pte path")
    parser.add_argument(
        "--backend",
        type=str,
        default="none",
        choices=["none", "coreml", "qnn"],
        help="Target backend partitioner",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default=None,
        choices=["none", "int4", "int8_int4", "qat_int4", "qat_int8_int4"],
        help="Override checkpoint quantization mode",
    )
    parser.add_argument(
        "--allow-quantized-export",
        action="store_true",
        help="Attempt direct export of TorchAO quantized checkpoints (often fails on current toolchain)",
    )
    parser.add_argument(
        "--fallback-fp-checkpoint",
        type=str,
        default="auto",
        help="Fallback FP checkpoint path when quantized export is disabled/unsupported ('auto' to pick latest merged ckpt)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Example input batch size")
    parser.add_argument("--image-size", type=int, default=504, help="Example input image size")
    parser.add_argument("--seq-len", type=int, default=16, help="Example input token length")
    parser.add_argument(
        "--coreml-compute-unit",
        type=str,
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        help="CoreML compute unit when --backend coreml",
    )
    parser.add_argument(
        "--strict-export",
        action="store_true",
        help="Disable draft_export fallback if export() fails",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="Fail instead of backend=none fallback when partitioning fails",
    )
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="Skip one-step finite forward sanity check before export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("  EfficientSAM3 ExecuTorch Export")
    print("=" * 60)
    print(f"checkpoint: {ckpt_path}")
    print(f"backend:    {args.backend}")
    print(f"output:     {out_path}")
    print(f"torch:      {torch.__version__}")
    print()

    print("[1/5] Load model/checkpoint")
    wrapper, meta = build_model_from_checkpoint(ckpt_path, args.quant_mode)
    effective_ckpt_path = ckpt_path

    if meta.get("quant_mode_used") and not args.allow_quantized_export:
        requested_quant_ckpt = effective_ckpt_path
        fallback_ckpt: Path | None
        if args.fallback_fp_checkpoint == "auto":
            fallback_ckpt = find_latest_merged_checkpoint(exclude=requested_quant_ckpt)
        else:
            fallback_ckpt = Path(args.fallback_fp_checkpoint)
            if not fallback_ckpt.exists():
                raise FileNotFoundError(f"Fallback checkpoint not found: {fallback_ckpt}")

        if fallback_ckpt is None:
            raise RuntimeError(
                "Quantized checkpoint detected, but no FP fallback checkpoint is available. "
                "Provide --fallback-fp-checkpoint <merged_ckpt> or use --allow-quantized-export."
            )

        print(
            "WARN: quantized checkpoint export is unstable in current torch/executorch "
            f"runtime. Falling back to FP checkpoint: {fallback_ckpt}"
        )
        wrapper, fallback_meta = build_model_from_checkpoint(fallback_ckpt, quant_mode_override="none")
        meta["requested_quantized_checkpoint"] = str(requested_quant_ckpt)
        meta["quantized_export_fallback"] = True
        effective_ckpt_path = fallback_ckpt
        # overwrite load metadata with the effective model metadata
        for key, value in fallback_meta.items():
            meta[key] = value

    print("[2/5] Build example inputs")
    example_inputs = make_example_inputs(
        batch_size=args.batch_size,
        image_size=args.image_size,
        seq_len=args.seq_len,
    )

    if args.skip_forward_check:
        print("[3/5] Forward sanity check skipped")
    else:
        print("[3/5] Forward finite sanity check")
        check_forward_finite(wrapper, example_inputs)

    print("[4/5] torch.export -> Edge")
    try:
        exported_program, export_method = export_program(
            wrapper,
            example_inputs,
            strict_export=args.strict_export,
        )
    except Exception as e:
        err_summary = str(e).splitlines()[0]
        raise RuntimeError(
            f"Export failed for checkpoint {effective_ckpt_path}: {type(e).__name__}: {err_summary}"
        )

    from executorch import exir

    edge_program = exir.to_edge(exported_program)
    edge_program, backend_used, backend_fallback_reason = maybe_partition_backend(
        edge_program=edge_program,
        backend=args.backend,
        strict_backend=args.strict_backend,
        coreml_compute_unit=args.coreml_compute_unit,
    )

    print("[5/5] To ExecuTorch + save")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    et_program = edge_program.to_executorch()
    et_program.save(str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    metadata = {
        "requested_checkpoint": str(ckpt_path),
        "checkpoint": str(effective_ckpt_path),
        "output": str(out_path),
        "output_size_mb": size_mb,
        "requested_backend": args.backend,
        "used_backend": backend_used,
        "backend_fallback_reason": backend_fallback_reason,
        "export_method": export_method,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seq_len": args.seq_len,
        "torch_version": torch.__version__,
    }
    metadata.update(meta)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
    print(f"Metadata: {meta_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
