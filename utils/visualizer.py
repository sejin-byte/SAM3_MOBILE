from __future__ import annotations

import sys
import os
from typing import Tuple

import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")  # headless-friendly backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if not torch.is_tensor(t):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(t)}")
    return t.detach().cpu().numpy()


def _squeeze_to_hw(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a mask-like tensor/array to HxW float32 numpy array."""
    m = _as_numpy(mask).astype(np.float32)
    # Common shapes: (H,W), (1,H,W), (H,W,1), (B,1,H,W), (B,H,W)
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        # Prefer channel-first (1,H,W) -> (H,W)
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            # If it's multi-channel, take the first channel.
            m = m[0]
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeezing; got shape {m.shape}")
    return m


def _to_rgb_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert an image-like tensor/array to HxWx3 float32 numpy array in [0,1].

    Accepts common shapes: (C,H,W), (H,W,C), (1,H,W), (H,W), (B,C,H,W), (B,H,W,C).
    """
    x = _as_numpy(image).astype(np.float32)

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    elif x.ndim == 3:
        # If channel-first, move to channel-last.
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        # If grayscale channel-last.
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if x.shape[-1] != 3:
            # If still not RGB, take first 3 channels (or repeat first channel).
            if x.shape[-1] > 3:
                x = x[..., :3]
            else:
                x = np.repeat(x[..., :1], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape {x.shape}")

    # Heuristic normalization to [0,1]
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if np.isfinite(x_min) and np.isfinite(x_max):
        if x_max <= 1.0 and x_min >= 0.0:
            pass
        elif x_max <= 255.0 and x_min >= 0.0:
            x = x / 255.0
        else:
            denom = (x_max - x_min) if (x_max - x_min) > 1e-8 else 1.0
            x = (x - x_min) / denom

    x = np.clip(x, 0.0, 1.0)
    return x


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        return np.zeros_like(x, dtype=np.float32)
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _sigmoid01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mask_strength(mask: np.ndarray) -> np.ndarray:
    """
    Convert a raw mask (binary/probability/logits) into a [0,1] strength map.

    This is more appropriate than pure min-max normalization for overlays, and
    handles constant masks (all-zeros/all-ones) sensibly.
    """
    m = mask.astype(np.float32)
    if m.size == 0:
        return m

    m_min = float(np.nanmin(m))
    m_max = float(np.nanmax(m))
    if not (np.isfinite(m_min) and np.isfinite(m_max)):
        return np.zeros_like(m, dtype=np.float32)

    # Common cases: probabilities in [0,1] or uint8-like [0,255].
    if m_min >= 0.0 and m_max <= 1.0:
        return np.clip(m, 0.0, 1.0)
    if m_min >= 0.0 and m_max <= 255.0:
        return np.clip(m / 255.0, 0.0, 1.0)

    # Constant masks: show "on" if positive, otherwise "off".
    if abs(m_max - m_min) < 1e-8:
        return np.ones_like(m, dtype=np.float32) if m_max > 0.0 else np.zeros_like(m, dtype=np.float32)

    # Logits-ish (both negative and positive): sigmoid makes sense.
    if m_min < 0.0 and m_max > 0.0:
        return _sigmoid01(m)

    # Fallback: min-max scale.
    return _normalize_01(m)


def _resize_mask_hw(mask_hw: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a HxW mask to target (H,W) using bilinear interpolation."""
    if mask_hw.shape == target_hw:
        return mask_hw.astype(np.float32)
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None, ...]  # 1x1xH xW
    m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
    return m[0, 0].detach().cpu().numpy()


def plot_mask_comparison(
    original_image: torch.Tensor,
    teacher_mask: torch.Tensor,
    student_mask: torch.Tensor,
    epoch: int,
    save_dir: str,
    *,
    alpha: float = 0.45,
    dpi: int = 200,
) -> str:
    """
    Overlay teacher/student masks on the original image and save as:
      {save_dir}/epoch_{epoch}_mask.png

    Teacher mask: green, Student mask: red.
    """
    _ensure_dir(save_dir)

    img = _to_rgb_image(original_image)
    tmask = _squeeze_to_hw(teacher_mask)
    smask = _squeeze_to_hw(student_mask)

    img_hw = (img.shape[0], img.shape[1])
    tmask = _resize_mask_hw(tmask, img_hw)
    smask = _resize_mask_hw(smask, img_hw)

    t = _mask_strength(tmask)
    s = _mask_strength(smask)

    # RGBA overlays (alpha scaled by mask intensity)
    t_rgba = np.zeros((t.shape[0], t.shape[1], 4), dtype=np.float32)
    t_rgba[..., 1] = 1.0  # green
    t_rgba[..., 3] = np.clip(alpha * t, 0.0, 1.0)

    s_rgba = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.float32)
    s_rgba[..., 0] = 1.0  # red
    s_rgba[..., 3] = np.clip(alpha * s, 0.0, 1.0)

    out_path = os.path.join(save_dir, f"epoch_{epoch}_mask.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(t_rgba)
    ax.imshow(s_rgba)
    ax.set_title(f"Mask Overlay (epoch {epoch})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return out_path


def plot_feature_heatmap(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
    epoch: int,
    save_dir: str,
    *,
    cmap: str = "inferno",
    dpi: int = 200,
) -> str:
    """
    Compute |teacher_features - student_features| and visualize as a heatmap:
      {save_dir}/epoch_{epoch}_heatmap.png

    Expected feature shapes: (B,C,H,W) or (C,H,W) or (H,W).
    If channels exist, the heatmap is channel-mean over |diff|.
    """
    _ensure_dir(save_dir)

    t = teacher_features
    s = student_features
    if torch.is_tensor(t) and torch.is_tensor(s) and t.shape != s.shape:
        raise ValueError(f"teacher_features shape {tuple(t.shape)} != student_features shape {tuple(s.shape)}")

    diff = _as_numpy(t - s).astype(np.float32)
    diff = np.abs(diff)

    # Reduce to HxW.
    if diff.ndim == 4:
        diff = diff[0]  # C,H,W
    if diff.ndim == 3:
        # Assume channel-first: (C,H,W) -> (H,W)
        diff = diff.mean(axis=0)
    if diff.ndim != 2:
        raise ValueError(f"Unsupported diff shape after reduction: {diff.shape}")

    heat = _normalize_01(diff)
    out_path = os.path.join(save_dir, f"epoch_{epoch}_heatmap.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, cmap=cmap)
    ax.set_title(f"Feature |diff| Heatmap (epoch {epoch})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return out_path
