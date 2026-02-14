"""Cache teacher FPN L3 features for SA-V videos.

Pre-computes SAM3 teacher vision features at 504px resolution and saves
FPN level 3 ([256, 18, 18]) per annotated frame as FP16 .pt files.

Usage:
    python cache_teacher_features.py                    # Full run (919 videos)
    python cache_teacher_features.py --debug            # First 5 videos only
    python cache_teacher_features.py --batch-size 2     # Reduce if OOM
    python cache_teacher_features.py --resume            # Skip existing caches (default)

Output:
    data/sa_v/cached_features/sav_XXXXXX.pt  — [N_frames, 256, 18, 18] FP16
"""

import argparse
import os
import time
from pathlib import Path

import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_frames_cv2(video_path: str, frame_indices: list, target_size: int = 504):
    """Extract specific frames from MP4 and resize to target_size.

    Args:
        video_path: path to .mp4 file
        frame_indices: list of 0-based frame indices to extract
        target_size: resize to (target_size, target_size)

    Returns:
        tensor [N, 3, H, W] in [0, 1] float32, or None if video unreadable
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for fidx in sorted(frame_indices):
        if fidx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR to RGB, resize, normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)

    cap.release()

    if not frames:
        return None
    return torch.stack(frames)


def normalize_for_teacher(frames: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Apply teacher normalization. frames: [N, 3, H, W] in [0, 1]."""
    m = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    return (frames - m) / s


def discover_videos(sa_v_dir: str):
    """Find all SA-V videos with manual annotations.

    Returns:
        list of (video_id, mp4_path, manual_json_path)
    """
    sa_v_path = Path(sa_v_dir) / "sav_train"
    videos = []

    for shard_dir in sorted(sa_v_path.iterdir()):
        if not shard_dir.is_dir():
            continue
        for mp4 in sorted(shard_dir.glob("*.mp4")):
            video_id = mp4.stem
            manual_json = mp4.parent / (video_id + "_manual.json")
            if manual_json.exists():
                videos.append((video_id, str(mp4), str(manual_json)))

    return videos


def get_annotated_frame_count(manual_json_path: str) -> int:
    """Read manual annotation to get number of annotated frames."""
    import json
    with open(manual_json_path, "r") as f:
        data = json.load(f)
    return len(data["masklet"])


def main():
    parser = argparse.ArgumentParser(description="Cache teacher FPN L3 features for SA-V")
    parser.add_argument("--sa-v-dir", type=str, default="data/sa_v")
    parser.add_argument("--cache-dir", type=str, default="data/sa_v/cached_features")
    parser.add_argument("--teacher-model", type=str, default="jetjodh/sam3")
    parser.add_argument("--image-size", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Frames per batch for teacher forward")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--debug", action="store_true",
                        help="Process first 5 videos only")
    parser.add_argument("--no-resume", action="store_true",
                        help="Overwrite existing cache files")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # -- Discover videos --
    videos = discover_videos(args.sa_v_dir)
    print("Found {} SA-V videos with manual annotations".format(len(videos)))

    if args.debug:
        videos = videos[:5]
        print("  DEBUG: processing first {} videos only".format(len(videos)))

    # -- Load teacher --
    print("Loading teacher model...")
    from transformers import Sam3Model
    teacher = Sam3Model.from_pretrained(args.teacher_model, torch_dtype=torch.float16)
    teacher = teacher.to(args.device).half()
    teacher.requires_grad_(False)
    teacher.eval()
    print("  Teacher: {:.1f}M params".format(sum(p.numel() for p in teacher.parameters()) / 1e6))

    # Resize RoPE for target resolution
    native_size = teacher.config.vision_config.backbone_config.image_size
    if args.image_size != native_size:
        from distillation.trainer import resize_teacher_rope
        resize_teacher_rope(teacher, args.image_size)

    # -- Process videos --
    total_frames_cached = 0
    total_bytes = 0
    start_time = time.time()

    for vi, (video_id, mp4_path, json_path) in enumerate(videos):
        cache_path = os.path.join(args.cache_dir, video_id + ".pt")

        # Resume: skip if already cached
        if not args.no_resume and os.path.exists(cache_path):
            print("[{}/{}] {} — cached, skipping".format(vi + 1, len(videos), video_id))
            continue

        # Determine annotated frames
        num_annotated = get_annotated_frame_count(json_path)
        if num_annotated == 0:
            print("[{}/{}] {} — no annotations, skipping".format(vi + 1, len(videos), video_id))
            continue

        # SA-V annotates every 4th frame starting from 0
        frame_indices = list(range(0, num_annotated * 4, 4))[:num_annotated]

        # Extract frames from video
        frames = extract_frames_cv2(mp4_path, frame_indices, target_size=args.image_size)
        if frames is None or len(frames) == 0:
            print("[{}/{}] {} — frame extraction failed, skipping".format(
                vi + 1, len(videos), video_id))
            continue

        # Teacher forward in batches
        frames_normalized = normalize_for_teacher(frames)
        all_features = []

        for batch_start in range(0, len(frames_normalized), args.batch_size):
            batch = frames_normalized[batch_start:batch_start + args.batch_size]
            batch = batch.to(args.device).half()

            with torch.no_grad():
                vision_out = teacher.get_vision_features(batch)
                # FPN level 3: lowest resolution [batch, 256, 18, 18]
                fpn_l3 = vision_out.fpn_hidden_states[3]
                all_features.append(fpn_l3.cpu().half())

            del batch, vision_out, fpn_l3
            if args.device == "mps":
                torch.mps.empty_cache()

        # Stack and save
        cached_tensor = torch.cat(all_features, dim=0)  # [N_frames, 256, 18, 18]
        torch.save(cached_tensor, cache_path)

        file_size = os.path.getsize(cache_path)
        total_frames_cached += cached_tensor.shape[0]
        total_bytes += file_size

        elapsed = time.time() - start_time
        eta = elapsed / (vi + 1) * (len(videos) - vi - 1) if vi > 0 else 0

        print("[{}/{}] {} — {} frames, shape {}, {:.1f} MB, ETA {:.1f}h".format(
            vi + 1, len(videos), video_id,
            cached_tensor.shape[0], list(cached_tensor.shape),
            file_size / 1e6, eta / 3600))

        del frames, frames_normalized, all_features, cached_tensor

    elapsed = time.time() - start_time
    print("\nDone: {} frames cached, {:.2f} GB total, {:.1f} hours".format(
        total_frames_cached, total_bytes / 1e9, elapsed / 3600))


if __name__ == "__main__":
    main()
