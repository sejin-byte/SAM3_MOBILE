"""SA-V Video Dataset for temporal memory distillation.

Each sample is a video clip: T context frames (cached FPN L3) + 1 query frame (pixels).
GT masks come from SA-V manual annotations (RLE-encoded).
"""

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .video_config import VideoDistillationConfig


class SAVVideoDataset(Dataset):
    """SA-V video dataset for video distillation.

    Each sample returns:
    - context_features: [T, 256, 18, 18] — cached teacher FPN L3 for context frames
    - student_pixel_values: [3, H, W] — query frame preprocessed for student
    - student_input_ids: [seq_len] — tokenized text prompt for MobileCLIP
    - gt_masks: [N, H_mask, W_mask] — binary GT masks for query frame
    - gt_boxes: [N, 4] — normalized boxes (x1, y1, x2, y2)
    - num_objects: int — N (variable per sample)
    """

    def __init__(
        self,
        config: VideoDistillationConfig,
        student_tokenizer,
    ):
        self.config = config
        self.student_tokenizer = student_tokenizer

        # Discover videos with cached features
        self.clips = self._discover_clips()

        # Student image preprocessing
        self.student_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.student_mean, std=config.student_std),
        ])

    def _discover_clips(self):
        """Build list of valid (video_id, cache_path, json_path, mp4_path, num_frames).

        A clip is valid if:
        1. Cached features exist
        2. Manual annotation exists
        3. Has enough frames for context + query (T + 1)
        """
        clips = []
        sa_v_path = Path(self.config.sa_v_dir) / "sav_train"
        cache_dir = Path(self.config.cache_dir)

        for shard_dir in sorted(sa_v_path.iterdir()):
            if not shard_dir.is_dir():
                continue
            for mp4 in sorted(shard_dir.glob("*.mp4")):
                video_id = mp4.stem
                cache_path = cache_dir / (video_id + ".pt")
                json_path = mp4.parent / (video_id + "_" + self.config.annotation_type + ".json")

                if not cache_path.exists() or not json_path.exists():
                    continue

                # Read annotation to get frame count
                with open(json_path, "r") as f:
                    ann = json.load(f)
                num_annotated = len(ann["masklet"])

                # Need at least context_frames + 1 for a valid clip
                if num_annotated < self.config.context_frames + 1:
                    continue

                clips.append({
                    "video_id": video_id,
                    "cache_path": str(cache_path),
                    "json_path": str(json_path),
                    "mp4_path": str(mp4),
                    "num_annotated": num_annotated,
                })

        return clips

    def __len__(self):
        return len(self.clips)

    def _decode_rle(self, rle_dict, height, width):
        """Decode RLE mask from SA-V annotation format.

        Args:
            rle_dict: {"size": [H, W], "counts": str} — COCO-style uncompressed RLE

        Returns:
            binary mask [H, W] as uint8 tensor
        """
        from pycocotools import mask as mask_utils

        # pycocotools expects {"size": [H, W], "counts": bytes_or_str}
        rle = {"size": rle_dict["size"], "counts": rle_dict["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        binary = mask_utils.decode(rle)  # [H, W] numpy uint8
        return torch.from_numpy(binary).to(torch.uint8)

    def _mask_to_box(self, mask: torch.Tensor, img_h: int, img_w: int):
        """Convert binary mask to normalized (x1, y1, x2, y2) box.

        Args:
            mask: [H, W] binary uint8
            img_h, img_w: original image dimensions for normalization

        Returns:
            [4] tensor in (x1, y1, x2, y2) normalized coords, or None if empty
        """
        ys, xs = torch.where(mask > 0)
        if len(ys) == 0:
            return None
        x1 = xs.min().float() / img_w
        y1 = ys.min().float() / img_h
        x2 = xs.max().float() / img_w
        y2 = ys.max().float() / img_h
        return torch.tensor([x1, y1, x2, y2])

    def __getitem__(self, idx):
        clip = self.clips[idx]
        T = self.config.context_frames

        # Load cached teacher features: [num_annotated, 256, 18, 18]
        cached = torch.load(clip["cache_path"], map_location="cpu", weights_only=True)
        num_annotated = clip["num_annotated"]

        # Ensure cached tensor matches annotation count (take min for safety)
        num_frames = min(cached.shape[0], num_annotated)
        if num_frames < T + 1:
            # Shouldn't happen due to filtering, but guard anyway
            raise ValueError(
                "Video {} has {} frames but needs {}".format(clip["video_id"], num_frames, T + 1))

        # Sample a random query frame index (must have T context frames before it)
        query_idx = random.randint(T, num_frames - 1)
        context_indices = list(range(query_idx - T, query_idx))

        # Context features from cache
        context_features = cached[context_indices].float()  # [T, 256, 18, 18]

        # Query frame: extract from video
        import cv2
        cap = cv2.VideoCapture(clip["mp4_path"])

        # SA-V annotates every 4th frame starting from 0
        video_frame_idx = query_idx * 4
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # Fallback: return a black frame (training will handle gracefully)
            from PIL import Image
            query_image = Image.new("RGB", (self.config.image_size, self.config.image_size))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            query_image = Image.fromarray(frame)

        student_pixels = self.student_transform(query_image)

        # GT masks for query frame from annotation
        with open(clip["json_path"], "r") as f:
            ann = json.load(f)

        vid_h = int(ann["video_height"])
        vid_w = int(ann["video_width"])
        frame_masks = ann["masklet"][query_idx]  # list of RLE dicts (one per object)

        gt_masks = []
        gt_boxes = []
        for obj_mask in frame_masks:
            if obj_mask is None:
                continue
            mask = self._decode_rle(obj_mask, vid_h, vid_w)
            box = self._mask_to_box(mask, vid_h, vid_w)
            if box is None:
                continue
            gt_masks.append(mask)
            gt_boxes.append(box)

        # Limit objects per frame
        if len(gt_masks) > self.config.max_objects_per_frame:
            indices = random.sample(range(len(gt_masks)), self.config.max_objects_per_frame)
            gt_masks = [gt_masks[i] for i in indices]
            gt_boxes = [gt_boxes[i] for i in indices]

        if gt_masks:
            gt_masks = torch.stack(gt_masks)  # [N, H, W]
            gt_boxes = torch.stack(gt_boxes)  # [N, 4]
        else:
            # No valid objects — create dummy (1 empty mask)
            gt_masks = torch.zeros(1, vid_h, vid_w, dtype=torch.uint8)
            gt_boxes = torch.zeros(1, 4)

        # Tokenize text prompt (open_clip tokenizer)
        text = random.choice(self.config.text_prompts)
        try:
            tokens = self.student_tokenizer([text], context_length=77)
        except TypeError:
            tokens = self.student_tokenizer([text])
        if isinstance(tokens, torch.Tensor):
            input_ids = tokens.squeeze(0).long()
        elif isinstance(tokens, dict):
            input_ids = tokens["input_ids"].squeeze(0).long()
        elif hasattr(tokens, "input_ids"):
            input_ids = tokens.input_ids.squeeze(0).long()
        else:
            input_ids = torch.as_tensor(tokens).squeeze(0).long()

        return {
            "context_features": context_features,
            "student_pixel_values": student_pixels,
            "student_input_ids": input_ids,
            "gt_masks": gt_masks,
            "gt_boxes": gt_boxes,
            "num_objects": gt_masks.shape[0],
        }


def video_collate_fn(batch):
    """Collate for video distillation with variable GT counts.

    context_features and student_pixel_values are stacked normally.
    gt_masks and gt_boxes are kept as lists (variable N per sample).
    """
    context_features = torch.stack([b["context_features"] for b in batch])
    student_pixels = torch.stack([b["student_pixel_values"] for b in batch])
    student_input_ids = torch.stack([b["student_input_ids"] for b in batch])
    gt_masks = [b["gt_masks"] for b in batch]
    gt_boxes = [b["gt_boxes"] for b in batch]
    num_objects = [b["num_objects"] for b in batch]

    return {
        "context_features": context_features,
        "student_pixel_values": student_pixels,
        "student_input_ids": student_input_ids,
        "gt_masks": gt_masks,
        "gt_boxes": gt_boxes,
        "num_objects": num_objects,
    }
