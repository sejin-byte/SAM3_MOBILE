"""SA-1B Distillation Dataset with dual preprocessing and dual tokenization."""

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .config import DistillationConfig


class SA1BDistillDataset(Dataset):
    """SA-1B dataset for knowledge distillation.

    Each sample returns:
    - teacher_pixel_values: [3, H, W] normalized with teacher mean/std
    - student_pixel_values: [3, H, W] normalized with student mean/std
    - teacher_input_ids: [seq_len_teacher] tokenized text prompt (HF tokenizer)
    - student_input_ids: [seq_len_student] tokenized text prompt (open_clip tokenizer)
    - prompt_type: str — "text", "point", or "box"
    - prompt_coords: tensor — normalized coordinates for point/box prompts
    - prompt_text: str — effective text prompt used for this sample
    """

    def __init__(
        self,
        config: DistillationConfig,
        teacher_tokenizer=None,
        student_tokenizer=None,
        tokenizer=None,
        split: str = "train",
        phase: int = 1,
    ):
        self.config = config
        self.teacher_tokenizer = teacher_tokenizer or tokenizer
        self.student_tokenizer = student_tokenizer
        if self.teacher_tokenizer is None:
            raise ValueError("teacher_tokenizer is required")
        if self.student_tokenizer is None:
            raise ValueError("student_tokenizer is required")
        self.split = split
        self.phase = phase
        self.prompt_ratios = config.prompt_ratios_phase1 if phase == 1 else config.prompt_ratios_phase2

        # Discover image files
        sa1b_dir = Path(config.sa1b_dir)
        all_jsons = sorted(sa1b_dir.glob("sa_*.json"))
        if split == "train":
            self.json_files = all_jsons[:config.num_train]
        else:
            self.json_files = all_jsons[config.num_train:config.num_train + config.num_val]

        # Dual preprocessing transforms
        self.teacher_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.teacher_mean, std=config.teacher_std),
        ])
        self.student_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.student_mean, std=config.student_std),
        ])

    def __len__(self):
        return len(self.json_files)

    def _sample_prompt_type(self) -> str:
        """Randomly select prompt type based on phase ratios."""
        r = random.random()
        cumulative = 0.0
        for ptype, ratio in self.prompt_ratios.items():
            cumulative += ratio
            if r < cumulative:
                return ptype
        return "text"

    def _extract_prompts(self, annotations: list, img_w: int, img_h: int):
        """Extract point and box prompts from SA-1B annotations.

        Returns:
            points: list of [x_norm, y_norm] centroids
            boxes: list of [x1_norm, y1_norm, x2_norm, y2_norm]
        """
        points = []
        boxes = []
        for ann in annotations:
            # Point prompt from centroid
            if "point_coords" in ann and ann["point_coords"]:
                px, py = ann["point_coords"][0]
                points.append([px / img_w, py / img_h])

            # Box prompt from bbox (SA-1B format: [x, y, w, h])
            if "bbox" in ann and ann["bbox"]:
                x, y, w, h = ann["bbox"]
                boxes.append([
                    x / img_w,
                    y / img_h,
                    (x + w) / img_w,
                    (y + h) / img_h,
                ])
        return points, boxes

    def _tokenize_teacher(self, text: str) -> torch.Tensor:
        """Tokenize prompt for teacher (HF tokenizer)."""
        tokens = self.teacher_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.teacher_text_max_length,
            truncation=True,
        )
        if isinstance(tokens, dict):
            input_ids = tokens["input_ids"]
        else:
            input_ids = tokens.input_ids
        return input_ids.squeeze(0).long()

    def _tokenize_student(self, text: str) -> torch.Tensor:
        """Tokenize prompt for student (open_clip tokenizer)."""
        try:
            tokens = self.student_tokenizer([text], context_length=self.config.student_text_max_length)
        except TypeError:
            tokens = self.student_tokenizer([text])

        if isinstance(tokens, torch.Tensor):
            input_ids = tokens
        elif isinstance(tokens, dict):
            input_ids = tokens["input_ids"]
        elif hasattr(tokens, "input_ids"):
            input_ids = tokens.input_ids
        else:
            input_ids = torch.as_tensor(tokens)

        if input_ids.ndim == 1:
            return input_ids.long()
        return input_ids.squeeze(0).long()

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        img_path = json_path.with_suffix(".jpg")

        # Load annotation
        with open(json_path, "r") as f:
            data = json.load(f)

        img_info = data["image"]
        img_w, img_h = img_info["width"], img_info["height"]
        annotations = data.get("annotations", [])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Dual preprocessing
        teacher_pixels = self.teacher_transform(image)
        student_pixels = self.student_transform(image)

        # Select prompt type and generate prompt data
        prompt_type = self._sample_prompt_type()
        points, boxes = self._extract_prompts(annotations, img_w, img_h)

        # Fall back to text if no geometric prompts available
        if prompt_type == "point" and not points:
            prompt_type = "text"
        if prompt_type == "box" and not boxes:
            prompt_type = "text"

        if prompt_type == "text":
            prompt_text = random.choice(self.config.text_prompts)
            prompt_coords = torch.zeros(1, 4)  # placeholder
        elif prompt_type == "point":
            # Sample up to 5 random points
            sampled = random.sample(points, min(5, len(points)))
            # Pad to [N, 2] → store as [N, 4] with zeros for consistency
            coords = torch.tensor(sampled, dtype=torch.float32)
            prompt_coords = torch.zeros(coords.shape[0], 4)
            prompt_coords[:, :2] = coords
            prompt_text = self.config.neutral_prompt
        elif prompt_type == "box":
            # Sample up to 5 random boxes
            sampled = random.sample(boxes, min(5, len(boxes)))
            prompt_coords = torch.tensor(sampled, dtype=torch.float32)
            prompt_text = self.config.neutral_prompt

        # Use a single semantic prompt text for both towers.
        teacher_input_ids = self._tokenize_teacher(prompt_text)
        student_input_ids = self._tokenize_student(prompt_text)

        return {
            "teacher_pixel_values": teacher_pixels,
            "student_pixel_values": student_pixels,
            "teacher_input_ids": teacher_input_ids,
            "student_input_ids": student_input_ids,
            "prompt_type": prompt_type,
            "prompt_coords": prompt_coords,
            "prompt_text": prompt_text,
        }


def collate_fn(batch):
    """Custom collate for variable-length prompt_coords."""
    teacher_pixels = torch.stack([b["teacher_pixel_values"] for b in batch])
    student_pixels = torch.stack([b["student_pixel_values"] for b in batch])
    teacher_input_ids = torch.stack([b["teacher_input_ids"] for b in batch])
    student_input_ids = torch.stack([b["student_input_ids"] for b in batch])
    prompt_types = [b["prompt_type"] for b in batch]
    prompt_texts = [b["prompt_text"] for b in batch]

    # Pad prompt_coords to max length in batch
    max_n = max(b["prompt_coords"].shape[0] for b in batch)
    padded_coords = torch.zeros(len(batch), max_n, 4)
    for i, b in enumerate(batch):
        n = b["prompt_coords"].shape[0]
        padded_coords[i, :n] = b["prompt_coords"]

    return {
        "teacher_pixel_values": teacher_pixels,
        "student_pixel_values": student_pixels,
        "teacher_input_ids": teacher_input_ids,
        "student_input_ids": student_input_ids,
        "prompt_types": prompt_types,
        "prompt_coords": padded_coords,
        "prompt_texts": prompt_texts,
    }
