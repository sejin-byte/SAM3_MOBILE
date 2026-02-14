"""
SAM 3 데이터셋 다운로드 및 전처리 파이프라인
============================================
1단계 증류: SA-1B (1% 서브셋) — 이미지 인코더 Feature Alignment
2단계 증류: SA-V — 시간적 메모리 모듈 학습
3단계 증류: SA-Co — 엔드-투-엔드 PCS 미세 조정

데이터셋 접근:
- SA-1B: https://ai.meta.com/datasets/segment-anything/
- SA-V:  https://ai.meta.com/datasets/segment-anything-video/
- SA-Co: facebookresearch/sam3 GitHub 또는 HuggingFace
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── 경로 설정 ──────────────────────────────────────────
DATA_ROOT = Path(__file__).parent
SA1B_DIR = DATA_ROOT / "sa1b"
SAV_DIR = DATA_ROOT / "sa_v"
SACO_DIR = DATA_ROOT / "sa_co"

# MPS 호환: pin_memory=False 필수
PIN_MEMORY = False


# ══════════════════════════════════════════════════════════
# SA-1B 데이터셋 (이미지 + 마스크)
# ══════════════════════════════════════════════════════════

class SA1BDataset(Dataset):
    """SA-1B 서브셋 데이터셋 (1단계 인코더 증류용).

    디렉토리 구조:
        sa1b/
        ├── images/
        │   ├── sa_000000.jpg
        │   ├── sa_000001.jpg
        │   └── ...
        └── annotations/
            ├── sa_000000.json
            ├── sa_000001.json
            └── ...

    각 JSON에는 이미지의 모든 마스크 어노테이션이 포함됨.
    """

    def __init__(self, root: Path, processor, max_samples: Optional[int] = None):
        self.root = root
        self.processor = processor
        self.image_dir = root / "images"
        self.anno_dir = root / "annotations"

        if self.image_dir.exists():
            self.image_files = sorted(self.image_dir.glob("*.jpg"))
            if max_samples:
                self.image_files = self.image_files[:max_samples]
        else:
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # 대응하는 어노테이션 로드
        anno_path = self.anno_dir / f"{img_path.stem}.json"
        annotations = {}
        if anno_path.exists():
            with open(anno_path) as f:
                annotations = json.load(f)

        # 프로세서로 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        # 배치 차원 제거 (DataLoader가 다시 추가)
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return {
            "inputs": inputs,
            "annotations": annotations,
            "image_path": str(img_path),
        }


# ══════════════════════════════════════════════════════════
# SA-V 데이터셋 (비디오 프레임 시퀀스)
# ══════════════════════════════════════════════════════════

class SAVDataset(Dataset):
    """SA-V 비디오 데이터셋 (2단계 메모리 증류용).

    디렉토리 구조:
        sa_v/
        ├── videos/
        │   ├── video_001/
        │   │   ├── 00000.jpg
        │   │   ├── 00001.jpg
        │   │   └── ...
        │   └── video_002/
        └── annotations/
            ├── video_001.json
            └── video_002.json
    """

    def __init__(self, root: Path, processor, num_frames: int = 8,
                 max_samples: Optional[int] = None):
        self.root = root
        self.processor = processor
        self.num_frames = num_frames
        self.video_dir = root / "videos"
        self.anno_dir = root / "annotations"

        if self.video_dir.exists():
            self.video_dirs = sorted([d for d in self.video_dir.iterdir() if d.is_dir()])
            if max_samples:
                self.video_dirs = self.video_dirs[:max_samples]
        else:
            self.video_dirs = []

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        from PIL import Image

        video_dir = self.video_dirs[idx]
        frames = sorted(video_dir.glob("*.jpg"))

        # 균등 간격으로 프레임 샘플링
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        images = [Image.open(f).convert("RGB") for f in frames]

        # 프레임별 전처리
        processed_frames = []
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt")
            processed_frames.append(
                {k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            )

        # 어노테이션 로드
        anno_path = self.anno_dir / f"{video_dir.name}.json"
        annotations = {}
        if anno_path.exists():
            with open(anno_path) as f:
                annotations = json.load(f)

        return {
            "frames": processed_frames,
            "annotations": annotations,
            "video_path": str(video_dir),
        }


# ══════════════════════════════════════════════════════════
# SA-Co 데이터셋 (개념 분할 — 텍스트 + 이미지 + 마스크)
# ══════════════════════════════════════════════════════════

class SACoDataset(Dataset):
    """SA-Co 개념 분할 데이터셋 (3단계 PCS 미세 조정용).

    디렉토리 구조:
        sa_co/
        ├── images/
        │   ├── 000000.jpg
        │   └── ...
        └── annotations.json
            [{"image": "000000.jpg", "concept": "yellow school bus",
              "masks": [...], "presence": true}, ...]
    """

    def __init__(self, root: Path, processor, split: str = "train",
                 max_samples: Optional[int] = None):
        self.root = root
        self.processor = processor
        self.image_dir = root / "images"

        anno_file = root / f"annotations_{split}.json"
        if anno_file.exists():
            with open(anno_file) as f:
                self.annotations = json.load(f)
            if max_samples:
                self.annotations = self.annotations[:max_samples]
        else:
            self.annotations = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        from PIL import Image

        anno = self.annotations[idx]
        image = Image.open(self.image_dir / anno["image"]).convert("RGB")
        text_prompt = anno["concept"]

        # 텍스트 프롬프트 포함 전처리
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return {
            "inputs": inputs,
            "concept": text_prompt,
            "presence": anno.get("presence", True),
            "image_path": str(self.image_dir / anno["image"]),
        }


# ══════════════════════════════════════════════════════════
# DataLoader 유틸리티
# ══════════════════════════════════════════════════════════

def create_dataloader(dataset, batch_size=1, num_workers=0, shuffle=True):
    """MPS 호환 DataLoader 생성.

    주의: pin_memory=False 필수 (Apple Silicon MPS 호환성)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,  # MPS에서 True 사용 시 crash
    )


# ══════════════════════════════════════════════════════════
# 다운로드 가이드 출력
# ══════════════════════════════════════════════════════════

def print_download_guide():
    """데이터셋 다운로드 안내를 출력합니다."""
    print("=" * 60)
    print("  SAM 3 데이터셋 다운로드 가이드")
    print("=" * 60)

    print(f"""
[SA-1B] 이미지 분할 데이터셋 (1단계 증류용)
  접근: https://ai.meta.com/datasets/segment-anything/
  크기: 전체 ~11TB, 1% 서브셋 ~100GB
  저장: {SA1B_DIR}/
  구조:
    images/     → .jpg 이미지
    annotations/ → .json 마스크 어노테이션

[SA-V] 비디오 분할 데이터셋 (2단계 증류용)
  접근: https://ai.meta.com/datasets/segment-anything-video/
  저장: {SAV_DIR}/
  구조:
    videos/       → 비디오별 프레임 폴더
    annotations/  → .json 추적 어노테이션

[SA-Co] 개념 분할 데이터셋 (3단계 증류용)
  접근: facebookresearch/sam3 GitHub repo
  저장: {SACO_DIR}/
  구조:
    images/              → .jpg 이미지
    annotations_train.json → 텍스트 개념 + 마스크
    annotations_val.json   → 검증 세트
""")

    # 디렉토리 생성 상태 확인
    for name, path in [("SA-1B", SA1B_DIR), ("SA-V", SAV_DIR), ("SA-Co", SACO_DIR)]:
        status = "준비됨" if path.exists() else "미생성"
        has_data = False
        if path.exists():
            has_data = any(path.rglob("*.jpg")) or any(path.rglob("*.json"))
        data_status = "데이터 있음" if has_data else "데이터 없음"
        print(f"  [{name}] {path} — {status}, {data_status}")

    print("\n" + "=" * 60)


def setup_directories():
    """데이터셋 디렉토리 구조를 생성합니다."""
    dirs = [
        SA1B_DIR / "images",
        SA1B_DIR / "annotations",
        SAV_DIR / "videos",
        SAV_DIR / "annotations",
        SACO_DIR / "images",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("데이터셋 디렉토리 구조 생성 완료:")
    for d in dirs:
        print(f"  {d}")


def verify_datasets():
    """데이터셋 준비 상태를 검증합니다."""
    print("\n데이터셋 검증:")
    results = {}

    for name, path, pattern in [
        ("SA-1B images", SA1B_DIR / "images", "*.jpg"),
        ("SA-1B annotations", SA1B_DIR / "annotations", "*.json"),
        ("SA-V videos", SAV_DIR / "videos", "*"),
        ("SA-Co images", SACO_DIR / "images", "*.jpg"),
        ("SA-Co train annotations", SACO_DIR, "annotations_train.json"),
    ]:
        count = len(list(path.glob(pattern))) if path.exists() else 0
        status = "OK" if count > 0 else "EMPTY"
        results[name] = count
        print(f"  [{status}] {name}: {count} files")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3 데이터셋 준비")
    parser.add_argument("--setup", action="store_true", help="디렉토리 구조 생성")
    parser.add_argument("--verify", action="store_true", help="데이터셋 상태 검증")
    parser.add_argument("--guide", action="store_true", help="다운로드 가이드 출력")
    args = parser.parse_args()

    if args.setup:
        setup_directories()
    elif args.verify:
        verify_datasets()
    else:
        print_download_guide()

    if args.setup or (not args.verify and not args.guide):
        setup_directories()
