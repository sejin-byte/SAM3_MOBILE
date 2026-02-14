"""
SAM 3 베이스라인 추론 스크립트 (Mac Mini M4 Pro / MPS)
=====================================================
HuggingFace Transformers 구현체를 사용하여 Triton/CUDA 의존성을 우회합니다.
- device="mps" 명시 지정 (device_map="auto" 사용 금지)
- pin_memory=False (MPS 호환성)
- torch_dtype=float16 (24GB UMA 메모리 절약)
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# ── 환경 설정 ──────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# facebook/sam3 gated repo 접근 승인 완료
MODEL_ID = "facebook/sam3"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
OUTPUT_DIR = Path(__file__).parent / "outputs" / "baseline"


def load_model():
    """SAM 3 모델 및 프로세서를 MPS 디바이스에 로드합니다."""
    from transformers import Sam3Model, Sam3Processor

    print(f"[1/4] 모델 로드 중... (device={DEVICE}, dtype={DTYPE})")
    t0 = time.time()

    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        # device_map="auto" 사용 금지 — CUDA로 라우팅될 수 있음
    ).to(DEVICE)

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"      완료! ({elapsed:.1f}s)")
    print(f"      파라미터: {param_count / 1e6:.1f}M")
    print(f"      메모리 사용량: {mem_mb:.0f} MB ({DTYPE})")

    return model, processor


def run_image_inference(model, processor, image_path: str, text_prompt: str):
    """단일 이미지에 대해 텍스트 프롬프트 기반 분할을 수행합니다."""
    from PIL import Image

    print(f"\n[2/4] 이미지 추론: '{text_prompt}'")
    print(f"      입력: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # 프로세서로 입력 전처리
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    )
    # MPS로 이동
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 추론
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - t0

    print(f"      추론 시간: {elapsed * 1000:.1f} ms")
    print(f"      출력 키: {list(outputs.keys()) if hasattr(outputs, 'keys') else type(outputs)}")

    return outputs, elapsed


def save_baseline_results(outputs, elapsed, text_prompt, output_path):
    """베이스라인 추론 결과를 저장합니다 (교사 모델 데이터로 활용)."""
    output_path.mkdir(parents=True, exist_ok=True)

    # 메타데이터 저장
    metadata = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "text_prompt": text_prompt,
        "inference_time_ms": round(elapsed * 1000, 1),
        "output_keys": list(outputs.keys()) if hasattr(outputs, "keys") else [],
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 출력 텐서 저장 (증류 학습 시 교사 데이터로 사용)
    tensors_to_save = {}
    if hasattr(outputs, "keys"):
        for key in outputs.keys():
            val = outputs[key]
            if isinstance(val, torch.Tensor):
                tensors_to_save[key] = val.cpu().float()

    if tensors_to_save:
        torch.save(tensors_to_save, output_path / "teacher_outputs.pt")
        print(f"\n[3/4] 교사 모델 출력 저장: {output_path}")
        for k, v in tensors_to_save.items():
            print(f"      {k}: {list(v.shape)}")


def create_test_image(output_dir: Path) -> str:
    """테스트용 샘플 이미지를 생성합니다."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test_image.png"

    if not test_path.exists():
        # 간단한 테스트 이미지 생성 (640x480, 컬러 블록)
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img.save(test_path)
        print(f"      테스트 이미지 생성: {test_path}")

    return str(test_path)


def main():
    print("=" * 60)
    print("  SAM 3 베이스라인 추론 (Mac Mini M4 Pro / MPS)")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: {DEVICE}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print()

    # 1. 모델 로드
    model, processor = load_model()

    # 2. 테스트 이미지 준비
    test_image = create_test_image(OUTPUT_DIR)

    # 3. 추론 실행
    text_prompt = "a blue object"
    outputs, elapsed = run_image_inference(model, processor, test_image, text_prompt)

    # 4. 결과 저장
    save_baseline_results(outputs, elapsed, text_prompt, OUTPUT_DIR)

    print(f"\n[4/4] 베이스라인 추론 완료!")
    print(f"      결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
