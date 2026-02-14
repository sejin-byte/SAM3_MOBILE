# Codex 업무 이관서 — SAM 3 Mobile Deployment

> **작성일**: 2026-02-10
> **이전 담당**: Claude Code (Opus 4.6)
> **프로젝트**: SAM 3 (848M) → EfficientSAM3 (100.7M) → 모바일 배포
> **장비**: Mac Mini M4 Pro, 24GB UMA, macOS Darwin 25.2.0

---

## 1. 프로젝트 현황 요약

> **업데이트 (2026-02-12)**  
> - Phase 2(이미지) + 비디오 증류 완료  
> - Stage 4 양자화/시각검증 파이프라인 실행 완료  
> - Stage 5 `scripts/export_executorch.py` 구현 완료 (`.pte` 생성 가능)

```
[완료] 1단계: 환경 구축
[완료] 2단계: 아키텍처 경량화 (EfficientSAM3 student model 100.7M)
[완료] 3단계: 지식 증류
       ├── [완료] Phase 1: Feature Alignment (이미지)
       ├── [완료] Phase 2: Output Refinement (이미지)
       └── [완료] 비디오 증류 (Perceiver + MemoryCrossAttention)
[완료] 4단계: TorchAO 양자화 + 시각 검증
[완료] 5단계: ExecuTorch export 스크립트 구현 및 .pte 생성 검증
```

---

## 2. 다음 작업 순서 (정확히 이 순서로 진행)

### 작업 A: Phase 2 완료 확인
Phase 2가 이미 실행 중입니다. 완료되었는지 확인하세요.
```bash
ls -la checkpoints/distillation/phase2_*
```
Phase 2 체크포인트가 없으면 다시 실행:
```bash
conda activate sam3_mobile
python train_distill.py --phase 2 --device mps
```

### 작업 B: 교사 FPN L3 특징 캐싱 (~16시간)
Phase 2가 **완전히 끝난 후** MPS를 점유하여 실행합니다.
```bash
python cache_teacher_features.py --device mps
```
- 919 SA-V 비디오에서 교사(SAM3) FPN level 3 특징 추출
- 출력: `data/sa_v/cached_features/sav_XXXXXX.pt` (영상당 하나)
- 각 파일: `[N_frames, 256, 18, 18]` FP16 텐서
- **resumable**: 중단 후 재실행 시 기존 파일 스킵
- 총 ~12GB 디스크

검증:
```bash
python3 -c "
import torch, glob
files = sorted(glob.glob('data/sa_v/cached_features/*.pt'))
print(f'{len(files)} cached files')
t = torch.load(files[0], weights_only=True)
print(f'Shape: {t.shape}, dtype: {t.dtype}')
assert t.shape[1:] == (256, 18, 18) and t.dtype == torch.float16
print('OK')
"
```

### 작업 C: 비디오 증류 스모크 테스트
```bash
python train_video_distill.py --student-ckpt checkpoints/distillation/phase2_epoch2_step<최신>.pt --debug --device mps
```
확인할 것:
- 모든 loss가 finite (NaN/Inf 없음)
- "Trainable: 1.862M | Frozen: 98.8M" 출력
- 5 step 완료

### 작업 D: 비디오 증류 풀 학습
```bash
python train_video_distill.py --student-ckpt checkpoints/distillation/phase2_epoch2_step<최신>.pt --device mps
```
- 5 epochs, lr=1e-4, warmup=200
- 체크포인트: `checkpoints/video_distillation/video_epoch*_step*.pt`

### 작업 E: 4단계 양자화 (PLAN.md 4단계 참조)
비디오 증류 완료 후 진행. `PLAN.md`의 4단계 섹션을 따르세요.

---

## 3. 프로젝트 파일 구조

```
SAM3_M4/
├── PLAN.md                          ★ 전체 계획서 (5단계 체크박스)
├── run_sam3.py                       교사 모델 베이스라인 추론
├── verify_student.py                 학생 모델 검증
│
├── models/                          ★ EfficientSAM3 학생 모델
│   ├── __init__.py                   exports
│   ├── configuration.py              EfficientSAM3Config 하이퍼파라미터
│   ├── efficient_sam3.py            ★ 핵심: forward(), forward_with_intermediates(), forward_video()
│   ├── backbone_repvit.py            RepViT-M2.3 백본 + FPN
│   ├── text_encoder_mobileclip.py    MobileCLIP-S1 텍스트 인코더
│   ├── perceiver_resampler.py        Perceiver Resampler (K=64 latents)
│   ├── memory_attention.py          ★ MemoryCrossAttention (비디오용, 신규)
│   ├── lightweight_detr.py           DETR encoder/decoder
│   ├── mask_decoder.py               마스크 디코더
│   └── utils.py                      inverse_sigmoid, box_cxcywh_to_xyxy, DecoderMLP
│
├── distillation/                    ★ 증류 학습 패키지
│   ├── __init__.py                   exports
│   ├── config.py                     이미지 증류 설정 (DistillationConfig)
│   ├── dataset.py                    SA-1B 데이터셋 (SA1BDistillDataset)
│   ├── prompt_encoder.py             GeometricPromptEncoder
│   ├── greedy_matcher.py             MPS-native GreedyMatcher
│   ├── losses.py                     이미지 9-term 손실 (DistillationLoss)
│   ├── trainer.py                    이미지 증류 트레이너 + resize_teacher_rope()
│   ├── video_config.py              ★ 비디오 증류 설정 (VideoDistillationConfig)
│   ├── video_dataset.py             ★ SA-V 데이터셋 (SAVVideoDataset)
│   ├── video_losses.py              ★ 비디오 5-term GT 손실 (VideoDistillationLoss)
│   └── video_trainer.py             ★ 비디오 증류 트레이너 (VideoDistillationTrainer)
│
├── train_distill.py                  이미지 증류 CLI (--phase 1/2)
├── train_video_distill.py           ★ 비디오 증류 CLI (신규)
├── cache_teacher_features.py        ★ 교사 FPN 캐싱 CLI (신규)
│
├── data/
│   ├── sa1b/                         SA-1B 이미지 33,558장 (.jpg + .json)
│   ├── sa_v/
│   │   ├── sav_train/sav_000/        SA-V 919 비디오 (.mp4 + _manual.json + _auto.json)
│   │   └── cached_features/          [미생성] 교사 FPN 캐시 (.pt)
│   └── sa_co/                        SA-Co gold/silver
│
├── checkpoints/
│   ├── distillation/                 이미지 증류 체크포인트
│   │   ├── phase1_epoch0_step8139.pt   Phase 1 최종 (1.1GB)
│   │   └── phase2_epoch*_step*.pt      Phase 2 (생성 중)
│   └── video_distillation/           [미생성] 비디오 증류 체크포인트
│
└── docs/
    └── HANDOFF_CODEX.md              이 문서
```

---

## 4. 핵심 기술 정보

### 환경
```
conda env: sam3_mobile (Python 3.10)
torch: 2.11.0.dev20260207 (nightly, MPS backend)
transformers: 5.2.0.dev0 (Sam3Model, Sam3Processor)
timm: 1.0.24 (repvit_m2_3)
open_clip_torch: 3.2.0 (MobileCLIP-S1, pretrained='datacompdr')
torchao: 0.17.0 (소스 빌드)
executorch: 1.1.0
pycocotools: 2.0.11
opencv-python: 4.13.0.92
```

### MPS 주의사항 (반드시 준수)
1. `pin_memory=False` — True이면 MPS에서 크래시
2. `device="mps"` 명시 — `device_map="auto"` 사용 금지 (CUDA로 라우팅될 수 있음)
3. `batch_size=1` — 24GB UMA에서 batch>1은 메모리 압력 유발
4. `torch.mps.empty_cache()` — 큰 텐서 삭제 후 호출하여 MPS 메모리 반환
5. Box RPB는 `float32`로 계산 — `log2`/`sign` 정밀도 문제

### 교사 모델 (SAM 3)
- HuggingFace: `Sam3Model`, `Sam3Processor` (NOT Sam3ForConceptSegmentation)
- 미러: `jetjodh/sam3` (facebook/sam3은 gated)
- 840.4M params, FP16 1,681 MB
- 504px 해상도 사용 시 `resize_teacher_rope(teacher, 504)` 필수
- `teacher.get_vision_features()` → `fpn_hidden_states` (4 levels)

### 학생 모델 (EfficientSAM3)
- 100.7M params, FP16 ~201 MB
- 3개 forward 메서드:
  - `forward()`: 이미지 추론 (배포용)
  - `forward_with_intermediates()`: 이미지 증류 (중간 특징 반환)
  - `forward_video()`: 비디오 추론 (메모리 주입)
- 비디오용 모듈: `perceiver_resampler` (1.6M) + `memory_cross_attn` (0.26M)
- `strict=False` 로딩 필수 — 이전 체크포인트에 `memory_cross_attn` 키 없음

### SA-V 어노테이션 구조
```python
# _manual.json
{
  "masklet": [              # masklet[frame_idx][object_idx]
    [                        # frame 0
      {"size": [848, 480], "counts": "...RLE..."},  # object 0
      {"size": [848, 480], "counts": "...RLE..."},  # object 1
      null,                                          # object 2 (not visible)
    ],
    [...],                   # frame 1
  ],
  "masklet_num": 5,          # total objects tracked
  "video_height": 848,
  "video_width": 480,
  "video_frame_count": 483,  # total video frames
}
# SA-V는 4프레임 간격으로 어노테이션 → annotation frame i = video frame i*4
```

### 비디오 증류 아키텍처
```
Context T=8 frames  →  cached FPN L3 [T, 256, 18, 18]
                              ↓ flatten
                        [batch, T*324, 256]
                              ↓
                    Perceiver Resampler (TRAINABLE, 1.6M)
                              ↓
                        [batch, 64, 256]     ← 40x 압축
                              ↓
Query frame → backbone(FROZEN) → FPN → DETR encoder(FROZEN) → encoder_out
                              ↓
              MemoryCrossAttention(encoder_out, memory_tokens) (TRAINABLE, 0.26M)
                gate * tanh(attn_output) + residual
                              ↓
              DETR decoder(FROZEN) → masks, boxes, logits
                              ↓
              Loss vs SA-V GT (5 terms: mask, box_l1, giou, iou, presence)
```

- gate 파라미터는 0으로 초기화 → 처음엔 이미지 성능 그대로 유지
- 1-2 optimizer step 후 gate가 열리면서 perceiver에 gradient 흐름 시작
- 체크포인트는 trainable 모듈(perceiver + memory_cross_attn)만 저장 (~7MB)

---

## 5. 알려진 이슈 및 주의사항

| 이슈 | 설명 | 해결 |
|------|------|------|
| Phase 2 load 실패 | `memory_cross_attn` 키 없음 | `strict=False` 적용 완료 |
| MPS OOM | 교사+학생 동시 로드 시 | `torch.mps.empty_cache()`, batch=1 |
| 교사 RoPE | 504px에서 위치 인코딩 불일치 | `resize_teacher_rope()` 호출 |
| open_clip tag | 잘못된 pretrained name | `('MobileCLIP-S1', 'datacompdr')` 사용 |
| TorchAO 설치 | torch nightly 호환성 | `pip install --no-build-isolation git+...ao.git` |
| TensorBoard | sam3_mobile env에 미설치 | import 가드 처리됨 (optional) |

---

## 6. 검증 커맨드 모음

```bash
# 환경 확인
conda activate sam3_mobile
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"

# 학생 모델 빌드 + forward_video 테스트
python -c "
import torch
from models import EfficientSAM3, EfficientSAM3Config
m = EfficientSAM3(EfficientSAM3Config())
out = m.forward_video(torch.randn(1,3,504,504), torch.randint(0,1000,(1,16)), torch.randn(1,8*324,256))
print({k: list(v.shape) for k, v in out.items()})
"

# freeze/unfreeze 확인
python -c "
from models import EfficientSAM3, EfficientSAM3Config
m = EfficientSAM3(EfficientSAM3Config())
m.requires_grad_(False)
m.perceiver_resampler.requires_grad_(True)
m.memory_cross_attn.requires_grad_(True)
t = sum(p.numel() for p in m.parameters() if p.requires_grad)
f = sum(p.numel() for p in m.parameters() if not p.requires_grad)
print(f'Trainable: {t/1e6:.3f}M, Frozen: {f/1e6:.1f}M')
"

# 캐시 상태 확인
ls data/sa_v/cached_features/ | wc -l   # 919이면 캐싱 완료

# 비디오 증류 디버그 실행
python train_video_distill.py --student-ckpt <path> --debug --device mps
```

---

## 7. 연락 사항

- 전체 계획: `PLAN.md` (5단계 체크박스 형태)
- 기술 메모: `.claude/projects/-Users-sejinkim-developers-SAM3-M4/memory/MEMORY.md`
- 이전 대화 전문: `.claude/projects/-Users-sejinkim-developers-SAM3-M4/4c7faf1a-28dd-4d94-becb-4a1be075f3d6.jsonl`
