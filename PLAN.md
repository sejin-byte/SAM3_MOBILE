# SAM 3 통합 실행 계획서 (Integrated Implementation Plan)

> **프로젝트**: Meta SAM 3 → 모바일 디바이스 배포
> **타겟 하드웨어**: Mac Mini M4 Pro (24GB UMA) — 개발 환경
> **배포 타겟**: iOS (CoreML/ANE), Android (QNN/Hexagon NPU)
> **핵심 전략**: EfficientSAM3 PHD(Progressive Hierarchical Distillation) + TorchAO 양자화 + ExecuTorch 배포
> **원본 모델**: SAM 3 — 848M 파라미터, Promptable Concept Segmentation (PCS)

---

## 1단계: 환경 구축 (Environment Setup)

### 1.1 macOS 개발 도구 설치

- [x] **Homebrew 설치** — v5.0.13 ✅
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- [x] **Node.js 설치** — v25.5.0 ✅
  ```bash
  brew install node
  node -v
  ```

- [x] **Miniforge 설치** — conda v25.11.0 ✅
  ```bash
  brew install miniforge
  conda init zsh
  source ~/.zshrc
  ```

### 1.2 AI 에이전트 도구 설정

- [x] **Claude Code 설치 및 인증** ✅
  ```bash
  brew install --cask claude-code
  claude login
  ```
  > 터미널 명령어 자율 실행 권한을 Allow 모드로 설정하여 작업 속도 향상

- [x] **Google Antigravity 설치** ✅
  - `antigravity.google/download` 에서 Apple Silicon용 `.dmg` 다운로드
  - Google 계정 로그인 후 Mission Control Setup 완료

- [x] **OpenAI Codex CLI 설치** ✅
  ```bash
  npm install -g @openai/codex
  codex login
  ```

- [x] **Antigravity-Claude 프록시 설정** — v2.6.2 ✅
  ```bash
  npm install -g antigravity-claude-proxy
  antigravity-claude-proxy start
  ```
  > Antigravity 설정에서 모델 엔드포인트를 `http://localhost:8080`으로 지정

### 1.3 Python 가상환경 및 딥러닝 프레임워크

- [x] **conda 가상환경 생성** — `sam3_mobile` (Python 3.10) ✅
  ```bash
  conda create -n sam3_mobile python=3.10
  conda activate sam3_mobile
  ```

- [x] **PyTorch Nightly 설치** — v2.11.0.dev20260207, MPS ✅
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
  ```
  > MPS 가속 확인:
  > ```python
  > import torch
  > print(torch.backends.mps.is_available())  # True
  > ```

- [x] **HuggingFace Transformers 설치** — v5.2.0.dev0, Sam3Model 포함 ✅
  ```bash
  pip install git+https://github.com/huggingface/transformers.git
  ```
  > 공식 SAM 3 리포지토리는 Triton/CUDA 의존성이 있어 Apple Silicon에서 직접 사용 불가.
  > HuggingFace 구현체로 우회하여 MPS 백엔드에서 추론 가능.

- [x] **ExecuTorch 설치** ✅
  ```bash
  pip install executorch
  ./install_requirements.sh --pybind coreml
  ```

- [x] **TorchAO (PyTorch Architecture Optimization)** — v0.17.0 (소스 빌드) ✅
  ```bash
  # torch 2.11.0.dev 호환을 위해 소스에서 설치
  pip install --no-build-isolation git+https://github.com/pytorch/ao.git
  # API 변경: int4_weight_only() → Int4WeightOnlyConfig
  ```

### 1.4 SAM 3 원본 모델 다운로드 및 베이스라인 추론 확인

- [x] **SAM 3 모델 체크포인트 다운로드** — 840.4M params, FP16 1,681MB ✅
  ```python
  from transformers import Sam3Model, Sam3Processor

  # facebook/sam3 gated repo 접근 승인 완료
  processor = Sam3Processor.from_pretrained("facebook/sam3")
  model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16)
  ```

- [x] **MPS 디바이스 추론 테스트 스크립트** (`run_sam3.py`) 작성 ✅
  ```python
  import torch

  device = "mps" if torch.backends.mps.is_available() else "cpu"
  model = model.to(device)

  # DataLoader에서 pin_memory=False 필수 (MPS 호환성)
  # device_map="auto" 대신 명시적으로 device="mps" 지정
  ```
  > **트러블슈팅**:
  > - `RuntimeError: Triton packages are not available` → `device="mps"` 명시 지정
  > - `pin_memory` 충돌 → `DataLoader(pin_memory=False)` 설정

- [x] **베이스라인 추론 결과 저장** — `outputs/baseline/` ✅
  - 출력 키: pred_masks[1,200,288,288], pred_boxes[1,200,4], pred_logits[1,200]
  - 존재 토큰(Presence Token): presence_logits[1,1]
  - 시맨틱 분할: semantic_seg[1,1,288,288]
  - 교사 출력: `teacher_outputs.pt` (증류 학습 데이터로 활용)

### 1.5 데이터셋 다운로드 및 전처리

- [x] **SA-1B 데이터셋** — 33,558 이미지 ✅
  - 3개 tar 서브셋: sa_000020, sa_000097, sa_000524 (~33GB)
  - 저장: `data/sa1b/` — 이미지(.jpg) + 어노테이션(.json) 쌍
  - 용도: 이미지 증류 Phase 1 (Feature Alignment) + Phase 2 (Output Refinement)

- [x] **SA-V (Segment Anything Video) 데이터셋** — 919 비디오 ✅
  - 파일: sav_000.tar (~8.10GB)
  - 저장: `data/sa_v/sav_train/sav_000/` — MP4 + 수동/자동 어노테이션 JSON
  - 구조: `masklet[frame_idx][object_idx]` = RLE {size, counts}, 4프레임 간격 어노테이션
  - 용도: 비디오 증류 (시간적 메모리 모듈 학습)

- [x] **SA-Co (Segment Anything with Concepts) 데이터셋** — Gold + Silver 다운로드 완료 ✅
  - Gold: 465MB (24 파일), Silver: 631MB (13 파일)
  - 저장: `data/sa_co/gold/`, `data/sa_co/silver/`
  - VEval (32.25GB): 3단계 증류 검증 시 다운로드 예정
  - 용도: 3단계 증류 (엔드-투-엔드 PCS 미세 조정 및 검증)

- [x] **데이터셋 전처리 파이프라인 구축** — `data/prepare_datasets.py` ✅
  - SA1BDataset, SAVDataset, SACoDataset 클래스 구현
  - DataLoader 구성 (pin_memory=False, batch_size=1)
  - 디렉토리 구조 생성 완료
  - `python data/prepare_datasets.py --verify` 로 상태 검증 가능

---

## 2단계: 아키텍처 경량화 (RepViT Student Model)

> **목표**: SAM 3의 848M 파라미터를 모바일 친화적인 경량 구조로 교체.
> 단순히 레이어 수를 줄인 ViT가 아니라, 모바일 NPU의 메모리 접근 패턴과 캐시 효율성을 고려한 아키텍처 선정.

### 2.1 이미지 인코더: RepViT-M2.3 백본

- [x] **RepViT-M2.3 아키텍처 선정 및 구현** — `models/backbone_repvit.py` ✅
  - 구조적 재매개변수화(Structural Re-parameterization) 기반
  - timm `repvit_m2_3` (features_only=True) + FPN 채널 어댑터 (80/160/320/640→256)
  - Feature map sizes at 1008x1008: 252x252, 126x126, 63x63, 32x32

- [x] **ImageNet 사전 학습 가중치 로드** ✅
  - `timm.create_model('repvit_m2_3', pretrained=True, features_only=True)`
  - 25.07M params

- [x] **SAM 3의 Perception Encoder를 RepViT-M2.3으로 교체** ✅
  - FPN 채널 어댑터: Conv1x1 + BN + ReLU + Conv3x3 + BN + ReLU
  - SinePositionEmbedding per level

### 2.2 텍스트 인코더: MobileCLIP-S1

- [x] **MobileCLIP-S1 모델 통합** — `models/text_encoder_mobileclip.py` ✅
  - open_clip `MobileCLIP-S1` (pretrained='datacompdr'), 12-layer TextTransformer
  - 시퀀스 전체 hidden states 추출 (pooled만이 아님)
  - 63.30M params

- [x] **텍스트 임베딩 차원 호환성 검증** ✅
  - nn.Linear(512, 256) 프로젝션으로 hidden_size 정렬
  - 출력: [batch, seq_len, 256]

### 2.3 비디오 메모리 모듈: Perceiver Resampler 기반 압축 메모리

- [x] **Perceiver Resampler 모듈 구현** — `models/perceiver_resampler.py` ✅
  - K=64 learnable latent tokens, 2-layer cross-attention + FFN (pre-norm)
  - nn.MultiheadAttention(batch_first=True) for ExecuTorch compatibility
  - 1.60M params

- [x] **SAM 3의 Dense Memory Bank를 Perceiver 압축 메모리로 대체** ✅
  - 고정 출력 shape [batch, 64, 256] — 비디오 길이 무관

### 2.4 디코더 경량화

- [x] **DETR 인코더/디코더 경량화 설계** — `models/lightweight_detr.py` ✅
  - Encoder: 3 layers (teacher: 6), FFN=1024 (teacher: 2048), 3.16M params
  - Decoder: 3 layers, 100 queries (teacher: 200), FFN=1024, 4.45M params
  - 존재 헤드(Presence Head), Box RPB, iterative box refinement 모두 유지
  - DotProductScoring: 0.66M params

- [x] **마스크 디코더** — `models/mask_decoder.py` ✅
  - PixelDecoder (3-stage FPN), MaskEmbedder (3-layer MLP), semantic seg head
  - 교사와 동일 구조 (hidden_size=256), 2.04M params

- [x] **정적 그래프 호환성** ✅
  - nn.MultiheadAttention(batch_first=True) 사용 — ExecuTorch export 호환
  - 동적 제어 흐름 최소화 (고정 layer count, 고정 query count)

---

## 3단계: 지식 증류 (Progressive Hierarchical Distillation)

> **목표**: 교사 모델(SAM 3, 848M)의 지식을 학생 모델(EfficientSAM3, 100.7M)에 단계적으로 전이.
> 이미지 증류 → 비디오 증류 → 엔드-투-엔드 미세 조정 순서로 진행.

### 3.1 이미지 증류 Phase 1: Feature Alignment (인코더 특징 정렬)

- [x] **증류 인프라 구축** ✅
  - `distillation/` 패키지: config, dataset, prompt_encoder, greedy_matcher, losses, trainer
  - IoU Head 추가: `DecoderMLP(256, 256, 1, num_layers=3)` → iou_scores[batch, 100]
  - `forward_with_intermediates()`: FPN features, encoder output, decoder hidden states 반환 + prompt injection
  - GeometricPromptEncoder: 사인 위치 인코딩 + 타입 임베딩 → 256-dim
  - GreedyMatcher: MPS-native (no scipy), cost = mask_iou + box_l1 + logit_sim, greedy assignment

- [x] **9개 손실 항목 설계 및 검증** ✅
  - 출력 손실 (항상 활성): mask(Dice+BCE), box_L1, box_GIoU, logit, iou_token, presence, semantic_seg
  - 특징 손실 (Phase 1만): fpn_feature(P1), encoder_feature(P1)
  - 모든 손실 MPS에서 finite 확인, backward pass 정상 작동

- [x] **Phase 1 학습 완료** ✅
  - 설정: 1 epoch, lr=1e-4, warmup=500, batch=4, grad_accum=2
  - 동적 프롬프트: text 50% / point 25% / box 25%
  - 504px 해상도 (teacher RoPE resize 적용)
  - 체크포인트: `checkpoints/distillation/phase1_epoch0_step8139.pt`

### 3.2 이미지 증류 Phase 2: Output Refinement (출력 정제) — 🔄 진행 중

- [x] **Phase 2 학습 코드 준비** ✅
  - 특징 손실 비활성화, 출력 손실만 사용
  - lr=5e-5, 동적 프롬프트: text 30% / point 35% / box 35%
  - Phase 1 체크포인트 자동 로드 (`strict=False` — memory_cross_attn 신규 모듈 대응)

- [ ] **Phase 2 학습 실행**
  ```bash
  python train_distill.py --phase 2 --device mps
  ```
  - 3 epochs, SA-1B 32,558 images
  - 예상: ~24시간

### 3.3 비디오 증류: Temporal Memory (시간적 메모리 학습) — ⏳ 코드 완성, 캐싱 대기

> **아키텍처**: 학생 모델의 Perceiver Resampler (1.6M) + MemoryCrossAttention (0.26M)만 학습.
> 나머지 ~98.8M 파라미터는 이미지 증류 결과를 동결(freeze).
> 교사 FPN L3 특징을 사전 캐싱하여 학습 시 교사 모델 불필요.

```
Context T=8 frames → cached FPN L3 [T, 256, 18, 18] → flatten [T*324, 256]
                                                              ↓
                                                    Perceiver Resampler (TRAINABLE)
                                                              ↓
                                                       [batch, 64, 256]
                                                              ↓
Query frame → Student backbone (FROZEN) → FPN → DETR encoder (FROZEN)
                                                              ↓
                                          MemoryCrossAttention(encoder_out, memory) (TRAINABLE)
                                                              ↓
                                          DETR decoder (FROZEN) → masks, boxes
                                                              ↓
                                          Loss vs SA-V GT masks (Dice+BCE, L1+GIoU)
```

- [x] **MemoryCrossAttention 모듈 구현** — `models/memory_attention.py` ✅
  - Pre-norm cross-attention: vision features(Q) × memory tokens(K,V) + gated residual
  - gate=0 초기화로 이미지 성능 무영향 보장, 학습 시 점진적 활성화
  - 264K params, batch_first=True (ExecuTorch 호환)

- [x] **EfficientSAM3.forward_video() 구현** — `models/efficient_sam3.py` ✅
  - Perceiver compress → MemoryCrossAttention → decoder → predictions
  - 출력 shape 검증 완료: pred_masks[1,100,126,126], pred_boxes[1,100,4] 등
  - freeze/unfreeze 검증: perceiver_resampler + memory_cross_attn만 requires_grad=True

- [x] **교사 FPN L3 캐싱 스크립트** — `cache_teacher_features.py` ✅
  - 919 SA-V 비디오 × ~121 프레임/비디오 → FPN level 3 [256, 18, 18] FP16
  - 배치 처리 (batch=4), resumable (기존 캐시 스킵)
  - 예상 디스크: ~12 GB, 예상 시간: ~16시간

- [ ] **교사 특징 캐싱 실행**
  ```bash
  python cache_teacher_features.py --device mps
  ```

- [x] **비디오 증류 패키지 구현** ✅
  - `distillation/video_config.py` — VideoDistillationConfig
  - `distillation/video_dataset.py` — SAVVideoDataset + video_collate_fn
    - 클립 샘플링: T context (캐시) + 1 query (MP4에서 추출)
    - GT: pycocotools RLE 디코딩 → binary masks + boxes
  - `distillation/video_losses.py` — 5개 GT 기반 손실 (mask, box_l1, box_giou, iou_token, presence)
  - `distillation/video_trainer.py` — freeze/unfreeze, 코사인 LR, 체크포인트 (trainable만 저장)
  - `train_video_distill.py` — CLI 진입점 (이미지 체크포인트 자동 탐색)

- [ ] **비디오 증류 학습 실행**
  ```bash
  python train_video_distill.py --student-ckpt <phase2_checkpoint> --debug  # 스모크 테스트
  python train_video_distill.py --student-ckpt <phase2_checkpoint>          # 풀 학습
  ```
  - 설정: 5 epochs, lr=1e-4, warmup=200, batch=1, grad_accum=4
  - 학습 가능 파라미터: ~1.86M (Perceiver 1.6M + MemoryCrossAttn 0.26M)

- [ ] **비디오 추적 정확도 중간 검증**
  - 가려짐(Occlusion) 상황에서의 추적 지속성 확인
  - 메모리 사용량 고정 여부 확인 (K=64 유지)

### 3.4 Stage 3: 엔드-투-엔드 PCS 미세 조정 (End-to-End Fine-Tuning)

- [ ] **데이터셋: SA-Co (Segment Anything with Concepts) 로드**

- [ ] **전체 파이프라인 동결 해제(Unfreeze) 및 미세 조정**
  - 백본(RepViT) + 텍스트 인코더(MobileCLIP) + 메모리(Perceiver) + 디코더
  - 존재 헤드(Presence Head)가 경량 백본 특징 맵에 적응하도록 학습
  - 미세한 의미론적 차이 학습 (예: "빨간 옷 사람" vs "파란 옷 사람")

- [ ] **QAT 준비: Fake Quantization 노드 삽입** (4단계 연계)
  ```python
  from torchao.quantization import quantize_, Int4WeightOnlyConfig
  # 미세 조정 후반부에서 fake quantization 활성화
  ```

### 3.5 중간 검증: 교사 모델 대비 성능 확인

- [ ] **SA-Co 검증 세트 평가**
  - **목표: 교사 모델 대비 85% 이상 성능 달성**
  - mIoU (Mean Intersection over Union) 측정
  - Presence Token 정확도 (존재 판단 F1 스코어)
  - 비디오 추적 J&F 스코어

- [ ] **성능 미달 시 대응 전략**
  - 학습률 조정 및 추가 에포크 훈련
  - 손실 함수 가중치 재조정
  - 백본 크기 상향 검토 (RepViT-M2.3 → M3.0)

### 3.6 사용자 시각 검증 (Visual QA)

> **목표**: 학습 완료 후, "사용자 관점"에서 학생 모델이 실제 이미지에 대해
> 프롬프트에 반응하며 합리적인 마스크를 내는지 빠르게 눈으로 확인.

- [ ] **시각 검증 리포트 생성 (이미지)**
  - 스크립트: `scripts/visual_eval_student.py`
  - 산출물: `outputs/visual_eval/<run>/index.html` + PNG 갤러리
  - 예시:
    ```bash
    conda activate sam3_mobile

    # 학생만 (original | student)
    python scripts/visual_eval_student.py \
      --student-ckpt checkpoints/distillation/phase2_epoch2_step<FINAL>.pt \
      --image outputs/baseline/test_image.png \
      --prompt "objects in the image" \
      --top-k 5 \
      --out-dir outputs/visual_eval/phase2_final_student_only

    # 교사 비교 (original | teacher | student) — distillation과 동일 토크나이저 사용
    python scripts/visual_eval_student.py \
      --student-ckpt checkpoints/distillation/phase2_epoch2_step<FINAL>.pt \
      --image outputs/baseline/test_image.png \
      --prompt "objects in the image" \
      --top-k 5 \
      --compare-teacher \
      --out-dir outputs/visual_eval/phase2_final_vs_teacher
    ```

- [ ] **(선택) 실제 사용자 이미지 폴더로 일괄 테스트**
  ```bash
  python scripts/visual_eval_student.py \
    --student-ckpt checkpoints/distillation/phase2_epoch2_step<FINAL>.pt \
    --image-dir <your_images_dir> \
    --prompt "segment everything" \
    --top-k 5 \
    --out-dir outputs/visual_eval/user_images
  ```

- [ ] **체크 포인트(눈으로 확인)**
  - 작은/얇은 물체, 다중 객체, 배경(Stuff)에서 누락/과분할/배경 오염 여부
  - 프롬프트 변화에 대한 반응성(예: "person", "car", "food" 등)
  - teacher 비교 시: top-k 마스크의 대략적 coverage와 노이즈 수준

### 3.7 학습 종료 후 용량 확보 (Artifact Pruning)

> **목표**: 최종 체크포인트만 남기고 중간 산출물(대용량)을 정리하여 디스크 용량 확보.
> (정리 작업은 모델 최종 성능 확인 + 내보내기 산출물 저장 후에만 수행)

- [ ] **정리 전 반드시 보존할 것**
  - 최종 이미지 증류 체크포인트: `checkpoints/distillation/phase2_epoch2_step<FINAL>.pt`
  - (비디오 증류 완료 시) 최종 비디오 증류 체크포인트: `checkpoints/video_distillation/video_epoch*_step*.pt`
  - (양자화/배포 진행 시) 양자화 모델 산출물 + `.pte` + 성능 리포트

- [ ] **대표적인 용량 회수 대상**
  - `checkpoints/distillation/phase1_*.pt`, `checkpoints/distillation/phase2_*.pt` 중 "중간 step" 파일들 (각 ~1.1GB)
  - `logs/distillation/vis/*.png` (시각화 이미지)
  - TensorBoard 이벤트 파일(`logs/distillation/**/events.*`) (생성된 경우)
  - (비디오 증류 종료 후, 재학습 계획이 없고 용량이 급할 때) `data/sa_v/cached_features/*.pt` (~12GB)

- [ ] **정리 스크립트 (기본: dry-run)**
  - 스크립트: `scripts/prune_artifacts.py`
  - 예시:
    ```bash
    # 무엇이 얼마나 지워지는지 먼저 확인
    python scripts/prune_artifacts.py \
      --prune-distillation-ckpt --prune-vis --prune-tensorboard \
      --keep-last-n 1

    # 실제 삭제
    python scripts/prune_artifacts.py \
      --apply \
      --prune-distillation-ckpt --prune-vis --prune-tensorboard \
      --keep-last-n 1

    # (선택) 교사 캐시까지 삭제
    python scripts/prune_artifacts.py --apply --prune-cache
    ```

---

## 4단계: 양자화 (Quantization with TorchAO)

> **목표**: 모델 크기 추가 압축 및 NPU 연산 가속.
> FP16 → Int4/Int8 변환으로 모델 크기 1/4, 추론 속도 2배 이상 향상.
> **정확도 기준**: FP16 대비 mIoU 2% 이내 하락.

### 4.1 가중치 양자화 (Weight-Only Quantization — Int4 Group-wise)

- [x] **TorchAO Int4 Group-wise 양자화 적용** ✅ (코드 완성)
  - `quantize_model.py --mode int4`: Int4WeightOnlyConfig(group_size=128)
  - 민감 레이어 보호: iou_head, dot_product_scoring, perceiver_resampler, memory_cross_attn (14 Linear → FP16 유지)
  - 양자화 대상: 79 Linear layers (RepViT, MobileCLIP, DETR, Mask Decoder)
  - `should_quantize` filter_fn으로 선택적 양자화 적용

- [x] **양자화 후 모델 크기 확인** ✅ (코드 완성)
  - `quantize_model.py --mode compare`: FP16 vs Int4 vs Int8+Int4 비교표 출력
  - 모델 크기, mIoU, Presence F1, 추론 시간 비교

### 4.2 동적 활성화 양자화 (Dynamic Activation Quantization — Int8)

- [x] **TorchAO Int8 Dynamic 활성화 양자화 적용** ✅ (코드 완성)
  - `quantize_model.py --mode int8_int4`: Int8DynamicActivationInt4WeightConfig()
  - 동일한 민감 레이어 보호 적용

### 4.3 양자화 인지 학습 (QAT — Quantization-Aware Training)

- [x] **QAT 적용 여부 판단** ✅ (코드 완성)
  - `quantize_model.py --mode compare` 결과에서 mIoU 2% 초과 하락 시 경고 출력

- [x] **QAT 학습 실행** ✅ (코드 완성)
  - `train_qat.py`: TorchAO QATConfig(step="prepare") → 학습 → QATConfig(step="convert")
  - SA-1B 데이터셋으로 1~2 에포크 fine-tuning (lr=1e-5, warmup=100)
  - Phase 2 distillation loss (output-only) 사용
  - 전체 모델 unfreeze + fake quant → 양자화 노이즈 적응

### 4.4 양자화 정확도 검증

- [x] **SA-1B 검증 세트 평가** ✅ (코드 완성)
  - SA1BAssessmentDataset: SA-1B 마지막 N장을 검증 세트로 사용
  - mIoU (GreedyMatcher 매칭), Presence F1, 추론 시간 측정
  - RLE 디코딩 (compressed + uncompressed 형식 지원)
  - NaN/Inf 출력 검증

- [x] **민감 레이어 Mixed Precision** ✅ (구현 완료)
  - iou_head (3 Linear), dot_product_scoring (4 Linear), perceiver_resampler (6 Linear), memory_cross_attn (1 Linear)
  - 총 14개 Linear 레이어 FP16 유지, 나머지 79개 양자화

> **실행 대기**: Phase 2 distillation 완료 후 실행
> ```bash
> # Step 1: PTQ 비교
> python quantize_model.py --mode compare --num-val 200
>
> # Step 2: mIoU 2% 초과 시만
> python train_qat.py --mode int4 --epochs 2
> ```

---

## 5단계: ExecuTorch 배포 (Deployment)

> **목표**: 양자화된 Mobile-SAM 3 모델을 iOS/Android에서 실시간 구동.
> PyTorch 네이티브 경로(ExecuTorch)를 통해 NPU 가속 바이너리(.pte) 생성.

### 5.1 ExecuTorch Lowering Pipeline

- [ ] **Step 1: Export (내보내기)**
  ```python
  import torch
  from torch.export import export

  # ATen 연산자 단위로 그래프 캡처
  exported_model = export(model, example_inputs)
  ```

- [ ] **Step 2: To Edge (엣지 변환)**
  ```python
  from executorch.exir import to_edge

  # Edge Dialect IR로 변환 (불필요 연산 제거, 메모리 레이아웃 최적화)
  edge_model = to_edge(exported_model)
  ```

- [ ] **Step 3: Partition & Delegate (분할 및 위임)**
  - NPU 연산: Conv, MatMul 등 → CoreML/QNN 파티셔너로 위임
  - CPU 폴백: 복잡한 제어 흐름, DETR 동적 쿼리 처리 → XNNPACK
  ```python
  # 플랫폼별 파티셔너 적용 (아래 5.2, 5.3에서 상세)
  edge_model = edge_model.to_backend(partitioner)
  ```

- [ ] **Step 4: Memory Planning & .pte 생성**
  ```python
  # 정적 메모리 할당 (런타임 동적 malloc 오버헤드 제거)
  et_program = edge_model.to_executorch()

  with open("mobile_sam3.pte", "wb") as f:
      f.write(et_program.buffer)
  ```

### 5.2 iOS: CoreML Backend (ANE 가속)

- [ ] **CoreML 파티셔너 설정**
  ```python
  from executorch.backends.apple.coreml.partition import (
      CoreMLPartitioner, CoreMLCompileSpec
  )
  import coremltools as ct

  partitioner = CoreMLPartitioner(
      compile_spec=CoreMLCompileSpec(
          compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + NPU(ANE) 모두 활용
          precision=ct.Precision.FLOAT16
      )
  )
  ```

- [ ] **SDPA 연산자 매핑 확인**
  - `torch.nn.functional.scaled_dot_product_attention` → CoreML 레이어 매핑
  - ANE에서 Multi-Head Attention 효율적 실행 보장

- [ ] **iOS .pte 파일 생성 및 데스크탑 검증**

- [ ] **Xcode 프로젝트 통합** (Swift/C++)
  - ExecuTorch iOS 라이브러리 탑재
  - 모델 바이너리(.pte) 번들 포함
  - A18 Pro ANE 35 TOPS 활용 (FP16/Int8 최적화)

### 5.3 Android: QNN Backend (Hexagon NPU)

- [ ] **QNN 파티셔너 설정**
  ```python
  from executorch.backends.qualcomm.partition import QnnPartitioner

  partitioner = QnnPartitioner(
      # Snapdragon 8 Elite HTP(Hexagon Tensor Processor) 타겟
      # 양자화된 모델 전달하여 DSP 가속 활성화
  )
  ```

- [ ] **QNN 개발 환경 구축**
  - Qualcomm AI Hub Docker 이미지 활용 권장
  - QNN SDK 설치 및 환경 설정

- [ ] **Android .pte 파일 생성 및 검증**

- [ ] **Android Studio 프로젝트 통합** (Kotlin/JNI)
  - ExecuTorch Android 라이브러리 탑재
  - 모델 바이너리(.pte) 에셋 포함
  - Snapdragon 8 Elite NPU 45+ TOPS 활용

### 5.4 온디바이스 프로파일링 및 성능 최적화

- [ ] **추론 지연시간(Latency) 측정**
  - 이미지: 단일 프레임 추론 시간 (목표: < 100ms)
  - 비디오: FPS (목표: > 15 FPS)

- [ ] **메모리 사용량 프로파일링**
  - 피크 메모리 사용량 측정
  - Perceiver Resampler 메모리 고정(K=64) 확인
  - 장시간 비디오에서 OOM 발생 여부 테스트

- [ ] **NPU 활용률 확인**
  - iOS: Xcode Instruments → Neural Engine 활용률
  - Android: Snapdragon Profiler → Hexagon NPU 활용률
  - CPU 폴백 비율 최소화 (목표: < 10% 연산만 CPU)

- [ ] **병목 구간 최적화**
  - NPU 미지원 연산자 식별 및 대체
  - 텐서 타일링(Tiling) 최적화 (Hexagon TCM 활용)
  - 전처리/후처리 파이프라인 GPU 가속

- [ ] **최종 성능 리포트 작성**
  | 지표 | 목표값 |
  |------|--------|
  | 모델 크기 | < 50MB (.pte) |
  | 이미지 추론 | < 100ms |
  | 비디오 FPS | > 15 FPS |
  | mIoU (SA-Co) | 교사 대비 85%+ |
  | 피크 메모리 | < 500MB |
  | NPU 활용률 | > 90% |

---

## 참고: 메모리 관리 전략 (24GB UMA)

> 개발 환경(Mac Mini M4 Pro)에서의 메모리 관리 지침

- 모델 로드 (FP16): ~1.7GB
- KV Cache 및 Activation: 비디오 처리 시 수 GB~십수 GB
- 시스템 오버헤드 (macOS + 개발 도구): ~4~6GB
- **Batch Size = 1 유지** (스와핑 방지)
- **Activity Monitor에서 Memory Pressure 노란색 미만 유지**
- **불필요한 브라우저 탭 닫기** (2~4GB 확보 가능)

---

## 참고: 핵심 트러블슈팅

| 에러 | 원인 | 해결 |
|------|------|------|
| `RuntimeError: Triton packages are not available` | NVIDIA 전용 경로 사용 | `device="mps"` 명시 지정, CUDA import 비활성화 |
| `pin_memory` 관련 오류 | MPS + pin_memory 비호환 | `DataLoader(pin_memory=False)` |
| NPU 폴백 과다 | 미지원 연산자 | 연산자 대체 또는 XNNPACK 파티셔닝 |
| OOM (Out of Memory) | 메모리 뱅크 선형 증가 | Perceiver Resampler (K=64) 적용 |
