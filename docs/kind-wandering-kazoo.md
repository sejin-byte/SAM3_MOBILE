# Plan: Stage 3 — EfficientSAM3 Knowledge Distillation

## Context
Stage 2 완료: EfficientSAM3 학생 모델 (100.3M params, 201 MB FP16, ~248ms MPS inference).
교사 모델(Sam3Model, 840M)에서 학생 모델로 지식 증류(PHD)를 구현합니다.
SA-1B 33,558장을 사용하여 교사의 출력과 중간 피처를 학생에게 전달합니다.

## 데이터셋
- **SA-1B**: 33,558 images 로컬 (`data/sa1b/sa_XXXXXXX.jpg` + `.json`)
- 텍스트 프롬프트 없음 → generic prompts 사용 ("objects in the image" 등)
- Train: 32,558장 / Val: 1,000장 (마지막 1000장)

## 핵심 기술 과제

### 1. Dual Preprocessing
- **Teacher**: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], resize=1008
- **Student**: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], resize=1008
- 하나의 PIL 이미지에서 양쪽 텐서를 동시에 생성

### 2. Hungarian Matching (100 vs 200 queries)
- Cost: mask IoU + box L1 + logit similarity
- `scipy.optimize.linear_sum_assignment`

### 3. FPN Spatial Mismatch
- Teacher: [288,288], [144,144], [72,72] / Student: [252,252], [126,126], [63,63]
- F.interpolate로 학생을 교사 크기에 맞춤

### 4. MPS 최적화
- FP16 모델, FP32 loss / pin_memory=False / batch_size=1, grad_accum=8

## 파일 구조

```
distillation/
  __init__.py
  config.py                      # DistillationConfig
  dataset.py                     # SA1BDistillDataset (dual preprocessing)
  hungarian.py                   # HungarianMatcher
  losses.py                      # DistillationLoss (8 terms)
  trainer.py                     # DistillationTrainer
train_distill.py                 # Entry point
```

## Loss Functions

| Loss | Weight | Description |
|------|--------|-------------|
| mask_loss | 5.0 | Dice + BCE on matched pred_masks |
| box_l1_loss | 5.0 | L1 on matched pred_boxes |
| box_giou_loss | 2.0 | GIoU on matched pred_boxes |
| logit_loss | 2.0 | BCE on matched pred_logits |
| presence_loss | 1.0 | MSE on presence_logits |
| semantic_seg_loss | 2.0 | Dice + BCE on semantic_seg |
| fpn_feature_loss | 1.0 | MSE on FPN features (Phase 1 only) |
| encoder_feature_loss | 1.0 | MSE on DETR encoder output (Phase 1 only) |

## Training Phases

### Phase 1: Feature Alignment (5 epochs, lr=1e-4)
- Cosine schedule, warmup 500 steps
- Feature alignment + output losses 동시 사용
- FPN 3 levels + DETR encoder alignment (MSE)

### Phase 2: Output Refinement (10 epochs, lr=5e-5)
- Cosine schedule
- Output losses only (feature alignment 제거)

## Memory Budget: ~5-6GB / 24GB UMA

## 구현 순서 (8 steps)

### Step 1: `distillation/__init__.py`
### Step 2: `distillation/config.py` — DistillationConfig dataclass
### Step 3: `distillation/dataset.py` — SA1BDistillDataset
- SA-1B 파일 스캔, dual preprocessing, generic text prompts
### Step 4: `distillation/hungarian.py` — HungarianMatcher
- 200→100 query matching, cost matrix: mask+box+logit
### Step 5: `distillation/losses.py` — DistillationLoss
- 8 loss terms, spatial interpolation, FP32 계산
### Step 6: `distillation/trainer.py` — DistillationTrainer
- Teacher/Student manual forward for intermediate capture
- Gradient accumulation, checkpointing, TensorBoard
### Step 7: `train_distill.py` — CLI entry point
### Step 8: 검증 — 1 epoch 실행, loss 수렴 확인

## 검증 방법
1. Dataset: 33,558장 로드, dual tensor 생성
2. Forward: teacher + student 동시, intermediates 캡처
3. Matching: 100 matched pairs 정상
4. Loss: 8 terms 모두 finite
5. Training: 1 epoch loss 감소
6. Memory: < yellow pressure

## 참조 파일
- 학생 모델: `models/efficient_sam3.py` (Stage 2 완료)
- 교사 모델: `.../transformers/models/sam3/modeling_sam3.py`
- 데이터: `data/sa1b/sa_*.jpg` + `sa_*.json`
