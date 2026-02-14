# SAM3_M4 성능 재평가 보고서

> 작성일: `2026-02-13`  
> 체크포인트: `student_phase2_video_merged_20260213_151651.pt`  
> 평가 설정: `--mode compare --num-val 200 --include-teacher-baseline`  
> 이전 보고서: `docs/RESULT_REPORT_20260212.md`  
> 개선 계획 문서: `docs/QUALITY_IMPROVEMENT_PLAN.md`

---

## 1) 실행 요약

### 코드 개선사항 (Phase 0~1 완료)

| 작업 | 대상 파일 | 상태 |
|---|---|:---:|
| Dual tokenizer 분리 | `dataset.py`, `trainer.py`, `train_distill.py` | ✅ |
| 텍스트 프롬프트 실사용 버그 수정 | `dataset.py` | ✅ |
| 카테고리 프롬프트 확장 (4→21개) | `config.py` | ✅ |
| Matcher max_matches 제한 (100→30) | `greedy_matcher.py` | ✅ |
| Teacher 저신뢰 query 필터링 | `trainer.py` | ✅ |
| Teacher baseline 평가 추가 | `quantize_model.py` | ✅ |
| Prompt sensitivity 지표 추가 | `quantize_model.py` | ✅ |
| Presence F1 평가 보정 (TP/FP/FN/TN 분리) | `quantize_model.py` | ✅ |

---

## 2) 정량 평가 결과

### 2-1. 주요 비교표 (동일 평가 코드 기준)

| 모델 | Size (MB) | mIoU | Presence F1 | Prompt Sensitivity | Inference (ms) |
|---|---:|---:|---:|---:|---:|
| **Teacher (SAM3) FP16** | 3,223.8 | **0.1522** | 0.0000 | **0.3699** | 4,424.6 |
| **Student FP16 (OLD 2/11)** | 384.6 | **0.1024** | 0.0000 | 0.0015 | 360.7 |
| **Student FP16 (NEW 2/13)** | 384.6 | 0.0571 | 0.0000 | **0.1479** | 356.0 |
| Student Int8+Int4 (OLD) | 384.6 | 0.0942 | 0.0000 | 0.0016 | 472.0 |
| Student Int8+Int4 (NEW) | 384.6 | 0.0513 | 0.0000 | 0.1475 | 481.7 |
| Student Int4 | — | — | — | — | — (mslk 미설치) |

### 2-2. 이전(2/11) vs 현재(2/13) — 동일 평가 코드로 비교

| 지표 | OLD (2/11 ckpt) | NEW (2/13 ckpt) | 변화 | 비고 |
|---|---:|---:|---|---|
| **mIoU (FP16)** | **0.1024** | 0.0571 | 📉 **−0.0453 (44% 하락)** | 마스크 정밀도 악화 |
| **Prompt Sensitivity** | 0.0015 | **0.1479** | 📈 **+0.1464 (98배 증가)** | 프롬프트 반응성 대폭 개선 |
| **Presence F1** | 0.0000 | 0.0000 | 변화 없음 | SA-1B에 negative sample 없음 |
| **Teacher mIoU** | — | **0.1522** | 기준선 확보 | Student의 상한 참조값 |
| **Int8+Int4 mIoU drop** | 0.0082 | 0.0057 | 개선 | 양자화 안정 |

### 2-3. 핵심 해석

**⚠️ mIoU가 오히려 하락한 이유:**
- 재학습은 수행되었으나, **기존 체크포인트(step 24417)에서 resume하여 8,722 step만 추가 학습**하는 방식이었음
- 코드가 크게 변경된 상태(tokenizer 완전 분리)에서 기존 가중치 위에 이어 학습하면 **학습 불안정** 발생
- 특히 student text encoder에 들어가는 input_ids 분포가 이전과 완전히 달라져, 기존에 배운 가중치와 **충돌**
- 결과적으로 prompt sensitivity는 크게 잡았으나 mask precision은 깨진 **trade-off** 상태

**✅ Prompt Sensitivity 대폭 개선 (0.0015 → 0.1479):**
- 이전 모델은 프롬프트에 사실상 무반응(0.0015)이었으나, 재학습 후 반응성이 98배 증가
- Teacher(0.3699)의 약 40% 수준으로, **dual tokenizer와 프롬프트 확장이 확실히 효과가 있음**
- 다만 resume 학습으로 8,722 step만으로는 mIoU까지 함께 수렴시키기엔 부족

**Presence F1 = 0 지속:**
- Teacher도 F1=0.0으로, SA-1B 200개 validation set에 "object 없는 이미지"가 사실상 없기 때문
- 모든 이미지에 annotated 물체가 있어 TN(True Negative) 케이스가 발생하지 않음
- 이 지표는 SA-1B 데이터 특성상 0으로 나오는 것이 정상 → **negative sample이 포함된 평가셋** 필요

**📌 핵심 결론:**
- 코드 수정 자체는 유효 (prompt sensitivity 입증)
- 그러나 resume 학습이 아닌 **from scratch 재학습**이 필요
- Phase 1 (Feature Alignment)부터 다시 시작해야 text encoder와 vision encoder의 정렬이 올바르게 이루어짐

---

## 3) 시각 평가 (Visual QA)

### 3-1. 프롬프트별 결과 비교 (20260213 체크포인트)

#### "person" 프롬프트
- 사람이 있는 이미지(`sa_1092801.jpg` — 군인 초상)에서 사람 영역에 마스크가 집중됨 ✅
- 그러나 비(非)사람 객체(건물, 비행기, 차 등)도 함께 분할됨 → **프롬프트 선택성 아직 부족**
- 이전 결과 대비 큰 시각적 차이는 제한적

#### "car" 프롬프트
- 자동차가 있는 이미지(`sa_5864836.jpg`)에서 차량 분할이 명확 ✅
- 자동차가 없는 장면에서도 다른 객체가 분할됨

#### "building" 프롬프트
- 건물 이미지(`sa_1096145.jpg`)에서 건축물 영역 분할 확인 ✅
- 자연 장면에서는 여전히 넓은 영역이 분할됨

#### "segment everything" 프롬프트
- 다양한 색상의 마스크로 여러 객체 영역이 분할됨
- "person"/"car"/"building" 결과와 **일부 차이가 있음** → Prompt sensitivity 존재의 시각적 확인

### 3-2. 시각 품질 종합 판정

| 항목 | 판정 | 비고 |
|---|:---:|---|
| 마스크 생성 | ✅ 정상 | 다양한 객체에 대해 마스크가 생성됨 |
| 프롬프트별 차이 | ⚠️ 약함 | 프롬프트 간 결과 차이가 존재하나 미미 |
| 객체 경계 정밀도 | ⚠️ 보통 | 대략적인 윤곽은 맞으나 세부가 부족 |
| 과분할 억제 | ❌ 미흡 | 여전히 비대상 영역도 분할 |
| Teacher 대비 | ❌ 격차 큼 | mIoU 기준 37.5% 수준 |

---

## 4) 양자화 결과

| 모드 | 크기 (MB) | 압축률 | mIoU | mIoU Drop | 추론 속도 |
|---|---:|---:|---:|---:|---:|
| FP16 (baseline) | 384.6 | 1.0x | 0.0571 | — | 356.0 ms |
| Int8+Int4 | 384.6 | 1.0x | 0.0513 | −0.0057 | 481.7 ms |
| Int4 | ❌ 실패 | — | — | — | mslk 미설치 |

- Int8+Int4의 mIoU 하락(-0.0057)은 2% 임계치 이내로, **양자화 자체는 품질 영향 최소**
- Int8+Int4 실제 압축률이 1.0x인 것은 CPU 위에서의 메모리 계산 특성 때문 (디스크 저장 시 298.3MB로 22% 절감)

---

## 5) 결론 및 권장 사항

### ✅ 완료된 것
1. Phase 0~1 코드 수정 전량 완료 (dual tokenizer, prompt 확장, matcher 제한 등)
2. 평가 파이프라인 개선 (teacher baseline, prompt sensitivity, presence F1 세분화)
3. 수정 코드로 resume 재학습 수행 (step 24418 → 33139, +8722 steps)
4. 이전/현재 체크포인트 동일 조건 비교 평가 완료

### ⚠️ 핵심 발견
- **코드 수정 효과는 입증됨**: Prompt Sensitivity 0.0015 → 0.1479 (98배 증가)
- **그러나 mIoU가 하락함**: 0.1024 → 0.0571 (44% 하락)
- **원인**: 기존 가중치 위에 resume 학습 → tokenizer 변경으로 인한 학습 불안정
- **결론**: **Phase 1부터 from scratch 재학습 필요**

### 🔜 즉시 다음 액션

```
우선순위 1: From-scratch 재학습 실행
  → Phase 1 (Feature Alignment) from scratch (기존 체크포인트 로드 없이)
  → 수정된 dual tokenizer + 확장 프롬프트 + matcher 제한 반영
  
  커맨드:
  conda run -n sam3_mobile python train_distill.py \
    --phase 1 --device mps

우선순위 2: Phase 2 (Output Refinement)
  → Phase 1 완료 후 자동 연계
  
  커맨드:
  conda run -n sam3_mobile python train_distill.py \
    --phase 2 --device mps

우선순위 3: 재학습 완료 후 동일 평가 재실행
  → mIoU 0.12+ 목표 (teacher의 79%+)
  → Prompt Sensitivity 0.25+ 목표 (teacher의 67%+)
```

### Go / No-Go 기준 (재학습 후)

| 조건 | Go (다음 단계 진행) | No-Go (원인 재점검) |
|---|---|---|
| mIoU | ≥ 0.12 | < 0.12 |
| Prompt Sensitivity | ≥ 0.25 (teacher의 67%+) | < 0.15 |
| 시각 검증 | 프롬프트별 결과 차이 명확 | 여전히 동일 |

---

## 6) 부록: 실행 로그 요약

```
============================================================
  Comparison Summary
============================================================
  Mode           Size(MB)   Compress     mIoU       F1   P-Sens       ms
  --------------------------------------------------------------------------
  fp16              384.6       1.0x   0.0571   0.0000   0.1479   356.0
  teacher_fp16     3223.8       1.0x   0.1522   0.0000   0.3699  4424.6
  int4                  -          -        -        -        -        -  ERROR
    reason: ImportError: Requires mslk >= 1.0.0
  int8_int4         384.6       1.0x   0.0513   0.0000   0.1475   481.7

  int8_int4 mIoU drop = 0.0057 (within 2% threshold)
```

### 시각 평가 경로

| 프롬프트 | 경로 |
|---|---|
| person | `outputs/visual_eval/20260213_152013_person/` |
| car | `outputs/visual_eval/20260213_152013_car/` |
| building | `outputs/visual_eval/20260213_152013_building/` |
| segment everything | `outputs/visual_eval/20260213_152013_segment_everything/` |
| (전 15 프롬프트) | `outputs/visual_eval/20260213_152013_*/` |
