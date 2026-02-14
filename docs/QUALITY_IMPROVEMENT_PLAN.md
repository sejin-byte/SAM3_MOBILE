# SAM3_M4 품질 개선 실행안 (RESULT_REPORT 반영판)

> 작성일: `2026-02-12`  
> 기준 문서: `docs/RESULT_REPORT_20260212.md`  
> 코드 근거: `distillation/*.py`, `models/text_encoder_mobileclip.py`, `quantize_model.py`

---

## 1) 현재 상태 요약 (리포트 기준)

| 항목 | 현재 값 | 비고 |
|---|---:|---|
| mIoU (fp16 baseline) | 0.0605 | `quantize_model.py --mode compare --num-val 200` 결과 |
| Presence F1 | 0.0000 | 지표 계산 방식 점검 필요 |
| Video distill 완료 | Yes | `video_epoch4_step4595.pt` |
| 시각 검증 폴더 생성 | Yes | 15 prompts x 25 images |
| 시각 품질 육안 확정 | No | HTML 생성 확인만 완료 |

핵심: 파이프라인은 끝까지 동작하지만, 품질 지표가 낮아 바로 배포 품질로 보기 어렵다.

---

## 2) 원인 재정리 (확정/가설 구분)

### 확정 1: 텍스트 입력 파이프라인 불일치 + 텍스트 샘플 사용 버그

근거:
- `distillation/dataset.py:144-147`에서 HF tokenizer로 만든 `input_ids` 하나만 반환
- `distillation/trainer.py:299-312`에서 teacher/student가 동일 `input_ids` 사용
- `models/text_encoder_mobileclip.py:37`에서 student는 open_clip tokenizer 체계 사용
- `distillation/dataset.py:128-130`에서 뽑은 `text`가 실제 tokenization에 사용되지 않음

영향:
- 프롬프트 의미 전달이 깨져 프롬프트 반응성이 거의 사라질 가능성이 매우 높다.

### 확정 2: 프롬프트 다양성 부족

근거:
- `distillation/config.py:34-39` 프롬프트가 4개 범용 문장뿐

영향:
- `person/car/building` 등 카테고리 프롬프트를 학습에서 거의 보지 못함.

### 확정 3: Matcher 전량 매칭(100개)으로 과분할 학습 유도 가능

근거:
- `distillation/greedy_matcher.py:136-137` 기본 매칭 수 = `min(ns, nt)` = 100

영향:
- 저품질/빈 query까지 매칭되어 마스크 과생성 경향 강화 가능.

### 보정 필요 1: Presence F1=0 해석

기존 문서의 "head 학습 부재" 단정은 과함.  
평가 파이프라인에서 GT 없는 샘플 취급 방식 때문에 F1이 왜곡될 가능성이 있다.

정리:
- Presence 학습 자체 문제일 수도 있고
- 평가 계산 방식 문제일 수도 있으므로 먼저 지표 로직을 보정해야 한다.

### 보정 필요 2: Video distill 후 image mIoU 정체 해석

`RESULT_REPORT_20260212.md`에서 video distill 후 mIoU가 동일(0.0605)인 것은 이상 징후일 수 있지만,
현재 image-only eval에서는 video memory 경로 이득이 반영되지 않기 쉽다.

정리:
- Video distill 실패로 단정하지 말고, image 품질 개선은 image distill 파이프라인에서 먼저 해결.

---

## 3) 우선순위 로드맵 (실행 순서 고정)

## Phase 0: 지표 신뢰성 보정 (반드시 선행)

### 0-1. Presence F1 평가 로직 보정
- `quantize_model.py`의 평가에서 GT empty/negative 케이스 처리 방식 정리
- "정말 모델이 항상 positive인지"와 "평가가 그렇게 보이게 만드는지" 분리

### 0-2. Teacher 기준선 벤치마크 추가
- 동일 데이터/동일 프롬프트 세팅으로 teacher mIoU/F1 측정
- 이후 student 개선을 절대값 + teacher 대비 두 축으로 추적

### 0-3. 프롬프트 반응성 정량 지표 추가
- 동일 이미지에 다중 프롬프트 적용 시 mask 차이(예: IoU diversity) 계산
- "폴더 생성됨"이 아니라 "반응성이 수치로 존재함"을 확인

완료 기준:
- 개선 전/후 비교표에 `teacher baseline + student + prompt sensitivity`가 함께 표시됨

---

## Phase 1: 즉시 코드 수정 (최우선)

### 1-1. Dual tokenizer 분리 (최우선)

변경:
- dataset에서 `teacher_input_ids`, `student_input_ids`를 분리 생성
- trainer에서 teacher/student에 각자 올바른 ids 전달

대상:
- `distillation/dataset.py`
- `distillation/trainer.py`
- `train_distill.py` (tokenizer wiring)

### 1-2. 텍스트 샘플 실제 사용 버그 수정

변경:
- `prompt_type=="text"`일 때 선택한 `text`를 실제 tokenization 입력으로 사용
- geometric prompt의 텍스트 정책도 일관화(예: 고정 neutral prompt 또는 명시 전략)

대상:
- `distillation/dataset.py`

### 1-3. 프롬프트 세트 확장

변경:
- 4개 범용 프롬프트 -> 범용 + 카테고리형 프롬프트로 확대
- 학습 샘플링 비율 점검

대상:
- `distillation/config.py`

### 1-4. Matcher 제한

변경:
- `max_matches` 도입(예: 20~30)
- teacher low-confidence query 필터링 옵션 추가

대상:
- `distillation/greedy_matcher.py`
- `distillation/trainer.py` (호출 파라미터)

---

## Phase 2: 짧은 실험(AB) 후 본학습

권장 실험 순서:
1. Baseline(현행)
2. +Tokenizer/Prompt bug fix
3. +Prompt 확장
4. +Matcher 제한

각 실험마다:
- mIoU
- Presence F1
- Prompt sensitivity
- 시각 검증(최소 person/car/building 3개)

의사결정:
- 지표와 시각 품질이 동시에 개선된 조합만 본학습으로 승격

---

## Phase 3: 본학습 및 후속 파이프라인 재실행

1. Image distill 재학습 (Phase1 -> Phase2)  
2. Visual eval 재생성  
3. 필요 시 Video distill 재실행  
4. Quantization 재평가  
5. Stage5 export 갱신

---

## Phase 4: 고비용 개선 (조건부)

아래는 Phase 1~3로도 목표 미달일 때만 진행:
- Negative sample 강화
- GT-aware auxiliary loss
- Teacher 1024 / Student 504 분리 학습
- 아키텍처 확장(text-conditioning 경로 강화)
- 데이터 확장(32K -> 200K+)

---

## 4) 기대 성능 (보수적 추정)

| 단계 | 예상 mIoU | 프롬프트 반응성 | 비고 |
|---|---:|---|---|
| 현재 | ~0.06 | 거의 없음 | 기준선 |
| Phase 1 완료 | 0.12~0.25 | 분명한 차이 시작 | 가장 높은 ROI |
| Phase 2~3 완료 | 0.20~0.40 | 주요 카테고리 반응 | 학습 안정화 필요 |
| Phase 4 포함 | 0.35+ | 안정적 | 시간/리소스 고비용 |

주의:
- 위 범위는 현재 코드/데이터 제약에서의 실무적 추정치
- "즉시 0.7" 목표는 현재 조건에서 비현실적

---

## 5) 구현 체크리스트

| 우선순위 | 작업 | 파일 | 완료 기준 | 상태(2026-02-12) |
|---|---|---|---|---|
| P0 | Presence 평가 보정 | `quantize_model.py` | F1 계산 근거 명확 | 완료 |
| P0 | Teacher baseline 추가 | `quantize_model.py` 또는 별도 eval script | teacher/student 동시 비교표 | 완료 |
| P0 | Prompt sensitivity 지표 추가 | `quantize_model.py` | 프롬프트 반응성 수치화 | 완료 |
| P1 | dual tokenizer 분리 | `distillation/dataset.py`, `distillation/trainer.py` | teacher/student ids 분리 전달 | 완료 |
| P1 | text prompt 사용 버그 수정 | `distillation/dataset.py` | 선택 prompt가 실제 입력됨 | 완료 |
| P1 | prompt 확장 | `distillation/config.py` | 카테고리 prompt 포함 | 완료 |
| P1 | matcher 제한 | `distillation/greedy_matcher.py` | max_matches/filter 적용 | 완료 |
| P2 | AB 실험 자동화 | `train_distill.py`/run scripts | 조합별 비교 로그 생성 | 진행 예정 |

검증 메모:
- 문법 검증: `python -m py_compile` (관련 파일 전체 통과)
- 토크나이저 스모크: teacher=16 token / student=77 token 확인
- SA-V 스모크: `student_input_ids` 배치 shape 정상 확인

---

## 6) Go / No-Go 기준

Go (다음 단계 진행):
- mIoU가 baseline 대비 유의미 상승
- Prompt sensitivity가 0에 가깝지 않음
- 시각 검증에서 프롬프트별 결과 차이가 확인됨

No-Go (원인 재점검):
- P1 완료 후에도 mIoU < 0.12
- 프롬프트별 출력이 거의 동일
- Presence F1가 여전히 계산 로직 문제인지 모델 문제인지 분리 안 됨

---

## 7) 즉시 실행 권장 순서

1. Phase 0 지표 보정 먼저 적용  
2. Phase 1 (tokenizer/prompt/matcher) 코드 수정  
3. 짧은 AB 실험으로 유효 조합 결정  
4. 본학습 재실행 및 시각/정량 재평가

---

## 8) 체크포인트 기반 실행 커맨드

아래는 **처음부터 재학습이 아니라**, 기존 체크포인트를 출발점으로 이어서 개선하는 실행 예시.

### 8-1. 개선 후 fp16/teacher 비교 재평가

```bash
conda run -n sam3_mobile python quantize_model.py \
  --mode compare \
  --device cpu \
  --checkpoint checkpoints/final/student_phase2_video_merged_20260211_235518.pt \
  --num-val 200 \
  --include-teacher-baseline \
  --eval-prompt "segment everything" \
  --prompt-sensitivity-prompts "segment everything,person,car,building" \
  --prompt-sensitivity-batches 10
```

### 8-2. 이미지 distillation 재학습 (체크포인트 이어받기)

```bash
conda run -n sam3_mobile python train_distill.py \
  --phase 2 \
  --resume checkpoints/distillation/phase2_epoch2_step24417.pt \
  --device mps
```

### 8-3. 비디오 distillation 재실행 (개선 student 기반)

```bash
conda run -n sam3_mobile python train_video_distill.py \
  --student-ckpt checkpoints/distillation/phase2_epoch2_step24417.pt \
  --device mps
```
