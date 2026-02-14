# SAM3_M4 결과 리포트 템플릿

> 작성일: `YYYY-MM-DD`
> 작성자: `<name>`
> 실행 환경: `Mac Mini M4 Pro / sam3_mobile / torch ...`

## 1) 실행 요약

| 항목 | 값 |
|---|---|
| 최종 이미지 ckpt | `checkpoints/distillation/phase2_epoch2_step....pt` |
| 최종 비디오 ckpt | `checkpoints/video_distillation/video_epoch...pt` |
| 양자화 모드 | `fp16 / int4 / int8_int4 / qat_int4 / qat_int8_int4` |
| 리포트 범위 | `Phase2 -> Video Distill -> Quantization -> Deploy` |

## 2) Stage 3 성능 (학습 모델 품질)

| 모델 | mIoU | Presence F1 | J&F (video) | 비고 |
|---|---:|---:|---:|---|
| Teacher (SAM3) |  |  |  | 기준선 |
| Student (Phase2 final) |  |  | - |  |
| Student (+Video Distill) |  |  |  |  |

### Stage 3 코멘트

- 주요 개선점:
- 실패/회귀 항목:
- 다음 액션:

## 3) 사용자 시각 검증 (Visual QA)

### 실행 정보

- 이미지 manifest: `configs/visual_eval/image_manifest_phase2.txt`
- 프롬프트 세트: `configs/visual_eval/prompts_core.txt`
- 생성 결과: `outputs/visual_eval/<run>/index.html`

### 관찰 체크리스트

| 항목 | Pass/Fail | 메모 |
|---|---|---|
| 작은 물체 분할 안정성 |  |  |
| 다중 객체 분리 품질 |  |  |
| 배경 누수(오탐) |  |  |
| 프롬프트 반응성 |  |  |
| Teacher 대비 품질 차이(시각) |  |  |

## 4) Stage 4 양자화 성능 비교

> `python quantize_model.py --mode compare --num-val 200` 결과 반영

| Mode | Size (MB) | Compression | mIoU | Presence F1 | Avg ms |
|---|---:|---:|---:|---:|---:|
| fp16 |  | 1.0x |  |  |  |
| int4 |  |  |  |  |  |
| int8_int4 |  |  |  |  |  |
| qat_int4 (optional) |  |  |  |  |  |
| qat_int8_int4 (optional) |  |  |  |  |  |

### 양자화 의사결정

- mIoU 하락 기준(2% 이내) 충족 여부:
- 최종 채택 모드:
- QAT 수행 여부 및 근거:

## 5) Stage 5 배포 준비/검증

| 항목 | iOS(CoreML) | Android(QNN) | 비고 |
|---|---|---|---|
| Export 성공 |  |  |  |
| .pte 크기 (MB) |  |  | 목표 `< 50MB` |
| 이미지 추론 지연 (ms) |  |  | 목표 `< 100ms` |
| 비디오 FPS |  |  | 목표 `> 15` |
| NPU 활용률 |  |  | 목표 `> 90%` |

## 6) 산출물 목록

- 체크포인트:
- 양자화 모델:
- 배포 바이너리(.pte):
- 로그:
- 시각 검증 리포트:

## 7) 용량 정리 전/후

| 항목 | 정리 전 | 정리 후 | 절감 |
|---|---:|---:|---:|
| checkpoints/distillation |  |  |  |
| checkpoints/video_distillation |  |  |  |
| logs/distillation/vis |  |  |  |
| data/sa_v/cached_features |  |  |  |
| 전체 디스크 사용량 |  |  |  |

### 정리 실행 로그

- Dry-run: `logs/storage_prune_dry_run_*.log`
- Apply-run: `logs/storage_prune_apply_*.log`

