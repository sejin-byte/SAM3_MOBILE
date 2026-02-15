# SAM3_MOBILE

## 1) 프로젝트 목적
SAM3 Teacher(`jetjodh/sam3`)를 기반으로, 모바일 환경에서 동작 가능한 경량 Student(EfficientSAM3)를 지식 증류하고 양자화하여
- 이미지 분할 품질을 최대한 유지하면서
- 모바일 배포 가능한 형태(`.pt`/`.pte`)로 전환
하는 것을 목표로 진행한 프로젝트입니다.

## 2) 프로젝트 종료 상태 (2026-02-15)
현재 학습은 종료되었고, 본 저장소는 **실험 종료/아카이브 상태**입니다.

- Phase2 최종 이미지 증류 체크포인트: `checkpoints/distillation/phase2_epoch2_step24417.pt`
- 비디오 모듈 병합 체크포인트: `checkpoints/final/student_phase2_video_merged_20260215_161400.pt`
- 양자화 체크포인트: `checkpoints/quantized/quantized_int8_int4.pt`

## 3) 실험 진행 이력 요약

### A. 구조 안정화
- Dual tokenizer 경로(teacher HF tokenizer / student open_clip tokenizer) 정리
- From-scratch 재학습(Phase1 -> Phase2)으로 학습 안정성 회복

### B. 재학습 효과 (Before vs After)
출처: `outputs/reports/retrain_impact_20260215.md`

- mIoU: `0.0570 -> 0.0898` (`+57.46%`)
- Prompt Sensitivity(lower better): `0.1479 -> 0.1343` (`-9.17%`)
- Presence F1: `0.0000 -> 0.0000` (변화 없음)
- Inference(ms/img): `54.2616 -> 54.2522` (거의 동일)

### C. Teacher 대비 성능
출처: `outputs/reports/teacher_vs_student_miou_tiny_20260215_165149.json`

- Student mIoU: `0.0787`
- Teacher mIoU: `0.1459`
- Student/Teacher mIoU: `0.5392` (약 `53.9%`)

### D. 양자화 검증
출처: `logs/quantization/compare_20260215_161405.log`

- FP16 mIoU: `0.0898`
- int8_int4 mIoU: `0.0879`
- mIoU drop: `0.0019` (허용 범위 내)
- int4 단독 모드는 환경 의존성(`mslk`) 이슈로 실패

## 4) 현재 해석
- 재학습으로 품질은 유의미하게 회복됨
- 다만 Teacher 대비 격차가 여전히 큼(약 54%)
- Presence F1가 0.0이라 무객체/오탐 관점의 신뢰도는 아직 낮음
- 따라서 공개 출시보다는 추가 실험 또는 제한적 베타 성격이 적합한 상태

## 5) 배포 가정 기준 점검 항목
배포 전 최종 확인이 필요한 KPI(참고: `scripts/stage5_deploy_commands.sh`)

- model size `< 50MB`
- image latency `< 100ms`
- video fps `> 15`
- NPU utilization `> 90%`

현재 저장소에는 위 KPI의 실기기(on-device) 확정치가 모두 채워진 상태는 아님.

## 6) 실험 재현/확인 커맨드

### 6-1. 학습 완료 체크
```bash
ls -lh checkpoints/distillation/phase2_epoch2_step24417.pt
ps aux | rg "train_distill.py|train_video_distill.py" -n
```

### 6-2. 품질 비교(학생 모델)
```bash
conda run -n sam3_mobile python quantize_model.py \
  --mode compare \
  --checkpoint checkpoints/final/student_phase2_video_merged_20260215_161400.pt \
  --num-val 200 \
  --device mps
```

### 6-3. 배포 커맨드 시트 확인
```bash
bash scripts/stage5_deploy_commands.sh
```

## 7) 프로젝트 종료 메모
- 본 저장소는 학습 실험 결과를 보존하는 목적의 종료 상태
- 다음 사이클에서 성능을 더 끌어올리려면
  - Phase2 연장 학습
  - Negative sample 추가로 Presence head 개선
  - 온디바이스 KPI 실측 기반 튜닝
  순으로 진행 권장
