# Stage4/5 Runbook

비디오 증류 완료 후 사용할 실행 스크립트:

## Stage 4 (Quantization)

```bash
bash scripts/stage4_quantization_pipeline.sh
```

옵션:

```bash
DEVICE=mps NUM_VAL=200 RUN_QAT=1 QAT_MODE=int4 QAT_EPOCHS=2 \
  bash scripts/stage4_quantization_pipeline.sh
```

주요 동작:

1. `phase2` 최종 ckpt + `video_distillation` 최종 ckpt 자동 탐색
2. `scripts/merge_video_modules_into_student_ckpt.py`로 통합 ckpt 생성
3. `quantize_model.py --mode compare` 실행
4. 필요 시 `train_qat.py` 실행

## Stage 5 (ExecuTorch export)

```bash
bash scripts/stage5_deploy_commands.sh
```

실제 export 실행:

```bash
LATEST_MERGED=$(ls -t checkpoints/final/student_phase2_video_merged_*.pt | head -n 1)

# iOS CoreML partition
python scripts/export_executorch.py \
  --checkpoint "$LATEST_MERGED" \
  --backend coreml \
  --output artifacts/executorch/mobile_sam3_ios_coreml.pte

# Android QNN partition (QNN 모듈 없으면 자동 fallback=none)
python scripts/export_executorch.py \
  --checkpoint "$LATEST_MERGED" \
  --backend qnn \
  --output artifacts/executorch/mobile_sam3_android_qnn.pte
```

검증:

```bash
ls -lh artifacts/executorch/mobile_sam3_ios_coreml.pte artifacts/executorch/mobile_sam3_android_qnn.pte
```

참고:
- `checkpoints/quantized/quantized_int8_int4.pt`를 입력으로 주면 현재 툴체인 제약으로 FP merged ckpt로 fallback될 수 있음
- fallback 여부는 `.meta.json`에서 `quantized_export_fallback` 값으로 확인
