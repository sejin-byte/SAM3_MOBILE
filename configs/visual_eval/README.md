# Visual Eval Set

Prepared assets for user-facing visual QA:

- `configs/visual_eval/image_manifest_phase2.txt`
  - 이미지 경로 목록 (한 줄 1개)
- `configs/visual_eval/prompts_core.txt`
  - 검증용 텍스트 프롬프트 목록

## Generate / refresh image manifest

```bash
python scripts/prepare_visual_eval_set.py \
  --sa1b-dir data/sa1b \
  --output configs/visual_eval/image_manifest_phase2.txt \
  --num-images 24 \
  --scan-limit 3000 \
  --include-baseline
```

## Run visual eval batch (after checkpoint is ready)

```bash
bash scripts/run_visual_eval_from_manifest.sh \
  --student-ckpt checkpoints/distillation/phase2_epoch2_step24417.pt \
  --compare-teacher
```

Outputs are written to `outputs/visual_eval/`.

