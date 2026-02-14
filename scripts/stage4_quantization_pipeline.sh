#!/usr/bin/env bash
set -euo pipefail

# Stage 4 quantization pipeline runner (execute AFTER video distillation).
#
# What it does:
# 1) Pick latest phase2 image ckpt + latest video ckpt
# 2) Merge video modules into full student checkpoint
# 3) Run PTQ compare (fp16 vs int4 vs int8_int4)
# 4) Optional QAT (only when RUN_QAT=1)
#
# Usage:
#   bash scripts/stage4_quantization_pipeline.sh
#
# Env vars:
#   DEVICE=mps
#   NUM_VAL=200
#   SKIP_ASSESSMENT=0
#   RUN_QAT=0
#   QAT_MODE=int4            # int4 | int8_int4
#   QAT_EPOCHS=2

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-mps}"
NUM_VAL="${NUM_VAL:-200}"
SKIP_ASSESSMENT="${SKIP_ASSESSMENT:-0}"
RUN_QAT="${RUN_QAT:-0}"
QAT_MODE="${QAT_MODE:-int4}"
QAT_EPOCHS="${QAT_EPOCHS:-2}"

IMAGE_CKPT="${IMAGE_CKPT:-$(ls -t checkpoints/distillation/phase2_epoch2_step*.pt 2>/dev/null | head -n 1 || true)}"
VIDEO_CKPT="${VIDEO_CKPT:-$(ls -t checkpoints/video_distillation/video_epoch*_step*.pt 2>/dev/null | head -n 1 || true)}"

if [[ -z "${IMAGE_CKPT}" ]]; then
  echo "No phase2 final checkpoint found."
  exit 2
fi
if [[ -z "${VIDEO_CKPT}" ]]; then
  echo "No video distillation checkpoint found."
  exit 2
fi

mkdir -p checkpoints/final logs/quantization

if ! conda run --no-capture-output -n sam3_mobile python -c "import mslk" >/dev/null 2>&1; then
  echo "WARN: python package 'mslk' is not available in sam3_mobile."
  echo "      Int4 PTQ may fail with: ImportError: Requires mslk >= 1.0.0"
fi

MERGED_CKPT="checkpoints/final/student_phase2_video_merged_$(date '+%Y%m%d_%H%M%S').pt"
echo "[1/4] Merge image ckpt + video modules"
conda run --no-capture-output -n sam3_mobile python scripts/merge_video_modules_into_student_ckpt.py \
  --image-ckpt "${IMAGE_CKPT}" \
  --video-ckpt "${VIDEO_CKPT}" \
  --output "${MERGED_CKPT}"

echo
echo "[2/4] PTQ compare (fp16/int4/int8_int4)"
COMPARE_LOG="logs/quantization/compare_$(date '+%Y%m%d_%H%M%S').log"
compare_cmd=(
  conda run --no-capture-output -n sam3_mobile
  python quantize_model.py
  --mode compare
  --checkpoint "${MERGED_CKPT}"
  --device "${DEVICE}"
  --num-val "${NUM_VAL}"
)
if [[ "${SKIP_ASSESSMENT}" == "1" ]]; then
  compare_cmd+=(--skip-assessment)
fi
"${compare_cmd[@]}" | tee "${COMPARE_LOG}"

echo
echo "[3/4] PTQ done"
echo "compare_log=${COMPARE_LOG}"
echo "merged_ckpt=${MERGED_CKPT}"

if [[ "${RUN_QAT}" != "1" ]]; then
  echo
  echo "[4/4] QAT skipped (RUN_QAT=${RUN_QAT})"
  echo "If needed: RUN_QAT=1 QAT_MODE=int4|int8_int4 QAT_EPOCHS=1|2 bash scripts/stage4_quantization_pipeline.sh"
  exit 0
fi

echo
echo "[4/4] Run QAT"
QAT_LOG="logs/quantization/qat_${QAT_MODE}_$(date '+%Y%m%d_%H%M%S').log"
conda run --no-capture-output -n sam3_mobile python train_qat.py \
  --mode "${QAT_MODE}" \
  --checkpoint "${MERGED_CKPT}" \
  --device "${DEVICE}" \
  --epochs "${QAT_EPOCHS}" | tee "${QAT_LOG}"

echo "qat_log=${QAT_LOG}"
