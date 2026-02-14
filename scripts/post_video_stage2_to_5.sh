#!/usr/bin/env bash
set -euo pipefail

# One-command runner for post-video steps:
# 2) Stage4 quantization pipeline
# 3) (optional) QAT inside Stage4 via RUN_QAT=1
# 4) Visual evaluation gallery generation
# 5) Stage5 deployment command sheet print
#
# Usage:
#   bash scripts/post_video_stage2_to_5.sh
#
# Optional env vars:
#   DEVICE=mps
#   NUM_VAL=200
#   SKIP_ASSESSMENT=0
#   RUN_QAT=0
#   QAT_MODE=int4
#   QAT_EPOCHS=2
#   RUN_VISUAL=1
#   COMPARE_TEACHER=0
#   VIS_OUT_ROOT=outputs/visual_eval
#   VIS_TOP_K=5
#   VIS_THRESHOLD=0.5
#   VIS_ALPHA=0.5
#   VIS_MANIFEST=configs/visual_eval/image_manifest_phase2.txt
#   VIS_PROMPTS=configs/visual_eval/prompts_core.txt

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-mps}"
NUM_VAL="${NUM_VAL:-200}"
SKIP_ASSESSMENT="${SKIP_ASSESSMENT:-0}"
RUN_QAT="${RUN_QAT:-0}"
QAT_MODE="${QAT_MODE:-int4}"
QAT_EPOCHS="${QAT_EPOCHS:-2}"

RUN_VISUAL="${RUN_VISUAL:-1}"
COMPARE_TEACHER="${COMPARE_TEACHER:-0}"
VIS_OUT_ROOT="${VIS_OUT_ROOT:-outputs/visual_eval}"
VIS_TOP_K="${VIS_TOP_K:-5}"
VIS_THRESHOLD="${VIS_THRESHOLD:-0.5}"
VIS_ALPHA="${VIS_ALPHA:-0.5}"
VIS_MANIFEST="${VIS_MANIFEST:-configs/visual_eval/image_manifest_phase2.txt}"
VIS_PROMPTS="${VIS_PROMPTS:-configs/visual_eval/prompts_core.txt}"

echo "== Post-Video Steps 2-5 =="
echo "ROOT: ${ROOT}"
echo "DEVICE: ${DEVICE}"
echo "NUM_VAL: ${NUM_VAL}"
echo "SKIP_ASSESSMENT: ${SKIP_ASSESSMENT}"
echo "RUN_QAT: ${RUN_QAT} (mode=${QAT_MODE}, epochs=${QAT_EPOCHS})"
echo "RUN_VISUAL: ${RUN_VISUAL} (compare_teacher=${COMPARE_TEACHER})"
echo

if [[ "${DEVICE}" == "mps" ]]; then
  if ! conda run --no-capture-output -n sam3_mobile python -c "import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)"; then
    echo "WARN: MPS is not available in current runtime. Falling back DEVICE=cpu."
    DEVICE="cpu"
  fi
fi

echo "[2/5] Stage4 quantization pipeline"
DEVICE="${DEVICE}" \
NUM_VAL="${NUM_VAL}" \
SKIP_ASSESSMENT="${SKIP_ASSESSMENT}" \
RUN_QAT="${RUN_QAT}" \
QAT_MODE="${QAT_MODE}" \
QAT_EPOCHS="${QAT_EPOCHS}" \
  bash scripts/stage4_quantization_pipeline.sh
echo

MERGED_CKPT="$(ls -t checkpoints/final/student_phase2_video_merged_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${MERGED_CKPT}" ]]; then
  echo "No merged checkpoint found under checkpoints/final/"
  exit 2
fi
echo "Merged checkpoint: ${MERGED_CKPT}"

echo "[4/5] Visual evaluation from manifest"
if [[ "${RUN_VISUAL}" == "1" ]]; then
  if [[ "${COMPARE_TEACHER}" == "1" ]]; then
    bash scripts/run_visual_eval_from_manifest.sh \
      --student-ckpt "${MERGED_CKPT}" \
      --manifest "${VIS_MANIFEST}" \
      --prompts "${VIS_PROMPTS}" \
      --device "${DEVICE}" \
      --out-root "${VIS_OUT_ROOT}" \
      --top-k "${VIS_TOP_K}" \
      --threshold "${VIS_THRESHOLD}" \
      --alpha "${VIS_ALPHA}" \
      --compare-teacher
  else
    bash scripts/run_visual_eval_from_manifest.sh \
      --student-ckpt "${MERGED_CKPT}" \
      --manifest "${VIS_MANIFEST}" \
      --prompts "${VIS_PROMPTS}" \
      --device "${DEVICE}" \
      --out-root "${VIS_OUT_ROOT}" \
      --top-k "${VIS_TOP_K}" \
      --threshold "${VIS_THRESHOLD}" \
      --alpha "${VIS_ALPHA}"
  fi
else
  echo "RUN_VISUAL=0 -> skipping visual evaluation"
fi
echo

echo "[5/5] Stage5 deployment command sheet"
QUANT_CKPT="checkpoints/quantized/quantized_int4.pt"
if [[ "${RUN_QAT}" == "1" && -f "checkpoints/quantized/qat_${QAT_MODE}.pt" ]]; then
  QUANT_CKPT="checkpoints/quantized/qat_${QAT_MODE}.pt"
elif [[ -f "checkpoints/quantized/quantized_int4.pt" ]]; then
  QUANT_CKPT="checkpoints/quantized/quantized_int4.pt"
elif [[ -f "checkpoints/quantized/quantized_int8_int4.pt" ]]; then
  QUANT_CKPT="checkpoints/quantized/quantized_int8_int4.pt"
fi

QUANT_CKPT="${QUANT_CKPT}" bash scripts/stage5_deploy_commands.sh
echo

echo "Done. Post-video steps 2-5 completed."
