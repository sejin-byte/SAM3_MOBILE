#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for quality improvement plan:
# Phase 0 -> Phase 2 image distill -> video distill -> Stage4 -> visual eval -> Stage5.
#
# Example:
#   bash scripts/run_quality_upgrade_phase0_to_end.sh
#
# Important env vars:
#   DEVICE=mps
#   OFFLINE=1
#   RUN_PHASE0=1
#   RUN_PHASE2=1
#   RUN_VIDEO=1
#   RUN_STAGE4=1
#   RUN_VISUAL=1
#   RUN_STAGE5=1
#
#   PHASE2_RESUME=checkpoints/distillation/phase2_epoch2_step24417.pt
#   PHASE2_EPOCHS=1
#   PHASE2_MAX_STEPS=
#   PHASE2_NUM_WORKERS=0
#
#   VIDEO_EPOCHS=5
#   VIDEO_MAX_STEPS=
#   VIDEO_NUM_WORKERS=0
#
#   NUM_VAL=200
#   SKIP_ASSESSMENT=0
#   RUN_QAT=0
#   QAT_MODE=int4
#   QAT_EPOCHS=2
#
#   VIS_MANIFEST=configs/visual_eval/image_manifest_phase2.txt
#   VIS_PROMPTS=configs/visual_eval/prompts_core.txt

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/quality_upgrade
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="logs/quality_upgrade/full_pipeline_${RUN_TAG}.log"
touch "$MASTER_LOG"
# Mirror all stdout/stderr to master log for reliable postmortem and monitoring.
# Some sandboxes disallow /dev/fd process substitution; fall back to direct append.
if [[ "${QUALITY_UPGRADE_NO_TEE:-0}" == "1" ]]; then
  exec >> "$MASTER_LOG" 2>&1
else
  if ! exec > >(tee -a "$MASTER_LOG") 2>&1; then
    echo "WARN: tee mirroring unavailable; using direct MASTER_LOG append."
    exec >> "$MASTER_LOG" 2>&1
  fi
fi

DEVICE="${DEVICE:-mps}"
OFFLINE="${OFFLINE:-1}"

RUN_PHASE0="${RUN_PHASE0:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_VIDEO="${RUN_VIDEO:-1}"
RUN_STAGE4="${RUN_STAGE4:-1}"
RUN_VISUAL="${RUN_VISUAL:-1}"
RUN_STAGE5="${RUN_STAGE5:-1}"

PHASE0_CKPT="${PHASE0_CKPT:-$(ls -t checkpoints/final/student_phase2_video_merged_*.pt 2>/dev/null | head -n 1 || true)}"
if [[ -z "${PHASE0_CKPT}" ]]; then
  PHASE0_CKPT="$(ls -t checkpoints/distillation/phase2_epoch*_step*.pt 2>/dev/null | head -n 1 || true)"
fi
PHASE0_NUM_VAL="${PHASE0_NUM_VAL:-200}"
PHASE0_INCLUDE_TEACHER="${PHASE0_INCLUDE_TEACHER:-1}"
PHASE0_EVAL_PROMPT="${PHASE0_EVAL_PROMPT:-segment everything}"
PHASE0_SENS_PROMPTS="${PHASE0_SENS_PROMPTS:-segment everything,person,car,building}"
PHASE0_SENS_BATCHES="${PHASE0_SENS_BATCHES:-10}"

PHASE2_RESUME="${PHASE2_RESUME:-$(ls -t checkpoints/distillation/phase2_epoch*_step*.pt 2>/dev/null | head -n 1 || true)}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-1}"
PHASE2_MAX_STEPS="${PHASE2_MAX_STEPS:-}"
PHASE2_NUM_WORKERS="${PHASE2_NUM_WORKERS:-0}"

VIDEO_EPOCHS="${VIDEO_EPOCHS:-5}"
VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-}"
VIDEO_NUM_WORKERS="${VIDEO_NUM_WORKERS:-0}"

NUM_VAL="${NUM_VAL:-200}"
SKIP_ASSESSMENT="${SKIP_ASSESSMENT:-0}"
RUN_QAT="${RUN_QAT:-0}"
QAT_MODE="${QAT_MODE:-int4}"
QAT_EPOCHS="${QAT_EPOCHS:-2}"

VIS_MANIFEST="${VIS_MANIFEST:-configs/visual_eval/image_manifest_phase2.txt}"
VIS_PROMPTS="${VIS_PROMPTS:-configs/visual_eval/prompts_core.txt}"
VIS_OUT_ROOT="${VIS_OUT_ROOT:-outputs/visual_eval}"
VIS_TOP_K="${VIS_TOP_K:-5}"
VIS_THRESHOLD="${VIS_THRESHOLD:-0.5}"
VIS_ALPHA="${VIS_ALPHA:-0.5}"
COMPARE_TEACHER="${COMPARE_TEACHER:-0}"

if [[ "${DEVICE}" == "mps" ]]; then
  if ! conda run --no-capture-output -n sam3_mobile python -c "import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)"; then
    echo "WARN: MPS is not available in this runtime. Falling back DEVICE=cpu."
    DEVICE="cpu"
  fi
fi

if [[ "${OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "== Quality Upgrade Full Pipeline =="
echo "ROOT: ${ROOT}"
echo "MASTER_LOG: ${MASTER_LOG}"
echo "DEVICE: ${DEVICE}"
echo "OFFLINE: ${OFFLINE}"
echo "RUN_PHASE0=${RUN_PHASE0} RUN_PHASE2=${RUN_PHASE2} RUN_VIDEO=${RUN_VIDEO} RUN_STAGE4=${RUN_STAGE4} RUN_VISUAL=${RUN_VISUAL} RUN_STAGE5=${RUN_STAGE5}"
echo

LATEST_PHASE2_CKPT="${PHASE2_RESUME}"
LATEST_VIDEO_CKPT=""

if [[ "${RUN_PHASE0}" == "1" ]]; then
  if [[ -z "${PHASE0_CKPT}" || ! -f "${PHASE0_CKPT}" ]]; then
    echo "Phase0 checkpoint not found: ${PHASE0_CKPT}"
    exit 2
  fi
  echo "[Phase 0] Metric baseline compare"
  phase0_cmd=(
    conda run --no-capture-output -n sam3_mobile
    python quantize_model.py
    --mode compare
    --checkpoint "${PHASE0_CKPT}"
    --device "${DEVICE}"
    --num-val "${PHASE0_NUM_VAL}"
    --eval-prompt "${PHASE0_EVAL_PROMPT}"
    --prompt-sensitivity-prompts "${PHASE0_SENS_PROMPTS}"
    --prompt-sensitivity-batches "${PHASE0_SENS_BATCHES}"
  )
  if [[ "${PHASE0_INCLUDE_TEACHER}" == "1" ]]; then
    phase0_cmd+=(--include-teacher-baseline)
  fi
  "${phase0_cmd[@]}"
  echo
fi

if [[ "${RUN_PHASE2}" == "1" ]]; then
  if [[ -z "${PHASE2_RESUME}" || ! -f "${PHASE2_RESUME}" ]]; then
    echo "Phase2 resume checkpoint not found: ${PHASE2_RESUME}"
    exit 2
  fi
  echo "[Phase 2] Image distillation continue"
  phase2_cmd=(
    conda run --no-capture-output -n sam3_mobile
    python train_distill.py
    --phase 2
    --resume "${PHASE2_RESUME}"
    --epochs "${PHASE2_EPOCHS}"
    --device "${DEVICE}"
    --num-workers "${PHASE2_NUM_WORKERS}"
  )
  if [[ -n "${PHASE2_MAX_STEPS}" ]]; then
    phase2_cmd+=(--max-steps "${PHASE2_MAX_STEPS}")
  fi
  "${phase2_cmd[@]}"
  LATEST_PHASE2_CKPT="$(ls -t checkpoints/distillation/phase2_epoch*_step*.pt 2>/dev/null | head -n 1 || true)"
  echo "latest_phase2_ckpt=${LATEST_PHASE2_CKPT}"
  echo
fi

if [[ "${RUN_VIDEO}" == "1" ]]; then
  if [[ -z "${LATEST_PHASE2_CKPT}" || ! -f "${LATEST_PHASE2_CKPT}" ]]; then
    echo "Latest phase2 checkpoint not found: ${LATEST_PHASE2_CKPT}"
    exit 2
  fi
  echo "[Video] Video distillation"
  video_cmd=(
    conda run --no-capture-output -n sam3_mobile
    python train_video_distill.py
    --student-ckpt "${LATEST_PHASE2_CKPT}"
    --epochs "${VIDEO_EPOCHS}"
    --device "${DEVICE}"
    --num-workers "${VIDEO_NUM_WORKERS}"
  )
  if [[ -n "${VIDEO_MAX_STEPS}" ]]; then
    video_cmd+=(--max-steps "${VIDEO_MAX_STEPS}")
  fi
  "${video_cmd[@]}"
  LATEST_VIDEO_CKPT="$(ls -t checkpoints/video_distillation/video_epoch*_step*.pt 2>/dev/null | head -n 1 || true)"
  echo "latest_video_ckpt=${LATEST_VIDEO_CKPT}"
  echo
fi

if [[ "${RUN_STAGE4}" == "1" ]]; then
  if [[ -z "${LATEST_PHASE2_CKPT}" || ! -f "${LATEST_PHASE2_CKPT}" ]]; then
    echo "Stage4 image checkpoint not found: ${LATEST_PHASE2_CKPT}"
    exit 2
  fi
  if [[ -z "${LATEST_VIDEO_CKPT}" || ! -f "${LATEST_VIDEO_CKPT}" ]]; then
    LATEST_VIDEO_CKPT="$(ls -t checkpoints/video_distillation/video_epoch*_step*.pt 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "${LATEST_VIDEO_CKPT}" || ! -f "${LATEST_VIDEO_CKPT}" ]]; then
    echo "Stage4 video checkpoint not found: ${LATEST_VIDEO_CKPT}"
    exit 2
  fi
  echo "[Stage4] Quantization pipeline"
  IMAGE_CKPT="${LATEST_PHASE2_CKPT}" \
  VIDEO_CKPT="${LATEST_VIDEO_CKPT}" \
  DEVICE="${DEVICE}" \
  NUM_VAL="${NUM_VAL}" \
  SKIP_ASSESSMENT="${SKIP_ASSESSMENT}" \
  RUN_QAT="${RUN_QAT}" \
  QAT_MODE="${QAT_MODE}" \
  QAT_EPOCHS="${QAT_EPOCHS}" \
    bash scripts/stage4_quantization_pipeline.sh
  echo
fi

MERGED_CKPT="$(ls -t checkpoints/final/student_phase2_video_merged_*.pt 2>/dev/null | head -n 1 || true)"

if [[ "${RUN_VISUAL}" == "1" ]]; then
  if [[ -z "${MERGED_CKPT}" || ! -f "${MERGED_CKPT}" ]]; then
    echo "Merged checkpoint for visual eval not found: ${MERGED_CKPT}"
    exit 2
  fi
  echo "[Visual] Visual evaluation gallery"
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
  echo
fi

if [[ "${RUN_STAGE5}" == "1" ]]; then
  echo "[Stage5] Deployment command sheet"
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
fi

echo "All done."
echo "master_log=${MASTER_LOG}"
echo "latest_phase2_ckpt=${LATEST_PHASE2_CKPT}"
echo "latest_video_ckpt=${LATEST_VIDEO_CKPT}"
echo "latest_merged_ckpt=${MERGED_CKPT}"
