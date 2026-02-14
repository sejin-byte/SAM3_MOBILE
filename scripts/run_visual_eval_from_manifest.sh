#!/usr/bin/env bash
set -euo pipefail

# Run visual_eval_student.py for each prompt against a manifest of images.
# Default behavior: student-only columns (original | student).
#
# Usage:
#   bash scripts/run_visual_eval_from_manifest.sh \
#     --student-ckpt checkpoints/distillation/phase2_epoch2_step24417.pt
#
# Optional:
#   --compare-teacher   # original | teacher | student
#   --video-ckpt PATH   # accepted but note: image-only forward does not use it

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STUDENT_CKPT=""
VIDEO_CKPT=""
MANIFEST="configs/visual_eval/image_manifest_phase2.txt"
PROMPTS="configs/visual_eval/prompts_core.txt"
OUT_ROOT="outputs/visual_eval"
DEVICE="mps"
TOP_K="5"
THRESHOLD="0.5"
ALPHA="0.5"
COMPARE_TEACHER="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-ckpt) STUDENT_CKPT="$2"; shift 2 ;;
    --video-ckpt) VIDEO_CKPT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --compare-teacher) COMPARE_TEACHER="1"; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "${STUDENT_CKPT}" ]]; then
  echo "--student-ckpt is required"
  exit 2
fi

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "student ckpt not found: ${STUDENT_CKPT}"
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}"
  exit 2
fi

if [[ ! -f "${PROMPTS}" ]]; then
  echo "prompts file not found: ${PROMPTS}"
  exit 2
fi

mkdir -p "${OUT_ROOT}"
run_tag="$(date '+%Y%m%d_%H%M%S')"

# Build repeated --image args from manifest.
image_args=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" =~ ^# ]] && continue
  image_args+=(--image "${line}")
done < "${MANIFEST}"

if [[ ${#image_args[@]} -eq 0 ]]; then
  echo "No images found in manifest: ${MANIFEST}"
  exit 2
fi

while IFS= read -r prompt; do
  [[ -z "${prompt}" ]] && continue
  [[ "${prompt}" =~ ^# ]] && continue

  slug="$(echo "${prompt}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//')"
  out_dir="${OUT_ROOT}/${run_tag}_${slug}"

  cmd=(
    conda run --no-capture-output -n sam3_mobile
    python scripts/visual_eval_student.py
    --student-ckpt "${STUDENT_CKPT}"
    --prompt "${prompt}"
    --out-dir "${out_dir}"
    --device "${DEVICE}"
    --top-k "${TOP_K}"
    --threshold "${THRESHOLD}"
    --alpha "${ALPHA}"
  )

  if [[ -n "${VIDEO_CKPT}" ]]; then
    cmd+=(--video-ckpt "${VIDEO_CKPT}")
  fi
  if [[ "${COMPARE_TEACHER}" == "1" ]]; then
    cmd+=(--compare-teacher)
  fi

  cmd+=("${image_args[@]}")

  echo "Running prompt: ${prompt}"
  "${cmd[@]}"
done < "${PROMPTS}"

echo
echo "Done. Outputs under: ${OUT_ROOT}"

