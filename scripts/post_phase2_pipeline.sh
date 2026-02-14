#!/usr/bin/env bash
set -euo pipefail

# Post-Phase2 automation:
# 1) Wait for Phase 2 image distillation to fully finish (final checkpoint present + process gone)
# 3) Cache teacher FPN L3 features for SA-V
# 4) Verify cache count + tensor shape/dtype
# 5) Run video distillation smoke test (--debug)
# 6) Run full video distillation
#
# Run this in a detached session (recommended) since caching/training are long:
#   screen -dmS sam3_post_phase2 bash -lc 'cd /path/to/SAM3_M4 && bash scripts/post_phase2_pipeline.sh'
#
# Env vars:
#   SLEEP_SECS=300         Poll interval while waiting for Phase2
#   DEVICE=mps             Device for caching + video distill
#   CACHE_EXPECTED=919     Expected number of cached video feature files

SLEEP_SECS="${SLEEP_SECS:-300}"
DEVICE="${DEVICE:-mps}"
CACHE_EXPECTED="${CACHE_EXPECTED:-919}"
POST_CKPT_SLEEP_SECS="${POST_CKPT_SLEEP_SECS:-60}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

latest_phase2_ckpt() {
  ls -t checkpoints/distillation/phase2_*.pt 2>/dev/null | head -n 1 || true
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 10
  fi
}

require_cmd conda
require_cmd python3

mkdir -p logs/post_phase2_pipeline
LOG_PATH="logs/post_phase2_pipeline/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "== Post Phase2 Pipeline =="
echo "ROOT: $ROOT"
echo "Device: $DEVICE"
echo "Log: $LOG_PATH"
echo "Poll: ${SLEEP_SECS}s"
echo "Post-ckpt sleep: ${POST_CKPT_SLEEP_SECS}s"
echo

# Compute expected final checkpoint from config.
EXPECTED_CKPT="${EXPECTED_CKPT:-}"
if [[ -z "${EXPECTED_CKPT}" ]]; then
  EXPECTED_CKPT="$(PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from distillation.config import DistillationConfig
cfg = DistillationConfig()
steps_per_epoch = cfg.num_train // cfg.batch_size
total_steps = steps_per_epoch * cfg.phase2_epochs
final_epoch = cfg.phase2_epochs - 1
print(f"checkpoints/distillation/phase2_epoch{final_epoch}_step{total_steps}.pt")
PY
  )"
fi

echo "Expected final Phase2 checkpoint: ${EXPECTED_CKPT}"
echo

echo "[WAIT] Phase2 completion (final ckpt appears on disk)"
while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  latest="$(latest_phase2_ckpt)"

  if [[ -f "${EXPECTED_CKPT}" ]]; then
    echo "[${ts}] Final ckpt detected: ${EXPECTED_CKPT}"
    break
  fi

  echo "[${ts}] Waiting... latest_ckpt=${latest:-<none>}"
  sleep "${SLEEP_SECS}"
done

FINAL_CKPT="${EXPECTED_CKPT}"
echo "[OK] Using student checkpoint: ${FINAL_CKPT}"
echo

echo "[WAIT] Final checkpoint write completion (size stabilization)"
prev_size="-1"
stable="no"
for _ in $(seq 1 20); do
  size="$(stat -f%z "${FINAL_CKPT}" 2>/dev/null || echo 0)"
  if [[ "${size}" == "${prev_size}" ]] && [[ "${size}" -gt 0 ]]; then
    stable="yes"
    break
  fi
  prev_size="${size}"
  sleep 15
done
if [[ "${stable}" != "yes" ]]; then
  echo "Final checkpoint size did not stabilize; aborting."
  exit 3
fi

echo "[OK] Final checkpoint size stable. Sleeping ${POST_CKPT_SLEEP_SECS}s to avoid GPU overlap."
sleep "${POST_CKPT_SLEEP_SECS}"
echo

echo "[STEP 3] Cache teacher FPN features (if needed)"
mkdir -p data/sa_v/cached_features logs/cache_teacher_features

cached_count="$(find data/sa_v/cached_features -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
echo "Current cache count: ${cached_count}/${CACHE_EXPECTED}"

if [[ "${cached_count}" -lt "${CACHE_EXPECTED}" ]]; then
  cache_log="logs/cache_teacher_features/cache_teacher_features_$(date '+%Y%m%d_%H%M%S').log"
  echo "Running: conda run -n sam3_mobile python cache_teacher_features.py --device ${DEVICE}"
  echo "Caching log: ${cache_log}"
  conda run --no-capture-output -n sam3_mobile python cache_teacher_features.py --device "${DEVICE}" | tee "${cache_log}"
else
  echo "Cache already complete (skipping)."
fi
echo

echo "[STEP 4] Verify cache"
cached_count="$(find data/sa_v/cached_features -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
echo "Cache count: ${cached_count}/${CACHE_EXPECTED}"
if [[ "${cached_count}" -ne "${CACHE_EXPECTED}" ]]; then
  echo "Cache count mismatch; expected ${CACHE_EXPECTED}, got ${cached_count}"
  exit 2
fi

conda run --no-capture-output -n sam3_mobile python3 - <<'PY'
import glob
import torch

files = sorted(glob.glob("data/sa_v/cached_features/*.pt"))
print(f"cached_files={len(files)}")
t = torch.load(files[0], weights_only=True)
print(f"shape={tuple(t.shape)} dtype={t.dtype}")
assert t.ndim == 4 and t.shape[1:] == (256, 18, 18)
assert t.dtype == torch.float16
print("cache_tensor_ok=1")
PY
echo

echo "[STEP 5] Video distillation smoke test (--debug, 5 steps)"
mkdir -p checkpoints/video_distillation logs/video_distillation
vd_debug_log="logs/video_distillation/video_debug_$(date '+%Y%m%d_%H%M%S').log"
conda run --no-capture-output -n sam3_mobile python train_video_distill.py \
  --student-ckpt "${FINAL_CKPT}" \
  --debug \
  --device "${DEVICE}" | tee "${vd_debug_log}"
echo

echo "[STEP 6] Video distillation full training"
vd_full_log="logs/video_distillation/video_full_$(date '+%Y%m%d_%H%M%S').log"
conda run --no-capture-output -n sam3_mobile python train_video_distill.py \
  --student-ckpt "${FINAL_CKPT}" \
  --device "${DEVICE}" | tee "${vd_full_log}"

echo
echo "All steps completed."
