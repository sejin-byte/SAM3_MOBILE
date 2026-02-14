#!/usr/bin/env bash
set -euo pipefail

# Wait for Phase 2 image distillation to finish, then start teacher FPN caching.
#
# Usage:
#   bash scripts/wait_phase2_then_cache.sh
#
# Env vars:
#   SLEEP_SECS=300        Poll interval (seconds)
#   EXPECTED_CKPT=...     Override the expected final Phase2 checkpoint path
#   REQUIRE_FINAL=1       If set to 1, only start caching after the final Phase2 checkpoint exists

SLEEP_SECS="${SLEEP_SECS:-300}"
REQUIRE_FINAL="${REQUIRE_FINAL:-1}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

latest_ckpt() {
  ls -t checkpoints/distillation/phase2_*.pt 2>/dev/null | head -n 1 || true
}

mkdir -p data/sa_v/cached_features logs/cache_teacher_features

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

echo "Watching for final Phase 2 checkpoint: ${EXPECTED_CKPT}"
echo "Poll interval: ${SLEEP_SECS}s"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  ckpt="$(latest_ckpt)"

  if [[ -f "${EXPECTED_CKPT}" ]]; then
    echo "[${ts}] Final ckpt detected: ${EXPECTED_CKPT}"
    break
  fi

  if [[ -n "${ckpt}" ]]; then
    echo "[${ts}] Waiting... Latest ckpt: ${ckpt}"
  else
    echo "[${ts}] Waiting... No Phase 2 checkpoints found yet."
  fi
  sleep "${SLEEP_SECS}"
done

final_ckpt="${EXPECTED_CKPT}"

if [[ "${REQUIRE_FINAL}" == "1" ]] && [[ ! -f "${final_ckpt}" ]]; then
  echo "Final checkpoint not found; not starting caching."
  exit 2
fi

cached_count="$(find data/sa_v/cached_features -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
if [[ "${cached_count}" -ge 919 ]]; then
  echo "Teacher cache already complete (${cached_count}/919). Nothing to do."
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Activate your shell environment and re-run."
  exit 3
fi

log_path="logs/cache_teacher_features/cache_teacher_features_$(date '+%Y%m%d_%H%M%S').log"
echo "Starting caching into data/sa_v/cached_features/ (current: ${cached_count}/919)"
echo "Logging to ${log_path}"

conda run --no-capture-output -n sam3_mobile python cache_teacher_features.py --device mps | tee "${log_path}"
