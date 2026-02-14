#!/usr/bin/env bash
set -euo pipefail

# Stage 5 deployment command sheet (working commands).
#
# Usage:
#   bash scripts/stage5_deploy_commands.sh
#   EXPORT_CKPT=checkpoints/final/student_phase2_video_merged_*.pt bash scripts/stage5_deploy_commands.sh

LATEST_MERGED="$(ls -t checkpoints/final/student_phase2_video_merged_*.pt 2>/dev/null | head -n 1 || true)"
DEFAULT_EXPORT_CKPT="${LATEST_MERGED}"
if [[ -z "${DEFAULT_EXPORT_CKPT}" ]]; then
  DEFAULT_EXPORT_CKPT="checkpoints/final/student_phase2_video_merged_<timestamp>.pt"
fi

QUANT_CKPT="${QUANT_CKPT:-checkpoints/quantized/quantized_int8_int4.pt}"
EXPORT_CKPT="${EXPORT_CKPT:-${DEFAULT_EXPORT_CKPT}}"
PTE_DIR="${PTE_DIR:-artifacts/executorch}"
IOS_PTE="${IOS_PTE:-$PTE_DIR/mobile_sam3_ios_coreml.pte}"
ANDROID_PTE="${ANDROID_PTE:-$PTE_DIR/mobile_sam3_android_qnn.pte}"

cat <<EOF
[Stage5] Deployment commands

1) Export -> ToEdge -> CoreML partition (iOS)
python scripts/export_executorch.py \\
  --checkpoint ${EXPORT_CKPT} \\
  --backend coreml \\
  --output ${IOS_PTE}

2) Export -> ToEdge -> QNN partition (Android, env에 QNN 모듈 없으면 자동 fallback=none)
python scripts/export_executorch.py \\
  --checkpoint ${EXPORT_CKPT} \\
  --backend qnn \\
  --output ${ANDROID_PTE}

3) If you must pass quantized checkpoint as input (toolchain 이슈로 FP fallback 가능)
python scripts/export_executorch.py \\
  --checkpoint ${QUANT_CKPT} \\
  --backend coreml \\
  --output ${IOS_PTE}

4) Sanity check produced binaries
ls -lh ${IOS_PTE} ${ANDROID_PTE}

5) iOS integration notes
- Xcode app bundle에 .pte 포함
- ExecuTorch iOS runtime 링크
- Instruments로 ANE/CPU fallback 비율 확인

6) Android integration notes
- Android assets에 .pte 포함
- ExecuTorch Android runtime + JNI 연동
- Snapdragon profiler로 NPU offload 확인

7) Target KPI
- model size < 50MB
- image latency < 100ms
- video fps > 15
- NPU utilization > 90%

EOF

if [[ ! -f "${EXPORT_CKPT}" ]]; then
  echo "NOTE: export checkpoint not found yet: ${EXPORT_CKPT}"
fi

if [[ -n "${LATEST_MERGED}" ]]; then
  echo "latest_merged_ckpt=${LATEST_MERGED}"
fi

if [[ ! -f "${QUANT_CKPT}" ]]; then
  echo "note: quantized input ckpt not found (optional): ${QUANT_CKPT}"
fi
