# Release Notes - 2026-02-15

## Scope
Closeout release for the current SAM3_MOBILE training cycle.

## Highlights
- Completed From-scratch retraining cycle (Phase1 -> Phase2).
- Finalized project documentation for handoff and release planning.
- Added repository guardrails (`.gitignore`) to exclude local-only artifacts.
- Applied script stability fixes used during evaluation and resume flows.

## Metrics Snapshot
Source: `outputs/reports/retrain_impact_20260215.md`

- mIoU: `0.0570 -> 0.0898` (`+57.46%`)
- Prompt sensitivity (lower better): `0.1479 -> 0.1343`
- Presence F1: `0.0000 -> 0.0000`
- Inference (ms/img): `54.2616 -> 54.2522`

Teacher comparison snapshot:
- Student/Teacher mIoU: `0.5392` (~53.9%)
- Reference: `outputs/reports/teacher_vs_student_miou_tiny_20260215_165149.json`

## Artifacts
- Final distillation checkpoint:
  - `checkpoints/distillation/phase2_epoch2_step24417.pt`
- Final merged checkpoint:
  - `checkpoints/final/student_phase2_video_merged_20260215_161400.pt`
- Quantized checkpoint:
  - `checkpoints/quantized/quantized_int8_int4.pt`

## Known Limitations
- Presence F1 remains 0.0 on current evaluation setup.
- Teacher parity not yet reached.
- Full on-device KPI validation remains pending.

## Deployment Readiness Note
Current state is suitable for internal/demo or limited beta iteration,
not a final public-quality release for broad segmentation use cases.

## Next Iteration Recommendations
1. Extend Phase2 training.
2. Introduce negative samples to improve presence behavior.
3. Complete on-device KPI measurements and tune for target thresholds.
