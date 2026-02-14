# Video Distillation Implementation Plan

## Context

Image distillation (Stage 3, Phase 1+2) trains the student backbone, DETR, mask decoder.
Now we need **video distillation** to activate the Perceiver Resampler for temporal memory compression.
Strategies applied: (1) Backbone freeze, (2) Teacher FPN L3 caching ~12GB, (5) 504px resolution.

## Architecture

```
Context T=8 frames → cached FPN L3 [T, 256, 18, 18] → flatten [T*324, 256]
                                                              ↓
                                                    Perceiver Resampler (TRAINABLE, 1.3M)
                                                              ↓
                                                       [batch, 64, 256]
                                                              ↓
Query frame → Student backbone (FROZEN) → FPN → DETR encoder (FROZEN) → [H*W, 256]
                                                              ↓
                                          MemoryCrossAttention(encoder_out, memory) (TRAINABLE, 0.2M)
                                                              ↓
                                          DETR decoder (FROZEN) → masks, boxes
                                                              ↓
                                          Loss vs SA-V GT masks (Dice+BCE, L1+GIoU)
```

Trainable: ~1.5M (Perceiver 1.3M + MemoryCrossAttention 0.2M)
Frozen: backbone, text encoder, DETR encoder, DETR decoder, mask decoder (~99M)

## Training Pipeline (4 Steps)

| Step | Task | Estimated Time |
|------|------|---------------|
| 1. Image Phase 1 | Feature alignment (1 epoch) | ~7.9h (DONE) |
| 2. Image Phase 2 | Output refinement (3 epochs) | ~7.1h |
| 3. Teacher Caching | Cache FPN L3 for SA-V 919 videos | ~16h |
| 4. Video Distillation | Train Perceiver + MemoryCrossAttention (5 epochs) | ~4.5h |
| **Total** | | **~35.5h (~1.5 days)** |

## Files to Create (7)

### 1. `models/memory_attention.py` — MemoryCrossAttention module
- Pre-norm cross-attention: vision features (queries) attend to compressed memory (keys/values)
- Residual connection, batch_first=True (ExecuTorch compatible)
- ~198K params (LayerNorm×2 + MHA(256, 8 heads))

### 2. `cache_teacher_features.py` — Teacher FPN L3 caching script
- Load SAM3 teacher at 504px (with `resize_teacher_rope`)
- For each SA-V video (919 total):
  - Read `_manual.json` to determine annotated frame count
  - Extract frames from MP4 via cv2, resize to 504px
  - Run `teacher.get_vision_features()` → save `fpn_hidden_states[3]` ([256,18,18]) as FP16
- Output: `data/sa_v/cached_features/sav_XXXXXX.pt` per video
- Batch processing (batch=4), resumable (skip existing cache files)
- ~12 GB total, ~16 hours estimated

### 3. `distillation/video_config.py` — VideoDistillationConfig
- SA-V paths, cache dir, annotation_type="manual"
- context_frames=8, max_objects_per_frame=20
- epochs=5, lr=1e-4, warmup=200, batch_size=1, grad_accum=4
- Loss weights: mask=5.0, box_l1=5.0, box_giou=2.0, iou=2.0, presence=1.0

### 4. `distillation/video_dataset.py` — SAVVideoDataset
- `__getitem__`: sample clip (T context + 1 query frame)
  - Context: load cached FPN L3 from .pt file → [T, 256, 18, 18]
  - Query: extract frame from MP4 via cv2, student preprocessing
  - GT: decode RLE masks from SA-V JSON via pycocotools → binary masks + boxes
- `video_collate_fn`: stack pixel/features, keep GT masks as list (variable N per frame)

### 5. `distillation/video_losses.py` — VideoDistillationLoss
- Loss against SA-V GT masks (not teacher predictions)
- Reuse `dice_loss`, `compute_giou` from `distillation/losses.py`
- Match 100 predictions to N GT objects via GreedyMatcher
- Terms: mask (Dice+BCE), box_l1, box_giou, iou_token, presence

### 6. `distillation/video_trainer.py` — VideoDistillationTrainer
- Load student from image distillation checkpoint
- Freeze all → unfreeze only `perceiver_resampler` + `memory_cross_attn`
- `_prepare_memory_features`: [batch, T, 256, 18, 18] → [batch, T*324, 256]
- Training loop follows `distillation/trainer.py` patterns exactly
- Checkpoints save only trainable module state dicts + optimizer
- No teacher model loaded (features pre-cached)

### 7. `train_video_distill.py` — CLI entry point
- `--student-ckpt`, `--resume`, `--debug`, `--device`, `--lr`, `--epochs`
- Auto-find latest image distillation checkpoint if not specified
- Tokenizer from Sam3Processor (same as image distillation)

## Files to Modify (3)

### 8. `models/efficient_sam3.py`
- Import `MemoryCrossAttention`
- Add `self.memory_cross_attn = MemoryCrossAttention(config.hidden_size)` in `__init__`
- Add `forward_video(pixel_values, input_ids, memory_features)` method:
  - Perceiver compress → memory_tokens [batch, 64, 256]
  - Standard vision+text → DETR encoder → encoder_out
  - MemoryCrossAttention(encoder_out, memory_tokens) → enriched features
  - DETR decoder + mask decoder with enriched features → predictions

### 9. `models/__init__.py` — Add MemoryCrossAttention export

### 10. `distillation/__init__.py` — Add video component exports

## Implementation Order

1. `models/memory_attention.py` — standalone, no deps
2. `models/efficient_sam3.py` — add memory_cross_attn + forward_video()
3. `models/__init__.py` — update exports
4. `cache_teacher_features.py` — create and run (~16h)
5. `distillation/video_config.py` — standalone dataclass
6. `distillation/video_dataset.py` — requires cached features
7. `distillation/video_losses.py` — reuse from losses.py
8. `distillation/video_trainer.py` — follows trainer.py patterns
9. `distillation/__init__.py` — update exports
10. `train_video_distill.py` — CLI entry point

## Verification

1. **Cache integrity**: load .pt, assert shape [N, 256, 18, 18] FP16
2. **forward_video smoke test**: random tensors → verify output shapes
3. **Freeze check**: only perceiver_resampler.* and memory_cross_attn.* have requires_grad=True
4. **Loss sanity**: `python train_video_distill.py --debug` → all losses finite, backward OK
5. **Gradient flow**: non-zero gradients for trainable params, zero for frozen

## Key Design Decisions

- **Late fusion (not encoder modification)**: MemoryCrossAttention sits BETWEEN encoder and decoder. Frozen encoder produces normal features; memory enriches them before decoder.
- **GT-based loss**: No teacher online during video training. Teacher only provides cached features for context frames. Supervision from SA-V GT masks.
- **FPN level 3 ([256, 18, 18])**: Lowest-res FPN output. 324 tokens/frame × 8 frames = 2,592 → Perceiver compresses to 64 (40x). Disk: 0.16 MB/frame ≈ 12 GB total.
- **DETR decoder frozen**: Prevents catastrophic forgetting of image capabilities. Can unfreeze later if needed.
- **strict=False for checkpoint loading**: memory_cross_attn is new, won't exist in image distillation checkpoints.
