"""EfficientSAM3 Student Model Verification Script.

Checks:
1. Model instantiation with pretrained backbone/text encoder
2. Parameter count within expected range (50-200M)
3. FP16 memory usage
4. MPS forward pass with batch_size=1, 1008x1008 input
5. Output shape validation (5 keys)
6. Component-level parameter breakdown
"""

import sys
import time
import torch

# MPS setup
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32


def count_params(module, name=""):
    """Count parameters and print breakdown."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name:40s} {total/1e6:8.2f}M params ({trainable/1e6:.2f}M trainable)")
    return total


def main():
    print("=" * 70)
    print("  EfficientSAM3 Student Model Verification")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device:  {DEVICE}")
    print(f"  Dtype:   {DTYPE}")
    print()

    # -- 1. Instantiation --
    print("[1/6] Instantiating model...")
    t0 = time.time()

    from models import EfficientSAM3, EfficientSAM3Config
    config = EfficientSAM3Config()
    model = EfficientSAM3(config)

    print(f"      Done in {time.time() - t0:.1f}s")

    # -- 2. Parameter count --
    print("\n[2/6] Parameter breakdown:")
    total = 0
    total += count_params(model.vision_encoder, "Vision Encoder (RepViT + FPN)")
    total += count_params(model.text_encoder, "Text Encoder (MobileCLIP + Proj)")
    total += count_params(model.perceiver_resampler, "Perceiver Resampler")
    total += count_params(model.detr_encoder, "DETR Encoder (3L)")
    total += count_params(model.detr_decoder, "DETR Decoder (3L, 100Q)")
    total += count_params(model.dot_product_scoring, "Dot Product Scoring")
    total += count_params(model.mask_decoder, "Mask Decoder")
    print(f"  {'TOTAL':40s} {total/1e6:8.2f}M params")

    assert 50_000_000 < total < 200_000_000, f"Parameter count {total/1e6:.1f}M outside expected range [50M, 200M]"
    print(f"  [OK] Parameter count within expected range")

    # -- 3. FP16 memory --
    print("\n[3/6] FP16 memory estimate:")
    fp16_bytes = sum(p.numel() * 2 for p in model.parameters())
    fp16_mb = fp16_bytes / 1_000_000
    print(f"  FP16 model size: {fp16_mb:.0f} MB")
    assert fp16_mb < 500, f"FP16 memory {fp16_mb:.0f} MB exceeds 500 MB limit"
    print(f"  [OK] Below 500 MB threshold (teacher: 1,681 MB)")

    # -- 4. Move to device --
    print(f"\n[4/6] Moving model to {DEVICE} ({DTYPE})...")
    t0 = time.time()
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    print(f"      Done in {time.time() - t0:.1f}s")

    # -- 5. Forward pass --
    print("\n[5/6] Running forward pass (batch=1, 1008x1008)...")

    # Create dummy inputs
    pixel_values = torch.randn(1, 3, 1008, 1008, device=DEVICE, dtype=DTYPE)

    # Tokenize a test prompt using the model's tokenizer
    tokenizer = model.text_encoder.tokenizer
    input_ids = tokenizer(["a blue object"]).to(DEVICE)

    # Warmup
    with torch.no_grad():
        _ = model(pixel_values, input_ids)

    # Timed run
    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(pixel_values, input_ids)
    if DEVICE == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    print(f"      Inference time: {elapsed * 1000:.1f} ms")

    # -- 6. Output validation --
    print("\n[6/6] Output validation:")
    expected_keys = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg"]
    num_queries = config.detr_decoder_num_queries  # 100

    for key in expected_keys:
        assert key in outputs, f"Missing output key: {key}"
        val = outputs[key]
        print(f"  {key:20s} shape={list(val.shape)} dtype={val.dtype}")

    # Shape checks
    assert outputs["pred_masks"].dim() == 4, "pred_masks should be [B, Q, H, W]"
    assert outputs["pred_masks"].shape[0] == 1, "batch size should be 1"
    assert outputs["pred_masks"].shape[1] == num_queries, f"expected {num_queries} queries"

    assert outputs["pred_boxes"].shape == (1, num_queries, 4), "pred_boxes should be [B, Q, 4]"
    assert outputs["pred_logits"].shape == (1, num_queries), "pred_logits should be [B, Q]"
    assert outputs["presence_logits"].shape[0] == 1, "presence_logits batch should be 1"
    assert outputs["semantic_seg"].dim() == 4, "semantic_seg should be [B, 1, H, W]"
    assert outputs["semantic_seg"].shape[1] == 1, "semantic_seg channels should be 1"

    print(f"\n  [OK] All 5 output keys present with correct shapes")

    # -- Summary --
    print("\n" + "=" * 70)
    print("  VERIFICATION PASSED")
    reduction_pct = (1 - total / 840_000_000) * 100
    print(f"  Total params:    {total/1e6:.1f}M (teacher: 840M, reduction: {reduction_pct:.0f}%)")
    print(f"  FP16 memory:     {fp16_mb:.0f} MB (teacher: 1,681 MB)")
    print(f"  Inference time:  {elapsed * 1000:.1f} ms")
    print(f"  Output keys:     {expected_keys}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
