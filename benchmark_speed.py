"""Benchmark distillation step time at 504px to estimate total training duration."""

import time
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "mps"
NUM_WARMUP = 2
NUM_MEASURE = 5


def main():
    print("=" * 60)
    print("  Distillation Speed Benchmark (504px, batch=4)")
    print("=" * 60)

    # ── Load config ──
    from distillation.config import DistillationConfig
    config = DistillationConfig()
    print(f"  image_size: {config.image_size}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  grad_accum_steps: {config.grad_accum_steps}")
    print(f"  num_train: {config.num_train}")
    print(f"  phase1_epochs: {config.phase1_epochs}")
    print(f"  phase2_epochs: {config.phase2_epochs}")

    steps_per_epoch = config.num_train // config.batch_size
    print(f"  steps_per_epoch: {steps_per_epoch}")
    print()

    # ── Load teacher ──
    print("[1/3] Loading teacher model...")
    t0 = time.time()
    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained(config.teacher_model_name)
    teacher = Sam3Model.from_pretrained(
        config.teacher_model_name, torch_dtype=torch.float16,
    ).to(DEVICE).half()
    teacher.requires_grad_(False)
    teacher.train(False)

    # Resize teacher RoPE for 504px
    from distillation.trainer import resize_teacher_rope
    resize_teacher_rope(teacher, config.image_size)
    print(f"  Teacher loaded in {time.time() - t0:.1f}s")

    # ── Load student ──
    print("[2/3] Loading student model...")
    from models import EfficientSAM3, EfficientSAM3Config
    student = EfficientSAM3(EfficientSAM3Config()).to(DEVICE).float()
    student.train()
    if config.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    print(f"  Student loaded")

    # ── Build dataloader ──
    print("[3/3] Building dataloader...")
    from distillation.dataset import SA1BDistillDataset, collate_fn
    from distillation.prompt_encoder import GeometricPromptEncoder
    from distillation.greedy_matcher import GreedyMatcher
    from distillation.losses import DistillationLoss
    from torch.utils.data import DataLoader

    dataset = SA1BDistillDataset(config, processor.tokenizer, split="train", phase=1)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=False,
        collate_fn=collate_fn, drop_last=True,
    )

    prompt_encoder = GeometricPromptEncoder(hidden_size=256).to(DEVICE).float()
    matcher = GreedyMatcher()
    loss_fn = DistillationLoss(config)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(prompt_encoder.parameters()),
        lr=config.phase1_lr, weight_decay=config.weight_decay,
    )

    print(f"\nRunning {NUM_WARMUP} warmup + {NUM_MEASURE} measured steps...")
    print("-" * 60)

    step_times = []
    data_iter = iter(loader)

    for i in range(NUM_WARMUP + NUM_MEASURE):
        batch = next(data_iter)

        step_start = time.time()

        # Move to device
        teacher_pixels = batch["teacher_pixel_values"].to(DEVICE).half()
        student_pixels = batch["student_pixel_values"].to(DEVICE).float()
        input_ids = batch["input_ids"].to(DEVICE)
        prompt_types = batch["prompt_types"]
        prompt_coords = batch["prompt_coords"].to(DEVICE).float()

        # Teacher forward
        with torch.no_grad():
            vision_outputs = teacher.get_vision_features(teacher_pixels)
            fpn_features = list(vision_outputs.fpn_hidden_states[:-1])
            t_out = teacher(vision_embeds=vision_outputs, input_ids=input_ids, output_hidden_states=True)
            teacher_out = {
                "pred_masks": t_out.pred_masks,
                "pred_boxes": t_out.pred_boxes,
                "pred_logits": t_out.pred_logits,
                "presence_logits": t_out.presence_logits,
                "semantic_seg": t_out.semantic_seg,
                "fpn_features": fpn_features,
                "encoder_output": t_out.encoder_hidden_states[-1] if t_out.encoder_hidden_states else None,
            }

        del teacher_pixels
        torch.mps.empty_cache()

        # Student forward
        prompt_embeddings = prompt_encoder(prompt_types, prompt_coords)
        has_geometric = any(p in ("point", "box") for p in prompt_types)
        if not has_geometric:
            prompt_embeddings = None

        student_out = student.forward_with_intermediates(
            pixel_values=student_pixels, input_ids=input_ids,
            prompt_embeddings=prompt_embeddings,
        )

        # Matching
        bs = student_out["pred_masks"].shape[0]
        all_s_idx, all_t_idx = [], []
        with torch.no_grad():
            for b in range(bs):
                s_idx, t_idx = matcher.match(
                    student_out["pred_masks"][b].float(),
                    teacher_out["pred_masks"][b].float(),
                    student_out["pred_boxes"][b].float(),
                    teacher_out["pred_boxes"][b].float(),
                    student_out["pred_logits"][b].float(),
                    teacher_out["pred_logits"][b].float(),
                )
                all_s_idx.append(s_idx)
                all_t_idx.append(t_idx)

        # Loss + backward
        losses = loss_fn(student_out, teacher_out, all_s_idx, all_t_idx, phase=1)
        scaled_loss = losses["total_loss"] / config.grad_accum_steps
        scaled_loss.backward()

        if (i + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(prompt_encoder.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        # Sync MPS for accurate timing
        torch.mps.synchronize()

        del student_out, teacher_out, losses, scaled_loss
        torch.mps.empty_cache()

        elapsed = time.time() - step_start

        label = "WARMUP" if i < NUM_WARMUP else "MEASURE"
        print(f"  Step {i+1:2d} [{label}]: {elapsed:.2f}s  loss={batch.get('total_loss', '-')}")

        if i >= NUM_WARMUP:
            step_times.append(elapsed)

    # ── Report ──
    avg_step = sum(step_times) / len(step_times)
    min_step = min(step_times)
    max_step = max(step_times)

    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Avg step time:  {avg_step:.2f}s")
    print(f"  Min/Max:        {min_step:.2f}s / {max_step:.2f}s")
    print()

    # Phase 1 estimate
    p1_steps = steps_per_epoch * config.phase1_epochs
    p1_time_s = p1_steps * avg_step
    p1_hours = p1_time_s / 3600
    print(f"  Phase 1: {config.phase1_epochs} epoch x {steps_per_epoch} steps = {p1_steps} total steps")
    print(f"           Estimated: {p1_time_s:.0f}s = {p1_hours:.1f} hours")

    # Phase 2 estimate
    p2_steps = steps_per_epoch * config.phase2_epochs
    p2_time_s = p2_steps * avg_step
    p2_hours = p2_time_s / 3600
    print(f"  Phase 2: {config.phase2_epochs} epochs x {steps_per_epoch} steps = {p2_steps} total steps")
    print(f"           Estimated: {p2_time_s:.0f}s = {p2_hours:.1f} hours")

    total_hours = p1_hours + p2_hours
    print()
    print(f"  TOTAL: {total_hours:.1f} hours ({total_hours / 24:.1f} days)")
    print("=" * 60)


if __name__ == "__main__":
    main()
