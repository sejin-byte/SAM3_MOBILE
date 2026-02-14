"""Distillation configuration — all hyperparameters for Stage 3 training."""

from dataclasses import dataclass, field


@dataclass
class DistillationConfig:
    # ── Paths ──
    sa1b_dir: str = "data/sa1b"
    teacher_model_name: str = "jetjodh/sam3"
    checkpoint_dir: str = "checkpoints/distillation"
    log_dir: str = "logs/distillation"

    # ── Dataset ──
    num_train: int = 32558
    num_val: int = 1000
    image_size: int = 504  # 504/14=36 patches (must be multiple of patch_size=14)

    # ── Teacher preprocessing ──
    teacher_mean: tuple = (0.5, 0.5, 0.5)
    teacher_std: tuple = (0.5, 0.5, 0.5)

    # ── Student preprocessing ──
    student_mean: tuple = (0.485, 0.456, 0.406)
    student_std: tuple = (0.229, 0.224, 0.225)

    # ── Dynamic Prompt Ratios (phase 1 / phase 2) ──
    prompt_ratios_phase1: dict = field(default_factory=lambda: {
        "text": 0.50, "point": 0.25, "box": 0.25,
    })
    prompt_ratios_phase2: dict = field(default_factory=lambda: {
        "text": 0.30, "point": 0.35, "box": 0.35,
    })
    text_prompts: list = field(default_factory=lambda: [
        "objects in the image",
        "segment everything",
        "all objects",
        "things and stuff",
        "person",
        "people",
        "face",
        "hand",
        "car",
        "vehicle",
        "road",
        "building",
        "house",
        "tree",
        "sky",
        "animal",
        "food",
        "chair",
        "table",
        "phone",
        "screen",
    ])
    neutral_prompt: str = "segment everything"
    teacher_text_max_length: int = 16
    student_text_max_length: int = 77

    # ── Loss weights ──
    mask_loss_weight: float = 5.0
    box_l1_loss_weight: float = 5.0
    box_giou_loss_weight: float = 2.0
    logit_loss_weight: float = 2.0
    iou_token_loss_weight: float = 2.0
    presence_loss_weight: float = 1.0
    semantic_seg_loss_weight: float = 2.0
    fpn_feature_loss_weight: float = 1.0
    encoder_feature_loss_weight: float = 1.0

    # ── Greedy Matcher cost weights ──
    matcher_mask_iou_weight: float = 1.0
    matcher_box_l1_weight: float = 1.0
    matcher_logit_sim_weight: float = 1.0
    matcher_max_matches: int = 30
    matcher_teacher_score_threshold: float = 0.05

    # ── Training — Phase 1: Feature Alignment ──
    phase1_epochs: int = 1
    phase1_lr: float = 1e-4
    phase1_warmup_steps: int = 500

    # ── Training — Phase 2: Output Refinement ──
    phase2_epochs: int = 3
    phase2_lr: float = 5e-5

    # ── Shared training ──
    batch_size: int = 4
    grad_accum_steps: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False
    gradient_checkpointing: bool = True
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 10
    vis_every_n_steps: int = 200
    vis_dir: str = "logs/distillation/vis"
    vis_top_k: int = 5
