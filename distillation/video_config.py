"""Video distillation configuration — hyperparameters for temporal memory training."""

from dataclasses import dataclass


@dataclass
class VideoDistillationConfig:
    # -- Paths --
    sa_v_dir: str = "data/sa_v"
    cache_dir: str = "data/sa_v/cached_features"
    checkpoint_dir: str = "checkpoints/video_distillation"
    log_dir: str = "logs/video_distillation"

    # -- Dataset --
    annotation_type: str = "manual"
    context_frames: int = 8
    max_objects_per_frame: int = 20
    image_size: int = 504

    # -- Student preprocessing --
    student_mean: tuple = (0.485, 0.456, 0.406)
    student_std: tuple = (0.229, 0.224, 0.225)

    # -- Text prompts (for text encoder input) --
    text_prompts: list = None

    def __post_init__(self):
        if self.text_prompts is None:
            self.text_prompts = [
                "objects in the video",
                "segment everything",
                "all objects",
                "things and stuff",
            ]

    # -- Loss weights --
    mask_loss_weight: float = 5.0
    box_l1_loss_weight: float = 5.0
    box_giou_loss_weight: float = 2.0
    iou_token_loss_weight: float = 2.0
    presence_loss_weight: float = 1.0

    # -- Greedy Matcher cost weights --
    matcher_mask_iou_weight: float = 1.0
    matcher_box_l1_weight: float = 1.0
    matcher_logit_sim_weight: float = 1.0

    # -- Training --
    epochs: int = 5
    lr: float = 1e-4
    warmup_steps: int = 200
    batch_size: int = 1
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False  # MPS: must be False
    save_every_n_steps: int = 500
    log_every_n_steps: int = 10
