"""EfficientSAM3 Knowledge Distillation (Stage 3)."""

from .config import DistillationConfig
from .dataset import SA1BDistillDataset
from .prompt_encoder import GeometricPromptEncoder
from .greedy_matcher import GreedyMatcher
from .losses import DistillationLoss
from .trainer import DistillationTrainer
from .video_config import VideoDistillationConfig
from .video_dataset import SAVVideoDataset
from .video_losses import VideoDistillationLoss
from .video_trainer import VideoDistillationTrainer

__all__ = [
    "DistillationConfig",
    "SA1BDistillDataset",
    "GeometricPromptEncoder",
    "GreedyMatcher",
    "DistillationLoss",
    "DistillationTrainer",
    "VideoDistillationConfig",
    "SAVVideoDataset",
    "VideoDistillationLoss",
    "VideoDistillationTrainer",
]
