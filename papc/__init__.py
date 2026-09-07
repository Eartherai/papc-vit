"""PAPC — Predictive Coding auxiliary losses for pretrained Vision Transformers.

Reference implementation for the paper *"Predictive Coding Auxiliary Losses Are
Calibration Levers, Not Accuracy Levers: A Cross-Domain Empirical Study"*
(CAISc 2026).
"""

from .model import DiagonalSSM, PCLayer, PAPCViT, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE
from .train import train_eval
from .config import (CFG, PUB, mk_vanilla, mk_fixed, mk_prog, mk_adaptive,
                     mk_papc, mk_clamped, CONDITIONS)

__version__ = "1.0.0"

__all__ = [
    "DiagonalSSM", "PCLayer", "PAPCViT", "DEFAULT_MODEL_NAME", "DEFAULT_IMG_SIZE",
    "train_eval", "CFG", "PUB", "CONDITIONS",
    "mk_vanilla", "mk_fixed", "mk_prog", "mk_adaptive", "mk_papc", "mk_clamped",
]
