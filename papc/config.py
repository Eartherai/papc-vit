"""Per-dataset hyper-parameters, published baselines, and condition factories.

Single source of truth shared by the CLI, the experiment sessions, and the
analysis scripts. Values are calibrated for ViT-Base at 224 px on an A100 40GB.
"""

from __future__ import annotations

# Per-dataset: epochs / batch size / learning rate.
CFG = {
    "pathmnist":      dict(ep=7,  bs=128, lr=1.5e-4),
    "bloodmnist":     dict(ep=18, bs=128, lr=1.5e-4),
    "dermamnist":     dict(ep=22, bs=96,  lr=1e-4),
    "breastmnist":    dict(ep=35, bs=32,  lr=5e-5),
    "pneumoniamnist": dict(ep=22, bs=96,  lr=1e-4),
    "retinamnist":    dict(ep=35, bs=32,  lr=5e-5),
    "organamnist":    dict(ep=10, bs=128, lr=1e-4),
    "organcmnist":    dict(ep=12, bs=128, lr=1e-4),
}

# Published specialist baselines: {dataset: {model: (AUC, ACC)}}.
PUB = {
    "pathmnist":      {"MedMamba-B": (0.999, 0.964), "MedViT-S": (0.993, 0.942)},
    "bloodmnist":     {"MedMamba-S": (0.999, 0.984), "MedViT-S": (0.997, 0.951)},
    "dermamnist":     {"MedViT-S": (0.937, 0.780), "MedMamba-B": (0.925, 0.757)},
    "pneumoniamnist": {"MedViT-S": (0.995, 0.961), "MedMamba-S": (0.976, 0.936)},
    "retinamnist":    {"MedMamba-X": (0.719, 0.570), "MedViT-S": (0.773, 0.561)},
    "breastmnist":    {"MedViT-S": (0.938, 0.897), "MedMamba-B": (0.849, 0.891)},
    "organamnist":    {"MedMamba-B": (0.998, 0.953), "MedViT-S": (0.997, 0.944)},
    "organcmnist":    {"MedMamba-S": (0.997, 0.925), "MedViT-S": (0.995, 0.917)},
}


# --- Model condition factories (kwargs for PAPCViT) ---------------------------
def mk_vanilla():
    """No predictive coding — plain fine-tuned ViT-Base."""
    return dict(use_pc=False)


def mk_fixed(w=0.05):
    """Fixed-weight PC auxiliary loss."""
    return dict(use_pc=True, pred_loss_weight=w)


def mk_prog(w=0.005):
    """Progressive (cosine-warmed) fixed-weight PC."""
    return dict(use_pc=True, pred_loss_weight=w, progressive=True)


def mk_adaptive():
    """Learnable per-layer PC weights."""
    return dict(use_pc=True, adaptive=True, w_max=0.01, w_l2=1e-3)


def mk_papc():
    """PAPC: progressive + adaptive (the proposed method)."""
    return dict(use_pc=True, adaptive=True, progressive=True, w_max=0.01, w_l2=1e-3)


def mk_clamped(w=0.05):
    """Gate-clamped PC: auxiliary loss on, error integration off."""
    return dict(use_pc=True, pred_loss_weight=w, clamp_gate=True)


CONDITIONS = {
    "vanilla": mk_vanilla,
    "fixed": mk_fixed,
    "prog": mk_prog,
    "adaptive": mk_adaptive,
    "papc": mk_papc,
    "clamped": mk_clamped,
}
