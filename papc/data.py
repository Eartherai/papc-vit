"""Datasets, augmentation, EMA, and evaluation metrics.

Covers the eight MedMNIST benchmarks used in the paper (via the ``medmnist``
package) plus CIFAR-100 sub-samples for the cross-domain study. All images are
resized to 224 px to match the pretrained ViT-Base backbone.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score

# ImageNet normalisation stats (backbone was pretrained on ImageNet).
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def build_tf(sz: int, in_chans: int, train: bool):
    """Build train/eval transforms; grayscale inputs use single-channel stats."""
    m, s = (IMAGENET_STATS if in_chans == 3
            else ((sum(IMAGENET_STATS[0]) / 3,), (sum(IMAGENET_STATS[1]) / 3,)))
    if train:
        aug = T.RandAugment(2, 9) if in_chans == 3 else T.RandomAffine(10, (0.05, 0.05))
        return T.Compose([
            T.Resize(int(sz * 1.15), interpolation=3),
            T.RandomResizedCrop(sz, scale=(0.7, 1.0), interpolation=3),
            T.RandomHorizontalFlip(), aug, T.ToTensor(), T.Normalize(m, s),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    return T.Compose([
        T.Resize(int(sz * 1.15), interpolation=3),
        T.CenterCrop(sz), T.ToTensor(), T.Normalize(m, s)])


class EMA:
    """Exponential moving average of model weights, applied at eval time."""

    def __init__(self, model, decay: float = 0.9995):
        self.decay = decay
        self.sh = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.sh[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.sh[k] = v.detach().clone()

    def apply(self, model):
        bk = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.sh, strict=True)
        return bk

    def restore(self, model, bk):
        model.load_state_dict(bk, strict=True)


def compute_metrics(yt, ys, task, nc):
    """Return (accuracy, macro-AUC) handling binary / multi-class / multi-label."""
    yt = np.asarray(yt)
    ys = np.nan_to_num(np.asarray(ys, dtype=np.float64))
    if task == "multi-label, binary-class":
        yc = np.clip(ys, 0, 1)
        acc = float(((yc > 0.5).astype(int) == yt).mean())
        a = [roc_auc_score(yt[:, c], yc[:, c]) for c in range(nc)
             if 0 < yt[:, c].sum() < len(yt)]
        return acc, float(np.mean(a)) if a else float("nan")
    y1 = yt.squeeze().astype(np.int64)
    if ys.ndim > 1:
        rs = ys.sum(1, keepdims=True)
        yn = ys / np.where(rs > 0, rs, 1)
        yp = ys.argmax(1)
    else:
        yn = ys
        yp = (ys > 0.5).astype(int)
    acc = float((yp == y1).mean())
    if task == "binary-class" or nc == 2:
        sc = yn if yn.ndim == 1 else yn[:, 1]
        try:
            return acc, (float(roc_auc_score(y1, sc)) if len(np.unique(y1)) > 1
                         else float("nan"))
        except Exception:
            return acc, float("nan")
    pc = []
    for c in range(nc):
        yb = (y1 == c).astype(int)
        if 0 < yb.sum() < len(yb):
            try:
                pc.append(roc_auc_score(yb, yn[:, c]))
            except Exception:
                pass
    return acc, float(np.mean(pc)) if pc else float("nan")


def ece(yt, ys, nb: int = 15):
    """Expected Calibration Error with ``nb`` equal-width confidence bins."""
    y1 = np.asarray(yt).squeeze().astype(int)
    ys2 = np.nan_to_num(np.asarray(ys, dtype=np.float64))
    if ys2.ndim == 1:
        ys2 = np.stack([1 - ys2, ys2], 1)
    conf = ys2.max(1)
    pred = ys2.argmax(1)
    correct = (pred == y1).astype(float)
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    for i in range(nb):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def load_med(flag: str, output_dir: str = ".", sz: int = 224):
    """Load a MedMNIST dataset (downloads on first use)."""
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    DC = getattr(medmnist, info["python_class"])
    ic = info["n_channels"]
    task = info["task"]
    nc = len(info["label"]) if isinstance(info["label"], dict) else int(info["label"])
    root = os.path.join(output_dir, "medmnist_data")
    os.makedirs(root, exist_ok=True)
    ssz = sz if sz in (28, 64, 128, 224) else 224
    tr = DC(split="train", transform=build_tf(sz, ic, True), download=True, size=ssz, root=root)
    va = DC(split="val", transform=build_tf(sz, ic, False), download=True, size=ssz, root=root)
    te = DC(split="test", transform=build_tf(sz, ic, False), download=True, size=ssz, root=root)
    return tr, va, te, task, nc, ic


def load_cifar(n: int, output_dir: str = ".", sz: int = 224):
    """Load a fixed random CIFAR-100 sub-sample of ``n`` training images."""
    root = os.path.join(output_dir, "cifar")
    os.makedirs(root, exist_ok=True)
    full = torchvision.datasets.CIFAR100(root, True, download=True, transform=build_tf(sz, 3, True))
    te = torchvision.datasets.CIFAR100(root, False, download=True, transform=build_tf(sz, 3, False))
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(0))[:n].tolist()
    return Subset(full, idx), te, te, "multi-class", 100, 3
