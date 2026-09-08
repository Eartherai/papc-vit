"""Training / evaluation driver for a single (condition, seed) run.

Fine-tunes a :class:`~papc.model.PAPCViT` with layer-wise LR decay, cosine
schedule, EMA weights, Mixup/CutMix (multi-class only), and test-time
horizontal-flip augmentation. Returns a dict of accuracy, AUC, ECE, mean
absolute gate value, learned per-layer weights, and wall-clock minutes.
"""

from __future__ import annotations

import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import PAPCViT, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE
from .data import EMA, compute_metrics, ece


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_backends():
    """Enable TF32 / cuDNN autotuning and pick an AMP dtype."""
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        has_bf16 = torch.cuda.is_bf16_supported()
    else:
        has_bf16 = False
    return device, (torch.bfloat16 if has_bf16 else torch.float16)


def train_eval(model_kw, tr, va, te, task, nc, ic, epochs, bs, lr, seed,
               model_name=DEFAULT_MODEL_NAME, img_size=DEFAULT_IMG_SIZE,
               save_path=None):
    """Train one model and evaluate it. ``model_kw`` are kwargs for ``PAPCViT``.

    If ``save_path`` is given, the EMA weights and enough config to rebuild the
    model are written there (loadable by ``scripts/predict.py``).
    """
    from timm.data import Mixup

    device, amp_dtype = configure_backends()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    is_ml = task == "multi-label, binary-class"

    model = PAPCViT(model_name, img_size, nc, ic, pretrained=True, **model_kw).to(device)
    use_pc = model_kw.get("use_pc", True)
    nw = min(10, os.cpu_count() or 4)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    tl = DataLoader(tr, bs, shuffle=True, drop_last=True, **kw)
    el = DataLoader(te, bs * 2, **kw)

    # Keyword args ensure each value lands in the right Mixup slot.
    mfn = Mixup(
        mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
        mode="batch", label_smoothing=0.1, num_classes=nc,
    ) if (not is_ml and nc > 2) else None

    opt = torch.optim.AdamW(model.param_groups(lr), weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = len(tl) * epochs
    warmup = max(1, int(total_steps * 0.05))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))))
    ema = EMA(model)
    t0 = time.time()
    gs = 0

    for _ in range(epochs):
        model.train()
        for x, y in tl:
            model.set_progress(gs, total_steps)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yl = y.long().squeeze(-1) if y.ndim > 1 else y.long()
            if mfn and not is_ml:
                x, tgt = mfn(x, yl)
            elif is_ml:
                tgt = y.float()
            else:
                tgt = F.one_hot(yl, nc).float() * 0.9 + 0.1 / nc
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                lo, pl = model(x)
                main = (F.binary_cross_entropy_with_logits(lo, tgt) if is_ml
                        else -(tgt * F.log_softmax(lo, -1)).sum(-1).mean())
                loss = main + pl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            ema.update(model)
            gs += 1

    bk = ema.apply(model)   # evaluate (and save) with EMA weights
    if save_path:
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_kw": model_kw, "num_classes": nc, "in_chans": ic, "task": task,
            "model_name": model_name, "img_size": img_size,
        }, save_path)
    model.eval()
    ys_all, ss_all = [], []
    with torch.no_grad():
        for x, y in el:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                la, _ = model(x)
                lb, _ = model(torch.flip(x, [-1]))
            avg = (la + lb) / 2
            s = torch.sigmoid(avg) if is_ml else F.softmax(avg, dim=-1)
            ys_all.append(y.cpu().numpy())
            ss_all.append(s.float().cpu().numpy())
    ema.restore(model, bk)

    ya = np.concatenate(ys_all)
    sa = np.concatenate(ss_all)
    acc, auc = compute_metrics(ya, sa, task, nc)
    ecv = ece(ya, sa)
    gv = float(np.mean(np.abs(model.gate_values()))) if use_pc else 0.0
    lw = model.learned_weights()

    del model, ema, opt, sch, tl, el
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return dict(acc=acc, auc=auc, ece=ecv, gate=gv, lw=lw, t_min=(time.time() - t0) / 60)
