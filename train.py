"""Training entry point for PAPC.

Contains the dataset/condition configuration, the single-run train+eval driver,
the four experimental sessions from the paper, and the command-line interface.

Reproduce a paper session (resumable, time-budgeted)::

    python train.py --session A --output-dir results_repro

Run an ad-hoc grid on one dataset, saving checkpoints for predict.py::

    python train.py --dataset dermamnist --conditions vanilla fixed:0.05 papc \
        --seeds 3 --save-models checkpoints --output-dir results_repro
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from papc import PAPCViT, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE
from datasets import EMA, compute_metrics, ece, load_med, load_cifar

DEFAULT_TIME_BUDGET_SEC = int(8.5 * 3600)  # safe margin inside a 9h session


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-run training / evaluation
# ---------------------------------------------------------------------------

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
    model are written there (loadable by ``predict.py``).
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


# ---------------------------------------------------------------------------
# Experimental sessions
# ---------------------------------------------------------------------------

  # safe margin inside a 9h session


class Session:
    """Runs a session, persisting results after every completed run."""

    def __init__(self, name, output_dir=".", time_budget_sec=DEFAULT_TIME_BUDGET_SEC,
                 seeds=3):
        self.name = name
        self.output_dir = output_dir
        self.time_budget_sec = time_budget_sec
        self.seeds = seeds
        os.makedirs(output_dir, exist_ok=True)
        self.out = os.path.join(output_dir, f"papc_{name}_results.json")
        self.res = json.load(open(self.out)) if os.path.exists(self.out) else {}
        self.t0 = time.time()

    # -- persistence / budget helpers ----------------------------------------
    def save(self):
        json.dump(self.res, open(self.out, "w"), indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    def time_left(self):
        return self.time_budget_sec - (time.time() - self.t0)

    # -- generic grid runner --------------------------------------------------
    def run_grid(self, tag, datasets, conditions, seeds=None, save_dir=None):
        seeds = self.seeds if seeds is None else seeds
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for flag, cfg in datasets.items():
            if self.time_left() < 600:
                print(f"TIME: skipping {flag}")
                break
            try:
                tr, va, te, task, nc, ic = load_med(flag, self.output_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  {flag}: SKIP ({e})")
                continue
            print(f"\n=== {flag} (n={len(tr)}) ===")
            self.res.setdefault(tag, {}).setdefault(flag, {"n": len(tr)})
            for cname, mkw in conditions:
                self.res[tag][flag].setdefault(cname, [])
                existing = len(self.res[tag][flag][cname])
                for s in range(existing, seeds):
                    if self.time_left() < 300:
                        break
                    sp = (os.path.join(save_dir, f"{flag}_{cname}_s{s}.pt")
                          if save_dir else None)
                    r = train_eval(mkw, tr, va, te, task, nc, ic,
                                   cfg["ep"], cfg["bs"], cfg["lr"], s, save_path=sp)
                    self.res[tag][flag][cname].append(r)
                    print(f"  {cname:14s} s{s}: AUC={r['auc']:.4f} ACC={r['acc']:.4f} "
                          f"ECE={r['ece']:.4f} gate={r['gate']:.3f} ({r['t_min']:.1f}m)")
                self.save()
            for cname, _ in conditions:
                aucs = [r["auc"] for r in self.res[tag][flag].get(cname, [])]
                if aucs:
                    best_pub = max((v[0] for v in PUB.get(flag, {}).values()),
                                   default=float("nan"))
                    print(f"  {cname:14s} mean AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f}"
                          f"  (best pub={best_pub:.3f})")


def run_session_a(output_dir=".", **kw):
    s = Session("A", output_dir, **kw)
    print("SESSION A: Main SOTA table (vanilla / HP w=0.05 / PAPC)")
    s.run_grid("sota", CFG, [
        ("vanilla", mk_vanilla()),
        ("hp_0.05", mk_fixed(0.05)),
        ("PAPC", mk_papc()),
    ], seeds=3)
    s.save()
    return s.out


def run_session_b(output_dir=".", **kw):
    s = Session("B", output_dir, **kw)
    print("SESSION B: Full ablation (6 conditions) + gate-clamping")
    abl_ds = {k: CFG[k] for k in
              ["dermamnist", "pneumoniamnist", "bloodmnist", "breastmnist"] if k in CFG}
    s.run_grid("ablation", abl_ds, [
        ("vanilla", mk_vanilla()),
        ("fixed_0.05", mk_fixed(0.05)),
        ("fixed_0.001", mk_fixed(0.001)),
        ("prog_0.005", mk_prog(0.005)),
        ("adaptive", mk_adaptive()),
        ("PAPC", mk_papc()),
    ], seeds=3)
    print("\n--- Gate-clamping ablation ---")
    clamp_ds = {k: CFG[k] for k in ["dermamnist", "pneumoniamnist"] if k in CFG}
    s.run_grid("clamp", clamp_ds, [
        ("vanilla", mk_vanilla()),
        ("full_005", mk_fixed(0.05)),
        ("clamped", mk_clamped(0.05)),
    ], seeds=3)
    s.save()
    return s.out


def run_session_c(output_dir=".", **kw):
    s = Session("C", output_dir, **kw)
    print("SESSION C: CIFAR-100 cross-domain + MedMNIST weight scaling sweep")

    # C1: CIFAR-100 at three sub-sample sizes.
    for n in [2000, 10000, 50000]:
        if s.time_left() < 800:
            print(f"TIME: skipping CIFAR n={n}")
            break
        tr, va, te, task, nc, ic = load_cifar(n, s.output_dir)
        ep = {2000: 25, 10000: 12, 50000: 6}[n]
        print(f"\n=== CIFAR-100 n={n} ===")
        s.res.setdefault("cifar", {}).setdefault(str(n), {})
        for cname, mkw in [("vanilla", mk_vanilla()), ("PAPC", mk_papc()),
                           ("fixed_001", mk_fixed(0.001)), ("fixed_05", mk_fixed(0.05))]:
            runs = []
            for seed in range(2):
                if s.time_left() < 300:
                    break
                r = train_eval(mkw, tr, va, te, task, nc, ic, ep, 128, 1.5e-4, seed)
                runs.append(r)
                print(f"  {cname:12s} s{seed}: ACC={r['acc']:.4f} AUC={r['auc']:.4f} "
                      f"({r['t_min']:.1f}m)")
            s.res["cifar"][str(n)][cname] = runs
        s.save()

    # C2: weight scaling sweep across four MedMNIST datasets.
    print("\n--- Weight scaling sweep ---")
    weights = [0.0005, 0.001, 0.005, 0.01, 0.05]
    for flag in ["breastmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]:
        if s.time_left() < 600:
            break
        try:
            tr, va, te, task, nc, ic = load_med(flag, s.output_dir)
        except Exception as e:  # noqa: BLE001
            print(f"{flag}: {e}")
            continue
        cfg = CFG[flag]
        print(f"\n=== SWEEP {flag} (n={len(tr)}) ===")
        s.res.setdefault("sweep", {}).setdefault(flag, {"n": len(tr)})
        for w in weights:
            runs = []
            for seed in range(2):
                if s.time_left() < 200:
                    break
                r = train_eval(mk_fixed(w), tr, va, te, task, nc, ic,
                               cfg["ep"], cfg["bs"], cfg["lr"], seed)
                runs.append(r)
            au = [r["auc"] for r in runs]
            ec = [r["ece"] for r in runs]
            s.res["sweep"][flag][f"w{w}"] = {"auc": au, "ece": ec}
            print(f"  w={w:<7} AUC={np.mean(au):.4f} ECE={np.mean(ec):.4f}")
        s.save()

    fit_scaling_law(s.res)
    s.save()
    sl = s.res.get("scaling_law")
    if sl:
        print(f"\nSCALING LAW: w* = {sl['k']:.4g} * n^(-{sl['alpha']:.3f})  R2={sl['r2']:.3f}")
    return s.out


def run_session_d(output_dir=".", **kw):
    s = Session("D", output_dir, **kw)
    print("SESSION D: 5-seed runs for tight CIs + learned-weight extraction")
    s.run_grid("extended", CFG, [
        ("vanilla", mk_vanilla()),
        ("PAPC", mk_papc()),
    ], seeds=5)

    print("\n--- Learned per-layer weights (PAPC) ---")
    for flag in CFG:
        d = s.res.get("extended", {}).get(flag, {})
        lws = [r["lw"] for r in d.get("PAPC", []) if r.get("lw")]
        if lws:
            avg = np.mean(lws, axis=0)
            print(f"  {flag:16s}: {' '.join(f'{v:.4f}' for v in avg)}")
            s.res.setdefault("learned_weights", {})[flag] = avg.tolist()
    s.save()
    return s.out


def fit_scaling_law(res):
    """Fit w* = k * n^(-alpha) to the best-AUC weight per dataset in the sweep."""
    sizes, wopt = [], []
    for flag, d in res.get("sweep", {}).items():
        if "n" not in d:
            continue
        best = max(((float(k[1:]), np.mean(v["auc"])) for k, v in d.items()
                    if k.startswith("w") and isinstance(v, dict) and "auc" in v),
                   key=lambda t: t[1], default=None)
        if best:
            sizes.append(d["n"])
            wopt.append(best[0])
    if len(sizes) >= 3:
        x = np.log(np.array(sizes, float))
        y = np.log(np.array(wopt, float))
        A = np.vstack([x, np.ones_like(x)]).T
        slope, inter = np.linalg.lstsq(A, y, rcond=None)[0]
        yh = A @ [slope, inter]
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res["scaling_law"] = {"k": float(math.exp(inter)), "alpha": float(-slope),
                              "r2": float(r2), "sizes": sizes, "wopt": wopt}
    return res.get("scaling_law")


SESSIONS = {"A": run_session_a, "B": run_session_b,
            "C": run_session_c, "D": run_session_d}


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_condition(spec):
    """Parse a CLI condition like ``fixed:0.05`` -> ('fixed_0.05', kwargs)."""
    name, _, arg = spec.partition(":")
    if name not in CONDITIONS:
        raise SystemExit(f"Unknown condition '{name}'. Choose from {list(CONDITIONS)}.")
    factory = CONDITIONS[name]
    if arg:
        kwargs = factory(float(arg))
        return f"{name}_{arg}", kwargs
    return name, factory()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", choices=sorted(SESSIONS),
                    help="Run a full paper session (A/B/C/D).")
    ap.add_argument("--dataset", help="Single MedMNIST dataset for an ad-hoc grid.")
    ap.add_argument("--conditions", nargs="+", default=["vanilla", "papc"],
                    help="Conditions, e.g. vanilla fixed:0.05 papc clamped:0.05.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output-dir", default="results_repro")
    ap.add_argument("--time-budget-hours", type=float, default=8.5)
    ap.add_argument("--save-models", metavar="DIR", default=None,
                    help="Ad-hoc grid only: save each trained checkpoint into DIR "
                         "(loadable by predict.py).")
    args = ap.parse_args()

    budget = int(args.time_budget_hours * 3600)

    if args.session:
        out = SESSIONS[args.session](
            output_dir=args.output_dir, time_budget_sec=budget, seeds=args.seeds)
        print(f"\nDone. Wrote {out}")
        return

    if not args.dataset:
        ap.error("Provide either --session or --dataset.")
    if args.dataset not in CFG:
        ap.error(f"Unknown dataset '{args.dataset}'. Choose from {list(CFG)}.")

    conds = [parse_condition(c) for c in args.conditions]
    s = Session("adhoc", output_dir=args.output_dir, time_budget_sec=budget, seeds=args.seeds)
    s.run_grid("adhoc", {args.dataset: CFG[args.dataset]}, conds,
               seeds=args.seeds, save_dir=args.save_models)
    s.save()
    print(f"\nDone. Wrote {s.out}")


if __name__ == "__main__":
    main()
