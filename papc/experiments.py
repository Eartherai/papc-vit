"""The four experimental sessions from the paper.

Each session writes a JSON results file (``papc_<SESSION>_results.json``)
matching the schema of the checked-in ``results/`` files. Runs are resumable:
re-running a session skips (dataset, condition, seed) combinations already
present in the output file, and stops gracefully when the time budget is spent.

  Session A  Main SOTA table (vanilla / fixed w=0.05 / PAPC) over 8 datasets
  Session B  Full 6-condition ablation + gate-clamping mechanism ablation
  Session C  CIFAR-100 cross-domain sizes + weight sweep + scaling-law fit
  Session D  5-seed runs (tight CIs) + learned per-layer weight extraction
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .config import (CFG, PUB, mk_vanilla, mk_fixed, mk_prog, mk_adaptive,
                     mk_papc, mk_clamped)
from .data import load_med, load_cifar
from .train import train_eval

DEFAULT_TIME_BUDGET_SEC = int(8.5 * 3600)  # safe margin inside a 9h session


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
