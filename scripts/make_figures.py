#!/usr/bin/env python
"""Generate the paper's figures from the checked-in result JSONs.

Produces (into ``--out-dir``, default ``assets/``):
  * ``scaling_law.png``      — optimal weight w* vs dataset size n (log-log fit)
  * ``learned_weights.png``  — PAPC per-layer learned weights across datasets
  * ``calibration.png``      — ECE vs auxiliary weight from the Session C sweep

No GPU required.

    python scripts/make_figures.py --results-dir results --out-dir assets
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(results_dir, name):
    p = os.path.join(results_dir, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def fig_scaling_law(C, out_dir):
    sweep = C.get("sweep", {})
    sizes, wopt = [], []
    for flag, d in sweep.items():
        if "n" not in d:
            continue
        best = max(((float(k[1:]), np.mean(v["auc"])) for k, v in d.items()
                    if k.startswith("w") and isinstance(v, dict) and "auc" in v),
                   key=lambda t: t[1], default=None)
        if best:
            sizes.append(d["n"])
            wopt.append(best[0])
    if len(sizes) < 3:
        print("scaling_law: not enough points, skipping")
        return
    x = np.log(sizes)
    y = np.log(wopt)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, inter = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ [slope, inter]) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(sizes, wopt, s=70, zorder=3, color="#2b6cb0")
    xs = np.linspace(min(sizes) * 0.8, max(sizes) * 1.2, 100)
    ax.plot(xs, math.exp(inter) * xs ** slope, "--", color="#e53e3e",
            label=fr"$w^* = {math.exp(inter):.0f}\,n^{{{slope:.2f}}}$  ($R^2={r2:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training-set size $n$")
    ax.set_ylabel("AUC-optimal auxiliary weight $w^*$")
    ax.set_title("Dataset-size scaling law")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "scaling_law.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def fig_learned_weights(D, out_dir):
    lw = D.get("learned_weights", {})
    if not lw:
        # fall back to averaging the raw PAPC runs
        for flag, fd in D.get("extended", {}).items():
            lws = [r["lw"] for r in fd.get("PAPC", []) if r.get("lw")]
            if lws:
                lw[flag] = np.mean(lws, axis=0).tolist()
    if not lw:
        print("learned_weights: none found, skipping")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for flag, vals in lw.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3, label=flag)
    ax.set_xlabel("ViT block index")
    ax.set_ylabel("Learned auxiliary weight $w_i$")
    ax.set_title("PAPC learned per-layer weights")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "learned_weights.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def fig_calibration(C, out_dir):
    sweep = C.get("sweep", {})
    if not sweep:
        print("calibration: no sweep data, skipping")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4))
    for flag, d in sweep.items():
        ws, eces = [], []
        for k, v in sorted(d.items()):
            if k.startswith("w") and isinstance(v, dict) and "ece" in v:
                ws.append(float(k[1:]))
                eces.append(float(np.mean(v["ece"])))
        if ws:
            order = np.argsort(ws)
            ax.plot(np.array(ws)[order], np.array(eces)[order],
                    marker="o", ms=4, label=f"{flag} (n={d.get('n','?')})")
    ax.set_xscale("log")
    ax.set_xlabel("Auxiliary loss weight $w$")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("PC losses are calibration levers")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "calibration.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    C = load(args.results_dir, "papc_C_results.json")
    D = load(args.results_dir, "papc_D_results.json")
    fig_scaling_law(C, args.out_dir)
    fig_learned_weights(D, args.out_dir)
    fig_calibration(C, args.out_dir)


if __name__ == "__main__":
    main()
