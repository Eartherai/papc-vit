#!/usr/bin/env python
"""Regenerate the paper's tables, figures, and per-run logs from ``results/``.

Everything here is CPU-only and reads the recorded metrics in ``results/*.json``
-- no GPU and no training required.

    python analyze.py                 # tables + figures + logs
    python analyze.py --tables        # just print the tables
    python analyze.py --figures       # just rebuild assets/*.png
    python analyze.py --logs          # just rebuild results/logs/*.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(results_dir, name):
    p = os.path.join(results_dir, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def agg(runs, key):
    v = [r[key] for r in runs if isinstance(r, dict) and r.get(key) is not None]
    return ((float(np.mean(v)), float(np.std(v)), len(v)) if v
            else (float("nan"), float("nan"), 0))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
PUB_BEST = {  # best published AUC per dataset (for the "beats SOTA?" column)
    "pathmnist": 0.999, "bloodmnist": 0.999, "dermamnist": 0.937,
    "pneumoniamnist": 0.995, "retinamnist": 0.773, "breastmnist": 0.938,
    "organamnist": 0.998, "organcmnist": 0.997,
}


def load(results_dir, name):
    p = os.path.join(results_dir, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def agg(runs, key):
    v = [r[key] for r in runs if isinstance(r, dict) and r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)


def sota_table(results_dir):
    A = load(results_dir, "papc_A_results.json")
    F = load(results_dir, "papc_final_runs.json")
    merged = {}
    for src in (A.get("sota", {}), F.get("sota", {})):
        for flag, fd in src.items():
            merged.setdefault(flag, {"n": fd.get("n", "?")})
            for c in ("vanilla", "hp_0.05", "PAPC"):
                merged[flag].setdefault(c, [])
                merged[flag][c] += fd.get(c, [])

    print("\n" + "=" * 78)
    print("SOTA TABLE  (test AUC, mean over seeds)")
    print("=" * 78)
    print(f"{'Dataset':16s} {'n':>7}  {'Vanilla':>16}  {'HP w=.05':>16}  "
          f"{'PAPC':>16}  {'BestPub':>8}  {'V>Pub':>6}")
    for flag, fd in sorted(merged.items(), key=lambda x: -(x[1].get("n", 0) or 0)):
        cells = []
        for c in ("vanilla", "hp_0.05", "PAPC"):
            m, s, k = agg(fd[c], "auc")
            cells.append(f"{m:.4f}±{s:.4f}" if k else "     -      ")
        pub = PUB_BEST.get(flag, float("nan"))
        va = agg(fd["vanilla"], "auc")[0]
        win = "yes" if va > pub else ""
        print(f"{flag:16s} {str(fd['n']):>7}  {cells[0]:>16}  {cells[1]:>16}  "
              f"{cells[2]:>16}  {pub:>8.3f}  {win:>6}")


def clamp_table(results_dir):
    B = load(results_dir, "papc_B_results.json")
    F = load(results_dir, "papc_final_runs.json")
    print("\n" + "=" * 78)
    print("GATE-CLAMPING ABLATION  (aux-loss gradient is the mechanism)")
    print("=" * 78)
    print(f"{'Dataset':16s} {'Condition':10s} {'AUC':>10} {'ECE':>10} {'n':>4}  Interpretation")
    tags = {"vanilla": "baseline", "full_005": "aux+integration", "clamped": "aux only"}
    for flag in ("dermamnist", "pneumoniamnist"):
        for c in ("vanilla", "full_005", "clamped"):
            runs = (B.get("clamp", {}).get(flag, {}).get(c, [])
                    + F.get("clamp", {}).get(flag, {}).get(c, []))
            if not runs:
                src = "fixed_0.05" if c == "full_005" else c
                runs = B.get("ablation", {}).get(flag, {}).get(src, [])
            a = agg(runs, "auc")
            e = agg(runs, "ece")
            if a[2]:
                print(f"{flag:16s} {c:10s} {a[0]:>10.4f} {e[0]:>10.4f} {a[2]:>4}  {tags[c]}")


def scaling_law(results_dir):
    C = load(results_dir, "papc_C_results.json")
    sizes, wopt = [], []
    for flag, d in C.get("sweep", {}).items():
        if "n" not in d:
            continue
        best = max(((float(k[1:]), np.mean(v["auc"])) for k, v in d.items()
                    if k.startswith("w") and isinstance(v, dict) and "auc" in v),
                   key=lambda t: t[1], default=None)
        if best:
            sizes.append(d["n"])
            wopt.append(best[0])
    print("\n" + "=" * 78)
    print("DATASET-SIZE SCALING LAW  for the AUC-optimal auxiliary weight")
    print("=" * 78)
    for flag, d in C.get("sweep", {}).items():
        if "n" in d:
            print(f"  {flag:16s} n={d['n']:>6}")
    if len(sizes) >= 3:
        x = np.log(sizes)
        y = np.log(wopt)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, inter = np.linalg.lstsq(A, y, rcond=None)[0]
        yh = A @ [slope, inter]
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\n  w* = {math.exp(inter):.4g} * n^(-{-slope:.3f})   (R^2 = {r2:.3f})")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Per-run logs
# ---------------------------------------------------------------------------
SESSIONS = {
    "papc_A_results.json": ("session_A", "Session A - main SOTA table"),
    "papc_B_results.json": ("session_B", "Session B - ablation + gate clamping"),
    "papc_C_results.json": ("session_C", "Session C - CIFAR-100 + weight sweep"),
    "papc_D_results.json": ("session_D", "Session D - 5-seed runs + learned weights"),
    "papc_final_runs.json": ("final_runs", "Follow-up consolidated runs"),
}

HEADER = "{:<16} {:<14} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "dataset", "condition", "seed", "ACC", "AUC", "ECE", "gate", "min")


def fmt_run(flag, cond, seed, r):
    return (f"{flag:<16} {cond:<14} {seed:>4} "
            f"{r.get('acc', float('nan')):8.4f} {r.get('auc', float('nan')):8.4f} "
            f"{r.get('ece', float('nan')):8.4f} {r.get('gate', float('nan')):8.4f} "
            f"{r.get('t_min', float('nan')):8.2f}")


def export(path, title, out_path):
    data = json.load(open(path))
    lines = [
        "=" * 88,
        f"{title}",
        f"exported from {os.path.basename(path)} on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "metrics recorded per run; derived from the stored results JSON",
        "=" * 88, "",
    ]
    total = 0
    gpu_min = 0.0

    for section, sdata in data.items():
        if not isinstance(sdata, dict):
            continue
        if section in ("scaling_law", "learned_weights"):
            lines += [f"[{section}]", json.dumps(sdata, indent=2), ""]
            continue
        lines += [f"[{section}]", HEADER, "-" * 88]
        for flag, fd in sdata.items():
            if not isinstance(fd, dict):
                continue
            for cond, runs in fd.items():
                if cond == "n" or not isinstance(runs, list):
                    continue
                for seed, r in enumerate(runs):
                    if not isinstance(r, dict):
                        continue
                    lines.append(fmt_run(flag, cond, seed, r))
                    total += 1
                    if isinstance(r.get("t_min"), (int, float)):
                        gpu_min += r["t_min"]
                aucs = [r["auc"] for r in runs if isinstance(r, dict) and r.get("auc") is not None]
                eces = [r["ece"] for r in runs if isinstance(r, dict) and r.get("ece") is not None]
                if aucs:
                    lines.append(
                        f"{'  -> mean':<16} {cond:<14} {len(aucs):>4} {'':>8} "
                        f"{np.mean(aucs):8.4f} {np.mean(eces):8.4f}")
        lines.append("")

    lines += ["=" * 88,
              f"runs logged: {total}    total train+eval time: {gpu_min/60:.1f} GPU-hours",
              "=" * 88, ""]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}  ({total} runs, {gpu_min/60:.1f} GPU-h)")
    return total, gpu_min

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--assets-dir", default="assets")
    ap.add_argument("--logs-dir", default="results/logs")
    ap.add_argument("--tables", action="store_true", help="Print the paper tables.")
    ap.add_argument("--figures", action="store_true", help="Rebuild the result figures.")
    ap.add_argument("--logs", action="store_true", help="Rebuild the per-run logs.")
    args = ap.parse_args()

    do_all = not (args.tables or args.figures or args.logs)

    if do_all or args.tables:
        sota_table(args.results_dir)
        clamp_table(args.results_dir)
        scaling_law(args.results_dir)
        print()

    if do_all or args.figures:
        os.makedirs(args.assets_dir, exist_ok=True)
        C = load(args.results_dir, "papc_C_results.json")
        D = load(args.results_dir, "papc_D_results.json")
        fig_scaling_law(C, args.assets_dir)
        fig_learned_weights(D, args.assets_dir)
        fig_calibration(C, args.assets_dir)

    if do_all or args.logs:
        os.makedirs(args.logs_dir, exist_ok=True)
        grand, gmin = 0, 0.0
        for fname, (stem, title) in SESSIONS.items():
            path = os.path.join(args.results_dir, fname)
            if not os.path.exists(path):
                continue
            t, m = export(path, title, os.path.join(args.logs_dir, f"{stem}.log"))
            grand += t
            gmin += m
        print(f"\nTOTAL: {grand} runs, {gmin/60:.1f} GPU-hours")


if __name__ == "__main__":
    main()
