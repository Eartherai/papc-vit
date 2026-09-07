#!/usr/bin/env python
"""Regenerate the paper's headline tables from the checked-in result JSONs.

Reads ``results/papc_{A,B,C,D}_results.json`` (+ ``papc_final_runs.json``) and
prints: the SOTA table (vs. published baselines), the gate-clamping mechanism
ablation, and the fitted dataset-size scaling law. No GPU required.

    python scripts/analyze_results.py --results-dir results
"""

import argparse
import json
import math
import os

import numpy as np

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    sota_table(args.results_dir)
    clamp_table(args.results_dir)
    scaling_law(args.results_dir)
    print()


if __name__ == "__main__":
    main()
