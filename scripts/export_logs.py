#!/usr/bin/env python
"""Export per-run logs from the result JSONs into ``logs/``.

Each run recorded during the paper's experiments is written out as a readable,
greppable line (dataset, condition, seed, ACC, AUC, ECE, mean gate, minutes),
followed by per-condition aggregates. These files are derived from
``results/*.json`` -- they are an export of the recorded metrics, not raw
training console output.

    python scripts/export_logs.py --results-dir results --out-dir logs
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grand, gmin = 0, 0.0
    for fname, (stem, title) in SESSIONS.items():
        path = os.path.join(args.results_dir, fname)
        if not os.path.exists(path):
            continue
        t, m = export(path, title, os.path.join(args.out_dir, f"{stem}.log"))
        grand += t
        gmin += m
    print(f"\nTOTAL: {grand} runs, {gmin/60:.1f} GPU-hours")


if __name__ == "__main__":
    main()
