#!/usr/bin/env python
"""Run PAPC experiments from the command line.

Examples
--------
Run a full paper session (A/B/C/D), resumable and time-budgeted::

    python scripts/run_experiments.py --session A --output-dir results_repro

Run a single (dataset, condition) grid for a quick check::

    python scripts/run_experiments.py --dataset pneumoniamnist \
        --conditions vanilla fixed:0.05 papc --seeds 2 --output-dir results_repro
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from papc.config import CFG, CONDITIONS  # noqa: E402
from papc.experiments import SESSIONS, Session, DEFAULT_TIME_BUDGET_SEC  # noqa: E402


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
    s.run_grid("adhoc", {args.dataset: CFG[args.dataset]}, conds, seeds=args.seeds)
    s.save()
    print(f"\nDone. Wrote {s.out}")


if __name__ == "__main__":
    main()
