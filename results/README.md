# Raw results

JSON metrics from the paper's four experimental sessions (plus a consolidated
`papc_final_runs.json` covering runs finished in a follow-up session). These are
the exact numbers behind every table and figure; regenerate them with
`python scripts/analyze_results.py --results-dir results`.

## Files

| File | Session | Top-level keys |
|---|---|---|
| `papc_A_results.json` | A | `sota` |
| `papc_B_results.json` | B | `ablation`, `clamp` |
| `papc_C_results.json` | C | `cifar`, `sweep`, `scaling_law` |
| `papc_D_results.json` | D | `extended`, `learned_weights` |
| `papc_final_runs.json` | follow-up | `clamp`, `sota`, `extended` |

## Schema

Runs are nested `results[tag][dataset][condition] = [run, ...]`, one entry per
seed. Each run is:

```json
{
  "acc":   0.9919,        // top-1 accuracy
  "auc":   0.9974,        // macro one-vs-rest ROC-AUC
  "ece":   0.0182,        // 15-bin Expected Calibration Error
  "gate":  0.0072,        // mean |tanh(gate)| over PC layers (0 for vanilla)
  "lw":    [0.002, ...],  // learned per-layer weights (PAPC only; else null)
  "t_min": 38.5           // wall-clock minutes for the run
}
```

The weight sweep (`sweep`) stores condensed `{"auc": [...], "ece": [...]}` per
weight instead of full run dicts. `scaling_law` holds the fitted
`{k, alpha, r2, sizes, wopt}` for `w* = k · n^(-alpha)`.
