# Notebooks

The original, **fully executed** notebooks from the CAISc 2026 submission — every
cell is run and its output (metrics, tables, plots) is preserved in the `.ipynb`.
They are the ground-truth artifacts behind [`results/`](../results/); the
[`papc/`](../papc/) package is a clean refactor of the same code.

| Notebook | Session | Contents |
|---|:-:|---|
| [`results_of_a_APAPC_CAISc_final.ipynb`](results_of_a_APAPC_CAISc_final.ipynb) | A | Main SOTA table across the 8 MedMNIST datasets (vanilla / fixed `w=0.05` / PAPC) |
| [`results_b_B_PAPC_CAISc_final.ipynb`](results_b_B_PAPC_CAISc_final.ipynb) | B | Six-condition ablation + gate-clamping mechanism ablation |
| [`results_of_c_papc_PAPC_CAISc_final.ipynb`](results_of_c_papc_PAPC_CAISc_final.ipynb) | C | CIFAR-100 cross-domain sizes + weight sweep + scaling-law fit |
| [`results_of_d_DPAPC_CAISc_final.ipynb`](results_of_d_DPAPC_CAISc_final.ipynb) | D | 5-seed runs (tight CIs) + learned per-layer weight extraction |
| [`PAPC_final_runs.ipynb`](PAPC_final_runs.ipynb) | — | Consolidated follow-up runs that top up A/B/D (resumable driver) |

The first four share one code base with a `SESSION = 'A'|'B'|'C'|'D'` switch at the
top; set it and run top-to-bottom on an A100-class GPU (~9h each). Prefer the
packaged CLI for new runs:

```bash
python ../scripts/run_experiments.py --session A --output-dir ../results_repro
```

> Open these on [nbviewer](https://nbviewer.org/) if GitHub is slow to render the
> stored outputs.
