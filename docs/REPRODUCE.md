# Reproduction guide

Two levels of reproduction are supported:

1. **Verify the paper's numbers** from the recorded results — minutes, CPU only.
2. **Re-run the experiments** from scratch — hours to days, needs an A100-class GPU.

---

## 1. Clone and set up

```bash
git clone https://github.com/Eartherai/papc-vit.git
cd papc-vit

python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -e ".[dev]"
```

Verify the install:

```bash
pytest -q                 # 6 CPU smoke tests, no downloads, no GPU
```

Expected: `6 passed`.

---

## 2. Verify the paper's numbers (CPU, ~seconds)

All headline tables are recomputed from `results/*.json`:

```bash
python scripts/analyze_results.py --results-dir results
```

Expected output (exact values):

```
SOTA TABLE  (test AUC, mean over seeds)
Dataset                n           Vanilla          HP w=.05              PAPC   BestPub   V>Pub
pathmnist          89996     0.9976±0.0004     0.9915±0.0009     0.9976±0.0003     0.999
organamnist        34561     0.9962±0.0009     0.9962±0.0009     0.9957±0.0011     0.998
organcmnist        12975     0.9875±0.0021     0.9858±0.0029     0.9881±0.0023     0.997
bloodmnist         11959     0.9993±0.0001     0.9991±0.0001     0.9994±0.0000     0.999     yes
dermamnist          7007     0.9818±0.0004     0.9637±0.0081     0.9806±0.0005     0.937     yes
pneumoniamnist      4708     0.9919±0.0008     0.9892±0.0017     0.9907±0.0006     0.995
retinamnist         1080     0.8566±0.0158     0.8612±0.0039     0.8602±0.0057     0.773     yes
breastmnist          546     0.8175±0.0037     0.8594±0.0171     0.8315±0.0226     0.938

GATE-CLAMPING ABLATION  (aux-loss gradient is the mechanism)
dermamnist       vanilla        0.9818     0.0811    6  baseline
dermamnist       full_005       0.9637     0.1820    3  aux+integration
dermamnist       clamped        0.9633     0.1584    3  aux only
pneumoniamnist   vanilla        0.9919     0.0558    3  baseline
pneumoniamnist   full_005       0.9892     0.0182    3  aux+integration
pneumoniamnist   clamped        0.9896     0.0168    3  aux only

DATASET-SIZE SCALING LAW
  w* = 256.7 * n^(-1.412)   (R^2 = 0.831)
```

Regenerate the figures and the per-run logs:

```bash
python scripts/make_figures.py --results-dir results --out-dir assets
python scripts/export_logs.py  --results-dir results --out-dir logs
```

`export_logs.py` reports `269 runs, 36.2 GPU-hours` of recorded train+eval time.

---

## 3. Re-run the experiments (GPU)

### Hardware and software used

| | |
|---|---|
| GPU | NVIDIA A100, 40 GB and 80 GB variants |
| Sessions | 4, each budgeted at 8.5 h wall-clock (~60 GPU-h total incl. overhead) |
| Recorded train+eval | 36.2 GPU-hours across 269 runs (see `logs/`) |
| Python | 3.11 |
| Key packages | `torch`, `timm>=1.0`, `medmnist>=3.0`, `scikit-learn`, `numpy`, `scipy` |
| Backbone | `vit_base_patch16_224.augreg_in21k_ft_in1k` (timm, downloads on first run) |

Datasets download automatically on first use (MedMNIST v2 via `medmnist`,
CIFAR-100 via `torchvision`); budget a few GB of disk.

### Run a session

Sessions are **resumable** and **time-budgeted**: re-running skips
(dataset, condition, seed) triples already present in the output JSON and stops
gracefully as the budget runs out, so an interrupted run can simply be restarted.

```bash
python scripts/run_experiments.py --session A --output-dir results_repro
python scripts/run_experiments.py --session B --output-dir results_repro
python scripts/run_experiments.py --session C --output-dir results_repro
python scripts/run_experiments.py --session D --output-dir results_repro
```

| Session | Contents | Output file |
|:-:|---|---|
| A | Main SOTA table (vanilla / fixed `w=0.05` / PAPC), 8 datasets, 3 seeds | `papc_A_results.json` |
| B | Six-condition ablation + gate-clamping ablation | `papc_B_results.json` |
| C | CIFAR-100 at 3 sizes + weight sweep + scaling-law fit | `papc_C_results.json` |
| D | 5-seed runs for tight CIs + learned per-layer weights | `papc_D_results.json` |

Then compare against the shipped results:

```bash
python scripts/analyze_results.py --results-dir results_repro
```

### Smaller ad-hoc runs

```bash
# one dataset, a few conditions, saving checkpoints for inference
python scripts/run_experiments.py --dataset dermamnist \
    --conditions vanilla fixed:0.05 clamped:0.05 papc \
    --seeds 3 --save-models checkpoints --output-dir results_repro
```

Conditions: `vanilla`, `fixed:<w>`, `prog:<w>`, `adaptive`, `papc`, `clamped:<w>`.
Other flags: `--seeds`, `--output-dir`, `--time-budget-hours`, `--save-models`.

---

## 4. Expected variance

Runs are seeded (`torch`, `numpy`, CUDA), but bitwise determinism is **not**
guaranteed: cuDNN autotuning, TF32/bf16 autocast, and non-deterministic reduction
kernels introduce small run-to-run differences. Expect agreement within roughly
the seed spread reported in the tables (typically ±0.001–0.02 AUC, larger on the
smallest datasets such as BreastMNIST, n=546).

The single most seed-sensitive setting is naive fixed `w=0.05` on DermaMNIST,
where individual seeds range from 0.957 to 0.975 AUC — that instability is itself
one of the paper's findings, and is what PAPC removes.

---

## 5. Results and logs layout

```
results/
  papc_A_results.json     sota
  papc_B_results.json     ablation, clamp
  papc_C_results.json     cifar, sweep, scaling_law
  papc_D_results.json     extended, learned_weights
  papc_final_runs.json    clamp, sota, extended  (follow-up top-up runs)
  README.md               JSON schema

logs/
  session_A.log ... session_D.log, final_runs.log
```

Every run is stored as
`{"acc", "auc", "ece", "gate", "lw", "t_min"}` and nested as
`results[section][dataset][condition] = [run per seed]`. See
[`../results/README.md`](../results/README.md) for the full schema.
