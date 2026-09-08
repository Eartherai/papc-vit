<div align="center">

<img src="assets/banner.png" width="100%" alt="PAPC - Predictive Coding for Pretrained Vision Transformers"/>

<h3>Predictive Coding Auxiliary Losses Are Calibration Levers, Not Accuracy Levers</h3>

<p><i>A Cross-Domain Empirical Study</i><br>
1st Conference for AI Scientists (CAISc 2026)</p>

<p><b>Khamir Desai</b></p>

<p>
<a href="https://openreview.net/forum?id=Kcsv2jUROe"><img src="https://img.shields.io/badge/Paper-OpenReview-8C1B13?style=for-the-badge" alt="Paper"/></a>
<a href="https://openreview.net/pdf?id=Kcsv2jUROe"><img src="https://img.shields.io/badge/PDF-Download-b31b1b?style=for-the-badge" alt="PDF"/></a>
</p>

<a href="https://openreview.net/forum?id=Kcsv2jUROe"><img src="https://img.shields.io/badge/venue-CAISc%202026-1f6feb.svg" alt="Venue"/></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-3776AB.svg" alt="Python"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg" alt="PyTorch"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/></a>
<a href="tests/"><img src="https://img.shields.io/badge/tests-6%20passing-brightgreen.svg" alt="Tests"/></a>

</div>

---

Predictive coding (PC) attaches a lightweight predictor to each block of a
network and trains it to anticipate the next block's activations. This work asks
whether such a **PC sidecar improves a pretrained Vision Transformer** during
fine-tuning. Across eight MedMNIST benchmarks and CIFAR-100 at three dataset
sizes, the answer reframes the question:

> **PC auxiliary losses are calibration levers, not accuracy levers.** They
> systematically reshape a model's confidence distribution (ECE) while leaving
> its discriminability (AUC) essentially unchanged.

**Paper:** [openreview.net/forum?id=Kcsv2jUROe](https://openreview.net/forum?id=Kcsv2jUROe)
· **PDF:** [openreview.net/pdf?id=Kcsv2jUROe](https://openreview.net/pdf?id=Kcsv2jUROe)
· **Local copy:** [`paper/CAISc2026_SUBMIT_v3.pdf`](paper/CAISc2026_SUBMIT_v3.pdf)

---

## Contents

[Abstract](#abstract) ·
[Key findings](#key-findings) ·
[Main results](#main-results) ·
[Calibration lever](#the-calibration-lever-effect) ·
[Mechanism](#mechanism-gradient-not-gate) ·
[Scaling law](#the-scaling-law) ·
[Ablation](#full-ablation) ·
[Method](#method) ·
[Installation](#installation) ·
[Reproducing the paper](#reproducing-the-paper) ·
[Training](#training) ·
[Inference](#inference) ·
[API](#using-the-model-in-your-own-code) ·
[Repository layout](#repository-layout) ·
[Citation](#citation)

---

## Abstract

We investigate whether predictive-coding auxiliary losses can improve pretrained
Vision Transformer classifiers. Attaching a diagonal state-space predictor to
each ViT block and training with an inter-block prediction loss, we run a
controlled study across eight MedMNIST benchmarks and CIFAR-100 at three dataset
sizes. Our central finding is that PC losses are calibration levers, not accuracy
levers: they systematically reshape confidence distributions (ECE) while leaving
discriminability (AUC) nearly unchanged. At the commonly used weight `w=0.05`,
calibration degrades on most datasets — ECE nearly doubles on CIFAR-100 (n=50k)
while AUC moves by only 0.02 pp — yet on PneumoniaMNIST the same weight improves
ECE by 3.2x with negligible AUC cost. We identify a dataset-size scaling law for
the optimal weight, w\* ≈ 257·n<sup>−1.41</sup> (R²=0.83), and show through a
gate-clamping ablation that the auxiliary-loss gradient is the primary mechanism.
A progressive adaptive variant (PAPC) prevents all catastrophic failures while
preserving calibration benefits. As a secondary finding, vanilla ViT-Base at
224 px already outperforms published specialist architectures (MedMamba, MedViT)
on DermaMNIST by +4.5 AUC points and RetinaMNIST by +8.3 points, suggesting
training recipes dominate architectural novelty on these benchmarks.

<div align="center">
<img src="assets/paper/fig1_architecture.png" width="92%" alt="HP-ViT architecture"/><br>
<sub><b>Figure 1.</b> A diagonal SSM predicts each ViT block's output from the previous block's activations. The prediction error is folded back through a learned gate <i>g<sub>i</sub></i>, initialised to zero; the auxiliary loss uses a stop-gradient target. In PAPC, <i>w<sub>i</sub></i> is learnable per layer and cosine-warmed from zero.</sub>
</div>

---

## Key findings

**1. Calibration, not discriminability.** On PneumoniaMNIST, sweeping `w` from
`5e-4` to `5e-2` improves ECE 3.2x (0.048 to 0.015) while AUC moves ~0.2 pp. On
CIFAR-100 (n=50k) the same weight nearly doubles ECE (0.104 to 0.195) for a
0.02 pp AUC change. Multi-seed testing finds no significant AUC gain from PC
anywhere (Wilcoxon *p* = 0.72).

**2. A dataset-size scaling law** for the AUC-optimal weight,
w\* ≈ 257·n<sup>−1.41</sup> (R² = 0.83) — a training-free heuristic for choosing
the auxiliary weight from dataset size alone.

**3. The auxiliary-loss gradient is the mechanism**, not error integration. A
gate-clamping ablation that disables the residual pathway yields near-identical
AUC to the full model.

**4. PAPC prevents catastrophic failures** while matching vanilla ViT on AUC,
where naive fixed-weight PC can lose 1.8 AUC points.

**5. Recipe beats architecture (secondary).** A plain ViT-Base at 224 px already
beats MedMamba and MedViT on several MedMNIST datasets.

This work began from a hypothesis that was refuted: PC was expected to improve
accuracy. Systematically examining what the auxiliary loss *did* change is what
produced the calibration-lever result.

---

## Main results

Test AUC (mean ± std over 3 seeds), ViT-Base at 224 px. The final column marks
where the plain vanilla baseline already exceeds the best published specialist
model. Reproduce with `python scripts/analyze_results.py`.

| Dataset | n | Vanilla | HP (w=0.05) | PAPC | Best published | Vanilla > published |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| PathMNIST | 89,996 | 0.9976 | 0.9915 | 0.9976 | 0.999 | |
| OrganAMNIST | 34,561 | 0.9962 | 0.9962 | 0.9957 | 0.998 | |
| OrganCMNIST | 12,975 | 0.9875 | 0.9858 | 0.9881 | 0.997 | |
| BloodMNIST | 11,959 | 0.9993 | 0.9991 | 0.9994 | 0.999 | yes |
| DermaMNIST | 7,007 | **0.9818** | 0.9637 | 0.9806 | 0.937 | yes (+4.5 pts) |
| PneumoniaMNIST | 4,708 | 0.9919 | 0.9892 | 0.9907 | 0.995 | |
| RetinaMNIST | 1,080 | 0.8566 | 0.8612 | 0.8602 | 0.773 | yes (+8.3 pts) |
| BreastMNIST | 546 | 0.8175 | 0.8594 | 0.8315 | 0.938 | |

PAPC matches vanilla on AUC across the board — the auxiliary loss buys no
discriminability — while naive fixed `w=0.05` can actively hurt (DermaMNIST,
0.982 to 0.964), a failure mode PAPC removes.

---

## The calibration-lever effect

<div align="center">
<img src="assets/paper/fig2_weight_sweep.png" width="80%" alt="AUC and ECE versus auxiliary weight"/><br>
<sub><b>Figure 2.</b> AUC (top) and ECE (bottom) against auxiliary weight <i>w</i>. On DermaMNIST a larger <i>w</i> worsens ECE; on PneumoniaMNIST it improves ECE 3.2x. AUC stays essentially flat on both.</sub>
</div>

<div align="center">
<img src="assets/paper/fig3_cifar_ece.png" width="45%" alt="CIFAR-100 ECE"/><br>
<sub><b>Figure 3.</b> On CIFAR-100, <i>w</i>=0.05 degrades calibration (+88% ECE at n=50k) for a negligible AUC change.</sub>
</div>

The sign of the effect flips across datasets, which is precisely why a
dataset-size-aware weight is needed.

---

## Mechanism: gradient, not gate

The PC layer both adds an auxiliary MSE loss and folds the prediction error back
through a learned gate. Clamping the gate isolates the two: the auxiliary loss
stays active, the integration pathway is forced off.

<div align="center">
<img src="assets/paper/fig5_gate_clamping.png" width="80%" alt="Gate-clamping ablation"/><br>
<sub><b>Figure 5.</b> Gate-clamping ablation at <i>w</i>=0.05. Clamped and Full track each other on AUC, so the effect is driven by the loss gradient rather than by re-injecting the error.</sub>
</div>

| Dataset | Condition | AUC | ECE | |
|---|---|:---:|:---:|---|
| DermaMNIST | vanilla | 0.9818 | 0.0811 | baseline |
| DermaMNIST | full (aux + integration) | 0.9637 | 0.1820 | |
| DermaMNIST | clamped (aux only) | **0.9633** | 0.1584 | AUC matches full |
| PneumoniaMNIST | vanilla | 0.9919 | 0.0558 | baseline |
| PneumoniaMNIST | full (aux + integration) | 0.9892 | 0.0182 | |
| PneumoniaMNIST | clamped (aux only) | **0.9896** | 0.0168 | AUC matches full |

---

## The scaling law

<div align="center">
<img src="assets/paper/fig4_scaling_law.png" width="45%" alt="Scaling law"/><br>
<sub><b>Figure 4.</b> The AUC-optimal auxiliary weight against training-set size across four MedMNIST datasets.</sub>
</div>

$$w^{*} \approx 256.7 \cdot n^{-1.412}, \qquad R^{2} = 0.831$$

| Dataset | BreastMNIST | PneumoniaMNIST | DermaMNIST | BloodMNIST |
|---|:---:|:---:|:---:|:---:|
| n | 546 | 4,708 | 7,007 | 11,959 |
| AUC-optimal `w*` | 0.05 | 0.0005 | 0.001 | 0.001 |

---

## Full ablation

Six conditions, three seeds, AUC and ECE. Every calibrated variant lands within
seed noise of vanilla on AUC; only naive fixed `w=0.05` swings widely.

| Dataset | vanilla | fixed 0.05 | fixed 0.001 | prog 0.005 | adaptive | PAPC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| DermaMNIST (AUC) | 0.9818 | 0.9637 | 0.9800 | 0.9796 | 0.9798 | **0.9806** |
| DermaMNIST (ECE) | 0.0811 | 0.1820 | 0.0938 | 0.0894 | 0.0891 | 0.0897 |
| PneumoniaMNIST (AUC) | 0.9919 | 0.9892 | 0.9912 | 0.9911 | 0.9918 | **0.9907** |
| PneumoniaMNIST (ECE) | 0.0558 | 0.0182 | 0.0438 | 0.0368 | 0.0404 | 0.0466 |
| BloodMNIST (AUC) | 0.9993 | 0.9991 | 0.9994 | 0.9994 | 0.9994 | **0.9994** |
| BreastMNIST (AUC) | 0.8175 | 0.8594 | 0.8405 | 0.8365 | 0.8342 | **0.8315** |

---

## Method

Given a pretrained ViT with blocks $f_1,\dots,f_L$, a Predictive Coding Layer
sits after each block. A per-channel diagonal SSM predicts the next block's
output from the current activations; the resulting error is optionally folded
back through a learned gate $g_i$ initialised to zero, so an untrained model is
identical to the plain backbone because $\tanh(0)=0$.

```
x_{i+1}  = f_i(x_i)                                            # block output
x_hat    = SSM(LayerNorm(x_i))                                 # PC prediction
x_tilde  = x_{i+1} + tanh(g_i) * W_i * LN(x_{i+1} - x_hat)     # gated integration
L_aux    = sum_i w_i * || x_hat - stopgrad(x_{i+1}) ||^2       # auxiliary loss
L        = L_task + L_aux
```

The SSM is implemented as a causal depthwise convolution, numerically equal to
the linear recurrence to about 2e-7, but evaluated as a single `conv1d`.

| Variant | Weight `w_i` | Flags |
|---|---|---|
| vanilla | none (no PC) | `use_pc=False` |
| fixed | constant `w` | `pred_loss_weight=w` |
| progressive | `w * 0.5(1 - cos(pi*t))` | `progressive=True` |
| adaptive | `softplus(theta_i) * w_max`, learned | `adaptive=True` |
| PAPC | learned and cosine-warmed | `adaptive=True, progressive=True` |
| gate-clamped | fixed `w`, integration off | `clamp_gate=True` |

Full equation-to-code mapping: [`docs/METHOD.md`](docs/METHOD.md).

---

## Installation

```bash
git clone https://github.com/Eartherai/papc-vit.git
cd papc-vit
pip install -e ".[dev]"
```

Requires Python 3.9+, PyTorch 2.1+, `timm` 1.0+, `medmnist` 3.0+, plus
`scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy` and `pillow`. A CUDA GPU
is required only for training; analysis, figures, inference and tests run on CPU.

Verify:

```bash
pytest -q          # 6 passed
```

---

## Reproducing the paper

Every table and figure is regenerated from the recorded results in
[`results/`](results/) — no GPU, a few seconds:

```bash
python scripts/analyze_results.py --results-dir results     # tables
python scripts/make_figures.py --results-dir results --out-dir assets
python scripts/export_logs.py  --results-dir results --out-dir logs
```

`analyze_results.py` prints the SOTA table, the gate-clamping ablation and the
fitted scaling law. `export_logs.py` writes per-run logs to [`logs/`](logs/):
**269 runs, 36.2 GPU-hours** of recorded train and eval time.

Make targets: `make analyze`, `make figures`, `make logs`, `make test`.

Full instructions, expected output, hardware notes and expected run-to-run
variance: **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

---

## Training

Sessions are resumable and time-budgeted: re-running skips
(dataset, condition, seed) triples already recorded and stops gracefully near the
budget, so an interrupted run can simply be restarted. Datasets and the
pretrained backbone download automatically on first use.

```bash
python scripts/run_experiments.py --session A --output-dir results_repro
```

| Session | Contents | Output keys |
|:---:|---|---|
| A | Main SOTA table: vanilla / fixed `w=0.05` / PAPC across 8 datasets | `sota` |
| B | Six-condition ablation plus gate-clamping ablation | `ablation`, `clamp` |
| C | CIFAR-100 at three sizes, weight sweep, scaling-law fit | `cifar`, `sweep`, `scaling_law` |
| D | Five-seed runs for tight CIs, learned per-layer weights | `extended`, `learned_weights` |

Ad-hoc grid on a single dataset, saving checkpoints for inference:

```bash
python scripts/run_experiments.py --dataset dermamnist \
    --conditions vanilla fixed:0.05 clamped:0.05 papc \
    --seeds 3 --save-models checkpoints --output-dir results_repro
```

Conditions: `vanilla`, `fixed:<w>`, `prog:<w>`, `adaptive`, `papc`, `clamped:<w>`.
Flags: `--seeds`, `--output-dir`, `--time-budget-hours`, `--save-models`.

---

## Inference

Classify a single image with a checkpoint saved by `--save-models`. CPU is fine:

```bash
python scripts/predict.py \
    --checkpoint checkpoints/dermamnist_papc_s0.pt \
    --image path/to/image.png --topk 3
```

```
Image:      path/to/image.png
Checkpoint: checkpoints/dermamnist_papc_s0.pt  (task=multi-class, classes=7)
Prediction: class 5  (confidence 0.8123)

Top-k:
  class  5  0.8123  ████████████████████████
  class  1  0.0904  ███
  class  0  0.0431  █
```

The reported confidence is exactly the quantity this paper shows PC losses
reshape.

---

## Using the model in your own code

```python
import torch
from papc import PAPCViT, mk_papc

model = PAPCViT(num_classes=8, in_chans=3, **mk_papc())

logits, aux_loss = model(torch.randn(4, 3, 224, 224))
loss = task_loss(logits, targets) + aux_loss    # add the auxiliary term

model.set_progress(step, total_steps)   # drives the cosine warmup
model.gate_values()                     # per-layer tanh(gate)
model.learned_weights()                 # per-layer w_i (PAPC only)
```

Condition factories in [`papc/config.py`](papc/config.py): `mk_vanilla`,
`mk_fixed(w)`, `mk_prog(w)`, `mk_adaptive`, `mk_papc`, `mk_clamped(w)`.

---

## Repository layout

```
papc-vit/
├── papc/                    Installable package
│   ├── model.py             DiagonalSSM, PCLayer, PAPCViT (all variants)
│   ├── data.py              MedMNIST/CIFAR loaders, transforms, EMA, AUC/ECE
│   ├── train.py             train_eval: fine-tune, evaluate, save checkpoints
│   ├── config.py            Per-dataset hyper-parameters, baselines, conditions
│   └── experiments.py       Sessions A-D and the scaling-law fit
├── scripts/
│   ├── run_experiments.py   Training entry point (sessions or ad-hoc grids)
│   ├── predict.py           Single-image inference from a checkpoint
│   ├── analyze_results.py   Regenerate the paper's tables
│   ├── make_figures.py      Regenerate result figures
│   ├── export_logs.py       Export per-run logs from the results JSON
│   └── make_banner.py       Regenerate the README banner
├── results/                 Recorded metrics (JSON) + schema documentation
├── logs/                    Per-run logs exported from results/
├── paper/                   CAISc 2026 manuscript PDF
├── assets/                  Banner and figures (assets/paper/ from the PDF)
├── docs/                    METHOD.md, REPRODUCE.md
└── tests/                   CPU smoke tests
```

---

## Datasets

| Source | Datasets | Access |
|---|---|---|
| [MedMNIST v2](https://medmnist.com/) | Path, Blood, Derma, Breast, Pneumonia, Retina, OrganA, OrganC | downloaded automatically via `medmnist` |
| CIFAR-100 | subsamples at n = 2,000 / 10,000 / 50,000 | downloaded automatically via `torchvision` |

All images are upsampled to 224 px for the ImageNet-pretrained ViT-Base backbone
(`vit_base_patch16_224.augreg_in21k_ft_in1k`). No dataset is redistributed in
this repository; please observe the MedMNIST and CIFAR licenses.

---

## Citation

```bibtex
@inproceedings{desai2026papc,
  title     = {Predictive Coding Auxiliary Losses Are Calibration Levers,
               Not Accuracy Levers: A Cross-Domain Empirical Study},
  author    = {Khamir Desai},
  booktitle = {1st Conference for AI Scientists (CAISc)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Kcsv2jUROe}
}
```

## License

Released under the [MIT License](LICENSE). The manuscript PDF and the datasets
retain their respective licenses.
