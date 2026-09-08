# Method notes

A companion to §3 of the [paper](../paper/CAISc2026_SUBMIT_v3.pdf). This
documents what the code in [`papc/`](../papc/) actually implements, so the
mapping from equations to modules is unambiguous.

## Architecture (HP-ViT)

A pretrained ViT with blocks `f₁ … f_L` gets one **Predictive Coding Layer**
([`PCLayer`](../papc/model.py)) after each block. Layer `i` sees the block output
`x_{i+1} = f_i(x_i)` of shape `(B, L_tok, D)` and:

1. **Predicts** the *next* block's output from the *current* activations with a
   per-channel **diagonal SSM** ([`DiagonalSSM`](../papc/model.py)):

   ```
   x̂_{i+1} = SSM(LN(x_i))
   ```

   The SSM is implemented as a **causal depthwise convolution** whose kernel is
   the materialised impulse response `ψ`. This is numerically equivalent to the
   linear recurrence `hₜ = Ā·hₜ₋₁ + B̄·xₜ`, `yₜ = C·hₜ + D·xₜ` (max deviation
   ~2.4e-7), but runs as one `conv1d` instead of a Python loop.

2. **Integrates** the prediction error back through a learned gate (optional):

   ```
   x̃_{i+1} = x_{i+1} + tanh(g_i) · W_i · LN(x_{i+1} − x̂_{i+1})
   ```

   `g_i` is initialised to **zero**, so `tanh(g_i) = 0` and an untrained model is
   bit-for-bit the plain backbone. `clamp_gate=True` forces this pathway off
   while keeping the auxiliary loss — the key mechanism ablation.

3. Contributes to the **auxiliary loss** with a stop-gradient target:

   ```
   L_aux = Σ_i w_i · ‖x̂_{i+1} − sg(x_{i+1})‖²_F
   L     = L_task + L_aux
   ```

The first layer has no error pathway (`has_error=False`): there is no previous
block to predict from.

## Variants (the `w_i` schedule)

| Variant | `w_i` | Flags |
|---|---|---|
| vanilla | — (no PC) | `use_pc=False` |
| fixed | constant `w` | `pred_loss_weight=w` |
| progressive | `w · ½(1 − cos(πt))` | `progressive=True` |
| adaptive | `softplus(θ_i) · w_max` (learned) | `adaptive=True` |
| **PAPC** | learned **and** cosine-warmed | `adaptive=True, progressive=True` |
| gate-clamped | fixed `w`, integration off | `clamp_gate=True` |

`t = step / total_steps`. The adaptive weights carry an L2 penalty
(`w_l2 · Σ w_i²`) that keeps them from drifting large.

## Training recipe ([`train.py`](../papc/train.py))

- **Backbone:** `vit_base_patch16_224.augreg_in21k_ft_in1k` (timm), 224 px.
- **Optimiser:** AdamW, `betas=(0.9, 0.95)`, weight decay 0.05.
- **LR schedule:** 5% linear warmup → cosine decay.
- **Layer-wise LR decay** (`param_groups`): factor 0.75 per block from the top;
  head ×10, PC params ×5 relative to `base_lr`.
- **Augmentation:** RandAugment / affine, random-resized-crop, h-flip,
  random-erasing; **Mixup + CutMix** for multi-class tasks (label smoothing 0.1).
- **EMA** weights (decay 0.9995) applied at evaluation.
- **Eval:** test-time horizontal-flip averaging; metrics = accuracy, macro-AUC,
  and 15-bin **ECE**.

Per-dataset epochs / batch size / LR live in
[`papc/config.py`](../papc/config.py) (`CFG`).

## Results schema

Each run returns and stores:

```json
{"acc": 0.99, "auc": 0.997, "ece": 0.018, "gate": 0.007,
 "lw": [...per-layer learned weights or null...], "t_min": 38.5}
```

Sessions nest these as `results[tag][dataset][condition] = [run, run, ...]`
(one entry per seed). See [`results/`](../results/) for the paper's runs and
[`scripts/analyze_results.py`](../scripts/analyze_results.py) for how the tables
are computed from them.
