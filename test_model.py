"""CPU smoke tests: shapes, the zero-gate identity property, and variant wiring.

Run with ``pytest -q``. These build the model with ``pretrained=False`` so they
need no network access and run in seconds on CPU.
"""

import torch

from papc import PAPCViT
from train import mk_vanilla, mk_fixed, mk_papc, mk_clamped


def _build(num_classes=8, in_chans=3, **kw):
    return PAPCViT(num_classes=num_classes, in_chans=in_chans, pretrained=False, **kw).eval()


def test_forward_shapes():
    model = _build(**mk_papc())
    logits, ploss = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 8)
    assert ploss.ndim == 0


def test_grayscale_input_expands():
    model = _build(in_chans=1, num_classes=2, **mk_vanilla())
    logits, _ = model(torch.randn(2, 1, 224, 224))
    assert logits.shape == (2, 2)


def test_vanilla_has_no_pc_and_zero_ploss():
    model = _build(**mk_vanilla())
    _, ploss = model(torch.randn(2, 3, 224, 224))
    assert model.pc is None
    assert float(ploss) == 0.0


def test_gate_initialised_to_zero():
    # tanh(0)=0 => an untrained PC model equals the plain backbone.
    model = _build(**mk_fixed(0.05))
    assert all(abs(g) < 1e-6 for g in model.gate_values())


def test_clamped_still_reports_aux_loss():
    # Gate-clamping disables integration but keeps the auxiliary loss term.
    model = _build(**mk_clamped(0.05))
    _, ploss = model(torch.randn(2, 3, 224, 224))
    assert model.clamp_gate is True
    assert float(ploss.detach()) > 0.0


def test_adaptive_learns_per_layer_weights():
    model = _build(**mk_papc())
    weights = model.learned_weights()
    assert weights is not None
    assert len(weights) == model.depth
