"""PAPC model: a predictive-coding sidecar for pretrained Vision Transformers.

The architecture (``HP-ViT`` in the paper) attaches a Predictive Coding Layer
after every ViT block. Each layer contains a diagonal state-space model (SSM)
predictor and an optional gated error-integration pathway. The inter-block
prediction error is added to the task loss as an auxiliary term.

Mode flags select the variant studied in the paper:

===============  =========================================================
Variant          kwargs
===============  =========================================================
vanilla          ``use_pc=False``
fixed-weight     ``use_pc=True, pred_loss_weight=w``
progressive      ``use_pc=True, pred_loss_weight=w, progressive=True``
adaptive         ``use_pc=True, adaptive=True``
**PAPC**         ``use_pc=True, adaptive=True, progressive=True``
gate-clamped     ``use_pc=True, pred_loss_weight=w, clamp_gate=True``
===============  =========================================================

Because the error gate is initialised to zero (``tanh(0) = 0``), an untrained
model is numerically identical to the unmodified pretrained backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

DEFAULT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"
DEFAULT_IMG_SIZE = 224


class DiagonalSSM(nn.Module):
    """Per-channel diagonal SSM applied along the token dimension.

    Implemented as a causal depthwise convolution, which is numerically
    equivalent to the underlying linear recurrence (max deviation ~2.4e-7 in
    the paper's verification).
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        a = torch.log(torch.linspace(1.0, float(d_state), d_state))
        self.A_log = nn.Parameter(a.repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.dt_log = nn.Parameter(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        dt = torch.exp(self.dt_log.float())
        A_bar = torch.exp(-torch.exp(self.A_log.float()) * dt.unsqueeze(-1))
        B_bar = self.B.float() * dt.unsqueeze(-1)
        logA = torch.log(A_bar.clamp(min=1e-38))
        k = torch.arange(L, device=x.device, dtype=torch.float32)
        A_pow = torch.exp(k.view(-1, 1, 1) * logA.unsqueeze(0))
        psi = (A_pow * (self.C.float() * B_bar).unsqueeze(0)).sum(-1).to(x.dtype)
        w = psi.t().contiguous().unsqueeze(1).flip(-1)
        xp = F.pad(x.transpose(1, 2).contiguous(), (L - 1, 0))
        h = F.conv1d(xp, w, groups=D)
        return (h + self.D.to(x.dtype).view(1, -1, 1) * x.transpose(1, 2)).transpose(1, 2)


class PCLayer(nn.Module):
    """A predictive-coding layer: SSM predictor + gated error integration."""

    def __init__(self, dim: int, d_state: int = 16, has_error: bool = True):
        super().__init__()
        self.has_error = has_error
        self.pred_norm = nn.LayerNorm(dim, eps=1e-6)
        self.predictor = DiagonalSSM(dim, d_state=d_state)
        if has_error:
            self.err_norm = nn.LayerNorm(dim, eps=1e-6)
            self.err_proj = nn.Linear(dim, dim)
            self.gate = nn.Parameter(torch.zeros(1))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.pred_norm(x))

    def integrate(self, actual: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return actual + torch.tanh(self.gate) * self.err_proj(self.err_norm(actual - pred))

    def gate_val(self) -> float:
        return 0.0 if not self.has_error else float(torch.tanh(self.gate).detach().cpu())


class PAPCViT(nn.Module):
    """Vision Transformer with an optional predictive-coding sidecar.

    ``PAPC`` (the paper's proposed method) = ``use_pc + adaptive + progressive``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME,
                 img_size: int = DEFAULT_IMG_SIZE, num_classes: int = 10,
                 in_chans: int = 3, d_state: int = 16, pred_loss_weight: float = 0.05,
                 drop_path: float = 0.1, pretrained: bool = True, use_pc: bool = True,
                 adaptive: bool = False, progressive: bool = False,
                 clamp_gate: bool = False, w_max: float = 0.01, w_l2: float = 1e-3):
        super().__init__()
        self.use_pc = use_pc
        self.pred_loss_weight = pred_loss_weight if use_pc else 0.0
        self.clamp_gate = clamp_gate
        self.adaptive = adaptive and use_pc
        self.progressive = progressive
        self.w_max = w_max
        self.w_l2 = w_l2
        self._step = 0
        self._total = 1

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, img_size=img_size,
            num_classes=0, drop_path_rate=drop_path, in_chans=3)
        self.dim = self.backbone.embed_dim
        self.depth = len(self.backbone.blocks)
        self.in_chans = in_chans
        self.expand_gray = (in_chans == 1)

        if use_pc:
            self.pc = nn.ModuleList([
                PCLayer(self.dim, d_state, has_error=(i > 0))
                for i in range(self.depth)])
            if self.adaptive:
                # init so each learnable weight starts near 0.002
                init_val = math.log(max(1e-6, math.exp(0.002 / max(1e-6, w_max)) - 1))
                self.log_w = nn.Parameter(torch.full((self.depth,), init_val))
        else:
            self.pc = None

        self.head = nn.Linear(self.dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def set_progress(self, step: int, total: int) -> None:
        self._step = step
        self._total = total

    def _prog_scale(self) -> float:
        # cosine warmup of the auxiliary loss: 0 at start -> 1 at end
        if not self.progressive:
            return 1.0
        t = self._step / max(1, self._total)
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def _layer_weights(self):
        if self.adaptive:
            return F.softplus(self.log_w) * self.w_max * self._prog_scale()
        return None

    def forward(self, x: torch.Tensor):
        if self.expand_gray:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)

        pred_loss = x.new_zeros(())
        if self.use_pc:
            lw = self._layer_weights()
            prev = None
            for i, block in enumerate(self.backbone.blocks):
                actual = block(x)
                pc = self.pc[i]
                if prev is not None and pc.has_error:
                    wi = lw[i] if lw is not None else self._prog_scale()
                    pred_loss = pred_loss + wi * F.mse_loss(prev, actual.detach())
                    if not self.clamp_gate:
                        actual = pc.integrate(actual, prev)
                prev = pc.predict(actual)
                x = actual
            x = self.backbone.norm(x)
            if prev is not None:
                wl = lw[-1] if lw is not None else self._prog_scale()
                pred_loss = pred_loss + wl * F.mse_loss(prev, x.detach())
            if self.adaptive:
                pred_loss = pred_loss + self.w_l2 * (self._layer_weights() ** 2).sum()
            else:
                pred_loss = pred_loss * self.pred_loss_weight
        else:
            for block in self.backbone.blocks:
                x = block(x)
            x = self.backbone.norm(x)

        return self.head(x[:, 0]), pred_loss

    def gate_values(self):
        return [] if not self.use_pc else [pc.gate_val() for pc in self.pc]

    def learned_weights(self):
        if not self.adaptive or not self.use_pc:
            return None
        with torch.no_grad():
            return (F.softplus(self.log_w) * self.w_max).cpu().tolist()

    def param_groups(self, base_lr: float, head_mult: float = 10.0,
                     pc_mult: float = 5.0, decay: float = 0.75):
        """Layer-wise LR decay: deeper blocks and the head get larger LRs."""
        groups = []
        for i, block in enumerate(self.backbone.blocks):
            groups.append({"params": list(block.parameters()),
                           "lr": base_lr * decay ** (self.depth - 1 - i)})
        early = list(self.backbone.patch_embed.parameters())
        for attr in ("cls_token", "pos_embed", "reg_token"):
            pa = getattr(self.backbone, attr, None)
            if isinstance(pa, nn.Parameter):
                early.append(pa)
        groups.append({"params": early, "lr": base_lr * decay ** self.depth})
        groups.append({"params": list(self.backbone.norm.parameters()), "lr": base_lr})
        if self.use_pc:
            pp = list(self.pc.parameters())
            if self.adaptive:
                pp.append(self.log_w)
            groups.append({"params": pp, "lr": base_lr * pc_mult})
        groups.append({"params": list(self.head.parameters()), "lr": base_lr * head_mult})
        return [{"params": [p for p in g["params"] if isinstance(p, nn.Parameter)], "lr": g["lr"]}
                for g in groups
                if any(isinstance(p, nn.Parameter) for p in g["params"])]
