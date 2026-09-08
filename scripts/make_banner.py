#!/usr/bin/env python
"""Render the repository hero banner to ``assets/banner.png``.

Self-contained matplotlib script (no external assets), rendered at 300 dpi so it
stays crisp on high-density displays.

    python scripts/make_banner.py
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG_TOP = "#122a49"
BG_BOT = "#060d18"
WHITE = "#ffffff"
MUTED = "#93a6c0"
ACCENT = "#e8833a"
BLUE = "#63a8ee"


def _track(text, em=0.18):
    """Poor-man's letter-spacing: matplotlib has no tracking control."""
    sep = " " * max(1, int(round(em * 4)))
    return sep.join(list(text))


def make_banner(out_path, dpi=300):
    fig = plt.figure(figsize=(8.0, 2.0), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 25)
    ax.axis("off")

    # background: vertical gradient + a soft diagonal light
    grad = np.linspace(0, 1, 512).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOT])
    ax.imshow(grad, extent=[0, 100, 0, 25], aspect="auto", cmap=cmap,
              zorder=0, interpolation="bilinear")
    gx, gy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 120))
    glow = np.exp(-(((gx - 0.16) ** 2) / 0.10 + ((gy - 0.75) ** 2) / 0.30))
    ax.imshow(glow, extent=[0, 100, 0, 25], aspect="auto", cmap="Blues_r",
              alpha=0.10, zorder=1, interpolation="bilinear")

    # ---- left: wordmark ------------------------------------------------------
    ax.text(6, 15.4, "PAPC", color=WHITE, fontsize=44, fontweight="bold",
            ha="left", va="center", zorder=5)
    # thin accent rule under the wordmark
    ax.plot([6.4, 20.5], [10.6, 10.6], color=ACCENT, lw=2.0,
            solid_capstyle="butt", zorder=5)

    ax.text(6.4, 7.4, _track("PREDICTIVE CODING FOR PRETRAINED ViTs"),
            color=MUTED, fontsize=6.2, ha="left", va="center", zorder=5)
    ax.text(6.4, 3.6,
            r"CAISc 2026      $w^{*}\approx257\,n^{-1.41}$      8 MedMNIST + CIFAR-100",
            color="#5f7495", fontsize=5.6, ha="left", va="center", zorder=5)

    # ---- right: the thesis, drawn as a minimal chart -------------------------
    x0, x1 = 62.0, 94.0
    yb, yt = 6.0, 19.0
    ax.plot([x0, x1], [yb, yb], color="#2b3f5c", lw=0.9, zorder=3)
    ax.plot([x0, x0], [yb, yt], color="#2b3f5c", lw=0.9, zorder=3)

    t = np.linspace(0, 1, 200)
    # AUC: flat (discriminability unchanged)
    auc = np.full_like(t, 0.80) + 0.005 * np.sin(t * 6)
    # ECE: moves a lot (calibration is the lever)
    ece = 0.14 + 0.44 * t ** 2.2
    ax.plot(x0 + t * (x1 - x0), yb + auc * (yt - yb), color=BLUE, lw=2.3, zorder=4)
    ax.plot(x0 + t * (x1 - x0), yb + ece * (yt - yb), color=ACCENT, lw=2.3, zorder=4)

    ax.text(x1 + 1.2, yb + 0.80 * (yt - yb), "AUC", color=BLUE, fontsize=5.8,
            ha="left", va="center", fontweight="bold", zorder=5)
    ax.text(x1 + 1.2, yb + 0.58 * (yt - yb), "ECE", color=ACCENT, fontsize=5.8,
            ha="left", va="center", fontweight="bold", zorder=5)
    ax.text(x0, yb - 1.9, _track("AUXILIARY WEIGHT  w"), color="#4c6083",
            fontsize=4.8, ha="left", va="center", zorder=5)

    # ---- centre: the claim ---------------------------------------------------
    ax.text(34.5, 15.0, "Calibration levers,", color=ACCENT, fontsize=10.5,
            fontweight="bold", ha="left", va="center", zorder=5)
    ax.text(34.5, 10.2, "not accuracy levers", color=BLUE, fontsize=10.5,
            fontweight="bold", ha="left", va="center", zorder=5)

    fig.savefig(out_path, dpi=dpi, facecolor=BG_BOT)
    plt.close(fig)
    print(f"wrote {out_path}  ({int(8.0 * dpi)}x{int(2.0 * dpi)} px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/banner.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_banner(args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
