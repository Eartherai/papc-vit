#!/usr/bin/env python
"""Render the repository hero banner to ``assets/banner.png``.

A self-contained matplotlib script (no external assets) so the banner can be
regenerated or restyled at will.

    python scripts/make_banner.py
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle  # noqa: E402


INK = "#0b1f3a"       # deep navy
INK2 = "#12305c"      # lighter navy for the gradient
ACCENT = "#e8833a"    # warm orange (the "calibration" accent)
BLUE = "#5aa0e6"      # cool blue (the "AUC" accent)
CREAM = "#f4ede2"


def make_banner(out_path, w=1600, h=460, dpi=200):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # vertical gradient background
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [INK2, INK])
    ax.imshow(grad, extent=[0, 16, 0, 4.6], aspect="auto", cmap=cmap, zorder=0)

    # --- left: schematic ViT blocks with a PC sidecar --------------------------
    y = 3.05
    xs = [0.7, 2.15, 3.6]
    for i, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x, y), 1.05, 0.72, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc="#20406e", ec=BLUE, lw=1.6, zorder=3))
        ax.text(x + 0.52, y + 0.36, f"$f_{i+1}$", color="white", ha="center", va="center",
                fontsize=13, zorder=4)
        if i < len(xs) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.05, y + 0.36), (xs[i + 1], y + 0.36),
                                         arrowstyle="-|>", mutation_scale=13, color="white", lw=1.6, zorder=3))
    # PC predictor + gate motif under block 1 -> 2
    ax.add_patch(FancyBboxPatch((2.05, 1.75), 1.1, 0.5, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fc="#3a2a14", ec=ACCENT, lw=1.5, zorder=3))
    ax.text(2.6, 2.0, "Diag. SSM", color=ACCENT, ha="center", va="center", fontsize=8.5, zorder=4)
    ax.add_patch(FancyArrowPatch((1.22, y), (2.35, 2.25), arrowstyle="-|>", mutation_scale=10,
                                 color=BLUE, lw=1.3, zorder=2))
    ax.add_patch(Circle((3.4, 2.7), 0.18, fc="#1f5d3a", ec="#6fdc9b", lw=1.4, zorder=4))
    ax.text(3.4, 2.7, "$g$", color="#c9ffe0", ha="center", va="center", fontsize=9, zorder=5)
    ax.add_patch(FancyArrowPatch((3.15, 2.0), (3.55, y - 0.02), arrowstyle="-|>", mutation_scale=9,
                                 color="#6fdc9b", lw=1.3, zorder=2))

    # --- right: wordmark + tagline --------------------------------------------
    X0 = 5.6
    ax.text(X0, 3.15, "PAPC", color="white", fontsize=60, fontweight="bold",
            ha="left", va="center", zorder=5)
    ax.text(X0 + 0.06, 2.18, "Predictive Coding for Pretrained ViTs",
            color=CREAM, fontsize=15, ha="left", va="center", zorder=5)

    # colored tagline laid out by measuring rendered text extents
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def run(x, y, parts, size):
        for text, color, weight in parts:
            t = ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
                        ha="left", va="center", zorder=5)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=rend)
            x = inv.transform((bb.x1, 0))[0]

    run(X0 + 0.08, 1.52, [
        ("Calibration levers", ACCENT, "bold"),
        (", not ", "#b9c6da", "normal"),
        ("accuracy levers", BLUE, "bold"),
    ], 13)

    # scaling-law chip
    ax.text(X0 + 0.08, 0.74,
            r"$w^* \approx 257\,n^{-1.41}$    ·    8 MedMNIST + CIFAR-100    ·    CAISc 2026",
            color="#8fa3bf", fontsize=10.5, ha="left", va="center", zorder=5)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/banner.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_banner(args.out)


if __name__ == "__main__":
    main()
