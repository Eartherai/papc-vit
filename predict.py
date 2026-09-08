#!/usr/bin/env python
"""Run a trained PAPC/ViT checkpoint on a single image.

Produces a checkpoint first by training with ``--save-models``::

    python train.py --dataset dermamnist \
        --conditions papc --seeds 1 --save-models checkpoints --output-dir results_repro

then classify any image::

    python predict.py --checkpoint checkpoints/dermamnist_papc_s0.pt \
        --image path/to/lesion.png --topk 3

The script reports the predicted class and its confidence — the quantity this
paper shows PAPC reshapes. It needs only a CPU.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


from papc import PAPCViT
from datasets import build_tf


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PAPCViT(ckpt["model_name"], ckpt["img_size"], ckpt["num_classes"],
                    ckpt["in_chans"], pretrained=False, **ckpt["model_kw"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to a .pt saved by --save-models.")
    ap.add_argument("--image", required=True, help="Path to an input image.")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    ic = ckpt["in_chans"]

    img = Image.open(args.image).convert("RGB" if ic == 3 else "L")
    tf = build_tf(ckpt["img_size"], ic, train=False)
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits + model(torch.flip(x, [-1]))[0]      # test-time flip averaging
        probs = F.softmax(logits / 2, dim=-1)[0].cpu().numpy()

    k = min(args.topk, len(probs))
    order = np.argsort(probs)[::-1][:k]
    print(f"\nImage:      {args.image}")
    print(f"Checkpoint: {args.checkpoint}  (task={ckpt['task']}, classes={ckpt['num_classes']})")
    print(f"Prediction: class {int(order[0])}  (confidence {probs[order[0]]:.4f})\n")
    print("Top-k:")
    for r in order:
        bar = "█" * int(round(probs[r] * 30))
        print(f"  class {int(r):>2}  {probs[r]:6.4f}  {bar}")


if __name__ == "__main__":
    main()
