#!/usr/bin/env python
"""Plot the HELD-OUT (validation) dish3 frames: annotations (green) vs model
predictions (red). Tight overlap confirms the model reproduces the hand labels.

Run: .venv/bin/python scripts/dl_dish3_plot.py  ->  annotations/dl_dish3_heldout.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from dl_dish3_train import CACHE, KPNet, MODEL_OUT, decode, preprocess  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

lab = pd.read_csv(CACHE / "labels.csv").set_index("frame")
val_frames = json.load(open(CACHE / "split.json"))["val"]
net = KPNet().to(DEVICE); net.load_state_dict(torch.load(MODEL_OUT, weights_only=True)); net.eval()

ann = {"ant": [], "post": []}; pred = {"ant": [], "post": []}; errs = {"ant": [], "post": []}
rows = []
with torch.no_grad():
    for f in val_frames:
        r = lab.loc[f]
        img = cv2.imread(str(CACHE / "frames" / f"{int(f)}.png"))
        p = decode(net(preprocess(img)[None].to(DEVICE)).cpu().numpy()[0])
        truth = [(r.ant_x, r.ant_y), (r.post_x, r.post_y)]
        for k, name in enumerate(("ant", "post")):
            ann[name].append(truth[k]); pred[name].append(p[k])
            errs[name].append(float(np.hypot(p[k][0] - truth[k][0], p[k][1] - truth[k][1])))
        rows.append(f)

ann = {k: np.array(v) for k, v in ann.items()}
pred = {k: np.array(v) for k, v in pred.items()}

fig, ax = plt.subplots(1, 2, figsize=(13, 6))

# Panel 1: spatial — held-out annotations (green) vs predictions (red), error lines
a = ax[0]
for name, mk in (("ant", "o"), ("post", "^")):
    a.scatter(ann[name][:, 0], ann[name][:, 1], c="green", s=18, marker=mk,
              label=f"annotation {name}", alpha=0.7)
    a.scatter(pred[name][:, 0], pred[name][:, 1], c="red", s=12, marker=mk,
              label=f"prediction {name}", alpha=0.7)
    for (tx, ty), (px, py) in zip(ann[name], pred[name]):
        a.plot([tx, px], [ty, py], color="gray", lw=0.4, alpha=0.5)
a.invert_yaxis(); a.set_aspect("equal"); a.set_title(f"Held-out frames (n={len(rows)}): annotation (green) vs prediction (red)")
a.set_xlabel("x (px)"); a.set_ylabel("y (px)"); a.legend(fontsize=8)

# Panel 2: per-frame error over time
b = ax[1]
order = np.argsort(rows); frames_sorted = np.array(rows)[order]
for name, col in (("ant", "tab:red"), ("post", "tab:blue")):
    b.plot(frames_sorted, np.array(errs[name])[order], ".-", ms=4, lw=0.6, color=col, label=name)
bl = float(lab["bl"].median())
b.axhline(0.1 * bl, color="green", ls="--", label=f"10% body len ({0.1*bl:.1f}px)")
b.set_xlabel("frame"); b.set_ylabel("keypoint error (px)")
b.set_title(f"Held-out error vs frame (median {np.median(errs['ant']+errs['post']):.1f}px)")
b.legend(fontsize=8)

fig.tight_layout()
out = ROOT / "DL" / "dl_dish3_heldout.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}  | held-out frames {len(rows)}  median err "
      f"{np.median(errs['ant']+errs['post']):.2f}px")
