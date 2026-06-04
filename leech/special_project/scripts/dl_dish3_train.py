#!/usr/bin/env python
"""Train a ResNet18 -> 2-heatmap keypoint model on the IMG_2859_dish3 hand annotations
to predict leech anterior/posterior, and validate on a held-out split.

Confirms the annotations: if the model hits low median error (vs ~45px body length) on
frames it never trained on, the labels carry a consistent, learnable ant/post signal.

Run:  .venv/bin/python scripts/dl_dish3_train.py
Writes into DL/: dish3_kpnet.pt , dl_dish3_val_report.txt , dl_dish3_val_scatter.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VIDEO = ROOT / "clips" / "IMG_2859_dish3.mp4"
CSV = ROOT / "annotations" / "annotations.csv"
CACHE = ROOT / "data" / "dl_dish3"
DL = ROOT / "DL"                          # all DL results live here
MODEL_OUT = DL / "dish3_kpnet.pt"
S = 256                                   # model input size
# Fixed crop covering the leech band (all labels fall in x[77,186] y[33,243]); cropping
# makes the ~45px leech large relative to input — full-frame lost the signal. Origin (0,0)
# so normalized target = (x/CROP_W, y/CROP_H).
CROP_X0, CROP_Y0, CROP_W, CROP_H = 0, 0, 300, 300
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def preprocess(bgr):
    """BGR full frame -> (3,S,S) float tensor of the fixed crop, /255 grayscale->3ch."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[CROP_Y0:CROP_Y0 + CROP_H, CROP_X0:CROP_X0 + CROP_W]
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(g)[None].repeat(3, 1, 1)


def decode(pred_row):
    """4 normalized coords -> [(ant_x,ant_y),(post_x,post_y)] in full-frame pixels."""
    ax, ay, px, py = pred_row
    return [(float(ax) * CROP_W + CROP_X0, float(ay) * CROP_H + CROP_Y0),
            (float(px) * CROP_W + CROP_X0, float(py) * CROP_H + CROP_Y0)]


def build_dataset():
    """Single-leech labeled frames with both ant+post; cache each frame to PNG once."""
    df = pd.read_csv(CSV)
    d = df[(df.video == "IMG_2859_dish3.mp4")].dropna(subset=["ant_x", "ant_y", "post_x", "post_y"])
    d = d[d.groupby("frame")["frame"].transform("size") == 1]          # single-leech frames
    d = d.copy()
    d["bl"] = np.hypot(d.ant_x - d.post_x, d.ant_y - d.post_y)
    d = d[d.bl >= 5].sort_values("frame").reset_index(drop=True)        # drop degenerate
    (CACHE / "frames").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    W = int(cap.get(3)); H = int(cap.get(4))
    rows = []
    for r in d.itertuples():
        p = CACHE / "frames" / f"{int(r.frame)}.png"
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, img = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(p), img)
        rows.append((int(r.frame), r.ant_x, r.ant_y, r.post_x, r.post_y, r.bl))
    cap.release()
    lab = pd.DataFrame(rows, columns=["frame", "ant_x", "ant_y", "post_x", "post_y", "bl"])
    lab.to_csv(CACHE / "labels.csv", index=False)
    return lab, W, H


class LeechKP(Dataset):
    """Cropped frame -> 4 normalized target coords (ant_x,ant_y,post_x,post_y in [0,1])."""

    def __init__(self, lab, W, H, train):
        self.lab = lab.reset_index(drop=True); self.train = train

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]
        img = cv2.imread(str(CACHE / "frames" / f"{int(r.frame)}.png"))
        x = preprocess(img)
        if self.train and np.random.rand() < 0.5:
            x = torch.clamp(x * float(np.random.uniform(0.85, 1.15)), 0, 1)  # label-safe
        t = torch.tensor([(r.ant_x - CROP_X0) / CROP_W, (r.ant_y - CROP_Y0) / CROP_H,
                          (r.post_x - CROP_X0) / CROP_W, (r.post_y - CROP_Y0) / CROP_H],
                         dtype=torch.float32)
        return x, t


class KPNet(nn.Module):
    """resnet18 backbone -> 4 sigmoid coords (ant_x,ant_y,post_x,post_y), normalized."""

    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        bb.fc = nn.Linear(512, 4)
        self.bb = bb

    def forward(self, x):
        return torch.sigmoid(self.bb(x))


def main():
    torch.manual_seed(0); np.random.seed(0)
    lab, W, H = build_dataset()
    frames = lab.frame.values
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    n_val = int(round(0.2 * len(frames)))
    val_idx = set(order[:n_val].tolist())
    tr = lab.iloc[[i for i in range(len(lab)) if i not in val_idx]]
    va = lab.iloc[sorted(val_idx)]
    print(f"dataset: {len(lab)} frames  train {len(tr)}  val {len(va)}  device {DEVICE}")
    json.dump({"train": tr.frame.tolist(), "val": va.frame.tolist()},
              open(CACHE / "split.json", "w"))

    dl_tr = DataLoader(LeechKP(tr, W, H, True), batch_size=16, shuffle=True)
    dl_va = DataLoader(LeechKP(va, W, H, False), batch_size=32)
    net = KPNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()

    DL.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(40):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = lossf(net(x), y); loss.backward(); opt.step()
        # validate: median pixel error
        net.eval(); errs = []
        with torch.no_grad():
            vi = 0
            for x, y in dl_va:
                pred = net(x.to(DEVICE)).cpu().numpy()        # (B,4) normalized
                for b in range(len(pred)):
                    r = va.iloc[vi]; vi += 1
                    for (kx, ky), (tx, ty) in zip(decode(pred[b]),
                                                  [(r.ant_x, r.ant_y), (r.post_x, r.post_y)]):
                        errs.append(np.hypot(kx - tx, ky - ty))
        med = float(np.median(errs))
        if med < best:
            best = med
            torch.save(net.state_dict(), MODEL_OUT)
        print(f"ep {ep:2d}  val median err {med:5.2f}px  (best {best:5.2f})")

    # final report on best model
    net.load_state_dict(torch.load(MODEL_OUT, weights_only=True)); net.eval()
    bl = float(lab.bl.median())
    per = {"ant": [], "post": []}
    rows_out = []
    with torch.no_grad():
        vi = 0
        for x, y in dl_va:
            pred = net(x.to(DEVICE)).cpu().numpy()
            for b in range(len(pred)):
                r = va.iloc[vi]; vi += 1
                kp = decode(pred[b])
                e = {}
                for (kx, ky), name, (tx, ty) in [(kp[0], "ant", (r.ant_x, r.ant_y)),
                                                 (kp[1], "post", (r.post_x, r.post_y))]:
                    e[name] = np.hypot(kx - tx, ky - ty); per[name].append(e[name])
                rows_out.append((int(r.frame), e["ant"], e["post"]))
    allerr = np.array(per["ant"] + per["post"])
    rep = []
    rep.append(f"Held-out val: {len(va)} frames | median body length {bl:.1f}px | device {DEVICE}")
    for name in ("ant", "post"):
        a = np.array(per[name])
        rep.append(f"  {name}: median {np.median(a):.2f}px  mean {a.mean():.2f}  p90 {np.percentile(a,90):.2f}")
    rep.append(f"  overall median {np.median(allerr):.2f}px  ({100*np.median(allerr)/bl:.0f}% of body length)")
    rep.append(f"  PCK@0.1 (<{0.1*bl:.1f}px) {100*np.mean(allerr<0.1*bl):.0f}%   "
               f"PCK@0.2 (<{0.2*bl:.1f}px) {100*np.mean(allerr<0.2*bl):.0f}%")
    worst = sorted(rows_out, key=lambda r: max(r[1], r[2]), reverse=True)[:10]
    rep.append("  worst-10 val frames (frame, ant_err, post_err) — inspect for label issues:")
    for f, ea, ep_ in worst:
        rep.append(f"    frame {f}: ant {ea:.1f}px post {ep_:.1f}px")
    report = "\n".join(rep)
    print("\n" + report)
    (DL / "dl_dish3_val_report.txt").write_text(report + "\n")

    # scatter pred-vs-true magnitude
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].hist(allerr, bins=30); ax[0].axvline(0.1*bl, color="r", ls="--")
    ax[0].set_title("held-out keypoint error (px)"); ax[0].set_xlabel("px")
    ax[1].scatter(per["ant"], per["post"], s=8, alpha=0.5)
    ax[1].set_xlabel("ant err px"); ax[1].set_ylabel("post err px"); ax[1].set_title("per-frame")
    fig.tight_layout(); fig.savefig(DL / "dl_dish3_val_scatter.png", dpi=110)
    print("saved model + report + scatter")


if __name__ == "__main__":
    main()
