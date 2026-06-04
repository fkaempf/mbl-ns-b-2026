#!/usr/bin/env python
"""Train a dish2-ONLY ant/post model (analog of the dish3-only model), on the SAME
dish2 held-out frames the combined model used, so dish2 val error is comparable:
does training on dish2 alone beat the combined (dish2+dish3) model on dish2?

Run:  .venv/bin/python scripts/dl_dish2_train.py
Writes into DL/: dish2_only_kpnet.pt , dish2_only_val_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dl_dopamine_train import CACHE, DEVICE, DL, KPNet, VIDEOS, decode, preprocess  # noqa: E402

VID = "IMG_2859_dish2.mp4"
CROP = VIDEOS[VID]["crop"]
CSV = ROOT / "annotations" / "annotationsdish2.csv"
MODEL_OUT = DL / "dish2_only_kpnet.pt"
CDIR = CACHE / Path(VID).stem


def build():
    d = pd.read_csv(CSV).dropna(subset=["ant_x", "ant_y", "post_x", "post_y"])
    d = d[(d.video == VID)]
    d = d[d.groupby("frame").frame.transform("size") == 1].copy()
    d["bl"] = np.hypot(d.ant_x - d.post_x, d.ant_y - d.post_y)
    d = d[d.bl >= 5]
    CDIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEOS[VID]["clip"]))
    rows = []
    for r in d.itertuples():
        p = CDIR / f"{int(r.frame)}.png"
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame)); ok, img = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(p), img)
        rows.append((int(r.frame), r.ant_x, r.ant_y, r.post_x, r.post_y, r.bl))
    cap.release()
    return pd.DataFrame(rows, columns=["frame", "ant_x", "ant_y", "post_x", "post_y", "bl"])


class KP(Dataset):
    def __init__(self, lab, train):
        self.lab = lab.reset_index(drop=True); self.train = train

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]
        img = cv2.imread(str(CDIR / f"{int(r.frame)}.png"))
        x = preprocess(img, CROP)
        if self.train and np.random.rand() < 0.5:
            x = torch.clamp(x * float(np.random.uniform(0.85, 1.15)), 0, 1)
        x0, y0, w, h = CROP
        t = torch.tensor([(r.ant_x - x0) / w, (r.ant_y - y0) / h,
                          (r.post_x - x0) / w, (r.post_y - y0) / h], dtype=torch.float32)
        return x, t


def main():
    torch.manual_seed(0); np.random.seed(0)
    lab = build()
    # SAME dish2 split as the combined model: rng(0) shuffle of dish2 frames, 20% val
    rng = np.random.default_rng(0); f = lab.frame.values.copy(); rng.shuffle(f)
    val_f = set(f[:int(0.2 * len(f))].tolist())
    tr = lab[~lab.frame.isin(val_f)]; va = lab[lab.frame.isin(val_f)]
    print(f"dish2-only: train {len(tr)}  val {len(va)}  dev {DEVICE}")
    dl_tr = DataLoader(KP(tr, True), batch_size=16, shuffle=True)
    net = KPNet().to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.SmoothL1Loss()
    DL.mkdir(parents=True, exist_ok=True)

    def errs():
        net.eval(); e = []
        with torch.no_grad():
            for r in va.itertuples():
                img = cv2.imread(str(CDIR / f"{int(r.frame)}.png"))
                kp = decode(net(preprocess(img, CROP)[None].to(DEVICE)).cpu().numpy()[0], CROP)
                for (kx, ky), (tx, ty) in zip(kp, [(r.ant_x, r.ant_y), (r.post_x, r.post_y)]):
                    e.append(np.hypot(kx - tx, ky - ty))
        return np.array(e)

    best = 1e9
    for ep in range(40):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
        med = float(np.median(errs()))
        if med < best:
            best = med; torch.save(net.state_dict(), MODEL_OUT)
        print(f"ep {ep:2d}  dish2-val med {med:5.2f}px  (best {best:5.2f})")

    net.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
    e = errs(); bl = float(lab.bl.median())
    rep = (f"DISH2-ONLY model | val {len(va)} frames | body {bl:.0f}px | device {DEVICE}\n"
           f"  val median {np.median(e):.2f}px  mean {e.mean():.2f}  p90 {np.percentile(e,90):.2f}  "
           f"PCK@0.2 {100*np.mean(e<0.2*bl):.0f}%\n"
           f"  Compare to COMBINED model on dish2 (DL/dopamine_val_report.txt: 6.83px / PCK 70%).")
    print("\n" + rep)
    (DL / "dish2_only_val_report.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
