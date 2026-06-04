#!/usr/bin/env python
"""Train a combined ant/post keypoint model on dish3 + dish2 (dopamine_all_annotations.csv).

Per-video crops (the two dishes differ in size and leech region). The dish3 held-out
frames are the SAME as the dish3-only model's val split (data/dl_dish3/split.json), so the
dish3 val error is directly comparable: does adding dish2 data help or hurt dish3?

Run:  .venv/bin/python scripts/dl_dopamine_train.py
Writes into DL/: dopamine_kpnet.pt , dopamine_val_report.txt
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
CSV = ROOT / "annotations" / "dopamine_all_annotations.csv"
CACHE = ROOT / "data" / "dl_dopamine"
DL = ROOT / "DL"
MODEL_OUT = DL / "dopamine_kpnet.pt"
DISH3_SPLIT = ROOT / "data" / "dl_dish3" / "split.json"   # reuse dish3 val frames
S = 256
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# per-video square crops covering each dish's leech band (origin 0,0; leeches sit top-left)
VIDEOS = {
    "IMG_2859_dish3.mp4": {"clip": ROOT / "clips" / "IMG_2859_dish3.mp4", "crop": (0, 0, 440, 440)},
    "IMG_2859_dish2.mp4": {"clip": ROOT / "clips" / "IMG_2859_dish2.mp4", "crop": (0, 0, 480, 480)},
}


def preprocess(bgr, crop):
    x0, y0, w, h = crop
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y0:y0 + h, x0:x0 + w]
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(g)[None].repeat(3, 1, 1)


def decode(pred, crop):
    x0, y0, w, h = crop
    ax, ay, px, py = pred
    return [(float(ax) * w + x0, float(ay) * h + y0), (float(px) * w + x0, float(py) * h + y0)]


def build():
    df = pd.read_csv(CSV)
    rows = []
    for vid, cfg in VIDEOS.items():
        d = df[(df.video == vid)].dropna(subset=["ant_x", "ant_y", "post_x", "post_y"])
        d = d[d.groupby("frame").frame.transform("size") == 1].copy()
        d["bl"] = np.hypot(d.ant_x - d.post_x, d.ant_y - d.post_y)
        d = d[d.bl >= 5]
        cdir = CACHE / Path(vid).stem
        cdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(cfg["clip"]))
        for r in d.itertuples():
            p = cdir / f"{int(r.frame)}.png"
            if not p.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame)); ok, img = cap.read()
                if not ok:
                    continue
                cv2.imwrite(str(p), img)
            rows.append((vid, int(r.frame), r.ant_x, r.ant_y, r.post_x, r.post_y, r.bl))
        cap.release()
    return pd.DataFrame(rows, columns=["video", "frame", "ant_x", "ant_y", "post_x", "post_y", "bl"])


class KP(Dataset):
    def __init__(self, lab, train):
        self.lab = lab.reset_index(drop=True); self.train = train

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]
        crop = VIDEOS[r.video]["crop"]
        img = cv2.imread(str(CACHE / Path(r.video).stem / f"{int(r.frame)}.png"))
        x = preprocess(img, crop)
        if self.train and np.random.rand() < 0.5:
            x = torch.clamp(x * float(np.random.uniform(0.85, 1.15)), 0, 1)
        x0, y0, w, h = crop
        t = torch.tensor([(r.ant_x - x0) / w, (r.ant_y - y0) / h,
                          (r.post_x - x0) / w, (r.post_y - y0) / h], dtype=torch.float32)
        return x, t


class KPNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        bb.fc = nn.Linear(512, 4); self.bb = bb

    def forward(self, x):
        return torch.sigmoid(self.bb(x))


def main():
    torch.manual_seed(0); np.random.seed(0)
    lab = build()
    # dish3 val = the dish3-only model's held-out frames (fair comparison)
    d3_val = set(json.load(open(DISH3_SPLIT))["val"]) if DISH3_SPLIT.exists() else set()
    is_d3 = lab.video == "IMG_2859_dish3.mp4"
    d3 = lab[is_d3]; d2 = lab[~is_d3]
    d3_val_mask = d3.frame.isin(d3_val)
    # dish2 random 80/20
    rng = np.random.default_rng(0); d2f = d2.frame.values.copy(); rng.shuffle(d2f)
    d2_val = set(d2f[:int(0.2 * len(d2f))].tolist())
    d2_val_mask = d2.frame.isin(d2_val)
    train = pd.concat([d3[~d3_val_mask], d2[~d2_val_mask]])
    val = pd.concat([d3[d3_val_mask], d2[d2_val_mask]])
    print(f"train {len(train)} (d3 {len(d3[~d3_val_mask])} + d2 {len(d2[~d2_val_mask])})  "
          f"val {len(val)} (d3 {int(d3_val_mask.sum())} + d2 {int(d2_val_mask.sum())})  dev {DEVICE}")

    dl_tr = DataLoader(KP(train, True), batch_size=16, shuffle=True)
    dl_va = DataLoader(KP(val, False), batch_size=32)
    net = KPNet().to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.SmoothL1Loss()
    DL.mkdir(parents=True, exist_ok=True)

    def eval_errs(rows):
        net.eval(); per = {}
        with torch.no_grad():
            for vi in range(len(rows)):
                r = rows.iloc[vi]; crop = VIDEOS[r.video]["crop"]
                img = cv2.imread(str(CACHE / Path(r.video).stem / f"{int(r.frame)}.png"))
                pr = net(preprocess(img, crop)[None].to(DEVICE)).cpu().numpy()[0]
                kp = decode(pr, crop)
                for (kx, ky), (tx, ty) in zip(kp, [(r.ant_x, r.ant_y), (r.post_x, r.post_y)]):
                    per.setdefault(r.video, []).append(np.hypot(kx - tx, ky - ty))
        return per

    best = 1e9
    for ep in range(40):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
        per = eval_errs(val)
        med = float(np.median(np.concatenate(list(per.values()))))
        if med < best:
            best = med; torch.save(net.state_dict(), MODEL_OUT)
        d3m = np.median(per.get("IMG_2859_dish3.mp4", [np.nan]))
        print(f"ep {ep:2d}  val med {med:5.2f}px  dish3-val {d3m:5.2f}px  (best {best:5.2f})")

    net.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
    per = eval_errs(val)
    rep = [f"COMBINED (dish2+dish3) model | device {DEVICE}"]
    for vid, e in per.items():
        e = np.array(e); bl = float(lab[lab.video == vid].bl.median())
        rep.append(f"  {vid}: val median {np.median(e):.2f}px  mean {e.mean():.2f}  p90 "
                   f"{np.percentile(e,90):.2f}  (body {bl:.0f}px)  PCK@0.2 {100*np.mean(e<0.2*bl):.0f}%")
    rep.append("\nCompare dish3 val median to the dish3-only model (see DL/dish3_only_val_report.txt: 2.85px).")
    report = "\n".join(rep); print("\n" + report)
    (DL / "dopamine_val_report.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
