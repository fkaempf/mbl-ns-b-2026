#!/usr/bin/env python
"""General per-dish ant/post keypoint trainer + full-clip predictor, handling N leeches.

Each dish: a ResNet18 -> 4*N coord regressor (N leeches x ant/post). For multi-leech
dishes the leeches are given a canonical order (sorted by centroid x) so targets are
consistent frame-to-frame. Reads canonical per-dish CSVs in annotations/by_dish/.

Train:    .venv/bin/python scripts/dl_dish_general.py train   dish1
Predict:  .venv/bin/python scripts/dl_dish_general.py predict dish4   (full clip)
Writes into DL/: dish{N}_kpnet.pt , dish{N}_val_report.txt , <stem>_dishonly.mp4 ,
                 dishonly_pred_<stem>.csv
"""
from __future__ import annotations

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
DL = ROOT / "DL"
S = 256
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DISHES = {
    "dish1": {"video": "IMG_2859_dish1.mp4", "csv": "annotations/by_dish/dish1.csv",
              "crop": (0, 40, 460, 520), "n": 1},
    "dish4": {"video": "IMG_2859_dish4.mp4", "csv": "annotations/by_dish/dish4.csv",
              "crop": (0, 0, 588, 588), "n": 2},
    "v2dish0": {"video": "Video2_dish0.mp4",
                "csv": "annotations/juv_v2dish0annotations.csv",
                "crop": (0, 0, 548, 548), "n": 2},
    "v1dish0": {"video": "Video1_dish0.mp4",
                "csv": "annotations/juv_v1_dish01_annotations.csv",
                "crop": (0, 0, 550, 550), "n": 2},
    "v1dish1": {"video": "Video1_dish1.mp4",
                "csv": "annotations/juv_v1_dish01_annotations.csv",
                "crop": (0, 0, 534, 534), "n": 2},
    "v2dish1": {"video": "Video2_dish1.mp4",
                "csv": "annotations/juv_v2_dish1_annotations.csv",
                "crop": (0, 0, 518, 518), "n": 2},
}


def preprocess(bgr, crop):
    x0, y0, w, h = crop
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y0:y0 + h, x0:x0 + w]
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(g)[None].repeat(3, 1, 1)


def decode(pred, crop, n):
    x0, y0, w, h = crop
    out = []
    for i in range(n):
        ax, ay, px, py = pred[4 * i:4 * i + 4]
        out.append([(float(ax) * w + x0, float(ay) * h + y0),
                    (float(px) * w + x0, float(py) * h + y0)])
    return out


def build_lab(cfg):
    """Frames with exactly n leeches each having ant+post; leeches sorted by centroid x."""
    d = pd.read_csv(ROOT / cfg["csv"]).dropna(subset=["ant_x", "ant_y", "post_x", "post_y"])
    d = d[d.video == cfg["video"]]
    n = cfg["n"]
    rows = []
    cdir = ROOT / "data" / "dl_general" / Path(cfg["video"]).stem
    cdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "clips" / cfg["video"]))
    for f, g in d.groupby("frame"):
        if len(g) != n:
            continue
        g = g.assign(cx=(g.ant_x + g.post_x) / 2).sort_values("cx")
        bl = np.hypot(g.ant_x - g.post_x, g.ant_y - g.post_y)
        if (bl < 5).any():
            continue
        p = cdir / f"{int(f)}.png"
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(p), img)
        coords = []
        for r in g.itertuples():
            coords += [r.ant_x, r.ant_y, r.post_x, r.post_y]
        rows.append((int(f), float(bl.median()), *coords))
    cap.release()
    cols = ["frame", "bl"] + [c for i in range(n) for c in
                              (f"ax{i}", f"ay{i}", f"px{i}", f"py{i}")]
    return pd.DataFrame(rows, columns=cols), cdir


class KP(Dataset):
    def __init__(self, lab, cfg, cdir, train):
        self.lab = lab.reset_index(drop=True); self.cfg = cfg; self.cdir = cdir; self.train = train

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]; crop = self.cfg["crop"]; n = self.cfg["n"]
        img = cv2.imread(str(self.cdir / f"{int(r.frame)}.png"))
        x = preprocess(img, crop)
        if self.train and np.random.rand() < 0.5:
            x = torch.clamp(x * float(np.random.uniform(0.85, 1.15)), 0, 1)
        x0, y0, w, h = crop
        t = []
        for i2 in range(n):
            t += [(r[f"ax{i2}"] - x0) / w, (r[f"ay{i2}"] - y0) / h,
                  (r[f"px{i2}"] - x0) / w, (r[f"py{i2}"] - y0) / h]
        return x, torch.tensor(t, dtype=torch.float32)


class KPNet(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        bb.fc = nn.Linear(512, n_out); self.bb = bb

    def forward(self, x):
        return torch.sigmoid(self.bb(x))


def model_path(key):
    return DL / f"{key}_kpnet.pt"


def train(key):
    cfg = DISHES[key]; n = cfg["n"]
    torch.manual_seed(0); np.random.seed(0)
    lab, cdir = build_lab(cfg)
    rng = np.random.default_rng(0); idx = rng.permutation(len(lab))
    nv = int(0.2 * len(lab)); va = lab.iloc[sorted(idx[:nv])]; tr = lab.iloc[sorted(idx[nv:])]
    print(f"{key}: {len(lab)} frames (n={n})  train {len(tr)}  val {len(va)}  dev {DEVICE}")
    dl_tr = DataLoader(KP(tr, cfg, cdir, True), batch_size=16, shuffle=True)
    net = KPNet(4 * n).to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.SmoothL1Loss()
    DL.mkdir(parents=True, exist_ok=True)

    def errs():
        net.eval(); e = []
        with torch.no_grad():
            for r in va.itertuples():
                img = cv2.imread(str(cdir / f"{int(r.frame)}.png"))
                pr = net(preprocess(img, cfg["crop"])[None].to(DEVICE)).cpu().numpy()[0]
                kp = decode(pr, cfg["crop"], n)
                for i2 in range(n):
                    for (kx, ky), (tx, ty) in zip(kp[i2],
                            [(getattr(r, f"ax{i2}"), getattr(r, f"ay{i2}")),
                             (getattr(r, f"px{i2}"), getattr(r, f"py{i2}"))]):
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
            best = med; torch.save(net.state_dict(), model_path(key))
        print(f"ep {ep:2d}  val med {med:5.2f}px  (best {best:5.2f})")
    net.load_state_dict(torch.load(model_path(key), weights_only=True))
    e = errs(); bl = float(lab.bl.median())
    rep = (f"{key.upper()} model ({cfg['video']}, {n} leech) | val {len(va)} frames | body {bl:.0f}px\n"
           f"  val median {np.median(e):.2f}px  mean {e.mean():.2f}  p90 {np.percentile(e,90):.2f}  "
           f"PCK@0.2 {100*np.mean(e<0.2*bl):.0f}%")
    print("\n" + rep); (DL / f"{key}_val_report.txt").write_text(rep + "\n")


def predict(key, start=0, nf=10_000_000):
    cfg = DISHES[key]; n = cfg["n"]; crop = cfg["crop"]; stem = Path(cfg["video"]).stem
    net = KPNet(4 * n).to(DEVICE); net.load_state_dict(torch.load(model_path(key), weights_only=True)); net.eval()
    cap = cv2.VideoCapture(str(ROOT / "clips" / cfg["video"]))
    W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30
    stop = min(int(cap.get(7)), start + nf)
    out_mp4 = DL / f"{stem}_dishonly.mp4"; out_csv = DL / f"dishonly_pred_{stem}.csv"
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    COL = [(0, 0, 255), (255, 90, 0), (0, 200, 0), (200, 0, 200)]
    rows, bi, bf, bo = [], [], [], []

    def flush():
        if not bi:
            return
        pred = net(torch.stack(bi).to(DEVICE)).detach().cpu().numpy()
        for j, fi in enumerate(bf):
            frame = bo[j]
            for li, (a, p) in enumerate(decode(pred[j], crop, n)):
                rows.append({"frame": fi, "time_s": round(fi / FPS, 3), "track": li, "node": 0,
                             "x": round(a[0], 1), "y": round(a[1], 1)})
                rows.append({"frame": fi, "time_s": round(fi / FPS, 3), "track": li, "node": 1,
                             "x": round(p[0], 1), "y": round(p[1], 1)})
                c = COL[li % len(COL)]
                cv2.line(frame, (int(a[0]), int(a[1])), (int(p[0]), int(p[1])), c, 2)
                cv2.circle(frame, (int(a[0]), int(a[1])), 5, c, -1)
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, c, 2)
            cv2.putText(frame, f"f{fi}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            vw.write(frame)
        bi.clear(); bf.clear(); bo.clear()

    for fi in range(start, stop):
        ok, img = cap.read()
        if not ok:
            break
        bi.append(preprocess(img, crop)); bf.append(fi); bo.append(img)
        if len(bi) >= 64:
            flush()
        if fi % 5000 == 0:
            print(f"  {fi}/{stop}", flush=True)
    flush()
    cap.release(); vw.release()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} rows) and {out_mp4} ({stop-start} frames)")


if __name__ == "__main__":
    mode, key = sys.argv[1], sys.argv[2]
    if mode == "train":
        train(key)
    else:
        predict(key, *(int(a) for a in sys.argv[3:]))
