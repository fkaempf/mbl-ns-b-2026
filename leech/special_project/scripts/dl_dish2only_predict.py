#!/usr/bin/env python
"""Predict ant/post on the full dish2 clip with the DISH2-ONLY model.
Writes DL/IMG_2859_dish2_dish2only.mp4 + DL/dish2only_pred_IMG_2859_dish2.csv.

Run: .venv/bin/python scripts/dl_dish2only_predict.py [start] [n_frames]  (default whole clip)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dl_dopamine_train import KPNet, VIDEOS, decode, preprocess  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DL = ROOT / "DL"
VID = "IMG_2859_dish2.mp4"
CROP = VIDEOS[VID]["crop"]
MODEL = DL / "dish2_only_kpnet.pt"
OUT_MP4 = DL / "IMG_2859_dish2_dish2only.mp4"
OUT_CSV = DL / "dish2only_pred_IMG_2859_dish2.csv"
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000
BATCH = 64


def main():
    net = KPNet().to(DEVICE); net.load_state_dict(torch.load(MODEL, weights_only=True)); net.eval()
    cap = cv2.VideoCapture(str(VIDEOS[VID]["clip"]))
    W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30
    stop = min(int(cap.get(7)), start + nf)
    vw = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows, bi, bf, bo = [], [], [], []

    def flush():
        if not bi:
            return
        pred = net(torch.stack(bi).to(DEVICE)).detach().cpu().numpy()
        for j, fi in enumerate(bf):
            frame = bo[j]; pts = decode(pred[j], CROP)
            for k, (kx, ky) in enumerate(pts):
                rows.append({"frame": fi, "time_s": round(fi / FPS, 3), "track": 0,
                             "node": k, "x": round(kx, 1), "y": round(ky, 1)})
            (ax, ay), (px, py) = pts
            cv2.line(frame, (int(ax), int(ay)), (int(px), int(py)), (0, 200, 255), 2)
            cv2.circle(frame, (int(ax), int(ay)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(px), int(py)), 5, (255, 90, 0), -1)
            cv2.putText(frame, f"f{fi}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            vw.write(frame)
        bi.clear(); bf.clear(); bo.clear()

    for fi in range(start, stop):
        ok, img = cap.read()
        if not ok:
            break
        bi.append(preprocess(img, CROP)); bf.append(fi); bo.append(img)
        if len(bi) >= BATCH:
            flush()
        if fi % 5000 == 0:
            print(f"  {fi}/{stop}", flush=True)
    flush()
    cap.release(); vw.release()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_MP4} ({stop-start} frames)")


if __name__ == "__main__":
    main()
