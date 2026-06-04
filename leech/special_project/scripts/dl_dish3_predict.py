#!/usr/bin/env python
"""Predict leech ant/post on IMG_2859_dish3 with the trained KPNet, write a CSV under a
distinct name and an overlay video so the results can be checked against the annotations.

Run:  .venv/bin/python scripts/dl_dish3_predict.py [start] [n_frames]
Writes: annotations/dl_pred_IMG_2859_dish3.csv , tracked/IMG_2859_dish3_dl.mp4
By default predicts a continuous window (start=0, n=9000 ~5 min) for an easy visual check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl_dish3_train import KPNet, decode, preprocess  # reuse model + utils  # noqa: E402

VIDEO = ROOT / "clips" / "IMG_2859_dish3.mp4"
MODEL = ROOT / "DL" / "dish3_kpnet.pt"
OUT_CSV = ROOT / "DL" / "dl_pred_IMG_2859_dish3.csv"
OUT_MP4 = ROOT / "DL" / "IMG_2859_dish3_dl.mp4"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000   # default: whole clip
BATCH = 64


def main():
    net = KPNet().to(DEVICE)
    net.load_state_dict(torch.load(MODEL, weights_only=True))
    net.eval()

    cap = cv2.VideoCapture(str(VIDEO))
    W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30
    total = int(cap.get(7)); stop = min(total, start + nf)
    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    rows = []
    buf_imgs, buf_idx, buf_orig = [], [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs).to(DEVICE)
        with torch.no_grad():
            pred = net(x).cpu().numpy()                    # (B,4) normalized
        for j, fi in enumerate(buf_idx):
            frame = buf_orig[j]
            pts = decode(pred[j])
            for k, (kx, ky) in enumerate(pts):
                rows.append({"frame": fi, "time_s": round(fi / FPS, 3), "track": 0,
                             "node": k, "x": round(kx, 1), "y": round(ky, 1)})
            (ax, ay), (px, py) = pts
            cv2.line(frame, (int(ax), int(ay)), (int(px), int(py)), (0, 200, 255), 2)
            cv2.circle(frame, (int(ax), int(ay)), 5, (0, 0, 255), -1)    # ant red
            cv2.circle(frame, (int(px), int(py)), 5, (255, 90, 0), -1)   # post blue
            cv2.putText(frame, f"f{fi}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            vw.write(frame)
        buf_imgs.clear(); buf_idx.clear(); buf_orig.clear()

    for fi in range(start, stop):
        ok, img = cap.read()
        if not ok:
            break
        buf_imgs.append(preprocess(img)); buf_idx.append(fi); buf_orig.append(img)
        if len(buf_imgs) >= BATCH:
            flush()
        if fi % 3000 == 0:
            print(f"  {fi}/{stop}", flush=True)
    flush()
    cap.release(); vw.release()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_MP4} ({stop-start} frames)")


if __name__ == "__main__":
    main()
