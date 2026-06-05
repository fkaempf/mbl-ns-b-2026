#!/usr/bin/env python
"""Condensed (<=20 page) report PDF: cover, findings/methods, then one section per
page with a grid of figures + one-line captions. Reuses FIGURE_GUIDE.md + plots/."""
import re
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_calcium")
GUIDE = BASE / "FIGURE_GUIDE.md"
PDF = BASE / "CALCIUM_REPORT_CONDENSED.pdf"
A4 = (8.27, 11.69)
ACCENT = "#1f3a4d"
GREY = "#666666"


def strip_md(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", s)
    return s


# ---- parse figures grouped by section ----
sections = []          # [(section_title, [(name, img, oneliner), ...])]
cur = None
lines = GUIDE.read_text().splitlines()
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("# ") and not ln.startswith("## "):
        cur = (strip_md(ln[2:].strip()), [])
        sections.append(cur)
        i += 1
        continue
    if ln.startswith("### "):
        name = strip_md(ln[4:].strip())
        i += 1
        img = None
        what = ""
        readit = ""
        while i < len(lines) and not lines[i].startswith("#"):
            s = lines[i].strip()
            m = re.match(r"!\[.*?\]\((.+?)\)", s)
            if m:
                img = m.group(1)
            elif s.startswith("- **What**") or s.startswith("- What"):
                what = strip_md(re.sub(r"^- \**What\**:?\s*", "", s))
            elif s.startswith("- **How to read**") or s.startswith("- How to read"):
                readit = strip_md(re.sub(r"^- \**How to read\**:?\s*", "", s))
            i += 1
        # one-liner: prefer "what" first sentence
        base = what or readit
        one = base.split(". ")[0].rstrip(".")
        if cur is not None and img:
            cur[1].append((name, img, one))
        continue
    i += 1

# keep only sections that actually hold figures, in order
sections = [s for s in sections if s[1]]


def fit(ax_box, ar):
    """Given a box (x,y,w,h) in fig coords and image aspect ratio (w/h), return the
    sub-box that fits the image preserving aspect, centered, accounting for A4 shape."""
    x, y, w, h = ax_box
    page_ar = A4[0] / A4[1]
    box_phys_ar = (w / h) * page_ar       # physical aspect of the box
    if ar > box_phys_ar:
        iw = w
        ih = w * (A4[0] / A4[1]) / ar
    else:
        ih = h
        iw = h * ar * (A4[1] / A4[0])
    return (x + (w - iw) / 2, y + (h - ih) / 2, iw, ih)


def footer(fig, page, total):
    fig.add_artist(plt.Line2D([0.085, 0.915], [0.045, 0.045], color="#cccccc", lw=0.6,
                              transform=fig.transFigure))
    fig.text(0.5, 0.032, f"helobdella_LeechNo2_elavgcamp6m-17    page {page} of {total}",
             ha="center", fontsize=7.5, color=GREY)


NARR1_TITLE = "What we did"
NARR1 = [
    "Five-stage pipeline on the raw movie (692 frames, 700x700, fs 2.882 Hz, ~240 s, "
    "single-channel pan-neuronal GCaMP6m, 10x juvenile leech, after dopamine).",
    "Stage 1 QC and motion: photobleaching, rigid and non-rigid drift, correlation image.",
    "Stage 2 segmentation: suite2p (24 ROIs, kept as a cross-check) and a primary "
    "correlation-blob plus watershed method (180 active patches).",
    "Stage 3 and 4: temporal (spectra, spectrogram, autocorrelation, Hilbert) and spatial "
    "(per-pixel power and phase, traveling-wave test, region coherence) characterisation.",
    "Intricate visualization pass: 25 detailed figures plus a cycle-averaged movie.",
    "10x and a preliminary line, so everything is read at the ganglion or regional level, "
    "not as single neurons. The 180 patches are active regions, not confirmed cells.",
]
NARR2_TITLE = "Headline result and how to interpret it"
NARR2 = [
    "One strong, clock-like, ganglion-wide calcium oscillation at 0.372 Hz (period 2.69 s), "
    "sustained undamped for the full ~240 s and synchronous across the ganglion.",
    "Frequency: dominant peak 0.372 Hz with a 0.73 Hz harmonic, ~230x above the spectral "
    "floor, well below the 1.44 Hz Nyquist, so the fundamental is not aliased.",
    "Regularity: 88 cycles, inter-peak interval CV 0.077, tight Poincare cluster, stable "
    "spectrogram band.",
    "Synchrony: vertical kymograph stripes, whole-ganglion brightening together, flat "
    "phase-versus-position, inter-region coherence ~0.88. No traveling wave.",
    "Nuance: a strong subset carries the rhythm. ~93% of patches peak within 0.05 Hz of "
    "0.37 Hz and ~40% follow the ganglion mean at r>0.5; the rest are noisy (low-photon 10x).",
    "Interpretation: a synchronous ganglion-wide oscillation suggests a shared network "
    "drive. Labelled after dopamine, but there is no pre-dopamine baseline here, so no "
    "causal claim. Motion correction was verified unnecessary (0 px drift, ~0.5% bleaching).",
]
METRICS = [
    ("frame rate", "2.882 Hz"), ("duration", "~240 s (692 frames)"),
    ("dominant rhythm", "0.372 Hz (period 2.69 s)"), ("cycles / interval CV", "88 / 0.077"),
    ("active patches (primary)", "180 (suite2p cross-check 24)"),
    ("inter-region coherence at 0.37 Hz", "~0.88"),
    ("tissue pixels rhythmic", "~93%"), ("phase circular std (tissue)", "1.13 rad"),
    ("photobleaching / rigid drift", "~0.5% / 0 px (no motion correction)"),
]

NCOLS, NROWS = 2, 3      # 6 figures per page; every section has <=6 figures
total_pages = 1 + 2 + len(sections)


def narrative_page(pdf, page, title, bullets, metrics=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.085, 0.93, title, fontsize=16, weight="bold", color=ACCENT)
    fig.add_artist(plt.Line2D([0.085, 0.915], [0.915, 0.915], color=ACCENT, lw=1.2,
                              transform=fig.transFigure))
    y = 0.875
    for b in bullets:
        wrapped = textwrap.fill("- " + b, 96, subsequent_indent="  ")
        n = wrapped.count("\n") + 1
        fig.text(0.085, y, wrapped, fontsize=10, va="top")
        y -= 0.0165 * n + 0.012
    if metrics:
        y -= 0.01
        fig.text(0.085, y, "Key numbers", fontsize=12, weight="bold", color=ACCENT)
        y -= 0.03
        for k, v in metrics:
            fig.text(0.10, y, k, fontsize=9.5, va="top")
            fig.text(0.56, y, v, fontsize=9.5, va="top", weight="bold")
            y -= 0.022
    footer(fig, page, total_pages)
    pdf.savefig(fig); plt.close(fig)


with PdfPages(PDF) as pdf:
    # ---- cover ----
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.74, "Calcium imaging report", ha="center", fontsize=24,
             weight="bold", color=ACCENT)
    fig.add_artist(plt.Line2D([0.30, 0.70], [0.71, 0.71], color=ACCENT, lw=1.2,
                              transform=fig.transFigure))
    fig.text(0.5, 0.675, "helobdella_LeechNo2_elavgcamp6m-17", ha="center", fontsize=13,
             weight="bold")
    fig.text(0.5, 0.645, "Pan-neuronal GCaMP6m, 10x juvenile leech ganglion, after dopamine",
             ha="center", fontsize=10, style="italic", color=GREY)
    abstract = ("Condensed report. A single strong, clock-like, ganglion-wide calcium "
                "oscillation at 0.37 Hz (period 2.7 s), sustained for the full ~240 s "
                "recording and spatially synchronous, with no traveling wave.")
    fig.text(0.5, 0.50, textwrap.fill(abstract, 74), ha="center", fontsize=10.5)
    fig.text(0.5, 0.10, "Prepared 2026-06-05", ha="center", fontsize=9, color=GREY)
    pdf.savefig(fig); plt.close(fig)

    # ---- narrative ----
    narrative_page(pdf, 2, NARR1_TITLE, NARR1)
    narrative_page(pdf, 3, NARR2_TITLE, NARR2, metrics=METRICS)

    # ---- figure plates: one section per page ----
    page = 4
    for sect_title, figs in sections:
        fig = plt.figure(figsize=A4)
        fig.text(0.085, 0.955, sect_title, fontsize=13, weight="bold", color=ACCENT)
        fig.add_artist(plt.Line2D([0.085, 0.915], [0.942, 0.942], color=ACCENT, lw=1.0,
                                  transform=fig.transFigure))
        # grid area: y from 0.07 to 0.925
        gx0, gx1, gy0, gy1 = 0.06, 0.94, 0.075, 0.925
        cw = (gx1 - gx0) / NCOLS
        ch = (gy1 - gy0) / NROWS
        for idx, (name, img, one) in enumerate(figs[:NCOLS * NROWS]):
            r = idx // NCOLS
            c = idx % NCOLS
            cx = gx0 + c * cw
            cy = gy1 - (r + 1) * ch
            # caption uses bottom ~0.052 of the cell; image uses the rest
            cap_h = 0.050
            img_box = (cx + 0.012, cy + cap_h, cw - 0.024, ch - cap_h - 0.012)
            p = BASE / img
            if p.exists():
                im = mpimg.imread(p)
                ar = im.shape[1] / im.shape[0]
                bx, by, bw, bh = fit(img_box, ar)
                ax = fig.add_axes([bx, by, bw, bh])
                ax.imshow(im); ax.axis("off")
            fig.text(cx + 0.012, cy + cap_h - 0.006, name, fontsize=7.2,
                     weight="bold", color=ACCENT, va="top")
            cap = textwrap.fill(one, 58)
            cap = "\n".join(cap.split("\n")[:2])
            fig.text(cx + 0.012, cy + cap_h - 0.020, cap, fontsize=6.6, va="top")
        footer(fig, page, total_pages)
        pdf.savefig(fig); plt.close(fig)
        page += 1

    d = pdf.infodict()
    d["Title"] = "Calcium imaging report (condensed): helobdella_LeechNo2_elavgcamp6m-17"

# compress
try:
    tmp = str(PDF) + ".tmp"
    subprocess.run(["gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
                    "-dPDFSETTINGS=/ebook", "-dNOPAUSE", "-dQUIET", "-dBATCH",
                    f"-sOutputFile={tmp}", str(PDF)], check=True)
    Path(tmp).replace(PDF)
except Exception as e:
    print("gs compress skipped:", e)

print(f"wrote {PDF.name}: {total_pages} pages, {len(sections)} section plates, "
      f"{sum(len(s[1]) for s in sections)} figures, {PDF.stat().st_size/1e6:.1f} MB")
