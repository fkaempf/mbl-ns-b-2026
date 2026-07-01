"""Compile the key analysis plots into one multi-page review PDF.

PDF-source plots (the vector barrier traces and the vector cluster overview) are
**embedded as real vector pages** (zoomable, not screenshots); PNG plots are placed
one per page. Section dividers separate the groups.

Run from this directory:  python make_review_pdf.py
"""

import glob
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pypdf import PdfReader, PdfWriter

from utils import PLOTS_DIR

OUT = os.path.join(PLOTS_DIR, sys.argv[1] if len(sys.argv) > 1 else "review.pdf")
TMP = tempfile.mkdtemp()
_seq = [0]


def _tmp(ext):
    _seq[0] += 1
    return os.path.join(TMP, f"{_seq[0]}.{ext}")


def divider_pdf(title):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=22, weight="bold")
    p = _tmp("pdf"); fig.savefig(p); plt.close(fig)
    return p


def png_pdf(png, caption):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.imshow(mpimg.imread(png)); ax.axis("off")
    fig.text(0.5, 0.965, caption, ha="center", va="center", fontsize=9)
    p = _tmp("pdf"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def g(*parts):
    return sorted(glob.glob(os.path.join(PLOTS_DIR, *parts)))


def _good_traces():
    """The GOOD_*.pdf raw traces, restricted to the runs of the active group
    (BARRIER_WIDTH + FLY_STIMULUS, via bo.barrier_experiments) so a per-stimulus
    review only shows that group's traces - e.g. white runs stay out of review_green."""
    import bounces as bo
    keys = {e.replace("/", "_") for e in bo.barrier_experiments("good")}
    out = []
    for p in g("barriers", "angles", "GOOD_*.pdf"):
        key = os.path.basename(p)[len("GOOD_"):-len("_barriers.pdf")]
        if key in keys:
            out.append(p)
    return out


_ROSES = (g("exploratory", "heading_meanvector_rose*.png")   # 200 + 300 + window grids
          + g("exploratory", "meanvector_roses*.png")        # 200 + 300
          + g("exploratory", "heading_rose*.png"))           # rose, rose_pair, rose_all
_EXPLORATORY = [p for p in g("exploratory", "*.png") if p not in _ROSES]


SECTIONS = [
    ("Good barrier traces (vector)\n(wall + heading angle, red = shock)", _good_traces()),
    ("Wall ON -> OFF trials\n(wall_trials.py)", g("wall_trials", "*.png")),
    ("Wall OFF -> ON trials\n(interwall_analysis.py)", g("interwall", "*.png")),
    ("Wall response: speed, reflect/around, heading change, tortuosity\n(wall_response.py)", g("wall_response", "*.png")),
    ("Bounce analysis, good runs\n(bounces / 30 s)", g("bounces", "30s", "*.png")),
    ("Bounce analysis - other windows\n(10 s / 60 s / 120 s)",
     g("bounces", "10s", "*.png") + g("bounces", "60s", "*.png") + g("bounces", "120s", "*.png")),
    ("Bounce types, unbiased k-means (vector)\n(all bounces, no preselection)", g("bounce_gallery", "30pre_60post", "cluster_overview.pdf")),
    ("Exploratory findings\n(shock/post-escape, wall-through, recovery, persistence)", _EXPLORATORY),
    ("Heading mean-vector roses\n(200-sample window; 100-1000 ms standing/moving grids; per-run drift pair)", _ROSES),
    ("Inter-trial heading\n(mean vector all runs + per-run roses)",
     g("heading", "intertrial_meanvectors.png") + g("heading", "intertrial", "*.png")),
]


def main():
    writer = PdfWriter()

    def add(pdf_path):
        for page in PdfReader(pdf_path).pages:
            writer.add_page(page)

    add(divider_pdf("Fly barrier project - all analysis plots\n(review)"))
    n = 0
    for title, paths in SECTIONS:
        if not paths:
            continue
        add(divider_pdf(title))
        for p in paths:
            if p.lower().endswith(".pdf"):          # embed the real vector page
                add(p)
            else:
                add(png_pdf(p, os.path.relpath(p, PLOTS_DIR)))
            n += 1
    with open(OUT, "wb") as fh:
        writer.write(fh)
    print(f"wrote {n} plots ({len(writer.pages)} pages) -> {OUT}")


if __name__ == "__main__":
    main()
