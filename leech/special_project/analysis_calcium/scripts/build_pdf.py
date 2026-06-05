#!/usr/bin/env python
"""Combine all figures + captions from FIGURE_GUIDE.md into one PDF.
One figure per page with its caption; title page, section dividers, text pages."""
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_calcium")
GUIDE = BASE / "FIGURE_GUIDE.md"
PDF = BASE / "CALCIUM_FIGURE_GUIDE.pdf"
A4 = (8.27, 11.69)

lines = GUIDE.read_text().splitlines()


def strip_md(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)       # bold
    s = re.sub(r"`(.+?)`", r"\1", s)             # code
    return s


# ---- parse into an ordered list of blocks ----
blocks = []            # (kind, payload)
title = "Figure guide"
i = 0
cur_section = None
intro_text = []
seen_first_section = False

while i < len(lines):
    ln = lines[i]
    if ln.startswith("# ") and not ln.startswith("## "):
        h = strip_md(ln[2:].strip())
        if not blocks and not seen_first_section:
            title = h
        else:
            seen_first_section = True
            blocks.append(("section", h))
            cur_section = h
        i += 1
        continue
    if ln.startswith("### "):
        name = strip_md(ln[4:].strip())
        i += 1
        img = None
        cap = []
        while i < len(lines) and not lines[i].startswith("#"):
            m = re.match(r"!\[.*?\]\((.+?)\)", lines[i].strip())
            if m:
                img = m.group(1)
            elif lines[i].strip().startswith("- "):
                cap.append(strip_md(lines[i].strip()[2:]))
            elif lines[i].strip():
                cap.append(strip_md(lines[i].strip()))
            i += 1
        blocks.append(("figure", (name, img, cap, cur_section)))
        continue
    if ln.startswith("## "):
        h = strip_md(ln[3:].strip())
        i += 1
        body = []
        while i < len(lines) and not lines[i].startswith("#"):
            if lines[i].strip():
                body.append(strip_md(lines[i].strip()))
            i += 1
        target = intro_text if not seen_first_section else None
        if target is not None:
            intro_text.append(("h", h))
            intro_text += [("p", b) for b in body]
        else:
            blocks.append(("text", (h, body)))
        continue
    # loose prose before first section -> intro
    if ln.strip() and not seen_first_section:
        intro_text.append(("p", strip_md(ln.strip())))
    i += 1


def wrap(s, w=95):
    return "\n".join(textwrap.fill(line, w) for line in s.split("\n"))


# ---- narrative: what we did + how to interpret (shown before the figure walkthrough) ----
# items: ("h", heading) | ("p", paragraph) | ("b", bullet)
NARRATIVE = [
    ("What we did: the analysis pipeline", [
        ("p", "The recording was analysed in five stages, each a script in scripts/, "
              "followed by an intricate visualization pass. All outputs are derived from "
              "the single raw movie; no manual edits to traces were made."),
        ("b", "Stage 1 (QC and motion): measured photobleaching, rigid drift (phase "
              "cross-correlation of every frame to a reference), non-rigid block-wise "
              "jitter, and built the local correlation image that reveals cell bodies."),
        ("b", "Stage 2 (suite2p): ran suite2p with registration on to detect ROIs and "
              "extract fluorescence. It accepted only 24 ROIs, far fewer than the visible "
              "somata, so it was kept only as a cross-check."),
        ("b", "Stage 2b (correlation-blob segmentation): the primary method. Detected "
              "somata on the correlation image times the std projection (peak detection "
              "plus watershed to split touching cells), giving 180 active patches, and "
              "extracted dF/F with local-annulus neuropil subtraction."),
        ("b", "Stage 3 (temporal): characterised the rhythm of the whole-ganglion mean "
              "signal with Welch spectra, a spectrogram, autocorrelation, peak detection "
              "and a Hilbert transform."),
        ("b", "Stage 4 (spatial): mapped per-pixel oscillation power and phase, tested "
              "for a traveling wave, and measured region-by-region coherence."),
        ("b", "Intricate pass: 25 detailed multi-panel figures (fig_*.py) plus a "
              "cycle-averaged movie, covering population, spectral, spatial, "
              "cycle-resolved and dashboard views."),
        ("p", "Units note: this is 10x, a preliminary line, so all signals are read at "
              "the ganglion / regional level, not as single neurons. The 180 patches are "
              "active regions, not confirmed cells."),
    ]),
    ("Motion correction: not needed (verified)", [
        ("p", "Whether to motion-correct was decided from measurements, not assumed."),
        ("b", "Rigid drift was 0.00 px across all 692 frames (cross-checked at full "
              "resolution). The preparation is stationary; the movement flag no is correct."),
        ("b", "Non-rigid jitter was 0.00 px in a 3x3 block check. No local warping."),
        ("b", "Photobleaching was 0.5 to 0.8% over 240 s, negligible; dF/F absorbs it. No "
              "saturation, no dropped frames."),
        ("p", "Conclusion: analysis runs directly on the raw movie with no interpolation "
              "artifacts. The moderate frame-to-frame correlation (~0.55) is real calcium "
              "dynamics, not motion, and must not be mistaken for a motion problem. The "
              "dataset recordings flagged leech moves or david moves are the ones that "
              "would need correction, ideally using their palmmcherry static channel."),
    ]),
    ("Headline result and how to interpret it", [
        ("p", "The ganglion shows one strong, clock-like, ganglion-wide calcium "
              "oscillation at 0.37 Hz (period 2.7 s) that is sustained undamped for the "
              "full ~240 s and is spatially synchronous across the whole ganglion."),
        ("b", "Frequency: dominant peak 0.372 Hz with a harmonic at ~0.73 Hz (a real "
              "periodic, non-sinusoidal waveform), ~230x above the spectral floor, well "
              "below the 1.44 Hz Nyquist so the fundamental is not aliased."),
        ("b", "Regularity: 88 cycles, inter-peak interval CV 0.077, a tight Poincare "
              "cluster, and a stable spectrogram band. A very clock-like rhythm."),
        ("b", "Synchrony: vertical kymograph stripes, whole-ganglion brightening and "
              "dimming together in the phase montage, and a flat phase-vs-position slope. "
              "Inter-region coherence at 0.37 Hz is ~0.88. No traveling wave."),
        ("b", "Nuance: the rhythm is carried by a strong subset. About 93% of patches "
              "peak within 0.05 Hz of 0.37 Hz and ~40% follow the ganglion mean at r>0.5; "
              "the rest are noisy (low-photon 10x). Pooling the rhythmic subset is what "
              "yields the clean ganglion signal, so all-patch pairwise correlation looks "
              "low while region-level coherence is high."),
        ("p", "How to interpret: a single synchronous ganglion-wide oscillation points to "
              "a shared network drive rather than independent local pacemakers. The file "
              "is labelled after dopamine pharyngals, but there is no pre-dopamine baseline "
              "here, so no causal claim about dopamine can be made. The 0.37 Hz rhythm is "
              "a candidate motor or visceral rhythm; identifying it needs a labelled "
              "baseline and a behavioural or electrophysiological readout."),
    ]),
]


def emit_text_pages(pdf, heading, items, A4=A4):
    """Render heading + items across as many A4 pages as needed."""
    y = None
    fig = None

    def newpage(first):
        nonlocal fig, y
        fig = plt.figure(figsize=A4)
        if first:
            fig.text(0.08, 0.93, heading, fontsize=15, weight="bold")
            y = 0.87
        else:
            fig.text(0.08, 0.93, heading + " (cont.)", fontsize=12, weight="bold", color="#555")
            y = 0.88

    newpage(True)
    for kind, txt in items:
        if kind == "h":
            indent, size, pre = 0.08, 12, ""
        elif kind == "b":
            indent, size, pre = 0.10, 9.5, "- "
        else:
            indent, size, pre = 0.08, 9.5, ""
        width = 96 if kind != "b" else 92
        wrapped = textwrap.fill(pre + txt, width,
                                subsequent_indent="  " if kind == "b" else "")
        n = wrapped.count("\n") + 1
        need = 0.018 * n + 0.010
        if y - need < 0.07:
            pdf.savefig(fig); plt.close(fig); newpage(False)
        weight = "bold" if kind == "h" else "normal"
        fig.text(indent, y, wrapped, fontsize=size, va="top", weight=weight)
        y -= need
    pdf.savefig(fig); plt.close(fig)


with PdfPages(PDF) as pdf:
    # ---- title page ----
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.88, "Calcium imaging figure guide", ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.84, title.replace("Figure guide: ", ""), ha="center", fontsize=11,
             style="italic", color="#444")
    y = 0.76
    for kind, val in intro_text:
        if kind == "h":
            y -= 0.018
            fig.text(0.08, y, val, fontsize=12, weight="bold")
            y -= 0.028
        else:
            wrapped = textwrap.fill(val, 92)
            n = wrapped.count("\n") + 1
            fig.text(0.08, y, wrapped, fontsize=9, va="top", family="DejaVu Sans")
            y -= 0.018 * n + 0.006
        if y < 0.08:
            break
    pdf.savefig(fig); plt.close(fig)

    # ---- narrative: what we did + how to interpret ----
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.55, "What we did and how to interpret", ha="center", va="center",
             fontsize=20, weight="bold")
    fig.add_artist(plt.Line2D([0.2, 0.8], [0.5, 0.5], color="#888", lw=1,
                              transform=fig.transFigure))
    pdf.savefig(fig); plt.close(fig)
    for h, items in NARRATIVE:
        emit_text_pages(pdf, h, items)

    # ---- figure walkthrough divider ----
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.55, "Figure walkthrough", ha="center", va="center",
             fontsize=20, weight="bold")
    fig.text(0.5, 0.49, "every plot, with how it was made and how to read it",
             ha="center", fontsize=10, style="italic", color="#555")
    fig.add_artist(plt.Line2D([0.2, 0.8], [0.46, 0.46], color="#888", lw=1,
                              transform=fig.transFigure))
    pdf.savefig(fig); plt.close(fig)

    # ---- content blocks ----
    for kind, payload in blocks:
        if kind == "section":
            fig = plt.figure(figsize=A4)
            fig.text(0.5, 0.55, payload, ha="center", va="center", fontsize=20,
                     weight="bold", wrap=True)
            fig.add_artist(plt.Line2D([0.2, 0.8], [0.5, 0.5], color="#888", lw=1,
                                      transform=fig.transFigure))
            pdf.savefig(fig); plt.close(fig)
        elif kind == "text":
            h, body = payload
            fig = plt.figure(figsize=A4)
            fig.text(0.08, 0.92, h, fontsize=15, weight="bold")
            y = 0.86
            for b in body:
                wrapped = textwrap.fill(b, 92)
                n = wrapped.count("\n") + 1
                fig.text(0.08, y, wrapped, fontsize=9.5, va="top")
                y -= 0.02 * n + 0.008
                if y < 0.08:
                    break
            pdf.savefig(fig); plt.close(fig)
        elif kind == "figure":
            name, img, cap, section = payload
            fig = plt.figure(figsize=A4)
            fig.text(0.5, 0.965, name, ha="center", fontsize=13, weight="bold")
            if section:
                fig.text(0.5, 0.945, section, ha="center", fontsize=8, color="#777")
            ip = BASE / img if img else None
            if ip and ip.exists():
                im = mpimg.imread(ip)
                h, w = im.shape[:2]
                ar = w / h
                # fit image into top region [0.05,0.42]-[0.95,0.93]
                box_w, box_h = 0.90, 0.50
                box_x, box_y = 0.05, 0.42
                if ar > box_w / box_h:
                    iw = box_w; ih = box_w / ar
                else:
                    ih = box_h; iw = box_h * ar
                ax = fig.add_axes([box_x + (box_w - iw) / 2,
                                   box_y + (box_h - ih), iw, ih])
                ax.imshow(im); ax.axis("off")
            # caption
            ytxt = 0.38 if (ip and ip.exists()) else 0.85
            for c in cap:
                wrapped = textwrap.fill(c, 96)
                n = wrapped.count("\n") + 1
                fig.text(0.06, ytxt, "- " + wrapped, fontsize=9.2, va="top")
                ytxt -= 0.018 * n + 0.006
                if ytxt < 0.04:
                    break
            pdf.savefig(fig); plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "Calcium imaging figure guide: helobdella_LeechNo2_elavgcamp6m-17"
    d["Author"] = "analysis_calcium pipeline"

n_fig = sum(1 for k, _ in blocks if k == "figure")
print(f"wrote {PDF.name}: {n_fig} figure pages + dividers/text + title")
