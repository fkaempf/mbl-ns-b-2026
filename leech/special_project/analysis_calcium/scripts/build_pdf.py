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

# ---- shared layout constants (figure coordinates, 0..1) ----
LM = 0.085          # left text margin
RM = 0.915          # right text margin
TXT_W = RM - LM     # usable text width
TOP = 0.93          # first baseline on body pages
BOT = 0.075         # bottom margin: nothing drawn below this

# typographic scale
FS_PAGE_TITLE = 15.0    # body-page H1 (e.g. text pages)
FS_DIVIDER = 21.0       # section / divider titles
FS_FIG_TITLE = 13.5     # figure name
FS_SUBTITLE = 8.5       # small section subtitle under a figure title
FS_HEAD = 12.0          # in-flow heading inside narrative/text pages
FS_BODY = 9.5           # caption + paragraph body
LINE = 0.0150           # vertical advance per wrapped text line
GAP_PARA = 0.009        # extra gap after a paragraph/bullet
GAP_HEAD = 0.006        # extra gap before/after a heading

GREY = "#666666"
# one restrained dark accent used consistently for divider/section titles,
# rules, and level-0 TOC entries so the whole document reads as one piece.
ACCENT = "#1f3a4d"      # dark slate-teal
RULE = "#9aa7ad"        # muted accent-tinted rule

FOOTER_LABEL = "helobdella_LeechNo2_elavgcamp6m-17"

# ---- page tracking / footer state (reset per render pass) ----
# PAGE_NO: 1-based number of the page currently being built.
# TOTAL_PAGES: filled in after pass 1 so the footer can show "page N of M".
# OUTLINE: list of (level, title, page_index0) collected during a pass for
#          building both the visible TOC and the pypdf bookmark outline.
PAGE_NO = 0
TOTAL_PAGES = None
OUTLINE = []


def _reset_pass():
    global PAGE_NO, OUTLINE
    PAGE_NO = 0
    OUTLINE = []


def record_outline(level, title):
    """Record a TOC/bookmark entry pointing at the *next* page to be saved."""
    OUTLINE.append((level, title, PAGE_NO))   # PAGE_NO = pages saved so far == 0-based index of next page


def save_page(pdf, fig, dpi=None, footer=True):
    """Stamp a discreet footer, save the figure, advance the page counter."""
    global PAGE_NO
    if footer:
        if TOTAL_PAGES is not None:
            txt = f"{FOOTER_LABEL}    page {PAGE_NO + 1} of {TOTAL_PAGES}"
        else:
            txt = f"{FOOTER_LABEL}    page {PAGE_NO + 1}"
        # thin accent-tinted footer rule, consistent across every page
        fig.add_artist(plt.Line2D([LM, RM], [0.045, 0.045], color=RULE, lw=0.6,
                                  transform=fig.transFigure))
        fig.text(0.5, 0.030, txt, ha="center", va="center", fontsize=7.0,
                 color=GREY)
    if dpi is not None:
        pdf.savefig(fig, dpi=dpi)
    else:
        pdf.savefig(fig)
    plt.close(fig)
    PAGE_NO += 1


lines = GUIDE.read_text().splitlines()


def strip_md(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)       # bold
    s = re.sub(r"\*(.+?)\*", r"\1", s)           # single-asterisk italic
    s = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"\1", s)  # _italic_
    s = re.sub(r"`(.+?)`", r"\1", s)             # code
    s = re.sub(r"(?<!!)\[(.+?)\]\((.+?)\)", r"\1", s)  # links -> text (skip ![]() images)
    return s


# ---- document identity (used on the title page) ----
# Hardcoded so the title page is always correct regardless of how the source
# headings are ordered. The first H1 of the source is informational only.
DOC_TITLE = "helobdella_LeechNo2_elavgcamp6m-17"
DOC_SUBTITLE = ("Pan-neuronal GCaMP6m, 10x juvenile leech ganglion, "
                "after dopamine")

# ---- parse into an ordered list of blocks ----
blocks = []            # (kind, payload)
title = DOC_TITLE
first_h1_seen = False   # only the very first H1 is the document title
i = 0
cur_section = None
intro_text = []
seen_first_section = False

while i < len(lines):
    ln = lines[i]
    if ln.startswith("# ") and not ln.startswith("## "):
        h = strip_md(ln[2:].strip())
        if not first_h1_seen:
            # the leading document H1: consume as title, do not emit a section
            first_h1_seen = True
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
            s = lines[i].strip()
            m = re.match(r"!\[.*?\]\((.+?)\)", s)
            if s in ("---", "***", "___"):
                i += 1
                continue
            if m:
                img = m.group(1)
            elif s.startswith("- "):
                cap.append(strip_md(s[2:]))
            elif s:
                # continuation of the current bullet/paragraph: join, do not split
                frag = strip_md(s)
                if cap:
                    cap[-1] = (cap[-1].rstrip() + " " + frag).strip()
                else:
                    cap.append(frag)
            i += 1
        blocks.append(("figure", (name, img, cap, cur_section)))
        continue
    if ln.startswith("## "):
        h = strip_md(ln[3:].strip())
        i += 1
        body = []          # list of ("p"|"b", text); paragraphs join across soft wraps
        while i < len(lines) and not lines[i].startswith("#"):
            s = lines[i].strip()
            if s in ("---", "***", "___"):
                i += 1
                continue
            if not s:
                if body and body[-1][1]:
                    body.append(("p", ""))   # blank -> paragraph break marker
            elif s.startswith("- "):
                body.append(("b", strip_md(s[2:])))
            else:
                frag = strip_md(s)
                if body and body[-1][0] in ("p", "b") and body[-1][1]:
                    body[-1] = (body[-1][0], (body[-1][1] + " " + frag).strip())
                else:
                    body.append(("p", frag))
            i += 1
        body = [b for b in body if b[1]]     # drop empty break markers
        if not seen_first_section:
            intro_text.append(("h", h))
            intro_text += body
        else:
            blocks.append(("text", (h, body)))
        continue
    # loose prose before first section -> intro
    s = ln.strip()
    if s in ("---", "***", "___"):
        i += 1
        continue
    if s and not seen_first_section:
        frag = strip_md(s)
        if intro_text and intro_text[-1][0] == "p":
            intro_text[-1] = ("p", (intro_text[-1][1] + " " + frag).strip())
        else:
            intro_text.append(("p", frag))
    i += 1


def wrap(s, w=95):
    return "\n".join(textwrap.fill(line, w) for line in s.split("\n"))


# average glyph width as a fraction of font size, in figure-x units (A4 width).
# Used to pick a wrap column that respects the right margin for a given font size.
def _wrap_cols(fontsize):
    # empirically ~0.50 * fontsize points per char for DejaVu Sans
    char_w_fig = (0.50 * fontsize) / 72.0 / A4[0]
    return max(20, int(TXT_W / char_w_fig))


def _layout_block(kind, txt):
    """Return (wrapped_text, fontsize, weight, x, line_count, gap_after)
    for a heading ('h'), paragraph ('p') or bullet ('b')."""
    if kind == "h":
        fs, weight, x, pre, hang = FS_HEAD, "bold", LM, "", ""
    elif kind == "b":
        fs, weight, x, pre, hang = FS_BODY, "normal", LM + 0.015, "- ", "  "
    else:
        fs, weight, x, pre, hang = FS_BODY, "normal", LM, "", ""
    cols = _wrap_cols(fs)
    if kind == "b":
        cols -= 2
    wrapped = textwrap.fill(pre + txt, cols, subsequent_indent=hang)
    n = wrapped.count("\n") + 1
    gap = GAP_HEAD if kind == "h" else GAP_PARA
    return wrapped, fs, weight, x, n, gap


def flow_items(pdf, heading, items, subtitle=None, cont_label=" (cont.)"):
    """Render heading + a list of ('h'|'p'|'b', text) items across A4 pages."""
    fig = y = None

    def newpage(first):
        nonlocal fig, y
        fig = plt.figure(figsize=A4)
        if first:
            fig.text(LM, TOP, heading, fontsize=FS_PAGE_TITLE, weight="bold",
                     color=ACCENT)
            # thin accent rule under the page title for a consistent header look
            ry = TOP - 0.014
            fig.add_artist(plt.Line2D([LM, RM], [ry, ry], color=RULE, lw=0.8,
                                      transform=fig.transFigure))
            yy = TOP - 0.030
            if subtitle:
                fig.text(LM, yy, subtitle, fontsize=FS_SUBTITLE, color=GREY,
                         style="italic")
                yy -= 0.022
            y = yy - 0.012
        else:
            fig.text(LM, TOP, heading + cont_label, fontsize=FS_HEAD,
                     weight="bold", color=GREY)
            y = TOP - 0.030

    newpage(True)
    for kind, txt in items:
        wrapped, fs, weight, x, n, gap = _layout_block(kind, txt)
        need = LINE * n + gap
        # extra breathing room before a heading that is not at the page top
        if kind == "h" and y < TOP - 0.05:
            y -= GAP_HEAD
        if y - need < BOT:
            save_page(pdf, fig); newpage(False)
        fig.text(x, y, wrapped, fontsize=fs, va="top", weight=weight)
        y -= need
    save_page(pdf, fig)


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
        ("b", "Photobleaching was ~0.5% over 240 s, negligible; dF/F absorbs it. No "
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
    flow_items(pdf, heading, items)


def divider(pdf, title, subtitle=None):
    """A clean section / part divider with centered vertical rhythm."""
    fig = plt.figure(figsize=A4)
    wrapped = "\n".join(textwrap.fill(title, 34).split("\n"))
    n = wrapped.count("\n") + 1
    cy = 0.55
    fig.text(0.5, cy, wrapped, ha="center", va="center", fontsize=FS_DIVIDER,
             weight="bold", color=ACCENT)
    rule_y = cy - 0.018 * n - 0.022
    fig.add_artist(plt.Line2D([0.30, 0.70], [rule_y, rule_y], color=RULE, lw=1.0,
                              transform=fig.transFigure))
    if subtitle:
        fig.text(0.5, rule_y - 0.030, subtitle, ha="center", fontsize=10,
                 style="italic", color=GREY)
    save_page(pdf, fig)


def render_caption(fig, cap, ytop, ybot):
    """Render caption bullets with hanging indent between ytop and ybot.
    Returns the list of bullets that did NOT fit (for pagination)."""
    y = ytop
    leftover = []
    for idx, c in enumerate(cap):
        wrapped, fs, weight, x, n, gap = _layout_block("b", c)
        need = LINE * n + gap
        if y - need < ybot:
            leftover = cap[idx:]
            break
        fig.text(x, y, wrapped, fontsize=fs, va="top", weight=weight)
        y -= need
    return leftover


def caption_height(cap):
    """Total figure-y height needed to render all caption bullets."""
    h = 0.0
    for c in cap:
        _, _, _, _, n, gap = _layout_block("b", c)
        h += LINE * n + gap
    return h


# A4 page is taller than wide; in figure (0..1) coords the page aspect (w/h)
# of the *drawable area* matters when converting image aspect to a box.
PAGE_AR = A4[0] / A4[1]          # ~0.707 (width / height of the sheet)


def _draw_title(fig, name, section, y_title=0.965, y_sub=0.945):
    fig.text(0.5, y_title, name, ha="center", fontsize=FS_FIG_TITLE,
             weight="bold", color=ACCENT)
    if section:
        fig.text(0.5, y_sub, section, ha="center", fontsize=FS_SUBTITLE, color=GREY)


def _place_image(fig, im, box_x, box_y, box_w, box_h, valign="center"):
    """Fit image into the (figure-coord) box preserving aspect, return (x,y,w,h)
    of the placed axes. valign: 'center' | 'top' | 'bottom'."""
    h, w = im.shape[:2]
    ar = w / h                                   # image pixel aspect
    # box aspect in *physical* units: figure-coord box scaled by page aspect
    box_ar = (box_w * A4[0]) / (box_h * A4[1])
    if ar > box_ar:                              # width-limited
        iw = box_w
        ih = (box_w * A4[0]) / ar / A4[1]
    else:                                        # height-limited
        ih = box_h
        iw = (box_h * A4[1]) * ar / A4[0]
    x = box_x + (box_w - iw) / 2
    if valign == "top":
        y = box_y + (box_h - ih)
    elif valign == "bottom":
        y = box_y
    else:
        y = box_y + (box_h - ih) / 2
    ax = fig.add_axes([x, y, iw, ih])
    ax.imshow(im, interpolation="antialiased")
    ax.axis("off")
    return x, y, iw, ih


# caption band caps and slack
CAP_BAND_MAX = 0.34
SLACK = 0.012
# dpi for figure pages that embed a raster image (source PNGs are 170 dpi).
# matplotlib embeds the placed image resampled to this dpi over its on-page
# size; 150 keeps wide panels crisp while holding the PDF well under 15 MB.
IMG_DPI = 130


def emit_figure(pdf, name, img, cap, section):
    ip = BASE / img if img else None
    has_img = bool(ip and ip.exists())

    if not has_img:
        # text-only "figure": flow caption as bullets across pages
        flow_items(pdf, name, [("b", c) for c in cap], subtitle=section)
        return

    im = mpimg.imread(ip)
    h, w = im.shape[:2]
    ar = w / h

    cap_h = caption_height(cap)
    fig = plt.figure(figsize=A4)
    _draw_title(fig, name, section)

    # available vertical span between the title block and the bottom margin
    AVAIL_TOP = 0.935
    leftover = []

    if ar >= 2.2:
        # ULTRA-WIDE: scale to full text width, sit it just above the caption
        # with both vertically centered in the available span -> no dead band.
        box_x, box_w = LM, TXT_W
        ih = (box_w * A4[0]) / ar / A4[1]        # height the image will occupy
        cap_band = min(cap_h + SLACK, CAP_BAND_MAX)
        gap = 0.030
        group_h = ih + gap + cap_band
        span = AVAIL_TOP - BOT
        # Position the (image + gap + caption) group with an upward bias: keep a
        # comfortable margin under the title and let any slack fall to the bottom
        # rather than as a dead band between the title and the image.
        top_margin = 0.045                       # gap below the title block
        free = span - group_h                    # leftover vertical space
        top = AVAIL_TOP - top_margin - 0.30 * max(0.0, free)
        img_y = top - ih
        _place_image(fig, im, box_x, img_y, box_w, ih, valign="bottom")
        cap_top = img_y - gap
        leftover = render_caption(fig, cap, cap_top, BOT)
    elif ar <= 0.95:
        # VERY TALL: use the full height; caption gets whatever band is left,
        # paginating overflow. Image hugs the left, caption can share width.
        cap_band = min(cap_h + SLACK, 0.22)
        cap_top = BOT + cap_band
        box_y = cap_top + 0.020
        box_h = AVAIL_TOP - box_y
        _place_image(fig, im, 0.05, box_y, 0.90, box_h, valign="top")
        leftover = render_caption(fig, cap, cap_top, BOT)
    else:
        # NORMAL: caption bottom-anchored, image fills the space above it.
        cap_band = min(cap_h + SLACK, CAP_BAND_MAX)
        cap_top = BOT + cap_band
        box_y = cap_top + 0.024
        box_h = AVAIL_TOP - box_y
        _place_image(fig, im, 0.05, box_y, 0.90, box_h, valign="center")
        leftover = render_caption(fig, cap, cap_top, BOT)

    # render embedded images at a higher dpi than the 100-dpi default so the
    # 170-dpi source PNGs stay crisp; ~200 dpi keeps file size reasonable.
    save_page(pdf, fig, dpi=IMG_DPI)

    # paginate any caption overflow onto continuation pages
    if leftover:
        flow_items(pdf, name, [("b", c) for c in leftover],
                   subtitle=section, cont_label=" (caption cont.)")


# ---- table of contents -------------------------------------------------------
# A TOC entry: (level, title, page_number_or_None). level 0 = part/section
# divider, 1 = sub-section, 2 = figure. Page numbers are 1-based and filled in
# on the second pass once the layout is known.
TOC_TITLE = "Contents"
TOC_FS = {0: 11.0, 1: 9.5, 2: 9.0}
TOC_INDENT = {0: LM, 1: LM + 0.030, 2: LM + 0.060}
TOC_WEIGHT = {0: "bold", 1: "normal", 2: "normal"}
TOC_LINE = 0.0165          # vertical advance per TOC row
TOC_GAP0 = 0.008           # extra gap before a level-0 entry


def toc_capacity_pages(entries):
    """How many pages the TOC will need (layout is independent of page numbers)."""
    pages = 1
    y = TOP - 0.055
    for level, title, _ in entries:
        if level == 0 and y < TOP - 0.06:
            y -= TOC_GAP0
        if y < BOT + TOC_LINE:
            pages += 1
            y = TOP - 0.055
        y -= TOC_LINE
    return pages


def render_toc(pdf, entries):
    """Render the visible TOC across however many pages it needs."""
    def newpage(first):
        f = plt.figure(figsize=A4)
        if first:
            f.text(LM, TOP, TOC_TITLE, fontsize=FS_PAGE_TITLE, weight="bold",
                   color=ACCENT)
            ry = TOP - 0.014
            f.add_artist(plt.Line2D([LM, RM], [ry, ry], color=RULE, lw=0.8,
                                    transform=f.transFigure))
        else:
            f.text(LM, TOP, TOC_TITLE + " (cont.)", fontsize=FS_HEAD,
                   weight="bold", color=GREY)
        return f, TOP - 0.055

    fig, y = newpage(True)
    for level, title, pageno in entries:
        if level == 0 and y < TOP - 0.06:
            y -= TOC_GAP0
        if y < BOT + TOC_LINE:
            save_page(pdf, fig)
            fig, y = newpage(False)
        fs = TOC_FS[level]
        x = TOC_INDENT[level]
        weight = TOC_WEIGHT[level]
        color = ACCENT if level == 0 else "#333333"
        # title (clipped if absurdly long) on the left, page number flush right
        t = title if len(title) <= 78 else title[:77].rstrip() + "…"
        fig.text(x, y, t, fontsize=fs, va="center", weight=weight, color=color)
        if pageno is not None:
            fig.text(RM, y, str(pageno), fontsize=fs, va="center", ha="right",
                     weight=weight, color=color)
        y -= TOC_LINE
    save_page(pdf, fig)


def build(pdf, toc_entries, toc_pages):
    """Render the whole document. `toc_entries` is what to print on the TOC
    page(s) (empty placeholder rows on pass 1 so the page count is stable);
    `toc_pages` is how many pages the TOC occupies. Records OUTLINE as it goes."""
    _reset_pass()

    # ---- cover page (no footer): clean centered title block ----
    fig = plt.figure(figsize=A4)
    # title block
    fig.text(0.5, 0.760, "Calcium imaging figure guide", ha="center",
             fontsize=23, weight="bold", color=ACCENT)
    fig.add_artist(plt.Line2D([0.34, 0.66], [0.730, 0.730], color=RULE, lw=1.2,
                              transform=fig.transFigure))
    fig.text(0.5, 0.700, DOC_TITLE, ha="center",
             fontsize=13.5, weight="bold", color="#222")
    fig.text(0.5, 0.673, DOC_SUBTITLE, ha="center",
             fontsize=10.5, style="italic", color="#444")
    # short abstract of the headline result (2-3 lines, centered)
    abstract = ("This guide walks through every figure of the analysis. The headline "
                "result: the ganglion shows a single strong, clock-like, ganglion-wide "
                "calcium oscillation at 0.37 Hz (period 2.7 s) that is sustained for the "
                "full ~240 s recording and is spatially synchronous, with no traveling "
                "wave.")
    aw = textwrap.fill(abstract, 72)
    fig.text(0.5, 0.560, aw, ha="center", va="top", fontsize=10.5,
             color="#333", linespacing=1.5)
    # recording id + date metadata block
    fig.add_artist(plt.Line2D([0.40, 0.60], [0.300, 0.300], color=RULE, lw=0.8,
                              transform=fig.transFigure))
    fig.text(0.5, 0.270, "Recording: " + DOC_TITLE, ha="center",
             fontsize=9.5, color="#444")
    fig.text(0.5, 0.248, "Prepared 2026-06-05", ha="center",
             fontsize=9.5, color="#444")
    fig.text(0.5, 0.110, "Contents follow", ha="center",
             fontsize=9.0, style="italic", color=GREY)
    save_page(pdf, fig, footer=False)

    # ---- table of contents ----
    if toc_pages > 0:
        render_toc(pdf, toc_entries)

    # ---- about this guide (intro prose moved off the cover) ----
    if intro_text:
        record_outline(0, "About this guide")
        flow_items(pdf, "About this guide", intro_text)

    # ---- narrative: what we did + how to interpret ----
    record_outline(0, "What we did and how to interpret")
    divider(pdf, "What we did and how to interpret")
    for h, items in NARRATIVE:
        record_outline(1, h)
        flow_items(pdf, h, items)

    # ---- figure walkthrough divider ----
    record_outline(0, "Figure walkthrough")
    divider(pdf, "Figure walkthrough",
            subtitle="every plot, with how it was made and how to read it")

    # ---- content blocks ----
    for kind, payload in blocks:
        if kind == "section":
            record_outline(1, payload)
            divider(pdf, payload)
        elif kind == "text":
            h, body = payload
            record_outline(1, h)
            flow_items(pdf, h, body)
        elif kind == "figure":
            name, img, cap, section = payload
            record_outline(2, name)
            emit_figure(pdf, name, img, cap, section)

    d = pdf.infodict()
    d["Title"] = "Calcium imaging figure guide: helobdella_LeechNo2_elavgcamp6m-17"
    d["Author"] = "analysis_calcium pipeline"


# ---- build the TOC entry list up front (titles known without rendering) -------
# Mirror the order in build(): narrative section, its subsections, figure
# walkthrough, then each section divider / text block / figure.
def make_toc_skeleton():
    entries = []
    if intro_text:
        entries.append((0, "About this guide"))
    entries.append((0, "What we did and how to interpret"))
    for h, _ in NARRATIVE:
        entries.append((1, h))
    entries.append((0, "Figure walkthrough"))
    for kind, payload in blocks:
        if kind == "section":
            entries.append((1, payload))
        elif kind == "text":
            entries.append((1, payload[0]))
        elif kind == "figure":
            entries.append((2, payload[0]))
    return entries


toc_skeleton = make_toc_skeleton()
TOC_PAGES = toc_capacity_pages([(l, t, None) for l, t in toc_skeleton])

# ---- PASS 1: render with placeholder TOC to learn each entry's page number ----
import io
_buf = io.BytesIO()
with PdfPages(_buf) as _pdf:
    placeholder = [(l, t, None) for l, t in toc_skeleton]
    build(_pdf, placeholder, TOC_PAGES)
pass1_outline = list(OUTLINE)
pass1_total = PAGE_NO

# Map each TOC entry (by its position in the recorded outline) to a page number.
# OUTLINE order matches make_toc_skeleton order exactly.
toc_entries = []
for (level, title, idx0), (slevel, stitle) in zip(pass1_outline, toc_skeleton):
    toc_entries.append((level, title, idx0 + 1))   # 1-based page number

# ---- PASS 2: real render with correct TOC page numbers + "N of M" footers ----
TOTAL_PAGES = pass1_total
with PdfPages(PDF) as pdf:
    build(pdf, toc_entries, TOC_PAGES)
final_outline = list(OUTLINE)
final_total = PAGE_NO

# ---- POST-PROCESS: clickable PDF bookmarks via pypdf ----
bookmarks_added = False
try:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(PDF))
    writer = PdfWriter()
    writer.append(reader)
    parents = {}            # level -> last bookmark item at that level
    for level, title, idx0 in final_outline:
        parent = parents.get(level - 1) if level > 0 else None
        item = writer.add_outline_item(title, idx0, parent=parent)
        parents[level] = item
        # invalidate deeper levels so a new section starts a fresh subtree
        for d in list(parents):
            if d > level:
                parents.pop(d)
    if reader.metadata:
        writer.add_metadata({k: v for k, v in reader.metadata.items()
                             if isinstance(v, str)})
    with open(PDF, "wb") as fh:
        writer.write(fh)
    bookmarks_added = True
except Exception as e:                       # pragma: no cover
    print(f"bookmark post-process skipped: {e}")

# ---- POST-PROCESS: shrink with ghostscript if available and it helps ----
# gs preserves the page count and the pypdf outline; downsampling embedded
# rasters to the on-page dpi keeps the file comfortably under the ~15 MB cap.
import shutil
import subprocess
compressed = False
gs = shutil.which("gs")
if gs:
    tmp = PDF.with_suffix(".gs.pdf")
    try:
        subprocess.run([
            gs, "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/printer",
            "-dDownsampleColorImages=true", f"-dColorImageResolution={IMG_DPI}",
            "-dDownsampleGrayImages=true", f"-dGrayImageResolution={IMG_DPI}",
            "-dNOPAUSE", "-dBATCH", "-dQUIET",
            f"-sOutputFile={tmp}", str(PDF),
        ], check=True)
        # sanity: gs output must keep all pages before we trust it
        from pypdf import PdfReader as _R
        if (tmp.stat().st_size < PDF.stat().st_size
                and len(_R(str(tmp)).pages) == final_total):
            tmp.replace(PDF)
            compressed = True
        else:
            tmp.unlink(missing_ok=True)
    except Exception as e:                   # pragma: no cover
        print(f"gs compression skipped: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)

n_fig = sum(1 for k, _ in blocks if k == "figure")
size_mb = PDF.stat().st_size / 1e6
print(f"wrote {PDF.name}: {n_fig} figure pages, {final_total} total pages, "
      f"TOC={TOC_PAGES}p, bookmarks={bookmarks_added}, "
      f"compressed={compressed}, {size_mb:.1f} MB")
