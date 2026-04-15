#!/usr/bin/env python3
"""Figure 6 — Mechanistic vignettes.

Design philosophy: AlphaGenome native plots are ~10:1 landscape. Shrinking them
into a 3-column grid makes labels unreadable. Instead, stack vertically: each
panel gets full canvas width to preserve legibility.

Structure (9 rows, each ~1.2" tall, 16" wide):
  Header: LMNA cluster
  A: LMNA 156115264 ISM
  B: LMNA 156115279 ISM
  C: LMNA 156115286 ISM
  Header: LMNA REF/ALT (reduced ALT transcription, all three variants)
  D: LMNA 156115264 REF/ALT
  E: LMNA 156115279 REF/ALT
  F: LMNA 156115286 REF/ALT
  Header: Multimodal evidence and readthrough
  G: DUSP1 TF ISM
  H: DUSP1 histone ISM
  I: VIM REF/ALT (readthrough)
"""
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread

mpl.use("Agg")
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9, "figure.dpi":300, "savefig.dpi":300,
    "savefig.bbox":"tight", "pdf.fonttype":42,
})

ACCENT = "#CC79A7"
GREY = "#333333"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    vdir = Path(args.vignettes)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Ordered sequence: (kind, content, panel_label, title_or_header_text)
    # kind = "header" | "panel"
    sections = [
        ("header", ACCENT,
         "LMNA promoter VUS cluster — three independent ClinVar VUS with convergent ISM signatures"),
        ("panel", "ism_logo_chr1_156115264_C_T_DNASE.png", "A",
         "LMNA chr1:156115264:C>T — DNase ISM"),
        ("panel", "ism_logo_chr1_156115279_G_A_DNASE.png", "B",
         "LMNA chr1:156115279:G>A — DNase ISM"),
        ("panel", "ism_logo_chr1_156115286_C_A_DNASE.png", "C",
         "LMNA chr1:156115286:C>A — DNase ISM"),
        ("header", ACCENT,
         "LMNA REF vs ALT — reduced predicted transcription on alternate allele (Heart LV)"),
        ("panel", "ref_alt_chr1_156115264_C_T.png", "D",
         "LMNA chr1:156115264 — REF (grey) vs ALT (red)"),
        ("panel", "ref_alt_chr1_156115279_G_A.png", "E",
         "LMNA chr1:156115279 — REF vs ALT"),
        ("panel", "ref_alt_chr1_156115286_C_A.png", "F",
         "LMNA chr1:156115286 — REF vs ALT"),
        ("header", GREY,
         "Multimodal evidence: DUSP1 (invisible to DNase-only) and VIM (ALT-allele readthrough)"),
        ("panel", "ism_logo_chr5_172770036_T_C_CHIP_TF.png", "G",
         "DUSP1 chr5:172770036:T>C — ChIP-TF ISM (max = 3.17; DNase max = 0.02)"),
        ("panel", "ism_logo_chr5_172770036_T_C_CHIP_HISTONE.png", "H",
         "DUSP1 chr5:172770036:T>C — ChIP-histone ISM (max = 20.01)"),
        ("panel", "ref_alt_chr10_17229111_G_C.png", "I",
         "VIM chr10:17229111:G>C — REF vs ALT (Heart LV): ALT-allele readthrough downstream of variant"),
    ]

    # Height ratios: headers thin (0.4), image panels tall (1.8)
    # With 3 headers and 9 image panels -> total ratio units: 3*0.4 + 9*1.8 = 1.2 + 16.2 = 17.4
    # Use 16" wide, compute height so each 1.8-unit panel is ~1.5" tall:
    #   panel_inches = 1.5 -> ratio_to_inches = 1.5/1.8 = 0.833
    #   total_height = 17.4 * 0.833 = 14.5 inches
    height_ratios = []
    for kind, *_ in sections:
        height_ratios.append(0.4 if kind == "header" else 1.8)
    total_ratio = sum(height_ratios)
    inches_per_ratio = 0.85
    fig_height = total_ratio * inches_per_ratio  # ~14.8 inches

    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(
        len(sections), 1,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.05, right=0.99, top=0.99, bottom=0.01,
    )

    for i, sec in enumerate(sections):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        if sec[0] == "header":
            _, color, text = sec
            ax.text(0.0, 0.5, text, fontsize=12, fontweight="bold",
                    color=color, va="center", ha="left",
                    transform=ax.transAxes)
            # Underline
            ax.axhline(0.1, xmin=0.0, xmax=1.0, color=color,
                       linewidth=0.8, alpha=0.5)
        else:
            _, fname, label, title = sec
            fp = vdir / fname
            if fp.exists():
                img = imread(fp)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"MISSING: {fname}", ha="center", va="center",
                        color="red", transform=ax.transAxes, fontsize=10)
            # Panel label in top-left corner, outside image if possible
            ax.text(-0.015, 1.0, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="right")
            # Title above image, left-aligned, in a weight that won't compete with image labels
            ax.set_title(title, fontsize=9.5, loc="left", pad=2, fontweight="bold")

    plt.savefig(out / "fig6_vignettes.png", dpi=300)
    plt.savefig(out / "fig6_vignettes.pdf")
    plt.close()
    print(f"wrote {out/'fig6_vignettes.png'}")
    print(f"wrote {out/'fig6_vignettes.pdf'}")


if __name__ == "__main__":
    main()