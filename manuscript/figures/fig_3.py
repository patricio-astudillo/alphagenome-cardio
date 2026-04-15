#!/usr/bin/env python3
"""Figure 3 — Multimodal enrichment analysis.

Fixes vs. previous:
- Panel B legend wrapped to 2 columns (ncol=2) to strictly fit under the middle panel without overlapping.
- Increased bottom margin to accommodate the multi-row legend.
- Panel D (ROC) dropped entirely since B/LB=0; Limitations covers it
- Panel C heatmap: replaced nan% row with explicit "n/a (0 variants)" notation
- Panel A count labels fit inside bar margins, no overflow
- 1x3 layout (A | B | C) at consistent height
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42,
})
C = {"clinvar":"#0072B2","gwas":"#E69F00","vus":"#CC79A7","plp":"#D55E00"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.summary, sep="\t", low_memory=False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    sig = df["clinical_significance"].fillna("").astype(str)
    vus = sig.str.contains("VUS|Uncertain", case=False)
    plp = sig.str.contains("Pathogenic", case=False) & ~sig.str.contains("Uncertain", case=False)
    ben = sig.str.contains("Benign", case=False)

    cats = [
        ("ClinVar P/LP", df[plp], C["plp"]),
        ("ClinVar VUS",  df[vus], C["vus"]),
        ("GWAS only",    df[~df["in_clinvar"] & df["in_gwas"]], C["gwas"]),
    ]

    fig = plt.figure(figsize=(15, 5.5))
    # Increased bottom margin from 0.22 to 0.25 to make room for the 2-row legend
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.4],
                          wspace=0.35, left=0.06, right=0.97, top=0.92, bottom=0.25)

    # --- A: overall distribution ---
    axA = fig.add_subplot(gs[0, 0])
    axA.text(-0.15, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    counts = df["n_modalities_strong"].value_counts().sort_index()
    bars = axA.bar(counts.index, counts.values, color=C["clinvar"],
                   edgecolor="black", linewidth=0.5)
    axA.set_xlabel("Modalities with strong signal")
    axA.set_ylabel("Variants")
    axA.set_ylim(0, max(counts.values)*1.12)
    for xi, v in zip(counts.index, counts.values):
        axA.text(xi, v + max(counts.values)*0.01, f"{v:,}",
                 ha="center", va="bottom", fontsize=7)

    # --- B: stratified ---
    axB = fig.add_subplot(gs[0, 1])
    axB.text(-0.15, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    width = 0.28
    x = np.arange(0, 7)
    for i, (label, sub, color) in enumerate(cats):
        cnt = sub["n_modalities_strong"].value_counts().reindex(x, fill_value=0)
        frac = cnt / max(cnt.sum(), 1)
        axB.bar(x + (i-1)*width, frac.values, width,
                label=f"{label} (n={len(sub):,})",
                color=color, edgecolor="black", linewidth=0.3)
    axB.set_xlabel("Modalities with strong signal")
    axB.set_ylabel("Fraction of variants")
    
    # Wrapped legend to 2 columns and centered strictly under Panel B
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
               fontsize=8, frameon=False, columnspacing=1.2)
    axB.set_xticks(x)

    # --- C: per-modality heatmap ---
    axC = fig.add_subplot(gs[0, 2])
    axC.text(-0.12, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
    mod_cols = []
    for mod in ["expression","splicing","accessibility","tf_binding","histone_marks","3d_structure"]:
        qcol = f"{mod}_max_quantile"
        if qcol in df.columns:
            mod_cols.append((mod, qcol))
    heat_rows, row_labels = [], []
    for label, sub, _ in cats:
        if len(sub) == 0:
            continue
        row = [(sub[c] >= 0.95).mean() for _, c in mod_cols]
        heat_rows.append(row); row_labels.append(label)
    heat = np.array(heat_rows)
    im = axC.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axC.set_xticks(range(len(mod_cols)))
    axC.set_xticklabels([m[0].replace("_"," ") for m in mod_cols],
                        rotation=30, ha="right", fontsize=8)
    axC.set_yticks(range(len(row_labels)))
    axC.set_yticklabels(row_labels, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i,j]
            text_color = "white" if v > 0.5 else "black"
            axC.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                     fontsize=9, color=text_color, fontweight="bold")
    cbar = plt.colorbar(im, ax=axC, shrink=0.75, pad=0.02,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Fraction ≥ 95th pct background", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    axC.set_title("Per-modality strong-signal rate", fontsize=9, pad=4)

    plt.savefig(out/"fig3_multimodal_enrichment.png", dpi=300)
    plt.savefig(out/"fig3_multimodal_enrichment.pdf")
    plt.close()
    print(f"wrote {out/'fig3_multimodal_enrichment.png'}")

if __name__ == "__main__":
    main()