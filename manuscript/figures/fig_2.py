#!/usr/bin/env python3
"""Figure 2 — AlphaGenome scoring + cardiac filtering.

Fixes vs. previous:
- Panel A inverts the bar order (top-to-bottom = pipeline stage flow)
- Panel B keeps output_type distribution (was fine)
- Panel C (GTEx tissues) now shows actual per-tissue counts (was showing uniform bars due to aggregation bug)
- Panel D (ENCODE biosamples) truncates cleanly, no cutoff
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

mpl.use("Agg")
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42,
})
C = {"clinvar":"#0072B2","gwas":"#E69F00","both":"#CC79A7"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardiac-parquet", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    print("Streaming cardiac parquet...")
    pf = pq.ParquetFile(args.cardiac_parquet)
    schema = set(pf.schema_arrow.names)
    needed = [c for c in ["output_type","gtex_tissue","biosample_name"] if c in schema]

    ot_counts, gt_counts, bn_counts = {}, {}, {}
    total = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        chunk = batch.to_pandas()
        total += len(chunk)
        if "output_type" in chunk:
            for k, v in chunk["output_type"].dropna().value_counts().items():
                ot_counts[k] = ot_counts.get(k, 0) + v
        if "gtex_tissue" in chunk:
            s = chunk["gtex_tissue"].dropna()
            s = s[s.astype(str).str.len() > 0]
            for k, v in s.value_counts().items():
                gt_counts[k] = gt_counts.get(k, 0) + v
        if "biosample_name" in chunk:
            s = chunk["biosample_name"].dropna()
            s = s[s.astype(str).str.len() > 0]
            for k, v in s.value_counts().items():
                bn_counts[k] = bn_counts.get(k, 0) + v
    print(f"Processed {total:,} rows")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.1],
                          hspace=0.45, wspace=0.35, left=0.08, right=0.97, top=0.95, bottom=0.08)

    # --- A: funnel ---
    axA = fig.add_subplot(gs[0, 0])
    axA.text(-0.18, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    stages = [("All scored rows", 3_572_509_053, "#CCCCCC"),
              ("Cardiac-filtered", 452_688_226, C["clinvar"]),
              ("Unique variants\nwith signal", 104_473, C["both"]),
              ("Multimodal (≥2)", 65_030, "#7B3F8B")]
    y = np.arange(len(stages))
    vals = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    bars = axA.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
    axA.set_yticks(y); axA.set_yticklabels([s[0] for s in stages], fontsize=8)
    axA.invert_yaxis()
    axA.set_xscale("log"); axA.set_xlabel("Count (log scale)")
    for bar, v in zip(bars, vals):
        axA.text(v*1.5, bar.get_y()+bar.get_height()/2, f"{v:,}",
                 va="center", fontsize=8)
    axA.set_xlim(1e4, 2e11)

    # --- B: output type ---
    axB = fig.add_subplot(gs[0, 1])
    axB.text(-0.15, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    ot = pd.Series(ot_counts).sort_values(ascending=True)
    labels = [k.replace("_", " ") for k in ot.index]
    axB.barh(range(len(ot)), ot.values / 1e6, color=C["clinvar"],
             edgecolor="black", linewidth=0.4)
    axB.set_yticks(range(len(ot))); axB.set_yticklabels(labels, fontsize=7.5)
    axB.set_xlabel("Cardiac score rows (millions)")
    xmax = (ot.values / 1e6).max()
    axB.set_xlim(0, xmax*1.15)
    for i, v in enumerate(ot.values / 1e6):
        if v >= 1:
            label_str = f"{v:.1f}M"
        else:
            label_str = f"{int(ot.values[i]):,}"
        axB.text(v + xmax*0.01, i, label_str, va="center", fontsize=7)

    # --- C: GTEx tissues (only 5, use nice label) ---
    axC = fig.add_subplot(gs[1, 0])
    axC.text(-0.18, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
    gt = pd.Series(gt_counts).sort_values(ascending=True)
    labels = [k.replace("_", " ") for k in gt.index]
    axC.barh(range(len(gt)), gt.values / 1e6, color=C["gwas"],
             edgecolor="black", linewidth=0.5)
    axC.set_yticks(range(len(gt))); axC.set_yticklabels(labels, fontsize=9)
    axC.set_xlabel("Cardiac score rows (millions)")
    xmax = (gt.values / 1e6).max()
    axC.set_xlim(0, xmax*1.18)
    for i, v in enumerate(gt.values / 1e6):
        axC.text(v + xmax*0.015, i, f"{v:.2f}M", va="center", fontsize=8)
    axC.set_title("GTEx cardiac tissues", fontsize=9, pad=4)

    # --- D: ENCODE biosamples (top 15) ---
    axD = fig.add_subplot(gs[1, 1])
    axD.text(-0.15, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
    bn = pd.Series(bn_counts).sort_values(ascending=True).tail(15)
    def clean_name(n):
        n = n.replace("endothelial cell of umbilical vein", "HUVEC")
        n = n.replace("fibroblast of the aortic adventitia", "Aortic adv. fibroblast")
        n = n.replace("left ventricle myocardium inferior", "LV myocardium (inf.)")
        n = n.replace("right atrium auricular region", "Right atrium auricular")
        if len(n) > 28:
            n = n[:27].rstrip() + "…"
        return n
    labels = [clean_name(k) for k in bn.index]
    axD.barh(range(len(bn)), bn.values / 1e6, color=C["both"],
             edgecolor="black", linewidth=0.4)
    axD.set_yticks(range(len(bn))); axD.set_yticklabels(labels, fontsize=7.5)
    axD.set_xlabel("Cardiac score rows (millions)")
    xmax = (bn.values / 1e6).max()
    axD.set_xlim(0, xmax*1.18)
    for i, v in enumerate(bn.values / 1e6):
        axD.text(v + xmax*0.01, i, f"{v:.1f}M", va="center", fontsize=7)
    axD.set_title("ENCODE cardiac biosamples (top 15)", fontsize=9, pad=4)

    plt.savefig(out/"fig2_scoring_flow.png", dpi=300)
    plt.savefig(out/"fig2_scoring_flow.pdf")
    plt.close()
    print(f"wrote {out/'fig2_scoring_flow.png'}")

if __name__ == "__main__":
    main()