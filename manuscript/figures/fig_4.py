#!/usr/bin/env python3
"""Figure 4 — Top-ranked variants + cardiac panel enrichment.

Fixes vs. previous:
- Panel A: Left side y-tick labels use standard fixed-width spacing for perfect vertical alignment.
- Panel A: VUS star mark moved inside the barplot (white with black outline for visibility).
- Panel A: legend positioned below plot, not overlapping bars
- Panel B: fill region clipped above baseline only
- Panel C: heatmap annotations use readable white/black based on cell value
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

mpl.use("Agg")
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42,
})
C = {"panel":"#CC79A7","nonpanel":"#0072B2","vus":"#CC79A7"}

CARDIAC_GENE_PANEL = {
    "MYH7","MYBPC3","TNNT2","TNNI3","TPM1","ACTC1","MYL2","MYL3","TNNC1","CSRP3","TCAP",
    "MYH6","ACTN2","PLN","JPH2","NEXN","LMNA","DES","DSP","DSG2","DSC2","PKP2","JUP",
    "TMEM43","BAG3","FLNC","TTN","RBM20","SCN5A","LDB3","VCL","ANKRD1","KCNQ1","KCNH2",
    "KCNE1","KCNE2","KCNJ2","KCNJ5","CACNA1C","CACNB2","RYR2","CASQ2","CALM1","CALM2",
    "CALM3","TRDN","AKAP9","ANK2","SNTA1","CAV3","FBN1","FBN2","TGFBR1","TGFBR2","TGFB2",
    "TGFB3","SMAD3","ACTA2","MYH11","MYLK","PRKG1","COL3A1","LOX","LDLR","APOB","PCSK9",
    "APOE","LDLRAP1","LIPA","ABCG5","ABCG8","LPL","APOC2","APOC3","APOA1","APOA5","CETP",
    "SORT1","HMGCR","NPC1L1","BMPR2","ACVRL1","ENG","GATA4","GATA6","NKX2-5","TBX5","TBX1",
    "TBX20","NOTCH1","JAG1","PITX2","ZFHX3","KCNN3","GJA5","LPA","PHACTR1","TCF21","CDKN2A",
    "CDKN2B","ACE","AGT","AGTR1","HCN4","GJA1",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.ranked, sep="\t", low_memory=False)
    gene_col = next((c for c in ["expression_top_gene","resolved_gene","l2g_gene"]
                     if c in df.columns), None)
    df["gene"] = df[gene_col].fillna("?")
    df["in_panel"] = df["gene"].isin(CARDIAC_GENE_PANEL)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1],
                          hspace=0.40, wspace=0.30,
                          left=0.06, right=0.97, top=0.95, bottom=0.10)

    # --- A: top 20 ---
    axA = fig.add_subplot(gs[:, 0])
    axA.text(-0.22, 1.02, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    top20 = df.head(20).copy()
    top20["is_vus"] = top20.get("clinical_significance", pd.Series([""]*len(top20)))\
                         .astype(str).str.contains("VUS|Uncertain", case=False)
    
    # Align text labels properly by setting variant_id to a standard fixed size
    labels = []
    for _, row in top20.iterrows():
        var_id = str(row['variant_id'])[:24].ljust(24)
        gene = str(row['gene'])
        labels.append(f"{var_id}   {gene}")
        
    colors = [C["panel"] if p else C["nonpanel"] for p in top20["in_panel"]]
    y = np.arange(len(top20))[::-1]
    
    # Draw bars
    bars = axA.barh(y, top20["composite_score"].values, color=colors,
             edgecolor="black", linewidth=0.4)
             
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=7.5, family="monospace")
    
    max_score = top20["composite_score"].max()
    axA.set_xlabel("Composite score")
    axA.set_xlim(0, max_score * 1.03)

    # Insert VUS stars inside the bars
    star_x = max_score * 0.02 # Position star slightly offset from the left inside the bar
    for i, (_, row) in enumerate(top20.iterrows()):
        if row["is_vus"]:
            txt = axA.text(star_x, y[i], "★", color="white",
                           ha="left", va="center", fontsize=10)
            # Add a subtle black outline so it pops on both dark/light backgrounds
            txt.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='black'),
                                  path_effects.Normal()])

    # Legend below the plot
    legend = [Patch(color=C["panel"], label="Cardiac panel gene"),
              Patch(color=C["nonpanel"], label="Non-panel gene"),
              Patch(color="white", label="★ = ClinVar VUS")]
    
    # Adjust legend star patch outline to match what we drew
    leg = axA.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                     ncol=3, fontsize=8, frameon=False)
    
    # Fix for newer matplotlib versions: legend_handles instead of legendHandles
    leg_handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", None))
    if leg_handles is not None and len(leg_handles) > 2:
        leg_handles[2].set_edgecolor("black")
        leg_handles[2].set_linewidth(1.0)

    # --- B: enrichment curve ---
    axB = fig.add_subplot(gs[0, 1])
    axB.text(-0.15, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    cumfrac = df["in_panel"].expanding().mean()
    N = min(5000, len(df))
    xs = np.arange(1, N+1)
    ys = cumfrac.values[:N]
    baseline = df["in_panel"].mean()
    axB.fill_between(xs, baseline, ys, where=(ys > baseline),
                     color=C["panel"], alpha=0.25,
                     label="Enrichment over baseline")
    axB.plot(xs, ys, color=C["panel"], linewidth=1.5)
    axB.axhline(baseline, ls="--", color="gray", linewidth=0.8,
                label=f"Baseline ({baseline*100:.2f}%)")
    axB.set_xscale("log")
    axB.set_xlabel("Top N variants (composite-ranked)")
    axB.set_ylabel("Fraction in cardiac panel")
    axB.legend(loc="upper right", fontsize=7, frameon=False)

    # --- C: per-gene heatmap (top 20 panel genes) ---
    axC = fig.add_subplot(gs[1, 1])
    axC.text(-0.15, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
    panel_df = df[df["in_panel"]].copy()
    gene_counts = panel_df["gene"].value_counts().head(20)
    panel_df["comp_bin"] = pd.qcut(panel_df["composite_score"],
                                    q=4, labels=["Q1","Q2","Q3","Q4"],
                                    duplicates="drop")
    heat = pd.crosstab(panel_df["gene"], panel_df["comp_bin"]).loc[gene_counts.index]
    vmax = heat.values.max()
    im = axC.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    axC.set_yticks(range(len(heat))); axC.set_yticklabels(heat.index, fontsize=7)
    axC.set_xticks(range(len(heat.columns))); axC.set_xticklabels(heat.columns, fontsize=8)
    axC.set_xlabel("Composite quartile")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i,j]
            if v > 0:
                color = "white" if v > vmax*0.55 else "black"
                axC.text(j, i, f"{v:,}", ha="center", va="center",
                         fontsize=6.5, color=color, fontweight="bold")
    cbar = plt.colorbar(im, ax=axC, shrink=0.8, pad=0.02)
    cbar.set_label("Variants", fontsize=8); cbar.ax.tick_params(labelsize=7)

    plt.savefig(out/"fig4_top_ranked.png", dpi=300)
    plt.savefig(out/"fig4_top_ranked.pdf")
    plt.close()
    print(f"wrote {out/'fig4_top_ranked.png'}")

if __name__ == "__main__":
    main()