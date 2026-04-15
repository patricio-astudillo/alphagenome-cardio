#!/usr/bin/env python3
"""Figure 5 — GWAS direction-of-effect.

Fixes vs. previous:
- Panel C left-side text (Variant/Gene) shifted further left to prevent plot overlap
- Panel C dot plot repositioned slightly to the right to create more column space
- Panel C label moved to absolute left position for better alignment with A and B
- Panel C right-side text (Traits) truncation limit increased from 32 to 65
- Three panels on their own vertical axes, no title/label overlap
- Panel B scatter uses density-aware alpha; highlights top 10 with rings
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.use("Agg")
mpl.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9, "axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":300,"savefig.dpi":300,
    "pdf.fonttype":42,
})
C = {"up":"#D55E00","down":"#009E73",
     "highconf":"#0072B2","modconf":"#56B4E9","lowconf":"#CCCCCC"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.doe, sep="\t", low_memory=False)
    cardiac = df[df["is_cardiac_trait"] == True].copy()
    resolved = cardiac[cardiac["direction"] != "UNCERTAIN"]
    hc = cardiac[cardiac["confidence"] == "high"].copy()
    hc = hc[hc["max_pip"] > 0]
    top10 = hc.reindex(hc["max_abs_cardiac_lfc"].abs().sort_values(ascending=False).index).head(10).copy()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Two-row layout: top row has A+B, bottom row is C (full width)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1, 1.3],
                          hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.07)

    # ---------- A: stacked confidence bars ----------
    axA = fig.add_subplot(gs[0, 0])
    axA.text(-0.18, 1.06, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    data = {"UP":{}, "DOWN":{}}
    for d in ("UP","DOWN"):
        for conf in ("low","moderate","high"):
            data[d][conf] = int(((resolved["direction"]==d) & (resolved["confidence"]==conf)).sum())
    bottoms = {"UP":0,"DOWN":0}
    for conf, col in [("low",C["lowconf"]),("moderate",C["modconf"]),("high",C["highconf"])]:
        axA.bar(["UP","DOWN"], [data["UP"][conf], data["DOWN"][conf]],
                bottom=[bottoms["UP"], bottoms["DOWN"]], color=col,
                edgecolor="black", linewidth=0.4, label=conf)
        bottoms["UP"] += data["UP"][conf]; bottoms["DOWN"] += data["DOWN"][conf]
    axA.set_ylabel("Variant-gene pairs")
    axA.legend(title="Confidence", loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=7, title_fontsize=7, frameon=False)
    axA.set_title(f"{len(resolved):,} resolved pairs of {len(cardiac):,}", fontsize=9, pad=4)
    for i, d in enumerate(("UP","DOWN")):
        tot = sum(data[d].values())
        axA.text(i, tot*1.02, f"{tot:,}", ha="center", va="bottom", fontsize=8)
    axA.set_ylim(0, max(bottoms.values())*1.15)

    # ---------- B: PIP vs |LFC| scatter ----------
    axB = fig.add_subplot(gs[0, 1])
    axB.text(-0.12, 1.06, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    colors = hc["direction"].map({"UP":C["up"],"DOWN":C["down"]})
    axB.scatter(hc["max_pip"], hc["max_abs_cardiac_lfc"], s=6, alpha=0.3,
                c=colors, edgecolors="none")
    # Highlight top 10 with ring markers
    axB.scatter(top10["max_pip"], top10["max_abs_cardiac_lfc"], s=50,
                facecolors="none", edgecolors="black", linewidth=1.1, zorder=5)
    axB.set_xlabel("GWAS max PIP"); axB.set_ylabel("Max |cardiac LFC|")
    axB.set_title(f"High-confidence pairs (n={len(hc):,}; circled = top 10)", fontsize=9, pad=4)
    axB.legend(handles=[Line2D([0],[0],marker='o',color=C["up"],ls='',markersize=6,label="UP"),
                        Line2D([0],[0],marker='o',color=C["down"],ls='',markersize=6,label="DOWN")],
               loc="upper right", fontsize=7, frameon=False)

    # ---------- C: top 10 dot plot ----------
    axC = fig.add_subplot(gs[1, :])
    
    # Place "C" label absolutely so it aligns nicely with the left side of the figure
    fig.text(0.04, 0.46, "C", fontsize=13, fontweight="bold")
    
    top = top10.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))
    colors_c = top["direction"].map({"UP":C["up"],"DOWN":C["down"]})
    axC.scatter(top["mean_cardiac_lfc"], y,
                s=top["max_pip"]*600 + 40, c=colors_c,
                edgecolors="black", linewidth=0.6, alpha=0.85, zorder=3)
    axC.axvline(0, color="gray", linewidth=0.6, ls="--", zorder=1)
    
    # LFC values placed on the side AWAY from the zero-line
    xrange = top["mean_cardiac_lfc"].abs().max() * 1.25
    axC.set_xlim(-xrange, xrange)
    for yi, (_, row) in zip(y, top.iterrows()):
        lfc = row["mean_cardiac_lfc"]
        offset_x = xrange * 0.04 * (1 if lfc >= 0 else -1)
        ha = "left" if lfc >= 0 else "right"
        axC.text(lfc + offset_x, yi, f"{lfc:+.3f}",
                 va="center", ha=ha, fontsize=7.5,
                 color=C["up"] if lfc >= 0 else C["down"])

    axC.set_yticks([]); axC.set_xlabel("Mean cardiac LFC")
    axC.set_ylim(-0.7, len(top)-0.3)
    axC.set_title("Top 10 high-confidence direction-of-effect predictions (dot size ∝ PIP)",
                  fontsize=9, pad=6)
                  
    # Adjust plot position: start further right (0.31) and slightly narrow it (0.26 width) 
    # to give the left and right text columns enough horizontal room.
    axC.set_position([0.31, 0.07, 0.26, 0.38])
    
    # Draw all tabular texts using absolute Figure coordinates
    for yi, (_, row) in zip(y, top.iterrows()):
        disp_y = axC.transData.transform((0, yi))[1]
        fig_y = fig.transFigure.inverted().transform((0, disp_y))[1]
        
        # --- Left Side Columns (Variant -> Gene) ---
        vid = str(row["variant_id"])
        gene = str(row["gene_name"])
        
        # Shifted left (0.16 -> 0.18) to prevent crashing into the plot at 0.31
        fig.text(0.16, fig_y, vid, ha="right", va="center", fontsize=8, family="monospace", color="#333")
        fig.text(0.17, fig_y, "→", ha="center", va="center", fontsize=8, family="monospace", color="#888")
        fig.text(0.18, fig_y, gene, ha="left", va="center", fontsize=8, family="monospace", color="#333")
        
        # --- Right Side Columns (Trait and PIP) ---
        trait = str(row.get("gwas_trait",""))
        trait_clean = (trait
                       .replace("High density lipoprotein cholesterol levels", "HDL cholesterol")
                       .replace("High density lipoprotein cholesterol", "HDL cholesterol")
                       .replace("Low density lipoprotein cholesterol levels", "LDL cholesterol")
                       .replace("Atrial fibrillation and flutter with reimbursement", "AF/flutter")
                       .replace("Atrial fibrillation and flutter", "AF/flutter")
                       .replace("Waist-to-hip ratio adjusted for BMI", "WHR (BMI-adj)")
                       .replace("Essential hypertension (PheCode 401)", "Essential hypertension")
                       .replace("Hypertension, essential", "Essential hypertension")
                       .replace(" (UKB data field)", "")
                       .replace(" (UKB data f...)", ""))
                       
        if len(trait_clean) > 65:
            trait_clean = trait_clean[:64].rstrip() + "…"
            
        pip = row["max_pip"]
        
        fig.text(0.59, fig_y, trait_clean, ha="left", va="center", fontsize=7.5, color="#333")
        fig.text(0.90, fig_y, f"PIP={pip:.2f}", ha="left", va="center", fontsize=7, color="#888", family="monospace")

    plt.savefig(out/"fig5_gwas_doe.png", dpi=300, facecolor="white")
    plt.savefig(out/"fig5_gwas_doe.pdf", facecolor="white")
    plt.close()
    print(f"wrote {out/'fig5_gwas_doe.png'}")

if __name__ == "__main__":
    main()