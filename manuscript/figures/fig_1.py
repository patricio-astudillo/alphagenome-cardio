#!/usr/bin/env python3
"""Figure 1 — Variant curation overview.

Fixes vs. previous version:
- Panel A schematic moved to dedicated top row, no vertical whitespace below
- Panel C: uses explicit column for values, no overlap with bar labels
- Panel D: trait labels rendered in full with wider horizontal space
- Panel B-D heights equalized

Usage: python fig1.py --merged <path> --clinvar <path> --output <dir>
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
C = {"clinvar":"#0072B2","gwas":"#E69F00","both":"#CC79A7","bg":"#F5F5F5"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    merged = pd.read_csv(args.merged, sep="\t", low_memory=False)
    clinvar = pd.read_csv(args.clinvar, sep="\t", low_memory=False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Two-row layout: schematic (1), then three panels side by side (1)
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.55,
                          left=0.06, right=0.97, top=0.95, bottom=0.10)

    # ---------- Panel A: schematic ----------
    axA = fig.add_subplot(gs[0, :])
    axA.set_xlim(0, 10.5); axA.set_ylim(0, 3); axA.axis("off")
    axA.text(-0.02, 1.02, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    def box(x,y,w,h,txt,fc,bold=False):
        axA.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor=fc, edgecolor="#333", linewidth=0.8))
        axA.text(x+w/2,y+h/2,txt, ha="center",va="center",fontsize=9,
                 fontweight=("bold" if bold else "normal"))
    def arr(x1,y1,x2,y2):
        axA.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="->,head_width=4,head_length=6",
                                      color="#555",linewidth=1))

    # ClinVar lane
    box(0.2,2.05,1.8,0.7,"ClinVar\nVariant Summary\n(2024-03-31)", C["clinvar"]+"33")
    box(2.4,2.05,1.8,0.7,"Non-coding +\ncardiac phenotype\nfilters", C["bg"])
    box(4.6,2.05,1.7,0.7,f"{int(merged['in_clinvar'].sum()):,}\nClinVar variants",
        C["clinvar"]+"66", bold=True)
    arr(2.0,2.4,2.4,2.4); arr(4.2,2.4,4.6,2.4)
    # GWAS lane
    box(0.2,0.5,1.8,0.7,"Open Targets\nGraphQL API\n(release 26.03)", C["gwas"]+"33")
    box(2.4,0.5,1.8,0.7,"27 cardiovascular\nEFO terms\nPIP ≥ 0.01", C["bg"])
    box(4.6,0.5,1.7,0.7,f"{int(merged['in_gwas'].sum()):,}\nGWAS variants",
        C["gwas"]+"66", bold=True)
    arr(2.0,0.85,2.4,0.85); arr(4.2,0.85,4.6,0.85)
    # Merge
    box(6.8,1.3,1.6,0.7,"Merge + dedup\non chr:pos:ref>alt", C["bg"])
    arr(6.3,2.4,6.8,1.9); arr(6.3,0.85,6.8,1.7)
    box(8.8,1.3,1.2,0.7,f"{len(merged):,}\nunique", C["both"]+"66", bold=True)
    arr(8.4,1.65,8.8,1.65)

    # ---------- Panel B: source breakdown ----------
    axB = fig.add_subplot(gs[1, 0])
    axB.text(-0.22, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    n_both = int((merged["in_clinvar"] & merged["in_gwas"]).sum())
    vals = [int(merged["in_clinvar"].sum()-n_both),
            int(merged["in_gwas"].sum()-n_both), n_both]
    labels = ["ClinVar\nonly","GWAS\nonly","Both"]
    bars = axB.bar(labels, vals, color=[C["clinvar"],C["gwas"],C["both"]],
                   edgecolor="black", linewidth=0.5)
    axB.set_yscale("log"); axB.set_ylabel("Variants")
    axB.set_ylim(1, max(vals)*6)
    for b,v in zip(bars,vals):
        axB.text(b.get_x()+b.get_width()/2, v*1.4, f"{v:,}",
                 ha="center", va="bottom", fontsize=8)

    # ---------- Panel C: ClinVar significance ----------
    axC = fig.add_subplot(gs[1, 1])
    axC.text(-0.22, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
    sig = clinvar["clinical_significance"].fillna("Unknown").value_counts().head(6)
    clean = [s.replace("_"," ") for s in sig.index]
    y = np.arange(len(sig))[::-1]
    axC.barh(y, sig.values, color=C["clinvar"], edgecolor="black", linewidth=0.5)
    axC.set_yticks(y); axC.set_yticklabels(clean, fontsize=8)
    axC.set_xlabel("Variants")
    # Push count labels past bar ends, outside any possible tick collision
    xmax = sig.values.max()
    axC.set_xlim(0, xmax*1.18)
    for yi, v in zip(y, sig.values):
        axC.text(v + xmax*0.015, yi, f"{v:,}", va="center", fontsize=7.5)

    # ---------- Panel D: top GWAS traits ----------
    axD = fig.add_subplot(gs[1, 2])
    axD.text(-0.22, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
    if "gwas_trait" in merged.columns:
        traits = merged[merged["in_gwas"]]["gwas_trait"].fillna("").value_counts()
        traits = traits[traits.index != ""].head(10)
        # Rewrite the verbose Open Targets trait strings to clean short labels
        def clean_trait(t):
            t = t.strip()
            rewrites = {
                "High density lipoprotein cholesterol levels": "HDL cholesterol",
                "High density lipoprotein cholesterol": "HDL cholesterol",
                "Low density lipoprotein cholesterol levels": "LDL cholesterol",
                "Low density lipoprotein cholesterol": "LDL cholesterol",
                "Serum low density lipoprotein cholesterol levels": "Serum LDL cholesterol",
                "Serum high density lipoprotein cholesterol levels": "Serum HDL cholesterol",
                "systolic blood pressure (SBP, maximum)": "Systolic BP (max)",
                "systolic blood pressure (SBP, automated)": "Systolic BP (automated)",
                "Heel bone mineral density (BMD) (UKB data field 3148)": "Heel BMD",
                "Heel bone mineral density (BMD)": "Heel BMD",
                "Waist-to-hip ratio adjusted for BMI": "WHR (BMI-adjusted)",
                "Triglyceride levels (UKB data field)": "Triglyceride levels",
                "Atrial fibrillation and flutter": "Atrial fibrillation/flutter",
            }
            for k, v in rewrites.items():
                if t.startswith(k):
                    return v
            # Generic truncate for anything else
            return t if len(t) <= 32 else t[:31].rstrip() + "…"
        clean_t = [clean_trait(t) for t in traits.index]
        y = np.arange(len(traits))[::-1]
        axD.barh(y, traits.values, color=C["gwas"], edgecolor="black", linewidth=0.5)
        axD.set_yticks(y); axD.set_yticklabels(clean_t, fontsize=8)
        axD.set_xlabel("Variants")
        xmax = traits.values.max()
        axD.set_xlim(0, xmax*1.15)
        for yi, v in zip(y, traits.values):
            axD.text(v + xmax*0.015, yi, f"{v:,}", va="center", fontsize=7.5)

    plt.savefig(out/"fig1_variant_curation.png", dpi=300)
    plt.savefig(out/"fig1_variant_curation.pdf")
    plt.close()
    print(f"wrote {out/'fig1_variant_curation.png'}")

if __name__ == "__main__":
    main()