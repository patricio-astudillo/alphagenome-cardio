#!/usr/bin/env python3
"""Build all four manuscript tables + S1/S2 supplementary tables.

Fixes vs. previous version:
- Table 1 no longer miscounts VUS/VUS_conflicting as subtraction from Pathogenic
- Table 1 splits columns clearly: Pathogenic | Likely Pathogenic | VUS | VUS_conflicting | LB/B
- Table 3 filters on correct columns (confidence == 'high' AND is_cardiac_trait == True)
- Table 4 drops clinvar_gene (not present in ranked_variant_table), uses expression_top_gene fallback

Usage:
    python build_tables.py \
        --ws567-dir /mnt/local_data/alphagenome-cardio/ws567_output \
        --clinvar /mnt/local_data/alphagenome-cardio/clinvar/clinvar_cardio_extract.tsv \
        --merged /mnt/local_data/alphagenome-cardio/variant_interval/merged_cardio_variants.tsv \
        --output /mnt/local_data/alphagenome-cardio/manuscript/tables/

Produces:
    table1_sources.tsv
    table2_top50.tsv
    table3_doe_top20.tsv
    table4_vus_candidates.tsv
    supp_S1_gene_panel.tsv
    supp_S2_trait_keywords.tsv
"""
import argparse
from pathlib import Path
import pandas as pd


def classify_clinvar(sig_series):
    """Return a dict of counts: pathogenic / likely_pathogenic / vus / vus_conflicting / benign.

    Uses exact-match labels — no ambiguous substring heuristics. ClinVar emits these
    specific values in `clinical_significance` after the standard filtering pipeline.
    """
    s = sig_series.fillna("").astype(str)
    # Exact matches first (after case normalization). Order matters: check conflicting
    # before non-conflicting so we don't double-count.
    lower = s.str.lower()
    vus_conflicting = lower.isin({"vus_conflicting"})
    is_vus = lower.isin({"vus", "uncertain_significance", "uncertain significance"})
    is_pathogenic = lower.isin({"pathogenic"})
    is_likely_pathogenic = lower.isin({"likely_pathogenic", "likely pathogenic"})
    is_benign = lower.isin({"benign", "likely_benign", "likely benign", "benign/likely_benign"})
    return {
        "pathogenic": int(is_pathogenic.sum()),
        "likely_pathogenic": int(is_likely_pathogenic.sum()),
        "vus": int(is_vus.sum()),
        "vus_conflicting": int(vus_conflicting.sum()),
        "benign": int(is_benign.sum()),
        "unclassified": int((~(is_pathogenic | is_likely_pathogenic | is_vus |
                              vus_conflicting | is_benign)).sum()),
    }


def build_table1(clinvar_path, merged_path, outdir):
    merged = pd.read_csv(merged_path, sep="\t", low_memory=False)
    clinvar = pd.read_csv(clinvar_path, sep="\t", low_memory=False)

    cat = classify_clinvar(clinvar["clinical_significance"])
    print(f"  ClinVar source categorization: {cat}")

    n_both = int((merged["in_clinvar"] & merged["in_gwas"]).sum())
    n_cv_only = int(merged["in_clinvar"].sum() - n_both)
    n_gwas_only = int(merged["in_gwas"].sum() - n_both)

    rows = [
        {"Source": "ClinVar only", "Total": n_cv_only,
         "Pathogenic": cat["pathogenic"],
         "Likely Pathogenic": cat["likely_pathogenic"],
         "VUS": cat["vus"],
         "VUS (conflicting)": cat["vus_conflicting"],
         "B/LB": cat["benign"]},
        {"Source": "GWAS only", "Total": n_gwas_only,
         "Pathogenic": "—", "Likely Pathogenic": "—",
         "VUS": "—", "VUS (conflicting)": "—", "B/LB": "—"},
        {"Source": "Dual-source (both)", "Total": n_both,
         "Pathogenic": 0, "Likely Pathogenic": 0,
         "VUS": n_both, "VUS (conflicting)": 0, "B/LB": 0},
        {"Source": "TOTAL UNIQUE", "Total": len(merged),
         "Pathogenic": "", "Likely Pathogenic": "",
         "VUS": "", "VUS (conflicting)": "", "B/LB": ""},
    ]
    pd.DataFrame(rows).to_csv(outdir / "table1_sources.tsv", sep="\t", index=False)
    print(f"  Table 1 -> {outdir/'table1_sources.tsv'}")


def build_table2(ranked_path, outdir, n=50):
    """Top N variants by composite score."""
    df = pd.read_csv(ranked_path, sep="\t", low_memory=False, nrows=n)
    # Pick publication-relevant columns if present, skip clutter
    wanted = ["variant_id", "chromosome", "position",
              "expression_top_gene", "composite_score", "n_modalities_strong",
              "in_clinvar", "in_gwas", "clinical_significance",
              "gwas_trait", "l2g_gene", "max_pip"]
    keep = [c for c in wanted if c in df.columns]
    df[keep].to_csv(outdir / "table2_top50.tsv", sep="\t", index=False)
    print(f"  Table 2 -> {outdir/'table2_top50.tsv'} ({len(df)} rows, {len(keep)} cols)")


def build_table3(doe_path, outdir, n=20):
    """Top N cardiac, high-confidence direction-of-effect predictions, by |LFC|."""
    df = pd.read_csv(doe_path, sep="\t", low_memory=False)
    sel = df[(df["confidence"] == "high") & (df["is_cardiac_trait"] == True)]
    sel = sel.reindex(
        sel["max_abs_cardiac_lfc"].abs().sort_values(ascending=False).index
    ).head(n)
    sel.to_csv(outdir / "table3_doe_top20.tsv", sep="\t", index=False)
    print(f"  Table 3 -> {outdir/'table3_doe_top20.tsv'} ({len(sel)} rows)")


def build_table4(ranked_path, outdir, n=25):
    """Top N VUS candidates for functional validation, by composite score."""
    df = pd.read_csv(ranked_path, sep="\t", low_memory=False)
    if "clinical_significance" not in df.columns:
        print("  WARNING: no clinical_significance in ranked_variant_table; skipping Table 4")
        return
    mask = df["clinical_significance"].astype(str).str.contains(
        "VUS|Uncertain", case=False, na=False
    )
    vus = df[mask].nlargest(n, "composite_score").copy()

    # Synthesize a PP3 evidence summary for each variant — a short manuscript-ready phrase
    # combining composite-score strength, modality count, top modality, and tissue.
    def pp3_text(row):
        n_mods = int(row.get("n_modalities_strong", 0))
        # Identify strongest modality by max_quantile
        mod_qs = {}
        for mod in ["expression", "splicing", "accessibility", "tf_binding",
                    "histone_marks", "3d_structure"]:
            col = f"{mod}_max_quantile"
            if col in row and pd.notna(row[col]):
                mod_qs[mod] = row[col]
        top_mod = max(mod_qs, key=mod_qs.get) if mod_qs else "n/a"
        tissue = row.get("expression_top_tissue", "")
        strength = ("strong" if n_mods >= 5 else
                    "moderate" if n_mods >= 3 else "weak")
        return (f"{strength.capitalize()} multimodal signal "
                f"({n_mods} modalities; top: {top_mod.replace('_', ' ')}"
                + (f", {tissue}" if tissue and str(tissue) not in ("nan", "") else "")
                + ")")
    vus["PP3_evidence_summary"] = vus.apply(pp3_text, axis=1)

    wanted = ["variant_id", "chromosome", "position", "clinical_significance",
              "composite_score", "n_modalities_strong",
              "expression_top_gene", "expression_top_tissue",
              "in_gwas", "gwas_trait", "max_pip",
              "PP3_evidence_summary"]
    keep = [c for c in wanted if c in vus.columns]
    vus[keep].to_csv(outdir / "table4_vus_candidates.tsv", sep="\t", index=False)
    print(f"  Table 4 -> {outdir/'table4_vus_candidates.tsv'} ({len(vus)} rows)")


def build_s1_gene_panel(outdir):
    """Supplementary Table S1: cardiac gene panel used for vignette selection."""
    # Grouped for readability; change as needed
    panel_groups = [
        ("Cardiomyopathy (HCM/DCM/RCM/ARVC)", [
            "MYH7","MYBPC3","TNNT2","TNNI3","TPM1","ACTC1","MYL2","MYL3","TNNC1","CSRP3",
            "TCAP","MYH6","ACTN2","PLN","JPH2","NEXN","LMNA","DES","DSP","DSG2","DSC2",
            "PKP2","JUP","TMEM43","BAG3","FLNC","TTN","RBM20","LDB3","VCL","ANKRD1"]),
        ("Channelopathy (LQTS/SQTS/Brugada/CPVT)", [
            "SCN5A","KCNQ1","KCNH2","KCNE1","KCNE2","KCNJ2","KCNJ5","CACNA1C","CACNB2",
            "RYR2","CASQ2","CALM1","CALM2","CALM3","TRDN","AKAP9","ANK2","SNTA1","CAV3"]),
        ("Aortopathy / connective tissue", [
            "FBN1","FBN2","TGFBR1","TGFBR2","TGFB2","TGFB3","SMAD3","ACTA2","MYH11",
            "MYLK","PRKG1","COL3A1","LOX"]),
        ("Lipid metabolism / atherosclerosis", [
            "LDLR","APOB","PCSK9","APOE","LDLRAP1","LIPA","ABCG5","ABCG8","LPL","APOC2",
            "APOC3","APOA1","APOA5","CETP","SORT1","HMGCR","NPC1L1"]),
        ("Pulmonary arterial hypertension", ["BMPR2","ACVRL1","ENG"]),
        ("Congenital heart disease / development", [
            "GATA4","GATA6","NKX2-5","TBX5","TBX1","TBX20","NOTCH1","JAG1"]),
        ("Atrial fibrillation", ["PITX2","ZFHX3","KCNN3","GJA5"]),
        ("CAD / GWAS-implicated", ["LPA","PHACTR1","TCF21","CDKN2A","CDKN2B"]),
        ("Hypertension", ["ACE","AGT","AGTR1"]),
        ("Conduction / inherited arrhythmia", ["HCN4","GJA1"]),
    ]
    rows = []
    for cat, genes in panel_groups:
        for g in genes:
            rows.append({"Category": cat, "Gene": g})
    pd.DataFrame(rows).to_csv(outdir / "supp_S1_gene_panel.tsv", sep="\t", index=False)
    print(f"  Supp S1 -> {outdir/'supp_S1_gene_panel.tsv'}")


def build_s2_trait_keywords(outdir):
    """Supplementary Table S2: trait keyword inclusion/exclusion lists."""
    cardiac_keywords = [
        "heart", "cardiac", "cardio", "myocardi", "ventric", "atria", "coron",
        "aort", "valve", "mitral", "tricuspid", "septal",
        "atrial fibrillation", "arrhythm", "sudden cardiac", "QT", "brugada",
        "tachy", "bradycardi", "wolff",
        "blood pressure", "hypertens", "hypotens", "vascul", "arteri", "vein",
        "thromb", "stroke", "ischemi", "infarct",
        "cholesterol", "ldl", "hdl", "lipid", "triglycerid", "lipoprotein", "apoa", "apob",
        "diabet", "glucose", "obesity", "bmi", "waist",
        "cad", "chd", "ihd", "cardiovascular", "circulation",
        "pulmonary hypertension", "pulmonary arterial",
    ]
    exclude_keywords = [
        "bone mineral density", "bmd ", "heel bone", "osteoporo", "fracture",
        "calcium levels", "vitamin d", "magnesium", "phosphate",
        "kidney function", "egfr", "urate",
        "alzheimer", "parkinson", "schizophren", "depression", "bipolar",
        "asthma", "lung function", "fev1",
        "skin", "hair", "balding", "eczema",
        "intelligence", "education", "income",
        "cancer", "tumor", "neoplasm", "leukemia", "lymphoma",
    ]
    rows = []
    for k in cardiac_keywords:
        rows.append({"List": "Cardiac (include)", "Keyword": k})
    for k in exclude_keywords:
        rows.append({"List": "Exclusion (drop even if cardiac kw matches)", "Keyword": k})
    pd.DataFrame(rows).to_csv(outdir / "supp_S2_trait_keywords.tsv", sep="\t", index=False)
    print(f"  Supp S2 -> {outdir/'supp_S2_trait_keywords.tsv'}")


def build_s3_full_ranked(ranked_path, outdir):
    """Supplementary Table S3: full ranked variant table (all scored variants).
    Large — should be submitted as .xlsx supplementary.
    """
    df = pd.read_csv(ranked_path, sep="\t", low_memory=False)
    out_tsv = outdir / "supp_S3_full_ranked_variants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Supp S3 -> {out_tsv} ({len(df):,} rows)")
    # Also emit as xlsx if openpyxl is available
    # try:
    #     out_xlsx = outdir / "supp_S3_full_ranked_variants.xlsx"
    #     df.to_excel(out_xlsx, index=False, engine="openpyxl")
    #     print(f"  Supp S3 -> {out_xlsx} (Excel copy)")
    # except ImportError:
    #     print(f"  (install openpyxl to also emit .xlsx)")


def build_s4_doe_cardiac(doe_path, outdir):
    """Supplementary Table S4: full GWAS DoE filtered to cardiac-trait pairs."""
    df = pd.read_csv(doe_path, sep="\t", low_memory=False)
    cardiac = df[df["is_cardiac_trait"] == True].copy()
    # Sort by |LFC| so the strongest hits are at the top
    cardiac = cardiac.reindex(
        cardiac["max_abs_cardiac_lfc"].abs().sort_values(ascending=False).index
    )
    out_tsv = outdir / "supp_S4_doe_cardiac_full.tsv"
    cardiac.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Supp S4 -> {out_tsv} ({len(cardiac):,} rows)")
    # try:
    #     out_xlsx = outdir / "supp_S4_doe_cardiac_full.xlsx"
    #     cardiac.to_excel(out_xlsx, index=False, engine="openpyxl")
    #     print(f"  Supp S4 -> {out_xlsx} (Excel copy)")
    # except ImportError:
    #     print(f"  (install openpyxl to also emit .xlsx)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws567-dir", required=True)
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ws567 = Path(args.ws567_dir)
    ranked = ws567 / "ranked_variant_table.tsv"
    doe = ws567 / "gwas_direction_of_effect.tsv"

    print("=== Building tables ===")
    build_table1(args.clinvar, args.merged, out)
    build_table2(ranked, out)
    build_table3(doe, out)
    build_table4(ranked, out)
    build_s1_gene_panel(out)
    build_s2_trait_keywords(out)
    build_s3_full_ranked(ranked, out)
    build_s4_doe_cardiac(doe, out)
    print("\nDone. Tables in:", out)


if __name__ == "__main__":
    main()