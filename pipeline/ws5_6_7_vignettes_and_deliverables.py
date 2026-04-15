#!/usr/bin/env python3
"""Workstreams 5-7: Mechanistic Vignettes, GWAS Sign Prediction & Deliverables.

WS5: Run ISM on top variants to reveal disrupted TF motifs, generate figures.
WS6: For GWAS variants, determine direction-of-effect on cardiac genes.
WS7: Package everything into ranked tables and a summary report.

Usage:
    # Full run (requires AlphaGenome API)
    python ws5_6_7_vignettes_and_deliverables.py \
        --variant-summary /path/to/ws3_output/variant_summary.tsv \
        --cardiac-scores /path/to/ws3_output/cardiac_scores.parquet \
        --merged-variants /path/to/variant_interval/merged_cardio_variants.tsv \
        --output-dir /path/to/ws567_output/ \
        --api-key YOUR_KEY

    # Analysis only (skip ISM, use existing scores)
    python ws5_6_7_vignettes_and_deliverables.py \
        --variant-summary /path/to/ws3_output/variant_summary.tsv \
        --cardiac-scores /path/to/ws3_output/cardiac_scores.parquet \
        --merged-variants /path/to/variant_interval/merged_cardio_variants.tsv \
        --output-dir /path/to/ws567_output/ \
        --skip-ism
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as _pa
import pyarrow.parquet as _pq
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HAS_MPL = True



from alphagenome.data import genome
from alphagenome.data import gene_annotation
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
HAS_AG = True


# ---------------------------------------------------------------------------
# 0. CONSTANTS: Cardiac gene panel and trait filter keywords
# ---------------------------------------------------------------------------

# Curated cardiovascular gene panel — combines OMIM cardiac genes, ClinGen
# cardiac panels (HCM, DCM, ARVC, LQTS, Brugada, AF), and lipid metabolism.
# Used to prefer biologically meaningful vignette candidates over uncharacterized
# pseudogenes and ENSG IDs.
CARDIAC_GENE_PANEL = {
    # Cardiomyopathies (HCM/DCM/RCM/ARVC)
    "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3",
    "TNNC1", "CSRP3", "TCAP", "MYH6", "ACTN2", "PLN", "JPH2", "NEXN",
    "LMNA", "DES", "DSP", "DSG2", "DSC2", "PKP2", "JUP", "TMEM43",
    "BAG3", "FLNC", "TTN", "RBM20", "SCN5A", "LDB3", "VCL", "ANKRD1",
    # Channelopathies (LQTS/SQTS/Brugada/CPVT)
    "KCNQ1", "KCNH2", "KCNE1", "KCNE2", "KCNJ2", "KCNJ5", "CACNA1C",
    "CACNB2", "RYR2", "CASQ2", "CALM1", "CALM2", "CALM3", "TRDN", "AKAP9",
    "ANK2", "SNTA1", "CAV3",
    # Aortopathies / connective tissue
    "FBN1", "FBN2", "TGFBR1", "TGFBR2", "TGFB2", "TGFB3", "SMAD3", "ACTA2",
    "MYH11", "MYLK", "PRKG1", "COL3A1", "LOX", "FOXE3", "MFAP5", "EFEMP2",
    # Lipid metabolism / atherosclerosis
    "LDLR", "APOB", "PCSK9", "APOE", "LDLRAP1", "LIPA", "ABCG5", "ABCG8",
    "LPL", "APOC2", "APOC3", "APOA1", "APOA5", "ANGPTL3", "ANGPTL4",
    "CETP", "SORT1", "HMGCR", "NPC1L1",
    # Pulmonary arterial hypertension
    "BMPR2", "ACVRL1", "ENG", "SMAD9", "CAV1", "KCNK3", "EIF2AK4",
    # Congenital heart disease / development
    "GATA4", "GATA6", "NKX2-5", "TBX5", "TBX1", "TBX20", "NOTCH1", "JAG1",
    "ZIC3", "CFC1", "CRELD1", "MED13L", "PTPN11", "RAF1", "SOS1", "BRAF",
    "MAP2K1", "MAP2K2", "HRAS", "KRAS", "NRAS", "SHOC2", "CBL", "RIT1",
    "NF1",
    # Atrial fibrillation
    "PITX2", "ZFHX3", "KCNN3", "PRRX1", "CAV1", "GREM2", "MYL4", "GJA5",
    # Coronary artery disease (top GWAS-implicated genes)
    "SORT1", "PCSK9", "LPA", "PHACTR1", "TCF21", "CXCL12", "CDKN2A",
    "CDKN2B", "MIA3", "SH2B3", "ZC3HC1", "WDR12", "MRAS", "CYP17A1",
    # Hypertension / blood pressure
    "ACE", "AGT", "AGTR1", "REN", "NR3C2", "SCNN1A", "SCNN1B", "SCNN1G",
    "WNK1", "WNK4", "KLHL3", "CUL3", "HSD11B2", "CYP11B1", "CYP11B2",
    # Sudden cardiac death / inherited arrhythmia
    "HCN4", "GPD1L", "SCN1B", "SCN3B", "SCN4B", "SCN10A", "GJA1",
}

# Cardiovascular trait keywords for filtering WS6 direction-of-effect output.
# Used to drop musculoskeletal and other non-cardiovascular GWAS signals
# that get tagged along when querying Open Targets with cardiac EFO terms.
CARDIAC_TRAIT_KEYWORDS = [
    # Cardiac structure/function
    "heart", "cardiac", "cardio", "myocardi", "ventric", "atria", "coron",
    "aort", "valve", "mitral", "tricuspid", "septal",
    # Rhythm
    "atrial fibrillation", "arrhythm", "sudden cardiac", "QT", "brugada",
    "tachy", "bradycardi", "wolff",
    # Vascular
    "blood pressure", "hypertens", "hypotens", "vascul", "arteri", "vein",
    "thromb", "stroke", "ischemi", "infarct",
    # Lipids
    "cholesterol", "ldl", "hdl", "lipid", "triglycerid", "lipoprotein",
    "apoa", "apob",
    # Metabolic CV risk factors that ARE cardiac-relevant
    "diabet", "glucose", "obesity", "bmi", "waist",
    # Heart disease umbrella terms
    "cad", "chd", "ihd", "cardiovascular", "circulation",
    # Pulmonary vascular
    "pulmonary hypertension", "pulmonary arterial",
]

# Trait keywords that should NEVER appear in cardiac results (drop these)
NON_CARDIAC_TRAIT_KEYWORDS = [
    "bone mineral density", "bmd ", "heel bone", "osteoporo", "fracture",
    "calcium levels", "vitamin d", "magnesium", "phosphate",
    "kidney function", "egfr", "urate",
    "alzheimer", "parkinson", "schizophren", "depression", "bipolar",
    "asthma", "lung function", "fev1",
    "skin", "hair", "balding", "eczema",
    "intelligence", "education", "income",
    "cancer", "tumor", "neoplasm", "leukemia", "lymphoma",
]


# Per-gene ISM scorer overrides. Default DNASE always runs; the listed extra
# scorers are run in addition for the named genes. Use TF or Histone for
# vignette candidates whose composite score is dominated by transcription-
# factor or histone-mark modalities rather than chromatin accessibility.
ISM_EXTRA_SCORERS = {
    "DUSP1":  ["CHIP_TF", "CHIP_HISTONE"],
    "CAVIN1": ["CHIP_TF", "CHIP_HISTONE"],
}


def is_cardiac_trait(trait: str) -> bool:
    """Check if a GWAS trait string is cardiovascular.

    Returns True if the trait matches a cardiac keyword and does NOT match
    any explicit non-cardiac exclusion.
    """
    if not trait or str(trait).lower() in ("nan", "none", ""):
        return False
    t = str(trait).lower()
    # Hard exclusions first
    for kw in NON_CARDIAC_TRAIT_KEYWORDS:
        if kw in t:
            return False
    # Then cardiac keywords
    for kw in CARDIAC_TRAIT_KEYWORDS:
        if kw in t:
            return True
    return False


def resolve_gene_name(row, merged_variants_lookup: dict | None = None) -> str:
    """Resolve a gene name from a variant summary row using the full fallback chain.

    Tries in order:
      1. expression_top_gene (from GeneMaskLFCScorer top hit)
      2. splicing_top_gene (from splicing scorers)
      3. accessibility_top_gene
      4. tf_binding_top_gene
      5. clinvar_gene (from WS1A metadata)
      6. l2g_gene (from Open Targets WS1B metadata)
      7. merged_variants lookup (if provided, for any other gene field)
      8. "?" if nothing resolves

    Filters out "nan", "None", "", and ENSG IDs unless nothing else is available.
    Returns a clean string.
    """
    fallback_cols = [
        "expression_top_gene",
        "splicing_top_gene",
        "accessibility_top_gene",
        "tf_binding_top_gene",
        "histone_marks_top_gene",
        "clinvar_gene",
        "l2g_gene",
    ]

    # Try each in order, preferring HGNC symbols over ENSG IDs
    candidates_hgnc = []
    candidates_ensg = []

    for col in fallback_cols:
        try:
            val = row.get(col) if hasattr(row, "get") else row[col] if col in row else None
        except (KeyError, AttributeError):
            val = None
        if val is None:
            continue
        s = str(val).strip()
        if s in ("", "nan", "None", "?"):
            continue
        if s.startswith("ENSG") or s.startswith("ENST"):
            candidates_ensg.append(s)
        else:
            candidates_hgnc.append(s)

    # Prefer HGNC over ENSG
    if candidates_hgnc:
        return candidates_hgnc[0]
    if candidates_ensg:
        return candidates_ensg[0]
    return "?"


# ---------------------------------------------------------------------------
# 1. VIGNETTE CANDIDATE SELECTION
# ---------------------------------------------------------------------------

def select_vignette_candidates(summary: pd.DataFrame,
                               n_candidates: int = 5) -> pd.DataFrame:
    """Select the best variants for mechanistic deep-dive vignettes.

    Prioritizes:
    1. Variants in known cardiac panel genes (FBN1, KCNQ1, SCN5A, etc.)
    2. Highest composite score
    3. Multimodal (high n_modalities_strong)
    4. SNVs preferred over indels (cleaner ISM)
    5. HGNC symbols over ENSG IDs (publishable gene names)

    A variant gets a large bonus if its resolved gene is in the cardiac panel.
    Within panel genes, ranks by composite score. If fewer than n_candidates
    panel genes are found, fills the remaining slots from top non-panel hits.
    """
    df = summary.copy()

    # Resolve gene name for every variant using the full fallback chain
    df["resolved_gene"] = df.apply(lambda r: resolve_gene_name(r), axis=1)

    # Flag variants in the cardiac panel
    df["in_cardiac_panel"] = df["resolved_gene"].isin(CARDIAC_GENE_PANEL)

    # Flag variants with a real HGNC gene name (not "?" or ENSG)
    df["has_named_gene"] = df["resolved_gene"].apply(
        lambda g: g != "?" and not g.startswith("ENSG") and not g.startswith("ENST")
    )

    # Prefer SNVs for ISM (shorter variant_id = simpler variant)
    df["vid_len"] = df["variant_id"].str.len()
    df["is_snv"] = df["vid_len"] < 30

    # Composite-based ranking; panel membership handled by slot reservation below.
    df["vignette_score"] = (
        df["composite_score"] * 1.0
        + df["n_modalities_strong"] * 2.0
        + df["has_named_gene"].astype(float) * 20.0
        + df["is_snv"].astype(float) * 5.0
    )
    df = df.sort_values("vignette_score", ascending=False)

    # Slot reservation strategy: dedicate ~60% of slots to cardiac-panel hits,
    # remainder to top non-panel hits. This guarantees panel representation
    # even when non-panel composite scores are much higher.
    # For panel genes, allow up to MAX_PER_PANEL_GENE variants per gene (cluster
    # vignettes — useful when one panel gene has multiple co-localized variants,
    # e.g. an LMNA promoter VUS cluster).
    MAX_PER_PANEL_GENE = 3
    n_panel_slots = max(1, (n_candidates * 3 + 4) // 5)  # ceil(n*0.6)

    panel_df = df[df["in_cardiac_panel"]]
    other_df = df[~df["in_cardiac_panel"]]

    selected = []
    panel_gene_counts = {}   # gene -> count, panel pass only (allows clusters)
    seen_other_genes = set()  # non-panel pass enforces strict diversity

    # Pass 1: fill panel slots, allowing up to MAX_PER_PANEL_GENE per gene
    for _, row in panel_df.iterrows():
        gene = row["resolved_gene"]
        if panel_gene_counts.get(gene, 0) >= MAX_PER_PANEL_GENE:
            continue
        selected.append(row)
        panel_gene_counts[gene] = panel_gene_counts.get(gene, 0) + 1
        if len(selected) >= n_panel_slots:
            break

    # Pass 2: fill remaining slots from non-panel with strict gene diversity
    seen_other_genes = set(panel_gene_counts.keys())  # don't repeat panel genes
    for _, row in other_df.iterrows():
        gene = row["resolved_gene"]
        if gene in seen_other_genes:
            continue
        selected.append(row)
        seen_other_genes.add(gene)
        if len(selected) >= n_candidates:
            break

    # Pass 3: fallback — if still short, allow any remaining variant
    if len(selected) < n_candidates:
        existing_vids = {s["variant_id"] for s in selected}
        for _, row in df.iterrows():
            if row["variant_id"] not in existing_vids:
                selected.append(row)
                existing_vids.add(row["variant_id"])
                if len(selected) >= n_candidates:
                    break

    result = pd.DataFrame(selected)

    # Print breakdown for transparency
    if not result.empty:
        n_panel = result["in_cardiac_panel"].sum()
        n_named = result["has_named_gene"].sum()
        print(f"  Vignette breakdown: {n_panel}/{len(result)} in cardiac panel "
              f"({n_panel_slots} reserved, max {MAX_PER_PANEL_GENE}/gene), "
              f"{n_named}/{len(result)} with HGNC names")

    return result


# ---------------------------------------------------------------------------
# 2. WS5: ISM VIGNETTES
# ---------------------------------------------------------------------------

# Map scorer name → (OutputType, ylabel suffix). Add more as needed.
# Defined as a function to avoid module-load failure when alphagenome isn't installed.
def _ism_scorer_presets():
    return {
        "DNASE":        (dna_client.OutputType.DNASE,        "DNase"),
        "CHIP_TF":      (dna_client.OutputType.CHIP_TF,      "TF"),
        "CHIP_HISTONE": (dna_client.OutputType.CHIP_HISTONE, "Histone"),
    }


def run_ism_vignette(client, variant_str: str, gene_name: str,
                     output_dir: Path, ism_width: int = 128,
                     scorer_name: str = "DNASE") -> dict:
    """Run ISM analysis for a single variant and generate plots.

    Args:
        client: AlphaGenome DnaClient
        variant_str: Variant in chr:pos:ref>alt format
        gene_name: Target gene name for labeling
        output_dir: Where to save plots and data
        ism_width: Width of ISM window around variant (bp)
        scorer_name: One of "DNASE", "CHIP_TF", "CHIP_HISTONE".
            Use TF or Histone for variants whose composite score is dominated
            by transcription-factor or histone-mark modalities rather than
            chromatin accessibility.

    Returns:
        dict with ISM results metadata
    """
    presets = _ism_scorer_presets()
    if scorer_name not in presets:
        raise ValueError(f"Unknown scorer {scorer_name}; "
                         f"choose from {list(presets)}")
    output_type, scorer_label = presets[scorer_name]

    variant = genome.Variant.from_str(variant_str)
    sequence_interval = variant.reference_interval.resize(
        dna_client.SEQUENCE_LENGTH_16KB
    )
    ism_interval = variant.reference_interval.resize(ism_width)

    scorer = variant_scorers.CenterMaskScorer(
        requested_output=output_type,
        width=501,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
    )

    print(f"    Running ISM ({scorer_name}, {ism_width}bp window, "
          f"{ism_width * 3} variants)...", end="", flush=True)

    ism_scores = client.score_ism_variants(
        interval=sequence_interval,
        ism_interval=ism_interval,
        variant_scorers=[scorer],
    )
    print(f" done ({len(ism_scores)} variants scored)")

    def extract_mean(adata):
        return float(adata.X.mean())

    ism_result = ism.ism_matrix(
        [extract_mean(x[0]) for x in ism_scores],
        variants=[x[0].uns["variant"] for x in ism_scores],
    )

    # Filename includes scorer suffix so multiple scorers per variant don't overwrite
    safe_name = (variant_str.replace(":", "_").replace(">", "_")[:50]
                 + f"_{scorer_name}")
    np.save(output_dir / f"ism_{safe_name}.npy", ism_result)

    result = {
        "variant": variant_str,
        "gene": gene_name,
        "scorer": scorer_name,
        "ism_interval": str(ism_interval),
        "ism_shape": list(ism_result.shape),
        "max_ism_score": float(np.abs(ism_result).max()),
    }

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(20, 3))
        try:
            plot_components.plot(
                [plot_components.SeqLogo(
                    scores=ism_result,
                    scores_interval=ism_interval,
                    ylabel=f"ISM {scorer_label}\n{gene_name}",
                )],
                interval=ism_interval,
                annotations=[plot_components.VariantAnnotation(
                    [variant], alpha=0.8
                )],
                fig_width=20,
            )
            plt.savefig(output_dir / f"ism_logo_{safe_name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            result["plot"] = f"ism_logo_{safe_name}.png"
        except Exception as e:
            plt.close()
            result["plot_error"] = str(e)

    # REF vs ALT track plot — only for the DNase pass to avoid duplicate work
    # (the RNA_SEQ output is the same regardless of which ISM scorer ran).
    if scorer_name == "DNASE":
        try:
            full_interval = variant.reference_interval.resize(
                dna_client.SEQUENCE_LENGTH_1MB
            )
            variant_output = client.predict_variant(
                interval=full_interval,
                variant=variant,
                requested_outputs=[dna_client.OutputType.RNA_SEQ],
                ontology_terms=["UBERON:0006566"],  # Heart Left Ventricle
            )

            if HAS_MPL and variant_output:
                plot_components.plot(
                    [plot_components.OverlaidTracks(
                        tdata={
                            "REF": variant_output.reference.rna_seq,
                            "ALT": variant_output.alternate.rna_seq,
                        },
                        colors={"REF": "dimgrey", "ALT": "red"},
                    )],
                    interval=variant_output.reference.rna_seq.interval.resize(2**15),
                    annotations=[plot_components.VariantAnnotation(
                        [variant], alpha=0.8
                    )],
                )
                # Strip _DNASE suffix for ref_alt filename — there's only one per variant
                base_name = safe_name.rsplit("_DNASE", 1)[0]
                plt.savefig(output_dir / f"ref_alt_{base_name}.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
                result["ref_alt_plot"] = f"ref_alt_{base_name}.png"
        except Exception as e:
            result["ref_alt_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# 3. WS6: GWAS DIRECTION-OF-EFFECT
# ---------------------------------------------------------------------------

def compute_direction_of_effect(cardiac_scores,
                                merged_variants: pd.DataFrame) -> pd.DataFrame:
    """For GWAS variants, determine the predicted direction of effect
    on cardiac gene expression.

    Uses the signed RNA_SEQ scores (GeneMaskLFCScorer) to predict whether
    a variant increases or decreases expression of nearby genes in cardiac tissue.
    """
    # Filter to GWAS variants only
    gwas_vids = set(
        merged_variants[merged_variants["in_gwas"] == True]["variant_id"]
    )

    # Stream the cardiac parquet to accumulate per-(variant, gene) sign statistics.
    # Accumulator pattern avoids loading 276M rows into memory.
    from pathlib import Path as _Path
    is_path_based = isinstance(cardiac_scores, _Path) or (
        isinstance(cardiac_scores, str) and cardiac_scores.endswith(".parquet")
    )

    # Pre-build a fast metadata lookup dict.
    # The previous version did a linear scan over merged_variants for every
    # variant-gene pair, which is O(n*m) and catastrophically slow at scale
    # (1M pairs × 60K variants = 60 billion comparisons). This makes it O(n+m).
    print("  Building metadata lookup index...")
    metadata_lookup = {}

    def _clean_str(val):
        """Convert pandas NaN/None to empty string, otherwise to str."""
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        s = str(val)
        return "" if s in ("nan", "None", "NaN") else s

    def _clean_float(val):
        """Convert pandas NaN/None to 0.0, otherwise to float."""
        if val is None:
            return 0.0
        try:
            if pd.isna(val):
                return 0.0
        except (TypeError, ValueError):
            pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    for _, mrow in merged_variants.iterrows():
        vid = mrow["variant_id"]
        if vid not in metadata_lookup:
            metadata_lookup[vid] = {
                "gwas_trait": _clean_str(mrow.get("gwas_trait")),
                "max_pip": _clean_float(mrow.get("max_pip")),
                "l2g_gene": _clean_str(mrow.get("l2g_gene")),
            }
    print(f"  Indexed {len(metadata_lookup):,} variants for O(1) lookup")

    # Accumulator: (variant_id, gene_name) -> list of raw scores
    pair_scores = {}

    if is_path_based:
        pf = _pq.ParquetFile(str(cardiac_scores))
        schema_cols = set(pf.schema_arrow.names)
        needed = [c for c in ["variant_id", "output_type", "gene_name", "raw_score", "variant_scorer"]
                  if c in schema_cols]
        has_scorer_col = "variant_scorer" in needed
        total = pf.metadata.num_rows
        batch_size = 500_000
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Streaming {total_batches} batches for GWAS direction-of-effect...")

        bn = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=needed):
            bn += 1
            chunk = batch.to_pandas()
            chunk = chunk[(chunk["variant_id"].isin(gwas_vids)) &
                          (chunk["output_type"] == "RNA_SEQ")]
            # Filter to GeneMaskLFCScorer only (signed). Exclude GeneMaskActiveScorer
            # which is unsigned and would bias everything positive.
            if has_scorer_col and not chunk.empty:
                chunk = chunk[chunk["variant_scorer"].astype(str).str.contains(
                    "LFCScorer", case=False, na=False
                )]
            if chunk.empty:
                del chunk; continue
            for vid, gene_id, raw in zip(chunk["variant_id"].values,
                                         chunk["gene_name"].values,
                                         chunk["raw_score"].values):
                if gene_id is None or str(gene_id) in ("None", "nan", ""):
                    continue
                key = (vid, gene_id)
                if key not in pair_scores:
                    pair_scores[key] = []
                pair_scores[key].append(float(raw))
            if bn % 50 == 0:
                print(f"    Batch {bn}/{total_batches} | tracking {len(pair_scores):,} pairs")
            del chunk
    else:
        mask = ((cardiac_scores["variant_id"].isin(gwas_vids)) &
                (cardiac_scores["output_type"] == "RNA_SEQ"))
        if "variant_scorer" in cardiac_scores.columns:
            mask &= cardiac_scores["variant_scorer"].astype(str).str.contains(
                "LFCScorer", case=False, na=False
            )
        expr_scores = cardiac_scores[mask]
        for (vid, gene_id), group in expr_scores.groupby(["variant_id", "gene_name"]):
            if str(gene_id) in ("None", "nan", ""):
                continue
            pair_scores[(vid, gene_id)] = group["raw_score"].tolist()

    if not pair_scores:
        print("  No GWAS variants with cardiac expression scores found.")
        return pd.DataFrame()

    print(f"  Computing direction calls for {len(pair_scores):,} variant-gene pairs...")
    # For each variant-gene pair, determine sign
    results = []
    for (vid, gene_id), raw_list in pair_scores.items():
        raw_scores = np.asarray(raw_list, dtype=float)
        mean_score = float(np.mean(raw_scores))
        max_abs_score = float(np.max(np.abs(raw_scores)))

        # Determine confident direction
        # Threshold: mean score must be consistently signed across tissues
        positive_frac = (raw_scores > 0).mean()

        # Use mean sign to determine direction (avoids symmetric-threshold failure
        # on asymmetric score distributions). Use positive_frac for confidence.
        if max_abs_score <= 0.01:
            direction = "UNCERTAIN"
            confidence = "low"
        elif mean_score > 0:
            direction = "UP"
            if positive_frac >= 0.9:
                confidence = "high"
            elif positive_frac >= 0.7:
                confidence = "moderate"
            else:
                confidence = "low"
        elif mean_score < 0:
            direction = "DOWN"
            if positive_frac <= 0.1:
                confidence = "high"
            elif positive_frac <= 0.3:
                confidence = "moderate"
            else:
                confidence = "low"
        else:
            direction = "UNCERTAIN"
            confidence = "low"

        # O(1) metadata lookup
        meta = metadata_lookup.get(vid, {})
        trait_str = meta.get("gwas_trait", "")

        results.append({
            "variant_id": vid,
            "gene_name": gene_id,
            "mean_cardiac_lfc": mean_score,
            "max_abs_cardiac_lfc": max_abs_score,
            "direction": direction,
            "confidence": confidence,
            "positive_tissue_fraction": positive_frac,
            "n_cardiac_tissues": len(raw_scores),
            "gwas_trait": trait_str,
            "is_cardiac_trait": is_cardiac_trait(trait_str),
            "max_pip": meta.get("max_pip", 0),
            "l2g_gene": meta.get("l2g_gene", ""),
        })

    doe_df = pd.DataFrame(results)
    if not doe_df.empty:
        doe_df = doe_df.sort_values("max_abs_cardiac_lfc", ascending=False)

    return doe_df


# ---------------------------------------------------------------------------
# 4. WS7: DELIVERABLES
# ---------------------------------------------------------------------------

def generate_ranked_table(summary: pd.DataFrame,
                          merged_variants: pd.DataFrame,
                          output_path: Path) -> None:
    """Generate the publication-ready ranked variant table (WS7A)."""

    cols = ["variant_id"]

    # Source info
    for c in ["chromosome", "position", "in_clinvar", "in_gwas",
              "clinical_significance", "clinvar_condition", "variant_location"]:
        if c in summary.columns:
            cols.append(c)

    # GWAS info
    for c in ["max_pip", "gwas_trait", "l2g_gene", "l2g_score"]:
        if c in summary.columns:
            cols.append(c)

    # Scores
    cols.append("composite_score")
    cols.append("n_modalities_strong")

    for mod in ["expression", "splicing", "accessibility", "tf_binding",
                "histone_marks", "3d_structure"]:
        for suffix in ["_max_raw", "_top_gene", "_top_tissue"]:
            c = f"{mod}{suffix}"
            if c in summary.columns:
                cols.append(c)

    available = [c for c in cols if c in summary.columns]
    out = summary[available].copy()
    out = out.sort_values("composite_score", ascending=False)
    out.to_csv(output_path, sep="\t", index=False)


def generate_summary_report(output_dir: Path,
                            summary: pd.DataFrame,
                            multimodal: pd.DataFrame,
                            doe: pd.DataFrame | None,
                            vignettes: list[dict]) -> None:
    """Generate a markdown summary report (WS7B — manuscript seed)."""

    report_path = output_dir / "REPORT.md"
    n_variants = len(summary)
    n_multimodal = len(multimodal) if multimodal is not None else 0

    with open(report_path, "w") as f:
        f.write("# AlphaGenome Cardiovascular Non-Coding Variant Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Variants scored:** {n_variants}\n")
        f.write(f"- **Multimodal hits (2+ modalities):** {n_multimodal} "
                f"({n_multimodal/n_variants*100:.0f}%)\n")

        if "in_clinvar" in summary.columns:
            n_cv = summary["in_clinvar"].sum()
            n_gw = summary["in_gwas"].sum() if "in_gwas" in summary.columns else 0
            n_both = ((summary.get("in_clinvar", False) == True) &
                      (summary.get("in_gwas", False) == True)).sum()
            f.write(f"- **ClinVar variants:** {n_cv}\n")
            f.write(f"- **GWAS variants:** {n_gw}\n")
            f.write(f"- **Dual-source (ClinVar + GWAS):** {n_both}\n")

        f.write("\n## Top 10 Variants by Cardiac Composite Score\n\n")
        f.write("| Rank | Variant | Gene | Composite | Modalities | ClinSig | Trait |\n")
        f.write("|------|---------|------|-----------|------------|---------|-------|\n")
        for i, (_, row) in enumerate(summary.head(10).iterrows(), 1):
            vid = str(row["variant_id"])[:40]
            gene = resolve_gene_name(row)
            comp = f"{row['composite_score']:.2f}"
            mods = str(row["n_modalities_strong"])
            csig_raw = row.get("clinical_significance", "")
            csig = "" if str(csig_raw) in ("nan", "None", "") else str(csig_raw)
            trait_raw = row.get("gwas_trait", "")
            trait = "" if str(trait_raw) in ("nan", "None", "") else str(trait_raw)[:30]
            f.write(f"| {i} | {vid} | {gene} | {comp} | {mods} | {csig} | {trait} |\n")

        if vignettes:
            f.write("\n## Mechanistic Vignettes (ISM Analysis)\n\n")
            for vig in vignettes:
                f.write(f"### {vig['gene']} — {vig['variant']}\n\n")
                f.write(f"- ISM window: {vig['ism_interval']}\n")
                f.write(f"- Max ISM score: {vig['max_ism_score']:.4f}\n")
                if "plot" in vig:
                    f.write(f"- Sequence logo: `{vig['plot']}`\n")
                if "ref_alt_plot" in vig:
                    f.write(f"- REF vs ALT: `{vig['ref_alt_plot']}`\n")
                f.write("\n")

        if doe is not None and not doe.empty:
            f.write("\n## GWAS Direction-of-Effect (Cardiac Traits Only)\n\n")
            f.write("*Filtered to cardiovascular traits — non-cardiac contamination "
                    "(BMD, calcium, kidney function, etc.) excluded.*\n\n")
            n_up = (doe["direction"] == "UP").sum()
            n_down = (doe["direction"] == "DOWN").sum()
            n_unc = (doe["direction"] == "UNCERTAIN").sum()
            n_conf = ((doe["direction"] != "UNCERTAIN") &
                      (doe["confidence"] == "high")).sum()
            f.write(f"- Variant-gene pairs analyzed: {len(doe):,}\n")
            f.write(f"- Upregulated: {n_up:,}\n")
            f.write(f"- Downregulated: {n_down:,}\n")
            f.write(f"- Uncertain: {n_unc:,}\n")
            f.write(f"- High-confidence predictions: {n_conf:,}\n\n")

            f.write("| Variant | Gene | Direction | LFC | Confidence | Trait |\n")
            f.write("|---------|------|-----------|-----|------------|-------|\n")
            for _, row in doe[doe["confidence"] == "high"].head(10).iterrows():
                trait = str(row.get('gwas_trait', ''))
                trait = "" if trait in ("nan", "None") else trait[:25]
                f.write(f"| {row['variant_id'][:35]} | {row['gene_name']} | "
                        f"{row['direction']} | {row['mean_cardiac_lfc']:.3f} | "
                        f"{row['confidence']} | {trait} |\n")

        f.write("\n## Methods Summary\n\n")
        f.write("Variants were scored using AlphaGenome (Avsec et al., Nature 2026) "
                "with the full set of 19 recommended variant scorers spanning 11 "
                "modalities: gene expression (GeneMaskLFCScorer and GeneMaskActiveScorer "
                "on RNA-seq), splicing (SpliceJunctionScorer plus GeneMaskSplicingScorer "
                "for both splice sites and splice site usage), chromatin accessibility "
                "(CenterMaskScorer with both DIFF_LOG2_SUM and ACTIVE_SUM aggregations "
                "for DNase-seq and ATAC-seq), transcription factor binding (CenterMaskScorer "
                "for ChIP-TF, both directional and active), histone modifications "
                "(CenterMaskScorer for ChIP-histone, both aggregations), CAGE, PRO-cap, "
                "polyadenylation (PolyadenylationScorer), and 3D chromatin contact maps "
                "(ContactMapScorer). Each variant was scored against a 1,048,576 bp "
                "(2^20) context window centered on the variant. Scores were filtered to "
                "cardiac-relevant GTEx tissues (Heart_Left_Ventricle, Heart_Atrial_Appendage, "
                "Artery_Aorta, Artery_Coronary, Artery_Tibial) and cardiac ENCODE biosamples "
                "(cardiomyocytes, cardiac smooth muscle, vascular endothelium). Multimodal "
                "hits were defined as variants exceeding the 95th quantile of background "
                "common variant effects in 2 or more modality groups simultaneously. "
                "Direction-of-effect calls for GWAS variants were derived from signed "
                "GeneMaskLFCScorer outputs requiring >90%% agreement across cardiac tissue "
                "tracks for high-confidence calls. GWAS results were further filtered to "
                "cardiovascular traits to exclude tag-along musculoskeletal and metabolic "
                "phenotypes from broad EFO term queries. In silico mutagenesis (ISM) "
                "vignettes used a 128 bp window centered on each variant and the DNase "
                "scorer to identify disrupted transcription factor binding motifs. "
                "Vignette candidates were preferentially selected from a curated panel "
                "of cardiovascular disease genes covering cardiomyopathies, channelopathies, "
                "aortopathies, lipid metabolism, congenital heart disease, atrial fibrillation, "
                "coronary artery disease, and hypertension.\n")

    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(
    variant_summary_path: str,
    cardiac_scores_path: str,
    merged_variants_path: str,
    output_dir: str,
    api_key: str | None = None,
    skip_ism: bool = False,
    n_vignettes: int = 5,
    ism_width: int = 128,
) -> None:
    """Run WS5 + WS6 + WS7."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vignettes").mkdir(exist_ok=True)

    print("=" * 70)
    print("WORKSTREAMS 5-7: VIGNETTES, DIRECTION-OF-EFFECT & DELIVERABLES")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading data...")
    summary = pd.read_csv(variant_summary_path, sep="\t", low_memory=False)
    print(f"  Variant summary: {len(summary):,} variants")

    # DON'T load 276M rows into memory — pass the path for streaming
    from pathlib import Path as _Path
    cardiac_scores = _Path(cardiac_scores_path) if cardiac_scores_path.endswith(".parquet") else cardiac_scores_path
    if cardiac_scores_path.endswith(".parquet"):
        _total = _pq.ParquetFile(cardiac_scores).metadata.num_rows
        print(f"  Cardiac scores parquet: {_total:,} rows (streaming, not loaded)")
    else:
        cardiac_scores = pd.read_csv(cardiac_scores_path, sep="\t", low_memory=False)
        print(f"  Cardiac scores: {len(cardiac_scores):,} rows (loaded)")

    merged_variants = pd.read_csv(merged_variants_path, sep="\t",
                                  low_memory=False, dtype={"chromosome": str})
    print(f"  Merged variants: {len(merged_variants):,} records")

    # =====================================================================
    # WS5: MECHANISTIC VIGNETTES
    # =====================================================================
    print("\n" + "=" * 70)
    print("WORKSTREAM 5: MECHANISTIC VIGNETTES")
    print("=" * 70)

    candidates = select_vignette_candidates(summary, n_vignettes)
    print(f"\nSelected {len(candidates)} vignette candidates:")
    for _, row in candidates.iterrows():
        gene = resolve_gene_name(row)
        in_panel = "*" if row.get("in_cardiac_panel", False) else " "
        print(f"  {in_panel} {row['variant_id'][:50]:50s} gene={gene:12s} "
              f"composite={row['composite_score']:.2f}  "
              f"mods={row['n_modalities_strong']}")
    print("  (* = in cardiac gene panel)")

    # Save candidates
    candidates.to_csv(output_dir / "vignette_candidates.tsv",
                      sep="\t", index=False)

    vignette_results = []

    if skip_ism or not HAS_AG:
        print(f"\n  {'Skipping ISM (--skip-ism)' if skip_ism else 'AlphaGenome not available'}")
        print("  Vignette plots will be generated when ISM is run.")
        for _, row in candidates.iterrows():
            vignette_results.append({
                "variant": row["variant_id"],
                "gene": resolve_gene_name(row),
                "in_cardiac_panel": bool(row.get("in_cardiac_panel", False)),
                "ism_interval": "pending",
                "ism_shape": [],
                "max_ism_score": 0,
            })
    else:
        # Initialize client
        key = api_key or os.environ.get("ALPHAGENOME_API_KEY", "")
        if not key:
            key_file = Path.home() / ".alphagenome" / "api_key.txt"
            if key_file.exists():
                key = key_file.read_text().strip()
        if not key:
            print("  WARNING: No API key — skipping ISM")
        else:
            client = dna_client.create(key)
            print(f"\n  Running ISM for {len(candidates)} variants "
                  f"({ism_width}bp windows)...")

            vignette_dir = output_dir / "vignettes"
            for i, (_, row) in enumerate(candidates.iterrows(), 1):
                vid = row["variant_id"]
                gene = resolve_gene_name(row)
                in_panel = bool(row.get("in_cardiac_panel", False))
                print(f"\n  Vignette {i}/{len(candidates)}: {gene} — {vid[:50]}")

                # Always run default DNase ISM
                try:
                    result = run_ism_vignette(
                        client, vid, gene, vignette_dir,
                        ism_width=ism_width, scorer_name="DNASE",
                    )
                    result["in_cardiac_panel"] = in_panel
                    vignette_results.append(result)
                except Exception as e:
                    print(f"    FAILED (DNASE): {e}")
                    vignette_results.append({
                        "variant": vid, "gene": gene, "scorer": "DNASE",
                        "in_cardiac_panel": in_panel,
                        "ism_interval": "failed", "ism_shape": [],
                        "max_ism_score": 0, "error": str(e),
                    })

                # Optional extra scorers for genes where DNase signal is weak
                for extra in ISM_EXTRA_SCORERS.get(gene, []):
                    print(f"  → Extra ISM ({extra}) for {gene}")
                    try:
                        extra_result = run_ism_vignette(
                            client, vid, gene, vignette_dir,
                            ism_width=ism_width, scorer_name=extra,
                        )
                        extra_result["in_cardiac_panel"] = in_panel
                        vignette_results.append(extra_result)
                    except Exception as e:
                        print(f"    FAILED ({extra}): {e}")
                        vignette_results.append({
                            "variant": vid, "gene": gene, "scorer": extra,
                            "in_cardiac_panel": in_panel,
                            "ism_interval": "failed", "ism_shape": [],
                            "max_ism_score": 0, "error": str(e),
                        })

    # Save vignette metadata
    with open(output_dir / "vignette_results.json", "w") as f:
        json.dump(vignette_results, f, indent=2, default=str)

    # =====================================================================
    # WS6: GWAS DIRECTION-OF-EFFECT
    # =====================================================================
    print("\n" + "=" * 70)
    print("WORKSTREAM 6: GWAS DIRECTION-OF-EFFECT")
    print("=" * 70)

    doe_df = compute_direction_of_effect(cardiac_scores, merged_variants)

    if not doe_df.empty:
        doe_path = output_dir / "gwas_direction_of_effect.tsv"
        doe_df.to_csv(doe_path, sep="\t", index=False)
        print(f"\n  Variant-gene pairs (all traits): {len(doe_df):,}")

        # Filter to cardiac traits only — this is the publication-ready subset
        doe_cardiac = doe_df[doe_df["is_cardiac_trait"] == True].copy()
        doe_cardiac_path = output_dir / "gwas_direction_of_effect_cardiac.tsv"
        doe_cardiac.to_csv(doe_cardiac_path, sep="\t", index=False)

        n_cardiac = len(doe_cardiac)
        n_dropped = len(doe_df) - n_cardiac
        print(f"  Cardiac-trait pairs: {n_cardiac:,} "
              f"({n_cardiac/len(doe_df)*100:.0f}%)")
        print(f"  Dropped non-cardiac (BMD/calcium/etc): {n_dropped:,}")

        # Direction stats on cardiac subset
        n_resolved = (doe_cardiac["direction"] != "UNCERTAIN").sum()
        print(f"\n  Cardiac direction resolved: {n_resolved:,} "
              f"({n_resolved/max(n_cardiac,1)*100:.0f}%)")
        print(f"    UP:   {(doe_cardiac['direction'] == 'UP').sum():,}")
        print(f"    DOWN: {(doe_cardiac['direction'] == 'DOWN').sum():,}")

        n_high = (doe_cardiac["confidence"] == "high").sum()
        print(f"  High confidence (cardiac only): {n_high:,}")

        if n_resolved > 0:
            print(f"\n  Top confident cardiac direction-of-effect predictions:")
            top_doe = doe_cardiac[doe_cardiac["confidence"] == "high"].head(10)
            for _, row in top_doe.iterrows():
                print(f"    {row['variant_id'][:40]:40s} → {row['gene_name']:15s} "
                      f"{row['direction']:4s} (LFC={row['mean_cardiac_lfc']:.3f}) "
                      f"trait={str(row.get('gwas_trait', ''))[:35]}")

        # Use cardiac-filtered version for the report
        doe_for_report = doe_cardiac
    else:
        print("  No GWAS variants in current scored set (run full WS2 first).")
        doe_df = None
        doe_for_report = None

    # =====================================================================
    # WS7: DELIVERABLES
    # =====================================================================
    print("\n" + "=" * 70)
    print("WORKSTREAM 7: DELIVERABLES")
    print("=" * 70)

    # 7A: Ranked variant table
    ranked_path = output_dir / "ranked_variant_table.tsv"
    generate_ranked_table(summary, merged_variants, ranked_path)
    print(f"\n  7A. Ranked variant table: {ranked_path}")

    # 7B: Summary report (manuscript seed)
    multimodal = summary[summary["n_modalities_strong"] >= 2]
    generate_summary_report(output_dir, summary, multimodal,
                            doe_for_report, vignette_results)
    print(f"  7B. Summary report: {output_dir / 'REPORT.md'}")

    # 7C: CRF pitch data package
    pitch_dir = output_dir / "crf_pitch"
    pitch_dir.mkdir(exist_ok=True)

    # Top 20 variants for pitch
    top20 = summary.nlargest(20, "composite_score")
    top20.to_csv(pitch_dir / "top20_cardiac_variants.tsv", sep="\t", index=False)

    # Key stats for pitch deck
    pitch_stats = {
        "total_variants_scored": int(len(summary)),
        "multimodal_hits": int(len(multimodal)),
        "multimodal_pct": round(len(multimodal) / len(summary) * 100, 1),
        "top_gene": resolve_gene_name(summary.iloc[0]),
        "top_variant": str(summary.iloc[0]["variant_id"])[:50],
        "top_composite_score": round(float(summary.iloc[0]["composite_score"]), 2),
        "n_vignettes": len([v for v in vignette_results if v.get("max_ism_score", 0) > 0]),
        "n_vignettes_in_cardiac_panel": len([v for v in vignette_results if v.get("in_cardiac_panel", False)]),
        "direction_of_effect_resolved_all": int((doe_df["direction"] != "UNCERTAIN").sum()) if doe_df is not None else 0,
        "direction_of_effect_resolved_cardiac": int((doe_for_report["direction"] != "UNCERTAIN").sum()) if doe_for_report is not None else 0,
        "direction_of_effect_high_confidence_cardiac": int((doe_for_report["confidence"] == "high").sum()) if doe_for_report is not None else 0,
    }
    with open(pitch_dir / "pitch_stats.json", "w") as f:
        json.dump(pitch_stats, f, indent=2)

    print(f"  7C. CRF pitch package: {pitch_dir}/")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("ALL WORKSTREAMS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nDeliverables:")
    print(f"  ranked_variant_table.tsv    — Full ranked list for publication Table 1")
    print(f"  REPORT.md                   — Manuscript seed with key findings")
    print(f"  vignette_candidates.tsv     — Selected variants for Fig deep-dives")
    print(f"  vignettes/                  — ISM logos and REF/ALT plots")
    if doe_df is not None:
        print(f"  gwas_direction_of_effect.tsv — Sign predictions for GWAS loci")
    print(f"  crf_pitch/                  — CRF collaboration pitch materials")
    print(f"\nNext: Run full WS2 scoring (61K variants), then re-run this script.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WS5-7: Vignettes, direction-of-effect, and deliverables.",
    )
    parser.add_argument("--variant-summary", required=True,
                        help="variant_summary.tsv from WS3")
    parser.add_argument("--cardiac-scores", required=True,
                        help="cardiac_scores.parquet from WS3")
    parser.add_argument("--merged-variants", required=True,
                        help="merged_cardio_variants.tsv from WS1C")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--api-key", default=None,
                        help="AlphaGenome API key")
    parser.add_argument("--skip-ism", action="store_true",
                        help="Skip ISM (WS5 API calls)")
    parser.add_argument("--n-vignettes", type=int, default=5)
    parser.add_argument("--ism-width", type=int, default=128,
                        help="ISM window width in bp (default: 128)")

    args = parser.parse_args()
    if args.api_key:
        os.environ["ALPHAGENOME_API_KEY"] = args.api_key

    run_pipeline(
        args.variant_summary, args.cardiac_scores, args.merged_variants,
        args.output_dir, args.api_key, args.skip_ism,
        args.n_vignettes, args.ism_width,
    )


if __name__ == "__main__":
    main()