#!/usr/bin/env python3
"""Workstream 3: Cardiac Tissue Filtering and Multimodal Enrichment.

Reads the tidy scores from Workstream 2, filters to cardiac-relevant tracks,
identifies top-scoring variants per modality, detects multimodal hits, and
produces ranked variant tables for the CRF pitch and publication.

Usage:
    python cardiac_filter_analysis.py \
        --scores /path/to/scoring_output/all_scores_tidy.parquet \
        --variants /path/to/merged_cardio_variants.tsv \
        --output-dir /path/to/ws3_output/

Outputs:
    cardiac_scores.parquet          — All scores filtered to cardiac tracks
    variant_summary.tsv             — One row per variant: max scores per modality
    multimodal_hits.tsv             — Variants scoring high in 2+ modalities
    top_expression_variants.tsv     — Top variants by cardiac expression effect
    top_splicing_variants.tsv       — Top variants by splicing disruption
    top_accessibility_variants.tsv  — Top variants by chromatin accessibility
    modality_enrichment.tsv         — Enrichment stats (like AlphaGenome Fig 6f)
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as _pa
import pyarrow.parquet as _pq


# ---------------------------------------------------------------------------
# 1. CARDIAC TISSUE DEFINITIONS
# ---------------------------------------------------------------------------
# GTEx tissue names and biosample names relevant to cardiovascular biology.
# These match the gtex_tissue and biosample_name columns in tidy_scores output.

CARDIAC_GTEX_TISSUES = {
    "Heart_Left_Ventricle",
    "Heart_Atrial_Appendage",
    "Artery_Aorta",
    "Artery_Coronary",
    "Artery_Tibial",
}

CARDIAC_BIOSAMPLE_KEYWORDS = [
    "heart", "cardiac", "cardiomyocyte", "ventricle", "atrial", "atrium",
    "aorta", "aortic", "artery", "arterial", "coronary",
    "endothelial",  # vascular endothelium
    "smooth muscle",  # vascular smooth muscle
    "myocardium",
]

# Combined regex pattern for vectorized filtering (much faster than looping)
CARDIAC_PATTERN = "|".join(CARDIAC_BIOSAMPLE_KEYWORDS)


# Broader vascular set for secondary analysis
VASCULAR_GTEX_TISSUES = CARDIAC_GTEX_TISSUES | {
    "Whole_Blood",
    "Cells_EBV-transformed_lymphocytes",
}

# Modality grouping based on output_type column
MODALITY_MAP = {
    "RNA_SEQ": "expression",
    "CAGE": "expression",
    "PROCAP": "expression",
    "SPLICE_JUNCTIONS": "splicing",
    "SPLICE_SITES": "splicing",
    "SPLICE_SITE_USAGE": "splicing",
    "DNASE": "accessibility",
    "ATAC": "accessibility",
    "CHIP_TF": "tf_binding",
    "CHIP_HISTONE": "histone_marks",
    "CONTACT_MAPS": "3d_structure",
}

# Canonical six modality groups used for composite scoring. This set is the
# authoritative definition of which per-(variant, modality) max-absolute-score
# columns are summed into composite_score. Output types that are not in
# MODALITY_MAP (e.g., polyadenylation/PA_QTL) fall into a separate "other"
# bucket and are EXCLUDED from the composite score to match the manuscript
# formula (§3.4: sum over six modality groups only).
COMPOSITE_MODALITIES = (
    "expression",
    "splicing",
    "accessibility",
    "tf_binding",
    "histone_marks",
    "3d_structure",
)


# ---------------------------------------------------------------------------
# 2. CARDIAC FILTERING
# ---------------------------------------------------------------------------

def stream_filter_parquet_to_cardiac(parquet_path: str, output_dir: Path,
                                      batch_size: int = 500_000) -> pd.DataFrame:
    """Stream a large Parquet file in chunks, filtering each chunk to cardiac
    tracks immediately, and writing the filtered result to disk.

    Returns the filtered DataFrame (now small enough to fit in memory).

    This avoids loading the full ~700M row Parquet file into memory.
    """
    pf = _pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    file_size_gb = Path(parquet_path).stat().st_size / 1e9
    print(f"  File size: {file_size_gb:.1f} GB on disk")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Streaming in batches of {batch_size:,}...")

    cardiac_path = output_dir / "cardiac_scores.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which columns are present (for biosample_name optionality)
    schema_cols = set(pf.schema_arrow.names)
    has_biosample = "biosample_name" in schema_cols
    has_gtex = "gtex_tissue" in schema_cols

    writer = None
    total_kept = 0
    total_processed = 0
    batch_num = 0

    # Project only the columns we need - cuts I/O and memory by ~60%
    needed_cols = [c for c in [
        "variant_id", "output_type", "gene_name", "gtex_tissue",
        "biosample_name", "raw_score", "quantile_score", "variant_scorer",
        "track_name", "biosample_type", "transcription_factor", "histone_mark"
    ] if c in schema_cols]

    for batch in pf.iter_batches(batch_size=batch_size, columns=needed_cols):
        batch_num += 1
        chunk = batch.to_pandas()
        total_processed += len(chunk)

        # Build cardiac mask vectorized
        gtex_mask = (chunk["gtex_tissue"].isin(CARDIAC_GTEX_TISSUES)
                     if has_gtex else pd.Series(False, index=chunk.index))

        biosample_mask = pd.Series(False, index=chunk.index)
        if has_biosample:
            biosample_mask = chunk["biosample_name"].fillna("").str.lower().str.contains(
                CARDIAC_PATTERN, na=False, regex=True
            )

        contact_mask = chunk["output_type"] == "CONTACT_MAPS"
        cardiac_mask = gtex_mask | biosample_mask | contact_mask
        kept = chunk[cardiac_mask]

        if len(kept) > 0:
            kept_table = pa_table_from_pandas(kept)
            if writer is None:
                writer = _pq.ParquetWriter(str(cardiac_path), kept_table.schema)
            writer.write_table(kept_table)
            total_kept += len(kept)
            del kept_table

        if batch_num % 10 == 0 or batch_num == 1:
            pct = total_processed / total_rows * 100
            print(f"    Batch {batch_num}: processed {total_processed:,}/{total_rows:,} "
                  f"({pct:.1f}%) | cardiac kept: {total_kept:,}")

        del chunk, kept, gtex_mask, biosample_mask, contact_mask, cardiac_mask
        if batch_num % 20 == 0:
            import gc; gc.collect()

    if writer is not None:
        writer.close()

    print(f"  Cardiac scores written: {cardiac_path}")
    if total_processed == 0:
        print("  WARNING: input parquet was empty")
        return pd.DataFrame()

    pct = (total_kept / total_processed * 100) if total_processed > 0 else 0
    print(f"  Filtered {total_kept:,} / {total_processed:,} rows ({pct:.1f}%)")

    if total_kept == 0:
        print("  WARNING: zero cardiac rows found. Check biosample names and "
              "GTEx tissue values in your input parquet.")
        return pd.DataFrame()

    if not cardiac_path.exists():
        print(f"  ERROR: cardiac parquet was not written to {cardiac_path}")
        return pd.DataFrame()

    print(f"  Cardiac parquet ready at {cardiac_path}")
    print(f"  (NOT loading into memory — downstream uses streaming)")
    return cardiac_path


def pa_table_from_pandas(df: pd.DataFrame):
    """Convert pandas DataFrame to pyarrow Table."""
    return _pa.Table.from_pandas(df, preserve_index=False)


def is_cardiac_track(row: pd.Series) -> bool:
    """Check if a score row is from a cardiac-relevant track."""
    # Check GTEx tissue
    gtex = str(row.get("gtex_tissue", "")).strip()
    if gtex in CARDIAC_GTEX_TISSUES:
        return True

    # Check biosample name
    biosample = str(row.get("biosample_name", "")).lower()
    for kw in CARDIAC_BIOSAMPLE_KEYWORDS:
        if kw in biosample:
            return True

    # Contact maps are tissue-averaged, always include
    if row.get("output_type") == "CONTACT_MAPS":
        return True

    return False


def filter_cardiac_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the full score DataFrame to cardiac-relevant tracks."""
    print("  Filtering to cardiac tracks...")

    # Vectorized filtering for speed on large DataFrames
    gtex_mask = df["gtex_tissue"].isin(CARDIAC_GTEX_TISSUES) if "gtex_tissue" in df.columns else pd.Series(False, index=df.index)

    biosample_mask = pd.Series(False, index=df.index)
    if "biosample_name" in df.columns:
        biosample_lower = df["biosample_name"].fillna("").str.lower()
        for kw in CARDIAC_BIOSAMPLE_KEYWORDS:
            biosample_mask = biosample_mask | biosample_lower.str.contains(kw, na=False)

    # Contact maps are always cardiac-relevant (tissue-averaged)
    contact_mask = df["output_type"] == "CONTACT_MAPS"

    cardiac_mask = gtex_mask | biosample_mask | contact_mask
    cardiac_df = df[cardiac_mask].copy()

    print(f"  Cardiac scores: {len(cardiac_df):,} / {len(df):,} "
          f"({len(cardiac_df)/len(df)*100:.1f}%)")

    return cardiac_df


# ---------------------------------------------------------------------------
# 3. VARIANT SUMMARY (one row per variant)
# ---------------------------------------------------------------------------

def build_variant_summary(cardiac_df: pd.DataFrame,
                          variants_meta: pd.DataFrame | None = None
                          ) -> pd.DataFrame:
    """Build a summary table with one row per variant, aggregating scores
    across modalities.

    For each variant, computes:
    - Max absolute raw_score per modality (expression, splicing, accessibility, etc.)
    - Max quantile_score per modality (if available)
    - Number of modalities with a score above threshold
    - Top affected gene per modality

    Gene name resolution strategy:
    - Gene-centric scorers (GeneMaskLFCScorer for expression, GeneMaskSplicingScorer
      for splicing) produce rows with gene_name populated. Use that directly.
    - Variant-centric scorers (CenterMaskScorer for accessibility/TF/histone/CAGE,
      ContactMapScorer for 3d_structure) have no gene_name. Fall back to the
      variant's metadata gene from variants_meta (clinvar_gene or l2g_gene),
      so every modality gets a meaningful gene label even when the scorer
      itself is not gene-specific.
    """
    cardiac_df = cardiac_df.copy()
    cardiac_df["modality"] = cardiac_df["output_type"].map(MODALITY_MAP).fillna("other")

    has_quantile = "quantile_score" in cardiac_df.columns
    records = []

    # Build a fast lookup: variant_id -> metadata gene fallback
    # Used when a variant-centric scorer modality has no gene_name from the scorer itself
    metadata_gene_lookup = {}
    if variants_meta is not None:
        for _, mrow in variants_meta.iterrows():
            vid = mrow["variant_id"]
            # Prefer clinvar_gene (HGNC, manually curated) over l2g_gene (ML predicted)
            for col in ["clinvar_gene", "l2g_gene"]:
                if col in mrow.index:
                    val = mrow[col]
                    if val and str(val) not in ("None", "nan", ""):
                        metadata_gene_lookup[vid] = str(val)
                        break

    for variant_id, vdf in cardiac_df.groupby("variant_id"):
        rec = {"variant_id": variant_id}
        metadata_gene = metadata_gene_lookup.get(variant_id)

        # Per-modality aggregation
        modalities_above_threshold = 0

        for modality, mdf in vdf.groupby("modality"):
            abs_scores = mdf["raw_score"].abs()
            max_abs_idx = abs_scores.idxmax()
            max_raw = mdf.loc[max_abs_idx, "raw_score"]

            rec[f"{modality}_max_raw"] = max_raw
            rec[f"{modality}_max_abs"] = abs(max_raw)

            if has_quantile and "quantile_score" in mdf.columns:
                q_scores = mdf["quantile_score"].abs()
                max_q_idx = q_scores.idxmax()
                rec[f"{modality}_max_quantile"] = mdf.loc[max_q_idx, "quantile_score"]

            # Top gene for this modality - try scorer output first, then metadata fallback
            top_gene = None
            if "gene_name" in mdf.columns:
                top_gene_row = mdf.loc[abs_scores.idxmax()]
                gene = top_gene_row.get("gene_name")
                if gene and str(gene) not in ("None", "nan", ""):
                    top_gene = str(gene)

            # Variant-centric scorer fallback: use metadata gene
            # This populates _top_gene for accessibility, tf_binding, histone_marks,
            # cage, 3d_structure, polyadenylation modalities that lack gene_name
            if not top_gene and metadata_gene:
                top_gene = metadata_gene

            if top_gene:
                rec[f"{modality}_top_gene"] = top_gene

            # Top tissue for this modality
            if "gtex_tissue" in mdf.columns:
                top_tissue_row = mdf.loc[abs_scores.idxmax()]
                tissue = top_tissue_row.get("gtex_tissue")
                if tissue and str(tissue) not in ("None", "nan", ""):
                    rec[f"{modality}_top_tissue"] = str(tissue)

            # Count modalities with strong signal
            if has_quantile and "quantile_score" in mdf.columns:
                if mdf["quantile_score"].abs().max() >= 0.95:
                    modalities_above_threshold += 1
            else:
                # Fallback: use raw score percentile within the data
                if abs_scores.max() > abs_scores.quantile(0.95):
                    modalities_above_threshold += 1

        rec["n_modalities_strong"] = modalities_above_threshold
        rec["n_cardiac_scores"] = len(vdf)

        # Composite score: sum of absolute-max raw scores across the six
        # canonical modality groups (expression, splicing, accessibility,
        # tf_binding, histone_marks, 3d_structure). Columns from output types
        # outside these groups (e.g., polyadenylation, which maps to "other")
        # are deliberately excluded here to match the manuscript definition
        # (§3.4).
        modality_abs_cols = [f"{m}_max_abs" for m in COMPOSITE_MODALITIES]
        rec["composite_score"] = sum(rec.get(c, 0) for c in modality_abs_cols)

        records.append(rec)

    summary = pd.DataFrame(records)

    # Merge with variant metadata if provided
    if variants_meta is not None and not summary.empty:
        meta_cols = ["variant_id", "chromosome", "position", "in_clinvar",
                     "in_gwas", "clinical_significance", "clinvar_gene",
                     "max_pip", "gwas_trait", "l2g_gene", "l2g_score",
                     "variant_location", "clinvar_condition"]
        available_cols = [c for c in meta_cols if c in variants_meta.columns]
        summary = summary.merge(
            variants_meta[available_cols].drop_duplicates(subset="variant_id"),
            on="variant_id", how="left",
        )

    # Sort by composite score descending
    summary = summary.sort_values("composite_score", ascending=False)

    return summary


# ---------------------------------------------------------------------------
# 4. MULTIMODAL HIT DETECTION
# ---------------------------------------------------------------------------

def find_multimodal_hits(summary: pd.DataFrame,
                         min_modalities: int = 2) -> pd.DataFrame:
    """Find variants that score strongly in multiple modalities.

    This replicates the multimodal enrichment analysis from AlphaGenome Fig 6f,
    which showed causal variants are enriched for multimodal effects.
    """
    hits = summary[summary["n_modalities_strong"] >= min_modalities].copy()
    hits = hits.sort_values("n_modalities_strong", ascending=False)
    return hits


# ---------------------------------------------------------------------------
# 5. TOP VARIANTS PER MODALITY
# ---------------------------------------------------------------------------

def top_variants_by_modality(cardiac_parquet_path,
                             modality_output_types: list[str],
                             top_n: int = 50,
                             batch_size: int = 500_000) -> pd.DataFrame:
    """Streaming: extract top N variants for a specific modality group.

    Iterates the cardiac parquet in batches, keeping per-variant max-abs
    statistics in a dict. Filters to modality_output_types at the column
    projection stage. Peak memory ~1-2 GB regardless of cardiac dataset size.
    """
    pf = _pq.ParquetFile(cardiac_parquet_path)

    needed_cols = ["variant_id", "output_type", "gene_name",
                   "gtex_tissue", "track_name", "raw_score"]
    schema_cols = set(pf.schema_arrow.names)
    needed_cols = [c for c in needed_cols if c in schema_cols]

    target_types = set(modality_output_types)
    accum = {}  # variant_id -> {max_abs, max_raw, n_tracks, top_gene, top_tissue, top_track}

    for batch in pf.iter_batches(batch_size=batch_size, columns=needed_cols):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["output_type"].isin(target_types)]
        if chunk.empty:
            del chunk
            continue

        chunk["abs_score"] = chunk["raw_score"].abs()

        # Per-variant top row in this batch
        idx = chunk.groupby("variant_id")["abs_score"].idxmax()
        top_rows = chunk.loc[idx]

        # Per-variant track count contribution from this batch
        track_counts = chunk.groupby("variant_id").size()

        for _, row in top_rows.iterrows():
            vid = row["variant_id"]
            abs_s = float(row["abs_score"])
            cur = accum.get(vid)
            this_count = int(track_counts.get(vid, 0))
            if cur is None:
                accum[vid] = {
                    "max_abs_raw": abs_s,
                    "max_raw": float(row["raw_score"]),
                    "n_tracks": this_count,
                    "top_gene": str(row.get("gene_name", "") or ""),
                    "top_tissue": str(row.get("gtex_tissue", "") or ""),
                    "top_track": str(row.get("track_name", "") or ""),
                }
            else:
                cur["n_tracks"] += this_count
                if abs_s > cur["max_abs_raw"]:
                    cur["max_abs_raw"] = abs_s
                    cur["max_raw"] = float(row["raw_score"])
                    cur["top_gene"] = str(row.get("gene_name", "") or "")
                    cur["top_tissue"] = str(row.get("gtex_tissue", "") or "")
                    cur["top_track"] = str(row.get("track_name", "") or "")

        del chunk, top_rows, track_counts

    if not accum:
        return pd.DataFrame()

    records = [{"variant_id": v, **stats} for v, stats in accum.items()]
    result = pd.DataFrame(records)
    result = result.nlargest(top_n, "max_abs_raw")
    return result


# ---------------------------------------------------------------------------
# 6. MODALITY ENRICHMENT SUMMARY
# ---------------------------------------------------------------------------

def compute_modality_enrichment(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute per-modality detection rates and multimodal breakdown.

    Mimics the enrichment analysis from AlphaGenome Fig 6f, showing
    what fraction of variants are detected in each modality category.
    """
    modalities = ["expression", "splicing", "accessibility", "tf_binding",
                  "histone_marks", "3d_structure"]

    records = []
    total = len(summary)

    for threshold_name, threshold_col, threshold_val in [
        ("any_signal", "_max_abs", 0),
        ("top_10pct", "_max_abs", None),  # computed per-modality
        ("strong_quantile", "_max_quantile", 0.95),
    ]:
        for mod in modalities:
            abs_col = f"{mod}_max_abs"
            q_col = f"{mod}_max_quantile"

            if abs_col not in summary.columns:
                continue

            if threshold_name == "top_10pct":
                thresh = summary[abs_col].quantile(0.9)
                detected = (summary[abs_col] > thresh).sum()
            elif threshold_name == "strong_quantile" and q_col in summary.columns:
                detected = (summary[q_col].abs() >= threshold_val).sum()
            else:
                detected = (summary[abs_col] > threshold_val).sum()

            records.append({
                "modality": mod,
                "threshold": threshold_name,
                "n_detected": detected,
                "pct_detected": detected / total * 100 if total > 0 else 0,
            })

    # Multimodal breakdown
    for n in range(0, 6):
        count = (summary["n_modalities_strong"] == n).sum()
        records.append({
            "modality": f"multimodal_{n}+",
            "threshold": "n_modalities_strong",
            "n_detected": count,
            "pct_detected": count / total * 100 if total > 0 else 0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------



def build_variant_summary_streaming(cardiac_parquet_path,
                                     variants_meta=None,
                                     batch_size: int = 500_000) -> pd.DataFrame:
    """Streaming variant summary that never loads the full cardiac dataset.

    Iterates over the cardiac_scores.parquet file in batches, accumulating
    per-(variant, modality) maximum scores in a dict. Peak memory ~2 GB
    regardless of cardiac dataset size.
    """
    pf = _pq.ParquetFile(cardiac_parquet_path)

    # Build metadata gene lookup once for variant-centric scorer fallback
    metadata_gene_lookup = {}
    if variants_meta is not None:
        for _, mrow in variants_meta.iterrows():
            vid = mrow["variant_id"]
            for col in ["clinvar_gene", "l2g_gene"]:
                if col in mrow.index:
                    val = mrow[col]
                    if val and str(val) not in ("None", "nan", ""):
                        metadata_gene_lookup[vid] = str(val)
                        break

    # Project only the columns we need
    needed_cols = ["variant_id", "output_type", "gene_name",
                   "gtex_tissue", "raw_score", "quantile_score"]
    schema_cols = set(pf.schema_arrow.names)
    needed_cols = [c for c in needed_cols if c in schema_cols]
    has_quantile = "quantile_score" in needed_cols

    # Accumulator: variant_id -> modality -> stats dict
    accum = {}

    total_rows = pf.metadata.num_rows
    total_batches = (total_rows + batch_size - 1) // batch_size
    print(f"  Streaming variant summary over {total_batches} batches "
          f"({total_rows:,} rows)...")

    batch_num = 0
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=needed_cols):
        batch_num += 1
        chunk = batch.to_pandas()
        rows_seen += len(chunk)

        chunk["modality"] = chunk["output_type"].map(MODALITY_MAP).fillna("other")
        chunk["abs_score"] = chunk["raw_score"].abs()

        # For each (variant, modality) in this batch, find the row with the
        # largest absolute raw score
        idx = chunk.groupby(["variant_id", "modality"])["abs_score"].idxmax()
        top_rows = chunk.loc[idx]

        for _, row in top_rows.iterrows():
            vid = row["variant_id"]
            mod = row["modality"]
            abs_s = float(row["abs_score"])

            if vid not in accum:
                accum[vid] = {}

            cur = accum[vid].get(mod)
            if cur is None or abs_s > cur["max_abs"]:
                gene_val = row.get("gene_name") if "gene_name" in row.index else None
                top_gene = None
                if gene_val is not None and pd.notna(gene_val) and str(gene_val) not in ("None", "nan", ""):
                    top_gene = str(gene_val)
                if not top_gene:
                    top_gene = metadata_gene_lookup.get(vid)

                tissue_val = row.get("gtex_tissue") if "gtex_tissue" in row.index else None
                top_tissue = None
                if tissue_val is not None and pd.notna(tissue_val) and str(tissue_val) not in ("None", "nan", ""):
                    top_tissue = str(tissue_val)

                accum[vid][mod] = {
                    "max_abs": abs_s,
                    "max_raw": float(row["raw_score"]),
                    "max_quantile": float(row["quantile_score"] or 0) if has_quantile and pd.notna(row.get("quantile_score")) else 0.0,
                    "top_gene": top_gene,
                    "top_tissue": top_tissue,
                }

        if batch_num % 50 == 0 or batch_num == 1:
            pct = rows_seen / total_rows * 100
            print(f"    Batch {batch_num}/{total_batches} ({pct:.1f}%) | "
                  f"tracking {len(accum):,} variants")

        del chunk, top_rows
        if batch_num % 20 == 0:
            import gc; gc.collect()

    # Build final DataFrame from accumulator
    print(f"  Building final summary for {len(accum):,} variants...")
    records = []
    for vid, mod_dict in accum.items():
        rec = {"variant_id": vid, "n_modalities_strong": 0}
        for mod, stats in mod_dict.items():
            rec[f"{mod}_max_raw"] = stats["max_raw"]
            rec[f"{mod}_max_abs"] = stats["max_abs"]
            rec[f"{mod}_max_quantile"] = stats["max_quantile"]
            if stats["top_gene"]:
                rec[f"{mod}_top_gene"] = stats["top_gene"]
            if stats["top_tissue"]:
                rec[f"{mod}_top_tissue"] = stats["top_tissue"]
            if abs(stats["max_quantile"]) >= 0.95:
                rec["n_modalities_strong"] += 1
        # Composite score: sum over the six canonical modality groups only
        # (see COMPOSITE_MODALITIES definition above). Excludes "other" bucket
        # (e.g., polyadenylation) to match manuscript §3.4.
        modality_abs_cols = [f"{m}_max_abs" for m in COMPOSITE_MODALITIES]
        rec["composite_score"] = sum(rec.get(c, 0) for c in modality_abs_cols)
        records.append(rec)

    summary = pd.DataFrame(records)

    # Merge with variant metadata
    if variants_meta is not None and not summary.empty:
        meta_cols = ["variant_id", "chromosome", "position", "in_clinvar",
                     "in_gwas", "clinical_significance", "clinvar_gene",
                     "max_pip", "gwas_trait", "l2g_gene", "l2g_score"]
        meta_cols = [c for c in meta_cols if c in variants_meta.columns]
        summary = summary.merge(variants_meta[meta_cols], on="variant_id", how="left")

    summary = summary.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return summary



def run_analysis(scores_path: str, variants_path: str | None,
                 output_dir: str) -> None:
    """Run the full Workstream 3 analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WORKSTREAM 3: CARDIAC TISSUE FILTERING & MULTIMODAL ANALYSIS")
    print("=" * 70)

    # --- Load scores (streaming to handle multi-billion-row Parquet files) ---
    print(f"\nLoading scores from {scores_path}...")
    if scores_path.endswith(".parquet"):
        scores_df = stream_filter_parquet_to_cardiac(scores_path, output_dir)
    else:
        scores_df = pd.read_csv(scores_path, sep="\t", low_memory=False)
        scores_df = filter_cardiac_scores(scores_df)
    print(f"  Cardiac parquet on disk; downstream operations will stream it.")

    # --- Load variant metadata ---
    variants_meta = None
    if variants_path:
        print(f"\nLoading variant metadata from {variants_path}...")
        variants_meta = pd.read_csv(variants_path, sep="\t", low_memory=False,
                                    dtype={"chromosome": str})
        print(f"  Loaded {len(variants_meta):,} variant records")

    # --- 3A: Cardiac Tissue Filtering ---
    # Already done during streaming load. scores_df is now the parquet path.
    print("\n--- 3A: Cardiac Tissue Filtering ---")
    cardiac_parquet_path = output_dir / "cardiac_scores.parquet"
    print(f"  Cardiac parquet saved to: {cardiac_parquet_path}")
    cardiac_df = None  # Never materialized — downstream uses streaming

    # (Per-output-type / per-tissue stats already printed during the streaming
    # filter step — see stream_filter_parquet_to_cardiac output above.)

    # --- 3B: Variant Summary (one row per variant) ---
    print("\n--- 3B: Gene-Target Mapping & Variant Summary ---")
    cardiac_parquet_path = output_dir / "cardiac_scores.parquet"
    summary = build_variant_summary_streaming(cardiac_parquet_path, variants_meta)
    summary_path = output_dir / "variant_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"  Variant summary: {len(summary):,} variants")
    print(f"  Saved to {summary_path}")

    # Show top 10
    print("\n  Top 10 variants by cardiac composite score:")
    display_cols = ["variant_id", "composite_score", "n_modalities_strong"]
    for mod in ["expression", "splicing", "accessibility"]:
        gene_col = f"{mod}_top_gene"
        if gene_col in summary.columns:
            display_cols.append(gene_col)
    if "clinical_significance" in summary.columns:
        display_cols.append("clinical_significance")
    if "max_pip" in summary.columns:
        display_cols.append("max_pip")

    avail_cols = [c for c in display_cols if c in summary.columns]
    for _, row in summary.head(10).iterrows():
        parts = [f"{row['variant_id']:45s}"]
        parts.append(f"composite={row['composite_score']:.4f}")
        parts.append(f"modalities={row['n_modalities_strong']}")

        genes = set()
        for mod in ["expression", "splicing", "accessibility"]:
            g = row.get(f"{mod}_top_gene")
            if g and str(g) not in ("nan", "None", ""):
                genes.add(str(g))
        if genes:
            parts.append(f"genes={','.join(sorted(genes))}")

        clinsig = row.get("clinical_significance", "")
        if clinsig and str(clinsig) not in ("nan", "None", ""):
            parts.append(f"[{clinsig}]")

        pip = row.get("max_pip")
        if pip and float(pip) > 0:
            parts.append(f"PIP={float(pip):.3f}")

        print(f"    {' | '.join(parts)}")

    # --- 3C: Multimodal Hit Detection ---
    print("\n--- 3C: Multimodal Enrichment Analysis ---")
    multimodal = find_multimodal_hits(summary, min_modalities=2)
    multi_path = output_dir / "multimodal_hits.tsv"
    multimodal.to_csv(multi_path, sep="\t", index=False)
    print(f"  Multimodal hits (2+ modalities): {len(multimodal):,}")

    if not multimodal.empty:
        print(f"\n  Top multimodal variants:")
        for _, row in multimodal.head(5).iterrows():
            mods = []
            for mod in ["expression", "splicing", "accessibility",
                        "tf_binding", "histone_marks", "3d_structure"]:
                abs_col = f"{mod}_max_abs"
                if abs_col in row and row[abs_col] > 0:
                    mods.append(f"{mod}={row[abs_col]:.4f}")
            gene = row.get("expression_top_gene") or row.get("clinvar_gene") or "?"
            print(f"    {row['variant_id']:45s} gene={str(gene):10s} "
                  f"mods={row['n_modalities_strong']}  {', '.join(mods[:3])}")

    # Enrichment stats
    enrichment = compute_modality_enrichment(summary)
    enrich_path = output_dir / "modality_enrichment.tsv"
    enrichment.to_csv(enrich_path, sep="\t", index=False)
    print(f"\n  Modality enrichment saved to {enrich_path}")

    # --- Per-modality top lists ---
    print("\n--- Per-Modality Top Variant Lists ---")

    modality_configs = [
        ("expression", ["RNA_SEQ", "CAGE", "PROCAP"], "top_expression_variants.tsv"),
        ("splicing", ["SPLICE_JUNCTIONS", "SPLICE_SITES", "SPLICE_SITE_USAGE"], "top_splicing_variants.tsv"),
        ("accessibility", ["DNASE", "ATAC"], "top_accessibility_variants.tsv"),
        ("tf_binding", ["CHIP_TF"], "top_tf_binding_variants.tsv"),
        ("3d_structure", ["CONTACT_MAPS"], "top_contact_map_variants.tsv"),
    ]

    for mod_name, output_types, filename in modality_configs:
        top = top_variants_by_modality(cardiac_parquet_path, output_types, top_n=50)
        if not top.empty:
            top_path = output_dir / filename
            top.to_csv(top_path, sep="\t", index=False)
            print(f"  {mod_name:20s}: {len(top):3d} variants → {filename}")
            # Show top 3
            for _, row in top.head(3).iterrows():
                vid = str(row.get('variant_id', '?'))[:45]
                score = row.get('max_raw', 0) or 0
                gene = str(row.get('top_gene', '?') or '?')[:15]
                tissue = str(row.get('top_tissue', '?') or '?')
                print(f"    {vid:45s} "
                      f"score={float(score):.4f}  "
                      f"gene={gene:15s}  "
                      f"tissue={tissue}")
        else:
            print(f"  {mod_name:20s}: no cardiac scores found")

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("WORKSTREAM 3 SUMMARY")
    print("=" * 70)
    try:
        input_rows = _pq.ParquetFile(scores_path).metadata.num_rows
        cardiac_rows = _pq.ParquetFile(output_dir / "cardiac_scores.parquet").metadata.num_rows
        pct = cardiac_rows / input_rows * 100 if input_rows else 0
        print(f"Total input scores:           {input_rows:>12,}")
        print(f"Cardiac-filtered scores:      {cardiac_rows:>12,} ({pct:.1f}%)")
    except Exception as e:
        print(f"Row counts unavailable: {e}")
    print(f"Unique variants scored:       {summary['variant_id'].nunique():>12,}")
    print(f"Multimodal hits (2+ mods):    {len(multimodal):>12,}")

    if "in_clinvar" in summary.columns:
        clinvar_multi = multimodal[multimodal.get("in_clinvar", False) == True]
        gwas_multi = multimodal[multimodal.get("in_gwas", False) == True]
        print(f"  ClinVar multimodal:         {len(clinvar_multi):>12,}")
        print(f"  GWAS multimodal:            {len(gwas_multi):>12,}")

    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey files for Workstream 4 (Benchmarking) and CRF pitch:")
    print(f"  variant_summary.tsv       — Ranked variants with per-modality scores")
    print(f"  multimodal_hits.tsv       — Your strongest candidates")
    print(f"  top_expression_variants   — For eQTL comparison story")
    print(f"  top_splicing_variants     — For SpliceAI comparison story")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cardiac tissue filtering and multimodal enrichment "
                    "analysis (Workstream 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cardiac_filter_analysis.py \\
      --scores /mnt/local_data/alphagenome-cardio/scoring_output/all_scores_tidy.parquet \\
      --variants /mnt/local_data/alphagenome-cardio/variant_interval/merged_cardio_variants.tsv \\
      --output-dir /mnt/local_data/alphagenome-cardio/ws3_output/
        """,
    )
    parser.add_argument("--scores", "-s", required=True,
                        help="Tidy scores Parquet/TSV from Workstream 2")
    parser.add_argument("--variants", "-v", default=None,
                        help="Merged variants TSV from Workstream 1C (for metadata)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory")

    args = parser.parse_args()
    run_analysis(args.scores, args.variants, args.output_dir)


if __name__ == "__main__":
    main()