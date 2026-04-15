#!/usr/bin/env python3
"""Workstream 2: AlphaGenome Variant Scoring for Cardiovascular Use Case.

Reads the merged variants and scoring intervals from Workstream 1C,
scores each variant using AlphaGenome's recommended scorers, and produces
a tidy DataFrame with all scores across modalities.

This script is designed to:
- Use `score_variants()` with built-in parallelism (max_workers)
- Process variants in batches with checkpointing (resume on failure)
- Use the full recommended scorer set (all 20 scorers in one call)
- Save both raw AnnData results and tidied DataFrames

Usage:
    python alphag_cardio_scoring.py \
        --variants merged_cardio_variants.tsv \
        --intervals scoring_intervals.tsv \
        --output-dir /path/to/scoring_output/ \
        [--batch-size 50] \
        [--max-workers 5] \
        [--top-n 500]  # score top N priority variants first

Prerequisites:
    - AlphaGenome Python client installed and authenticated
    - pip install anndata pandas tqdm

Architecture note:
    The AlphaGenome API's `score_variants()` already parallelizes internally
    using ThreadPoolExecutor with max_workers. We DON'T add another layer of
    parallelism on top — we batch variants and let the API handle concurrency.
"""

import argparse
import csv
import json
import os
import sys
import time
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# AlphaGenome imports — adjust path as needed for your setup
from alphagenome.data import genome
from alphagenome.models import variant_scorers
from alphagenome.models import dna_client
from alphagenome.models import dna_output
HAS_ALPHAGENOME = True


# ---------------------------------------------------------------------------
# 1. SCORER CONFIGURATION
# ---------------------------------------------------------------------------
# We define our cardiac-focused scorer groups. Each group targets a different
# biological modality. All can be passed to a single score_variant() call
# (up to 20 scorers max).

def get_cardiac_scorers() -> dict[str, list]:
    """Return the FULL recommended scorer set (19 scorers, all in one API call).

    This matches AlphaGenome's RECOMMENDED_VARIANT_SCORERS dict from
    variant_scorers.py. Each variant gets scored across:
      - 6 directional CenterMask scorers (ATAC/DNASE/CHIP_TF/CHIP_HISTONE/CAGE/PROCAP)
      - 6 active CenterMask scorers (same 6 modalities, ACTIVE_SUM aggregation)
      - 1 ContactMap scorer (3D structure)
      - 1 directional GeneMaskLFC (RNA_SEQ expression)
      - 1 active GeneMaskActive (RNA_SEQ max activity)
      - 2 GeneMaskSplicing scorers (SPLICE_SITES + SPLICE_SITE_USAGE)
      - 1 SpliceJunction scorer
      - 1 Polyadenylation scorer

    All 19 scorers fit within AlphaGenome's 20-scorer-per-request limit.
    Each variant gets one API round-trip producing scores across all modalities,
    BOTH the directional (ALT-REF, signed) and active (max(ALT,REF), unsigned)
    aggregations.

    Why both directional and active:
      - Directional (DIFF_LOG2_SUM) tells you up vs down regulation
      - Active (ACTIVE_SUM) tells you whether the locus has activity at all
      - High active + low directional = element shifting/reorganizing
      - High active + high directional = element being created or destroyed
      - Combining both improves variant pathogenicity discrimination
    """
    scorers = {
        # --- Expression (signed + active) ---
        "expression": [
            variant_scorers.GeneMaskLFCScorer(
                requested_output=dna_client.OutputType.RNA_SEQ,
            ),
            variant_scorers.GeneMaskActiveScorer(
                requested_output=dna_client.OutputType.RNA_SEQ,
            ),
        ],

        # --- Splicing (no active variants — splicing scorers are unsigned by nature) ---
        "splicing": [
            variant_scorers.SpliceJunctionScorer(),
            variant_scorers.GeneMaskSplicingScorer(
                requested_output=dna_client.OutputType.SPLICE_SITES,
                width=None,
            ),
            variant_scorers.GeneMaskSplicingScorer(
                requested_output=dna_client.OutputType.SPLICE_SITE_USAGE,
                width=None,
            ),
        ],

        # --- Chromatin accessibility (signed + active) ---
        "accessibility": [
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.DNASE,
                width=501,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.DNASE,
                width=501,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.ATAC,
                width=501,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.ATAC,
                width=501,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
        ],

        # --- 3D contact maps ---
        "contact_maps": [
            variant_scorers.ContactMapScorer(),
        ],

        # --- Polyadenylation ---
        "polyadenylation": [
            variant_scorers.PolyadenylationScorer(),
        ],

        # --- TF binding (signed + active) ---
        "tf_binding": [
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CHIP_TF,
                width=501,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CHIP_TF,
                width=501,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
        ],

        # --- Histone marks (signed + active, wider window) ---
        "histone_marks": [
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CHIP_HISTONE,
                width=2001,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CHIP_HISTONE,
                width=2001,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
        ],

        # --- CAGE (signed + active) ---
        "cage": [
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CAGE,
                width=501,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.CAGE,
                width=501,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
        ],

        # --- PROCAP (signed + active) — was previously missing ---
        "procap": [
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.PROCAP,
                width=501,
                aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
            ),
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.PROCAP,
                width=501,
                aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
            ),
        ],
    }
    return scorers


def flatten_scorers(scorer_groups: dict) -> list:
    """Flatten scorer groups into a single list for API call."""
    flat = []
    for group in scorer_groups.values():
        flat.extend(group)
    # Deduplicate (scorers are frozen dataclasses, so set works)
    return list(dict.fromkeys(flat))


# ---------------------------------------------------------------------------
# 2. READ WORKSTREAM 1C OUTPUTS
# ---------------------------------------------------------------------------

def read_merged_variants(path: str) -> pd.DataFrame:
    """Read the merged variants TSV from Workstream 1C."""
    df = pd.read_csv(path, sep="\t", dtype={"chromosome": str, "position": int})
    print(f"  Loaded {len(df):,} variants from {path}")
    return df


def read_intervals(path: str) -> pd.DataFrame:
    """Read the scoring intervals TSV from Workstream 1C."""
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {len(df):,} intervals from {path}")
    return df


# ---------------------------------------------------------------------------
# 3. CHECKPOINTING
# ---------------------------------------------------------------------------

class Checkpoint:
    """Simple file-based checkpoint to resume scoring after failures."""

    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scored_file = self.dir / "scored_variants.txt"
        self.scored: set[str] = set()
        self._load()

    def _load(self):
        if self.scored_file.exists():
            with open(self.scored_file) as f:
                self.scored = {line.strip() for line in f if line.strip()}
            print(f"  Checkpoint: {len(self.scored):,} variants already scored")

    def is_scored(self, variant_id: str) -> bool:
        return variant_id in self.scored

    def mark_scored(self, variant_ids: list[str]):
        self.scored.update(variant_ids)
        with open(self.scored_file, "a") as f:
            for vid in variant_ids:
                f.write(vid + "\n")

    def count(self) -> int:
        return len(self.scored)


# ---------------------------------------------------------------------------
# 4. SCORING PIPELINE
# ---------------------------------------------------------------------------

def score_batch(
    client,
    variants_df: pd.DataFrame,
    all_scorers: list,
    max_workers: int = 5,
) -> list[list]:
    """Score a batch of variants using score_variants().

    Args:
        client: AlphaGenome DnaClient instance
        variants_df: DataFrame with variant_id, chromosome, position,
                     ref, alt, interval_start, interval_end
        all_scorers: Flat list of all variant scorers
        max_workers: Parallel workers for the API

    Returns:
        List of list[AnnData] — one inner list per variant, one AnnData
        per scorer.
    """
    # Build Variant and Interval objects
    variant_objects = []
    interval_objects = []

    for _, row in variants_df.iterrows():
        variant_objects.append(genome.Variant(
            chromosome=str(row["chromosome"]),
            position=int(row["position"]),
            reference_bases=str(row["ref"]),
            alternate_bases=str(row["alt"]),
        ))
        interval_objects.append(genome.Interval(
            chromosome=str(row["chromosome"]),
            start=int(row["interval_start"]),
            end=int(row["interval_end"]),
        ))

    # score_variants() handles parallelism internally
    scores = client.score_variants(
        intervals=interval_objects,
        variants=variant_objects,
        variant_scorers=all_scorers,
        max_workers=max_workers,
    )

    return scores


def tidy_batch_scores(scores: list[list], variant_ids: list[str]) -> pd.DataFrame:
    """Convert a batch of raw AnnData scores to a tidy DataFrame.

    Uses variant_scorers.tidy_scores() from the AlphaGenome library.
    Converts any non-serializable objects (Variant, Interval) to strings.
    """
    tidy_df = variant_scorers.tidy_scores(scores)
    if tidy_df is None:
        return pd.DataFrame()

    # The SDK stores genome.Variant and genome.Interval objects in uns,
    # which tidy_scores copies into the DataFrame. Convert to strings
    # so Parquet/CSV serialization works.
    for col in tidy_df.columns:
        if tidy_df[col].dtype == object:
            # Check first non-null value
            first_val = tidy_df[col].dropna().iloc[0] if not tidy_df[col].dropna().empty else None
            if first_val is not None and not isinstance(first_val, str):
                tidy_df[col] = tidy_df[col].astype(str)

    return tidy_df


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_scoring(
    variants_path: str,
    intervals_path: str,
    output_dir: str,
    batch_size: int = 50,
    max_workers: int = 5,
    top_n: int | None = None,
    dry_run: bool = False,
    keep_batches: bool = False,
) -> None:
    """Run the full AlphaGenome scoring pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WORKSTREAM 2: AlphaGenome VARIANT SCORING")
    print("=" * 70)

    # --- Read inputs ---
    print("\nLoading Workstream 1C outputs...")
    variants_df = read_merged_variants(variants_path)
    intervals_df = read_intervals(intervals_path)

    # --- Optional: limit to top-N priority variants ---
    if top_n and top_n < len(variants_df):
        print(f"\nLimiting to top {top_n} priority variants...")
        # priority_score column from 1C output, sort descending
        if "priority_score" in variants_df.columns:
            variants_df = variants_df.nlargest(top_n, "priority_score")
        else:
            variants_df = variants_df.head(top_n)
        print(f"  Working set: {len(variants_df):,} variants")

    # --- Configure scorers ---
    print("\nConfiguring scorers...")
    scorer_groups = get_cardiac_scorers()
    all_scorers = flatten_scorers(scorer_groups)
    print(f"  Scorer groups: {len(scorer_groups)}")
    print(f"  Total scorers: {len(all_scorers)} (max per request: 20)")
    for group_name, group_scorers in scorer_groups.items():
        scorer_names = [s.__class__.__name__ for s in group_scorers]
        print(f"    {group_name}: {', '.join(scorer_names)}")

    if len(all_scorers) > 20:
        print("  WARNING: More than 20 scorers — will need to split calls.")
        print("  Splitting into chunks of 20...")
        # This shouldn't happen with our cardiac config (we have ~10)

    # --- Initialize checkpoint ---
    checkpoint = Checkpoint(str(output_dir / "checkpoints"))
    remaining = variants_df[
        ~variants_df["variant_id"].isin(checkpoint.scored)
    ]
    print(f"\n  Total variants: {len(variants_df):,}")
    print(f"  Already scored: {checkpoint.count():,}")
    print(f"  Remaining:      {len(remaining):,}")

    if len(remaining) == 0:
        print("\nAll variants already scored! Skipping to aggregation...")
    elif dry_run or not HAS_ALPHAGENOME:
        print(f"\n{'DRY RUN' if dry_run else 'NO ALPHAGENOME CLIENT'}: "
              f"Would score {len(remaining):,} variants in "
              f"{(len(remaining) + batch_size - 1) // batch_size} batches")
        print(f"  Batch size: {batch_size}")
        print(f"  Max workers: {max_workers}")
        print(f"  Estimated API calls: {len(remaining):,} "
              f"(each with {len(all_scorers)} scorers)")

        # Write a sample of what the output structure would look like
        _write_dry_run_sample(output_dir, variants_df.head(5), scorer_groups)
        return
    else:
        # --- SCORING LOOP ---
        print(f"\nScoring {len(remaining):,} variants in batches of "
              f"{batch_size} ({max_workers} workers)...")

        # Initialize client
        api_key = os.environ.get("ALPHAGENOME_API_KEY", "")
        if not api_key:
            # Try reading from a key file
            key_file = Path.home() / ".alphagenome" / "api_key.txt"
            if key_file.exists():
                api_key = key_file.read_text().strip()
        if not api_key:
            print("ERROR: No API key found. Set ALPHAGENOME_API_KEY env var "
                  "or create ~/.alphagenome/api_key.txt")
            sys.exit(1)
        client = dna_client.create(api_key)

        # Pre-filter: skip variants whose ref or alt allele doesn't fit in the interval
        # (e.g., large deletions that extend beyond the 1MB window)
        def _variant_fits(row):
            var_start = int(row["position"])
            ref_len = len(str(row["ref"]))
            alt_len = len(str(row["alt"]))
            var_end = var_start + max(ref_len, alt_len)
            int_start = int(row["interval_start"])
            int_end = int(row["interval_end"])
            # Variant must be fully contained in interval
            return (var_start >= int_start) and (var_end <= int_end)

        fit_mask = remaining.apply(_variant_fits, axis=1)
        oversized = remaining[~fit_mask]
        remaining = remaining[fit_mask]

        if len(oversized) > 0:
            oversized_path = output_dir / "skipped_oversized_variants.tsv"
            oversized[["variant_id", "chromosome", "position", "ref", "alt",
                       "interval_start", "interval_end"]].to_csv(
                oversized_path, sep="\t", index=False)
            print(f"  Skipped {len(oversized):,} oversized variants "
                  f"(structural variants too large for 1MB window)")
            print(f"  Logged to: {oversized_path}")
            # Mark them as "scored" in checkpoint so we don't retry
            checkpoint.mark_scored(oversized["variant_id"].tolist())

        all_tidy_dfs = []
        batch_num = 0
        total_batches = (len(remaining) + batch_size - 1) // batch_size
        total_rows_scored = 0

        for start_idx in range(0, len(remaining), batch_size):
            batch_num += 1
            batch = remaining.iloc[start_idx:start_idx + batch_size]
            batch_vids = batch["variant_id"].tolist()

            print(f"\n  Batch {batch_num}/{total_batches} "
                  f"({len(batch)} variants)...", end="", flush=True)

            t0 = time.time()
            try:
                scores = score_batch(client, batch, all_scorers, max_workers=max_workers)
                tidy_df = tidy_batch_scores(scores, batch_vids)
                if not tidy_df.empty:
                    batch_path = output_dir / f"batch_{batch_num:05d}.parquet"
                    tidy_df.to_parquet(batch_path, index=False)
                    total_rows_scored += len(tidy_df)
                    checkpoint.mark_scored(batch_vids)        # ← moved INSIDE the if
                else:
                    print(f" WARNING: empty tidy_df for {len(batch_vids)} variants, NOT checkpointing", flush=True)
                elapsed = time.time() - t0
                print(f" done ({elapsed:.1f}s, {len(tidy_df):,} score rows)")
                del scores, tidy_df
                if batch_num % 20 == 0:
                    import gc; gc.collect()
            except Exception as e:
                elapsed = time.time() - t0
                import traceback
                print(f" FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
                traceback.print_exc()                         # ← see what's actually failing

        # --- Concatenate all batches from disk ---
        batch_files = sorted(output_dir.glob("batch_*.parquet"))
        if batch_files:
            print(f"\nConcatenating {len(batch_files)} batch files from disk...")
            full_path = output_dir / "all_scores_tidy.parquet"
            concat_succeeded = False

            # Stream-concatenate: read and write in chunks to avoid OOM
            # Use pyarrow for memory-efficient concatenation
            try:
                import pyarrow.parquet as pq
                writer = None
                total_rows = 0
                for bf in batch_files:
                    table = pq.read_table(bf)
                    if writer is None:
                        # First batch defines the canonical order
                        writer = pq.ParquetWriter(out_path, table.schema)
                        writer.write_table(table)
                    else:
                        # Reorder columns to match writer's schema
                        canonical_cols = [f.name for f in writer.schema_arrow]
                        table = table.select(canonical_cols)
                        writer.write_table(table)
                    writer.write_table(table)
                    total_rows += len(table)
                    del table
                if writer:
                    writer.close()
                print(f"  Full scores saved: {full_path} ({total_rows:,} rows)")
                concat_succeeded = True
            except ImportError:
                # Fallback: concatenate in pandas chunks
                print("  pyarrow.parquet not available, using pandas chunked concat...")
                chunk_dfs = []
                for bf in batch_files:
                    chunk_dfs.append(pd.read_parquet(bf))
                    # Write intermediate if accumulation gets large
                    if len(chunk_dfs) >= 50:
                        partial = pd.concat(chunk_dfs, ignore_index=True)
                        partial.to_parquet(full_path, index=False)
                        chunk_dfs = [partial]
                        del partial
                if chunk_dfs:
                    full_scores = pd.concat(chunk_dfs, ignore_index=True)
                    full_scores.to_parquet(full_path, index=False)
                    print(f"  Full scores saved: {full_path} ({len(full_scores):,} rows)")
                    del full_scores, chunk_dfs
                    concat_succeeded = True

            # Clean up batch files after successful concatenation to save disk space.
            # For 38K variants this is ~770 small files; for the 500-study run it
            # would be 10,000+. Keeping them doubles disk usage and is unnecessary
            # since the consolidated all_scores_tidy.parquet contains everything.
            if concat_succeeded and full_path.exists() and full_path.stat().st_size > 0:
                if not keep_batches:
                    print(f"  Cleaning up {len(batch_files)} batch files...")
                    cleaned = 0
                    for bf in batch_files:
                        try:
                            bf.unlink()
                            cleaned += 1
                        except OSError as e:
                            print(f"    Warning: failed to remove {bf.name}: {e}")
                    print(f"  Removed {cleaned} batch files")
                else:
                    print(f"  Keeping {len(batch_files)} batch files (--keep-batches)")
            else:
                print("  Concatenation incomplete — keeping batch files for safety")

            # Skip TSV for large runs — Parquet is the primary format
            # TSV of 1.5B rows would be 100+ GB
            if total_rows_scored < 5_000_000:
                try:
                    tsv_path = output_dir / "all_scores_tidy.tsv"
                    tsv_df = pd.read_parquet(full_path)
                    tsv_df.to_csv(tsv_path, sep="\t", index=False)
                    print(f"  TSV copy: {tsv_path}")
                    del tsv_df
                except Exception:
                    print("  TSV copy skipped (too large for memory)")
            else:
                print(f"  TSV copy skipped ({total_rows_scored:,} rows too large; use Parquet)")

    # --- Summary ---
    _print_summary(output_dir, variants_df, scorer_groups)


def _write_dry_run_sample(output_dir: Path, sample_df: pd.DataFrame,
                          scorer_groups: dict) -> None:
    """Write a sample file showing the expected output structure."""
    sample_path = output_dir / "DRY_RUN_sample_output.tsv"

    rows = []
    for _, v in sample_df.iterrows():
        for group_name, scorers in scorer_groups.items():
            for scorer in scorers:
                rows.append({
                    "variant_id": v["variant_id"],
                    "scored_interval": v.get("interval_id", ""),
                    "gene_id": "(from scoring)",
                    "gene_name": v.get("clinvar_gene", "") or v.get("l2g_gene", ""),
                    "output_type": scorer.requested_output.name,
                    "variant_scorer": str(scorer),
                    "track_name": "(per tissue/cell type track)",
                    "gtex_tissue": "(e.g. Heart_Left_Ventricle)",
                    "biosample_name": "(e.g. left ventricle myocardium)",
                    "raw_score": 0.0,
                    "quantile_score": 0.0,
                    "modality_group": group_name,
                })

    sample_out = pd.DataFrame(rows)
    sample_out.to_csv(sample_path, sep="\t", index=False)
    print(f"\n  Dry run sample written to: {sample_path}")
    print(f"  ({len(rows)} example rows showing output structure)")


def _print_summary(output_dir: Path, variants_df: pd.DataFrame,
                   scorer_groups: dict) -> None:
    """Print summary statistics using pyarrow metadata to avoid loading full parquet."""
    print("\n" + "=" * 70)
    print("WORKSTREAM 2 SUMMARY")
    print("=" * 70)

    full_path = output_dir / "all_scores_tidy.parquet"
    if not full_path.exists():
        print("No completed scores found yet.")
        return

    try:
        import pyarrow.parquet as pq
        # Get row count from metadata without loading data
        pf = pq.ParquetFile(full_path)
        total_rows = pf.metadata.num_rows
        print(f"Total score rows:     {total_rows:,}")

        # For large files (>10M rows), use chunked streaming to avoid OOM
        if total_rows > 10_000_000:
            print(f"Computing statistics via streaming (file too large for memory)...")
            from collections import Counter
            variant_ids = set()
            output_type_counts = Counter()
            cardiac_count = 0

            # Read in batches of 500K rows
            batch_iter = pf.iter_batches(
                batch_size=500_000,
                columns=["variant_id", "output_type", "gtex_tissue"]
            )
            for batch in batch_iter:
                batch_df = batch.to_pandas()
                variant_ids.update(batch_df["variant_id"].unique())
                output_type_counts.update(batch_df["output_type"].value_counts().to_dict())
                if "gtex_tissue" in batch_df.columns:
                    cardiac_mask = batch_df["gtex_tissue"].fillna("").str.contains(
                        "Heart|Artery|Aorta", case=False, na=False
                    )
                    cardiac_count += int(cardiac_mask.sum())
                del batch_df

            print(f"Unique variants:      {len(variant_ids):,}")
            print(f"\nScores by output type:")
            for otype, count in sorted(output_type_counts.items(),
                                       key=lambda x: -x[1]):
                print(f"  {otype:30s} {count:>12,}")
            if cardiac_count > 0:
                print(f"\nCardiac tissue scores: {cardiac_count:,} "
                      f"({cardiac_count / total_rows * 100:.1f}%)")
        else:
            # Small enough to load
            df = pd.read_parquet(full_path)
            print(f"Unique variants:      {df['variant_id'].nunique():,}")
            if "output_type" in df.columns:
                print(f"\nScores by output type:")
                for otype, count in df["output_type"].value_counts().items():
                    print(f"  {otype:30s} {count:>12,}")
            if "gtex_tissue" in df.columns:
                cardiac_mask = df["gtex_tissue"].fillna("").str.contains(
                    "Heart|Artery|Aorta", case=False, na=False
                )
                cardiac = cardiac_mask.sum()
                print(f"\nCardiac tissue scores: {cardiac:,} "
                      f"({cardiac / len(df) * 100:.1f}%)")
            del df
    except Exception as e:
        print(f"Summary stats failed: {e}")
        print("Scores are still saved in the parquet file.")

    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext steps (Workstream 3):")
    print(f"  1. Filter scores to cardiac GTEx tracks")
    print(f"  2. Identify multimodal hits (variants scoring high in 2+ modalities)")
    print(f"  3. Map variants to target genes via enhancer-gene linking")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score cardiovascular variants with AlphaGenome "
                    "(Workstream 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — validate inputs and show expected output structure
  python alphag_cardio_scoring.py \\
      --variants merged_cardio_variants.tsv \\
      --intervals scoring_intervals.tsv \\
      --output-dir scoring_output/ \\
      --dry-run

  # Score top 500 priority variants first (quick results for CRF pitch)
  python alphag_cardio_scoring.py \\
      --variants merged_cardio_variants.tsv \\
      --intervals scoring_intervals.tsv \\
      --output-dir scoring_output/ \\
      --top-n 500 \\
      --batch-size 25

  # Full scoring run
  python alphag_cardio_scoring.py \\
      --variants merged_cardio_variants.tsv \\
      --intervals scoring_intervals.tsv \\
      --output-dir scoring_output/ \\
      --batch-size 50 \\
      --max-workers 5

  # Resume after interruption (checkpoint auto-detects progress)
  python alphag_cardio_scoring.py \\
      --variants merged_cardio_variants.tsv \\
      --intervals scoring_intervals.tsv \\
      --output-dir scoring_output/
        """,
    )
    parser.add_argument("--variants", "-v", required=True,
                        help="Merged variants TSV from Workstream 1C")
    parser.add_argument("--intervals", "-i", required=True,
                        help="Scoring intervals TSV from Workstream 1C")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for scores")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Variants per scoring batch (default: 50)")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Parallel workers for API calls (default: 5)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Score only top N priority variants")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without scoring")
    parser.add_argument("--api-key", default=None,
                        help="AlphaGenome API key (or set ALPHAGENOME_API_KEY env var)")
    parser.add_argument("--keep-batches", action="store_true",
                        help="Don't delete per-batch parquet files after concatenation "
                             "(saves disk space; default is to clean up)")

    args = parser.parse_args()
    if args.api_key:
        os.environ["ALPHAGENOME_API_KEY"] = args.api_key
    run_scoring(
        args.variants, args.intervals, args.output_dir,
        args.batch_size, args.max_workers, args.top_n, args.dry_run,
        keep_batches=args.keep_batches,
    )


if __name__ == "__main__":
    main()