#!/usr/bin/env python3
"""Merge ClinVar + GWAS variants and create 1Mb AlphaGenome scoring intervals.

Workstream 1C: Variant-to-Interval Mapping.

Takes the outputs of 1A (ClinVar) and 1B (GWAS), merges them, deduplicates
by genomic position, creates centered 1Mb intervals for AlphaGenome, and
clusters nearby variants to minimize redundant API calls.

Usage:
    python variant_interval_mapper.py \
        --clinvar clinvar_cardio_noncoding.tsv \
        --gwas gwas_cardio_credible_sets.tsv \
        --output-variants merged_variants.tsv \
        --output-intervals scoring_intervals.tsv

Output is directly compatible with:
    genome.Variant.from_str(variant_id)
    genome.Interval(chromosome, start, end)
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. CHROMOSOME LENGTHS (GRCh38)
# ---------------------------------------------------------------------------
# Used to clamp intervals at chromosome boundaries.

CHROM_LENGTHS_GRCH38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}

#WINDOW_SIZE = 1_000_000  # 1 Mb — AlphaGenome's context window
WINDOW_SIZE = 2**20  # 1,048,576 bp


# ---------------------------------------------------------------------------
# 2. DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class MergedVariant:
    """A unified variant record from ClinVar and/or GWAS."""
    variant_id: str         # chr:pos:ref>alt (AlphaGenome format)
    chromosome: str
    position: int
    ref: str
    alt: str
    # Source tracking
    in_clinvar: bool = False
    in_gwas: bool = False
    # ClinVar metadata
    clinical_significance: str = ""
    variant_location: str = ""
    clinvar_condition: str = ""
    clinvar_gene: str = ""
    review_stars: int = 0
    # GWAS metadata
    max_pip: float = 0.0
    gwas_trait: str = ""
    l2g_gene: str = ""
    l2g_score: float = 0.0
    gwas_study_count: int = 0
    consequence: str = ""
    # Interval assignment
    interval_id: str = ""
    interval_start: int = 0
    interval_end: int = 0


# ---------------------------------------------------------------------------
# 3. FILE READERS
# ---------------------------------------------------------------------------

def read_clinvar(path: str) -> dict[str, MergedVariant]:
    """Read ClinVar TSV from Workstream 1A."""
    variants = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row["variant_id"]
            if vid in variants:
                # Same variant, different condition — keep higher review stars
                existing = variants[vid]
                if int(row.get("review_stars", 0)) > existing.review_stars:
                    existing.review_stars = int(row["review_stars"])
                    existing.clinical_significance = row["clinical_significance"]
                    existing.clinvar_condition = row["condition"]
                continue

            variants[vid] = MergedVariant(
                variant_id=vid,
                chromosome=row["chromosome"],
                position=int(row["position"]),
                ref=row["ref"],
                alt=row["alt"],
                in_clinvar=True,
                clinical_significance=row.get("clinical_significance", ""),
                variant_location=row.get("variant_location", ""),
                clinvar_condition=row.get("condition", ""),
                clinvar_gene=row.get("gene_symbol", ""),
                review_stars=int(row.get("review_stars", 0)),
            )
    return variants


def read_gwas(path: str) -> dict[str, MergedVariant]:
    """Read GWAS TSV from Workstream 1B."""
    variants: dict[str, MergedVariant] = {}
    study_counts: dict[str, set] = defaultdict(set)

    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row["variant_id"]
            study_counts[vid].add(row.get("study_id", ""))

            pip = float(row.get("pip", 0))

            if vid in variants:
                existing = variants[vid]
                if pip > existing.max_pip:
                    existing.max_pip = pip
                    existing.gwas_trait = row.get("trait", "")
                    existing.l2g_gene = row.get("l2g_gene", "")
                    existing.l2g_score = float(row.get("l2g_score", 0))
                    existing.consequence = row.get("consequence", "")
                continue

            variants[vid] = MergedVariant(
                variant_id=vid,
                chromosome=row["chromosome"],
                position=int(row["position"]),
                ref=row["ref"],
                alt=row["alt"],
                in_gwas=True,
                max_pip=pip,
                gwas_trait=row.get("trait", ""),
                l2g_gene=row.get("l2g_gene", ""),
                l2g_score=float(row.get("l2g_score", 0)),
                consequence=row.get("consequence", ""),
            )

    # Update study counts
    for vid, studies in study_counts.items():
        if vid in variants:
            variants[vid].gwas_study_count = len(studies)

    return variants


# ---------------------------------------------------------------------------
# 4. MERGE LOGIC
# ---------------------------------------------------------------------------

def merge_variants(clinvar: dict[str, MergedVariant],
                   gwas: dict[str, MergedVariant]) -> list[MergedVariant]:
    """Merge ClinVar and GWAS variants, flagging those in both."""
    merged: dict[str, MergedVariant] = {}

    # Start with ClinVar
    for vid, v in clinvar.items():
        merged[vid] = v

    # Add/merge GWAS
    for vid, gv in gwas.items():
        if vid in merged:
            # Variant exists in both — merge GWAS metadata into ClinVar record
            existing = merged[vid]
            existing.in_gwas = True
            existing.max_pip = gv.max_pip
            existing.gwas_trait = gv.gwas_trait
            existing.l2g_gene = gv.l2g_gene
            existing.l2g_score = gv.l2g_score
            existing.gwas_study_count = gv.gwas_study_count
            if not existing.consequence:
                existing.consequence = gv.consequence
        else:
            merged[vid] = gv

    return list(merged.values())


# ---------------------------------------------------------------------------
# 5. INTERVAL CREATION
# ---------------------------------------------------------------------------

def create_centered_interval(chrom: str, position: int) -> tuple[int, int]:
    """Create a 1Mb interval centered on the variant position.

    Handles chromosome boundary edge cases by shifting the window.
    """
    chrom_len = CHROM_LENGTHS_GRCH38.get(chrom, 250_000_000)
    half_window = WINDOW_SIZE // 2

    start = position - half_window
    end = position + half_window

    # Clamp to chromosome boundaries
    if start < 0:
        start = 0
        end = WINDOW_SIZE
    if end > chrom_len:
        end = chrom_len
        start = max(0, end - WINDOW_SIZE)

    # Ensure exactly 1Mb
    if end - start != WINDOW_SIZE:
        # Chromosome is shorter than 1Mb (shouldn't happen for human, but safe)
        end = min(chrom_len, start + WINDOW_SIZE)

    return start, end


def cluster_variants_into_intervals(
    variants: list[MergedVariant],
) -> list[tuple[str, int, int, list[MergedVariant]]]:
    """Cluster nearby variants to share scoring intervals where possible.

    Strategy: greedily assign variants to intervals. If a variant falls within
    an already-created interval on the same chromosome, reuse it. Otherwise
    create a new centered interval.

    This reduces redundant API calls when multiple variants are close together.
    """
    # Sort by chromosome and position
    variants.sort(key=lambda v: (v.chromosome, v.position))

    # Group by chromosome
    by_chrom: dict[str, list[MergedVariant]] = defaultdict(list)
    for v in variants:
        by_chrom[v.chromosome].append(v)

    intervals: list[tuple[str, int, int, list[MergedVariant]]] = []

    for chrom in sorted(by_chrom.keys()):
        chrom_variants = by_chrom[chrom]
        current_start = None
        current_end = None
        current_group: list[MergedVariant] = []

        for v in chrom_variants:
            pos = v.position

            if (current_start is not None and
                    current_start <= pos <= current_end):
                # Variant fits in current interval
                current_group.append(v)
            else:
                # Save previous interval if exists
                if current_group:
                    intervals.append(
                        (chrom, current_start, current_end, current_group))

                # Create new interval centered on this variant
                current_start, current_end = create_centered_interval(chrom, pos)
                current_group = [v]

            # Assign interval to variant
            v.interval_start = current_start
            v.interval_end = current_end
            v.interval_id = f"{chrom}:{current_start}-{current_end}"

        # Don't forget last group
        if current_group:
            intervals.append(
                (chrom, current_start, current_end, current_group))

    return intervals


# ---------------------------------------------------------------------------
# 6. PRIORITY SCORING
# ---------------------------------------------------------------------------

def compute_priority(v: MergedVariant) -> float:
    """Compute a priority score for variant ordering.

    Higher = should be scored first. Considers:
    - Being in both ClinVar + GWAS (highest priority)
    - ClinVar pathogenic > likely pathogenic > VUS
    - Higher GWAS PIP
    - Higher L2G score
    """
    score = 0.0

    # Dual-source bonus
    if v.in_clinvar and v.in_gwas:
        score += 100.0

    # ClinVar significance
    clinsig_scores = {
        "Pathogenic": 50.0,
        "Likely_pathogenic": 40.0,
        "VUS": 20.0,
        "VUS_conflicting": 15.0,
    }
    score += clinsig_scores.get(v.clinical_significance, 0.0)

    # Review stars (0-4)
    score += v.review_stars * 2.0

    # GWAS PIP (0-1)
    score += v.max_pip * 30.0

    # L2G score (0-1)
    score += v.l2g_score * 10.0

    # Multi-study support
    score += min(v.gwas_study_count, 5) * 2.0

    return score


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def run(clinvar_path: str | None, gwas_path: str | None,
        output_variants: str, output_intervals: str,
        pip_filter: float = 0.0) -> None:
    """Run the merge and interval mapping pipeline."""

    stats = Counter()

    # --- Read inputs ---
    clinvar_variants = {}
    gwas_variants = {}

    if clinvar_path:
        print(f"Reading ClinVar variants from {clinvar_path}...")
        clinvar_variants = read_clinvar(clinvar_path)
        print(f"  Loaded {len(clinvar_variants):,} unique ClinVar variants")
        stats["clinvar_input"] = len(clinvar_variants)

    if gwas_path:
        print(f"Reading GWAS variants from {gwas_path}...")
        gwas_variants = read_gwas(gwas_path)
        print(f"  Loaded {len(gwas_variants):,} unique GWAS variants")
        stats["gwas_input"] = len(gwas_variants)

    if not clinvar_variants and not gwas_variants:
        print("ERROR: No input files provided.")
        sys.exit(1)

    # --- Optional PIP filter for GWAS ---
    if pip_filter > 0 and gwas_variants:
        before = len(gwas_variants)
        gwas_variants = {
            k: v for k, v in gwas_variants.items() if v.max_pip >= pip_filter
        }
        print(f"  PIP filter >= {pip_filter}: {before:,} -> {len(gwas_variants):,}")

    # --- Merge ---
    print(f"\nMerging variants...")
    merged = merge_variants(clinvar_variants, gwas_variants)
    print(f"  Total unique variants after merge: {len(merged):,}")

    both = sum(1 for v in merged if v.in_clinvar and v.in_gwas)
    only_clinvar = sum(1 for v in merged if v.in_clinvar and not v.in_gwas)
    only_gwas = sum(1 for v in merged if v.in_gwas and not v.in_clinvar)
    print(f"  ClinVar only:  {only_clinvar:>10,}")
    print(f"  GWAS only:     {only_gwas:>10,}")
    print(f"  Both sources:  {both:>10,}  <-- highest priority!")
    stats["both_sources"] = both

    # --- Filter to valid chromosomes ---
    valid_chroms = set(CHROM_LENGTHS_GRCH38.keys())
    merged = [v for v in merged if v.chromosome in valid_chroms]
    print(f"  After chromosome filter: {len(merged):,}")

    # --- Compute priority and sort ---
    for v in merged:
        v._priority = compute_priority(v)
    merged.sort(key=lambda v: -v._priority)

    # --- Create intervals and cluster ---
    print(f"\nCreating 1Mb scoring intervals...")
    intervals = cluster_variants_into_intervals(merged)
    print(f"  Unique scoring intervals: {len(intervals):,}")
    print(f"  Variants per interval (mean): "
          f"{len(merged) / max(len(intervals), 1):.1f}")

    # Interval size distribution
    sizes = [len(group) for _, _, _, group in intervals]
    single = sum(1 for s in sizes if s == 1)
    multi = sum(1 for s in sizes if s > 1)
    max_cluster = max(sizes) if sizes else 0
    print(f"  Single-variant intervals: {single:,}")
    print(f"  Multi-variant intervals:  {multi:,} (max cluster: {max_cluster})")

    # --- Write variant TSV ---
    print(f"\nWriting {len(merged):,} variants to {output_variants}...")
    # Re-sort by priority for the output
    merged.sort(key=lambda v: -v._priority)

    with open(output_variants, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "variant_id", "chromosome", "position", "ref", "alt",
            "interval_id", "interval_start", "interval_end",
            "in_clinvar", "in_gwas", "priority_score",
            "clinical_significance", "variant_location",
            "clinvar_condition", "clinvar_gene", "review_stars",
            "max_pip", "gwas_trait", "l2g_gene", "l2g_score",
            "gwas_study_count", "consequence",
        ])
        for v in merged:
            writer.writerow([
                v.variant_id, v.chromosome, v.position, v.ref, v.alt,
                v.interval_id, v.interval_start, v.interval_end,
                v.in_clinvar, v.in_gwas, f"{v._priority:.1f}",
                v.clinical_significance, v.variant_location,
                v.clinvar_condition, v.clinvar_gene, v.review_stars,
                f"{v.max_pip:.6f}", v.gwas_trait,
                v.l2g_gene, f"{v.l2g_score:.4f}",
                v.gwas_study_count, v.consequence,
            ])

    # --- Write interval TSV (for batch scoring) ---
    print(f"Writing {len(intervals):,} intervals to {output_intervals}...")
    # Sort intervals by number of variants (largest clusters first for efficiency)
    intervals.sort(key=lambda x: -len(x[3]))

    with open(output_intervals, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "interval_id", "chromosome", "start", "end",
            "num_variants", "variant_ids",
            "has_clinvar", "has_gwas", "has_both",
            "max_priority",
        ])
        for chrom, start, end, group in intervals:
            interval_id = f"{chrom}:{start}-{end}"
            variant_ids = "|".join(v.variant_id for v in group)
            has_clinvar = any(v.in_clinvar for v in group)
            has_gwas = any(v.in_gwas for v in group)
            has_both = any(v.in_clinvar and v.in_gwas for v in group)
            max_pri = max(v._priority for v in group)

            writer.writerow([
                interval_id, chrom, start, end,
                len(group), variant_ids,
                has_clinvar, has_gwas, has_both,
                f"{max_pri:.1f}",
            ])

    # --- Summary ---
    print("\n" + "=" * 70)
    print("WORKSTREAM 1C SUMMARY")
    print("=" * 70)
    print(f"Input ClinVar variants:       {stats.get('clinvar_input', 0):>10,}")
    print(f"Input GWAS variants:          {stats.get('gwas_input', 0):>10,}")
    print(f"Merged unique variants:       {len(merged):>10,}")
    print(f"  In both sources:            {both:>10,}")
    print(f"Unique 1Mb intervals:         {len(intervals):>10,}")
    print()

    # Chromosome distribution
    chrom_counts = Counter(v.chromosome for v in merged)
    print("Variants per chromosome:")
    for c in sorted(chrom_counts, key=lambda x: (len(x), x)):
        print(f"  {c:6s}  {chrom_counts[c]:>8,}")

    # Top priority variants
    print(f"\nTop 10 highest-priority variants:")
    for v in merged[:10]:
        sources = []
        if v.in_clinvar:
            sources.append(f"ClinVar:{v.clinical_significance}")
        if v.in_gwas:
            sources.append(f"GWAS:PIP={v.max_pip:.3f}")
        gene = v.clinvar_gene or v.l2g_gene or "?"
        print(f"  {v.variant_id:45s} gene={gene:10s} "
              f"pri={v._priority:.0f}  {', '.join(sources)}")

    # API call estimate
    print(f"\nEstimated AlphaGenome API calls for Workstream 2:")
    print(f"  Intervals to score:  {len(intervals):,}")
    print(f"  × ~5 scorer types  = ~{len(intervals) * 5:,} API calls")
    print(f"  (variants sharing an interval are scored together)")

    print(f"\nOutputs:")
    print(f"  Variants: {output_variants}")
    print(f"  Intervals: {output_intervals}")
    print(f"\nReady for Workstream 2 (AlphaGenome scoring)!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge ClinVar + GWAS variants and create 1Mb "
                    "AlphaGenome scoring intervals (Workstream 1C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full merge
  python variant_interval_mapper.py \\
      --clinvar clinvar_cardio_noncoding.tsv \\
      --gwas gwas_cardio_credible_sets.tsv \\
      --output-variants merged_cardio_variants.tsv \\
      --output-intervals scoring_intervals.tsv

  # GWAS only, high-PIP filter
  python variant_interval_mapper.py \\
      --gwas gwas_cardio_credible_sets.tsv \\
      --output-variants gwas_high_pip_variants.tsv \\
      --output-intervals gwas_high_pip_intervals.tsv \\
      --pip-filter 0.1

  # ClinVar only
  python variant_interval_mapper.py \\
      --clinvar clinvar_cardio_noncoding.tsv \\
      --output-variants clinvar_variants_with_intervals.tsv \\
      --output-intervals clinvar_intervals.tsv
        """,
    )
    parser.add_argument("--clinvar", help="ClinVar TSV from Workstream 1A")
    parser.add_argument("--gwas", help="GWAS TSV from Workstream 1B")
    parser.add_argument("--output-variants", "-ov", required=True,
                        help="Output: merged variants with interval assignments")
    parser.add_argument("--output-intervals", "-oi", required=True,
                        help="Output: unique scoring intervals for batch processing")
    parser.add_argument("--pip-filter", type=float, default=0.0,
                        help="Min GWAS PIP to include (default: 0, keep all)")

    args = parser.parse_args()

    if not args.clinvar and not args.gwas:
        parser.error("At least one of --clinvar or --gwas is required")

    run(args.clinvar, args.gwas,
        args.output_variants, args.output_intervals,
        args.pip_filter)


if __name__ == "__main__":
    main()