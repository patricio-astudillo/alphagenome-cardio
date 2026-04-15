#!/usr/bin/env python3
"""Extract cardiovascular non-coding variants from ClinVar variant_summary.txt.

Produces a clean TSV ready for AlphaGenome variant scoring (Workstream 1A).

Usage:
    python clinvar_cardio_extract.py \
        --input /path/to/variant_summary.txt \
        --output clinvar_cardio_noncoding.tsv \
        [--include-coding]  # optional: also keep coding variants for comparison

Input:  ClinVar variant_summary.txt (tab-delimited, ~4 GB)
        Download from: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/
Output: TSV with columns formatted for AlphaGenome's genome.Variant.from_str()
"""

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. CARDIOVASCULAR CONDITION KEYWORDS
# ---------------------------------------------------------------------------
# Broad-net keyword matching on PhenotypeList and PhenotypeIDS columns.
# Includes MedGen/OMIM/Orphanet IDs for major cardiac Mendelian conditions,
# plus text keywords to catch less-standardized submissions.

CARDIO_KEYWORDS = [
    # Arrhythmia syndromes
    "long qt", "long-qt", "brugada", "short qt", "short-qt",
    "catecholaminergic polymorphic ventricular tachycardia", "cpvt",
    "atrial fibrillation", "atrial flutter",
    "cardiac conduction", "sick sinus", "wolff-parkinson-white",
    "ventricular tachycardia", "ventricular fibrillation",
    "torsade", "arrhythmogenic",
    # Cardiomyopathies
    "hypertrophic cardiomyopathy", "dilated cardiomyopathy",
    "restrictive cardiomyopathy", "cardiomyopathy",
    "arrhythmogenic right ventricular",
    "left ventricular noncompaction", "noncompaction",
    "takotsubo",
    # Vascular / atherosclerotic
    "coronary artery disease", "coronary heart disease",
    "myocardial infarction", "atherosclerosis",
    "aortic aneurysm", "marfan", "loeys-dietz", "ehlers-danlos",  # vascular type
    "familial hypercholesterolemia", "hypercholesterol",
    "familial hyperlipidemia", "dyslipidemia",
    # Heart failure / structural
    "heart failure", "cardiac arrest",
    "congenital heart", "tetralogy", "transposition of the great",
    "ebstein", "atrioventricular septal", "ventricular septal",
    "atrial septal",
    # Blood pressure / hypertension
    "hypertension", "pulmonary arterial hypertension",
    "blood pressure",
    # Valvular
    "mitral valve", "aortic valve", "tricuspid", "bicuspid aortic",
    "valve prolapse", "valve stenosis",
    # Other cardiac
    "cardiac", "cardiogenic", "cardio",
    "qt interval", "pr interval", "qt prolongation",
    "sudden cardiac death", "sudden death",
    "myocarditis", "pericarditis", "endocarditis",
    "kawasaki",
]

# Specific MedGen / OMIM / Orphanet IDs for high-confidence matching
CARDIO_IDS = {
    # Long QT subtypes
    "OMIM:192500", "OMIM:613688", "OMIM:603830", "OMIM:600919",
    "OMIM:613695", "OMIM:613693", "OMIM:611820", "OMIM:618447",
    # Brugada
    "OMIM:601144",
    # HCM
    "OMIM:192600", "OMIM:115197",
    # DCM
    "OMIM:604145", "OMIM:611880", "OMIM:613172",
    # ARVC
    "OMIM:107970", "OMIM:610476", "OMIM:609040",
    # CPVT
    "OMIM:604772", "OMIM:611938",
    # Familial hypercholesterolemia
    "OMIM:143890", "OMIM:603776",
    # Marfan
    "OMIM:154700",
    # CAD
    "OMIM:608320",
    # AF
    "OMIM:608583",
    # MedGen cardiovascular terms
    "MedGen:C0023976",  # Long QT
    "MedGen:C1142166",  # Brugada
    "MedGen:C0007194",  # Cardiomyopathy, dilated
    "MedGen:C0007196",  # Cardiomyopathy, hypertrophic
    "MedGen:C0004238",  # Atrial fibrillation
    "MedGen:C0027051",  # Myocardial infarction
    # Orphanet
    "Orphanet:768",     # Familial hypercholesterolemia
    "Orphanet:217604",  # DCM
    "Orphanet:217569",  # HCM
}


# ---------------------------------------------------------------------------
# 2. NON-CODING CLASSIFICATION LOGIC
# ---------------------------------------------------------------------------
# The variant_summary.txt doesn't have a MolecularConsequence column.
# We infer coding vs non-coding from:
#   - The Name column (HGVS notation)
#   - The Type column
#
# HGVS patterns:
#   c.1234A>G          = coding (exonic)
#   c.1234+56A>G       = intronic (non-coding) — the +/- after a number
#   c.1234-56A>G       = intronic (non-coding)
#   c.*1234A>G         = 3' UTR (non-coding)
#   c.-1234A>G         = 5' UTR (non-coding)
#   n.1234A>G          = non-coding RNA transcript
#   g.1234A>G          = genomic (non-coding context)
#   p.Xxx123Yyy        = protein change (coding)

# Regex patterns for HGVS-based classification
# Intronic: c.DIGITS+DIGITS or c.DIGITS-DIGITS (but NOT c.-DIGITS which is 5'UTR)
RE_INTRONIC = re.compile(r":c\.\d+[\+\-]\d+")
# Also catch deep intronic with * (3'UTR-adjacent intron): c.*DIGITS+DIGITS
RE_INTRONIC_STAR = re.compile(r":c\.\*\d+[\+\-]\d+")
# 5' UTR: c.-DIGITS (negative position, no splice offset)
RE_5UTR = re.compile(r":c\.-\d+[ACGT>]")
# 3' UTR: c.*DIGITS (no splice offset)
RE_3UTR = re.compile(r":c\.\*\d+[ACGT>del]")
# Non-coding transcript
RE_NCRNA = re.compile(r":n\.\d+")
# Protein change in Name = coding
RE_PROTEIN = re.compile(r"\(p\.[A-Z][a-z]{2}\d+")
# Synonymous (p.Xxx123=) or p.(=) — coding but no AA change
RE_SYNONYMOUS = re.compile(r"\(p\.\(?[A-Z][a-z]{2}\d+=\)?|\(p\.\(=\)\)")

# Coding HGVS patterns to EXCLUDE (missense, nonsense, frameshift, inframe)
CODING_PROTEIN_PATTERNS = [
    re.compile(r"\(p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}"),   # missense: p.Arg27Leu
    re.compile(r"\(p\.[A-Z][a-z]{2}\d+Ter"),               # nonsense: p.Gln232Ter
    re.compile(r"\(p\.[A-Z][a-z]{2}\d+fs"),                 # frameshift: p.Leu473fs
    re.compile(r"\(p\.[A-Z][a-z]{2}\d+_"),                  # inframe indel
    re.compile(r"\(p\.[A-Z][a-z]{2}\d+del"),                # inframe del
]


def classify_variant_location(name: str, var_type: str) -> str:
    """Classify a variant as coding or non-coding based on HGVS name and type.

    Returns one of:
        'intronic', '5_prime_utr', '3_prime_utr', 'non_coding_transcript',
        'splice_region', 'synonymous', 'coding', 'intergenic', 'unknown'
    """
    name_lower = name.lower()

    # Structural variants and copy number — skip these
    if var_type in ("copy number gain", "copy number loss",
                    "Microsatellite", "Translocation", "Inversion",
                    "Complex"):
        return "structural"

    # Check for intronic (splice offset notation)
    if RE_INTRONIC.search(name) or RE_INTRONIC_STAR.search(name):
        # Distinguish "near-splice" from "deep intronic"
        # Near-splice: offset <= 10bp from exon boundary
        offsets = re.findall(r":c\.[\d\*]+[\+\-](\d+)", name)
        if offsets:
            max_offset = max(int(o) for o in offsets)
            if max_offset <= 2:
                return "splice_site"       # canonical splice (well-characterized)
            elif max_offset <= 10:
                return "splice_region"     # near-splice (interesting!)
            else:
                return "deep_intronic"     # deep intronic (prime AlphaGenome territory)
        return "intronic"

    # 5' UTR
    if RE_5UTR.search(name):
        return "5_prime_utr"

    # 3' UTR
    if RE_3UTR.search(name) and not RE_INTRONIC_STAR.search(name):
        return "3_prime_utr"

    # Non-coding RNA transcript
    if RE_NCRNA.search(name):
        return "non_coding_transcript"

    # Synonymous (coding position, but no AA change — interesting for splicing)
    if RE_SYNONYMOUS.search(name):
        return "synonymous"

    # Coding with protein change — these are NOT our focus
    for pattern in CODING_PROTEIN_PATTERNS:
        if pattern.search(name):
            return "coding"

    # Intergenic or upstream/downstream (no transcript reference)
    if not re.search(r"NM_|NR_|ENST", name):
        return "intergenic"

    return "unknown"


# ---------------------------------------------------------------------------
# 3. CLINICAL SIGNIFICANCE NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_clinsig(raw_clinsig: str) -> str | None:
    """Normalize ClinVar clinical significance to our three buckets.

    Returns: 'Pathogenic', 'Likely_pathogenic', 'VUS', or None (to skip).
    """
    sig = raw_clinsig.lower().strip()

    if "pathogenic/likely pathogenic" in sig:
        return "Pathogenic"  # conservative: treat as pathogenic
    if "pathogenic" in sig and "likely" not in sig and "conflict" not in sig:
        return "Pathogenic"
    if "likely pathogenic" in sig and "conflict" not in sig:
        return "Likely_pathogenic"
    if "uncertain significance" in sig:
        return "VUS"
    if "conflicting" in sig:
        return "VUS_conflicting"  # keep these — often interesting

    # Skip: Benign, Likely benign, not provided, drug response, etc.
    return None


# ---------------------------------------------------------------------------
# 4. REVIEW STATUS → STAR RATING
# ---------------------------------------------------------------------------

REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion for the individual variant": 0,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
}


def get_review_stars(review_status: str) -> int:
    """Convert ClinVar review status to star rating (0-4)."""
    return REVIEW_STARS.get(review_status.strip().lower(), 0)


# ---------------------------------------------------------------------------
# 5. CARDIOVASCULAR CONDITION MATCHING
# ---------------------------------------------------------------------------

def is_cardiovascular(phenotype_list: str, phenotype_ids: str) -> bool:
    """Check if variant is associated with a cardiovascular condition."""
    text = (phenotype_list + " " + phenotype_ids).lower()

    # Check keyword matches
    for kw in CARDIO_KEYWORDS:
        if kw in text:
            return True

    # Check specific IDs
    for cid in CARDIO_IDS:
        if cid in phenotype_ids:
            return True

    return False


# ---------------------------------------------------------------------------
# 6. MAIN EXTRACTION
# ---------------------------------------------------------------------------

@dataclass
class ExtractedVariant:
    variant_id: str           # chr:pos:ref>alt (AlphaGenome format)
    chromosome: str
    position: int
    ref: str
    alt: str
    clinical_significance: str
    variant_location: str     # intronic, 5_prime_utr, etc.
    condition: str
    gene_symbol: str
    allele_id: int
    variation_id: int
    review_stars: int
    rs_id: str
    hgvs_name: str


def process_clinvar(input_path: str, output_path: str,
                    include_coding: bool = False) -> None:
    """Process ClinVar variant_summary.txt and write filtered TSV."""

    stats = Counter()
    variants_seen = set()  # dedup by variant_id + condition
    results: list[ExtractedVariant] = []

    # Non-coding categories to keep
    noncoding_categories = {
        "deep_intronic", "splice_region", "intronic",
        "5_prime_utr", "3_prime_utr",
        "non_coding_transcript", "intergenic",
        "synonymous",  # keep: may affect splicing
    }
    if include_coding:
        noncoding_categories.add("coding")
        noncoding_categories.add("splice_site")

    print(f"Reading {input_path} ...")
    print(f"This is a large file (~4 GB), processing line by line.\n")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i % 500_000 == 0 and i > 0:
                print(f"  Processed {i:,} rows | "
                      f"cardio hits: {stats['cardio_match']:,} | "
                      f"non-coding kept: {stats['kept']:,}")

            stats["total"] += 1

            # --- Filter 1: GRCh38 only ---
            if row.get("Assembly") != "GRCh38":
                stats["skipped_assembly"] += 1
                continue

            # --- Filter 2: Must have VCF-style coordinates ---
            pos_vcf = row.get("PositionVCF", "").strip()
            ref_vcf = row.get("ReferenceAlleleVCF", "").strip()
            alt_vcf = row.get("AlternateAlleleVCF", "").strip()
            chrom = row.get("Chromosome", "").strip()

            if not pos_vcf or pos_vcf == "-1" or not ref_vcf or ref_vcf == "na":
                stats["skipped_no_coords"] += 1
                continue
            if not alt_vcf or alt_vcf == "na" or alt_vcf == "-":
                stats["skipped_no_coords"] += 1
                continue
            if chrom not in [str(c) for c in range(1, 23)] + ["X", "Y"]:
                stats["skipped_chrom"] += 1
                continue

            # --- Filter 3: Clinical significance ---
            raw_sig = row.get("ClinicalSignificance", "")
            norm_sig = normalize_clinsig(raw_sig)
            if norm_sig is None:
                stats["skipped_clinsig"] += 1
                continue

            # --- Filter 4: Cardiovascular condition ---
            phenotype_list = row.get("PhenotypeList", "")
            phenotype_ids = row.get("PhenotypeIDS", "")
            if not is_cardiovascular(phenotype_list, phenotype_ids):
                stats["skipped_not_cardio"] += 1
                continue

            stats["cardio_match"] += 1

            # --- Filter 5: Non-coding classification ---
            name = row.get("Name", "")
            var_type = row.get("Type", "")
            location = classify_variant_location(name, var_type)

            if location not in noncoding_categories:
                stats[f"skipped_location_{location}"] += 1
                continue

            # --- Build variant in AlphaGenome format ---
            chr_str = f"chr{chrom}"
            try:
                position = int(pos_vcf)
            except ValueError:
                stats["skipped_bad_pos"] += 1
                continue

            variant_id = f"{chr_str}:{position}:{ref_vcf}>{alt_vcf}"

            # Dedup: same variant + same condition bucket
            dedup_key = (variant_id, norm_sig)
            if dedup_key in variants_seen:
                stats["skipped_dup"] += 1
                continue
            variants_seen.add(dedup_key)

            # --- Extract metadata ---
            gene = row.get("GeneSymbol", "").strip()
            if gene == "-" or gene == "":
                gene = "intergenic"

            review_stars = get_review_stars(row.get("ReviewStatus", ""))
            rs_id = row.get("RS# (dbSNP)", "").strip()
            if rs_id == "-1" or rs_id == "-":
                rs_id = ""
            else:
                rs_id = f"rs{rs_id}"

            allele_id = int(row.get("#AlleleID", row.get("AlleleID", 0)))
            variation_id = int(row.get("VariationID", 0))

            # Truncate condition to first entry for readability
            condition = phenotype_list.split("|")[0].strip()

            results.append(ExtractedVariant(
                variant_id=variant_id,
                chromosome=chr_str,
                position=position,
                ref=ref_vcf,
                alt=alt_vcf,
                clinical_significance=norm_sig,
                variant_location=location,
                condition=condition,
                gene_symbol=gene,
                allele_id=allele_id,
                variation_id=variation_id,
                review_stars=review_stars,
                rs_id=rs_id,
                hgvs_name=name[:200],  # truncate very long HGVS names
            ))
            stats["kept"] += 1

    # --- Sort: pathogenic first, then by chromosome and position ---
    sig_order = {"Pathogenic": 0, "Likely_pathogenic": 1,
                 "VUS": 2, "VUS_conflicting": 3}
    results.sort(key=lambda v: (sig_order.get(v.clinical_significance, 9),
                                v.chromosome, v.position))

    # --- Write output TSV ---
    print(f"\nWriting {len(results):,} variants to {output_path} ...")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "variant_id", "chromosome", "position", "ref", "alt",
            "clinical_significance", "variant_location", "condition",
            "gene_symbol", "allele_id", "variation_id", "review_stars",
            "rs_id", "hgvs_name",
        ])
        for v in results:
            writer.writerow([
                v.variant_id, v.chromosome, v.position, v.ref, v.alt,
                v.clinical_significance, v.variant_location, v.condition,
                v.gene_symbol, v.allele_id, v.variation_id, v.review_stars,
                v.rs_id, v.hgvs_name,
            ])

    # --- Print summary statistics ---
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total rows in file:           {stats['total']:>10,}")
    print(f"Skipped (not GRCh38):         {stats['skipped_assembly']:>10,}")
    print(f"Skipped (no coordinates):     {stats['skipped_no_coords']:>10,}")
    print(f"Skipped (non-standard chrom): {stats['skipped_chrom']:>10,}")
    print(f"Skipped (benign/other sig):   {stats['skipped_clinsig']:>10,}")
    print(f"Skipped (not cardiovascular): {stats['skipped_not_cardio']:>10,}")
    print(f"Cardiovascular matches:       {stats['cardio_match']:>10,}")
    print()

    # Location breakdown of skipped cardio variants
    print("Location breakdown of cardiovascular variants (skipped):")
    for key in sorted(stats):
        if key.startswith("skipped_location_"):
            loc = key.replace("skipped_location_", "")
            print(f"  {loc:30s} {stats[key]:>8,}")
    print()

    print(f"Duplicates removed:           {stats.get('skipped_dup', 0):>10,}")
    print(f"VARIANTS KEPT:                {stats['kept']:>10,}")
    print()

    # Breakdown by clinical significance
    sig_counts = Counter(v.clinical_significance for v in results)
    print("By clinical significance:")
    for sig, count in sig_counts.most_common():
        print(f"  {sig:30s} {count:>8,}")

    # Breakdown by variant location
    loc_counts = Counter(v.variant_location for v in results)
    print("\nBy variant location:")
    for loc, count in loc_counts.most_common():
        print(f"  {loc:30s} {count:>8,}")

    # Breakdown by gene (top 20)
    gene_counts = Counter(v.gene_symbol for v in results)
    print("\nTop 20 genes:")
    for gene, count in gene_counts.most_common(20):
        print(f"  {gene:30s} {count:>8,}")

    print(f"\nOutput written to: {output_path}")
    print(f"\nThe variant_id column is directly compatible with:")
    print(f"  genome.Variant.from_str('{results[0].variant_id}')"
          if results else "  (no variants extracted)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract cardiovascular non-coding variants from ClinVar "
                    "for AlphaGenome scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (non-coding only)
  python clinvar_cardio_extract.py \\
      --input variant_summary.txt \\
      --output clinvar_cardio_noncoding.tsv

  # Include coding variants for benchmarking comparison
  python clinvar_cardio_extract.py \\
      --input variant_summary.txt \\
      --output clinvar_cardio_all.tsv \\
      --include-coding
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to ClinVar variant_summary.txt (uncompressed or .gz)")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output TSV path")
    parser.add_argument(
        "--include-coding", action="store_true",
        help="Also include coding variants (for benchmarking)")

    args = parser.parse_args()

    # Handle gzipped input
    input_path = args.input
    if input_path.endswith(".gz"):
        import gzip
        import shutil
        import tempfile
        print(f"Decompressing {input_path} ...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with gzip.open(input_path, "rb") as f_in:
            shutil.copyfileobj(f_in, tmp)
        tmp.close()
        input_path = tmp.name
        print(f"Decompressed to {input_path}")

    process_clinvar(input_path, args.output, args.include_coding)


if __name__ == "__main__":
    main()