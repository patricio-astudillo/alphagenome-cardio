#!/usr/bin/env python3
"""Download cardiovascular GWAS fine-mapped variants from Open Targets Platform.

Produces a clean TSV of fine-mapped GWAS variants ready for AlphaGenome scoring
(Workstream 1B). Parallelized with ThreadPoolExecutor for all pipeline stages.

Usage:
    python opentargets_cardio_gwas.py \
        --output gwas_cardio_credible_sets.tsv \
        [--pip-threshold 0.01] \
        [--max-studies 500] \
        [--workers 8]

Requires: requests (pip install requests)
"""

import argparse
import csv
import json
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. OPEN TARGETS API CONFIGURATION
# ---------------------------------------------------------------------------

OT_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Rate limiting: be polite but allow concurrency
REQUEST_DELAY = 0.05  # reduced per-request delay (concurrency handles throughput)
MAX_RETRIES = 4
DEFAULT_WORKERS = 8

# Thread-safe session pool
_session_local = threading.local()

def _get_session() -> requests.Session:
    """Get a thread-local requests.Session for connection pooling."""
    if not hasattr(_session_local, "session"):
        _session_local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=DEFAULT_WORKERS,
            pool_maxsize=DEFAULT_WORKERS * 2,
            max_retries=0,  # we handle retries ourselves
        )
        _session_local.session.mount("https://", adapter)
    return _session_local.session


# ---------------------------------------------------------------------------
# 2. CARDIOVASCULAR EFO DISEASE IDS
# ---------------------------------------------------------------------------
# These are the top-level EFO terms for cardiovascular traits.
# We use enableIndirect=true to also capture descendant terms.

CARDIO_DISEASES = {
    # Major cardiovascular conditions
    "EFO_0000318": "cardiovascular disease",
    "EFO_0001645": "coronary artery disease",
    "EFO_0000275": "atrial fibrillation",
    "EFO_0003144": "heart failure",
    "EFO_0000537": "hypertension",
    "MONDO_0005267": "heart disease",

    # Cardiomyopathies
    "EFO_0000407": "dilated cardiomyopathy",
    "EFO_0000538": "hypertrophic cardiomyopathy",
    "Orphanet_247": "arrhythmogenic right ventricular cardiomyopathy",

    # Arrhythmias and conduction
    "EFO_0004838": "QT interval",
    "EFO_0004462": "PR interval",
    "EFO_0005741": "cardiac conduction disease",
    "EFO_0004343": "resting heart rate",
    "EFO_0004682": "heart rate variability",

    # Blood pressure quantitative traits
    "EFO_0006335": "systolic blood pressure",
    "EFO_0006340": "diastolic blood pressure",
    "EFO_0005763": "pulse pressure",

    # Lipids / atherosclerosis
    "EFO_0004611": "LDL cholesterol",
    "EFO_0004612": "HDL cholesterol",
    "EFO_0009270": "total cholesterol",
    "EFO_0004530": "triglycerides",
    "EFO_0000612": "myocardial infarction",

    # Stroke / vascular
    "EFO_0000712": "stroke",
    "EFO_0003964": "aortic aneurysm",
    "HP_0001907": "thromboembolism",
    "EFO_0004298": "peripheral arterial disease",

    # Valvular
    "EFO_0004272": "aortic valve stenosis",
    "EFO_0004220": "mitral valve prolapse",
}


# ---------------------------------------------------------------------------
# 3. GRAPHQL QUERIES
# ---------------------------------------------------------------------------

# Step 1: Get studies for a disease (GWAS only)
STUDIES_QUERY = """
query CardioStudies($diseaseId: String!, $page: Pagination!) {
  studies(
    diseaseIds: [$diseaseId],
    enableIndirect: true,
    page: $page
  ) {
    count
    rows {
      id
      studyType
      traitFromSource
      nSamples
      nCases
      nControls
      pubmedId
      publicationFirstAuthor
      publicationDate
      hasSumstats
      diseases {
        id
        name
      }
    }
  }
}
"""

# Step 2: Get credible sets for a study
CREDIBLE_SETS_QUERY = """
query StudyCredibleSets($studyId: String!, $page: Pagination!) {
  credibleSets(
    studyIds: [$studyId],
    studyTypes: [gwas],
    page: $page
  ) {
    count
    rows {
      studyLocusId
      studyId
      chromosome
      position
      pValueMantissa
      pValueExponent
      beta
      standardError
      finemappingMethod
      confidence
      credibleSetlog10BF
      effectAlleleFrequencyFromSource
      variant {
        id
        chromosome
        position
        referenceAllele
        alternateAllele
        rsIds
        mostSevereConsequence {
          id
          label
        }
      }
      l2GPredictions(page: {index: 0, size: 5}) {
        rows {
          score
          target {
            id
            approvedSymbol
          }
        }
      }
    }
  }
}
"""

# Step 3: Get locus variants (individual variants in credible set with PIPs)
LOCUS_QUERY = """
query LocusVariants($studyLocusId: String!, $page: Pagination!) {
  credibleSet(studyLocusId: $studyLocusId) {
    studyLocusId
    studyId
    chromosome
    finemappingMethod
    confidence
    locus(page: $page) {
      count
      rows {
        posteriorProbability
        is95CredibleSet
        is99CredibleSet
        beta
        pValueMantissa
        pValueExponent
        variant {
          id
          chromosome
          position
          referenceAllele
          alternateAllele
          rsIds
          mostSevereConsequence {
            id
            label
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# 4. API HELPER
# ---------------------------------------------------------------------------

def graphql_request(query: str, variables: dict,
                    retries: int = MAX_RETRIES) -> dict | None:
    """Execute a GraphQL request with retry logic and connection pooling."""
    session = _get_session()
    for attempt in range(retries):
        try:
            resp = session.post(
                OT_API_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data:
                    return None
                return data.get("data")
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1) + (threading.current_thread().ident % 3)
                time.sleep(wait)
                continue
            elif resp.status_code in (502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            else:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return None
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# 5. DATA COLLECTION PIPELINE
# ---------------------------------------------------------------------------

# Coding consequences to EXCLUDE
CODING_CONSEQUENCES = {
    "missense_variant", "stop_gained", "stop_lost",
    "frameshift_variant", "inframe_insertion", "inframe_deletion",
    "start_lost", "protein_altering_variant",
    "incomplete_terminal_codon_variant",
}


@dataclass
class GWASVariant:
    variant_id: str            # AlphaGenome format: chr:pos:ref>alt
    ot_variant_id: str         # OT format: chr_pos_ref_alt
    chromosome: str
    position: int
    ref: str
    alt: str
    pip: float                 # posterior inclusion probability
    is_95_credset: bool
    study_id: str
    study_locus_id: str
    trait: str
    disease_efo: str
    consequence: str
    l2g_gene: str
    l2g_score: float
    rs_id: str
    finemapping_method: str
    confidence: str
    beta: float | None
    pvalue_exp: int | None
    effect_allele_freq: float | None


def collect_studies(disease_id: str, disease_name: str,
                    max_per_disease: int = 200) -> list[dict]:
    """Collect all GWAS studies for a disease."""
    studies = []
    page_size = 50
    index = 0

    while True:
        data = graphql_request(STUDIES_QUERY, {
            "diseaseId": disease_id,
            "page": {"index": index, "size": page_size},
        })
        if not data or not data.get("studies"):
            break

        rows = data["studies"]["rows"]
        total = data["studies"]["count"]

        # Filter to GWAS only
        gwas_rows = [r for r in rows if r.get("studyType") == "gwas"]
        studies.extend(gwas_rows)

        index += 1
        if index * page_size >= total or index * page_size >= max_per_disease:
            break
        time.sleep(REQUEST_DELAY)

    return studies


def collect_credible_sets(study_id: str) -> list[dict]:
    """Collect credible sets for a study."""
    credsets = []
    page_size = 100
    index = 0

    while True:
        data = graphql_request(CREDIBLE_SETS_QUERY, {
            "studyId": study_id,
            "page": {"index": index, "size": page_size},
        })
        if not data or not data.get("credibleSets"):
            break

        rows = data["credibleSets"]["rows"]
        total = data["credibleSets"]["count"]
        credsets.extend(rows)

        index += 1
        if index * page_size >= total:
            break
        time.sleep(REQUEST_DELAY)

    return credsets


def collect_locus_variants(study_locus_id: str) -> list[dict]:
    """Collect individual variants in a credible set with PIPs."""
    variants = []
    page_size = 200
    index = 0

    while True:
        data = graphql_request(LOCUS_QUERY, {
            "studyLocusId": study_locus_id,
            "page": {"index": index, "size": page_size},
        })
        if not data or not data.get("credibleSet"):
            break

        cs = data["credibleSet"]
        if not cs.get("locus"):
            break

        rows = cs["locus"].get("rows") or []
        total = cs["locus"]["count"]
        variants.extend([
            {**r, "_study_locus_meta": {
                "studyId": cs.get("studyId"),
                "finemappingMethod": cs.get("finemappingMethod"),
                "confidence": cs.get("confidence"),
            }}
            for r in rows
        ])

        index += 1
        if index * page_size >= total:
            break
        time.sleep(REQUEST_DELAY)

    return variants


def ot_to_alphag_id(ot_id: str) -> str:
    """Convert OT variant ID (1_154453788_C_T) to AlphaGenome format (chr1:154453788:C>T)."""
    parts = ot_id.split("_")
    if len(parts) < 4:
        return ot_id
    chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
    pos = parts[1]
    ref = parts[2]
    alt = "_".join(parts[3:])  # handle multi-part alts
    return f"{chrom}:{pos}:{ref}>{alt}"


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(output_path: str, pip_threshold: float = 0.01,
                 max_studies: int = 500, workers: int = DEFAULT_WORKERS) -> None:
    """Run the full GWAS collection pipeline with parallel execution."""

    stats = Counter()
    all_variants: list[GWASVariant] = []
    seen_study_ids: set[str] = set()
    seen_variants: set[str] = set()  # dedup by variant_id + study_id
    _lock = threading.Lock()

    print("=" * 70)
    print("OPEN TARGETS CARDIOVASCULAR GWAS VARIANT COLLECTION")
    print("=" * 70)
    print(f"Diseases to query: {len(CARDIO_DISEASES)}")
    print(f"PIP threshold: {pip_threshold}")
    print(f"Max studies per disease: {max_studies}")
    print(f"Parallel workers: {workers}")
    print()

    # --- Step 1: Collect studies (sequential — fast, needs dedup) ---
    all_studies: list[tuple[dict, str, str]] = []

    for efo_id, disease_name in CARDIO_DISEASES.items():
        print(f"Querying studies for: {disease_name} ({efo_id})...")
        studies = collect_studies(efo_id, disease_name, max_per_disease=max_studies)
        new_studies = 0
        for s in studies:
            if s["id"] not in seen_study_ids:
                seen_study_ids.add(s["id"])
                all_studies.append((s, efo_id, disease_name))
                new_studies += 1
        print(f"  Found {len(studies)} GWAS studies, {new_studies} new")
        stats["total_studies_raw"] += len(studies)
        time.sleep(REQUEST_DELAY)

    print(f"\nTotal unique GWAS studies: {len(all_studies)}")
    stats["unique_studies"] = len(all_studies)

    # --- Step 2: Collect credible sets (PARALLEL) ---
    print(f"\nCollecting credible sets from {len(all_studies)} studies "
          f"({workers} workers)...")

    all_credsets: list[tuple[dict, str, str, str]] = []
    completed_studies = [0]

    def _fetch_credsets_for_study(args):
        study, efo_id, disease_name = args
        study_id = study["id"]
        trait = study.get("traitFromSource", disease_name)
        credsets = collect_credible_sets(study_id)
        return [(cs, study_id, efo_id, trait) for cs in credsets]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_credsets_for_study, item): item
            for item in all_studies
        }
        for future in as_completed(futures):
            try:
                results = future.result()
                all_credsets.extend(results)
                completed_studies[0] += 1
                if completed_studies[0] % 50 == 0:
                    print(f"  Studies: {completed_studies[0]}/{len(all_studies)} "
                          f"| credible sets: {len(all_credsets):,}")
            except Exception as e:
                completed_studies[0] += 1

    stats["total_credsets"] = len(all_credsets)
    print(f"Total credible sets collected: {len(all_credsets):,}")

    # --- Step 3: Extract variants from credible sets (PARALLEL locus fetching) ---
    # Split into two passes:
    #   Pass A: Extract lead variants from all credible sets (fast, no API calls)
    #   Pass B: Fetch full locus for SuSiE credible sets (slow, parallelized)

    print(f"\nPass A: Extracting lead variants from {len(all_credsets):,} "
          f"credible sets...")

    susie_credsets = []  # credible sets needing locus fetching

    for cs, study_id, efo_id, trait in all_credsets:
        study_locus_id = cs.get("studyLocusId", "")

        l2g_gene = ""
        l2g_score = 0.0
        l2g_preds = cs.get("l2GPredictions", {})
        if l2g_preds and l2g_preds.get("rows"):
            top_l2g = l2g_preds["rows"][0]
            l2g_score = top_l2g.get("score", 0.0)
            target = top_l2g.get("target")
            if target:
                l2g_gene = target.get("approvedSymbol", "")

        confidence = cs.get("confidence", "")
        fm_method = cs.get("finemappingMethod", "")

        lead_variant = cs.get("variant")
        if lead_variant:
            ot_id = lead_variant.get("id", "")
            chrom = lead_variant.get("chromosome", "")
            pos = lead_variant.get("position", 0)
            ref = lead_variant.get("referenceAllele", "")
            alt = lead_variant.get("alternateAllele", "")
            rs_ids = lead_variant.get("rsIds") or []
            rs_id = rs_ids[0] if rs_ids else ""
            consequence_obj = lead_variant.get("mostSevereConsequence")
            consequence = consequence_obj.get("label", "") if consequence_obj else ""

            ag_id = ot_to_alphag_id(ot_id)
            dedup_key = f"{ag_id}|{study_id}"

            if dedup_key not in seen_variants and chrom and pos:
                seen_variants.add(dedup_key)
                all_variants.append(GWASVariant(
                    variant_id=ag_id,
                    ot_variant_id=ot_id,
                    chromosome=f"chr{chrom}" if not chrom.startswith("chr") else chrom,
                    position=pos, ref=ref, alt=alt,
                    pip=1.0, is_95_credset=True,
                    study_id=study_id, study_locus_id=study_locus_id,
                    trait=trait, disease_efo=efo_id,
                    consequence=consequence,
                    l2g_gene=l2g_gene, l2g_score=l2g_score, rs_id=rs_id,
                    finemapping_method=fm_method, confidence=confidence,
                    beta=cs.get("beta"), pvalue_exp=cs.get("pValueExponent"),
                    effect_allele_freq=cs.get("effectAlleleFrequencyFromSource"),
                ))
                stats["lead_variants"] += 1

        # Queue SuSiE credible sets for parallel locus fetching
        if confidence and "susie" in confidence.lower():
            susie_credsets.append((cs, study_id, efo_id, trait,
                                  l2g_gene, l2g_score))

    print(f"  Lead variants extracted: {stats['lead_variants']:,}")
    print(f"\nPass B: Fetching locus variants for {len(susie_credsets):,} "
          f"SuSiE credible sets ({workers} workers)...")

    # Build a dict for fast PIP updates of existing lead variants
    variant_lookup: dict[str, GWASVariant] = {}
    for v in all_variants:
        key = f"{v.variant_id}|{v.study_id}"
        variant_lookup[key] = v

    def _fetch_locus(args):
        """Fetch locus variants for a single credible set. Returns a list of
        (GWASVariant_or_None, dedup_key, pip, is95) tuples."""
        cs, study_id, efo_id, trait, l2g_gene, l2g_score = args
        study_locus_id = cs.get("studyLocusId", "")
        confidence = cs.get("confidence", "")
        fm_method = cs.get("finemappingMethod", "")

        locus_variants = collect_locus_variants(study_locus_id)
        results = []

        for lv in locus_variants:
            pip = lv.get("posteriorProbability", 0.0)
            if pip < pip_threshold:
                results.append(("below_threshold", None, None))
                continue

            var_data = lv.get("variant")
            if not var_data:
                continue

            ot_id = var_data.get("id", "")
            ag_id = ot_to_alphag_id(ot_id)
            dedup_key = f"{ag_id}|{study_id}"

            chrom = var_data.get("chromosome", "")
            pos = var_data.get("position", 0)
            ref = var_data.get("referenceAllele", "")
            alt = var_data.get("alternateAllele", "")
            rs_ids = var_data.get("rsIds") or []
            rs_id = rs_ids[0] if rs_ids else ""
            consequence_obj = var_data.get("mostSevereConsequence")
            consequence = consequence_obj.get("label", "") if consequence_obj else ""
            meta = lv.get("_study_locus_meta", {})

            variant = GWASVariant(
                variant_id=ag_id,
                ot_variant_id=ot_id,
                chromosome=f"chr{chrom}" if not chrom.startswith("chr") else chrom,
                position=pos, ref=ref, alt=alt,
                pip=pip,
                is_95_credset=lv.get("is95CredibleSet", False),
                study_id=study_id, study_locus_id=study_locus_id,
                trait=trait, disease_efo=efo_id,
                consequence=consequence,
                l2g_gene=l2g_gene, l2g_score=l2g_score, rs_id=rs_id,
                finemapping_method=meta.get("finemappingMethod", fm_method),
                confidence=meta.get("confidence", confidence),
                beta=lv.get("beta"), pvalue_exp=lv.get("pValueExponent"),
                effect_allele_freq=None,
            )
            results.append(("variant", dedup_key, variant))

        return results

    completed_loci = [0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_locus, item): item
            for item in susie_credsets
        }
        for future in as_completed(futures):
            completed_loci[0] += 1
            if completed_loci[0] % 200 == 0:
                print(f"  Loci fetched: {completed_loci[0]}/{len(susie_credsets)} "
                      f"| variants: {len(all_variants):,}")
            try:
                results = future.result()
                with _lock:
                    for result_type, dedup_key, variant in results:
                        if result_type == "below_threshold":
                            stats["below_pip_threshold"] += 1
                            continue
                        if dedup_key in seen_variants:
                            # Update PIP for existing lead variant
                            existing = variant_lookup.get(dedup_key)
                            if existing:
                                existing.pip = variant.pip
                                existing.is_95_credset = variant.is_95_credset
                            continue
                        seen_variants.add(dedup_key)
                        variant_lookup[dedup_key] = variant
                        all_variants.append(variant)
                        stats["locus_variants"] += 1
            except Exception:
                pass

    print(f"  Locus variants extracted: {stats['locus_variants']:,}")

    # --- Step 4: Filter to non-coding variants ---
    print(f"\nFiltering to non-coding variants...")
    noncoding_variants = []
    for v in all_variants:
        consequence_label = v.consequence.replace(" ", "_") if v.consequence else ""

        if consequence_label in CODING_CONSEQUENCES:
            stats["skipped_coding"] += 1
            continue

        noncoding_variants.append(v)
        stats["kept_noncoding"] += 1

    # --- Step 5: Sort and write output ---
    noncoding_variants.sort(key=lambda v: (-v.pip, v.chromosome, v.position))

    print(f"\nWriting {len(noncoding_variants)} variants to {output_path}...")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "variant_id", "ot_variant_id", "chromosome", "position",
            "ref", "alt", "pip", "is_95_credset",
            "study_id", "study_locus_id", "trait", "disease_efo",
            "consequence", "l2g_gene", "l2g_score", "rs_id",
            "finemapping_method", "confidence", "beta", "pvalue_exponent",
            "effect_allele_freq",
        ])
        for v in noncoding_variants:
            writer.writerow([
                v.variant_id, v.ot_variant_id, v.chromosome, v.position,
                v.ref, v.alt, f"{v.pip:.6f}", v.is_95_credset,
                v.study_id, v.study_locus_id, v.trait, v.disease_efo,
                v.consequence, v.l2g_gene, f"{v.l2g_score:.4f}", v.rs_id,
                v.finemapping_method, v.confidence,
                f"{v.beta:.6f}" if v.beta is not None else "",
                v.pvalue_exp if v.pvalue_exp is not None else "",
                f"{v.effect_allele_freq:.6f}" if v.effect_allele_freq is not None else "",
            ])

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Diseases queried:             {len(CARDIO_DISEASES):>10}")
    print(f"Unique GWAS studies found:    {stats['unique_studies']:>10,}")
    print(f"Total credible sets:          {stats['total_credsets']:>10,}")
    print(f"Lead variants extracted:      {stats['lead_variants']:>10,}")
    print(f"Locus variants extracted:     {stats['locus_variants']:>10,}")
    print(f"Below PIP threshold:          {stats['below_pip_threshold']:>10,}")
    print(f"Skipped (coding):             {stats['skipped_coding']:>10,}")
    print(f"VARIANTS KEPT (non-coding):   {stats['kept_noncoding']:>10,}")
    print()

    # Breakdown by consequence
    conseq_counts = Counter(v.consequence for v in noncoding_variants)
    print("By VEP consequence:")
    for c, count in conseq_counts.most_common(15):
        print(f"  {c or '(unknown)':40s} {count:>8,}")

    # Breakdown by trait
    trait_counts = Counter(v.trait for v in noncoding_variants)
    print("\nTop 15 traits:")
    for t, count in trait_counts.most_common(15):
        print(f"  {t[:50]:50s} {count:>8,}")

    # Breakdown by L2G gene
    gene_counts = Counter(v.l2g_gene for v in noncoding_variants if v.l2g_gene)
    print("\nTop 20 L2G-predicted target genes:")
    for g, count in gene_counts.most_common(20):
        print(f"  {g:30s} {count:>8,}")

    # PIP distribution
    high_pip = sum(1 for v in noncoding_variants if v.pip >= 0.5)
    med_pip = sum(1 for v in noncoding_variants if 0.1 <= v.pip < 0.5)
    low_pip = sum(1 for v in noncoding_variants if v.pip < 0.1)
    print(f"\nPIP distribution:")
    print(f"  PIP >= 0.5:        {high_pip:>8,}  (high confidence)")
    print(f"  0.1 <= PIP < 0.5:  {med_pip:>8,}  (moderate)")
    print(f"  PIP < 0.1:         {low_pip:>8,}  (low, consider filtering)")

    print(f"\nOutput written to: {output_path}")
    print(f"\nThe variant_id column is directly compatible with:")
    if noncoding_variants:
        print(f"  genome.Variant.from_str('{noncoding_variants[0].variant_id}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download cardiovascular GWAS fine-mapped variants from "
                    "Open Targets Platform for AlphaGenome scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default extraction (PIP >= 0.01)
  python opentargets_cardio_gwas.py --output gwas_cardio_credible_sets.tsv

  # Higher PIP threshold for a smaller, higher-confidence set
  python opentargets_cardio_gwas.py --output gwas_cardio_high_pip.tsv --pip-threshold 0.1
        """,
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output TSV path")
    parser.add_argument(
        "--pip-threshold", type=float, default=0.01,
        help="Minimum posterior inclusion probability (default: 0.01)")
    parser.add_argument(
        "--max-studies", type=int, default=500,
        help="Max studies to fetch per disease (default: 500)")
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")

    args = parser.parse_args()
    run_pipeline(args.output, args.pip_threshold, args.max_studies, args.workers)


if __name__ == "__main__":
    main()