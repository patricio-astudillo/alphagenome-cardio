# AlphaGenome Cardiovascular Non-Coding Variant Analysis

Code accompanying the manuscript *"Systematic Characterization of Cardiovascular Non-Coding Variants Using AlphaGenome"*.

Data deposit: [Zenodo DOI](https://doi.org/10.5281/zenodo.19590763)

## Overview

This repository contains the analysis pipeline used to score ~200,000 cardiovascular non-coding variants (from ClinVar and Open Targets GWAS) against Google DeepMind's AlphaGenome model, filter the resulting ~3.6 billion score rows to cardiac tissue context, and identify variants with convergent multimodal regulatory signal.

The pipeline spans seven workstreams, each implemented as a standalone Python script:

| Workstream | Script | Description |
|---|---|---|
| WS1 | `pipeline/clinvar_cardio_extract.py` | Filter ClinVar `variant_summary` to cardiovascular non-coding variants |
| WS2 | `pipeline/opentargets_cardio_gwas.py` | Query Open Targets GraphQL for fine-mapped GWAS credible sets |
| WS3 | `pipeline/variant_interval_mapper.py` | Merge ClinVar+GWAS variants, define 1 Mb scoring intervals |
| WS4 | `pipeline/alphag_cardio_scoring.py` | Score variants through AlphaGenome API, checkpointed |
| WS5 | `pipeline/cardiac_filter_analysis.py` | Filter scores to cardiac tissue context, compute modality summaries |
| WS6/7 | `pipeline/ws5_6_7_vignettes_and_deliverables.py` | Direction-of-effect analysis, vignette ISM, final deliverables |
| — | `manuscript/build_tables.py` | Build manuscript tables from WS5/6 output |
| — | `manuscript/figures/fig{1..6}.py` | Build manuscript figures |

## Requirements

- Python 3.11+
- AlphaGenome API access (https://github.com/google-deepmind/alphagenome)
- AlphaGenome API key (free tier sufficient for reproduction at this scale)
- ~500 GB disk for intermediate parquets
- Pipeline runs on CPU; GPU not required (all inference via API)

Install:
```bash
conda env create -f environment.yml
conda activate ag
```

## Running the full pipeline

Set your AlphaGenome API key (one of):
```bash
export ALPHAGENOME_API_KEY=your_key_here
# or
mkdir -p ~/.alphagenome && echo "your_key_here" > ~/.alphagenome/api_key.txt
```

Then run each stage:
```bash
# WS1 — ClinVar
python pipeline/clinvar_cardio_extract.py \
    --input variant_summary.txt.gz \
    --output data/clinvar/clinvar_cardio_extract.tsv

# WS2 — Open Targets
python pipeline/opentargets_cardio_gwas.py \
    --output data/opentargets/gwas_cardio_credible_sets.tsv

# WS3 — Merge intervals
python pipeline/variant_interval_mapper.py \
    --clinvar data/clinvar/clinvar_cardio_extract.tsv \
    --gwas data/opentargets/gwas_cardio_credible_sets.tsv \
    --output-dir data/variant_interval/

# WS4 — Score (long-running, checkpointed)
python pipeline/alphag_cardio_scoring.py \
    --variants data/variant_interval/merged_cardio_variants.tsv \
    --intervals data/variant_interval/scoring_intervals.tsv \
    --output-dir data/scoring_output/

# WS5 — Filter + summarize
python pipeline/cardiac_filter_analysis.py \
    --scores data/scoring_output/all_scores.parquet \
    --variants data/variant_interval/merged_cardio_variants.tsv \
    --output-dir data/ws3_output/

# WS6/7 — Direction of effect, vignettes, rankings
python pipeline/ws5_6_7_vignettes_and_deliverables.py \
    --ws3-dir data/ws3_output/ \
    --output-dir data/ws567_output/

# Manuscript tables
python manuscript/build_tables.py \
    --ws567-dir data/ws567_output/ \
    --clinvar data/clinvar/clinvar_cardio_extract.tsv \
    --merged data/variant_interval/merged_cardio_variants.tsv \
    --output manuscript/tables/

# Manuscript figures
for i in 1 2 3 4 5 6; do
    python manuscript/figures/fig${i}.py --output manuscript/figures/
done
```

## Skipping the expensive steps

WS4 (AlphaGenome scoring) takes days and costs API quota. The scored data is deposited on Zenodo — download it there and skip to WS5:

```bash
wget https://zenodo.org/records/XXXXXXX/files/cardiac_scores.parquet
python pipeline/cardiac_filter_analysis.py \
    --scores cardiac_scores.parquet \
    --variants merged_cardio_variants.tsv \
    --output-dir data/ws3_output/
```

## Citation

If you use this code or data, please cite:

> *<TODO: full citation once manuscript DOI assigned>*

And the underlying AlphaGenome model:

> Google DeepMind AlphaGenome team. AlphaGenome: a deep learning model for sequence-to-function prediction across 1 Mb of genomic context. *Nature* (2026).

## License

Code is released under MIT License (see `LICENSE`).

Note: input data from ClinVar and Open Targets retain their respective licenses. AlphaGenome model weights are not redistributed here; obtain from DeepMind.

## Contact

Questions or issues: please open a GitHub issue.
