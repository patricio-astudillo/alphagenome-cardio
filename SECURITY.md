# API Keys and Secrets

This repository **never** contains API keys, credentials, or secrets.

## AlphaGenome API key

Required for running `pipeline/alphag_cardio_scoring.py` and the vignette ISM
step in `pipeline/ws5_6_7_vignettes_and_deliverables.py`. Obtain one from
https://github.com/google-deepmind/alphagenome.

Provide the key via one of (in priority order):

1. `--api-key` CLI argument
2. `ALPHAGENOME_API_KEY` environment variable
3. `~/.alphagenome/api_key.txt` file (one line, no whitespace)

## Open Targets GraphQL

No key required — the endpoint is public.

## If you accidentally commit a secret

1. Revoke the key immediately at its issuer.
2. Force-push a clean history: `git filter-branch` or BFG Repo-Cleaner.
3. Note that pushed commits may already be cached by GitHub's API; assume the
   leaked key is permanently compromised and cannot be "recovered."
