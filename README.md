# Twice Told: claim-level review overlap at ICLR 2017–2026 and NeurIPS 2021–2025

Pipeline, analysis scripts and paper for the study of whether reviewers of the same paper raise the same critiques more often in the LLM era.

## Setup

```
uv sync --all-extras
set -a && . ./.env && set +a    # OPENAI_API_KEY
```

Raw review dumps are expected under `~/data/peer-review/` (ResearchArcade, davidheineman/iclr-2026, JerMa88/ICLR_Peer_Reviews_2026); see `src/build_corpus.py`.

## Pipeline (`src/`)

| Step | Script | Output |
|---|---|---|
| Corpus | `build_corpus.py`, `sample.py` | `data/reviews.parquet`, `papers.parquet`, `sample_reviews.parquet`, `panels*.jsonl` |
| Pass 1 (claims per review) | `MODEL=gpt-6-luna uv run python -I src/extract.py pass1 data/sample_reviews.parquet` | `data/claims_<model>.jsonl` |
| Pass 2 (clusters per panel) | `uv run python -I src/extract.py pass2 data/panels.jsonl` (also `panels_placebo*.jsonl`, `panels_sub3.jsonl`, `panels_blind.jsonl`) | `data/clusters_<model>.jsonl` |
| Outcomes | `outcomes.py <model> [tag]` | `panel_outcomes`, `placebo_outcomes`, `sub3_outcomes`, `review_outcomes` |
| Exposure | `detector.py` (style detector trained on IntelLabs pairs), `areas.py` | `detector_scores.parquet`, `paper_areas.parquet` |
| E1/E2 trends + placebo | `analysis_trends.py <model>` | `reports/year_*`, `placebo_*` |
| E3 within paper | `analysis_within.py <model>`, `analysis_pairs.py <model>` | `reports/within_paper_*`, `pairs_within_*`, `spillover_*` |
| E4 area dose | `analysis_did.py <model>` | `reports/did_*`, `event_study_*` |
| E5 LLM panels | `make_llm_panels.py`, `analysis_llm.py <model>` | `reports/llm_panels_*` |
| E6 cross-paper | `templates.py <model> embed|analyze`, `label_types.py <model>` | `reports/template_by_cell_*`, `critique_types_*`, `type_shift_*` |
| Robustness variants | `make_robust_panels.py`, `make_robust_panels_full.py` (all ICLR 2023/2026 re-cluster panels, no-questions reviews for 12,000 ICLR 2024–2026 papers), `analysis_variants.py <model>` | `reports/variant_*`, `variants_full_*` |
| Validation | `make_blind_set.py`, `validate_blind.py <model>`, `validate_ampere.py`, `make_pairs.py <model>`, `validate_pairs.py`, `make_spec_set.py`, `validate_spec.py` | `reports/validate_blind_*`, `pairs_validation.*`, `spec_validation.txt` |
| Open-weight replication (CPU) | `LOCAL_MODEL_PATH=data/local/<model>.gguf MODEL=qwen3-4b THREADS=1 uv run python -I src/extract.py pass1 data/blind_reviews.parquet` (then pass2 on `panels_blind.jsonl`, `validate_blind.py qwen3-4b`) | `reports/validate_blind_qwen3-4b.txt` |
| ICML 2026 policy comparison (needs an OpenReview login in `.env`: `OPENREVIEW_USERNAME`, `OPENREVIEW_PASSWORD`) | `fetch_icml2026.py`, then the pipeline with `TAG=_icml` and `analysis_icml.py` | `data/icml2026_*.parquet`, `reports/icml_*` |
| Pre-period placebo break, embedding level | `analysis_prebreak.py <model>`, `analysis_embed.py embed|analyze` | `reports/prebreak_*`, `embed_*` |
| Paper assets | `make_tables.py <model>`, `figures.py <model>` | `paper/tables/*.tex`, `paper/figs/*.pdf` |

Reference annotations (blind Opus annotators, protocol in `data/blind/PROTOCOL.md`), the 600 adjudicated claim pairs (`data/pairs/`) and the 300 generic-vs-specific labels (`data/spec/`) are included.

## Paper

```
cd paper && latexmk -pdf -bibtex main.tex
```
