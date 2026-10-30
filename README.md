# Twice Told: claim-level review overlap at ICLR 2017–2026 and NeurIPS 2021–2025

Pipeline, analysis scripts and paper for the study of whether reviewers of the same paper raise the same critiques more often in the LLM era.

## Setup

```
uv sync --all-extras
set -a && . ./.env && set +a    # OPENAI_API_KEY
```

Raw review dumps are expected under `~/data/peer-review/` (ResearchArcade, davidheineman/iclr-2026, JerMa88/ICLR_Peer_Reviews_2026); see `src/corpus.py`.

## Pipeline (`src/`)

Run each line as `uv run python -I src/<line>`, in this order. `# API` marks paid OpenAI calls (load `.env` first); every
command has `--help`. The model defaults to `gpt-6-luna`.

```
fetch.py scipost                                                  # also f1000, plos, bmc, copernicus -> data/other/
fetch.py icml --year 2025                                         # and --year 2026; needs OPENREVIEW_USERNAME/PASSWORD
corpus.py ml                                                      # data/reviews.parquet, papers.parquet
corpus.py other                                                   # data/other_reviews.parquet, other_papers.parquet
panels.py ml                                                      # sample_reviews.parquet, panels.jsonl, panels_placebo.jsonl
corpus.py areas                                                   # API; paper_areas.parquet
panels.py robust                                                  # panels_placebo_area, sub3(_full), noq(_full) + noq reviews
panels.py icml                                                    # icml_pass1.parquet, icml_sample.parquet, panels_icml.jsonl
panels.py other                                                   # other_pass1.parquet, panels_other(_placebo).jsonl
validate.py blind_set                                             # data/blind/, panels_blind.jsonl
extract.py pass1 data/sample_reviews.parquet                      # API; data/claims_gpt-6-luna.jsonl
extract.py pass2 data/panels.jsonl                                # API; data/clusters_gpt-6-luna.jsonl (same for the 5 lines below)
extract.py pass2 data/panels_placebo.jsonl                        # API
extract.py pass2 data/panels_placebo_area.jsonl                   # API
extract.py pass2 data/panels_sub3.jsonl                           # API
extract.py pass2 data/panels_sub3_full.jsonl                      # API
extract.py pass2 data/panels_blind.jsonl                          # API
extract.py pass1 data/noq_reviews.parquet --tag _noq              # API
extract.py pass1 data/noq_full_reviews.parquet --tag _noq         # API
extract.py pass2 data/panels_noq.jsonl --tag _noq                 # API
extract.py pass2 data/panels_noq_full.jsonl --tag _noq            # API
extract.py pass1 data/wd_reviews.parquet --tag _wd                # API; withdrawn ICLR 2026 papers (no generator script kept)
extract.py pass2 data/panels_wd.jsonl --tag _wd                   # API
extract.py pass1 data/sol_reviews.parquet --model gpt-6.1-sol     # API; second-model check (no generator script kept)
extract.py pass2 data/panels_sol.jsonl --model gpt-6.1-sol        # API
extract.py pass1 data/icml_pass1.parquet --tag _icml              # API
extract.py pass2 data/panels_icml.jsonl --tag _icml               # API
extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general            # API
extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general             # API
extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general     # API
extract.py pass1 data/blind_reviews.parquet --model qwen3-4b --threads 1 --local_model_path data/local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
extract.py pass2 data/panels_blind.jsonl --model qwen3-4b --threads 1 --local_model_path data/local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
outcomes.py panels                                                # panel_, placebo_, sub3_, review_outcomes_gpt-6-luna.parquet
outcomes.py panels --tag _noq                                     # *_outcomes_gpt-6-luna_noq.parquet
outcomes.py panels --tag _wd                                      # *_outcomes_gpt-6-luna_wd.parquet
outcomes.py panels --model gpt-6.1-sol                            # *_outcomes_gpt-6.1-sol.parquet
outcomes.py panels --tag _icml                                    # icml_panel_outcomes_gpt-6-luna.parquet
outcomes.py panels --tag _other                                   # other_panel_outcomes_gpt-6-luna.parquet
outcomes.py pairs                                                 # data/pairs_gpt-6-luna.parquet (review pairs, all sources)
outcomes.py standardize                                           # reports/match_std.csv (length- and size-standardized match rate)
ai.py reference                                                   # API; data/ai_reference.jsonl
ai.py wordshare                                                   # reports/llm_share.csv
ai.py wordshare_fields                                            # reports/llm_share_all.csv
ai.py cocodet data/cocodet_input.parquet main                     # also cocodet_input_icml/_other.parquet; needs the detect extra
ai.py style                                                       # data/detector_scores.parquet, reports/detector_by_cell.csv
between.py templates                                              # claims_table, critique_types, template_by_cell, type_shift
between.py embed                                                  # reports/embed_gpt-6-luna.csv
validate.py pairs_set                                             # data/pairs/ inputs and key
validate.py spec_set                                              # data/spec/ inputs and key
validate.py blind                                                 # pipeline vs blind Opus reference
validate.py blind --model qwen3-4b                                # open-weight replication
validate.py pairs                                                 # reports/pairs_validation.csv
validate.py spec                                                  # generic-vs-specific agreement
validate.py ampere                                                # API (embeddings); AMPERE segmentation check
validate.py variants                                              # reports/variant_{sub3,noq,wd}_gpt-6-luna.csv
figures.py table                                                  # reports/by_year.csv, paper/tables/by_year.tex
figures.py all                                                    # paper/figs/short_*.pdf
```

The sample on disk was drawn before `corpus.py ml` dropped copied reviews. `panels.py ml --data data/backup_predup` (the
earlier inputs) redraws `sample_reviews.parquet` and `panels_placebo.jsonl` exactly; `panels.jsonl` on disk is that
draw with the copied reviews removed from each panel.

Reference annotations (blind Opus annotators, protocol in `data/blind/PROTOCOL.md`), the 600 adjudicated claim pairs (`data/pairs/`) and the 300 generic-vs-specific labels (`data/spec/`) are included.

## Paper

```
cd paper && latexmk -pdf -bibtex main.tex
```
