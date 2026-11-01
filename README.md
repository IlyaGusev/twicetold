# Twice Told: claim-level review overlap at ICLR 2017–2026 and NeurIPS 2021–2025

Pipeline, analysis scripts and paper for the study of whether reviewers of the same paper raise the same critiques more often in the LLM era.

## Setup

```
uv sync --all-extras
```

Keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENREVIEW_USERNAME/PASSWORD`) go in `.env` at the repo root; `src/settings.py`
reads them with every path, the default model, the reasoning efforts, the price table and the OpenRouter routes. Any
field can be overridden by an environment variable of the same name. Prompts are jinja2 templates in `src/prompts/`.
Raw review dumps (ResearchArcade, davidheineman/iclr-2026, JerMa88/ICLR_Peer_Reviews_2026) are expected under
`RAW_ML_REVIEWS_DIR` in `src/settings.py`; `src/corpus.py` reads them.

## Pipeline (`src/`)

Run each line as `uv run python -I src/<line>`, in this order. `# API` marks paid model calls; every command has `--help`.
The model defaults to `gpt-6-luna-high` (gpt-6-luna through OpenRouter). Each model short name has fixed reasoning
efforts in `REASONING_EFFORTS` in `src/settings.py`: gpt-6-luna, gpt-6.1-sol, glm-5.3-flash and deepseek-v4.1-flash use
low in pass 1 and medium in pass 2, and gpt-6-luna-high uses high in both passes. A new effort needs a new short name.
Every claims and clusters row records its effort, and a resume into a file made at another effort stops.

Outputs carry the model short name (`<model>` below). The paper uses `gpt-6-luna`: to remake its reports, tables and
figures, pass `--model gpt-6-luna` or run `scripts/post.sh gpt-6-luna`. `(no generator script kept)` marks inputs that
no command here writes; they are not in the repository. `# done` marks one-time steps whose key files the existing
Claude Opus annotations depend on: these commands stop if the key exists.

```
fetch.py scipost                                                  # also f1000, plos, bmc, copernicus -> data/other/
fetch.py icml --year 2025                                         # and --year 2026; needs OPENREVIEW_USERNAME/PASSWORD
corpus.py ml                                                      # data/reviews.parquet, papers.parquet
corpus.py other                                                   # data/other_reviews.parquet, other_papers.parquet
panels.py ml                                                      # sample_reviews.parquet, panels.jsonl, panels_placebo.jsonl
corpus.py openreview_areas                                        # data/openreview_areas.parquet (ICLR 2023-2025 primary areas); needs OPENREVIEW_USERNAME/PASSWORD
corpus.py areas                                                   # API; reads openreview_areas.parquet; paper_areas.parquet (area, area_final), abstract_emb.npy
panels.py robust                                                  # panels_placebo_area, sub3(_full), noq(_full) + noq reviews
panels.py icml                                                    # icml_pass1.parquet, icml_sample.parquet, panels_icml.jsonl
panels.py other                                                   # other_pass1.parquet, panels_other(_placebo).jsonl
validate.py blind_set                                             # keeps an existing data/blind/blind_key.json; writes panels_blind.jsonl and blind_reviews.parquet from it
```

`scripts/extract_all.sh MODEL [PASS_BUDGET_USD=300]` runs the 21 passes of the next block for one model as parallel
chains (it takes the reasoning efforts from `REASONING_EFFORTS`; it has no effort argument). Each pass is its own
extract.py process with its own cap of PASS_BUDGET_USD, so one run can spend up to 21 times PASS_BUDGET_USD. Each
restart counts from zero, and calls already running finish after the cap. Each pass resumes from its output file, so a
rerun after a failure does only the missing items. Logs go to `logs/extract_<model>/<chain>.log`; at the end the script
prints `SOME CHAIN FAILED` and exits 1 if a pass exited with an error or a signal, skipped panels or reached its cap.

```
extract.py pass1 data/sample_reviews.parquet                      # API; data/claims_<model>.jsonl
extract.py pass1 data/llm_reviews.parquet --tag _llm              # API; LLM-written reviews; llm_reviews.parquet, panels_llm.jsonl (no generator script kept)
extract.py pass2 data/panels.jsonl                                # API; data/clusters_<model>.jsonl (same for the 5 lines below)
extract.py pass2 data/panels_placebo.jsonl                        # API
extract.py pass2 data/panels_placebo_area.jsonl                   # API
extract.py pass2 data/panels_sub3.jsonl                           # API
extract.py pass2 data/panels_sub3_full.jsonl                      # API
extract.py pass2 data/panels_blind.jsonl                          # API
extract.py pass2 data/panels_llm.jsonl --tag _llm --claims_files claims_gpt-6-luna-high_llm.jsonl,claims_gpt-6-luna-high.jsonl  # API; after both pass-1 lines above; no step below reads the _llm outputs
extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general            # API
extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general             # API
extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general     # API
extract.py pass1 data/noq_reviews.parquet --tag _noq              # API
extract.py pass1 data/noq_full_reviews.parquet --tag _noq         # API
extract.py pass2 data/panels_noq.jsonl --tag _noq                 # API
extract.py pass2 data/panels_noq_full.jsonl --tag _noq            # API
extract.py pass1 data/icml_pass1.parquet --tag _icml              # API
extract.py pass2 data/panels_icml.jsonl --tag _icml               # API
extract.py pass1 data/wd_reviews.parquet --tag _wd                # API; withdrawn ICLR 2026 papers; wd_reviews.parquet, panels_wd.jsonl (no generator script kept)
extract.py pass2 data/panels_wd.jsonl --tag _wd                   # API
extract.py pass1 data/ampere_reviews.parquet --tag _ampere        # API; for validate.py ampere; ampere_reviews.parquet (no generator script kept)
```

The second model and the open-weight model run outside `extract_all.sh`. `validate.py instruments` also reads
`claims_<m>.jsonl` and `clusters_<m>.jsonl` of every model in `INSTRUMENT_MODELS` on these second-model panels and on
`panels_blind.jsonl`; each is the same two passes with `--model <m>`.

```
extract.py pass1 data/sol_reviews.parquet --model gpt-6.1-sol     # API; second-model check; sol_reviews.parquet, panels_sol.jsonl (no generator script kept)
extract.py pass2 data/panels_sol.jsonl --model gpt-6.1-sol        # API
panels.py sol                                                     # already run once: panels_sol2.jsonl, sol2_reviews.parquet; draws from panel_outcomes_gpt-6-luna.parquet, so a rerun after an outcomes change draws other panels
extract.py pass1 data/sol2_reviews.parquet --model gpt-6.1-sol    # API
extract.py pass2 data/panels_sol2.jsonl --model gpt-6.1-sol       # API
extract.py pass1 data/blind_reviews.parquet --model qwen3-4b --threads 1 --local_model_path data/local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
extract.py pass2 data/panels_blind.jsonl --model qwen3-4b --threads 1 --local_model_path data/local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
outcomes.py panels --model gpt-6.1-sol                            # *_outcomes_gpt-6.1-sol.parquet (for validate.py robust and drift); reads data/review_cdates.parquet (no generator script kept)
outcomes.py pairs --model gpt-6.1-sol                             # data/pairs_gpt-6.1-sol.parquet (for validate.py rq1_evidence)
```

AI-likeness measures and the reference checks (all before `scripts/post.sh`, which reads their outputs):

```
ai.py reference                                                   # API; data/ai_reference_<model>.jsonl (default settings.AI_REFERENCE_MODEL = gpt-6-luna, effort from settings.REASONING_EFFORTS)
ai.py wordshare                                                   # reports/llm_share.csv; reads data/intel_reviews.parquet (no generator script kept)
ai.py wordshare_fields                                            # reports/llm_share_all.csv; reads data/ai_reference_gpt-6-luna.jsonl
ai.py cocodet data/cocodet_input.parquet main                     # data/cocodet_main.jsonl; needs the detect extra; cocodet_input*.parquet (no generator script kept)
ai.py cocodet data/cocodet_input_icml.parquet icml                # data/cocodet_icml.jsonl
ai.py cocodet data/cocodet_input_other.parquet other              # data/cocodet_other.jsonl
ai.py cocodet data/cocodet_input_abstracts.parquet abstracts      # data/cocodet_abstracts.jsonl (for validate.py paper_side)
ai.py style                                                       # data/detector_scores.parquet, reports/detector_by_cell.csv; reads intel_reviews.parquet, gptzero.parquet (no generator script kept)
validate.py pairs_set                                             # done: data/pairs/ is the key of the included Opus judgments; refuses to overwrite it
validate.py spec_set                                              # done: data/spec/ is the key of the included labels; refuses to overwrite it
validate.py blind --model qwen3-4b                                # open-weight replication against the blind Opus reference (data/blind/)
validate.py pairs                                                 # reports/pairs_validation.csv
validate.py spec                                                  # generic-vs-specific agreement
validate.py ampere                                                # API (embeddings); AMPERE segmentation check; data/ampere_full.parquet, ampere_reviews.parquet (no generator script kept)
validate.py flag_set                                              # done: data/blind3/ (detector-contrast set, not in the repository); refuses to overwrite it
validate.py flags                                                 # reports/flags_blind3.csv
validate.py swap_detector                                         # reports/swap_detector.csv; reads data/blind4/, the Opus style-swap rewrites (no generator script kept)
validate.py swap                                                  # reports/swap_blind4.csv
validate.py pair_audit_set                                        # done: data/blind5/ (pair-audit set, not in the repository); refuses to overwrite it
validate.py pair_audit                                            # reports/pair_audit_blind5.csv
```

`scripts/post.sh MODEL [OUT_DIR]` runs the next block with `--model MODEL`, in this order. It stops at the first step
that fails and prints that step's exit status. Without OUT_DIR it writes the paper's tables and figures
(`paper/tables/`, `paper/figs/`); with OUT_DIR it writes them to `OUT_DIR/tables` and `OUT_DIR/figs`. The model-named
reports go to `reports/` in both cases.

```
outcomes.py panels                                                # panel_, placebo_, sub3_, review_outcomes_<model>.parquet
outcomes.py panels --tag _noq                                     # *_outcomes_<model>_noq.parquet
outcomes.py panels --tag _wd                                      # *_outcomes_<model>_wd.parquet
outcomes.py panels --tag _icml                                    # icml_panel_outcomes_<model>.parquet
outcomes.py panels --tag _other                                   # other_panel_outcomes_<model>.parquet
outcomes.py pairs                                                 # data/pairs_<model>.parquet (review pairs, all sources)
outcomes.py standardize                                           # reports/match_std_<model>.csv (length- and size-standardized match rate)
outcomes.py withdrawals                                           # reports/match_std_withdrawals_<model>.csv
between.py claims_table                                           # data/claims_table_<model>.parquet; keeps emb_row of an existing file
between.py embed                                                  # reports/embed_<model>.csv; API only with --reembed (otherwise reads data/review_emb.npy)
validate.py detector_placebo                                      # reports/detector_placebo_<model>.csv
validate.py rq1_evidence                                          # reports/rq1_evidence_<model>.csv
validate.py variants                                              # reports/variant_{sub3,noq,wd}_<model>.csv
validate.py blind                                                 # pipeline vs blind Opus reference
validate.py reviewer_fe                                           # reports/reviewer_fe_f1000_<model>.csv
validate.py paper_side                                            # reports/paper_side_<model>.csv
validate.py instruments                                           # reports/instruments_<model>.csv (this model first, then INSTRUMENT_MODELS)
validate.py robust                                                # reports/robust_<model>.csv, paper/tables/robust.tex (--tex with OUT_DIR)
figures.py table                                                  # reports/by_year_<model>.csv, paper/tables/by_year_ml.tex, by_year_journals.tex
figures.py all                                                    # paper/figs/fig1_ai.pdf ... figA_*.pdf
```

Steps outside `post.sh`:

```
between.py templates                                              # API with --reembed (otherwise needs data/claim_emb_<model>.npy); claims_table, critique_types, template_by_cell, type_shift
validate.py drift_set                                             # done: data/blind2/ (drift set, not in the repository); refuses to overwrite it
validate.py drift                                                 # reports/drift_blind2.csv; reads panel_outcomes_gpt-6-luna and _gpt-6.1-sol
```

The sample on disk was drawn before `corpus.py ml` dropped copied reviews. `panels.py ml --data data/backup_predup` (the
earlier inputs) redraws `sample_reviews.parquet`, `panels.jsonl` and `panels_placebo.jsonl` exactly. `panels.jsonl`
leaves out reviews that are no longer in `data/reviews.parquet` and papers with fewer than 3 reviews left.
`panels.py robust` leaves out sub3 panels that hold a review outside `panels.jsonl`.

Reference annotations (blind Opus annotators, protocol in `data/blind/PROTOCOL.md`), the 600 adjudicated claim pairs
(`data/pairs/`) and the 300 generic-vs-specific labels (`data/spec/`) are included.

## Paper

```
cd paper && latexmk -pdf -bibtex main.tex
```

The paper reads `paper/tables/by_year_ml.tex`, `by_year_journals.tex`, `robust.tex` and the figures in `paper/figs/`;
`scripts/post.sh gpt-6-luna` without OUT_DIR writes them.
