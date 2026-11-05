# Twice Told

Code and paper for measuring whether reviewers repeat the same critiques in the LLM era.

## Setup

```
uv sync --locked --all-extras
```

API keys and OpenReview credentials go in `.env`. `src/settings.py` holds paths, model routes, prices and reasoning
efforts. The paper uses `gpt-6-luna-high`, with high effort in both extraction passes. Prompts live in
`src/prompts/`.

## Measure

For each pair of reviews, count the distinct problem groups they share and divide by the smaller problem set.
A score of 1 means that the smaller set adds no problem beyond the other review. Duplicate claims in one group
count once. A pair with an empty review scores 0. Average all pairs within each paper, then average papers within
each year.
Every paper has equal weight.

`src/metrics.py` defines the measure. The same function scores panels, pair tables and cross-paper controls.

## Data and extraction

Run commands as `uv run python -I src/<module>.py <command>`. Raw ML dumps live at `RAW_ML_REVIEWS_DIR`; journal
sources live at `RAW_JOURNAL_REVIEWS_DIR`, both set in `src/settings.py`.

1. Fetch source data with `fetch.py`.
2. Build the ML review tables with `corpus.py ml`.
3. Run `scripts/inputs.sh`. It builds the journal tables (`corpus.py other`), parses the ICML downloads
   (`fetch.py icml`), and builds the ICML, journal and robustness inputs (`panels.py icml`, `other`,
   `robust`, `nocq`) and the MELD input (`meld.py input`). Git does not store these inputs. `extract_all.sh` and `post.sh` run `inputs.sh` when the
   inputs are missing. No step calls a paid API.
4. Score the MELD input on a GPU with `meld.py scores`. It writes `data/meld_scores.jsonl`; no chain script runs it.
5. Run `scripts/extract_all.sh gpt-6-luna-high` for extraction and grouping. This step calls paid APIs. Each pass
   has its own budget cap; inspect the script's argument and logs before starting it.

Each model has one claims file and one clusters file in `data/` (`claims_<model>.jsonl`, `clusters_<model>.jsonl`).
They hold every sample; the panel id prefix names the sample (`icml_`, `other_`, `otherplacebo_`, `noq_`, `nocq_`, `wd_`,
`placebo_`, `placeboA_`, `sub3_`, or a bare paper id). Rows record the request settings. A different model or
reasoning effort needs a different short name. The extraction chains write part files and
`extract.py merge` joins them. Git LFS stores both files.

`data/gold/` holds the Sol 6.1, Opus 5.5 and Kimi K3.1 Pro council outputs (`claims_council.jsonl`,
`clusters_council.jsonl`), sample reviews and panels, and LLM reference labels. `adjudicate/` holds the claim judge
input and output; `adjudicate2/` holds the grouping judge input and output. JSONL and Parquet files use Git LFS. These are LLM labels, not human labels. `scripts/gold.sh OUT_DIR` runs the same steps into a new
folder (`src/gold.py`); it calls paid APIs and never writes into `data/gold/`. `tests/test_gold.py` checks that the
steps without model calls rebuild `data/gold/` byte for byte.

Each data file has a `<file>.md` next to it that says what it holds, how to make it and what reads it.

`sample_reviews.parquet` and `panels.jsonl` are the fixed sample; `panels.py ml` cannot redraw them from the current
`reviews.parquet`. Claim and review embeddings, MELD scores and reference annotations are saved model outputs,
not analysis estimates.

## Analysis

```
scripts/post.sh gpt-6-luna-high
```

Each run recomputes the analysis from saved model outputs. It does not reuse earlier analysis estimates or make
model calls. The stages rebuild MELD flags, panel and pair scores, the detector placebo, robustness checks, yearly tables,
paper numbers and figures.

- `data/overlap/`: recomputed panel and pair tables.
- `reports/overlap/`: recomputed statistical summaries and their sample sizes.
- `paper/tables/`: generated result tables and `numbers.tex`.
- `paper/figs/`: vector PDF figures.

These directories come from `ANALYSIS_DIR`, `ANALYSIS_REPORTS_DIR`, `TABLES_DIR` and `FIGURES_DIR` in settings.
The paper reads numeric commands from `paper/tables/numbers.tex`, generated from the current reports rather than
hand-entered values.

`scripts/post.sh MODEL OUT_DIR` writes the paper tables and figures under `OUT_DIR`. Analysis reports still use
`ANALYSIS_REPORTS_DIR`.

## Reproduce the detector-conditioned results

1. Use the same repository revision and `uv.lock`, Python 3.14, and the `detect` extra included by
   `uv sync --locked --all-extras`. Fetch the Git LFS objects with `git lfs pull`.
2. Restore the fixed sample texts and panels described above. Set the raw-data paths in `.env` and run
   `scripts/inputs.sh` if the derived inputs are absent. Do not redraw the ML sample: the current corpus cannot
   recover its original draw. The file notes under `data/` list the required source and saved-output files.
3. With the saved `data/meld_scores.jsonl`, run the existing analysis chain:

   ```
   OMP_NUM_THREADS=2 nice -n 19 scripts/post.sh gpt-6-luna-high
   ```

   This route uses saved extraction, grouping, embedding and detector outputs. It needs no paid API calls or GPU.
   It writes review-linked flags locally; do not publish those flags. Do not run `extract_all.sh` to reproduce
   analysis from saved responses: that script makes paid model calls. On 5 November 2026, the full post chain
   took 11 minutes 16 seconds and peaked at 3.75 GiB resident memory on an 8-vCPU AMD EPYC-Rome host with
   `OMP_NUM_THREADS=2`. This is one local run, not a hardware requirement.
4. Check these aggregate files under `reports/overlap/`:

   | File | Check |
   |---|---|
   | `identification_gpt-6-luna-high.csv` | `varying`: identifying papers; `two`, `two_se`: two-versus-none estimate and paper-clustered SE; `endpoint*`: direct comparison and its interval |
   | `pair_types_gpt-6-luna-high.csv` | Separate one-flagged, other two-flagged and two-AI-written contrasts against unflagged pairs |
   | `pair_type_sizes_gpt-6-luna-high.csv` | Shared-problem counts, set sizes, and size-adjusted contrasts |
   | `paper_numbers_gpt-6-luna-high.csv` | Numeric inputs to the paper |

   Expected `identification` rows, rounded as in the paper:

   | Sample | Identifying papers | Two vs. none (points) | Clustered SE |
   |---|---:|---:|---:|
   | ICLR 2024–2026, flagged | 6,275 | 3.3 | 0.4 |
   | ICLR 2024–2026, AI-written | 4,284 | 7.5 | 0.6 |
   | NeurIPS 2024–2025, flagged | 1,012 | 4.6 | 1.2 |
   | ICML 2025–2026, flagged | 2,335 | 5.0 | 0.6 |
   | All journals 2024–2026, flagged | 887 | 4.8 | 1.6 |

   The AI-written row uses pairs with no AI-written review as its reference; the other rows use unflagged pairs.
   The fixed-effects 95% intervals are the estimate ± 1.96 times its unrounded clustered SE. Direct-comparison
   intervals use the t distribution over papers. Source-year pair-type counts alone cannot reproduce either fit.

### Rerun the detector when scores are unavailable

The CPU lock file does **not** supply a CUDA build of PyTorch. These commands require a separate CUDA-capable
installation with the project's dependencies. Keep `uv.lock` unchanged and use `uv run --no-sync` so UV does not
replace that installation with CPU PyTorch. The original GPU environment lock and driver version are not in this
archive; the repository therefore does not establish a fully pinned GPU rerun.

In a working copy with the fixed `data/meld_input.parquet` and no `data/meld_scores.jsonl`, run:

```
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python -I -c 'import torch; assert torch.cuda.is_available()'
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python -I src/meld.py check --n 64 --batch_tokens 16384
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python -I src/meld.py scores --batch_tokens 16384
```

The scorer downloads `anon-review-meld-2026/meld` at revision
`ddfc9c789b8c8d0f92d3774265316d418c96dc67` from Hugging Face. It uses fp32, 2,048-token windows and the top 25% of
token scores. It resumes by skipping keys already in the output; move old scores aside before a fresh run.
After scoring, run step 3 in the CPU analysis environment. `meld.py flags`, the first post step, recomputes the
99.9th-percentile threshold on pre-2023 texts and rebuilds both detector labels.

The recorded initial scoring run took about 4.5 hours on one GPU. Later added-text runs used an RTX 3090; the
notes do not establish a full-current-corpus runtime or peak memory requirement. See
[`data/meld_scores.jsonl.md`](data/meld_scores.jsonl.md). Rerun timings depend on hardware and text lengths.

A release without review-linked scores needs this local detector rerun and the exact sampled texts. Released
pair overlap and source-year counts are not enough. Withholding scores limits direct disclosure; it does not
prevent someone from joining public reviews to released claims or rerunning the public detector. Detector labels
must not support action against individual reviewers.

## Paper

```
cd paper
latexmk -pdf -bibtex main.tex
```

The opening figure shows measured time trends. Section 3 defines the measure and its limits. The council
reference protocol and robustness checks appear in the appendix. Figures and prose follow the guides in `CLAUDE.md`.

## Checks

```
uv run python -I -m pytest
uv run ruff check src scripts tests
uv run black --check src scripts tests
uv run python -I scripts/check_no_comments.py
```
