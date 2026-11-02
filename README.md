# Twice Told

Code and paper for measuring whether reviewers repeat the same critiques in the LLM era.

## Setup

```
uv sync --all-extras
```

API keys and OpenReview credentials go in `.env`. `src/settings.py` holds paths, model routes, prices and reasoning
efforts. The paper uses `gpt-6-luna-high`, with high effort in both extraction passes. Prompts live in
`src/prompts/`.

## Measure

For each pair of reviews, count the distinct problem groups they share and divide by the smaller problem set.
A score of 1 means that the smaller set adds no problem beyond the other review. Duplicate claims in one group
count once. Empty reviews enter no pair. Average pairs within each paper, then average papers within each year.
Every paper has equal weight.

`src/metrics.py` defines the measure. The same function scores panels, pair tables and cross-paper controls.

## Data and extraction

Run commands as `uv run python -I src/<module>.py <command>`. Raw ML dumps live at `RAW_ML_REVIEWS_DIR`; journal
sources live at `RAW_JOURNAL_REVIEWS_DIR`, both set in `src/settings.py`.

1. Fetch source data with `fetch.py`.
2. Build the ML review tables with `corpus.py ml`.
3. Run `scripts/inputs.sh`. It builds the journal tables (`corpus.py other`), parses the ICML downloads
   (`fetch.py icml`), and builds the ICML, journal, robustness and CoCoDet inputs (`panels.py icml`, `other`,
   `robust`, `cocodet`). Git does not store these inputs. `extract_all.sh` and `post.sh` run `inputs.sh` when the
   inputs are missing. No step calls a paid API.
4. Run `scripts/extract_all.sh gpt-6-luna-high` for extraction and grouping. This step calls paid APIs. Each pass
   has its own budget cap; inspect the script's argument and logs before starting it.

Claims and groupings use model-named JSONL files in `data/`. Rows record the request settings. A different model or
reasoning effort needs a different short name; do not overwrite model outputs. Git LFS stores the main
`gpt-6-luna-high` files and their `_icml`, `_other`, `_noq` and `_wd` variants.

`data/gold/` holds the Sol 6.1, Opus 5.5 and Kimi K3.1 Pro council outputs, sample reviews and panels, and LLM
reference labels. `adjudicate/` holds claim judgments; `adjudicate2/` holds grouping judgments. JSONL and Parquet
files use Git LFS. These are LLM labels, not human labels. `scripts/gold.sh OUT_DIR` runs the same steps into a new
folder (`src/gold.py`); it calls paid APIs and never writes into `data/gold/`. `tests/test_gold.py` checks that the
steps without model calls rebuild `data/gold/` byte for byte.

Each data file has a `<file>.md` next to it that says what it holds, how to make it and what reads it.

`sample_reviews.parquet` and `panels.jsonl` are the fixed sample; `panels.py ml` cannot redraw them from the current
`reviews.parquet`. Claim and review embeddings, detector scores and reference annotations are saved model outputs,
not analysis estimates.

## Analysis

```
scripts/post.sh gpt-6-luna-high
```

Each run recomputes the analysis from saved model outputs. It does not reuse earlier analysis estimates or make
model calls. The stages rebuild panel and pair scores, detector comparisons, robustness checks, yearly tables,
paper numbers and figures.

- `data/overlap/`: recomputed panel and pair tables.
- `reports/overlap/`: recomputed statistical summaries and their sample sizes.
- `paper/tables/`: generated result tables and `numbers.tex`.
- `paper/figs/`: vector PDF figures.

These directories come from `ANALYSIS_DIR`, `ANALYSIS_REPORTS_DIR`, `TABLES_DIR` and `FIGURES_DIR` in settings.
The paper reads numeric commands from `paper/tables/numbers.tex`, generated from the current reports rather than
hand-entered values. `paper/tables/match_example.tex` supplies a labeled schematic example.

`scripts/post.sh MODEL OUT_DIR` writes the paper tables and figures under `OUT_DIR`. Analysis reports still use
`ANALYSIS_REPORTS_DIR`.

## Paper

```
cd paper
latexmk -pdf -bibtex main.tex
```

The opening figure shows time trends. The method and worked example follow the introduction. The council
reference protocol and robustness checks appear in the appendix. Figures and prose follow the guides in `CLAUDE.md`.

## Checks

```
uv run python -I -m pytest
uv run ruff check src scripts tests
uv run black --check src scripts tests
uv run python -I scripts/check_no_comments.py
```
