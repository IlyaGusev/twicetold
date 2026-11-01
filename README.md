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

`src/metrics.py` defines the measure. The same function scores panels, pair tables, cross-paper controls and blind
annotations. Direct pair audits first combine connected claim matches into distinct problems.

## Data and extraction

Run commands as `uv run python -I src/<module>.py <command>`. Raw ML dumps live at `RAW_ML_REVIEWS_DIR`; journal
sources live at `RAW_JOURNAL_REVIEWS_DIR`, both set in `src/settings.py`.

1. Fetch source data with `fetch.py`.
2. Build the review tables with `corpus.py ml` and `corpus.py other`.
3. Build conference, journal and control panels with `panels.py ml`, `panels.py robust`, `panels.py icml` and
   `panels.py other`.
4. Run `scripts/extract_all.sh gpt-6-luna-high` for extraction and grouping. This step calls paid APIs. Each pass
   has its own budget cap; inspect the script's argument and logs before starting it.

Claims and groupings use model-named JSONL files in `data/`. Rows record the request settings. A different model or
reasoning effort needs a different short name; do not overwrite model outputs.

The annotation keys in `data/blind/`, `data/pairs/` and `data/spec/` identify the saved reference labels. Blind
contrast, rewrite and pair-audit annotations live in `data/blind2/` through `data/blind5/`. Do not redraw keys for
existing labels. The panel generators refuse to overwrite them.

The original sample uses the source snapshot in `data/backup_predup/`. Preserve that snapshot to reproduce its
paper and review identifiers. Claim and review embeddings, detector scores and reference annotations are saved
model outputs, not analysis estimates.

## Analysis

```
scripts/post.sh gpt-6-luna-high
```

Each run recomputes the analysis from saved model outputs and annotations. It does not reuse earlier analysis
estimates or make model calls. The stages rebuild panel and pair scores, reference audits, detector comparisons,
robustness checks, yearly tables, paper numbers and figures.

- `data/overlap/`: recomputed panel, review and pair tables.
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

The opening figure shows time trends. The method and worked example follow the introduction. Detailed audits
and robustness checks appear in the appendix. Figures follow `paper/FIGURE_GUIDE.md`; prose follows
`paper/WRITING_GUIDE.md`.

## Checks

```
uv run python -I -m pytest
uv run ruff check src scripts tests
uv run black --check src scripts tests
uv run python -I scripts/check_no_comments.py
```
