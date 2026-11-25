# Twice Told

Code and paper for measuring whether reviewers repeat the same critiques in the LLM era.

[Paper (arXiv v1)](https://arxiv.org/abs/2611.26286v1) ·
[Dataset](https://huggingface.co/datasets/IlyaGusev/twicetold) ·
[Citation](#citation)

## Quick start

Rebuild the analysis from saved model outputs with Python 3.14 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-extras
OMP_NUM_THREADS=2 nice -n 19 scripts/post.sh gpt-6-luna-high
```

The analysis script pulls the pinned dataset revision when `data/final/` is missing. With the saved extraction,
grouping, embedding and detector outputs, the run needs no paid API calls or GPU. Do not run `extract_all.sh` to
reproduce the analysis: that script calls paid APIs.

Configuration lives in [`src/settings.py`](src/settings.py): paths, model routes, prices, reasoning efforts and
`HF_DATASET_REVISION`. Put API keys and OpenReview credentials in `.env` only for steps that need them. The paper
uses `gpt-6-luna-high`, with high effort in both extraction passes. Prompts live in [`src/prompts/`](src/prompts/).

## Measure

For each review pair, count the distinct problem groups the reviews share and divide by the smaller problem set.
A score of 1 means that the smaller set adds no problem beyond the other review. Duplicate claims in one group
count once. A pair with an empty problem set scores 0. Average all pairs within each paper, then average papers
within each year. Every paper has equal weight.

[`src/metrics.py`](src/metrics.py) defines the measure. The same function scores panels, pair tables and
cross-paper controls.

## Data

The [Hugging Face dataset](https://huggingface.co/datasets/IlyaGusev/twicetold) and local `data/` share this layout:

| Folder | Content | Use |
|---|---|---|
| `final/` | Reviews with claims and detector scores, papers, panels with problem groups, review variants | Analysis through `src/hub.py`; extraction |
| `gold/` | Council reference annotations and two external human benchmarks | Extraction and grouping checks |
| `additional/` | Rewrite test, LLM reviews of arXiv papers, known-origin detector texts, review embeddings | Side experiments |
| `raw/` | OpenReview dumps, predicted areas, journal statistics, panel files, detector scores | Corpus build |
| `work/` | Local work files and analysis tables | Local only; not uploaded |

Git does not store `data/`. `HF_DATASET_REVISION` pins the dataset commit used by the paper. To fetch the data
without running the analysis:

```bash
uv run python src/release.py pull
```

Add `--raw` to include the raw files listed in `settings.HF_RAW_PATTERNS`. The dataset card describes each config
and column. Most files in `gold/`, `additional/` and `raw/` have a `<file>.md` with their contents and build steps.
Embeddings, detector scores and reference annotations are saved model outputs, not analysis estimates.

## Analysis outputs

[`scripts/post.sh`](scripts/post.sh) recomputes panel and pair scores, detector controls, robustness checks,
quote checks, external benchmark checks, yearly tables, paper numbers and figures. It does not reuse earlier
analysis estimates. Keep the saved embeddings: missing review embeddings can cause paid API calls.

| Path | Output |
|---|---|
| `data/work/overlap/` | Panel and pair tables |
| `reports/overlap/` | Statistical summaries and sample sizes |
| `paper/tables/` | Result tables and `numbers.tex` |
| `paper/figs/` | Vector PDF figures |

These paths come from `ANALYSIS_DIR`, `ANALYSIS_REPORTS_DIR`, `TABLES_DIR` and `FIGURES_DIR` in settings. The paper
reads numeric commands from `numbers.tex`, which the script generates from current reports rather than hand-entered
values.

To write paper tables and figures to a separate folder:

```bash
OMP_NUM_THREADS=2 nice -n 19 scripts/post.sh gpt-6-luna-high OUT_DIR
```

Analysis reports still use `ANALYSIS_REPORTS_DIR`.

### Check the detector-conditioned results

Use the same repository revision, `uv.lock` and pinned dataset revision. The quick-start command includes the
`detect` extra. Check these files in `reports/overlap/`, each with the suffix `_gpt-6-luna-high.csv`:

| Report | Check |
|---|---|
| `rq1_evidence` | `est`, `se`: direct within-paper gap between two-flagged and unflagged pairs, averaged over papers; `identifying`: papers with both pair types |
| `pair_types` | One-flagged and two-flagged gaps against unflagged pairs; `est_fe`, `se_fe`: fixed-effects check |
| `identification` | `two`, `two_se`: fixed-effects estimate and paper-clustered SE; `endpoint*`: direct gap |
| `journal_gaps` | Pooled within-paper journal gap |
| `robust`, `withdrawn` | ICLR rise from 2026 to 2027 under each analysis choice; share and overlap of withdrawn 2026 papers |
| `pair_type_sizes` | Shared-problem counts, set sizes and size-adjusted contrasts |
| `paper_numbers` | Numeric inputs to the paper |

Each 95% interval is the estimate ± 1.96 times its unrounded SE.

A detector flag is not proof that a reviewer used an LLM. Do not use detector labels to act against individual
reviewers.

## Extract new claims and groups

This step calls paid APIs. Set the API keys in `.env`, check the model route and set a budget before running:

```bash
OMP_NUM_THREADS=2 nice -n 19 scripts/extract_all.sh gpt-6-luna-high 300
```

The second argument caps each pass at that many USD; the default is 300. Check the logs in
`logs/extract_<model>/` for failures and budget limits.

1. `extract.py final_pass1` reads sampled reviews and variants from `data/final/` and writes
   `data/work/claims_<model>.jsonl`.
2. `extract.py final_pass2` reads panels from `data/final/` and writes `data/work/clusters_<model>.jsonl`.
3. `release.py build` merges both work files into `data/final/`.

Both passes skip rows already in `data/final/`; `--redo` extracts them again. Rows record the request settings.
Use a new short name in `settings.OPENROUTER_ROUTES` for each model or reasoning effort. If panel outcomes already
exist, the script also runs the rewrite and LLM-review side experiments.

Panel `kind` names are `paper`, `icml`, `journal`, `subset3`, `noq`, `nocq`, and the cross-paper controls
`placebo_venue`, `placebo_area` and `journal_placebo`.

`uv run python src/release.py push` uploads the release files to the dataset. After a push, set
`HF_DATASET_REVISION` to the new commit.

## Reference checks

`data/gold/` contains model reference labels and two human benchmarks. Council agreement is not accuracy against
human ground truth.

| Files | Content |
|---|---|
| `claims_council.jsonl`, `clusters_council.jsonl` | Sol 6.1, DeepSeek V4.1 Flash and Kimi K3.1 Pro council outputs; the council groups Luna's claims |
| `claims_<model>.jsonl`, `clusters_<model>.jsonl` | Luna's claims and groups for the same reviews and panels |
| `claims_<member>.jsonl`, `clusters_<member>_own.jsonl` | Each council member's full extraction and grouping run |
| `kim_items.jsonl`, `kim_reviews.parquet` | 218 human-split Nature-family reviews from `settings.KIM_DATASET_REPO` |
| `claimcheck_items.jsonl`, `claimcheck_reviews.parquet` | 60 NeurIPS reviews with expert weakness spans from `settings.CLAIMCHECK_REPO` |
| `claims_<model>_<name>.jsonl` | Model claims extracted from each external benchmark |

`gold.py sample` draws 600 production panels with a fixed seed: 510 same-paper panels and 90 cross-paper controls.
The sample covers ICLR 2018–2027, NeurIPS 2024–2025, ICML 2025–2026 and five journals in 2024–2026. From 2024 on,
half of each cell has at least two flagged and one unflagged review. A quarter of each ML cell has at least two
LLM-written reviews. Each cell splits 70/30 into dev and test.

The analysis chain runs these checks through [`src/validate.py`](src/validate.py):

| Command | Check |
|---|---|
| `extraction` | Quoted spans and aligned claims from Luna and the council |
| `external` | Claims aligned to human items in each external benchmark |
| `council_agreement` | Luna's groupings against the majority of raw council votes |
| `council_pair_types` | Grouping agreement by pair type |
| `pipelines` | Panel overlap from Luna's full pipeline against each member's full pipeline |
| `grouper_swap` | Within-paper gaps after regrouping Luna's claims with each member and the council majority |

Grouping checks exclude pairs with a missing member, missing claim ID or repeated claim ID. Excluded pairs do
not count as separate problems. Unlabeled pairs do not define complete groups, so agreement checks cannot establish
reference overlap levels or trends. These checks read the gold folder and make no model calls.

To run only the council agreement check:

```bash
scripts/post.sh --council-agreement
```

`scripts/gold.sh OUT_DIR` rebuilds the reference data in a new folder. It calls paid APIs and refuses to write to
`data/gold/`. [`tests/test_gold.py`](tests/test_gold.py) checks that steps without model calls rebuild the saved
reference data byte for byte.

## Build the corpus from raw sources

Skip this section to reproduce the analysis from saved outputs. The original ML sample cannot be redrawn from
the current `reviews.parquet`; keep the fixed `sample_reviews.parquet` and `panels.jsonl`.

The corpus build needs `release.py pull --raw` and external dumps at `RAW_ML_REVIEWS_DIR` and
`RAW_JOURNAL_REVIEWS_DIR`. Their defaults are `data/raw/sources/` and `data/raw/sources/other/`; set other paths in
`.env`. Run module commands with `uv run python src/<module>.py <command>`.

1. Fetch source data with `fetch.py`. `fetch.py iclr_download` gets ICLR 2027 submissions and reviews from OpenReview.
2. Build ML review tables with `corpus.py ml`. To append ICLR 2027, use `corpus.py append_openreview_iclr`,
   `panels.py new_cell` and `panels.py robust_cell`.
3. Run `OMP_NUM_THREADS=2 nice -n 19 scripts/raw.sh`. The script builds journal tables, parses ICML downloads,
   builds ICML, journal and robustness panels, and writes the detector inputs.
4. When the script stops for missing detector scores, follow the [detector steps](docs/detectors.md) and restart
   the script. It needs review scores from MELD, EditLens, GPTTrace and GigaCheck, plus abstract scores from MELD,
   EditLens and GigaCheck.
5. Once scores exist, the script rebuilds detector flags and writes `data/final/` with `release.py from_raw`.
   It keeps existing claims and groups. The raw build makes no paid API calls.

GPU setup, scorer commands and known runtime limits are in [Rerun detectors](docs/detectors.md).

## Build the paper

Install LaTeX and `latexmk`, then run:

```bash
(cd paper && latexmk -pdf -bibtex main.tex)
uv run python scripts/check_layout.py
```

The layout check fails if the introduction ends after page 1. Writing and figure guides live in
[`CLAUDE.md`](CLAUDE.md).

## Checks

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run python -m pytest
uv run ruff check src scripts tests
uv run black --check src scripts tests
uv run python scripts/check_no_comments.py
```

## Citation

If you use the code or data, cite [the paper](https://arxiv.org/abs/2611.26286v1):

```bibtex
@misc{gusev2026twicetold,
  title = {Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the {LLM} Era},
  author = {Ilya Gusev},
  year = {2026},
  eprint = {2611.26286},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2611.26286v1}
}
```
