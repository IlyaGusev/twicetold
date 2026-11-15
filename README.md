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

## Dataset

The Hugging Face dataset [`IlyaGusev/twicetold`](https://huggingface.co/datasets/IlyaGusev/twicetold) has four
folders. The local folder `data/` has the same four, plus `data/work/` for files that stay local. `HF_DATASET_REVISION`
in `src/settings.py` pins the commit that the paper uses. Git does not store `data/`.

| Folder | Content | Read by |
|---|---|---|
| `final/` | Reviews with claims and MELD scores, papers, panels with problem groups, review variants | all analysis, through `src/hub.py`; extraction |
| `gold/` | Council reference annotations and the two external human benchmarks | the extraction and grouping checks |
| `additional/` | Rewrite test, LLM reviews of arXiv papers, known-origin detector texts, review embeddings | the side experiments |
| `raw/` | OpenReview dumps, predicted areas, journal statistics, panel files, MELD scores | only the corpus build (`scripts/raw.sh`) |

The pipeline has three steps:

1. `uv run python src/release.py pull` copies `final/`, `gold/` and `additional/` of the pinned revision into `data/`.
   The chain scripts run it when `data/final/` is missing. `--raw` adds `raw/`.
2. `scripts/extract_all.sh MODEL` extracts claims for sampled reviews without claims, groups panels without groups
   and writes the results into `data/final/` (`release.py build`). This step calls paid APIs.
3. `scripts/post.sh MODEL` recomputes every report, table and figure from `data/`.

`release.py push` uploads `data/` to the dataset. After a push, set `HF_DATASET_REVISION` to the new commit.

## Extraction

`extract.py final_pass1` reads the sampled reviews and the review variants from `data/final/` and writes new claims
to `data/work/claims_<model>.jsonl`. `extract.py final_pass2` reads the panels from `data/final/` and writes new
problem groups to `data/work/clusters_<model>.jsonl`. Both skip rows that `data/final/` already holds; `--redo`
extracts them again. `release.py build` copies the rows of both work files into `data/final/`. Rows record the request
settings. A different model or reasoning effort needs a different short name. Each pass has its own budget cap;
inspect the script's argument and logs before a run.

The panel `kind` names the sample: `paper`, `icml`, `journal`, `withdrawn`, `subset3`, `noq`, `nocq`, and the
cross-paper controls `placebo_venue`, `placebo_area` and `journal_placebo`.

## Corpus build from raw sources

`scripts/raw.sh` builds `data/final/` from the source dumps. It needs the files of `raw/` (`release.py pull --raw`)
and the external dumps at `RAW_ML_REVIEWS_DIR` and `RAW_JOURNAL_REVIEWS_DIR` (default `data/raw/sources/`; set them
in `.env`). All files of this path live in `data/raw/`. Run commands as `uv run python src/<module>.py <command>`.

1. Fetch source data with `fetch.py`. `fetch.py iclr_download` gets the ICLR 2027 submissions and reviews from the
   OpenReview API.
2. Build the ML review tables with `corpus.py ml`. `corpus.py append_openreview_iclr` appends ICLR 2027 to the
   existing tables; `panels.py new_cell` and `panels.py robust_cell` append its sample and robustness panels.
3. Run `scripts/raw.sh`. It builds the journal tables (`corpus.py other`), parses the ICML downloads
   (`fetch.py icml`), builds the ICML, journal and robustness inputs (`panels.py icml`, `other`, `robust`, `nocq`)
   and the MELD input (`meld.py input`). It stops when `data/raw/meld_scores.jsonl` is missing: score the MELD input
   on a GPU with `meld.py scores`, then run the script again. It stops in the same way when
   `data/raw/editlens_scores.jsonl` is missing (`editlens.py scores`). It then rebuilds the MELD flags
   (`meld.py flags`) and the EditLens flags (`editlens.py flags`) and
   writes `data/final/` (`release.py from_raw`), keeping the claims and groups that `data/final/` already holds.
   No step calls a paid API.

`sample_reviews.parquet` and `panels.jsonl` are the fixed sample; `panels.py ml` cannot redraw them from the current
`reviews.parquet`.

## Reference data

`data/gold/` holds the Sol 6.1, DeepSeek V4.1 Flash and Kimi K3.1 Pro council outputs (`claims_council.jsonl`,
`clusters_council.jsonl`), sample reviews and panels, and LLM reference labels. `gold.py sample` draws 600 production
panels with a fixed seed: 510 same-paper panels (ICLR 2018–2027, NeurIPS 2024–2025, ICML 2025–2026 and five journals
2024–2026) and 90 cross-paper placebo panels. From 2024 on, half of each cell holds at least two flagged and one
unflagged review, and a quarter of each ML cell holds at least two LLM-written reviews. Each cell splits 70/30 into
dev and test. `claims_<model>.jsonl` and `clusters_<model>.jsonl` hold Luna's claims and groupings for the same
reviews and panels; the council groups Luna's claim list (`clusters_council.jsonl`). `claims_<member>.jsonl` and
`clusters_<member>_own.jsonl` hold each member's own claims and its grouping of them, so every model runs the full pipeline. `kim_items.jsonl` and `kim_reviews.parquet` hold the 218 human-split Nature-family reviews from
`settings.KIM_DATASET_REPO`; `claimcheck_items.jsonl` and `claimcheck_reviews.parquet` hold the 60 NeurIPS reviews with
expert weakness spans from `settings.CLAIMCHECK_REPO`; `claims_<model>_<name>.jsonl` holds the claims extracted from each. The dataset holds these files under
`gold/`. These are LLM labels, not human labels. `scripts/gold.sh OUT_DIR` runs the same steps into a new
folder (`src/gold.py`); it calls paid APIs and never writes into `data/gold/`. `tests/test_gold.py` checks that the
steps without model calls rebuild `data/gold/` byte for byte.

The dataset card describes each config and column. Most files under `gold/`, `additional/` and `raw/` have a
`<file>.md` next to them that says what the file holds and how to make it.

Claim and review embeddings, MELD scores and reference annotations are saved model outputs, not analysis estimates.

## Analysis

```
scripts/post.sh gpt-6-luna-high
```

Each run recomputes the analysis from saved model outputs. It does not reuse earlier analysis estimates or make
model calls. The stages rebuild panel and pair scores, the detector placebo, robustness checks, the quote checks
(`validate.py quotes`: quotes found in the text, duplicate quotes, critique sentences no quote touches; `validate.py external`:
claims aligned to the human items of each external set), yearly tables, paper numbers and figures.

- `data/work/overlap/`: recomputed panel and pair tables.
- `reports/overlap/`: recomputed statistical summaries and their sample sizes.
- `paper/tables/`: generated result tables and `numbers.tex`.
- `paper/figs/`: vector PDF figures.

These directories come from `ANALYSIS_DIR`, `ANALYSIS_REPORTS_DIR`, `TABLES_DIR` and `FIGURES_DIR` in settings.
The paper reads numeric commands from `paper/tables/numbers.tex`, generated from the current reports rather than
hand-entered values.

`scripts/post.sh MODEL OUT_DIR` writes the paper tables and figures under `OUT_DIR`. Analysis reports still use
`ANALYSIS_REPORTS_DIR`.

### Grouping reference

Use the majority of the raw votes from Sol, DeepSeek and Kimi for grouping checks. Exclude pairs with a missing
member, missing claim ID or repeated claim ID. Do not treat excluded pairs as separate problems.

`validate.py extraction` compares the quoted spans and the aligned claims of Luna and the council (`reports/overlap/extraction*_MODEL.csv`).
Run `scripts/post.sh --council-agreement` to score Luna's groupings against the majority votes. `validate.py pipelines`
compares the panel overlap of Luna's full pipeline with each member's full pipeline (`reports/overlap/pipelines*_MODEL.csv`). `validate.py grouper_swap`
regroups Luna's claims with each member and with the council majority and compares the within-paper gaps (`reports/overlap/grouper_swap_MODEL.csv`). The commands read
the gold folder; `extraction` embeds the claim sentences with `settings.CLAIM_EMBED_MODEL` through OpenRouter, `council_agreement` and `grouper_swap` make no model calls. Agreement is not human-ground-truth accuracy. Unlabeled pairs do not define
complete groups. Do not use this check to infer reference overlap levels or trends.

## Reproduce the detector-conditioned results

1. Use the same repository revision and `uv.lock`, Python 3.14, and the `detect` extra included by
   `uv sync --locked --all-extras`.
2. Run the analysis chain:

   ```
   OMP_NUM_THREADS=2 nice -n 19 scripts/post.sh gpt-6-luna-high
   ```

   The chain reads `data/` and pulls the pinned dataset revision if `data/final/` is missing. It uses saved
   extraction, grouping, embedding and detector outputs and needs no paid API calls or GPU. Do not redraw the ML
   sample: the current corpus cannot recover its original draw. Do not run `extract_all.sh` to reproduce analysis
   from saved responses: that script makes paid model calls.
3. Check these aggregate files under `reports/overlap/`:

   | File | Check |
   |---|---|
   | `rq1_evidence_gpt-6-luna-high.csv` | `est`, `se`: direct within-paper gap (two flagged minus unflagged, mean over papers) and its SE over papers; `identifying`: papers with both pair types |
   | `pair_types_gpt-6-luna-high.csv` | One-flagged, other two-flagged and two-LLM-written gaps against unflagged pairs; `est_fe`, `se_fe`: the fixed-effects check |
   | `identification_gpt-6-luna-high.csv` | `two`, `two_se`: fixed-effects estimate and paper-clustered SE; `endpoint*`: the direct gap again |
   | `robust_gpt-6-luna-high.csv`, `withdrawn_gpt-6-luna-high.csv` | ICLR rise from 2026 to 2027 under each analysis choice; share and overlap of withdrawn 2026 papers |
   | `pair_type_sizes_gpt-6-luna-high.csv` | Shared-problem counts, set sizes, and size-adjusted contrasts |
   | `paper_numbers_gpt-6-luna-high.csv` | Numeric inputs to the paper |

   Expected `rq1_evidence` rows, rounded as in the paper:

   | Sample | Papers with both pair types | Gap (points) | SE |
   |---|---:|---:|---:|
   | ICLR 2024–27, flagged | 5,414 | 3.3 | 0.3 |
   | ICLR 2024–27, LLM-written | 1,559 | 7.0 | 0.7 |
   | NeurIPS 2024–25, flagged | 1,005 | 4.7 | 0.7 |
   | ICML 2025–26, flagged | 2,433 | 4.0 | 0.5 |

   The pooled journal gap is in `journal_gaps_gpt-6-luna-high.csv`: 5.0 points, SE 1.8, 230 papers. Each 95%
   interval is the estimate ± 1.96 times its unrounded SE.

### Rerun the detector when scores are unavailable

The CPU lock file does **not** supply a CUDA build of PyTorch. These commands require a separate CUDA-capable
installation with the project's dependencies. Keep `uv.lock` unchanged and use `uv run --no-sync` so UV does not
replace that installation with CPU PyTorch. The original GPU environment lock and driver version are not in this
archive; the repository therefore does not establish a fully pinned GPU rerun.

In a working copy with `data/raw/meld_input.parquet` (`meld.py input`) and no `data/raw/meld_scores.jsonl`, run:

```
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python -c 'import torch; assert torch.cuda.is_available()'
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/meld.py check --n 64 --batch_tokens 16384
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/meld.py scores --batch_tokens 16384
```

The scorer downloads `anon-review-meld-2026/meld` at revision
`ddfc9c789b8c8d0f92d3774265316d418c96dc67` from Hugging Face. It uses fp32, 2,048-token windows and the top 25% of
token scores. It resumes by skipping keys already in the output; move old scores aside before a fresh run.
After scoring, run `scripts/raw.sh` in the CPU analysis environment. `meld.py flags` recomputes the
99.9th-percentile threshold on pre-2023 texts and rebuilds both detector labels.

The recorded initial scoring run took about 4.5 hours on one GPU. Later added-text runs used an RTX 3090; the
notes do not establish a full-current-corpus runtime or peak memory requirement. See
`raw/meld_scores.jsonl.md` in the dataset. Rerun timings depend on hardware and text lengths.

EditLens is the second detector (appendix only). In the same CUDA environment, with access to the gated repository
`pangram/editlens_Llama-3.2-3B` and with the `detect` extra (`peft`, `bitsandbytes`, `emoji`), run:

```
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/editlens.py check --reference EditLens/data/test.csv --n 500
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/editlens.py scores
```

`check` needs `data/test.csv` from https://github.com/pangramlabs/EditLens and compares our scores with Pangram's
published scores. The scorer loads the adapter on `unsloth/Llama-3.2-3B` in 4 bits and scores 1,024-token windows;
the recorded run took 9 h 45 min on an RTX 3090. See `raw/editlens_scores.jsonl.md` in the dataset.

The dataset holds review-linked MELD and EditLens scores and flags. A flag is a detector output, not proof that a reviewer
used an LLM. Detector labels must not support action against individual reviewers.

## Paper

```
cd paper
latexmk -pdf -bibtex main.tex
cd ..
uv run python scripts/check_layout.py
```

`check_layout.py` fails when the introduction ends after page 1.

The opening figure shows measured time trends. Section 3 defines the measure and its limits. The council
reference protocol and robustness checks appear in the appendix. Figures and prose follow the guides in `CLAUDE.md`.

## Checks

```
uv run python -m pytest
uv run ruff check src scripts tests
uv run black --check src scripts tests
uv run python scripts/check_no_comments.py
```
