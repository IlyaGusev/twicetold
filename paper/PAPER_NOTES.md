# Twice-Told Critiques: paper-specific notes

How WRITING_GUIDE.md and FIGURE_GUIDE.md apply to this paper.

## Core claims

1. Within a paper, two AI-flagged reviews share more critique claims than two unflagged reviews of the same paper.
2. Same-paper overlap was flat 2020–2025 and rose in 2026; cross-paper overlap did not rise.
3. The mix of critiques shifted from clarity and typos toward requests for more experiments.

Confidence: claim 1 is systematic (all instruments, detectors, blind references). Claim 2 is hedged: the ICLR rise is instrument-sensitive and drift cannot be excluded. Journal results (RQ5) are exploratory.

## Holes to red-team

- Detector false positives.
- Review length and panel size.
- Venue mix and decision mix (rejected and withdrawn papers match more).
- Year-by-venue confounds; instrument drift over time.
- Reviewer turnover: checked at F1000 (signed reports), `validate.py reviewer_fe`; same reviewers rose as much as the pool.
- Paper-side AI-likeness (LLM-written papers with uniform flaws): CoCoDet on abstracts, `validate.py paper_side`.
- Venue policy: ICML 2025 banned LLM use in reviewing; ICML 2026 dual policy. Per-review policy is not public.

## Pending: position-track difference in differences

ICML's position-paper track kept the LLM ban in both years and the same review form, while the main track relaxed the
ban and changed its form in 2026. Reviews are fetched: `data/icml_position_reviews.parquet` (2025: 73 papers,
280 reviews; 2026: 213 papers, 818 reviews) and `data/panels_icml_position.jsonl`. To run (about USD 2 with
gpt-6-luna; blocked by the OpenAI spend limit on 2026-10-30):

    set -a; . ./.env; set +a
    uv run python -I src/extract.py pass1 data/icml_position_reviews.parquet --tag _icmlpos --budget 5
    uv run python -I src/extract.py pass2 data/panels_icml_position.jsonl --tag _icmlpos --budget 5

Then compare (main 2026 − main 2025) with (position 2026 − position 2025). The position-track sample is small
(73 papers in 2025), so expect a standard error near 2 points.

## Extraction models compared (`validate.py instruments`, reports/instruments_gpt-6-luna.csv)

`validate.py instruments --model M` writes `reports/instruments_M.csv` with M as the reference model. The committed
`reports/instruments.csv` is the gpt-6-luna run made before that rename; the next `scripts/post.sh gpt-6-luna` writes
`reports/instruments_gpt-6-luna.csv` and does not change `reports/instruments.csv`.

Same prompts, schema and reasoning effort on the second-model set (1,797 ICLR panels) and the 40 blind papers.
OpenRouter models run through `extract.py` under short names (`glm-5.3-flash`, `deepseek-v4.1-flash`).

| | gpt-6-luna | gpt-6.1-sol | glm-5.3-flash | deepseek-v4.1-flash |
|---|---|---|---|---|
| Groups per panel | 21.4 | 22.6 | 13.9 | 20.6 |
| Std. change 2023-2026 | +2.1 (0.6) | +0.5 (0.5) | +0.7 (0.8) | +1.4 (0.6) |
| Std. change 2025-2026 | +3.7 (0.9) | +2.6 (0.8) | +4.7 (1.2) | +3.2 (0.9) |
| Gap two flagged vs none, 2024-26 | +3.7 (1.0) | +4.7 (0.9) | +4.1 (1.4) | +3.1 (1.0) |
| Blind bias vs Opus (pts) | -4.9 (1.3) | - | +3.3 (1.4) | -4.1 (1.1) |
| Blind Spearman vs Opus | 0.68 | - | 0.75 | 0.70 |
| Cost for this set (USD) | - | - | 3.8 | 27.2 |

The within-paper gap and the 2025-2026 rise hold under every model; the 2023-2026 rise is largest under the main
model. GLM merges about twice as often (levels about 2x) but ranks papers like the reference.

### Provider mix (GLM full run, 2026-10-31)

Unpinned `z-ai/glm-5.3-flash` on OpenRouter is served by about 40 endpoints at fp4/nvfp4/fp8 whose groupings differ.
The GLM full run is a provider mix and cannot be reproduced; replicas on one provider agree with each other. Pinned
first-party endpoint = short name `glm-zai`. Category agreement with the council gold: 0.52 mixed, 0.77 pinned.

### LLM-council gold (data/tune/) and prompt tuning

60 tuning panels (40 same-paper, 20 cross-paper placebo; dev 40 / test 20). Stage 1: gpt-6.1-sol, Opus 5.5 and
Kimi K3.1 Pro extract claims; 12 Opus 5.5 agents adjudicate gold claims with a support count. Stage 2: the three
models group the gold claims; 6 Opus agents adjudicate the gold grouping. This is an LLM-council reference, never
"human" in the paper. The council models fed the gold, so their own scores are upper bounds.

Against gold (dev / test): luna stage-1 recall 0.94 / 0.95, category 0.79 / 0.79; GLM pinned 0.92 / 0.91, 0.76 /
0.78. Stage-2 pairwise F1 on gold claims: council 0.82-0.88; luna base prompt low 0.78 / 0.64, medium 0.80-0.81,
high 0.84-0.86 / 0.73; GLM tuned strict rule high 0.80-0.83 / 0.70. End to end, same-paper hit-rate bias vs gold:
luna main run -1.2 (SE 0.9) / -6.0 (2.4); luna high-effort grouping -0.6 (0.6) / -4.8 (1.4); GLM tuned -1.2 / -3.4.
All cross-paper leakage of every model came from one "typos throughout" bucket per panel; luna at high effort has
none. Prompt rules beyond the base prompt gave no measurable gain for luna (a strict merge rule cut recall to 0.42),
and the 28 dev panels cannot resolve F1 changes under 0.04 or bias under 1 point. Decision: base prompts, luna through
OpenRouter at high effort in both passes (short name `gpt-6-luna-high`). Cheap alternatives screened on stage 2 (F1,
dev): Haiku 5.5 0.72, DeepSeek 4.1 flash 0.70, MiMo 0.79 with failures, MiniMax 0.54; none beats luna at luna's price.

## Terms (define once, never drift)

- "AI-flagged" means the style detector fired, not "AI-written".
- "Critique claim": one problem the authors would answer with one fix; give one example.
- "Match rate" (raw) and "standardized match rate": the only two names for the overlap measure. Do not use "homogenization" as a measured term; use "overlap" with its formula.
- Journals have "reports", conferences have "reviews"; say so once.

## Calibrated language

- Finding: "overlap rose". Speculation: "reviewers paste LLM output". Keep the mechanism in one paragraph labeled as speculation; the journal correlation (Fig. 4) is correlational.

## Baselines next to each claim

- Unflagged same-paper pairs (claim 1).
- Cross-paper pairs (claims 1–2).
- Length- and panel-size-standardized rate (claim 2).
- Confidence intervals at the paper level, not the review level.
- Mark the per-venue trend comparison as post hoc.

## New vs not new

- LLM-based critique extraction exists. The same-paper vs cross-paper contrast, the within-paper AI-flag comparison and the 2020–2026 trend are ours.

## Related work placement

- Three sentences in the introduction (LLM-review detection, review-quality studies, critique extraction); full related work, if space allows, before Limitations.

## Limitations (main text)

- Detector error, English only, venue coverage, LLM extraction noise, correlation not causation, LLM (not human) validation.

## Data window

- ICLR starts in 2020: before 2020 the public data lack almost all withdrawn papers (later 13–26% of submissions), and reviews of withdrawn (weak) papers match more. Evidence: `src/outcomes.py withdrawals`, Appendix A.
- ICLR 2026 papers that were withdrawn or desk-rejected after the early heineman snapshot: `corpus.py ml` takes their status from the final venue of the submission note (decision `withdrawn` or `desk_reject`). The analysis drops every panel that holds a review of a withdrawn or desk-rejected paper (`outcomes.excluded_review_ids`), so the existing extraction gives the corrected sample; `outcomes.py panels` and `pairs` print the number of dropped panels. They also drop a same-paper panel that holds a review no longer in `reviews.parquet` (for example an older version of a reviewer's own review, which `corpus.py ml` now removes); without that drop its pairs have no length and the standardized fit fails. The same rule applies to the per-review tables (`ai.py wordshare` and `wordshare_fields`, the `ai.py style` cell table, `between.py`, the flag columns of `figures.py table`, Fig. 3, Fig. A likeness, `validate.py instruments` and `drift`, and the 2026 rating ICC in `outcomes.py withdrawals`). Same-venue placebo panels mix reviews of four papers, so the 2026 placebo cells lose more panels than the same-paper cell. `outcomes.py withdrawals` ranks 2026 papers for the rating quintiles among the main sample only, as in every other year.

## Counting rules

- A review with no critique claim counts as a review with an empty claim set in Reviews, Claims/rev. and the rarefied S_k, D_k, U_k (also in the `_spec` and `_m8` subsets); it still enters no pair. `n_reviews` in the outcome tables is the panel size; `paper_n_reviews` is the paper's review count in `papers.parquet`.
- The summary+strengths style columns (`p_llm_ss`, `llm_flag_ss`) score the `[SUMMARY]` and `[STRENGTHS]` text up to the next section tag, and are missing (not False) when a review has no such text over 200 characters. They skip forms whose strengths field also holds the weaknesses (ICLR 2023, NeurIPS 2022 and 2025) but score summary-only forms (ICLR 2022, NeurIPS 2021). Their 99th-percentile threshold comes from the pre-period ICLR rows with such text, which are ICLR 2022 summaries alone, so it is not comparable with the summary+strengths text from 2024 on. No analysis reads these columns. `reports/within_gpt-6-luna.txt` comes from the removed `analysis_within.py` and an older version of these columns, so it is stale.

## Figure messages

| Fig | Message |
|---|---|
| 1 | Within a paper, two AI-flagged reviews share more critiques than two unflagged ones; every check gives the same sign. |
| 2 | Under the main extraction model, the same-paper match rate rose at ICLR and ICML in 2026 and at NeurIPS in 2025; the rate across papers did not rise. |
| 3 | Clarity and typo critiques fell while requests for more experiments rose. |
| 4 | Exploratory: the two journal sources whose AI-likeness index rose most had the largest rise in match rate. |

Appendix figures, one message each: AI-likeness by year; cross-paper rate by venue; same-area control; journal sources by year; journal cross-paper rates; whole-review embedding similarity.

## Color map

- ICLR blue, NeurIPS vermillion, ICML bluish green: venues only.
- Orange: both reviews AI-flagged; light orange: one flagged. Grey: unflagged, context and baselines.
- Squares: blind Claude Opus estimates; open circles: placebos.
- Black: single-series plots with no venue identity.
- ChatGPT period: one light grey band, the same in every time plot.

## Format

- ACL short paper: 4 pages of main text plus references and appendix.
- Figure widths: 3.03 in (one column), 6.3 in (full width).
- Font: Times-like serif to match the ACL template.
