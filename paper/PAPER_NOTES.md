# Twice-Told Critiques: paper-specific notes

How WRITING_GUIDE.md and FIGURE_GUIDE.md apply to this paper.

## Core claims

1. Within a paper, two AI-flagged reviews share more critique claims than two unflagged reviews of the same paper
   (+3.6 points, SE 0.3, ICLR 2024-26; gpt-6-luna-high run of 2026-11-01).
2. Same-paper overlap was flat 2020–2025 and rose in 2026 (ICLR +1.4 over 2023, SE 0.2); cross-paper overlap did not
   rise (0.5% to 0.3%).
3. The mix of critiques shifted from clarity (23% to 14%) and typos toward requests for more experiments.

Every number in main.tex comes from the gpt-6-luna-high run (reports/, tables/, claim embeddings through OpenRouter,
and the ad-hoc checks in the session scratch script adhoc.py: third reviewer, CoCoDet splits, controls, audit papers,
panel-size and length effects, second-model fine print). The two blind sets disagree on the sign of the main model's
drift (+3.3, SE 1.5, on the 2023/2026 set; -2.9, SE 2.1, on the first set). The word-use length-quarter check was not
rerun and its sentence was dropped.

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

## Extraction models compared (`validate.py instruments`, reports/instruments_<model>.csv)

`validate.py instruments --model M` writes `reports/instruments_M.csv` with M as the reference model; `scripts/post.sh`
runs it for the main model.

Same prompts, schema and reasoning effort on the second-model set (1,797 ICLR panels) and the 40 blind papers.
The table is `reports/instruments_<model>.csv` after `scripts/post.sh`; the reference model is the main model.

### GLM dropped (2026-11-01)

GLM-5.3-flash was tried as the second model family (full run 2026-10-31). Dropped: unpinned `z-ai/glm-5.3-flash` on
OpenRouter is served by about 40 endpoints at fp4/nvfp4/fp8 whose groupings differ, so the run was a provider mix
that cannot be reproduced; pinned to one provider and tuned it reached luna's agreement with the council gold at a
higher price. Its routes, reports and data files are removed; the remaining second-model checks are gpt-6.1-sol and
deepseek-v4.1-flash.

### LLM-council gold (data/tune/) and prompt tuning

60 tuning panels (40 same-paper, 20 cross-paper placebo; dev 40 / test 20). Stage 1: gpt-6.1-sol, Opus 5.5 and
Kimi K3.1 Pro extract claims; 12 Opus 5.5 agents adjudicate gold claims with a support count. Stage 2: the three
models group the gold claims; 6 Opus agents adjudicate the gold grouping. This is an LLM-council reference, never
"human" in the paper. The council models fed the gold, so their own scores are upper bounds.

Against gold (dev / test): luna stage-1 recall 0.94 / 0.95, category 0.79 / 0.79. Stage-2 pairwise F1 on gold
claims: council 0.82-0.88; luna base prompt low 0.78 / 0.64, medium 0.80-0.81, high 0.84-0.86 / 0.73. End to end,
same-paper hit-rate bias vs gold: luna main run -1.2 (SE 0.9) / -6.0 (2.4); luna high-effort grouping -0.6 (0.6) /
-4.8 (1.4).
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
