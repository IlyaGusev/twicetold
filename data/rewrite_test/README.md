# data/rewrite_test

The rewrite control (paper Appendix I). Two random reviews of each sampled pre-2023 paper are rewritten by
`gpt-6-luna` in two ways: from an outline of their own points (`ai_outline`, `ai_write`) and by a language polish that
keeps every point (`ai_polish`). On the 600 ICLR papers, `kimi-k3.1-pro` (`settings.REWRITE_SECOND_WRITER`) makes
both changes a second time with the same prompts. Each panel is grouped again with the originals and with each set of
two new texts in place of the two originals.

| File | Content |
|---|---|
| `papers.jsonl` | 800 papers: 150 ICLR 2020–2022 and 40 per journal source up to 2022 with at least 3 reports (seed 0), then 450 more ICLR 2020–2022 papers (seed 1); `panel_id`, `conf`, `year`, `review_ids`, `rewritten` (2 ids) |
| `rewrites_gpt-6-luna.jsonl` | Outline rewrites, fields as in `ai_reference_gpt-6-luna.jsonl` |
| `polishes_gpt-6-luna.jsonl` | Polished reviews: `review_id`, `conf`, `year`, `text` |
| `rewrites_kimi-k3.1-pro.jsonl`, `polishes_kimi-k3.1-pro.jsonl` | Kimi outline rewrites and polished reviews of the 600 ICLR papers (1,200 each), fields as above |
| `reviews_ml.parquet`, `reviews_general.parquet` | Luna rewrite texts with id `rw_<review_id>`, Luna polished texts `pl_<review_id>`, Kimi rewrite texts `rk_<review_id>` and Kimi polished texts `pk_<review_id>` (ICLR only), input of pass 1 |
| `panels_ml.jsonl`, `panels_general.jsonl` | `rw0_<paper>` (original reviews), `rw2_<paper>` (two outline rewrites in place), `rwp_<paper>` (two polished reviews in place), `rk2_<paper>` and `rkp_<paper>` (the same with Kimi texts, ICLR only), `pres_<review_id>`, `ppres_<review_id>`, `kpres_<review_id>` and `kppres_<review_id>` (original and its new text) |
| `panels_pairs_ml.jsonl`, `panels_pairs_general.jsonl` | The two changed reviews of each paper alone: `pair_rw0_<paper>` (originals) and `pair_<panel prefix><paper>` for each condition (`pair_rw2_`, `pair_rwp_`, `pair_rk2_`, `pair_rkp_`); 3,000 ML and 600 general panels |
| `panels_audit.jsonl` | 100 panels each of `pres_`, `ppres_`, `kpres_` and `kppres_` (seed 0) for the Opus 5.5 match |
| `claims_gpt-6-luna-high.jsonl` | Pass-1 claims of the original reviews (copied from `data/`) and of the rewrites |
| `clusters_gpt-6-luna-high.jsonl` | Pass-2 groups of all panels, including the pair panels; one journal pair panel (`pair_rw2_other_2107.14637v2`) holds an `error` row after two retries, and `rewrite_test` skips it |
| `clusters_opus-5.5.jsonl` | Opus 5.5 groups of audit panels; `rewrite_test` reads only the panels in `panels_audit.jsonl`; Opus returns no content for 4 `pres_`, 1 `kpres_` and 1 `kppres_` audit panels, which hold `error` rows |

## Make

The rewrite chain of `scripts/extract_all.sh`: `src/ai.py rewrite_sample`, `rewrite_panels`, `rewrite_inputs`, `rewrite_pairs`, then
`src/extract.py pass1`, `pass2`, `merge` with `--data data/rewrite_test`, then the Opus pass 2 on `panels_audit.jsonl`.
Paid APIs (OpenAI for the Luna rewrites, OpenRouter for the Kimi rewrites and the passes). Model output is not deterministic.

## Read by

`src/meld.py input` (sets `rewrite_panel`, `polish_panel`, `rewrite_panel_second`, `polish_panel_second`), `src/validate.py rewrite_test`.
