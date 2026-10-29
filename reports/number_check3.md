# Number check 3: paper/main.tex after the full-sample sub3/noq rerun

Checked against main.tex md5 8c035492d623b5cf2792382d1af5b09b (07:37) and main.pdf (07:37, same text).
No file other than this report was changed. No model API was called.

## Rerun status

- `outcomes.py` rerun did not move the main outputs. New `panel_outcomes` year means equal `year_means_gpt-6-luna.csv` (max diff 0.000 on S3, D3, U3, share shared, hit rate, pair rate, claims/review, specific share, paper counts). New `placebo_outcomes` equals `placebo_gpt-6-luna.csv`. New `review_outcomes` gives the same ICLR 2024-26 share-shared mean (0.2358) as `within_paper_gpt-6-luna.csv`.
- All ten `paper/tables/*.tex` are byte-identical to a fresh `make_tables.py` run on the current reports (run into the scratchpad).
- So only the variant numbers and the API cost changed.

## Mismatches

| # | Line | Quoted text | Correct value | Source |
|---|---|---|---|---|
| M1 | 85 | "about \$120 of the study's \$170 for the full sample and its placebo and robustness panels" | gpt-6-luna cost is now about \$158 (\$120.12 before the rerun + \$37.43 for the rerun: \$12.94 pass 1 noq, \$9.56 pass 2 sub3, \$14.93 pass 2 noq). Logged gpt calls alone total \$198 (luna \$157.55 + sol \$40.48), before Opus annotation. | logs/chain_full.log; finished lines in logs/pass1_*.log, pass2*_*.log, sol.log, pilot_sol.log |
| M2 | 217 | "The study used about 170 USD of commercial API calls" | At least 198 USD from logged gpt calls; about 208 USD if the unlogged rest (about 10 USD) is unchanged. | same as M1 |
| M3 | 127 | "though 2023 itself was 1.2 below 2021--2022" | 1.3. D3: 2021 19.493, 2022 19.355, 2023 18.127; mean gap 1.297. Raw year effects: 2021 +1.366, 2022 +1.227. | reports/year_means_gpt-6-luna.csv; reports/year_effects_gpt-6-luna.csv |
| M4 | 137 | "$+0.37$ (SE 0.14) on specific claims" | SE 0.13 (S3_spec 2026, raw: +0.3697, SE 0.1348). Check 2 (V56) rounded 0.1348 to 0.135 and then to 0.14. | reports/year_effects_gpt-6.1-sol.csv |
| M5 | 328 | "$+0.37$ (0.14) for $S_3$ on specific claims" | (0.13). Same cause as M4. | reports/year_effects_gpt-6.1-sol.csv |
| M6 | 334 | "with the true 2023--2026 areas and the share of reviews flagged, $-0.23$ (0.44)" | -0.232 (0.436) uses `dose_any`, the share of review pairs with at least one flagged review (dose range 0.01-0.77), not the share of reviews flagged. No report has a true-area estimate with the share of reviews flagged. Write "the share of review pairs with a flagged review". | reports/extra3_gpt-6-luna.txt; src/analysis_extra3.py line 31 |

## Verified (focus items)

| Line | Quoted text | Source value |
|---|---|---|
| 88 | "re-cluster a random three-review subset from scratch (all 2023 and 2026 papers with four or more reviews)" | 2,622 and 4,918 = all sampled ICLR 2023 and 2026 papers with n_reviews >= 4 (panel_outcomes count); sub3 total 9,786 panels (chain_full2.log) |
| 137 | "gives $+0.25$ (SE 0.03) from 2023 to 2026" | variant_sub3: 1.7165 - 1.4634 = 0.253; SE sqrt(0.0265^2 + 0.0201^2) = 0.033 |
| 137 | "against $+0.29$ rarefied on the same papers" | 1.7038 - 1.4134 = 0.290 |
| 137 | "re-running both passes on 12,000 papers" | variant_noq n: 3,000 + 3,000 + 6,000 |
| 137 | "lowers every level" | noq below full in every year for S3, share shared, claims/review, D3 |
| 137 | "$S_3$ $+0.37$ from 2024 to 2026 against $+0.44$ with the full text" | 1.2662 - 0.8923 = 0.374; 1.7042 - 1.2604 = 0.444 |
| 317 | caption: "2,622 and 4,918; 400 papers in other years", "3,000 papers per year, 6,000 in 2026", "SEs in parentheses" | variant_sub3 n (400 in 2021, 2022, 2024, 2025); variant_noq n; all 30 variants.tex cells match the CSVs |
| 127 | "$+0.275$ vs. 2023, SE 0.024, $+19\%$; $+0.29$ vs. 2021--2023 pooled" | 0.2749/1.429 = 19.2%; extra: +0.289 |
| 127 | "$+0.44$ from 2024 to 2026 (share shared 0.177 to 0.235)"; "2025 is 0.08 below" | 1.704 - 1.260; year_effects -0.078 |
| 137 | excess 1.60 vs 1.17 ($+0.43$, SE 0.03); placebo 0.10 vs 0.17-0.31 | placebo.tex (2022 placebo 0.1748) |
| 137 | $+0.276$; $+0.42$ on base 1.01; 1.13 to 1.52; $+0.24$ to $+0.31$; $+0.17$ (SE 0.17) | year_effects.tex; panel_outcomes S3_spec 2023 1.007; extra2 substantive; year_effects_gpt-6.1-sol |
| 139 | NeurIPS 1.08-1.11, 1.27, $+0.16$ (0.04), $+0.041$ (0.006) | year_means; year_effects |
| 145 | 1.44 (mean 6.8), 1.48, 0.047 (0.002, mean 0.24), $+0.030$, $+0.034$, $+0.027$, $+0.021$ | within.tex; extra |
| 154 | 0.038 (0.003), base 0.089, +43%, 0.035, 0.005, $+0.040$; mix -0.003 of +0.018, +0.005 of +0.031; 0.083 to 0.100; 0.009 to 0.022 (2.49x); $+0.016$ (0.001); 0.081 to 0.097 | within.tex; pairs_note.tex; decomposition csv; extra2; extra3 |
| 157 | -0.002 (0.002), +0.08 (0.02); +0.042 (0.019), -0.77; 0.21 vs 0.20, +0.03 (0.04) | spillover csv; within.tex; reliability csv |
| 160 | 0.50/0.19, 0.275/0.091, 15.4/24.4, 0.70/0.86, 15% | llm_panels.tex; (0.235-0.189)/(0.498-0.189) = 0.149 |
| 163 | -0.58 (0.35), +0.11, +0.62, 0.32 to 0.54 | extra; did txt |
| 169 | 24.5 to 15.3, 5.8 to 3.1, 21% to 34%, 10%, 15%, 0.27 to 0.20, 267/273/230, 260 to 244, 214 vs 254 | categories.tex; category_shift; types.tex; template_by_cell; extra3 (259.7/260.0, 243.5, 213.9, 253.8) |
| 186, 189 | 43%, more than forty times, 17% | pairs_within; detector_by_cell 0.4448 |
| 195, 207 | +-0.06; 2.05 vs 1.70, 0.264 vs 0.235 (n 1,000) | validate_blind SE 0.029; variant_wd (rerun, same values) |
| 282 | 30 and 7,312 both-flagged pairs; 667 papers | pairs_within n_t; check 2 |
| 328-336 | all other appendix values (sol +0.029, +0.057, 1,177, +0.042, 1.5; blind biases; replicate; dose; event study; -1.07 (0.50); terciles; decomposition; third reader) | year_effects/within_paper sol; validate_blind; did csv; event_study; extra3; decomposition; reliability |

## Unverifiable

- Line 85 and 217: the non-gpt part of the total cost (Opus reference annotation, embeddings). No log or report gives it, so the exact new total is open. Only the lower bound (198 USD) is in logs.

## Notes (not counted)

1. Lines 127 and 189: the noq rerun shows that the questions field raises overlap. In 2024, S3 is 1.26 with it and 0.89 without; share shared 0.177 and 0.167. So "The year the form added a questions field, 2024, sits below 2023" and "the 2024 form change lowered overlap" cannot rest on the questions field. Only the S/W split is left as a cause, and no variant tests it.
2. Line 169: "related-work claims by a third": 4.80% to 3.37% is -30% (category_shift). "by 30%" is exact.
3. Line 169: "a combined 21% to 34%" is correct on unrounded values (21.14, 34.49). The five rounded table cells sum to 21.2 and 34.5, so a reader who adds the table gets 35%.
4. Lines 127 and 137 both give "+0.29" for two different contrasts (2026 vs pooled 2021-23 on all papers; 2023 to 2026 rarefied on the sub3 papers). Both are correct.
5. Lines 94 and 296: "300 panels per year". placebo csv same_area3 has 297 (2017), 299 (2019) and 298 (2023) panels with claims.
6. Data section items that checks 1 and 2 could not find now match data_facts.txt: 2,187 withdrawn, 95% accepted (0.951), 0.1% after release (0.0011), 23% and 31% rating changes (0.229, 0.314).

Mismatches found: 6.
