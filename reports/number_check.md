# Number check: paper/main.tex against reports/

Scope: Abstract, Introduction (Findings), Data, Results, Discussion, Limitations, Appendix "Additional Results", and paper/tables/*.tex.
Checked against main.tex md5 1c96f0c0a3ad7b4a7c5311827d2655a1. The file changed during the check, so all quotes come from that final version.
No file other than this report was changed. No model API was called.

Sources outside the listed reports (used only when no report had the number): logs/outcomes_luna.log, logs/detector2.log, data/llm_panel_outcomes_gpt-6-luna.parquet, data/panel_outcomes_gpt-6-luna.parquet, data/pairs_outcomes_gpt-6-luna.parquet.

## Mismatches

| # | Paper fragment | Source value | Status |
|---|---|---|---|
| M1 | Intro Findings: "shared 1.4 of their critiques per three reviews every year from 2021 to 2025" | year_means S3: 2021 1.373, 2022 1.436, 2023 1.429, 2024 1.260, 2025 1.352. 2024 rounds to 1.3. | MISMATCH. Write "1.3--1.4" or "1.4 on average" (mean 1.37). |
| M2 | Results: "17.5--19.5 critiques were distinct" (2021--2025) | year_means D3: 19.49, 19.36, 18.13, 17.87, 18.55 | MISMATCH. Correct: 17.9--19.5. |
| M3 | Results: "claims per review do not change ($-0.05$)" | year_effects claims_per_review 2026: raw (primary) +0.075 (SE 0.051). The value -0.050 (0.041) is the full-controls spec. The same sentence gives D3 from the raw spec. | MISMATCH. Primary spec gives +0.08. |
| M4 | Results: "move the 2026 coefficients by less than one standard error of each other" | year_effects 2026, raw/full/unedited/area-reweighted: S3 0.242 to 0.308 (spread 0.066, SE 0.022--0.029, about 2.5 SE); share shared 0.041 to 0.047 (spread 0.006 = 2 SE); hit rate 0.016 to 0.018 (about 1 SE). | MISMATCH. True only for hit rate. |
| M5 | Abstract: "with 45\% of reviews in an LLM style" | detector_by_cell ICLR 2026 flag 0.4448 | MISMATCH. 44\%. |
| M6 | Data: "45\% in 2026 at ICLR" | detector_by_cell 0.4448 | MISMATCH. 44\%. |
| M7 | Spillover: "raised the share of reviews in that style from 1\% to 45\%" | detector_by_cell 0.4448 | MISMATCH. 44\%. |
| M8 | Results: "2024 and 2025, with 16\% and 32\% of reviews flagged" | detector_by_cell 2024 0.1651, 2025 0.3243. The Data section gives 16.5\%. | MISMATCH (2024). Use 17\% or 16.5\%. 32\% is correct. |
| M9 | E3: "makes 1.44 fewer claims than its co-reviews (on a mean of 6.6)" | within_gpt-6-luna.txt ICLR 2024--2026 llm_flag n_claims -1.4421, mean 6.844. 6.64 is the 2024-only mean. | MISMATCH (mean). Correct: 6.8. The value 1.44 is correct. |
| M10 | E3: "a share of shared claims 0.047 higher (SE 0.002, on a mean of 0.20)" | same block: +0.0474 (SE 0.0023), mean 0.236. 0.203 is the 2024-only mean. | MISMATCH (mean). Correct: 0.24. The coefficient and SE are correct. |
| M11 | Pairs: "$0.21\times0.048+0.46\times0.013=0.016$" | pairs_within ICLR 2026-2026 hit: LL 0.04747, HL 0.01248. These round to 0.047 and 0.012, as Table within prints. Shares: LL 7312/34311 = 21.3\%, HL 15903/34311 = 46.3\%. Recomputed: 0.2131 x 0.0475 + 0.4635 x 0.0125 = 0.0159. | MISMATCH (inputs). Use 0.047 and 0.012. The result 0.016 and the shares 21\% and 46\% stay correct. |
| M12 | E5: "with a hit rate of 0.275 against 0.091" | llm_panels CSV and Table llm: llm4 0.280, hum4 0.089 (equal-weight mean of 2022, 2023 and 2024). The pooled mean over the 667 papers (parquet) is 0.275 and 0.091. | MISMATCH against the table and CSV. The text uses a different weighting than the table. Choose one weighting for both. |
| M13 | E5: "15.4 distinct critiques against 24.4 per four reviews" | Table llm D4: 16.0 and 24.3. Pooled: 15.37 and 24.44. | MISMATCH against the table and CSV. Same cause as M12. |
| M14 | E5: "a specific share of 0.69 against 0.85" | Table llm: 0.717 and 0.859. Pooled: 0.699 and 0.855. | MISMATCH under both weightings. Pooled gives 0.70 and 0.86. Table gives 0.72 and 0.86. |
| M15 | Discussion: "raise 37\% fewer distinct critiques than four humans" | Table llm D4 16.04/24.34 gives 34\% fewer. Pooled 15.37/24.44 gives 37.1\% fewer. | MISMATCH against the table and CSV. Same cause as M12. |
| M16 | E4: "2026 flagged shares run from 0.37 to 0.52 across areas" | did_gpt-6-luna.txt 2026 column: min 0.315 (learning theory), max 0.535 (datasets and benchmarks) | MISMATCH. Correct: 0.32 to 0.54. |
| M17 | E6: "(272 in 2020)" | template_by_cell hill_types_20k ICLR 2020 = 272.57. 2021 = 271.92. | MISMATCH. 2020 rounds to 273. 272 is the 2021 value, as Table types shows. |
| M18 | Validation: "its share of claims shared is 0.09 below the reference" | validate_blind: 40-paper means 0.286 (Opus) and 0.205 (model), gap 0.081. 2026-only bias is -0.093. | MISMATCH. Overall gap is 0.08. |
| M19 | Validation: "moves it between 1.09 and 1.21 around the raw 1.13" | pair_rate_strict 2026/2023 = 0.018167/0.016099 = 1.128. Precision only: x 0.87/0.81 = 1.212. Recall only: x 0.7168/0.7982 = 1.013. Both corrections together: 1.128 x 1.074 x 0.898 = 1.089. | MISMATCH under the two one-sided formulas. The range is 1.01--1.21. The value 1.09 is the joint precision-and-recall correction, not a bound. Raw 1.13 and 1.21 are correct. |
| M20 | Appendix: "Within papers (ICLR 2024--2026, 1{,}177 panels)" | within_paper_gpt-6.1-sol: n = 1177 in the review-level regression. year_means sol: 100 papers per year, about 3.9 reviews per paper, so 300 papers. | MISMATCH. 1,177 is reviews, not panels. |
| M21 | Validation: "Two independent Opus replicates agree with the reference at $\rho{=}0.93$--$0.97$ with no bias" | validate_blind shows one reference-vs-replicate comparison on 20 papers. rho: S3 0.96, D3 0.97, share shared 0.94, shared clusters 0.93, but hit rate 0.87, pair rate 0.91, specific share 0.85 (bias -0.028, SE 0.011), lenient share 0.54 (bias -0.046, SE 0.027). | MISMATCH (scope). The range and "no bias" hold only for S3, D3, share shared and shared clusters. Name those outcomes. |

## Not found

| # | Paper fragment | Where looked | Status |
|---|---|---|---|
| N1 | Validation: "On AMPERE, extracted claims track evaluative and request propositions ($\rho{=}0.78$)" | reports/, logs/. src/validate_ampere.py prints this value, but no saved output exists. | NOT FOUND |
| N2 | Validation: "and cover 68\% of requests" | same | NOT FOUND |
| N3 | Data: "dropping the 2,188 withdrawn ICLR 2026 submissions that have reviews" | reports/, logs/ | NOT FOUND |
| N4 | Data: "NeurIPS publishes mainly accepted papers (95\%)" | reports/, logs/ | NOT FOUND |
| N5 | Data: "59\% five-fold accuracy" | reports/, logs/ | NOT FOUND |
| N6 | Data: "(26\% and 32\% did)" (rating changed after response, ICLR 2024--2025) | reports/, logs/ | NOT FOUND |
| N7 | Data: "All ICLR 2023 reviews but 0.1\% were created before 30 November 2022" | reports/, logs/. logs/detector2.log has no timestamp count. | NOT FOUND |
| N8 | Data: "trained on 2,000 human reviews of ICLR 2021--2023 papers" | logs/detector2.log gives "train 57692" rows only | NOT FOUND |

## Verified

| # | Paper fragment | Source value | Status |
|---|---|---|---|
| V1 | Abstract: "117k public reviews" | logs/outcomes_luna.log: reviews 116847 | VERIFIED |
| V2 | Abstract: "31k ... papers" | year_means papers sum 30,937 | VERIFIED |
| V3 | Abstract: "793k atomic critique claims" | template_by_cell claims sum 793,306 | VERIFIED |
| V4 | Abstract: "From 2021 to 2025 three reviewers of the same ICLR paper shared 1.4 critiques" | mean of S3 2021--2025 = 1.37 | VERIFIED (as an average; see M1 for "every year") |
| V5 | Abstract: "in 2026 ... they share 1.7" | S3 2026 1.704 | VERIFIED |
| V6 | Abstract: "a 19\% rise" | 0.275/1.429 = 19.2\% | VERIFIED |
| V7 | Abstract: "survives re-clustering" | variant_sub3 2026 1.73 vs rarefied 1.71 | VERIFIED |
| V8 | Abstract: "removing the questions field" | variant_noq S3 0.889 (2024) to 1.233 (2026) | VERIFIED |
| V9 | Abstract: "keeping only specific claims" | S3_spec 2026 +0.418 (0.021) | VERIFIED |
| V10 | Abstract: "and a second extractor" | year_effects sol S3 2026 +0.169 (0.170), p = 0.32 | VERIFIED (sign only; not significant) |
| V11 | Abstract: "NeurIPS rose the same way in 2025" | NeurIPS S3 2025 vs 2022 +0.156 (0.043) | VERIFIED |
| V12 | Abstract: "two such reviews match each other half again as often as two human reviews" | pairs_within LL +0.038 on HH mean 0.089 = +43\%; 2026 +0.047 on 0.100 = +47\% | VERIFIED (approx.) |
| V13 | Abstract: "from 267 to 230 effective critique types" | hill_types_20k 2023 267.3, 2026 230.5 | VERIFIED |
| V14 | Abstract: "600 adjudicated claim pairs" | pairs_validation: 600 pairs with both adjudications | VERIFIED |
| V15 | Intro Findings: "and 1.7 in 2026" | 1.704 | VERIFIED |
| V16 | Intro Findings: "a 19\% rise" | 19.2\% | VERIFIED |
| V17 | Intro Findings: "half again as often" | +43\% | VERIFIED (approx.) |
| V18 | Intro Findings: "from 267 to 230" | 267.3, 230.5 | VERIFIED |
| V19 | Data: "249{,}903 reviews of 64{,}436 papers" | sum of Table corpus rows: 249,903 reviews, 64,436 papers | VERIFIED (internal consistency) |
| V20 | Data: "3{,}000 random papers each for ICLR 2024 and 2025, 6{,}000 for ICLR 2026, and 1{,}000 per year for NeurIPS" | year_means papers | VERIFIED |
| V21 | Data: "30{,}937 papers and 116{,}847 reviews" | year_means; logs/outcomes_luna.log | VERIFIED |
| V22 | Data: "primary area (21 areas)" | did_gpt-6-luna.txt: 21 area rows | VERIFIED |
| V23 | Data: "tested on the held-out 2024 pairs (AUC 0.99)" | logs/detector2.log AUC 0.9939 | VERIFIED |
| V24 | Data: "pre-ChatGPT false-positive rate is 1\% by construction" | detector_by_cell ICLR 2017--2022: 0.65\% to 1.54\% | VERIFIED |
| V25 | Data: "2.8\% in 2023" | 0.0282 | VERIFIED |
| V26 | Data: "16.5\% in 2024" | 0.1651 | VERIFIED |
| V27 | Data: "32\% in 2025" | 0.3243 | VERIFIED |
| V28 | Data: "14\%, 27\% and 41\% at NeurIPS 2023--2025" | 0.1439, 0.2660, 0.4093 | VERIFIED |
| V29 | Data: "AUC 0.92; 71\% of flagged reviews are labelled AI, against 18\% overall" | logs/detector2.log ICLR 2024: AUC 0.918, P(isAI given flag) 0.711, isAI 0.177 | VERIFIED |
| V30 | Results: "793{,}306 claims from the 116{,}847 sampled reviews (6.8 per review)" | 793,306 / 116,847 = 6.79 | VERIFIED |
| V31 | Fig. 1 caption: "shared 1.43 of their critiques in 2023 and 1.70 in 2026, while the placebo fell" | S3 1.429, 1.704; same-area placebo 0.310 to 0.104 | VERIFIED |
| V32 | Validation: "On the 40 blind papers ... $D_3$: $\rho{=}0.90$, $S_3$: 0.76, share shared 0.68, hit rate 0.68" | validate_blind: 0.90, 0.76, 0.68, 0.68 | VERIFIED |
| V33 | Validation: "29.8 against 25.0 per paper in 2026" | n_claims 2026 model 29.8, Opus 25.0 | VERIFIED |
| V34 | Validation: "$-0.002$ (SE 0.029) ... $+0.20$ (0.28) ... $+0.02$ (0.02)" | TOST block: -0.002 (0.029), +0.200 (0.279), +0.020 (0.024) | VERIFIED |
| V35 | Merge: "agree on same problem for 97\% of pairs ($\kappa{=}0.93$)" | 0.968, kappa 0.933 | VERIFIED |
| V36 | Merge: "precision 0.84 and recall 0.76" | 0.840, 0.757 | VERIFIED |
| V37 | Merge: "lower in 2017--2022 (0.81) than in 2026 (0.87)" | 0.81, 0.87 | VERIFIED |
| V38 | Merge: "flagged review (0.88) as without (0.86)" | 0.88, 0.86 | VERIFIED |
| V39 | Merge: "recall is lower on flagged pairs (0.76 against 0.84)" | 0.759, 0.843 | VERIFIED |
| V40 | Merge: "the raw 1.13" and upper value "1.21" | 1.128; 1.212 | VERIFIED (see M19 for 1.09) |
| V41 | ICLR: "a three-review panel shared 1.26--1.44 critiques" (2021--2025) | 1.260 to 1.436 | VERIFIED |
| V42 | ICLR: "share of claims in a shared cluster stayed between 0.175 and 0.193" | 0.175 to 0.193 | VERIFIED |
| V43 | ICLR: "1.70 critiques ($+0.275$ vs. 2023, SE 0.024, $+19\%$)" | 1.704; +0.2749 (0.0239); 19.2\% | VERIFIED |
| V44 | ICLR: "share of claims shared is 0.235 ($+0.045$, SE 0.003, $+24\%$)" | 0.235; +0.0452 (0.0029); 0.045/0.190 = 23.8\% | VERIFIED |
| V45 | ICLR: "hit rate is 0.115 ($+0.017$, SE 0.002)" | 0.115; +0.0174 (0.0018) | VERIFIED |
| V46 | ICLR: "falls from 0.916 to 0.899" | U3 0.916, 0.899 | VERIFIED |
| V47 | ICLR: "$D_3$: $-0.11$, SE 0.13" | -0.109 (0.133) | VERIFIED |
| V48 | ICLR: "overlap fell by 0.16 per year from 2017 to 2023" | trends txt pre-trend S3 -0.1647 (0.0089), with length control | VERIFIED |
| V49 | Placebo: "share 0.10 critiques, down from 0.31 in 2023" | same_area3 0.1037, 0.3103 | VERIFIED |
| V50 | Placebo: "excess ... grows from 1.12 to 1.60 ($+0.48$, SE 0.05)" | 1.119 to 1.600; diff 0.481; SE sqrt(0.040^2 + 0.025^2) = 0.047 | VERIFIED |
| V51 | "(2026: 1.73 against 1.71)" | variant_sub3 1.7318, 1.7100 | VERIFIED |
| V52 | "share shared 0.157 in 2024, 0.211 in 2026" (questions removed) | variant_noq 0.1569, 0.2107 | VERIFIED |
| V53 | "Capping claims at eight ... ($S_3$: $+0.276$)" | S3_m8 2026 +0.276 | VERIFIED |
| V54 | "($+0.42$ on a 2023 base of 1.01; share shared $+0.066$ on 0.147)" | S3_spec +0.418; share_spec +0.066; 2023 bases from panel_outcomes parquet: 1.007, 0.147 | VERIFIED |
| V55 | "\texttt{gpt-6.1-sol} on 100 papers per year, gives $+0.17$ (SE 0.17) ... $+0.37$ (SE 0.14)" | sol S3 +0.169 (0.170); S3_spec +0.370 (0.135); 100 papers per year | VERIFIED |
| V56 | "its sample is thirty times smaller" | ICLR papers 25,937 vs 1,000 = 26x | VERIFIED (approx.) |
| V57 | NeurIPS: "$S_3$ is flat from 2022 to 2024 (1.08--1.11) and rises to 1.27 in 2025 ($+0.16$ vs. 2022, SE 0.04)" | 1.114, 1.087, 1.081; 1.270; +0.156 (0.043) | VERIFIED |
| V58 | NeurIPS: "the share shared rises by 0.041 (SE 0.006)" | +0.041 (0.006) | VERIFIED |
| V59 | E3: "has 1.48 fewer claims that no co-reviewer raises" | n_unique -1.4763 | VERIFIED |
| V60 | E3: "less often specific, less often about clarity and typos, and more often about the scope of the experiments" | -0.0213, -0.0670, -0.0202, +0.0318 | VERIFIED (signs) |
| V61 | E3: "GPTZero labels for 2024 give the same signs ($+0.021$ for the share shared, SE 0.006)" | gz_isAI +0.0209 (0.0057); other signs match | VERIFIED |
| V62 | E3: "NeurIPS 2023--2025 replicates every coefficient" | all signs match | VERIFIED |
| V63 | Pairs: "hit rate 0.038 above ... (SE 0.003, on a human--human mean of 0.089)" | LL 0.0380 (0.0026); mean_HH 0.0894 | VERIFIED |
| V64 | Pairs: "0.035 above on specific claims" | hit_spec LL 0.0351 | VERIFIED |
| V65 | Pairs: "a human--flagged pair is only 0.005 above" | HL 0.0046 | VERIFIED |
| V66 | Pairs: "The gap holds in every year from 2024 to 2026" | not in reports; recomputed from pairs_outcomes parquet with paper fixed effects: LL 2024 +0.025, 2025 +0.028, 2026 +0.047 (2026 equals the report) | VERIFIED (point estimates only) |
| V67 | Pairs: "and at NeurIPS ($+0.040$)" | NeurIPS LL 0.0396 | VERIFIED |
| V68 | Pairs: "flagged--flagged pairs are 21\% of ICLR 2026 pairs and human--flagged pairs 46\%" | 21.3\%, 46.3\% | VERIFIED |
| V69 | Pairs: "matches the observed rise ... ($+0.017$)" | hit_rate_strict 2026 +0.0174 | VERIFIED |
| V70 | Spillover: "$-0.002$ (SE 0.002) ... $+0.08$ (SE 0.02)" | share_shared -0.0022 (0.0017); n_unique +0.0791 (0.0230) | VERIFIED (see note 2) |
| V71 | Spillover: "($+0.042$ for the share shared, SE 0.019; $-0.77$ claims)" | ICLR 2019--2022 llm_flag +0.0418 (0.0186); -0.7663 | VERIFIED |
| V72 | Spillover: "The one percent of 2019--2022 reviews that the detector flags" | 0.69\%, 0.84\%, 0.65\%, 1.54\% | VERIFIED |
| V73 | E5: "On the same 667 papers" | parquet: 667 panels per kind | VERIFIED |
| V74 | E5: "four LLM reviews share 0.50 of their claims against 0.19" | table 0.503, 0.186; pooled 0.498, 0.189 | VERIFIED |
| V75 | E5: "one seventh of the way from the human floor to the all-LLM ceiling" | (0.235 - 0.186)/(0.503 - 0.186) = 0.153 (table); 0.150 (pooled). 1/7 = 0.143, 1/6 = 0.167. | VERIFIED (approx.; closer to 1/6.5) |
| V76 | E4: "changes $S_3$ by $-0.58$ (SE 0.35)" | -0.575 (0.350) | VERIFIED |
| V77 | E4: "an event study ... shows neither a pre-trend nor a 2026 divergence" | event_study S3: no p < 0.05 | VERIFIED |
| V78 | E6: "specific claims rises from 0.84 in 2023 to 0.90 in 2026" | 0.842 / 0.901 (template_by_cell); 0.835 / 0.899 (year_means) | VERIFIED |
| V79 | E6: "template share ... falls from 0.27 to 0.20" | 0.266, 0.203 | VERIFIED |
| V80 | E6: "on 20{,}000 claims per year, falls from 267 in 2023 ... to 230 in 2026" | 267.3, 230.5 | VERIFIED |
| V81 | E6: "share of claims in the 150 rarest pre-period types from 0.37 to 0.33" | tail_share 0.372, 0.332 | VERIFIED |
| V82 | E6: "NeurIPS follows the same path" | NeurIPS hill 259 to 243; tail 0.372 to 0.349 | VERIFIED |
| V83 | Discussion: "a kind of review that LLMs made forty times more common" | 1\% to 44.5\% = 44x | VERIFIED (approx.) |
| V84 | Discussion: "Four LLM reviews of one paper share half their claims" | 0.498 to 0.503 | VERIFIED |
| V85 | Discussion: "passed 40\% at each venue (ICLR 2026, NeurIPS 2025) and not before" | ICLR 2026 44\%, NeurIPS 2025 41\%; ICLR 2025 32\%, NeurIPS 2024 27\% | VERIFIED |
| V86 | Discussion: "flagged reviews already overlapped more in 2024" | ICLR 2024 llm_flag share_shared +0.0280 (0.0059) | VERIFIED |
| V87 | Limitations: "claim-pair set we release (600 pairs)" | 600 | VERIFIED |
| V88 | Limitations: "1{,}000 of them that we re-ran ... ($S_3$ 2.05 against 1.70; share shared 0.264 against 0.235)" | variant_wd n 1000; 2.0451 vs 1.7042; 0.2637 vs 0.2348 | VERIFIED |
| V89 | Appendix sol: "$+0.029$ (0.018) for the share shared ... $+0.057$ (0.018) ... on specific claims" | +0.029 (0.018); +0.057 (0.018) | VERIFIED |
| V90 | Appendix sol: "$+0.042$ higher (SE 0.015) and 1.5 fewer claims (0.25)" | +0.0420 (0.0153); -1.505 (0.252) | VERIFIED |
| V91 | Appendix: share shared by year "2018 $-0.109$ (0.039), 2022 $-0.074$ (0.032), 2024 $-0.049$ (0.045), 2026 $-0.093$ (0.014)" | identical | VERIFIED |
| V92 | Appendix: S3 by year "$-0.70$ (0.37), $-0.20$ (0.24), $-0.10$ (0.19), $-0.25$ (0.17)" | -0.700 (0.367), -0.200 (0.238), -0.100 (0.194), -0.250 (0.171) | VERIFIED |
| V93 | Appendix: "over 10 papers per year" | 40 papers, 4 years | VERIFIED |
| V94 | Appendix: "$S_3$ 1.80 against 1.88 ($\rho{=}0.96$), $D_3$ 17.1 against 17.0 (0.97), share shared 0.269 against 0.269 (0.94)" | 1.800/1.875 (0.96); 17.137/16.988 (0.97); 0.269/0.269 (0.94) | VERIFIED |
| V95 | Appendix dose: "with area trends $-0.25$ (0.43); 2024--2026 only $-0.21$ (0.39); share shared $-0.058$ (0.039)" | -0.248 (0.434); -0.209 (0.393); -0.058 (0.039) | VERIFIED |
| V96 | Appendix dose: "lie between $-0.09$ and $+0.24$ with SEs of 0.05--0.21, and none is significant" | min -0.093, max +0.237; SE 0.053 to 0.206; min p 0.076 | VERIFIED |
| V97 | tables/year_effects.tex: all 66 cells and stars | regenerated from year_effects_gpt-6-luna.csv; every cell matches | VERIFIED |
| V98 | tables/placebo.tex: all cells | placebo_adjusted_gpt-6-luna.csv | VERIFIED |
| V99 | tables/types.tex: all cells | template_by_cell_gpt-6-luna.csv | VERIFIED |
| V100 | tables/type_shift.tex: all 16 rows | type_shift_gpt-6-luna.csv (x100) | VERIFIED |
| V101 | tables/variants.tex: all cells | variant_sub3, variant_noq CSVs | VERIFIED |
| V102 | tables/pairs_validation.tex: all cells | pairs_validation.csv | VERIFIED |
| V103 | tables/within.tex: all cells | within_gpt-6-luna.txt, pairs_within_gpt-6-luna.csv | VERIFIED (see note 1) |
| V104 | tables/pairs_note.tex: "HH mean hit 0.089; n pairs 69741; n HH 31583; n LL 9924" | pairs_within ICLR 2024-2026 | VERIFIED |
| V105 | tables/llm_panels.tex: all cells | llm_panels_gpt-6-luna.csv, equal-weight mean of 3 years | VERIFIED against CSV (conflicts with text, see M12--M15) |

## Notes (not counted)

1. tables/within.tex: the bottom panel reuses the top column headers. Under "GPTZero (2024)" it prints specific-claims values. Under "Flag, S+S fields" it prints ICLR 2026 values. The caption explains this, but the headers mislead. The top "Flag, S+S fields" column is all "--", although the caption describes that exposure.
2. Spillover: the text says "Unflagged reviews are not changed by flagged co-reviews". The unique-claims effect is +0.079 (SE 0.023, p = 0.0006), which is significant. The numbers are correct, but the wording conflicts with them.
3. Second extractor: the abstract says the rise "survives ... a second extractor". The sol S3 effect has p = 0.32 and the share-shared effect has p = 0.10. Only the specific-claims outcomes are significant (p < 0.01).
4. E6 says "generic novelty complaints" fall. But "Limited novelty and incremental contributions" is the third-largest rise (0.48\% to 1.28\%, type_shift.tex).
5. "One seventh" compares a 2026 three-review panel (share 0.235) with four-review E5 panels. Share shared depends on panel size.
