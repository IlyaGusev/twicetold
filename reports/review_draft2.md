# Area-chair review, draft 2 (full paper, 2026-10-29)

**Verdict.** The data and the within-paper design are strong. Three headline claims go past the numbers, and the body runs past 8 pages. The data in hand can fix both.

## 1. Overclaims and logical gaps (ranked)

**1.1 Pair composition does not explain the 2026 jump.** The text says "$0.21\times0.048+0.46\times0.013=0.016$ matches ... $+0.017$."
- The match works only because the 2026 human-human rate (0.100, Fig. 2) is close to the overall 2023 rate (0.098). In Fig. 2 all three pair types rise from 2025 to 2026 (0.083→0.100, 0.085→0.112, 0.111→0.143). Most of the jump is *within* pair types.
- E2 calls 2024–2026 the clean same-form window. There the hit rate rises by +0.030. Composition predicts +0.008 with the pooled coefficients and +0.011 with the 2026 ones, a quarter to a third.
- For 2024 and 2025 composition predicts +0.002 and +0.006. The observed values are −0.013 and −0.009.
- Composition also predicts that claims per review fall by about 0.65 (0.45 × 1.44). The observed change is −0.05.
- Remove "predicts the size of the jump" from the abstract, Findings and Discussion. Report composition and within-type parts. Rising unflagged overlap contradicts "not in the papers or the year".

**1.2 The "square of the share" model in "Why 2026" fails too.** It predicts a detectable rise in 2025 (SE 0.002), but 2025 is below 2023. "Slightly below 2023" is also wrong for $S_3$ in 2024: −0.169 (SE 0.026). The drop matches the new Questions field. Explain it.

**1.3 The NeurIPS form argument points the wrong way.** ICLR went from S&W joint to S, W, Q in 2023–2024, and $S_3$ fell by 0.17. So the reverse change at NeurIPS 2025 predicts a *rise*, as observed. Remove "opposite direction to the one a form-driven explanation would need", or list the form as a confound. Also, NeurIPS 2025 reviews came a few months before ICLR 2026 reviews, not "one year earlier".

**1.4 The abstract's "survives re-clustering, removing the questions field ... and a second extractor" is not supported.**
- Second extractor: $S_3$ +0.17 (SE 0.17) and share shared +0.029 (0.018) are not significant. Only the specific-claims rows survive.
- Questions removed: 2026 $S_3$ is 1.23, *below* 2023 (1.41–1.43). The text quotes only the change from 2024 to 2026.
- Re-clustering: Table 8 gives 2023 = 1.54 and 2026 = 1.73, so +0.19 (SE ≈ 0.10) against +0.30 rarefied. "Panel size plays no role" cites 2026 only. In 2022–2023 the re-clustered value is 0.13 higher.

**1.5 The 40-paper bias test does not support "does not move with the year", or the Limitations line "we bound ... the bias".**
- The share-shared drift CI (±0.057) covers the whole effect (+0.045).
- The $S_3$ drift is +0.20 (0.28): the point estimate is 73% of the effect.
- The by-year $S_3$ bias rises steadily: −0.70, −0.20, −0.10, −0.25. Pooling 2018 with 2022 hides this.
- Write "cannot exclude a drift as large as the effect", and add the equivalence bound.
- If each annotator covered one year, annotator and year are confounded.

**1.6 The pair validation misreports recall and FPR.**
- The output in `pairs_validation.txt` says recall uses "sampling weights ignored". The sample is stratified by the pipeline's decision, so unweighted recall is not the population recall. The caption claim that precision and recall "are the comparable quantities" is false for recall.
- The FPR caption describes a quantity that is zero by definition.
- Each 2026 cell has 50 merged pairs, so the precision SE is about 0.05. The 0.81 vs 0.87 difference is noise.
- The point correction is 1.13 × (0.87/0.80)/(0.81/0.72) = 1.09, which cuts a third of the excess. Say so, and define 1.13: it is neither 0.235/0.190 nor 0.115/0.098.
- "Two independent reference annotators" are two runs of one model.

**1.7 The pre-period flagged reviews weaken the LLM attribution.** The share-shared coefficient is +0.042 (SE 0.019, t = 2.2), below your own threshold of 2.58. The reading "a style of review that overlaps more whoever writes it" admits that a pre-LLM style predicts overlap. That style is short and vague (−0.77 claims, −0.129 specific share). This fits low-effort reviewing or instrument self-preference as well as LLM use. Discuss these readings.

**1.8 E4 is evidence against the effect, not only a low-power null.** −0.58 (SE 0.35) gives an upper bound of about +0.11. The 2026 year effect read linearly (+0.275 at 45%) implies about +0.6 per unit of dose, which this CI excludes. Report the minimum detectable effect. Your own quadratic argument implies that the dose should be the share of flagged-flagged pairs. The appendix promises results with the true 2024–2026 areas, but none are given.

**1.9 The reliability rebuttal does not tell the two readings apart.**
- A constant $D_3$ with more sharing is also what better reliability predicts. $D_3$ also depends on the specification: −0.34* with controls, +0.44* specific only, +0.51* with the 8-claim cap.
- "Specific claims rose", yet flagged reviews are *less* specific (−0.021; E5 0.72 vs 0.86). So the rise cannot clear LLM reviews of vagueness. 2023 is also a dip: 0.842 against 0.887 in 2021.
- Table 7 contradicts the text. "Limited novelty and incremental contributions" rises from 0.48 to 1.28. "Experiments are insufficiently thorough" falls. "Missing ablations", "hyperparameter sensitivity", "typos", "derivations" and "reads the proof" have no support in the table.
- Many types that move are near-synonyms (two reproducibility types, four novelty types). Pre-period k-means loses entropy under *any* change in wording or topic.
- The repertoire was already falling in 2023, on pre-ChatGPT reviews (272→267). It also fell through 2024–2025 while $S_3$ dipped, so it is not tied to the jump.

**1.10 Missing confound: LLM-polished papers.** Fewer writing and typo critiques, and part of the year trend, can come from author-side LLM use. E3 controls for it; E1, E2 and E6 do not. Add it to Limitations.

**1.11 "Human" means "unflagged".** Fig. 2, Table 3 and the abstract ("two human reviews") call unflagged reviews human. Your Related Work says detectors miss polishing. Use "unflagged".

## 2. Internal inconsistencies

1. **Withdrawn papers.** Data says the 2024–2026 dumps omit them. Limitations says the 2026 dump includes them. "Toward zero" also needs the gap between withdrawn and kept papers in a pre-period year.
2. **E5 numbers.** The text gives D 15.4 vs 24.4, hit rate 0.275 vs 0.091, specific share 0.69 vs 0.85. Table 9 gives 16.0 vs 24.3, 0.280 vs 0.089, 0.717 vs 0.859. "37% fewer" is 34% by the table.
3. **LLM count.** Detector training uses "five LLMs", but E5 uses four.
4. **Cost.** Methods says $120 and Ethics says 170 USD.
5. **Table 3 headers.** The headers "GPTZero (2024)" and "Flag, S+S fields" sit over columns that the caption calls specific-only and 2026-only. The S+S measure has no results ("--"). Report it or cut it.
6. **"No edited review".** In 2024–2025 it means a rating change or a mention of the response. In other years it means a mention only. Hence 2025 moves from −0.078 to +0.096.
7. **Pre-trend.** "Fell by 0.16 per year" is wrong. An OLS fit of the plotted $S_3$ for 2017–2023 gives −0.094 per year, and 2020–2023 is flat. So "decade-long decline" is wrong too. Table 2 has no 2017–2022 rows, although E1 promises a pre-trend test.
8. **Baseline 2023.** 2.8% of 2023 reviews are flagged (1% by construction). 2023 is also an outlier on the placebo (0.31), tail share (0.372) and specific share (0.842). This inflates the excess (+0.48) and the E6 changes. Add a pooled 2021–2023 baseline.
9. **Placebo.** A floor of about 0.1 against 1.7 has little power to detect drift in how readily the pipeline merges. Change "The instrument cannot produce the rise" to "... on unrelated reviews".
10. **Spillover.** "Not changed" conflicts with a significant +0.08 (SE 0.02) unique claims.

## 3. One more analysis each (all feasible with current data)

1. **Decompose** the 2024→2026 and 2023→2026 changes into composition and within-type parts, for hit rate, share shared, claims and $D_3$.
2. **Control claim count in E3.** Sample m = 4 claims per review, or add claim-count fixed effects. The cap at 8 barely binds at a mean of 6.8.
3. **Test reliability directly.** In papers with ≥2 flagged reviews and ≥1 unflagged review, measure how many flagged-flagged shared critiques the unflagged reviewer also raises. Compare with unflagged-unflagged shared critiques raised by a third reviewer.
4. **Explain the 2026 rise among unflagged reviews.** Split it by paper-level LLM scores, confidence and area.
5. **Rerun E4** with the share of flagged-flagged pairs as the dose, the true areas, and a minimum detectable effect.
6. **Test type robustness.** Refit k-means on pooled years and on 2026, and compute entropy within area. Compare flagged and unflagged 2026 reviews at the same sample size.
7. **Run the costly variants on all 2023 and 2026 papers**, not 100–400 per year.

## 4. Writing

- The abstract has 241 words. Check it against the venue limit (ACL usually asks for 200 or fewer). Cut the list of checks. Say "LLM reference annotations".
- Define a shared critique in the abstract as one "raised by at least two of three reviewers".
- "Half again as often" is an idiom. Write "43% more often".
- Undefined on first use: HC3, "S+S fields", "dose", "2026-dose tercile".
- Fig. 1: say how a four-review placebo yields $S_3$. Fig. 2: give n per point. Fig. 3: say what the shading means.
- Table 6's caption does not stand alone. Say how you made Table 7's labels.

## 5. Compliance

1. **The page limit is broken.** The Discussion "Implications" paragraph runs 9 lines onto page 9 (lines 638–646). The Mechanism paragraph repeats the Findings, so cut it.
2. **Key claims sit only in the appendix.** These include the placebo excess (Table 5), re-clustering and no-Questions (Table 8), the second extractor, the by-year validation bias, the E4 event study and the type list (Table 7). Put the excess row and the validation drift with its CI into the main tables.
3. **Anonymity.** "Released" needs an anonymized link. The PDF metadata is clean.
4. **Limitations** must drop the bound claim (1.5) and add the confound from 1.10.
5. **Add to the Ethics statement:**
   - the licences of the three Hugging Face dumps, and a check that every OpenReview year is CC BY 4.0;
   - a statement that the JerMa88 records hold no reviewer identity fields and were not collected during the incident;
   - exact model IDs, access dates, settings and single-run nondeterminism for all three LLMs;
   - "no human subjects", or recruitment, pay and consent details if human adjudication goes ahead;
   - a note that released review IDs plus the detector let anyone rebuild per-review labels.

**Next action:** cut the Mechanism paragraph so the body ends on page 8.
