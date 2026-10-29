# Area-chair review: "Twice Told" (draft of 2026-10-28)

**Overall.** The question fits the theme, the data are strong, and within-paper control is the right idea. Today the paper does not hold, for three reasons. The measuring instrument can produce the trend by itself. The validation uses only LLMs. Some designs include post-treatment controls. All three can be fixed with the data you already have.

---

## 1. Three most serious threats, ranked

**T1. The instrument can make overlap rise even if critiques do not converge.** S_k and the shared share go up when reviews have more claims, more generic claims, or bigger panels to cluster. All three changed over time.

- From 2024 there is a separate Questions field. Pass-1 rule 2 turns defect-implying questions into claims, so claims per review go up at the form break.
- LLM-styled reviews tend to say generic things like "limited novelty" or "missing ablations". Pass 2 merges these easily, even with rule 4 in place. Pass 2 also shows the pass-1 category and target labels "as hints" (prompts.tex line 37), so the labels can steer merges.
- Reviews per paper rose from 3.0 to 4.1, so pass 2 clusters bigger panels. Rarefying a 5-review clustering to k=3 is not the same as clustering 3 reviews.
- The placebo is too weak. Panels of "four reviews of four different papers" from mixed areas are easy to tell apart. They also use 4 reviews where the outcome uses k=3. And a level floor does not control a trend.
- The template measure has the same flaw. Maximum cosine similarity to other papers in the venue-year grows with pool size. That pool is 490 papers in 2017 and 3,792 in 2023.

*Fix:*
1. Make the primary estimate a difference: observed minus a placebo built from same-area, same-year shuffles at k=3, tracked year by year.
2. Repeat the main results on specific claims only, and with the Questions field dropped in all years.
3. Add a claim-count-matched version: sample m claims per review before you cluster.
4. Re-cluster a random 3-review subset per paper. This is cheap on a subsample.
5. Compute template similarity against a fixed-size random pool, for example 400 papers.

**T2. The validation cannot support a trend claim.** The reference is Claude Opus. "Produced by human-style reading" does not change that. The year test uses 40 papers across four years, about 10 per year, so it is underpowered. If you find no year dependence, it does not show that the bias is stable. The bigger risk is not year. It is whether the pipeline merges LLM-styled claims more readily. Panickssery et al. (NeurIPS 2024) show that LLM evaluators favour their own generations.

*Fix:*
1. Finish human adjudication of the 600 pairs before you submit. Stratify the pairs by year and by detector label.
2. Report the pipeline's bias against those human labels by detector label as well as by year.
3. Use an equivalence test, with the bound set to about half the headline effect.
4. Use AMPERE (Hua et al., NAACL 2019) only to check pass-1 segmentation recall. It covers ICLR 2018, so it cannot test year drift. Say so.
5. Report agreement among the four Opus annotations as the ceiling for reference noise.

**T3. Identification and framing.**

- E1 controls for "mean and spread of ratings" and "log total review length". LLM use changes both, and your own related-work paragraph cites rating compression. Rating spread also tracks critique agreement closely, so this is a bad control.
- E3 has interference. One review's label changes its co-reviewers' outcomes.
- E3 also has a direct link between detector and outcome. A style detector rewards templated text, and templated text overlaps more by construction.
- E4 is a continuous-treatment DiD. It needs a stronger parallel-trends assumption: Callaway, Goodman-Bacon and Sant'Anna (NBER WP 32117, 2024).
- Pool changes go untreated: field growth, the 2026 reciprocal-review rule, and the growing share of LLM-topic papers.
- Framing: more overlap can mean better reliability, not homogenization. Panels may now agree more on real flaws. The paper must show what is lost, not only that overlap rose.

*Fix:*
1. Make E1 without post-treatment controls the primary model. Move the controlled model to robustness.
2. Run E3 at the pair level: human-human, human-LLM and LLM-LLM pairs with paper fixed effects.
3. In E3, measure exposure on fields that do not feed the claims (summary or strengths), or use the GPTZero labels.
4. Reweight every year to the 2017-2022 area mix.
5. Use the 2023-2025 timestamps to keep only reviews last changed before the rebuttal opened.
6. Use NeurIPS 2025 as a natural test. Its form went back to "S&W joint", so if the effect follows the form, it should drop there.
7. Make E6 (the loss of specific and tail critiques) part of the main argument, not an add-on. This is the answer to the reliability reading.

---

## 2. Unclear, overclaiming or contradictory text

1. "it publishes reviews of all submissions, so there is no selection on outcome." Appendix A says "The ICLR 2024 and 2025 dumps omit withdrawn submissions", and you also drop withdrawn papers in 2026. Withdrawal depends on the reviews and has grown over time. State this as selection and bound it.
2. "258{,}414 reviews of 69{,}063 papers (Table 1)". Table 1 sums to 249,903 reviews and 64,436 papers.
3. "(Section~\ref{sec:validation})" prints as "Section ??". No such label exists.
4. "validate it against a blind reference annotation". The team brief says there are four Opus references plus AMPERE. The text names one reference and does not mention AMPERE.
5. The Data and Methods sections never describe the exposure measures. Name the style detector, how you validated it and the threshold. Name the source of the GPTZero labels (aireviewlottery2024data?).
6. "Their shared-critique rate is the floor that style alone produces". Shuffled panels measure generic content plus the model's merge tendency, not style alone. "any real within-paper signal must exceed it" should be about the change over time, not the level.
7. "under six designs that make different assumptions". E5 and E6 do not identify an effect. E6 is descriptive.
8. "(4) ... place the human trend between its floor and its ceiling." The trend is not human. It mixes human and LLM reviews. The E5 ceiling depends on which model writes and with which prompt, and the paper does not say. If GPT writes and GPT extracts, self-preference applies.
9. "rose from about 10% of sentences in 2024 ... to 21% of reviews labelled fully AI-generated in 2026". These are different units and different detectors, so "rose" is not supported.
10. "GPT-4-matched points give a 35% hit rate at ICLR". This is the human-human overlap that GPT-4 matched (35.25%). Say so.
11. "Pass 2 receives the claims of all reviews of a paper" conflicts with "at most five randomly chosen reviews". Also, "at least three reviews" is stated for 2017-2023 only.
12. "following the two-step protocol of our pilot annotation". This protocol is not defined anywhere.
13. "the extractor cannot harmonize wording across co-reviewers". It can still turn every review into its own standard vocabulary. Say that, then point to the placebo as the control.
14. "with gpt-6.1-sol on a stratified subsample as a second annotator". Call it a second model, not an annotator. Give the subsample size and strata.
15. Limitations: "We show that ... the pipeline's bias relative to the reference does not depend on the year". This states a null as proof. Report the equivalence bound instead.
16. "we cannot recover first drafts without API access to revision histories". Data says only ICLR 2026 has timestamps ("the ICLR 2026 dump carries creation and modification times"), but the brief says 2023-2025 have them too. Make the text match, and use them.
17. "this low overlap is a feature, not a bug". This is a stock phrase and a claim you have not argued. Disagreement is also noise (Cortes and Lawrence 2021).
18. "their summary presents it as holding for real reviews". Quote Baumann et al.'s abstract word for word, or remove this. Area chairs react badly to unsupported claims about other papers.
19. "Classic studies found" cites one study, and it covers psychology journals. Say that.
20. E1 states the 2023 baseline is pre-ChatGPT because "reviews were due in November 2022". Give the exact review and discussion end dates. Final versions edited after 30 November 2022 would leak exposure into the baseline.
21. Area labels have 59% accuracy and come from a classifier trained on 2026. Its errors likely grow for older years because topics shift. This weakens area fixed effects and E4. Report E4 on true 2024-2026 areas in the main text, not only as robustness.
22. Hill-Shannon diversity depends on sample size. Rarefy it or use coverage-based estimates (chao2012coverage is already in the bib).

---

## 3. Related work

- **Mis-cited: stelmakh2023herding.** The paper studies herding in reviewer discussions, after the first drafts. It is not about "independent first drafts".
- **Add these:**
  - Gao et al., "Does My Rebuttal Matter? Insights from a Major NLP Conference", NAACL 2019, on rebuttal effects.
  - Bornmann, Mutz and Daniel, PLoS ONE 2010, on low reviewer reliability, for the framing.
  - Panickssery et al., "LLM Evaluators Recognize and Favor Their Own Generations", NeurIPS 2024, on LLM self-preference.
  - Callaway, Goodman-Bacon and Sant'Anna, "Difference-in-Differences with a Continuous Treatment", NBER WP 32117, 2024.
  - Hurlbert, Ecology 1971, the source of rarefaction. The current cf. points to kim2026argument.
  - Kirk et al., ICLR 2024, on RLHF reducing output diversity.
  - Agarwal, Naaman and Vashistha, CHI 2025, on AI suggestions homogenizing writing.
- **Cite AMPERE (hua2019argument) in the text.** It is in the bib but the text never cites it.

---

## 4. What Results must contain, in order

1. **Validation table.** Pipeline vs human pairs and vs the Opus references, broken out by year and by detector label. Include the equivalence bound and the AMPERE segmentation recall. Put it first: a sceptic will not read further without it.
2. **Figure 1.** Shared share and S_3 by year, observed vs same-area placebo, ICLR and NeurIPS. Show the 2017-2022 pre-trend.
3. **E1/E2 table.** The model without controls first. Then specific-only, Questions field dropped, claim-count matched, and pre-rebuttal timestamps only.
4. **E3 pair-level table.** Human-human vs human-LLM vs LLM-LLM match rates within the same paper, on specific claims.
5. **E4 event-study plot** by dose tercile, with pre-period coefficients.
6. **E5 ceiling,** naming the model and prompt.
7. **E6 loss decomposition.** Specific and tail share and category shifts. This answers "is it just more agreement on real flaws?"
8. **Appendix robustness:** second model, rebuttal-mention exclusion, true areas, strict vs lenient match rule.

The numbers that will persuade a sceptic most:

- The placebo-adjusted change from 2023 to 2026, with a CI.
- A flat pre-trend slope.
- The LLM-LLM minus human-human gap on specific claims within the same paper.
- D_3 in plain counts, for example "a three-reviewer panel now raises X fewer distinct critiques".
- The pipeline's year bias against human labels, which must be well under the effect.

Pick one primary outcome and one primary design ahead of time. There are 7 outcomes x 6 designs x 2 venues, which invites a forking-paths critique.

---

## 5. ACL and ARR compliance

1. **Bibliography padding.** `\nocite{*}` prints all 61 bib entries, many of them uncited. Remove it.
2. **Page budget.** The body ends on page 4. That leaves about 4.5 pages for Results, Discussion and figures. Validation and E3 must be in the main body. Reviewers do not have to read appendices.
3. **Placeholders.** The abstract, cost figures and Section ?? are unfinished.
4. **Section order.** Limitations and Ethics are present and in the right place. Neither counts toward the 8 pages.
5. **Anonymity.** "we release" needs an anonymized link. The PDF metadata is clean, and the third-party Hugging Face names are fine.
6. **ARR Responsible NLP checklist.** Disclose dataset licences, LLMs used as annotators, and use of AI writing help. If humans adjudicate the 600 pairs, give their pay and recruitment.
7. **Reproducibility.** Give model version or date, reasoning effort and sampling settings for both closed models.
8. **Theme wording.** Check "The ACL 2027 theme asks whether..." against the exact CFP text.

**Next action:** fix the `\nocite{*}` and the "Section ??" today. Both take under two minutes in main.tex.
