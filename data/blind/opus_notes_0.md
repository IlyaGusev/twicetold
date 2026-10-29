# Hard cases, batch 0

- B24_R1: the update says "my initial concerns have been addressed" as a blanket remark. I kept all claims, because no single point was marked as addressed. A strict reading of rule 4 gives an empty list.
- B20_R3: the reviewer named two points as addressed after the rebuttal (RLSearch peer review, sample sizes). I dropped them and kept the RLSearch-detail point that the reviewer repeated.
- B24_R3: the update keeps the iid and Theorem 1 points. I kept the original theory and practice questions too, because the reviewer did not mark them as addressed.
- B32: four reviewers say the certificate re-uses Cohen et al. randomized smoothing, with different wording. I merged them into one cluster. Theorems 2 and 3 got their own cluster.
- B16_R2: this review has many small notation and wording notes. I grouped them into three claims (page 3 wording, variable notation, objective naming), not one claim per symbol.
- B04: R0 "results only on MBench, risk of overfit" and R3 "no HumanML3D/BEAT2 results" have the same fix, so I merged them. Training-data tuning on MBench (R0 vocabulary overlap, R3 MoCap subset choice) got a separate target.
- B04: R1 "perturbed ground truth does not simulate video references" and R3 "noise-perturbed ground truth makes a train/inference gap" are in one cluster.
- B36: R2 "Observation 1 is assumed, not derived" and R3 "Observations are assumptions, not proofs" are in one cluster. "No margin bound" and "Observation 2 has no intensity definition" are separate clusters.
- B16: "classification is a poor test of topic posteriors" (R3) is vague. I put it with R1 "real-data results only on semi-supervised classification".
- Category choice was often unclear for dataset-quality and data-contamination points (B04). I used "other" or "unsupported-claim" for them.
- Process: another agent overwrote a shared scratchpad helper with blind_input_3. I caught this before I wrote anything and used a helper with a unique name. All outputs come from blind_input_0.
