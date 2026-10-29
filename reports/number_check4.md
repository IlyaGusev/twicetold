# Number check 4 (new analyses only)

## Mismatches

1. main.tex line 332 (Placebo break at 2021):
   "$D_3$ shifts by $+1.2$ to $+1.3$ (SE 0.3) at 2021"
   Correct: the 2017-2022 window gives $-0.40$ (SE 0.29); only 2018-2022 (+1.16, SE 0.33)
   and 2019-2022 (+1.34, SE 0.36) give +1.2 to +1.3. Fix: "$-0.4$ to $+1.3$ (SE 0.3--0.4)",
   or name the two later windows. Source: reports/prebreak_gpt-6-luna.csv.

## Notes (not mismatches)

- line 127 "rises between unrelated reviews in 2026 too": true for same-area InterSim
  (0.498 -> 0.530), not for any-pair InterSim (2026 0.498, flat). Line 330 says "same
  primary area", so it agrees. Source: reports/embed_gpt-6-luna.txt.
- line 330 "rises in every year": 2022 -> 2023 is 0.7839 -> 0.7843 (+0.0004); true but tiny.
- line 328 "SE about 0.05" for the period gaps and line 330 "300 random pairs per
  area-year": no value in reports/ to check against.

## Checked and correct

- Generic vs specific (lines 97, 328): 300 = 150 + 150; annotator shares 0.78/0.77;
  agreement 99% (0.987); kappa 0.96; pipeline 0.88; 296 agreed items; accuracy 0.83,
  precision 0.85, recall 0.96; kappa 0.41 (0.413 vs A, 0.401 vs B); period accuracy
  0.85/0.82 (0.845/0.818); gaps +0.09 (0.880-0.787) and +0.12 (0.880-0.760).
  Source: reports/spec_validation.txt.
- Placebo (lines 127, 332): S3 +0.06 (0.05), -0.05 (0.05), +0.08 (0.06), 2026 +0.33 (0.03);
  hit rate +0.010, -0.010, -0.005 (0.004), 2026 +0.024 (0.002); share shared +0.07, +0.02,
  +0.04 (0.006), 2026 +0.027 (0.003); 2018-2020 share shared 0.155/0.202/0.146
  ("0.15 to 0.20", reports/trends_gpt-6-luna.txt).
- Embedding (lines 127, 330): all ten ICLR IntraSim values; SE max 0.0032; 2026 vs 2023
  +0.040 (SE 0.0009); pre-trend about +0.0095/yr; InterSim 0.49-0.51 to 2025, 0.530 in 2026;
  excess 0.272/0.308/0.294; NeurIPS 0.789 -> 0.822 (0.8215); n = 30,937; Spearman
  0.15/0.17/0.06.
- Limitations (lines 195-204): "600 adjudicated pairs" (pairs_validation.txt: 600);
  "40-paper reference"; "95% CI +-0.06 on the share shared" = 1.96 x SE 0.029 of the
  2026 vs 2018-2022 bias gap (validate_blind_gpt-6-luna.txt). "Exposure is measured" has
  no numbers to contradict.

## Ethics cost (line 217, "about 210 USD")

- logs/*.log "finished" lines: 20 lines, sum 197.30 USD.
- Embeddings: 85M tokens x 0.02 USD/M = 1.70 USD.
- Unlogged Opus/embeddings: about 10 USD.
- Total about 209 USD. "About 210 USD" is still a fair rounding.

Mismatches found: 1
