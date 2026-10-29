# Hard cases, batch 1 (opus annotator)

- B37_R1 holds a revised review and the original review. I dropped the original novelty point because the reviewer says the revision fixed the positioning. I kept the original minor comments, since the reviewer does not mark them as fixed.
- B05_R1 post-rebuttal: I dropped the runtime/memory and Section 5 clarity points because the reviewer says they are fixed. I kept the new baseline and thinning-bound requests.
- B25_R1: one novelty point sits in STRENGTHS ("follows directly from PCP sketch properties"). I kept it as a softened critique.
- B21: four claims ask for deeper failure analysis (stage breakdown, prompting failures, OLS default). I put them in one cluster because one added analysis section answers all four. This cluster is a judgment call.
- B21: B21_R0's "wrong method but low MRE" overlaps B21_R3's Table 4 point. I clustered it with the metric-design problem, because MSA design is its main point.
- B17: "MAT worse than MSD" and "hyper-parameter excuse is unfair or unconvincing" share a target but sit in different clusters.
- B33: I merged three utility complaints (downstream tasks, perplexity is too narrow, MMLU/NQ) into one cluster. The data-removal baseline is a separate cluster from the unlearning baselines.
- B01 has no shared clusters. The reviewers comment on different parts (theory setup, baselines, estimator bias).
- B13: Figure 1 terms and "Task I+II" from three reviewers form one cluster, though B13_R3 frames the point as page 2 exposition.
- Categories for "practical use case / significance" (B25) and "metric design" (B21) have no exact match. I used "other".
