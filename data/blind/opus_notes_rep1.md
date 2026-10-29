# Hard cases, replicate 1 (batch 1)

- B37_R1 holds an updated review plus the original. I dropped the original novelty point (the update says positioning improved) but kept the original minor comments, since the update does not mark them addressed.
- B05_R1: dropped runtime/memory and Section 5 clarity points; the post-rebuttal update says both are addressed.
- Generic "paper is hard to read" lines: kept as one non-specific claim when no concrete example follows (B05, B37); merged into the specific claim when the reviewer gives examples right after (B13_R1).
- Pure clarification questions skipped (B01_R2 max over s,a; B25_R1 "how good is SAT vs A_k"; B21_R0 data-generation framework idea).
- B21: four reviews criticize metrics and failure analysis from close angles. I split into "MSA/MRE too coarse" vs "pipeline steps not evaluated", and "stage-level failure categories" vs "why prompting/OLS fail". The boundary is soft.
- B33: merged unlearning baselines (R1), generic baselines (R2) and data-removal baseline (R3) into one cluster, since R1 lists a deletion oracle explicitly.
- B17: MSD comparison claims from all four reviewers form one cluster, even though they cite different tables (Tab. 5, 6, 9).
- B25_R0 Lemma 3.2: split into two claims, one per proof step, since each needs a separate fix.
- Category choice for evaluation-metric design and use-case critiques often fell to "other"; no category fits them well.
