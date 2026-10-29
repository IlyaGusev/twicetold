# Hard cases, batch 2 (opus annotator)

- B26_R0 ends with "Thanks for addressing the issues I raised". This is a general thank-you, not a mark on each point, so I kept all 3 claims. Under a strict reading of rule 4, the list would be empty.
- B10_R3 and B22_R1 add text after the rebuttal. I kept the new points (for example, the loose eigenvalue statement in B22_R1) and also kept the points from before the rebuttal, because none of them is struck through.
- B06: R1 says "no baselines in Highway" and "none in grid-world" as two points. R2 says "baselines only on Cart-Pole". I put all three in one cluster because one fix (run baselines on all environments) answers them.
- B22: R1 and R2 name different missing papers on spectral and heterophily GCL. I merged them into one related-work cluster because one positioning section answers both. The two lists of papers do not overlap.
- B10: three convergence-rate claims (an f(eps) bound, a rate for eps against n, and sample complexity) are in one cluster. The proofs are different, but each reply is "add a rate".
- B10: "nearly optimal" comes up for transport (R0, R3) and for the generator (R2). I split them. R2 attacks the cost function, not weak convergence, so its claim goes with R1-8.
- B14_R3 and B30_R3 put soft critiques inside praise ("not a surprising result", "theory not particularly novel"). I kept them as novelty claims because rule 4 keeps softened points.
- B14: R0 cites FID only as a missing reference, and R1 asks to add FID as a metric. I merged them.
- B18: "hypothesis testing not validated" (R1, R2), "desideratum 4 ill-defined" (R3), and "no fidelity metric" (R3) have the same target but go in different clusters.
- B30_R1 asks about expanding subspaces like BAxUS. I dropped this because it is a suggestion and does not point to a defect.
- B02: "time complexity of Eq. 11" (R0) went into the server-cost cluster with R1 and R2, not into the "Eq. 11 solver unspecified" cluster.
- Category choice was often unclear between unsupported-claim and clarity, and between assumptions and theory/correctness. I chose by the fix the authors would make.
