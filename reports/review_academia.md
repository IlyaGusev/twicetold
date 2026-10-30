# academia MCP `review_pdf_paper` on paper/main.pdf (2026-10-29)

Scores: soundness 3, clarity 3, significance 4, originality 4, overall 5, confidence 4.

## Questions
1. Detector threshold: report within-paper results (Table 2, pair hit-rate gaps) and year-level S3 for flag thresholds 95th–99.9th percentile and for the continuous detector score.
2. Human adjudication: dual-human labels on 100–150 cross-review claim pairs, stratified by year and flag; compare with Opus (precision, recall, bias by year).
3. Form/policy confounds: restrict ICLR 2024–2026 to reviews that do not mention the response and whose rating did not change; use earliest versions if available; control for the 2026 "authors review" policy (count of author-reviewers).
4. Repertoire robustness: effective number of types and template share vs k-means k (200/400), embedding model, template-similarity threshold (60/75/90th pct).
5. D3: Table 1 with controls shows D3 falling in 2026, the primary rarefied estimate is flat. Reconcile, say which estimate supports "distinct critiques did not fall", add a D3 uncertainty plot.

## Weaknesses
- Proprietary LLMs as instrument; 40-paper reference set and 600 LLM-adjudicated pairs may be too small to rule out year-dependent drift; the 2026 effect is near what the CI allows.
- Form and policy confounds; only final reviews; discussion leakage may inflate overlap.
- Detector exposure is noisy, mixes polishing and drafting, depends on a 99th-percentile threshold; no full threshold sweep.
- "Distinct critiques did not fall" looks fragile (Table 1, D3 with controls).
- Repertoire analysis depends on embedding model, k, thresholds; little sensitivity in the main text.
- Clarity: dense; needs a pipeline/outcome/design schematic; Fig. 1 labels small; Tables 1–3 overloaded.
- Significance: causal claims limited without randomization (ICML 2026); external validity beyond OpenReview CS venues.

## Limitations to add
- Pass-1 paraphrase length sensitivity (15 vs 25 words), in case wording harmonization raises 2026 mergeability.
- Ethics note on chilling effects of measuring LLM use in reviews.
- Pre-register E1–E3 on ICML 2026 assignments; human annotation; journal replication.

## Format
- Enlarge Fig. 1 fonts and legend; split Table 1; hyphenation ("co-reviewers"); keep model versions explicit.
