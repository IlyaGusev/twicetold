# Twice-Told Critiques: paper notes

## Main claims

1. Same-paper critique overlap rises in the latest ML review cycles.
2. AI-flagged pairs overlap more within one paper, including in blind reference annotations and style-swapped texts.
3. Cross-paper comparisons distinguish repeated critiques within panels from convergence across unrelated papers.

Category changes and the five-source journal comparison provide separate evidence about content and scope.
Journal comparisons are exploratory. Detector labels describe text, not authorship.

## Measure

For review problem sets A and B, overlap is the size of their intersection divided by the smaller set size.
Each set contains distinct problems, not raw claims or broad topic categories. Claims assigned to the same problem
group within one review count once. Full overlap means containment of the smaller set, not identical reviews.
Empty problem sets enter no pair. Average valid pair scores within each paper, then average papers within each
year. Every paper has equal weight.

`src/metrics.py` holds the common overlap function. Panel outcomes, pair outcomes and blind panel annotations use
that function. In the direct pair audit, connected claim matches define one problem before scoring. Character
length does not enter the overlap formula. The within-paper association and its length-control check both use
the same outcome, with paper-clustered intervals.

## Reproducibility

`scripts/post.sh gpt-6-luna-high` recomputes all analysis tables from the fixed model outputs and reference labels.
No stage substitutes values from an earlier analysis. Raw claims, groupings, embeddings and annotation keys are
fixed model outputs. The analysis does not make model calls.

Current derived data live in `data/overlap/`; reports live in `reports/overlap/`. The settings fields
`ANALYSIS_DIR` and `ANALYSIS_REPORTS_DIR` define those paths. The numbers command generates
`paper/tables/numbers.tex` from the reports. Main estimates and intervals in the manuscript use those commands.

The unit for a yearly mean is a paper, not a review pair. The within-paper detector regression clusters by paper.
Cross-paper control intervals describe variation over sampled comparison panels conditional on the corpus.
The comparison baseline for ICLR is 2023; the latest-cycle comparisons use NeurIPS 2024--2025 and ICML 2025--2026.
All analyses are exploratory rather than preregistered.

## Validation

Claude Opus annotations come from another model family, not human annotators. The independent checks cover
complete panels, style-swapped panels and direct review pairs. The two drift samples assess year-dependent
measurement error. Their estimates and sample sizes come from the current overlap reports.

The 600-pair merge audit uses an earlier low/medium-effort extraction configuration. It is not a validation of the
high-effort run's scores. Generic-versus-specific labels and AMPERE proposition labels assess separate extraction
tasks. Their targets differ from distinct-problem overlap.

## Data

ICLR starts in 2020 because withdrawals and missing withdrawn papers distort the earliest public years. The
analysis excludes withdrawn and desk-rejected papers and review identifiers absent from the current corpus.
The withdrawal robustness check adds the 2026 withdrawn sample. Preserve `data/backup_predup/` to redraw the
original panel sample exactly.

Journal sources retain first-round referee reports and exclude author replies and editor notes. Journal data for
2026 cover January--October; BMC Medicine ends in 2025. Review forms, LLM policies and access to earlier reports
vary across sources. Final public reviews do not recover independent first drafts.

## Figure messages

| Figure | Message |
|---|---|
| 1 | Same-paper critique overlap rose in the latest ML review cycles. |
| 2 | AI-flagged review pairs overlap more within one paper. |
| 3 | Clarity and typo critiques fell while experiment requests rose. |
| 4 | Journal sources with larger AI-likeness increases have larger overlap increases. |

Figure 1 belongs in the opening, before method definitions. Its axis and caption explain the ratio directly.
The worked example follows in the method. Detailed detector, control, blind and placebo estimates belong in the
appendix. Use the same overlap definition throughout and do not present containment as full review identity.

## Figure style

- ICLR: blue. NeurIPS: vermillion. ICML: bluish green.
- Both reviews flagged: orange. One flagged: light orange. Neither flagged: grey.
- Blind Opus estimates: squares. Placebo estimates: open circles.
- Single-series plots without venue identity: black.
- Review cycles from 2023 on: the same light-grey band in every applicable time plot.
- One-column figures: 3.03 inches. Full-width figures: 6.3 inches.
- Times-like serif text at 7--8 pt. Render and inspect every PDF page.
