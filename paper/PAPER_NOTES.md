# Twice-Told Critiques: paper-specific notes

How WRITING_GUIDE.md and FIGURE_GUIDE.md apply to this paper.

## Core claims

1. Within a paper, two AI-flagged reviews share more critique claims than two unflagged reviews of the same paper.
2. Same-paper overlap was flat 2020–2025 and rose in 2026; cross-paper overlap did not rise.
3. The mix of critiques shifted from clarity and typos toward requests for more experiments.

Confidence: claim 1 is systematic (all instruments, detectors, blind references). Claim 2 is hedged: the ICLR rise is instrument-sensitive and drift cannot be excluded. Journal results (RQ5) are exploratory.

## Holes to red-team

- Detector false positives.
- Review length and panel size.
- Venue mix and decision mix (rejected and withdrawn papers match more).
- Year-by-venue confounds; instrument drift over time.

## Terms (define once, never drift)

- "AI-flagged" means the style detector fired, not "AI-written".
- "Critique claim": one problem the authors would answer with one fix; give one example.
- "Match rate" (raw) and "standardized match rate": the only two names for the overlap measure. Do not use "homogenization" as a measured term; use "overlap" with its formula.
- Journals have "reports", conferences have "reviews"; say so once.

## Calibrated language

- Finding: "overlap rose". Speculation: "reviewers paste LLM output". Keep the mechanism in one paragraph labeled as speculation; the journal correlation (Fig. 4) is correlational.

## Baselines next to each claim

- Unflagged same-paper pairs (claim 1).
- Cross-paper pairs (claims 1–2).
- Length- and panel-size-standardized rate (claim 2).
- Confidence intervals at the paper level, not the review level.
- Mark the per-venue trend comparison as post hoc.

## New vs not new

- LLM-based critique extraction exists. The same-paper vs cross-paper contrast, the within-paper AI-flag comparison and the 2020–2026 trend are ours.

## Related work placement

- Three sentences in the introduction (LLM-review detection, review-quality studies, critique extraction); full related work, if space allows, before Limitations.

## Limitations (main text)

- Detector error, English only, venue coverage, LLM extraction noise, correlation not causation, LLM (not human) validation.

## Data window

- ICLR starts in 2020: before 2020 the public data lack almost all withdrawn papers (later 13–26% of submissions), and reviews of withdrawn (weak) papers match more. Evidence: `src/outcomes.py withdrawals`, Appendix A.

## Figure messages

| Fig | Message |
|---|---|
| 1 | Within a paper, two AI-flagged reviews share more critiques than two unflagged ones; every check gives the same sign. |
| 2 | Under the main extraction model, the same-paper match rate rose at ICLR and ICML in 2026 and at NeurIPS in 2025; the rate across papers did not rise. |
| 3 | Clarity and typo critiques fell while requests for more experiments rose. |
| 4 | Exploratory: the two journal sources whose AI-likeness index rose most had the largest rise in match rate. |

Fig 1a shows ICLR AI-likeness as context for the gap. Appendix figure: whole-review embedding similarity.

## Color map

- ICLR blue, NeurIPS vermillion, ICML bluish green: venues only.
- Orange: both reviews AI-flagged; light orange: one flagged. Grey: unflagged, context and baselines.
- Squares: blind Claude Opus estimates; open circles: placebos.
- Black: single-series plots with no venue identity.
- ChatGPT period: one light grey band, the same in every time plot.

## Format

- ACL short paper: 4 pages of main text plus references and appendix.
- Figure widths: 3.03 in (one column), 6.3 in (full width).
- Font: Times-like serif to match the ACL template.
