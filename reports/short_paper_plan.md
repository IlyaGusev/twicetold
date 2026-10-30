# Short paper plan (4 pages + appendix)

One measure everywhere: **match rate** = share of one review's critiques that a second review also raises
(pass-2 cluster = same target + same criticism). It needs no panel-size correction and has known baselines
(Liang et al. 2023: 35% with a lenient GPT-4 matcher; Kim et al. 2026: 3.4% human-human, strict).

| RQ | Question | Measure | Figure |
|---|---|---|---|
| RQ1 | Do reviewers of the **same paper** raise the same critiques more often since ChatGPT? | Match rate of review pairs of one paper, by venue-year. Score agreement (rating SD) on the same axis. | Fig 1 |
| RQ2 | Do reviews of **different papers** become more alike? | Match rate of review pairs from different papers, same area and year (placebo pairs); share of template claims; embedding similarity for contrast (style vs content). | Fig 1, second panel |
| RQ3 | **Which critiques** gain and which disappear? | Category shares, 2017–2022 vs 2026; effective number of critique types. | Fig 2 |
| RQ4 | How much review text is **AI-written**, and do AI-like reviews explain the overlap? | Corpus AI share (Liang MLE); per-review detector (EditLens if access, else style detector); match rate by pair type (0/1/2 AI-like) within paper. | Fig 3 |
| RQ5 | Does it hold in **other fields** and review systems? | RQ1 + RQ4 on SciPost, F1000, BMC Medicine, PLOS, Copernicus. | Fig 4 |

Main text: intro (RQs + answers), data (1 table), method (1 paragraph + 1 schematic), 4 result paragraphs,
limitations. Everything else (validation, prompts, robustness, designs E4/E5, sol replication) goes to appendices.
