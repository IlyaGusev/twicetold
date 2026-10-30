# Writing guide

Sources:
- [SPJ] Peyton Jones. *How to write a great research paper.* Microsoft Research slides (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf).
- [LS] Lipton, Steinhardt. *Troubling Trends in Machine Learning Scholarship.* arXiv:1807.03341, 2018.
- [JS] Schulman. *An Opinionated Guide to ML Research.* 2020 (http://joschu.net/blog/opinionated-guide-ml-research.html). Mostly on research process, not writing; cited sparingly.
- [NN] Nanda. *Highly Opinionated Advice on How to Write ML Papers.* Alignment Forum, 2025.
- [CL] Pechmann, after Widom. *Tips for Writing Technical Papers (Checklist).* Hochschule Emden/Leer.

Each rule cites its source. For figures, see FIGURE_GUIDE.md; this guide does not repeat it.

## 1. Core claims first

- Write the 1–3 claims before any prose. Everything else serves them; cut a sentence if it serves none [NN, SPJ: "one ping"].
- Give each claim a confidence level: systematic ("across all settings we checked"), hedged ("suggestive"), or narrow ("in these three settings") [NN]. Do not turn a hedged claim into a systematic one in the abstract.
- Give each claim 1–3 key experiments; put the rest in the appendix [NN].
- Red-team before writing: assume there is a hole, find it, close it [NN].

## 2. Abstract

- Order: one sentence of context, one of the gap, then the claims with one concrete number each, then what the reader should do with the result [NN]. CL says use little or no background; NN allows two sentences to orient a cold reader. In a short paper, use one.
- State the standard of evidence in the last sentence: "provides evidence that", not "shows that", unless the result survives every check [NN, LS].
- Name the paper's genre at once, e.g. "a measurement study of N items from sources S" [NN].

## 3. Introduction

- Open with the problem, not the field. Use a concrete example rather than "X is important [1–6]" [SPJ: molehills, not mountains].
- CL's six questions, in this order: what the problem is; why it matters; why it is hard; what prior work missed; what we did and its limits; our contributions [CL].
- Write contributions as a bullet list of refutable claims, each with a forward reference: "A is X% higher than B (§4, Fig. 1)", not "we study A" [SPJ].
- Do not write "The rest of this paper is organized as follows" [SPJ]. CL allows a structure paragraph; this guide follows SPJ, because the forward references do that job.
- Say what is new and what is not [NN].
- Budget: about half a page in a short paper.

## 4. Calibrated language

- Separate what you measured from why you think it happened. A measured change is a finding; a mechanism you did not test is speculation. Label speculation as such and keep it in one paragraph [LS §3.1].
- Test: "Would I rely on this explanation to predict new data?" If not, it is speculation [LS §5.1].
- No suggestive terms. A proxy is not the thing it proxies: a detector's flag is not ground truth. Say so once, then never drift [LS §3.4.1]. Define each new term once, with one example.
- No overloaded terms: words such as "collapse" or "bias" carry other meanings. Use a neutral word and give its formula [LS §3.4.2–3].
- No mathiness: equations only where they make a definition exact; no theorem or notation you use only once [LS §3.3].
- Inform, do not persuade. If a check weakens a claim, say so in the main text, not only in an appendix [NN].
- Credit prior work warmly; a competitor's weakness does not make your work stronger [SPJ].

## 5. Experiments as evidence

- Each experiment answers "which hypothesis does this rule out?" If it rules out nothing, cut it [NN].
- Baselines are mandatory and must be strong [NN, CL].
- Every number has a sample size and a 95% CI. CIs come from the right unit: the cluster, when observations share one [NN; FIGURE_GUIDE §7].
- Never let a p-value between .01 and .05 be the only support for a central claim; report the effect size with its CI instead [NN].
- Show robustness to each analysis choice you made [LS §5.1, NN]. Give one sentence per check in the main text and the full table in the appendix.
- Mark which analyses you planned before seeing the data and which are post hoc [NN].
- Do not cherry-pick qualitative examples; show random ones, or say how you chose them [NN].
- A simple statistic that answers the question beats a complex model that impresses [JS: a small gain must come with small complexity; LS §3.2].

## 6. Main text vs appendix

- Main text: everything needed to understand and believe the core claims [CL]. Appendix: everything a typical reader skips [CL, NN].
- NN says to polish the appendix little; CL says it must still fully back the claims. Follow CL for numbers and tables, and NN for prose.
- A reader must be able to replicate the work from the main text plus the appendix: prompts, model versions, thresholds, data dates, code link [NN].
- Robustness checks, validation and extra settings go to the appendix, each with one forward reference from the main text.

## 7. Related work

- SPJ: put it late, after the idea, because a comparison is hard to read before the reader knows the idea. CL: put it early only if it is short or the paper must defend its position. NN: second to last, unless prior work motivates the paper.
- In a short paper, a few sentences in the introduction can place the work, with the full section at the end. Each paragraph says how the work differs, not only that other work exists [NN].

## 8. Limitations and conclusion

- Limitations go in the main text, not the appendix [NN, SPJ: acknowledge weaknesses].
- Say what is open and what is settled [LS §5.1].
- Conclusion: CL wants one short paragraph that makes the claims concrete with numbers; NN says skip it. In a short paper, fold it into the discussion.

## 9. Prose

- Active voice: "we ran", not "experiments were run" [SPJ].
- Plain words: "find out", not "ascertain"; "yearly", not "on an annual basis" [SPJ].
- Short sentences, one idea each [NN].
- Define each term once, before first use; then use only that term. No synonyms for one object [CL].
- No bare "this" or "it": write "this rise", "this check" [CL].
- Never "for various reasons"; give the reasons. No "etc." unless the list is obvious [CL].
- Numbers with units and uncertainty: "12.3% (95% CI 10.1–14.5), n = 4,210" [NN; FIGURE_GUIDE §7].

## 10. Revision

- Write in this order: claims, abstract, bullet outline, introduction, draft figures, full draft; get feedback after each step [NN].
- A reader can read a draft fresh only once. Ask "where did you get lost?", not for typos [SPJ].
- Read as a hostile reviewer: for each claim, find the evidence, then the easiest objection [NN, SPJ].
- Treat every critical comment as a request to make the text clearer, not as an error by the reader [SPJ].
- Send the draft to the authors of the closest prior work and ask if you describe it fairly [SPJ].
- Print the final PDF and read it on paper; run a spell checker [CL, SPJ].
- Keep the lab notebook current: every number in the paper traces to a dated entry and a script [JS].

## Pre-submission checklist

1. Each core claim is one sentence and appears in the abstract, the introduction and the section headings.
2. Each contribution bullet is refutable and has a section and figure reference.
3. The abstract names the genre, the data size and one number per claim.
4. Every claim in the introduction has evidence in the body.
5. Every speculation is labeled and sits in one marked paragraph.
6. Proxies are named as proxies and never used as the thing they proxy.
7. Each term is defined once; no synonyms for one object.
8. Every estimate has n and a 95% CI from the correct unit.
9. Each main result has a strong baseline next to it.
10. Robustness checks are listed in one sentence in the main text and tabled in the appendix.
11. Post hoc analyses are marked.
12. Limitations appear in the main text.
13. Prompts, model versions, thresholds and data dates are in the appendix; the code link works.
14. No passive voice in results sentences; no bare "this"; no "etc."; no "various reasons".
15. No roadmap paragraph.
16. Related work says how each cited line of work differs.
17. At least two readers outside the project read the introduction and said where they got lost.
18. Figures pass the FIGURE_GUIDE checklist.
19. Spell checked; printed and read once.
20. Within the page limit plus references and appendix; submitted before the deadline [SPJ].
