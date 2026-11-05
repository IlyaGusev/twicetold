# Gold critique claims: adjudication rules

The input is one peer review as JSON: `review_id`, `text`, and `council`, the claim lists that three extraction
models (openai/gpt-6.1-sol, anthropic/claude-opus-5.5, moonshotai/kimi-k3.1-pro) produced with the same prompt. Your job: write the gold claim list for the review. Judge
against the review text; the council lists are evidence, not truth.

Definition. One claim = one distinct problem with the paper that the authors would answer with one fix or one reply.

Rules
1. Skip praise, the paper summary, pleasantries, pointers to text elsewhere ("see questions"), and struck-through or
   "addressed" text. A softened point stays.
2. A question counts only if it implies a defect or a missing result ("Why no error bars?" counts; "How is X swapped?" does not).
3. Merge restatements of one problem inside the review. Split a long bullet only if the parts need different fixes.
4. Keep a claim that two or three council members list, unless the text does not support it. Keep a claim that one
   member lists only if the text clearly raises it as a distinct problem. Add a claim no member lists only if the text
   clearly raises it.
5. `claim`: a paraphrase of at most 25 words that keeps the concrete referents (section, figure, equation, dataset,
   baseline, number). Pick the council wording that is most precise, or write a better one.
6. `category`, one of: novelty, baselines/comparisons, ablation, experiments-scope, statistical-rigor,
   theory/correctness, assumptions, unsupported-claim, weak-results, clarity/presentation, related-work,
   reproducibility/details, compute-cost/scalability, generalization, limitations/societal, typo/formatting, other.
7. `target`, one of: method, theory, experiments, results, claims, related-work, presentation, other.
8. `specific`: true only if the claim names a concrete element of THIS paper (section, equation, figure, table,
   dataset, named method or baseline, number); false if the sentence could be pasted into a review of most papers.
9. `support`: how many of the three council members list this claim (0, 1, 2 or 3). Count a member's claim as the
   same if it points to the same problem, even when the wording differs.

Output: one JSON object `{"claims": [{"claim": ..., "category": ..., "target": ..., "specific": true/false,
"support": 0-3}, ...]}`. Claims in reading order. An empty list is valid when the review raises no critique.