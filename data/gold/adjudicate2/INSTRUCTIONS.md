# Gold claim grouping: adjudication rules

The input is one panel as JSON: `panel_id`, `kind`, `year`, `n_reviews`, `claims` and `council`.
- `claims`: the gold critique claims of the panel's reviews. Id `R2-5` means review 2, claim 5. Each claim has its
  category and specific flag.
- `kind`: `same` means the reviews are of ONE paper. `cross` means each review is of a DIFFERENT paper (a placebo
  panel): claims from different reviews then concern different papers, and a merge across reviews is right only when
  the two critiques would be answered by the same fix even so (for example, two reviews that both name the same
  missing baseline method). Expect very few such merges.
- `council`: the groupings that three models (openai/gpt-6.1-sol, anthropic/claude-opus-5.5, moonshotai/kimi-k3.1-pro) produced for these
  claims. Each group has a problem statement and the claim ids it holds. They are evidence, not truth.

Your job: write the gold grouping of the claims into problems.

Rules
1. Two claims share a group only if they point to the same underlying problem, so the authors would answer both with
   one fix or one reply. Judge the problem raised, not the wording or the category.
2. Do not merge claims only because they share a category or a generic request. "More baselines" and "no comparison
   with method X" are one group only if the text of both points to the same missing comparison. "Unclear writing in
   Section 3" and "notation in Section 5 is confusing" are two groups. "More datasets" and "only tested on CIFAR-10"
   are one group only if both ask for the same kind of extra evaluation.
3. A claim that merges two problems goes into the group it covers most.
4. Every claim id appears in exactly one group. A claim that matches nothing forms its own group.
5. Two claims from the SAME review are usually two groups (the extractor already merged restatements). Put them in
   one group only if they clearly restate one problem.
6. Where the council members disagree, read the two claims and decide by rule 1. Where all three agree, follow them
   unless rule 1 clearly says otherwise.

Output: one JSON object `{"clusters": [{"problem": "<one short sentence>", "claim_ids": ["R0-3", "R2-1"]}, ...]}`.
Singleton groups are listed too. Every claim id of the panel must appear exactly once over all groups.