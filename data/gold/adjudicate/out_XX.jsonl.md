# data/gold/adjudicate/out_00.jsonl … out_11.jsonl

Reference claims: the judge's claim list for each review of `in_XX.jsonl`, same order. One line per review:
`review_id`, `claims` (each claim: `claim`, `category`, `target`, `specific`, `support`; `support` counts the council
members that list the claim).

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

On 2026-10-31 between 15:05 and 15:11 UTC, the session started 12 Claude Code subagents (`general-purpose`, model
Opus 5.5), one per input file. Each read `INSTRUCTIONS.md` and its `in_XX.jsonl`, wrote `out_XX.jsonl` with a Python
script and ran the check in step 5 of its prompt. Prompt of the agent for `in_00` (the others differ only in the
file number):

```
You build gold critique-claim annotations for a peer-review study. Work only inside /home/yallen/twicetold/data/tune/adjudicate/.

1. Read INSTRUCTIONS.md in that directory and follow its rules exactly.
2. Read in_00.jsonl (20 peer reviews, each with three model-produced claim lists under "council").
3. For every review, judge the claims against the review text and write the gold claim list. Do the reasoning yourself per review; do not just copy one council member.
4. Write out_00.jsonl in the same directory: one JSON object per line, one line per input review, in input order, with the exact schema from INSTRUCTIONS.md (fields review_id, claims; each claim has claim, category, target, specific, support). Write the file with the Write tool or a Python script via `uv run python` (never plain python); do not run anything heavy.
5. Validate at the end with this command and fix any error it reports:
   cd /home/yallen/twicetold && uv run python -I -c "import json; C={'novelty','baselines/comparisons','ablation','experiments-scope','statistical-rigor','theory/correctness','assumptions','unsupported-claim','weak-results','clarity/presentation','related-work','reproducibility/details','compute-cost/scalability','generalization','limitations/societal','typo/formatting','other'}; T={'method','theory','experiments','results','claims','related-work','presentation','other'}; ins=[json.loads(l)['review_id'] for l in open('data/tune/adjudicate/in_00.jsonl')]; outs=[json.loads(l) for l in open('data/tune/adjudicate/out_00.jsonl')]; assert [o['review_id'] for o in outs]==ins, 'order/ids'; bad=[(o['review_id'],c) for o in outs for c in o['claims'] if c['category'] not in C or c['target'] not in T or not isinstance(c['specific'],bool) or c['support'] not in (0,1,2,3) or len(c['claim'].split())>25]; print('ok' if not bad else bad[:5])"

Report back in at most five lines: number of reviews done, mean gold claims per review, and anything in the reviews you could not resolve.
```

These are LLM labels, not human labels.

## Reproduce

`src/gold.py judge_claims --out <dir>` calls the judge model (`settings.GOLD_JUDGE_MODEL`, `opus-5.5` through
OpenRouter) once per review with the rules of `src/prompts/gold_claims.j2`, a strict JSON schema and the 25-word
check. This is an API call, not a Claude Code agent, so it can differ from the original labels.

## Read by

`src/gold.py merge_claims`.
