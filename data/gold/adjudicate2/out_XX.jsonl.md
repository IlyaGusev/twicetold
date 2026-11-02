# data/gold/adjudicate2/out_00.jsonl … out_05.jsonl

Reference grouping: the judge's grouping for each panel of `in_XX.jsonl`, same order. One line per panel: `panel_id`,
`clusters` (each cluster: `problem`, `claim_ids`). Every claim id of the panel appears once. Mean groups per panel:
23.8 for same-paper panels, 32.2 for cross-paper panels.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

On 2026-10-31 between 15:26 and 15:30 UTC, the session started 6 Claude Code subagents (`general-purpose`, model Opus
5.5), one per input file. Prompt of the agent for `in_00` (the others differ only in the file number):

```
You build gold groupings of critique claims for a peer-review study. Work only inside /home/yallen/twicetold/data/tune/adjudicate2/.

1. Read INSTRUCTIONS.md in that directory and follow its rules exactly.
2. Read in_00.jsonl (10 panels; each has the gold claims with ids and three model-produced groupings under "council").
3. For every panel, decide the grouping yourself by the rules; use the council as evidence, not as the answer. Mind the "kind" field: cross-paper panels should have very few merges across reviews.
4. Write out_00.jsonl in the same directory: one JSON object per line, one line per input panel, in input order, with the exact schema from INSTRUCTIONS.md (panel_id, clusters; each cluster has problem and claim_ids). Write it with the Write tool or a Python script via `uv run python` (never plain python); do not run anything heavy.
5. Validate at the end with this command and fix any error it reports:
   cd /home/yallen/twicetold && uv run python -I -c "import json; ins=[json.loads(l) for l in open('data/tune/adjudicate2/in_00.jsonl')]; outs=[json.loads(l) for l in open('data/tune/adjudicate2/out_00.jsonl')]; assert [o['panel_id'] for o in outs]==[i['panel_id'] for i in ins], 'order/ids'; bad=[]; [bad.append((i['panel_id'], sorted(set(c['id'] for c in i['claims'])^set(x for cl in o['clusters'] for x in cl['claim_ids'])), sum(len(cl['claim_ids']) for cl in o['clusters'])-len(i['claims']))) for i,o in zip(ins,outs) if set(c['id'] for c in i['claims'])!=set(x for cl in o['clusters'] for x in cl['claim_ids']) or sum(len(cl['claim_ids']) for cl in o['clusters'])!=len(i['claims'])]; print('ok' if not bad else bad)"

Report back in at most five lines: number of panels done, mean groups per panel for same-paper and for cross-paper panels, and any panel you could not resolve.
```

These are LLM labels, not human labels.

## Reproduce

`src/gold.py judge_groups --out <dir>`: one API call per panel to `settings.GOLD_JUDGE_MODEL` with the rules of
`src/prompts/gold_groups.j2`, a strict JSON schema and a check that the groups cover each claim id once. It can differ
from the original labels.

## Read by

No code in the repo reads it now. The paper appendix (`app:reference`) describes it.
