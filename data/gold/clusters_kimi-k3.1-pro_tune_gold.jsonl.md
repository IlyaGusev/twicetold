# data/gold/clusters_kimi-k3.1-pro_tune_gold.jsonl

Council grouping of the reference claims by `kimi-k3.1-pro` (OpenRouter `moonshotai/kimi-k3.1-pro`). One line per panel: `panel_id`,
`model`, `review_ids`, `targets`, `clusters` (each cluster: `target_id`, `problem`, `claim_ids` such as `R2-5`).
60 lines, 60 panels.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/extract.py pass2 data/tune/panels_tune.jsonl --model kimi-k3.1-pro --tag _tune_gold --claims_files claims_gold_tune.jsonl
--effort medium --threads 12 --budget 15` on 2026-10-31 at about 15:20 UTC. The prompt is identical to the current
`src/prompts/pass2.j2`. The same command also ran a fourth model, `glm-zai`; its file was not kept and the grouping
judges did not see it.

## Reproduce

`src/gold.py council_groups --out <dir>`. It runs each model twice, so the second run retries failed panels.

## Read by

`src/gold.py group_inputs`.
