# data/cocodet_icml.jsonl

CoCoDet class probabilities for the 11,500 ICML reviews of `cocodet_input_icml.parquet`. One JSON object per line:
`review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). The line order is the scoring order, not the input
order.

## How we got it

1. Model, script and settings: the same as `cocodet_main.jsonl` (CoCoDet `khaaaaaan/CoCoDet` `CoCoDet.bin`, cached
   revision `7f60a5dcdad3af02767cb57da842ead41424c89d`; `src/detect_cocodet.py`; 512 tokens, batch 8, CPU, this
   machine).
2. The script removed only the tags `[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`,
   `[LIMITATIONS]`, `[OTHER]`. The ICML tags, for example `[CLAIMS AND EVIDENCE]` and `[THEORY]`, stayed in the text
   that the model read. All 11,500 input rows have such a tag.
3. Command, Claude Code session `881519d6` (project `~/ideas`), 2026-10-29 18:23:03 UTC:

   ```
   HF_TOKEN=$(cat ~/.cache/huggingface/token) setsid nohup uv run python -I src/detect_cocodet.py data/cocodet_input_icml.parquet icml > logs/cocodet_icml.log 2>&1
   ```

   The log is `logs/cocodet_icml.log`: 0 done, 11,500 to score, "finished in 111.1 min". The last write was on
   2026-10-29 at 20:14:24 UTC. No paid API.

## Reproduce

```
DATA_DIR=<dir> uv run python -I src/ai.py cocodet data/cocodet_input_icml.parquet --name icml
```

The current `src/ai.py` also removes the ICML tags (`SECTION_TAGS`), so its input text differs from the input of
this file. Its output can differ.

## Check

On 2026-11-02 we did not rerun the model. We checked the file against `cocodet_input_icml.parquet`:

- 11,500 lines, 11,500 distinct ids; the id set equals the input id set.
- No NaN. Each probability is in [3.9e-14, 1.0]; each row sums to 1.000000.
- Largest class: human 10,728; mix 632; ai 140.

## Read by

`src/validate.py rq1_evidence`, `src/figures.py by_year` (list `COCODET`).
