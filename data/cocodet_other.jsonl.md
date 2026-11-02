# data/cocodet_other.jsonl

CoCoDet class probabilities for the 10,042 journal reviews of `cocodet_input_other.parquet`. One JSON object per
line: `review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). The line order is the scoring order, not the
input order.

## How we got it

1. Model, script and settings: the same as `cocodet_main.jsonl` (CoCoDet `khaaaaaan/CoCoDet` `CoCoDet.bin`, cached
   revision `7f60a5dcdad3af02767cb57da842ead41424c89d`; `src/detect_cocodet.py`; 512 tokens, batch 8, CPU, this
   machine).
2. The script removed only the tags `[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`,
   `[LIMITATIONS]`, `[OTHER]`. Other tags, for example `[REPORT]` and `[REQUESTED CHANGES]`, stayed in the text.
   1,990 input rows have such a tag.
3. Two runs in Claude Code session `881519d6` (project `~/ideas`) wrote the file:
   - 2026-10-29 21:59 UTC: the first run scored 986 reviews. The session stopped it at 22:06:10 UTC and restarted it with `nice -n 19` and 2
     threads.
   - 2026-10-29 22:06:18 UTC: the second run skipped the 986 done ids and scored the other 9,056:

     ```
     OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_TOKEN=$(cat ~/.cache/huggingface/token) setsid nohup nice -n 19 uv run python -I src/detect_cocodet.py data/cocodet_input_other.parquet other > logs/cocodet_other.log 2>&1
     ```

   The log is `logs/cocodet_other.log` (second run only): "986 done, 9056 to score", "finished in 139.7 min". The
   last write was on 2026-10-30 at 00:26:10 UTC. No paid API.

## Reproduce

```
DATA_DIR=<dir> uv run python -I src/ai.py cocodet data/cocodet_input_other.parquet --name other
```

The current `src/ai.py` also removes the `[REPORT]` and `[REQUESTED CHANGES]` tags, so its input text differs for
1,990 rows. Its output can differ.

## Check

On 2026-11-02 we did not rerun the model. We checked the file against `cocodet_input_other.parquet`:

- 10,042 lines, 10,042 distinct ids; the id set equals the input id set. The restart left no duplicate lines.
- No NaN. Each probability is in [3.4e-14, 1.0]; each row sums to 1.000000.
- Largest class: human 9,553; mix 355; ai 134.

## Read by

`src/figures.py by_year` (list `COCODET`).
