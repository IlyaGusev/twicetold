# data/cocodet_main.jsonl

CoCoDet class probabilities for the 26,484 ICLR and NeurIPS reviews of `cocodet_input.parquet`. One JSON object per
line: `review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). The line order is the scoring order (see
step 2), not the input order. A review counts as AI or mixed when `human` is not the largest of the three.

## How we got it

1. Model: CoCoDet, a ModernBERT classifier with three classes (human, mix, ai).
   - Weights: `CoCoDet.bin` from Hugging Face `khaaaaaan/CoCoDet` (`settings.COCODET_REPO`,
     `settings.COCODET_WEIGHTS_FILE`). The code pins no revision. The local cache holds revision
     `7f60a5dcdad3af02767cb57da842ead41424c89d` (blob sha256
     `70506dfe4afbc11612d0c3aa480918250b68b5b9294ffe52aac6490f438b4143`), downloaded on 2026-10-29 at 13:02 UTC.
   - Base model and tokenizer: `answerdotai/ModernBERT-base` (`settings.COCODET_BASE_MODEL`), cached revision
     `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
   - Head: cosine between the CLS vector and three class vectors, times 30, then softmax.
2. Script: `src/detect_cocodet.py` (never committed; deleted on 2026-10-30 at 00:32 UTC). On 2026-11-02 we
   rewrote it as `src/cocodet.py` from a saved copy, with the same logic. It:
   - removes the tags `[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`,
     `[OTHER]`;
   - shuffles the rows (`random_state=0`), then sorts by text length inside chunks of 256 rows;
   - truncates to 512 tokens, batch 8, CPU, on this machine.
3. Command, Claude Code session `881519d6` (project `~/ideas`), 2026-10-29 13:14:51 UTC:

   ```
   HF_TOKEN=$(cat ~/.cache/huggingface/token) setsid nohup uv run python -I src/detect_cocodet.py data/cocodet_input.parquet main > logs/cocodet_main.log 2>&1
   ```

   The log is `logs/cocodet_main.log`: 0 done, 26,484 to score, "finished in 208.7 min". The last write was on
   2026-10-29 at 16:43:47 UTC. No paid API.

## Reproduce

```
uv run python -I src/cocodet.py scores --input data/cocodet_input.parquet --out <dir>/cocodet_main.jsonl
```

`src/cocodet.py` is `src/detect_cocodet.py` (the script that made this file) without comments, with paths and model
names from `src/settings.py`. It removes only the seven ML section tags, as the original did. The command appends to
an existing output and skips ids that it holds.

## Check

On 2026-11-02 we did not rerun the model. We checked the file against `cocodet_input.parquet`:

- 26,484 lines, 26,484 distinct ids; the id set equals the input id set.
- No NaN. Each probability is in [2.1e-14, 1.0]; each row sums to 1.000000.
- Largest class: human 24,335; mix 1,136; ai 1,013.

## Read by

`src/validate.py rq1_evidence`, `src/figures.py by_year` (list `COCODET`).
