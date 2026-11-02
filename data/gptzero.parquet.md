# data/gptzero.parquet

GPTZero scores for ICLR 2018–2024 reviews, from the AI Review Lottery release (Latona et al., arXiv 2405.02150). We
did not call GPTZero. 70,473 rows, one per review that is both in that release and in `reviews.parquet` of
2026-10-28.

| Column | Type | Meaning |
|---|---|---|
| `review_id`, `paper_id` | string | OpenReview note id and forum id |
| `conf`, `year`, `reviewer` | string, int, string | Copied from `reviews.parquet` of 2026-10-28 |
| `gz_ai`, `gz_human`, `gz_mixed` | float | Source columns `ai`, `human`, `mixed`: GPTZero probabilities, as released |
| `gz_isAI` | bool | Source column `isAI`: the release's AI-assisted label |

Rows per year: 2018: 2,779; 2019: 4,329; 2020: 6,717; 2021: 10,021; 2022: 10,202; 2023: 14,327; 2024: 22,098. The row
order follows `reviews.parquet` of 2026-10-28.

`gz_mixed` is below 0 in 1,115 rows (minimum −0.136). The source CSV has the same values (1,596 rows below 0).

## How we got it

No script in the repo makes this file.

1. On 2026-10-28 at 22:51:35 UTC, a subagent (`data-scout`) of Claude Code session `1e2ee420` (project
   `~/twicetold`) downloaded
   `https://github.com/epfl-dlab/AIReviewLottery/raw/HEAD/analyses/data/reviews_iclr_2018_2024_annotated.csv.zip`
   to `/tmp/scout/lottery/rev.zip` (90,174,742 bytes, sha256 `12b5a0e1…4e07d`). No commit was pinned. The repo shows
   no licence. The CSV inside has 86,690 rows; the repo README says the authors labelled the reviews with GPTZero.
2. On 2026-10-28 at 23:30:31 UTC, an inline Python command in the same session wrote this file:
   - read `id`, `review_id`, `year`, `ai`, `human`, `mixed`, `isAI` from the CSV;
   - renamed them to `paper_id` and `gz_*`, dropped duplicate `review_id` (86,682 left);
   - inner join with `reviews.parquet` (`review_id`, `conf`, `year`, `paper_id`, `reviewer`) on `review_id`.

   The same command also wrote `review_cdates.parquet` and changed `paper_areas.parquet`. The last write was on
   2026-10-28 at 23:30:34 UTC.

The files in `/tmp/scout/lottery/` still exist (2026-11-02), but `/tmp` is not permanent.

## Reproduce

```
uv run python -I - <<'EOF'
import pandas as pd
L = pd.read_csv("/tmp/scout/lottery/x/reviews_iclr_2018_2024_annotated.csv", usecols=["id","review_id","year","ai","human","mixed","isAI"])
L = L.rename(columns={"id":"paper_id","ai":"gz_ai","human":"gz_human","mixed":"gz_mixed","isAI":"gz_isAI"}).drop_duplicates("review_id")
R = pd.read_parquet("data/reviews.parquet", columns=["review_id","conf","year","paper_id","reviewer"])
R.merge(L[["review_id","gz_ai","gz_human","gz_mixed","gz_isAI"]], on="review_id", how="inner").to_parquet("<dir>/gptzero.parquet")
EOF
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the CSV in `/tmp/scout` and the current `reviews.parquet` into a
scratch folder. The result has 70,470 rows. The 3 missing rows (`S1eH2aMfqB`, `HkxH37tRFH` from 2020, `S1eNZj4bpX`
from 2019) are reviews that the 2026-11-01 rebuild of `reviews.parquet` dropped. With those 3 rows removed, this file
is identical to the result (`pandas.DataFrame.equals`). No NaN, no duplicate ids.

## Read by

`src/ai.py style` (prints the comparison with our detector for ICLR 2023 and 2024).
