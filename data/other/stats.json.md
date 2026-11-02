# data/other/stats.json

Filter and parse counts of the journal fetches. One key per source (`scipost`, `f1000`, `bmc`, `plos`, `copernicus`);
each value is a map of counts for that source.

Keys in every source:

| Key | Counts |
|---|---|
| `parsed_reviews` | reviews before filters |
| `short_dropped` | texts with fewer than 100 characters |
| `dup_dropped` | duplicate (paper, text) pairs |

| Source | `parsed_reviews` | `short_dropped` | `dup_dropped` |
|---|---|---|---|
| `scipost` | 8,603 | 209 | 15 |
| `f1000` | 11,846 | 0 | 0 |
| `bmc` | 8,109 | 302 | 2 |
| `plos` | 11,251 | 0 | 1 |
| `copernicus` | 9,682 | 34 | 0 |

Keys of one source only:

| Source | Key | Value | Counts |
|---|---|---|---|
| `scipost` | `api_missing` | 3 | submissions that the API pages did not return |
| `plos` | `resubmission` | 138 | papers that the parser skipped because they have an author response to a previous submission |
| `plos` | `earlier_reviews` | 3 | papers that the parser skipped because the text names an earlier review round |
| `plos` | `letters` | 5,086 | parsed decision letters |
| `plos` | `letters_split_2plus` | 4,500 | letters that split into 2 or more reviewer texts |
| `copernicus` | `pdf with manuscript pages` | 92 | PDFs that the parser skipped because they hold manuscript pages |

A skip reason of `plos` or `copernicus` has a key only when its count is above 0.

## Make

```
uv run python -I src/fetch.py <source>
```

`finalize` in `src/fetch.py` replaces the key of `<source>` and keeps the other keys. It runs in the same step as
`<source>_reviews.parquet`; see `<source>_reviews.parquet.md`. No script in `scripts/` runs it.

## Read by

`src/corpus.py other` (`inventory`) copies the counts of each source into `reports/other_inventory.txt` and reports
`letters_split_2plus` / `letters` for `plos`.
