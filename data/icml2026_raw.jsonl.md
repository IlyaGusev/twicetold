# data/icml2026_raw.jsonl

Raw OpenReview notes for ICML 2026 accepted papers, with all replies. One JSON line per paper:
`{"id": <forum id>, "content": <submission content>, "replies": [<every reply note>]}`. Replies include official
reviews, comments, rebuttals and decisions. 6,341 papers, 95,957 replies.

## Make

```
uv run python -I src/fetch.py icml --year 2026
```

`scripts/inputs.sh` runs this step. It downloads the file (`icml_download`) only when the file does not exist. It
calls OpenReview API v2 (`https://api2.openreview.net`) with `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` from
`.env`: `client.get_notes(content={"venueid": "ICML.cc/2026/Conference"}, details="replies")`, 1,000 papers per request. Needs network
and credentials; no cost.

## Notes

- This venue id holds accepted papers only. Rejected papers are not in the file.
- OpenReview content can change, so a new download can differ.
- Git LFS stores it.

## Read by

`src/fetch.py icml` (`icml_reviews`), which writes `icml2026_reviews.parquet`.
