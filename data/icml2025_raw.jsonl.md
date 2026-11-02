# data/icml2025_raw.jsonl

Raw OpenReview notes for ICML 2025 accepted papers, with all replies. One JSON line per paper:
`{"id": <forum id>, "content": <submission content>, "replies": [<every reply note>]}`. Replies include official
reviews, comments, rebuttals and decisions. 3,257 papers, 37,139 replies.

## Make

```
uv run python -I src/fetch.py icml --year 2025
```

`scripts/inputs.sh` runs this step. It downloads the file (`icml_download`) only when the file does not exist. It
calls OpenReview API v2 (`https://api2.openreview.net`) with `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` from
`.env`: `client.get_all_notes(content={"venueid": "ICML.cc/2025/Conference"}, details="replies")`. Needs network
and credentials; no cost.

## Notes

- This venue id holds accepted papers only. Rejected papers are not in the file.
- OpenReview content can change, so a new download can differ.
- Git LFS stores it.

## Read by

`src/fetch.py icml` (`icml_reviews`), which writes `icml2025_reviews.parquet`.
