# data/iclr2027_raw.jsonl

Raw OpenReview notes for all ICLR 2027 submissions, with their replies, downloaded on 7 November 2026, the day the
reviews became public. One JSON line per submission:
`{"id": <forum id>, "content": <submission content>, "replies": [<every reply note>]}`. 42,790 submissions (39,950
under review, 1,682 desk-rejected, 1,158 withdrawn), 125,740 replies, 125,728 of them official reviews.

## Make

```
uv run python -I src/fetch.py iclr_download --year 2027
```

It downloads the file only when the file does not exist. It calls OpenReview API v2 (`https://api2.openreview.net`)
with `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` from `.env`:
`client.get_notes(invitation="ICLR.cc/2027/Conference/-/Submission", details="replies")`, 1,000 submissions per
request. Needs network and credentials; no cost.

## Notes

- The download came before author responses and decisions, so `content.venueid` gives only the status at that day.
- The review form changed in 2027: `paper_summary`, `critical_strengths`, `critical_weaknesses`,
  `questions_for_authors`, and ratings on a 1–4 scale.
- OpenReview content can change, so a new download can differ.

## Read by

`src/corpus.py append_openreview_iclr` (`from_openreview_iclr`), which appends the reviews and papers to
`reviews.parquet` and `papers.parquet`.
