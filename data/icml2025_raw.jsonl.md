# data/icml2025_raw.jsonl

Raw OpenReview notes for ICML 2025 accepted papers, with all replies. One JSON line per paper.
3,257 papers, 37,139 replies.

Each line: `{"id": <forum id>, "content": <submission content>, "replies": [<every reply note>]}`. Replies include
official reviews, comments, rebuttals and decisions.

## How we got it

`src/fetch.py icml --year 2025` calls `icml_download` when this file does not exist:

- OpenReview API v2 (`https://api2.openreview.net`), logged in with `OPENREVIEW_USERNAME` and
  `OPENREVIEW_PASSWORD` from `.env`;
- `client.get_all_notes(content={"venueid": "ICML.cc/2025/Conference"}, details="replies")`. This venue id holds
  accepted papers only, so rejected papers are not in the file.

The download ran on 2026-10-29 at about 14:40 UTC in Claude Code session `881519d6` (project `~/ideas`). The code
then was `src/fetch_icml2026.py`, which later became `fetch.py icml`. The query and the line format are the same.

## Reproduce

Delete or move the file, then run `uv run python -I src/fetch.py icml --year 2025`. OpenReview content can change
after the download (edits, withdrawn notes), so a new download can differ.

## Read by

`src/fetch.py icml` (`icml_reviews`), which writes `icml2025_reviews.parquet`.
