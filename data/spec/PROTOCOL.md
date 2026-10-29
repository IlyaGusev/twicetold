# Generic-vs-specific annotation protocol

Each line of `spec_input.jsonl` is one critique claim extracted from an ICLR review. Label it alone, from the
claim text only.

`specific`: true only if the claim names a concrete element of THIS paper (section, equation, figure, table,
dataset, named method or baseline, number); false if the sentence could be pasted into a review of most papers
in the field.

Output: one JSON line per item, in input order: {"item_id": "...", "specific": true|false}
Do not skip items. Do not consult any other file.
