# twicetold

## Code rules

1. No comments, no docstrings, no file docstrings, in any file (Python, shell, jinja2, TOML, YAML). The only
   allowed `#` line is a shebang on line 1. CI runs `scripts/check_no_comments.py` and fails on the first one.
   Explain a decision in the commit message or in `paper/PAPER_NOTES.md`, not in the code.
2. Configuration lives in `src/settings.py` (pydantic-settings, upper-case field names): paths, default model,
   reasoning efforts, prices, OpenRouter routes, API keys from `.env`. No module defines a path, a model name or
   reads `os.environ`.
3. Prompts are jinja2 templates in `src/prompts/`, rendered by `src/prompts.py`. No prompt text in Python.
4. Run chains are `scripts/extract_all.sh` and `scripts/post.sh`. A new pass or post step goes there, not in a
   scratch script.
5. Always UV: `uv run python -I src/<module>.py <command>`. Keep CPU use low (`nice -n 19`, `OMP_NUM_THREADS=2`).
6. Model outputs are named by model short name (`claims_<model><tag>.jsonl`, `reports/<name>_<model>.csv`). A new
   model or effort setting gets a new short name in `settings.OPENROUTER_ROUTES`; existing files are never overwritten.

## Writing rules

1. Reference annotations from Claude Opus or an LLM council are named as such in the paper, never as human.
2. Numbers in the paper come from files in `reports/` or `paper/tables/`; none are typed by hand.
