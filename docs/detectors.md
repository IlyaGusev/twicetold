# Rerun detectors

Use these steps only when saved detector scores are unavailable. To rebuild the analysis from saved outputs,
follow the [README](../README.md#quick-start) instead. Set up source data with the
[raw corpus build](../README.md#build-the-corpus-from-raw-sources).

The CPU lock file does **not** supply CUDA PyTorch. Use a separate CUDA-capable environment with the project
dependencies. Keep `uv.lock` unchanged and use `uv run --no-sync` so uv does not replace CUDA PyTorch with the CPU
build. The archive does not include the original GPU environment lock or driver version, so GPU reruns are not
fully pinned.

## MELD

With `data/raw/meld_input.parquet` from `meld.py input`, run:

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python -c 'import torch; assert torch.cuda.is_available()'
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/meld.py check --n 64 --batch_tokens 16384
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/meld.py scores --batch_tokens 16384
```

The scorer uses `anon-review-meld-2026/meld` at revision `ddfc9c789b8c8d0f92d3774265316d418c96dc67`, fp32,
2,048-token windows and the top 25% of token scores. It skips keys already in the output; move old scores aside
for a fresh run. `meld.py flags` sets the 99.9th-percentile threshold from pre-2023 texts and rebuilds both labels.

The initial run took about 4.5 hours on one GPU. Later added-text runs used an RTX 3090. These records do not
establish a runtime or peak memory need for the full current corpus. See `raw/meld_scores.jsonl.md` in the dataset.

## EditLens

EditLens is the second detector, used in the appendix. In the CUDA environment, install the `detect` extra and
get access to the gated `pangram/editlens_Llama-3.2-3B` repository. Run:

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/editlens.py check --reference EditLens/data/test.csv --n 500
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/editlens.py scores
```

`check` needs `data/test.csv` from [EditLens](https://github.com/pangramlabs/EditLens) and compares our scores with
Pangram's published scores. The scorer loads the adapter on `unsloth/Llama-3.2-3B` in 4 bits and scores 1,024-token
windows. The recorded run took 9 h 45 min on an RTX 3090. See `raw/editlens_scores.jsonl.md` in the dataset.

## GPTTrace and GigaCheck

These detectors appear in the dataset, not the paper. Run GPTTrace in the CUDA environment:

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/gpttrace.py scores
```

GPTTrace uses `nakas/gpttrace-v8s-1b` in bfloat16 with 512-token windows. See `raw/gpttrace_scores.jsonl.md`.

GigaCheck needs a separate Python 3.11 environment with the
[`gigacheck` package](https://github.com/ai-forever/gigacheck), `fire`, `openai` and `pydantic-settings`. With that
environment active, run outside the project's Python requirement:

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run --no-project --active --no-sync python src/gigacheck_detector.py scores
```

GigaCheck uses `iitolstykh/GigaCheck-Classifier-Multi` in bfloat16 and the first 1,024 tokens of each text. See
`raw/gigacheck_scores.jsonl.md` in the dataset.

## Abstract scores

`abstracts.py input` writes `data/raw/abstract_input.parquet` from the `papers` config. In the GPU environments,
run:

```bash
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/meld.py scores --input data/raw/abstract_input.parquet --out data/raw/meld_abstract_scores.jsonl
OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python src/editlens.py scores --input data/raw/abstract_input.parquet --out data/raw/editlens_abstract_scores.jsonl
OMP_NUM_THREADS=2 nice -n 19 uv run --no-project --active --no-sync python src/gigacheck_detector.py scores --input data/raw/abstract_input.parquet --out data/raw/gigacheck_abstract_scores.jsonl
```

`abstracts.py flags` sets thresholds from abstracts of papers from 2022 and earlier. It writes
`data/raw/abstract_flags.parquet` and `reports/abstract_*.csv`. `scripts/raw.sh` runs both abstract input and flag
steps. After scoring, rerun the raw build in the CPU analysis environment.

