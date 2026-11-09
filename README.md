# TAP-Merge

**Trajectory-Aligned Projection-Aware Training for Reliability-Guided Multitask Model Merging**

## Install and check

Python 3.10–3.12 is the tested compatibility target. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# CPU checks; on a GPU machine install the appropriate CUDA PyTorch build instead.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf,dev,plots]'
python -m pytest -q
python -m tap_merge smoke --output runs/smoke --all-ablations
```

## Prepare experiment

All experts must use the identical base weights and tokenizer.

```bash
python -m tap_merge init --kind hf --model-id openai-community/gpt2 \
  --revision MODEL_COMMIT --output runs/base_gpt2

python -m tap_merge fetch --corpus c4 --revision DATA_COMMIT \
  --max-documents 1000000 --output data/c4_source
python -m tap_merge clean --corpus c4 --input data/c4_source/documents.jsonl \
  --output data/c4_clean
python -m tap_merge pack --input data/c4_clean --output data/c4_tokens \
  --tokenizer runs/base_gpt2/tokenizer

python -m tap_merge train --config configs/train_gpt2.yaml
```

`fetch` exports a bounded selection; increase the document cap to provide at
least 100,000,001 raw train tokens and sufficiently large heldout streams. 

Learning-rate tuning uses development NLL only:

```bash
python scripts/tune_lr.py --config configs/train_gpt2.yaml --output runs/c4_lr
```

To resume at the next update, retain the original output directory and early files:

```bash
python -m tap_merge train --config configs/train_gpt2.yaml --resume runs/c4_seed42/resume.pt
```

## Merge and evaluate

Train every domain independently by changing `data` and `output` in its config,
keeping `base` identical. Merging reads checkpoints and statistics only.

```bash
python -m tap_merge merge --base runs/base_gpt2 \
  --tasks runs/github/task runs/arxiv/task runs/wikipedia/task runs/stackexchange/task \
  --config configs/merge.yaml --output runs/merged

python -m tap_merge evaluate --base runs/base_gpt2 --weights runs/merged/model.pt \
  --tokens data/github_tokens/test.bin --device cuda --output runs/merged/github.json
python -m tap_merge export-hf --base runs/base_gpt2 --weights runs/merged/model.pt \
  --output runs/merged_hf
```

## Run all ablations

```bash
# Inspect the concrete run matrix first; it does not start training.
python -m tap_merge suite --config configs/suites/slimpajama.yaml
python -m tap_merge suite --config configs/suites/c4_fineweb.yaml
# Execute after preparing the base and all task token streams.
python -m tap_merge suite --config configs/suites/slimpajama.yaml --execute
python -m tap_merge suite --config configs/suites/c4_fineweb.yaml --execute
python -m tap_merge report --input runs/c4_fineweb_ablation/results.json \
  --output runs/c4_fineweb_ablation/summary.json
python scripts/export_tables.py --input runs/c4_fineweb_ablation/results.json --output runs/tables
python scripts/plot_results.py --input runs/c4_fineweb_ablation/results.json --output runs/merged_ppl.png
```

## What is included

| Area | Main files |
|---|---|
| Optimizer, adaptive rank, moment transport, EF | `src/tap_merge/optimizer.py`, `linalg.py` |
| Checkpoint statistics and compressed early vectors | `archive.py` |
| Reliability, localization, conflict gates, weighted merge | `merge.py` |
| Dataset export, splits, overlap removal, packing | `data.py` |
| Causal training, exact target normalization, resume | `train.py` |
| Prediction quality, step reduction, merge gap | `metrics.py` |
| Peak memory and synchronized timing | `profiling.py` |
| Ablation suites, means/std, merging controls | `experiments.py`, `baselines.py` |
| Tiny offline LM and HF GPT-2/LLaMA support | `models.py`, `demo.py` |
| Paper defaults and benchmark contracts | `configs/` |
| Tests, CI, container, packaging | `tests/`, `.github/`, `Dockerfile`, `pyproject.toml` |

