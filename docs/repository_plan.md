# Required repository files and responsibilities

This inventory maps the paper to the code required before implementation. Paths
are relative to the repository root; `MANIFEST.json` lists every packaged file.

| Required files | Purpose |
|---|---|
| `README.md` | Installation, offline check, real workflow, ablations, GitHub upload |
| `pyproject.toml`, `requirements.txt` | Python package, dependencies, CLI, optional HF/dev/plot dependencies |
| `LICENSE`, `CITATION.cff` | Generated-code terms and manuscript citation without invented publication metadata |
| `.gitignore`, `CONTRIBUTING.md`, `SECURITY.md` | Repository hygiene and contribution guidance |
| `Dockerfile`, `Makefile` | CPU container and common commands |
| `MANIFEST.in` | Include docs/configs/scripts/reference material in source distributions |
| `.github/workflows/tests.yml` | Unit/integration tests, lint, full ablation smoke, wheel build |
| `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/pull_request_template.md` | Reproducible issue/PR reporting |
| `src/tap_merge/config.py` | Validated optimizer and merge defaults |
| `src/tap_merge/linalg.py` | Orientation, QR completion, robust scales, sparse residuals, stable top support |
| `src/tap_merge/optimizer.py` | Proposal, adaptive rank, power updates, transport, Adam direction, EF, budget |
| `src/tap_merge/archive.py` | Base identity, unique/tied layout, checkpoint precision/saliency, early/final packages |
| `src/tap_merge/merge.py` | Decomposition, reliability/prior fallback, masks, sign consensus, quadratic aggregation, global alpha_L |
| `src/tap_merge/baselines.py` | Uniform averaging, TIES and DARE+TIES controls |
| `src/tap_merge/models.py` | HF causal LM loading/export and tied-weight offline causal LM |
| `src/tap_merge/data.py` | Corpus export, split hashing, exact/MinHash overlap screening, EOS packing, resumable streams |
| `src/tap_merge/train.py` | All-parameter adaptation, valid-target accumulation, clipping, schedule, validation/resume |
| `src/tap_merge/metrics.py` | NLL/PPL, domain merge gap, steps to loss, step reduction |
| `src/tap_merge/profiling.py` | Synchronized update times, full-run CUDA peaks, environment |
| `src/tap_merge/experiments.py` | Five ablation groups, seed orchestration, same-expert controls, result aggregation |
| `src/tap_merge/demo.py` | Offline two-domain train/archive/merge/eval demonstration |
| `src/tap_merge/cli.py`, `__main__.py`, `__init__.py` | Unified executable interface and importable API |
| `configs/train_gpt2.yaml`, `configs/merge.yaml` | Paper general protocol and merge defaults |
| `configs/suites/slimpajama.yaml`, `configs/suites/c4_fineweb.yaml` | Four-domain and two-corpus experiment matrices |
| `configs/benchmark_registry.yaml` | Distinct comparison benchmarks and missing source contracts |
| `scripts/tune_lr.py` | Development-only grid selection |
| `scripts/audit_cross_task.py` | Optional global heldout overlap audit across cleaned task corpora |
| `scripts/export_tables.py`, `scripts/plot_results.py` | Tables and plots from measured experiment records |
| `scripts/smoke.sh`, `scripts/check_artifact.py` | Complete local check and package-integrity verification |
| `tests/test_optimizer.py` | Equation-level optimizer, orientation, transport, rank, budget and resume checks |
| `tests/test_merge.py` | Decomposition, zero cases, stable masks, consensus, stationarity and shrinkage |
| `tests/test_data_metrics.py` | Split/decontamination, exact targets, accumulation, causal metrics and stream resume |
| `tests/test_integration.py` | Real tiny training, archives, merge variants, tied aliases and identity rejection |
| `tests/test_hf_local.py` | Local HF GPT-2/tokenizer, packing, exact interrupted resume, export and optimizer adapters |
| `docs/paper_analysis.md` | Critical protocol/memory/result audit |
| `docs/method.md` | Equation-to-code traceability and algorithm conventions |
| `docs/data.md`, `docs/experiments.md`, `docs/ablations.md` | Corpus handling, training/evaluation and every ablation definition |
| `docs/archive_format.md`, `docs/reproducibility.md`, `docs/sources.md` | Saved package format, evidence limits and official technical references |
| `paper/manuscript.tex`, `paper/README.md` | Supplied source and missing original compilation assets |
| `results/paper_reference/*.csv`, `results/paper_reference/README.md` | Original reported values, kept apart from measured outputs |
| `VALIDATION.md`, `validation/` | Commands, actual local test evidence and synthetic smoke summary |
| `MANIFEST.json` | SHA-256 hashes of packaged files |

Training data, pretrained models, optimizer resume states and large outputs belong
in ignored `data/` and `runs/` directories, not version-controlled source.
