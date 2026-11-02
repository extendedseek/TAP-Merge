# Validation performed for this package

Validated on 2026-11-01 in a CPU Linux environment with Python 3.12.14,
PyTorch 2.5.1+cpu, Transformers 4.46.3 and Datasets 3.1.0.
`validation/environment.json` records the exact installed versions.

| Check actually run | Outcome |
|---|---|
| `python -m pytest -q` | 46 passed; no skipped optional HF tests in this environment |
| `python -m ruff check src tests scripts` | Passed |
| `python -m compileall -q src tests scripts` | Passed |
| `python -m tap_merge smoke --output ... --all-ablations` | Completed two synthetic tasks across 7 training configurations and 12 merge configurations; 18 evaluated comparisons |
| Local HF tokenizer -> packing -> GPT-2 training -> task archive -> merging -> safe HF export/reload | Passed |
| Interrupted HF training at an update boundary -> resume | Exact tensor agreement with uninterrupted training |
| Persistent optimizer tensor budget | Enforced and tested, including separate dense EF and global vector allocation |
| Base mismatch rejection and tied-weight aliases | Passed |
| MinHash candidate plus verified Jaccard overlap removal | Passed on a local cross-partition example |
| Adam and externally imported optimizer benchmark paths | Passed with Adam and PyTorch SGD |
| Both manuscript suite configs in plan mode | Parsed and produced expected matrices |
| Report/table/plot generation from measured smoke results | Passed |
| Python source distribution and wheel build | Passed |

Tests check rectangular state transport against a dense diagonal-covariance
reference, first-step and rotated-basis EF equations, identity-transport clipping,
rank expansion/reduction, orthogonal completion, row/column statistics before EF,
resume, exact support counts/ties, decomposition, reliability zeros/fallback,
quadratic stationarity, shrinkage, sign gates, target normalization and data order.

`validation/synthetic_ablation_summary.json` and `synthetic_ablation_records.json`
contain actual CPU integration outputs. These are not manuscript results or evidence
of the proposed method's scientific superiority. Fixed48/fixed128 tiny controls
can coincide after dimension caps; full paper backbones have different dimensions.

Not performed: full C4/FineWeb/SlimPajama downloads, pretrained GPT-2/LLaMA benchmark
runs, CUDA BF16 timing/peak measurements, 1.1B/5.4B training, official third-party
optimizer comparisons, Docker image execution, or remote GitHub Actions runs.
The workflow files enable future CI; they do not claim a completed remote run.
Paper-scale reproduction remains subject to the missing source settings listed
in `docs/paper_analysis.md` and `configs/benchmark_registry.yaml`.
