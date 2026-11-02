# Reproduction scope and checklist

This reference implements the paper's mechanisms; it does not certify its reported
scientific results. Supplied materials contain manuscript source only: no author
weights, raw experiment logs, complete bibliography, original figure data or exact
benchmark architecture/config files. The results tables are transcribed for comparison.

Before a new full experiment, fix a shared model revision, tokenizer revision,
dataset revisions, selection caps/data ordering, split salt, overlap-removal scope,
hardware/backend and seed list. Inspect actual token counts after preparation.
Choose learning rates on development NLL only and freeze every merge setting before
testing. Record the general adaptation protocol separately from source-study pretraining
benchmarks. Retain data source manifests and cleaning reports with packed tokens.

Each run records its resolved config, environment, base identity, target counts,
optimizer states, archive metadata and measured results. Single-device BF16 training
uses FP32 state. No GPU benchmark has been performed during package creation; CPU
smoke timing is not a substitute for reported GPU throughput or memory. CUDA SVD,
QR and attention backends can vary across devices/versions; seeded execution does
not guarantee bitwise cross-hardware equality. Match backend and software versions
for controlled comparisons.

Run mathematical tests, offline integration, optional local HF integration, lint
and build checks before scientific experiments. `VALIDATION.md` distinguishes
checks actually performed from steps requiring user hardware/data. Result scripts
only aggregate measured files; reference paper numbers are never used as fallback
outputs for failed or missing runs.

After a resume, update-level profiling describes the current invocation. End-to-end
elapsed time incorporates saved prior training elapsed time (excluding process
restart/model reload). Preserve prior summaries/logs for a full hardware peak audit;
the resumed process cannot reconstruct a peak that was not recorded before interruption.

Full publication parity still requires architecture/source schedules for the LLaMA,
NanoGPT and NorMuon benchmarks; matched official baseline implementations; actual
GPU runs; and recovered per-domain values from the missing ablation figures.
