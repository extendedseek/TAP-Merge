# Critical analysis of the supplied manuscript

## Scientific specification

The contribution is a training/merging pipeline rather than a standalone merge
of ordinary final checkpoints. The optimizer preserves a moving low-rank basis,
projected first and second moments, factorized gradient-square statistics, a dense
error buffer and trajectory indicators. The early checkpoint provides the factors
needed to reconstruct a compressed early task vector. These artifacts are necessary
for faithful use of the proposed merge equations.

Training uses the shorter matrix dimension as the projected dimension. The first
basis uses a reduced SVD of the first clipped task gradient. Later bases follow a
momentum/current-signal barycenter through block power steps. Rank increases when
old-basis proposal energy is low and decreases after repeated high-energy checks.
State transport maps stored first moments into new coordinates; second-moment
transport uses a clipped, bias-corrected diagonal centered proxy. Error feedback
retains both missed corrected gradient and discarded first-moment information.

The checkpoint precision map combines normalized row-column gradient-square
statistics with reconstructed projected second moments. These are relative
importance proxies, not measured Hessians, inverse covariance matrices or exact
Fisher information. Early/final alignment, subspace overlap and projection fidelity
form a bounded geometric reliability score. The merge decomposes final displacement
into a bounded aligned early component and its exact residual, selects stage-specific
supports, attenuates confident sign conflicts and solves a fixed-weight quadratic
objective coordinate by coordinate. A global retained-residual-mass average of raw
reliability scales the late update.

## Reproduction findings and decisions

| Finding | Consequence and repository treatment |
|---|---|
| Main setup describes GPT-2 124M/355M; comparison section explicitly switches to LLaMA/NanoGPT/large Transformers | Keep distinct entries in `configs/benchmark_registry.yaml`; do not claim a GPT-2 recipe reproduces every table. |
| Exact architecture files, base checkpoint revisions, tokenizer revisions and dataset commits are absent | Require real revisions or explicit architecture JSON. Random initialization is labeled; no guessed LLaMA checkpoint is called the original. |
| NanoGPT 0.73B-token and GPT-2 1,000-update benchmarks omit complete source schedules | Preserve reported values; require external source protocol for exact reproduction. |
| Paper cites many source baselines but the supplied bibliography file is absent | Do not invent paper DOIs or reproduce unspecified optimizers under their published names. Provide Adam and external optimizer integration plus conventional merge controls. |
| FineWeb timing uses 300.5 ms and 3,408 tokens/s | Their product is about 1,024 tokens/update. General adaptation uses 64 × 1,024 = 65,536 targets/update. These cannot be the same throughput/batch scope. Logs report actual processed tokens and wall clock. |
| Estimated C4 memory uses an unspecified `G` convention | At 130M parameters, FP32 weights alone are about 0.52 decimal GB; FP32 weights plus two Adam moments are about 1.56 GB, excluding gradients. The reported 0.76 G is not directly this accounting. Keep its original metric label; measure actual tensor bytes separately. |
| Dense EF remains persistent even if a gradient buffer is reused | This reference stores EF separately and counts its full size; it makes no buffer-aliasing memory claim. |
| A 75% Adam-moment budget cannot hold each vector block independently | Enforce a global tensor-state budget across unique parameters. Vector blocks use an implicit U=1 and no redundant zero EF tensor. Python-object overhead/workspaces are measured separately. |
| Paper leaves layer grouping somewhat broad | Treat every unique parameter tensor as a block, including biases/norm vectors. Tied parameters are processed once and aliases restored. |
| Early checkpoint at exactly 10% may fall inside an update | Save the first post-update checkpoint reaching the threshold and record requested/actual token counts. The final target budget is exact. |
| Rank checks and refreshes may be configured on different schedules | Checks happen at refresh steps also divisible by `check_every`; paper defaults 10/50 coincide as intended. |
| Robust normalization offers a median example | Fix lower median of strictly positive entries, with zero map -> zero; use the same rule everywhere. |
| Tied or quantized saliency changes quantile supports | Stable descending sort, flattened-index ties, exactly ceil(kN) retained coordinates. |
| Some ablation semantics are not mathematically specified | Define each operational control in `ablations.md`; do not imply these were recovered from author code. |
| Two ablation figures and the architecture figure are referenced, but image assets are absent | Transcribe only numbers in the supplied text. Missing domain values remain absent rather than reconstructed from guesses. |
| Results lack comprehensive multi-seed uncertainty | Preserve source point estimates. New suites use configurable repeated seeds and sample standard deviation, with `null` for single-seed uncertainty. |
| SlimPajama original endpoint failed authenticated/public retrieval during verification | Keep original provenance and allow an explicitly chosen pinned mirror; never substitute automatically. |

## Limits of the mathematics

The scalar early fit has a positive regularization denominator and a [0,c_max]
constraint. It does not force the late residual orthogonal to the early vector.
Compression error remains in the exact final residual. The geometrically averaged
quality score is empirical and may be zero; normalized task priors can still be
used when all scores vanish, while the raw-score late coefficient remains zero.
Subspace overlap can be one for nested unequal-rank subspaces.

The second-moment transport is approximate because off-diagonal covariance is not
stored and beta1/beta2 can differ. The practical defaults do not meet the restricted
analytical beta2 relation: kappa=9 gives 81/82 ≈ 0.987805, rather than 0.999. The
optional analytical denominator changes the update; enabling it alone does not
establish convergence of the adaptive-rank clipped stochastic optimizer.

The closed-form merge minimizes the fixed-weight quadratic surrogate. Its
coordinate shrinkage bound is an amplitude bound, not a guarantee of downstream
task accuracy. Early support being broader and late regularization stronger are
design choices. These qualifications are preserved in the implementation docs.

## Coverage boundary

The proposed TAP mechanism, general training protocol, three corpus preparation
paths, metrics, memory/timing collection and all five ablation groups are implemented.
Large pretrained checkpoints and source-study optimizer implementations are not
contained in the manuscript. Exact reproduction of all numerical comparisons
requires those assets, matched hardware and complete external benchmark settings.
