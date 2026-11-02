# Method-to-code traceability

The manuscript labels below are preserved verbatim from its LaTeX source. A block
means a unique parameter tensor, with matrices transposed when rows exceed columns.
Scalar/vector tensors become 1×m and use implicit U=1. This convention keeps full
forward models intact; there is no LoRA adapter or parameter freezing.

| Manuscript equation/section | Implementation |
|---|---|
| Gradient clipping and corrected accumulator | `train.py`: divide accumulated gradients by actual targets, global norm clip; `optimizer.py`: A=G+xi |
| `eq:momentum_gradient_barycenter` | `optimizer.py`: factorized B/B^T operators and explicit mixed-energy scalar |
| Rank hysteresis and seed construction | `TAPOptimizer.step`: scheduled checks, small eigendecomposition on reduction, residual probes on expansion |
| `eq:dynamic_basis` | `linalg.orth` and B(B^TQ); no n×n B B^T allocation |
| First-moment transport | `optimizer.transport`: U_new^T U_old m_old |
| `eq:second_moment_transport_unbiased` | `optimizer.transport`: clipped centered bias-corrected diagonal proxy; label name does not imply unbiased covariance |
| `eq:first_moment_update` | `TAPOptimizer.step`: current-coordinate moments, practical bias correction |
| Practical and analytical adaptive directions | `TAPOptimizer.step`: optional scalar denominator envelope, independent of practical defaults |
| `eq:generalized_error` | `TAPOptimizer.step`: gradient omission plus kappa times lost historical momentum; first step omits historical term |
| `eq:row_curvature` | Gradient-square mean reductions before EF; no gradient covariance matrices formed |
| `eq:combined_precision` | `archive.precision`: RC/LD maps with positive-median scales |
| Checkpoint saliency | `archive.save_early` and `save_task`: delta² sqrt(H), created only at checkpoints |
| `eq:early_reconstruction` | `save_early`, `reconstructed_early`: U Z plus exact sparse residual values/indices |
| `eq:adaptive_rank_stability` | Per-refresh overlap EMA; per-update corrected-gradient fidelity EMA, initialized to one |
| `eq:early_scaling_coefficient` | `merge.decompose`: bounded nonnegative ridge scalar and exact final residual |
| `eq:trajectory_reliability` | `merge.reliability`: geometric scores; zero exponent omitted, exact positive-exponent zero preserved; prior fallback |
| `eq:reliability_retention` | `merge_block`: raw reliability changes retained fraction, not a uniform rescaling of scores |
| `eq:early_positive_support` | `merge.aggregate`: weighted mask × normalized importance × magnitude support |
| `eq:early_conflict_gate` | `merge.aggregate`: confidence-dependent attenuation of opposing signs, sign(0)=0 |
| `eq:early_curvature_aggregation` | `merge.aggregate`: sum(aD)/(lambda+sum(a)) |
| `eq:data_free_late_coefficient` | `merge_archives`: global retained mass before gates/precision; raw reliability numerator; zero-mass -> zero |
| `eq:final_merged_model` | Base plus scaled early and global-alpha late stage; aliases restored |
| `eq:archive_memory` | `archive.py` stores its factors plus exact int64 sparse-index overhead and dense host saliency |
| `eq:merge_shrinkage_bound` | Tested through the fixed-weight coordinate optimum, not claimed for task losses |

## Important operational conventions

- The basis is seeded by reduced SVD once. Later orthogonalization uses input-order
  rank detection, two projection passes, ambient coordinate completion if deficient,
  and reduced QR. Expansion probes use PyTorch's seeded random generator.
- The budget is `0.75 * 2 * parameter_count * 4` bytes by default, covering persistent
  FP32 tensor states only. It reserves uninitialized blocks and limits every rank
  expansion. Rank initialization may be reduced within configured minimums if the
  global allocation would not fit; failure at minimum rank is explicit. Persistent
  Python/scalar metadata, parameters, gradients and transient workspaces are outside
  this *state tensor* budget and remain in measured GPU peaks.
- The optimizer uses a separate EF tensor for matrix blocks. Gradient accumulation
  and norm clipping cannot overwrite it. No residual is needed for full-rank vectors.
- Checkpoint parameters are post-update; bases/moments/statistics are those used in
  that update. Row/column EMAs are not independently bias-corrected, matching the
  paper. Projected second moments are bias-corrected at the checkpoint.
- When the basis is unchanged, identity transport still clips the centered proxy:
  half-step v is max(old v, old squared bias-corrected mean times its v bias factor).
  Omitting this operation would change the written equations when beta1/beta2 differ.
- Early saliency uses the original early displacement before compression. The final
  saliency is saved as metadata; late saliency is recomputed from the late residual.
  Compression uses FP32 values and int64 flattened indices, density 0.01.
- Checkpoint saliency is generated with the paper's 0.5/0.5 precision mixture. Changing
  precision mixture at merge time does not regenerate original precompression early
  saliency; retain the defaults for paper-equivalent processing.
- Lower positive median is the fixed robust scale rule. Stable sorting breaks all
  top-support ties by flattened block index. This is O(N log N), rather than an
  assumed linear-time selection implementation.
- The global alpha_L requires two block passes. The implementation keeps only one
  block's stacked task maps, and the CPU base/result models; it does not retain all
  early and late stage maps for the entire model. Conventional TIES controls use a
  whole-model vector and therefore have a different host-memory profile.
- Parameter checksums identify the shared base. Tokenizer fingerprints identify
  tokenization rules/vocabulary, and task packages must agree on parameter aliases.

There is no task-data access in `merge.py`, no inference-based coefficient search,
and no Hessian construction. A different merge coefficient tuned on heldout task
data would be a different protocol and must be identified as such.
