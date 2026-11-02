# All five manuscript ablation groups

The orchestration distinguishes merge-only controls (same experts) from optimizer
controls (new independent expert training). Each seed uses the same data order and
base across applicable controls. Test evaluation occurs only after settings are fixed.

| Group / source | Variant name | Executed change |
|---|---|---|
| Dual-stage vs single-stage, SlimPajama four domains | `full` | Aligned early + residual; early/late settings; global alpha_L |
| | `single_early` | Merge full final delta once using early support/gate/regularization and early checkpoint precision; importance recomputed for full delta; coefficient 1 |
| | `single_late` | Merge full final delta once using late rules/final precision; coefficient 1 |
| | `ties`, `average`, `dare_ties` | Same final TAP-trained experts; no early decomposition |
| State transport/error feedback, SlimPajama four domains | `no_transport` | Keep old coordinate moment slots without basis rotation; zero-pad or truncate on rank change; EF remains active |
| | `no_error_feedback` | A=G, omit dense xi; moment transport remains active |
| | `no_transport_or_feedback` | Apply both preceding changes |
| Reliability-aware localization, C4+FineWeb | `uniform_budgets` | Use stage midpoint .65 early/.125 late for every task/block; preserve task reliability weights |
| | `no_sign_attenuation` | Every conflict gate =1; preserve all other stages |
| | `magnitude_only` | Rank selected coordinates by abs(stage displacement), keep original saliency in sign support and original precision weights |
| Curvature/trajectory, C4+FineWeb | `no_curvature` | H=1 in both stages; early importance normalized reconstructed-early delta²; late importance normalized residual² |
| | `no_trajectory` | Remove overlap/fidelity factors, retain directional q only with exponent 1 |
| | `random_masking` | Seeded uniform random selection scores, same exact per-block counts; original saliency/gates/precision retained |
| Adaptive rank, C4+FineWeb | `fixed_average_rank` | Rank48 (dimension-capped), refresh basis, transport/EF active |
| | `fixed_max_rank` | Rank128 (dimension-capped), refresh basis, transport/EF active |
| | `static_basis` | Fixed rank48 initialized once by SVD; no later basis refresh |

The supplied paper names these variants but does not fully specify no-transport
moment handling, single-stage precision choices, uniform-budget value, no-trajectory
exponent renormalization or the exact scope of magnitude-only masking. The table
above defines reproducible choices for this reference. They are not asserted to
be the original author's undocumented choices. Change them explicitly if author
clarification becomes available.

Static/fixed rank48 and fixed rank128 preserve the same global state-budget cap.
If a required rank cannot fit, training raises a budget error; it does not silently
grant that variant more memory. Adaptive initialization may reduce planned rank
within its configured bounds when necessary. The tiny all-ablation smoke uses a
larger state cap deliberately, since full-rank tiny matrices have a different
overhead ratio; it is clearly marked synthetic.

`configs/suites/slimpajama.yaml` runs groups 1–2 on four domains.
`configs/suites/c4_fineweb.yaml` runs groups 3–5 on two corpora.
Both use three seeds by default. This adds uncertainty estimates to new runs;
the source paper's point estimates are not converted into fictitious seed runs.

The runner trains the full baseline once/seed/task and reuses its archives for all
merge-only variants. Optimizer controls retrain their experts, then use the full
dual-stage merge. Outputs include Expert Avg, Merged Avg, Merge Gap, per-domain
NLL/PPL, actual target counts, full merge diagnostics and each training summary.
`report` computes sample standard deviation (n-1); one-seed standard deviation is
null. `scripts/export_tables.py` and `plot_results.py` use only these measured files.
