# Training, evaluation and efficiency protocol

## General adaptation recipe

| Setting | Paper default / implementation |
|---|---|
| Objective | Causal next-token cross entropy, all unique parameters trainable |
| Shared backbone | One immutable common pretrained GPT-2 124M or 355M base |
| Train budget | 100,000,000 scored nonpadding target tokens per task |
| Context length | 1,024 target positions per block; S+1 raw tokens |
| Effective batch | 4 sequences × 16 gradient accumulation = 64 sequences |
| Precision | BF16 forward autocast; FP32 parameters, gradients, bases, moments, EF |
| Gradient handling | Sum token losses, accumulate, divide by actual valid targets, global clip 1.0, then add EF |
| Learning rate | Development selection from 1e-5, 3e-5, 1e-4; fixed during test evaluation |
| Schedule | 5% linear warmup then cosine to 10% of peak; progress measured in scored tokens |
| Adam-type parameters | beta1=.9, beta2=.999, rho=.9, kappa_ef=9, eps=1e-8; no weight decay |
| Rank schedule | Initial/min/max 16/8/128, dimension-capped; increment 8 |
| Basis refresh/rank checks | Every 10/50 updates, one block-power iteration |
| Hysteresis | low .85, high .95, three consecutive high checks |
| Curvature/quality EMA | beta_c=.99; beta_b=beta_f=.9; quality initialized to one |
| Early/final capture | First completed update reaching 10M; final at 100M targets |
| Validation | Development every 10M targets and at end; test excluded from training |
| Hardware mode | One device/run; no offload, quantization or activation checkpointing |

The CLI creates a base bundle once; tasks load identical tensor values from it.
For random-initialized benchmark models, provide explicit Hugging Face architecture
JSON to `init --kind hf --config-json ...`; train/load every task from that bundle.
A guessed architecture with a similar parameter count is not source-benchmark parity.
The source does not include the exact LLaMA/NanoGPT/large-Transformer JSONs.

Train logs contain token counts, rates, gradient norm, update time, state bytes and
current rank. Environment and resolved configs record software/device/precision.
Resume stores full model, optimizer, CPU/CUDA RNG, Python RNG, stream order/cursor,
completed updates and actual token counts. Keep early files in the original run.
The implementation is single-device; distributed reduction has not been implemented.

## Metrics

NLL is total negative log likelihood divided by *scored targets*. PPL is exp(NLL).
Do not average batch NLLs when target counts differ. Domain-average PPL is the
arithmetic mean of each domain's PPL; it differs from exp(mean domain NLL).
Expert Avg evaluates each independent expert on its own domain. Merge Gap is the
arithmetic merged-domain average minus that expert average, using the same heldout
token streams. Negative gaps are allowed. The suite saves all per-domain values.

`steps_to_loss` finds the first measured validation update at/below a reference
NLL; no interpolation is assumed. `step_reduction` computes
100*(1-method_steps/Adam_steps), returning missing when a method has not reached
the reference. It does not imply equal wall-clock savings. FineWeb reported source
losses remain named losses; the reference tables do not silently convert them to PPL.

## Timing and memory

CUDA synchronization brackets every update. Per-update time includes data delivery,
forward/backward, gradient normalization/clipping, basis/scoring/statistics work and
optimizer updates. The first five updates are excluded from profiled update means.
Processed-token throughput counts all supplied input positions, including padding
and masked suffixes, divided by synchronized profiled update wall clock.

End-to-end time separately includes development evaluation, checkpoint construction
and writes, training input handling and final saving. It starts after base/model/
optimizer setup and ends after final outputs are saved. It is not `steps * mean`.
Environment initialization and package installation are outside this timing scope.

GPU peaks are captured from initial updates onward. Before each update, prior peaks
are retained and PyTorch peak counters reset; later capture includes early/final
archive creation, resume saves and development evaluation. Report allocated and
reserved CUDA bytes separately. CPU reports CUDA memory as null, never as zero-GPU
training. CUDA allocator metrics are not whole-system device consumption.

`optimizer_state_bytes` counts actual persistent tensor values including dense EF,
bases, moments and row/column statistics. It excludes model weights, gradients,
Python object overhead, transient workspaces and host archives. It is checked
against the state budget, whereas measured peak includes transient allocations.
Decimal GB = bytes/1e9; GiB = bytes/2^30. The original paper's `G` estimated memory
is retained verbatim and is not substituted for these measured quantities.

The arithmetic mean of rank first averages across optimizer steps for each block,
then across unique blocks. It is neither a parameter-size-weighted mean nor a
guarantee of an exact average rank of 48. Fixed-rank controls use the paper's stated
48 and 128 regardless of a new run's observed average, and dimension caps remain.

## Baseline implementation boundary

Adam is directly supported through `method: adam`; it produces ordinary model
checkpoints and optimization metrics. It does not fabricate TAP early statistics.
For official optimizer implementations installed separately, use:

```yaml
method: external
external_optimizer: package.module:OptimizerClass
external_kwargs: {}
```

Supply the class's documented arguments (including its parameter-group structure
where required) and preserve the official commit/environment. The generic adapter
passes all unique trainable parameters and the configured learning rate. If an
official optimizer needs nontrivial parameter groups, write its documented grouping
in a separate adapter before comparing it. The repo does not claim this adapter
alone reproduces GaLore, LDAdam, UMTAM, APOLLO, SUMO, SARA, NorMuon or other source
studies. No approximate substitute is named after those methods.

Uniform averaging and conventional TIES/DARE+TIES are implemented here. TIES selects
the top magnitude fraction of a whole-model task vector, elects sign by the sum
of retained values, then averages nonzero agreeing values. DARE drops independent
coordinates and rescales survivors by 1/(1-p) before TIES. Density .2, drop .5 and
scale 1 are explicit reference choices; the manuscript does not give these settings.

## Distinct comparison benchmarks

The registry records C4 LLaMA 130M/350M PPL, FineWeb NanoGPT 0.73B-token loss,
FineWeb GPT-2 124M rank32/1,000-update LR comparison, SlimPajama LLaMA60M/130M PPL,
and 50B-token Transformer1.1B/5.4B step reduction. These must be configured from
their respective original studies. The general GPT-2 adaptation recipe does not
resolve their missing settings. No full-scale scores are produced in this package.
