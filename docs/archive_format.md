# Saved artifact format

## Shared base bundle

`model.json` records kind, architecture/config, initialization, seed, revision when
available and SHA-256 fingerprint. `model.pt` is the CPU FP32 state dictionary.
HF initialization also saves `tokenizer/`; random custom architectures require an
explicitly selected compatible tokenizer at packing time. The model/weight tensor
fingerprint includes names, shapes, dtype and bytes, sorted by name.

## Task package

Each training run writes `task/manifest.json` with schema=1, base fingerprint,
canonical parameter blocks, tied aliases, tokenizer fingerprint, model/data
provenance and final token count. Block filenames are the first 24 hexadecimal
characters of SHA-256(parameter name), avoiding unsafe parameter-path filenames.

For each canonical parameter:

| File | Contents |
|---|---|
| `task/early/ID.pt` | U, bias-corrected projected v, R, C, b/f summaries, step; early coordinates Z; sparse Q flattened int64 indices/FP32 values/shape; dense original early saliency |
| `task/final/ID.pt` | Final full parameter tensor; U, bias-corrected projected v, R, C, b/f summaries, step; dense final saliency |

Everything is stored on the host using FP32 (indices int64). Dense H is not stored;
it is reconstructed on demand. Saliency maps are intentionally dense host metadata,
as allowed by the manuscript. Final saliency remains archived even though the
late-stage merge recomputes its own saliency from the decomposed residual.

Merge verifies common base identity, compatible blocks/shapes/aliases and tokenizer
identity. Tied weights are updated once and all aliases point to the same merged
tensor. Only parameters are merged; a model with buffers changed during adaptation
is rejected during task saving. Current target backbones have fixed buffers.

The archive includes enough information to reconstruct an ordinary expert state
dictionary (`export-expert`) and the dual-stage merged model (`merge`). A versioned
manifest separates this format from arbitrary author checkpoints. Files use
`torch.load(weights_only=True)`; load trusted local artifacts only.

## Training state and outputs

`resolved_config.json`, `environment.json`, `train.jsonl`, `validation.jsonl`,
`early_checkpoint.json`, `summary.json`, `resume.pt`, and `model.pt` live beside
the package. `resume.pt` is an atomic update-level checkpoint containing model,
optimizer, data order/cursor, RNG, progress, config/base/data identities. Resume
requires identical config and input metadata and the original early files.
It does not resume inside a partial gradient-accumulation update.

`merged/model.pt` is an ordinary state dictionary. `merged/merge.json` records
config, alpha_L, per-block reliability and normalized weights, alignment scales,
selection counts and residual masses. Export an HF-compatible directory with
`export-hf`; the base architecture and tokenizer remain explicit inputs.
