# Source provenance and technical references

Primary scientific specification: user-supplied LaTeX manuscript, copied unchanged
to `paper/manuscript.tex`. It is the authority for algorithm/default choices.
The manuscript references `sample.bib` and figures but did not include them.

Official technical references consulted for interface/data provenance verification
(accessed 2026-11-01; these do not supply missing author experiment configurations):

- PyTorch SVD: https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html
- PyTorch optimizer API: https://docs.pytorch.org/docs/stable/optim.html
- Hugging Face GPT-2 API: https://huggingface.co/docs/transformers/model_doc/gpt2
- C4 dataset card: https://huggingface.co/datasets/allenai/c4
- FineWeb dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb paper: https://arxiv.org/abs/2406.17557
- SlimPajama original release: https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama
- SlimPajama official preprocessing: https://training-api.cerebras.ai/en/latest/wsc/port/prepare-data/slim_pajama.html
- Original SlimPajama dataset address: https://huggingface.co/datasets/cerebras/SlimPajama-627B

The last dataset endpoint returned 401 during verification. A historical release
description and a live accessible data endpoint are different evidence. The fetch
command therefore exposes an explicit dataset-ID override without guessing a mirror.
External baseline citation keys in the supplied manuscript are retained as source
labels in transcribed tables; missing bibliographic records are not invented.
