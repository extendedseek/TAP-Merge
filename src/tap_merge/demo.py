"""Offline integration demo."""
import json
from pathlib import Path
import numpy as np
from .models import create_base
from .experiments import run_suite, aggregate_results


def synthetic_tokens(directory, seed=0, vocab=256, count=8193):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split in ("train", "dev", "test"):
        # Distinct streams; intentionally simple periodic domain signals plus noise.
        values = (np.arange(count) * (seed + 1) + rng.integers(0, 5, size=count)) % vocab
        values.astype(np.uint32).tofile(directory / (split + ".bin"))
    (directory / "tokens.json").write_text(json.dumps(dict(dtype="uint32", tokenizer_fingerprint="synthetic-v1-256",
                                                          synthetic=True, counts={k: count for k in ("train", "dev", "test")})))


def smoke(output, all_ablations=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "base" / "model.pt").exists():
        create_base(output / "base")
    tasks = {}
    for i, name in enumerate(("domain_a", "domain_b")):
        path = output / "data" / name
        synthetic_tokens(path, i)
        tasks[name] = str(path)
    config = dict(output=str(output / "experiments"), seeds=[42],
                  groups=["dual_stage", "transport", "localization", "curvature", "rank"] if all_ablations else ["dual_stage"],
                  tasks=tasks, training=dict(base=str(output / "base"), device="cpu", precision="fp32", cpu_threads=1,
                      train_tokens=2048, early_tokens=512, validation_every_tokens=1024, eval_tokens=256,
                      sequence_length=32, microbatch=2, accumulation=2, warmup_profile_steps=1,
                      optimizer=dict(lr=0.001, rank=2, rank_min=1, rank_max=8, rank_increment=1,
                                     refresh_every=2, check_every=2, hysteresis=2, state_budget_ratio=2.0)))
    if all_ablations:
        # Paper fixed ranks (48/128) are capped by matrix dimensions in this tiny model.
        # Larger budget permits these full-rank smoke controls.
        config["training"]["optimizer"]["state_budget_ratio"] = 3.0
    records = run_suite(config, execute=True)
    report = aggregate_results(records)
    (output / "smoke_report.json").write_text(json.dumps(dict(synthetic=True, results=report), indent=2))
    return report
