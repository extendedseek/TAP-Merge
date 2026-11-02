"""Ablation matrix and sequential experiment orchestration."""
from copy import deepcopy
import json
from pathlib import Path
import statistics
import torch
from .archive import safe_load, atomic_save, TaskArchive
from .config import MergeConfig
from .merge import merge_archives
from .baselines import average, ties, dare_ties
from .models import load_model
from .metrics import evaluate
from .train import train

TRAIN_VARIANTS = {
    "full": {},
    "no_transport": {"state_transport": False},
    "no_error_feedback": {"error_feedback": False},
    "no_transport_or_feedback": {"state_transport": False, "error_feedback": False},
    "fixed_average_rank": {"rank": 48, "rank_min": 48, "rank_max": 48, "adaptive_rank": False},
    "fixed_max_rank": {"rank": 128, "rank_min": 128, "rank_max": 128, "adaptive_rank": False},
    "static_basis": {"rank": 48, "rank_min": 48, "rank_max": 48, "adaptive_rank": False, "refresh_basis": False},
}

MERGE_VARIANTS = {
    "full": {},
    "single_early": {"mode": "single_early"},
    "single_late": {"mode": "single_late"},
    "uniform_budgets": {"uniform_budgets": True},
    "no_sign_attenuation": {"sign_attenuation": False},
    "magnitude_only": {"masking": "magnitude"},
    "no_curvature": {"curvature": False},
    "no_trajectory": {"trajectory": False},
    "random_masking": {"masking": "random"},
}


def baseline_merge(base, tasks, method="average", density=0.2, drop_rate=0.5, scale=1.0, seed=42):
    archives = [TaskArchive(p) for p in tasks]
    # Export validates the common base fingerprint.
    experts = [a.export(base) for a in archives]
    names = list(archives[0].manifest["blocks"])
    for a in archives:
        if a.manifest["blocks"] != archives[0].manifest["blocks"]:
            raise ValueError("Incompatible baseline experts")
    deltas = [torch.cat([(e[n] - base[n]).flatten() for n in names]) for e in experts]
    if method == "average":
        merged = average(deltas)
    elif method == "ties":
        merged = ties(deltas, density)
    elif method == "dare_ties":
        merged = dare_ties(deltas, density, drop_rate, seed)
    else:
        raise ValueError("Unknown baseline merge")
    output, offset = dict(base), 0
    for name in names:
        size = base[name].numel()
        output[name] = base[name] + scale * merged[offset:offset + size].reshape_as(base[name])
        offset += size
    for alias, canonical in archives[0].manifest["aliases"].items():
        output[alias] = output[canonical]
    return output


def run_suite(config, execute=False):
    """Print a concrete plan by default. --execute performs the requested full runs.

    Task configs identify data directories and a COMMON base. Full-trained archives
    are reused for all merge-only controls. Optimizer controls retrain every task.
    """
    root = Path(config["output"])
    groups = config.get("groups", ["dual_stage", "transport", "localization", "curvature", "rank"])
    training = {"full"}
    merging = {"full"}
    if "transport" in groups:
        training |= {"no_transport", "no_error_feedback", "no_transport_or_feedback"}
    if "rank" in groups:
        training |= {"fixed_average_rank", "fixed_max_rank", "static_basis"}
    if "dual_stage" in groups:
        merging |= {"single_early", "single_late"}
    if "localization" in groups:
        merging |= {"uniform_budgets", "no_sign_attenuation", "magnitude_only"}
    if "curvature" in groups:
        merging |= {"no_curvature", "no_trajectory", "random_masking"}
    plan = dict(seeds=config.get("seeds", [42, 43, 44]), groups=groups,
                tasks=list(config["tasks"]), training_variants=sorted(training), merge_variants=sorted(merging),
                baseline_merges=["average", "ties", "dare_ties"] if "dual_stage" in groups else [])
    if not execute:
        return plan
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in plan["seeds"]:
        for variant in plan["training_variants"]:
            paths, expert_ppls = [], []
            for task, data_path in config["tasks"].items():
                cfg = deepcopy(config["training"])
                cfg.update(seed=seed, data=data_path, output=str(root / str(seed) / variant / task))
                cfg.setdefault("optimizer", {}).update(TRAIN_VARIANTS[variant])
                run_dir = Path(cfg["output"])
                if not (run_dir / "summary.json").exists():
                    train(cfg, resume=run_dir / "resume.pt" if (run_dir / "resume.pt").exists() else None)
                saved = json.loads((run_dir / "resolved_config.json").read_text())
                if saved != cfg:
                    raise ValueError(f"Cached run config differs: {run_dir}")
                paths.append(str(run_dir / "task"))
                model, _ = load_model(cfg["base"], run_dir / "model.pt", cfg.get("device", "cpu"))
                value = evaluate(model, Path(data_path) / "test.bin", cfg.get("sequence_length", 1024),
                                 cfg.get("microbatch", 4), cfg.get("eval_tokens", 1000000), cfg.get("device", "cpu"))
                expert_ppls.append(value["perplexity"])
                del model
            expert_avg = statistics.mean(expert_ppls)
            base = safe_load(Path(config["training"]["base"]) / "model.pt")
            variants = plan["merge_variants"] + plan["baseline_merges"] if variant == "full" else ["full"]
            for merge_name in variants:
                if merge_name in MERGE_VARIANTS:
                    mcfg = MergeConfig(**(config.get("merge", {}) | MERGE_VARIANTS[merge_name] | {"seed": seed}))
                    weights, diagnostics = merge_archives(base, paths, mcfg)
                else:
                    weights = baseline_merge(base, paths, merge_name, **config.get("baseline", {}), seed=seed)
                    diagnostics = dict(method=merge_name)
                dest = root / str(seed) / variant / ("merged_" + merge_name)
                dest.mkdir(exist_ok=True)
                atomic_save(weights, dest / "model.pt")
                (dest / "merge.json").write_text(json.dumps(diagnostics, indent=2))
                cfg = config["training"]
                model, _ = load_model(cfg["base"], dest / "model.pt", cfg.get("device", "cpu"))
                domains = {}
                for task, path in config["tasks"].items():
                    domains[task] = evaluate(model, Path(path) / "test.bin", cfg.get("sequence_length", 1024),
                                              cfg.get("microbatch", 4), cfg.get("eval_tokens", 1000000), cfg.get("device", "cpu"))
                merged_avg = statistics.mean(v["perplexity"] for v in domains.values())
                record = dict(seed=seed, training_variant=variant, merge_variant=merge_name,
                              expert_avg=expert_avg, merged_avg=merged_avg, merge_gap=merged_avg - expert_avg,
                              expert_domain_ppl=dict(zip(config["tasks"], expert_ppls)), domains=domains)
                (dest / "metrics.json").write_text(json.dumps(record, indent=2))
                records.append(record)
                del model
    (root / "results.json").write_text(json.dumps(records, indent=2))
    return records


def aggregate_results(records):
    groups = {}
    for row in records:
        key = (row["training_variant"], row["merge_variant"])
        groups.setdefault(key, []).append(row)
    report = []
    for (training, merging), values in sorted(groups.items()):
        row = dict(training_variant=training, merge_variant=merging, seeds=[v["seed"] for v in values], n=len(values))
        for name in ("expert_avg", "merged_avg", "merge_gap"):
            scores = [v[name] for v in values]
            row[name + "_mean"] = statistics.mean(scores)
            row[name + "_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        report.append(row)
    return report
