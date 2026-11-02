"""Command-line entry points; run tap-merge --help for the full workflow."""
import argparse
import json
from pathlib import Path
from .archive import atomic_save, safe_load, TaskArchive
from .config import load_config, MergeConfig


def write_json(value, path=None):
    text = json.dumps(value, indent=2)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TAP-Merge paper-derived reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="Create the immutable shared base once")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["tiny", "hf"], default="tiny")
    p.add_argument("--model-id")
    p.add_argument("--revision")
    p.add_argument("--config-json")
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("fetch", help="Export pinned HF text documents; no automatic mirror substitution")
    p.add_argument("--corpus", choices=["c4", "fineweb", "slimpajama"], required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--dataset-id")
    p.add_argument("--domain", choices=["github", "arxiv", "wikipedia", "stackexchange"])
    p.add_argument("--max-documents", type=int, default=1000000)
    p.add_argument("--output", required=True)
    p = sub.add_parser("clean", help="Partition and remove heldout overlaps before tokenization")
    p.add_argument("--input", required=True)
    p.add_argument("--corpus", choices=["c4", "fineweb", "slimpajama"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exact-only", action="store_true", help="Smoke/debug only; deviates from paper cleaning")
    p.add_argument("--salt", default="tap-merge-v1")
    p = sub.add_parser("pack", help="EOS-delimited uint32 token streams")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--revision")
    p.add_argument("--max-tokens", type=int)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--max-updates", type=int, help="Pause with a resume checkpoint after this many updates")
    p = sub.add_parser("merge")
    p.add_argument("--base", required=True, help="Base bundle directory")
    p.add_argument("--tasks", nargs="+", required=True, help="Task archive directories")
    p.add_argument("--config")
    p.add_argument("--method", choices=["tap", "average", "ties", "dare_ties"], default="tap")
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--drop-rate", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--priors", nargs="+", type=float)
    p.add_argument("--output", required=True)
    p = sub.add_parser("export-expert")
    p.add_argument("--base", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("export-hf", help="Export HF backbone weights and config for from_pretrained")
    p.add_argument("--base", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--base", required=True)
    p.add_argument("--weights")
    p.add_argument("--tokens", required=True)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1000000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output")
    p = sub.add_parser("suite", help="Five ablation groups; --execute launches training")
    p.add_argument("--config", required=True)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("report", help="Summarize actual run results with mean and sample standard deviation")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("smoke", help="Offline synthetic two-domain train/archive/merge/eval integration")
    p.add_argument("--output", required=True)
    p.add_argument("--all-ablations", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "init":
        from .models import create_base
        write_json(create_base(args.output, args.kind, args.model_id, args.revision, args.seed, args.config_json))
    elif args.command == "fetch":
        from .data import export_hf
        write_json(export_hf(args.corpus, args.revision, args.output, args.domain, args.dataset_id, args.max_documents))
    elif args.command == "clean":
        from .data import clean_documents
        write_json(clean_documents(args.input, args.output, args.corpus, not args.exact_only, args.salt))
    elif args.command == "pack":
        from .data import pack
        write_json(pack(args.input, args.output, args.tokenizer, args.revision, args.max_tokens))
    elif args.command == "train":
        from .train import train
        write_json(train(load_config(args.config), args.resume, args.max_updates))
    elif args.command == "merge":
        from .merge import merge_archives
        from .experiments import baseline_merge
        base = safe_load(Path(args.base) / "model.pt")
        if args.method == "tap":
            cfg = MergeConfig(**(load_config(args.config) if args.config else {}))
            weights, info = merge_archives(base, args.tasks, cfg, args.priors)
        else:
            weights = baseline_merge(base, args.tasks, args.method, args.density, args.drop_rate, args.scale, args.seed)
            info = dict(method=args.method, density=args.density, drop_rate=args.drop_rate, scale=args.scale, seed=args.seed)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_save(weights, output / "model.pt")
        write_json(info, output / "merge.json")
    elif args.command == "export-expert":
        weights = TaskArchive(args.task).export(safe_load(Path(args.base) / "model.pt"))
        atomic_save(weights, args.output)
    elif args.command == "export-hf":
        from .models import load_model
        model, info = load_model(args.base, args.weights)
        if info["kind"] != "hf":
            raise ValueError("export-hf requires a Hugging Face backbone")
        model.save_pretrained(args.output, safe_serialization=True)
        if (Path(args.base) / "tokenizer").exists():
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(Path(args.base) / "tokenizer").save_pretrained(args.output)
    elif args.command == "evaluate":
        from .models import load_model
        from .metrics import evaluate
        model, _ = load_model(args.base, args.weights, args.device)
        write_json(evaluate(model, args.tokens, args.sequence_length, args.batch_size, args.max_tokens, args.device), args.output)
    elif args.command == "suite":
        from .experiments import run_suite
        write_json(run_suite(load_config(args.config), args.execute))
    elif args.command == "report":
        from .experiments import aggregate_results
        write_json(aggregate_results(json.loads(Path(args.input).read_text())), args.output)
    elif args.command == "smoke":
        from .demo import smoke
        write_json(smoke(args.output, args.all_ablations))


if __name__ == "__main__":
    main()
