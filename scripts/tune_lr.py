"""Development-only learning-rate selection. Never opens test.bin.

Run from repository root: python scripts/tune_lr.py --config configs/train_gpt2.yaml --output runs/lr
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tap_merge.config import load_config
from tap_merge.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--rates", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4])
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
base = load_config(args.config)
records = []
for rate in args.rates:
    cfg = deepcopy(base)
    cfg["output"] = str(root / f"lr_{rate:g}")
    cfg.setdefault("optimizer", {})["lr"] = rate
    dest = Path(cfg["output"])
    if not (dest / "summary.json").exists():
        train(cfg, resume=dest / "resume.pt" if (dest / "resume.pt").exists() else None)
    if json.loads((dest / "resolved_config.json").read_text()) != cfg:
        raise ValueError("Cached tuning config differs")
    rows = [json.loads(line) for line in (dest / "validation.jsonl").read_text().splitlines()]
    records.append(dict(lr=rate, final_dev_nll=rows[-1]["nll"], run=cfg["output"]))
best = min(records, key=lambda row: row["final_dev_nll"])
result = dict(selection_metric="final development NLL", selected=best, trials=records)
(root / "selection.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
