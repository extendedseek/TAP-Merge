"""Convert actually measured suite output into CSV and Markdown."""
import argparse
import csv
import json
from pathlib import Path
from tap_merge.experiments import aggregate_results

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
rows = aggregate_results(json.loads(Path(args.input).read_text()))
if not rows:
    raise ValueError("No measured results")
with (root / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ["| Training | Merge | Seeds | Merged PPL | Merge gap |", "|---|---|---:|---:|---:|"]
for row in rows:
    def fmt(name):
        mean, std = row[name + "_mean"], row[name + "_std"]
        return f"{mean:.4f}" + (f" ± {std:.4f}" if std is not None else " (one seed)")
    lines.append(f"| {row['training_variant']} | {row['merge_variant']} | {row['n']} | {fmt('merged_avg')} | {fmt('merge_gap')} |")
(root / "results.md").write_text("\n".join(lines) + "\n")
