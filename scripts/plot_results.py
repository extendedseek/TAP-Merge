"""Plot measured suite results with sample-standard-deviation error bars."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tap_merge.experiments import aggregate_results

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = aggregate_results(json.loads(Path(args.input).read_text()))
labels = [r["training_variant"] + "/" + r["merge_variant"] for r in rows]
fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.36)))
ax.barh(labels, [r["merged_avg_mean"] for r in rows],
        xerr=[r["merged_avg_std"] or 0 for r in rows], color="#2b6e8a", capsize=3)
ax.set_xlabel("Mean merged-domain perplexity (lower is better)")
ax.invert_yaxis()
fig.tight_layout()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=200)
