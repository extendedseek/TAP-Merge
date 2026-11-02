"""Cross-task heldout decontamination before tokenization.

python scripts/audit_cross_task.py --tasks c4=data/c4_clean fineweb=data/fineweb_clean --output data/audited
The strict audit also removes exact duplicates across task training sets.
"""
import argparse
import json
from pathlib import Path
import shutil
from tap_merge.data import clean_documents

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", required=True, help="name=clean_directory")
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
tasks = dict(item.split("=", 1) for item in args.tasks)
combined = root / "combined.jsonl"
with combined.open("w") as out:
    for name, directory in tasks.items():
        for split in ("train", "dev", "test"):
            with (Path(directory) / (split + ".jsonl")).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    row.update(task=name, split="validation" if split == "dev" else split)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
report = clean_documents(combined, root / "global", "slimpajama", near_duplicates=True)
for name, directory in tasks.items():
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with (root / "global" / (split + ".jsonl")).open() as stream, (dest / (split + ".jsonl")).open("w") as out:
            for line in stream:
                row = json.loads(line)
                if row.get("task") == name:
                    out.write(line)
    if (Path(directory) / "source.json").exists():
        shutil.copyfile(Path(directory) / "source.json", dest / "source.json")
    (dest / "cleaning.json").write_text(json.dumps(dict(cross_task_audit=report, tasks=tasks), indent=2))
print(json.dumps(report, indent=2))
