"""Integrity check against the packaged source manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text())
failures = []
for name, digest in manifest["sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
if failures:
    raise SystemExit("Integrity mismatch: " + ", ".join(failures))
print(f"Verified {len(manifest['sha256'])} source files")
