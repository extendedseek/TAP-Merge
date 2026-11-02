"""Pinned dataset export, split-before-tokenize cleaning, and packed token streams."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
import numpy as np
import torch

DOMAINS = {"github": "RedPajamaGithub", "arxiv": "RedPajamaArXiv",
           "wikipedia": "RedPajamaWikipedia", "stackexchange": "RedPajamaStackExchange"}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def doc_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def partition(text, corpus, official_split, salt="tap-merge-v1"):
    value = int(hashlib.sha256((salt + doc_hash(text)).encode()).hexdigest()[:16], 16) / 2**64
    if corpus == "fineweb":
        return "train" if value < 0.8 else "dev" if value < 0.9 else "test"
    if corpus == "c4":
        return "train" if official_split == "train" else "dev" if value < 0.5 else "test"
    if corpus == "slimpajama":
        return {"train": "train", "validation": "dev", "test": "test"}[official_split]
    raise ValueError("Unknown corpus")


def export_hf(corpus, revision, output, domain=None, dataset_id=None, max_documents=1000000):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    ids = dict(c4="allenai/c4", fineweb="HuggingFaceFW/fineweb", slimpajama="cerebras/SlimPajama-627B")
    dataset_id = dataset_id or ids[corpus]
    if not revision:
        raise ValueError("An explicit dataset revision is required; it will be resolved to a commit")
    resolved = HfApi().dataset_info(dataset_id, revision=revision).sha
    name = "en" if corpus == "c4" else "sample-10BT" if corpus == "fineweb" else None
    splits = ["train", "validation"] if corpus == "c4" else ["train"] if corpus == "fineweb" else ["train", "validation", "test"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with (output / "documents.jsonl").open("w") as out:
        for split in splits:
            counts[split] = 0
            stream = load_dataset(dataset_id, name=name, split=split, revision=resolved, streaming=True)
            for row in stream:
                if domain:
                    meta = row.get("meta", {})
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    label = meta.get("redpajama_set_name", meta.get("redpajama_set"))
                    if label != DOMAINS[domain]:
                        continue
                out.write(json.dumps(dict(text=row["text"], split=split), ensure_ascii=False) + "\n")
                counts[split] += 1
                if counts[split] >= max_documents:
                    break
    info = dict(corpus=corpus, dataset_id=dataset_id, revision=resolved, domain=domain,
                counts=counts, max_documents_per_official_split=max_documents)
    (output / "source.json").write_text(json.dumps(info, indent=2))
    return info


def shingles(text):
    words = re.findall(r"\w+", normalized(text).lower())
    if len(words) < 5:
        return {" ".join(words)}
    return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}


def clean_documents(input_jsonl, output, corpus, near_duplicates=True, salt="tap-merge-v1"):
    """Disk-backed document store; heldout priority test > dev > train.

    MinHash only proposes candidates. Word-5-gram Jaccard >= .8 verifies each match.
    The LSH candidate index is held in RAM; see docs/data.md for large-corpus limits.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dbpath = output / "documents.sqlite"
    if dbpath.exists():
        raise FileExistsError("Use a fresh data output directory")
    db = sqlite3.connect(dbpath)
    db.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, digest TEXT, split TEXT, text TEXT, task TEXT)")
    db.execute("CREATE INDEX doc_digest ON docs(digest)")
    with Path(input_jsonl).open() as stream:
        for line in stream:
            row = json.loads(line)
            text = row["text"]
            if not normalized(text):
                continue
            split = partition(text, corpus, row.get("split", "train"), salt)
            db.execute("INSERT INTO docs(digest,split,text,task) VALUES(?,?,?,?)", (doc_hash(text), split, text, row.get("task")))
    db.commit()
    lsh = None
    if near_duplicates:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=0.8, num_perm=128)
    seen, counts, removed = set(), {}, {}
    for split in ("test", "dev", "train"):
        counts[split], removed[split] = 0, 0
        with (output / (split + ".jsonl")).open("w") as out:
            for row_id, digest, text, task in db.execute("SELECT id,digest,text,task FROM docs WHERE split=? ORDER BY id", (split,)):
                duplicate = digest in seen
                sketch = None
                if lsh is not None and not duplicate:
                    words = shingles(text)
                    sketch = MinHash(num_perm=128, seed=42)
                    sketch.update_batch([w.encode() for w in sorted(words)])
                    for key in lsh.query(sketch):
                        other_split, other_text = db.execute("SELECT split,text FROM docs WHERE id=?", (int(key),)).fetchone()
                        if other_split == split:
                            continue
                        other = shingles(other_text)
                        if len(words & other) / max(1, len(words | other)) >= 0.8:
                            duplicate = True
                            break
                if duplicate:
                    removed[split] += 1
                    continue
                seen.add(digest)
                if lsh is not None:
                    lsh.insert(str(row_id), sketch)
                value = dict(text=text, sha256=digest)
                if task is not None:
                    value["task"] = task
                out.write(json.dumps(value, ensure_ascii=False) + "\n")
                counts[split] += 1
    db.close()
    report = dict(corpus=corpus, split_salt=salt, retained=counts, removed=removed,
                  near_duplicates=near_duplicates, minhash_seed=42, num_perm=128,
                  jaccard_threshold=0.8, priority=["test", "dev", "train"])
    (output / "cleaning.json").write_text(json.dumps(report, indent=2))
    source = Path(input_jsonl).parent / "source.json"
    if source.exists():
        shutil.copyfile(source, output / "source.json")
    return report


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_fingerprint(tokenizer):
    vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    # Include tokenization rules, special tokens and added tokens as well as vocabulary.
    backend = tokenizer.backend_tokenizer.to_str() if getattr(tokenizer, "is_fast", False) else str(tokenizer.init_kwargs)
    return hashlib.sha256((vocab + backend + str(tokenizer.special_tokens_map)).encode()).hexdigest()


def pack(input_directory, output, tokenizer_path, revision=None, max_tokens=None):
    from transformers import AutoTokenizer
    from huggingface_hub import model_info
    resolved = None
    if Path(tokenizer_path).is_dir():
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        if not revision:
            raise ValueError("Remote tokenizers require an explicit revision")
        resolved = model_info(tokenizer_path, revision=revision).sha
        tok = AutoTokenizer.from_pretrained(tokenizer_path, revision=resolved)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must define an end-of-text token")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = dict(dtype="uint32", tokenizer=tokenizer_path, tokenizer_revision=resolved, sha256={},
                tokenizer_fingerprint=tokenizer_fingerprint(tok), eos=tok.eos_token_id, counts={})
    for split in ("train", "dev", "test"):
        count = 0
        with (Path(input_directory) / (split + ".jsonl")).open() as stream, (output / (split + ".bin")).open("wb") as out:
            for line in stream:
                ids = tok.encode(json.loads(line)["text"], add_special_tokens=False) + [tok.eos_token_id]
                if max_tokens:
                    ids = ids[:max(0, max_tokens - count)]
                np.asarray(ids, dtype=np.uint32).tofile(out)
                count += len(ids)
                if max_tokens and count >= max_tokens:
                    break
        info["counts"][split] = count
        info["sha256"][split] = file_sha256(output / (split + ".bin"))
    for name in ("source.json", "cleaning.json"):
        source = Path(input_directory) / name
        if source.exists():
            info[name.removesuffix(".json")] = json.loads(source.read_text())
    (output / "tokens.json").write_text(json.dumps(info, indent=2))
    return info


class TokenStream:
    """Each block uses S inputs and S explicit next-token labels (S+1 raw tokens).

    Adjacent blocks share one boundary context token. All available targets are
    scored once per pass, including a padded tail. Order is shuffled by block.
    """
    def __init__(self, path, sequence_length, seed=42, shuffle=True):
        self.tokens = np.memmap(path, mode="r", dtype=np.uint32) if Path(path).stat().st_size else np.array([], dtype=np.uint32)
        self.length = sequence_length
        self.total = max(0, len(self.tokens) - 1)
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(math.ceil(self.total / self.length), generator=self.rng) if shuffle else torch.arange(math.ceil(self.total / self.length))
        self.cursor = 0

    def next(self, batch_size):
        indices = self.order[self.cursor:self.cursor + batch_size]
        if not len(indices):
            raise StopIteration
        self.cursor += len(indices)
        x = torch.zeros((len(indices), self.length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, block in enumerate(indices):
            start = int(block) * self.length
            count = min(self.length, self.total - start)
            x[row, :count] = torch.from_numpy(np.array(self.tokens[start:start + count], dtype=np.int64))
            y[row, :count] = torch.from_numpy(np.array(self.tokens[start + 1:start + count + 1], dtype=np.int64))
        return x, y

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, rng=self.rng.get_state())

    def load_state_dict(self, value):
        self.order, self.cursor = value["order"], value["cursor"]
        self.rng.set_state(value["rng"])


def truncate_targets(labels, count):
    labels = labels.clone()
    flat = labels.flatten()
    valid = torch.nonzero(flat != -100, as_tuple=False).flatten()
    flat[valid[count:]] = -100
    return labels
