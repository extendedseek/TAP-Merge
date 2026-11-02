"""Optional HF integration with locally constructed GPT-2/tokenizer; no downloads."""
# ruff: noqa: E402 -- optional dependency checks precede imports intentionally.
import json
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
from tap_merge.models import create_base, load_model
from tap_merge.archive import safe_load, fingerprint
from tap_merge.data import pack
from tap_merge.train import train
from tap_merge.merge import merge_archives


def local_setup(tmp_path):
    arch = tmp_path / "arch.json"
    arch.write_text(json.dumps(dict(model_type="gpt2", vocab_size=8, n_embd=16, n_layer=1,
                                    n_head=2, n_positions=32, n_ctx=32, bos_token_id=1,
                                    eos_token_id=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)))
    base = tmp_path / "base"
    create_base(base, kind="hf", config_json=arch)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2,
                                 "hello": 3, "world": 4, "science": 5, "text": 6, "code": 7}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
    tokenizer.save_pretrained(base / "tokenizer")
    docs = tmp_path / "docs"
    docs.mkdir()
    for split in ("train", "dev", "test"):
        (docs / (split + ".jsonl")).write_text("\n".join(json.dumps(dict(text="hello world science text code")) for _ in range(100)))
    data = tmp_path / "tokens"
    pack(docs, data, str(base / "tokenizer"))
    return dict(base=str(base), data=str(data), output=str(tmp_path / "run"),
                precision="fp32", device="cpu", sequence_length=16, microbatch=2, accumulation=2,
                train_tokens=128, early_tokens=64, validation_every_tokens=64, eval_tokens=64,
                optimizer=dict(rank=2, rank_min=1, rank_max=4, refresh_every=2, check_every=2, state_budget_ratio=3.0))


def test_hf_train_pack_merge_export(tmp_path):
    cfg = local_setup(tmp_path)
    result = train(cfg)
    assert result["scored_tokens"] == 128
    from pathlib import Path
    base = safe_load(Path(cfg["base"]) / "model.pt")
    merged, _ = merge_archives(base, [str(Path(cfg["output"]) / "task")])
    model, _ = load_model(cfg["base"])
    model.load_state_dict(merged)
    target = tmp_path / "export"
    model.save_pretrained(target, safe_serialization=True)
    restored = AutoModelForCausalLM.from_pretrained(target, local_files_only=True)
    assert fingerprint(model.state_dict()) == fingerprint(restored.state_dict())


def test_real_update_level_training_resume_is_exact(tmp_path):
    cfg = local_setup(tmp_path)
    from pathlib import Path
    full = dict(cfg, output=str(tmp_path / "full"))
    train(full)
    paused = train(cfg, max_updates=1)
    assert paused["status"] == "paused"
    train(cfg, resume=paused["resume"])
    uninterrupted = safe_load(Path(full["output"]) / "model.pt")
    resumed = safe_load(Path(cfg["output"]) / "model.pt")
    assert all(torch.equal(uninterrupted[k], resumed[k]) for k in resumed)


@pytest.mark.parametrize("method,extra", [("adam", {}), ("external", {"external_optimizer": "torch.optim:SGD"})])
def test_plain_optimizer_benchmark_paths(tmp_path, method, extra):
    cfg = local_setup(tmp_path)
    result = train(cfg | dict(method=method) | extra)
    assert result["scored_tokens"] == 128
    assert result["method"] == method
