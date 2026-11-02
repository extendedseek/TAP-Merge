import json
import math
import numpy as np
import pytest
import torch
from tap_merge.data import TokenStream, truncate_targets, partition, doc_hash, clean_documents
from tap_merge.metrics import nll_sum, merge_gap, step_reduction, steps_to_loss


def test_stream_targets_and_masked_tail(tmp_path):
    path = tmp_path / "x.bin"
    np.arange(12, dtype=np.uint32).tofile(path)
    stream = TokenStream(path, 4, shuffle=False)
    x, y = stream.next(3)
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 0]]
    assert y.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, -100]]
    assert (truncate_targets(y, 5) != -100).sum() == 5
    with pytest.raises(StopIteration):
        stream.next(1)


def test_unequal_microbatch_accumulation_matches_full_valid_token_gradient():
    torch.manual_seed(12)
    x = torch.randn(3, 4, 7, requires_grad=True)
    y = torch.randint(0, 7, (3, 4))
    y[-1, -2:] = -100
    full = nll_sum(x, y) / (y != -100).sum()
    expected = torch.autograd.grad(full, x)[0]
    (nll_sum(x[:2], y[:2]) + nll_sum(x[2:], y[2:])).backward()
    x.grad.div_((y != -100).sum())
    assert torch.allclose(x.grad, expected)


def test_metrics_count_actual_targets():
    logits = torch.zeros(2, 3, 5)
    y = torch.tensor([[1, 2, 3], [0, -100, -100]])
    assert float(nll_sum(logits, y)) == pytest.approx(4 * math.log(5))
    assert merge_gap([10, 14], [9, 11]) == 2
    assert step_reduction(70, 100) == pytest.approx(30)
    assert step_reduction(None, 100) is None
    assert steps_to_loss([dict(step=2, nll=3), dict(step=3, nll=2)], 2.5) == 3


def test_hash_split_is_stable_before_tokenization():
    assert doc_hash("hello  world") == doc_hash("hello\nworld")
    assert partition("same text", "fineweb", "train") == partition("same text", "fineweb", "test")
    assert partition("x", "slimpajama", "validation") == "dev"


def test_holdout_priority_exact_cleaning(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = [dict(text="one document for overlap", split=s) for s in ("train", "validation", "test")]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    result = clean_documents(source, tmp_path / "clean", "slimpajama", near_duplicates=False)
    assert result["retained"] == dict(test=1, dev=0, train=0)


def test_stream_resume_keeps_order(tmp_path):
    path = tmp_path / "x.bin"
    np.arange(100, dtype=np.uint32).tofile(path)
    first = TokenStream(path, 4, seed=5)
    first.next(3)
    other = TokenStream(path, 4, seed=99)
    other.load_state_dict(first.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(first.next(2), other.next(2)))


def test_minhash_cross_partition_verification_and_provenance(tmp_path):
    pytest.importorskip("datasketch")
    source = tmp_path / "input.jsonl"
    words = " ".join("word" + str(i) for i in range(150))
    rows = [dict(text=words, split="test", task="a"),
            dict(text=words + " differentword", split="train", task="b")]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "source.json").write_text(json.dumps(dict(revision="test-local")))
    result = clean_documents(source, tmp_path / "clean", "slimpajama", near_duplicates=True)
    assert result["retained"] == dict(test=1, dev=0, train=0)
    row = json.loads((tmp_path / "clean" / "test.jsonl").read_text())
    assert row["task"] == "a"
    assert json.loads((tmp_path / "clean" / "source.json").read_text())["revision"] == "test-local"
