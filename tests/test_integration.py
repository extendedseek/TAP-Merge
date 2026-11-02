import json
import torch
import pytest
from tap_merge.archive import safe_load, TaskArchive, fingerprint
from tap_merge.models import create_base
from tap_merge.demo import synthetic_tokens
from tap_merge.train import train
from tap_merge.merge import merge_archives
from tap_merge.config import MergeConfig
from tap_merge.experiments import MERGE_VARIANTS, run_suite


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("integration")
    create_base(out / "base")
    tasks = []
    for i in range(2):
        data = out / f"data{i}"
        synthetic_tokens(data, i, count=1025)
        cfg = dict(base=str(out / "base"), data=str(data), output=str(out / f"task{i}"),
                   device="cpu", precision="fp32", train_tokens=256, early_tokens=128,
                   validation_every_tokens=128, eval_tokens=64, sequence_length=16,
                   microbatch=2, accumulation=2, warmup_profile_steps=0,
                   optimizer=dict(lr=0.001, rank=2, rank_min=1, rank_max=4,
                                  refresh_every=2, check_every=2, state_budget_ratio=2.0))
        train(cfg)
        tasks.append(str(out / f"task{i}" / "task"))
    return out, tasks


def test_archive_merge_all_controls(trained):
    out, tasks = trained
    base = safe_load(out / "base" / "model.pt")
    for options in MERGE_VARIANTS.values():
        state, diagnostics = merge_archives(base, tasks, MergeConfig(**options))
        assert 0 <= diagnostics["alpha_late"] <= 1
        assert state["head.weight"].data_ptr() == state["token.weight"].data_ptr()
        assert all(torch.isfinite(x).all() for x in state.values())
        assert state.keys() == base.keys()
    exported = TaskArchive(tasks[0]).export(base)
    assert fingerprint(exported) == fingerprint(safe_load(out / "task0" / "model.pt"))


def test_bad_shared_base_is_rejected(trained):
    out, tasks = trained
    base = safe_load(out / "base" / "model.pt")
    base["norm.bias"] = base["norm.bias"] + 1
    with pytest.raises(ValueError, match="exact supplied base"):
        merge_archives(base, tasks)


def test_full_training_counts_and_resume(trained):
    out, _ = trained
    cfg = json.loads((out / "task0" / "resolved_config.json").read_text())
    before = safe_load(out / "task0" / "model.pt")
    result = train(cfg, resume=out / "task0" / "resume.pt")
    assert result["scored_tokens"] == 256
    assert result["optimizer_state_bytes"] <= result["budget_bytes"]
    assert fingerprint(before) == fingerprint(safe_load(out / "task0" / "model.pt"))


def test_suite_plan_covers_five_ablation_groups():
    result = run_suite(dict(output="unused", tasks={"c4": "a", "fineweb": "b"}, training={}))
    assert len(result["training_variants"]) == 7
    assert len(result["merge_variants"]) == 9
    assert result["seeds"] == [42, 43, 44]
