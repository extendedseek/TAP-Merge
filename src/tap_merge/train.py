"""Single-device all-parameter adaptation with exact valid-target normalization."""
import json
import math
from pathlib import Path
import random
import time
from contextlib import nullcontext
import torch
from .archive import atomic_save, safe_load, cpu_tree, save_early, save_task, fingerprint
from .config import OptimizerConfig
from .optimizer import TAPOptimizer
from .models import load_model, logits
from .data import TokenStream, truncate_targets, file_sha256
from .metrics import evaluate, nll_sum
from .profiling import Profiler, environment, sync


def lr_multiplier(progress, warmup=0.05, floor=0.1):
    if progress < warmup:
        return progress / warmup
    phase = min(1.0, (progress - warmup) / (1 - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * phase))


def train(config, resume=None, max_updates=None):
    cfg = dict(config)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "resolved_config.json").exists() and resume is None:
        raise FileExistsError("Run output exists; use --resume or a new directory")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 1))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("precision", "bf16") == "bf16" and not str(device).startswith("cuda"):
        raise ValueError("BF16 paper training requires CUDA; set precision=fp32 for CPU smoke tests")
    if cfg.get("precision", "bf16") == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16")
    model, model_info = load_model(cfg["base"], device=device)
    base_state = safe_load(Path(cfg["base"]) / "model.pt")
    if fingerprint(base_state) != model_info["fingerprint"]:
        raise ValueError("Base model metadata fingerprint mismatch")
    data_dir = Path(cfg["data"])
    token_info = json.loads((data_dir / "tokens.json").read_text())
    if model_info["kind"] == "hf" and (Path(cfg["base"]) / "tokenizer").exists():
        from transformers import AutoTokenizer
        from .data import tokenizer_fingerprint
        base_tok = AutoTokenizer.from_pretrained(Path(cfg["base"]) / "tokenizer", local_files_only=True)
        if tokenizer_fingerprint(base_tok) != token_info["tokenizer_fingerprint"]:
            raise ValueError("Packed data tokenizer differs from base tokenizer")
    if "tokenizer_fingerprint" not in token_info:
        raise ValueError("tokens.json must identify the tokenizer")
    for split, digest in token_info.get("sha256", {}).items():
        if file_sha256(data_dir / (split + ".bin")) != digest:
            raise ValueError(f"Packed {split} tokens changed since preparation")
    opt_cfg = OptimizerConfig(**cfg.get("optimizer", {}))
    method = cfg.get("method", "tap")
    if method == "tap":
        optimizer = TAPOptimizer(model.parameters(), opt_cfg)
    elif method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg.lr,
                                     betas=(opt_cfg.beta1, opt_cfg.beta2), eps=opt_cfg.eps, weight_decay=0)
    elif method == "external":
        from importlib import import_module
        module_name, class_name = cfg["external_optimizer"].rsplit(":", 1)
        optimizer_class = getattr(import_module(module_name), class_name)
        optimizer = optimizer_class(model.parameters(), lr=opt_cfg.lr, **cfg.get("external_kwargs", {}))
    else:
        raise ValueError("Supported methods: tap, adam, external (official optimizer class)")
    sequence = cfg.get("sequence_length", 1024)
    batch = cfg.get("microbatch", 4)
    accumulation = cfg.get("accumulation", 16)
    budget = cfg.get("train_tokens", 100000000)
    early_at = cfg.get("early_tokens", budget // 10)
    validate_every = cfg.get("validation_every_tokens", 10000000)
    if min(sequence, batch, accumulation, budget, early_at, validate_every) < 1 or early_at > budget:
        raise ValueError("Invalid training or checkpoint budget")
    stream = TokenStream(data_dir / "train.bin", sequence, seed)
    if stream.total < budget:
        raise ValueError(f"Training data has {stream.total} targets, budget needs {budget}; prepare enough data")
    step, seen, processed, early_saved, previous_elapsed = 0, 0, 0, False, 0.0
    profiler = Profiler(device, cfg.get("warmup_profile_steps", 5))
    if resume:
        state = safe_load(resume)
        if state["config"] != cfg:
            raise ValueError("Resume requires the identical resolved training config")
        if state["base_fingerprint"] != model_info["fingerprint"] or state["token_info"] != token_info:
            raise ValueError("Resume base/data metadata changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["stream"])
        step, seen, processed, early_saved = (state[k] for k in ("step", "seen", "processed", "early_saved"))
        previous_elapsed = state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if early_saved and method == "tap" and not (out / "task" / "early").exists():
            raise ValueError("Resume needs the original early archive directory")
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    runtime = environment(device)
    runtime.update(precision=cfg.get("precision", "bf16"), parameter_dtype="float32",
                   microbatch=batch, accumulation=accumulation, sequence_length=sequence,
                   activation_checkpointing=False, parameter_offload=False,
                   attention_implementation=getattr(getattr(model, "config", None), "_attn_implementation", "torch_sdpa"),
                   unique_parameters=sum(p.numel() for p in model.parameters()))
    (out / "environment.json").write_text(json.dumps(runtime, indent=2))
    next_validation = (seen // validate_every + 1) * validate_every
    sync(device)
    run_start = time.perf_counter()
    model.train()
    invocation_updates = 0

    def save_resume():
        sync(device)
        state = dict(config=cfg, base_fingerprint=model_info["fingerprint"], token_info=token_info,
                     model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     stream=stream.state_dict(), step=step, seen=seen, processed=processed,
                     early_saved=early_saved, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     elapsed_seconds=previous_elapsed + time.perf_counter() - run_start)
        atomic_save(state, out / "resume.pt")

    while seen < budget:
        started = profiler.begin()
        optimizer.zero_grad(set_to_none=True)
        targets_this, processed_this, loss_this = 0, 0, 0.0
        for _ in range(accumulation):
            if seen + targets_this >= budget:
                break
            x, y = stream.next(batch)
            y = truncate_targets(y, budget - seen - targets_this)
            valid = int((y != -100).sum())
            if not valid:
                continue
            x, y = x.to(device), y.to(device)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if cfg.get("precision", "bf16") == "bf16" else nullcontext()
            with context:
                loss = nll_sum(logits(model, x), y)
            loss.backward()  # Sum first; divide all accumulated gradients by actual valid targets below.
            loss_this += float(loss.detach())
            targets_this += valid
            processed_this += x.numel()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(targets_this)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_norm", 1.0), error_if_nonfinite=True)
        progress = min(1.0, (seen + targets_this) / budget)
        rate = opt_cfg.lr * lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()  # Error feedback is added only after global clipping.
        step += 1
        seen += targets_this
        processed += processed_this
        seconds = profiler.end(started, step, processed_this)
        row = dict(step=step, scored_tokens=seen, processed_tokens=processed, train_nll=loss_this / targets_this,
                   lr=rate, grad_norm_before_clip=float(grad_norm), update_seconds=seconds)
        if method == "tap":
            row["optimizer_state_bytes"] = optimizer.actual_state_bytes()
            row["mean_current_rank"] = sum(s["m"].shape[0] for s in optimizer.state.values()) / len(optimizer.state)
        with (out / "train.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        if method == "tap" and not early_saved and seen >= early_at:
            save_early(model, optimizer, base_state, out / "task", cfg.get("early_residual_density", 0.01))
            early_saved = True
            (out / "early_checkpoint.json").write_text(json.dumps(dict(step=step, scored_tokens=seen, requested_tokens=early_at)))
        if seen >= next_validation or seen == budget:
            measured = evaluate(model, data_dir / "dev.bin", sequence, batch,
                                cfg.get("eval_tokens", 1000000), device)
            measured.update(step=step, scored_training_tokens=seen, split="dev")
            with (out / "validation.jsonl").open("a") as log:
                log.write(json.dumps(measured) + "\n")
            next_validation = (seen // validate_every + 1) * validate_every
            save_resume()
        profiler.capture_peak()  # Include archive construction and validation workspaces.
        invocation_updates += 1
        if max_updates and invocation_updates >= max_updates and seen < budget:
            save_resume()
            return dict(status="paused", step=step, scored_tokens=seen, resume=str(out / "resume.pt"))
    if method == "tap":
        save_task(model, optimizer, base_state, out / "task",
                  dict(seed=seed, scored_tokens=seen, model=model_info, data=token_info,
                       tokenizer_fingerprint=token_info["tokenizer_fingerprint"]))
    atomic_save(cpu_tree(model.state_dict()), out / "model.pt")
    sync(device)
    profiler.capture_peak()
    elapsed = previous_elapsed + time.perf_counter() - run_start
    summary = dict(method=method, steps=step, scored_tokens=seen, processed_tokens=processed,
                   end_to_end_seconds=elapsed, end_to_end_processed_tokens_s=processed / elapsed,
                   profile_scope="current invocation only; end_to_end includes prior resume invocations",
                   **profiler.report())
    if method == "tap":
        summary.update(optimizer_state_bytes=optimizer.actual_state_bytes(), budget_bytes=optimizer.budget_bytes,
                       mean_time_rank=sum(s["rank_sum"] / s["step"] for s in optimizer.state.values()) / len(optimizer.state))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
