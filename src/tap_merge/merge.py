"""Data-free dual-stage merge. This module has no dataset/model forward dependency."""
from dataclasses import asdict
import hashlib
import torch
from .config import MergeConfig
from .linalg import matrix, restore, normalize, top_mask
from .archive import TaskArchive, fingerprint, precision, reconstructed_early


def decompose(early, final, c_max=1.0, eps=1e-8):
    dot = (early * final).sum()
    scale = (dot / (early.square().sum() + eps)).clamp(0, c_max)
    aligned = scale * early
    q = (dot / (early.norm() * final.norm() + eps)).clamp(0, 1)
    return aligned, final - aligned, q, scale


def reliability(q, b, f, priors, cfg):
    exponents = (cfg.gamma_q, cfg.gamma_b, cfg.gamma_f)
    if not cfg.trajectory:
        # Ablation retains q only, as implied by the paper's discussion.
        exponents = (1.0, 0.0, 0.0)
    raw = torch.ones_like(q)
    for indicator, exponent in zip((q, b, f), exponents):
        if exponent > 0:
            raw *= indicator.clamp(0, 1).pow(exponent)
    total = (priors * raw).sum()
    weights = priors * raw / total if total > 0 else priors.clone()
    return raw, weights


def aggregate(d, j, h, masks, weights, kappa, regularizer, eps=1e-8, attenuation=True):
    w = weights.reshape((-1,) + (1,) * (d.ndim - 1))
    support = w * masks * j * d.abs()
    positive = (support * (d > 0)).sum(dim=0)
    negative = (support * (d < 0)).sum(dim=0)
    elected = torch.sign(positive - negative)
    confidence = (positive - negative).abs() / (positive + negative + eps)
    gate = torch.where(torch.sign(d) == elected, 1.0, (1 - confidence).pow(kappa))
    if not attenuation:
        gate = torch.ones_like(d)
    a = w * h * masks * gate
    total = a.sum(dim=0)
    return (a * d).sum(dim=0) / (regularizer + total), total, confidence


def merge_block(base, early_stats, final_stats, priors, cfg, block_name=""):
    e_raw = torch.stack([reconstructed_early(e) for e in early_stats])
    finals = torch.stack([matrix(f["final"] - base) for f in final_stats])
    early, late, q, scales = [], [], [], []
    for e, f in zip(e_raw, finals):
        de, dl, qi, ci = decompose(e, f, cfg.c_max, cfg.eps)
        early.append(de)
        late.append(dl)
        q.append(qi)
        scales.append(float(ci))
    early, late, q = torch.stack(early), torch.stack(late), torch.stack(q)
    raw, weights = reliability(q, torch.tensor([s["b_bar"] for s in final_stats]),
                               torch.tensor([s["f_bar"] for s in final_stats]), priors, cfg)
    if cfg.curvature:
        he = torch.stack([precision(s, cfg.lambda_rc, cfg.lambda_ld, cfg.eps) for s in early_stats])
        hf = torch.stack([precision(s, cfg.lambda_rc, cfg.lambda_ld, cfg.eps) for s in final_stats])
    else:
        he, hf = torch.ones_like(early), torch.ones_like(late)
    je = torch.stack([normalize(e["saliency"], cfg.eps) if cfg.curvature
                      else normalize(reconstructed_early(e).square(), cfg.eps) for e in early_stats])
    jl = torch.stack([normalize(d.square() * h.sqrt(), cfg.eps) for d, h in zip(late, hf)])
    if cfg.mode == "single_early":
        early = finals
        je = torch.stack([normalize(d.square() * h.sqrt(), cfg.eps) for d, h in zip(finals, he)])
    elif cfg.mode == "single_late":
        late = finals
        jl = torch.stack([normalize(d.square() * h.sqrt(), cfg.eps) for d, h in zip(finals, hf)])
    generator = torch.Generator().manual_seed(cfg.seed + int(hashlib.sha256(block_name.encode()).hexdigest()[:8], 16))
    stages = []
    masses, reliable_masses = 0.0, 0.0
    budgets = []
    for stage, (d, j, h, lo, hi, kappa, lam) in enumerate((
        (early, je, he, cfg.early_min, cfg.early_max, cfg.kappa_early, cfg.lambda_early),
        (late, jl, hf, cfg.late_min, cfg.late_max, cfg.kappa_late, cfg.lambda_late),
    )):
        masks, counts = [], []
        for i in range(len(d)):
            k = (lo + hi) / 2 if cfg.uniform_budgets else lo + (hi - lo) * float(raw[i])
            score = j[i]
            if cfg.masking == "magnitude":
                score = d[i].abs()
            elif cfg.masking == "random":
                score = torch.rand(d[i].shape, generator=generator)
            masks.append(top_mask(score, k))
            counts.append(int(masks[-1].sum()))
        masks = torch.stack(masks)
        update, _, _ = aggregate(d, j, h, masks, weights, kappa, lam, cfg.eps, cfg.sign_attenuation)
        stages.append(update)
        budgets.append(counts)
        if stage == 1:
            mass = (masks * d).abs().flatten(1).sum(dim=1)
            masses = float(mass.sum())
            reliable_masses = float((mass * raw).sum())
    diagnostics = dict(reliability=raw.tolist(), weights=weights.tolist(), scales=scales,
                       selected=budgets, residual_mass=masses, reliable_residual_mass=reliable_masses)
    return stages[0], stages[1], diagnostics


@torch.no_grad()
def merge_archives(base_state, directories, cfg=None, priors=None):
    cfg = cfg or MergeConfig()
    archives = [TaskArchive(x) for x in directories]
    if not archives:
        raise ValueError("At least one task archive is required")
    digest = fingerprint(base_state)
    first = archives[0].manifest
    for a in archives:
        if a.manifest["base_fingerprint"] != digest:
            raise ValueError("Experts must share the exact supplied base checkpoint")
        if a.manifest["blocks"] != first["blocks"] or a.manifest["aliases"] != first["aliases"]:
            raise ValueError("Incompatible expert parameter layout")
        if a.manifest["metadata"].get("tokenizer_fingerprint") != first["metadata"].get("tokenizer_fingerprint"):
            raise ValueError("Expert tokenizer metadata mismatch")
    priors = torch.tensor(priors if priors is not None else [1 / len(archives)] * len(archives), dtype=torch.float32)
    if len(priors) != len(archives) or not torch.isfinite(priors).all() or (priors < 0).any() or priors.sum() <= 0:
        raise ValueError("Invalid task priors")
    priors /= priors.sum()
    # Global alpha_L needs TWO block passes; do not accumulate model-sized stage tensors.
    diagnostics = {}
    mass, reliable_mass = 0.0, 0.0
    for name in first["blocks"]:
        loaded = [a.load(name) for a in archives]
        _, _, info = merge_block(base_state[name], [x[0] for x in loaded], [x[1] for x in loaded], priors, cfg, name)
        diagnostics[name] = info
        mass += info["residual_mass"]
        reliable_mass += info["reliable_residual_mass"]
    alpha_late = reliable_mass / mass if mass > 0 else 0.0
    state = dict(base_state)
    for name in first["blocks"]:
        loaded = [a.load(name) for a in archives]
        de, dl, _ = merge_block(base_state[name], [x[0] for x in loaded], [x[1] for x in loaded], priors, cfg, name)
        delta = cfg.alpha_early * de + alpha_late * dl
        if cfg.mode == "single_early":
            delta = de
        elif cfg.mode == "single_late":
            delta = dl  # Explicit single-stage control; no dual-stage residual coefficient.
        state[name] = base_state[name] + restore(delta, base_state[name].shape)
    for alias, canonical in first["aliases"].items():
        state[alias] = state[canonical]
    return state, dict(config=asdict(cfg), alpha_late=alpha_late, blocks=diagnostics)
