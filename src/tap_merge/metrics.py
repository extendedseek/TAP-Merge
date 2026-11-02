"""Token-weighted causal likelihood and explicitly named efficiency metrics."""
import math
import statistics
import torch
from torch.nn import functional as F
from .models import logits
from .data import TokenStream, truncate_targets


def nll_sum(predictions, labels):
    return F.cross_entropy(predictions.float().reshape(-1, predictions.shape[-1]),
                           labels.reshape(-1), ignore_index=-100, reduction="sum")


@torch.no_grad()
def evaluate(model, token_path, sequence_length=1024, batch_size=4, max_tokens=1000000, device="cpu"):
    was_training = model.training
    model.eval()
    stream = TokenStream(token_path, sequence_length, shuffle=False)
    total_nll, targets, processed = 0.0, 0, 0
    try:
        while targets < max_tokens:
            try:
                x, y = stream.next(batch_size)
            except StopIteration:
                break
            y = truncate_targets(y, max_tokens - targets)
            total_nll += float(nll_sum(logits(model, x.to(device)), y.to(device)))
            targets += int((y != -100).sum())
            processed += x.numel()
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("Evaluation data contains no next-token targets")
    nll = total_nll / targets
    return dict(nll=nll, perplexity=math.exp(nll) if nll < 700 else float("inf"),
                scored_tokens=targets, processed_tokens=processed)


def merge_gap(merged_domain_ppl, expert_domain_ppl):
    if len(merged_domain_ppl) != len(expert_domain_ppl) or not merged_domain_ppl:
        raise ValueError("Need matched nonempty domains")
    return statistics.mean(merged_domain_ppl) - statistics.mean(expert_domain_ppl)


def steps_to_loss(records, reference):
    reached = [r["step"] for r in records if r["nll"] <= reference]
    return min(reached) if reached else None


def step_reduction(method_steps, adam_steps):
    if method_steps is None:
        return None
    if adam_steps <= 0:
        raise ValueError("Adam step count must be positive")
    return 100 * (1 - method_steps / adam_steps)
