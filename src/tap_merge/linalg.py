"""Matrix orientation, deterministic selection, and robust scale conventions."""
import math
import torch


def matrix(x):
    if x.ndim > 2:
        raise ValueError("Only scalar, vector and matrix parameter blocks are supported")
    y = x.reshape(1, -1) if x.ndim < 2 else x
    return y.T if y.shape[0] > y.shape[1] else y


def restore(x, shape):
    if len(shape) == 2 and shape[0] > shape[1]:
        return x.T.reshape(shape)
    return x.reshape(shape)


def orth(x, rank, eps=1e-7):
    """Reduced QR with explicit rank-deficiency detection and orthogonal completion.

    Columns accepted in input order; completion uses ambient coordinate order.
    Small r-column loop avoids a dense n-by-n completion matrix.
    """
    if rank > x.shape[0]:
        raise ValueError("Rank exceeds ambient dimension")
    accepted = []
    candidates = [x[:, i] for i in range(x.shape[1])]
    def append(v):
        for _ in range(2):
            if accepted:
                u = torch.stack(accepted, dim=1)
                v = v - u @ (u.T @ v)
        norm = torch.linalg.vector_norm(v)
        if norm > eps:
            accepted.append(v / norm)
    for v in candidates:
        append(v)
        if len(accepted) == rank:
            break
    if len(accepted) < rank:
        for i in range(x.shape[0]):
            e = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            e[i] = 1
            append(e)
            if len(accepted) == rank:
                break
    q, r = torch.linalg.qr(torch.stack(accepted, dim=1), mode="reduced")
    return q * torch.where(r.diag() < 0, -1.0, 1.0)


def normalize(x, eps=1e-8):
    positive = x[x > 0]
    if not positive.numel():
        return torch.zeros_like(x)
    # torch.median selects the lower middle for even lengths; pin this convention.
    return x / (positive.median() + eps)


def top_mask(scores, fraction):
    """Exactly ceil(k*N) entries; stable sorting breaks ties by flattened index."""
    if not 0 <= float(fraction) <= 1:
        raise ValueError("Selection fraction outside [0,1]")
    n = scores.numel()
    count = min(n, math.ceil(float(fraction) * n))
    order = torch.argsort(scores.flatten(), descending=True, stable=True)
    out = torch.zeros(n, dtype=torch.bool, device=scores.device)
    out[order[:count]] = True
    return out.reshape_as(scores)


def sparse_encode(x, fraction):
    keep = top_mask(x.abs(), fraction).flatten()
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    return {"indices": idx.cpu(), "values": x.flatten()[idx].cpu(), "shape": list(x.shape)}


def sparse_decode(value, device="cpu"):
    x = torch.zeros(math.prod(value["shape"]), dtype=value["values"].dtype, device=device)
    x[value["indices"].to(device)] = value["values"].to(device)
    return x.reshape(value["shape"])
