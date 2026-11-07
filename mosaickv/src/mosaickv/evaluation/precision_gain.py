"""Label-free signed benefit of exact K/V restoration, including normalization.

This measures local attention-output error, not vocabulary-logit or task quality.
Keep per-query errors until the final mean; cap workspace by token chunks.
"""

import torch


@torch.inference_mode()
def exact_restoration_gain(queries, original, parent, chunk=64):
    if (
        len(original) != 2
        or len(parent) != 2
        or queries.ndim != 2
        or original[0].ndim != 2
        or any(t.shape != original[0].shape for t in (*original, *parent))
        or queries.shape[-1] != original[0].shape[-1]
        or type(chunk) is not int
        or chunk < 1
        or any(not bool(torch.isfinite(t).all()) for t in (queries, *original, *parent))
    ):
        raise ValueError("finite individual queries and matching original/parent K/V required")
    q = queries.float() / queries.shape[-1] ** 0.5
    k, v = (t.float() for t in original)
    pk, pv = (t.float() for t in parent)
    full = torch.softmax(q @ k.T, -1) @ v
    probability = torch.softmax(q @ pk.T, -1)
    score = q @ pk.T
    ids = torch.arange(k.shape[0], device=k.device)
    anchor = probability @ pv
    base = (full - anchor).square().mean()
    out = []
    for first in range(0, k.shape[0], chunk):
        last = first + chunk
        # Stable single-token replacement: recompute softmax over parent
        # log-partition plus that token's original score, without exp overflow.
        new_score = q @ k[first:last].T
        other_score = (
            score[:, None, :]
            .expand(-1, min(chunk, k.shape[0] - first), -1)
            .masked_fill(ids[None, None, :] == ids[None, first:last, None], -torch.inf)
        )
        other_log = torch.logsumexp(other_score, -1)
        new_log_z = torch.logaddexp(other_log, new_score)
        new_p = torch.exp(new_score - new_log_z)
        corrected = (
            torch.exp(other_score - new_log_z[..., None]) @ pv
            + new_p[..., None] * v[None, first:last]
        )
        out.append(base - (full[:, None] - corrected).square().mean((0, 2)))
    if not bool(torch.isfinite(torch.cat(out)).all()):
        raise ValueError("nonfinite signed restoration gains")
    return torch.cat(out), float(base)


def repair_risk(attention_mass, dispersion, residual_norm):
    import math

    if any(not math.isfinite(x) for x in (attention_mass, dispersion, residual_norm)) or (
        not 0 <= attention_mass <= 1 or dispersion < 0 or residual_norm <= 0
    ):
        raise ValueError("finite actual mass, nonnegative dispersion and positive norm required")
    return attention_mass * math.sqrt(dispersion) / residual_norm
