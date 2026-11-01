"""Bounded one-head-per-layer intervention schedule; labels are not inputs."""

from mosaickv.evaluation.last_layer_tail import order


def proposals(ranking, *, layers):
    if type(layers) is not int or not 1 <= layers <= 32:
        raise ValueError("one to 32 decoder layers required")
    if not ranking or any(
        type(r["layer"]) is not int
        or not 0 <= r["layer"] < layers
        or type(r["head"]) is not int
        or r["head"] < 0
        or not isinstance(r["score"], (float, int))
        or not 0 <= r["score"] < float("inf")
        for r in ranking
    ):
        raise ValueError("finite nonnegative route evidence required")
    result = []
    for layer in range(layers):
        candidates = [r for r in ranking if r["layer"] == layer]
        if not candidates:
            raise ValueError("every decoder layer must be represented")
        best = min(candidates, key=lambda r: (-r["score"], r["head"]))
        result.append(dict(layer=layer, head=best["head"]))
    return tuple(result)


def choose(records, anchor):
    if not records or len({(r["layer"], r["head"]) for r in records}) != len(records):
        raise ValueError("unique fully evaluated causal routes required")
    eligible = [
        r
        for r in records
        if r["statistics"]["guard_pass"]
        and r["statistics"]["mean_KL_FP32"] <= anchor["mean_KL_FP32"]
    ]
    ranked = sorted(eligible, key=lambda r: (*order(r)[:-1], r["layer"], r["head"]))
    return ranked[0] if ranked else None
