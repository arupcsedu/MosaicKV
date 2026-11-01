"""CPU-only opportunity screen, not a nonlinear-model stability certificate."""

from mosaickv.evaluation.last_layer_tail import statistics


def screen(reference, anchor, heads, tokens):
    if (
        reference.device.type != "cpu"
        or reference.shape != anchor.shape
        or reference.ndim != 2
        or len(reference) != 32
    ):
        raise ValueError("CPU 32-prefix logits and identical conditioning required")
    result = []
    for layer, a in sorted(heads.items()):
        for position, b in sorted(tokens.items()):
            if a.shape != anchor.shape or b.shape != anchor.shape:
                raise ValueError("proposal logits do not share the anchor shape")
            # Both individual changes already include propagation through the
            # COMPLETE remaining transformer. Joint interactions are unknown.
            predicted = anchor + (a - anchor) + (b - anchor)
            values = statistics(reference, predicted)
            result.append(dict(layer=layer, position=position, statistics=values))
    if not result:
        raise ValueError("both nonempty intervention families required")
    result.sort(
        key=lambda r: (
            not r["statistics"]["guard_pass"],
            r["statistics"]["token_mismatches"],
            r["statistics"]["maximum_KL_FP32"],
            r["statistics"]["maximum_logit_difference"],
            r["statistics"]["mean_KL_FP32"],
            r["layer"],
            r["position"],
        )
    )
    return result
