"""Numerical counterfactuals only, not realizable same-byte cache formats."""

from mosaickv.evaluation.projection_source_feasibility import attention_output, errors


def compare(
    query, full_key, full_value, old_key, old_value, new_key, new_value, positions, query_positions
):
    reference = attention_output(query, full_key, full_value, positions, query_positions)
    pairs = dict(
        refitted_K_old_quantized_V=(new_key, old_value),
        old_quantized_K_refitted_V=(old_key, new_value),
        refitted_K_FullKV_V=(new_key, full_value),
        FullKV_K_refitted_V=(full_key, new_value),
        old_quantized_K_FullKV_V=(old_key, full_value),
        FullKV_K_old_quantized_V=(full_key, old_value),
    )
    return {
        name: errors(reference, attention_output(query, key, value, positions, query_positions))
        for name, (key, value) in pairs.items()
    }
