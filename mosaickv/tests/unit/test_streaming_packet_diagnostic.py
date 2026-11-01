import torch
from evicastkv_streaming_packet_diagnostic import differences


def test_bitwise_metadata_difference_and_shape_mismatch_do_not_pass_silently():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert differences((a,), (a.clone(),)) == []
    changed = a.clone()
    changed[1] = 4
    record = differences((a,), (changed,))[0]
    assert record["changed_numeric_values"] == 1 and record["first_indices"] == [1]
    assert record["old_values"] == [2.0] and record["new_values"] == [4.0]
    assert differences((a,), ()) == [dict(tensor_count_old=1, tensor_count_new=0)]
    assert differences((a,), (a[None],))[0]["old_shape"] == [3]


def test_signed_zero_bit_difference_is_separate_from_numerical_equality():
    row = differences((torch.tensor([-0.0]),), (torch.tensor([0.0]),))[0]
    assert row["numerically_equal"] and row["changed_numeric_values"] == 0
