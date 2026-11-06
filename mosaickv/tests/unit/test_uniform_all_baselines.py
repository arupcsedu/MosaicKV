import evicastkv_uniform_all_baselines as control
import pytest


def records():
    return {
        method: {
            arm: dict(
                packet_sha256=f"{method}:{arm}",
                prompt_graph_reader=True,
                direct_policy={"method": method},
            )
            for arm in ("graph", "graph-original", "graph-lt")
        }
        for method in (*control.METHODS, "mosaickv_exact", "prefixkv_reimpl")
    }


def plan():
    return dict(
        frozen_runtime_policies=dict(
            segment=True, rotary=True, copy=False, frozen=True, overlap=True
        ),
        common_original_norm_replay=True,
        selected_common_input_reuse=True,
    )


@pytest.mark.parametrize("method", control.METHODS)
def test_all_ports_keep_their_own_packet_policy_and_common_backend(method):
    values = control.configurations(records(), method, plan())
    assert len(values) == 3 and control.CANDIDATE in values
    assert values[control.CANDIDATE]["verified_constructor"] == "async_handoff"
    for key, record in values.items():
        assert record["uniform_attention_core"] and not record["fused_transport_predicate"]
        assert record["native_history_reader"] and record["native_history_graph"]
        if key != control.CANDIDATE:
            assert record["packet_sha256"] == f"{method}:{key[1]}"
            assert record["direct_policy"] == {"method": method}


def test_unregistered_baseline_is_rejected():
    with pytest.raises(ValueError, match="registered"):
        control.configurations(records(), "official_lookm", plan())


def test_synthetic_statistical_fixture_is_not_a_quality_or_freshness_gate():
    config = control.configurations(records(), control.METHODS[0], plan())
    rows = [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(total_wall_seconds=6.0 if (m, a) == control.CANDIDATE else 7.5),
        )
        for m, a in config
        for t in range(-5, 20)
    ]
    audited = {
        key: {k: 1 if key == control.CANDIDATE else 2 for k in control.driver.STORAGE}
        for key in config
    }
    value = control.result_summary(rows, audited, config, 3, 5, 20)
    assert value["four_percent_engineering_gate"]
    assert value["independent_quality_N"] == 0 and value["no_paper_quality_superiority_claim"]
    with pytest.raises(ValueError, match="unique paired"):
        control.result_summary(rows[:-1], audited, config, 3, 5, 20)


def test_real_source_boundary_ignores_legacy_import_defaults():
    from mosaickv.evaluation.joint_incoming_projection import validate_source_rows

    value = control.configurations(records(), control.METHODS[0], plan())[control.CANDIDATE]
    validate_source_rows(
        value["source_tile_rows"],
        value["joint_incoming_projection"],
        value["frozen_joint_plan"],
        value["native_history_reader"],
    )
    assert value["source_tile_rows"] == 192 and value["joint_incoming_projection"] == "kv"
    assert value["frozen_joint_plan"] is True
    assert value["reuse_source_input_copy"] is False
    assert value["post_attention_segment"] and value["exact_rotary_dispatch"]
    assert value["input_norm_segment"] and value["reuse_norm_input"]
