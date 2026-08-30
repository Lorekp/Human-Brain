from __future__ import annotations

import dataclasses
import inspect
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import pytest
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4
from experiments.r3_pilot import prompt_factor_multispan_event_bridge_v3 as v3
from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as v2,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace

EXPECTED_ROOTS = {
    "interface_sha256": "4b1fc40f03cf28174c24055bed6082d7d2a06a3a367047d86c88275cf7d0b811",
    "atomic_input_sha256": "9462d22cd614bd83ebbd548aef0bdcb26e2b53a8c5b26ac194c51bc13bf0f9f9",
    "curriculum_sha256": "70a73a51f00a6a3cc62ff36ac21f2ddf13c58d60806f2f7e1943a28aa099bb3a",
    "split_sha256": "f0a654960e7d915036f0ff2557ef8ff852a065952e96bb9d2cd88728276c2ded",
    "v3_reference_input_sha256": "6ee1d8a07a11bac570cfaaec0202939064441e8ff2c650571068d45e7b393e84",
    "neutral_sha256": "0cf1aa942f470519c915fa135721d90d6a02b3577d79b092891a72fc497225fe",
    "composition_sha256": "19252fc99a61fdc419edb3304ec3431d70820f4dc2798e3a3f20f5ea5af0e82c",
    "duplicate_controls_sha256": "1367d3f4ac1851446b91204bf15f0d95acb2df3ae24eb0dd979b24b89d154c03",
    "overlap_controls_sha256": "546e6cbff3e6f733cb1a555c218cfc5bb7110e2790d60c96f2d4934445db8880",
    "capacity_controls_sha256": "b621786c870f20424de6605dd6a0b6407b429006e37bd4bb58e6bc4dbf0d89bd",
    "glif_sha256": "c5bb461ab6fb964b5314f4a210327993d08735475e3fea88ba7d84a96afd716e",
    "latency_table_sha256": "8d896eba62371a1a38ed4e071a4a82b02da7d2a3eb3debdd810d8aa36482d471",
    "wta_topology_sha256": "68a9f3fbdcdae30053e84b644819e1308f26cdd130d484b7c72709e78785595f",
    "wta_electrical_controls_sha256": (
        "4f9ff76e89eccf7294f992d97a4d2c38530e9b4283fb03ffca99f3bc8fe010e2"
    ),
    "local_rule_sha256": "1ea847dc847c52c5244c70500a9112919114c04913f80ae113a113eb577b1279",
    "protocol_sha256": "37aa868988319bd70cea735ad715736dabaef2ac59a45ee0c70677fa89ac2124",
}


def _walk(value: object):
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def test_atomic_freeze_has_exact_disjoint_matched_capacity() -> None:
    protocol = v4.PROTOCOL
    lessons = protocol.lessons
    fit_ids = set(protocol.fit_lesson_sha256s)
    heldout_ids = set(protocol.heldout_lesson_sha256s)

    assert len(v4.TRIADS) == 3
    assert len({factor for triad in v4.TRIADS for factor in triad}) == 9
    assert tuple(factor for triad in v4.TRIADS for factor in triad) == v4.TAUGHT_FACTORS
    assert len(protocol.atomic_groups) == v4.GROUP_COUNT == 18
    assert len(lessons) == v4.LESSON_COUNT == 54
    assert len(fit_ids) == v4.FIT_COUNT == 36
    assert len(heldout_ids) == v4.HELDOUT_COUNT == 18
    assert fit_ids.isdisjoint(heldout_ids)
    assert {lesson.sha256 for lesson in lessons if lesson.split == v4.FIT} == fit_ids
    assert {
        lesson.sha256 for lesson in lessons if lesson.split == v4.HELDOUT
    } == heldout_ids

    expected_rows = (
        lambda t, d1, d2: (t, d1, d2, v4.MARKER_TOKEN),
        lambda t, d1, d2: (d1, t, v4.MARKER_TOKEN, d2),
        lambda t, d1, d2: (d2, v4.MARKER_TOKEN, t, d1),
        lambda t, d1, d2: (v4.MARKER_TOKEN, d2, d1, t),
        lambda t, d1, d2: (t, d1, v4.MARKER_TOKEN, d2),
        lambda t, d1, d2: (d1, t, d2, v4.MARKER_TOKEN),
    )
    for lesson in lessons:
        triad = v4.TRIADS[lesson.triad_index]
        d1 = triad[(lesson.target_index + 1) % 3]
        d2 = triad[(lesson.target_index + 2) % 3]
        assert lesson.tokens == expected_rows[lesson.row_index](lesson.factor, d1, d2)

    for group in protocol.atomic_groups:
        assert len(group.lessons) == 3
        assert len({tuple(sorted(lesson.tokens)) for lesson in group.lessons}) == 1
        assert {lesson.split for lesson in group.lessons} == {group.split}
        assert len({lesson.factor for lesson in group.lessons}) == 3
        assert all(
            lesson.tokens[lesson.target_slot] == lesson.factor
            for lesson in group.lessons
        )
        assert all(lesson.tokens.count(v4.MARKER_TOKEN) == 1 for lesson in group.lessons)

    fit_counts = Counter(
        lesson.factor for lesson in lessons if lesson.sha256 in fit_ids
    )
    heldout_counts = Counter(
        lesson.factor for lesson in lessons if lesson.sha256 in heldout_ids
    )
    assert fit_counts == Counter({factor: 4 for factor in v4.TAUGHT_FACTORS})
    assert heldout_counts == Counter({factor: 2 for factor in v4.TAUGHT_FACTORS})

    for factor in v4.TAUGHT_FACTORS:
        fit_slots = {
            lesson.target_slot
            for lesson in lessons
            if lesson.factor == factor and lesson.sha256 in fit_ids
        }
        heldout_slots = {
            lesson.target_slot
            for lesson in lessons
            if lesson.factor == factor and lesson.sha256 in heldout_ids
        }
        assert heldout_slots <= fit_slots

        fit_features = tuple(set() for _bank in range(3))
        for lesson in lessons:
            if lesson.factor != factor or lesson.sha256 not in fit_ids:
                continue
            for destination, source in zip(
                fit_features,
                v2._feature_keys(lesson.tokens),
                strict=True,
            ):
                destination.update(source)
        for lesson in lessons:
            if lesson.factor != factor or lesson.sha256 not in heldout_ids:
                continue
            token_keys, slot_keys, transition_keys = v2._feature_keys(lesson.tokens)
            assert token_keys <= fit_features[0]
            assert slot_keys <= fit_features[1]
            assert len(transition_keys & fit_features[2]) >= 2

    fit_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in fit_ids}
    heldout_inputs = {
        lesson.tokens for lesson in lessons if lesson.sha256 in heldout_ids
    }
    v3_groups, _ = v3.self_posed_local_span_curriculum()
    v3_inputs = {
        lesson.teacher_span.canonical_tokens
        for group in v3_groups
        for lesson in group.lessons
    }
    assert fit_inputs.isdisjoint(heldout_inputs)
    assert (fit_inputs | heldout_inputs).isdisjoint(v3_inputs)


def test_neutral_and_two_event_composition_panels_are_frozen() -> None:
    protocol = v4.PROTOCOL
    heldout_ids = set(protocol.heldout_lesson_sha256s)
    neutral_ids = {span.sha256 for span in protocol.neutral_spans}
    atomic_inputs = {lesson.tokens for lesson in protocol.lessons}

    assert len(protocol.neutral_spans) == v4.NEUTRAL_SPAN_COUNT == 6
    assert len(neutral_ids) == 6
    assert {span.tokens for span in protocol.neutral_spans}.isdisjoint(atomic_inputs)
    assert len(protocol.composition_cases) == v4.COMPOSITION_CASE_COUNT == 18

    for triad_index, triad in enumerate(v4.TRIADS):
        cases = [
            case
            for case in protocol.composition_cases
            if case.triad_index == triad_index
        ]
        assert {(case.left_factor, case.right_factor) for case in cases} == {
            (left, right) for left in triad for right in triad if left != right
        }
        for case in cases:
            assert case.left_factor != case.right_factor
            assert case.expected_factors == (case.left_factor, case.right_factor)
            assert tuple(
                (span.start_token, span.stop_token, span.kind) for span in case.spans
            ) == ((0, 4, "factor"), (4, 8, "neutral"), (8, 12, "factor"))
            assert case.spans[0].source_sha256 in heldout_ids
            assert case.spans[1].source_sha256 in neutral_ids
            assert case.spans[2].source_sha256 in heldout_ids


def test_compositor_controls_cover_duplicate_overlap_and_capacity() -> None:
    protocol = v4.PROTOCOL

    assert len(protocol.duplicate_controls) == v4.DUPLICATE_CONTROL_COUNT == 9
    assert {control.expected_factors[0] for control in protocol.duplicate_controls} == set(
        v4.TAUGHT_FACTORS
    )
    for control in protocol.duplicate_controls:
        assert control.kind == "exact-duplicate"
        assert control.events[0] == control.events[1]
        assert control.expected_status == "ONE_EVENT"
        assert control.expected_factors == (control.events[0].factor,)

    assert len(protocol.overlap_controls) == v4.OVERLAP_CONTROL_COUNT == 3
    for control in protocol.overlap_controls:
        left, right = control.events
        assert left.factor != right.factor
        assert max(left.start_token, right.start_token) < min(
            left.stop_token, right.stop_token
        )
        assert control.expected_status == "STOP_OVERLAPPING_FACTOR_EVENTS"
        assert control.expected_factors == ()

    assert len(protocol.capacity_controls) == v4.CAPACITY_CONTROL_COUNT == 3
    for control in protocol.capacity_controls:
        assert len(control.events) == 3 > v4.EVENT_CAPACITY
        assert all(
            left.stop_token <= right.start_token
            for left, right in zip(
                control.events[:-1], control.events[1:], strict=True
            )
        )
        assert control.expected_status == "STOP_EVENT_CAPACITY_2"
        assert control.expected_factors == ()


def test_glif_latency_table_is_exact_and_replays_through_race_many() -> None:
    protocol = v4.PROTOCOL
    table = protocol.latency_table

    assert v4.SCORE_MIN == -528
    assert v4.SCORE_MAX == 528
    assert v4.STOP_SCORE == 12
    assert len(table) == 529
    assert tuple(entry.score for entry in table) == tuple(range(529))
    assert tuple(entry.first_spike_step for entry in table) == tuple(
        range(v4.ZERO_SCORE_SPIKE_STEP, v4.MIN_SCORE_SPIKE_STEP - 1, -1)
    )
    assert all(
        left.current_pa < right.current_pa
        for left, right in pairwise(table)
    )
    assert protocol.stop_current_pa == table[v4.STOP_SCORE].current_pa
    assert v4.RAW_TO_CURRENT_SCORE_RULE == "current_score=max(raw_score,0)"
    assert tuple(v4._current_score(score) for score in (-528, -1, 0, 1, 528)) == (
        0,
        0,
        0,
        1,
        528,
    )
    with pytest.raises(ValueError):
        v4._current_score(-529)
    with pytest.raises(ValueError):
        v4._current_score(529)

    assert protocol.latency_table_sha256 == v4._namespaced_sha256(
        v4._LATENCY_NAMESPACE,
        {
            "glif_sha256": protocol.glif_sha256,
            "raw_score_min": v4.SCORE_MIN,
            "raw_score_max": v4.SCORE_MAX,
            "raw_to_current_score_rule": v4.RAW_TO_CURRENT_SCORE_RULE,
            "current_score_min": 0,
            "current_score_max": v4.SCORE_MAX,
            "stop_score": v4.STOP_SCORE,
            "entries": tuple(entry.record() for entry in table),
        },
    )

    currents = np.asarray([entry.current_pa for entry in table], dtype=np.float64)
    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        currents[:, None]
    )
    assert tuple(race.spike_step for race in races) == tuple(
        entry.first_spike_step for entry in table
    )
    assert all(race.simultaneous_spike_candidates == 1 for race in races)


def test_wta_contract_binds_exclusive_stop_and_batched_authentication() -> None:
    topology = v4.WTA_TOPOLOGY_RECORD

    assert topology["principal_factor_cells"] == 24
    assert topology["principal_stop_cells"] == 1
    assert topology["candidate_count"] == 25
    assert topology["stop_candidate_index"] == 24
    assert topology["raw_score_domain"] == (-528, 528)
    assert topology["raw_to_current_score_rule"] == v4.RAW_TO_CURRENT_SCORE_RULE
    assert topology["current_score_domain"] == (0, 528)
    assert topology["stop_score"] == v4.STOP_SCORE
    assert topology["race_adapter"] == "CompiledRestingGlifRace.race_many"
    assert topology["batched_current_shape"] == ("authenticated_span_count", 25)
    assert "before and one after" in topology["batch_authentication"]
    assert topology["per_span_checkpoint_hashing"] is False
    assert topology["race_require_unique"] is False
    assert topology["host_argmax_or_tie_break"] is False
    assert "tie routes" in topology["tie_semantics"]
    assert topology["basket_delay_ticks"] == 1
    assert topology["factor_relay_delay_ticks"] == 3
    assert topology["stop_relay_delay_ticks"] == 1
    assert topology["collision_detector_delay_ticks"] == 1
    assert topology["collision_relay_delay_ticks"] == 1
    assert topology["collision_stop_delay_ticks"] == 2
    assert topology["basket_precedes_principal_threshold_on_equal_step"] is True
    assert topology["teacher_clamp_current_pa_hex"] == (5_000.0).hex()
    assert topology["relay_copy_current_pa_hex"] == (5_000.0).hex()
    assert topology["basket_inhibition_current_pa_hex"] == (-5_000.0).hex()


def test_wta_electrical_controls_freeze_25_way_tiers_and_deferred_schedule() -> None:
    protocol = v4.PROTOCOL
    controls = protocol.wta_electrical_controls
    table = protocol.latency_table

    assert len(controls) == v4.WTA_ELECTRICAL_CONTROL_COUNT == 5
    assert tuple(control.name for control in controls) == (
        "empty-stop",
        "negative-one-clamps-to-empty-stop",
        "factor-stop-score-12-tie",
        "two-factor-score-13-collision",
        "unique-factor-score-13-accepted",
    )
    assert tuple(control.principal_first_spike_tier for control in controls) == (
        (24,),
        (24,),
        (0, 24),
        (0, 1),
        (0,),
    )
    assert tuple(control.simultaneous_spike_candidates for control in controls) == (
        1,
        1,
        2,
        2,
        1,
    )
    assert tuple(control.expected_decision for control in controls) == (
        "STOP",
        "STOP",
        "STOP",
        "STOP",
        "FACTOR",
    )
    assert tuple(control.expected_factor for control in controls) == (
        None,
        None,
        None,
        None,
        v4.FACTOR_REGISTRY[0],
    )

    current_rows = np.asarray(
        [control.principal_currents_pa for control in controls],
        dtype=np.float64,
    )
    assert current_rows.shape == (5, 25)
    for control in controls:
        assert len(control.principal_raw_scores) == 25
        assert len(control.principal_current_scores) == 25
        assert control.principal_raw_scores[-1] == v4.STOP_SCORE
        assert control.principal_current_scores == tuple(
            max(raw_score, 0) for raw_score in control.principal_raw_scores
        )
        assert control.principal_currents_pa == tuple(
            table[score].current_pa for score in control.principal_current_scores
        )
        assert control.full_circuit_execution_deferred is True

    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        current_rows,
        require_unique=False,
    )
    for control, race in zip(controls, races, strict=True):
        tier = tuple(
            cell_index
            for cell_index, score in enumerate(control.principal_current_scores)
            if table[score].first_spike_step == race.spike_step
        )
        assert race.spike_step == control.principal_first_spike_step
        assert race.simultaneous_spike_candidates == len(tier)
        assert tier == control.principal_first_spike_tier
        assert control.basket_inhibition_step == race.spike_step + 1

    empty, negative, factor_stop_tie, factor_collision, unique_factor = controls
    assert (empty.stop_relay_step, empty.expected_decision_step) == (557, 557)
    assert empty.collision_detection_step is None
    assert empty.factor_relay_steps == ()
    assert negative.principal_raw_scores[0] == -1
    assert negative.principal_current_scores[0] == 0
    assert negative.principal_currents_pa == empty.principal_currents_pa
    assert (negative.stop_relay_step, negative.expected_decision_step) == (557, 557)
    assert (
        factor_stop_tie.stop_relay_step,
        factor_stop_tie.collision_detection_step,
        factor_stop_tie.collision_stop_step,
        factor_stop_tie.factor_relay_steps,
        factor_stop_tie.expected_decision_step,
    ) == (None, 557, 558, (), 558)
    assert (
        factor_collision.stop_relay_step,
        factor_collision.collision_detection_step,
        factor_collision.collision_stop_step,
        factor_collision.factor_relay_steps,
        factor_collision.expected_decision_step,
    ) == (None, 556, 557, (), 557)
    assert (
        unique_factor.stop_relay_step,
        unique_factor.collision_detection_step,
        unique_factor.collision_stop_step,
        unique_factor.factor_relay_steps,
        unique_factor.expected_decision_step,
    ) == (None, None, None, ((0, 558),), 558)


def test_deep_interface_and_local_rule_are_metadata_only() -> None:
    operations = v4.INTERFACE_RECORD["operations"]
    assert len(operations) == 2
    assert tuple(operation["name"] for operation in operations) == (
        "build_frozen_exclusive_factor_bridge",
        "FrozenExclusiveFactorBridge.resolve",
    )
    assert v4.INTERFACE_RECORD["protocol_module_exposes_operations"] is False
    resolve_invariants = operations[1]["invariants"]
    assert any("no caller factor selection" in item for item in resolve_invariants)
    assert any("at most two physical factor events" in item for item in resolve_invariants)

    assert v4.LOCAL_RULE_RECORD["tick_dtype"] == "uint8"
    assert v4.LOCAL_RULE_RECORD["tick_cap"] == 4
    assert v4.LOCAL_RULE_RECORD["state_shape"] == (24, 1536)
    assert v4.LOCAL_RULE_RECORD["training_epochs"] == 4
    assert "heldout updates" in v4.LOCAL_RULE_RECORD["forbidden"]
    schedule = v4.LOCAL_RULE_RECORD["fit_schedule"]
    assert schedule["sort_key"] == (
        "canonical_tokens",
        "factor",
        "lesson_sha256",
    )
    assert schedule["sort_direction"] == "ascending"
    assert schedule["shuffle"] is False
    assert schedule["causal_order_source"] == "this lesson_sha256s tuple only"
    assert "never causal" in schedule["split_fit_ids_semantics"]
    assert schedule["epoch_count"] == 4
    assert schedule["heldout_scheduled"] is False
    ordered_fit = tuple(
        sorted(
            (
                lesson
                for lesson in v4.PROTOCOL.lessons
                if lesson.sha256 in set(v4.PROTOCOL.fit_lesson_sha256s)
            ),
            key=lambda lesson: (lesson.tokens, lesson.factor, lesson.sha256),
        )
    )
    expected_schedule_ids = tuple(lesson.sha256 for lesson in ordered_fit)
    assert schedule["lesson_sha256s"] == expected_schedule_ids
    assert len(set(expected_schedule_ids)) == v4.FIT_COUNT
    assert set(expected_schedule_ids) == set(v4.PROTOCOL.fit_lesson_sha256s)
    expected_local_rule = dict(v4._LOCAL_RULE_BASE_RECORD)
    expected_local_rule["fit_schedule"] = {
        "sort_key": schedule["sort_key"],
        "sort_direction": schedule["sort_direction"],
        "shuffle": schedule["shuffle"],
        "lesson_sha256s": schedule["lesson_sha256s"],
        "causal_order_source": schedule["causal_order_source"],
        "split_fit_ids_semantics": schedule["split_fit_ids_semantics"],
        "epoch_count": schedule["epoch_count"],
        "epoch_policy": schedule["epoch_policy"],
        "heldout_scheduled": schedule["heldout_scheduled"],
    }
    assert v4.PROTOCOL.local_rule_sha256 == v4._namespaced_sha256(
        v4._LOCAL_RULE_NAMESPACE,
        expected_local_rule,
    )

    forbidden_surfaces = {
        "build_frozen_exclusive_factor_bridge",
        "FrozenExclusiveFactorBridge",
        "fit",
        "resolve",
        "ExclusiveFactorResolution",
    }
    assert forbidden_surfaces.isdisjoint(vars(v4))
    public_functions = {
        name
        for name, value in inspect.getmembers(v4, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v4.__name__
    }
    assert public_functions == set()
    assert not any(isinstance(value, np.ndarray) for value in _walk(v4.PROTOCOL))


def test_protocol_roots_counts_and_no_run_flags_are_exact() -> None:
    protocol = v4.PROTOCOL
    actual_roots = {
        field.name: getattr(protocol, field.name)
        for field in dataclasses.fields(protocol)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert protocol.protocol_sha256 == v4._namespaced_sha256(
        v4._PROTOCOL_NAMESPACE,
        protocol.root_record(include_protocol_sha256=False),
    )
    assert all(
        len(root) == 64 and root == root.lower() and int(root, 16) >= 0
        for root in actual_roots.values()
    )

    assert (
        protocol.public_input_count,
        protocol.canonical_input_count,
        protocol.protected_input_count,
        protocol.learning_run_count,
        protocol.composition_result_count,
    ) == (0, 0, 0, 0, 0)
    root = protocol.root_record()
    assert root["learner_state_constructed"] is False
    assert root["composition_executed"] is False
    assert root["execution_surface_present"] is False
    assert root["learner_electrical_preflight_executed"] is False
    assert root["wta_electrical_control_count"] == 5


def test_protocol_and_nested_contract_records_are_immutable() -> None:
    protocol = v4.PROTOCOL
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "RUN"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.wta_electrical_controls[0].name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        v4.WTA_TOPOLOGY_RECORD["candidate_count"] = 24  # type: ignore[index]
    with pytest.raises(TypeError):
        v4.INTERFACE_RECORD["operations"][0]["name"] = "fit"  # type: ignore[index]
    with pytest.raises(TypeError):
        v4.LOCAL_RULE_RECORD["state_shape"] = (1, 1)  # type: ignore[index]
    with pytest.raises(TypeError):
        v4.LOCAL_RULE_RECORD["fit_schedule"]["shuffle"] = True  # type: ignore[index]
