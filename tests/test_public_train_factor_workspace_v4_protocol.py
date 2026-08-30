from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v3_protocol as v3
from experiments.r3_pilot import public_train_factor_workspace_v4_protocol as v4

EXPECTED_ROOTS = {
    "dependency_sha256": "fda76ef19c429a93e8cf0ef6f910466fe8a7bf6312a89edec3c8bfed3feab753",
    "localization_sha256": "1788ac1fa483544e648f2ed724aabb40f4447edadfddaf0ae927421dd5b64f6b",
    "cohort_sha256": "ddcca1ef73611bbc3db2e6f2e018a43b11a4c30affaafb8d9099aae2f7eaaaf3",
    "identifiability_sha256": "6c7a6807c9cefeedb55f37535a2ea594b0bc29d5d53a1655264d5b4970ce53e8",
    "role_binding_topology_sha256": (
        "ecccad6fa14801f462c852d9efcb26e3e209d2f1193e8416390fbafd1806a6df"
    ),
    "membership_topology_sha256": (
        "af184ecd2ec3b4b97593fd3fa753d07ec2a53bc1e09f2abf51978e44d3a2f5c3"
    ),
    "local_rule_sha256": "a4d9803878b9e777d49ff1fafde507961cce8324bf160fcfbfd1a63335843e65",
    "controls_sha256": "ba96b380da55588aa0ab4d2961c51d20c1cc40ca7e3605209e9fb0e3cb917609",
    "surface_boundary_sha256": "69760eb8790a2190a1287245f1d17c1fafea6e354da02bb3e391c6d8c6c56f72",
    "protocol_sha256": "83335e57fb896e9ba509a3f7f34e1bba39e7394d700c80813a702c4b34cc0f12",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_v3_fit_only_collision_localization_is_exact_and_heldout_remains_unread() -> None:
    localization = v4.PROTOCOL.v3_localization
    v3_protocol_path = Path(v3.__file__).resolve()
    v3_learner_path = v3_protocol_path.with_name("public_train_factor_workspace_v3.py")
    v3_test_path = (
        v3_protocol_path.parents[2] / "tests" / "test_public_train_factor_workspace_v3.py"
    )

    assert sha256(v3_protocol_path.read_bytes()).hexdigest() == v4.V3_PROTOCOL_SOURCE_SHA256
    assert sha256(v3_learner_path.read_bytes()).hexdigest() == v4.V3_LEARNER_SOURCE_SHA256
    assert sha256(v3_test_path.read_bytes()).hexdigest() == v4.V3_LEARNER_TEST_SHA256
    assert v4.V3_PROTOCOL_COMMIT == "398818f"
    assert v4.V3_LEARNER_COMMIT == "640d3d7"
    assert localization.status == "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_FIT_READINESS"
    assert (
        localization.fit_attempt_count,
        localization.fit_exact_count,
        localization.fit_collision_count,
        localization.heldout_treatment_attempt_count,
    ) == (48, 43, 5, 0)
    assert (
        localization.multilabel_equal_score_collision_count,
        localization.once_seen_stop_tie_count,
    ) == (4, 1)
    assert (
        localization.committed_excitatory_ticks,
        localization.committed_inhibitory_ticks,
        localization.excitatory_max_ticks,
        localization.inhibitory_max_ticks,
    ) == (396, 0, 2, 0)
    assert localization.causal_key_geometry_discarded_by_scorer is True
    assert localization.add_seed_overlap_counts == ((12, 0, 1), (0, 12, 0), (1, 0, 12))
    assert v4.HELDOUT_TREATMENT_RESULT_COUNT == 0


def test_fresh_72_lesson_cohort_is_balanced_paired_and_disjoint_from_v3() -> None:
    protocol = v4.PROTOCOL
    lessons = protocol.lessons

    assert len(protocol.binding_classes) == 12
    assert len(protocol.templates) == 6
    assert len(lessons) == 72
    assert sum(row.split == v4.FIT for row in lessons) == 48
    assert sum(row.split == v4.HELDOUT for row in lessons) == 24
    balance = Counter((row.target_factors, row.split) for row in lessons)
    assert all(
        balance[(target, v4.FIT)] == 4 and balance[(target, v4.HELDOUT)] == 2
        for target in v4.FACTOR_SET_CLASSES
    )

    for pair_index in range(v4.CARRIER_PAIR_COUNT):
        pair = protocol.binding_classes[pair_index * 2 : pair_index * 2 + 2]
        assert pair[0].carrier_assembly_cells == pair[1].carrier_assembly_cells
        assert (pair[0].direction, pair[1].direction) == ("NEXT", "PREV")
        assert pair[0].target_factors != pair[1].target_factors

    proof = protocol.identifiability_proof
    assert proof.unique_full_input_count == 72
    assert proof.fresh_v3_full_input_overlap_count == 0
    assert proof.fresh_v3_lesson_sha_overlap_count == 0


def test_role_bound_projection_is_the_smallest_total_causal_view() -> None:
    protocol = v4.PROTOCOL
    proof = protocol.identifiability_proof

    assert proof.unique_role_bound_projection_count == 48
    assert proof.full_keys_with_multiple_target_sets == 0
    assert proof.role_bound_keys_with_multiple_target_sets == 0
    assert proof.content_only_keys_with_multiple_target_sets == 6
    assert proof.direction_only_keys_with_multiple_target_sets == 2
    assert proof.min_heldout_supported_representation_cells == 24
    assert proof.total_function is True

    for lesson in protocol.lessons:
        causal = lesson.causal_input.record()
        serialized = _canonical_json(causal).lower()
        assert not any(
            fragment in serialized
            for fragment in (
                "factor",
                "profile",
                "sha256",
                "lesson",
                "audit",
                "split",
                "class_index",
                "template_index",
                "pair_index",
            )
        )
        assert len(lesson.causal_input.selector_role_cells) == 3
        assert len(lesson.causal_input.representation_spike_cells) == 48
        assert len(set(lesson.causal_input.representation_spike_cells)) == 48
        assert min(lesson.causal_input.representation_spike_cells) >= 0
        assert max(lesson.causal_input.representation_spike_cells) < 5_632


def test_unordered_membership_topology_and_falsifiers_are_frozen_without_cap_growth() -> None:
    membership = v4.MEMBERSHIP_TOPOLOGY_RECORD
    role_binding = v4.ROLE_BINDING_TOPOLOGY_RECORD
    local_rule = v4.LOCAL_RULE_RECORD
    controls = {row.name: row for row in v4.PROTOCOL.future_controls}

    assert v4.EVENT_CAPACITY == 4
    assert v4.MEMBERSHIP_SCORE_THRESHOLD == 12
    assert membership["simultaneous_distinct_factor_relays_are_legal"] is True
    assert membership["factor_factor_collision_poison"] is False
    assert membership["factor_stop_equal_score_collision"] is False
    assert membership["factor_set_permutation_invariant"] is True
    assert membership["ordered_factor_role_present"] is False
    assert membership["host_union_or_filter"] is False
    assert role_binding["projection_uses_factor_label"] is False
    assert role_binding["projection_uses_hash_or_audit_identity"] is False
    assert role_binding["projection_uses_lesson_or_class_ordinal"] is False
    assert local_rule["tick_cap"] == 4
    assert local_rule["state_shape"] == (24, 5_632)
    assert local_rule["state_bytes"] == 270_336
    assert controls["once-exposed-membership"].expected_status == "READY_FACTOR_SET"
    assert controls["two-membership-unordered-set"].expected_factor_membership_count == 2
    assert controls["role-binding-off"].expected_status == "STOP_CONTENT_ONLY_ALIAS"
    assert controls["direction-role-off"].expected_status == "STOP_ROLE_CAUSAL_ALIAS"
    assert controls["fifth-membership-poison"].expected_status == "STOP_EVENT_CAPACITY_4"
    assert controls["duplicate-relay-poison"].expected_status == "STOP_DUPLICATE_FACTOR"
    assert controls["factor-set-permutation"].expected_exact is True
    membership_json = v4._canonical_json(membership).decode("utf-8")
    assert "minimum" not in membership_json
    assert "maximum" not in membership_json


def test_protocol_is_data_only_surface_closed_immutable_and_rooted() -> None:
    protocol = v4.PROTOCOL

    assert protocol.canonical_identities == v3.PROTOCOL.canonical_identities
    assert protocol.canonical_fit_prompt_input_sha256s == (
        v3.PROTOCOL.canonical_fit_prompt_input_sha256s
    )
    assert protocol.canonical_heldout_prompt_input_sha256s == (
        v3.PROTOCOL.canonical_heldout_prompt_input_sha256s
    )
    assert all(
        value == 0
        for value in (
            v4.DEV_INPUT_COUNT,
            v4.PROTECTED_INPUT_COUNT,
            v4.DESIRED_OUTPUT_COUNT,
            v4.PROGRAM_ARTIFACT_COUNT,
            v4.PROPERTY_ROW_COUNT,
            v4.FAMILY_IDENTIFIER_COUNT,
            v4.TASK_IDENTIFIER_COUNT,
            v4.LEARNER_ARRAY_COUNT,
            v4.LEARNING_RUN_COUNT,
            v4.ELECTRICAL_EXECUTION_COUNT,
            v4.HELDOUT_TREATMENT_RESULT_COUNT,
        )
    )
    assert v4.SURFACE_BOUNDARY_RECORD["data_only"] is True
    assert v4.SURFACE_BOUNDARY_RECORD["execution_surface_present"] is False
    assert dataclasses.is_dataclass(protocol)
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "PASS"  # type: ignore[misc]

    source = inspect.getsource(v4)
    assert "import numpy" not in source
    assert "public_train_factor_workspace_v3 as" not in source
    assert "teacher_gated_factor_workspace_v5 as" not in source
    assert "AdaptiveGlifPopulation" not in source

    root = protocol.root_record()
    assert {key: root[key] for key in EXPECTED_ROOTS} == EXPECTED_ROOTS
    assert v4._build_protocol() == protocol
    root_without_protocol = dict(root)
    root_without_protocol.pop("protocol_sha256")
    assert protocol.protocol_sha256 == v4._namespaced_sha256(
        v4._PROTOCOL_NAMESPACE,
        root_without_protocol,
    )
