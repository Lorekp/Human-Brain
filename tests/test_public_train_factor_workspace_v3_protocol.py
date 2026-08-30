from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as v2
from experiments.r3_pilot import public_train_factor_workspace_v3_protocol as v3

EXPECTED_ROOTS = {
    "dependency_sha256": "e26b9c3ab42a9fad2e383ff345cbd64a19fb501c1729acc10287dc22087636e8",
    "curriculum_sha256": "356a7533ebdd0c0768160aa56da72dba95f6c7ec87c0960c8248acf01ffd6a47",
    "alias_collapse_sha256": "83f96f6d54e0a52e972862615553e65ba194c42883ead25e7370c8219686b6dd",
    "identifiability_sha256": "ab4f325688d91237e0c1ecad482f996347b47699952c0e82dbfef52fe6ed5cb9",
    "teacher_rule_sha256": "45f3f598104cf411aeb1f9f059e7c4d79c83cc2e7b54e8a6b1060c99abbb0c20",
    "surface_boundary_sha256": "ea50ef3fc0fa983ecd265b897af383b2375e78e7e51898dd89c3dafdc2a9d8a2",
    "protocol_sha256": "1722424f3cb540ed152d0035f916f85b9c7a8e9306242ecfe026b39b9685fdec",
}
EXPECTED_V2_PROTOCOL_SOURCE_SHA256 = (
    "1931f1748d0943a35be7ba348a5460ef010656cc1865fd6a92434e206b3cc4cd"
)
EXPECTED_V2_PROTOCOL_TEST_SHA256 = (
    "8ee79e0c5da6575b531b7c0ce1d33bcf6a89fa2a0380ce7c5dfe636ab3aed218"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_exact_canonical_42_inventory_and_24_18_split_are_retained() -> None:
    protocol = v3.PROTOCOL

    assert protocol.canonical_identities == v2.PROTOCOL.canonical_identities
    assert (
        protocol.canonical_fit_prompt_input_sha256s
        == v2.PROTOCOL.canonical_fit_prompt_input_sha256s
    )
    assert (
        protocol.canonical_heldout_prompt_input_sha256s
        == v2.PROTOCOL.canonical_heldout_prompt_input_sha256s
    )
    assert len(protocol.canonical_identities) == 42
    assert len(protocol.canonical_fit_prompt_input_sha256s) == 24
    assert len(protocol.canonical_heldout_prompt_input_sha256s) == 18
    assert sum(row.appearance_count for row in protocol.canonical_identities) == 90
    assert not (
        set(protocol.canonical_fit_prompt_input_sha256s)
        & set(protocol.canonical_heldout_prompt_input_sha256s)
    )
    assert v3.V2_PROTOCOL_COMMIT == "e04953f"
    assert v2.PROTOCOL.protocol_sha256 == v3.V2_PROTOCOL_SHA256

    v2_source_path = Path(v2.__file__).resolve()
    v2_test_path = (
        v2_source_path.parents[2] / "tests" / "test_public_train_factor_workspace_v2_protocol.py"
    )
    assert sha256(v2_source_path.read_bytes()).hexdigest() == (EXPECTED_V2_PROTOCOL_SOURCE_SHA256)
    assert sha256(v2_test_path.read_bytes()).hexdigest() == (EXPECTED_V2_PROTOCOL_TEST_SHA256)


def test_twelve_factor_set_classes_have_exact_four_two_balance_and_72_rows() -> None:
    protocol = v3.PROTOCOL
    lessons = protocol.lessons

    assert tuple(row.target_factors for row in protocol.factor_set_classes) == (
        v3.FACTOR_SET_CLASSES
    )
    assert len(protocol.factor_set_classes) == 12
    assert Counter(
        factor
        for factor_class in protocol.factor_set_classes
        for factor in factor_class.target_factors
    ) == Counter(v3.ACTIVE_FACTORS)
    assert len(lessons) == 72
    assert sum(row.split == v3.FIT for row in lessons) == 48
    assert sum(row.split == v3.HELDOUT for row in lessons) == 24

    balance = Counter((row.target_factors, row.split) for row in lessons)
    for factor_set in v3.FACTOR_SET_CLASSES:
        class_rows = tuple(row for row in lessons if row.target_factors == factor_set)
        assert balance[(factor_set, v3.FIT)] == 4
        assert balance[(factor_set, v3.HELDOUT)] == 2
        assert len(class_rows) == 6
        assert len({row.causal_input for row in class_rows}) == 6

    v2_lesson_sha256s = {
        lesson.sha256 for group in v2.PROTOCOL.counterfactual_groups for lesson in group.lessons
    }
    assert not ({row.sha256 for row in lessons} & v2_lesson_sha256s)


def test_stripped_causal_keys_form_a_total_function_to_exact_factor_sets() -> None:
    protocol = v3.PROTOCOL
    targets_by_key: dict[str, set[tuple[str, ...]]] = defaultdict(set)

    for lesson in protocol.lessons:
        causal_record = lesson.causal_input_record()
        assert tuple(causal_record) == (
            "direction",
            "cue_slot",
            "selected_slot",
            "slot_assembly_cells",
        )
        serialized = _canonical_json(causal_record).lower()
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
            )
        )
        assert (
            causal_record["slot_assembly_cells"][lesson.causal_input.cue_slot]
            == (v3.CUE_ASSEMBLY_CELLS[lesson.causal_input.direction])
        )
        targets_by_key[_canonical_json(causal_record)].add(lesson.target_factors)

    assert len(targets_by_key) == 72
    assert all(len(targets) == 1 for targets in targets_by_key.values())
    proof = protocol.identifiability_proof
    assert proof.lesson_count == 72
    assert proof.unique_causal_input_key_count == 72
    assert proof.keys_with_multiple_target_sets == 0
    assert proof.total_function is True
    assert proof.fit_inputs_per_class == 4
    assert proof.heldout_inputs_per_class == 2
    assert proof.excluded_noncausal_fields == (
        "factor_label",
        "profile_sha256",
        "lesson_id",
        "lesson_sha256",
        "input_sha256",
        "audit_hash",
        "split",
        "class_index",
        "template_index",
    )


def test_both_v2_minimum_maximum_aliases_are_collapsed_into_one_set_target() -> None:
    collapses = v3.PROTOCOL.v2_alias_collapses

    assert len(collapses) == 2
    assert {row.v2_lesson_indices for row in collapses} == {
        (1, 2, 30, 39, 59),
        (22, 31, 51, 67, 68),
    }
    assert all(row.v2_factor_labels == ("minimum", "maximum") for row in collapses)
    assert all(row.v3_target_factors == ("minimum", "maximum") for row in collapses)
    assert all(len(row.v3_lesson_indices) == 3 for row in collapses)

    minmax_class = next(
        row
        for row in v3.PROTOCOL.factor_set_classes
        if row.target_factors == ("minimum", "maximum")
    )
    assert set(minmax_class.physical_seed_assemblies) == {
        row.physical_assembly_cells for row in collapses
    }
    for collapse in collapses:
        v3_rows = tuple(v3.PROTOCOL.lessons[index] for index in collapse.v3_lesson_indices)
        assert all(row.target_factors == ("minimum", "maximum") for row in v3_rows)
        assert all(
            row.causal_input.selected_assembly_cells == collapse.physical_assembly_cells
            for row in v3_rows
        )
    assert v3.PROTOCOL.identifiability_proof.v2_alias_collapse_count == 2


def test_capacity_four_topology_surfaces_and_all_roots_are_frozen() -> None:
    protocol = v3.PROTOCOL

    assert protocol.masked_wta_controls == v2.PROTOCOL.masked_wta_controls
    assert protocol.compositor_controls == v2.PROTOCOL.compositor_controls
    assert v3.EVENT_CAPACITY == 4
    compositor_by_name = {row.name: row for row in protocol.compositor_controls}
    assert (
        compositor_by_name["same-span-different-factor-overlap"].expected_status
        == "READY_FACTOR_SET"
    )
    assert compositor_by_name["same-factor-duplicate"].expected_status == ("STOP_DUPLICATE_FACTOR")
    assert compositor_by_name["fifth-distinct-factor"].expected_status == ("STOP_EVENT_CAPACITY_4")
    assert v3.TEACHER_RULE_RECORD["different_factor_overlap"] == "legal"
    assert v3.TEACHER_RULE_RECORD["different_factor_nesting"] == "legal"
    assert v3.TEACHER_RULE_RECORD["host_union_or_filter"] is False

    assert all(
        value == 0
        for value in (
            v3.DEV_INPUT_COUNT,
            v3.PROTECTED_INPUT_COUNT,
            v3.DESIRED_OUTPUT_COUNT,
            v3.PROGRAM_ARTIFACT_COUNT,
            v3.PROPERTY_ROW_COUNT,
            v3.FAMILY_IDENTIFIER_COUNT,
            v3.TASK_IDENTIFIER_COUNT,
            v3.LEARNER_ARRAY_COUNT,
            v3.LEARNING_RUN_COUNT,
            v3.ELECTRICAL_EXECUTION_COUNT,
        )
    )
    assert v3.SURFACE_BOUNDARY_RECORD["data_only"] is True
    assert v3.SURFACE_BOUNDARY_RECORD["execution_surface_present"] is False
    assert v3.SURFACE_BOUNDARY_RECORD["learner_or_inference_interface_present"] is False
    assert dataclasses.is_dataclass(protocol)
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "PASS"  # type: ignore[misc]

    source = inspect.getsource(v3)
    assert "import numpy" not in source
    assert "teacher_gated_factor_workspace_v5 as" not in source
    assert "AdaptiveGlifPopulation" not in source

    root = protocol.root_record()
    assert {key: root[key] for key in EXPECTED_ROOTS} == EXPECTED_ROOTS
    assert protocol == v3._build_protocol()
    root_without_protocol = dict(root)
    root_without_protocol.pop("protocol_sha256")
    assert protocol.protocol_sha256 == v3._namespaced_sha256(
        v3._PROTOCOL_NAMESPACE,
        root_without_protocol,
    )
