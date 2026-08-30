from __future__ import annotations

import dataclasses
import inspect
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v2 as workspace_v2
from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as protocol_v2

EXPECTED_PROTOCOL_ROOT = "54aca3e9f9f7ae0bf1b1743510938522400305c64530db9b3f1467b241ee3171"
EXPECTED_RESULT_ROOT = "50a7efab1daf703df828b5f52fc727346533f00ff7e806a8033adff6e400188c"
EXPECTED_PROTOCOL_SOURCE_SHA256 = "1931f1748d0943a35be7ba348a5460ef010656cc1865fd6a92434e206b3cc4cd"
EXPECTED_PROTOCOL_TEST_SHA256 = "8ee79e0c5da6575b531b7c0ce1d33bcf6a89fa2a0380ce7c5dfe636ab3aed218"


def _lessons_by_index() -> dict[int, protocol_v2.CounterfactualFactorLesson]:
    return {
        lesson.lesson_index: lesson
        for group in protocol_v2.PROTOCOL.counterfactual_groups
        for lesson in group.lessons
    }


def test_exact_protocol_authentication_returns_the_preallocation_alias_stop() -> None:
    result = workspace_v2.RESULT

    assert workspace_v2.run_public_train_factor_workspace_v2_preflight() == result
    assert result.status == workspace_v2.STOP_INPUT_ALIAS
    assert result.protocol_sha256 == EXPECTED_PROTOCOL_ROOT
    assert result.protocol_authentication_count == 2
    assert result.counterfactual_lesson_scan_count == 78
    assert result.ambiguous_physical_input_count == 2
    assert result.ambiguous_lesson_count == 10
    assert result.ambiguous_fit_lesson_count == 7
    assert result.ambiguous_heldout_lesson_count == 3
    assert result.sha256 == EXPECTED_RESULT_ROOT
    assert {alias.audit_lesson_indices for alias in result.aliases} == {
        (1, 2, 30, 39, 59),
        (22, 31, 51, 67, 68),
    }
    assert all(alias.audit_factor_labels == ("minimum", "maximum") for alias in result.aliases)


def test_erasing_labels_and_hashes_exposes_identical_minimum_maximum_inputs() -> None:
    lessons = _lessons_by_index()

    for minimum_index, maximum_index in ((1, 2), (22, 31)):
        minimum = lessons[minimum_index]
        maximum = lessons[maximum_index]
        assert minimum.factor == "minimum"
        assert maximum.factor == "maximum"
        assert minimum.sha256 != maximum.sha256
        assert minimum.target_span_profile_sha256 != maximum.target_span_profile_sha256
        assert minimum.input_slots != maximum.input_slots

        minimum_physical = workspace_v2.branch_local_physical_input_key(minimum)
        maximum_physical = workspace_v2.branch_local_physical_input_key(maximum)
        assert minimum_physical == maximum_physical
        assert minimum_physical.record() == maximum_physical.record()
        causal_record_keys = tuple(minimum_physical.record())
        assert causal_record_keys == (
            "causal_stage",
            "selected_assembly_cells",
            "selected_active_cell_count",
        )
        assert not any(
            fragment in key
            for key in causal_record_keys
            for fragment in ("factor", "label", "hash", "sha", "profile")
        )

    fit_lessons = tuple(lesson for lesson in lessons.values() if lesson.split == protocol_v2.FIT)
    fit_labels_by_physical_key: dict[
        workspace_v2.BranchLocalPhysicalInputKey,
        set[str],
    ] = {}
    for lesson in fit_lessons:
        physical_key = workspace_v2.branch_local_physical_input_key(lesson)
        fit_labels_by_physical_key.setdefault(physical_key, set()).add(lesson.factor)
    assert sum(len(labels) > 1 for labels in fit_labels_by_physical_key.values()) == 2


def test_protocol_failure_precedes_alias_scan_and_every_downstream_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding_scan(_protocol: object) -> object:
        raise AssertionError("alias scan ran before protocol authentication")

    monkeypatch.setattr(workspace_v2, "_scan_physical_input_aliases", exploding_scan)
    tampered = replace(protocol_v2.PROTOCOL, protocol_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="protocol authentication failed"):
        workspace_v2.run_public_train_factor_workspace_v2_preflight(tampered)

    result = workspace_v2.RESULT
    attempt_fields = (
        "electrical_control_attempt_count",
        "learner_array_allocation_count",
        "learning_run_count",
        "counterfactual_fit_lesson_attempt_count",
        "canonical_fit_identity_attempt_count",
        "teacher_off_control_attempt_count",
        "cell_off_control_attempt_count",
        "factor_derangement_control_attempt_count",
        "counterfactual_replay_attempt_count",
        "canonical_fit_gate_attempt_count",
        "canonical_heldout_gate_attempt_count",
        "deployment_rebuild_attempt_count",
    )
    assert result.downstream_attempt_count == 0
    assert all(getattr(result, field) == 0 for field in attempt_fields)
    assert result.dev_input_count == 0
    assert result.protected_input_count == 0
    assert result.desired_output_read_count == 0
    assert result.program_artifact_read_count == 0
    assert result.property_row_read_count == 0
    assert result.family_identifier_read_count == 0


def test_stop_is_immutable_data_only_and_preserves_the_protocol_bytes() -> None:
    assert dataclasses.is_dataclass(workspace_v2.RESULT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        workspace_v2.RESULT.status = "PASS"  # type: ignore[misc]

    source = inspect.getsource(workspace_v2)
    assert "import numpy" not in source
    assert "teacher_gated_factor_workspace_v5 as" not in source
    assert "AdaptiveGlifPopulation" not in source
    assert "CompiledRestingGlifRace" not in source

    protocol_source = Path(protocol_v2.__file__).read_bytes()
    protocol_test = (
        Path(protocol_v2.__file__).resolve().parents[2]
        / "tests"
        / "test_public_train_factor_workspace_v2_protocol.py"
    ).read_bytes()
    assert sha256(protocol_source).hexdigest() == EXPECTED_PROTOCOL_SOURCE_SHA256
    assert sha256(protocol_test).hexdigest() == EXPECTED_PROTOCOL_TEST_SHA256
    assert protocol_v2.PROTOCOL.protocol_sha256 == EXPECTED_PROTOCOL_ROOT
