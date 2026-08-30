from __future__ import annotations

import dataclasses
import inspect
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from experiments.r3_pilot import public_train_factor_workspace_v4 as learner
from experiments.r3_pilot import public_train_factor_workspace_v4_protocol as protocol_v4

EXPECTED_PREFLIGHT_SHA256 = "e80fffa6cecdd56ff4b44dc885fb89027b754c8dd9860b08f44e6a4e4abbe55c"
EXPECTED_CHECKPOINT_SHA256 = "63fd83a72f23aeb759bd72d032bb9b7c44adfcdead8557992602279b99847578"
EXPECTED_FIT_REPLAY_SHA256 = "6599ac4269fcd981ceb5ee79d00cac7b0fb0d0b8398f69ed56dab82421da1250"
EXPECTED_HELDOUT_REPLAY_SHA256 = "8a9f2c5f59a3df75be434e004adbd6af7497f9e51d816b3b017a5eefbe4b0ca0"
EXPECTED_CELL_OFF_SHA256 = "f4e4557780c2f7c10dd1eb18e79f562b77d7e1fe4fc3cbde0a2ad82d6bc6d4c9"
EXPECTED_SHUFFLED_SHA256 = "384dba86657373f8699d27cff260caf01ee21d0a60a17d603912f2143caaaa07"
EXPECTED_RESULT_SHA256 = "96ab4c088382271e2208bbc71d8f308497cd5a560d601529e124938dda73cc4e"


def test_frozen_v4_dependency_and_all_13_controls_precede_state_allocation() -> None:
    protocol_path = Path(protocol_v4.__file__).resolve()
    protocol_test_path = (
        protocol_path.parents[2] / "tests" / "test_public_train_factor_workspace_v4_protocol.py"
    )
    preflight = learner.ELECTRICAL_PREFLIGHT

    assert learner.V4_PROTOCOL_COMMIT == "c9da529"
    assert sha256(protocol_path.read_bytes()).hexdigest() == learner.V4_PROTOCOL_SOURCE_SHA256
    assert sha256(protocol_test_path.read_bytes()).hexdigest() == learner.V4_PROTOCOL_TEST_SHA256
    assert protocol_v4.PROTOCOL.protocol_sha256 == learner.V4_PROTOCOL_SHA256
    assert protocol_v4._build_protocol() == protocol_v4.PROTOCOL
    assert preflight.sha256 == EXPECTED_PREFLIGHT_SHA256
    assert preflight.control_count == preflight.pass_count == 13
    assert preflight.arrays_allocated_before_completion == 0
    assert tuple(row.protocol_control_sha256 for row in preflight.controls) == tuple(
        row.sha256 for row in protocol_v4.PROTOCOL.future_controls
    )
    assert tuple(
        (row.status, row.factor_membership_count, row.exact) for row in preflight.controls
    ) == tuple(
        (
            row.expected_status,
            row.expected_factor_membership_count,
            row.expected_exact,
        )
        for row in protocol_v4.PROTOCOL.future_controls
    )
    assert preflight.sha256 == learner._namespaced_sha256(
        learner._PREFLIGHT_NAMESPACE,
        preflight.record(include_sha256=False),
    )


def test_one_local_fit_pass_freezes_two_arrays_and_replays_fit_and_heldout_exactly() -> None:
    result = learner.RESULT

    assert result.status == learner.PASS_STATUS
    assert result.arrays_allocated_after_preflight is True
    assert result.state_array_allocation_count == 2
    assert result.learned_state_bytes == 270_336
    assert result.fit_lesson_attempt_count == 48
    assert result.fit_pass_count == 1
    assert result.fit_replay.attempt_count == result.fit_replay.exact_count == 48
    assert result.fit_replay.mismatch_count == 0
    assert result.fit_replay.status_counts == (("READY_FACTOR_SET", 48),)
    assert result.fit_replay.checkpoint_authentication_count == 2
    assert result.fit_replay.sha256 == EXPECTED_FIT_REPLAY_SHA256
    assert result.heldout_replay is not None
    assert result.heldout_attempt_count == 24
    assert result.heldout_replay.attempt_count == result.heldout_replay.exact_count == 24
    assert result.heldout_replay.mismatch_count == 0
    assert result.heldout_replay.status_counts == (("READY_FACTOR_SET", 24),)
    assert result.heldout_replay.checkpoint_authentication_count == 2
    assert result.heldout_replay.sha256 == EXPECTED_HELDOUT_REPLAY_SHA256
    assert result.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert (
        result.excitatory_update_events,
        result.inhibitory_update_events,
        result.committed_excitatory_ticks,
        result.committed_inhibitory_ticks,
        result.excitatory_max_ticks,
        result.inhibitory_max_ticks,
    ) == (52, 1_100, 2_496, 45_632, 4, 4)


def test_teacher_cell_and_deranged_controls_remain_negative_after_fit_readiness() -> None:
    result = learner.RESULT

    assert result.teacher_off_attempt_count == 24
    assert result.teacher_off_excitatory_update_events == 0
    assert result.teacher_off_inhibitory_update_events == 0
    assert result.cell_off_replay is not None
    assert result.cell_off_replay.attempt_count == 24
    assert result.cell_off_replay.exact_count == 0
    assert result.cell_off_replay.status_counts == (("STOP_ZERO", 24),)
    assert result.cell_off_replay.sha256 == EXPECTED_CELL_OFF_SHA256
    assert result.shuffled_replay is not None
    assert result.shuffled_replay.attempt_count == 24
    assert result.shuffled_replay.exact_count == 0
    assert result.shuffled_replay.status_counts == (("STOP_FACTOR_SET_MISMATCH", 24),)
    assert result.shuffled_replay.sha256 == EXPECTED_SHUFFLED_SHA256
    assert result.duplicate_control_stop_count == 1
    assert result.fifth_factor_control_stop_count == 1
    mapping = learner._teacher_derangement()
    assert all(mapping[factor] != factor for factor in learner.ACTIVE_FACTORS)
    assert all(
        frozenset(mapping[factor] for factor in target).isdisjoint(target)
        for target in protocol_v4.FACTOR_SET_CLASSES
    )


def test_role_coincidence_and_unordered_physical_membership_are_authoritative() -> None:
    left_class, right_class = protocol_v4.PROTOCOL.binding_classes[:2]
    template = protocol_v4.PROTOCOL.templates[0]
    left_input = protocol_v4._causal_input(left_class, template)
    right_input = protocol_v4._causal_input(right_class, template)
    left, left_publication = learner._physical_representation(left_input)
    right, right_publication = learner._physical_representation(right_input)
    left_content, _ = learner._physical_representation(
        left_input,
        suppress_all_conjunctions=True,
    )
    right_content, _ = learner._physical_representation(
        right_input,
        suppress_all_conjunctions=True,
    )
    left_direction_off, _ = learner._physical_representation(
        left_input,
        suppress_directional_geometry=True,
    )
    right_direction_off, _ = learner._physical_representation(
        right_input,
        suppress_directional_geometry=True,
    )

    assert np.count_nonzero(left) == np.count_nonzero(right) == 48
    assert left_publication.spike_cells == left_input.representation_spike_cells
    assert right_publication.spike_cells == right_input.representation_spike_cells
    assert not np.array_equal(left, right)
    assert np.array_equal(left_content, right_content)
    assert np.array_equal(left_direction_off, right_direction_off)

    raw = [0] * learner.FACTOR_COUNT
    raw[:2] = (protocol_v4.MEMBERSHIP_SCORE_THRESHOLD,) * 2
    decision = learner._control_membership(
        tuple(raw),
        tuple(reversed(learner.FACTOR_REGISTRY[:2])),
    )
    assert decision.status == "READY_FACTOR_SET"
    assert decision.candidate_exact is True
    assert decision.membership_count == 2
    assert decision.physical_occupancy[:2] == (True, True)
    assert not any(decision.physical_occupancy[2:])
    assert decision.stop_spike is False


def test_checkpoint_result_and_forbidden_surface_are_immutable_and_closed() -> None:
    result = learner.RESULT

    assert result.result_sha256 == EXPECTED_RESULT_SHA256
    assert result.forbidden_read_count == 0
    assert all(
        value == 0
        for value in (
            learner.DEV_INPUT_COUNT,
            learner.PROTECTED_INPUT_COUNT,
            learner.DESIRED_OUTPUT_COUNT,
            learner.PROGRAM_ARTIFACT_COUNT,
            learner.PROPERTY_ROW_COUNT,
            learner.FAMILY_IDENTIFIER_COUNT,
            learner.TASK_IDENTIFIER_COUNT,
            learner.FORBIDDEN_READ_COUNT,
        )
    )
    assert not hasattr(result, "excitatory_ticks")
    assert not hasattr(result, "inhibitory_ticks")
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = learner.FIT_STOP_STATUS  # type: ignore[misc]

    source = inspect.getsource(learner)
    representation_source = inspect.getsource(learner._physical_representation)
    assert "lesson.sha256" not in source
    assert ".class_index" not in source
    assert "audit_" not in source
    assert "profile_sha" not in source
    assert "public_train_factor_workspace_v3 as" not in source
    assert "ordered_factor_relays" not in source
    assert "target_factors" not in representation_source
    assert result.result_sha256 == learner._namespaced_sha256(
        learner._RESULT_NAMESPACE,
        result.record(include_sha256=False),
    )
