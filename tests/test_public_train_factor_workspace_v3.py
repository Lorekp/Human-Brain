from __future__ import annotations

import dataclasses
import inspect
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v3 as learner
from experiments.r3_pilot import public_train_factor_workspace_v3_protocol as protocol_v3

EXPECTED_PROTOCOL_SOURCE_SHA256 = "30f9393169caa57313250cc9323fe064ae2f9b5a1ce1970df41f5177da2410e7"
EXPECTED_PROTOCOL_TEST_SHA256 = "c12062d8421a0390a67247ed4eed1a93c82de33570db91afa7d05ecfc6254661"
EXPECTED_PREFLIGHT_SHA256 = "53c6358e7a37c6847123f1f41def09d91d2dcfa71cff7e1eaf374fd1b6656924"
EXPECTED_CHECKPOINT_SHA256 = "fa44141c1dfa5c089b99cb56f467d048ab51022095d10451583519912852718b"
EXPECTED_FIT_REPLAY_SHA256 = "9d581cb8f005390ea148db747e352d3cc81a08c950419db0f516b294dddbe1c1"
EXPECTED_RESULT_SHA256 = "af73e68a8897611c7ae34003acbb58fe18f6d8184fa5eb9fe8f770b170d1612f"
EXPECTED_MISMATCH_CAUSAL_SOURCES = (
    "6ec8784c8728ad194e45b147ea5162579706d64d85296ed70c328b8405eb6fef",
    "d09efe147fa0945e471ef0aadb2fb270036c3b6372fd386c08ffbb628699684b",
    "3826ab0c0e08946b1eab60724a547810ef99ad0b8f518b1d291e8022604af669",
    "bf8b9bc2ce83e285151da27ad635c08f018e4090fa4f85c8208c40e64ed4790e",
    "735cadb43deb17b6b3e0ba435f0e7fee2b7790267df46be2d1c802f3392f2097",
)


def test_frozen_v3_protocol_and_v5_commit_are_authenticated_unchanged() -> None:
    protocol_path = Path(protocol_v3.__file__).resolve()
    protocol_test_path = (
        protocol_path.parents[2] / "tests" / "test_public_train_factor_workspace_v3_protocol.py"
    )

    assert sha256(protocol_path.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SOURCE_SHA256
    assert sha256(protocol_test_path.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_TEST_SHA256
    assert protocol_v3.PROTOCOL.protocol_sha256 == learner.V3_PROTOCOL_SHA256
    assert learner.V5_COMMIT == "5cb01cc"
    assert protocol_v3._build_protocol() == protocol_v3.PROTOCOL


def test_complete_controls_precede_one_fit_and_all_negative_arms_remain_negative() -> None:
    result = learner.RESULT

    assert result.electrical_preflight_sha256 == EXPECTED_PREFLIGHT_SHA256
    assert result.electrical_control_count == 68
    assert result.arrays_allocated_after_preflight is True
    assert result.fit_lesson_attempt_count == 48
    assert result.fit_pass_count == 1
    assert result.untrained_exact_count == 0
    assert result.untrained_stop_count == 72
    assert result.label_off_exact_count == 0
    assert result.label_off_stop_count == 72
    assert result.shuffled_true_exact_count == 0
    assert result.duplicate_control_attempt_count == 1
    assert result.duplicate_control_stop_count == 1
    assert result.fifth_factor_control_attempt_count == 1
    assert result.fifth_factor_control_stop_count == 1


def test_first_honest_fit_stops_atomically_before_untouched_heldout_replay() -> None:
    result = learner.RESULT
    fit = result.fit_replay

    assert result.status == learner.FIT_STOP_STATUS
    assert fit.attempt_count == 48
    assert fit.exact_count == 43
    assert fit.mismatch_count == 5
    assert fit.status_counts == (("COLLISION_STOP", 5), ("READY_FACTOR_SET", 43))
    assert fit.mismatch_causal_source_sha256s == EXPECTED_MISMATCH_CAUSAL_SOURCES
    assert fit.checkpoint_authentication_count == 2
    assert fit.sha256 == EXPECTED_FIT_REPLAY_SHA256
    assert result.heldout_replay is None
    assert result.heldout_attempt_count == 0

    mismatch_rows = tuple(
        lesson
        for lesson in protocol_v3.PROTOCOL.lessons
        if learner._causal_source_sha256(lesson.causal_input)
        in set(EXPECTED_MISMATCH_CAUSAL_SOURCES)
    )
    assert tuple(row.target_factors for row in mismatch_rows) == (
        ("minimum", "maximum"),
        ("minimum", "maximum"),
        ("minimum", "maximum"),
        ("minimum", "maximum"),
        ("add",),
    )
    assert tuple(row.split for row in mismatch_rows) == (protocol_v3.FIT,) * 5


def test_checkpoint_publication_and_forbidden_surface_are_closed() -> None:
    result = learner.RESULT

    assert result.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert result.result_sha256 == EXPECTED_RESULT_SHA256
    assert result.learned_state_bytes == 24_576
    assert (
        result.excitatory_update_events,
        result.inhibitory_update_events,
        result.committed_excitatory_ticks,
        result.committed_inhibitory_ticks,
        result.excitatory_max_ticks,
        result.inhibitory_max_ticks,
    ) == (33, 0, 396, 0, 2, 0)
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
        result.status = learner.PASS_STATUS  # type: ignore[misc]

    source = inspect.getsource(learner)
    assert "lesson.sha256" not in source
    assert "audit_" not in source
    assert "profile_sha" not in source
    assert ".class_index" not in source
    assert (
        learner._namespaced_sha256(
            learner._RESULT_NAMESPACE,
            result.record(include_sha256=False),
        )
        == result.result_sha256
    )
