from __future__ import annotations

from copy import copy
from dataclasses import replace
from hashlib import sha256
from inspect import signature

import pytest
from humanbrain.raw_source_motor_controller import (
    RawSourceMotorController,
    SuccessfulSourceMotorExperience,
)
from humanbrain.raw_source_motor_online_growth import (
    AppendOnlyRawSourceMotorStack,
    OnlineGrowthAblation,
    SourceMotorExecutionAuthority,
    SourceMotorExecutionReceipt,
    SourceMotorSuccessEvidence,
)

_BASE_CONTEXT = bytes.fromhex("c701")
_ONLINE_CONTEXT = bytes.fromhex("c702")
_PRE = bytes.fromhex("f701")
_RIGHT = bytes.fromhex("f702")
_LINK = bytes.fromhex("f703")
_SEPARATOR = bytes.fromhex("f704")
_MIDDLE = bytes.fromhex("f705")
_LEFT = bytes.fromhex("f706")
_POST = bytes.fromhex("f707")


def _tapes(seed: int) -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes, bytes]:
    left_value = bytes((seed, 0xA1))
    right_value = bytes((seed, 0xB1))
    before = (
        _PRE,
        _RIGHT,
        _LINK,
        right_value,
        _SEPARATOR,
        _MIDDLE,
        _LEFT,
        _LINK,
        left_value,
        _SEPARATOR,
        _POST,
    )
    after = (left_value, _SEPARATOR, right_value, _SEPARATOR)
    return before, after, left_value, right_value


def _fresh_tapes(
    seed: int,
) -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes, bytes]:
    before, after, left_value, right_value = _tapes(seed)
    shifted = (
        bytes((seed, 0xD1)),
        bytes((seed, 0xD2)),
        *before,
        bytes((seed, 0xD3)),
    )
    return shifted, after, left_value, right_value


def _base_controller() -> RawSourceMotorController:
    experiences = []
    for seed in (0xA1, 0xA2):
        before, after, _, _ = _tapes(seed)
        experiences.append(
            SuccessfulSourceMotorExperience(
                context_event=_BASE_CONTEXT,
                before_tokens=before,
                after_tokens=after,
                provenance_id=f"base-{seed}",
            )
        )
    return RawSourceMotorController.grow(tuple(experiences))


def _stack() -> AppendOnlyRawSourceMotorStack:
    return AppendOnlyRawSourceMotorStack.seed(_base_controller())


def _online_evidence(
    learner_checkpoint_sha256: str,
    *,
    emitted_after_tokens: tuple[bytes, ...] | None = None,
    provenance_id: str = "online-success-b1",
) -> SourceMotorSuccessEvidence:
    before, learned_after, _, _ = _tapes(0xB1)
    after = learned_after if emitted_after_tokens is None else emitted_after_tokens
    evaluator_calls: list[str] = []

    def evaluator(**query: object) -> bool:
        evaluator_calls.append(sha256(repr(query).encode("utf-8")).hexdigest())
        return (
            query["context_event"] == _ONLINE_CONTEXT
            and query["before_tokens"] == before
            and query["emitted_after_tokens"] == after
        )

    authority = SourceMotorExecutionAuthority(
        evaluator=evaluator,
        evaluator_sha256=sha256(b"frozen-evaluator").hexdigest(),
    )
    execution_receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=learner_checkpoint_sha256,
        context_event=_ONLINE_CONTEXT,
        before_tokens=before,
        emitted_after_tokens=after,
    )
    assert len(evaluator_calls) == 1
    return SourceMotorSuccessEvidence.seal(
        context_event=_ONLINE_CONTEXT,
        before_tokens=before,
        emitted_after_tokens=after,
        execution_receipt=execution_receipt,
        provenance_id=provenance_id,
    )


def test_append_only_successor_learns_later_recurrence_and_retains_parent() -> None:
    caller_controller = _base_controller()
    parent = AppendOnlyRawSourceMotorStack.seed(caller_controller)
    parent_root = parent.checkpoint_sha256
    parent_summary = parent.training_summary
    base_source, base_after, _, _ = _fresh_tapes(0xA3)
    online_source, online_after, left_value, right_value = _fresh_tapes(0xB2)

    base_before = parent.respond((_BASE_CONTEXT,), base_source)
    unknown_before = parent.respond((_ONLINE_CONTEXT,), online_source)
    evidence = _online_evidence(parent_root)
    transition = parent.extend_success(evidence)
    successor = transition.successor
    sibling = parent.extend_success(evidence).successor
    receipt = transition.receipt
    stored_base = parent._segments[0].adapter.controller

    assert stored_base is not caller_controller
    assert successor._segments[0] is parent._segments[0]
    assert sibling._segments[0] is parent._segments[0]
    assert stored_base._pending is None
    assert stored_base._tape._pending is None
    assert base_before.decision == "EMIT"
    assert base_before.emissions == base_after
    assert unknown_before.decision == "STOP"
    assert receipt.status == "APPENDED"
    assert receipt.parent_checkpoint_sha256 == parent_root
    assert receipt.successor_checkpoint_sha256 == successor.checkpoint_sha256
    assert receipt.successor_checkpoint_sha256 != parent_root
    assert receipt.reserved_cell_address_start == (parent_summary.next_reserved_cell_address)
    assert receipt.reserved_synapse_address_start == (parent_summary.next_reserved_synapse_address)
    assert receipt.reserved_cell_capacity_added == 4_377
    assert receipt.reserved_synapse_capacity_estimate_added == 38
    assert receipt.excitatory_support_tick_estimate_added == 68
    assert receipt.coactivation_support_tick_estimate_added == 38
    assert receipt.positive_success_support_tick_estimate_added == 30
    assert receipt.inhibitory_support_tick_estimate_added == 0
    assert receipt.source_experience_count == 1
    assert receipt.committed_evidence_tick_phases == 2
    assert receipt.parent_experience_replay_count == 0
    assert receipt.raw_training_sequences_retained == 0
    assert successor.training_summary.next_reserved_cell_address == (
        parent_summary.next_reserved_cell_address + receipt.reserved_cell_capacity_added
    )
    assert successor.training_summary.next_reserved_synapse_address == (
        parent_summary.next_reserved_synapse_address
        + receipt.reserved_synapse_capacity_estimate_added
    )

    learned = successor.respond((_ONLINE_CONTEXT,), online_source)
    base_after_growth = successor.respond((_BASE_CONTEXT,), base_source)
    collision = successor.respond(
        (_BASE_CONTEXT, _ONLINE_CONTEXT),
        online_source,
    )

    assert online_source.index(right_value) < online_source.index(left_value)
    assert learned.decision == "EMIT"
    assert learned.emissions == online_after
    assert learned.winning_segment_ordinal == 1
    assert learned.winning_selector_cell_id == receipt.reserved_cell_address_start
    assert learned.component_receipt_sha256 is not None
    assert base_after_growth.decision == "EMIT"
    assert base_after_growth.emissions == base_before.emissions
    assert base_after_growth.winning_segment_ordinal == 0
    assert base_after_growth.winning_selector_cell_id == (base_before.winning_selector_cell_id)
    assert collision.decision == "STOP"
    assert collision.emissions == ()
    assert collision.stop_reason == "collision"
    assert collision.race.simultaneous_spike_candidates == 2
    assert parent.checkpoint_sha256 == parent_root
    assert parent.respond((_BASE_CONTEXT,), base_source).emissions == base_after
    assert sibling.respond((_ONLINE_CONTEXT,), online_source).emissions == online_after
    assert sibling.respond((_BASE_CONTEXT,), base_source).emissions == base_after
    assert stored_base._pending is None
    assert stored_base._tape._pending is None
    assert successor.checkpoint_sha256 == sibling.checkpoint_sha256
    assert parent.checkpoint_sha256 == parent_root


def test_no_update_conflict_and_tamper_commit_no_structural_state() -> None:
    parent = _stack()
    evidence = _online_evidence(parent.checkpoint_sha256)
    no_update = parent.extend_success(
        evidence,
        ablation=OnlineGrowthAblation.NO_UPDATE,
    )
    online_source, _, _, _ = _fresh_tapes(0xB2)

    assert no_update.successor is parent
    assert no_update.receipt.status == "NO_UPDATE"
    assert no_update.receipt.successor_checkpoint_sha256 == parent.checkpoint_sha256
    assert no_update.receipt.successor_provenance_sha256 == (
        no_update.receipt.prior_provenance_sha256
    )
    assert no_update.receipt.reserved_cell_capacity_added == 0
    assert no_update.receipt.reserved_synapse_capacity_estimate_added == 0
    assert no_update.receipt.excitatory_support_tick_estimate_added == 0
    assert no_update.receipt.coactivation_support_tick_estimate_added == 0
    assert no_update.receipt.positive_success_support_tick_estimate_added == 0
    assert no_update.receipt.committed_evidence_tick_phases == 0
    assert no_update.successor.respond((_ONLINE_CONTEXT,), online_source).decision == "STOP"

    appended = parent.extend_success(evidence).successor
    with pytest.raises(ValueError, match="does not bind this parent learner"):
        appended.extend_success(evidence)
    duplicate = appended.extend_success(_online_evidence(appended.checkpoint_sha256))
    before, _, left_value, right_value = _tapes(0xB1)
    conflicting_after = (right_value, _SEPARATOR, left_value, _SEPARATOR)
    conflicting = _online_evidence(
        appended.checkpoint_sha256,
        emitted_after_tokens=conflicting_after,
        provenance_id="conflicting-online-success",
    )
    conflict = appended.extend_success(conflicting)

    assert duplicate.successor is appended
    assert duplicate.receipt.status == "DUPLICATE"
    assert duplicate.receipt.reserved_cell_capacity_added == 0
    assert conflict.successor is appended
    assert conflict.receipt.status == "CONFLICT_STOP"
    assert conflict.receipt.stop_reason == "incompatible_context_program"
    assert conflict.receipt.admission_race is not None
    assert conflict.receipt.admission_race.stopped
    assert conflict.receipt.admission_race.stop_reason == "collision"
    assert conflict.receipt.reserved_cell_capacity_added == 0
    assert conflict.receipt.reserved_synapse_capacity_estimate_added == 0
    assert appended.checkpoint_sha256 == conflict.receipt.successor_checkpoint_sha256

    tampered = replace(evidence, action_sha256="0" * 64)
    with pytest.raises(ValueError, match="action custody changed"):
        parent.extend_success(tampered)

    failed_receipt = copy(evidence.execution_receipt)
    object.__setattr__(failed_receipt, "outcome", "FAIL")
    with pytest.raises(ValueError, match="requires a PASS receipt"):
        parent.extend_success(replace(evidence, execution_receipt=failed_receipt))

    evaluator_calls = 0

    def failing_evaluator(**_query: object) -> bool:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return False

    failing_authority = SourceMotorExecutionAuthority(
        evaluator=failing_evaluator,
        evaluator_sha256=sha256(b"failing-frozen-evaluator").hexdigest(),
    )
    with pytest.raises(ValueError, match="did not PASS"):
        failing_authority.execute_and_issue(
            learner_checkpoint_sha256=parent.checkpoint_sha256,
            context_event=_ONLINE_CONTEXT,
            before_tokens=before,
            emitted_after_tokens=conflicting_after,
        )
    assert evaluator_calls == 1
    assert not hasattr(SourceMotorExecutionReceipt, "seal_pass")

    assert tuple(signature(AppendOnlyRawSourceMotorStack.respond).parameters) == (
        "self",
        "context_events",
        "source_tokens",
    )
