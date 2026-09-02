from __future__ import annotations

from copy import copy
from dataclasses import replace
from hashlib import sha256
from inspect import signature

import pytest
from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorAblation,
    BoundaryMultiscaleSourceMotorController,
    SuccessfulBoundaryMultiscaleSourceMotorExperience,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.multiscale_source_motor_online_growth import (
    AppendOnlyMultiscaleSourceMotorStack,
    MultiscaleSourceMotorExecutionAuthority,
    MultiscaleSourceMotorExecutionReceipt,
    MultiscaleSourceMotorSuccessEvidence,
    OnlineMultiscaleGrowthAblation,
)


def _buggy_source(function_name: str) -> bytes:
    return (
        f"def {function_name}():\n"
        '    return "ready"\n'
        f'print({function_name}(), end="")\n'
        'raise RuntimeError("obsolete")\n'
    ).encode()


def _correct_source(function_name: str) -> bytes:
    return (
        f'def {function_name}():\n    return "ready"\nprint({function_name}(), end="")\n'
    ).encode()


def _short_source(function_name: str) -> bytes:
    return f'def {function_name}():\n    return "ready"\n'.encode()


def _workspace(
    rows: tuple[tuple[str, str], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, _buggy_source(function_name)) for path, function_name in rows
    )


def _experience(
    *,
    target_path: str,
    rows: tuple[tuple[str, str], ...],
    provenance_id: str,
    short: bool = False,
) -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    before = _workspace(rows)
    function_by_path = dict(rows)
    replacement = _short_source if short else _correct_source
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            (
                replacement(function_by_path[item.path])
                if item.path == target_path
                else item.content
            ),
        )
        for item in before
    )
    query_identity = target_path.rsplit("/", 1)[-1].removesuffix(".py")
    return SuccessfulBoundaryMultiscaleSourceMotorExperience(
        query=f"repair {query_identity} using existing bytes".encode(),
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def _base_controller() -> BoundaryMultiscaleSourceMotorController:
    # The base learns a query-stem -> flat-path role. The appended shard learns
    # the same reusable relation through one stable nested-path envelope, a new
    # support region rather than an absolute workspace-position patch.
    first = _experience(
        target_path="mira_worker.py",
        rows=(
            ("atlas_worker.py", "atlas_cycle"),
            ("mira_worker.py", "mira_cycle"),
            ("zenith_worker.py", "zenith_cycle"),
        ),
        provenance_id="base-mira",
    )
    second = _experience(
        target_path="tango_worker.py",
        rows=(
            ("ember_worker.py", "ember_cycle"),
            ("tango_worker.py", "tango_cycle"),
            ("willow_worker.py", "willow_cycle"),
        ),
        provenance_id="base-tango",
    )
    return BoundaryMultiscaleSourceMotorController.grow((first, second))


def _online_experience(
    *,
    short: bool = False,
) -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    return _experience(
        target_path="pkg/cinder_worker.py",
        rows=(
            ("pkg/cinder_worker.py", "cinder_cycle"),
            ("pkg/opal_worker.py", "opal_cycle"),
            ("pkg/sierra_worker.py", "sierra_cycle"),
        ),
        provenance_id="online-cinder-short" if short else "online-cinder",
        short=short,
    )


def _evidence(
    learner_checkpoint_sha256: str,
    *,
    experience: SuccessfulBoundaryMultiscaleSourceMotorExperience | None = None,
) -> MultiscaleSourceMotorSuccessEvidence:
    row = _online_experience() if experience is None else experience
    evaluator_calls: list[tuple[bytes, int, int]] = []

    def evaluator(**observation: object) -> bool:
        query = observation["query"]
        before = observation["before_files"]
        after = observation["after_files"]
        assert isinstance(query, bytes)
        assert isinstance(before, tuple)
        assert isinstance(after, tuple)
        evaluator_calls.append((query, len(before), len(after)))
        return query == row.query and before == row.before_files and after == row.after_files

    authority = MultiscaleSourceMotorExecutionAuthority(
        evaluator=evaluator,
        evaluator_sha256=sha256(b"frozen-multiscale-evaluator").hexdigest(),
    )
    receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=learner_checkpoint_sha256,
        query=row.query,
        before_files=row.before_files,
        after_files=row.after_files,
    )
    assert evaluator_calls == [(row.query, 3, 3)]
    return MultiscaleSourceMotorSuccessEvidence.seal(
        query=row.query,
        before_files=row.before_files,
        after_files=row.after_files,
        execution_receipt=receipt,
        provenance_id=row.provenance_id,
    )


def _fresh_first_file_experience() -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    return _experience(
        target_path="pkg/lumen_worker.py",
        rows=(
            ("pkg/lumen_worker.py", "lumen_cycle"),
            ("pkg/quartz_worker.py", "quartz_cycle"),
            ("pkg/vireo_worker.py", "vireo_cycle"),
        ),
        provenance_id="fresh-lumen",
    )


def _fresh_middle_file_experience() -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    return _experience(
        target_path="nova_worker.py",
        rows=(
            ("delta_worker.py", "delta_cycle"),
            ("nova_worker.py", "nova_cycle"),
            ("umber_worker.py", "umber_cycle"),
        ),
        provenance_id="fresh-nova",
    )


def _materialized(
    decision_emissions: tuple[bytes, ...],
    row: SuccessfulBoundaryMultiscaleSourceMotorExperience,
) -> tuple[OpaqueWorkspaceFile, ...]:
    observation = MultiscaleOpaqueEventTape.observe(row.query, row.before_files)
    return observation.materialize_motor_emission(decision_emissions)


def test_append_retains_base_and_adds_replay_free_full_workspace_support() -> None:
    caller_controller = _base_controller()
    parent = AppendOnlyMultiscaleSourceMotorStack.seed(caller_controller)
    parent_root = parent.checkpoint_sha256
    parent_summary = parent.training_summary
    base_fresh = _fresh_middle_file_experience()
    online_training = _online_experience()
    online_fresh = _fresh_first_file_experience()

    base_before = parent.respond(base_fresh.query, base_fresh.before_files)
    online_before = parent.respond(online_training.query, online_training.before_files)
    evidence = _evidence(parent_root)
    transition = parent.extend_success(evidence)
    successor = transition.successor
    receipt = transition.receipt
    stored_base = parent._segments[0].adapter.controller

    assert stored_base is not caller_controller
    assert stored_base._action_tape._pending is None
    assert successor._segments[0] is parent._segments[0]
    assert base_before.decision == "EMIT"
    assert _materialized(base_before.emissions, base_fresh) == base_fresh.after_files
    assert online_before.decision == "STOP"
    assert online_before.stop_reason == "tonic_stop"

    assert receipt.status == "APPENDED"
    assert receipt.parent_checkpoint_sha256 == parent_root
    assert receipt.successor_checkpoint_sha256 == successor.checkpoint_sha256
    assert receipt.successor_checkpoint_sha256 != parent_root
    assert receipt.reserved_cell_address_start == parent_summary.next_reserved_cell_address
    assert receipt.reserved_synapse_address_start == (parent_summary.next_reserved_synapse_address)
    assert receipt.reserved_cell_capacity_estimate_added > 0
    assert receipt.reserved_synapse_capacity_estimate_added > 0
    assert receipt.learned_event_bytes_retained_estimate_added > 0
    assert receipt.excitatory_support_tick_estimate_added == (
        receipt.coactivation_support_tick_estimate_added
        + receipt.positive_success_support_tick_estimate_added
    )
    assert receipt.coactivation_support_tick_estimate_added > 0
    assert receipt.coactivation_support_tick_estimate_added == (
        receipt.positive_success_support_tick_estimate_added
    )
    assert receipt.inhibitory_support_tick_estimate_added == 0
    assert receipt.source_experience_count == 1
    assert receipt.current_experience_training_rows == 2
    assert receipt.committed_evidence_tick_phases == 2
    assert receipt.parent_experience_replay_count == 0
    assert receipt.raw_training_sequences_retained == 0
    assert successor.training_summary.segment_count == 2
    assert successor.training_summary.next_reserved_cell_address == (
        parent_summary.next_reserved_cell_address + receipt.reserved_cell_capacity_estimate_added
    )
    assert successor.training_summary.next_reserved_synapse_address == (
        parent_summary.next_reserved_synapse_address
        + receipt.reserved_synapse_capacity_estimate_added
    )
    assert successor.training_summary.learned_event_bytes_retained_estimate == (
        parent_summary.learned_event_bytes_retained_estimate
        + receipt.learned_event_bytes_retained_estimate_added
    )

    learned = successor.respond(online_fresh.query, online_fresh.before_files)
    base_after_growth = successor.respond(base_fresh.query, base_fresh.before_files)
    unknown = successor.respond(
        b"repair absent_worker using existing bytes",
        online_fresh.before_files,
    )
    ambiguous = successor.respond(
        b"repair lumen_worker or quartz_worker using existing bytes",
        online_fresh.before_files,
    )

    assert learned.decision == "EMIT"
    assert learned.winning_segment_ordinal == 1
    assert learned.winning_selector_cell_id == receipt.reserved_cell_address_start
    assert learned.component_receipt_sha256 is not None
    assert _materialized(learned.emissions, online_fresh) == online_fresh.after_files
    assert base_after_growth.decision == "EMIT"
    assert base_after_growth.winning_segment_ordinal == 0
    assert _materialized(base_after_growth.emissions, base_fresh) == base_fresh.after_files
    assert unknown.decision == "STOP"
    assert unknown.stop_reason == "tonic_stop"
    assert ambiguous.decision == "STOP"
    assert ambiguous.emissions == ()
    assert ambiguous.stop_reason == "tonic_stop"

    for ablation in (
        BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF,
        BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF,
        BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF,
    ):
        stopped = successor.respond(
            online_fresh.query,
            online_fresh.before_files,
            ablation=ablation,
        )
        assert stopped.decision == "STOP"
        assert stopped.stop_reason == "tonic_stop"

    assert stored_base._action_tape._pending is None
    assert successor._segments[1].adapter.controller._action_tape._pending is None
    assert successor.checkpoint_sha256 == receipt.successor_checkpoint_sha256
    assert parent.checkpoint_sha256 == parent_root
    assert parent.respond(base_fresh.query, base_fresh.before_files).sha256 == base_before.sha256
    assert parent.respond(online_fresh.query, online_fresh.before_files).decision == "STOP"


def test_no_update_duplicate_conflict_and_tamper_commit_no_state() -> None:
    parent = AppendOnlyMultiscaleSourceMotorStack.seed(_base_controller())
    evidence = _evidence(parent.checkpoint_sha256)
    no_update = parent.extend_success(
        evidence,
        ablation=OnlineMultiscaleGrowthAblation.NO_UPDATE,
    )

    assert no_update.successor is parent
    assert no_update.receipt.status == "NO_UPDATE"
    assert no_update.receipt.successor_checkpoint_sha256 == parent.checkpoint_sha256
    assert no_update.receipt.successor_provenance_sha256 == (
        no_update.receipt.prior_provenance_sha256
    )
    assert no_update.receipt.reserved_cell_capacity_estimate_added == 0
    assert no_update.receipt.reserved_synapse_capacity_estimate_added == 0
    assert no_update.receipt.learned_event_bytes_retained_estimate_added == 0
    assert no_update.receipt.excitatory_support_tick_estimate_added == 0
    assert no_update.receipt.committed_evidence_tick_phases == 0
    assert (
        no_update.successor.respond(
            evidence.query,
            evidence.before_files,
        ).decision
        == "STOP"
    )

    appended = parent.extend_success(evidence).successor
    with pytest.raises(ValueError, match="does not bind this parent learner"):
        appended.extend_success(evidence)

    duplicate = appended.extend_success(_evidence(appended.checkpoint_sha256))
    conflicting_row = _online_experience(short=True)
    conflict = appended.extend_success(
        _evidence(
            appended.checkpoint_sha256,
            experience=conflicting_row,
        )
    )

    assert duplicate.successor is appended
    assert duplicate.receipt.status == "DUPLICATE"
    assert duplicate.receipt.reserved_cell_capacity_estimate_added == 0
    assert conflict.successor is appended
    assert conflict.receipt.status == "CONFLICT_STOP"
    assert conflict.receipt.stop_reason == "incompatible_observation_action"
    assert conflict.receipt.admission_race is not None
    assert conflict.receipt.admission_race.stopped
    assert conflict.receipt.admission_race.stop_reason == "collision"
    assert conflict.receipt.admission_race.simultaneous_spike_candidates == 2
    assert conflict.receipt.reserved_cell_capacity_estimate_added == 0
    assert conflict.receipt.reserved_synapse_capacity_estimate_added == 0
    assert conflict.receipt.learned_event_bytes_retained_estimate_added == 0
    assert appended.checkpoint_sha256 == conflict.receipt.successor_checkpoint_sha256

    tampered = replace(evidence, action_sha256="0" * 64)
    with pytest.raises(ValueError, match="action custody changed"):
        parent.extend_success(tampered)

    failed_receipt = copy(evidence.execution_receipt)
    object.__setattr__(failed_receipt, "outcome", "FAIL")
    with pytest.raises(ValueError, match="requires a PASS receipt"):
        parent.extend_success(replace(evidence, execution_receipt=failed_receipt))

    evaluator_calls = 0

    def failing_evaluator(**_observation: object) -> bool:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return False

    authority = MultiscaleSourceMotorExecutionAuthority(
        evaluator=failing_evaluator,
        evaluator_sha256=sha256(b"failing-multiscale-evaluator").hexdigest(),
    )
    with pytest.raises(ValueError, match="did not PASS"):
        authority.execute_and_issue(
            learner_checkpoint_sha256=parent.checkpoint_sha256,
            query=evidence.query,
            before_files=evidence.before_files,
            after_files=evidence.after_files,
        )
    assert evaluator_calls == 1
    assert not hasattr(MultiscaleSourceMotorExecutionReceipt, "seal_pass")

    assert tuple(signature(AppendOnlyMultiscaleSourceMotorStack.respond).parameters) == (
        "self",
        "query",
        "files",
        "ablation",
    )
