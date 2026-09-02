"""Public-TRAIN continual source-edit evidence for append-only neuronal growth.

Two independently seeded stacks start from the same frozen cue-free Python
controller.  Both first STOP on one genuinely new exact context.  A public
teaching correction is applied and executed once, then its exact observation
and action are sealed in a PASS receipt.  Treatment appends one frozen neuronal
shard; NO_UPDATE follows the same admission path while committing no state.

Fresh tasks contain only issue text and complete repository files.  Each arm
receives one whole-workspace query per task.  The narrow claim is later
same-context recurrence across renamed values, changed paths, and shifted
distractors.  Exact context routing and the externally supplied teaching
trajectory remain scaffolding; this is not context abstraction or autonomous
program induction.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.r4_integrated import cue_free_python_source_edit_pilot_v1 as _transport
from humanbrain.raw_source_motor_online_growth import (
    AppendOnlyRawSourceMotorStack,
    OnlineGrowthAblation,
    SourceMotorExecutionAuthority,
    SourceMotorSuccessEvidence,
)

__all__ = [
    "STATUS_PASS",
    "STATUS_STOP",
    "public_online_growth_tasks",
    "run_experiment",
]


STATUS_PASS = "PASS_CUE_FREE_PYTHON_ONLINE_GROWTH_V1"
STATUS_STOP = "STOP_CUE_FREE_PYTHON_ONLINE_GROWTH_V1"

_ONLINE_CONTEXT = b"dependency"
_TEACHING_FUNCTION = "prepare_citrine"


def public_online_growth_tasks() -> tuple[_transport.PublicPythonRepositoryTask, ...]:
    """Return teaching plus fresh transfer observations, never target source."""

    return (
        _transport.PublicPythonRepositoryTask(
            "online-teaching",
            b"the dependency is called before it is defined; reorder existing source.",
            (
                _transport.WorkspaceFile(
                    "a_docs/lesson.md",
                    b"public teaching repository; execution decides success\n",
                ),
                _transport.WorkspaceFile(
                    "middle/lesson.py",
                    _transport._buggy_source(_TEACHING_FUNCTION),
                ),
                _transport.WorkspaceFile("z_settings.toml", b'mode = "teach"\n'),
            ),
        ),
        _transport.PublicPythonRepositoryTask(
            "online-transfer-early-path",
            b"move the dependency definition before its first use; preserve all lines.",
            (
                _transport.WorkspaceFile(
                    "a_runtime.py",
                    _transport._buggy_source("open_ultramarine"),
                ),
                _transport.WorkspaceFile("m_notes.txt", b"renamed transfer alpha\n"),
                _transport.WorkspaceFile(
                    "z_docs/details.md",
                    b"non-executable distractor after the live file\n",
                ),
            ),
        ),
        _transport.PublicPythonRepositoryTask(
            "online-transfer-late-path",
            b"the dependency call occurs too early; reorder only existing source.",
            (
                _transport.WorkspaceFile(
                    "a_assets/data.json",
                    b'{"kind": "position-shifting distractor"}\n',
                ),
                _transport.WorkspaceFile(
                    "b_docs/guide.md",
                    b"another renamed repository\n\nwith extra lines\n",
                ),
                _transport.WorkspaceFile("m_config.ini", b"enabled=true\n"),
                _transport.WorkspaceFile(
                    "zeta/jobs/worker.py",
                    _transport._buggy_source("stage_vermilion"),
                ),
            ),
        ),
    )


def _query(
    stack: AppendOnlyRawSourceMotorStack,
    task: _transport.PublicPythonRepositoryTask,
):
    context_events = _transport.tokenize_issue_losslessly(task.issue)
    source_tape = _transport.canonical_workspace_tape(task.files)
    return stack.respond(context_events, source_tape), source_tape


def _teaching_action(
    task: _transport.PublicPythonRepositoryTask,
) -> tuple[bytes, ...]:
    python_files = tuple(item for item in task.files if item.path.endswith(".py"))
    if len(python_files) != 1:
        raise RuntimeError("teaching correction requires one live Python file")
    path = python_files[0].path
    return (
        *_transport._path_tokens(path),
        *_transport._source_tokens(_transport._correct_source(_TEACHING_FUNCTION)),
    )


def _all_emissions_are_current(
    emissions: tuple[bytes, ...],
    source_tape: tuple[bytes, ...],
) -> bool:
    emitted_counts = Counter(emissions)
    source_counts = Counter(source_tape)
    return all(count <= source_counts[token] for token, count in emitted_counts.items())


def _execute(
    files: tuple[_transport.WorkspaceFile, ...],
    root: Path,
) -> bool:
    return _transport._run_python_behavior(files, root)


def _evaluator_sha256() -> str:
    transport_sha256 = sha256(Path(_transport.__file__).read_bytes()).digest()
    issuer_sha256 = sha256(Path(__file__).read_bytes()).digest()
    return sha256(
        b"humanbrain.cue-free-python-online-growth.evaluator-v1\0"
        + transport_sha256
        + issuer_sha256
    ).hexdigest()


def _seal_executed_teaching_success(
    task: _transport.PublicPythonRepositoryTask,
    *,
    learner_checkpoint_sha256: str,
    root: Path,
) -> tuple[SourceMotorSuccessEvidence, dict[str, object]]:
    before_tokens = _transport.canonical_workspace_tape(task.files)
    action = _teaching_action(task)
    if not _all_emissions_are_current(action, before_tokens):
        raise RuntimeError("teaching correction must reuse only current tape occurrences")
    before_passed = _execute(task.files, root / "before")
    if before_passed:
        raise RuntimeError("teaching repository must fail before correction")
    evaluator_calls = 0

    def frozen_evaluator(**query: object) -> bool:
        nonlocal evaluator_calls
        evaluator_calls += 1
        if (
            query["context_event"] != _ONLINE_CONTEXT
            or query["before_tokens"] != before_tokens
            or query["emitted_after_tokens"] != action
        ):
            return False
        corrected = _transport.apply_emitted_source_action(task.files, action)
        return _execute(corrected, root / "after")

    authority = SourceMotorExecutionAuthority(
        evaluator=frozen_evaluator,
        evaluator_sha256=_evaluator_sha256(),
    )
    execution_receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=learner_checkpoint_sha256,
        context_event=_ONLINE_CONTEXT,
        before_tokens=before_tokens,
        emitted_after_tokens=action,
    )
    if evaluator_calls != 1:
        raise RuntimeError("execution authority did not invoke the evaluator exactly once")
    execution_receipt.authenticate()
    evidence = SourceMotorSuccessEvidence.seal(
        context_event=_ONLINE_CONTEXT,
        before_tokens=before_tokens,
        emitted_after_tokens=action,
        execution_receipt=execution_receipt,
        provenance_id="public-online-teaching-success-v1",
    )
    evidence.authenticate()
    return evidence, {
        "action_sha256": evidence.action_sha256,
        "after_behavior_passed": execution_receipt.outcome == "PASS",
        "before_behavior_failed": not before_passed,
        "execution_receipt_sha256": execution_receipt.sha256,
        "observation_sha256": evidence.observation_sha256,
    }


def _run_transfer_arm(
    *,
    name: str,
    stack: AppendOnlyRawSourceMotorStack,
    tasks: tuple[_transport.PublicPythonRepositoryTask, ...],
    root: Path,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for task in tasks:
        decision, source_tape = _query(stack, task)
        current_occurrences = _all_emissions_are_current(
            decision.emissions,
            source_tape,
        )
        behavior_passed = False
        if decision.decision == "EMIT":
            applied = _transport.apply_emitted_source_action(
                task.files,
                decision.emissions,
            )
            behavior_passed = _execute(applied, root / name / task.task_id)
        records.append(
            {
                "behavior_passed": behavior_passed,
                "checkpoint_sha256": decision.checkpoint_sha256,
                "current_tape_occurrences_only": current_occurrences,
                "decision": decision.decision,
                "physical_stop": decision.race.stopped,
                "query_count": 1,
                "query_sha256": decision.query_sha256,
                "task_id": task.task_id,
            }
        )
    return {
        "behavior_passes": sum(bool(record["behavior_passed"]) for record in records),
        "emits": sum(record["decision"] == "EMIT" for record in records),
        "records": tuple(records),
        "stops": sum(record["decision"] == "STOP" for record in records),
    }


def run_experiment(root: Path) -> dict[str, object]:
    """Run one teaching update and fresh recurrence stream without an artifact."""

    if not isinstance(root, Path):
        raise TypeError("root must be pathlib.Path")
    teaching, *fresh_tasks_list = public_online_growth_tasks()
    fresh_tasks = tuple(fresh_tasks_list)

    treatment_parent = AppendOnlyRawSourceMotorStack.seed(_transport._controller())
    no_update_parent = AppendOnlyRawSourceMotorStack.seed(_transport._controller())
    initial_checkpoint = treatment_parent.checkpoint_sha256
    initial_roots_equal = (
        initial_checkpoint == no_update_parent.checkpoint_sha256
        and treatment_parent.training_summary.provenance_sha256
        == no_update_parent.training_summary.provenance_sha256
    )

    base_task = _transport.public_train_tasks()[0]
    treatment_base_before, base_tape = _query(treatment_parent, base_task)
    no_update_base_before, _ = _query(no_update_parent, base_task)
    base_before_passed = _execute(
        _transport.apply_emitted_source_action(
            base_task.files,
            treatment_base_before.emissions,
        ),
        root / "base" / "before",
    )

    treatment_teaching_before, teaching_tape = _query(treatment_parent, teaching)
    no_update_teaching_before, _ = _query(no_update_parent, teaching)
    evidence, teaching_execution = _seal_executed_teaching_success(
        teaching,
        learner_checkpoint_sha256=initial_checkpoint,
        root=root / "teaching",
    )

    treatment_transition = treatment_parent.extend_success(evidence)
    no_update_transition = no_update_parent.extend_success(
        evidence,
        ablation=OnlineGrowthAblation.NO_UPDATE,
    )
    treatment = treatment_transition.successor
    no_update = no_update_transition.successor
    growth = treatment_transition.receipt
    control_growth = no_update_transition.receipt

    treatment_results = _run_transfer_arm(
        name="treatment",
        stack=treatment,
        tasks=fresh_tasks,
        root=root / "transfer",
    )
    no_update_results = _run_transfer_arm(
        name="no_update",
        stack=no_update,
        tasks=fresh_tasks,
        root=root / "transfer",
    )

    treatment_base_after, _ = _query(treatment, base_task)
    no_update_base_after, _ = _query(no_update, base_task)
    base_after_passed = _execute(
        _transport.apply_emitted_source_action(
            base_task.files,
            treatment_base_after.emissions,
        ),
        root / "base" / "after",
    )

    collision = treatment.respond(
        (b"prerequisite", _ONLINE_CONTEXT),
        _transport.canonical_workspace_tape(fresh_tasks[0].files),
    )
    checks = {
        "identical_frozen_roots": initial_roots_equal,
        "both_initially_stop_on_new_context": (
            treatment_teaching_before.decision == no_update_teaching_before.decision == "STOP"
            and treatment_teaching_before.race.stopped
            and no_update_teaching_before.race.stopped
        ),
        "teaching_correction_executed_before_receipt": (
            teaching_execution["before_behavior_failed"]
            and teaching_execution["after_behavior_passed"]
        ),
        "receipt_binds_exact_observation_and_action": (
            evidence.execution_receipt.observation_sha256 == evidence.observation_sha256
            and evidence.execution_receipt.action_sha256 == evidence.action_sha256
            and growth.execution_receipt_sha256 == evidence.execution_receipt.sha256
            and control_growth.execution_receipt_sha256 == evidence.execution_receipt.sha256
        ),
        "identical_evidence_entered_both_arms": (
            growth.evidence_sha256 == control_growth.evidence_sha256 == evidence.sha256
        ),
        "treatment_appended_without_reset": (
            growth.status == "APPENDED"
            and growth.parent_checkpoint_sha256 == initial_checkpoint
            and treatment.checkpoint_sha256 == growth.successor_checkpoint_sha256
            and treatment.training_summary.segment_count
            == treatment_parent.training_summary.segment_count + 1
        ),
        "no_update_committed_zero_state": (
            no_update_transition.successor is no_update_parent
            and control_growth.status == "NO_UPDATE"
            and control_growth.parent_checkpoint_sha256
            == control_growth.successor_checkpoint_sha256
            == initial_checkpoint
            and control_growth.reserved_cell_capacity_added
            == control_growth.reserved_synapse_capacity_estimate_added
            == control_growth.excitatory_support_tick_estimate_added
            == 0
        ),
        "treatment_transfers_and_executes": (
            treatment_results["emits"] == len(fresh_tasks)
            and treatment_results["behavior_passes"] == len(fresh_tasks)
            and all(
                record["current_tape_occurrences_only"] and record["query_count"] == 1
                for record in treatment_results["records"]
            )
        ),
        "no_update_physically_stops": (
            no_update_results["stops"] == len(fresh_tasks)
            and all(
                record["physical_stop"] and record["query_count"] == 1
                for record in no_update_results["records"]
            )
        ),
        "base_behavior_retained": (
            treatment_base_before.decision
            == no_update_base_before.decision
            == treatment_base_after.decision
            == no_update_base_after.decision
            == "EMIT"
            and treatment_base_before.emissions == treatment_base_after.emissions
            and no_update_base_before.emissions == no_update_base_after.emissions
            and treatment_base_before.emissions == no_update_base_before.emissions
            and _all_emissions_are_current(treatment_base_after.emissions, base_tape)
            and base_before_passed
            and base_after_passed
        ),
        "growth_capacity_estimates_are_consistent_and_replay_free": (
            growth.reserved_cell_capacity_added > 0
            and growth.reserved_synapse_capacity_estimate_added > 0
            and growth.excitatory_support_tick_estimate_added
            == growth.coactivation_support_tick_estimate_added
            + growth.positive_success_support_tick_estimate_added
            and growth.inhibitory_support_tick_estimate_added == 0
            and growth.source_experience_count == 1
            and growth.committed_evidence_tick_phases == 2
            and growth.parent_experience_replay_count == 0
            and growth.raw_training_sequences_retained == 0
        ),
        "mixed_context_collision_physically_stops": (
            collision.decision == "STOP"
            and collision.stop_reason == "collision"
            and collision.race.stopped
            and collision.race.simultaneous_spike_candidates == 2
        ),
        "teaching_action_used_current_tape": _all_emissions_are_current(
            evidence.emitted_after_tokens,
            teaching_tape,
        ),
    }
    return {
        "arms": {
            "no_update": no_update_results,
            "treatment": treatment_results,
        },
        "checks": checks,
        "claim_boundary": (
            "one receipt-bound append enables exact-context same-layout recurrence "
            "through grounded whole-workspace tokens; exact context and an external "
            "successful teaching trajectory remain scaffolding"
        ),
        "fresh_transfer_count": len(fresh_tasks),
        "growth": {
            "coactivation_support_tick_estimate_added": (
                growth.coactivation_support_tick_estimate_added
            ),
            "excitatory_support_tick_estimate_added": (
                growth.excitatory_support_tick_estimate_added
            ),
            "parent_experience_replay_count": growth.parent_experience_replay_count,
            "positive_success_support_tick_estimate_added": (
                growth.positive_success_support_tick_estimate_added
            ),
            "raw_training_sequences_retained": growth.raw_training_sequences_retained,
            "reserved_cell_capacity_added": growth.reserved_cell_capacity_added,
            "reserved_synapse_capacity_estimate_added": (
                growth.reserved_synapse_capacity_estimate_added
            ),
        },
        "initial_checkpoint_sha256": initial_checkpoint,
        "status": STATUS_PASS if all(checks.values()) else STATUS_STOP,
        "teaching": teaching_execution,
    }


def main() -> None:
    with TemporaryDirectory(prefix="humanbrain-online-growth-v1-") as directory:
        print(json.dumps(run_experiment(Path(directory)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
