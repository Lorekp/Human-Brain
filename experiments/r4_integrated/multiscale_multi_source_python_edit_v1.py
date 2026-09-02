"""Fresh integrated issue-to-path grounding and source-motor transfer probe.

The experiment trains on two public successful repositories, then gives one
persistent neuronal controller three fresh TRAIN-only repositories.  Every
inference call receives exactly one raw issue plus every byte of all three
workspace files.  The learned computation is:

    query/source equality -> opaque enclosing path role -> learned motor chain

The action tasks transfer a renamed, same-layout prefix-truncation motor program
that removes one trailing crashing line by copying only live source units.  The
claim is intentionally narrower than arbitrary source editing, language
understanding, operation composition, or continual learning.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
    OpaqueWorkspaceFile,
)
from humanbrain.multiscale_source_motor_controller import (
    MultiscaleSourceMotorAblation,
    MultiscaleSourceMotorController,
    SuccessfulMultiscaleSourceMotorExperience,
)
from humanbrain.source_relative_action_tape import TapeValence

_EXPECTED_STDOUT = b"ready"
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_RESULT_NAMESPACE = b"humanbrain.multiscale-multi-source-python-edit-v1\x00"


@dataclass(frozen=True, slots=True)
class PublicMultiSourceTask:
    """The entire fresh public inference surface; it contains no answer field."""

    task_id: str
    issue: bytes
    files: tuple[OpaqueWorkspaceFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be non-empty text")
        if not isinstance(self.issue, bytes) or not self.issue:
            raise ValueError("issue must be non-empty bytes")
        MultiscaleOpaqueEventTape.observe(self.issue, self.files)


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


def _workspace(rows: tuple[tuple[str, str], ...]) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, _buggy_source(function_name)) for path, function_name in rows
    )


def _successful_experience(
    *,
    provenance_id: str,
    query: bytes,
    rows: tuple[tuple[str, str], ...],
    changed_path: str,
) -> SuccessfulMultiscaleSourceMotorExperience:
    before = _workspace(rows)
    function_by_path = dict(rows)
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            (
                _correct_source(function_by_path[item.path])
                if item.path == changed_path
                else item.content
            ),
        )
        for item in before
    )
    return SuccessfulMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def public_training_experiences() -> tuple[SuccessfulMultiscaleSourceMotorExperience, ...]:
    """Two public successes; their repaired sources exist only at training."""

    return (
        _successful_experience(
            provenance_id="public-train-orbit",
            query=b"repair mira_worker using existing bytes",
            rows=(
                ("atlas_worker.py", "atlas_cycle"),
                ("mira_worker.py", "mira_cycle"),
                ("zenith_worker.py", "zenith_cycle"),
            ),
            changed_path="mira_worker.py",
        ),
        _successful_experience(
            provenance_id="public-train-grove",
            query=b"repair tango_worker using existing bytes",
            rows=(
                ("ember_worker.py", "ember_cycle"),
                ("tango_worker.py", "tango_cycle"),
                ("willow_worker.py", "willow_cycle"),
            ),
            changed_path="tango_worker.py",
        ),
    )


def public_fresh_tasks() -> tuple[PublicMultiSourceTask, ...]:
    """Two fresh action tasks and one fresh genuine equality collision."""

    return (
        PublicMultiSourceTask(
            task_id="fresh-lumen",
            issue=b"repair lumen_worker using existing bytes",
            files=_workspace(
                (
                    ("cobalt_worker.py", "cobalt_cycle"),
                    ("lumen_worker.py", "lumen_cycle"),
                    ("sable_worker.py", "sable_cycle"),
                )
            ),
        ),
        PublicMultiSourceTask(
            task_id="fresh-vireo",
            issue=b"repair vireo_worker using existing bytes",
            files=_workspace(
                (
                    ("vireo_worker.py", "vireo_cycle"),
                    ("wren_worker.py", "wren_cycle"),
                    ("zephyr_worker.py", "zephyr_cycle"),
                )
            ),
        ),
        PublicMultiSourceTask(
            task_id="fresh-ambiguous-prism-quartz",
            issue=b"repair prism_worker or quartz_worker using existing bytes",
            files=_workspace(
                (
                    ("prism_worker.py", "prism_cycle"),
                    ("quartz_worker.py", "quartz_cycle"),
                    ("raven_worker.py", "raven_cycle"),
                )
            ),
        ),
    )


def public_inference_schema() -> tuple[str, ...]:
    """Expose the exact fresh task fields for evidence-integrity checks."""

    return tuple(item.name for item in fields(PublicMultiSourceTask))


def _issue_related_paths(
    issue: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    """Evaluator-only exact identity relation, invoked after model response."""

    issue_identities = frozenset(_IDENTIFIER.findall(issue))
    return tuple(
        item.path
        for item in sorted(files, key=lambda candidate: candidate.path)
        if PurePosixPath(item.path).stem.encode("utf-8") in issue_identities
    )


def _materialize_workspace(
    files: tuple[OpaqueWorkspaceFile, ...],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)


def _run_python_file(
    files: tuple[OpaqueWorkspaceFile, ...],
    path: str,
    root: Path,
) -> bool:
    if sum(item.path == path for item in files) != 1:
        return False
    _materialize_workspace(files, root)
    execution = subprocess.run(
        [sys.executable, str(root.joinpath(*PurePosixPath(path).parts))],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return execution.returncode == 0 and execution.stdout == _EXPECTED_STDOUT


def _changed_paths(
    before: tuple[OpaqueWorkspaceFile, ...],
    after: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    before_by_path = {item.path: item.content for item in before}
    after_by_path = {item.path: item.content for item in after}
    if before_by_path.keys() != after_by_path.keys():
        return tuple(sorted(before_by_path.keys() | after_by_path.keys()))
    return tuple(
        path for path in sorted(before_by_path) if before_by_path[path] != after_by_path[path]
    )


def _decoys_preserved(
    before: tuple[OpaqueWorkspaceFile, ...],
    after: tuple[OpaqueWorkspaceFile, ...],
    intended_path: str,
) -> bool:
    before_by_path = {item.path: item.content for item in before}
    after_by_path = {item.path: item.content for item in after}
    return before_by_path.keys() == after_by_path.keys() and all(
        after_by_path[path] == content
        for path, content in before_by_path.items()
        if path != intended_path
    )


def _record_digest(record: object) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(_RESULT_NAMESPACE + payload).hexdigest()


def _run_task(
    controller: MultiscaleSourceMotorController,
    task: PublicMultiSourceTask,
    *,
    ablation: MultiscaleSourceMotorAblation,
    root: Path,
) -> dict[str, object]:
    observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
    before = observation.reconstruct_workspace()

    # This is the sole model query.  No evaluator-derived relation exists yet.
    decision = controller.respond(task.issue, task.files, ablation=ablation)

    source_occurrences = Counter(observation.source_events)
    emitted_occurrences = Counter(decision.emissions)
    emissions_from_current_tape = all(
        count <= source_occurrences[event] for event, count in emitted_occurrences.items()
    )
    updated = before
    materialization_valid: bool | None = None
    materialization_error: str | None = None
    if decision.decision == "EMIT":
        try:
            updated = observation.materialize_motor_emission(decision.emissions)
            materialization_valid = True
        except (OpaqueEventTapeError, TypeError, ValueError) as error:
            materialization_valid = False
            materialization_error = type(error).__name__
        consequence = controller.resolve(TapeValence.NEUTRAL)
        neutral_checkpoint_retained = (
            consequence.prior_training_sha256 == consequence.training_sha256
            and controller.checkpoint_sha256 == decision.checkpoint_sha256
        )
    else:
        neutral_checkpoint_retained = True

    # Private scoring starts only after the response and neutral consequence.
    related_paths = _issue_related_paths(task.issue, task.files)
    ambiguous = len(related_paths) != 1
    intended_path = None if ambiguous else related_paths[0]
    changed_paths = _changed_paths(before, updated)
    if intended_path is None:
        before_behavior_passed: bool | None = None
        after_behavior_passed: bool | None = None
        decoys_preserved: bool | None = None
        capability_success = decision.decision == "STOP" and not changed_paths
    else:
        before_behavior_passed = _run_python_file(
            before,
            intended_path,
            root / "before",
        )
        after_behavior_passed = _run_python_file(
            updated,
            intended_path,
            root / "after",
        )
        decoys_preserved = _decoys_preserved(before, updated, intended_path)
        capability_success = (
            decision.decision == "EMIT"
            and materialization_valid is True
            and before_behavior_passed is False
            and after_behavior_passed is True
            and changed_paths == (intended_path,)
            and decoys_preserved
            and emissions_from_current_tape
        )

    if capability_success:
        outcome = "correct_ambiguity_stop" if ambiguous else "correct_edit"
    elif decision.decision == "STOP":
        outcome = "grounding_or_motor_stop"
    elif materialization_valid is not True:
        outcome = "invalid_materialization"
    elif intended_path not in changed_paths:
        outcome = "wrong_file_or_no_change"
    else:
        outcome = "behavior_failure"

    return {
        "after_behavior_passed": after_behavior_passed,
        "ambiguous": ambiguous,
        "before_behavior_passed": before_behavior_passed,
        "capability_success": capability_success,
        "changed_paths": changed_paths,
        "decision": decision.decision,
        "decoys_preserved": decoys_preserved,
        "emission_count": len(decision.emissions),
        "emissions_from_current_tape": emissions_from_current_tape,
        "materialization_error": materialization_error,
        "materialization_valid": materialization_valid,
        "neutral_checkpoint_retained": neutral_checkpoint_retained,
        "outcome": outcome,
        "physical_cell_steps": decision.physical_cell_steps,
        "query_count": 1,
        "query_sha256": decision.query_sha256,
        "related_path_count": len(related_paths),
        "stop_reason": decision.stop_reason,
        "stop_stage": decision.stop_stage,
        "task_id": task.task_id,
        "workspace_lossless": observation.reconstruct_workspace() == before,
    }


def _run_arm(
    *,
    ablation: MultiscaleSourceMotorAblation,
    tasks: tuple[PublicMultiSourceTask, ...],
    root: Path,
) -> dict[str, object]:
    controller = MultiscaleSourceMotorController.grow(public_training_experiences())
    initial_checkpoint = controller.checkpoint_sha256
    task_records = tuple(
        _run_task(
            controller,
            task,
            ablation=ablation,
            root=root / task.task_id,
        )
        for task in tasks
    )
    action_records = tuple(row for row in task_records if not row["ambiguous"])
    ambiguity_records = tuple(row for row in task_records if row["ambiguous"])
    return {
        "ablation": ablation.value,
        "action_required_success_count": sum(
            bool(row["capability_success"]) for row in action_records
        ),
        "ambiguity_stop_success_count": sum(
            bool(row["capability_success"]) for row in ambiguity_records
        ),
        "checkpoint_retained": controller.checkpoint_sha256 == initial_checkpoint,
        "emit_count": sum(row["decision"] == "EMIT" for row in task_records),
        "stop_count": sum(row["decision"] == "STOP" for row in task_records),
        "tasks": task_records,
        "training_summary": asdict(controller.training_summary),
    }


def _run_experiment_at(root: Path) -> dict[str, object]:
    tasks = public_fresh_tasks()
    arms = {
        ablation.value: _run_arm(
            ablation=ablation,
            tasks=tasks,
            root=root / ablation.value,
        )
        for ablation in MultiscaleSourceMotorAblation
    }
    treatment = arms[MultiscaleSourceMotorAblation.NONE.value]
    equality_off = arms[MultiscaleSourceMotorAblation.EQUALITY_OFF.value]
    propagation_off = arms[MultiscaleSourceMotorAblation.PROPAGATION_OFF.value]
    integrity = {
        "fresh_task_fields": public_inference_schema(),
        "fresh_tasks_have_three_files": all(len(task.files) == 3 for task in tasks),
        "all_queries_single_shot": all(
            row["query_count"] == 1 for arm in arms.values() for row in arm["tasks"]
        ),
        "all_workspaces_lossless": all(
            row["workspace_lossless"] for arm in arms.values() for row in arm["tasks"]
        ),
        "all_emissions_live": all(
            row["emissions_from_current_tape"] for arm in arms.values() for row in arm["tasks"]
        ),
        "all_neutral_checkpoints_retained": all(
            arm["checkpoint_retained"]
            and all(row["neutral_checkpoint_retained"] for row in arm["tasks"])
            for arm in arms.values()
        ),
    }
    gate_passed = (
        treatment["action_required_success_count"] == 2
        and treatment["ambiguity_stop_success_count"] == 1
        and equality_off["emit_count"] == 0
        and equality_off["stop_count"] == len(tasks)
        and propagation_off["emit_count"] == 0
        and propagation_off["stop_count"] == len(tasks)
        and all(integrity.values())
    )
    result: dict[str, object] = {
        "arms": arms,
        "claim": (
            "Fresh renamed same-layout issue-to-path grounding plus learned "
            "prefix-truncation motor-program transfer across three-file workspaces."
        ),
        "fresh_action_task_count": 2,
        "fresh_ambiguity_task_count": 1,
        "gate_passed": gate_passed,
        "integrity": integrity,
        "limitations": (
            "Exact opaque identifier equality supplies the issue-to-path seed.",
            "The transferred edit keeps the demonstrated physical-line layout.",
            "The action is one learned contiguous prefix copy, not novel composition.",
            "Fresh tasks do not update the learner; this is not continual learning.",
            "The result is not arbitrary language understanding or source editing.",
        ),
        "public_training_experience_count": len(public_training_experiences()),
    }
    result["result_sha256"] = _record_digest(result)
    return result


def run_experiment(root: Path | None = None) -> dict[str, object]:
    """Run the fresh TRAIN-only cohort once without writing an artifact."""

    if root is not None:
        return _run_experiment_at(root)
    with tempfile.TemporaryDirectory(prefix="multiscale-source-edit-") as directory:
        return _run_experiment_at(Path(directory))


def main() -> None:
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PublicMultiSourceTask",
    "public_fresh_tasks",
    "public_inference_schema",
    "public_training_experiences",
    "run_experiment",
]
