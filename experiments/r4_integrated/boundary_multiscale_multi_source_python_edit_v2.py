"""Fresh boundary-invariant multi-source Python repair cohort.

The frozen neuronal controller grows from two successful public TRAIN
transitions whose targets occupy interior segments of four-file workspaces.
One persistent controller then receives three fresh action repositories with
the target segment at the first, middle, and last sorted positions, plus one
genuinely ambiguous repository.  Every decision sees exactly one raw issue and
the entire lossless workspace; target identity and execution scoring begin only
after that decision and neutral consequence resolution.

The learned motor remains deliberately narrow: copy a live target path and the
live prefix of its source while dropping one trailing crashing line.  This
cohort tests boundary-relative grounding across workspace position, not general
Python synthesis or natural-language understanding.
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

from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorAblation,
    BoundaryMultiscaleSourceMotorController,
    SuccessfulBoundaryMultiscaleSourceMotorExperience,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
    OpaqueWorkspaceFile,
)
from humanbrain.source_relative_action_tape import TapeValence

_EXPECTED_STDOUT = b"settled"
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_RESULT_NAMESPACE = b"humanbrain.boundary-multiscale-multi-source-python-edit-v2\x00"


@dataclass(frozen=True, slots=True)
class PublicBoundaryMultiSourceTask:
    """The complete public inference surface, with no answer-bearing field."""

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
        '    return "settled"\n'
        f'print({function_name}(), end="")\n'
        'raise RuntimeError("obsolete")\n'
    ).encode()


def _correct_source(function_name: str) -> bytes:
    return (
        f'def {function_name}():\n    return "settled"\nprint({function_name}(), end="")\n'
    ).encode()


def _sorted_workspace(
    rows: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, content)
        for path, content in sorted(rows, key=lambda item: item[0])
    )


def _target_workspace(
    *,
    target_path: str,
    function_name: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return _sorted_workspace((*decoys, (target_path, _buggy_source(function_name))))


def _successful_experience(
    *,
    provenance_id: str,
    query: bytes,
    target_path: str,
    function_name: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    before = _target_workspace(
        target_path=target_path,
        function_name=function_name,
        decoys=decoys,
    )
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            _correct_source(function_name) if item.path == target_path else item.content,
        )
        for item in before
    )
    return SuccessfulBoundaryMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def public_training_experiences() -> tuple[SuccessfulBoundaryMultiscaleSourceMotorExperience, ...]:
    """Two public successes with targets at distinct interior positions."""

    return (
        _successful_experience(
            provenance_id="public-train-helios",
            query=b"repair helios_worker using existing bytes",
            target_path="helios_worker.py",
            function_name="helios_cycle",
            decoys=(
                ("amber_notes.md", b"# amber\nquiet observational notes\n"),
                ("quill_data.json", b'{"quill": 17, "active": true}\n'),
                ("zinc_helper.py", b'LABEL = "zinc"\n'),
            ),
        ),
        _successful_experience(
            provenance_id="public-train-nivalis",
            query=b"repair nivalis_worker using existing bytes",
            target_path="nivalis_worker.py",
            function_name="nivalis_cycle",
            decoys=(
                ("birch_config.toml", b"enabled = true\nlimit = 23\n"),
                ("copper_helper.py", b"COPPER_LEVEL = 5\n"),
                ("willow_notes.txt", b"willow ledger\nsecond retained row\n"),
            ),
        ),
    )


def public_fresh_tasks() -> tuple[PublicBoundaryMultiSourceTask, ...]:
    """Three positional action tasks and one equality-ambiguous task."""

    return (
        PublicBoundaryMultiSourceTask(
            task_id="fresh-auric-first",
            issue=b"repair auric_worker using existing bytes",
            files=_target_workspace(
                target_path="auric_worker.py",
                function_name="auric_cycle",
                decoys=(
                    ("ember_notes.md", b"# ember index\nretained prose\n"),
                    ("quartz_data.json", b'{"quartz": [2, 3, 5]}\n'),
                    ("zephyr_helper.py", b"def zephyr_value():\n    return 31\n"),
                ),
            ),
        ),
        PublicBoundaryMultiSourceTask(
            task_id="fresh-selene-middle",
            issue=b"repair selene_worker using existing bytes",
            files=_target_workspace(
                target_path="selene_worker.py",
                function_name="selene_cycle",
                decoys=(
                    ("cinder_notes.txt", b"cinder one\ncinder two\ncinder three\n"),
                    ("frost_config.toml", b'name = "frost"\nweight = 0.75\n'),
                    ("willow_helper.py", b"WILLOW_READY = False\n"),
                ),
            ),
        ),
        PublicBoundaryMultiSourceTask(
            task_id="fresh-wyvern-last",
            issue=b"repair wyvern_worker using existing bytes",
            files=_target_workspace(
                target_path="wyvern_worker.py",
                function_name="wyvern_cycle",
                decoys=(
                    ("atlas_notes.md", b"# atlas\nlonger retained reference material\n"),
                    ("cobalt_config.json", b'{"mode": "cobalt", "count": 11}\n'),
                    ("lumen_helper.py", b"LUMEN_SCALE = (1, 4, 9)\n"),
                ),
            ),
        ),
        PublicBoundaryMultiSourceTask(
            task_id="fresh-ambiguous-datura-xiphos",
            issue=b"repair datura_worker or xiphos_worker using existing bytes",
            files=_sorted_workspace(
                (
                    ("datura_worker.py", _buggy_source("datura_cycle")),
                    ("granite_notes.md", b"# granite\nunaltered ambiguity decoy\n"),
                    ("tundra_config.json", b'{"tundra": true, "depth": 8}\n'),
                    ("xiphos_worker.py", _buggy_source("xiphos_cycle")),
                )
            ),
        ),
    )


def public_inference_schema() -> tuple[str, ...]:
    """Expose the precise answer-free fresh task fields."""

    return tuple(item.name for item in fields(PublicBoundaryMultiSourceTask))


def _issue_related_paths(
    issue: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    """Evaluator-only equality relation, first invoked after a decision."""

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
    controller: BoundaryMultiscaleSourceMotorController,
    task: PublicBoundaryMultiSourceTask,
    *,
    ablation: BoundaryMultiscaleSourceMotorAblation,
    root: Path,
) -> dict[str, object]:
    observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
    before = observation.reconstruct_workspace()

    # The one and only model query precedes every evaluator-derived relation.
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
        if emissions_from_current_tape:
            try:
                updated = observation.materialize_motor_emission(decision.emissions)
                materialization_valid = True
            except (OpaqueEventTapeError, TypeError, ValueError) as error:
                materialization_valid = False
                materialization_error = type(error).__name__
        else:
            materialization_valid = False
            materialization_error = "non_live_emission"
        consequence = controller.resolve(TapeValence.NEUTRAL)
        neutral_checkpoint_retained = (
            consequence.prior_training_sha256 == consequence.training_sha256
            and controller.checkpoint_sha256 == decision.checkpoint_sha256
        )
    else:
        neutral_checkpoint_retained = True

    # Private scoring begins only after response and consequence resolution.
    related_paths = _issue_related_paths(task.issue, task.files)
    ambiguous = len(related_paths) != 1
    intended_path = None if ambiguous else related_paths[0]
    sorted_paths = tuple(item.path for item in before)
    intended_position = None if intended_path is None else sorted_paths.index(intended_path)
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
        "boundary_match_counts": tuple(
            binding.boundary_match_count for binding in decision.bindings
        ),
        "capability_success": capability_success,
        "changed_paths": changed_paths,
        "decision": decision.decision,
        "decoys_preserved": decoys_preserved,
        "emission_count": len(decision.emissions),
        "emissions_from_current_tape": emissions_from_current_tape,
        "intended_position": intended_position,
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
    ablation: BoundaryMultiscaleSourceMotorAblation,
    tasks: tuple[PublicBoundaryMultiSourceTask, ...],
    root: Path,
) -> dict[str, object]:
    controller = BoundaryMultiscaleSourceMotorController.grow(public_training_experiences())
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
        for ablation in BoundaryMultiscaleSourceMotorAblation
    }
    treatment = arms[BoundaryMultiscaleSourceMotorAblation.NONE.value]
    controls = tuple(
        arms[ablation.value]
        for ablation in (
            BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF,
            BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF,
            BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF,
        )
    )
    treatment_actions = tuple(row for row in treatment["tasks"] if not row["ambiguous"])
    integrity = {
        "fresh_task_fields": public_inference_schema(),
        "fresh_tasks_have_four_files": all(len(task.files) == 4 for task in tasks),
        "fresh_action_target_positions": tuple(
            row["intended_position"] for row in treatment_actions
        ),
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
        treatment["action_required_success_count"] == 3
        and treatment["ambiguity_stop_success_count"] == 1
        and all(control["emit_count"] == 0 for control in controls)
        and all(control["stop_count"] == len(tasks) for control in controls)
        and integrity["fresh_task_fields"] == ("task_id", "issue", "files")
        and integrity["fresh_tasks_have_four_files"] is True
        and integrity["fresh_action_target_positions"] == (0, 2, 3)
        and integrity["all_queries_single_shot"] is True
        and integrity["all_workspaces_lossless"] is True
        and integrity["all_emissions_live"] is True
        and integrity["all_neutral_checkpoints_retained"] is True
    )
    result: dict[str, object] = {
        "arms": arms,
        "claim": (
            "Fresh renamed boundary-invariant issue-to-path grounding plus learned "
            "live-token prefix-truncation transfer across varied four-file workspaces."
        ),
        "fresh_action_task_count": 3,
        "fresh_ambiguity_task_count": 1,
        "gate_passed": gate_passed,
        "integrity": integrity,
        "limitations": (
            "Exact opaque identifier equality supplies the issue-to-path seed.",
            "The target edit retains the demonstrated physical-line layout.",
            "The motor is one learned contiguous prefix copy, not novel composition.",
            "Fresh tasks do not update the learner; this is not continual learning.",
            "The result is not arbitrary language understanding or source editing.",
        ),
        "public_training_experience_count": len(public_training_experiences()),
    }
    result["result_sha256"] = _record_digest(result)
    return result


def run_experiment(root: Path | None = None) -> dict[str, object]:
    """Run this fresh public-TRAIN cohort without writing an artifact."""

    if root is not None:
        return _run_experiment_at(root)
    with tempfile.TemporaryDirectory(prefix="boundary-multiscale-source-edit-v2-") as directory:
        return _run_experiment_at(Path(directory))


def main() -> None:
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PublicBoundaryMultiSourceTask",
    "public_fresh_tasks",
    "public_inference_schema",
    "public_training_experiences",
    "run_experiment",
]
