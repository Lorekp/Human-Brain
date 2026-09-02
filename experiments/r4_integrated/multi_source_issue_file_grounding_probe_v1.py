"""Fresh public-TRAIN localization probe for issue-to-file grounding.

Every repository contains three to five ``*_worker.py`` files with the same
call-before-definition layout.  A named issue uniquely relates one fresh symbol
to one file; the ambiguity case relates two.  Public inference receives only
the complete lossless issue token stream and one complete lossless workspace
tape, followed by exactly one neuronal controller query.

Successful training pairs may identify their demonstrated edit.  Fresh target
relations are used only after inference by the evaluator and never enter the
controller frame, action adapter, consequence, or checkpoint.  A STOP on the
named fresh tasks is an expected localization of the missing relational
computation, not a failed experiment execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from experiments.r4_integrated import cue_free_python_source_edit_pilot_v1 as transport
from humanbrain import raw_source_motor_controller as controller_source
from humanbrain import raw_tape_role_binder as binder_source
from humanbrain.raw_source_motor_controller import (
    RawSourceMotorAblation,
    RawSourceMotorController,
    SuccessfulSourceMotorExperience,
)
from humanbrain.source_relative_action_tape import TapeValence

STATUS_PASS = "PASS_MULTI_SOURCE_ISSUE_FILE_GROUNDING_PROBE_V1"
STATUS_INVALID = "INVALID_MULTI_SOURCE_ISSUE_FILE_GROUNDING_PROBE_V1"
OUTCOME_TRANSFER = "PASS_MULTI_SOURCE_ISSUE_FILE_GROUNDING"
OUTCOME_LOCALIZED = "STOP_MISSING_ISSUE_TO_FILE_RELATIONAL_BINDING"

_CONTEXT_EVENT = b"prerequisite"
_EXPECTED_STDOUT = b"ready"
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_CALL_SYMBOL = re.compile(rb"print\(([A-Za-z_][A-Za-z0-9_]*)\(\), end=\"\"\)")
_EXPECTED_DEPENDENCY_CUSTODY = {
    "raw_source_motor_controller": (
        "0c5d85867922f6e4f77d25b7b7d1720d86abbd051b08cc5d9d739065bfebcac9"
    ),
    "raw_tape_role_binder": (
        "c7aaf52301a724ffcb960ed6705450193b40023430a1f2384a5846421509379a"
    ),
    "source_edit_transport": (
        "d9de5ca9f6b245029f5c9665b606df59235e5721a2b0c1fd4c32d5266c1e5152"
    ),
}


@dataclass(frozen=True, slots=True)
class PublicMultiSourceTask:
    """The entire fresh inference authority: identifier, issue, and workspace."""

    task_id: str
    issue: bytes
    files: tuple[transport.WorkspaceFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.issue, bytes) or not self.issue:
            raise ValueError("issue must be non-empty bytes")
        if type(self.files) is not tuple or not 3 <= len(self.files) <= 5:
            raise ValueError("files must contain three to five workspace files")
        if any(not isinstance(item, transport.WorkspaceFile) for item in self.files):
            raise TypeError("files must contain WorkspaceFile values")
        if len({item.path for item in self.files}) != len(self.files):
            raise ValueError("workspace paths must be unique")


@dataclass(frozen=True, slots=True)
class PublicInferenceFrame:
    """Only the two complete raw streams presented to neuronal inference."""

    issue_tokens: tuple[bytes, ...]
    workspace_tape: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class _SuccessfulTrainingRepository:
    provenance_id: str
    issue: bytes
    before: tuple[transport.WorkspaceFile, ...]
    after: tuple[transport.WorkspaceFile, ...]


def _buggy_source(symbol: str) -> bytes:
    return (
        f'print({symbol}(), end="")\n\n\ndef {symbol}():\n    return "ready"\n'.encode(
            "ascii"
        )
    )


def _correct_source(symbol: str) -> bytes:
    return (
        f'def {symbol}():\n    return "ready"\nprint({symbol}(), end="")\n\n\n'.encode(
            "ascii"
        )
    )


def _workspace(
    rows: tuple[tuple[str, str], ...],
) -> tuple[transport.WorkspaceFile, ...]:
    return tuple(
        transport.WorkspaceFile(path, _buggy_source(symbol)) for path, symbol in rows
    )


def _issue_related_paths(
    issue: bytes,
    files: tuple[transport.WorkspaceFile, ...],
) -> tuple[str, ...]:
    """Derive the scoring relation from public bytes, never an inference cue."""

    issue_identifiers = frozenset(_IDENTIFIER.findall(issue))
    return tuple(
        sorted(
            item.path
            for item in files
            if issue_identifiers.intersection(_IDENTIFIER.findall(item.source))
        )
    )


def _successful_repository(
    provenance_id: str,
    issue: bytes,
    rows: tuple[tuple[str, str], ...],
    demonstrated_symbol: str,
) -> _SuccessfulTrainingRepository:
    before = _workspace(rows)
    symbol_by_path = {path: symbol for path, symbol in rows}
    demonstrated_paths = tuple(
        path for path, symbol in rows if symbol == demonstrated_symbol
    )
    if len(demonstrated_paths) != 1:
        raise RuntimeError("training demonstration must identify one symbol")
    demonstrated_path = demonstrated_paths[0]
    if _issue_related_paths(issue, before) != (demonstrated_path,):
        raise RuntimeError("training issue must uniquely relate its demonstrated symbol")
    after = tuple(
        transport.WorkspaceFile(
            item.path,
            (
                _correct_source(symbol_by_path[item.path])
                if item.path == demonstrated_path
                else item.source
            ),
        )
        for item in before
    )
    return _SuccessfulTrainingRepository(provenance_id, issue, before, after)


def _training_repositories() -> tuple[_SuccessfulTrainingRepository, ...]:
    return (
        _successful_repository(
            "multi-source-success-alpha",
            b"the prerequisite tied to warm_amber is used too early; reorder it only.",
            (
                ("alpha/amber_worker.py", "warm_amber"),
                ("alpha/cobalt_worker.py", "prime_cobalt"),
                ("alpha/denim_worker.py", "cool_denim"),
            ),
            "warm_amber",
        ),
        _successful_repository(
            "multi-source-success-gamma",
            b"the prerequisite tied to wake_sable is used too early; reorder it only.",
            (
                ("gamma/jade_worker.py", "lift_jade"),
                ("gamma/russet_worker.py", "calm_russet"),
                ("gamma/sable_worker.py", "wake_sable"),
            ),
            "wake_sable",
        ),
    )


def public_train_tasks() -> tuple[PublicMultiSourceTask, ...]:
    """Return fresh TRAIN-only repositories; no corrected fresh source is stored."""

    return (
        PublicMultiSourceTask(
            "fresh-named-aurora",
            b"the prerequisite tied to ignite_aurora is used too early; reorder it only.",
            _workspace(
                (
                    ("runtime/aurora_worker.py", "ignite_aurora"),
                    ("runtime/boreal_worker.py", "settle_boreal"),
                    ("runtime/cirrus_worker.py", "guard_cirrus"),
                )
            ),
        ),
        PublicMultiSourceTask(
            "fresh-named-ember",
            b"the prerequisite tied to route_ember is used too early; reorder it only.",
            _workspace(
                (
                    ("service/delta_worker.py", "settle_delta"),
                    ("service/ember_worker.py", "route_ember"),
                    ("service/fjord_worker.py", "calm_fjord"),
                )
            ),
        ),
        PublicMultiSourceTask(
            "fresh-ambiguous-maple-nickel",
            (
                b"the prerequisite could refer equally to open_maple or bind_nickel; "
                b"do not guess."
            ),
            _workspace(
                (
                    ("jobs/maple_worker.py", "open_maple"),
                    ("jobs/nickel_worker.py", "bind_nickel"),
                    ("jobs/opal_worker.py", "guard_opal"),
                )
            ),
        ),
    )


def public_inference_frame(task: PublicMultiSourceTask) -> PublicInferenceFrame:
    return PublicInferenceFrame(
        issue_tokens=transport.tokenize_issue_losslessly(task.issue),
        workspace_tape=transport.canonical_workspace_tape(task.files),
    )


def _experience(
    repository: _SuccessfulTrainingRepository,
) -> SuccessfulSourceMotorExperience:
    before = {item.path: item.source for item in repository.before}
    after = {item.path: item.source for item in repository.after}
    changed = tuple(path for path in sorted(before) if before[path] != after[path])
    if before.keys() != after.keys() or len(changed) != 1:
        raise RuntimeError("training repository must contain exactly one observed edit")
    path = changed[0]
    return SuccessfulSourceMotorExperience(
        context_event=_CONTEXT_EVENT,
        before_tokens=transport.canonical_workspace_tape(repository.before),
        after_tokens=(
            *transport._path_tokens(path),
            *transport._source_tokens(after[path]),
        ),
        provenance_id=repository.provenance_id,
    )


def _controller() -> RawSourceMotorController:
    return RawSourceMotorController.grow(
        tuple(_experience(repository) for repository in _training_repositories())
    )


def _materialize_workspace(
    files: tuple[transport.WorkspaceFile, ...],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.source)


def _run_python_file(
    files: tuple[transport.WorkspaceFile, ...],
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
    before: tuple[transport.WorkspaceFile, ...],
    after: tuple[transport.WorkspaceFile, ...],
) -> tuple[str, ...]:
    before_by_path = {item.path: item.source for item in before}
    after_by_path = {item.path: item.source for item in after}
    if before_by_path.keys() != after_by_path.keys():
        raise RuntimeError("generic adapter changed the workspace path set")
    return tuple(
        path
        for path in sorted(before_by_path)
        if before_by_path[path] != after_by_path[path]
    )


def _decoys_preserved(
    before: tuple[transport.WorkspaceFile, ...],
    after: tuple[transport.WorkspaceFile, ...],
    intended_path: str,
) -> bool:
    before_by_path = {item.path: item.source for item in before}
    after_by_path = {item.path: item.source for item in after}
    return all(
        before_by_path[path] == after_by_path.get(path)
        for path in before_by_path
        if path != intended_path
    )


def _failure_class(
    *,
    ambiguous: bool,
    decision: str,
    stop_stage: str | None,
    materialization_valid: bool | None,
    changed_paths: tuple[str, ...],
    intended_path: str | None,
    after_behavior_passed: bool | None,
) -> str:
    if ambiguous:
        return "ambiguity_stop" if decision == "STOP" else "ambiguity_emit"
    if decision == "STOP":
        return f"{stop_stage or 'unknown'}_stop"
    if materialization_valid is not True:
        return "invalid_emit"
    if not changed_paths:
        return "no_op_emit"
    if changed_paths != (intended_path,):
        return "wrong_file_emit"
    if after_behavior_passed is not True:
        return "behavior_failure_after_emit"
    return "correct_action"


def _run_task(
    controller: RawSourceMotorController,
    task: PublicMultiSourceTask,
    root: Path,
) -> dict[str, object]:
    frame = public_inference_frame(task)
    decision = controller.respond(
        frame.issue_tokens,
        frame.workspace_tape,
        ablation=RawSourceMotorAblation.NONE,
    )

    source_occurrences = Counter(frame.workspace_tape)
    emitted_occurrences = Counter(decision.emissions)
    emissions_from_current_tape = all(
        count <= source_occurrences[token]
        for token, count in emitted_occurrences.items()
    )
    ordered_before = tuple(sorted(task.files, key=lambda item: item.path))
    updated_files = ordered_before
    materialization_valid: bool | None = None
    materialization_error: str | None = None
    if decision.decision == "EMIT":
        try:
            updated_files = transport.apply_emitted_source_action(
                task.files,
                decision.emissions,
            )
            materialization_valid = True
        except (TypeError, ValueError) as error:
            materialization_valid = False
            materialization_error = type(error).__name__
        update = controller.resolve(TapeValence.NEUTRAL)
        neutral_checkpoint_retained = (
            update.prior_training_sha256 == update.training_sha256
            and controller.checkpoint_sha256 == update.training_sha256
        )
    else:
        neutral_checkpoint_retained = True

    # Scoring begins only after the neuronal query and neutral consequence.
    related_paths = _issue_related_paths(task.issue, task.files)
    ambiguous = len(related_paths) != 1
    intended_path = None if ambiguous else related_paths[0]
    if intended_path is None:
        before_behavior_passed: bool | None = None
        after_behavior_passed: bool | None = None
        decoys_preserved: bool | None = None
    else:
        before_behavior_passed = _run_python_file(
            ordered_before,
            intended_path,
            root / "before",
        )
        after_behavior_passed = _run_python_file(
            updated_files,
            intended_path,
            root / "after",
        )
        decoys_preserved = _decoys_preserved(
            ordered_before,
            updated_files,
            intended_path,
        )
    changed_paths = _changed_paths(ordered_before, updated_files)
    failure_class = _failure_class(
        ambiguous=ambiguous,
        decision=decision.decision,
        stop_stage=decision.stop_stage,
        materialization_valid=materialization_valid,
        changed_paths=changed_paths,
        intended_path=intended_path,
        after_behavior_passed=after_behavior_passed,
    )
    return {
        "after_behavior_passed": after_behavior_passed,
        "before_behavior_passed": before_behavior_passed,
        "changed_path_count": len(changed_paths),
        "decision": decision.decision,
        "decoys_preserved": decoys_preserved,
        "emissions_from_current_tape": emissions_from_current_tape,
        "failure_class": failure_class,
        "issue_lossless": b"".join(frame.issue_tokens) == task.issue,
        "materialization_error": materialization_error,
        "materialization_valid": materialization_valid,
        "neutral_checkpoint_retained": neutral_checkpoint_retained,
        "physical_cell_steps": decision.physical_cell_steps,
        "query_count": 1,
        "query_sha256": decision.query_sha256,
        "related_path_count": len(related_paths),
        "source_tape_sha256": _event_tape_sha256(frame.workspace_tape),
        "stop_reason": decision.stop_reason,
        "stop_stage": decision.stop_stage,
        "task_id": task.task_id,
        "workspace_lossless": transport.decode_canonical_workspace_tape(
            frame.workspace_tape
        )
        == ordered_before,
    }


def _layout_shape(source: bytes) -> bytes:
    return _IDENTIFIER.sub(b"<identifier>", source)


def _source_symbols(
    files: tuple[transport.WorkspaceFile, ...],
) -> frozenset[bytes]:
    symbols: set[bytes] = set()
    for item in files:
        matches = _CALL_SYMBOL.findall(item.source)
        if len(matches) != 1:
            raise RuntimeError("fixture source must contain one observable call symbol")
        symbols.add(matches[0])
    return frozenset(symbols)


def _matched_multi_source_fixture(
    files: tuple[transport.WorkspaceFile, ...],
) -> bool:
    return (
        3 <= len(files) <= 5
        and all(item.path.endswith("_worker.py") for item in files)
        and len({_layout_shape(item.source) for item in files}) == 1
        and len(_source_symbols(files)) == len(files)
    )


def _training_behavior_is_observable(
    repository: _SuccessfulTrainingRepository,
    root: Path,
) -> bool:
    related_paths = _issue_related_paths(repository.issue, repository.before)
    if len(related_paths) != 1:
        return False
    path = related_paths[0]
    return (
        not _run_python_file(repository.before, path, root / "before")
        and _run_python_file(repository.after, path, root / "after")
        and _decoys_preserved(repository.before, repository.after, path)
    )


def _event_tape_sha256(events: tuple[bytes, ...]) -> str:
    encoded = json.dumps(
        tuple(event.hex() for event in events),
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _record_sha256(record: dict[str, object]) -> str:
    return sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_file_sha256(module_file: str | None) -> str:
    if module_file is None:
        raise RuntimeError("source dependency has no filesystem path")
    return sha256(Path(module_file).read_bytes()).hexdigest()


def run_probe(root: Path) -> dict[str, object]:
    """Run the fixed cohort once and return evidence plus failure localization."""

    training = _training_repositories()
    tasks = public_train_tasks()
    dependency_custody = {
        "raw_source_motor_controller": _source_file_sha256(controller_source.__file__),
        "raw_tape_role_binder": _source_file_sha256(binder_source.__file__),
        "source_edit_transport": _source_file_sha256(transport.__file__),
    }
    controller = _controller()
    initial_checkpoint = controller.checkpoint_sha256
    training_behavior = tuple(
        _training_behavior_is_observable(
            repository,
            root / "training" / repository.provenance_id,
        )
        for repository in training
    )
    records = tuple(
        _run_task(controller, task, root / "fresh" / task.task_id) for task in tasks
    )

    action_records = tuple(
        record for record in records if record["related_path_count"] == 1
    )
    ambiguous_records = tuple(
        record for record in records if record["related_path_count"] != 1
    )
    failure_distribution = dict(
        sorted(Counter(str(record["failure_class"]) for record in records).items())
    )
    training_symbols = frozenset().union(
        *(_source_symbols(repository.before) for repository in training)
    )
    fresh_symbols = frozenset().union(*(_source_symbols(task.files) for task in tasks))
    integrity_checks = {
        "all_emissions_have_current_tape_provenance": all(
            bool(record["emissions_from_current_tape"]) for record in records
        ),
        "all_fresh_frames_are_lossless": all(
            bool(record["issue_lossless"]) and bool(record["workspace_lossless"])
            for record in records
        ),
        "dependency_custody_matches_freeze": (
            dependency_custody == _EXPECTED_DEPENDENCY_CUSTODY
        ),
        "fresh_and_training_symbols_are_disjoint": training_symbols.isdisjoint(
            fresh_symbols
        ),
        "fresh_named_programs_fail_before": all(
            record["before_behavior_passed"] is False for record in action_records
        ),
        "genuine_two_way_ambiguity": len(ambiguous_records) == 1
        and ambiguous_records[0]["related_path_count"] == 2,
        "matched_multi_source_layouts": all(
            _matched_multi_source_fixture(repository.before)
            for repository in training
        )
        and all(_matched_multi_source_fixture(task.files) for task in tasks),
        "neutral_consequences_retain_checkpoint": all(
            bool(record["neutral_checkpoint_retained"]) for record in records
        )
        and controller.checkpoint_sha256 == initial_checkpoint,
        "one_controller_query_per_task": all(
            record["query_count"] == 1 for record in records
        ),
        "successful_training_edits_are_observable": all(training_behavior),
    }
    capability_checks = {
        "all_named_relations_grounded_and_executed": all(
            record["failure_class"] == "correct_action" for record in action_records
        ),
        "genuine_ambiguity_stopped": len(ambiguous_records) == 1
        and ambiguous_records[0]["decision"] == "STOP",
    }
    scientific_outcome = (
        OUTCOME_TRANSFER if all(capability_checks.values()) else OUTCOME_LOCALIZED
    )
    result: dict[str, object] = {
        "capability_checks": capability_checks,
        "claim_boundary": (
            "fresh public-TRAIN multi-source issue-named-symbol grounding among "
            "matched same-layout source decoys; no continual learning, operation "
            "induction, or broad repository intelligence"
        ),
        "dependency_custody": dependency_custody,
        "failure_distribution": failure_distribution,
        "fresh_task_count": len(tasks),
        "integrity_checks": integrity_checks,
        "missing_computation": (
            "learned issue-token-to-workspace-token identity binding followed by "
            "enclosing-file/path-role propagation; the current controller selects an "
            "exact context schema and ranks independent source-token neighborhoods"
        ),
        "public_inference_fields": tuple(PublicInferenceFrame.__dataclass_fields__),
        "scientific_outcome": scientific_outcome,
        "status": STATUS_PASS if all(integrity_checks.values()) else STATUS_INVALID,
        "task_records": records,
        "training_repository_count": len(training),
        "training_summary": asdict(controller.training_summary),
    }
    result["result_sha256"] = _record_sha256(result)
    return result


def main() -> None:
    with TemporaryDirectory(prefix="humanbrain-multi-source-grounding-") as directory:
        result = run_probe(Path(directory))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
