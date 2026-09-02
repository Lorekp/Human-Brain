"""Public-TRAIN cue-free neuronal Python source-edit pilot.

Two successful public repositories train a raw-source motor controller on one
lossless whole-workspace tape each.  Every fresh task is then presented once as
all generic issue tokens plus one canonical tape containing every sorted file.
The controller must emit a live path token followed by a complete replacement
source tape, or physically STOP.  A generic adapter applies exactly that one
neuronal output before real Python execution.

The pilot intentionally targets one source file among non-source repository
evidence.  Exact lexical context, teacher-visible successful before/after
experiences, and compiler-produced motor trajectories remain scaffolding.  A
pass demonstrates renamed same-layout motor-program transfer grounded in source
rewriting, not operation induction, continual learning, or broad repository
intelligence.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from humanbrain.raw_source_motor_controller import (
    RawSourceMotorAblation,
    RawSourceMotorController,
    SuccessfulSourceMotorExperience,
)
from humanbrain.source_relative_action_tape import TapeValence

__all__ = [
    "PublicPythonRepositoryTask",
    "WorkspaceFile",
    "apply_emitted_source_action",
    "canonical_workspace_tape",
    "decode_canonical_workspace_tape",
    "public_train_tasks",
    "run_pilot",
    "tokenize_issue_losslessly",
]


STATUS_PASS = "PASS_CUE_FREE_PYTHON_SOURCE_EDIT_PILOT_V1"
STATUS_STOP = "STOP_CUE_FREE_PYTHON_SOURCE_EDIT_PILOT_V1"

_CONTEXT_EVENT = b"prerequisite"
_EXPECTED_PYTHON_STDOUT = b"ready"
_ISSUE_TOKEN = re.compile(rb"[A-Za-z0-9_]+|[^A-Za-z0-9_]+")

_WORKSPACE_START = b"\x00workspace:start"
_WORKSPACE_END = b"\x00workspace:end"
_FILE_START = b"\x00file:start"
_FILE_END = b"\x00file:end"
_PATH_TAG = b"\x00path"
_CONTENT_TAG = b"\x00content"
_PATH_STEM_PREFIX = b"\x01"
_PATH_SUFFIX_PREFIX = b"\x02"
_SOURCE_PREFIX = b"\x03"


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    path: str
    source: bytes

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.path)
        if (
            not isinstance(self.path, str)
            or not self.path
            or normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != self.path
        ):
            raise ValueError("path must be a canonical relative POSIX path")
        if not isinstance(self.source, bytes):
            raise TypeError("source must be bytes")


@dataclass(frozen=True, slots=True)
class PublicPythonRepositoryTask:
    task_id: str
    issue: bytes
    files: tuple[WorkspaceFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.issue, bytes) or not self.issue:
            raise ValueError("issue must be non-empty bytes")
        if type(self.files) is not tuple or not self.files:
            raise ValueError("files must be a non-empty tuple")
        if any(not isinstance(item, WorkspaceFile) for item in self.files):
            raise TypeError("files must contain WorkspaceFile values")
        paths = tuple(item.path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("workspace paths must be unique")


@dataclass(frozen=True, slots=True)
class _SuccessfulTrainingRepository:
    provenance_id: str
    before: tuple[WorkspaceFile, ...]
    after: tuple[WorkspaceFile, ...]


def _path_tokens(path: str) -> tuple[bytes, bytes]:
    suffix = PurePosixPath(path).suffix
    stem = path[: -len(suffix)] if suffix else path
    return (
        _PATH_STEM_PREFIX + stem.encode("utf-8"),
        _PATH_SUFFIX_PREFIX + suffix.encode("utf-8"),
    )


def _decode_path_tokens(stem_token: bytes, suffix_token: bytes) -> str:
    if not stem_token.startswith(_PATH_STEM_PREFIX) or len(stem_token) == 1:
        raise ValueError("workspace path stem token changed")
    if not suffix_token.startswith(_PATH_SUFFIX_PREFIX):
        raise ValueError("workspace path suffix token changed")
    path = stem_token[1:].decode("utf-8") + suffix_token[1:].decode("utf-8")
    if _path_tokens(path) != (stem_token, suffix_token):
        raise ValueError("workspace path tokenization is not canonical")
    return path


def _source_tokens(source: bytes) -> tuple[bytes, ...]:
    tokens = tuple(_SOURCE_PREFIX + line for line in source.splitlines(keepends=True))
    if b"".join(token[len(_SOURCE_PREFIX) :] for token in tokens) != source:
        raise RuntimeError("physical-line tokenization ceased to be lossless")
    return tokens


def tokenize_issue_losslessly(issue: bytes) -> tuple[bytes, ...]:
    """Return every opaque issue token without dropping whitespace or punctuation."""

    if not isinstance(issue, bytes) or not issue:
        raise ValueError("issue must be non-empty bytes")
    tokens = tuple(_ISSUE_TOKEN.findall(issue))
    if not tokens or b"".join(tokens) != issue:
        raise RuntimeError("issue tokenization ceased to be lossless")
    return tokens


def canonical_workspace_tape(files: tuple[WorkspaceFile, ...]) -> tuple[bytes, ...]:
    """Encode every sorted workspace file into one complete lossless tape."""

    if type(files) is not tuple or not files:
        raise ValueError("files must be a non-empty tuple")
    if any(not isinstance(item, WorkspaceFile) for item in files):
        raise TypeError("files must contain WorkspaceFile values")
    ordered = tuple(sorted(files, key=lambda item: item.path))
    if len({item.path for item in ordered}) != len(ordered):
        raise ValueError("workspace paths must be unique")

    tape: list[bytes] = [_WORKSPACE_START]
    for item in ordered:
        tape.extend(
            (
                _FILE_START,
                _PATH_TAG,
                *_path_tokens(item.path),
                _CONTENT_TAG,
                *_source_tokens(item.source),
                _FILE_END,
            )
        )
    tape.append(_WORKSPACE_END)
    encoded = tuple(tape)
    if decode_canonical_workspace_tape(encoded) != ordered:
        raise RuntimeError("canonical workspace tape ceased to be lossless")
    return encoded


def decode_canonical_workspace_tape(
    tape: tuple[bytes, ...],
) -> tuple[WorkspaceFile, ...]:
    """Decode the generic workspace transport; no edit decision is made here."""

    if type(tape) is not tuple or len(tape) < 2:
        raise ValueError("workspace tape must be a nonempty tuple")
    if tape[0] != _WORKSPACE_START or tape[-1] != _WORKSPACE_END:
        raise ValueError("workspace tape boundary changed")

    files: list[WorkspaceFile] = []
    cursor = 1
    while cursor < len(tape) - 1:
        if tape[cursor : cursor + 2] != (_FILE_START, _PATH_TAG):
            raise ValueError("workspace file header changed")
        cursor += 2
        path = _decode_path_tokens(tape[cursor], tape[cursor + 1])
        cursor += 2
        if tape[cursor] != _CONTENT_TAG:
            raise ValueError("workspace content tag changed")
        cursor += 1

        source_parts: list[bytes] = []
        while cursor < len(tape) - 1 and tape[cursor] != _FILE_END:
            token = tape[cursor]
            if not token.startswith(_SOURCE_PREFIX):
                raise ValueError("workspace source token changed")
            source_parts.append(token[1:])
            cursor += 1
        if cursor >= len(tape) - 1:
            raise ValueError("workspace file end is missing")
        cursor += 1
        files.append(WorkspaceFile(path, b"".join(source_parts)))

    ordered = tuple(files)
    if tuple(item.path for item in ordered) != tuple(
        sorted(item.path for item in ordered)
    ):
        raise ValueError("workspace file order is not canonical")
    return ordered


def apply_emitted_source_action(
    files: tuple[WorkspaceFile, ...],
    emissions: tuple[bytes, ...],
) -> tuple[WorkspaceFile, ...]:
    """Apply one neuronal ``(live path, complete source tape...)`` action."""

    if type(emissions) is not tuple or len(emissions) < 3:
        raise ValueError(
            "emissions must contain a complete live path and at least one source token"
        )
    path = _decode_path_tokens(emissions[0], emissions[1])
    if any(not token.startswith(_SOURCE_PREFIX) for token in emissions[2:]):
        raise ValueError("emitted replacement contains a non-source token")
    replacement = b"".join(token[1:] for token in emissions[2:])
    if not replacement:
        raise ValueError("emitted replacement must contain non-empty source")

    matches = sum(item.path == path for item in files)
    if matches != 1:
        raise ValueError("emitted path is not one unique live workspace file")
    return tuple(
        sorted(
            (
                WorkspaceFile(item.path, replacement) if item.path == path else item
                for item in files
            ),
            key=lambda item: item.path,
        )
    )


def _buggy_source(name: str) -> bytes:
    return (
        f'print({name}(), end="")\n\n\ndef {name}():\n    return "ready"\n'.encode(
            "ascii"
        )
    )


def _correct_source(name: str) -> bytes:
    return (
        f'def {name}():\n    return "ready"\nprint({name}(), end="")\n\n\n'.encode(
            "ascii"
        )
    )


def _training_repositories() -> tuple[_SuccessfulTrainingRepository, ...]:
    first_before = (
        WorkspaceFile("notes/readme.md", b"public training repository alpha\n"),
        WorkspaceFile("alpha_entry.py", _buggy_source("prime_alpha")),
    )
    first_after = (
        WorkspaceFile("notes/readme.md", b"public training repository alpha\n"),
        WorkspaceFile("alpha_entry.py", _correct_source("prime_alpha")),
    )
    second_before = (
        WorkspaceFile("zulu_worker.py", _buggy_source("stage_zulu")),
        WorkspaceFile("config/settings.toml", b"mode = \"public-train\"\n"),
        WorkspaceFile("guide.txt", b"second successful repository\n"),
    )
    second_after = (
        WorkspaceFile("zulu_worker.py", _correct_source("stage_zulu")),
        WorkspaceFile("config/settings.toml", b"mode = \"public-train\"\n"),
        WorkspaceFile("guide.txt", b"second successful repository\n"),
    )
    return (
        _SuccessfulTrainingRepository(
            "public-success-alpha",
            first_before,
            first_after,
        ),
        _SuccessfulTrainingRepository(
            "public-success-zulu",
            second_before,
            second_after,
        ),
    )


def _experience(
    repository: _SuccessfulTrainingRepository,
) -> SuccessfulSourceMotorExperience:
    before = {item.path: item.source for item in repository.before}
    after = {item.path: item.source for item in repository.after}
    if before.keys() != after.keys():
        raise RuntimeError("successful training repository path set changed")
    changed = tuple(
        path for path in sorted(before) if before[path] != after[path]
    )
    if len(changed) != 1:
        raise RuntimeError("pilot training repositories must have one observed edit")
    path = changed[0]
    after_action = (*_path_tokens(path), *_source_tokens(after[path]))
    return SuccessfulSourceMotorExperience(
        context_event=_CONTEXT_EVENT,
        before_tokens=canonical_workspace_tape(repository.before),
        after_tokens=after_action,
        provenance_id=repository.provenance_id,
    )


def _controller() -> RawSourceMotorController:
    return RawSourceMotorController.grow(
        tuple(_experience(repository) for repository in _training_repositories())
    )


def public_train_tasks() -> tuple[PublicPythonRepositoryTask, ...]:
    """Return the short fresh TRAIN-only stream; no fresh after source exists."""

    return (
        PublicPythonRepositoryTask(
            "fresh-action-a",
            b"the prerequisite runs before it exists; reorder existing source only.",
            (
                WorkspaceFile("readme.md", b"fresh renamed alpha repository\n"),
                WorkspaceFile("a_runtime.py", _buggy_source("ignite_aurora")),
            ),
        ),
        PublicPythonRepositoryTask(
            "fresh-action-b",
            b"distractor documents are healthy, but the prerequisite is used too early.",
            (
                WorkspaceFile("m_service.py", _buggy_source("prepare_cobalt")),
                WorkspaceFile("a_notes.txt", b"nothing executable lives here\n"),
                WorkspaceFile("z_config.toml", b"enabled = true\n"),
            ),
        ),
        PublicPythonRepositoryTask(
            "fresh-action-c",
            b"preserve every line while moving the prerequisite before its first use.",
            (
                WorkspaceFile("docs/design.md", b"layout distractor one\n\ntext\n"),
                WorkspaceFile("assets/data.json", b"{\"kind\": \"distractor\"}\n"),
                WorkspaceFile("zeta/worker.py", _buggy_source("wire_deneb")),
                WorkspaceFile("zz_notes.txt", b"layout distractor two\n"),
            ),
        ),
        PublicPythonRepositoryTask(
            "fresh-ambiguous-stop",
            b"two equally plausible prerequisite reorderings are present; do not guess.",
            (
                WorkspaceFile("guide.md", b"ambiguity is intentional\n"),
                WorkspaceFile("alpha_option.py", _buggy_source("open_ember")),
                WorkspaceFile("zeta_option.py", _buggy_source("open_fjord")),
            ),
        ),
    )


def _materialize_workspace(files: tuple[WorkspaceFile, ...], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.source)


def _run_python_behavior(files: tuple[WorkspaceFile, ...], root: Path) -> bool:
    _materialize_workspace(files, root)
    python_paths = tuple(
        item.path for item in sorted(files, key=lambda item: item.path) if item.path.endswith(".py")
    )
    if not python_paths:
        return False
    for path in python_paths:
        execution = subprocess.run(
            [sys.executable, str(root.joinpath(*PurePosixPath(path).parts))],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if execution.returncode != 0 or execution.stdout != _EXPECTED_PYTHON_STDOUT:
            return False
    return True


def _run_arm(
    *,
    name: str,
    ablation: RawSourceMotorAblation,
    tasks: tuple[PublicPythonRepositoryTask, ...],
    root: Path,
) -> dict[str, object]:
    controller = _controller()
    initial_checkpoint = controller.checkpoint_sha256
    task_records: list[dict[str, object]] = []
    for task in tasks:
        issue_tokens = tokenize_issue_losslessly(task.issue)
        source_tape = canonical_workspace_tape(task.files)
        before_passed = _run_python_behavior(
            task.files,
            root / name / task.task_id / "before",
        )
        decision = controller.respond(
            issue_tokens,
            source_tape,
            ablation=ablation,
        )
        source_occurrences = Counter(source_tape)
        emitted_occurrences = Counter(decision.emissions)
        emissions_from_current_tape = all(
            count <= source_occurrences[token]
            for token, count in emitted_occurrences.items()
        )
        if decision.decision == "EMIT":
            updated_files = apply_emitted_source_action(task.files, decision.emissions)
        else:
            updated_files = tuple(sorted(task.files, key=lambda item: item.path))
        after_passed = _run_python_behavior(
            updated_files,
            root / name / task.task_id / "after",
        )
        if decision.decision == "EMIT":
            update = controller.resolve(TapeValence.NEUTRAL)
            if update.prior_training_sha256 != update.training_sha256:
                raise RuntimeError("neutral pilot inference changed controller state")

        task_records.append(
            {
                "after_behavior_passed": after_passed,
                "before_behavior_passed": before_passed,
                "decision": decision.decision,
                "emissions_from_current_tape": emissions_from_current_tape,
                "issue_lossless": b"".join(issue_tokens) == task.issue,
                "issue_token_count": len(issue_tokens),
                "physical_cell_steps": decision.physical_cell_steps,
                "query_count": 1,
                "source_tape_sha256": sha256(b"".join(source_tape)).hexdigest(),
                "source_token_count": len(source_tape),
                "stop_reason": decision.stop_reason,
                "stop_stage": decision.stop_stage,
                "task_id": task.task_id,
                "workspace_changed": updated_files
                != tuple(sorted(task.files, key=lambda item: item.path)),
                "workspace_lossless": decode_canonical_workspace_tape(source_tape)
                == tuple(sorted(task.files, key=lambda item: item.path)),
            }
        )

    action_records = tuple(
        record for record in task_records if record["task_id"] != "fresh-ambiguous-stop"
    )
    ambiguity = next(
        record for record in task_records if record["task_id"] == "fresh-ambiguous-stop"
    )
    return {
        "action_behavior_passes": sum(
            bool(record["after_behavior_passed"]) for record in action_records
        ),
        "action_pre_edit_failures": sum(
            not bool(record["before_behavior_passed"]) for record in action_records
        ),
        "action_emits": sum(record["decision"] == "EMIT" for record in action_records),
        "all_emissions_from_current_tape": all(
            bool(record["emissions_from_current_tape"]) for record in task_records
        ),
        "ambiguity_stop": ambiguity["decision"] == "STOP",
        "binder_stops": sum(record["stop_stage"] == "binder" for record in task_records),
        "checkpoint_retained": controller.checkpoint_sha256 == initial_checkpoint,
        "single_query_tasks": sum(record["query_count"] == 1 for record in task_records),
        "tape_stops": sum(record["stop_stage"] == "tape" for record in task_records),
        "task_records": tuple(task_records),
        "total_emits": sum(record["decision"] == "EMIT" for record in task_records),
        "total_stops": sum(record["decision"] == "STOP" for record in task_records),
    }


def run_pilot(root: Path) -> dict[str, object]:
    """Run the fresh stream and causal controls without writing an artifact."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    training = _training_repositories()
    training_before_failures = 0
    training_after_passes = 0
    for repository in training:
        training_before_failures += int(
            not _run_python_behavior(
                repository.before,
                root / "training" / repository.provenance_id / "before",
            )
        )
        training_after_passes += int(
            _run_python_behavior(
                repository.after,
                root / "training" / repository.provenance_id / "after",
            )
        )

    tasks = public_train_tasks()
    arms = {
        "treatment": _run_arm(
            name="treatment",
            ablation=RawSourceMotorAblation.NONE,
            tasks=tasks,
            root=root,
        ),
        "binder_off": _run_arm(
            name="binder_off",
            ablation=RawSourceMotorAblation.BINDER_OFF,
            tasks=tasks,
            root=root,
        ),
        "tape_off": _run_arm(
            name="tape_off",
            ablation=RawSourceMotorAblation.TAPE_OFF,
            tasks=tasks,
            root=root,
        ),
    }
    treatment = arms["treatment"]
    binder_off = arms["binder_off"]
    tape_off = arms["tape_off"]
    checks = {
        "two_successful_training_repositories": (
            len(training) == training_before_failures == training_after_passes == 2
        ),
        "three_fresh_actions_fail_then_execute": treatment["action_pre_edit_failures"] == 3
        and treatment["action_behavior_passes"] == 3
        and treatment["action_emits"] == 3,
        "fresh_ambiguity_stops": treatment["ambiguity_stop"]
        and treatment["binder_stops"] == 1,
        "binder_off_is_causal": binder_off["total_emits"] == 0
        and binder_off["binder_stops"] == 4,
        "tape_off_is_causal": tape_off["total_emits"] == 0
        and tape_off["tape_stops"] == 3
        and tape_off["binder_stops"] == 1,
        "one_whole_workspace_query_per_task": all(
            arm["single_query_tasks"] == 4 for arm in arms.values()
        ),
        "all_transport_is_lossless": all(
            record["issue_lossless"] and record["workspace_lossless"]
            for arm in arms.values()
            for record in arm["task_records"]
        ),
        "all_emissions_have_current_tape_provenance": all(
            arm["all_emissions_from_current_tape"] for arm in arms.values()
        ),
        "neutral_stream_retains_checkpoint": all(
            arm["checkpoint_retained"] for arm in arms.values()
        ),
    }
    return {
        "arms": arms,
        "checks": checks,
        "claim_boundary": (
            "renamed same-layout motor-program transfer through grounded whole-workspace "
            "lexical source rewriting with exact context and successful-trajectory "
            "scaffolding; no continual learning or operation induction"
        ),
        "fresh_task_count": len(tasks),
        "status": STATUS_PASS if all(checks.values()) else STATUS_STOP,
        "training_repository_count": len(training),
    }


def main() -> None:
    with TemporaryDirectory(prefix="humanbrain-cue-free-pilot-") as directory:
        result = run_pilot(Path(directory))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
