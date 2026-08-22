"""Preregistered one-shot public DEV gate for typed programs.

Loading or planning never restores a checkpoint or completes a task.  The
production runner opens the fixed public DEV set only after exact raw-byte
authorization and exact TRAIN input validation.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import os
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from humanbrain.typed_curriculum import DevelopmentTask, load_public_curriculum
from humanbrain.typed_program import (
    Completion,
    FrozenCheckpoint,
    complete,
)
from humanbrain.typed_program import (
    compile as compile_program,
)
from humanbrain.typed_program_types import ProgramType

_SCHEMA = "humanbrain-public-typed-program-dev-protocol-v1"
_PROTOCOL_ID = "humanbrain-public-typed-program-dev-v1"
_STATUS = "preregistered-not-executed"
_SOURCE_COMMIT = "1ae5418b92e81dc48c6a2e36fb8822ceec7ce202"
FROZEN_PUBLIC_DEV_GATE_SHA256 = "c7be743617c5fbc090891275f77a636bf1a7a6a89d27a1d1e7cf0e889d19ae56"
_PUBLIC_DEV_PROTOCOL_BYTES = 9_704
_TRAIN_PROTOCOL_FILENAME = "typed_program_public_train_protocol_v1.json"
_TRAIN_ARTIFACT_COMMIT = "8291fe1a0039db7141817c438854a1817a0a8300"
_TRAIN_PROTOCOL_RELATIVE_PATH = Path("benchmarks/typed_program_public_train_protocol_v1.json")
_TRAIN_PROTOCOL_SHA256 = "1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191"
_TRAIN_PROTOCOL_GIT_BLOB = "afdacef6f9d53a1e16b76155d8accc3a1ecf9e46"
_TRAIN_PROTOCOL_BYTES = 7_690
_TRAIN_CHECKPOINT_RELATIVE_PATH = Path("artifacts/typed_program_public_train_v1/checkpoint.json")
_TRAIN_CHECKPOINT_RAW_SHA256 = "0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d"
_TRAIN_CHECKPOINT_CELL_SYNAPSE_SHA256 = (
    "86b40631c49db55124dff3212645dc571555c5939fc4004a5d94c54b49e99779"
)
_TRAIN_CHECKPOINT_GIT_BLOB = "430faa1221c7721d97f43a9c311671e30bb65317"
_TRAIN_CHECKPOINT_BYTES = 986_677
_TRAIN_SUMMARY_RELATIVE_PATH = Path(
    "artifacts/typed_program_public_train_v1/evidence/005-summary.json"
)
_TRAIN_SUMMARY_RAW_SHA256 = "71133044d55e48a3bbe5cb147a33ce2f543e0a0dad855ae11e2e1b140b6b02d7"
_TRAIN_SUMMARY_GIT_BLOB = "5ec6d687ed62aeb674231941c7ac52591d6f980f"
_EVIDENCE_RELATIVE_DIRECTORY = Path("artifacts/typed_program_public_dev_v1/evidence")
_TRAIN_ARTIFACT_FILE_IDENTITIES = (
    (
        _TRAIN_CHECKPOINT_RELATIVE_PATH,
        _TRAIN_CHECKPOINT_RAW_SHA256,
        _TRAIN_CHECKPOINT_GIT_BLOB,
        _TRAIN_CHECKPOINT_BYTES,
    ),
    (
        Path("artifacts/typed_program_public_train_v1/evidence/000-run-start.json"),
        "79dff01017e63072a3d32eb38d3a2435d1cfd9b008eae5501e55612a2be64b1d",
        "41571c9002570095383fdb85f8a036dc364d18de",
        494,
    ),
    (
        Path("artifacts/typed_program_public_train_v1/evidence/001-training.json"),
        "926daebc386eac7e13c77dcb1746d8e8e920f2c7716b6040d10faa9330195daf",
        "da307535dbd6404342a1cd29f401907d2dd67787",
        1_000,
    ),
    (
        Path("artifacts/typed_program_public_train_v1/evidence/002-train_int_identity.json"),
        "6e223fed4fc66c7932ad4f3c1ac0ca590ac8e1ec0ce565c79eb08a8456f8cdeb",
        "1b484988e065ecf9f6e6cdfba90967f741a31d96",
        3_082,
    ),
    (
        Path(
            "artifacts/typed_program_public_train_v1/evidence/003-train_int_add_negative_two.json"
        ),
        "c10f903fc114e988ab6c3a78338bf04bf9531f0c20b8ce61f902664b8b29812d",
        "2e880f281d84ec4f3113ac32ec0753dfaf570342",
        4_928,
    ),
    (
        Path(
            "artifacts/typed_program_public_train_v1/evidence/"
            "004-train_list_map_add_positive_three.json"
        ),
        "9df04197196402cde5762c1222f8f6dd631682b2d30a89234ec69241e8b6cdce",
        "ba64c3e702c9e00e3dc693fb7c3642cd0e64cd1b",
        6_027,
    ),
    (
        _TRAIN_SUMMARY_RELATIVE_PATH,
        _TRAIN_SUMMARY_RAW_SHA256,
        _TRAIN_SUMMARY_GIT_BLOB,
        2_336,
    ),
)
_SOURCE_FILE_IDENTITIES = (
    (
        Path("benchmarks/typed_program_public_train.py"),
        "c45dbc15f55b7ad8a8c4dd971c46096b48f4488f935f962eb503412d1c499bc1",
        "8d8b6aed9958b70a7d5583ffab1f8b61315766c6",
        421,
    ),
    (
        Path("src/humanbrain/typed_curriculum.py"),
        "d171f12aca180f87295468ef282b10d267dddd5f1f8265ba353dc3a1bb4eb8e6",
        "c680b488a5bb2a5e970e244244af5da03d1808c0",
        49_661,
    ),
    (
        Path("src/humanbrain/typed_program.py"),
        "447570e75264bb4419cd725f6a7508f92236088299a4d8fd9742f8ee334a819b",
        "93e1bb1c8d426d3dade24b7cbd5d201243998ba6",
        49_243,
    ),
    (
        Path("src/humanbrain/typed_program_types.py"),
        "5f06a43bc771a8d61881d7ebf7c7decefcadbe245d77ce10640dbfff5174d099",
        "5c94b46ec632702bdf062c3104d4c1bb013e38c5",
        6_527,
    ),
)
_PREREQUISITE_GATES = (
    "development_protocol_identity",
    "train_protocol_identity",
    "train_source_identity",
    "checkpoint_identity",
    "evidence_nonexistence",
    "teacher_free_definition",
)
_TASK_GATES = (
    "fresh_restore_and_attempt_accounting",
    "teacher_free_label_boundary",
    "typed_lowering",
    "syntax",
    "compile",
    "semantics",
    "prompt_identity",
    "signature_identity",
    "checkpoint_raw_immutability",
    "checkpoint_cell_synapse_immutability",
    "cell_race_audit",
    "timing_memory_accounting",
)
_RUN_GATES = (
    "all_tasks_recorded",
    "pass_at_1_accounting",
    "evidence_accounting",
    "timing_memory_accounting",
)
_PRODUCTION_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_PROTOCOL_PATH = (
    _PRODUCTION_REPOSITORY_ROOT / "benchmarks" / "typed_program_public_dev_protocol_v1.json"
)


class ExecutionRefusedError(RuntimeError):
    """The explicit execution authorization did not match the frozen protocol."""


class EvidenceExistsError(RuntimeError):
    """The fixed public DEV evidence root already exists."""


class DevGateError(RuntimeError):
    """The preregistered public DEV promotion milestone was not met."""


def _git_blob_at_commit(
    repository_root: Path,
    commit: str,
    relative_path: Path,
) -> str:
    root = repository_root.resolve()
    result = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            "rev-parse",
            f"{commit}:{relative_path.as_posix()}",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


@dataclass(frozen=True, slots=True)
class _RunDependencies:
    """Complete private adapter set used to keep tests off the DEV runtime."""

    repository_root: Path
    evidence_repository_root: Path
    protocol_path: Path
    complete: Callable[..., Completion]
    restore_checkpoint: Callable[[bytes], object]
    git_blob_at_commit: Callable[[Path, str, Path], str] = _git_blob_at_commit


def _production_dependencies() -> _RunDependencies:
    return _RunDependencies(
        repository_root=_PRODUCTION_REPOSITORY_ROOT,
        evidence_repository_root=_PRODUCTION_REPOSITORY_ROOT,
        protocol_path=_PRODUCTION_PROTOCOL_PATH,
        complete=complete,
        restore_checkpoint=FrozenCheckpoint.from_bytes,
    )


@dataclass(frozen=True, slots=True)
class PublicDevGate:
    protocol_id: str
    raw_sha256: str
    source_commit: str
    tasks: tuple[DevelopmentTask, ...]
    train_teacher_tokens: tuple[tuple[str, ...], ...]
    train_artifact_commit: str
    train_protocol_relative_path: Path
    train_protocol_sha256: str
    train_protocol_git_blob: str
    train_protocol_byte_count: int
    checkpoint_relative_path: Path
    checkpoint_raw_sha256: str
    checkpoint_cell_synapse_sha256: str
    checkpoint_git_blob: str
    checkpoint_byte_count: int
    train_summary_relative_path: Path
    train_summary_raw_sha256: str
    train_summary_git_blob: str
    train_artifact_file_identities: tuple[tuple[Path, str, str, int], ...]
    source_file_identities: tuple[tuple[Path, str, str, int], ...]
    evidence_relative_directory: Path


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _record(value: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _task_record(task: DevelopmentTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "family_id": task.family_id,
        "template_id": task.template_id,
        "prompt": task.prompt,
        "signature": task.signature.to_record(),
        "literals": list(task.literals),
        "semantic_cases": [
            {
                "case_id": case.case_id,
                "arguments": _json_value(case.arguments),
                "expected": _json_value(case.expected),
            }
            for case in task.semantic_cases
        ],
    }


def _read_exact_file_bytes(
    path: Path,
    *,
    expected_bytes: int,
    label: str,
    display_path: Path | None = None,
) -> bytes:
    """Read one committed file with an exact-size plus-one allocation bound."""
    suffix = "" if display_path is None else f": {display_path}"
    try:
        with path.open("rb") as stream:
            raw = stream.read(expected_bytes + 1)
    except OSError as error:
        raise ValueError(f"{label} is unavailable{suffix}") from error
    if len(raw) != expected_bytes:
        raise ValueError(f"{label} size changed{suffix}")
    return raw


def load_public_dev_gate(path: str | Path) -> PublicDevGate:
    """Load the frozen teacher-free DEV definition without executing it."""
    protocol_path = Path(path)
    raw = _read_exact_file_bytes(
        protocol_path,
        expected_bytes=_PUBLIC_DEV_PROTOCOL_BYTES,
        label="public DEV protocol",
    )
    raw_sha256 = sha256(raw).hexdigest()
    if raw_sha256 != FROZEN_PUBLIC_DEV_GATE_SHA256:
        raise ValueError("public DEV protocol raw SHA-256 is not preregistered")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("public DEV protocol is not strict UTF-8 JSON") from error
    root = _record(
        parsed,
        frozenset(
            {
                "schema",
                "protocol_id",
                "status",
                "source_commit",
                "scope",
                "train_inputs",
                "fixed_configuration",
                "development",
                "execution",
                "memorization_control",
                "reporting",
                "expected_cost",
                "prohibitions",
            }
        ),
        "public DEV protocol",
    )
    if (
        root["schema"],
        root["protocol_id"],
        root["status"],
        root["source_commit"],
    ) != (_SCHEMA, _PROTOCOL_ID, _STATUS, _SOURCE_COMMIT):
        raise ValueError("public DEV identity is not the preregistered v1 identity")

    expected_train_inputs = {
        "artifact_commit": _TRAIN_ARTIFACT_COMMIT,
        "protocol": {
            "path": _TRAIN_PROTOCOL_RELATIVE_PATH.as_posix(),
            "raw_sha256": _TRAIN_PROTOCOL_SHA256,
            "git_blob": _TRAIN_PROTOCOL_GIT_BLOB,
        },
        "checkpoint": {
            "path": _TRAIN_CHECKPOINT_RELATIVE_PATH.as_posix(),
            "raw_sha256": _TRAIN_CHECKPOINT_RAW_SHA256,
            "cell_synapse_sha256": _TRAIN_CHECKPOINT_CELL_SYNAPSE_SHA256,
            "git_blob": _TRAIN_CHECKPOINT_GIT_BLOB,
        },
        "train_summary": {
            "path": _TRAIN_SUMMARY_RELATIVE_PATH.as_posix(),
            "raw_sha256": _TRAIN_SUMMARY_RAW_SHA256,
            "git_blob": _TRAIN_SUMMARY_GIT_BLOB,
        },
        "artifact_files": [
            {
                "path": relative_path.as_posix(),
                "raw_sha256": raw_sha256,
                "git_blob": git_blob,
            }
            for relative_path, raw_sha256, git_blob, _expected_bytes in (
                _TRAIN_ARTIFACT_FILE_IDENTITIES
            )
        ],
        "source": {
            "commit": _SOURCE_COMMIT,
            "files": [
                {
                    "path": relative_path.as_posix(),
                    "raw_sha256": raw_sha256,
                    "git_blob": git_blob,
                }
                for relative_path, raw_sha256, git_blob, _expected_bytes in (
                    _SOURCE_FILE_IDENTITIES
                )
            ],
        },
    }
    if root["train_inputs"] != expected_train_inputs:
        raise ValueError("TRAIN input commitments changed")
    if root["scope"] != {
        "operation": "public-development-one-shot",
        "claim": ("two public teacher-free development tasks; no broad generalization claim"),
        "primary_metric": ("exact semantic pass@1 reported as an integer from 0 through 2"),
        "promotion_milestone": (
            "at least 1 of 2 tasks passes semantics with valid syntax and compilation "
            "and no boundary violations"
        ),
    }:
        raise ValueError("public DEV scope commitments changed")
    if root["fixed_configuration"] != {
        "module": "humanbrain.typed_program",
        "completion": "humanbrain.typed_program.complete",
        "cell_path": "recurrent-fitted-cell-only",
        "checkpoint_count": 1,
        "checkpoint_restore_attempts_per_task": 1,
        "completion_attempts_per_task": 1,
        "task_count": 2,
        "seed_options": 0,
        "configuration_options": 0,
        "max_output_tokens": 128,
    }:
        raise ValueError("fixed public DEV configuration commitments changed")

    development = _record(
        root["development"],
        frozenset(
            {
                "partition",
                "teacher_programs_present",
                "completion_opened",
                "run_count",
                "tasks",
            }
        ),
        "development",
    )
    if (
        development["partition"],
        development["teacher_programs_present"],
        development["completion_opened"],
        development["run_count"],
    ) != ("development", False, False, 0):
        raise ValueError("development must remain teacher-free, unopened, and unrun")

    train_protocol_path = protocol_path.with_name(_TRAIN_PROTOCOL_FILENAME)
    train_protocol_bytes = _read_exact_file_bytes(
        train_protocol_path,
        expected_bytes=_TRAIN_PROTOCOL_BYTES,
        label="public TRAIN protocol",
    )
    if sha256(train_protocol_bytes).hexdigest() != _TRAIN_PROTOCOL_SHA256:
        raise ValueError("TRAIN protocol identity changed")
    train = load_public_curriculum(train_protocol_path)
    if train.raw_sha256 != _TRAIN_PROTOCOL_SHA256:
        raise ValueError("TRAIN protocol identity changed")
    expected_tasks = [_task_record(task) for task in train.development_tasks]
    if development["tasks"] != expected_tasks:
        raise ValueError("public DEV tasks differ from the frozen TRAIN protocol definitions")

    execution = _record(
        root["execution"],
        frozenset(
            {
                "authorization",
                "attempts",
                "task_failure_policy",
                "prerequisite_gates",
                "task_gates",
                "run_gates",
            }
        ),
        "execution",
    )
    if (
        tuple(_sequence(execution["prerequisite_gates"], "execution.prerequisite_gates"))
        != _PREREQUISITE_GATES
    ):
        raise ValueError("prerequisite gate commitments changed")
    if tuple(_sequence(execution["task_gates"], "execution.task_gates")) != (_TASK_GATES):
        raise ValueError("task gate commitments changed")
    if tuple(_sequence(execution["run_gates"], "execution.run_gates")) != _RUN_GATES:
        raise ValueError("run gate commitments changed")
    if (
        execution["authorization"],
        execution["attempts"],
        execution["task_failure_policy"],
    ) != (
        "require --execute and the exact raw DEV protocol SHA-256",
        (
            "one fresh checkpoint restore attempt and, after a valid restore, one "
            "completion attempt per task; no retry or replacement"
        ),
        "record every task in protocol order even after an earlier task fails",
    ):
        raise ValueError("execution attempt or authorization commitment changed")
    if root["memorization_control"] != {
        "method": (
            "compare each returned token stream with the three public TRAIN teacher token streams"
        ),
        "model_calls": 0,
        "categories": [
            "exact public TRAIN-label replay",
            "not exact public TRAIN-label replay",
        ],
        "interpretation": (
            "not exact public TRAIN-label replay does not establish broad generalization"
        ),
    }:
        raise ValueError("memorization-control commitments changed")
    if root["reporting"] != {
        "evidence_directory": _EVIDENCE_RELATIVE_DIRECTORY.as_posix(),
        "incremental": True,
        "nonoverwriting": True,
        "all_tasks_recorded": True,
        "report_all_failures": True,
        "pass_at_1_range": [0, 2],
        "promotion_semantic_passes_required": 1,
        "promotion_task_count": 2,
        "measurements": [
            "wall_time_ns",
            "process_time_ns",
            "tracemalloc_current_bytes",
            "tracemalloc_peak_bytes",
            "checkpoint_bytes",
            "evidence_bytes",
            "checkpoint_restore_attempts",
            "completion_attempts",
            "completion_tokens",
            "cell_races",
        ],
    }:
        raise ValueError("reporting commitments changed")
    if root["expected_cost"] != {
        "plan_model_calls": 0,
        "plan_wall_time": "under 1 second",
        "execution_warm_wall_time": "1-3 minutes",
        "execution_cold_wall_time": "up to 10 minutes",
        "process_memory": "under 512 MiB",
        "status": "planning estimates; not measured results",
    }:
        raise ValueError("expected-cost commitments changed")
    if root["prohibitions"] != [
        "teacher programs for development tasks",
        "retraining",
        "tuning",
        "retries",
        "alternate checkpoints",
        "alternate configurations",
        "alternate seeds",
        "search",
        "repair",
        "fallback execution paths",
        "Simple-SWE",
        "canonical evaluation",
        "private evaluation",
        "external language models",
        "additional model calls for the memorization control",
        "development execution before explicit authorization",
    ]:
        raise ValueError("public DEV prohibitions changed")

    return PublicDevGate(
        protocol_id=_PROTOCOL_ID,
        raw_sha256=raw_sha256,
        source_commit=_SOURCE_COMMIT,
        tasks=train.development_tasks,
        train_teacher_tokens=tuple(task.teacher_tokens for task in train.train_tasks),
        train_artifact_commit=_TRAIN_ARTIFACT_COMMIT,
        train_protocol_relative_path=_TRAIN_PROTOCOL_RELATIVE_PATH,
        train_protocol_sha256=_TRAIN_PROTOCOL_SHA256,
        train_protocol_git_blob=_TRAIN_PROTOCOL_GIT_BLOB,
        train_protocol_byte_count=_TRAIN_PROTOCOL_BYTES,
        checkpoint_relative_path=_TRAIN_CHECKPOINT_RELATIVE_PATH,
        checkpoint_raw_sha256=_TRAIN_CHECKPOINT_RAW_SHA256,
        checkpoint_cell_synapse_sha256=_TRAIN_CHECKPOINT_CELL_SYNAPSE_SHA256,
        checkpoint_git_blob=_TRAIN_CHECKPOINT_GIT_BLOB,
        checkpoint_byte_count=_TRAIN_CHECKPOINT_BYTES,
        train_summary_relative_path=_TRAIN_SUMMARY_RELATIVE_PATH,
        train_summary_raw_sha256=_TRAIN_SUMMARY_RAW_SHA256,
        train_summary_git_blob=_TRAIN_SUMMARY_GIT_BLOB,
        train_artifact_file_identities=_TRAIN_ARTIFACT_FILE_IDENTITIES,
        source_file_identities=_SOURCE_FILE_IDENTITIES,
        evidence_relative_directory=_EVIDENCE_RELATIVE_DIRECTORY,
    )


def _checked_file_bytes(
    *,
    dependencies: _RunDependencies,
    relative_path: Path,
    expected_commit: str,
    expected_raw_sha256: str,
    expected_git_blob: str,
    expected_bytes: int,
) -> bytes:
    path = dependencies.repository_root / relative_path
    raw = _read_exact_file_bytes(
        path,
        expected_bytes=expected_bytes,
        label="checked TRAIN input",
        display_path=relative_path,
    )
    if sha256(raw).hexdigest() != expected_raw_sha256:
        raise ValueError(f"checked TRAIN input bytes changed: {relative_path}")
    try:
        actual_blob = dependencies.git_blob_at_commit(
            dependencies.repository_root,
            expected_commit,
            relative_path,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"checked TRAIN Git blob is unavailable: {relative_path}") from error
    if actual_blob != expected_git_blob:
        raise ValueError(f"checked TRAIN Git blob changed: {relative_path}")
    return raw


def _validate_train_summary(gate: PublicDevGate, raw: bytes) -> None:
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checked TRAIN summary is not strict UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("checked TRAIN summary is not a record")
    checkpoint = parsed.get("checkpoint")
    task_results = parsed.get("task_results")
    if not isinstance(checkpoint, Mapping) or not isinstance(task_results, list):
        raise ValueError("checked TRAIN summary has invalid result records")
    if (
        parsed.get("all_train_gates_passed") is not True
        or parsed.get("protocol_id") != "humanbrain-public-typed-program-train-v1"
        or parsed.get("protocol_sha256") != gate.train_protocol_sha256
        or parsed.get("development_completion_opened") is not False
        or parsed.get("development_run_count") != 0
        or parsed.get("completion_attempts") != 3
        or parsed.get("train_task_count") != 3
        or checkpoint.get("raw_sha256") != gate.checkpoint_raw_sha256
        or checkpoint.get("sha256") != gate.checkpoint_raw_sha256
        or checkpoint.get("cell_synapse_sha256") != gate.checkpoint_cell_synapse_sha256
        or [task.get("task_id") for task in task_results if isinstance(task, Mapping)]
        != [
            "train_int_identity",
            "train_int_add_negative_two",
            "train_list_map_add_positive_three",
        ]
        or not all(
            isinstance(task, Mapping) and task.get("all_gates_passed") is True
            for task in task_results
        )
    ):
        raise ValueError("checked TRAIN summary does not authorize the DEV gate")


def _checkpoint_cell_synapse_sha256(raw: bytes) -> str:
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checked TRAIN checkpoint is not strict UTF-8 JSON") from error
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("circuit"), Mapping):
        raise ValueError("checked TRAIN checkpoint has no circuit record")
    value = parsed["circuit"].get("sha256")
    if not isinstance(value, str):
        raise ValueError("checked TRAIN checkpoint has no cell-state SHA-256")
    return value


def _preflight_train_inputs(
    gate: PublicDevGate,
    dependencies: _RunDependencies,
) -> bytes:
    expected_protocol_path = dependencies.repository_root / Path(
        "benchmarks/typed_program_public_dev_protocol_v1.json"
    )
    if dependencies.protocol_path.resolve() != expected_protocol_path.resolve():
        raise ValueError("public DEV protocol path is not fixed to the checked source root")
    train_path = dependencies.repository_root / gate.train_protocol_relative_path
    if (
        train_path.resolve()
        != dependencies.protocol_path.with_name(_TRAIN_PROTOCOL_FILENAME).resolve()
    ):
        raise ValueError("TRAIN protocol path is not fixed to the checked source root")
    _checked_file_bytes(
        dependencies=dependencies,
        relative_path=gate.train_protocol_relative_path,
        expected_commit=gate.train_artifact_commit,
        expected_raw_sha256=gate.train_protocol_sha256,
        expected_git_blob=gate.train_protocol_git_blob,
        expected_bytes=gate.train_protocol_byte_count,
    )
    if load_public_curriculum(train_path).raw_sha256 != gate.train_protocol_sha256:
        raise ValueError("TRAIN protocol identity changed")
    for (
        relative_path,
        expected_sha256,
        expected_git_blob,
        expected_bytes,
    ) in gate.source_file_identities:
        _checked_file_bytes(
            dependencies=dependencies,
            relative_path=relative_path,
            expected_commit=gate.source_commit,
            expected_raw_sha256=expected_sha256,
            expected_git_blob=expected_git_blob,
            expected_bytes=expected_bytes,
        )
    artifact_bytes: dict[Path, bytes] = {}
    for (
        relative_path,
        expected_sha256,
        expected_git_blob,
        expected_bytes,
    ) in gate.train_artifact_file_identities:
        artifact_bytes[relative_path] = _checked_file_bytes(
            dependencies=dependencies,
            relative_path=relative_path,
            expected_commit=gate.train_artifact_commit,
            expected_raw_sha256=expected_sha256,
            expected_git_blob=expected_git_blob,
            expected_bytes=expected_bytes,
        )
    checkpoint_bytes = artifact_bytes[gate.checkpoint_relative_path]
    if _checkpoint_cell_synapse_sha256(checkpoint_bytes) != (gate.checkpoint_cell_synapse_sha256):
        raise ValueError("checked TRAIN checkpoint cell-state identity changed")
    summary_bytes = artifact_bytes[gate.train_summary_relative_path]
    _validate_train_summary(gate, summary_bytes)
    return checkpoint_bytes


def _evidence_directory(
    gate: PublicDevGate,
    dependencies: _RunDependencies,
) -> Path:
    return dependencies.evidence_repository_root / gate.evidence_relative_directory


def _reject_existing_evidence(evidence_directory: Path) -> None:
    artifact_root = evidence_directory.parent
    if artifact_root.exists():
        raise EvidenceExistsError(f"public DEV evidence root already exists: {artifact_root}")


def _plan(gate: PublicDevGate) -> dict[str, object]:
    return {
        "checkpoint": {
            "cell_synapse_sha256": gate.checkpoint_cell_synapse_sha256,
            "git_blob": gate.checkpoint_git_blob,
            "path": gate.checkpoint_relative_path.as_posix(),
            "raw_sha256": gate.checkpoint_raw_sha256,
        },
        "checkpoint_restore_attempts": len(gate.tasks),
        "completion_attempts": len(gate.tasks),
        "development_semantic_case_count": sum(len(task.semantic_cases) for task in gate.tasks),
        "development_task_count": len(gate.tasks),
        "development_task_ids": [task.task_id for task in gate.tasks],
        "evidence_directory": gate.evidence_relative_directory.as_posix(),
        "execute": False,
        "expected_execution": {
            "cold_wall_time": "up to 10 minutes",
            "process_memory": "under 512 MiB",
            "warm_wall_time": "1-3 minutes",
        },
        "mode": "plan",
        "model_calls": 0,
        "pass_at_1_range": [0, len(gate.tasks)],
        "promotion": {
            "semantic_passes_required": 1,
            "task_count": len(gate.tasks),
            "requires_no_boundary_violations": True,
            "requires_syntax_and_compile": True,
        },
        "protocol_id": gate.protocol_id,
        "protocol_sha256": gate.raw_sha256,
        "source": {
            "commit": gate.source_commit,
            "files": [
                {
                    "git_blob": git_blob,
                    "path": relative_path.as_posix(),
                    "raw_sha256": raw_sha256,
                }
                for relative_path, raw_sha256, git_blob, _expected_bytes in (
                    gate.source_file_identities
                )
            ],
        },
        "train_artifact_commit": gate.train_artifact_commit,
        "train_artifact_file_count": len(gate.train_artifact_file_identities),
        "train_protocol": {
            "git_blob": gate.train_protocol_git_blob,
            "path": gate.train_protocol_relative_path.as_posix(),
            "raw_sha256": gate.train_protocol_sha256,
        },
        "train_summary": {
            "git_blob": gate.train_summary_git_blob,
            "path": gate.train_summary_relative_path.as_posix(),
            "raw_sha256": gate.train_summary_raw_sha256,
        },
    }


def _canonical_evidence(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_file(path: Path, data: bytes) -> None:
    if type(data) is not bytes:
        raise TypeError("atomic evidence payload must be bytes")
    if path.exists():
        raise EvidenceExistsError(f"evidence target already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise EvidenceExistsError(f"evidence target already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_evidence(
    evidence_directory: Path,
    ordinal: int,
    label: str,
    body: Mapping[str, object],
) -> Path:
    path = evidence_directory / f"{ordinal:03d}-{label}.json"
    _write_new_file(path, _canonical_evidence(body))
    return path


def _evidence_byte_count(evidence_directory: Path) -> int:
    return sum(path.stat().st_size for path in evidence_directory.glob("*.json"))


def _gate(passed: bool, detail: str) -> dict[str, object]:
    return {
        "detail": detail,
        "passed": passed,
        "status": "passed" if passed else "failed",
    }


def _not_run(detail: str) -> dict[str, object]:
    return {"detail": detail, "passed": False, "status": "not_run"}


def _materialize_semantic(value: object, program_type: ProgramType) -> object:
    if program_type == ProgramType.Int():
        return value
    if program_type.constructor == "List":
        if not isinstance(value, tuple):
            raise ValueError("frozen List semantic value is not immutable")
        return [_materialize_semantic(item, program_type.arguments[0]) for item in value]
    raise ValueError("public DEV gate supports only Int and List semantic values")


def _sha256_is_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _race_audit(completion: Completion) -> tuple[bool, list[dict[str, object]]]:
    reports: list[dict[str, object]] = []
    all_passed = completion.race_count == len(completion.program.tokens)
    for index, (token, race) in enumerate(
        zip(completion.program.tokens, completion.races, strict=False)
    ):
        expected_prefix_sha256 = sha256(
            json.dumps(
                list(completion.program.tokens[:index]),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        allowed = set(race.allowed_cell_ids)
        excluded = set(race.excluded_cell_ids)
        passed = (
            race.accepted_token == token
            and race.winner_cell_id in allowed
            and race.winner_cell_id not in excluded
            and not (allowed & excluded)
            and race.prefix_sha256 == expected_prefix_sha256
            and type(race.simulated_steps) is int
            and race.simulated_steps > 0
            and math.isfinite(race.spike_time_ms)
            and race.spike_time_ms >= 0.0
            and _sha256_is_hex(race.output_current_sha256)
        )
        all_passed = all_passed and passed
        reports.append(
            {
                "accepted_token": race.accepted_token,
                "allowed_cell_ids": list(race.allowed_cell_ids),
                "excluded_cell_ids": list(race.excluded_cell_ids),
                "index": index,
                "output_current_sha256": race.output_current_sha256,
                "passed": passed,
                "prefix_sha256": race.prefix_sha256,
                "simulated_steps": race.simulated_steps,
                "spike_time_ms": race.spike_time_ms,
                "winner_cell_id": race.winner_cell_id,
            }
        )
    return all_passed, reports


def _timing_accounting_gate(
    measurements: Mapping[str, object],
    *,
    run_summary: bool,
) -> dict[str, object]:
    required = {
        "checkpoint_restore_attempts",
        "completion_attempts",
        "completion_tokens",
        "cell_races",
        "process_time_ns",
        "tracemalloc_current_bytes",
        "tracemalloc_peak_bytes",
        "wall_time_ns",
    }
    if run_summary:
        required.update({"checkpoint_bytes", "evidence_bytes"})
    values_are_counts = all(
        type(measurements.get(name)) is int and int(measurements[name]) >= 0 for name in required
    )
    peak_covers_current = (
        type(measurements.get("tracemalloc_peak_bytes")) is int
        and type(measurements.get("tracemalloc_current_bytes")) is int
        and int(measurements["tracemalloc_peak_bytes"])
        >= int(measurements["tracemalloc_current_bytes"])
    )
    return _gate(
        values_are_counts and peak_covers_current,
        "all preregistered timing, memory, and accounting counters are valid",
    )


_OUTCOME_GATES = frozenset({"syntax", "compile", "semantics"})
_BOUNDARY_GATES = frozenset(
    {
        "fresh_restore_and_attempt_accounting",
        "teacher_free_label_boundary",
        "prompt_identity",
        "signature_identity",
        "checkpoint_raw_immutability",
        "checkpoint_cell_synapse_immutability",
        "cell_race_audit",
        "timing_memory_accounting",
    }
)


def _empty_result(error: Exception) -> dict[str, object]:
    gates = {
        name: _not_run("completion output was unavailable")
        for name in _TASK_GATES
        if name not in {"fresh_restore_and_attempt_accounting", "timing_memory_accounting"}
    }
    return {
        "all_gates_passed": False,
        "attempt_error": f"{type(error).__name__}: {error}",
        "boundary_violations": [],
        "compiled_source": None,
        "compiled_source_sha256": None,
        "gates": gates,
        "memorization_control": {
            "category": "unavailable",
            "exact_train_label_match": None,
            "matched_train_task_ids": [],
            "model_calls": 0,
            "note": "this control does not establish broad generalization",
        },
        "race_count": 0,
        "races": [],
        "semantic_cases": [],
        "tokens": [],
    }


def _audit_completion(
    *,
    gate: PublicDevGate,
    task: DevelopmentTask,
    completion: Completion,
    checkpoint: object,
    checkpoint_bytes: bytes,
    checkpoint_path: Path,
) -> dict[str, object]:
    train_task_ids = (
        "train_int_identity",
        "train_int_add_negative_two",
        "train_list_map_add_positive_three",
    )
    matches = [
        task_id
        for task_id, teacher_tokens in zip(
            train_task_ids,
            gate.train_teacher_tokens,
            strict=True,
        )
        if completion.program.tokens == teacher_tokens
    ]
    exact_train_label_match = bool(matches)
    memorization = {
        "category": (
            "exact public TRAIN-label replay"
            if exact_train_label_match
            else "not exact public TRAIN-label replay"
        ),
        "exact_train_label_match": exact_train_label_match,
        "matched_train_task_ids": matches,
        "model_calls": 0,
        "note": "this control does not establish broad generalization",
    }

    typed_lowering_passed = False
    syntax_passed = False
    compile_passed = False
    semantic_passed = False
    compiled_source: str | None = None
    compiled_source_sha256: str | None = None
    lowering_detail = "typed structural lowering did not run"
    syntax_detail = "syntax parsing did not run"
    compile_detail = "Python compilation did not run"
    semantic_detail = "public DEV semantic cases did not run"
    semantic_cases: list[dict[str, object]] = []
    function: object = None
    try:
        compiled = compile_program(completion.program)
        compiled_source = compiled.source
        compiled_source_sha256 = compiled.sha256
        typed_lowering_passed = True
        lowering_detail = "typed Program lowered structurally"
    except Exception as error:
        lowering_detail = f"{type(error).__name__}: {error}"
    if compiled_source is not None:
        try:
            ast.parse(compiled_source)
            syntax_passed = True
            syntax_detail = "compiled source parses as Python"
        except SyntaxError as error:
            syntax_detail = f"SyntaxError: {error}"
        try:
            code = builtins.compile(compiled_source, f"<{task.task_id}>", "exec")
            namespace: dict[str, object] = {}
            exec(code, {"__builtins__": {}}, namespace)
            function = namespace.get(task.signature.entry_point)
            compile_passed = callable(function)
            compile_detail = (
                "registered entry point loaded from compiled source"
                if compile_passed
                else "compiled source omitted the registered entry point"
            )
        except Exception as error:
            compile_detail = f"{type(error).__name__}: {error}"
    if callable(function):
        semantic_passed = True
        for semantic_case in task.semantic_cases:
            arguments = tuple(
                _materialize_semantic(argument, parameter.type)
                for argument, parameter in zip(
                    semantic_case.arguments,
                    task.signature.parameters,
                    strict=True,
                )
            )
            expected = _materialize_semantic(
                semantic_case.expected,
                task.signature.result,
            )
            try:
                actual = function(*arguments)
                passed = actual == expected
                semantic_cases.append(
                    {
                        "actual": actual,
                        "arguments": list(arguments),
                        "case_id": semantic_case.case_id,
                        "expected": expected,
                        "passed": passed,
                    }
                )
            except Exception as error:
                passed = False
                semantic_cases.append(
                    {
                        "arguments": list(arguments),
                        "case_id": semantic_case.case_id,
                        "error": f"{type(error).__name__}: {error}",
                        "expected": expected,
                        "passed": False,
                    }
                )
            semantic_passed = semantic_passed and passed
        semantic_detail = (
            "all three registered cases for this public DEV task passed"
            if semantic_passed
            else "one or more public DEV semantic cases failed"
        )

    race_passed, race_reports = _race_audit(completion)
    try:
        checkpoint_file_bytes = _read_exact_file_bytes(
            checkpoint_path,
            expected_bytes=gate.checkpoint_byte_count,
            label="checked TRAIN checkpoint immutability input",
        )
        raw_checkpoint_passed = (
            checkpoint.to_bytes() == checkpoint_bytes  # type: ignore[attr-defined]
            and checkpoint_file_bytes == checkpoint_bytes
            and sha256(checkpoint_bytes).hexdigest() == gate.checkpoint_raw_sha256
            and completion.checkpoint_sha256 == gate.checkpoint_raw_sha256
            and checkpoint.sha256 == gate.checkpoint_raw_sha256  # type: ignore[attr-defined]
        )
    except Exception:
        raw_checkpoint_passed = False
    state_checkpoint_passed = (
        getattr(checkpoint, "cell_synapse_sha256", None) == gate.checkpoint_cell_synapse_sha256
        and completion.checkpoint_cell_synapse_before == gate.checkpoint_cell_synapse_sha256
        and completion.checkpoint_cell_synapse_after == gate.checkpoint_cell_synapse_sha256
    )
    prompt_passed = completion.prompt_sha256 == sha256(task.prompt.encode("utf-8")).hexdigest()
    signature_passed = completion.program.signature == task.signature
    gates = {
        "teacher_free_label_boundary": _gate(
            not exact_train_label_match,
            "returned tokens are not an exact public TRAIN teacher stream",
        ),
        "typed_lowering": _gate(typed_lowering_passed, lowering_detail),
        "syntax": _gate(syntax_passed, syntax_detail),
        "compile": _gate(compile_passed, compile_detail),
        "semantics": _gate(semantic_passed, semantic_detail),
        "prompt_identity": _gate(
            prompt_passed,
            "completion used the registered public DEV prompt",
        ),
        "signature_identity": _gate(
            signature_passed,
            "completion retained the registered typed signature",
        ),
        "checkpoint_raw_immutability": _gate(
            raw_checkpoint_passed,
            "committed TRAIN checkpoint bytes and SHA-256 are unchanged",
        ),
        "checkpoint_cell_synapse_immutability": _gate(
            state_checkpoint_passed,
            "restored fitted-cell state is unchanged by completion",
        ),
        "cell_race_audit": _gate(
            race_passed,
            "every emitted token is backed by one valid fitted-cell race",
        ),
    }
    return {
        "all_gates_passed": False,
        "boundary_violations": [],
        "checkpoint_cell_synapse_after": completion.checkpoint_cell_synapse_after,
        "checkpoint_cell_synapse_before": completion.checkpoint_cell_synapse_before,
        "checkpoint_sha256": completion.checkpoint_sha256,
        "compiled_source": compiled_source,
        "compiled_source_sha256": compiled_source_sha256,
        "gates": gates,
        "memorization_control": memorization,
        "model_input_sha256": completion.model_input_sha256,
        "program_sha256": completion.program.sha256,
        "prompt_sha256": completion.prompt_sha256,
        "race_count": completion.race_count,
        "races": race_reports,
        "semantic_cases": semantic_cases,
        "tokens": list(completion.program.tokens),
    }


def _finish_task_result(
    result: dict[str, object],
    *,
    fresh_restore: bool,
    restore_attempts: int,
    completion_attempts: int,
    measurements: Mapping[str, object],
) -> None:
    gates = result["gates"]
    assert isinstance(gates, dict)
    gates["fresh_restore_and_attempt_accounting"] = _gate(
        fresh_restore and restore_attempts == 1 and completion_attempts == 1,
        "one fresh checkpoint restore and one completion attempt were used",
    )
    gates["timing_memory_accounting"] = _timing_accounting_gate(
        measurements,
        run_summary=False,
    )
    result["boundary_violations"] = [
        name
        for name, gate_result in gates.items()
        if name in _BOUNDARY_GATES
        and isinstance(gate_result, Mapping)
        and gate_result.get("status") == "failed"
    ]
    result["all_gates_passed"] = all(
        isinstance(gate_result, Mapping) and gate_result.get("passed") is True
        for gate_result in gates.values()
    )


def _task_failure(task_result: Mapping[str, object]) -> dict[str, object] | None:
    result = task_result["result"]
    assert isinstance(result, Mapping)
    gates = result["gates"]
    assert isinstance(gates, Mapping)
    failed_gates = [
        {
            "detail": gate_result.get("detail"),
            "name": name,
            "status": gate_result.get("status"),
        }
        for name, gate_result in gates.items()
        if isinstance(gate_result, Mapping) and gate_result.get("passed") is not True
    ]
    semantic_cases = result.get("semantic_cases")
    failed_cases = (
        [
            semantic_case
            for semantic_case in semantic_cases
            if isinstance(semantic_cases, list)
            and isinstance(semantic_case, Mapping)
            and semantic_case.get("passed") is not True
        ]
        if isinstance(semantic_cases, list)
        else []
    )
    if not failed_gates and not failed_cases and "attempt_error" not in result:
        return None
    failure: dict[str, object] = {
        "failed_gates": failed_gates,
        "failed_semantic_cases": failed_cases,
        "scope": "task",
        "task_id": task_result["task_id"],
    }
    if "attempt_error" in result:
        failure["attempt_error"] = result["attempt_error"]
    return failure


def _execute_gate(
    *,
    gate: PublicDevGate,
    checkpoint_bytes: bytes,
    evidence_directory: Path,
    dependencies: _RunDependencies,
) -> dict[str, object]:
    artifact_root = evidence_directory.parent
    artifact_root.mkdir(parents=True, exist_ok=False)
    evidence_directory.mkdir()
    tracemalloc_was_tracing = tracemalloc.is_tracing()
    if not tracemalloc_was_tracing:
        tracemalloc.start()
    run_wall_start = time.perf_counter_ns()
    run_process_start = time.process_time_ns()
    checkpoint_path = dependencies.repository_root / gate.checkpoint_relative_path
    try:
        _write_evidence(
            evidence_directory,
            0,
            "run-start",
            {
                "development_task_ids": [task.task_id for task in gate.tasks],
                "event": "run-start",
                "mode": "public-development-one-shot",
                "protocol_id": gate.protocol_id,
                "protocol_sha256": gate.raw_sha256,
                "teacher_programs_present": False,
                "train_artifact_commit": gate.train_artifact_commit,
                "train_checkpoint_cell_synapse_sha256": (gate.checkpoint_cell_synapse_sha256),
                "train_checkpoint_raw_sha256": gate.checkpoint_raw_sha256,
                "train_protocol_sha256": gate.train_protocol_sha256,
            },
        )
        task_reports: list[dict[str, object]] = []
        restored_checkpoints: list[object] = []
        for ordinal, task in enumerate(gate.tasks, start=1):
            task_wall_start = time.perf_counter_ns()
            task_process_start = time.process_time_ns()
            restore_attempts = 1
            completion_attempts = 0
            completion_returned = False
            fresh_restore = False
            checkpoint: object | None = None
            completion: Completion | None = None
            try:
                checkpoint = dependencies.restore_checkpoint(checkpoint_bytes)
                fresh_restore = not any(checkpoint is previous for previous in restored_checkpoints)
                restored_checkpoints.append(checkpoint)
                checkpoint_identity_valid = (
                    getattr(checkpoint, "sha256", None) == gate.checkpoint_raw_sha256
                    and getattr(checkpoint, "cell_synapse_sha256", None)
                    == gate.checkpoint_cell_synapse_sha256
                    and checkpoint.to_bytes() == checkpoint_bytes  # type: ignore[attr-defined]
                )
                if not fresh_restore:
                    raise ValueError("checkpoint restore reused an earlier object")
                if not checkpoint_identity_valid:
                    raise ValueError("restored checkpoint identity changed")
                completion_attempts = 1
                completion = dependencies.complete(
                    checkpoint,
                    task.prompt,
                    task.signature,
                )
                completion_returned = True
                result = _audit_completion(
                    gate=gate,
                    task=task,
                    completion=completion,
                    checkpoint=checkpoint,
                    checkpoint_bytes=checkpoint_bytes,
                    checkpoint_path=checkpoint_path,
                )
            except Exception as error:
                result = _empty_result(error)
                if completion is not None:
                    result.update(
                        {
                            "checkpoint_cell_synapse_after": (
                                completion.checkpoint_cell_synapse_after
                            ),
                            "checkpoint_cell_synapse_before": (
                                completion.checkpoint_cell_synapse_before
                            ),
                            "checkpoint_sha256": completion.checkpoint_sha256,
                            "model_input_sha256": completion.model_input_sha256,
                            "program_sha256": completion.program.sha256,
                            "prompt_sha256": completion.prompt_sha256,
                            "race_count": completion.race_count,
                            "tokens": list(completion.program.tokens),
                        }
                    )
            task_wall_ns = time.perf_counter_ns() - task_wall_start
            task_process_ns = time.process_time_ns() - task_process_start
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            task_measurements = {
                "cell_races": int(result["race_count"]),
                "checkpoint_restore_attempts": restore_attempts,
                "completion_attempts": completion_attempts,
                "completion_tokens": len(result["tokens"]),
                "process_time_ns": task_process_ns,
                "tracemalloc_current_bytes": current_bytes,
                "tracemalloc_peak_bytes": peak_bytes,
                "wall_time_ns": task_wall_ns,
            }
            _finish_task_result(
                result,
                fresh_restore=fresh_restore,
                restore_attempts=restore_attempts,
                completion_attempts=completion_attempts,
                measurements=task_measurements,
            )
            task_report = {
                "completion_attempt_count": completion_attempts,
                "completion_returned": completion_returned,
                "event": "development-task-complete",
                "fresh_checkpoint_restore": fresh_restore,
                "measurements": task_measurements,
                "restore_attempt_count": restore_attempts,
                "result": result,
                "task_id": task.task_id,
            }
            _write_evidence(
                evidence_directory,
                ordinal,
                task.task_id,
                task_report,
            )
            task_reports.append(task_report)

        run_wall_ns = time.perf_counter_ns() - run_wall_start
        run_process_ns = time.process_time_ns() - run_process_start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        pass_at_1 = 0
        eligible_task_count = 0
        failures: list[dict[str, object]] = []
        task_results: list[dict[str, object]] = []
        for task_report in task_reports:
            result = task_report["result"]
            assert isinstance(result, Mapping)
            gates = result["gates"]
            assert isinstance(gates, Mapping)
            outcome_passed = all(
                isinstance(gates.get(name), Mapping) and gates[name].get("passed") is True  # type: ignore[union-attr]
                for name in _OUTCOME_GATES
            )
            boundary_violations = result["boundary_violations"]
            assert isinstance(boundary_violations, list)
            pass_at_1 += int(outcome_passed)
            eligible_task_count += int(outcome_passed and not boundary_violations)
            failure = _task_failure(task_report)
            if failure is not None:
                failures.append(failure)
            task_results.append(
                {
                    "boundary_violations": boundary_violations,
                    "completion_attempt_count": task_report["completion_attempt_count"],
                    "completion_returned": task_report["completion_returned"],
                    "failed_gate_count": len(failure["failed_gates"]) if failure is not None else 0,
                    "pass_at_1": outcome_passed,
                    "restore_attempt_count": task_report["restore_attempt_count"],
                    "task_id": task_report["task_id"],
                }
            )

        measurements: dict[str, object] = {
            "cell_races": sum(
                int(report["measurements"]["cell_races"])  # type: ignore[index]
                for report in task_reports
            ),
            "checkpoint_bytes": len(checkpoint_bytes),
            "checkpoint_restore_attempts": sum(
                int(report["restore_attempt_count"]) for report in task_reports
            ),
            "completion_attempts": sum(
                int(report["completion_attempt_count"]) for report in task_reports
            ),
            "completion_tokens": sum(
                int(report["measurements"]["completion_tokens"])  # type: ignore[index]
                for report in task_reports
            ),
            "evidence_bytes": 0,
            "process_time_ns": run_process_ns,
            "tracemalloc_current_bytes": current_bytes,
            "tracemalloc_peak_bytes": peak_bytes,
            "wall_time_ns": run_wall_ns,
        }
        run_gates = {
            "all_tasks_recorded": _gate(
                [report["task_id"] for report in task_reports]
                == [task.task_id for task in gate.tasks],
                "every preregistered public DEV task has one ordinal record",
            ),
            "pass_at_1_accounting": _gate(
                type(pass_at_1) is int and 0 <= pass_at_1 <= len(gate.tasks),
                "exact semantic pass@1 is an integer in the preregistered range",
            ),
            "evidence_accounting": _gate(
                len(list(evidence_directory.glob("*.json"))) == len(gate.tasks) + 1
                and not list(evidence_directory.glob("*.tmp")),
                "incremental evidence ordinals are complete and contain no temporary files",
            ),
            "timing_memory_accounting": _timing_accounting_gate(
                measurements,
                run_summary=True,
            ),
        }
        failures.extend(
            {
                "detail": gate_result["detail"],
                "gate": name,
                "scope": "run",
                "status": gate_result["status"],
            }
            for name, gate_result in run_gates.items()
            if gate_result["passed"] is not True
        )
        summary: dict[str, object] = {
            "event": "run-summary",
            "failures": failures,
            "gates": run_gates,
            "measurements": measurements,
            "pass_at_1": pass_at_1,
            "pass_at_1_range": [0, len(gate.tasks)],
            "promotion": {
                "achieved": False,
                "eligible_task_count": eligible_task_count,
                "semantic_passes_required": 1,
                "task_count": len(gate.tasks),
            },
            "protocol_id": gate.protocol_id,
            "protocol_sha256": gate.raw_sha256,
            "task_results": task_results,
            "train_artifact_commit": gate.train_artifact_commit,
            "train_checkpoint_raw_sha256": gate.checkpoint_raw_sha256,
        }
        evidence_bytes_before_summary = _evidence_byte_count(evidence_directory)
        summary_path = evidence_directory / "003-summary.json"
        while True:
            run_gates["timing_memory_accounting"] = _timing_accounting_gate(
                measurements,
                run_summary=True,
            )
            all_run_gates_passed = all(
                gate_result["passed"] is True for gate_result in run_gates.values()
            )
            summary["all_run_gates_passed"] = all_run_gates_passed
            promotion = summary["promotion"]
            assert isinstance(promotion, dict)
            promotion["achieved"] = eligible_task_count >= 1 and all_run_gates_passed
            summary_bytes = _canonical_evidence(summary)
            total_bytes = evidence_bytes_before_summary + len(summary_bytes)
            if measurements["evidence_bytes"] == total_bytes:
                break
            measurements["evidence_bytes"] = total_bytes
        _write_new_file(summary_path, summary_bytes)
        if _evidence_byte_count(evidence_directory) != total_bytes:
            raise RuntimeError("incremental public DEV evidence byte accounting mismatch")
        return summary
    finally:
        if not tracemalloc_was_tracing:
            tracemalloc.stop()


def run(argv: Sequence[str] | None = None) -> int:
    """Plan by default; execute only with the exact frozen DEV protocol hash."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    arguments = parser.parse_args(argv)

    dependencies = _production_dependencies()
    gate = load_public_dev_gate(dependencies.protocol_path)
    evidence_directory = _evidence_directory(gate, dependencies)
    if arguments.execute and arguments.protocol_sha256 != gate.raw_sha256:
        raise ExecutionRefusedError(
            "--execute requires the exact raw DEV protocol SHA-256 in --protocol-sha256"
        )
    _reject_existing_evidence(evidence_directory)
    checkpoint_bytes = _preflight_train_inputs(gate, dependencies)
    if not arguments.execute:
        print(json.dumps(_plan(gate), sort_keys=True))
        return 0
    summary = _execute_gate(
        gate=gate,
        checkpoint_bytes=checkpoint_bytes,
        evidence_directory=evidence_directory,
        dependencies=dependencies,
    )
    print(json.dumps(summary, sort_keys=True))
    promotion = summary["promotion"]
    assert isinstance(promotion, Mapping)
    if promotion.get("achieved") is not True:
        raise DevGateError("public DEV promotion milestone was not met")
    return 0
