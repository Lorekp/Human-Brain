"""Preregistered public train-only curriculum for the typed program seam.

Loading this protocol validates public data and teacher programs.  It never
grows a circuit, completes a task, or opens a development task for completion.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import os
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from humanbrain.typed_program import (
    Completion,
    Example,
    FrozenCheckpoint,
    Layer1Protocol,
    Program,
    compile as compile_program,
    complete,
    grow,
)
from humanbrain.typed_program_types import ProgramType, TypedSignature


_SCHEMA = "humanbrain-public-typed-program-train-protocol-v1"
_PROTOCOL_ID = "humanbrain-public-typed-program-train-v1"
_STATUS = "preregistered-not-executed"
_TYPED_STACK_BASE_COMMIT = "600ef82bd2edddb00b5df237754bec62be93c0b1"
_MAX_PROTOCOL_BYTES = 131_072
FROZEN_PUBLIC_CURRICULUM_SHA256 = (
    "1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191"
)
_CHECKPOINT_RELATIVE_PATH = Path(
    "artifacts/typed_program_public_train_v1/checkpoint.json"
)
_EVIDENCE_RELATIVE_DIRECTORY = Path(
    "artifacts/typed_program_public_train_v1/evidence"
)


class ExecutionRefusedError(RuntimeError):
    """The explicit execution authorization did not match the protocol."""


class EvidenceExistsError(RuntimeError):
    """A fixed checkpoint or evidence target already exists."""


class CurriculumGateError(RuntimeError):
    """One or more preregistered TRAIN gates failed."""


@dataclass(frozen=True, slots=True)
class SemanticCase:
    case_id: str
    arguments: tuple[object, ...]
    expected: object


@dataclass(frozen=True, slots=True)
class TrainTask:
    task_id: str
    family_id: str
    template_id: str
    prompt: str
    signature: TypedSignature
    teacher_tokens: tuple[str, ...]
    literals: tuple[int, ...]
    semantic_cases: tuple[SemanticCase, ...]

    def to_example(self) -> Example:
        return Example(
            prompt=self.prompt,
            signature=self.signature,
            teacher_tokens=self.teacher_tokens,
        )


@dataclass(frozen=True, slots=True)
class DevelopmentTask:
    task_id: str
    family_id: str
    template_id: str
    prompt: str
    signature: TypedSignature
    literals: tuple[int, ...]
    semantic_cases: tuple[SemanticCase, ...]


@dataclass(frozen=True, slots=True)
class PublicCurriculum:
    protocol_id: str
    raw_sha256: str
    typed_stack_base_commit: str
    train_tasks: tuple[TrainTask, ...]
    development_tasks: tuple[DevelopmentTask, ...]
    checkpoint_relative_path: Path
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


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered sequence")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _matches_type(value: object, program_type: ProgramType) -> bool:
    if program_type == ProgramType.Int():
        return type(value) is int
    if program_type.constructor == "List":
        return isinstance(value, list) and all(
            _matches_type(item, program_type.arguments[0]) for item in value
        )
    return False


def _freeze_semantic(value: object, program_type: ProgramType) -> object:
    if program_type == ProgramType.Int():
        return value
    if program_type.constructor == "List":
        assert isinstance(value, list)
        return tuple(
            _freeze_semantic(item, program_type.arguments[0]) for item in value
        )
    raise ValueError("public curriculum supports only Int and List semantic values")


def _materialize_semantic(value: object, program_type: ProgramType) -> object:
    if program_type == ProgramType.Int():
        return value
    if program_type.constructor == "List":
        assert isinstance(value, tuple)
        return [
            _materialize_semantic(item, program_type.arguments[0]) for item in value
        ]
    raise ValueError("public curriculum supports only Int and List semantic values")


def _semantic_cases(
    value: object,
    signature: TypedSignature,
    label: str,
) -> tuple[SemanticCase, ...]:
    records = _sequence(value, label)
    if not records:
        raise ValueError(f"{label} must not be empty")
    cases: list[SemanticCase] = []
    for index, raw_case in enumerate(records):
        case = _record(
            raw_case,
            frozenset({"case_id", "arguments", "expected"}),
            f"{label}[{index}]",
        )
        raw_arguments = tuple(
            _sequence(case["arguments"], f"{label}[{index}].arguments")
        )
        if len(raw_arguments) != len(signature.parameters):
            raise ValueError(f"{label}[{index}] argument count does not match signature")
        if not all(
            _matches_type(argument, parameter.type)
            for argument, parameter in zip(
                raw_arguments,
                signature.parameters,
                strict=True,
            )
        ):
            raise ValueError(f"{label}[{index}] argument type does not match signature")
        expected = case["expected"]
        if not _matches_type(expected, signature.result):
            raise ValueError(f"{label}[{index}] result type does not match signature")
        arguments = tuple(
            _freeze_semantic(argument, parameter.type)
            for argument, parameter in zip(
                raw_arguments,
                signature.parameters,
                strict=True,
            )
        )
        cases.append(
            SemanticCase(
                case_id=_nonempty_text(case["case_id"], f"{label}[{index}].case_id"),
                arguments=arguments,
                expected=_freeze_semantic(expected, signature.result),
            )
        )
    ids = tuple(case.case_id for case in cases)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} case IDs must be unique")
    return tuple(cases)


def _task_common(
    value: object,
    keys: frozenset[str],
    label: str,
) -> tuple[
    Mapping[str, object],
    str,
    str,
    str,
    str,
    TypedSignature,
    tuple[int, ...],
    tuple[SemanticCase, ...],
]:
    task = _record(value, keys, label)
    signature = TypedSignature.from_record(task["signature"])
    literals = tuple(
        _integer(item, f"{label}.literals")
        for item in _sequence(task["literals"], f"{label}.literals")
    )
    if len(literals) != len(set(literals)):
        raise ValueError(f"{label} literals must be unique")
    return (
        task,
        _nonempty_text(task["task_id"], f"{label}.task_id"),
        _nonempty_text(task["family_id"], f"{label}.family_id"),
        _nonempty_text(task["template_id"], f"{label}.template_id"),
        _nonempty_text(task["prompt"], f"{label}.prompt"),
        signature,
        literals,
        _semantic_cases(task["semantic_cases"], signature, f"{label}.semantic_cases"),
    )


def _train_task(value: object, index: int) -> TrainTask:
    label = f"train.tasks[{index}]"
    task, task_id, family_id, template_id, prompt, signature, literals, cases = (
        _task_common(
            value,
            frozenset(
                {
                    "task_id",
                    "family_id",
                    "template_id",
                    "prompt",
                    "signature",
                    "teacher_tokens",
                    "literals",
                    "semantic_cases",
                }
            ),
            label,
        )
    )
    teacher_tokens = tuple(
        _nonempty_text(token, f"{label}.teacher_tokens")
        for token in _sequence(task["teacher_tokens"], f"{label}.teacher_tokens")
    )
    compiled = compile_program(Program(teacher_tokens, signature))
    namespace: dict[str, object] = {}
    exec(compiled.source, {"__builtins__": {}}, namespace)
    function = namespace[signature.entry_point]
    if not callable(function):
        raise ValueError(f"{label} teacher program did not compile to an entry point")
    for case in cases:
        arguments = tuple(
            _materialize_semantic(argument, parameter.type)
            for argument, parameter in zip(
                case.arguments,
                signature.parameters,
                strict=True,
            )
        )
        expected = _materialize_semantic(case.expected, signature.result)
        if function(*arguments) != expected:
            raise ValueError(f"{label} teacher program fails semantic case {case.case_id}")
    return TrainTask(
        task_id=task_id,
        family_id=family_id,
        template_id=template_id,
        prompt=prompt,
        signature=signature,
        teacher_tokens=teacher_tokens,
        literals=literals,
        semantic_cases=cases,
    )


def _development_task(value: object, index: int) -> DevelopmentTask:
    label = f"development.tasks[{index}]"
    _, task_id, family_id, template_id, prompt, signature, literals, cases = (
        _task_common(
            value,
            frozenset(
                {
                    "task_id",
                    "family_id",
                    "template_id",
                    "prompt",
                    "signature",
                    "literals",
                    "semantic_cases",
                }
            ),
            label,
        )
    )
    return DevelopmentTask(
        task_id=task_id,
        family_id=family_id,
        template_id=template_id,
        prompt=prompt,
        signature=signature,
        literals=literals,
        semantic_cases=cases,
    )


def _assert_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_fixed_configuration(value: object) -> None:
    config = _record(
        value,
        frozenset(
            {
                "module",
                "growth",
                "completion",
                "cell_path",
                "checkpoint_count",
                "layer1_protocol",
            }
        ),
        "fixed_configuration",
    )
    expected_interface = (
        "humanbrain.typed_program",
        "humanbrain.typed_program.grow",
        "humanbrain.typed_program.complete",
        "recurrent-fitted-cell-only",
        1,
    )
    actual_interface = (
        config["module"],
        config["growth"],
        config["completion"],
        config["cell_path"],
        config["checkpoint_count"],
    )
    if actual_interface != expected_interface:
        raise ValueError("fixed configuration does not name the sole recurrent fitted-cell path")
    layer1 = _record(
        config["layer1_protocol"],
        frozenset(
            {
                "protocol_version",
                "compiler_id",
                "primitive_table_sha256",
                "circuit_config_sha256",
                "growth_epochs",
                "allen_glif5_config_sha256",
                "max_training_examples",
                "max_checkpoint_bytes",
                "literal_digit_bound",
                "max_ast_nodes",
                "max_ast_depth",
                "max_output_tokens",
            }
        ),
        "fixed_configuration.layer1_protocol",
    )
    protocol = Layer1Protocol()
    expected = {
        "protocol_version": protocol.protocol_version,
        "compiler_id": protocol.compiler_id,
        "primitive_table_sha256": protocol.primitive_table_sha256,
        "circuit_config_sha256": protocol.circuit_config_sha256,
        "growth_epochs": protocol.growth_epochs,
        "allen_glif5_config_sha256": protocol.allen_glif5_config_sha256,
        "max_training_examples": protocol.max_training_examples,
        "max_checkpoint_bytes": protocol.max_checkpoint_bytes,
        "literal_digit_bound": protocol.literal_digit_bound,
        "max_ast_nodes": protocol.max_ast_nodes,
        "max_ast_depth": protocol.max_ast_depth,
        "max_output_tokens": protocol.max_output_tokens,
    }
    if dict(layer1) != expected:
        raise ValueError("fixed Layer1Protocol commitments do not match this implementation")


def load_public_curriculum(path: str | Path) -> PublicCurriculum:
    """Load and validate the preregistered public curriculum without running it."""
    protocol_path = Path(path)
    raw = protocol_path.read_bytes()
    if len(raw) > _MAX_PROTOCOL_BYTES:
        raise ValueError("public curriculum protocol exceeds its byte limit")
    raw_sha256 = sha256(raw).hexdigest()
    if raw_sha256 != FROZEN_PUBLIC_CURRICULUM_SHA256:
        raise ValueError("public curriculum raw SHA-256 is not preregistered")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("public curriculum protocol is not strict UTF-8 JSON") from error
    root = _record(
        parsed,
        frozenset(
            {
                "schema",
                "protocol_id",
                "status",
                "typed_stack_base_commit",
                "scope",
                "fixed_configuration",
                "train",
                "development",
                "execution",
                "reporting",
                "prohibitions",
            }
        ),
        "public curriculum protocol",
    )
    identity = (
        root["schema"],
        root["protocol_id"],
        root["status"],
        root["typed_stack_base_commit"],
    )
    if identity != (_SCHEMA, _PROTOCOL_ID, _STATUS, _TYPED_STACK_BASE_COMMIT):
        raise ValueError("public curriculum identity is not the preregistered v1 identity")
    _validate_fixed_configuration(root["fixed_configuration"])

    train = _record(
        root["train"],
        frozenset({"partition", "teacher_programs_public", "tasks"}),
        "train",
    )
    if (train["partition"], train["teacher_programs_public"]) != ("train", True):
        raise ValueError("TRAIN must be the public teacher-program partition")
    train_tasks = tuple(
        _train_task(task, index)
        for index, task in enumerate(_sequence(train["tasks"], "train.tasks"))
    )
    if len(train_tasks) != 3:
        raise ValueError("the v1 public TRAIN curriculum contains exactly three tasks")

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
        raise ValueError("development must remain teacher-free, completion-unopened, and unrun")
    development_tasks = tuple(
        _development_task(task, index)
        for index, task in enumerate(
            _sequence(development["tasks"], "development.tasks")
        )
    )
    if len(development_tasks) != 2:
        raise ValueError("the v1 public development definition contains exactly two tasks")

    all_ids = tuple(task.task_id for task in (*train_tasks, *development_tasks))
    _assert_unique(all_ids, "curriculum task IDs")
    train_families = {task.family_id for task in train_tasks}
    development_families = {task.family_id for task in development_tasks}
    train_templates = {task.template_id for task in train_tasks}
    development_templates = {task.template_id for task in development_tasks}
    train_literals = {literal for task in train_tasks for literal in task.literals}
    development_literals = {
        literal for task in development_tasks for literal in task.literals
    }
    if train_families & development_families:
        raise ValueError("TRAIN and development families must be disjoint")
    if train_templates & development_templates:
        raise ValueError("TRAIN and development templates must be disjoint")
    if train_literals & development_literals:
        raise ValueError("TRAIN and development literals must be disjoint")

    scope = _record(
        root["scope"],
        frozenset({"operation", "claim", "development_policy"}),
        "scope",
    )
    if dict(scope) != {
        "operation": "public-train-only",
        "claim": "TRAIN label replay is memorization, not generalization",
        "development_policy": "definition is public; no completion or execution",
    }:
        raise ValueError("public curriculum scope commitments changed")

    # The remaining records are commitments consumed by the guarded runner.
    execution = _record(
        root["execution"],
        frozenset({"authorization", "attempts", "restore", "gates"}),
        "execution",
    )
    if execution["authorization"] != (
        "require --execute and the exact raw protocol SHA-256"
    ):
        raise ValueError("execution authorization commitment changed")
    reporting = _record(
        root["reporting"],
        frozenset(
            {
                "checkpoint_path",
                "evidence_directory",
                "incremental",
                "nonoverwriting",
                "measurements",
            }
        ),
        "reporting",
    )
    if (
        reporting["checkpoint_path"],
        reporting["evidence_directory"],
        reporting["incremental"],
        reporting["nonoverwriting"],
    ) != (
        _CHECKPOINT_RELATIVE_PATH.as_posix(),
        _EVIDENCE_RELATIVE_DIRECTORY.as_posix(),
        True,
        True,
    ):
        raise ValueError("evidence path or write-policy commitment changed")
    prohibitions = tuple(
        _nonempty_text(item, "prohibitions")
        for item in _sequence(root["prohibitions"], "prohibitions")
    )
    if len(prohibitions) != len(set(prohibitions)):
        raise ValueError("prohibitions must be unique")

    return PublicCurriculum(
        protocol_id=_PROTOCOL_ID,
        raw_sha256=raw_sha256,
        typed_stack_base_commit=_TYPED_STACK_BASE_COMMIT,
        train_tasks=train_tasks,
        development_tasks=development_tasks,
        checkpoint_relative_path=_CHECKPOINT_RELATIVE_PATH,
        evidence_relative_directory=_EVIDENCE_RELATIVE_DIRECTORY,
    )


def _plan(curriculum: PublicCurriculum) -> dict[str, object]:
    return {
        "checkpoint_count": 1,
        "completion_attempts_per_train_task": 1,
        "development_completion_opened": False,
        "development_run_count": 0,
        "development_task_count": len(curriculum.development_tasks),
        "execute": False,
        "mode": "plan",
        "protocol_id": curriculum.protocol_id,
        "protocol_sha256": curriculum.raw_sha256,
        "train_semantic_case_count": sum(
            len(task.semantic_cases) for task in curriculum.train_tasks
        ),
        "train_task_count": len(curriculum.train_tasks),
        "train_task_ids": [task.task_id for task in curriculum.train_tasks],
        "train_teacher_token_count": sum(
            len(task.teacher_tokens) for task in curriculum.train_tasks
        ),
    }


def _evidence_paths(
    protocol_path: Path,
    curriculum: PublicCurriculum,
) -> tuple[Path, Path]:
    repository_root = protocol_path.resolve().parents[1]
    return (
        repository_root / curriculum.checkpoint_relative_path,
        repository_root / curriculum.evidence_relative_directory,
    )


def _reject_existing_evidence(checkpoint_path: Path, evidence_directory: Path) -> None:
    if checkpoint_path.exists():
        raise EvidenceExistsError(
            f"checkpoint evidence already exists: {checkpoint_path}"
        )
    if evidence_directory.exists():
        raise EvidenceExistsError(
            f"incremental evidence directory already exists: {evidence_directory}"
        )
    if checkpoint_path.parent.exists():
        raise EvidenceExistsError(
            f"artifact evidence already exists: {checkpoint_path.parent}"
        )


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
    """Publish bytes atomically and fail instead of replacing an existing path."""
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


def _sha256_is_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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


def _gate(passed: bool, detail: str) -> dict[str, object]:
    return {"detail": detail, "passed": passed, "status": "passed" if passed else "failed"}


def _audit_completion(
    *,
    task: TrainTask,
    completion: Completion,
    checkpoint: FrozenCheckpoint,
    checkpoint_bytes: bytes,
    checkpoint_path: Path,
) -> dict[str, object]:
    label_passed = completion.program.tokens == task.teacher_tokens
    typed_lowering_passed = False
    syntax_passed = False
    compile_passed = False
    semantic_passed = False
    compiled_source: str | None = None
    compiled_source_sha256: str | None = None
    lowering_detail = "typed structural lowering did not run"
    syntax_detail = "syntax parsing did not run"
    compile_detail = "Python compilation did not run"
    semantic_detail = "public semantic cases did not run"
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
            code = builtins.compile(
                compiled_source,
                f"<{task.task_id}>",
                "exec",
            )
            namespace: dict[str, object] = {}
            exec(code, {"__builtins__": {}}, namespace)
            function = namespace.get(task.signature.entry_point)
            compile_passed = callable(function)
            compile_detail = (
                "entry point loaded from compiled source"
                if compile_passed
                else "compiled source did not define the registered entry point"
            )
        except Exception as error:
            compile_detail = f"{type(error).__name__}: {error}"

    if callable(function):
        semantic_passed = True
        for case in task.semantic_cases:
            arguments = tuple(
                _materialize_semantic(argument, parameter.type)
                for argument, parameter in zip(
                    case.arguments,
                    task.signature.parameters,
                    strict=True,
                )
            )
            expected = _materialize_semantic(case.expected, task.signature.result)
            try:
                actual = function(*arguments)
                passed = actual == expected
                semantic_cases.append(
                    {
                        "actual": actual,
                        "arguments": list(arguments),
                        "case_id": case.case_id,
                        "expected": expected,
                        "passed": passed,
                    }
                )
            except Exception as error:
                passed = False
                semantic_cases.append(
                    {
                        "arguments": list(arguments),
                        "case_id": case.case_id,
                        "error": f"{type(error).__name__}: {error}",
                        "expected": expected,
                        "passed": False,
                    }
                )
            semantic_passed = semantic_passed and passed
        semantic_detail = (
            "all registered public TRAIN cases passed"
            if semantic_passed
            else "one or more registered public TRAIN cases failed"
        )

    race_passed, race_reports = _race_audit(completion)
    raw_checkpoint_passed = (
        checkpoint.to_bytes() == checkpoint_bytes
        and checkpoint_path.read_bytes() == checkpoint_bytes
        and sha256(checkpoint_bytes).hexdigest() == checkpoint.sha256
        and completion.checkpoint_sha256 == checkpoint.sha256
    )
    state_checkpoint_passed = (
        completion.checkpoint_cell_synapse_before == checkpoint.cell_synapse_sha256
        and completion.checkpoint_cell_synapse_after == checkpoint.cell_synapse_sha256
    )
    prompt_passed = completion.prompt_sha256 == sha256(
        task.prompt.encode("utf-8")
    ).hexdigest()
    signature_passed = completion.program.signature == task.signature
    gates = {
        "cell_race_audit": _gate(
            race_passed,
            "every emitted token is backed by one valid fitted-cell race",
        ),
        "checkpoint_cell_synapse_immutability": _gate(
            state_checkpoint_passed,
            "restored fitted-cell state is unchanged by completion",
        ),
        "checkpoint_raw_immutability": _gate(
            raw_checkpoint_passed,
            "common checkpoint bytes and SHA-256 are unchanged",
        ),
        "compile": _gate(compile_passed, compile_detail),
        "label_replay": _gate(
            label_passed,
            "emitted tokens exactly replay the public TRAIN teacher label",
        ),
        "prompt_identity": _gate(prompt_passed, "completion used the registered prompt"),
        "semantics": _gate(semantic_passed, semantic_detail),
        "signature_identity": _gate(
            signature_passed,
            "completion retained the registered typed signature",
        ),
        "syntax": _gate(syntax_passed, syntax_detail),
        "typed_lowering": _gate(typed_lowering_passed, lowering_detail),
    }
    return {
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "compiled_source": compiled_source,
        "compiled_source_sha256": compiled_source_sha256,
        "gates": gates,
        "model_input_sha256": completion.model_input_sha256,
        "checkpoint_cell_synapse_after": completion.checkpoint_cell_synapse_after,
        "checkpoint_cell_synapse_before": completion.checkpoint_cell_synapse_before,
        "checkpoint_sha256": completion.checkpoint_sha256,
        "program_sha256": completion.program.sha256,
        "prompt_sha256": completion.prompt_sha256,
        "race_count": completion.race_count,
        "races": race_reports,
        "semantic_cases": semantic_cases,
        "tokens": list(completion.program.tokens),
    }


def _failed_completion_report(
    error: Exception,
    *,
    completion_returned: bool,
) -> dict[str, object]:
    detail = "completion audit failed" if completion_returned else "completion did not return"
    gates = {
        name: {
            "detail": detail,
            "passed": False,
            "status": "not_run",
        }
        for name in (
            "cell_race_audit",
            "checkpoint_cell_synapse_immutability",
            "checkpoint_raw_immutability",
            "compile",
            "label_replay",
            "prompt_identity",
            "semantics",
            "signature_identity",
            "syntax",
            "typed_lowering",
        )
    }
    return {
        "all_gates_passed": False,
        "attempt_error": f"{type(error).__name__}: {error}",
        "gates": gates,
        "race_count": 0,
        "races": [],
        "semantic_cases": [],
        "tokens": [],
    }


def _evidence_byte_count(evidence_directory: Path) -> int:
    return sum(
        path.stat().st_size for path in evidence_directory.iterdir() if path.is_file()
    )


def _execute_curriculum(
    *,
    curriculum: PublicCurriculum,
    checkpoint_path: Path,
    evidence_directory: Path,
) -> dict[str, object]:
    """Execute only the fixed public TRAIN partition exactly once."""
    artifact_root = checkpoint_path.parent
    artifact_root.mkdir(parents=True, exist_ok=False)
    evidence_directory.mkdir()
    tracemalloc.start()
    run_wall_start = time.perf_counter_ns()
    run_process_start = time.process_time_ns()
    try:
        _write_evidence(
            evidence_directory,
            0,
            "run-start",
            {
                "claim": "TRAIN label replay is memorization, not generalization",
                "development_completion_opened": False,
                "development_run_count": 0,
                "event": "run-start",
                "mode": "public-train-only",
                "protocol_id": curriculum.protocol_id,
                "protocol_sha256": curriculum.raw_sha256,
                "train_task_ids": [task.task_id for task in curriculum.train_tasks],
                "typed_stack_base_commit": curriculum.typed_stack_base_commit,
            },
        )

        training_wall_start = time.perf_counter_ns()
        training_process_start = time.process_time_ns()
        try:
            checkpoint = grow(
                Layer1Protocol(),
                tuple(task.to_example() for task in curriculum.train_tasks),
            )
        except Exception as error:
            _write_evidence(
                evidence_directory,
                1,
                "training-failure",
                {
                    "error": f"{type(error).__name__}: {error}",
                    "event": "training-failure",
                    "training_attempts": 1,
                },
            )
            raise
        training_wall_ns = time.perf_counter_ns() - training_wall_start
        training_process_ns = time.process_time_ns() - training_process_start
        checkpoint_bytes = checkpoint.to_bytes()
        _write_new_file(checkpoint_path, checkpoint_bytes)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        _write_evidence(
            evidence_directory,
            1,
            "training",
            {
                "checkpoint": {
                    "bytes": len(checkpoint_bytes),
                    "cell_synapse_sha256": checkpoint.cell_synapse_sha256,
                    "layer1_protocol_sha256": checkpoint.protocol_sha256,
                    "raw_sha256": sha256(checkpoint_bytes).hexdigest(),
                    "sha256": checkpoint.sha256,
                    "training_digest": checkpoint.training_digest,
                    "training_input_sha256": list(checkpoint.training_input_sha256),
                    "training_sequence_token_counts": list(
                        checkpoint.training_sequence_token_counts
                    ),
                },
                "event": "training-complete",
                "measurements": {
                    "process_time_ns": training_process_ns,
                    "tracemalloc_current_bytes": current_bytes,
                    "tracemalloc_peak_bytes": peak_bytes,
                    "wall_time_ns": training_wall_ns,
                },
                "training_attempts": 1,
                "training_examples": len(curriculum.train_tasks),
                "training_sequence_tokens": sum(
                    checkpoint.training_sequence_token_counts
                ),
                "training_teacher_tokens": sum(
                    len(task.teacher_tokens) for task in curriculum.train_tasks
                ),
            },
        )

        task_summaries: list[dict[str, object]] = []
        for task_index, task in enumerate(curriculum.train_tasks, start=1):
            task_wall_start = time.perf_counter_ns()
            task_process_start = time.process_time_ns()
            restored_checkpoint = FrozenCheckpoint.from_bytes(checkpoint_bytes)
            try:
                # This is the sole completion call for this registered TRAIN task.
                completion = complete(
                    restored_checkpoint,
                    task.prompt,
                    task.signature,
                )
                completion_returned = True
            except Exception as error:
                audit = _failed_completion_report(
                    error,
                    completion_returned=False,
                )
                completion_returned = False
            else:
                try:
                    audit = _audit_completion(
                        task=task,
                        completion=completion,
                        checkpoint=restored_checkpoint,
                        checkpoint_bytes=checkpoint_bytes,
                        checkpoint_path=checkpoint_path,
                    )
                except Exception as error:
                    audit = _failed_completion_report(
                        error,
                        completion_returned=True,
                    )
            task_wall_ns = time.perf_counter_ns() - task_wall_start
            task_process_ns = time.process_time_ns() - task_process_start
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            task_report = {
                "attempt_count": 1,
                "claim": "public TRAIN label replay; memorization only",
                "common_checkpoint_sha256": checkpoint.sha256,
                "completion_returned": completion_returned,
                "event": "train-task-complete",
                "fresh_checkpoint_restore": True,
                "measurements": {
                    "process_time_ns": task_process_ns,
                    "tracemalloc_current_bytes": current_bytes,
                    "tracemalloc_peak_bytes": peak_bytes,
                    "wall_time_ns": task_wall_ns,
                },
                "result": audit,
                "task_id": task.task_id,
            }
            _write_evidence(
                evidence_directory,
                task_index + 1,
                task.task_id,
                task_report,
            )
            task_summaries.append(
                {
                    "all_gates_passed": audit["all_gates_passed"],
                    "attempt_count": 1,
                    "completion_token_count": len(audit["tokens"]),
                    "completion_returned": completion_returned,
                    "race_count": audit["race_count"],
                    "task_id": task.task_id,
                }
            )

        run_wall_ns = time.perf_counter_ns() - run_wall_start
        run_process_ns = time.process_time_ns() - run_process_start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        summary_ordinal = len(curriculum.train_tasks) + 2
        summary_path = evidence_directory / f"{summary_ordinal:03d}-summary.json"
        summary: dict[str, object] = {
            "all_train_gates_passed": all(
                bool(task["all_gates_passed"]) for task in task_summaries
            ),
            "checkpoint": {
                "bytes": len(checkpoint_bytes),
                "cell_synapse_sha256": checkpoint.cell_synapse_sha256,
                "raw_sha256": sha256(checkpoint_bytes).hexdigest(),
                "sha256": checkpoint.sha256,
            },
            "claim": "TRAIN label replay is memorization, not generalization",
            "completion_attempts": len(task_summaries),
            "development_completion_opened": False,
            "development_run_count": 0,
            "event": "run-summary",
            "measurements": {
                "cell_races": sum(int(task["race_count"]) for task in task_summaries),
                "completion_tokens": sum(
                    int(task["completion_token_count"]) for task in task_summaries
                ),
                "completion_attempts": len(task_summaries),
                "checkpoint_bytes": len(checkpoint_bytes),
                "evidence_bytes": 0,
                "process_time_ns": run_process_ns,
                "training_sequences": len(checkpoint.training_sequence_token_counts),
                "training_tokens": sum(checkpoint.training_sequence_token_counts),
                "tracemalloc_current_bytes": current_bytes,
                "tracemalloc_peak_bytes": peak_bytes,
                "wall_time_ns": run_wall_ns,
            },
            "protocol_id": curriculum.protocol_id,
            "protocol_sha256": curriculum.raw_sha256,
            "task_results": task_summaries,
            "train_task_count": len(curriculum.train_tasks),
        }
        evidence_bytes_before_summary = _evidence_byte_count(evidence_directory)
        measurements = summary["measurements"]
        assert isinstance(measurements, dict)
        while True:
            summary_bytes = _canonical_evidence(summary)
            total_bytes = evidence_bytes_before_summary + len(summary_bytes)
            if measurements["evidence_bytes"] == total_bytes:
                break
            measurements["evidence_bytes"] = total_bytes
        _write_new_file(summary_path, summary_bytes)
        if _evidence_byte_count(evidence_directory) != total_bytes:
            raise RuntimeError("incremental evidence byte accounting mismatch")
        return summary
    finally:
        tracemalloc.stop()


def run(argv: Sequence[str] | None = None) -> int:
    """Plan by default; execute only with an exact raw protocol hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).parents[2]
        / "benchmarks"
        / "typed_program_public_train_protocol_v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    arguments = parser.parse_args(argv)

    curriculum = load_public_curriculum(arguments.protocol)
    checkpoint_path, evidence_directory = _evidence_paths(
        arguments.protocol,
        curriculum,
    )
    if arguments.execute and arguments.protocol_sha256 != curriculum.raw_sha256:
        raise ExecutionRefusedError(
            "--execute requires the exact raw protocol SHA-256 in "
            "--protocol-sha256"
        )
    _reject_existing_evidence(checkpoint_path, evidence_directory)
    if not arguments.execute:
        print(json.dumps(_plan(curriculum), sort_keys=True))
        return 0
    summary = _execute_curriculum(
        curriculum=curriculum,
        checkpoint_path=checkpoint_path,
        evidence_directory=evidence_directory,
    )
    print(json.dumps(summary, sort_keys=True))
    if not summary["all_train_gates_passed"]:
        raise CurriculumGateError("one or more public TRAIN gates failed")
    return 0


__all__ = [
    "CurriculumGateError",
    "DevelopmentTask",
    "EvidenceExistsError",
    "ExecutionRefusedError",
    "FROZEN_PUBLIC_CURRICULUM_SHA256",
    "PublicCurriculum",
    "SemanticCase",
    "TrainTask",
    "load_public_curriculum",
    "run",
]
