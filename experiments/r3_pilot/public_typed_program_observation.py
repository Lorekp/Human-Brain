"""Read-only root observations for the already-current typed program.

The sole program-semantics boundary is :func:`humanbrain.typed_program.compile`.
This module compiles one supplied incumbent, executes only that trusted emitted
source on a declared case grid, and projects each raw root output into the
factorized typed-output cells.  It has no edit, parser, subtree, or hypothetical
program-construction seam.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramType, TypedSignature


STATUS_OK = "OK"
STATUS_OUTPUT_UNKNOWN = "OUTPUT_UNKNOWN"
STATUS_RUNTIME_UNKNOWN = "RUNTIME_UNKNOWN"
STATUS_UNSUPPORTED_RESULT_TYPE = "UNSUPPORTED_RESULT_TYPE"

_UNKNOWN_RUNTIME_VALUE = object()
_TRUSTED_BUILTINS = {
    "abs": abs,
    "max": max,
    "min": min,
    "sum": sum,
}

CaseArgument = int | tuple[int, ...]
RawOutput = bool | int | str | tuple[object, ...] | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_int(program_type: ProgramType) -> bool:
    return program_type.constructor == "Int" and not program_type.arguments


def _is_bool(program_type: ProgramType) -> bool:
    return program_type.constructor == "Bool" and not program_type.arguments


def _is_list_int(program_type: ProgramType) -> bool:
    return (
        program_type.constructor == "List"
        and len(program_type.arguments) == 1
        and _is_int(program_type.arguments[0])
    )


def _parameter_mode(signature: TypedSignature) -> str:
    parameter_types = tuple(parameter.type for parameter in signature.parameters)
    if len(parameter_types) == 1 and _is_int(parameter_types[0]):
        return "one-int"
    if len(parameter_types) == 2 and all(_is_int(item) for item in parameter_types):
        return "two-int"
    if len(parameter_types) == 1 and _is_list_int(parameter_types[0]):
        return "one-list-int"
    raise ValueError(
        "root observation supports one/two Int parameters or one List[Int] parameter"
    )


def _declared_output_type(result: ProgramType) -> output_cells.OutputType | None:
    if _is_int(result):
        return output_cells.OutputType.INT
    if _is_bool(result):
        return output_cells.OutputType.BOOL
    if _is_list_int(result):
        return output_cells.OutputType.LIST_INT
    return None


def _normalize_cases(
    signature: TypedSignature,
    cases: Sequence[tuple[object, ...]],
) -> tuple[tuple[CaseArgument, ...], ...]:
    mode = _parameter_mode(signature)
    supplied = tuple(cases)
    if not supplied:
        raise ValueError("current-program observation needs at least one case")

    normalized: list[tuple[CaseArgument, ...]] = []
    expected_arity = len(signature.parameters)
    for case in supplied:
        if not isinstance(case, tuple) or len(case) != expected_arity:
            raise ValueError("every case must be a tuple matching signature arity")
        if mode in {"one-int", "two-int"}:
            if not all(type(value) is int for value in case):
                raise TypeError("Int case arguments must be exact Python integers")
            normalized.append(tuple(case))
            continue

        values = case[0]
        if not isinstance(values, (list, tuple)) or not all(
            type(value) is int for value in values
        ):
            raise TypeError("List[Int] cases must contain one integer list")
        normalized.append((tuple(values),))
    return tuple(normalized)


def _execution_arguments(
    signature: TypedSignature,
    case: tuple[CaseArgument, ...],
) -> tuple[object, ...]:
    if _parameter_mode(signature) == "one-list-int":
        return (list(case[0]),)
    return tuple(case)


def _snapshot_output(value: object) -> RawOutput:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) in {list, tuple}:
        return tuple(_snapshot_output(item) for item in value)
    return None


def _trusted_current_callable(
    compiled: tp.CompiledPython,
    entry_point: str,
):
    namespace: dict[str, object] = {
        "__builtins__": dict(_TRUSTED_BUILTINS),
    }
    exec(compiled.source, namespace)
    current = namespace.get(entry_point)
    if not callable(current):
        raise RuntimeError("trusted compiler output did not define its entry point")
    return current


@dataclass(frozen=True, slots=True)
class RootCaseObservation:
    case_index: int
    arguments: tuple[CaseArgument, ...]
    raw_output: RawOutput
    status: str
    output_encoding: output_cells.TypedOutputEncoding
    sha256: str


@dataclass(frozen=True, slots=True)
class CurrentProgramRootObservation:
    program_sha256: str
    compiled_sha256: str
    cases_sha256: str
    output_sha256: str
    output_projection_sha256: str
    cases: tuple[RootCaseObservation, ...]
    sha256: str

    @property
    def physical_output_cells(self) -> np.ndarray:
        cells = np.ascontiguousarray(
            np.stack(tuple(case.output_encoding.cells for case in self.cases))
        )
        cells.setflags(write=False)
        return cells

    @property
    def all_supported(self) -> bool:
        return all(case.status == STATUS_OK for case in self.cases)


def observe_current_program_root(
    program: tp.Program,
    cases: Sequence[tuple[object, ...]],
) -> CurrentProgramRootObservation:
    """Compile and observe one incumbent program over one typed case grid."""

    if not isinstance(program, tp.Program):
        raise TypeError("root observation requires the current typed Program")
    normalized_cases = _normalize_cases(program.signature, cases)
    cases_record = {
        "signature": program.signature.to_record(),
        "cases": normalized_cases,
    }
    cases_sha = sha256(_canonical_json(cases_record)).hexdigest()

    compiled = tp.compile(program)
    current = _trusted_current_callable(compiled, program.signature.entry_point)
    declared_type = _declared_output_type(program.signature.result)

    observations: list[RootCaseObservation] = []
    for case_index, arguments in enumerate(normalized_cases):
        try:
            raw_value = current(*_execution_arguments(program.signature, arguments))
        except Exception:
            raw_value = _UNKNOWN_RUNTIME_VALUE
            encoding = output_cells.TypedOutputCellBank.encode(
                raw_value,
                declared_type=declared_type,
            ) if declared_type is not None else output_cells.TypedOutputCellBank.encode(
                raw_value
            )
            status = STATUS_RUNTIME_UNKNOWN
            raw_snapshot: RawOutput = None
        else:
            raw_snapshot = _snapshot_output(raw_value)
            if declared_type is None:
                encoding = output_cells.TypedOutputCellBank.encode(
                    _UNKNOWN_RUNTIME_VALUE
                )
                status = STATUS_UNSUPPORTED_RESULT_TYPE
            else:
                encoding = output_cells.TypedOutputCellBank.encode(
                    raw_value,
                    declared_type=declared_type,
                )
                status = STATUS_OK if encoding.supported else STATUS_OUTPUT_UNKNOWN

        case_record = {
            "case_index": case_index,
            "arguments": arguments,
            "raw_output": raw_snapshot,
            "status": status,
            "output_encoding_sha256": encoding.sha256,
        }
        observations.append(
            RootCaseObservation(
                case_index=case_index,
                arguments=arguments,
                raw_output=raw_snapshot,
                status=status,
                output_encoding=encoding,
                sha256=sha256(_canonical_json(case_record)).hexdigest(),
            )
        )

    frozen_cases = tuple(observations)
    output_sha = sha256(
        _canonical_json(tuple(case.sha256 for case in frozen_cases))
    ).hexdigest()
    projection_sha = output_cells.TypedOutputCellBank.projection_sha256()
    observation_record = {
        "program_sha256": program.sha256,
        "compiled_sha256": compiled.sha256,
        "cases_sha256": cases_sha,
        "output_sha256": output_sha,
        "output_projection_sha256": projection_sha,
    }
    return CurrentProgramRootObservation(
        program_sha256=program.sha256,
        compiled_sha256=compiled.sha256,
        cases_sha256=cases_sha,
        output_sha256=output_sha,
        output_projection_sha256=projection_sha,
        cases=frozen_cases,
        sha256=sha256(_canonical_json(observation_record)).hexdigest(),
    )


__all__ = [
    "CurrentProgramRootObservation",
    "RootCaseObservation",
    "STATUS_OK",
    "STATUS_OUTPUT_UNKNOWN",
    "STATUS_RUNTIME_UNKNOWN",
    "STATUS_UNSUPPORTED_RESULT_TYPE",
    "observe_current_program_root",
]
