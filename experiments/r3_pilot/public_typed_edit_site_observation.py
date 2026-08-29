"""Read-only observations for one value site of the current typed program.

The sole program-semantics boundary is
:func:`humanbrain.typed_program.compile_edit_site`.  Explicit local scope
contexts are passed to its trusted ``edit_site`` function in de-Bruijn order,
then each raw result is projected into the factorized typed-output cells.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain import typed_program as tp


STATUS_OK = "OK"
STATUS_OUTPUT_UNKNOWN = "OUTPUT_UNKNOWN"
STATUS_RUNTIME_UNKNOWN = "RUNTIME_UNKNOWN"
STATUS_UNSUPPORTED_RESULT_TYPE = "UNSUPPORTED_RESULT_TYPE"

_UNKNOWN_RUNTIME_VALUE = object()
_SUPPORTED_TYPE_LABELS = frozenset(output_type.value for output_type in output_cells.OutputType)
_TRUSTED_BUILTINS = {
    "abs": abs,
    "max": max,
    "min": min,
    "sum": sum,
}

ScopeValue = bool | int | tuple[int, ...]
RawOutput = bool | int | str | tuple[object, ...] | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalize_contexts(
    scope_type_labels: tuple[str, ...],
    contexts: Sequence[tuple[object, ...]],
) -> tuple[tuple[ScopeValue, ...], ...]:
    if any(label not in _SUPPORTED_TYPE_LABELS for label in scope_type_labels):
        raise ValueError("edit-site observation supports Int, Bool, and List[Int] scopes")
    supplied = tuple(contexts)
    if not supplied:
        raise ValueError("edit-site observation needs at least one local scope context")

    normalized: list[tuple[ScopeValue, ...]] = []
    for context in supplied:
        if not isinstance(context, tuple) or len(context) != len(scope_type_labels):
            raise ValueError("every context must be a tuple matching local scope arity")
        values: list[ScopeValue] = []
        for label, value in zip(scope_type_labels, context, strict=True):
            if label == output_cells.OutputType.INT.value:
                if type(value) is not int:
                    raise TypeError("Int scope values must be exact Python integers")
                values.append(value)
            elif label == output_cells.OutputType.BOOL.value:
                if type(value) is not bool:
                    raise TypeError("Bool scope values must be exact Python booleans")
                values.append(value)
            else:
                if not isinstance(value, (list, tuple)) or not all(
                    type(item) is int for item in value
                ):
                    raise TypeError("List[Int] scope values must be integer lists")
                values.append(tuple(value))
        normalized.append(tuple(values))
    return tuple(normalized)


def _execution_arguments(
    scope_type_labels: tuple[str, ...],
    context: tuple[ScopeValue, ...],
) -> tuple[object, ...]:
    return tuple(
        list(value) if label == output_cells.OutputType.LIST_INT.value else value
        for label, value in zip(scope_type_labels, context, strict=True)
    )


def _declared_output_type(label: str) -> output_cells.OutputType | None:
    try:
        return output_cells.OutputType(label)
    except ValueError:
        return None


def _snapshot_output(value: object) -> RawOutput:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) in {list, tuple}:
        return tuple(_snapshot_output(item) for item in value)
    return None


def _trusted_edit_site_callable(compiled: tp.CompiledEditSite):
    namespace: dict[str, object] = {
        "__builtins__": dict(_TRUSTED_BUILTINS),
    }
    exec(compiled.source, namespace)
    current = namespace.get("edit_site")
    if not callable(current):
        raise RuntimeError("trusted compiler output did not define edit_site")
    return current


@dataclass(frozen=True, slots=True)
class SiteContextObservation:
    context_index: int
    scope_values: tuple[ScopeValue, ...]
    raw_output: RawOutput
    status: str
    output_encoding: output_cells.TypedOutputEncoding
    sha256: str


@dataclass(frozen=True, slots=True)
class CurrentEditSiteObservation:
    program_sha256: str
    path: tuple[int, ...]
    result_type_label: str
    scope_type_labels: tuple[str, ...]
    compiled_sha256: str
    contexts_sha256: str
    output_sha256: str
    output_projection_sha256: str
    contexts: tuple[SiteContextObservation, ...]
    sha256: str

    @property
    def physical_output_cells(self) -> np.ndarray:
        cells = np.ascontiguousarray(
            np.stack(tuple(context.output_encoding.cells for context in self.contexts))
        )
        cells.setflags(write=False)
        return cells

    @property
    def all_supported(self) -> bool:
        return all(context.status == STATUS_OK for context in self.contexts)


def observe_current_edit_site(
    program: tp.Program,
    path: tuple[int, ...],
    contexts: Sequence[tuple[object, ...]],
) -> CurrentEditSiteObservation:
    """Observe one already-current value site over explicit local contexts."""

    if not isinstance(program, tp.Program):
        raise TypeError("edit-site observation requires the current typed Program")
    compiled = tp.compile_edit_site(program, path)
    normalized_contexts = _normalize_contexts(compiled.scope_type_labels, contexts)
    contexts_record = {
        "path": compiled.path,
        "scope_type_labels": compiled.scope_type_labels,
        "contexts": normalized_contexts,
    }
    contexts_sha = sha256(_canonical_json(contexts_record)).hexdigest()

    current = _trusted_edit_site_callable(compiled)
    declared_type = _declared_output_type(compiled.result_type_label)
    observations: list[SiteContextObservation] = []
    for context_index, scope_values in enumerate(normalized_contexts):
        try:
            raw_value = current(
                *_execution_arguments(compiled.scope_type_labels, scope_values)
            )
        except Exception:
            raw_value = _UNKNOWN_RUNTIME_VALUE
            encoding = (
                output_cells.TypedOutputCellBank.encode(
                    raw_value,
                    declared_type=declared_type,
                )
                if declared_type is not None
                else output_cells.TypedOutputCellBank.encode(raw_value)
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

        context_record = {
            "context_index": context_index,
            "scope_values": scope_values,
            "raw_output": raw_snapshot,
            "status": status,
            "output_encoding_sha256": encoding.sha256,
        }
        observations.append(
            SiteContextObservation(
                context_index=context_index,
                scope_values=scope_values,
                raw_output=raw_snapshot,
                status=status,
                output_encoding=encoding,
                sha256=sha256(_canonical_json(context_record)).hexdigest(),
            )
        )

    frozen_contexts = tuple(observations)
    output_sha = sha256(
        _canonical_json(tuple(context.sha256 for context in frozen_contexts))
    ).hexdigest()
    projection_sha = output_cells.TypedOutputCellBank.projection_sha256()
    observation_record = {
        "program_sha256": program.sha256,
        "path": compiled.path,
        "result_type_label": compiled.result_type_label,
        "scope_type_labels": compiled.scope_type_labels,
        "compiled_sha256": compiled.sha256,
        "contexts_sha256": contexts_sha,
        "output_sha256": output_sha,
        "output_projection_sha256": projection_sha,
    }
    return CurrentEditSiteObservation(
        program_sha256=program.sha256,
        path=compiled.path,
        result_type_label=compiled.result_type_label,
        scope_type_labels=compiled.scope_type_labels,
        compiled_sha256=compiled.sha256,
        contexts_sha256=contexts_sha,
        output_sha256=output_sha,
        output_projection_sha256=projection_sha,
        contexts=frozen_contexts,
        sha256=sha256(_canonical_json(observation_record)).hexdigest(),
    )


__all__ = [
    "CurrentEditSiteObservation",
    "STATUS_OK",
    "STATUS_OUTPUT_UNKNOWN",
    "STATUS_RUNTIME_UNKNOWN",
    "STATUS_UNSUPPORTED_RESULT_TYPE",
    "SiteContextObservation",
    "observe_current_edit_site",
]
