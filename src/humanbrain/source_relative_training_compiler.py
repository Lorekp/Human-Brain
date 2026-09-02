"""Training-only compiler from successful token tapes to generic motor traces.

The compiler may inspect one public successful before/after pair while learning.
It partitions the after tape into deterministic source-contiguous runs whose
first token is an unambiguous source occurrence, then emits only REBIND, COPY,
ADVANCE, EMIT, and END activity for :class:`SourceRelativeMotorDemonstration`.

This is explicit trajectory scaffolding, not neuronal program induction.  The
compiled alignment and after tape are never inference inputs.  Novel tokens,
ambiguous run anchors, and transformations outside the forward-copy motor
language fail closed.  There is no diff, syntax, task, patch, or source-position
interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from humanbrain.source_relative_action_tape import (
    MotorStep,
    SourceRelativeMotorDemonstration,
    TapeOperation,
)

__all__ = [
    "SourceRelativeTrainingCompilerError",
    "compile_successful_token_tapes",
]


class SourceRelativeTrainingCompilerError(ValueError):
    """The successful pair is outside the lossless generic motor subset."""


@dataclass(frozen=True, slots=True)
class _ContiguousRun:
    source_start: int
    length: int


def _require_event(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    return value


def _require_tape(value: object, name: str, *, allow_empty: bool) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    tape = value
    if not allow_empty and not tape:
        raise ValueError(f"{name} must not be empty")
    for index, token in enumerate(tape):
        _require_event(token, f"{name}[{index}]")
    return tape


def _partition_contiguous_runs(
    before: tuple[bytes, ...],
    after: tuple[bytes, ...],
) -> tuple[_ContiguousRun, ...]:
    positions: dict[bytes, list[int]] = {}
    for source_index, token in enumerate(before):
        positions.setdefault(token, []).append(source_index)

    runs: list[_ContiguousRun] = []
    after_index = 0
    while after_index < len(after):
        anchors = positions.get(after[after_index], ())
        if not anchors:
            raise SourceRelativeTrainingCompilerError(
                f"after token at offset {after_index} is absent from the source tape"
            )
        if len(anchors) != 1:
            raise SourceRelativeTrainingCompilerError(
                f"after run at offset {after_index} has an ambiguous source anchor"
            )

        source_start = anchors[0]
        length = 1
        while (
            source_start + length < len(before)
            and after_index + length < len(after)
            and before[source_start + length] == after[after_index + length]
        ):
            length += 1
        runs.append(_ContiguousRun(source_start=source_start, length=length))
        after_index += length
    return tuple(runs)


def _compile_motor_steps(
    before: tuple[bytes, ...],
    runs: tuple[_ContiguousRun, ...],
) -> tuple[tuple[bytes, ...], tuple[MotorStep, ...]]:
    cues: list[bytes] = []
    cue_ordinals: dict[bytes, int] = {}
    steps: list[MotorStep] = []
    for run in runs:
        anchor = before[run.source_start]
        cue_ordinal = cue_ordinals.get(anchor)
        if cue_ordinal is None:
            cue_ordinal = len(cues)
            cues.append(anchor)
            cue_ordinals[anchor] = cue_ordinal
        steps.append(MotorStep(TapeOperation.REBIND, cue_ordinal))
        for offset in range(run.length):
            if offset:
                steps.append(MotorStep(TapeOperation.ADVANCE))
            steps.append(MotorStep(TapeOperation.COPY))
            steps.append(MotorStep(TapeOperation.EMIT))
    steps.append(MotorStep(TapeOperation.END))
    return tuple(cues), tuple(steps)


def _materialize(
    before: tuple[bytes, ...],
    cues: tuple[bytes, ...],
    steps: tuple[MotorStep, ...],
) -> tuple[bytes, ...]:
    head: int | None = None
    held: bytes | None = None
    emitted: list[bytes] = []
    for step in steps:
        if step.operation is TapeOperation.REBIND:
            cue_ordinal = step.cue_ordinal
            if cue_ordinal is None:
                raise RuntimeError("compiled REBIND lost its cue ordinal")
            matches = tuple(
                index for index, token in enumerate(before) if token == cues[cue_ordinal]
            )
            if len(matches) != 1:
                raise RuntimeError("compiled anchor lost its unique source occurrence")
            head = matches[0]
            held = None
        elif step.operation is TapeOperation.COPY:
            if head is None:
                raise RuntimeError("compiled COPY lost its source occurrence")
            held = before[head]
        elif step.operation is TapeOperation.ADVANCE:
            if head is None or head + 1 >= len(before):
                raise RuntimeError("compiled ADVANCE left the source tape")
            head += 1
            held = None
        elif step.operation is TapeOperation.EMIT:
            if held is None:
                raise RuntimeError("compiled EMIT lost its copied activation")
            emitted.append(held)
        elif step.operation is TapeOperation.END:
            break
    return tuple(emitted)


def compile_successful_token_tapes(
    *,
    context_event: bytes,
    before_tokens: tuple[bytes, ...],
    after_tokens: tuple[bytes, ...],
) -> SourceRelativeMotorDemonstration:
    """Compile one successful public pair into a lossless training trajectory.

    Only ``context_event`` and the returned cue inventory/source tape are later
    needed by the action-tape learner.  ``after_tokens`` and the private run
    partition have no inference path.
    """

    context = _require_event(context_event, "context_event")
    before = _require_tape(before_tokens, "before_tokens", allow_empty=False)
    after = _require_tape(after_tokens, "after_tokens", allow_empty=True)
    runs = _partition_contiguous_runs(before, after)
    cues, steps = _compile_motor_steps(before, runs)
    if _materialize(before, cues, steps) != after:
        raise SourceRelativeTrainingCompilerError(
            "compiled generic motor trajectory does not losslessly reproduce the after tape"
        )
    return SourceRelativeMotorDemonstration(
        context_event=context,
        cue_inventory=cues,
        source_tokens=before,
        motor_steps=steps,
    )
