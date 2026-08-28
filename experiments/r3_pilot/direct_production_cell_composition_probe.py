"""One-shot fresh composition probe for direct production-cell decoding.

The four teachers are public, binary-scalar micro-programs.  The target does
not exist until after the decoder is fitted, and its natural prompt is decoded
exactly once.  This is a pre-canonical architecture probe; it does not read any
Simple-SWE task or protected evaluation material.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

from experiments.r3_pilot.direct_production_cell_decoder import (  # noqa: E402
    DirectCompletion,
    DirectProductionCellDecoder,
    ProductionRace,
)

FORMAT = "humanbrain.direct-production-cell-composition-probe.v1"
PASS = "PASS"
STOP = "STOP_DIRECT_PROMPT_ASSEMBLY_COMPOSITION"
_FIT_EPOCHS = 8


@dataclass(frozen=True, slots=True)
class _Case:
    first: int
    second: int
    expected: int


@dataclass(frozen=True, slots=True)
class _FreshTarget:
    prompt: str
    signature: TypedSignature
    expected_tokens: tuple[str, ...]
    expected_source: str
    cases: tuple[_Case, ...]


def _binary_int_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter("first", integer),
            ProgramParameter("second", integer),
        ),
        result=integer,
    )


def _public_teachers() -> tuple[tp.Example, ...]:
    signature = _binary_int_signature()
    return (
        tp.Example(
            prompt="Given integers first and second, return first minus second.",
            signature=signature,
            teacher_tokens=("subtract", "variable-1", "variable-0"),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return the absolute value of "
                "first; second is unused."
            ),
            signature=signature,
            teacher_tokens=("absolute", "variable-1"),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return second unchanged; "
                "first is unused."
            ),
            signature=signature,
            teacher_tokens=("variable-0",),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return first unchanged; "
                "second is unused."
            ),
            signature=signature,
            teacher_tokens=("variable-1",),
        ),
    )


def _fresh_target() -> _FreshTarget:
    return _FreshTarget(
        prompt=(
            "Given integers first and second, return second minus the absolute "
            "value of first."
        ),
        signature=_binary_int_signature(),
        expected_tokens=("subtract", "variable-0", "absolute", "variable-1"),
        expected_source=(
            "def solve(first, second):\n"
            "    return second - abs(first)\n"
        ),
        cases=(
            _Case(3, 10, 7),
            _Case(-3, 10, 7),
            _Case(3, -10, -13),
            _Case(-3, -10, -13),
            _Case(0, 5, 5),
            _Case(5, 0, -5),
            _Case(-5, 0, -5),
            _Case(2, 2, 0),
        ),
    )


def _score_source(source: str, target: _FreshTarget) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__builtins__": {"abs": abs, "sum": sum},
    }
    try:
        exec(compile(source, "<direct-production-cell-probe>", "exec"), namespace)
        solve = namespace[target.signature.entry_point]
    except Exception as error:  # pragma: no cover - exercised only by a bad completion
        return {
            "all_pass": False,
            "passed": 0,
            "total": len(target.cases),
            "error": {"type": type(error).__name__, "message": str(error)},
            "cases": [],
        }

    records: list[dict[str, object]] = []
    for case in target.cases:
        try:
            actual = solve(case.first, case.second)  # type: ignore[operator]
            passed = actual == case.expected
            records.append(
                {
                    "args": [case.first, case.second],
                    "expected": case.expected,
                    "actual": actual,
                    "pass": passed,
                }
            )
        except Exception as error:  # pragma: no cover - bad programs may raise
            records.append(
                {
                    "args": [case.first, case.second],
                    "expected": case.expected,
                    "actual": None,
                    "pass": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
    passed = sum(bool(record["pass"]) for record in records)
    return {
        "all_pass": passed == len(target.cases),
        "passed": passed,
        "total": len(target.cases),
        "cases": records,
    }


def _race_audit(race: ProductionRace) -> dict[str, object]:
    accepted_is_legal = race.accepted_token in race.allowed_tokens
    assembly_present = (
        race.prompt_cells > 0
        and race.signature_cells > 0
        and race.role_cells > 0
        and race.active_assembly_cells > 0
    )
    physical = accepted_is_legal and assembly_present and race.spike_time_ms > 0.0
    return {
        "position": race.position,
        "accepted_token": race.accepted_token,
        "winner_cell": race.winner_cell,
        "allowed_tokens": list(race.allowed_tokens),
        "candidate_count": len(race.allowed_tokens),
        "accepted_is_legal": accepted_is_legal,
        "spike_time_ms": race.spike_time_ms,
        "prompt_cells": race.prompt_cells,
        "signature_cells": race.signature_cells,
        "role_cells": race.role_cells,
        "active_assembly_cells": race.active_assembly_cells,
        "physical": physical,
    }


def _completion_record(
    completion: DirectCompletion | None,
    error: Exception | None,
    target: _FreshTarget,
) -> dict[str, object]:
    if completion is None:
        assert error is not None
        return {
            "completed": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "tokens": [],
            "source": None,
            "exact_tokens": False,
            "exact_source": False,
            "execution": {
                "all_pass": False,
                "passed": 0,
                "total": len(target.cases),
                "cases": [],
            },
            "readout_immutable": False,
            "physical_audit_pass": False,
            "all_decisions_competitive": False,
            "races": [],
        }

    races = [_race_audit(race) for race in completion.races]
    readout_immutable = (
        completion.readout_before_sha256 == completion.readout_after_sha256
    )
    physical_audit_pass = (
        len(races) == len(completion.program.tokens)
        and readout_immutable
        and all(bool(race["physical"]) for race in races)
    )
    return {
        "completed": True,
        "tokens": list(completion.program.tokens),
        "source": completion.source,
        "exact_tokens": completion.program.tokens == target.expected_tokens,
        "exact_source": completion.source == target.expected_source,
        "execution": _score_source(completion.source, target),
        "readout_before_sha256": completion.readout_before_sha256,
        "readout_after_sha256": completion.readout_after_sha256,
        "readout_immutable": readout_immutable,
        "physical_audit_pass": physical_audit_pass,
        "all_decisions_competitive": bool(races)
        and all(int(race["candidate_count"]) > 1 for race in races),
        "races": races,
    }


def _teacher_execution_controls(
    teachers: tuple[tp.Example, ...], target: _FreshTarget
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for index, teacher in enumerate(teachers):
        source = tp.compile(tp.Program(teacher.teacher_tokens, teacher.signature)).source
        controls.append(
            {
                "teacher": index + 1,
                "tokens": list(teacher.teacher_tokens),
                "source": source,
                "execution": _score_source(source, target),
            }
        )
    return controls


def run_probe() -> dict[str, object]:
    teachers = _public_teachers()
    decoder = DirectProductionCellDecoder.fit(teachers, epochs=_FIT_EPOCHS)
    checkpoint_after_fit = decoder.checkpoint_sha256

    # The target is deliberately constructed only after fitting is complete.
    target = _fresh_target()
    target_completion: DirectCompletion | None = None
    target_error: Exception | None = None
    try:
        target_completion = decoder.complete(target.prompt, target.signature)
    except Exception as error:  # exactly one attempt; never retry
        target_error = error

    prompt_swap_completion: DirectCompletion | None = None
    prompt_swap_error: Exception | None = None
    try:
        prompt_swap_completion = decoder.complete(
            teachers[1].prompt,
            target.signature,
        )
    except Exception as error:  # one fixed control attempt
        prompt_swap_error = error

    target_record = _completion_record(target_completion, target_error, target)
    prompt_swap_record = _completion_record(
        prompt_swap_completion,
        prompt_swap_error,
        target,
    )
    whole_teacher_controls = _teacher_execution_controls(teachers, target)
    whole_teacher_retrieval_rejected = all(
        not bool(control["execution"]["all_pass"])  # type: ignore[index]
        for control in whole_teacher_controls
    )
    prompt_swap_rejected = (
        bool(prompt_swap_record["completed"])
        and not bool(prompt_swap_record["exact_tokens"])
        and not bool(prompt_swap_record["execution"]["all_pass"])  # type: ignore[index]
    )
    checkpoint_immutable = decoder.checkpoint_sha256 == checkpoint_after_fit
    passed = (
        bool(target_record["completed"])
        and bool(target_record["exact_tokens"])
        and bool(target_record["exact_source"])
        and bool(target_record["execution"]["all_pass"])  # type: ignore[index]
        and bool(target_record["physical_audit_pass"])
        and bool(target_record["all_decisions_competitive"])
        and decoder.local_updates > 0
        and decoder.structural_recruitments > 0
        and checkpoint_immutable
        and prompt_swap_rejected
        and bool(prompt_swap_record["physical_audit_pass"])
        and whole_teacher_retrieval_rejected
    )

    return {
        "format": FORMAT,
        "claim_boundary": (
            "Fresh pre-canonical binary-scalar composition probe only; no "
            "Simple-SWE, held-out, M0, C1, or spent task material was read."
        ),
        "protocol": {
            "fit_examples": len(teachers),
            "fit_epochs": _FIT_EPOCHS,
            "target_instantiated_after_fit": True,
            "target_completion_attempts": 1,
            "prompt_swap_completion_attempts": 1,
            "parameter_tuning": False,
        },
        "checkpoint": {
            "sha256": checkpoint_after_fit,
            "sha256_after_controls": decoder.checkpoint_sha256,
            "immutable": checkpoint_immutable,
            "local_updates": decoder.local_updates,
            "structural_recruitments": decoder.structural_recruitments,
        },
        "teachers": [
            {
                "prompt": teacher.prompt,
                "signature": teacher.signature.to_record(),
                "tokens": list(teacher.teacher_tokens),
            }
            for teacher in teachers
        ],
        "target": {
            "prompt": target.prompt,
            "signature": target.signature.to_record(),
            "expected_tokens": list(target.expected_tokens),
            "expected_source": target.expected_source,
            "cases": [
                {
                    "args": [case.first, case.second],
                    "expected": case.expected,
                }
                for case in target.cases
            ],
            "completion": target_record,
        },
        "controls": {
            "prompt_swap": {
                "prompt": teachers[1].prompt,
                "completion": prompt_swap_record,
                "rejected_target_solution": prompt_swap_rejected,
            },
            "whole_teacher_execution": whole_teacher_controls,
            "whole_teacher_retrieval_rejected": whole_teacher_retrieval_rejected,
        },
        "verdict": PASS if passed else STOP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe()
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
