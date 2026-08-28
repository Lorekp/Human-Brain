"""One-shot fresh composition probe for the role-gated production decoder."""

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

FORMAT = "humanbrain.direct-production-cell-role-gated-probe.v1"
PASS = "PASS"
STOP = "STOP_ROLE_GATED_COMPOSITION"
EPOCHS = 8


@dataclass(frozen=True, slots=True)
class _Case:
    alpha: int
    beta: int
    expected: int


@dataclass(frozen=True, slots=True)
class _Target:
    prompt: str
    signature: TypedSignature
    expected_tokens: tuple[str, ...]
    expected_source: str
    cases: tuple[_Case, ...]


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter("alpha", integer),
            ProgramParameter("beta", integer),
        ),
        result=integer,
    )


def _teachers() -> tuple[tp.Example, ...]:
    signature = _signature()
    return (
        tp.Example(
            prompt=(
                "For integers alpha and beta, return the magnitude of alpha; "
                "ignore beta."
            ),
            signature=signature,
            teacher_tokens=("absolute", "variable-1"),
        ),
        tp.Example(
            prompt=(
                "For integers alpha and beta, return the magnitude of beta; "
                "ignore alpha."
            ),
            signature=signature,
            teacher_tokens=("absolute", "variable-0"),
        ),
        tp.Example(
            prompt=(
                "For integers alpha and beta, apply magnitude twice to alpha; "
                "ignore beta."
            ),
            signature=signature,
            teacher_tokens=("absolute", "absolute", "variable-1"),
        ),
        tp.Example(
            prompt=(
                "For integers alpha and beta, return the quantity alpha plus "
                "beta, then add alpha again."
            ),
            signature=signature,
            teacher_tokens=(
                "add",
                "add",
                "variable-1",
                "variable-0",
                "variable-1",
            ),
        ),
    )


def _fresh_target() -> _Target:
    return _Target(
        prompt=(
            "For integers alpha and beta, return the magnitude of the quantity "
            "alpha plus beta."
        ),
        signature=_signature(),
        expected_tokens=("absolute", "add", "variable-1", "variable-0"),
        expected_source="def solve(alpha, beta):\n    return abs(alpha + beta)\n",
        cases=(
            _Case(3, 5, 8),
            _Case(-3, 5, 2),
            _Case(3, -5, 2),
            _Case(-3, -5, 8),
            _Case(0, 0, 0),
            _Case(9, -9, 0),
            _Case(-8, 3, 5),
            _Case(2, 7, 9),
        ),
    )


def _execute(source: str, target: _Target) -> dict[str, object]:
    namespace: dict[str, object] = {"__builtins__": {"abs": abs, "sum": sum}}
    try:
        exec(compile(source, "<role-gated-probe>", "exec"), namespace)
        solve = namespace[target.signature.entry_point]
    except Exception as error:
        return {
            "all_pass": False,
            "passed": 0,
            "total": len(target.cases),
            "cases": [],
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    records: list[dict[str, object]] = []
    for case in target.cases:
        try:
            actual = solve(case.alpha, case.beta)  # type: ignore[operator]
            records.append(
                {
                    "args": [case.alpha, case.beta],
                    "expected": case.expected,
                    "actual": actual,
                    "pass": actual == case.expected,
                }
            )
        except Exception as error:
            records.append(
                {
                    "args": [case.alpha, case.beta],
                    "expected": case.expected,
                    "actual": None,
                    "pass": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
    passed = sum(bool(record["pass"]) for record in records)
    return {
        "all_pass": passed == len(target.cases),
        "passed": passed,
        "total": len(target.cases),
        "cases": records,
    }


def _race_record(race: ProductionRace) -> dict[str, object]:
    physical = (
        race.accepted_token in race.allowed_tokens
        and race.spike_time_ms > 0.0
        and race.contextual_unigram_cells > 0
        and race.contextual_pair_cells > 0
        and race.active_assembly_cells > 0
    )
    return {
        "position": race.position,
        "accepted_token": race.accepted_token,
        "winner_cell": race.winner_cell,
        "allowed_tokens": list(race.allowed_tokens),
        "candidate_count": len(race.allowed_tokens),
        "accepted_is_legal": race.accepted_token in race.allowed_tokens,
        "spike_time_ms": race.spike_time_ms,
        "contextual_unigram_cells": race.contextual_unigram_cells,
        "contextual_pair_cells": race.contextual_pair_cells,
        "active_assembly_cells": race.active_assembly_cells,
        "physical": physical,
    }


def _completion_record(
    completion: DirectCompletion | None,
    error: Exception | None,
    target: _Target,
) -> dict[str, object]:
    if completion is None:
        assert error is not None
        return {
            "completed": False,
            "tokens": [],
            "source": None,
            "exact_tokens": False,
            "exact_source": False,
            "execution": {"all_pass": False, "passed": 0, "total": 8, "cases": []},
            "readout_immutable": False,
            "physical": False,
            "races": [],
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    races = [_race_record(race) for race in completion.races]
    readout_immutable = (
        completion.readout_before_sha256 == completion.readout_after_sha256
    )
    physical = (
        len(races) == len(completion.program.tokens)
        and readout_immutable
        and all(bool(race["physical"]) for race in races)
        and all(int(race["candidate_count"]) > 1 for race in races)
    )
    return {
        "completed": True,
        "tokens": list(completion.program.tokens),
        "source": completion.source,
        "exact_tokens": completion.program.tokens == target.expected_tokens,
        "exact_source": completion.source == target.expected_source,
        "execution": _execute(completion.source, target),
        "readout_before_sha256": completion.readout_before_sha256,
        "readout_after_sha256": completion.readout_after_sha256,
        "readout_immutable": readout_immutable,
        "physical": physical,
        "races": races,
    }


def _teacher_controls(
    teachers: tuple[tp.Example, ...], target: _Target
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for index, teacher in enumerate(teachers):
        source = tp.compile(tp.Program(teacher.teacher_tokens, teacher.signature)).source
        controls.append(
            {
                "teacher": index + 1,
                "tokens": list(teacher.teacher_tokens),
                "source": source,
                "execution": _execute(source, target),
            }
        )
    return controls


def run_probe() -> dict[str, object]:
    teachers = _teachers()
    decoder = DirectProductionCellDecoder.fit(teachers, epochs=EPOCHS)
    checkpoint_sha256 = decoder.checkpoint_sha256

    # This object, including its prompt and expected program, exists only post-fit.
    target = _fresh_target()
    target_completion: DirectCompletion | None = None
    target_error: Exception | None = None
    try:
        target_completion = decoder.complete(target.prompt, target.signature)
    except Exception as error:
        target_error = error

    swap_completion: DirectCompletion | None = None
    swap_error: Exception | None = None
    try:
        swap_completion = decoder.complete(teachers[3].prompt, target.signature)
    except Exception as error:
        swap_error = error

    target_record = _completion_record(target_completion, target_error, target)
    swap_record = _completion_record(swap_completion, swap_error, target)
    teacher_controls = _teacher_controls(teachers, target)
    teacher_controls_rejected = all(
        not bool(control["execution"]["all_pass"])  # type: ignore[index]
        for control in teacher_controls
    )
    swap_rejected = (
        bool(swap_record["completed"])
        and bool(swap_record["physical"])
        and not bool(swap_record["exact_tokens"])
        and not bool(swap_record["execution"]["all_pass"])  # type: ignore[index]
    )
    checkpoint_immutable = decoder.checkpoint_sha256 == checkpoint_sha256
    passed = (
        bool(target_record["completed"])
        and bool(target_record["physical"])
        and bool(target_record["exact_tokens"])
        and bool(target_record["exact_source"])
        and bool(target_record["execution"]["all_pass"])  # type: ignore[index]
        and decoder.local_updates > 0
        and decoder.structural_recruitments > 0
        and checkpoint_immutable
        and swap_rejected
        and teacher_controls_rejected
    )

    return {
        "format": FORMAT,
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "sha256_after_completions": decoder.checkpoint_sha256,
            "immutable": checkpoint_immutable,
            "local_updates": decoder.local_updates,
            "structural_recruitments": decoder.structural_recruitments,
        },
        "teachers": [
            {
                "prompt": teacher.prompt,
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
                {"args": [case.alpha, case.beta], "expected": case.expected}
                for case in target.cases
            ],
            "completion": target_record,
        },
        "controls": {
            "prompt_swap": {
                "prompt": teachers[3].prompt,
                "completion": swap_record,
                "rejected": swap_rejected,
            },
            "whole_teacher_execution": teacher_controls,
            "whole_teacher_execution_rejected": teacher_controls_rejected,
        },
        "verdict": PASS if passed else STOP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    result = run_probe()
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
