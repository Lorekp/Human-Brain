"""Freely rerunnable visible composition matrix for the direct cell decoder.

This is a development instrument, not frozen evidence.  One visible two-Int
curriculum is fitted once, replayed separately, then six absent combinations
are decoded once each.  Every operation, structural role, and variable used by
the compositions occurs in the public teachers; no hidden, canonical, or
spent task is imported.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.readout import SparseCellReadout  # noqa: E402
from humanbrain.sdr_union import SDRProjector  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

from experiments.r3_pilot import direct_production_cell_decoder as direct  # noqa: E402

_FIT_EPOCHS = 4


@dataclass(frozen=True, slots=True)
class VisibleProgram:
    name: str
    prompt: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    name: str
    expected_tokens: tuple[str, ...]
    actual_tokens: tuple[str, ...]
    exact: bool
    failure: str | None


@dataclass(frozen=True, slots=True)
class CompositionResult:
    name: str
    expected_tokens: tuple[str, ...]
    actual_tokens: tuple[str, ...]
    exact: bool
    first_mismatch: int | None
    expected_at_mismatch: str | None
    actual_at_mismatch: str | None
    root_allowed_currents_pa: tuple[tuple[str, float], ...]
    failure: str | None


@dataclass(frozen=True, slots=True)
class DevelopmentMatrixReport:
    epochs: int
    elapsed_seconds: float
    local_updates: int
    structural_recruitments: int
    teacher_exact: int
    teacher_total: int
    composition_exact: int
    composition_total: int
    teacher_replays: tuple[ReplayResult, ...]
    compositions: tuple[CompositionResult, ...]


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter("left", integer),
            ProgramParameter("right", integer),
        ),
        result=integer,
    )


def visible_teachers() -> tuple[VisibleProgram, ...]:
    return (
        VisibleProgram(
            "add_abs_left_left",
            "With integers left and right, add absolute left to left.",
            ("add", "absolute", "variable-1", "variable-1"),
        ),
        VisibleProgram(
            "add_right_abs_right",
            "With integers left and right, add right to absolute right.",
            ("add", "variable-0", "absolute", "variable-0"),
        ),
        VisibleProgram(
            "subtract_abs_right_right",
            "With integers left and right, subtract right from absolute right.",
            ("subtract", "absolute", "variable-0", "variable-0"),
        ),
        VisibleProgram(
            "subtract_left_abs_left",
            "With integers left and right, subtract absolute left from left.",
            ("subtract", "variable-1", "absolute", "variable-1"),
        ),
        VisibleProgram(
            "absolute_add_right_left",
            "With integers left and right, return absolute of right plus left.",
            ("absolute", "add", "variable-0", "variable-1"),
        ),
        VisibleProgram(
            "absolute_subtract_right_left",
            "With integers left and right, return absolute of right minus left.",
            ("absolute", "subtract", "variable-0", "variable-1"),
        ),
    )


def visible_compositions() -> tuple[VisibleProgram, ...]:
    return (
        VisibleProgram(
            "add_abs_left_right",
            "With integers left and right, add absolute left to right.",
            ("add", "absolute", "variable-1", "variable-0"),
        ),
        VisibleProgram(
            "add_left_abs_right",
            "With integers left and right, add left to absolute right.",
            ("add", "variable-1", "absolute", "variable-0"),
        ),
        VisibleProgram(
            "subtract_abs_left_right",
            "With integers left and right, subtract right from absolute left.",
            ("subtract", "absolute", "variable-1", "variable-0"),
        ),
        VisibleProgram(
            "subtract_left_abs_right",
            "With integers left and right, subtract absolute right from left.",
            ("subtract", "variable-1", "absolute", "variable-0"),
        ),
        VisibleProgram(
            "absolute_add_left_right",
            "With integers left and right, return absolute of left plus right.",
            ("absolute", "add", "variable-1", "variable-0"),
        ),
        VisibleProgram(
            "absolute_subtract_left_right",
            "With integers left and right, return absolute of left minus right.",
            ("absolute", "subtract", "variable-1", "variable-0"),
        ),
    )


def _first_mismatch(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> tuple[int | None, str | None, str | None]:
    for position, (expected_token, actual_token) in enumerate(zip(expected, actual)):
        if expected_token != actual_token:
            return position, expected_token, actual_token
    if len(expected) == len(actual):
        return None, None, None
    position = min(len(expected), len(actual))
    return (
        position,
        expected[position] if position < len(expected) else None,
        actual[position] if position < len(actual) else None,
    )


def _root_currents(
    decoder: direct.DirectProductionCellDecoder,
    prompt: str,
    signature: TypedSignature,
) -> tuple[tuple[str, float], ...]:
    eligibility = np.zeros(direct._ASSEMBLY_CELLS, dtype=np.float64)
    readout = SparseCellReadout.from_checkpoint(
        output_cells=len(direct._PRODUCTION_SPECS),
        reservoir_cells=direct._ASSEMBLY_CELLS,
        fan_in=direct._READOUT_FAN_IN,
        reservoir_eligibility=eligibility,
        output_heterogeneity_pa=decoder._output_heterogeneity_pa,
        pacemaker_synapses_pa=decoder._pacemaker_synapses_pa,
        arrays=direct._thaw_arrays(decoder._arrays),
        counters=decoder._counters,
    )
    projector = SDRProjector(
        n_bits=direct._ASSEMBLY_CELLS,
        w_ones=direct._ASSEMBLY_ONES,
        seed=direct._RANDOM_SEED,
    )
    state = tp._TypedHoleState(signature)
    role = direct._role_text(state, direct._HoleCursor((), "root", -1))
    eligibility[:], _ = direct._drive_assembly(
        decoder._tokenizer,
        projector,
        prompt,
        direct._signature_text(signature),
        role,
    )
    currents = readout.current(direct._OUTPUT_BASE_CURRENT_PA)
    return tuple(
        (spec.token, float(currents[direct._OUTPUT_BY_TOKEN[spec.token]]))
        for spec in state.allowed()
    )


def _decode(
    decoder: direct.DirectProductionCellDecoder,
    case: VisibleProgram,
    signature: TypedSignature,
) -> tuple[tuple[str, ...], str | None]:
    try:
        return decoder.complete(case.prompt, signature).program.tokens, None
    except Exception as error:  # diagnostic result, never a retry
        return (), f"{type(error).__name__}: {error}"


def run_development_matrix(*, epochs: int = _FIT_EPOCHS) -> DevelopmentMatrixReport:
    """Fit once and score visible replay and composition as separate axes."""
    started = time.perf_counter()
    signature = _signature()
    teachers = visible_teachers()
    compositions = visible_compositions()
    teacher_token_sets = {case.tokens for case in teachers}
    if any(case.tokens in teacher_token_sets for case in compositions):
        raise AssertionError("a visible composition duplicated a teacher program")
    examples = tuple(
        tp.Example(case.prompt, signature, case.tokens) for case in teachers
    )
    decoder = direct.DirectProductionCellDecoder.fit(examples, epochs=epochs)

    replay_results: list[ReplayResult] = []
    for case in teachers:
        actual, failure = _decode(decoder, case, signature)
        replay_results.append(
            ReplayResult(
                name=case.name,
                expected_tokens=case.tokens,
                actual_tokens=actual,
                exact=failure is None and actual == case.tokens,
                failure=failure,
            )
        )

    composition_results: list[CompositionResult] = []
    for case in compositions:
        root_currents = _root_currents(decoder, case.prompt, signature)
        actual, failure = _decode(decoder, case, signature)
        position, expected_token, actual_token = _first_mismatch(case.tokens, actual)
        composition_results.append(
            CompositionResult(
                name=case.name,
                expected_tokens=case.tokens,
                actual_tokens=actual,
                exact=failure is None and position is None,
                first_mismatch=position,
                expected_at_mismatch=expected_token,
                actual_at_mismatch=actual_token,
                root_allowed_currents_pa=root_currents,
                failure=failure,
            )
        )
    return DevelopmentMatrixReport(
        epochs=epochs,
        elapsed_seconds=time.perf_counter() - started,
        local_updates=decoder.local_updates,
        structural_recruitments=decoder.structural_recruitments,
        teacher_exact=sum(result.exact for result in replay_results),
        teacher_total=len(replay_results),
        composition_exact=sum(result.exact for result in composition_results),
        composition_total=len(composition_results),
        teacher_replays=tuple(replay_results),
        compositions=tuple(composition_results),
    )


def main() -> int:
    print(json.dumps(asdict(run_development_matrix()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
