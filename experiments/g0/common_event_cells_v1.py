"""Domain-neutral simulated-cell projection for opaque G0 event bytes.

The module has one narrow interface, :class:`CommonEventCells`.  It projects a
bounded byte frame without decoding it, then exposes the same factorized output
population and fitted-GLIF STOP race to every future G0 learner.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "G0_COMMON_EVENT_CELLS_V1"
STATUS = "COMMON_PROJECTION_ONLY_NO_LEARNER_OR_G0_RESULT"

MAX_EVENT_BYTES = 256
ASSEMBLY_ACTIVE_CELLS = 12
FEATURE_BANK_CELLS = 4_096
TRACE_LANES = 12

VALUE_CELL_OFFSET = 0
POSITION_CELL_OFFSET = VALUE_CELL_OFFSET + FEATURE_BANK_CELLS
BINDING_CELL_OFFSET = POSITION_CELL_OFFSET + FEATURE_BANK_CELLS
BOUNDARY_CELL_OFFSET = BINDING_CELL_OFFSET + FEATURE_BANK_CELLS
TRACE_CELL_OFFSET = BOUNDARY_CELL_OFFSET + FEATURE_BANK_CELLS
INPUT_CELL_COUNT = TRACE_CELL_OFFSET + TRACE_LANES * FEATURE_BANK_CELLS

OUTPUT_DIGIT_LANES = 4
OUTPUT_DIGIT_RADIX = 4
OUTPUT_TOKEN_COUNT = OUTPUT_DIGIT_RADIX**OUTPUT_DIGIT_LANES
OUTPUT_CELL_OFFSET = INPUT_CELL_COUNT
OUTPUT_DIGIT_CELL_COUNT = OUTPUT_DIGIT_LANES * OUTPUT_DIGIT_RADIX
STOP_CELL_ID = OUTPUT_CELL_OFFSET + OUTPUT_DIGIT_CELL_COUNT
OUTPUT_CELL_COUNT = OUTPUT_DIGIT_CELL_COUNT + 1
TOTAL_CELL_COUNT = INPUT_CELL_COUNT + OUTPUT_CELL_COUNT

RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
STOP_SCORE = 12
MIN_OUTPUT_SCORE = -48
MAX_OUTPUT_SCORE = 48

_PROJECTION_NAMESPACE = b"humanbrain.g0.common-event-cells-v1.projection\0"
_FRAME_NAMESPACE = b"humanbrain.g0.common-event-cells-v1.frame\0"
_OUTPUT_NAMESPACE = b"humanbrain.g0.common-event-cells-v1.output\0"
_GEOMETRY_NAMESPACE = b"humanbrain.g0.common-event-cells-v1.geometry\0"

_TRACE_MULTIPLIERS = (257, 263, 269, 271, 277, 281, 283, 293, 299, 301, 303, 307)
_TRACE_POSITION_GAINS = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_TRACE_BIASES = (17, 43, 71, 101, 137, 173, 211, 251, 293, 337, 383, 431)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _assembly(kind: bytes, payload: bytes, *, offset: int) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < ASSEMBLY_ACTIVE_CELLS:
        block = sha256(
            _PROJECTION_NAMESPACE
            + kind
            + b"\0"
            + len(payload).to_bytes(2, "big")
            + payload
            + counter.to_bytes(4, "big")
        ).digest()
        for cursor in range(0, len(block), 2):
            selected.add(
                offset
                + int.from_bytes(block[cursor : cursor + 2], "big")
                % FEATURE_BANK_CELLS
            )
            if len(selected) == ASSEMBLY_ACTIVE_CELLS:
                break
        counter += 1
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class ByteCellEmission:
    """One ordered byte event represented only by distributed physical cells."""

    value_cells: tuple[int, ...]
    position_cells: tuple[int, ...]
    binding_cells: tuple[int, ...]

    @property
    def active_cells(self) -> tuple[int, ...]:
        return (*self.value_cells, *self.position_cells, *self.binding_cells)

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectionWork:
    byte_events: int
    fixed_projection_cell_emissions: int
    recurrent_cell_transitions: int

    @property
    def total(self) -> int:
        return (
            self.byte_events
            + self.fixed_projection_cell_emissions
            + self.recurrent_cell_transitions
        )

    def record(self) -> dict[str, int]:
        return {
            **dataclasses.asdict(self),
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class CommonEventFrame:
    """Cell-only ordered observation returned by :meth:`CommonEventCells.project`."""

    byte_count: int
    start_cells: tuple[int, ...]
    byte_emissions: tuple[ByteCellEmission, ...]
    end_cells: tuple[int, ...]
    trace_cells: tuple[int, ...]
    work: ProjectionWork
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "byte_count": self.byte_count,
            "start_cells": self.start_cells,
            "byte_emissions": tuple(row.record() for row in self.byte_emissions),
            "end_cells": self.end_cells,
            "trace_cells": self.trace_cells,
            "work": self.work.record(),
            "raw_bytes_retained": False,
            "raw_integer_byte_handle_retained": False,
            "one_cell_byte_or_domain_identity_neural_input": False,
            "byte_count_is_serialization_and_boundary_metadata": True,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CommonOutputDecision:
    """One factorized byte-token emission or the physical STOP authority."""

    decision: str
    token: int | None
    active_cells: tuple[int, ...]
    lane_spike_steps: tuple[int, ...]
    lane_simultaneous_spike_candidates: tuple[int, ...]
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _build_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    unit_voltage, unit_voltage_threshold = population.compile_resting_response(RACE_STEPS)
    spike_gain = unit_voltage - unit_voltage_threshold
    crossing_current = population.parameters.threshold_inf_relative_mv / spike_gain
    currents: list[float] = []
    for score in range(MAX_OUTPUT_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("common output fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("common output fitted-GLIF latency calibration changed")
    return table, race


_CURRENT_TABLE, _OUTPUT_RACE = _build_current_table()


GEOMETRY_RECORD = {
    "scope": SCOPE,
    "maximum_event_bytes": MAX_EVENT_BYTES,
    "input": {
        "feature_banks": ("value", "position", "position-value-binding", "boundary"),
        "feature_bank_cells": FEATURE_BANK_CELLS,
        "assembly_active_cells": ASSEMBLY_ACTIVE_CELLS,
        "trace_lanes": TRACE_LANES,
        "trace_cells_per_lane": FEATURE_BANK_CELLS,
        "input_cell_count": INPUT_CELL_COUNT,
        "maximum_simultaneously_active_cells": 3 * ASSEMBLY_ACTIVE_CELLS,
    },
    "output": {
        "encoding": "four-sequential-base-four-fitted-glif-winners",
        "digit_lanes": OUTPUT_DIGIT_LANES,
        "digit_radix": OUTPUT_DIGIT_RADIX,
        "token_count": OUTPUT_TOKEN_COUNT,
        "token_active_cells": OUTPUT_DIGIT_LANES,
        "tonic_stop_cell_count": 1,
        "output_cell_count": OUTPUT_CELL_COUNT,
        "stop_score": STOP_SCORE,
        "race_steps": RACE_STEPS,
    },
    "total_cell_count": TOTAL_CELL_COUNT,
    "domain_decoder_count": 0,
    "task_or_family_identity_cell_count": 0,
    "one_cell_per_input_or_output_token": False,
}
GEOMETRY_SHA256 = _digest(_GEOMETRY_NAMESPACE, GEOMETRY_RECORD)


class CommonEventCells:
    """Project opaque bytes and race one shared factorized output population."""

    __slots__ = ()

    def project(self, event_bytes: bytes) -> CommonEventFrame:
        if type(event_bytes) is not bytes or not 1 <= len(event_bytes) <= MAX_EVENT_BYTES:
            raise ValueError("common event must be one nonempty bounded bytes value")

        start = _assembly(b"boundary-start", b"", offset=BOUNDARY_CELL_OFFSET)
        end = _assembly(
            b"boundary-end",
            len(event_bytes).to_bytes(2, "big"),
            offset=BOUNDARY_CELL_OFFSET,
        )
        state = tuple(
            (lane * 37 + 11) % FEATURE_BANK_CELLS for lane in range(TRACE_LANES)
        )
        emissions: list[ByteCellEmission] = []
        for position, value in enumerate(event_bytes):
            position_bytes = position.to_bytes(2, "big")
            value_bytes = bytes((value,))
            emissions.append(
                ByteCellEmission(
                    value_cells=_assembly(
                        b"value", value_bytes, offset=VALUE_CELL_OFFSET
                    ),
                    position_cells=_assembly(
                        b"position", position_bytes, offset=POSITION_CELL_OFFSET
                    ),
                    binding_cells=_assembly(
                        b"position-value-binding",
                        position_bytes + value_bytes,
                        offset=BINDING_CELL_OFFSET,
                    ),
                )
            )
            state = tuple(
                (
                    state[lane] * _TRACE_MULTIPLIERS[lane]
                    + value
                    + (position + 1) * _TRACE_POSITION_GAINS[lane]
                    + _TRACE_BIASES[lane]
                )
                % FEATURE_BANK_CELLS
                for lane in range(TRACE_LANES)
            )
        trace = tuple(
            TRACE_CELL_OFFSET + lane * FEATURE_BANK_CELLS + state[lane]
            for lane in range(TRACE_LANES)
        )
        work = ProjectionWork(
            byte_events=len(event_bytes),
            fixed_projection_cell_emissions=(3 * len(event_bytes) + 2) * ASSEMBLY_ACTIVE_CELLS,
            recurrent_cell_transitions=len(event_bytes) * TRACE_LANES,
        )
        record = {
            "byte_count": len(event_bytes),
            "start_cells": start,
            "byte_emissions": tuple(row.record() for row in emissions),
            "end_cells": end,
            "trace_cells": trace,
            "work": work.record(),
        }
        return CommonEventFrame(
            byte_count=len(event_bytes),
            start_cells=start,
            byte_emissions=tuple(emissions),
            end_cells=end,
            trace_cells=trace,
            work=work,
            sha256=_digest(_FRAME_NAMESPACE, record),
        )

    def race_output(
        self, raw_scores: tuple[tuple[int, ...], ...]
    ) -> CommonOutputDecision:
        if (
            type(raw_scores) is not tuple
            or len(raw_scores) != OUTPUT_DIGIT_LANES
            or any(type(row) is not tuple or len(row) != OUTPUT_DIGIT_RADIX for row in raw_scores)
            or any(
                type(score) is not int
                or not MIN_OUTPUT_SCORE <= score <= MAX_OUTPUT_SCORE
                for row in raw_scores
                for score in row
            )
        ):
            raise ValueError("common output scores must be an exact bounded 4x4 integer tuple")

        current_scores = tuple(
            (*tuple(max(score, 0) for score in row), STOP_SCORE)
            for row in raw_scores
        )
        currents = np.asarray(
            tuple(tuple(_CURRENT_TABLE[score] for score in row) for row in current_scores),
            dtype=np.float64,
        )
        races = _OUTPUT_RACE.race_many(currents, require_unique=False)
        collisions = tuple(row.simultaneous_spike_candidates != 1 for row in races)
        tonic_stops = tuple(
            not collision and row.winner_cell == OUTPUT_DIGIT_RADIX
            for collision, row in zip(collisions, races, strict=True)
        )
        physical_cell_steps = OUTPUT_DIGIT_LANES * (OUTPUT_DIGIT_RADIX + 1) * RACE_STEPS

        if any(collisions) or any(tonic_stops):
            stop_reason = "COLLISION" if any(collisions) else "TONIC_STOP"
            if any(collisions):
                stop_race = _OUTPUT_RACE.race_many(
                    np.asarray([[_CURRENT_TABLE[STOP_SCORE]]], dtype=np.float64)
                )[0]
                if stop_race.simultaneous_spike_candidates != 1:
                    raise RuntimeError("common collision STOP relay changed")
                physical_cell_steps += RACE_STEPS
            record = {
                "decision": "STOP",
                "token": None,
                "active_cells": (STOP_CELL_ID,),
                "lane_spike_steps": tuple(row.spike_step for row in races),
                "lane_simultaneous_spike_candidates": tuple(
                    row.simultaneous_spike_candidates for row in races
                ),
                "stop_reason": stop_reason,
                "physical_cell_steps": physical_cell_steps,
            }
            return CommonOutputDecision(
                **record,
                sha256=_digest(_OUTPUT_NAMESPACE, record),
            )

        digits = tuple(row.winner_cell for row in races)
        token = 0
        for digit in digits:
            token = token * OUTPUT_DIGIT_RADIX + digit
        active = tuple(
            OUTPUT_CELL_OFFSET + lane * OUTPUT_DIGIT_RADIX + digit
            for lane, digit in enumerate(digits)
        )
        record = {
            "decision": "TOKEN",
            "token": token,
            "active_cells": active,
            "lane_spike_steps": tuple(row.spike_step for row in races),
            "lane_simultaneous_spike_candidates": tuple(
                row.simultaneous_spike_candidates for row in races
            ),
            "stop_reason": None,
            "physical_cell_steps": physical_cell_steps,
        }
        return CommonOutputDecision(
            **record,
            sha256=_digest(_OUTPUT_NAMESPACE, record),
        )


__all__ = [
    "ASSEMBLY_ACTIVE_CELLS",
    "GEOMETRY_RECORD",
    "GEOMETRY_SHA256",
    "INPUT_CELL_COUNT",
    "MAX_EVENT_BYTES",
    "OUTPUT_CELL_COUNT",
    "OUTPUT_CELL_OFFSET",
    "OUTPUT_DIGIT_LANES",
    "OUTPUT_DIGIT_RADIX",
    "OUTPUT_TOKEN_COUNT",
    "SCOPE",
    "STATUS",
    "STOP_CELL_ID",
    "STOP_SCORE",
    "TOTAL_CELL_COUNT",
    "CommonEventCells",
    "CommonEventFrame",
    "CommonOutputDecision",
]
