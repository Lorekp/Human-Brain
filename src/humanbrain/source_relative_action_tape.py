"""Occurrence-preserving neuronal tape for source-relative generic actions.

The learner observes opaque context events, opaque source tokens, an externally
supplied opaque cue inventory, and previously executed generic motor steps.  It never accepts
target spans, desired outputs, syntax trees, candidate edits, repaired sources,
or task labels.  Complete motor programs are stored as recurrent simulated-cell
chains whose context, state, operation, role, and successor choices are made by
fitted GLIF races against a tonic STOP cell.

At inference, each source-token occurrence receives a distinct cell even when
several occurrences contain identical bytes.  A learned REBIND role activates
every occurrence equal to that cue; those cells, not Python ordering, compete
for focus.  COPY, ADVANCE, EMIT, REBIND, and END are all learned motor-cell
choices.  Host code is limited to opaque token activation and applying the
winning generic operation.

The cue inventory is explicit micro-gate scaffolding.  This is a supervised
motor-trajectory mechanism, not autonomous focus discovery or program
induction.  Its narrow claim is replay, fresh composition, and consequence
revision of generic source-relative action chains.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "ActionTapeAblation",
    "ActionTapeConfig",
    "ActionTapeConsequenceUpdate",
    "ActionTapeDecision",
    "ActionTapeEligibility",
    "ActionTapePendingError",
    "ActionTapeRaceTrace",
    "ActionTapeReceipt",
    "ActionTapeTrainingSummary",
    "MotorStep",
    "OccurrencePreservingActionTape",
    "SourceRelativeMotorDemonstration",
    "TapeOperation",
    "TapeValence",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 32
_TONIC_STOP_SCORE = 1
_RELATION_SCORE = 2
_REINFORCEMENT_TICKS = 1
_DEPRESSION_TICKS = 2

_CHECKPOINT_NAMESPACE = b"humanbrain.source-relative-action-tape.checkpoint-v1\0"
_TRAINING_NAMESPACE = b"humanbrain.source-relative-action-tape.training-v1\0"
_QUERY_NAMESPACE = b"humanbrain.source-relative-action-tape.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.source-relative-action-tape.race-v1\0"
_EDGE_NAMESPACE = b"humanbrain.source-relative-action-tape.edge-v1\0"
_RECEIPT_NAMESPACE = b"humanbrain.source-relative-action-tape.receipt-v1\0"
_DECISION_NAMESPACE = b"humanbrain.source-relative-action-tape.decision-v1\0"
_UPDATE_NAMESPACE = b"humanbrain.source-relative-action-tape.update-v1\0"
_CONSTRUCTION_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _require_event(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    return value


def _require_events(value: object, name: str, *, unique: bool = False) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    events = value
    for index, event in enumerate(events):
        _require_event(event, f"{name}[{index}]")
    if unique and len(set(events)) != len(events):
        raise ValueError(f"{name} must contain unique opaque events")
    return events


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (voltage - threshold)
    currents: list[float] = []
    for score in range(_SCORE_CAP + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("action-tape fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(row.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)):
        raise RuntimeError("action-tape fitted-GLIF latency calibration changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class TapeOperation(StrEnum):
    """Generic motor cells; values carry no source or task ontology."""

    REBIND = "rebind"
    COPY = "copy"
    ADVANCE = "advance"
    EMIT = "emit"
    END = "end"


class ActionTapeAblation(StrEnum):
    NONE = "none"
    RELATIONAL_TAPE_OFF = "relational_tape_off"
    ASSEMBLY_OFF = "assembly_off"


class TapeValence(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class ActionTapePendingError(RuntimeError):
    """Raised when the exact-once decision/consequence protocol is violated."""


@dataclass(frozen=True, slots=True)
class MotorStep:
    operation: TapeOperation
    cue_ordinal: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, TapeOperation):
            raise TypeError("operation must be a TapeOperation")
        if self.operation is TapeOperation.REBIND:
            if type(self.cue_ordinal) is not int or self.cue_ordinal < 0:
                raise ValueError("REBIND requires a non-negative cue ordinal")
        elif self.cue_ordinal is not None:
            raise ValueError("only REBIND may carry a cue ordinal")

    def record(self) -> dict[str, object]:
        return {"cue_ordinal": self.cue_ordinal, "operation": self.operation.value}


@dataclass(frozen=True, slots=True)
class SourceRelativeMotorDemonstration:
    """One observed generic motor trajectory over opaque occurrence cells."""

    context_event: bytes
    cue_inventory: tuple[bytes, ...]
    source_tokens: tuple[bytes, ...]
    motor_steps: tuple[MotorStep, ...]

    def __post_init__(self) -> None:
        _require_event(self.context_event, "context_event")
        cues = _require_events(self.cue_inventory, "cue_inventory", unique=True)
        source = _require_events(self.source_tokens, "source_tokens")
        if not source:
            raise ValueError("source_tokens must not be empty")
        if type(self.motor_steps) is not tuple or not self.motor_steps:
            raise ValueError("motor_steps must be a non-empty tuple")
        if any(not isinstance(step, MotorStep) for step in self.motor_steps):
            raise TypeError("motor_steps must contain MotorStep values")
        if self.motor_steps[-1].operation is not TapeOperation.END:
            raise ValueError("motor trajectory must finish with END")
        if any(step.operation is TapeOperation.END for step in self.motor_steps[:-1]):
            raise ValueError("END may occur only at the trajectory boundary")

        head: int | None = None
        held = False
        for step in self.motor_steps:
            if step.operation is TapeOperation.REBIND:
                ordinal = step.cue_ordinal
                if ordinal is None or ordinal >= len(cues):
                    raise ValueError("REBIND cue ordinal exceeds the demonstration inventory")
                matches = tuple(
                    index for index, token in enumerate(source) if token == cues[ordinal]
                )
                if len(matches) != 1:
                    raise ValueError(
                        "demonstration REBIND cue must activate one unambiguous occurrence"
                    )
                head = matches[0]
                held = False
            elif step.operation is TapeOperation.COPY:
                if head is None:
                    raise ValueError("COPY requires a prior REBIND")
                held = True
            elif step.operation is TapeOperation.ADVANCE:
                if head is None or head + 1 >= len(source):
                    raise ValueError("ADVANCE leaves the demonstration tape")
                head += 1
                held = False
            elif step.operation is TapeOperation.EMIT and not held:
                raise ValueError("EMIT requires a preceding COPY at the current occurrence")


@dataclass(frozen=True, slots=True)
class ActionTapeConfig:
    max_demonstrations: int = 512
    max_observation_events: int = 128
    max_roles: int = 256
    max_tape_tokens: int = 4_096
    max_motor_steps: int = 4_096
    max_event_bytes: int = 4_096
    max_cells: int = 65_536

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ActionTapeRaceTrace:
    population: str
    candidate_cells: tuple[int, ...]
    candidate_scores: tuple[int, ...]
    winner_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    stopped: bool
    stop_reason: str | None
    stop_relay_spike_step: int | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ActionTapeEligibility:
    bank: str
    row: int
    column: int
    source_cell: int
    target_cell: int
    plastic: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class ActionTapeReceipt:
    checkpoint_sha256: str
    query_sha256: str
    output_sha256: str
    transitions: tuple[ActionTapeEligibility, ...]
    race_sha256s: tuple[str, ...]
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ActionTapeDecision:
    decision: str
    emissions: tuple[bytes, ...]
    operations: tuple[TapeOperation, ...]
    focus_occurrence_cells: tuple[int, ...]
    stop_reason: str | None
    eligibility: ActionTapeReceipt | None
    races: tuple[ActionTapeRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ActionTapeConsequenceUpdate:
    valence: TapeValence
    receipt_sha256: str
    changed_transitions: tuple[ActionTapeEligibility, ...]
    prior_training_sha256: str
    training_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ActionTapeTrainingSummary:
    demonstrations: int
    context_cells: int
    program_cells: int
    recurrent_state_cells: int
    learned_synapses: int
    excitatory_ticks: int
    occurrence_capacity_cells: int
    total_cells: int


@dataclass(frozen=True, slots=True)
class _SynapseBank:
    rows: tuple[tuple[int, ...], ...]
    supports: tuple[tuple[int, ...], ...]
    column_count: int

    @classmethod
    def grow(
        cls,
        row_count: int,
        column_count: int,
        ticks: Mapping[tuple[int, int], int],
    ) -> _SynapseBank:
        columns_by_row: list[list[int]] = [[] for _ in range(row_count)]
        supports_by_row: list[list[int]] = [[] for _ in range(row_count)]
        for (row, column), support in sorted(ticks.items()):
            if not 0 <= row < row_count or not 0 <= column < column_count:
                raise RuntimeError("synapse address left its population")
            if type(support) is not int or not 1 <= support <= _SCORE_CAP:
                raise RuntimeError("synapse support left calibrated tick bounds")
            columns_by_row[row].append(column)
            supports_by_row[row].append(support)
        return cls(
            rows=tuple(tuple(row) for row in columns_by_row),
            supports=tuple(tuple(row) for row in supports_by_row),
            column_count=column_count,
        )

    def outgoing(self, row: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not 0 <= row < len(self.rows):
            raise RuntimeError("synapse row left its population")
        return self.rows[row], self.supports[row]

    @property
    def synapse_count(self) -> int:
        return sum(len(row) for row in self.rows)

    @property
    def excitatory_ticks(self) -> int:
        return sum(sum(row) for row in self.supports)

    def record(self) -> dict[str, object]:
        return {
            "column_count": self.column_count,
            "rows": self.rows,
            "supports": self.supports,
        }


@dataclass(frozen=True, slots=True)
class _CellGeometry:
    context_offset: int
    program_offset: int
    state_offset: int
    operation_offset: int
    role_offset: int
    occurrence_offset: int
    stop_cell: int
    total_cells: int

    def record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: ActionTapeRaceTrace


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    receipt: ActionTapeReceipt


def _program_key(program: tuple[MotorStep, ...]) -> bytes:
    return _canonical_json(tuple(step.record() for step in program))


def _increment(ticks: dict[tuple[int, int], int], row: int, column: int) -> None:
    key = (row, column)
    ticks[key] = min(_SCORE_CAP, ticks.get(key, 0) + 1)


def _physical_race(
    *,
    population: str,
    columns: tuple[int, ...],
    scores: tuple[int, ...],
    candidate_cells: tuple[int, ...],
    stop_cell: int,
) -> _RaceDecision:
    if len(columns) != len(scores) or len(columns) != len(candidate_cells):
        raise RuntimeError("physical race candidate custody differs")
    if any(not 0 <= score <= _SCORE_CAP for score in scores):
        raise RuntimeError("physical race score left calibrated bounds")
    all_scores = (*scores, _TONIC_STOP_SCORE)
    all_cells = (*candidate_cells, stop_cell)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_cells) * _RACE_STEPS
    relay_step: int | None = None
    stopped = False
    stop_reason: str | None = None
    selected: int | None = None

    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_STOP_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("action-tape collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        relay_step = relay.spike_step
        winner_cell = stop_cell
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(columns):
        winner_cell = stop_cell
        stopped = True
        stop_reason = "tonic_stop"
    else:
        selected = columns[result.winner_cell]
        winner_cell = candidate_cells[result.winner_cell]

    record = {
        "candidate_cells": all_cells,
        "candidate_scores": all_scores,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_candidates": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stop_relay_spike_step": relay_step,
        "stopped": stopped,
        "winner_cell": winner_cell,
    }
    return _RaceDecision(
        column=selected,
        trace=ActionTapeRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record)),
    )


class OccurrencePreservingActionTape:
    """Learned recurrent motor controller over a fresh occurrence-cell tape."""

    __slots__ = (
        "_adjustments",
        "_banks",
        "_base_checkpoint_sha256",
        "_checkpoint_sha256",
        "_config",
        "_context_index",
        "_contexts",
        "_geometry",
        "_operations",
        "_pending",
        "_programs",
        "_states",
        "_summary",
    )

    def __init__(
        self,
        *,
        contexts: tuple[bytes, ...],
        programs: tuple[tuple[MotorStep, ...], ...],
        states: tuple[tuple[int, int], ...],
        geometry: _CellGeometry,
        banks: Mapping[str, _SynapseBank],
        config: ActionTapeConfig,
        summary: ActionTapeTrainingSummary,
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use OccurrencePreservingActionTape.grow")
        self._contexts = contexts
        self._programs = programs
        self._states = states
        self._geometry = geometry
        self._banks = MappingProxyType(dict(banks))
        self._config = config
        self._summary = summary
        self._base_checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_sha256 = checkpoint_sha256
        self._adjustments: dict[tuple[str, int, int], int] = {}
        self._pending: _PendingDecision | None = None
        self._context_index = MappingProxyType(
            {event: index for index, event in enumerate(contexts)}
        )
        self._operations = tuple(TapeOperation)

    @classmethod
    def grow(
        cls,
        demonstrations: Iterable[SourceRelativeMotorDemonstration],
        config: ActionTapeConfig | None = None,
    ) -> OccurrencePreservingActionTape:
        """Recruit sparse neuronal motor chains from observed trajectories."""

        rows = tuple(demonstrations)
        if not rows:
            raise ValueError("at least one motor demonstration is required")
        if any(not isinstance(row, SourceRelativeMotorDemonstration) for row in rows):
            raise TypeError("demonstrations must contain SourceRelativeMotorDemonstration")
        active_config = config or ActionTapeConfig()
        if not isinstance(active_config, ActionTapeConfig):
            raise TypeError("config must be an ActionTapeConfig")
        if len(rows) > active_config.max_demonstrations:
            raise ValueError("demonstration count exceeds configured capacity")
        for row in rows:
            if len(row.cue_inventory) > active_config.max_roles:
                raise ValueError("demonstration cue inventory exceeds capacity")
            if len(row.source_tokens) > active_config.max_tape_tokens:
                raise ValueError("demonstration source tape exceeds capacity")
            if len(row.motor_steps) > active_config.max_motor_steps:
                raise ValueError("demonstration motor trajectory exceeds capacity")
            for event in (row.context_event, *row.cue_inventory, *row.source_tokens):
                if len(event) > active_config.max_event_bytes:
                    raise ValueError("opaque event exceeds configured byte capacity")

        contexts = tuple(sorted({row.context_event for row in rows}))
        programs = tuple(sorted({row.motor_steps for row in rows}, key=_program_key))
        states = tuple(
            (program_id, position)
            for program_id, program in enumerate(programs)
            for position in range(len(program))
        )
        context_index = {event: index for index, event in enumerate(contexts)}
        program_index = {program: index for index, program in enumerate(programs)}
        state_index = {state: index for index, state in enumerate(states)}
        operation_index = {operation: index for index, operation in enumerate(TapeOperation)}

        context_offset = 0
        program_offset = context_offset + len(contexts)
        state_offset = program_offset + len(programs)
        operation_offset = state_offset + len(states)
        role_offset = operation_offset + len(TapeOperation)
        occurrence_offset = role_offset + active_config.max_roles
        stop_cell = occurrence_offset + active_config.max_tape_tokens
        geometry = _CellGeometry(
            context_offset=context_offset,
            program_offset=program_offset,
            state_offset=state_offset,
            operation_offset=operation_offset,
            role_offset=role_offset,
            occurrence_offset=occurrence_offset,
            stop_cell=stop_cell,
            total_cells=stop_cell + 1,
        )
        if geometry.total_cells > active_config.max_cells:
            raise ValueError("recruited action-tape cells exceed configured capacity")

        tick_maps: dict[str, dict[tuple[int, int], int]] = {
            "context_program": {},
            "program_root": {},
            "state_operation": {},
            "state_next": {},
            "state_role": {},
        }
        for row in rows:
            context_id = context_index[row.context_event]
            program_id = program_index[row.motor_steps]
            _increment(tick_maps["context_program"], context_id, program_id)
            _increment(tick_maps["program_root"], program_id, state_index[(program_id, 0)])
            for position, step in enumerate(row.motor_steps):
                state_id = state_index[(program_id, position)]
                _increment(
                    tick_maps["state_operation"],
                    state_id,
                    operation_index[step.operation],
                )
                if step.operation is TapeOperation.REBIND:
                    if step.cue_ordinal is None:
                        raise RuntimeError("validated REBIND lost its cue ordinal")
                    _increment(tick_maps["state_role"], state_id, step.cue_ordinal)
                if position + 1 < len(row.motor_steps):
                    _increment(
                        tick_maps["state_next"],
                        state_id,
                        state_index[(program_id, position + 1)],
                    )

        banks = {
            "context_program": _SynapseBank.grow(
                len(contexts), len(programs), tick_maps["context_program"]
            ),
            "program_root": _SynapseBank.grow(
                len(programs), len(states), tick_maps["program_root"]
            ),
            "state_operation": _SynapseBank.grow(
                len(states), len(TapeOperation), tick_maps["state_operation"]
            ),
            "state_next": _SynapseBank.grow(len(states), len(states), tick_maps["state_next"]),
            "state_role": _SynapseBank.grow(
                len(states), active_config.max_roles, tick_maps["state_role"]
            ),
        }
        checkpoint = _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "banks": tuple((name, banks[name].record()) for name in tick_maps),
                "config": asdict(active_config),
                "contexts_hex": tuple(event.hex() for event in contexts),
                "geometry": geometry.record(),
                "programs": tuple(tuple(step.record() for step in program) for program in programs),
                "states": states,
            },
        )
        summary = ActionTapeTrainingSummary(
            demonstrations=len(rows),
            context_cells=len(contexts),
            program_cells=len(programs),
            recurrent_state_cells=len(states),
            learned_synapses=sum(bank.synapse_count for bank in banks.values()),
            excitatory_ticks=sum(bank.excitatory_ticks for bank in banks.values()),
            occurrence_capacity_cells=active_config.max_tape_tokens,
            total_cells=geometry.total_cells,
        )
        return cls(
            contexts=contexts,
            programs=programs,
            states=states,
            geometry=geometry,
            banks=banks,
            config=active_config,
            summary=summary,
            checkpoint_sha256=checkpoint,
            construction_token=_CONSTRUCTION_TOKEN,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> ActionTapeTrainingSummary:
        return self._summary

    def respond(
        self,
        observation_events: tuple[bytes, ...],
        cue_inventory: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
        *,
        ablation: ActionTapeAblation = ActionTapeAblation.NONE,
    ) -> ActionTapeDecision:
        """Run learned motor-cell chains over a fresh occurrence-cell tape."""

        if self._pending is not None:
            raise ActionTapePendingError("resolve the pending EMIT decision before responding")
        observations = _require_events(observation_events, "observation_events")
        if not observations:
            raise ValueError("observation_events must not be empty")
        cues = _require_events(cue_inventory, "cue_inventory", unique=True)
        tape = _require_events(source_tokens, "source_tokens")
        if not tape:
            raise ValueError("source_tokens must not be empty")
        if len(observations) > self._config.max_observation_events:
            raise ValueError("observation stream exceeds configured capacity")
        if len(cues) > self._config.max_roles:
            raise ValueError("cue inventory exceeds configured capacity")
        if len(tape) > self._config.max_tape_tokens:
            raise ValueError("source tape exceeds configured capacity")
        for event in (*observations, *cues, *tape):
            if len(event) > self._config.max_event_bytes:
                raise ValueError("opaque event exceeds configured byte capacity")
        if not isinstance(ablation, ActionTapeAblation):
            raise TypeError("ablation must be an ActionTapeAblation")

        query_record = {
            "ablation": ablation.value,
            "checkpoint_sha256": self._checkpoint_sha256,
            "cue_inventory_hex": tuple(event.hex() for event in cues),
            "observation_events_hex": tuple(event.hex() for event in observations),
            "source_tokens_hex": tuple(event.hex() for event in tape),
        }
        query_sha256 = _digest(_QUERY_NAMESPACE, query_record)
        races: list[ActionTapeRaceTrace] = []
        transitions: list[ActionTapeEligibility] = []
        operations: list[TapeOperation] = []
        focus_cells: list[int] = []
        emitted: list[bytes] = []

        active_contexts = tuple(
            dict.fromkeys(
                context_id
                for event in observations
                if (context_id := self._context_index.get(event)) is not None
            )
        )
        if not active_contexts:
            decision = _physical_race(
                population="context_program",
                columns=(),
                scores=(),
                candidate_cells=(),
                stop_cell=self._geometry.stop_cell,
            )
            races.append(decision.trace)
            return self._stop(query_sha256, races, f"context_program:{decision.trace.stop_reason}")

        program_decision, context_edges = self._select_active_program(active_contexts)
        races.append(program_decision.trace)
        if program_decision.column is None:
            return self._stop(
                query_sha256,
                races,
                f"context_program:{program_decision.trace.stop_reason}",
            )
        program_id = program_decision.column
        transitions.extend(context_edges)

        root_decision = self._select(
            "program_root",
            program_id,
            self._geometry.state_offset,
            ablated=ablation is ActionTapeAblation.ASSEMBLY_OFF,
        )
        races.append(root_decision.trace)
        if root_decision.column is None:
            return self._stop(
                query_sha256,
                races,
                f"program_root:{root_decision.trace.stop_reason}",
            )
        state_id = root_decision.column
        transitions.append(self._eligibility("program_root", program_id, state_id))
        head: int | None = None
        held: bytes | None = None

        for _ in range(self._config.max_motor_steps):
            operation_decision = self._select(
                "state_operation",
                state_id,
                self._geometry.operation_offset,
            )
            races.append(operation_decision.trace)
            if operation_decision.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"state_operation:{operation_decision.trace.stop_reason}",
                )
            operation_id = operation_decision.column
            operation = self._operations[operation_id]
            operations.append(operation)
            transitions.append(self._eligibility("state_operation", state_id, operation_id))

            if operation is TapeOperation.REBIND:
                role_decision = self._select(
                    "state_role",
                    state_id,
                    self._geometry.role_offset,
                    allowed_columns=len(cues),
                )
                races.append(role_decision.trace)
                if role_decision.column is None:
                    return self._stop(
                        query_sha256,
                        races,
                        f"state_role:{role_decision.trace.stop_reason}",
                    )
                cue_ordinal = role_decision.column
                transitions.append(self._eligibility("state_role", state_id, cue_ordinal))
                focus_decision = self._focus(
                    cues[cue_ordinal],
                    tape,
                    ablated=ablation is ActionTapeAblation.RELATIONAL_TAPE_OFF,
                )
                races.append(focus_decision.trace)
                if focus_decision.column is None:
                    return self._stop(
                        query_sha256,
                        races,
                        f"tape_focus:{focus_decision.trace.stop_reason}",
                    )
                head = focus_decision.column
                held = None
                focus_cells.append(self._geometry.occurrence_offset + head)
                transitions.append(
                    self._runtime_eligibility(
                        "tape_focus",
                        cue_ordinal,
                        head,
                        self._geometry.role_offset + cue_ordinal,
                        self._geometry.occurrence_offset + head,
                    )
                )
            elif operation is TapeOperation.COPY:
                if head is None:
                    raise RuntimeError("learned COPY lost its occurrence focus")
                held = tape[head]
            elif operation is TapeOperation.ADVANCE:
                if head is None:
                    raise RuntimeError("learned ADVANCE lost its occurrence focus")
                advance_decision = self._advance(
                    head,
                    len(tape),
                    ablated=ablation is ActionTapeAblation.RELATIONAL_TAPE_OFF,
                )
                races.append(advance_decision.trace)
                if advance_decision.column is None:
                    return self._stop(
                        query_sha256,
                        races,
                        f"tape_advance:{advance_decision.trace.stop_reason}",
                    )
                next_head = advance_decision.column
                transitions.append(
                    self._runtime_eligibility(
                        "tape_advance",
                        head,
                        next_head,
                        self._geometry.occurrence_offset + head,
                        self._geometry.occurrence_offset + next_head,
                    )
                )
                head = next_head
                held = None
            elif operation is TapeOperation.EMIT:
                if held is None:
                    raise RuntimeError("learned EMIT lost its copied occurrence activation")
                emitted.append(held)
            elif operation is TapeOperation.END:
                break

            next_decision = self._select(
                "state_next",
                state_id,
                self._geometry.state_offset,
            )
            races.append(next_decision.trace)
            if next_decision.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"state_next:{next_decision.trace.stop_reason}",
                )
            next_state = next_decision.column
            transitions.append(self._eligibility("state_next", state_id, next_state))
            state_id = next_state
        else:
            return self._stop(query_sha256, races, "motor_traversal:bounded_stop")

        output = tuple(emitted)
        output_sha256 = _digest(
            _DECISION_NAMESPACE,
            {"emissions_hex": tuple(event.hex() for event in output)},
        )
        physical_cell_steps = sum(race.physical_cell_steps for race in races)
        receipt_record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "output_sha256": output_sha256,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in races),
            "transition_sha256s": tuple(edge.sha256 for edge in transitions),
        }
        receipt = ActionTapeReceipt(
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            output_sha256=output_sha256,
            transitions=tuple(transitions),
            race_sha256s=tuple(race.sha256 for race in races),
            physical_cell_steps=physical_cell_steps,
            sha256=_digest(_RECEIPT_NAMESPACE, receipt_record),
        )
        decision_record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": "EMIT",
            "eligibility_sha256": receipt.sha256,
            "emissions_hex": tuple(event.hex() for event in output),
            "focus_occurrence_cells": tuple(focus_cells),
            "operations": tuple(operation.value for operation in operations),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": receipt.race_sha256s,
            "stop_reason": None,
        }
        result = ActionTapeDecision(
            decision="EMIT",
            emissions=output,
            operations=tuple(operations),
            focus_occurrence_cells=tuple(focus_cells),
            stop_reason=None,
            eligibility=receipt,
            races=tuple(races),
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, decision_record),
        )
        self._pending = _PendingDecision(receipt)
        return result

    def resolve(self, valence: TapeValence) -> ActionTapeConsequenceUpdate:
        """Apply one scalar consequence to only the exact selected learned chain."""

        if not isinstance(valence, TapeValence):
            raise TypeError("valence must be a TapeValence")
        pending = self._pending
        if pending is None:
            raise ActionTapePendingError("no pending EMIT decision to resolve")
        self._pending = None
        prior = self._checkpoint_sha256
        delta = 0
        if valence is TapeValence.POSITIVE:
            delta = _REINFORCEMENT_TICKS
        elif valence is TapeValence.NEGATIVE:
            delta = -_DEPRESSION_TICKS

        changed: list[ActionTapeEligibility] = []
        if delta:
            for edge in pending.receipt.transitions:
                if not edge.plastic:
                    continue
                bank = self._banks[edge.bank]
                columns, supports = bank.outgoing(edge.row)
                try:
                    offset = columns.index(edge.column)
                except ValueError as error:
                    raise RuntimeError("eligible synapse disappeared before consequence") from error
                key = (edge.bank, edge.row, edge.column)
                old_adjustment = self._adjustments.get(key, 0)
                base = supports[offset]
                old_score = max(0, min(_SCORE_CAP, base + old_adjustment))
                new_score = max(0, min(_SCORE_CAP, old_score + delta))
                if new_score != old_score:
                    adjustment = new_score - base
                    if adjustment:
                        self._adjustments[key] = adjustment
                    else:
                        self._adjustments.pop(key, None)
                    changed.append(edge)
            if changed:
                self._checkpoint_sha256 = self._training_sha256()

        record = {
            "changed_transition_sha256s": tuple(edge.sha256 for edge in changed),
            "prior_training_sha256": prior,
            "receipt_sha256": pending.receipt.sha256,
            "training_sha256": self._checkpoint_sha256,
            "valence": valence.value,
        }
        return ActionTapeConsequenceUpdate(
            valence=valence,
            receipt_sha256=pending.receipt.sha256,
            changed_transitions=tuple(changed),
            prior_training_sha256=prior,
            training_sha256=self._checkpoint_sha256,
            sha256=_digest(_UPDATE_NAMESPACE, record),
        )

    def _effective_scores(
        self,
        bank_name: str,
        row: int,
        columns: tuple[int, ...],
        supports: tuple[int, ...],
        *,
        ablated: bool,
        allowed_columns: int | None,
    ) -> tuple[int, ...]:
        if ablated:
            return tuple(0 for _ in supports)
        return tuple(
            0
            if allowed_columns is not None and column >= allowed_columns
            else max(
                0,
                min(
                    _SCORE_CAP,
                    support + self._adjustments.get((bank_name, row, column), 0),
                ),
            )
            for column, support in zip(columns, supports, strict=True)
        )

    def _select(
        self,
        bank_name: str,
        row: int,
        target_offset: int,
        *,
        ablated: bool = False,
        allowed_columns: int | None = None,
    ) -> _RaceDecision:
        columns, supports = self._banks[bank_name].outgoing(row)
        scores = self._effective_scores(
            bank_name,
            row,
            columns,
            supports,
            ablated=ablated,
            allowed_columns=allowed_columns,
        )
        return _physical_race(
            population=bank_name,
            columns=columns,
            scores=scores,
            candidate_cells=tuple(target_offset + column for column in columns),
            stop_cell=self._geometry.stop_cell,
        )

    def _select_active_program(
        self,
        active_contexts: tuple[int, ...],
    ) -> tuple[_RaceDecision, tuple[ActionTapeEligibility, ...]]:
        """Integrate all active context synapses in one neuronal program race."""

        summed_scores: dict[int, int] = {}
        contributing: dict[int, list[int]] = {}
        for context_id in active_contexts:
            columns, supports = self._banks["context_program"].outgoing(context_id)
            scores = self._effective_scores(
                "context_program",
                context_id,
                columns,
                supports,
                ablated=False,
                allowed_columns=None,
            )
            for program_id, score in zip(columns, scores, strict=True):
                summed_scores[program_id] = min(
                    _SCORE_CAP,
                    summed_scores.get(program_id, 0) + score,
                )
                if score > 0:
                    contributing.setdefault(program_id, []).append(context_id)

        programs = tuple(sorted(summed_scores))
        decision = _physical_race(
            population="context_program",
            columns=programs,
            scores=tuple(summed_scores[program_id] for program_id in programs),
            candidate_cells=tuple(
                self._geometry.program_offset + program_id for program_id in programs
            ),
            stop_cell=self._geometry.stop_cell,
        )
        if decision.column is None:
            return decision, ()
        edges = tuple(
            self._eligibility("context_program", context_id, decision.column)
            for context_id in contributing.get(decision.column, ())
        )
        if not edges:
            raise RuntimeError("winning program lost its active context synapse")
        return decision, edges

    def _focus(
        self,
        cue: bytes,
        tape: tuple[bytes, ...],
        *,
        ablated: bool,
    ) -> _RaceDecision:
        occurrences = tuple(index for index, token in enumerate(tape) if token == cue)
        scores = tuple(0 if ablated else _RELATION_SCORE for _ in occurrences)
        return _physical_race(
            population="tape_focus",
            columns=occurrences,
            scores=scores,
            candidate_cells=tuple(
                self._geometry.occurrence_offset + index for index in occurrences
            ),
            stop_cell=self._geometry.stop_cell,
        )

    def _advance(self, head: int, tape_size: int, *, ablated: bool) -> _RaceDecision:
        columns = (head + 1,) if head + 1 < tape_size else ()
        scores = tuple(0 if ablated else _RELATION_SCORE for _ in columns)
        return _physical_race(
            population="tape_advance",
            columns=columns,
            scores=scores,
            candidate_cells=tuple(self._geometry.occurrence_offset + index for index in columns),
            stop_cell=self._geometry.stop_cell,
        )

    def _eligibility(self, bank_name: str, row: int, column: int) -> ActionTapeEligibility:
        if bank_name == "context_program":
            source = self._geometry.context_offset + row
            target = self._geometry.program_offset + column
        elif bank_name == "program_root":
            source = self._geometry.program_offset + row
            target = self._geometry.state_offset + column
        elif bank_name == "state_operation":
            source = self._geometry.state_offset + row
            target = self._geometry.operation_offset + column
        elif bank_name == "state_next":
            source = self._geometry.state_offset + row
            target = self._geometry.state_offset + column
        elif bank_name == "state_role":
            source = self._geometry.state_offset + row
            target = self._geometry.role_offset + column
        else:
            raise RuntimeError("unknown learned action-tape synapse bank")
        return self._make_eligibility(bank_name, row, column, source, target, plastic=True)

    def _runtime_eligibility(
        self,
        bank_name: str,
        row: int,
        column: int,
        source: int,
        target: int,
    ) -> ActionTapeEligibility:
        return self._make_eligibility(bank_name, row, column, source, target, plastic=False)

    def _make_eligibility(
        self,
        bank_name: str,
        row: int,
        column: int,
        source: int,
        target: int,
        *,
        plastic: bool,
    ) -> ActionTapeEligibility:
        record = {
            "bank": bank_name,
            "checkpoint_sha256": self._checkpoint_sha256,
            "column": column,
            "plastic": plastic,
            "row": row,
            "source_cell": source,
            "target_cell": target,
        }
        return ActionTapeEligibility(
            bank=bank_name,
            row=row,
            column=column,
            source_cell=source,
            target_cell=target,
            plastic=plastic,
            sha256=_digest(_EDGE_NAMESPACE, record),
        )

    def _training_sha256(self) -> str:
        return _digest(
            _TRAINING_NAMESPACE,
            {
                "adjustments": tuple(
                    (*key, adjustment) for key, adjustment in sorted(self._adjustments.items())
                ),
                "base_checkpoint_sha256": self._base_checkpoint_sha256,
            },
        )

    def _stop(
        self,
        query_sha256: str,
        races: list[ActionTapeRaceTrace],
        reason: str,
    ) -> ActionTapeDecision:
        physical_cell_steps = sum(race.physical_cell_steps for race in races)
        record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": "STOP",
            "eligibility_sha256": None,
            "emissions_hex": (),
            "focus_occurrence_cells": (),
            "operations": (),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in races),
            "stop_reason": reason,
        }
        return ActionTapeDecision(
            decision="STOP",
            emissions=(),
            operations=(),
            focus_occurrence_cells=(),
            stop_reason=reason,
            eligibility=None,
            races=tuple(races),
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
