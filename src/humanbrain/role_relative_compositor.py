"""Generic role-relative sequence composition through simulated-cell races.

The module learns opaque context-event to sequence-chunk associations from
complete demonstrations.  Output events that exactly equal a member of the
demonstration's opaque role inventory are stored as ordinal role traversals;
all other events remain opaque literals.  Recall binds those ordinals to the
current inventory and can concatenate chunks owned by different
demonstrations.

Python dictionaries and hashes provide only stable cell addressing.  Chunk,
root, emission, CONTINUE/EXIT, recurrent-state, and role-binding selections are
made by fitted GLIF cell races with a tonic STOP competitor.  Successful
composition returns the exact selected synapse chain as an eligibility receipt
that a later continual learner can inhibit after a negative consequence.

There is no language, Python, AST, task, expected-output, or candidate-patch
ontology in this module.
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
    "CellRaceTrace",
    "CompositionAblation",
    "ConsequenceUpdate",
    "EligibilityTransition",
    "PendingConsequenceError",
    "RoleRelativeComposition",
    "RoleRelativeCompositorConfig",
    "RoleRelativeSequenceCompositor",
    "RoleRelativeTrainingSummary",
    "RoleSequenceDemonstration",
    "TransitionChainReceipt",
    "Valence",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 32
_TONIC_STOP_SCORE = 1
_NOGO_TICKS = 2

_CHECKPOINT_NAMESPACE = b"humanbrain.role-relative-compositor.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.role-relative-compositor.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.role-relative-compositor.race-v1\0"
_EDGE_NAMESPACE = b"humanbrain.role-relative-compositor.edge-v1\0"
_RECEIPT_NAMESPACE = b"humanbrain.role-relative-compositor.receipt-v1\0"
_DECISION_NAMESPACE = b"humanbrain.role-relative-compositor.decision-v1\0"
_TRAINING_NAMESPACE = b"humanbrain.role-relative-compositor.training-v1\0"
_CONSEQUENCE_NAMESPACE = b"humanbrain.role-relative-compositor.consequence-v1\0"
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


def _require_inventory(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    inventory = value
    for index, event in enumerate(inventory):
        _require_event(event, f"{name}[{index}]")
    if len(set(inventory)) != len(inventory):
        raise ValueError(f"{name} must contain unique opaque events")
    return inventory


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    unit_voltage, unit_voltage_threshold = population.compile_resting_response(_RACE_STEPS)
    spike_gain = unit_voltage - unit_voltage_threshold
    crossing_current = population.parameters.threshold_inf_relative_mv / spike_gain
    currents: list[float] = []
    for score in range(_SCORE_CAP + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("role-relative fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(row.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)):
        raise RuntimeError("role-relative fitted-GLIF latency calibration changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class CompositionAblation(StrEnum):
    NONE = "none"
    COMPOSITOR_OFF = "compositor_off"
    RECURRENCE_OFF = "recurrence_off"
    NOGO_OFF = "nogo_off"


class Valence(StrEnum):
    """Opaque consequence polarity supplied after one emitted decision."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class PendingConsequenceError(RuntimeError):
    """Raised when the exact-once respond/resolve protocol is violated."""


@dataclass(frozen=True, slots=True)
class RoleSequenceDemonstration:
    """One complete opaque context-to-emission demonstration."""

    context_event: bytes
    role_inventory: tuple[bytes, ...]
    emissions: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_event(self.context_event, "context_event")
        _require_inventory(self.role_inventory, "role_inventory")
        if type(self.emissions) is not tuple or not self.emissions:
            raise ValueError("emissions must be a non-empty tuple")
        for index, event in enumerate(self.emissions):
            _require_event(event, f"emissions[{index}]")


@dataclass(frozen=True, slots=True)
class RoleRelativeCompositorConfig:
    """Capacity bounds; electrical race parameters remain architecture-fixed."""

    max_demonstrations: int = 512
    max_observation_events: int = 128
    max_roles: int = 256
    max_chunk_emissions: int = 1_024
    max_event_bytes: int = 4_096
    max_cells: int = 65_536

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CellRaceTrace:
    """One observable fitted-cell selection, including physical STOP."""

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
class EligibilityTransition:
    """Stable selected synapse address for later consequence inhibition."""

    bank: str
    row: int
    column: int
    source_cell: int
    target_cell: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TransitionChainReceipt:
    """Checkpoint-bound eligibility for exactly one completed composition."""

    checkpoint_sha256: str
    query_sha256: str
    output_sha256: str
    transitions: tuple[EligibilityTransition, ...]
    race_sha256s: tuple[str, ...]
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RoleRelativeComposition:
    """One fail-closed composition decision at the module interface."""

    decision: str
    emissions: tuple[bytes, ...]
    stop_reason: str | None
    eligibility: TransitionChainReceipt | None
    races: tuple[CellRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ConsequenceUpdate:
    """Auditable local learning caused by resolving one pending decision."""

    valence: Valence
    receipt_sha256: str
    changed_transitions: tuple[EligibilityTransition, ...]
    prior_training_sha256: str
    training_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RoleRelativeTrainingSummary:
    demonstrations: int
    context_cells: int
    chunk_cells: int
    recurrent_state_cells: int
    emission_cells: int
    recurring_chunks: int
    learned_synapses: int
    excitatory_ticks: int
    total_cells: int


@dataclass(frozen=True, slots=True)
class _Atom:
    kind: str
    literal: bytes | None = None
    ordinal: int | None = None

    def record(self) -> dict[str, object]:
        if self.kind == "literal":
            if self.literal is None or self.ordinal is not None:
                raise RuntimeError("literal atom custody changed")
            return {"kind": self.kind, "literal_hex": self.literal.hex()}
        if self.kind == "role":
            if self.literal is not None or self.ordinal is None:
                raise RuntimeError("role atom custody changed")
            return {"kind": self.kind, "ordinal": self.ordinal}
        raise RuntimeError("unknown role-relative atom kind")


@dataclass(frozen=True, slots=True)
class _SparseSynapses:
    row_count: int
    column_count: int
    row_offsets: np.ndarray
    columns: np.ndarray
    excitatory: np.ndarray

    @classmethod
    def grow(
        cls,
        row_count: int,
        column_count: int,
        ticks: Mapping[tuple[int, int], int],
    ) -> _SparseSynapses:
        items = sorted(ticks.items())
        row_sizes = np.zeros(row_count, dtype=np.int32)
        columns = np.empty(len(items), dtype=np.int32)
        excitatory = np.empty(len(items), dtype=np.uint8)
        for index, ((row, column), support) in enumerate(items):
            if not 0 <= row < row_count or not 0 <= column < column_count:
                raise RuntimeError("sparse synapse address left its population")
            if type(support) is not int or not 1 <= support <= _SCORE_CAP:
                raise RuntimeError("sparse synapse support left its tick bounds")
            row_sizes[row] += 1
            columns[index] = column
            excitatory[index] = support
        offsets = np.zeros(row_count + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(row_sizes, dtype=np.int32)
        for array in (offsets, columns, excitatory):
            array.setflags(write=False)
        return cls(
            row_count=row_count,
            column_count=column_count,
            row_offsets=offsets,
            columns=columns,
            excitatory=excitatory,
        )

    def outgoing(self, row: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not 0 <= row < self.row_count:
            raise RuntimeError("sparse synapse row left its population")
        begin = int(self.row_offsets[row])
        end = int(self.row_offsets[row + 1])
        columns = tuple(int(value) for value in self.columns[begin:end])
        scores = tuple(int(value) for value in self.excitatory[begin:end])
        return columns, scores

    @property
    def synapse_count(self) -> int:
        return int(self.columns.size)

    @property
    def excitatory_ticks(self) -> int:
        return int(np.sum(self.excitatory, dtype=np.uint64))


@dataclass(frozen=True, slots=True)
class _CellGeometry:
    context_offset: int
    chunk_offset: int
    state_offset: int
    emission_offset: int
    role_offset: int
    continue_cell: int
    exit_cell: int
    stop_cell: int
    total_cells: int

    def record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: CellRaceTrace


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    receipt: TransitionChainReceipt
    ablation: CompositionAblation


def _atom_key(atom: _Atom) -> bytes:
    return _canonical_json(atom.record())


def _chunk_key(chunk: tuple[_Atom, ...]) -> bytes:
    return _canonical_json(tuple(atom.record() for atom in chunk))


def _normalize(demonstration: RoleSequenceDemonstration) -> tuple[_Atom, ...]:
    role_ordinals = {event: ordinal for ordinal, event in enumerate(demonstration.role_inventory)}
    return tuple(
        _Atom(kind="role", ordinal=role_ordinals[event])
        if event in role_ordinals
        else _Atom(kind="literal", literal=event)
        for event in demonstration.emissions
    )


def _increment(ticks: dict[tuple[int, int], int], row: int, column: int) -> None:
    key = (row, column)
    current = ticks.get(key, 0)
    ticks[key] = current + 1 if current < _SCORE_CAP else _SCORE_CAP


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json({"dtype": array.dtype.str, "name": name, "shape": array.shape})
    return len(header).to_bytes(4, "big") + header + array.tobytes(order="C")


def _checkpoint_sha256(
    *,
    config: RoleRelativeCompositorConfig,
    contexts: tuple[bytes, ...],
    chunks: tuple[tuple[_Atom, ...], ...],
    atoms: tuple[_Atom, ...],
    states: tuple[tuple[int, int], ...],
    geometry: _CellGeometry,
    banks: tuple[tuple[str, _SparseSynapses], ...],
) -> str:
    digest = sha256(_CHECKPOINT_NAMESPACE)
    digest.update(
        _canonical_json(
            {
                "atoms": tuple(atom.record() for atom in atoms),
                "chunks": tuple(tuple(atom.record() for atom in chunk) for chunk in chunks),
                "config": asdict(config),
                "contexts_hex": tuple(value.hex() for value in contexts),
                "geometry": geometry.record(),
                "states": states,
            }
        )
    )
    for name, bank in banks:
        digest.update(
            _canonical_json(
                {
                    "column_count": bank.column_count,
                    "name": name,
                    "row_count": bank.row_count,
                }
            )
        )
        digest.update(_array_frame(f"{name}.row_offsets", bank.row_offsets))
        digest.update(_array_frame(f"{name}.columns", bank.columns))
        digest.update(_array_frame(f"{name}.excitatory", bank.excitatory))
    return digest.hexdigest()


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
        raise RuntimeError("physical race score left its calibrated bounds")
    all_scores = (*scores, _TONIC_STOP_SCORE)
    all_cells = (*candidate_cells, stop_cell)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_cells) * _RACE_STEPS
    relay_step: int | None = None
    stopped = False
    stop_reason: str | None = None
    selected_column: int | None = None

    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_STOP_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("role-relative collision STOP relay changed")
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
        selected_column = columns[result.winner_cell]
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
    trace = CellRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record))
    return _RaceDecision(column=selected_column, trace=trace)


class RoleRelativeSequenceCompositor:
    """Deep module for learned role-relative chunk traversal and composition."""

    __slots__ = (
        "_atoms",
        "_banks",
        "_base_checkpoint_sha256",
        "_checkpoint_sha256",
        "_chunks",
        "_config",
        "_context_index",
        "_contexts",
        "_geometry",
        "_inhibitory",
        "_pending",
        "_states",
        "_summary",
    )

    def __init__(
        self,
        *,
        contexts: tuple[bytes, ...],
        chunks: tuple[tuple[_Atom, ...], ...],
        atoms: tuple[_Atom, ...],
        states: tuple[tuple[int, int], ...],
        geometry: _CellGeometry,
        banks: Mapping[str, _SparseSynapses],
        config: RoleRelativeCompositorConfig,
        summary: RoleRelativeTrainingSummary,
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use RoleRelativeSequenceCompositor.grow")
        self._contexts = contexts
        self._chunks = chunks
        self._atoms = atoms
        self._states = states
        self._geometry = geometry
        self._banks = MappingProxyType(dict(banks))
        self._config = config
        self._summary = summary
        self._base_checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_sha256 = checkpoint_sha256
        self._inhibitory: dict[tuple[str, int, int], int] = {}
        self._pending: _PendingDecision | None = None
        self._context_index = MappingProxyType(
            {value: index for index, value in enumerate(contexts)}
        )

    @classmethod
    def grow(
        cls,
        demonstrations: Iterable[RoleSequenceDemonstration],
        config: RoleRelativeCompositorConfig | None = None,
    ) -> RoleRelativeSequenceCompositor:
        """Recruit sparse synapses from complete opaque demonstrations."""

        rows = tuple(demonstrations)
        if not rows:
            raise ValueError("at least one complete demonstration is required")
        if any(not isinstance(row, RoleSequenceDemonstration) for row in rows):
            raise TypeError("demonstrations must contain RoleSequenceDemonstration")
        active_config = config or RoleRelativeCompositorConfig()
        if not isinstance(active_config, RoleRelativeCompositorConfig):
            raise TypeError("config must be a RoleRelativeCompositorConfig")
        if len(rows) > active_config.max_demonstrations:
            raise ValueError("demonstration count exceeds configured capacity")
        for row in rows:
            if len(row.role_inventory) > active_config.max_roles:
                raise ValueError("demonstration role inventory exceeds capacity")
            if len(row.emissions) > active_config.max_chunk_emissions:
                raise ValueError("demonstration emission sequence exceeds capacity")
            for event in (row.context_event, *row.role_inventory, *row.emissions):
                if len(event) > active_config.max_event_bytes:
                    raise ValueError("opaque event exceeds configured byte capacity")

        normalized = tuple(_normalize(row) for row in rows)
        contexts = tuple(sorted({row.context_event for row in rows}))
        chunks = tuple(sorted(set(normalized), key=_chunk_key))
        atoms = tuple(
            sorted(
                {atom for chunk in chunks for atom in chunk},
                key=_atom_key,
            )
        )
        context_index = {value: index for index, value in enumerate(contexts)}
        chunk_index = {value: index for index, value in enumerate(chunks)}
        atom_index = {value: index for index, value in enumerate(atoms)}

        states = tuple(
            (chunk_id, position)
            for chunk_id, chunk in enumerate(chunks)
            for position in range(len(chunk))
        )
        state_index = {value: index for index, value in enumerate(states)}

        context_offset = 0
        chunk_offset = context_offset + len(contexts)
        state_offset = chunk_offset + len(chunks)
        emission_offset = state_offset + len(states)
        role_offset = emission_offset + len(atoms)
        continue_cell = role_offset + active_config.max_roles
        exit_cell = continue_cell + 1
        stop_cell = exit_cell + 1
        geometry = _CellGeometry(
            context_offset=context_offset,
            chunk_offset=chunk_offset,
            state_offset=state_offset,
            emission_offset=emission_offset,
            role_offset=role_offset,
            continue_cell=continue_cell,
            exit_cell=exit_cell,
            stop_cell=stop_cell,
            total_cells=stop_cell + 1,
        )
        if geometry.total_cells > active_config.max_cells:
            raise ValueError("recruited compositor cells exceed configured capacity")

        tick_maps: dict[str, dict[tuple[int, int], int]] = {
            "context_chunk": {},
            "chunk_root": {},
            "state_emission": {},
            "state_boundary": {},
            "state_next": {},
            "role_binding": {},
        }
        chunk_support = [0] * len(chunks)
        for row, chunk in zip(rows, normalized, strict=True):
            context_id = context_index[row.context_event]
            chunk_id = chunk_index[chunk]
            chunk_support[chunk_id] += 1
            root_state = state_index[(chunk_id, 0)]
            _increment(tick_maps["context_chunk"], context_id, chunk_id)
            _increment(tick_maps["chunk_root"], chunk_id, root_state)
            for position, atom in enumerate(chunk):
                state_id = state_index[(chunk_id, position)]
                emission_id = atom_index[atom]
                _increment(tick_maps["state_emission"], state_id, emission_id)
                if atom.kind == "role":
                    if atom.ordinal is None:
                        raise RuntimeError("role atom lost its ordinal")
                    _increment(
                        tick_maps["role_binding"],
                        emission_id,
                        atom.ordinal,
                    )
                if position + 1 < len(chunk):
                    _increment(tick_maps["state_boundary"], state_id, 0)
                    next_state = state_index[(chunk_id, position + 1)]
                    _increment(tick_maps["state_next"], state_id, next_state)
                else:
                    _increment(tick_maps["state_boundary"], state_id, 1)

        banks = {
            "context_chunk": _SparseSynapses.grow(
                len(contexts), len(chunks), tick_maps["context_chunk"]
            ),
            "chunk_root": _SparseSynapses.grow(len(chunks), len(states), tick_maps["chunk_root"]),
            "state_emission": _SparseSynapses.grow(
                len(states), len(atoms), tick_maps["state_emission"]
            ),
            "state_boundary": _SparseSynapses.grow(len(states), 2, tick_maps["state_boundary"]),
            "state_next": _SparseSynapses.grow(len(states), len(states), tick_maps["state_next"]),
            "role_binding": _SparseSynapses.grow(
                len(atoms), active_config.max_roles, tick_maps["role_binding"]
            ),
        }
        ordered_banks = tuple((name, banks[name]) for name in tick_maps)
        checkpoint = _checkpoint_sha256(
            config=active_config,
            contexts=contexts,
            chunks=chunks,
            atoms=atoms,
            states=states,
            geometry=geometry,
            banks=ordered_banks,
        )
        summary = RoleRelativeTrainingSummary(
            demonstrations=len(rows),
            context_cells=len(contexts),
            chunk_cells=len(chunks),
            recurrent_state_cells=len(states),
            emission_cells=len(atoms),
            recurring_chunks=sum(support > 1 for support in chunk_support),
            learned_synapses=sum(bank.synapse_count for bank in banks.values()),
            excitatory_ticks=sum(bank.excitatory_ticks for bank in banks.values()),
            total_cells=geometry.total_cells,
        )
        return cls(
            contexts=contexts,
            chunks=chunks,
            atoms=atoms,
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
    def training_summary(self) -> RoleRelativeTrainingSummary:
        return self._summary

    def respond(
        self,
        observation_events: tuple[bytes, ...],
        role_inventory: tuple[bytes, ...],
        *,
        ablation: CompositionAblation = CompositionAblation.NONE,
    ) -> RoleRelativeComposition:
        """Discover learned contexts and compose their role-bound chunks.

        Unknown observation events carry no host-side routing meaning.  Learned
        context cells found in the stream define the recurrent traversal order.
        An EMIT decision remains pending until :meth:`resolve` is called.
        """

        if self._pending is not None:
            raise PendingConsequenceError("resolve the pending EMIT decision before responding")
        if type(observation_events) is not tuple or not observation_events:
            raise ValueError("observation_events must be a non-empty tuple")
        if len(observation_events) > self._config.max_observation_events:
            raise ValueError("observation stream exceeds configured capacity")
        for index, event in enumerate(observation_events):
            _require_event(event, f"observation_events[{index}]")
            if len(event) > self._config.max_event_bytes:
                raise ValueError("observation event exceeds configured byte capacity")
        roles = _require_inventory(role_inventory, "role_inventory")
        if len(roles) > self._config.max_roles:
            raise ValueError("role inventory exceeds configured capacity")
        if not isinstance(ablation, CompositionAblation):
            raise TypeError("ablation must be a CompositionAblation")

        query_record = {
            "ablation": ablation.value,
            "checkpoint_sha256": self._checkpoint_sha256,
            "observation_events_hex": tuple(value.hex() for value in observation_events),
            "role_inventory_hex": tuple(value.hex() for value in roles),
        }
        query_sha256 = _digest(_QUERY_NAMESPACE, query_record)
        races: list[CellRaceTrace] = []
        transitions: list[EligibilityTransition] = []
        emitted: list[bytes] = []

        active_contexts = tuple(
            context_id
            for event in observation_events
            if (context_id := self._context_index.get(event)) is not None
        )
        if not active_contexts:
            empty_decision = _physical_race(
                population="context_chunk",
                columns=(),
                scores=(),
                candidate_cells=(),
                stop_cell=self._geometry.stop_cell,
            )
            races.append(empty_decision.trace)
            return self._stop(
                query_sha256,
                races,
                f"context_chunk:{empty_decision.trace.stop_reason}",
            )

        for context_position, context_id in enumerate(active_contexts):
            ablated = ablation is CompositionAblation.COMPOSITOR_OFF and context_position > 0
            chunk_decision = self._select(
                "context_chunk",
                context_id,
                target_offset=self._geometry.chunk_offset,
                ablated=ablated,
            )
            races.append(chunk_decision.trace)
            if chunk_decision.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"context_chunk:{chunk_decision.trace.stop_reason}",
                )
            chunk_id = chunk_decision.column
            transitions.append(self._eligibility("context_chunk", context_id, chunk_id))

            root_decision = self._select(
                "chunk_root",
                chunk_id,
                target_offset=self._geometry.state_offset,
            )
            races.append(root_decision.trace)
            if root_decision.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"chunk_root:{root_decision.trace.stop_reason}",
                )
            state_id = root_decision.column
            transitions.append(self._eligibility("chunk_root", chunk_id, state_id))

            for _ in range(self._config.max_chunk_emissions):
                emission_decision = self._select(
                    "state_emission",
                    state_id,
                    target_offset=self._geometry.emission_offset,
                )
                races.append(emission_decision.trace)
                if emission_decision.column is None:
                    return self._stop(
                        query_sha256,
                        races,
                        f"state_emission:{emission_decision.trace.stop_reason}",
                    )
                emission_id = emission_decision.column
                transitions.append(self._eligibility("state_emission", state_id, emission_id))
                atom = self._atoms[emission_id]
                if atom.kind == "literal":
                    if atom.literal is None:
                        raise RuntimeError("selected literal atom has no event")
                    emitted.append(atom.literal)
                else:
                    binding_decision = self._select_role(emission_id, len(roles))
                    races.append(binding_decision.trace)
                    if binding_decision.column is None:
                        return self._stop(
                            query_sha256,
                            races,
                            f"role_binding:{binding_decision.trace.stop_reason}",
                        )
                    role_ordinal = binding_decision.column
                    transitions.append(self._eligibility("role_binding", emission_id, role_ordinal))
                    emitted.append(roles[role_ordinal])

                boundary_decision = self._select_boundary(state_id)
                races.append(boundary_decision.trace)
                if boundary_decision.column is None:
                    return self._stop(
                        query_sha256,
                        races,
                        f"state_boundary:{boundary_decision.trace.stop_reason}",
                    )
                boundary = boundary_decision.column
                transitions.append(self._eligibility("state_boundary", state_id, boundary))
                if boundary == 1:
                    break

                next_decision = self._select(
                    "state_next",
                    state_id,
                    target_offset=self._geometry.state_offset,
                    ablated=ablation is CompositionAblation.RECURRENCE_OFF,
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
                return self._stop(
                    query_sha256,
                    races,
                    "chunk_traversal:bounded_stop",
                )

        output = tuple(emitted)
        output_sha256 = _digest(
            _DECISION_NAMESPACE,
            {"emissions_hex": tuple(event.hex() for event in output)},
        )
        physical_cell_steps = sum(row.physical_cell_steps for row in races)
        receipt_record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "output_sha256": output_sha256,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(row.sha256 for row in races),
            "transition_sha256s": tuple(row.sha256 for row in transitions),
        }
        receipt = TransitionChainReceipt(
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            output_sha256=output_sha256,
            transitions=tuple(transitions),
            race_sha256s=tuple(row.sha256 for row in races),
            physical_cell_steps=physical_cell_steps,
            sha256=_digest(_RECEIPT_NAMESPACE, receipt_record),
        )
        decision_record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": "EMIT",
            "eligibility_sha256": receipt.sha256,
            "emissions_hex": tuple(event.hex() for event in output),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": receipt.race_sha256s,
            "stop_reason": None,
        }
        result = RoleRelativeComposition(
            decision="EMIT",
            emissions=output,
            stop_reason=None,
            eligibility=receipt,
            races=tuple(races),
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, decision_record),
        )
        self._pending = _PendingDecision(receipt=receipt, ablation=ablation)
        return result

    def resolve(self, valence: Valence) -> ConsequenceUpdate:
        """Resolve one pending EMIT and apply chain-local negative credit once."""

        if not isinstance(valence, Valence):
            raise TypeError("valence must be a Valence")
        pending = self._pending
        if pending is None:
            raise PendingConsequenceError("no pending EMIT decision to resolve")
        self._pending = None
        prior_training_sha256 = self._checkpoint_sha256
        changed: tuple[EligibilityTransition, ...] = ()

        if valence is Valence.NEGATIVE and pending.ablation is not CompositionAblation.NOGO_OFF:
            selected = next(
                (
                    transition
                    for transition in reversed(pending.receipt.transitions)
                    if transition.bank == "context_chunk"
                ),
                None,
            )
            if selected is None:
                raise RuntimeError("pending composition lost its competitive context binding")
            key = (selected.bank, selected.row, selected.column)
            old_ticks = self._inhibitory.get(key, 0)
            new_ticks = min(_SCORE_CAP, old_ticks + _NOGO_TICKS)
            if new_ticks != old_ticks:
                self._inhibitory[key] = new_ticks
                self._checkpoint_sha256 = self._training_sha256()
                changed = (selected,)

        record = {
            "changed_transition_sha256s": tuple(edge.sha256 for edge in changed),
            "prior_training_sha256": prior_training_sha256,
            "receipt_sha256": pending.receipt.sha256,
            "training_sha256": self._checkpoint_sha256,
            "valence": valence.value,
        }
        return ConsequenceUpdate(
            valence=valence,
            receipt_sha256=pending.receipt.sha256,
            changed_transitions=changed,
            prior_training_sha256=prior_training_sha256,
            training_sha256=self._checkpoint_sha256,
            sha256=_digest(_CONSEQUENCE_NAMESPACE, record),
        )

    def _training_sha256(self) -> str:
        return _digest(
            _TRAINING_NAMESPACE,
            {
                "base_checkpoint_sha256": self._base_checkpoint_sha256,
                "inhibitory_ticks": tuple(
                    (bank, row, column, ticks)
                    for (bank, row, column), ticks in sorted(self._inhibitory.items())
                ),
            },
        )

    def _select(
        self,
        bank_name: str,
        row: int,
        *,
        target_offset: int,
        ablated: bool = False,
    ) -> _RaceDecision:
        bank = self._banks[bank_name]
        columns, scores = bank.outgoing(row)
        active_scores = self._effective_scores(bank_name, row, columns, scores, ablated)
        return _physical_race(
            population=bank_name,
            columns=columns,
            scores=active_scores,
            candidate_cells=tuple(target_offset + column for column in columns),
            stop_cell=self._geometry.stop_cell,
        )

    def _select_boundary(self, state_id: int) -> _RaceDecision:
        bank = self._banks["state_boundary"]
        columns, scores = bank.outgoing(state_id)
        active_scores = self._effective_scores("state_boundary", state_id, columns, scores, False)
        control_cells = (self._geometry.continue_cell, self._geometry.exit_cell)
        return _physical_race(
            population="state_boundary",
            columns=columns,
            scores=active_scores,
            candidate_cells=tuple(control_cells[column] for column in columns),
            stop_cell=self._geometry.stop_cell,
        )

    def _select_role(self, emission_id: int, role_count: int) -> _RaceDecision:
        bank = self._banks["role_binding"]
        columns, scores = bank.outgoing(emission_id)
        active_scores = self._effective_scores("role_binding", emission_id, columns, scores, False)
        available_scores = tuple(
            score if column < role_count else 0
            for column, score in zip(columns, active_scores, strict=True)
        )
        return _physical_race(
            population="role_binding",
            columns=columns,
            scores=available_scores,
            candidate_cells=tuple(self._geometry.role_offset + column for column in columns),
            stop_cell=self._geometry.stop_cell,
        )

    def _effective_scores(
        self,
        bank_name: str,
        row: int,
        columns: tuple[int, ...],
        scores: tuple[int, ...],
        ablated: bool,
    ) -> tuple[int, ...]:
        if ablated:
            return tuple(0 for _ in scores)
        return tuple(
            max(0, score - self._inhibitory.get((bank_name, row, column), 0))
            for column, score in zip(columns, scores, strict=True)
        )

    def _eligibility(
        self,
        bank_name: str,
        row: int,
        column: int,
    ) -> EligibilityTransition:
        if bank_name == "context_chunk":
            source = self._geometry.context_offset + row
            target = self._geometry.chunk_offset + column
        elif bank_name == "chunk_root":
            source = self._geometry.chunk_offset + row
            target = self._geometry.state_offset + column
        elif bank_name == "state_emission":
            source = self._geometry.state_offset + row
            target = self._geometry.emission_offset + column
        elif bank_name == "state_boundary":
            source = self._geometry.state_offset + row
            target = self._geometry.continue_cell if column == 0 else self._geometry.exit_cell
        elif bank_name == "state_next":
            source = self._geometry.state_offset + row
            target = self._geometry.state_offset + column
        elif bank_name == "role_binding":
            source = self._geometry.emission_offset + row
            target = self._geometry.role_offset + column
        else:
            raise RuntimeError("unknown eligibility synapse bank")
        record = {
            "bank": bank_name,
            "checkpoint_sha256": self._checkpoint_sha256,
            "column": column,
            "row": row,
            "source_cell": source,
            "target_cell": target,
        }
        return EligibilityTransition(
            bank=bank_name,
            row=row,
            column=column,
            source_cell=source,
            target_cell=target,
            sha256=_digest(_EDGE_NAMESPACE, record),
        )

    def _stop(
        self,
        query_sha256: str,
        races: list[CellRaceTrace],
        reason: str,
    ) -> RoleRelativeComposition:
        physical_cell_steps = sum(row.physical_cell_steps for row in races)
        record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": "STOP",
            "eligibility_sha256": None,
            "emissions_hex": (),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(row.sha256 for row in races),
            "stop_reason": reason,
        }
        return RoleRelativeComposition(
            decision="STOP",
            emissions=(),
            stop_reason=reason,
            eligibility=None,
            races=tuple(races),
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
