"""Domain-neutral sparse neuronal pointers with reversible role/filler wiring.

Opaque bytes are deterministically allocated to a 64-of-4096 block code.  The
allocation hash is sensory scaffolding only.  Binding, unbinding, synaptic
superposition, and associative cleanup operate on simulated cell addresses.
Every returned transform or cleanup decision is the result of fitted GLIF5
first-spike competition; hashes only make immutable records auditable.

This is a reusable representational substrate, not a semantic learner or a
claim of language understanding.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256, shake_256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "SparseBlockPointer",
    "SparseCleanupDecision",
    "SparseCleanupRaceTrace",
    "SparseNeuronalRoleFiller",
    "SparsePointerSuperposition",
    "SparseRoleFillerAblation",
    "SparseSubstrateCounts",
    "SparseTransformDecision",
    "SparseTransformRaceTrace",
]


_BLOCK_COUNT = 64
_CELLS_PER_BLOCK = 64
_POINTER_CELL_COUNT = _BLOCK_COUNT * _CELLS_PER_BLOCK
_MAX_OPAQUE_BYTES = 4_096
_MAX_OBSERVATIONS = 4_096
_MAX_LIVE_POINTERS = 4_096
_SUPPORT_DTYPE = np.dtype("<u2")

_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = _BLOCK_COUNT
_TONIC_STOP_SCORE = 16
_ROUTED_SCORE = _SCORE_CAP

# The filler lane is permuted and multiplied by a unit in Z/64Z.  This makes
# the fixed wiring ordered while retaining an exact inverse for unbinding.
_FILLER_MULTIPLIER = 3
_FILLER_INVERSE = 43
_FILLER_BLOCK_STRIDE = 13
_FILLER_BLOCK_OFFSET = 7
_OUTPUT_OFFSET_STRIDE = 11
_OUTPUT_OFFSET = 5

_PROJECTION_NAMESPACE = b"humanbrain.sparse-role-filler.projection.v1\0"
_POINTER_NAMESPACE = b"humanbrain.sparse-role-filler.pointer.v1\0"
_SUPERPOSITION_NAMESPACE = b"humanbrain.sparse-role-filler.superposition.v1\0"
_RACE_NAMESPACE = b"humanbrain.sparse-role-filler.race.v1\0"
_DECISION_NAMESPACE = b"humanbrain.sparse-role-filler.decision.v1\0"


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


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _pointer_digest(block_cells: tuple[int, ...]) -> str:
    return _digest(_POINTER_NAMESPACE, {"block_cells": block_cells})


def _make_pointer(block_cells: Iterable[int]) -> SparseBlockPointer:
    cells = tuple(int(cell) for cell in block_cells)
    return SparseBlockPointer(block_cells=cells, sha256=_pointer_digest(cells))


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(_SCORE_CAP + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("sparse-pointer fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score
        for score, result in enumerate(controls)
    ):
        raise RuntimeError("sparse-pointer fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class SparseRoleFillerAblation(StrEnum):
    """Causal controls for the two reusable neuronal mechanisms."""

    NONE = "none"
    BINDING_OFF = "binding_off"
    CLEANUP_OFF = "cleanup_off"


@dataclass(frozen=True, slots=True)
class SparseSubstrateCounts:
    """Fixed logical size of the procedurally wired substrate.

    Each product branch has two afferent synapses and terminates on one output
    cell.  The branches are fixed arithmetic wiring, so the dense table is not
    materialized in host memory.
    """

    block_count: int = _BLOCK_COUNT
    cells_per_block: int = _CELLS_PER_BLOCK
    pointer_cells: int = _POINTER_CELL_COUNT
    binding_race_cells: int = _BLOCK_COUNT * (_CELLS_PER_BLOCK + 1)
    unbinding_race_cells: int = _BLOCK_COUNT * (_CELLS_PER_BLOCK + 1)
    cleanup_stop_cells: int = 2
    binding_wiring_synapses: int = 2 * _BLOCK_COUNT * _CELLS_PER_BLOCK**2
    unbinding_wiring_synapses: int = 2 * _BLOCK_COUNT * _CELLS_PER_BLOCK**2
    superposition_synapses: int = _POINTER_CELL_COUNT

    @property
    def fixed_cell_count(self) -> int:
        return (
            self.pointer_cells
            + self.binding_race_cells
            + self.unbinding_race_cells
            + self.cleanup_stop_cells
        )

    @property
    def fixed_synapse_count(self) -> int:
        return (
            self.binding_wiring_synapses
            + self.unbinding_wiring_synapses
            + self.superposition_synapses
        )


@dataclass(frozen=True, slots=True)
class SparseBlockPointer:
    """One immutable active cell in each fixed block."""

    block_cells: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if type(self.block_cells) is not tuple or len(self.block_cells) != _BLOCK_COUNT:
            raise ValueError(f"block_cells must contain exactly {_BLOCK_COUNT} cells")
        if any(
            type(cell) is not int or not 0 <= cell < _CELLS_PER_BLOCK
            for cell in self.block_cells
        ):
            raise ValueError("each block cell must be an in-range integer")
        _require_sha256(self.sha256, "sha256")
        if self.sha256 != _pointer_digest(self.block_cells):
            raise ValueError("pointer digest does not match its active cells")

    @property
    def active_cell_count(self) -> int:
        return _BLOCK_COUNT

    @property
    def global_cells(self) -> tuple[int, ...]:
        return tuple(
            block * _CELLS_PER_BLOCK + cell
            for block, cell in enumerate(self.block_cells)
        )

    def with_block_cells(
        self,
        updates: tuple[tuple[int, int], ...],
    ) -> SparseBlockPointer:
        """Return a fresh pointer with explicit block-level perturbations."""

        if type(updates) is not tuple:
            raise TypeError("updates must be a tuple")
        cells = list(self.block_cells)
        seen: set[int] = set()
        for block, cell in updates:
            if type(block) is not int or not 0 <= block < _BLOCK_COUNT:
                raise ValueError("updated block is out of range")
            if block in seen:
                raise ValueError("each block may be updated once")
            if type(cell) is not int or not 0 <= cell < _CELLS_PER_BLOCK:
                raise ValueError("updated cell is out of range")
            seen.add(block)
            cells[block] = cell
        return _make_pointer(cells)


def _superposition_digest(observation_count: int, support_uint16_le: bytes) -> str:
    header = observation_count.to_bytes(4, "little", signed=False)
    return sha256(_SUPERPOSITION_NAMESPACE + header + support_uint16_le).hexdigest()


@dataclass(frozen=True, slots=True)
class SparsePointerSuperposition:
    """Immutable compact support ticks for all 4,096 pointer-cell synapses."""

    observation_count: int
    support_uint16_le: bytes
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.observation_count) is not int
            or not 0 <= self.observation_count <= _MAX_OBSERVATIONS
        ):
            raise ValueError("observation_count is out of range")
        if not isinstance(self.support_uint16_le, bytes):
            raise TypeError("support_uint16_le must be bytes")
        if len(self.support_uint16_le) != _POINTER_CELL_COUNT * _SUPPORT_DTYPE.itemsize:
            raise ValueError("support state has the wrong byte length")
        support = np.frombuffer(self.support_uint16_le, dtype=_SUPPORT_DTYPE)
        if np.any(support > self.observation_count):
            raise ValueError("support ticks exceed the observation count")
        _require_sha256(self.sha256, "sha256")
        expected = _superposition_digest(self.observation_count, self.support_uint16_le)
        if self.sha256 != expected:
            raise ValueError("superposition digest does not match its synaptic state")

    @classmethod
    def empty(cls) -> SparsePointerSuperposition:
        support = bytes(_POINTER_CELL_COUNT * _SUPPORT_DTYPE.itemsize)
        return cls(
            observation_count=0,
            support_uint16_le=support,
            sha256=_superposition_digest(0, support),
        )

    @property
    def allocated_synapse_count(self) -> int:
        return _POINTER_CELL_COUNT

    @property
    def nonzero_synapse_count(self) -> int:
        return int(np.count_nonzero(np.frombuffer(self.support_uint16_le, dtype=_SUPPORT_DTYPE)))

    @property
    def storage_bytes(self) -> int:
        return len(self.support_uint16_le)

    @property
    def plastic_synapse_events(self) -> int:
        return self.observation_count * _BLOCK_COUNT

    def support_at(self, block: int, cell: int) -> int:
        if not 0 <= block < _BLOCK_COUNT or not 0 <= cell < _CELLS_PER_BLOCK:
            raise ValueError("support address is out of range")
        support = np.frombuffer(self.support_uint16_le, dtype=_SUPPORT_DTYPE)
        return int(support[block * _CELLS_PER_BLOCK + cell])

    def accumulate(self, pointer: SparseBlockPointer) -> SparsePointerSuperposition:
        """Potentiate the 64 active synapses and return a new checkpoint."""

        if not isinstance(pointer, SparseBlockPointer):
            raise TypeError("pointer must be a SparseBlockPointer")
        if self.observation_count >= _MAX_OBSERVATIONS:
            raise ValueError("superposition observation bound reached")
        support = np.frombuffer(self.support_uint16_le, dtype=_SUPPORT_DTYPE).copy()
        global_cells = np.asarray(pointer.global_cells, dtype=np.intp)
        support[global_cells] += 1
        support_bytes = support.astype(_SUPPORT_DTYPE, copy=False).tobytes()
        count = self.observation_count + 1
        return SparsePointerSuperposition(
            observation_count=count,
            support_uint16_le=support_bytes,
            sha256=_superposition_digest(count, support_bytes),
        )


@dataclass(frozen=True, slots=True)
class SparseTransformRaceTrace:
    """Auditable physical output races for one binding transformation."""

    population: str
    block_race_count: int
    cells_per_race: int
    winner_cells: tuple[int, ...]
    spike_steps: tuple[int, ...]
    simultaneous_spike_cells: tuple[int, ...]
    stopped: bool
    stop_reason: str | None
    collision_stop_spike_step: int | None
    physical_cell_steps: int
    active_synapses: int
    routed_cells_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SparseTransformDecision:
    """Immutable pointer assembly selected by independent block-cell races."""

    pointer: SparseBlockPointer | None
    trace: SparseTransformRaceTrace
    sha256: str

    @property
    def stopped(self) -> bool:
        return self.pointer is None


@dataclass(frozen=True, slots=True)
class SparseCleanupRaceTrace:
    """Auditable fitted-cell cleanup competition over live pointers."""

    population: str
    live_pointer_count: int
    live_support_ticks: tuple[int, ...]
    live_neural_scores: tuple[int, ...]
    tonic_stop_score: int
    winner_cell: int | None
    spike_step: int
    simultaneous_spike_cells: int
    stopped: bool
    stop_reason: str | None
    collision_stop_spike_step: int | None
    physical_cell_steps: int
    active_synapses: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SparseCleanupDecision:
    """Immutable cleanup result whose pointer, if any, won the cell race."""

    pointer: SparseBlockPointer | None
    trace: SparseCleanupRaceTrace
    sha256: str

    @property
    def stopped(self) -> bool:
        return self.pointer is None


def _collision_stop_relay() -> int:
    relay = _CELL_RACE.race_many(
        np.asarray([[_SCORE_CURRENTS[_TONIC_STOP_SCORE]]], dtype=np.float64)
    )[0]
    if relay.simultaneous_spike_candidates != 1 or relay.winner_cell != 0:
        raise RuntimeError("sparse-pointer collision STOP relay changed")
    return relay.spike_step


def _filler_block(block: int) -> int:
    return (_FILLER_BLOCK_STRIDE * block + _FILLER_BLOCK_OFFSET) % _BLOCK_COUNT


def _output_offset(block: int) -> int:
    return (_OUTPUT_OFFSET_STRIDE * block + _OUTPUT_OFFSET) % _CELLS_PER_BLOCK


def _bound_cells(role: SparseBlockPointer, filler: SparseBlockPointer) -> tuple[int, ...]:
    return tuple(
        (
            role.block_cells[block]
            + _FILLER_MULTIPLIER * filler.block_cells[_filler_block(block)]
            + _output_offset(block)
        )
        % _CELLS_PER_BLOCK
        for block in range(_BLOCK_COUNT)
    )


def _unbound_cells(role: SparseBlockPointer, bound: SparseBlockPointer) -> tuple[int, ...]:
    filler_cells = [0] * _BLOCK_COUNT
    for block in range(_BLOCK_COUNT):
        filler_cells[_filler_block(block)] = (
            _FILLER_INVERSE
            * (
                bound.block_cells[block]
                - role.block_cells[block]
                - _output_offset(block)
            )
        ) % _CELLS_PER_BLOCK
    return tuple(filler_cells)


def _transform_race(
    *,
    population: str,
    routed_cells: tuple[int, ...],
    ablated: bool,
) -> SparseTransformDecision:
    scores = np.zeros((_BLOCK_COUNT, _CELLS_PER_BLOCK + 1), dtype=np.intp)
    if not ablated:
        scores[np.arange(_BLOCK_COUNT), np.asarray(routed_cells, dtype=np.intp)] = (
            _ROUTED_SCORE
        )
    scores[:, -1] = _TONIC_STOP_SCORE
    results = _CELL_RACE.race_many(_SCORE_CURRENTS[scores], require_unique=False)
    winners = tuple(result.winner_cell for result in results)
    spike_steps = tuple(result.spike_step for result in results)
    simultaneous = tuple(result.simultaneous_spike_candidates for result in results)

    collision_step: int | None = None
    stop_reason: str | None = None
    if any(count != 1 for count in simultaneous):
        collision_step = _collision_stop_relay()
        stop_reason = "collision"
    elif any(winner == _CELLS_PER_BLOCK for winner in winners):
        stop_reason = "tonic_stop"

    stopped = stop_reason is not None
    physical_cell_steps = _BLOCK_COUNT * (_CELLS_PER_BLOCK + 1) * _RACE_STEPS
    if collision_step is not None:
        physical_cell_steps += _RACE_STEPS
    active_synapses = _BLOCK_COUNT + (0 if ablated else 2 * _BLOCK_COUNT)
    routed_sha256 = _digest(
        _RACE_NAMESPACE,
        {"ablated": ablated, "population": population, "routed_cells": routed_cells},
    )
    record = {
        "active_synapses": active_synapses,
        "block_race_count": _BLOCK_COUNT,
        "cells_per_race": _CELLS_PER_BLOCK + 1,
        "collision_stop_spike_step": collision_step,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "routed_cells_sha256": routed_sha256,
        "simultaneous_spike_cells": simultaneous,
        "spike_steps": spike_steps,
        "stop_reason": stop_reason,
        "stopped": stopped,
        "winner_cells": winners,
    }
    trace = SparseTransformRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )
    pointer = None if stopped else _make_pointer(winners)
    decision_record = {
        "pointer_sha256": None if pointer is None else pointer.sha256,
        "trace_sha256": trace.sha256,
    }
    return SparseTransformDecision(
        pointer=pointer,
        trace=trace,
        sha256=_digest(_DECISION_NAMESPACE, decision_record),
    )


@dataclass(frozen=True, slots=True)
class SparseNeuronalRoleFiller:
    """Fixed sparse pointer, product-wiring, and cleanup substrate."""

    counts: SparseSubstrateCounts = SparseSubstrateCounts()

    def project(self, opaque_bytes: bytes) -> SparseBlockPointer:
        """Allocate bounded opaque sensory bytes to one cell per block."""

        if not isinstance(opaque_bytes, bytes) or not opaque_bytes:
            raise ValueError("opaque_bytes must be non-empty bytes")
        if len(opaque_bytes) > _MAX_OPAQUE_BYTES:
            raise ValueError("opaque_bytes exceeds the sensory bound")
        payload = len(opaque_bytes).to_bytes(4, "little") + opaque_bytes
        allocation = shake_256(_PROJECTION_NAMESPACE + payload).digest(_BLOCK_COUNT)
        return _make_pointer(value % _CELLS_PER_BLOCK for value in allocation)

    def bind(
        self,
        role: SparseBlockPointer,
        filler: SparseBlockPointer,
        *,
        ablation: SparseRoleFillerAblation = SparseRoleFillerAblation.NONE,
    ) -> SparseTransformDecision:
        """Route a role and filler through fixed ordered product wiring."""

        self._require_pointer(role, "role")
        self._require_pointer(filler, "filler")
        self._require_ablation(ablation)
        return _transform_race(
            population="binding",
            routed_cells=_bound_cells(role, filler),
            ablated=ablation is SparseRoleFillerAblation.BINDING_OFF,
        )

    def unbind(
        self,
        role: SparseBlockPointer,
        bound: SparseBlockPointer,
        *,
        ablation: SparseRoleFillerAblation = SparseRoleFillerAblation.NONE,
    ) -> SparseTransformDecision:
        """Invert the role lane through its fixed product wiring."""

        self._require_pointer(role, "role")
        self._require_pointer(bound, "bound")
        self._require_ablation(ablation)
        return _transform_race(
            population="unbinding",
            routed_cells=_unbound_cells(role, bound),
            ablated=ablation is SparseRoleFillerAblation.BINDING_OFF,
        )

    def cleanup(
        self,
        superposition: SparsePointerSuperposition,
        live_pointers: tuple[SparseBlockPointer, ...],
        *,
        ablation: SparseRoleFillerAblation = SparseRoleFillerAblation.NONE,
    ) -> SparseCleanupDecision:
        """Race arbitrary live pointers using only accumulated synaptic support."""

        if not isinstance(superposition, SparsePointerSuperposition):
            raise TypeError("superposition must be SparsePointerSuperposition")
        if type(live_pointers) is not tuple or not 1 <= len(live_pointers) <= _MAX_LIVE_POINTERS:
            raise ValueError("live_pointers must be a bounded non-empty tuple")
        for index, pointer in enumerate(live_pointers):
            self._require_pointer(pointer, f"live_pointers[{index}]")
        self._require_ablation(ablation)

        support = np.frombuffer(
            superposition.support_uint16_le,
            dtype=_SUPPORT_DTYPE,
        ).reshape(_BLOCK_COUNT, _CELLS_PER_BLOCK)
        live_cells = np.asarray(
            tuple(pointer.block_cells for pointer in live_pointers),
            dtype=np.intp,
        )
        addressed_support = support[
            np.arange(_BLOCK_COUNT, dtype=np.intp)[None, :],
            live_cells,
        ]
        raw_support = addressed_support.sum(axis=1, dtype=np.uint64)
        if superposition.observation_count == 0:
            neural_scores = np.zeros(len(live_pointers), dtype=np.intp)
        else:
            denominator = _BLOCK_COUNT * superposition.observation_count
            neural_scores = (
                raw_support * _SCORE_CAP // denominator
            ).astype(np.intp, copy=False)
        cleanup_off = ablation is SparseRoleFillerAblation.CLEANUP_OFF
        if cleanup_off:
            neural_scores = np.zeros(len(live_pointers), dtype=np.intp)

        all_scores = np.concatenate(
            (neural_scores, np.asarray([_TONIC_STOP_SCORE], dtype=np.intp))
        )
        result = _CELL_RACE.race_many(
            _SCORE_CURRENTS[all_scores][None, :],
            require_unique=False,
        )[0]
        physical_cell_steps = len(all_scores) * _RACE_STEPS
        collision_step: int | None = None
        stop_reason: str | None = None
        winner_cell: int | None = result.winner_cell
        if result.simultaneous_spike_candidates != 1:
            collision_step = _collision_stop_relay()
            physical_cell_steps += _RACE_STEPS
            stop_reason = "collision"
            winner_cell = None
        elif result.winner_cell == len(live_pointers):
            stop_reason = "tonic_stop"
            winner_cell = None

        active_synapses = 1
        if not cleanup_off:
            active_synapses += int(np.count_nonzero(addressed_support))
        if collision_step is not None:
            active_synapses += 1
        raw_tuple = tuple(int(value) for value in raw_support)
        score_tuple = tuple(int(value) for value in neural_scores)
        record = {
            "active_synapses": active_synapses,
            "collision_stop_spike_step": collision_step,
            "live_neural_scores": score_tuple,
            "live_pointer_count": len(live_pointers),
            "live_support_ticks": raw_tuple,
            "physical_cell_steps": physical_cell_steps,
            "population": "associative_cleanup",
            "simultaneous_spike_cells": result.simultaneous_spike_candidates,
            "spike_step": result.spike_step,
            "stop_reason": stop_reason,
            "stopped": winner_cell is None,
            "tonic_stop_score": _TONIC_STOP_SCORE,
            "winner_cell": winner_cell,
        }
        trace = SparseCleanupRaceTrace(
            **record,
            sha256=_digest(_RACE_NAMESPACE, record),
        )
        pointer = None if winner_cell is None else live_pointers[winner_cell]
        decision_record = {
            "pointer_sha256": None if pointer is None else pointer.sha256,
            "superposition_sha256": superposition.sha256,
            "trace_sha256": trace.sha256,
        }
        return SparseCleanupDecision(
            pointer=pointer,
            trace=trace,
            sha256=_digest(_DECISION_NAMESPACE, decision_record),
        )

    @staticmethod
    def _require_pointer(pointer: object, name: str) -> None:
        if not isinstance(pointer, SparseBlockPointer):
            raise TypeError(f"{name} must be a SparseBlockPointer")

    @staticmethod
    def _require_ablation(ablation: object) -> None:
        if not isinstance(ablation, SparseRoleFillerAblation):
            raise TypeError("ablation must be SparseRoleFillerAblation")
