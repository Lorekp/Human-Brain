"""Variability-aware neuronal binding over complete opaque event streams.

Independent successful demonstrations expose only complete query/source streams
and the opaque role used by an action.  Learning compares the recurrent
seed-to-role path across demonstrations: identities that recur at one hop become
excitatory constraints, while identities that vary become unconstrained learned
propagation synapses.  The varying identities themselves are not retained.

Inference performs fitted-GLIF equality, recurrent propagation, exact nearest
boundary, and role-identity races.  No wildcard positions, boundaries, spans,
or semantic labels are supplied by the caller.  Unsupported structure and every
causal ablation physically select STOP.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "VariabilityAwareSegmentBinder",
    "VariabilityBinderAblation",
    "VariabilityBinderTrainingSummary",
    "VariabilityBindingDecision",
    "VariabilityRaceTrace",
    "VariabilitySegmentDemonstration",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_SCORE = 1
_MAX_DEMONSTRATIONS = 64
_MAX_QUERY_EVENTS = 256
_MAX_SOURCE_EVENTS = 4096
_MAX_EVENT_BYTES = 4096
_MAX_PROPAGATION_HOPS = 16
_MAX_BOUNDARY_HOPS = 64

_CHECKPOINT_NAMESPACE = b"humanbrain.variability-aware-segment-binder.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.variability-aware-segment-binder.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.variability-aware-segment-binder.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.variability-aware-segment-binder.decision-v1\0"
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
    if len(value) > _MAX_EVENT_BYTES:
        raise ValueError(f"{name} exceeds the opaque event byte bound")
    return value


def _require_events(value: object, name: str, *, maximum: int) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its event bound")
    for ordinal, event in enumerate(value):
        _require_event(event, f"{name}[{ordinal}]")
    return value


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (voltage - threshold)
    currents: list[float] = []
    for score in range(_DRIVE_SCORE + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("variability-binder fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, result in enumerate(controls)
    ):
        raise RuntimeError("variability-binder fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class VariabilityBinderAblation(StrEnum):
    NONE = "none"
    EQUALITY_OFF = "equality_off"
    PROPAGATION_OFF = "propagation_off"
    BOUNDARY_OFF = "boundary_off"


@dataclass(frozen=True, slots=True)
class VariabilitySegmentDemonstration:
    """One independent successful full-stream observation and its used role."""

    query_events: tuple[bytes, ...]
    source_events: tuple[bytes, ...]
    demonstrated_role: bytes

    def __post_init__(self) -> None:
        _require_events(self.query_events, "query_events", maximum=_MAX_QUERY_EVENTS)
        source = _require_events(
            self.source_events,
            "source_events",
            maximum=_MAX_SOURCE_EVENTS,
        )
        role = _require_event(self.demonstrated_role, "demonstrated_role")
        if source.count(role) != 1:
            raise ValueError("demonstrated_role must have one live source occurrence")


@dataclass(frozen=True, slots=True)
class VariabilityRaceTrace:
    population: str
    cell_count: int
    cell_scores_sha256: str
    spike_step: int
    simultaneous_spike_cells: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityBinderTrainingSummary:
    demonstration_count: int
    role_offset: int
    propagation_hop_count: int
    stable_intermediate_synapse_count: int
    wildcard_intermediate_hop_count: int
    wildcard_support_ticks: int
    boundary_synapse_count: int
    left_boundary_hop: int
    right_boundary_hop: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityBindingDecision:
    decision: str
    role_identity: bytes | None
    stop_stage: str | None
    stop_reason: str | None
    boundary_match_count: int
    races: tuple[VariabilityRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _PropagationSynapse:
    hop: int
    direction: int
    mode: str
    expected_event: bytes | None
    distinct_identity_count: int
    support_ticks: int


@dataclass(frozen=True, slots=True)
class _BoundarySynapse:
    direction: int
    event: bytes
    hop: int
    support_ticks: int


@dataclass(frozen=True, slots=True)
class _LearnedSchema:
    role_offset: int
    propagation: tuple[_PropagationSynapse, ...]
    boundaries: tuple[_BoundarySynapse, _BoundarySynapse]


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: VariabilityRaceTrace


def _trace(
    *,
    population: str,
    scores: tuple[int, ...],
    spike_step: int,
    simultaneous_spike_cells: int,
    stopped: bool,
    stop_reason: str | None,
    physical_cell_steps: int,
) -> VariabilityRaceTrace:
    score_root = _digest(_RACE_NAMESPACE, {"cell_scores": scores})
    record = {
        "cell_count": len(scores),
        "cell_scores_sha256": score_root,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_cells": simultaneous_spike_cells,
        "spike_step": spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
    }
    return VariabilityRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record))


def _physical_choice(*, population: str, scores: tuple[int, ...]) -> _RaceDecision:
    if not scores or any(score not in (0, _DRIVE_SCORE) for score in scores):
        raise ValueError("choice scores must be a non-empty binary drive vector")
    all_scores = (*scores, _TONIC_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_scores) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    selected: int | None = None
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("variability-binder collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(scores):
        stopped = True
        stop_reason = "tonic_stop"
    else:
        selected = result.winner_cell
    return _RaceDecision(
        column=selected,
        trace=_trace(
            population=population,
            scores=all_scores,
            spike_step=result.spike_step,
            simultaneous_spike_cells=result.simultaneous_spike_candidates,
            stopped=stopped,
            stop_reason=stop_reason,
            physical_cell_steps=physical_cell_steps,
        ),
    )


def _physical_detector(*, population: str, matched: bool) -> tuple[bool, VariabilityRaceTrace]:
    scores = (_DRIVE_SCORE if matched else 0, _TONIC_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    if result.simultaneous_spike_candidates != 1:
        raise RuntimeError("variability boundary detector race ceased to be unique")
    detected = result.winner_cell == 0
    return detected, _trace(
        population=population,
        scores=scores,
        spike_step=result.spike_step,
        simultaneous_spike_cells=result.simultaneous_spike_candidates,
        stopped=False,
        stop_reason=None,
        physical_cell_steps=len(scores) * _RACE_STEPS,
    )


def _physical_boundary_stop(population: str, reason: str) -> VariabilityRaceTrace:
    scores = (_TONIC_SCORE,)
    result = _CELL_RACE.race_many(_SCORE_CURRENTS[np.asarray(scores, dtype=np.intp)][None, :])[0]
    if result.simultaneous_spike_candidates != 1:
        raise RuntimeError("variability boundary STOP race changed")
    return _trace(
        population=population,
        scores=scores,
        spike_step=result.spike_step,
        simultaneous_spike_cells=result.simultaneous_spike_candidates,
        stopped=True,
        stop_reason=reason,
        physical_cell_steps=_RACE_STEPS,
    )


def _coincident_source_indices(
    query_events: tuple[bytes, ...],
    source_events: tuple[bytes, ...],
) -> tuple[int, ...]:
    query_identities = frozenset(query_events)
    return tuple(
        index
        for index, source_event in enumerate(source_events)
        if source_event in query_identities
    )


def _nearest_recurring_event(
    source: tuple[bytes, ...],
    role: int,
    direction: int,
    ignored: frozenset[bytes],
) -> tuple[bytes, int]:
    counts = Counter(source)
    for hop in range(1, _MAX_BOUNDARY_HOPS + 1):
        index = role + direction * hop
        if not 0 <= index < len(source):
            break
        event = source[index]
        if counts[event] >= 2 and event not in ignored:
            return event, hop
    side = "left" if direction < 0 else "right"
    raise ValueError(f"demonstration lacks a nearest recurring {side} boundary")


class VariabilityAwareSegmentBinder:
    """Learn and execute one variability-aware boundary-relative relation."""

    __slots__ = ("_checkpoint_sha256", "_schema", "_summary")

    def __init__(
        self,
        token: object,
        *,
        schema: _LearnedSchema,
        summary: VariabilityBinderTrainingSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use VariabilityAwareSegmentBinder.grow")
        self._schema = schema
        self._summary = summary
        self._checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def grow(
        cls,
        demonstrations: Iterable[VariabilitySegmentDemonstration],
    ) -> VariabilityAwareSegmentBinder:
        """Learn stable and variable recurrent hops from independent streams."""

        rows = tuple(demonstrations)
        if len(rows) < 2:
            raise ValueError("at least two independent demonstrations are required")
        if len(rows) > _MAX_DEMONSTRATIONS:
            raise ValueError("demonstrations exceed the training bound")
        if any(not isinstance(row, VariabilitySegmentDemonstration) for row in rows):
            raise TypeError("demonstrations must contain VariabilitySegmentDemonstration")
        if len(set(rows)) != len(rows):
            raise ValueError("demonstrations must be independent, not duplicate replays")

        seeds: list[int] = []
        roles: list[int] = []
        offsets: list[int] = []
        boundary_rows: list[tuple[tuple[bytes, int], tuple[bytes, int]]] = []
        for row in rows:
            coincidences = _coincident_source_indices(row.query_events, row.source_events)
            if len(coincidences) != 1:
                raise ValueError("each training stream must have one query/source coincidence")
            seed = coincidences[0]
            role = row.source_events.index(row.demonstrated_role)
            offset = role - seed
            if offset == 0:
                raise ValueError("demonstrated role must differ from the equality seed")
            if abs(offset) > _MAX_PROPAGATION_HOPS:
                raise ValueError("demonstrated role exceeds the propagation bound")
            direction = 1 if offset > 0 else -1
            path_events = tuple(
                row.source_events[seed + direction * hop] for hop in range(abs(offset) + 1)
            )
            ignored = frozenset((*row.query_events, *path_events))
            boundary_rows.append(
                (
                    _nearest_recurring_event(row.source_events, role, -1, ignored),
                    _nearest_recurring_event(row.source_events, role, 1, ignored),
                )
            )
            seeds.append(seed)
            roles.append(role)
            offsets.append(offset)

        seed_identities = {row.source_events[seed] for row, seed in zip(rows, seeds, strict=True)}
        role_identities = {row.source_events[role] for row, role in zip(rows, roles, strict=True)}
        if len(seed_identities) != len(rows) or len(role_identities) != len(rows):
            raise ValueError("independent demonstrations need distinct equality seeds and roles")
        if len(set(offsets)) != 1:
            raise ValueError("demonstrations do not share one recurrent role offset")

        role_offset = offsets[0]
        direction = 1 if role_offset > 0 else -1
        propagation: list[_PropagationSynapse] = []
        for hop in range(1, abs(role_offset) + 1):
            identities = {
                row.source_events[seed + direction * hop]
                for row, seed in zip(rows, seeds, strict=True)
            }
            distinct_identity_count = len(identities)
            if hop == abs(role_offset):
                mode = "role"
                expected_event = None
            elif len(identities) == 1:
                mode = "stable"
                expected_event = next(iter(identities))
            else:
                mode = "wildcard"
                expected_event = None
            propagation.append(
                _PropagationSynapse(
                    hop=hop,
                    direction=direction,
                    mode=mode,
                    expected_event=expected_event,
                    distinct_identity_count=distinct_identity_count,
                    support_ticks=len(rows),
                )
            )

        boundaries: list[_BoundarySynapse] = []
        for boundary_ordinal, boundary_direction in enumerate((-1, 1)):
            identities = {pair[boundary_ordinal][0] for pair in boundary_rows}
            hops = {pair[boundary_ordinal][1] for pair in boundary_rows}
            if len(identities) != 1 or len(hops) != 1:
                raise ValueError("demonstrations do not share one fixed nearest boundary envelope")
            boundaries.append(
                _BoundarySynapse(
                    direction=boundary_direction,
                    event=identities.pop(),
                    hop=hops.pop(),
                    support_ticks=len(rows),
                )
            )

        schema = _LearnedSchema(
            role_offset=role_offset,
            propagation=tuple(propagation),
            boundaries=(boundaries[0], boundaries[1]),
        )
        checkpoint_record = {
            "boundaries": tuple(
                {
                    "direction": boundary.direction,
                    "event_hex": boundary.event.hex(),
                    "hop": boundary.hop,
                    "support_ticks": boundary.support_ticks,
                }
                for boundary in schema.boundaries
            ),
            "demonstration_count": len(rows),
            "propagation": tuple(
                {
                    "direction": synapse.direction,
                    "distinct_identity_count": synapse.distinct_identity_count,
                    "expected_event_hex": (
                        None if synapse.expected_event is None else synapse.expected_event.hex()
                    ),
                    "hop": synapse.hop,
                    "mode": synapse.mode,
                    "support_ticks": synapse.support_ticks,
                }
                for synapse in schema.propagation
            ),
            "role_offset": schema.role_offset,
        }
        checkpoint_sha256 = _digest(_CHECKPOINT_NAMESPACE, checkpoint_record)
        wildcards = tuple(synapse for synapse in propagation if synapse.mode == "wildcard")
        summary = VariabilityBinderTrainingSummary(
            demonstration_count=len(rows),
            role_offset=role_offset,
            propagation_hop_count=len(propagation),
            stable_intermediate_synapse_count=sum(
                synapse.mode == "stable" for synapse in propagation
            ),
            wildcard_intermediate_hop_count=len(wildcards),
            wildcard_support_ticks=sum(synapse.support_ticks for synapse in wildcards),
            boundary_synapse_count=len(boundaries),
            left_boundary_hop=boundaries[0].hop,
            right_boundary_hop=boundaries[1].hop,
            checkpoint_sha256=checkpoint_sha256,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            schema=schema,
            summary=summary,
            checkpoint_sha256=checkpoint_sha256,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> VariabilityBinderTrainingSummary:
        return self._summary

    def respond(
        self,
        query_events: tuple[bytes, ...],
        source_events: tuple[bytes, ...],
        *,
        ablation: VariabilityBinderAblation = VariabilityBinderAblation.NONE,
    ) -> VariabilityBindingDecision:
        """Select one live opaque role from a supported stream or physically STOP."""

        query = _require_events(query_events, "query_events", maximum=_MAX_QUERY_EVENTS)
        source = _require_events(source_events, "source_events", maximum=_MAX_SOURCE_EVENTS)
        if not isinstance(ablation, VariabilityBinderAblation):
            raise TypeError("ablation must be VariabilityBinderAblation")
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "query_events_hex": tuple(event.hex() for event in query),
                "source_events_hex": tuple(event.hex() for event in source),
            },
        )
        races: list[VariabilityRaceTrace] = []
        query_identities = frozenset(query)
        equality = _physical_choice(
            population="query_source_equality",
            scores=tuple(
                _DRIVE_SCORE
                if (
                    ablation is not VariabilityBinderAblation.EQUALITY_OFF
                    and event in query_identities
                )
                else 0
                for event in source
            ),
        )
        races.append(equality.trace)
        if equality.column is None:
            return self._stop(
                query_sha256,
                races,
                stage="equality",
                reason=equality.trace.stop_reason,
                boundary_match_count=0,
            )

        active = equality.column
        for synapse in self._schema.propagation:
            candidate = active + synapse.direction
            permitted = (
                ablation is not VariabilityBinderAblation.PROPAGATION_OFF
                and 0 <= candidate < len(source)
                and (synapse.mode != "stable" or source[candidate] == synapse.expected_event)
            )
            scores = [0] * len(source)
            if permitted:
                scores[candidate] = _DRIVE_SCORE
            propagation = _physical_choice(
                population=f"recurrent_{synapse.mode}_hop_{synapse.hop}",
                scores=tuple(scores),
            )
            races.append(propagation.trace)
            if propagation.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    stage="propagation",
                    reason=propagation.trace.stop_reason,
                    boundary_match_count=0,
                )
            active = propagation.column

        boundary_matches = 0
        for side, boundary in zip(("left", "right"), self._schema.boundaries, strict=True):
            found = False
            for hop in range(1, boundary.hop + 1):
                column = active + boundary.direction * hop
                matched = (
                    ablation is not VariabilityBinderAblation.BOUNDARY_OFF
                    and 0 <= column < len(source)
                    and source[column] == boundary.event
                )
                detected, trace = _physical_detector(
                    population=f"{side}_boundary_walk_hop_{hop}",
                    matched=matched,
                )
                races.append(trace)
                if not detected:
                    continue
                if hop != boundary.hop:
                    stop_trace = _physical_boundary_stop(
                        f"{side}_boundary_early_stop",
                        "early_boundary",
                    )
                    races.append(stop_trace)
                    return self._stop(
                        query_sha256,
                        races,
                        stage="boundary",
                        reason=f"{side}_{stop_trace.stop_reason}",
                        boundary_match_count=boundary_matches,
                    )
                found = True
                boundary_matches += 1
                break
            if not found:
                stop_trace = _physical_boundary_stop(
                    f"{side}_boundary_missing",
                    "missing_boundary",
                )
                races.append(stop_trace)
                return self._stop(
                    query_sha256,
                    races,
                    stage="boundary",
                    reason=f"{side}_{stop_trace.stop_reason}",
                    boundary_match_count=boundary_matches,
                )

        role_identity = source[active]
        identity = _physical_choice(
            population="role_identity_uniqueness",
            scores=tuple(_DRIVE_SCORE if event == role_identity else 0 for event in source),
        )
        races.append(identity.trace)
        if identity.column is None:
            return self._stop(
                query_sha256,
                races,
                stage="identity",
                reason=identity.trace.stop_reason,
                boundary_match_count=boundary_matches,
            )
        return self._decision(
            decision="BIND",
            role_identity=source[identity.column],
            stop_stage=None,
            stop_reason=None,
            boundary_match_count=boundary_matches,
            query_sha256=query_sha256,
            races=races,
        )

    def _stop(
        self,
        query_sha256: str,
        races: list[VariabilityRaceTrace],
        *,
        stage: str,
        reason: str | None,
        boundary_match_count: int,
    ) -> VariabilityBindingDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its race reason")
        return self._decision(
            decision="STOP",
            role_identity=None,
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            boundary_match_count=boundary_match_count,
            query_sha256=query_sha256,
            races=races,
        )

    def _decision(
        self,
        *,
        decision: str,
        role_identity: bytes | None,
        stop_stage: str | None,
        stop_reason: str | None,
        boundary_match_count: int,
        query_sha256: str,
        races: list[VariabilityRaceTrace],
    ) -> VariabilityBindingDecision:
        race_tuple = tuple(races)
        physical_cell_steps = sum(race.physical_cell_steps for race in race_tuple)
        record = {
            "boundary_match_count": boundary_match_count,
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "role_identity_hex": None if role_identity is None else role_identity.hex(),
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return VariabilityBindingDecision(
            decision=decision,
            role_identity=role_identity,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            boundary_match_count=boundary_match_count,
            races=race_tuple,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
