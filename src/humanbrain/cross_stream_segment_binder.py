"""Neuronal equality-seeded relevance propagation over opaque event streams.

Training sees complete query/source sequences and the opaque role identity used
by a successful action.  It learns a recurrent offset path from the sole
query/source coincidence to that role plus stable relative containment
synapses.  No boundary identities, spans, positions, paths, or semantic labels
are inputs.

Inference activates one fitted-GLIF equality cell per source occurrence,
propagates the winner through learned recurrent hop synapses, verifies the
learned opaque containment envelope, and races the resulting role identity.
No match, any collision, or either causal ablation physically selects STOP.

This is a same-layout relational micro-mechanism, not language understanding or
an integrated source editor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "CrossStreamSegmentBinder",
    "SegmentBinderAblation",
    "SegmentBinderTrainingSummary",
    "SegmentBindingDecision",
    "SegmentRaceTrace",
    "SegmentRoleDemonstration",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_STOP_SCORE = 1
_CONTAINMENT_RADIUS = 5
_MIN_CONTAINMENT_SYNAPSES = 3
_MAX_DEMONSTRATIONS = 64
_MAX_QUERY_EVENTS = 256
_MAX_SOURCE_EVENTS = 4096
_MAX_EVENT_BYTES = 4096
_MAX_PROPAGATION_HOPS = 16

_CHECKPOINT_NAMESPACE = b"humanbrain.cross-stream-segment-binder.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.cross-stream-segment-binder.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.cross-stream-segment-binder.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.cross-stream-segment-binder.decision-v1\0"
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


def _require_events(
    value: object,
    name: str,
    *,
    maximum: int,
) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its event bound")
    for index, event in enumerate(value):
        _require_event(event, f"{name}[{index}]")
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
            raise RuntimeError("segment-binder fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, result in enumerate(controls)
    ):
        raise RuntimeError("segment-binder fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class SegmentBinderAblation(StrEnum):
    NONE = "none"
    EQUALITY_OFF = "equality_off"
    PROPAGATION_OFF = "propagation_off"


@dataclass(frozen=True, slots=True)
class SegmentRoleDemonstration:
    """One full-stream success; only the used opaque role is demonstrated."""

    query_events: tuple[bytes, ...]
    source_events: tuple[bytes, ...]
    demonstrated_role: bytes

    def __post_init__(self) -> None:
        _require_events(
            self.query_events,
            "query_events",
            maximum=_MAX_QUERY_EVENTS,
        )
        source = _require_events(
            self.source_events,
            "source_events",
            maximum=_MAX_SOURCE_EVENTS,
        )
        role = _require_event(self.demonstrated_role, "demonstrated_role")
        if source.count(role) != 1:
            raise ValueError("demonstrated_role must have one live source occurrence")


@dataclass(frozen=True, slots=True)
class SegmentRaceTrace:
    population: str
    candidate_count: int
    simultaneous_spike_candidates: int
    spike_step: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    candidate_scores_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SegmentBinderTrainingSummary:
    demonstration_count: int
    role_offset: int
    propagation_hop_count: int
    intermediate_event_synapse_count: int
    containment_synapse_count: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class SegmentBindingDecision:
    decision: str
    role_identity: bytes | None
    stop_stage: str | None
    stop_reason: str | None
    containment_match_count: int
    races: tuple[SegmentRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _PropagationSynapse:
    hop: int
    direction: int
    expected_event: bytes | None
    support_ticks: int


@dataclass(frozen=True, slots=True)
class _ContainmentSynapse:
    relative_to_role: int
    event: bytes
    support_ticks: int


@dataclass(frozen=True, slots=True)
class _LearnedSchema:
    role_offset: int
    propagation: tuple[_PropagationSynapse, ...]
    containment: tuple[_ContainmentSynapse, ...]


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: SegmentRaceTrace


def _physical_race(
    *,
    population: str,
    scores: tuple[int, ...],
) -> _RaceDecision:
    if not scores or any(score not in (0, _DRIVE_SCORE) for score in scores):
        raise ValueError("race scores must be a non-empty binary drive vector")
    all_scores = (*scores, _TONIC_STOP_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_scores) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    selected: int | None = None

    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_DRIVE_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("segment-binder collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(scores):
        stopped = True
        stop_reason = "tonic_stop"
    else:
        selected = result.winner_cell

    score_record = {"candidate_scores": scores}
    record = {
        "candidate_count": len(scores),
        "candidate_scores_sha256": _digest(_RACE_NAMESPACE, score_record),
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_candidates": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
    }
    return _RaceDecision(
        column=selected,
        trace=SegmentRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record)),
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


class CrossStreamSegmentBinder:
    """Learn and execute one equality-seeded opaque segment-role relation."""

    __slots__ = ("_checkpoint_sha256", "_schema", "_summary")

    def __init__(
        self,
        token: object,
        *,
        schema: _LearnedSchema,
        summary: SegmentBinderTrainingSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use CrossStreamSegmentBinder.grow")
        self._schema = schema
        self._summary = summary
        self._checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def grow(
        cls,
        demonstrations: Iterable[SegmentRoleDemonstration],
    ) -> CrossStreamSegmentBinder:
        """Learn recurrence and containment from repeated full opaque streams."""

        rows = tuple(demonstrations)
        if not rows:
            raise ValueError("demonstrations must not be empty")
        if len(rows) > _MAX_DEMONSTRATIONS:
            raise ValueError("demonstrations exceed the training bound")
        if any(not isinstance(row, SegmentRoleDemonstration) for row in rows):
            raise TypeError("demonstrations must contain SegmentRoleDemonstration")

        seeds: list[int] = []
        roles: list[int] = []
        offsets: list[int] = []
        for row in rows:
            coincidences = _coincident_source_indices(
                row.query_events,
                row.source_events,
            )
            if len(coincidences) != 1:
                raise ValueError("each training stream must have one query/source coincidence")
            seed = coincidences[0]
            role = row.source_events.index(row.demonstrated_role)
            offset = role - seed
            if offset == 0:
                raise ValueError("demonstrated role must differ from the equality seed")
            if abs(offset) > _MAX_PROPAGATION_HOPS:
                raise ValueError("demonstrated role exceeds the propagation bound")
            seeds.append(seed)
            roles.append(role)
            offsets.append(offset)
        if len(set(offsets)) != 1:
            raise ValueError("demonstrations do not share one recurrent role offset")

        role_offset = offsets[0]
        direction = 1 if role_offset > 0 else -1
        hop_count = abs(role_offset)
        propagation: list[_PropagationSynapse] = []
        for hop in range(1, hop_count + 1):
            expected_event: bytes | None = None
            if hop != hop_count:
                events = {
                    row.source_events[seed + direction * hop]
                    for row, seed in zip(rows, seeds, strict=True)
                }
                if len(events) != 1:
                    raise ValueError("demonstrations lack a stable opaque propagation synapse")
                expected_event = events.pop()
            propagation.append(
                _PropagationSynapse(
                    hop=hop,
                    direction=direction,
                    expected_event=expected_event,
                    support_ticks=len(rows),
                )
            )

        feature_sets: list[set[tuple[int, bytes]]] = []
        for row, seed, role in zip(rows, seeds, roles, strict=True):
            seed_relative_to_role = seed - role
            features: set[tuple[int, bytes]] = set()
            for relative in range(-_CONTAINMENT_RADIUS, _CONTAINMENT_RADIUS + 1):
                position = role + relative
                if relative not in (0, seed_relative_to_role) and 0 <= position < len(
                    row.source_events
                ):
                    features.add((relative, row.source_events[position]))
            feature_sets.append(features)
        stable_features = set.intersection(*feature_sets)
        if len(stable_features) < _MIN_CONTAINMENT_SYNAPSES:
            raise ValueError("demonstrations lack a stable opaque containment envelope")
        containment = tuple(
            _ContainmentSynapse(relative, event, len(rows))
            for relative, event in sorted(
                stable_features,
                key=lambda item: (item[0], item[1]),
            )
        )
        schema = _LearnedSchema(role_offset, tuple(propagation), containment)
        checkpoint_record = {
            "containment": tuple(
                {
                    "event_hex": synapse.event.hex(),
                    "relative_to_role": synapse.relative_to_role,
                    "support_ticks": synapse.support_ticks,
                }
                for synapse in containment
            ),
            "demonstration_count": len(rows),
            "propagation": tuple(
                {
                    "direction": synapse.direction,
                    "expected_event_hex": (
                        None if synapse.expected_event is None else synapse.expected_event.hex()
                    ),
                    "hop": synapse.hop,
                    "support_ticks": synapse.support_ticks,
                }
                for synapse in propagation
            ),
            "role_offset": role_offset,
        }
        checkpoint_sha256 = _digest(_CHECKPOINT_NAMESPACE, checkpoint_record)
        summary = SegmentBinderTrainingSummary(
            demonstration_count=len(rows),
            role_offset=role_offset,
            propagation_hop_count=hop_count,
            intermediate_event_synapse_count=sum(
                synapse.expected_event is not None for synapse in propagation
            ),
            containment_synapse_count=len(containment),
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
    def training_summary(self) -> SegmentBinderTrainingSummary:
        return self._summary

    def respond(
        self,
        query_events: tuple[bytes, ...],
        source_events: tuple[bytes, ...],
        *,
        ablation: SegmentBinderAblation = SegmentBinderAblation.NONE,
    ) -> SegmentBindingDecision:
        """Select one live opaque role from two complete event streams or STOP."""

        query = _require_events(
            query_events,
            "query_events",
            maximum=_MAX_QUERY_EVENTS,
        )
        source = _require_events(
            source_events,
            "source_events",
            maximum=_MAX_SOURCE_EVENTS,
        )
        if not isinstance(ablation, SegmentBinderAblation):
            raise TypeError("ablation must be a SegmentBinderAblation")
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "query_events_hex": tuple(event.hex() for event in query),
                "source_events_hex": tuple(event.hex() for event in source),
            },
        )
        races: list[SegmentRaceTrace] = []
        query_identities = frozenset(query)
        equality_scores = tuple(
            _DRIVE_SCORE
            if (ablation is not SegmentBinderAblation.EQUALITY_OFF and event in query_identities)
            else 0
            for event in source
        )
        equality = _physical_race(
            population="query_source_equality",
            scores=equality_scores,
        )
        races.append(equality.trace)
        if equality.column is None:
            return self._stop(
                query_sha256,
                races,
                stage="equality",
                reason=equality.trace.stop_reason,
            )

        active = equality.column
        for synapse in self._schema.propagation:
            candidate = active + synapse.direction
            propagation_scores = [0] * len(source)
            if (
                ablation is not SegmentBinderAblation.PROPAGATION_OFF
                and 0 <= candidate < len(source)
                and (synapse.expected_event is None or source[candidate] == synapse.expected_event)
            ):
                propagation_scores[candidate] = _DRIVE_SCORE
            propagation = _physical_race(
                population=f"recurrent_propagation_hop_{synapse.hop}",
                scores=tuple(propagation_scores),
            )
            races.append(propagation.trace)
            if propagation.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    stage="propagation",
                    reason=propagation.trace.stop_reason,
                )
            active = propagation.column

        containment_matches = sum(
            1
            for synapse in self._schema.containment
            if 0 <= active + synapse.relative_to_role < len(source)
            and source[active + synapse.relative_to_role] == synapse.event
        )
        role_scores = [0] * len(source)
        if containment_matches == len(self._schema.containment):
            role_scores[active] = _DRIVE_SCORE
        role_race = _physical_race(
            population="segment_containment_role",
            scores=tuple(role_scores),
        )
        races.append(role_race.trace)
        if role_race.column is None:
            return self._stop(
                query_sha256,
                races,
                stage="containment",
                reason=role_race.trace.stop_reason,
                containment_match_count=containment_matches,
            )

        role_identity = source[role_race.column]
        identity_race = _physical_race(
            population="role_identity_uniqueness",
            scores=tuple(_DRIVE_SCORE if event == role_identity else 0 for event in source),
        )
        races.append(identity_race.trace)
        if identity_race.column is None:
            return self._stop(
                query_sha256,
                races,
                stage="identity",
                reason=identity_race.trace.stop_reason,
                containment_match_count=containment_matches,
            )
        return self._decision(
            decision="BIND",
            role_identity=role_identity,
            stop_stage=None,
            stop_reason=None,
            containment_match_count=containment_matches,
            query_sha256=query_sha256,
            races=races,
        )

    def _stop(
        self,
        query_sha256: str,
        races: list[SegmentRaceTrace],
        *,
        stage: str,
        reason: str | None,
        containment_match_count: int = 0,
    ) -> SegmentBindingDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its race reason")
        return self._decision(
            decision="STOP",
            role_identity=None,
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            containment_match_count=containment_match_count,
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
        containment_match_count: int,
        query_sha256: str,
        races: list[SegmentRaceTrace],
    ) -> SegmentBindingDecision:
        race_tuple = tuple(races)
        physical_cell_steps = sum(race.physical_cell_steps for race in race_tuple)
        record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "containment_match_count": containment_match_count,
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "role_identity_hex": None if role_identity is None else role_identity.hex(),
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return SegmentBindingDecision(
            decision=decision,
            role_identity=role_identity,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            containment_match_count=containment_match_count,
            races=race_tuple,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
