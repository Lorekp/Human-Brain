"""Neuronal role binding from a complete opaque source-token tape.

This module closes one deliberately narrow piece of micro-gate scaffolding: it
derives the ordered opaque cue identities consumed by the source-relative motor
tape without receiving a curated cue inventory.  Training demonstrations may
identify the role tokens that were used.  The learned checkpoint retains only
recurring, center-free relative-neighborhood synapses for each ordered role.

At inference the host presents the complete context stream and lossless source
token tape.  It mechanically activates learned context/neighborhood synapses
for every source occurrence.  Fitted GLIF first-spike populations select each
role occurrence, verify that its opaque identity is unique on the tape, or
physically select STOP.  The only content-bearing output is an ordered tuple of
opaque cue identities; source positions and occurrence ordinals never leave the
binder.

This is a role-induction micro-gate, not autonomous language understanding or
end-to-end software intelligence.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.occurrence_neighborhood_scorer import OccurrenceNeighborhoodScorer
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "NeuronalRawTapeRoleBinder",
    "RawTapeRoleDemonstration",
    "RoleBinderAblation",
    "RoleBinderConfig",
    "RoleBinderRaceTrace",
    "RoleBinderTrainingSummary",
    "RoleBindingDecision",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 32
_RELATION_SCORE = 2

_CHECKPOINT_NAMESPACE = b"humanbrain.raw-tape-role-binder.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.raw-tape-role-binder.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.raw-tape-role-binder.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.raw-tape-role-binder.decision-v1\0"
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


def _require_events(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for index, event in enumerate(value):
        _require_event(event, f"{name}[{index}]")
    return value


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
            raise RuntimeError("role-binder fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(row.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)):
        raise RuntimeError("role-binder fitted-GLIF latency calibration changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class RoleBinderAblation(StrEnum):
    NONE = "none"
    BINDER_OFF = "binder_off"


@dataclass(frozen=True, slots=True)
class RoleBinderConfig:
    neighborhood_radius: int = 2
    minimum_stable_synapses: int = 2
    max_demonstrations: int = 512
    max_contexts: int = 128
    max_roles: int = 64
    max_source_tokens: int = 4096
    max_context_events: int = 256
    max_event_bytes: int = 4096

    def __post_init__(self) -> None:
        integer_fields = (
            "neighborhood_radius",
            "minimum_stable_synapses",
            "max_demonstrations",
            "max_contexts",
            "max_roles",
            "max_source_tokens",
            "max_context_events",
            "max_event_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if 2 * self.neighborhood_radius > _SCORE_CAP:
            raise ValueError("neighborhood radius exceeds the calibrated score cap")
        if self.minimum_stable_synapses > 2 * self.neighborhood_radius:
            raise ValueError("minimum stable synapses exceeds the neighborhood capacity")


@dataclass(frozen=True, slots=True)
class RawTapeRoleDemonstration:
    """One training observation; ordered role identities are teacher-visible only."""

    context_event: bytes
    source_tokens: tuple[bytes, ...]
    demonstrated_roles: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_event(self.context_event, "context_event")
        source = _require_events(self.source_tokens, "source_tokens")
        roles = _require_events(self.demonstrated_roles, "demonstrated_roles")
        if not source:
            raise ValueError("source_tokens must not be empty")
        if not roles:
            raise ValueError("demonstrated_roles must not be empty")
        if len(set(roles)) != len(roles):
            raise ValueError("demonstrated_roles must contain unique opaque identities")
        counts = Counter(source)
        if any(counts[role] != 1 for role in roles):
            raise ValueError("each demonstrated role must have one unambiguous training occurrence")


@dataclass(frozen=True, slots=True)
class RoleBinderRaceTrace:
    population: str
    candidate_count: int
    candidate_scores_sha256: str
    tonic_stop_score: int
    spike_step: int
    simultaneous_spike_candidates: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RoleBinderTrainingSummary:
    demonstration_count: int
    context_count: int
    role_slot_count: int
    neighborhood_synapse_count: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RoleBindingDecision:
    decision: str
    cue_inventory: tuple[bytes, ...]
    stop_reason: str | None
    races: tuple[RoleBinderRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _NeighborhoodSynapse:
    relative_offset: int
    neighbor_event: bytes
    support_ticks: int


@dataclass(frozen=True, slots=True)
class _RoleSchema:
    context_event: bytes
    role_synapses: tuple[tuple[_NeighborhoodSynapse, ...], ...]


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: RoleBinderRaceTrace


def _neighborhood_features(
    source: tuple[bytes, ...],
    center: int,
    radius: int,
    *,
    excluded_events: frozenset[bytes] = frozenset(),
) -> frozenset[tuple[int, bytes]]:
    features: set[tuple[int, bytes]] = set()
    for relative_offset in range(-radius, radius + 1):
        if relative_offset == 0:
            continue
        neighbor = center + relative_offset
        if not 0 <= neighbor < len(source):
            continue
        event = source[neighbor]
        if event not in excluded_events:
            features.add((relative_offset, event))
    return frozenset(features)


def _physical_race(
    *,
    population: str,
    scores: tuple[int, ...],
    tonic_stop_score: int,
) -> _RaceDecision:
    if not 0 <= tonic_stop_score <= _SCORE_CAP:
        raise RuntimeError("tonic STOP score left its calibrated bounds")
    if any(not 0 <= score <= _SCORE_CAP for score in scores):
        raise RuntimeError("candidate score left its calibrated bounds")

    all_scores = (*scores, tonic_stop_score)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_scores) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    selected: int | None = None

    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_RELATION_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("role-binder collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(scores):
        stopped = True
        stop_reason = "tonic_stop"
    else:
        selected = result.winner_cell

    score_record = {
        "candidate_scores": scores,
        "tonic_stop_score": tonic_stop_score,
    }
    record = {
        "candidate_count": len(scores),
        "candidate_scores_sha256": _digest(_RACE_NAMESPACE, score_record),
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_candidates": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
        "tonic_stop_score": tonic_stop_score,
    }
    return _RaceDecision(
        column=selected,
        trace=RoleBinderRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record)),
    )


class NeuronalRawTapeRoleBinder:
    """Learn ordered opaque roles from recurring raw-tape neighborhoods."""

    __slots__ = ("_checkpoint_sha256", "_config", "_schemas", "_summary")

    def __init__(
        self,
        token: object,
        *,
        config: RoleBinderConfig,
        schemas: tuple[_RoleSchema, ...],
        checkpoint_sha256: str,
        summary: RoleBinderTrainingSummary,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use NeuronalRawTapeRoleBinder.grow")
        self._config = config
        self._schemas = schemas
        self._checkpoint_sha256 = checkpoint_sha256
        self._summary = summary

    @classmethod
    def grow(
        cls,
        demonstrations: Iterable[RawTapeRoleDemonstration],
        config: RoleBinderConfig | None = None,
    ) -> NeuronalRawTapeRoleBinder:
        """Consolidate recurring center-free neighborhood evidence into synapses."""

        settings = config or RoleBinderConfig()
        if not isinstance(settings, RoleBinderConfig):
            raise TypeError("config must be a RoleBinderConfig")
        demos = tuple(demonstrations)
        if not demos:
            raise ValueError("demonstrations must not be empty")
        if len(demos) > settings.max_demonstrations:
            raise ValueError("demonstrations exceed configured capacity")
        if any(not isinstance(demo, RawTapeRoleDemonstration) for demo in demos):
            raise TypeError("demonstrations must contain RawTapeRoleDemonstration values")

        grouped: dict[bytes, list[RawTapeRoleDemonstration]] = defaultdict(list)
        for demo in demos:
            if len(demo.source_tokens) > settings.max_source_tokens:
                raise ValueError("training source tape exceeds configured capacity")
            if len(demo.demonstrated_roles) > settings.max_roles:
                raise ValueError("demonstrated roles exceed configured capacity")
            for event in (demo.context_event, *demo.source_tokens):
                if len(event) > settings.max_event_bytes:
                    raise ValueError("opaque training event exceeds configured byte capacity")
            grouped[demo.context_event].append(demo)
        if len(grouped) > settings.max_contexts:
            raise ValueError("training contexts exceed configured capacity")

        schemas: list[_RoleSchema] = []
        for context_event in sorted(grouped):
            context_demos = grouped[context_event]
            role_count = len(context_demos[0].demonstrated_roles)
            if any(len(demo.demonstrated_roles) != role_count for demo in context_demos):
                raise ValueError("one context cannot demonstrate different role counts")

            role_synapses: list[tuple[_NeighborhoodSynapse, ...]] = []
            for role_slot in range(role_count):
                feature_sets: list[frozenset[tuple[int, bytes]]] = []
                for demo in context_demos:
                    center = demo.source_tokens.index(demo.demonstrated_roles[role_slot])
                    feature_sets.append(
                        _neighborhood_features(
                            demo.source_tokens,
                            center,
                            settings.neighborhood_radius,
                            excluded_events=frozenset(demo.demonstrated_roles),
                        )
                    )
                stable_features = set.intersection(*(set(item) for item in feature_sets))
                if len(stable_features) < settings.minimum_stable_synapses:
                    raise ValueError(
                        f"context role {role_slot} has insufficient recurring "
                        "relative-neighborhood evidence"
                    )
                synapses = tuple(
                    _NeighborhoodSynapse(offset, event, len(context_demos))
                    for offset, event in sorted(
                        stable_features,
                        key=lambda item: (item[0], item[1]),
                    )
                )
                role_synapses.append(synapses)
            schemas.append(_RoleSchema(context_event, tuple(role_synapses)))

        schema_tuple = tuple(schemas)
        checkpoint_record = {
            "config": asdict(settings),
            "schemas": tuple(
                {
                    "context_event_hex": schema.context_event.hex(),
                    "roles": tuple(
                        tuple(
                            {
                                "neighbor_event_hex": synapse.neighbor_event.hex(),
                                "relative_offset": synapse.relative_offset,
                                "support_ticks": synapse.support_ticks,
                            }
                            for synapse in role
                        )
                        for role in schema.role_synapses
                    ),
                }
                for schema in schema_tuple
            ),
        }
        checkpoint_sha256 = _digest(_CHECKPOINT_NAMESPACE, checkpoint_record)
        summary = RoleBinderTrainingSummary(
            demonstration_count=len(demos),
            context_count=len(schema_tuple),
            role_slot_count=sum(len(schema.role_synapses) for schema in schema_tuple),
            neighborhood_synapse_count=sum(
                len(role) for schema in schema_tuple for role in schema.role_synapses
            ),
            checkpoint_sha256=checkpoint_sha256,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            config=settings,
            schemas=schema_tuple,
            checkpoint_sha256=checkpoint_sha256,
            summary=summary,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RoleBinderTrainingSummary:
        return self._summary

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
        *,
        ablation: RoleBinderAblation = RoleBinderAblation.NONE,
    ) -> RoleBindingDecision:
        """Bind ordered cue identities from the complete raw source tape or STOP."""

        contexts = _require_events(context_events, "context_events")
        source = _require_events(source_tokens, "source_tokens")
        if not contexts:
            raise ValueError("context_events must not be empty")
        if not source:
            raise ValueError("source_tokens must not be empty")
        if len(contexts) > self._config.max_context_events:
            raise ValueError("context stream exceeds configured capacity")
        if len(source) > self._config.max_source_tokens:
            raise ValueError("source tape exceeds configured capacity")
        for event in (*contexts, *source):
            if len(event) > self._config.max_event_bytes:
                raise ValueError("opaque event exceeds configured byte capacity")
        if not isinstance(ablation, RoleBinderAblation):
            raise TypeError("ablation must be a RoleBinderAblation")

        query_record = {
            "ablation": ablation.value,
            "checkpoint_sha256": self._checkpoint_sha256,
            "context_events_hex": tuple(event.hex() for event in contexts),
            "source_tokens_hex": tuple(event.hex() for event in source),
        }
        query_sha256 = _digest(_QUERY_NAMESPACE, query_record)
        races: list[RoleBinderRaceTrace] = []
        context_set = frozenset(contexts)
        context_scores = tuple(
            _RELATION_SCORE
            if (ablation is RoleBinderAblation.NONE and schema.context_event in context_set)
            else 0
            for schema in self._schemas
        )
        context_race = _physical_race(
            population="context_schema",
            scores=context_scores,
            tonic_stop_score=1,
        )
        races.append(context_race.trace)
        if context_race.column is None:
            return self._stop(
                query_sha256,
                races,
                f"context_schema:{context_race.trace.stop_reason}",
            )

        schema = self._schemas[context_race.column]
        occurrence_scorer = OccurrenceNeighborhoodScorer(
            source,
            self._config.neighborhood_radius,
        )
        selected_identities: list[bytes] = []
        for role_slot, synapses in enumerate(schema.role_synapses):
            learned_features = frozenset(
                (synapse.relative_offset, synapse.neighbor_event) for synapse in synapses
            )
            occurrence_scores = occurrence_scorer.score(
                learned_features,
                excluded_center_events=frozenset(selected_identities),
            )
            role_race = _physical_race(
                population=f"role_{role_slot}_neighborhood",
                scores=occurrence_scores,
                tonic_stop_score=len(synapses) - 1,
            )
            races.append(role_race.trace)
            if role_race.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"role_{role_slot}:{role_race.trace.stop_reason}",
                )

            selected_identity = source[role_race.column]
            identity_scores = tuple(
                _RELATION_SCORE if event == selected_identity else 0 for event in source
            )
            identity_race = _physical_race(
                population=f"role_{role_slot}_identity_uniqueness",
                scores=identity_scores,
                tonic_stop_score=1,
            )
            races.append(identity_race.trace)
            if identity_race.column is None:
                return self._stop(
                    query_sha256,
                    races,
                    f"role_{role_slot}_identity:{identity_race.trace.stop_reason}",
                )
            selected_identities.append(selected_identity)

        return self._decision(
            decision="BIND",
            cues=tuple(selected_identities),
            stop_reason=None,
            query_sha256=query_sha256,
            races=races,
        )

    def _stop(
        self,
        query_sha256: str,
        races: list[RoleBinderRaceTrace],
        reason: str,
    ) -> RoleBindingDecision:
        return self._decision(
            decision="STOP",
            cues=(),
            stop_reason=reason,
            query_sha256=query_sha256,
            races=races,
        )

    def _decision(
        self,
        *,
        decision: str,
        cues: tuple[bytes, ...],
        stop_reason: str | None,
        query_sha256: str,
        races: list[RoleBinderRaceTrace],
    ) -> RoleBindingDecision:
        physical_cell_steps = sum(race.physical_cell_steps for race in races)
        race_tuple = tuple(races)
        record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "cue_inventory_hex": tuple(event.hex() for event in cues),
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "stop_reason": stop_reason,
        }
        return RoleBindingDecision(
            decision=decision,
            cue_inventory=cues,
            stop_reason=stop_reason,
            races=race_tuple,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
