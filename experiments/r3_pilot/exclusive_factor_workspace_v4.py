"""Exclusive prompt-factor workspace V4 learner.

The public seam is deliberately narrow: build one immutable bridge from the
frozen protocol, then resolve authenticated span batches through that bridge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as spec
from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as v2,
)
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

TICK_DTYPE = np.dtype("uint8")
_CHECKPOINT_NAMESPACE = b"exclusive-factor-workspace-v4-checkpoint\0"
_ELECTRICAL_NAMESPACE = b"exclusive-factor-workspace-v4-electrical-preflight\0"
_PRINCIPAL_TRACE_NAMESPACE = b"exclusive-factor-workspace-v4-principal-trace\0"
_RELAY_TRACE_NAMESPACE = b"exclusive-factor-workspace-v4-relay-trace\0"
_PROJECTION_PANEL_NAMESPACE = b"exclusive-factor-workspace-v4-projection-panel\0"
_FACTOR_REGISTRY_NAMESPACE = b"exclusive-factor-workspace-v4-factor-registry\0"
_ATOMIC_RESULT_NAMESPACE = b"exclusive-factor-workspace-v4-atomic-result\0"
_FACTOR_INDEX = {factor: index for index, factor in enumerate(spec.FACTOR_REGISTRY)}
_COMPILED_RACE = CompiledRestingGlifRace.compile(step_count=spec.RACE_STEPS)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _authenticate_protocol(protocol: spec.FrozenExclusiveFactorProtocol) -> None:
    if not isinstance(protocol, spec.FrozenExclusiveFactorProtocol):
        raise TypeError("build requires FrozenExclusiveFactorProtocol")
    rebuilt, local_rule_record, local_rule_sha = spec._build_protocol()
    if (
        protocol != rebuilt
        or local_rule_sha != spec.LOCAL_RULE_SHA256
        or local_rule_record != spec.LOCAL_RULE_RECORD
        or protocol.protocol_sha256 != spec.PROTOCOL.protocol_sha256
    ):
        raise RuntimeError("exclusive V4 protocol authentication failed")


@dataclass(frozen=True, slots=True)
class ElectricalControlReplay:
    name: str
    principal_first_spike_step: int
    principal_first_spike_tier: tuple[int, ...]
    simultaneous_spike_candidates: int
    basket_inhibition_step: int
    collision_detection_step: int | None
    decision_step: int
    decision: str
    factor: str | None
    principal_trace_sha256: str
    relay_trace_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "principal_first_spike_step": self.principal_first_spike_step,
            "principal_first_spike_tier": self.principal_first_spike_tier,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
            "basket_inhibition_step": self.basket_inhibition_step,
            "collision_detection_step": self.collision_detection_step,
            "decision_step": self.decision_step,
            "decision": self.decision,
            "factor": self.factor,
            "principal_trace_sha256": self.principal_trace_sha256,
            "relay_trace_sha256": self.relay_trace_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SpanRaceDecision:
    source_sha256: str
    expected_factor: str | None
    decision: str
    factor: str | None
    principal_first_spike_step: int
    decision_step: int
    simultaneous_spike_candidates: int
    unique_winner_cell: int | None
    raw_factor_scores: tuple[int, ...]
    relay_trace_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "expected_factor": self.expected_factor,
            "decision": self.decision,
            "factor": self.factor,
            "principal_first_spike_step": self.principal_first_spike_step,
            "decision_step": self.decision_step,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
            "unique_winner_cell": self.unique_winner_cell,
            "raw_factor_scores": self.raw_factor_scores,
            "relay_trace_sha256": self.relay_trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedSpanBatch:
    spans: tuple[spec.SpanPlacement, ...]
    protocol_sha256: str
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.spans) is not tuple or not all(
            isinstance(span, spec.SpanPlacement) for span in self.spans
        ):
            raise TypeError("authenticated batch requires frozen span placements")
        if type(self.protocol_sha256) is not str:
            raise TypeError("authenticated batch requires a protocol SHA-256")
        record = {
            "protocol_sha256": self.protocol_sha256,
            "spans": tuple(span.record() for span in self.spans),
        }
        object.__setattr__(
            self,
            "authentication_sha256",
            _namespaced_sha256(
                b"exclusive-factor-workspace-v4-authenticated-span-batch\0",
                record,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExclusiveFactorResolution:
    status: str
    events: tuple[object, ...]
    span_decisions: tuple[SpanRaceDecision, ...]
    checkpoint_authentications: int
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class BridgeBuildSummary:
    status: str
    electrical_control_count: int
    electrical_pass_count: int
    electrical_preflight_passed: bool
    electrical_preflight_sha256: str
    electrical_circuit_off_decision_count: int
    electrical_tamper_rejected: bool
    factor_registry_sha256: str
    projection_panel_sha256: str
    epochs: int
    fit_lesson_count: int
    heldout_lesson_count: int
    atomic_lesson_count: int
    untrained_atomic_pass_count: int
    cell_off_atomic_pass_count: int
    atomic_pass_count: int
    atomic_mismatch_count: int
    fit_atomic_pass_count: int
    heldout_atomic_pass_count: int
    atomic_checkpoint_authentications: int
    composition_open: bool
    composition_attempt_count: int
    duplicate_control_attempt_count: int
    overlap_control_attempt_count: int
    capacity_control_attempt_count: int
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    excitatory_head_update_events: int
    inhibitory_head_update_events: int
    committed_cell_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    atomic_result_sha256: str


def _score_currents(
    protocol: spec.FrozenExclusiveFactorProtocol,
    raw_scores: tuple[int, ...],
) -> np.ndarray:
    current_scores = tuple(spec._current_score(score) for score in raw_scores)
    return np.asarray(
        [protocol.latency_table[score].current_pa for score in current_scores],
        dtype=np.float64,
    )


def _live_principal_tier(
    currents: np.ndarray,
    expected_step: int,
) -> tuple[tuple[int, ...], str]:
    population = AdaptiveGlifPopulation(spec.FACTOR_COUNT + 1)
    leading: tuple[int, ...] = ()
    leading_step = -1
    observed: list[dict[str, object]] = []
    final_step = min(spec.RACE_STEPS - 1, expected_step + spec.FACTOR_RELAY_DELAY_TICKS)
    for step in range(final_step + 1):
        step_currents = currents.copy()
        if leading and step >= leading_step + spec.BASKET_DELAY_TICKS:
            step_currents += spec.BASKET_INHIBITION_CURRENT_PA
        fired = tuple(int(cell) for cell in np.flatnonzero(population.step(step_currents)))
        observed.append({"step": step, "spike_cells": fired})
        if fired and not leading:
            leading = fired
            leading_step = step
        elif fired:
            raise RuntimeError("basket failed to suppress a trailing principal spike")
    if leading_step != expected_step or not leading:
        raise RuntimeError("live principal population changed its first-spike tier")
    trace_sha = _namespaced_sha256(
        _PRINCIPAL_TRACE_NAMESPACE,
        {
            "base_currents_pa_hex": tuple(float(value).hex() for value in currents),
            "observed": tuple(observed),
        },
    )
    return leading, trace_sha


class _DiscreteNeuronPopulation:
    """Custom current-integrating neurons for the frozen discrete relay circuit."""

    __slots__ = ("_membrane", "_threshold_pa")

    def __init__(self, size: int, *, threshold_pa: float) -> None:
        if type(size) is not int or size < 1 or threshold_pa <= 0.0:
            raise ValueError("discrete population requires positive size/threshold")
        self._membrane = np.zeros(size, dtype=np.float64)
        self._threshold_pa = float(threshold_pa)

    def step(self, currents_pa: np.ndarray) -> np.ndarray:
        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != self._membrane.shape or not np.all(np.isfinite(currents)):
            raise ValueError("discrete neuron current shape changed")
        self._membrane += currents
        fired = self._membrane >= self._threshold_pa
        self._membrane[fired] = 0.0
        return fired


def _run_relay_circuit(
    principal_step: int,
    tier: tuple[int, ...],
    *,
    synaptic_gain: float = 1.0,
    collision_enabled: bool = True,
) -> tuple[int | None, int | None, int | None, str | None, str | None, str]:
    if type(tier) is not tuple or not tier:
        raise ValueError("relay circuit requires a nonempty principal spike tier")
    relay_current = spec.RELAY_COPY_CURRENT_PA * synaptic_gain
    basket = _DiscreteNeuronPopulation(1, threshold_pa=spec.RELAY_COPY_CURRENT_PA)
    collision = _DiscreteNeuronPopulation(
        1,
        threshold_pa=2.0 * spec.RELAY_COPY_CURRENT_PA,
    )
    factor_relays = _DiscreteNeuronPopulation(
        spec.FACTOR_COUNT,
        threshold_pa=spec.RELAY_COPY_CURRENT_PA,
    )
    direct_stop_relay = _DiscreteNeuronPopulation(
        1,
        threshold_pa=spec.RELAY_COPY_CURRENT_PA,
    )
    collision_stop_relay = _DiscreteNeuronPopulation(
        1,
        threshold_pa=spec.RELAY_COPY_CURRENT_PA,
    )

    basket_step: int | None = None
    collision_step: int | None = None
    decision_step: int | None = None
    decision: str | None = None
    factor: str | None = None
    collision_latched = False
    observed: list[dict[str, object]] = []
    for relative_tick in range(1, spec.FACTOR_RELAY_DELAY_TICKS + 1):
        basket_input = np.asarray(
            [relay_current * len(tier) if relative_tick == spec.BASKET_DELAY_TICKS else 0.0]
        )
        if bool(basket.step(basket_input)[0]):
            basket_step = principal_step + relative_tick

        collision_input = np.asarray(
            [
                relay_current * len(tier)
                if collision_enabled
                and relative_tick == spec.COLLISION_DETECTOR_DELAY_TICKS
                else 0.0
            ]
        )
        if bool(collision.step(collision_input)[0]):
            collision_latched = True
            collision_step = principal_step + relative_tick

        factor_currents = np.zeros(spec.FACTOR_COUNT, dtype=np.float64)
        if relative_tick == spec.FACTOR_RELAY_DELAY_TICKS:
            for cell in tier:
                if cell < spec.FACTOR_COUNT:
                    factor_currents[cell] += relay_current
            if collision_latched:
                factor_currents += spec.BASKET_INHIBITION_CURRENT_PA
        factor_spikes = tuple(
            int(cell) for cell in np.flatnonzero(factor_relays.step(factor_currents))
        )

        direct_stop_current = 0.0
        if (
            relative_tick == spec.STOP_RELAY_DELAY_TICKS
            and spec.FACTOR_COUNT in tier
        ):
            direct_stop_current += relay_current
            if collision_latched:
                direct_stop_current += spec.BASKET_INHIBITION_CURRENT_PA
        direct_stop_spike = bool(
            direct_stop_relay.step(np.asarray([direct_stop_current]))[0]
        )

        collision_stop_current = (
            relay_current
            if collision_latched
            and relative_tick == spec.COLLISION_STOP_DELAY_TICKS
            else 0.0
        )
        collision_stop_spike = bool(
            collision_stop_relay.step(np.asarray([collision_stop_current]))[0]
        )
        stop_spike = direct_stop_spike or collision_stop_spike
        observed.append(
            {
                "relative_tick": relative_tick,
                "basket_spike": basket_step == principal_step + relative_tick,
                "collision_spike": collision_step
                == principal_step + relative_tick,
                "factor_relay_spikes": factor_spikes,
                "direct_stop_relay_spike": direct_stop_spike,
                "collision_stop_relay_spike": collision_stop_spike,
            }
        )

        if factor_spikes or stop_spike:
            if decision is not None:
                raise RuntimeError("relay circuit produced multiple decision ticks")
            if stop_spike and not factor_spikes:
                decision = "STOP"
                decision_step = principal_step + relative_tick
            elif len(factor_spikes) == 1 and not stop_spike:
                decision = "FACTOR"
                factor = spec.FACTOR_REGISTRY[factor_spikes[0]]
                decision_step = principal_step + relative_tick
            else:
                raise RuntimeError("relay circuit decision was not one-hot")
    relay_trace_sha = _namespaced_sha256(
        _RELAY_TRACE_NAMESPACE,
        {
            "principal_step": principal_step,
            "principal_tier": tier,
            "synaptic_gain_hex": float(synaptic_gain).hex(),
            "collision_enabled": collision_enabled,
            "observed": tuple(observed),
        },
    )
    return (
        basket_step,
        collision_step,
        decision_step,
        decision,
        factor,
        relay_trace_sha,
    )


def _execute_electrical_preflight(
    protocol: spec.FrozenExclusiveFactorProtocol,
) -> tuple[tuple[ElectricalControlReplay, ...], str, int, bool]:
    controls = protocol.wta_electrical_controls
    current_rows = np.stack(
        [
            _score_currents(protocol, control.principal_raw_scores)
            for control in controls
        ]
    )
    races = _COMPILED_RACE.race_many(
        current_rows,
        require_unique=False,
    )
    replays: list[ElectricalControlReplay] = []
    for control, currents, race in zip(controls, current_rows, races, strict=True):
        tier, principal_trace_sha = _live_principal_tier(currents, race.spike_step)
        if (
            tier != control.principal_first_spike_tier
            or race.simultaneous_spike_candidates != len(tier)
        ):
            raise RuntimeError("electrical principal control changed")
        basket, collision, decision_step, decision, factor, relay_trace_sha = (
            _run_relay_circuit(race.spike_step, tier)
        )
        if (
            basket != control.basket_inhibition_step
            or collision != control.collision_detection_step
            or decision_step != control.expected_decision_step
            or decision != control.expected_decision
            or factor != control.expected_factor
        ):
            raise RuntimeError("electrical basket/collision/relay control changed")
        record = {
            "name": control.name,
            "principal_first_spike_step": race.spike_step,
            "principal_first_spike_tier": tier,
            "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
            "basket_inhibition_step": basket,
            "collision_detection_step": collision,
            "decision_step": decision_step,
            "decision": decision,
            "factor": factor,
            "principal_trace_sha256": principal_trace_sha,
            "relay_trace_sha256": relay_trace_sha,
        }
        replays.append(
            ElectricalControlReplay(
                **record,
                sha256=_namespaced_sha256(_ELECTRICAL_NAMESPACE, record),
            )
        )
    replay_tuple = tuple(replays)
    circuit_off_decisions = sum(
        _run_relay_circuit(
            replay.principal_first_spike_step,
            replay.principal_first_spike_tier,
            synaptic_gain=0.0,
        )[3]
        is not None
        for replay in replay_tuple
    )
    tamper_rejected = False
    tie_replay = replay_tuple[2]
    try:
        tampered = _run_relay_circuit(
            tie_replay.principal_first_spike_step,
            tie_replay.principal_first_spike_tier,
            collision_enabled=False,
        )
        tamper_rejected = tampered[3:5] != (
            tie_replay.decision,
            tie_replay.factor,
        )
    except RuntimeError:
        tamper_rejected = True
    if circuit_off_decisions != 0 or not tamper_rejected:
        raise RuntimeError("electrical causal controls failed")
    root = _namespaced_sha256(
        _ELECTRICAL_NAMESPACE,
        {
            "replays": tuple(replay.record() for replay in replay_tuple),
            "circuit_off_decision_count": circuit_off_decisions,
            "collision_disabled_tamper_rejected": tamper_rejected,
        },
    )
    return replay_tuple, root, circuit_off_decisions, tamper_rejected


def _project_atomic_lessons(
    protocol: spec.FrozenExclusiveFactorProtocol,
) -> dict[str, object]:
    return {
        lesson.sha256: v2._project_tokens(lesson.tokens)
        for lesson in protocol.lessons
    }


def _projection_panel_sha256(
    protocol: spec.FrozenExclusiveFactorProtocol,
    projections: dict[str, object],
) -> str:
    rows = []
    for lesson in protocol.lessons:
        projected = projections[lesson.sha256]
        rows.append(
            {
                "lesson_sha256": lesson.sha256,
                "canonical_tokens": projected.canonical_tokens,
                "active_global_cells": projected.active_global_cells,
                "physical_drive_sha256": projected.physical_drive_sha256,
                "bank_trace_sha256s": tuple(
                    drive.trace_sha256 for drive in projected.bank_drives
                ),
            }
        )
    return _namespaced_sha256(_PROJECTION_PANEL_NAMESPACE, tuple(rows))


def _raw_factor_scores(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    projected: object,
) -> tuple[int, ...]:
    active = np.asarray(projected.active_global_cells, dtype=np.intp)
    if active.size < 1:
        raise RuntimeError("exclusive V4 projection produced no active cells")
    raw = np.sum(
        excitatory_ticks[:, active].astype(np.int16)
        - inhibitory_ticks[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    if np.any(raw < spec.SCORE_MIN) or np.any(raw > spec.SCORE_MAX):
        raise RuntimeError("exclusive V4 raw score left the frozen domain")
    return tuple(int(value) for value in raw)


def _race_span_rows(
    protocol: spec.FrozenExclusiveFactorProtocol,
    raw_rows: tuple[tuple[int, ...], ...],
    source_sha256s: tuple[str, ...],
    expected_factors: tuple[str | None, ...],
) -> tuple[SpanRaceDecision, ...]:
    if not (
        len(raw_rows) == len(source_sha256s) == len(expected_factors) > 0
        and all(len(row) == spec.FACTOR_COUNT for row in raw_rows)
    ):
        raise ValueError("exclusive V4 span race rows changed shape")
    current_score_rows = tuple(
        (*map(spec._current_score, row), spec.STOP_SCORE)
        for row in raw_rows
    )
    current_rows = np.asarray(
        [
            [protocol.latency_table[score].current_pa for score in score_row]
            for score_row in current_score_rows
        ],
        dtype=np.float64,
    )
    races = _COMPILED_RACE.race_many(current_rows, require_unique=False)
    decisions: list[SpanRaceDecision] = []
    for raw, current_scores, _currents, source, expected, race in zip(
        raw_rows,
        current_score_rows,
        current_rows,
        source_sha256s,
        expected_factors,
        races,
        strict=True,
    ):
        tier = tuple(
            cell
            for cell, score in enumerate(current_scores)
            if protocol.latency_table[score].first_spike_step == race.spike_step
        )
        if len(tier) != race.simultaneous_spike_candidates:
            raise RuntimeError("exclusive V4 principal spike tier changed")
        basket, collision, decision_step, relay_decision, factor, relay_trace_sha = (
            _run_relay_circuit(race.spike_step, tier)
        )
        if basket != race.spike_step + spec.BASKET_DELAY_TICKS:
            raise RuntimeError("exclusive V4 basket timing changed")
        if len(tier) > 1:
            decision = "COLLISION"
            unique_winner = None
            if collision is None or relay_decision != "STOP":
                raise RuntimeError("exclusive V4 collision failed closed")
        else:
            decision = relay_decision
            unique_winner = race.winner_cell
            if unique_winner != tier[0]:
                raise RuntimeError("exclusive V4 unique winner changed")
        if decision_step is None or decision is None:
            raise RuntimeError("exclusive V4 relay circuit emitted no decision")
        decisions.append(
            SpanRaceDecision(
                source_sha256=source,
                expected_factor=expected,
                decision=decision,
                factor=factor,
                principal_first_spike_step=race.spike_step,
                decision_step=decision_step,
                simultaneous_spike_candidates=race.simultaneous_spike_candidates,
                unique_winner_cell=unique_winner,
                raw_factor_scores=raw,
                relay_trace_sha256=relay_trace_sha,
            )
        )
    return tuple(decisions)


def _increment_tick_row(
    ticks: np.ndarray,
    factor_index: int,
    active_cells: np.ndarray,
) -> int:
    before = ticks[factor_index, active_cells].astype(np.uint16)
    after = np.minimum(before + 1, spec.TICK_CAP).astype(TICK_DTYPE)
    ticks[factor_index, active_cells] = after
    return int(np.sum(after.astype(np.int16) - before.astype(np.int16)))


def _checkpoint_payload(
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> bytes:
    metadata_bytes = _canonical_json(metadata)
    return b"".join(
        (
            _CHECKPOINT_NAMESPACE,
            len(metadata_bytes).to_bytes(8, "little"),
            metadata_bytes,
            _array_frame("excitatory_ticks", excitatory_ticks),
            _array_frame("inhibitory_ticks", inhibitory_ticks),
        )
    )


def _authenticate_state(
    checkpoint_sha256: str,
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> None:
    if (
        sha256(
            _checkpoint_payload(metadata, excitatory_ticks, inhibitory_ticks)
        ).hexdigest()
        != checkpoint_sha256
    ):
        raise RuntimeError("frozen exclusive V4 checkpoint changed")


def _array_frame(name: str, values: np.ndarray) -> bytes:
    header = _canonical_json(
        {"name": name, "dtype": values.dtype.str, "shape": values.shape}
    )
    raw = np.ascontiguousarray(values, dtype=TICK_DTYPE).tobytes(order="C")
    return (
        len(header).to_bytes(8, "little")
        + header
        + len(raw).to_bytes(8, "little")
        + raw
    )


def _atomic_result_record(
    *,
    summary: BridgeBuildSummary,
    protocol_sha256: str,
    checkpoint_sha256: str,
    electrical_preflight_sha256: str,
    projection_panel_sha256: str,
    factor_registry_sha256: str,
    atomic: tuple[SpanRaceDecision, ...],
) -> dict[str, object]:
    mismatches = tuple(
        row.record()
        for row in atomic
        if row.decision != "FACTOR" or row.factor != row.expected_factor
    )
    return {
        "status": summary.status,
        "claim_boundary": "self-posed atomic-readiness STOP only",
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "electrical_preflight_sha256": electrical_preflight_sha256,
        "projection_panel_sha256": projection_panel_sha256,
        "factor_registry_sha256": factor_registry_sha256,
        "atomic_pass_count": summary.atomic_pass_count,
        "atomic_mismatch_count": summary.atomic_mismatch_count,
        "fit_atomic_pass_count": summary.fit_atomic_pass_count,
        "heldout_atomic_pass_count": summary.heldout_atomic_pass_count,
        "decisions": tuple(row.record() for row in atomic),
        "ordered_mismatches": mismatches,
        "composition_attempt_count": summary.composition_attempt_count,
        "composition_result_count": 0,
        "duplicate_control_attempt_count": summary.duplicate_control_attempt_count,
        "overlap_control_attempt_count": summary.overlap_control_attempt_count,
        "capacity_control_attempt_count": summary.capacity_control_attempt_count,
    }


class FrozenExclusiveFactorBridge:
    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory_ticks",
        "_inhibitory_ticks",
        "_metadata",
        "atomic_replay",
        "build_summary",
        "electrical_replays",
    )

    def __init__(
        self,
        *,
        excitatory_ticks: np.ndarray,
        inhibitory_ticks: np.ndarray,
        metadata: dict[str, object],
        build_summary: BridgeBuildSummary,
        electrical_replays: tuple[ElectricalControlReplay, ...],
        atomic_replay: tuple[SpanRaceDecision, ...],
    ) -> None:
        for name, values in (
            ("excitatory_ticks", excitatory_ticks),
            ("inhibitory_ticks", inhibitory_ticks),
        ):
            if values.shape != (spec.FACTOR_COUNT, spec.INPUT_CELL_COUNT):
                raise ValueError(f"{name} must be uint8[24,1536]")
            if values.dtype != TICK_DTYPE or np.any(values > spec.TICK_CAP):
                raise ValueError(f"{name} violates the frozen uint8 cap")
        self._excitatory_ticks = np.array(excitatory_ticks, copy=True, order="C")
        self._inhibitory_ticks = np.array(inhibitory_ticks, copy=True, order="C")
        self._excitatory_ticks.setflags(write=False)
        self._inhibitory_ticks.setflags(write=False)
        self._metadata = dict(metadata)
        self.build_summary = build_summary
        self.electrical_replays = electrical_replays
        self.atomic_replay = atomic_replay
        self._checkpoint_sha256 = sha256(self._checkpoint_bytes()).hexdigest()
        self._authenticate_terminal_result()

    @property
    def excitatory_ticks(self) -> np.ndarray:
        values = self._excitatory_ticks.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        values = self._inhibitory_ticks.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def learned_state_bytes(self) -> int:
        return self._excitatory_ticks.nbytes + self._inhibitory_ticks.nbytes

    @property
    def checkpoint_sha256(self) -> str:
        self._authenticate_checkpoint()
        return self._checkpoint_sha256

    def _checkpoint_bytes(self) -> bytes:
        return _checkpoint_payload(
            self._metadata,
            self._excitatory_ticks,
            self._inhibitory_ticks,
        )

    def _authenticate_checkpoint(self) -> None:
        if sha256(self._checkpoint_bytes()).hexdigest() != self._checkpoint_sha256:
            raise RuntimeError("frozen exclusive V4 checkpoint changed")

    def _authenticate_terminal_result(self) -> None:
        result_record = _atomic_result_record(
            summary=self.build_summary,
            protocol_sha256=str(self._metadata["protocol_sha256"]),
            checkpoint_sha256=self._checkpoint_sha256,
            electrical_preflight_sha256=str(
                self._metadata["electrical_preflight_sha256"]
            ),
            projection_panel_sha256=str(self._metadata["projection_panel_sha256"]),
            factor_registry_sha256=str(self._metadata["factor_registry_sha256"]),
            atomic=self.atomic_replay,
        )
        if (
            _namespaced_sha256(_ATOMIC_RESULT_NAMESPACE, result_record)
            != self.build_summary.atomic_result_sha256
        ):
            raise RuntimeError("frozen exclusive V4 terminal result changed")

    def resolve(
        self,
        batch: AuthenticatedSpanBatch,
    ) -> ExclusiveFactorResolution:
        del batch
        self._authenticate_checkpoint()
        self._authenticate_terminal_result()
        raise RuntimeError(
            "exclusive V4 bridge stopped before atomic readiness; "
            "composition is sealed"
        )


def build_frozen_exclusive_factor_bridge(
    protocol: spec.FrozenExclusiveFactorProtocol,
) -> FrozenExclusiveFactorBridge:
    _authenticate_protocol(protocol)
    (
        electrical_replays,
        electrical_sha,
        circuit_off_decisions,
        tamper_rejected,
    ) = _execute_electrical_preflight(protocol)
    excitatory = np.zeros(
        (spec.FACTOR_COUNT, spec.INPUT_CELL_COUNT),
        dtype=TICK_DTYPE,
    )
    inhibitory = np.zeros_like(excitatory)
    projections = _project_atomic_lessons(protocol)
    projection_panel_sha = _projection_panel_sha256(protocol, projections)
    factor_registry_sha = _namespaced_sha256(
        _FACTOR_REGISTRY_NAMESPACE,
        spec.FACTOR_REGISTRY,
    )
    lesson_by_sha = {lesson.sha256: lesson for lesson in protocol.lessons}
    all_lessons = protocol.lessons
    zero_rows = tuple((0,) * spec.FACTOR_COUNT for _lesson in all_lessons)
    source_ids = tuple(lesson.sha256 for lesson in all_lessons)
    expected = tuple(lesson.factor for lesson in all_lessons)
    untrained = _race_span_rows(
        protocol,
        zero_rows,
        source_ids,
        expected,
    )
    untrained_pass_count = sum(
        row.decision == "FACTOR" and row.factor == row.expected_factor
        for row in untrained
    )

    excitatory_updates = 0
    inhibitory_updates = 0
    committed_ticks = 0
    schedule = tuple(spec.LOCAL_RULE_RECORD["fit_schedule"]["lesson_sha256s"])
    for _epoch in range(spec.LOCAL_RULE_RECORD["training_epochs"]):
        for lesson_sha256 in schedule:
            lesson = lesson_by_sha[lesson_sha256]
            projected = projections[lesson_sha256]
            raw = _raw_factor_scores(excitatory, inhibitory, projected)
            decision = _race_span_rows(
                protocol,
                (raw,),
                (lesson.sha256,),
                (lesson.factor,),
            )[0]
            if decision.decision == "FACTOR" and decision.factor == lesson.factor:
                continue
            active = np.asarray(projected.active_global_cells, dtype=np.intp)
            target_index = _FACTOR_INDEX[lesson.factor]
            committed_ticks += _increment_tick_row(
                excitatory,
                target_index,
                active,
            )
            excitatory_updates += 1
            if decision.decision == "FACTOR":
                if decision.unique_winner_cell is None:
                    raise RuntimeError("wrong factor lacks a physical unique winner")
                committed_ticks += _increment_tick_row(
                    inhibitory,
                    decision.unique_winner_cell,
                    active,
                )
                inhibitory_updates += 1

    cell_off = _race_span_rows(
        protocol,
        zero_rows,
        source_ids,
        expected,
    )
    cell_off_pass_count = sum(
        row.decision == "FACTOR" and row.factor == row.expected_factor
        for row in cell_off
    )
    metadata = {
        "schema": "exclusive-factor-workspace-v4-checkpoint",
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_root_record": protocol.root_record(),
        "electrical_preflight_sha256": electrical_sha,
        "factor_registry_sha256": factor_registry_sha,
        "projection_panel_sha256": projection_panel_sha,
        "local_rule_sha256": protocol.local_rule_sha256,
        "fit_schedule_sha256s": schedule,
        "training_epochs": spec.LOCAL_RULE_RECORD["training_epochs"],
        "untrained_atomic_pass_count": untrained_pass_count,
        "cell_off_atomic_pass_count": cell_off_pass_count,
        "excitatory_head_update_events": excitatory_updates,
        "inhibitory_head_update_events": inhibitory_updates,
        "committed_cell_ticks": committed_ticks,
        "array_shape": (spec.FACTOR_COUNT, spec.INPUT_CELL_COUNT),
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": spec.TICK_CAP,
        "composition_attempt_count": 0,
        "duplicate_control_attempt_count": 0,
        "overlap_control_attempt_count": 0,
        "capacity_control_attempt_count": 0,
        "public_input_count": protocol.public_input_count,
        "canonical_input_count": protocol.canonical_input_count,
        "protected_input_count": protocol.protected_input_count,
    }
    checkpoint_sha = sha256(
        _checkpoint_payload(metadata, excitatory, inhibitory)
    ).hexdigest()

    _authenticate_state(checkpoint_sha, metadata, excitatory, inhibitory)
    atomic_raw_rows = tuple(
        _raw_factor_scores(excitatory, inhibitory, projections[lesson.sha256])
        for lesson in all_lessons
    )
    atomic = _race_span_rows(
        protocol,
        atomic_raw_rows,
        source_ids,
        expected,
    )
    _authenticate_state(checkpoint_sha, metadata, excitatory, inhibitory)
    atomic_pass_count = sum(
        row.decision == "FACTOR" and row.factor == row.expected_factor
        for row in atomic
    )
    atomic_mismatch_count = len(atomic) - atomic_pass_count
    fit_id_set = set(protocol.fit_lesson_sha256s)
    fit_atomic_pass_count = sum(
        row.source_sha256 in fit_id_set
        and row.decision == "FACTOR"
        and row.factor == row.expected_factor
        for row in atomic
    )
    heldout_atomic_pass_count = atomic_pass_count - fit_atomic_pass_count
    composition_open = atomic_mismatch_count == 0
    status = (
        "PASS_SELF_POSED_EXCLUSIVE_FACTOR_WORKSPACE_V4_ATOMIC_READINESS"
        if composition_open
        else "STOP_SELF_POSED_EXCLUSIVE_FACTOR_WORKSPACE_V4_ATOMIC_READINESS"
    )
    summary = BridgeBuildSummary(
        status=status,
        electrical_control_count=len(protocol.wta_electrical_controls),
        electrical_pass_count=len(electrical_replays),
        electrical_preflight_passed=True,
        electrical_preflight_sha256=electrical_sha,
        electrical_circuit_off_decision_count=circuit_off_decisions,
        electrical_tamper_rejected=tamper_rejected,
        factor_registry_sha256=factor_registry_sha,
        projection_panel_sha256=projection_panel_sha,
        epochs=spec.LOCAL_RULE_RECORD["training_epochs"],
        fit_lesson_count=len(protocol.fit_lesson_sha256s),
        heldout_lesson_count=len(protocol.heldout_lesson_sha256s),
        atomic_lesson_count=len(all_lessons),
        untrained_atomic_pass_count=untrained_pass_count,
        cell_off_atomic_pass_count=cell_off_pass_count,
        atomic_pass_count=atomic_pass_count,
        atomic_mismatch_count=atomic_mismatch_count,
        fit_atomic_pass_count=fit_atomic_pass_count,
        heldout_atomic_pass_count=heldout_atomic_pass_count,
        atomic_checkpoint_authentications=2,
        composition_open=composition_open,
        composition_attempt_count=0,
        duplicate_control_attempt_count=0,
        overlap_control_attempt_count=0,
        capacity_control_attempt_count=0,
        public_input_count=protocol.public_input_count,
        canonical_input_count=protocol.canonical_input_count,
        protected_input_count=protocol.protected_input_count,
        excitatory_head_update_events=excitatory_updates,
        inhibitory_head_update_events=inhibitory_updates,
        committed_cell_ticks=committed_ticks,
        excitatory_max_ticks=int(np.max(excitatory)),
        inhibitory_max_ticks=int(np.max(inhibitory)),
        atomic_result_sha256="",
    )
    summary = replace(
        summary,
        atomic_result_sha256=_namespaced_sha256(
            _ATOMIC_RESULT_NAMESPACE,
            _atomic_result_record(
                summary=summary,
                protocol_sha256=protocol.protocol_sha256,
                checkpoint_sha256=checkpoint_sha,
                electrical_preflight_sha256=electrical_sha,
                projection_panel_sha256=projection_panel_sha,
                factor_registry_sha256=factor_registry_sha,
                atomic=atomic,
            ),
        ),
    )
    bridge = FrozenExclusiveFactorBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        build_summary=summary,
        electrical_replays=electrical_replays,
        atomic_replay=atomic,
    )
    if bridge.checkpoint_sha256 != checkpoint_sha:
        raise RuntimeError("exclusive V4 bridge froze a different checkpoint")
    return bridge


__all__ = [
    "AuthenticatedSpanBatch",
    "BridgeBuildSummary",
    "ElectricalControlReplay",
    "ExclusiveFactorResolution",
    "FrozenExclusiveFactorBridge",
    "SpanRaceDecision",
    "build_frozen_exclusive_factor_bridge",
]
