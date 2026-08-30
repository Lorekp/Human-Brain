"""Teacher-gated factor workspace V5 learner and frozen executor.

The public seam is deliberately narrow: authenticate one committed V5 protocol,
build one immutable bridge, then resolve authenticated batches through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4 as v4impl
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4spec
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as spec
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

TICK_DTYPE = np.dtype(spec.TICK_DTYPE)
_FULL_CURRENT_PA = float(v4spec.RELAY_COPY_CURRENT_PA)
_HALF_CURRENT_PA = _FULL_CURRENT_PA / 2.0
_FACTOR_INDEX = MappingProxyType(
    {factor: index for index, factor in enumerate(spec.FACTOR_REGISTRY)}
)
_STATUS_ORDER = tuple(spec.COMPOSITOR_EOS_STATUS_RECORD["status_order"])

_CHECKPOINT_NAMESPACE = b"teacher-gated-factor-workspace-v5-checkpoint\0"
_ELECTRICAL_NAMESPACE = b"teacher-gated-factor-workspace-v5-electrical-observed\0"
_SELECTOR_TRACE_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-observed\0"
_TEACHER_TRACE_NAMESPACE = b"teacher-gated-factor-workspace-v5-teacher-observed\0"
_COMPOSITOR_TRACE_NAMESPACE = b"teacher-gated-factor-workspace-v5-compositor-observed\0"
_PROJECTION_PANEL_NAMESPACE = b"teacher-gated-factor-workspace-v5-projection-panel\0"
_FACTOR_REGISTRY_NAMESPACE = b"teacher-gated-factor-workspace-v5-factor-registry\0"
_ATOMIC_RESULT_NAMESPACE = b"teacher-gated-factor-workspace-v5-atomic-result\0"
_COMPOSITION_RESULT_NAMESPACE = b"teacher-gated-factor-workspace-v5-composition-result\0"
_CONTROL_RESULT_NAMESPACE = b"teacher-gated-factor-workspace-v5-control-result\0"
_AUTHENTICATED_BATCH_NAMESPACE = b"teacher-gated-factor-workspace-v5-authenticated-batch\0"
_RESOLUTION_NAMESPACE = b"teacher-gated-factor-workspace-v5-resolution\0"


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authenticate_protocol(protocol: spec.FrozenTeacherGatedFactorProtocol) -> None:
    if not isinstance(protocol, spec.FrozenTeacherGatedFactorProtocol):
        raise TypeError("build requires FrozenTeacherGatedFactorProtocol")
    rebuilt = spec._build_protocol()
    expected_records = (
        spec.LOCAL_RULE_RECORD,
        spec.LOCAL_RULE_SHA256,
        spec.CONTROL_PANEL_RECORD,
        spec.CONTROL_PANEL_SHA256,
        spec.SELECTOR_FULL_PREFLIGHT_RECORD,
        spec.SELECTOR_FULL_PREFLIGHT_SHA256,
        spec.TEACHER_SCOUT_CIRCUIT_RECORD,
        spec.TEACHER_SCOUT_CIRCUIT_SHA256,
        spec.ELECTRICAL_PREFLIGHT_CONTRACT_RECORD,
        spec.ELECTRICAL_PREFLIGHT_CONTRACT_SHA256,
        spec.COMPOSITION_EXECUTION_RECORD,
        spec.COMPOSITION_EXECUTION_SHA256,
        spec.FUTURE_AUTHENTICATION_RECORD,
        spec.FUTURE_AUTHENTICATION_SHA256,
    )
    if (
        rebuilt != (protocol, *expected_records)
        or protocol != spec.PROTOCOL
        or protocol.protocol_sha256 != spec.PROTOCOL.protocol_sha256
        or protocol.root_record() != spec.PROTOCOL.root_record()
    ):
        raise RuntimeError("teacher-gated V5 protocol authentication failed")


@dataclass(frozen=True, slots=True)
class SelectorReplay:
    source_sha256: str
    decision: str
    selected_slot: int | None
    selected_trace_cells: tuple[int, ...]
    principal_first_spike_step: int
    principal_first_spike_tier: tuple[int, ...]
    decision_tick: int
    address_event_cells: tuple[int, ...]
    address_read_counts: tuple[int, ...]
    address_hard_reset_count: int
    scan_zero_clamp_count: int
    basket_tick: int | None
    collision_tick: int | None
    direct_collision_spike: bool
    stored_repeat_collision_spike: bool
    address_relay_tick: int | None
    stop_relay_tick: int | None
    selected_lexical_relay_tick: int | None
    selected_slot_post_reset_active_count: int
    trace_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "decision": self.decision,
            "selected_slot": self.selected_slot,
            "selected_trace_cells": self.selected_trace_cells,
            "principal_first_spike_step": self.principal_first_spike_step,
            "principal_first_spike_tier": self.principal_first_spike_tier,
            "decision_tick": self.decision_tick,
            "address_event_cells": self.address_event_cells,
            "address_read_counts": self.address_read_counts,
            "address_hard_reset_count": self.address_hard_reset_count,
            "scan_zero_clamp_count": self.scan_zero_clamp_count,
            "basket_tick": self.basket_tick,
            "collision_tick": self.collision_tick,
            "direct_collision_spike": self.direct_collision_spike,
            "stored_repeat_collision_spike": self.stored_repeat_collision_spike,
            "address_relay_tick": self.address_relay_tick,
            "stop_relay_tick": self.stop_relay_tick,
            "selected_lexical_relay_tick": self.selected_lexical_relay_tick,
            "selected_slot_post_reset_active_count": (self.selected_slot_post_reset_active_count),
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class FactorRaceDecision:
    source_sha256: str
    decision: str
    factor: str | None
    principal_first_spike_step: int
    decision_step: int
    simultaneous_spike_candidates: int
    unique_winner_cell: int | None
    factor_relay_spike_cells: tuple[int, ...]
    raw_factor_scores: tuple[int, ...]
    relay_trace_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "decision": self.decision,
            "factor": self.factor,
            "principal_first_spike_step": self.principal_first_spike_step,
            "decision_step": self.decision_step,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
            "unique_winner_cell": self.unique_winner_cell,
            "factor_relay_spike_cells": self.factor_relay_spike_cells,
            "raw_factor_scores": self.raw_factor_scores,
            "relay_trace_sha256": self.relay_trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class PlasticityReplay:
    lesson_sha256: str
    selected_trace_cells: tuple[int, ...]
    autonomous_decision: str
    autonomous_factor: str | None
    scout_spike_cells: tuple[int, ...]
    teacher_spike_cells: tuple[int, ...]
    target_match_spike_cells: tuple[int, ...]
    excitatory_update_spike_cells: tuple[int, ...]
    inhibitory_update_spike_cells: tuple[int, ...]
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    trace_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "lesson_sha256": self.lesson_sha256,
            "selected_trace_cells": self.selected_trace_cells,
            "autonomous_decision": self.autonomous_decision,
            "autonomous_factor": self.autonomous_factor,
            "scout_spike_cells": self.scout_spike_cells,
            "teacher_spike_cells": self.teacher_spike_cells,
            "target_match_spike_cells": self.target_match_spike_cells,
            "excitatory_update_spike_cells": self.excitatory_update_spike_cells,
            "inhibitory_update_spike_cells": self.inhibitory_update_spike_cells,
            "committed_excitatory_ticks": self.committed_excitatory_ticks,
            "committed_inhibitory_ticks": self.committed_inhibitory_ticks,
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompositorReplay:
    source_sha256: str
    input_event_sha256s: tuple[str, ...]
    status: str
    factors: tuple[str, ...]
    attempted_event_count: int
    accepted_event_count: int
    duplicate_count: int
    overlap_poison: bool
    capacity_poison: bool
    eos_status_spikes: tuple[bool, bool, bool, bool, bool]
    post_reset_active_count: int
    event_stage_spike_counts: tuple[tuple[tuple[int, int], ...], ...]
    event_stage_evaluated_counts: tuple[tuple[tuple[int, int], ...], ...]
    case_reset_relay_tick: int
    case_reset_arrival_tick: int
    next_case_ready_initialization_tick: int
    trace_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "input_event_sha256s": self.input_event_sha256s,
            "status": self.status,
            "factors": self.factors,
            "attempted_event_count": self.attempted_event_count,
            "accepted_event_count": self.accepted_event_count,
            "duplicate_count": self.duplicate_count,
            "overlap_poison": self.overlap_poison,
            "capacity_poison": self.capacity_poison,
            "eos_status_spikes": self.eos_status_spikes,
            "post_reset_active_count": self.post_reset_active_count,
            "event_stage_spike_counts": self.event_stage_spike_counts,
            "event_stage_evaluated_counts": self.event_stage_evaluated_counts,
            "case_reset_relay_tick": self.case_reset_relay_tick,
            "case_reset_arrival_tick": self.case_reset_arrival_tick,
            "next_case_ready_initialization_tick": (self.next_case_ready_initialization_tick),
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompositionCaseReplay:
    case_sha256: str
    selector_replays: tuple[SelectorReplay, SelectorReplay, SelectorReplay]
    span_decisions: tuple[FactorRaceDecision, FactorRaceDecision, FactorRaceDecision]
    emitted_events: tuple[spec.ProtocolFactorEvent, ...]
    expected_factors: tuple[str, str]
    compositor: CompositorReplay

    def record(self) -> dict[str, object]:
        return {
            "case_sha256": self.case_sha256,
            "selector_replays": tuple(row.record() for row in self.selector_replays),
            "span_decisions": tuple(row.record() for row in self.span_decisions),
            "emitted_events": tuple(event.record() for event in self.emitted_events),
            "expected_factors": self.expected_factors,
            "compositor": self.compositor.record(),
        }


@dataclass(frozen=True, slots=True)
class AtomicBatchResult:
    status: str
    decisions: tuple[FactorRaceDecision, ...]
    expected_factors: tuple[str, ...]
    pass_count: int
    mismatch_count: int
    fit_pass_count: int
    heldout_pass_count: int
    checkpoint_authentications: int
    composition_attempt_count: int
    control_attempt_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "decisions": tuple(row.record() for row in self.decisions),
            "expected_factors": self.expected_factors,
            "pass_count": self.pass_count,
            "mismatch_count": self.mismatch_count,
            "fit_pass_count": self.fit_pass_count,
            "heldout_pass_count": self.heldout_pass_count,
            "checkpoint_authentications": self.checkpoint_authentications,
            "composition_attempt_count": self.composition_attempt_count,
            "control_attempt_count": self.control_attempt_count,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CompositionBatchResult:
    status: str
    cases: tuple[CompositionCaseReplay, ...]
    exact_count: int
    mismatch_count: int
    attempt_count: int
    control_attempt_count: int
    checkpoint_authentications: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "cases": tuple(row.record() for row in self.cases),
            "exact_count": self.exact_count,
            "mismatch_count": self.mismatch_count,
            "attempt_count": self.attempt_count,
            "control_attempt_count": self.control_attempt_count,
            "checkpoint_authentications": self.checkpoint_authentications,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CompositorControlBatchResult:
    status: str
    controls: tuple[CompositorReplay, ...]
    exact_count: int
    mismatch_count: int
    duplicate_attempt_count: int
    overlap_attempt_count: int
    capacity_attempt_count: int
    checkpoint_authentications: int
    sha256: str

    @property
    def attempt_count(self) -> int:
        return (
            self.duplicate_attempt_count + self.overlap_attempt_count + self.capacity_attempt_count
        )

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "controls": tuple(row.record() for row in self.controls),
            "exact_count": self.exact_count,
            "mismatch_count": self.mismatch_count,
            "duplicate_attempt_count": self.duplicate_attempt_count,
            "overlap_attempt_count": self.overlap_attempt_count,
            "capacity_attempt_count": self.capacity_attempt_count,
            "checkpoint_authentications": self.checkpoint_authentications,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class BridgeBuildSummary:
    status: str
    electrical_preflight_passed: bool
    electrical_control_count: int
    electrical_pass_count: int
    electrical_preflight_sha256: str
    projection_panel_sha256: str
    factor_registry_sha256: str
    fit_pass_count: int
    excitatory_update_events: int
    inhibitory_update_events: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    teacher_off_stop_count: int
    cell_off_stop_count: int
    random_label_permuted_exact_count: int
    random_label_true_exact_count: int
    silent_row_nonzero_bytes: int
    atomic_pass_count: int
    atomic_mismatch_count: int
    composition_attempt_count: int
    composition_exact_count: int
    duplicate_control_attempt_count: int
    overlap_control_attempt_count: int
    capacity_control_attempt_count: int
    control_exact_count: int
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    checkpoint_sha256: str
    atomic_result_sha256: str
    composition_result_sha256: str | None
    compositor_control_result_sha256: str | None


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
        if not _is_sha256(self.protocol_sha256):
            raise ValueError("authenticated batch requires a SHA-256 protocol root")
        record = {
            "protocol_sha256": self.protocol_sha256,
            "spans": tuple(span.record() for span in self.spans),
        }
        object.__setattr__(
            self,
            "authentication_sha256",
            _namespaced_sha256(_AUTHENTICATED_BATCH_NAMESPACE, record),
        )


@dataclass(frozen=True, slots=True)
class TeacherGatedFactorResolution:
    status: str
    events: tuple[spec.ProtocolFactorEvent, ...]
    span_decisions: tuple[FactorRaceDecision, ...]
    compositor: CompositorReplay
    checkpoint_authentications: int
    semantic_sha256: str


def _zero_currents(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.float64)


def _source_sha256(tokens: tuple[str, str, str, str]) -> str:
    return _namespaced_sha256(
        b"teacher-gated-factor-workspace-v5-selector-source\0",
        tokens,
    )


class _ResettableCoincidencePopulation:
    """Current-integrating coincidence cells with rooted hard reset and zero clamp."""

    __slots__ = ("_membrane", "_threshold_pa")

    def __init__(self, size: int, *, threshold_pa: float) -> None:
        if type(size) is not int or size < 1 or threshold_pa <= 0.0:
            raise ValueError("coincidence population requires positive size/threshold")
        self._membrane = np.zeros(size, dtype=np.float64)
        self._threshold_pa = float(threshold_pa)

    @property
    def active_membrane_count(self) -> int:
        return int(np.count_nonzero(self._membrane))

    def step(self, currents_pa: np.ndarray) -> np.ndarray:
        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != self._membrane.shape or not np.all(np.isfinite(currents)):
            raise ValueError("coincidence population current shape changed")
        self._membrane += currents
        fired = self._membrane >= self._threshold_pa
        self._membrane[fired] = 0.0
        return fired

    def hard_reset(self) -> tuple[float, ...]:
        self._membrane += v4spec.BASKET_INHIBITION_CURRENT_PA
        self._membrane.fill(0.0)
        return tuple(float(value) for value in self._membrane)


def _assembly_lookup(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
) -> dict[str, spec.LexicalAssembly]:
    lookup = {assembly.token: assembly for assembly in protocol.lexical_assemblies}
    if len(lookup) != len(protocol.lexical_assemblies):
        raise RuntimeError("teacher-gated V5 lexical registry is not unique")
    return lookup


def _selector_principal_trace(
    *,
    address_latch: spec._NoLeakStorageReadoutPopulation,
    collision_storage_pa: tuple[float, ...],
    circuit_gain: float,
) -> tuple[int, tuple[int, ...], tuple[int, ...], str]:
    if circuit_gain not in (0.0, 1.0):
        raise ValueError("selector circuit gain must be binary")
    population = AdaptiveGlifPopulation(spec.SELECTOR_ADDRESS_COUNT + 1)
    leading_step = -1
    leading: tuple[int, ...] = ()
    read_counts = np.zeros(spec.SELECTOR_ADDRESS_COUNT, dtype=np.int64)
    observed: list[tuple[int, tuple[int, ...]]] = []
    for step in range(v4spec.RACE_STEPS):
        gates = np.full(spec.SELECTOR_ADDRESS_COUNT, _FULL_CURRENT_PA)
        address_spikes = address_latch.step(
            set_currents_pa=_zero_currents(spec.SELECTOR_ADDRESS_COUNT),
            read_gate_currents_pa=gates,
            direct_readout_currents_pa=_zero_currents(spec.SELECTOR_ADDRESS_COUNT),
            reset_currents_pa=_zero_currents(spec.SELECTOR_ADDRESS_COUNT),
        )
        read_counts += address_spikes.astype(np.int64)
        scores = (
            *(spec.SELECTOR_ADDRESS_SCORE if spike else 0 for spike in address_spikes),
            spec.SELECTOR_STOP_SCORE,
        )
        currents = np.asarray(
            [v4spec.PROTOCOL.latency_table[score].current_pa * circuit_gain for score in scores],
            dtype=np.float64,
        )
        if leading and step >= leading_step + v4spec.BASKET_DELAY_TICKS:
            currents += v4spec.BASKET_INHIBITION_CURRENT_PA
        fired = tuple(int(cell) for cell in np.flatnonzero(population.step(currents)))
        if fired and not leading:
            leading = fired
            leading_step = step
        elif fired:
            raise RuntimeError("selector basket failed to suppress trailing principals")
        if fired or step in (0, v4spec.RACE_STEPS - 1):
            observed.append((step, fired))
    expected_read_counts = tuple(
        v4spec.RACE_STEPS if charge == _FULL_CURRENT_PA else 0
        for charge in address_latch.storage_charge_pa
    )
    if tuple(int(value) for value in read_counts) != expected_read_counts:
        raise RuntimeError("selector address current did not persist for 600 reads")
    trace_sha = _namespaced_sha256(
        _SELECTOR_TRACE_NAMESPACE,
        {
            "address_storage_pa_hex": tuple(
                float(value).hex() for value in address_latch.storage_charge_pa
            ),
            "address_read_counts": tuple(int(value) for value in read_counts),
            "principal_observed": tuple(observed),
            "collision_storage_pa_hex": tuple(float(value).hex() for value in collision_storage_pa),
            "circuit_gain_hex": float(circuit_gain).hex(),
        },
    )
    return leading_step, leading, tuple(int(value) for value in read_counts), trace_sha


def _selector_delayed_relays(
    *,
    principal_step: int,
    tier: tuple[int, ...],
    collision_storage_pa: tuple[float, ...],
) -> tuple[str, int | None, int, int | None, dict[str, object]]:
    if principal_step < 0 or not tier:
        return (
            "SILENT",
            None,
            -1,
            None,
            {
                "basket_tick": None,
                "collision_tick": None,
                "address_relay_tick": None,
                "stop_relay_tick": None,
                "direct_collision_spike": False,
                "stored_repeat_collision_spike": False,
            },
        )
    if (
        collision_storage_pa not in ((0.0,), (_FULL_CURRENT_PA,))
        or len(tier) > 2
        or any(
            type(cell) is not int or not 0 <= cell <= spec.SELECTOR_ADDRESS_COUNT for cell in tier
        )
    ):
        raise ValueError("selector delayed relay domain changed")
    principal_tick = int(spec.SELECTOR_CIRCUIT_RECORD["principal_start_tick"]) + principal_step
    basket = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    collision = spec._NoLeakStorageReadoutPopulation(1)
    if collision_storage_pa == (_FULL_CURRENT_PA,):
        collision.step(
            set_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )
    address_relays = v4impl._DiscreteNeuronPopulation(
        spec.SELECTOR_ADDRESS_COUNT,
        threshold_pa=_FULL_CURRENT_PA,
    )
    direct_stop = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    collision_stop = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    collision_latched = False
    basket_tick: int | None = None
    collision_tick: int | None = None
    address_relay_cell: int | None = None
    address_relay_tick: int | None = None
    stop_relay_tick: int | None = None
    decision_tick = -1
    decision = "SILENT"
    selected_slot: int | None = None
    direct_collision_spike = False
    stored_repeat_collision_spike = False
    observed: list[dict[str, object]] = []
    for relative_tick in range(1, v4spec.FACTOR_RELAY_DELAY_TICKS + 1):
        basket_spike = bool(
            basket.step(
                np.asarray(
                    (
                        _FULL_CURRENT_PA * len(tier)
                        if relative_tick == v4spec.BASKET_DELAY_TICKS
                        else 0.0,
                    )
                )
            )[0]
        )
        if basket_spike:
            basket_tick = principal_tick + relative_tick
        direct_current = (
            2.0 * _FULL_CURRENT_PA
            if len(tier) == 2 and relative_tick == v4spec.COLLISION_DETECTOR_DELAY_TICKS
            else 0.0
        )
        read_gate = (
            _FULL_CURRENT_PA if relative_tick == v4spec.COLLISION_DETECTOR_DELAY_TICKS else 0.0
        )
        collision_spike = bool(
            collision.step(
                set_currents_pa=np.zeros(1),
                read_gate_currents_pa=np.asarray((read_gate,)),
                direct_readout_currents_pa=np.asarray((direct_current,)),
                reset_currents_pa=np.zeros(1),
            )[0]
        )
        if collision_spike:
            collision_latched = True
            collision_tick = principal_tick + relative_tick
            direct_collision_spike = direct_current > 0.0
            stored_repeat_collision_spike = collision_storage_pa == (_FULL_CURRENT_PA,)
        address_currents = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
        if relative_tick == v4spec.FACTOR_RELAY_DELAY_TICKS:
            for cell in tier:
                if cell < spec.SELECTOR_ADDRESS_COUNT:
                    address_currents[cell] += _FULL_CURRENT_PA
            if collision_latched:
                address_currents += v4spec.BASKET_INHIBITION_CURRENT_PA
        address_spikes = tuple(
            int(cell) for cell in np.flatnonzero(address_relays.step(address_currents))
        )
        stop_current = 0.0
        if relative_tick == v4spec.STOP_RELAY_DELAY_TICKS and spec.SELECTOR_ADDRESS_COUNT in tier:
            stop_current += _FULL_CURRENT_PA
            if collision_latched:
                stop_current += v4spec.BASKET_INHIBITION_CURRENT_PA
        direct_stop_spike = bool(direct_stop.step(np.asarray((stop_current,)))[0])
        collision_stop_spike = bool(
            collision_stop.step(
                np.asarray(
                    (
                        _FULL_CURRENT_PA
                        if collision_latched and relative_tick == v4spec.COLLISION_STOP_DELAY_TICKS
                        else 0.0,
                    )
                )
            )[0]
        )
        observed.append(
            {
                "relative_tick": relative_tick,
                "basket_spike": basket_spike,
                "collision_spike": collision_spike,
                "address_relay_spikes": address_spikes,
                "direct_stop_spike": direct_stop_spike,
                "collision_stop_spike": collision_stop_spike,
            }
        )
        if address_spikes or direct_stop_spike or collision_stop_spike:
            if decision != "SILENT":
                raise RuntimeError("selector emitted multiple delayed decisions")
            if len(address_spikes) == 1 and not direct_stop_spike and not collision_stop_spike:
                address_relay_cell = address_spikes[0]
                address_relay_tick = principal_tick + relative_tick
                decision_tick = address_relay_tick
                decision = "SELECTED_SLOT"
                selected_slot = spec.SELECTOR_ADDRESSES[address_relay_cell].selected_slot
            elif not address_spikes and (direct_stop_spike ^ collision_stop_spike):
                stop_relay_tick = principal_tick + relative_tick
                decision_tick = stop_relay_tick
                decision = "COLLISION_STOP" if collision_stop_spike else "STOP"
            else:
                raise RuntimeError("selector delayed decision was not one-hot")
    if decision == "SILENT":
        raise RuntimeError("selector delayed relay circuit emitted no decision")
    return (
        decision,
        selected_slot,
        decision_tick,
        address_relay_cell,
        {
            "basket_tick": basket_tick,
            "collision_tick": collision_tick,
            "address_relay_tick": address_relay_tick,
            "stop_relay_tick": stop_relay_tick,
            "direct_collision_spike": direct_collision_spike,
            "stored_repeat_collision_spike": stored_repeat_collision_spike,
            "observed": tuple(observed),
        },
    )


def _run_selector_assemblies(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    slot_assembly_cells: tuple[tuple[int, ...], ...],
    slot_tokens: tuple[str, ...],
    source_sha256: str,
    circuit_gain: float = 1.0,
    phase_gain: float = 1.0,
    cue_gain: float = 1.0,
    latch_persistence: bool = True,
    lexical_gain: float = 1.0,
    serial_repeats: tuple[tuple[int, int], ...] = (),
) -> tuple[SelectorReplay, spec._NoLeakStorageReadoutPopulation]:
    if (
        len(slot_assembly_cells) != 4
        or len(slot_tokens) != 4
        or any(
            type(cells) is not tuple
            or any(
                type(cell) is not int or not 0 <= cell < spec.LEXICAL_CELL_COUNT for cell in cells
            )
            for cells in slot_assembly_cells
        )
        or any(
            gain not in (0.0, 1.0) for gain in (circuit_gain, phase_gain, cue_gain, lexical_gain)
        )
        or type(latch_persistence) is not bool
        or not _is_sha256(source_sha256)
    ):
        raise ValueError("selector assembly fixture changed")
    assembly_by_token = _assembly_lookup(protocol)
    cue_cells = {
        direction: set(assembly_by_token[token].cells)
        for direction, token in spec.CUE_TOKENS.items()
    }
    address_latch = spec._NoLeakStorageReadoutPopulation(spec.SELECTOR_ADDRESS_COUNT)
    seen_address_latch = spec._NoLeakStorageReadoutPopulation(spec.SELECTOR_ADDRESS_COUNT)
    collision_latch = spec._NoLeakStorageReadoutPopulation(1)
    coincidence_population = _ResettableCoincidencePopulation(
        spec.SELECTOR_ADDRESS_COUNT,
        threshold_pa=_FULL_CURRENT_PA * (spec.ASSEMBLY_ACTIVE_CELL_COUNT + 1),
    )
    address_events: list[int] = []
    coincidence_rows: list[dict[str, object]] = []
    events_by_slot: dict[int, list[tuple[int, ...]]] = {
        slot: [slot_assembly_cells[slot]] for slot in range(4)
    }
    for slot, repeat_count in serial_repeats:
        if not 0 <= slot < 4 or repeat_count < 1:
            raise ValueError("selector serial repeat domain changed")
        events_by_slot[slot].extend([slot_assembly_cells[slot]] * repeat_count)
    for slot in range(4):
        for serial_index, active_cells in enumerate(events_by_slot[slot]):
            active = set(active_cells)
            currents = np.zeros(spec.SELECTOR_ADDRESS_COUNT, dtype=np.float64)
            for address in spec.SELECTOR_ADDRESSES:
                overlap = len(active & cue_cells[address.direction])
                cue_current = overlap * _FULL_CURRENT_PA * cue_gain
                phase_current = _FULL_CURRENT_PA * phase_gain if address.cue_slot == slot else 0.0
                currents[address.address_index] = cue_current + phase_current
            spikes = coincidence_population.step(currents)
            spike_cells = tuple(int(cell) for cell in np.flatnonzero(spikes))
            for cell in spike_cells:
                seen_set_currents = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
                seen_set_currents[cell] = _FULL_CURRENT_PA
                seen_read_gates = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
                seen_read_gates[cell] = _FULL_CURRENT_PA
                seen = seen_address_latch.step(
                    set_currents_pa=seen_set_currents,
                    read_gate_currents_pa=seen_read_gates,
                    direct_readout_currents_pa=np.zeros(spec.SELECTOR_ADDRESS_COUNT),
                    reset_currents_pa=np.zeros(spec.SELECTOR_ADDRESS_COUNT),
                )
                if bool(seen[cell]):
                    collision_latch.step(
                        set_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
                        read_gate_currents_pa=np.zeros(1),
                        direct_readout_currents_pa=np.zeros(1),
                        reset_currents_pa=np.zeros(1),
                    )
                set_currents = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
                set_currents[cell] = _FULL_CURRENT_PA
                address_latch.step(
                    set_currents_pa=set_currents,
                    read_gate_currents_pa=np.zeros(spec.SELECTOR_ADDRESS_COUNT),
                    direct_readout_currents_pa=np.zeros(spec.SELECTOR_ADDRESS_COUNT),
                    reset_currents_pa=np.zeros(spec.SELECTOR_ADDRESS_COUNT),
                )
                address_events.append(cell)
            reset_snapshot = coincidence_population.hard_reset()
            if any(reset_snapshot) or coincidence_population.active_membrane_count != 0:
                raise RuntimeError("selector address hard reset left residual membrane")
            coincidence_rows.append(
                {
                    "slot": slot,
                    "serial_index": serial_index,
                    "spike_cells": spike_cells,
                    "hard_reset_current_pa_hex": float(v4spec.BASKET_INHIBITION_CURRENT_PA).hex(),
                    "post_zero_clamp_active_count": 0,
                }
            )
    collision_storage_pa = collision_latch.storage_charge_pa
    principal_step, tier, read_counts, principal_trace = _selector_principal_trace(
        address_latch=address_latch,
        collision_storage_pa=collision_storage_pa,
        circuit_gain=circuit_gain,
    )
    if principal_step < 0:
        decision, selected_slot, decision_tick, address_relay_cell = (
            "SILENT",
            None,
            -1,
            None,
        )
        relay_observed = {
            "basket_tick": None,
            "collision_tick": None,
            "address_relay_tick": None,
            "stop_relay_tick": None,
            "direct_collision_spike": False,
            "stored_repeat_collision_spike": False,
            "observed": (),
        }
    else:
        (
            decision,
            selected_slot,
            decision_tick,
            address_relay_cell,
            relay_observed,
        ) = _selector_delayed_relays(
            principal_step=principal_step,
            tier=tier,
            collision_storage_pa=collision_storage_pa,
        )
    selected_latch = spec._NoLeakStorageReadoutPopulation(4)
    if address_relay_cell is not None and latch_persistence:
        set_currents = np.zeros(4)
        set_currents[spec.SELECTOR_ADDRESSES[address_relay_cell].selected_slot] = _FULL_CURRENT_PA
        selected_latch.step(
            set_currents_pa=set_currents,
            read_gate_currents_pa=np.zeros(4),
            direct_readout_currents_pa=np.zeros(4),
            reset_currents_pa=np.zeros(4),
        )
    selected_trace = np.zeros(spec.LEXICAL_CELL_COUNT, dtype=np.bool_)
    selected_trace_latch = spec._NoLeakStorageReadoutPopulation(spec.LEXICAL_CELL_COUNT)
    replay_rows: list[dict[str, object]] = []
    for slot, (token, cells) in enumerate(zip(slot_tokens, slot_assembly_cells, strict=True)):
        gates = np.zeros(4)
        gates[slot] = _FULL_CURRENT_PA
        latch_spikes = selected_latch.step(
            set_currents_pa=np.zeros(4),
            read_gate_currents_pa=gates,
            direct_readout_currents_pa=np.zeros(4),
            reset_currents_pa=np.zeros(4),
        )
        assembly_currents = np.zeros(spec.LEXICAL_CELL_COUNT, dtype=np.float64)
        assembly_currents[np.asarray(cells, dtype=np.intp)] = _FULL_CURRENT_PA * lexical_gain
        latch_gate_currents = np.full(
            spec.LEXICAL_CELL_COUNT,
            _FULL_CURRENT_PA if bool(latch_spikes[slot]) else 0.0,
        )
        lexical_relays = v4impl._DiscreteNeuronPopulation(
            spec.LEXICAL_CELL_COUNT,
            threshold_pa=2.0 * _FULL_CURRENT_PA,
        )
        relay_spikes = lexical_relays.step(assembly_currents + latch_gate_currents)
        relay_cells = tuple(int(cell) for cell in np.flatnonzero(relay_spikes))
        selected_trace |= relay_spikes
        selected_trace_latch.step(
            set_currents_pa=relay_spikes.astype(np.float64) * _FULL_CURRENT_PA,
            read_gate_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
            direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
            reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        )
        replay_rows.append(
            {
                "slot": slot,
                "token": token,
                "latch_spike": bool(latch_spikes[slot]),
                "relay_cells": relay_cells,
            }
        )
    reset = np.full(4, v4spec.BASKET_INHIBITION_CURRENT_PA)
    selected_latch.step(
        set_currents_pa=np.zeros(4),
        read_gate_currents_pa=np.zeros(4),
        direct_readout_currents_pa=np.zeros(4),
        reset_currents_pa=reset,
    )
    trace_cells = tuple(int(cell) for cell in np.flatnonzero(selected_trace))
    if trace_cells and len(trace_cells) != spec.ASSEMBLY_ACTIVE_CELL_COUNT:
        raise RuntimeError("selector lexical replay did not emit exactly 12 cells")
    trace_record = {
        "source_sha256": source_sha256,
        "coincidence_rows": tuple(coincidence_rows),
        "address_events": tuple(address_events),
        "address_read_counts": read_counts,
        "collision_storage_pa_hex": tuple(float(value).hex() for value in collision_storage_pa),
        "principal_trace_sha256": principal_trace,
        "principal_first_spike_step": principal_step,
        "principal_first_spike_tier": tier,
        "relay_observed": relay_observed,
        "decision": decision,
        "selected_slot": selected_slot,
        "decision_tick": decision_tick,
        "replay_rows": tuple(replay_rows),
        "selected_trace_cells": trace_cells,
        "post_reset_storage_pa_hex": tuple(
            float(value).hex() for value in selected_latch.storage_charge_pa
        ),
    }
    selected_relay_tick = (
        decision_tick + tuple(spec.SELECTOR_CIRCUIT_RECORD["content_gate_offsets"])[selected_slot]
        if selected_slot is not None
        else None
    )
    if (
        tuple(
            int(cell)
            for cell, charge in enumerate(selected_trace_latch.storage_charge_pa)
            if charge == _FULL_CURRENT_PA
        )
        != trace_cells
    ):
        raise RuntimeError("lexical relay did not set selected-trace storage directly")
    return (
        SelectorReplay(
            source_sha256=source_sha256,
            decision=decision,
            selected_slot=selected_slot,
            selected_trace_cells=trace_cells,
            principal_first_spike_step=principal_step,
            principal_first_spike_tier=tier,
            decision_tick=decision_tick,
            address_event_cells=tuple(address_events),
            address_read_counts=read_counts,
            address_hard_reset_count=len(address_events),
            scan_zero_clamp_count=len(coincidence_rows),
            basket_tick=relay_observed["basket_tick"],
            collision_tick=relay_observed["collision_tick"],
            direct_collision_spike=bool(relay_observed["direct_collision_spike"]),
            stored_repeat_collision_spike=bool(relay_observed["stored_repeat_collision_spike"]),
            address_relay_tick=relay_observed["address_relay_tick"],
            stop_relay_tick=relay_observed["stop_relay_tick"],
            selected_lexical_relay_tick=selected_relay_tick,
            selected_slot_post_reset_active_count=sum(
                value != 0.0 for value in selected_latch.storage_charge_pa
            ),
            trace_sha256=_namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, trace_record),
        ),
        selected_trace_latch,
    )


def _run_selector(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    tokens: tuple[str, str, str, str],
    *,
    source_sha256: str,
    circuit_gain: float = 1.0,
    phase_gain: float = 1.0,
    cue_gain: float = 1.0,
    latch_persistence: bool = True,
    lexical_gain: float = 1.0,
) -> tuple[SelectorReplay, spec._NoLeakStorageReadoutPopulation]:
    if type(tokens) is not tuple or len(tokens) != 4:
        raise ValueError("selector requires exactly four token relays")
    lookup = _assembly_lookup(protocol)
    if any(token not in lookup for token in tokens):
        raise ValueError("selector token lacks an authenticated lexical assembly")
    return _run_selector_assemblies(
        protocol,
        slot_assembly_cells=tuple(lookup[token].cells for token in tokens),
        slot_tokens=tokens,
        source_sha256=source_sha256,
        circuit_gain=circuit_gain,
        phase_gain=phase_gain,
        cue_gain=cue_gain,
        latch_persistence=latch_persistence,
        lexical_gain=lexical_gain,
    )


def _close_selector_preflight(
    execution: tuple[SelectorReplay, spec._NoLeakStorageReadoutPopulation],
) -> SelectorReplay:
    replay, trace_latch = execution
    trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.full(
            spec.LEXICAL_CELL_COUNT,
            v4spec.BASKET_INHIBITION_CURRENT_PA,
        ),
    )
    if any(trace_latch.storage_charge_pa):
        raise RuntimeError("selector preflight selected-trace reset failed")
    return replay


def _raw_factor_scores(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    selected_trace_cells: tuple[int, ...],
) -> tuple[int, ...]:
    if (
        excitatory_ticks.shape != spec.FUTURE_STATE_SHAPE
        or inhibitory_ticks.shape != spec.FUTURE_STATE_SHAPE
        or excitatory_ticks.dtype != TICK_DTYPE
        or inhibitory_ticks.dtype != TICK_DTYPE
        or type(selected_trace_cells) is not tuple
        or len(selected_trace_cells) not in (0, spec.ASSEMBLY_ACTIVE_CELL_COUNT)
        or any(
            type(cell) is not int or not 0 <= cell < spec.LEXICAL_CELL_COUNT
            for cell in selected_trace_cells
        )
        or len(set(selected_trace_cells)) != len(selected_trace_cells)
    ):
        raise ValueError("teacher-gated V5 selected-trace score domain changed")
    if not selected_trace_cells:
        return (0,) * spec.FACTOR_COUNT
    active = np.asarray(selected_trace_cells, dtype=np.intp)
    raw = np.sum(
        excitatory_ticks[:, active].astype(np.int16) - inhibitory_ticks[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    if np.any(raw < spec.SCORE_MIN) or np.any(raw > spec.SCORE_MAX):
        raise RuntimeError("teacher-gated V5 raw score left the frozen domain")
    return tuple(int(value) for value in raw)


def _factor_currents(raw_scores: tuple[int, ...]) -> np.ndarray:
    if (
        type(raw_scores) is not tuple
        or len(raw_scores) != spec.FACTOR_COUNT
        or any(
            type(score) is not int or not spec.SCORE_MIN <= score <= spec.SCORE_MAX
            for score in raw_scores
        )
    ):
        raise ValueError("teacher-gated V5 factor-score row changed")
    scores = (*(v4spec._current_score(score) for score in raw_scores), spec.STOP_SCORE)
    return np.asarray(
        [v4spec.PROTOCOL.latency_table[score].current_pa for score in scores],
        dtype=np.float64,
    )


def _relay_factor_decision(
    *,
    source_sha256: str,
    raw_scores: tuple[int, ...],
    principal_step: int,
    tier: tuple[int, ...],
) -> tuple[FactorRaceDecision, np.ndarray]:
    if not tier or principal_step < 0:
        raise RuntimeError("factor principal circuit emitted no leading tier")
    (
        basket,
        collision,
        decision_step,
        relay_decision,
        factor,
        factor_relay_spikes,
        relay_trace,
    ) = _run_factor_relays(principal_step=principal_step, tier=tier)
    if basket != principal_step + v4spec.BASKET_DELAY_TICKS:
        raise RuntimeError("factor basket timing changed")
    if len(tier) > 1:
        decision = "COLLISION"
        unique_winner = None
        if collision is None or relay_decision != "STOP" or factor is not None:
            raise RuntimeError("factor collision did not fail closed")
    else:
        decision = relay_decision
        unique_winner = tier[0]
        if collision is not None or decision not in ("FACTOR", "STOP"):
            raise RuntimeError("factor unique relay path changed")
    if decision_step is None:
        raise RuntimeError("factor delayed relay emitted no decision")
    return (
        FactorRaceDecision(
            source_sha256=source_sha256,
            decision=str(decision),
            factor=factor,
            principal_first_spike_step=principal_step,
            decision_step=decision_step,
            simultaneous_spike_candidates=len(tier),
            unique_winner_cell=unique_winner,
            factor_relay_spike_cells=tuple(
                int(cell) for cell in np.flatnonzero(factor_relay_spikes)
            ),
            raw_factor_scores=raw_scores,
            relay_trace_sha256=relay_trace,
        ),
        factor_relay_spikes,
    )


def _run_factor_relays(
    *,
    principal_step: int,
    tier: tuple[int, ...],
) -> tuple[int, int | None, int, str, str | None, np.ndarray, str]:
    if principal_step < 0 or not tier:
        raise ValueError("factor relay requires a physical leading tier")
    basket = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    collision = v4impl._DiscreteNeuronPopulation(
        1,
        threshold_pa=2.0 * _FULL_CURRENT_PA,
    )
    factor_relays = v4impl._DiscreteNeuronPopulation(
        spec.FACTOR_COUNT,
        threshold_pa=_FULL_CURRENT_PA,
    )
    direct_stop = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    collision_stop = v4impl._DiscreteNeuronPopulation(1, threshold_pa=_FULL_CURRENT_PA)
    basket_step = -1
    collision_step: int | None = None
    decision_step = -1
    decision = ""
    factor: str | None = None
    emitted_factor_mask = np.zeros(spec.FACTOR_COUNT, dtype=np.bool_)
    collision_latched = False
    observed: list[dict[str, object]] = []
    for relative_tick in range(1, v4spec.FACTOR_RELAY_DELAY_TICKS + 1):
        basket_spike = bool(
            basket.step(
                np.asarray(
                    (
                        _FULL_CURRENT_PA * len(tier)
                        if relative_tick == v4spec.BASKET_DELAY_TICKS
                        else 0.0,
                    )
                )
            )[0]
        )
        if basket_spike:
            basket_step = principal_step + relative_tick
        collision_spike = bool(
            collision.step(
                np.asarray(
                    (
                        _FULL_CURRENT_PA * len(tier)
                        if relative_tick == v4spec.COLLISION_DETECTOR_DELAY_TICKS
                        else 0.0,
                    )
                )
            )[0]
        )
        if collision_spike:
            collision_latched = True
            collision_step = principal_step + relative_tick
        factor_currents = np.zeros(spec.FACTOR_COUNT)
        if relative_tick == v4spec.FACTOR_RELAY_DELAY_TICKS:
            for cell in tier:
                if cell < spec.FACTOR_COUNT:
                    factor_currents[cell] += _FULL_CURRENT_PA
            if collision_latched:
                factor_currents += v4spec.BASKET_INHIBITION_CURRENT_PA
        factor_spikes = np.asarray(factor_relays.step(factor_currents), dtype=np.bool_)
        stop_current = 0.0
        if relative_tick == v4spec.STOP_RELAY_DELAY_TICKS and spec.FACTOR_COUNT in tier:
            stop_current += _FULL_CURRENT_PA
            if collision_latched:
                stop_current += v4spec.BASKET_INHIBITION_CURRENT_PA
        direct_stop_spike = bool(direct_stop.step(np.asarray((stop_current,)))[0])
        collision_stop_spike = bool(
            collision_stop.step(
                np.asarray(
                    (
                        _FULL_CURRENT_PA
                        if collision_latched and relative_tick == v4spec.COLLISION_STOP_DELAY_TICKS
                        else 0.0,
                    )
                )
            )[0]
        )
        observed.append(
            {
                "relative_tick": relative_tick,
                "basket_spike": basket_spike,
                "collision_spike": collision_spike,
                "factor_relay_spikes": tuple(int(cell) for cell in np.flatnonzero(factor_spikes)),
                "direct_stop_spike": direct_stop_spike,
                "collision_stop_spike": collision_stop_spike,
            }
        )
        if np.any(factor_spikes) or direct_stop_spike or collision_stop_spike:
            if decision:
                raise RuntimeError("factor relay emitted multiple decisions")
            if np.count_nonzero(factor_spikes) == 1 and not (
                direct_stop_spike or collision_stop_spike
            ):
                factor_cell = int(np.flatnonzero(factor_spikes)[0])
                emitted_factor_mask = factor_spikes.copy()
                decision = "FACTOR"
                factor = spec.FACTOR_REGISTRY[factor_cell]
            elif not np.any(factor_spikes) and (direct_stop_spike ^ collision_stop_spike):
                decision = "STOP"
            else:
                raise RuntimeError("factor relay output was not one-hot")
            decision_step = principal_step + relative_tick
    if basket_step < 0 or decision_step < 0 or not decision:
        raise RuntimeError("factor relay circuit did not close")
    trace_sha = _namespaced_sha256(
        b"teacher-gated-factor-workspace-v5-factor-relay\0",
        {
            "principal_step": principal_step,
            "tier": tier,
            "observed": tuple(observed),
        },
    )
    return (
        basket_step,
        collision_step,
        decision_step,
        decision,
        factor,
        emitted_factor_mask,
        trace_sha,
    )


def _run_training_factor_circuit(
    *,
    source_sha256: str,
    raw_scores: tuple[int, ...],
) -> tuple[
    FactorRaceDecision,
    spec._NoLeakStorageReadoutPopulation,
    spec._NoLeakStorageReadoutPopulation,
    str,
]:
    currents = _factor_currents(raw_scores)
    principals = AdaptiveGlifPopulation(spec.FACTOR_COUNT + 1)
    scouts = AdaptiveGlifPopulation(spec.FACTOR_COUNT)
    leading: tuple[int, ...] = ()
    leading_step = -1
    scout_latch = spec._NoLeakStorageReadoutPopulation(spec.FACTOR_COUNT)
    stop_close = v4spec.PROTOCOL.latency_table[spec.STOP_SCORE].first_spike_step
    observed: list[dict[str, object]] = []
    for step in range(stop_close + 1):
        principal_currents = currents.copy()
        if leading and step >= leading_step + v4spec.BASKET_DELAY_TICKS:
            principal_currents += v4spec.BASKET_INHIBITION_CURRENT_PA
        principal_spikes = tuple(
            int(cell) for cell in np.flatnonzero(principals.step(principal_currents))
        )
        if principal_spikes and not leading:
            leading = principal_spikes
            leading_step = step
        elif principal_spikes:
            raise RuntimeError("factor basket failed during FIT")
        scout_spikes = np.asarray(scouts.step(currents[:-1]), dtype=np.bool_)
        if np.any(scout_spikes):
            scout_latch.step(
                set_currents_pa=scout_spikes.astype(np.float64) * _FULL_CURRENT_PA,
                read_gate_currents_pa=np.zeros(spec.FACTOR_COUNT),
                direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
                reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
            )
        if principal_spikes or np.any(scout_spikes) or step in (0, stop_close):
            observed.append(
                {
                    "step": step,
                    "principal_spikes": principal_spikes,
                    "scout_spikes": tuple(int(cell) for cell in np.flatnonzero(scout_spikes)),
                }
            )
    decision, factor_relay_spikes = _relay_factor_decision(
        source_sha256=source_sha256,
        raw_scores=raw_scores,
        principal_step=leading_step,
        tier=leading,
    )
    autonomous_latch = spec._NoLeakStorageReadoutPopulation(spec.FACTOR_COUNT)
    autonomous_latch.step(
        set_currents_pa=factor_relay_spikes.astype(np.float64) * _FULL_CURRENT_PA,
        read_gate_currents_pa=np.zeros(spec.FACTOR_COUNT),
        direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
        reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
    )
    trace_sha = _namespaced_sha256(
        _TEACHER_TRACE_NAMESPACE,
        {
            "source_sha256": source_sha256,
            "raw_scores": raw_scores,
            "base_currents_pa_hex": tuple(float(value).hex() for value in currents),
            "stop_close_step": stop_close,
            "observed": tuple(observed),
            "leading_step": leading_step,
            "leading_tier": leading,
            "scout_storage_pa_hex": tuple(
                float(value).hex() for value in scout_latch.storage_charge_pa
            ),
            "autonomous_storage_pa_hex": tuple(
                float(value).hex() for value in autonomous_latch.storage_charge_pa
            ),
            "relay_trace_sha256": decision.relay_trace_sha256,
        },
    )
    return decision, autonomous_latch, scout_latch, trace_sha


def _teacher_spike_mask(factor: str, *, gain: float = 1.0) -> np.ndarray:
    if factor not in _FACTOR_INDEX or gain not in (0.0, 1.0):
        raise ValueError("teacher clamp domain changed")
    spikes = np.zeros(spec.FACTOR_COUNT, dtype=np.bool_)
    if gain == 1.0:
        currents = np.zeros(spec.FACTOR_COUNT, dtype=np.float64)
        currents[_FACTOR_INDEX[factor]] = v4spec.TEACHER_CLAMP_CURRENT_PA
        teacher = v4impl._DiscreteNeuronPopulation(
            spec.FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        )
        spikes = teacher.step(currents)
    return np.asarray(spikes, dtype=np.bool_)


def _target_match_relays(
    *,
    teacher_spikes: np.ndarray,
    autonomous_readout_spikes: np.ndarray,
) -> np.ndarray:
    teacher = np.asarray(teacher_spikes)
    autonomous = np.asarray(autonomous_readout_spikes)
    if (
        teacher.shape != (spec.FACTOR_COUNT,)
        or autonomous.shape != (spec.FACTOR_COUNT,)
        or teacher.dtype != np.bool_
        or autonomous.dtype != np.bool_
        or np.count_nonzero(teacher) > 1
        or np.count_nonzero(autonomous) > 1
    ):
        raise ValueError("teacher/scout spike-mask domain changed")
    match_population = v4impl._DiscreteNeuronPopulation(
        spec.FACTOR_COUNT,
        threshold_pa=_FULL_CURRENT_PA,
    )
    match_currents = (
        teacher.astype(np.float64) * _HALF_CURRENT_PA
        + autonomous.astype(np.float64) * _HALF_CURRENT_PA
    )
    return np.asarray(match_population.step(match_currents), dtype=np.bool_)


def _teacher_scout_update_relays(
    *,
    teacher_spikes: np.ndarray,
    scout_readout_spikes: np.ndarray,
    target_match_spikes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    teacher = np.asarray(teacher_spikes)
    scouts = np.asarray(scout_readout_spikes)
    matches = np.asarray(target_match_spikes)
    if (
        teacher.shape != (spec.FACTOR_COUNT,)
        or scouts.shape != (spec.FACTOR_COUNT,)
        or matches.shape != (spec.FACTOR_COUNT,)
        or teacher.dtype != np.bool_
        or scouts.dtype != np.bool_
        or matches.dtype != np.bool_
        or np.count_nonzero(teacher) > 1
        or np.count_nonzero(matches) > 1
    ):
        raise ValueError("teacher/scout update-relay spike domain changed")
    match_inhibition = _FULL_CURRENT_PA if np.any(matches) else 0.0
    e_population = v4impl._DiscreteNeuronPopulation(
        spec.FACTOR_COUNT,
        threshold_pa=_FULL_CURRENT_PA,
    )
    e_currents = teacher.astype(np.float64) * _FULL_CURRENT_PA - match_inhibition
    e_updates = e_population.step(e_currents)
    h_population = v4impl._DiscreteNeuronPopulation(
        spec.FACTOR_COUNT,
        threshold_pa=_FULL_CURRENT_PA,
    )
    teacher_open = _HALF_CURRENT_PA if np.any(teacher) else 0.0
    h_currents = (
        teacher_open
        + scouts.astype(np.float64) * _HALF_CURRENT_PA
        - teacher.astype(np.float64) * _FULL_CURRENT_PA
        - match_inhibition
    )
    h_updates = h_population.step(h_currents)
    return np.asarray(e_updates, dtype=np.bool_), np.asarray(h_updates, dtype=np.bool_)


def _commit_local_plasticity(
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    selected_trace_spikes: np.ndarray,
    excitatory_update_spikes: np.ndarray,
    inhibitory_update_spikes: np.ndarray,
) -> tuple[int, int]:
    selected = np.asarray(selected_trace_spikes)
    e_updates = np.asarray(excitatory_update_spikes)
    h_updates = np.asarray(inhibitory_update_spikes)
    if (
        excitatory_ticks.shape != spec.FUTURE_STATE_SHAPE
        or inhibitory_ticks.shape != spec.FUTURE_STATE_SHAPE
        or excitatory_ticks.dtype != TICK_DTYPE
        or inhibitory_ticks.dtype != TICK_DTYPE
        or selected.shape != (spec.LEXICAL_CELL_COUNT,)
        or selected.dtype != np.bool_
        or np.count_nonzero(selected) not in (0, spec.ASSEMBLY_ACTIVE_CELL_COUNT)
        or e_updates.shape != (spec.FACTOR_COUNT,)
        or h_updates.shape != (spec.FACTOR_COUNT,)
        or e_updates.dtype != np.bool_
        or h_updates.dtype != np.bool_
        or np.any(excitatory_ticks > spec.TICK_CAP)
        or np.any(inhibitory_ticks > spec.TICK_CAP)
    ):
        raise ValueError("teacher-gated V5 plasticity snapshot domain changed")
    active_cells = np.flatnonzero(selected)
    before_e = excitatory_ticks.astype(np.uint16, copy=True)
    before_h = inhibitory_ticks.astype(np.uint16, copy=True)
    after_e = before_e.copy()
    after_h = before_h.copy()
    if active_cells.size:
        e_rows = np.flatnonzero(e_updates)
        h_rows = np.flatnonzero(h_updates)
        if e_rows.size:
            selected_e = after_e[np.ix_(e_rows, active_cells)]
            after_e[np.ix_(e_rows, active_cells)] = np.minimum(
                selected_e + np.uint16(1),
                np.uint16(spec.TICK_CAP),
            )
        if h_rows.size:
            selected_h = after_h[np.ix_(h_rows, active_cells)]
            after_h[np.ix_(h_rows, active_cells)] = np.minimum(
                selected_h + np.uint16(1),
                np.uint16(spec.TICK_CAP),
            )
    delta_e = int(np.sum(after_e - before_e, dtype=np.uint64))
    delta_h = int(np.sum(after_h - before_h, dtype=np.uint64))
    excitatory_ticks[...] = after_e.astype(TICK_DTYPE)
    inhibitory_ticks[...] = after_h.astype(TICK_DTYPE)
    return delta_e, delta_h


def _execute_fit_lesson(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    lesson: spec.AtomicLesson,
    teacher_factor: str,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    teacher_gain: float = 1.0,
    lexical_gain: float = 1.0,
) -> PlasticityReplay:
    selector, selected_trace_latch = _run_selector(
        protocol,
        lesson.tokens,
        source_sha256=lesson.sha256,
        lexical_gain=lexical_gain,
    )
    if lexical_gain == 1.0 and (
        selector.decision != "SELECTED_SLOT"
        or selector.selected_slot != lesson.selected_slot
        or len(selector.selected_trace_cells) != spec.ASSEMBLY_ACTIVE_CELL_COUNT
    ):
        raise RuntimeError("FIT selector replay disagreed with preregistered metadata")
    score_trace_spikes = selected_trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.full(spec.LEXICAL_CELL_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
    )
    raw_scores = _raw_factor_scores(
        excitatory_ticks,
        inhibitory_ticks,
        tuple(int(cell) for cell in np.flatnonzero(score_trace_spikes)),
    )
    (
        autonomous,
        autonomous_latch,
        scout_latch,
        factor_trace_sha,
    ) = _run_training_factor_circuit(
        source_sha256=lesson.sha256,
        raw_scores=raw_scores,
    )
    teacher_mask = _teacher_spike_mask(teacher_factor, gain=teacher_gain)
    autonomous_mask = autonomous_latch.step(
        set_currents_pa=np.zeros(spec.FACTOR_COUNT),
        read_gate_currents_pa=np.full(spec.FACTOR_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
        reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
    )
    scout_mask = scout_latch.step(
        set_currents_pa=np.zeros(spec.FACTOR_COUNT),
        read_gate_currents_pa=np.full(spec.FACTOR_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
        reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
    )
    matches = _target_match_relays(
        teacher_spikes=teacher_mask,
        autonomous_readout_spikes=autonomous_mask,
    )
    e_updates, h_updates = _teacher_scout_update_relays(
        teacher_spikes=teacher_mask,
        scout_readout_spikes=scout_mask,
        target_match_spikes=matches,
    )
    selected_mask = selected_trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.full(spec.LEXICAL_CELL_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
    )
    committed_e, committed_h = _commit_local_plasticity(
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
        selected_trace_spikes=selected_mask,
        excitatory_update_spikes=e_updates,
        inhibitory_update_spikes=h_updates,
    )
    selected_trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.full(
            spec.LEXICAL_CELL_COUNT,
            v4spec.BASKET_INHIBITION_CURRENT_PA,
        ),
    )
    for latch in (autonomous_latch, scout_latch):
        latch.step(
            set_currents_pa=np.zeros(spec.FACTOR_COUNT),
            read_gate_currents_pa=np.zeros(spec.FACTOR_COUNT),
            direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
            reset_currents_pa=np.full(
                spec.FACTOR_COUNT,
                v4spec.BASKET_INHIBITION_CURRENT_PA,
            ),
        )
    if (
        any(selected_trace_latch.storage_charge_pa)
        or any(autonomous_latch.storage_charge_pa)
        or any(scout_latch.storage_charge_pa)
    ):
        raise RuntimeError("teacher/scout lesson reset left persistent charge")
    trace_record = {
        "lesson_sha256": lesson.sha256,
        "selector_trace_sha256": selector.trace_sha256,
        "selected_trace_cells": selector.selected_trace_cells,
        "factor_trace_sha256": factor_trace_sha,
        "autonomous": autonomous.record(),
        "scout_spike_cells": tuple(int(cell) for cell in np.flatnonzero(scout_mask)),
        "teacher_spike_cells": tuple(int(cell) for cell in np.flatnonzero(teacher_mask)),
        "target_match_spike_cells": tuple(int(cell) for cell in np.flatnonzero(matches)),
        "excitatory_update_spike_cells": tuple(int(cell) for cell in np.flatnonzero(e_updates)),
        "inhibitory_update_spike_cells": tuple(int(cell) for cell in np.flatnonzero(h_updates)),
        "committed_excitatory_ticks": committed_e,
        "committed_inhibitory_ticks": committed_h,
        "teacher_tick": spec.TEACHER_SCOUT_CIRCUIT_RECORD["teacher_spike_tick"],
        "match_tick": spec.TEACHER_SCOUT_CIRCUIT_RECORD["target_match_tick"],
        "update_tick": spec.TEACHER_SCOUT_CIRCUIT_RECORD["update_relay_tick"],
        "commit_tick": spec.TEACHER_SCOUT_CIRCUIT_RECORD["atomic_delta_commit_tick"],
        "reset_tick": spec.TEACHER_SCOUT_CIRCUIT_RECORD["lesson_reset_tick"],
        "post_reset_selected_storage_nonzero": sum(
            value != 0.0 for value in selected_trace_latch.storage_charge_pa
        ),
        "post_reset_autonomous_storage_nonzero": sum(
            value != 0.0 for value in autonomous_latch.storage_charge_pa
        ),
        "post_reset_scout_storage_nonzero": sum(
            value != 0.0 for value in scout_latch.storage_charge_pa
        ),
    }
    return PlasticityReplay(
        lesson_sha256=lesson.sha256,
        selected_trace_cells=selector.selected_trace_cells,
        autonomous_decision=autonomous.decision,
        autonomous_factor=autonomous.factor,
        scout_spike_cells=trace_record["scout_spike_cells"],
        teacher_spike_cells=trace_record["teacher_spike_cells"],
        target_match_spike_cells=trace_record["target_match_spike_cells"],
        excitatory_update_spike_cells=trace_record["excitatory_update_spike_cells"],
        inhibitory_update_spike_cells=trace_record["inhibitory_update_spike_cells"],
        committed_excitatory_ticks=committed_e,
        committed_inhibitory_ticks=committed_h,
        trace_sha256=_namespaced_sha256(_TEACHER_TRACE_NAMESPACE, trace_record),
    )


def _race_factor_batch(
    *,
    raw_rows: tuple[tuple[int, ...], ...],
    source_sha256s: tuple[str, ...],
) -> tuple[FactorRaceDecision, ...]:
    if (
        type(raw_rows) is not tuple
        or type(source_sha256s) is not tuple
        or len(raw_rows) != len(source_sha256s)
        or not raw_rows
    ):
        raise ValueError("teacher-gated V5 factor batch shape changed")
    current_rows = np.stack([_factor_currents(row) for row in raw_rows])
    races = CompiledRestingGlifRace.compile(step_count=v4spec.RACE_STEPS).race_many(
        current_rows,
        require_unique=False,
    )
    decisions: list[FactorRaceDecision] = []
    for raw_scores, source_sha, currents, race in zip(
        raw_rows,
        source_sha256s,
        current_rows,
        races,
        strict=True,
    ):
        tier, _principal_trace = v4impl._live_principal_tier(currents, race.spike_step)
        if len(tier) != race.simultaneous_spike_candidates:
            raise RuntimeError("factor batch principal tier changed")
        decision, _factor_relay_spikes = _relay_factor_decision(
            source_sha256=source_sha,
            raw_scores=raw_scores,
            principal_step=race.spike_step,
            tier=tier,
        )
        decisions.append(decision)
    return tuple(decisions)


def _threshold_spikes(currents_pa: np.ndarray, *, threshold_pa: float) -> np.ndarray:
    currents = np.asarray(currents_pa, dtype=np.float64)
    if currents.ndim != 1 or not np.all(np.isfinite(currents)):
        raise ValueError("threshold population current domain changed")
    population = v4impl._DiscreteNeuronPopulation(
        int(currents.size),
        threshold_pa=threshold_pa,
    )
    return np.asarray(population.step(currents), dtype=np.bool_)


def _ingress_identity(event: spec.ProtocolFactorEvent) -> np.ndarray:
    if (
        not isinstance(event, spec.ProtocolFactorEvent)
        or event.factor not in _FACTOR_INDEX
        or type(event.start_token) is not int
        or type(event.stop_token) is not int
        or not 0 <= event.start_token < event.stop_token <= 12
        or event != spec._event(event.factor, event.start_token, event.stop_token)
    ):
        raise ValueError("compositor event authentication failed")
    ingress = np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT, dtype=np.bool_)
    ingress[_FACTOR_INDEX[event.factor]] = True
    start_offset = spec.FACTOR_COUNT
    stop_offset = start_offset + spec._COMPOSITOR_BOUNDARY_CELL_COUNT
    footprint_offset = stop_offset + spec._COMPOSITOR_BOUNDARY_CELL_COUNT
    ingress[start_offset + event.start_token] = True
    ingress[stop_offset + event.stop_token] = True
    ingress[footprint_offset + event.start_token : footprint_offset + event.stop_token] = True
    return ingress


def _ingress_identity_from_factor_relay(
    event: spec.ProtocolFactorEvent,
    factor_relay_spike_cells: tuple[int, ...],
) -> np.ndarray:
    if (
        type(factor_relay_spike_cells) is not tuple
        or len(factor_relay_spike_cells) != 1
        or type(factor_relay_spike_cells[0]) is not int
        or not 0 <= factor_relay_spike_cells[0] < spec.FACTOR_COUNT
    ):
        raise ValueError("compositor factor relay must be physically one-hot")
    factor_cell = factor_relay_spike_cells[0]
    if event.factor != spec.FACTOR_REGISTRY[factor_cell]:
        raise RuntimeError("compositor event label disagreed with physical factor relay")
    ingress = _ingress_identity(event)
    ingress[: spec.FACTOR_COUNT] = False
    ingress[factor_cell] = True
    return ingress


def _duplicate_key_spikes(
    *,
    slot_identity: np.ndarray,
    ingress_identity: np.ndarray,
) -> np.ndarray:
    slots = np.asarray(slot_identity)
    ingress = np.asarray(ingress_identity)
    if (
        slots.shape != (spec.EVENT_CAPACITY, spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT)
        or ingress.shape != (spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,)
        or slots.dtype != np.bool_
        or ingress.dtype != np.bool_
    ):
        raise ValueError("compositor duplicate-key spike domain changed")
    boundary = spec._COMPOSITOR_BOUNDARY_CELL_COUNT
    start_offset = spec.FACTOR_COUNT
    stop_offset = start_offset + boundary
    rows: list[np.ndarray] = []
    leg = _FULL_CURRENT_PA / 5.0
    for slot in range(spec.EVENT_CAPACITY):
        stored_factor = slots[slot, : spec.FACTOR_COUNT].astype(np.float64)
        ingress_factor = ingress[: spec.FACTOR_COUNT].astype(np.float64)
        stored_start = slots[slot, start_offset:stop_offset].astype(np.float64)
        ingress_start = ingress[start_offset:stop_offset].astype(np.float64)
        stored_stop = slots[
            slot,
            stop_offset : stop_offset + boundary,
        ].astype(np.float64)
        ingress_stop = ingress[stop_offset : stop_offset + boundary].astype(np.float64)
        currents = leg * (
            stored_factor[:, None, None]
            + ingress_factor[:, None, None]
            + stored_start[None, :, None]
            + ingress_start[None, :, None]
            + stored_stop[None, None, :]
            + ingress_stop[None, None, :]
        )
        rows.append(
            _threshold_spikes(
                currents.reshape(-1),
                threshold_pa=6.0 * leg,
            )
        )
    return np.stack(rows)


def _overlap_coincidence_spikes(
    *,
    slot_identity: np.ndarray,
    ingress_identity: np.ndarray,
) -> np.ndarray:
    footprint_offset = spec.FACTOR_COUNT + 2 * spec._COMPOSITOR_BOUNDARY_CELL_COUNT
    stored = slot_identity[:, footprint_offset:].astype(np.float64)
    ingress = ingress_identity[footprint_offset:].astype(np.float64)
    currents = stored * _HALF_CURRENT_PA + ingress[None, :] * _HALF_CURRENT_PA
    return _threshold_spikes(
        currents.reshape(-1),
        threshold_pa=_FULL_CURRENT_PA,
    ).reshape(spec.EVENT_CAPACITY, spec._COMPOSITOR_TOKEN_POSITION_CELL_COUNT)


class _CompositorPersistentState:
    """Case-persistent compositor storage implemented only by no-leak populations."""

    __slots__ = ("_identity", "_occupied", "_poison", "_ready")

    def __init__(self) -> None:
        self._identity = spec._NoLeakStorageReadoutPopulation(
            spec.EVENT_CAPACITY * spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
        )
        self._occupied = spec._NoLeakStorageReadoutPopulation(spec.EVENT_CAPACITY)
        self._poison = spec._AbsorbingCompositorPoisonPopulation()
        self._ready = spec._NoLeakStorageReadoutPopulation(1)
        self.initialize_ready()

    @property
    def poison(self) -> spec._AbsorbingCompositorPoisonPopulation:
        return self._poison

    @property
    def ready_charge_pa(self) -> tuple[float, ...]:
        return self._ready.storage_charge_pa

    def initialize_ready(self) -> None:
        self._ready.step(
            set_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )

    def rearm_ready_from_completion(self, completion_relay_spike: bool) -> None:
        if type(completion_relay_spike) is not bool:
            raise ValueError("compositor completion relay domain changed")
        self._ready.step(
            set_currents_pa=np.asarray((_FULL_CURRENT_PA if completion_relay_spike else 0.0,)),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )

    def read_ready(self) -> bool:
        return bool(
            self._ready.step(
                set_currents_pa=np.zeros(1),
                read_gate_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
                direct_readout_currents_pa=np.zeros(1),
                reset_currents_pa=np.zeros(1),
            )[0]
        )

    def read_poison_halt(self) -> bool:
        return bool(
            self._poison._halt.step(
                set_currents_pa=np.zeros(1),
                read_gate_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
                direct_readout_currents_pa=np.zeros(1),
                reset_currents_pa=np.zeros(1),
            )[0]
        )

    def read_poison_status(self) -> np.ndarray:
        return np.asarray(
            self._poison._status.step(
                set_currents_pa=np.zeros(2),
                read_gate_currents_pa=np.full(2, _FULL_CURRENT_PA),
                direct_readout_currents_pa=np.zeros(2),
                reset_currents_pa=np.zeros(2),
            ),
            dtype=np.bool_,
        )

    def clear_ready(self) -> None:
        self._ready.step(
            set_currents_pa=np.zeros(1),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.asarray((v4spec.BASKET_INHIBITION_CURRENT_PA,)),
        )

    def read_slots(self) -> tuple[np.ndarray, np.ndarray]:
        identity_size = spec.EVENT_CAPACITY * spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
        identity = self._identity.step(
            set_currents_pa=np.zeros(identity_size),
            read_gate_currents_pa=np.full(identity_size, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(identity_size),
            reset_currents_pa=np.zeros(identity_size),
        ).reshape(
            spec.EVENT_CAPACITY,
            spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
        )
        occupied = self._occupied.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.full(spec.EVENT_CAPACITY, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )
        return np.asarray(identity, dtype=np.bool_), np.asarray(occupied, dtype=np.bool_)

    def set_copy(self, copied: np.ndarray) -> None:
        copy_spikes = np.asarray(copied)
        if (
            copy_spikes.shape
            != (
                spec.EVENT_CAPACITY,
                spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT + 1,
            )
            or copy_spikes.dtype != np.bool_
            or np.count_nonzero(copy_spikes[:, -1]) > 1
        ):
            raise ValueError("compositor persistent copy domain changed")
        identity_spikes = copy_spikes[:, :-1].reshape(-1)
        occupied_spikes = copy_spikes[:, -1]
        self._identity.step(
            set_currents_pa=identity_spikes.astype(np.float64) * _FULL_CURRENT_PA,
            read_gate_currents_pa=np.zeros(identity_spikes.size),
            direct_readout_currents_pa=np.zeros(identity_spikes.size),
            reset_currents_pa=np.zeros(identity_spikes.size),
        )
        self._occupied.step(
            set_currents_pa=occupied_spikes.astype(np.float64) * _FULL_CURRENT_PA,
            read_gate_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )

    def reset_slots(self) -> None:
        identity_size = spec.EVENT_CAPACITY * spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
        self._identity.step(
            set_currents_pa=np.zeros(identity_size),
            read_gate_currents_pa=np.zeros(identity_size),
            direct_readout_currents_pa=np.zeros(identity_size),
            reset_currents_pa=np.full(
                identity_size,
                v4spec.BASKET_INHIBITION_CURRENT_PA,
            ),
        )
        self._occupied.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.full(
                spec.EVENT_CAPACITY,
                v4spec.BASKET_INHIBITION_CURRENT_PA,
            ),
        )

    def reset_case_from_relay(self, case_reset_relay_spike: bool) -> None:
        if type(case_reset_relay_spike) is not bool:
            raise ValueError("compositor case-reset relay domain changed")
        if case_reset_relay_spike:
            self.reset_slots()
            self._poison.step(
                overlap_decision=False,
                capacity_decision=False,
                case_reset=True,
            )
            self.clear_ready()

    def active_count(self, *, include_ready: bool) -> int:
        return int(
            sum(value != 0.0 for value in self._identity.storage_charge_pa)
            + sum(value != 0.0 for value in self._occupied.storage_charge_pa)
            + sum(value != 0.0 for value in self._poison.status_charge_pa)
            + (sum(value != 0.0 for value in self._ready.storage_charge_pa) if include_ready else 0)
        )


def _execute_compositor(
    *,
    source_sha256: str,
    events: tuple[spec.ProtocolFactorEvent, ...],
    persistent_state: _CompositorPersistentState | None = None,
    factor_relay_spike_cells: tuple[tuple[int, ...], ...] | None = None,
) -> CompositorReplay:
    if (
        not _is_sha256(source_sha256)
        or type(events) is not tuple
        or any(not isinstance(event, spec.ProtocolFactorEvent) for event in events)
    ):
        raise ValueError("compositor execution input domain changed")
    if factor_relay_spike_cells is not None and len(factor_relay_spike_cells) != len(events):
        raise ValueError("compositor physical factor relay/event count changed")
    ingress_rows = tuple(
        _ingress_identity(event)
        if factor_relay_spike_cells is None
        else _ingress_identity_from_factor_relay(event, relay_cells)
        for event, relay_cells in zip(
            events,
            factor_relay_spike_cells
            if factor_relay_spike_cells is not None
            else ((),) * len(events),
            strict=True,
        )
    )
    state = persistent_state if persistent_state is not None else _CompositorPersistentState()
    if state.active_count(include_ready=False) != 0 or state.ready_charge_pa != (_FULL_CURRENT_PA,):
        raise RuntimeError("compositor case did not enter from reset/ready state")
    poison = state.poison
    phase = spec._CompositorPhaseRequestPopulation()
    copy_population = spec._BufferedIngressSlotCopyPopulation()
    duplicate_count = 0
    accepted_count = 0
    event_rows: list[dict[str, object]] = []
    for event_index, (event, ingress) in enumerate(zip(events, ingress_rows, strict=True)):
        ready_spike = state.read_ready()
        poison_halt_spike = state.read_poison_halt()
        request_spikes = phase.evaluate(
            request_spikes=np.asarray((True, False), dtype=np.bool_),
            event_ready_storage_pa=np.asarray((_FULL_CURRENT_PA if ready_spike else 0.0,)),
        )
        ingress_gate_currents = (
            ingress.astype(np.float64) * _HALF_CURRENT_PA
            + (request_spikes[0].astype(np.float64) * _HALF_CURRENT_PA)
            + (v4spec.BASKET_INHIBITION_CURRENT_PA if poison_halt_spike else 0.0)
        )
        ingress_gate_spikes = _threshold_spikes(
            ingress_gate_currents,
            threshold_pa=_FULL_CURRENT_PA,
        )
        accepted_factor_gate = bool(np.any(ingress_gate_spikes[: spec.FACTOR_COUNT]))
        if not accepted_factor_gate:
            event_rows.append(
                {
                    "event_index": event_index,
                    "event_sha256": event.event_sha256,
                    "request_tick": 11,
                    "gate_tick": 12,
                    "ingress_halted": True,
                    "ready_read_spike": ready_spike,
                    "poison_halt_read_spike": poison_halt_spike,
                    "stage_spike_counts": (
                        (-1, int(ready_spike) + int(poison_halt_spike)),
                        (0, int(np.count_nonzero(ingress_gate_spikes))),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 0),
                        (7, 0),
                        (8, 0),
                        (9, 0),
                        (10, 0),
                    ),
                    "stage_evaluated_counts": (
                        (-1, 2),
                        (0, spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 0),
                        (7, 0),
                        (8, 0),
                        (9, 0),
                        (10, 0),
                    ),
                    "poison_status_pa_hex": tuple(
                        float(value).hex() for value in poison.status_charge_pa
                    ),
                }
            )
            continue
        ingress_latch = spec._NoLeakStorageReadoutPopulation(
            spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
        )
        ingress_latch.step(
            set_currents_pa=ingress_gate_spikes.astype(np.float64) * _FULL_CURRENT_PA,
            read_gate_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            direct_readout_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            reset_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
        )
        state.clear_ready()
        slots, occupied_readout = state.read_slots()
        occupancy_snapshot_latch = spec._NoLeakStorageReadoutPopulation(spec.EVENT_CAPACITY)
        occupancy_snapshot_latch.step(
            set_currents_pa=occupied_readout.astype(np.float64) * _FULL_CURRENT_PA,
            read_gate_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )
        duplicate_keys = _duplicate_key_spikes(
            slot_identity=slots,
            ingress_identity=ingress_gate_spikes,
        )
        overlap_cells = _overlap_coincidence_spikes(
            slot_identity=slots,
            ingress_identity=ingress_gate_spikes,
        )
        duplicate_or = bool(
            _threshold_spikes(
                np.asarray((_FULL_CURRENT_PA * np.count_nonzero(duplicate_keys),)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        overlap_or = bool(
            _threshold_spikes(
                np.asarray((_FULL_CURRENT_PA * np.count_nonzero(overlap_cells),)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        duplicate_decision = bool(
            _threshold_spikes(
                np.asarray((_FULL_CURRENT_PA if duplicate_or else 0.0,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        overlap_current = (_FULL_CURRENT_PA if overlap_or else 0.0) + (
            v4spec.BASKET_INHIBITION_CURRENT_PA if duplicate_or else 0.0
        )
        overlap_decision = bool(
            _threshold_spikes(
                np.asarray((overlap_current,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        occupancy_for_capacity = occupancy_snapshot_latch.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.full(spec.EVENT_CAPACITY, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )
        capacity_current = np.count_nonzero(occupancy_for_capacity) * _HALF_CURRENT_PA + (
            v4spec.BASKET_INHIBITION_CURRENT_PA if duplicate_decision or overlap_decision else 0.0
        )
        capacity_decision = bool(
            _threshold_spikes(
                np.asarray((capacity_current,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        if sum((duplicate_decision, overlap_decision, capacity_decision)) > 1:
            raise RuntimeError("compositor classification lost precedence")
        status_latches, halted = poison.step(
            overlap_decision=overlap_decision,
            capacity_decision=capacity_decision,
        )
        admission_block_latch = spec._NoLeakStorageReadoutPopulation(1)
        classification_spike_count = int(
            duplicate_decision or overlap_decision or capacity_decision
        )
        admission_block_latch.step(
            set_currents_pa=np.asarray((_FULL_CURRENT_PA if classification_spike_count else 0.0,)),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )
        poison_slot_reset_relay = bool(
            _threshold_spikes(
                np.asarray((_FULL_CURRENT_PA if overlap_decision or capacity_decision else 0.0,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        if duplicate_decision:
            duplicate_count += 1
        if poison_slot_reset_relay:
            state.reset_slots()
        occupancy_for_slot_zero = occupancy_snapshot_latch.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.full(spec.EVENT_CAPACITY, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )
        block_for_slot_zero = admission_block_latch.step(
            set_currents_pa=np.zeros(1),
            read_gate_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )
        slot_zero_current = (
            _FULL_CURRENT_PA
            - occupancy_for_slot_zero[0].astype(np.float64) * _FULL_CURRENT_PA
            - block_for_slot_zero[0].astype(np.float64) * _FULL_CURRENT_PA
        )
        slot_zero_spike = bool(
            _threshold_spikes(
                np.asarray((slot_zero_current,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        occupancy_for_slot_one = occupancy_snapshot_latch.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.full(spec.EVENT_CAPACITY, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.zeros(spec.EVENT_CAPACITY),
        )
        block_for_slot_one = admission_block_latch.step(
            set_currents_pa=np.zeros(1),
            read_gate_currents_pa=np.asarray((_FULL_CURRENT_PA,)),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.zeros(1),
        )
        slot_one_current = (
            occupancy_for_slot_one[0].astype(np.float64) * _FULL_CURRENT_PA
            - occupancy_for_slot_one[1].astype(np.float64) * _FULL_CURRENT_PA
            - block_for_slot_one[0].astype(np.float64) * _FULL_CURRENT_PA
        )
        slot_one_spike = bool(
            _threshold_spikes(
                np.asarray((slot_one_current,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        admission_spikes = np.asarray((slot_zero_spike, slot_one_spike), dtype=np.bool_)
        admitted_slot_relays = _threshold_spikes(
            admission_spikes.astype(np.float64) * _FULL_CURRENT_PA,
            threshold_pa=_FULL_CURRENT_PA,
        )
        ingress_readout = ingress_latch.step(
            set_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            read_gate_currents_pa=np.full(
                spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
                _FULL_CURRENT_PA,
            ),
            direct_readout_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            reset_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
        )
        copied = copy_population.evaluate(
            ingress_readout_spikes=ingress_readout,
            admitted_slot_spikes=admitted_slot_relays,
        )
        completion_relay = bool(
            _threshold_spikes(
                np.asarray((_FULL_CURRENT_PA,)),
                threshold_pa=_FULL_CURRENT_PA,
            )[0]
        )
        state.set_copy(copied)
        accepted_count += int(np.count_nonzero(copied[:, -1]))
        ingress_latch.step(
            set_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            read_gate_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            direct_readout_currents_pa=np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
            reset_currents_pa=np.full(
                spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
                v4spec.BASKET_INHIBITION_CURRENT_PA,
            ),
        )
        occupancy_snapshot_latch.step(
            set_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            read_gate_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            direct_readout_currents_pa=np.zeros(spec.EVENT_CAPACITY),
            reset_currents_pa=np.full(
                spec.EVENT_CAPACITY,
                v4spec.BASKET_INHIBITION_CURRENT_PA,
            ),
        )
        admission_block_latch.step(
            set_currents_pa=np.zeros(1),
            read_gate_currents_pa=np.zeros(1),
            direct_readout_currents_pa=np.zeros(1),
            reset_currents_pa=np.asarray((v4spec.BASKET_INHIBITION_CURRENT_PA,)),
        )
        if not completion_relay:
            raise RuntimeError("compositor completion relay failed to fire")
        state.rearm_ready_from_completion(completion_relay)
        post_slots, post_occupied = state.read_slots()
        if any(ingress_latch.storage_charge_pa) or state.ready_charge_pa != (_FULL_CURRENT_PA,):
            raise RuntimeError("compositor ephemeral reset/ready rearm failed")
        event_rows.append(
            {
                "event_index": event_index,
                "event_sha256": event.event_sha256,
                "request_tick": 11,
                "gate_tick": 12,
                "ingress_halted": False,
                "duplicate_key_spike_count": int(np.count_nonzero(duplicate_keys)),
                "overlap_coincidence_spike_count": int(np.count_nonzero(overlap_cells)),
                "duplicate_or": duplicate_or,
                "overlap_or": overlap_or,
                "duplicate_decision": duplicate_decision,
                "overlap_decision": overlap_decision,
                "capacity_decision": capacity_decision,
                "poison_status": status_latches,
                "case_halted": halted,
                "occupancy_snapshot": tuple(bool(value) for value in occupied_readout),
                "poison_slot_reset_relay_spike": poison_slot_reset_relay,
                "admission_block_storage_set": bool(classification_spike_count),
                "slot_zero_admission_spike": slot_zero_spike,
                "slot_one_admission_spike": slot_one_spike,
                "admitted_slot_relay_spikes": tuple(bool(value) for value in admitted_slot_relays),
                "copy_spike_count": int(np.count_nonzero(copied)),
                "completion_relay_spike": completion_relay,
                "post_event_occupied": tuple(bool(value) for value in post_occupied),
                "post_event_identity_spike_count": int(np.count_nonzero(post_slots)),
                "stage_spike_counts": (
                    (-1, int(ready_spike) + int(poison_halt_spike)),
                    (
                        0,
                        int(np.count_nonzero(ingress_gate_spikes))
                        + int(np.count_nonzero(occupied_readout)),
                    ),
                    (
                        1,
                        int(np.count_nonzero(ingress_gate_spikes))
                        + int(np.count_nonzero(occupied_readout))
                        + int(np.count_nonzero(duplicate_keys))
                        + int(np.count_nonzero(overlap_cells)),
                    ),
                    (2, int(duplicate_or) + int(overlap_or)),
                    (
                        3,
                        int(duplicate_decision)
                        + int(overlap_decision)
                        + int(np.count_nonzero(occupancy_for_capacity)),
                    ),
                    (4, int(capacity_decision)),
                    (5, classification_spike_count + int(poison_slot_reset_relay)),
                    (
                        6,
                        int(slot_zero_spike)
                        + int(np.count_nonzero(occupancy_for_slot_zero))
                        + int(np.count_nonzero(block_for_slot_zero)),
                    ),
                    (
                        7,
                        int(slot_one_spike)
                        + int(np.count_nonzero(occupancy_for_slot_one))
                        + int(np.count_nonzero(block_for_slot_one)),
                    ),
                    (
                        8,
                        int(np.count_nonzero(admitted_slot_relays))
                        + int(np.count_nonzero(ingress_readout)),
                    ),
                    (9, int(np.count_nonzero(copied)) + int(completion_relay)),
                    (10, int(np.count_nonzero(copied[:, -1])) + 1),
                ),
                "stage_evaluated_counts": (
                    (-1, 2),
                    (0, spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
                    (
                        1,
                        spec._COMPOSITOR_DUPLICATE_KEY_CELL_COUNT
                        + spec._COMPOSITOR_OVERLAP_COINCIDENCE_CELL_COUNT
                        + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
                        + spec.EVENT_CAPACITY,
                    ),
                    (2, 2),
                    (3, 2 + spec.EVENT_CAPACITY),
                    (4, 1),
                    (5, 4),
                    (6, 1 + spec.EVENT_CAPACITY + 1),
                    (7, 1 + spec.EVENT_CAPACITY + 1),
                    (8, spec.EVENT_CAPACITY + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
                    (9, spec._COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT + 1),
                    (
                        10,
                        spec._COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT
                        + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
                        + spec.EVENT_CAPACITY
                        + 2,
                    ),
                ),
                "ephemeral_reset_relay_tick": 9,
                "ephemeral_reset_arrival_tick": 10,
                "ready_rearm_tick": 10,
            }
        )
    eos_ready_spike = state.read_ready()
    eos_request = phase.evaluate(
        request_spikes=np.asarray((False, True), dtype=np.bool_),
        event_ready_storage_pa=np.asarray((_FULL_CURRENT_PA if eos_ready_spike else 0.0,)),
    )
    if tuple(bool(value) for value in eos_request) != (False, True):
        raise RuntimeError("compositor EOS request barrier failed")
    eos = spec._CompositorEosStatusPopulation()
    slots, occupied = state.read_slots()
    poison_readout = state.read_poison_status()
    poison_storage = poison_readout.astype(np.float64) * _FULL_CURRENT_PA
    occupied_storage = occupied.astype(np.float64) * _FULL_CURRENT_PA
    status_candidates = eos.evaluate_status(
        occupied_storage_pa=occupied_storage,
        poison_storage_pa=poison_storage,
    )
    status_spikes = _threshold_spikes(
        status_candidates.astype(np.float64) * _FULL_CURRENT_PA,
        threshold_pa=_FULL_CURRENT_PA,
    )
    factor_rows = slots[:, : spec.FACTOR_COUNT].copy()
    factor_outputs = eos.evaluate_factor_outputs(
        factor_readout_spikes=factor_rows,
        status_spikes=status_spikes,
    )
    status_cell = int(np.flatnonzero(status_spikes)[0])
    status = _STATUS_ORDER[status_cell]
    factors = tuple(
        spec.FACTOR_REGISTRY[int(np.flatnonzero(row)[0])] for row in factor_outputs if np.any(row)
    )
    case_reset_relay = bool(
        _threshold_spikes(
            np.asarray((_FULL_CURRENT_PA * np.count_nonzero(status_spikes),)),
            threshold_pa=_FULL_CURRENT_PA,
        )[0]
    )
    state.reset_case_from_relay(case_reset_relay)
    post_reset_active = state.active_count(include_ready=True)
    if post_reset_active != 0 or poison.ingress_allowed is not True:
        raise RuntimeError("compositor case reset left active state")
    if not case_reset_relay:
        raise RuntimeError("compositor EOS case-reset relay did not fire")
    state.initialize_ready()
    if state.ready_charge_pa != (_FULL_CURRENT_PA,) or state.active_count(include_ready=False) != 0:
        raise RuntimeError("compositor next-case ready initialization failed")
    trace_record = {
        "source_sha256": source_sha256,
        "events": tuple(event.event_sha256 for event in events),
        "event_rows": tuple(event_rows),
        "eos_request": tuple(bool(value) for value in eos_request),
        "eos_read_tick": 0,
        "eos_candidate_tick": 1,
        "eos_candidate_spikes": tuple(bool(value) for value in status_candidates),
        "eos_status_tick": 2,
        "eos_status_spikes": tuple(bool(value) for value in status_spikes),
        "factor_output_spikes": tuple(
            tuple(int(cell) for cell in np.flatnonzero(row)) for row in factor_outputs
        ),
        "case_reset_relay_tick": 3,
        "case_reset_relay_spike": case_reset_relay,
        "case_reset_arrival_tick": 4,
        "next_case_ready_initialization_tick": 5,
        "next_case_ready_storage_pa_hex": tuple(
            float(value).hex() for value in state.ready_charge_pa
        ),
        "post_reset_active_count": post_reset_active,
    }
    return CompositorReplay(
        source_sha256=source_sha256,
        input_event_sha256s=tuple(event.event_sha256 for event in events),
        status=status,
        factors=factors,
        attempted_event_count=len(events),
        accepted_event_count=accepted_count,
        duplicate_count=duplicate_count,
        overlap_poison=status_spikes[3].item(),
        capacity_poison=status_spikes[4].item(),
        eos_status_spikes=tuple(bool(value) for value in status_spikes),
        post_reset_active_count=post_reset_active,
        event_stage_spike_counts=tuple(
            tuple(tuple(pair) for pair in row["stage_spike_counts"]) for row in event_rows
        ),
        event_stage_evaluated_counts=tuple(
            tuple(tuple(pair) for pair in row["stage_evaluated_counts"]) for row in event_rows
        ),
        case_reset_relay_tick=3,
        case_reset_arrival_tick=4,
        next_case_ready_initialization_tick=5,
        trace_sha256=_namespaced_sha256(_COMPOSITOR_TRACE_NAMESPACE, trace_record),
    )


def _execute_latch_preflight() -> dict[str, object]:
    latch = spec._NoLeakStorageReadoutPopulation(1)
    zero = np.zeros(1)
    full = np.asarray((_FULL_CURRENT_PA,))
    reset = np.asarray((v4spec.BASKET_INHIBITION_CURRENT_PA,))
    latch.step(
        set_currents_pa=full,
        read_gate_currents_pa=zero,
        direct_readout_currents_pa=zero,
        reset_currents_pa=zero,
    )
    for _ in range(1024):
        idle = latch.step(
            set_currents_pa=zero,
            read_gate_currents_pa=zero,
            direct_readout_currents_pa=zero,
            reset_currents_pa=zero,
        )
        if bool(idle[0]):
            raise RuntimeError("persistent latch leaked a readout spike while idle")
    repeated_reads = tuple(
        bool(
            latch.step(
                set_currents_pa=zero,
                read_gate_currents_pa=full,
                direct_readout_currents_pa=zero,
                reset_currents_pa=zero,
            )[0]
        )
        for _ in range(2)
    )
    if repeated_reads != (True, True) or latch.storage_charge_pa != (_FULL_CURRENT_PA,):
        raise RuntimeError("persistent latch repeated read changed")
    reset_spike = bool(
        latch.step(
            set_currents_pa=full,
            read_gate_currents_pa=full,
            direct_readout_currents_pa=np.asarray((2.0 * _FULL_CURRENT_PA,)),
            reset_currents_pa=reset,
        )[0]
    )
    if reset_spike or latch.storage_charge_pa != (0.0,):
        raise RuntimeError("persistent latch reset dominance changed")
    rejection_rows: list[str] = []
    invalid_vectors = (
        ("set", np.asarray((2500.0,)), zero, zero),
        ("read", zero, np.asarray((10000.0,)), zero),
        ("direct", zero, zero, np.asarray((15000.0,))),
    )
    for name, set_current, read_current, direct_current in invalid_vectors:
        before = latch.storage_charge_pa
        try:
            latch.step(
                set_currents_pa=set_current,
                read_gate_currents_pa=read_current,
                direct_readout_currents_pa=direct_current,
                reset_currents_pa=zero,
            )
        except ValueError:
            rejection_rows.append(name)
        else:
            raise RuntimeError("persistent latch accepted an invalid current")
        if latch.storage_charge_pa != before:
            raise RuntimeError("persistent latch invalid current mutated storage")
    direct_spike = bool(
        latch.step(
            set_currents_pa=zero,
            read_gate_currents_pa=zero,
            direct_readout_currents_pa=np.asarray((2.0 * _FULL_CURRENT_PA,)),
            reset_currents_pa=zero,
        )[0]
    )
    if not direct_spike or latch.storage_charge_pa != (0.0,):
        raise RuntimeError("direct two-principal collision channel changed")
    return {
        "idle_tick_count": 1024,
        "repeated_read_spikes": repeated_reads,
        "reset_dominance_spike": reset_spike,
        "invalid_current_rejections": tuple(rejection_rows),
        "direct_collision_spike": direct_spike,
        "final_storage_pa_hex": tuple(float(value).hex() for value in latch.storage_charge_pa),
    }


def _execute_selector_preflight(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
) -> tuple[dict[str, object], ...]:
    lookup = _assembly_lookup(protocol)
    rows: list[dict[str, object]] = []
    expected_rows = tuple(spec.SELECTOR_FULL_PREFLIGHT_RECORD["controls"])
    neutral_tokens = protocol.neutral_spans[0].tokens
    for expected in expected_rows:
        name = str(expected["name"])
        if expected["slot_tokens"]:
            tokens = tuple(expected["slot_tokens"])
            observed = _close_selector_preflight(
                _run_selector(
                    protocol,
                    tokens,
                    source_sha256=_namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, name),
                )
            )
        elif name == "incompatible-two-cue-collision-stop":
            cells = [lookup[token].cells for token in neutral_tokens]
            cells[0] = tuple(
                sorted(
                    set(lookup[spec.CUE_TOKENS["NEXT"]].cells)
                    | set(lookup[spec.CUE_TOKENS["PREV"]].cells)
                )
            )
            tokens = ("simultaneous-next-prev", *neutral_tokens[1:])
            observed = _close_selector_preflight(
                _run_selector_assemblies(
                    protocol,
                    slot_assembly_cells=tuple(cells),
                    slot_tokens=tokens,
                    source_sha256=_namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, name),
                )
            )
        elif name == "same-address-double-cue-collision-stop":
            tokens = (spec.CUE_TOKENS["NEXT"], *neutral_tokens[1:])
            observed = _close_selector_preflight(
                _run_selector_assemblies(
                    protocol,
                    slot_assembly_cells=tuple(lookup[token].cells for token in tokens),
                    slot_tokens=tokens,
                    source_sha256=_namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, name),
                    serial_repeats=((0, 1),),
                )
            )
        else:
            tokens = neutral_tokens
            observed = _close_selector_preflight(
                _run_selector(
                    protocol,
                    tokens,
                    source_sha256=_namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, name),
                )
            )
        if (
            observed.decision != expected["decision"]
            or observed.selected_slot != expected["selected_slot"]
            or observed.principal_first_spike_tier != tuple(expected["principal_first_spike_tier"])
            or observed.selected_trace_cells != tuple(expected["selected_lexical_relay_cells"])
            or observed.address_event_cells
            != tuple(cell for _tick, cell in expected["address_coincidence_spikes"])
            or observed.address_read_counts
            != tuple(
                v4spec.RACE_STEPS if cell in tuple(expected["address_current_latches_set"]) else 0
                for cell in range(spec.SELECTOR_ADDRESS_COUNT)
            )
            or observed.address_hard_reset_count != expected["address_hard_reset_count"]
            or observed.scan_zero_clamp_count
            != (5 if name == "same-address-double-cue-collision-stop" else 4)
            or observed.basket_tick != expected["basket_tick"]
            or observed.collision_tick != expected["collision_tick"]
            or observed.decision_tick != expected["decision_tick"]
            or observed.selected_lexical_relay_tick != expected["selected_lexical_relay_tick"]
            or observed.selected_slot_post_reset_active_count != 0
        ):
            raise RuntimeError(f"selector preflight control changed: {name}")
        rows.append({"name": name, "observed": observed.record()})
    base = expected_rows[0]
    base_tokens = tuple(base["slot_tokens"])
    base_source = _namespaced_sha256(_SELECTOR_TRACE_NAMESPACE, "causal-base")
    interventions = {
        "circuit_off": _close_selector_preflight(
            _run_selector(
                protocol,
                base_tokens,
                source_sha256=base_source,
                circuit_gain=0.0,
            )
        ),
        "phase_leg_off": _close_selector_preflight(
            _run_selector(
                protocol,
                base_tokens,
                source_sha256=base_source,
                phase_gain=0.0,
            )
        ),
        "cue_leg_off": _close_selector_preflight(
            _run_selector(
                protocol,
                base_tokens,
                source_sha256=base_source,
                cue_gain=0.0,
            )
        ),
        "latch_persistence_off": _close_selector_preflight(
            _run_selector(
                protocol,
                base_tokens,
                source_sha256=base_source,
                latch_persistence=False,
            )
        ),
        "lexical_gate_off": _close_selector_preflight(
            _run_selector(
                protocol,
                base_tokens,
                source_sha256=base_source,
                lexical_gain=0.0,
            )
        ),
    }
    if (
        interventions["circuit_off"].decision != "SILENT"
        or interventions["phase_leg_off"].decision != "STOP"
        or interventions["cue_leg_off"].decision != "STOP"
        or interventions["latch_persistence_off"].decision != "SELECTED_SLOT"
        or interventions["latch_persistence_off"].selected_trace_cells
        or interventions["lexical_gate_off"].decision != "SELECTED_SLOT"
        or interventions["lexical_gate_off"].selected_trace_cells
    ):
        raise RuntimeError("selector causal intervention panel changed")
    incompatible = next(
        row["observed"] for row in rows if row["name"] == "incompatible-two-cue-collision-stop"
    )
    repeated = next(
        row["observed"] for row in rows if row["name"] == "same-address-double-cue-collision-stop"
    )
    if (
        not incompatible["direct_collision_spike"]
        or incompatible["stored_repeat_collision_spike"]
        or incompatible["collision_tick"] is None
        or repeated["direct_collision_spike"]
        or not repeated["stored_repeat_collision_spike"]
        or repeated["collision_tick"] is None
    ):
        raise RuntimeError("selector collision provenance channels changed")
    residual_probe = _ResettableCoincidencePopulation(
        spec.SELECTOR_ADDRESS_COUNT,
        threshold_pa=_FULL_CURRENT_PA * (spec.ASSEMBLY_ACTIVE_CELL_COUNT + 1),
    )
    cue_only = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
    cue_only[0] = spec.ASSEMBLY_ACTIVE_CELL_COUNT * _FULL_CURRENT_PA
    cue_spikes = residual_probe.step(cue_only)
    residual_probe.hard_reset()
    phase_only = np.zeros(spec.SELECTOR_ADDRESS_COUNT)
    phase_only[0] = _FULL_CURRENT_PA
    phase_spikes = residual_probe.step(phase_only)
    residual_probe.hard_reset()
    if np.any(cue_spikes) or np.any(phase_spikes) or residual_probe.active_membrane_count:
        raise RuntimeError("selector serial reset allowed residual coincidence")
    rows.append(
        {
            "name": "serial-residual-current-rejection",
            "observed": {
                "cue_only_spike_count": int(np.count_nonzero(cue_spikes)),
                "phase_only_spike_count": int(np.count_nonzero(phase_spikes)),
                "post_reset_active_count": residual_probe.active_membrane_count,
            },
        }
    )
    rows.extend(
        {"name": name, "observed": replay.record()} for name, replay in interventions.items()
    )
    return tuple(rows)


def _execute_teacher_scratch(
    *,
    source_sha256: str,
    raw_scores: tuple[int, ...],
    teacher_factor: str,
    teacher_gain: float,
    selected_trace_spikes: np.ndarray,
) -> dict[str, object]:
    selected = np.asarray(selected_trace_spikes)
    if (
        selected.shape != (spec.LEXICAL_CELL_COUNT,)
        or selected.dtype != np.bool_
        or np.count_nonzero(selected) not in (0, spec.ASSEMBLY_ACTIVE_CELL_COUNT)
    ):
        raise ValueError("teacher scratch selected trace changed")
    selected_latch = spec._NoLeakStorageReadoutPopulation(spec.LEXICAL_CELL_COUNT)
    selected_latch.step(
        set_currents_pa=selected.astype(np.float64) * _FULL_CURRENT_PA,
        read_gate_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
    )
    decision, autonomous_latch, scout_latch, factor_trace = _run_training_factor_circuit(
        source_sha256=source_sha256,
        raw_scores=raw_scores,
    )
    teacher_spikes = _teacher_spike_mask(teacher_factor, gain=teacher_gain)
    autonomous = autonomous_latch.step(
        set_currents_pa=np.zeros(spec.FACTOR_COUNT),
        read_gate_currents_pa=np.full(spec.FACTOR_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
        reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
    )
    scouts = scout_latch.step(
        set_currents_pa=np.zeros(spec.FACTOR_COUNT),
        read_gate_currents_pa=np.full(spec.FACTOR_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.FACTOR_COUNT),
        reset_currents_pa=np.zeros(spec.FACTOR_COUNT),
    )
    matches = _target_match_relays(
        teacher_spikes=teacher_spikes,
        autonomous_readout_spikes=autonomous,
    )
    e_updates, h_updates = _teacher_scout_update_relays(
        teacher_spikes=teacher_spikes,
        scout_readout_spikes=scouts,
        target_match_spikes=matches,
    )
    trace_read = selected_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.full(spec.LEXICAL_CELL_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
    )
    for latch, size in (
        (selected_latch, spec.LEXICAL_CELL_COUNT),
        (autonomous_latch, spec.FACTOR_COUNT),
        (scout_latch, spec.FACTOR_COUNT),
    ):
        latch.step(
            set_currents_pa=np.zeros(size),
            read_gate_currents_pa=np.zeros(size),
            direct_readout_currents_pa=np.zeros(size),
            reset_currents_pa=np.full(size, v4spec.BASKET_INHIBITION_CURRENT_PA),
        )
        if any(latch.storage_charge_pa):
            raise RuntimeError("teacher scratch reset left persistent charge")
    return {
        "source_sha256": source_sha256,
        "factor_trace_sha256": factor_trace,
        "decision": decision.record(),
        "teacher_spikes": tuple(int(cell) for cell in np.flatnonzero(teacher_spikes)),
        "autonomous_readout_spikes": tuple(int(cell) for cell in np.flatnonzero(autonomous)),
        "scout_readout_spikes": tuple(int(cell) for cell in np.flatnonzero(scouts)),
        "target_match_spikes": tuple(int(cell) for cell in np.flatnonzero(matches)),
        "excitatory_update_spikes": tuple(int(cell) for cell in np.flatnonzero(e_updates)),
        "inhibitory_update_spikes": tuple(int(cell) for cell in np.flatnonzero(h_updates)),
        "selected_trace_read_spike_count": int(np.count_nonzero(trace_read)),
        "committed_excitatory_tick_count": int(
            np.count_nonzero(e_updates) * np.count_nonzero(trace_read)
        ),
        "committed_inhibitory_tick_count": int(
            np.count_nonzero(h_updates) * np.count_nonzero(trace_read)
        ),
        "teacher_tick": 1134,
        "read_tick": 1135,
        "match_tick": 1136,
        "update_tick": 1137,
        "commit_tick": 1138,
        "reset_tick": 1139,
        "post_reset_active_count": 0,
    }


def _execute_teacher_preflight() -> tuple[dict[str, object], ...]:
    target = _FACTOR_INDEX[spec.TRIADS[0][0]]
    rivals = tuple(_FACTOR_INDEX[factor] for factor in spec.TRIADS[0][1:])
    selected = np.zeros(spec.LEXICAL_CELL_COUNT, dtype=np.bool_)
    selected[: spec.ASSEMBLY_ACTIVE_CELL_COUNT] = True
    rows: list[dict[str, object]] = []
    specifications: tuple[tuple[str, tuple[int, ...], str, float, np.ndarray], ...] = (
        (
            "score-12-target-stop-tie",
            tuple(12 if cell == target else 0 for cell in range(spec.FACTOR_COUNT)),
            spec.TRIADS[0][0],
            1.0,
            selected,
        ),
        (
            "score-24-correct-unique",
            tuple(24 if cell == target else 0 for cell in range(spec.FACTOR_COUNT)),
            spec.TRIADS[0][0],
            1.0,
            selected,
        ),
        (
            "two-rival-scratch",
            tuple(24 if cell in rivals else 0 for cell in range(spec.FACTOR_COUNT)),
            spec.TRIADS[0][0],
            1.0,
            selected,
        ),
        (
            "nonselected-distractor-scratch",
            tuple(6 if cell == rivals[0] else 0 for cell in range(spec.FACTOR_COUNT)),
            spec.TRIADS[0][0],
            1.0,
            selected,
        ),
        (
            "teacher-off",
            (0,) * spec.FACTOR_COUNT,
            spec.TRIADS[0][0],
            0.0,
            selected,
        ),
        (
            "cell-off",
            (0,) * spec.FACTOR_COUNT,
            spec.TRIADS[0][0],
            1.0,
            np.zeros(spec.LEXICAL_CELL_COUNT, dtype=np.bool_),
        ),
    )
    for name, raw_scores, teacher_factor, teacher_gain, trace_spikes in specifications:
        rows.append(
            {
                "name": name,
                "observed": _execute_teacher_scratch(
                    source_sha256=_namespaced_sha256(_TEACHER_TRACE_NAMESPACE, name),
                    raw_scores=raw_scores,
                    teacher_factor=teacher_factor,
                    teacher_gain=teacher_gain,
                    selected_trace_spikes=trace_spikes,
                ),
            }
        )
    observed = {row["name"]: row["observed"] for row in rows}
    if (
        observed["score-12-target-stop-tie"]["decision"]["decision"] != "COLLISION"
        or observed["score-12-target-stop-tie"]["excitatory_update_spikes"] != (target,)
        or observed["score-12-target-stop-tie"]["inhibitory_update_spikes"]
        or observed["score-12-target-stop-tie"]["committed_excitatory_tick_count"] != 12
        or observed["score-12-target-stop-tie"]["committed_inhibitory_tick_count"] != 0
        or observed["score-24-correct-unique"]["decision"]["decision"] != "FACTOR"
        or observed["score-24-correct-unique"]["decision"]["factor"] != spec.TRIADS[0][0]
        or observed["score-24-correct-unique"]["target_match_spikes"] != (target,)
        or observed["score-24-correct-unique"]["excitatory_update_spikes"]
        or observed["score-24-correct-unique"]["inhibitory_update_spikes"]
        or observed["score-24-correct-unique"]["committed_excitatory_tick_count"] != 0
        or observed["score-24-correct-unique"]["committed_inhibitory_tick_count"] != 0
        or observed["two-rival-scratch"]["excitatory_update_spikes"] != (target,)
        or observed["two-rival-scratch"]["inhibitory_update_spikes"] != rivals
        or observed["two-rival-scratch"]["committed_excitatory_tick_count"] != 12
        or observed["two-rival-scratch"]["committed_inhibitory_tick_count"] != 24
        or observed["nonselected-distractor-scratch"]["inhibitory_update_spikes"]
        or observed["nonselected-distractor-scratch"]["decision"]["decision"] != "STOP"
        or observed["nonselected-distractor-scratch"]["scout_readout_spikes"]
        or observed["nonselected-distractor-scratch"]["excitatory_update_spikes"] != (target,)
        or observed["nonselected-distractor-scratch"]["committed_excitatory_tick_count"] != 12
        or observed["nonselected-distractor-scratch"]["committed_inhibitory_tick_count"] != 0
        or observed["teacher-off"]["teacher_spikes"]
        or observed["teacher-off"]["excitatory_update_spikes"]
        or observed["teacher-off"]["inhibitory_update_spikes"]
        or observed["teacher-off"]["committed_excitatory_tick_count"] != 0
        or observed["teacher-off"]["committed_inhibitory_tick_count"] != 0
        or observed["cell-off"]["decision"]["decision"] != "STOP"
        or observed["cell-off"]["selected_trace_read_spike_count"] != 0
        or observed["cell-off"]["excitatory_update_spikes"] != (target,)
        or observed["cell-off"]["inhibitory_update_spikes"]
        or observed["cell-off"]["committed_excitatory_tick_count"] != 0
        or observed["cell-off"]["committed_inhibitory_tick_count"] != 0
    ):
        raise RuntimeError("teacher/scout electrical preflight changed")
    return tuple(rows)


def _execute_compositor_preflight(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    phase = spec._CompositorPhaseRequestPopulation()
    ready = np.asarray((_FULL_CURRENT_PA,))
    for name, requests in (
        ("zero-phase-request", np.asarray((False, False), dtype=np.bool_)),
        ("two-phase-requests", np.asarray((True, True), dtype=np.bool_)),
    ):
        try:
            phase.evaluate(request_spikes=requests, event_ready_storage_pa=ready)
        except ValueError:
            rows.append({"name": name, "rejected_before_mutation": True})
        else:
            raise RuntimeError("compositor phase request accepted a non-one-hot input")
    early_state = _CompositorPersistentState()
    early_state.clear_ready()
    early_gate = phase.evaluate(
        request_spikes=np.asarray((True, False), dtype=np.bool_),
        event_ready_storage_pa=np.asarray((_FULL_CURRENT_PA if early_state.read_ready() else 0.0,)),
    )
    if np.any(early_gate) or early_state.active_count(include_ready=True) != 0:
        raise RuntimeError("compositor accepted ingress before ready")
    early_state.initialize_ready()
    rows.append(
        {
            "name": "early-ingress-before-ready",
            "gate_spikes": tuple(bool(value) for value in early_gate),
            "post_rearm_ready_pa_hex": tuple(
                float(value).hex() for value in early_state.ready_charge_pa
            ),
        }
    )
    admission = spec._SequentialFirstFreeSlotAdmissionPopulation()
    for name, occupied, blocked, expected in (
        ("admission-empty", (0.0, 0.0), 0.0, (True, False)),
        ("admission-slot-zero", (_FULL_CURRENT_PA, 0.0), 0.0, (False, True)),
        ("admission-blocked", (0.0, 0.0), _FULL_CURRENT_PA, (False, False)),
    ):
        observed = admission.evaluate(
            occupied_storage_pa=np.asarray(occupied),
            admission_block_storage_pa=np.asarray((blocked,)),
        )
        if tuple(bool(value) for value in observed) != expected:
            raise RuntimeError(f"compositor admission control changed: {name}")
        rows.append(
            {
                "name": name,
                "admission_spikes": tuple(bool(value) for value in observed),
            }
        )
    copy_population = spec._BufferedIngressSlotCopyPopulation()
    valid_event = spec._event(spec.TRIADS[0][0], 0, 4)
    valid_ingress = _ingress_identity(valid_event)
    copied = copy_population.evaluate(
        ingress_readout_spikes=valid_ingress,
        admitted_slot_spikes=np.asarray((True, False), dtype=np.bool_),
    )
    scratch_state = _CompositorPersistentState()
    scratch_state.set_copy(copied)
    first_identity, first_occupied = scratch_state.read_slots()
    second_identity, second_occupied = scratch_state.read_slots()
    if (
        not np.array_equal(first_identity, second_identity)
        or not np.array_equal(first_occupied, second_occupied)
        or tuple(bool(value) for value in first_occupied) != (True, False)
    ):
        raise RuntimeError("compositor slot storage read was destructive")
    scratch_state.reset_case_from_relay(False)
    retained_identity, retained_occupied = scratch_state.read_slots()
    if not np.array_equal(retained_identity, first_identity) or not np.array_equal(
        retained_occupied, first_occupied
    ):
        raise RuntimeError("suppressed compositor case-reset relay changed storage")
    scratch_state.reset_case_from_relay(True)
    scratch_state.initialize_ready()
    rows.append(
        {
            "name": "slot-storage-repeated-read-reset",
            "identity_spike_count": int(np.count_nonzero(first_identity)),
            "occupied_spikes": tuple(bool(value) for value in first_occupied),
            "repeated_read_equal": True,
            "suppressed_reset_retained_state": True,
            "post_reset_active_count": scratch_state.active_count(include_ready=False),
            "post_reset_ready_pa_hex": tuple(
                float(value).hex() for value in scratch_state.ready_charge_pa
            ),
        }
    )
    for name, ingress, admitted in (
        (
            "copy-empty-ingress-ghost",
            np.zeros(spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT, dtype=np.bool_),
            np.asarray((True, False), dtype=np.bool_),
        ),
        (
            "copy-multi-admission",
            valid_ingress,
            np.asarray((True, True), dtype=np.bool_),
        ),
    ):
        try:
            copy_population.evaluate(
                ingress_readout_spikes=ingress,
                admitted_slot_spikes=admitted,
            )
        except ValueError:
            rows.append({"name": name, "rejected_before_copy": True})
        else:
            raise RuntimeError("compositor copy accepted a malformed causal input")
    state = _CompositorPersistentState()
    scratch_factors = tuple(
        factor for factor in spec.FACTOR_REGISTRY if factor not in spec.TAUGHT_FACTORS
    )[:3]
    synthetic_cases = (
        (
            "synthetic-empty",
            (),
            "ZERO_EVENTS",
            (),
        ),
        (
            "synthetic-duplicate",
            (
                spec._event(scratch_factors[0], 1, 3),
                spec._event(scratch_factors[0], 1, 3),
            ),
            "ONE_EVENT",
            (scratch_factors[0],),
        ),
        (
            "synthetic-two",
            (
                spec._event(scratch_factors[0], 1, 3),
                spec._event(scratch_factors[1], 5, 7),
            ),
            "TWO_EVENTS",
            scratch_factors[:2],
        ),
        (
            "synthetic-overlap",
            (
                spec._event(scratch_factors[0], 1, 4),
                spec._event(scratch_factors[1], 3, 6),
            ),
            "STOP_OVERLAPPING_FACTOR_EVENTS",
            (),
        ),
        (
            "synthetic-capacity",
            (
                spec._event(scratch_factors[0], 0, 2),
                spec._event(scratch_factors[1], 4, 6),
                spec._event(scratch_factors[2], 8, 10),
            ),
            "STOP_EVENT_CAPACITY_2",
            (),
        ),
    )
    synthetic_replays = tuple(
        _execute_compositor(
            source_sha256=_namespaced_sha256(_COMPOSITOR_TRACE_NAMESPACE, name),
            events=events,
            persistent_state=state,
        )
        for name, events, _status, _factors in synthetic_cases
    )
    if any(
        replay.status != expected_status
        or replay.factors != expected_factors
        or replay.post_reset_active_count != 0
        or replay.case_reset_relay_tick != 3
        or replay.case_reset_arrival_tick != 4
        or replay.next_case_ready_initialization_tick != 5
        for replay, (_name, _events, expected_status, expected_factors) in zip(
            synthetic_replays,
            synthetic_cases,
            strict=True,
        )
    ):
        raise RuntimeError("compositor synthetic mechanism panel changed")
    first_event_evaluated = dict(synthetic_replays[1].event_stage_evaluated_counts[0])
    if tuple(first_event_evaluated) != tuple(range(-1, 11)) or any(
        count < 1 for count in first_event_evaluated.values()
    ):
        raise RuntimeError("compositor causal stage panel is incomplete")
    rows.extend(
        {"name": name, "observed": replay.record()}
        for (name, _events, _status, _factors), replay in zip(
            synthetic_cases,
            synthetic_replays,
            strict=True,
        )
    )
    overlap_events = tuple(
        spec._event(factor, start, stop)
        for factor, (start, stop) in zip(
            (
                scratch_factors[0],
                scratch_factors[1],
                scratch_factors[2],
                scratch_factors[0],
                scratch_factors[1],
            ),
            ((1, 4), (3, 6), (1, 4), (6, 8), (9, 11)),
            strict=True,
        )
    )
    capacity_events = tuple(
        spec._event(factor, start, stop)
        for factor, (start, stop) in zip(
            (
                scratch_factors[0],
                scratch_factors[1],
                scratch_factors[2],
                scratch_factors[0],
                scratch_factors[1],
            ),
            ((0, 2), (3, 5), (6, 8), (1, 4), (3, 6)),
            strict=True,
        )
    )
    absorbing_state = _CompositorPersistentState()
    overlap_absorbing = _execute_compositor(
        source_sha256=_namespaced_sha256(_COMPOSITOR_TRACE_NAMESPACE, "overlap-absorbing"),
        events=overlap_events,
        persistent_state=absorbing_state,
    )
    capacity_absorbing = _execute_compositor(
        source_sha256=_namespaced_sha256(_COMPOSITOR_TRACE_NAMESPACE, "capacity-absorbing"),
        events=capacity_events,
        persistent_state=absorbing_state,
    )
    zero_after_poison = _execute_compositor(
        source_sha256=_namespaced_sha256(_COMPOSITOR_TRACE_NAMESPACE, "zero-after-poison"),
        events=(),
        persistent_state=absorbing_state,
    )
    if (
        overlap_absorbing.status != "STOP_OVERLAPPING_FACTOR_EVENTS"
        or overlap_absorbing.factors
        or overlap_absorbing.eos_status_spikes != (False, False, False, True, False)
        or any(
            any(count for tick, count in stage_counts if tick >= 1)
            for stage_counts in overlap_absorbing.event_stage_spike_counts[2:]
        )
        or capacity_absorbing.status != "STOP_EVENT_CAPACITY_2"
        or capacity_absorbing.factors
        or capacity_absorbing.eos_status_spikes != (False, False, False, False, True)
        or zero_after_poison.status != "ZERO_EVENTS"
        or zero_after_poison.factors
    ):
        raise RuntimeError("compositor absorbing poison/reset control changed")
    rows.extend(
        (
            {"name": "overlap-first-absorbing", "observed": overlap_absorbing.record()},
            {"name": "capacity-first-absorbing", "observed": capacity_absorbing.record()},
            {"name": "zero-after-poison-reset", "observed": zero_after_poison.record()},
        )
    )
    return tuple(rows)


def _execute_electrical_preflight(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
) -> tuple[dict[str, object], str, int]:
    latch = _execute_latch_preflight()
    selector = _execute_selector_preflight(protocol)
    v4_replays, v4_root, circuit_off_count, tamper_rejected = v4impl._execute_electrical_preflight(
        v4spec.PROTOCOL
    )
    if (
        len(v4_replays) != len(protocol.factor_wta_electrical_controls)
        or circuit_off_count != 0
        or not tamper_rejected
    ):
        raise RuntimeError("V4 factor WTA electrical preflight changed")
    teacher = _execute_teacher_preflight()
    compositor = _execute_compositor_preflight(protocol)
    record = {
        "protocol_sha256": protocol.protocol_sha256,
        "electrical_preflight_contract_sha256": (protocol.electrical_preflight_contract_sha256),
        "persistent_latch_primitive_sha256": protocol.persistent_latch_primitive_sha256,
        "selector_full_preflight_sha256": protocol.selector_full_preflight_sha256,
        "v4_wta_reference_sha256": protocol.v4_wta_reference_sha256,
        "teacher_scout_circuit_sha256": protocol.teacher_scout_circuit_sha256,
        "compositor_circuit_sha256": protocol.compositor_circuit_sha256,
        "latch": latch,
        "selector": selector,
        "v4_factor_wta": tuple(replay.record() for replay in v4_replays),
        "v4_factor_wta_root_sha256": v4_root,
        "v4_circuit_off_decision_count": circuit_off_count,
        "v4_collision_tamper_rejected": tamper_rejected,
        "teacher_scout": teacher,
        "compositor": compositor,
        "completed_before_state_allocation": True,
    }
    root = _namespaced_sha256(_ELECTRICAL_NAMESPACE, record)
    if _namespaced_sha256(_ELECTRICAL_NAMESPACE, record) != root:
        raise RuntimeError("electrical preflight trace authentication failed")
    control_count = 7 + len(selector) + len(v4_replays) + len(teacher) + len(compositor)
    return record, root, control_count


def _array_frame(name: str, values: np.ndarray) -> bytes:
    if (
        values.shape != spec.FUTURE_STATE_SHAPE
        or values.dtype != TICK_DTYPE
        or not values.flags.c_contiguous
    ):
        raise ValueError("teacher-gated V5 checkpoint array framing changed")
    header = _canonical_json({"name": name, "dtype": values.dtype.str, "shape": values.shape})
    raw = values.tobytes(order="C")
    return len(header).to_bytes(8, "little") + header + len(raw).to_bytes(8, "little") + raw


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
    *,
    checkpoint_sha256: str,
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> None:
    if (
        not _is_sha256(checkpoint_sha256)
        or sha256(_checkpoint_payload(metadata, excitatory_ticks, inhibitory_ticks)).hexdigest()
        != checkpoint_sha256
    ):
        raise RuntimeError("frozen teacher-gated V5 checkpoint changed")


def _selector_score_row(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    tokens: tuple[str, str, str, str],
    source_sha256: str,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    lexical_gain: float = 1.0,
) -> tuple[SelectorReplay, tuple[int, ...]]:
    selector, trace_latch = _run_selector(
        protocol,
        tokens,
        source_sha256=source_sha256,
        lexical_gain=lexical_gain,
    )
    trace_spikes = trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.full(spec.LEXICAL_CELL_COUNT, _FULL_CURRENT_PA),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
    )
    raw = _raw_factor_scores(
        excitatory_ticks,
        inhibitory_ticks,
        tuple(int(cell) for cell in np.flatnonzero(trace_spikes)),
    )
    trace_latch.step(
        set_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        read_gate_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        direct_readout_currents_pa=np.zeros(spec.LEXICAL_CELL_COUNT),
        reset_currents_pa=np.full(
            spec.LEXICAL_CELL_COUNT,
            v4spec.BASKET_INHIBITION_CURRENT_PA,
        ),
    )
    if any(trace_latch.storage_charge_pa):
        raise RuntimeError("inference selected-trace reset failed")
    return selector, raw


def _projection_panel_sha256(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    selectors: tuple[SelectorReplay, ...],
) -> str:
    if len(selectors) != spec.LESSON_COUNT:
        raise ValueError("projection panel requires every atomic selector trace")
    return _namespaced_sha256(
        _PROJECTION_PANEL_NAMESPACE,
        {
            "protocol_projection_sha256": protocol.projection_sha256,
            "selector_circuit_sha256": protocol.selector_circuit_sha256,
            "ordered_selector_traces": tuple(selector.record() for selector in selectors),
        },
    )


def _atomic_result_record(
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
    electrical_preflight_sha256: str,
    projection_panel_sha256: str,
    factor_registry_sha256: str,
    result: AtomicBatchResult,
    telemetry: dict[str, object],
    causal_control_roots: dict[str, str],
) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "electrical_preflight_sha256": electrical_preflight_sha256,
        "projection_panel_sha256": projection_panel_sha256,
        "factor_registry_sha256": factor_registry_sha256,
        "local_rule_sha256": spec.LOCAL_RULE_SHA256,
        "atomic_gate_sha256": spec.ATOMIC_GATE_SHA256,
        "result": result.record(include_sha256=False),
        "update_telemetry": telemetry,
        "causal_control_roots": tuple(sorted(causal_control_roots.items())),
    }


def _composition_result_record(
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
    atomic_result_sha256: str,
    result: CompositionBatchResult,
) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "atomic_result_sha256": atomic_result_sha256,
        "composition_execution_sha256": spec.COMPOSITION_EXECUTION_SHA256,
        "compositor_circuit_sha256": spec.COMPOSITOR_CIRCUIT_SHA256,
        "result": result.record(include_sha256=False),
    }


def _control_result_record(
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
    composition_result_sha256: str,
    result: CompositorControlBatchResult,
) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "composition_result_sha256": composition_result_sha256,
        "composition_execution_sha256": spec.COMPOSITION_EXECUTION_SHA256,
        "compositor_circuit_sha256": spec.COMPOSITOR_CIRCUIT_SHA256,
        "result": result.record(include_sha256=False),
    }


def _execute_atomic_batch(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    metadata: dict[str, object],
    checkpoint_sha256: str,
    electrical_preflight_sha256: str,
    factor_registry_sha256: str,
    telemetry: dict[str, object],
    causal_control_roots: dict[str, str],
) -> tuple[AtomicBatchResult, tuple[SelectorReplay, ...], str]:
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    selectors: list[SelectorReplay] = []
    raw_rows: list[tuple[int, ...]] = []
    source_ids: list[str] = []
    for lesson in protocol.lessons:
        selector, raw = _selector_score_row(
            protocol,
            tokens=lesson.tokens,
            source_sha256=lesson.sha256,
            excitatory_ticks=excitatory_ticks,
            inhibitory_ticks=inhibitory_ticks,
        )
        selectors.append(selector)
        raw_rows.append(raw)
        source_ids.append(lesson.sha256)
    decisions = _race_factor_batch(
        raw_rows=tuple(raw_rows),
        source_sha256s=tuple(source_ids),
    )
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    expected_factors = tuple(lesson.factor for lesson in protocol.lessons)
    exact_mask = tuple(
        decision.decision == "FACTOR" and decision.factor == expected
        for decision, expected in zip(decisions, expected_factors, strict=True)
    )
    pass_count = sum(exact_mask)
    mismatch_count = spec.LESSON_COUNT - pass_count
    fit_ids = frozenset(protocol.fit_lesson_sha256s)
    fit_pass = sum(
        exact and decision.source_sha256 in fit_ids
        for exact, decision in zip(exact_mask, decisions, strict=True)
    )
    heldout_pass = pass_count - fit_pass
    status = (
        "PASS_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_ATOMIC_READINESS"
        if mismatch_count == 0
        else str(spec.COMPOSITION_EXECUTION_RECORD["atomic_failure_status"])
    )
    result = AtomicBatchResult(
        status=status,
        decisions=decisions,
        expected_factors=expected_factors,
        pass_count=pass_count,
        mismatch_count=mismatch_count,
        fit_pass_count=fit_pass,
        heldout_pass_count=heldout_pass,
        checkpoint_authentications=2,
        composition_attempt_count=0,
        control_attempt_count=0,
        sha256="",
    )
    projection_panel_sha = _projection_panel_sha256(protocol, tuple(selectors))
    result = replace(
        result,
        sha256=_namespaced_sha256(
            _ATOMIC_RESULT_NAMESPACE,
            _atomic_result_record(
                protocol_sha256=protocol.protocol_sha256,
                checkpoint_sha256=checkpoint_sha256,
                electrical_preflight_sha256=electrical_preflight_sha256,
                projection_panel_sha256=projection_panel_sha,
                factor_registry_sha256=factor_registry_sha256,
                result=result,
                telemetry=telemetry,
                causal_control_roots=causal_control_roots,
            ),
        ),
    )
    return result, tuple(selectors), projection_panel_sha


def _execute_composition_batch(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    metadata: dict[str, object],
    checkpoint_sha256: str,
    atomic_result_sha256: str,
) -> CompositionBatchResult:
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    ordered_spans = tuple(span for case in protocol.composition_cases for span in case.spans)
    raw_rows: list[tuple[int, ...]] = []
    source_ids: list[str] = []
    selector_replays: list[SelectorReplay] = []
    for span in ordered_spans:
        selector, raw = _selector_score_row(
            protocol,
            tokens=span.tokens,
            source_sha256=span.source_sha256,
            excitatory_ticks=excitatory_ticks,
            inhibitory_ticks=inhibitory_ticks,
        )
        raw_rows.append(raw)
        source_ids.append(span.source_sha256)
        selector_replays.append(selector)
    span_decisions = _race_factor_batch(
        raw_rows=tuple(raw_rows),
        source_sha256s=tuple(source_ids),
    )
    compositor_state = _CompositorPersistentState()
    case_replays: list[CompositionCaseReplay] = []
    for case_index, case in enumerate(protocol.composition_cases):
        decisions = span_decisions[case_index * 3 : (case_index + 1) * 3]
        selectors = selector_replays[case_index * 3 : (case_index + 1) * 3]
        event_rows: list[spec.ProtocolFactorEvent] = []
        relay_rows: list[tuple[int, ...]] = []
        for span, decision in zip(case.spans, decisions, strict=True):
            relay_cells = decision.factor_relay_spike_cells
            if relay_cells:
                if len(relay_cells) != 1:
                    raise RuntimeError("composition factor relay was not one-hot")
                event_rows.append(
                    spec._event(
                        spec.FACTOR_REGISTRY[relay_cells[0]],
                        span.start_token,
                        span.stop_token,
                    )
                )
                relay_rows.append(relay_cells)
        events = tuple(event_rows)
        compositor = _execute_compositor(
            source_sha256=case.sha256,
            events=events,
            persistent_state=compositor_state,
            factor_relay_spike_cells=tuple(relay_rows),
        )
        case_replays.append(
            CompositionCaseReplay(
                case_sha256=case.sha256,
                selector_replays=tuple(selectors),
                span_decisions=tuple(decisions),
                emitted_events=events,
                expected_factors=case.expected_factors,
                compositor=compositor,
            )
        )
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    exact_mask = tuple(
        replay.compositor.status == "TWO_EVENTS"
        and replay.compositor.factors == case.expected_factors
        for replay, case in zip(case_replays, protocol.composition_cases, strict=True)
    )
    exact_count = sum(exact_mask)
    mismatch_count = spec.COMPOSITION_CASE_COUNT - exact_count
    status = (
        "PASS_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_COMPOSITION"
        if mismatch_count == 0
        else str(spec.COMPOSITION_EXECUTION_RECORD["composition_failure_status"])
    )
    result = CompositionBatchResult(
        status=status,
        cases=tuple(case_replays),
        exact_count=exact_count,
        mismatch_count=mismatch_count,
        attempt_count=spec.COMPOSITION_CASE_COUNT,
        control_attempt_count=0,
        checkpoint_authentications=2,
        sha256="",
    )
    return replace(
        result,
        sha256=_namespaced_sha256(
            _COMPOSITION_RESULT_NAMESPACE,
            _composition_result_record(
                protocol_sha256=protocol.protocol_sha256,
                checkpoint_sha256=checkpoint_sha256,
                atomic_result_sha256=atomic_result_sha256,
                result=result,
            ),
        ),
    )


def _execute_control_batch(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    metadata: dict[str, object],
    checkpoint_sha256: str,
    composition_result_sha256: str,
) -> CompositorControlBatchResult:
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    controls = (
        *protocol.duplicate_controls,
        *protocol.overlap_controls,
        *protocol.capacity_controls,
    )
    compositor_state = _CompositorPersistentState()
    replays = tuple(
        _execute_compositor(
            source_sha256=control.sha256,
            events=control.events,
            persistent_state=compositor_state,
        )
        for control in controls
    )
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    exact_mask = tuple(
        replay.status == control.expected_status
        and replay.factors == control.expected_factors
        and (
            control.expected_eos_status_latch is None
            or (
                replay.overlap_poison
                if control.expected_eos_status_latch == "overlap"
                else replay.capacity_poison
            )
        )
        for replay, control in zip(replays, controls, strict=True)
    )
    exact_count = sum(exact_mask)
    mismatch_count = len(controls) - exact_count
    status = (
        str(spec.COMPOSITION_EXECUTION_RECORD["pass_status"])
        if mismatch_count == 0
        else str(spec.COMPOSITION_EXECUTION_RECORD["control_failure_status"])
    )
    result = CompositorControlBatchResult(
        status=status,
        controls=replays,
        exact_count=exact_count,
        mismatch_count=mismatch_count,
        duplicate_attempt_count=spec.DUPLICATE_CONTROL_COUNT,
        overlap_attempt_count=spec.OVERLAP_CONTROL_COUNT,
        capacity_attempt_count=spec.CAPACITY_CONTROL_COUNT,
        checkpoint_authentications=2,
        sha256="",
    )
    return replace(
        result,
        sha256=_namespaced_sha256(
            _CONTROL_RESULT_NAMESPACE,
            _control_result_record(
                protocol_sha256=protocol.protocol_sha256,
                checkpoint_sha256=checkpoint_sha256,
                composition_result_sha256=composition_result_sha256,
                result=result,
            ),
        ),
    )


def _full_atomic_inference(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    lexical_gain: float,
) -> tuple[FactorRaceDecision, ...]:
    raw_rows: list[tuple[int, ...]] = []
    source_ids: list[str] = []
    for lesson in protocol.lessons:
        _selector, raw = _selector_score_row(
            protocol,
            tokens=lesson.tokens,
            source_sha256=lesson.sha256,
            excitatory_ticks=excitatory_ticks,
            inhibitory_ticks=inhibitory_ticks,
            lexical_gain=lexical_gain,
        )
        raw_rows.append(raw)
        source_ids.append(lesson.sha256)
    return _race_factor_batch(
        raw_rows=tuple(raw_rows),
        source_sha256s=tuple(source_ids),
    )


def _execute_teacher_training_control(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    schedule: tuple[str, ...],
    teacher_mapping: dict[str, str],
    teacher_gain: float,
    name: str,
) -> tuple[dict[str, object], str]:
    lesson_by_sha = {lesson.sha256: lesson for lesson in protocol.lessons}
    excitatory = np.zeros(spec.FUTURE_STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(spec.FUTURE_STATE_SHAPE, dtype=TICK_DTYPE)
    replays = tuple(
        _execute_fit_lesson(
            protocol,
            lesson=lesson_by_sha[lesson_sha],
            teacher_factor=teacher_mapping[lesson_by_sha[lesson_sha].factor],
            excitatory_ticks=excitatory,
            inhibitory_ticks=inhibitory,
            teacher_gain=teacher_gain,
        )
        for lesson_sha in schedule
    )
    control_metadata = {
        "schema": "teacher-gated-factor-workspace-v5-causal-control",
        "name": name,
        "protocol_sha256": protocol.protocol_sha256,
        "control_panel_sha256": protocol.control_panel_sha256,
        "schedule": schedule,
        "teacher_mapping": tuple(sorted(teacher_mapping.items())),
        "teacher_gain_hex": float(teacher_gain).hex(),
        "fit_trace_roots": tuple(replay.trace_sha256 for replay in replays),
        "array_shape": spec.FUTURE_STATE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": spec.TICK_CAP,
    }
    checkpoint = sha256(_checkpoint_payload(control_metadata, excitatory, inhibitory)).hexdigest()
    excitatory.setflags(write=False)
    inhibitory.setflags(write=False)
    _authenticate_state(
        checkpoint_sha256=checkpoint,
        metadata=control_metadata,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
    )
    decisions = _full_atomic_inference(
        protocol,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        lexical_gain=1.0,
    )
    _authenticate_state(
        checkpoint_sha256=checkpoint,
        metadata=control_metadata,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
    )
    expected = tuple(teacher_mapping[lesson.factor] for lesson in protocol.lessons)
    exact_count = sum(
        decision.factor_relay_spike_cells == (_FACTOR_INDEX[target],)
        for decision, target in zip(decisions, expected, strict=True)
    )
    true_exact_count = sum(
        decision.factor_relay_spike_cells == (_FACTOR_INDEX[lesson.factor],)
        for decision, lesson in zip(decisions, protocol.lessons, strict=True)
    )
    record = {
        **control_metadata,
        "checkpoint_sha256": checkpoint,
        "checkpoint_authentications": 2,
        "checkpoint_nonzero_bytes": int(
            np.count_nonzero(excitatory) + np.count_nonzero(inhibitory)
        ),
        "decisions": tuple(decision.record() for decision in decisions),
        "expected_factors": expected,
        "exact_count": exact_count,
        "true_exact_count": true_exact_count,
        "excitatory_update_events": sum(
            bool(replay.excitatory_update_spike_cells) for replay in replays
        ),
        "inhibitory_update_events": sum(
            len(replay.inhibitory_update_spike_cells) for replay in replays
        ),
        "committed_excitatory_ticks": sum(replay.committed_excitatory_ticks for replay in replays),
        "committed_inhibitory_ticks": sum(replay.committed_inhibitory_ticks for replay in replays),
    }
    root = _namespaced_sha256(
        b"teacher-gated-factor-workspace-v5-training-control\0",
        record,
    )
    return record, root


def _execute_cell_off_control(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    *,
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    metadata: dict[str, object],
    checkpoint_sha256: str,
) -> tuple[dict[str, object], str]:
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    decisions = _full_atomic_inference(
        protocol,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
        lexical_gain=0.0,
    )
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    stop_count = sum(
        decision.decision == "STOP" and not decision.factor_relay_spike_cells
        for decision in decisions
    )
    record = {
        "protocol_sha256": protocol.protocol_sha256,
        "control_panel_sha256": protocol.control_panel_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "content_relay_gain_hex": (0.0).hex(),
        "checkpoint_authentications": 2,
        "decisions": tuple(decision.record() for decision in decisions),
        "stop_count": stop_count,
        "factor_event_count": sum(
            bool(decision.factor_relay_spike_cells) for decision in decisions
        ),
    }
    return record, _namespaced_sha256(
        b"teacher-gated-factor-workspace-v5-cell-off-control\0",
        record,
    )


def _atomic_gate_passes(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    result: AtomicBatchResult,
) -> bool:
    expected_ids = tuple(lesson.sha256 for lesson in protocol.lessons)
    expected_factors = tuple(lesson.factor for lesson in protocol.lessons)
    if (
        tuple(row.source_sha256 for row in result.decisions) != expected_ids
        or result.expected_factors != expected_factors
        or len(result.decisions) != spec.LESSON_COUNT
        or result.checkpoint_authentications != 2
        or result.composition_attempt_count != 0
        or result.control_attempt_count != 0
    ):
        raise RuntimeError("atomic publication schema/order changed")
    exact = tuple(
        row.decision == "FACTOR"
        and row.factor_relay_spike_cells == (_FACTOR_INDEX[expected],)
        and row.factor == expected
        for row, expected in zip(result.decisions, expected_factors, strict=True)
    )
    return (
        all(exact)
        and result.pass_count == spec.LESSON_COUNT
        and result.mismatch_count == 0
        and result.fit_pass_count == spec.FIT_COUNT
        and result.heldout_pass_count == spec.HELDOUT_COUNT
        and result.status == "PASS_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_ATOMIC_READINESS"
    )


def _composition_gate_passes(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    result: CompositionBatchResult,
) -> bool:
    if (
        len(result.cases) != spec.COMPOSITION_CASE_COUNT
        or result.attempt_count != spec.COMPOSITION_CASE_COUNT
        or result.control_attempt_count != 0
        or result.checkpoint_authentications != 2
        or tuple(case.case_sha256 for case in result.cases)
        != tuple(case.sha256 for case in protocol.composition_cases)
    ):
        raise RuntimeError("composition publication schema/order changed")
    exact_rows: list[bool] = []
    for observed, expected in zip(result.cases, protocol.composition_cases, strict=True):
        expected_source_sha256s = tuple(span.source_sha256 for span in expected.spans)
        left_span, _neutral_span, right_span = expected.spans
        left_decision, neutral_decision, right_decision = observed.span_decisions
        expected_events = (
            spec._event(
                expected.left_factor,
                left_span.start_token,
                left_span.stop_token,
            ),
            spec._event(
                expected.right_factor,
                right_span.start_token,
                right_span.stop_token,
            ),
        )
        if (
            observed.expected_factors != expected.expected_factors
            or tuple(span.kind for span in expected.spans) != ("factor", "neutral", "factor")
            or len(observed.selector_replays) != len(expected.spans)
            or len(observed.span_decisions) != len(expected.spans)
            or tuple(row.source_sha256 for row in observed.span_decisions)
            != expected_source_sha256s
            or tuple(row.source_sha256 for row in observed.selector_replays)
            != expected_source_sha256s
            or left_decision.decision != "FACTOR"
            or left_decision.factor != expected.left_factor
            or left_decision.factor_relay_spike_cells != (_FACTOR_INDEX[expected.left_factor],)
            or neutral_decision.decision != "STOP"
            or neutral_decision.factor is not None
            or neutral_decision.factor_relay_spike_cells
            or right_decision.decision != "FACTOR"
            or right_decision.factor != expected.right_factor
            or right_decision.factor_relay_spike_cells != (_FACTOR_INDEX[expected.right_factor],)
            or observed.emitted_events != expected_events
            or observed.compositor.source_sha256 != expected.sha256
            or observed.compositor.input_event_sha256s
            != tuple(event.event_sha256 for event in expected_events)
        ):
            raise RuntimeError("composition protocol-bound span/event publication changed")
        exact_rows.append(
            observed.compositor.status == "TWO_EVENTS"
            and observed.compositor.factors == expected.expected_factors
        )
    return (
        all(exact_rows)
        and result.exact_count == spec.COMPOSITION_CASE_COUNT
        and result.mismatch_count == 0
        and result.status == "PASS_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_COMPOSITION"
    )


def _validate_complete_compositor_stage_trace(
    replay: CompositorReplay,
    *,
    expected_event_count: int,
) -> None:
    expected_ticks = tuple(range(-1, 11))
    expected_evaluated = (
        (-1, 2),
        (0, spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
        (
            1,
            spec._COMPOSITOR_DUPLICATE_KEY_CELL_COUNT
            + spec._COMPOSITOR_OVERLAP_COINCIDENCE_CELL_COUNT
            + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
            + spec.EVENT_CAPACITY,
        ),
        (2, 2),
        (3, 2 + spec.EVENT_CAPACITY),
        (4, 1),
        (5, 4),
        (6, 1 + spec.EVENT_CAPACITY + 1),
        (7, 1 + spec.EVENT_CAPACITY + 1),
        (8, spec.EVENT_CAPACITY + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT),
        (9, spec._COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT + 1),
        (
            10,
            spec._COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT
            + spec._COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT
            + spec.EVENT_CAPACITY
            + 2,
        ),
    )
    if (
        len(replay.event_stage_spike_counts) != expected_event_count
        or len(replay.event_stage_evaluated_counts) != expected_event_count
        or any(
            tuple(tick for tick, _count in row) != expected_ticks
            for row in replay.event_stage_spike_counts
        )
        or any(row != expected_evaluated for row in replay.event_stage_evaluated_counts)
        or any(
            count < 0 or count > evaluated
            for spike_row, evaluated_row in zip(
                replay.event_stage_spike_counts,
                replay.event_stage_evaluated_counts,
                strict=True,
            )
            for (_tick, count), (_evaluated_tick, evaluated) in zip(
                spike_row,
                evaluated_row,
                strict=True,
            )
        )
    ):
        raise RuntimeError("compositor-control stage trace changed")


def _control_gate_passes(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
    result: CompositorControlBatchResult,
) -> bool:
    expected = (
        *protocol.duplicate_controls,
        *protocol.overlap_controls,
        *protocol.capacity_controls,
    )
    if (
        len(result.controls) != len(expected)
        or result.checkpoint_authentications != 2
        or result.duplicate_attempt_count != spec.DUPLICATE_CONTROL_COUNT
        or result.overlap_attempt_count != spec.OVERLAP_CONTROL_COUNT
        or result.capacity_attempt_count != spec.CAPACITY_CONTROL_COUNT
        or tuple(row.source_sha256 for row in result.controls)
        != tuple(row.sha256 for row in expected)
    ):
        raise RuntimeError("compositor-control publication schema/order changed")
    exact_rows: list[bool] = []
    for observed, control in zip(result.controls, expected, strict=True):
        expected_attempted = len(control.events)
        if control.kind == "exact-duplicate":
            expected_accepted, expected_duplicates = 1, 1
        elif control.kind == "exact-duplicate-after-full":
            expected_accepted, expected_duplicates = 2, 1
        elif control.kind == "distinct-overlap":
            expected_accepted, expected_duplicates = 1, 0
        elif control.kind in ("distinct-overlap-after-full", "capacity-three"):
            expected_accepted, expected_duplicates = 2, 0
        else:
            raise RuntimeError("compositor-control registered kind changed")
        expected_overlap = control.expected_eos_status_latch == "overlap"
        expected_capacity = control.expected_eos_status_latch == "capacity"
        expected_eos = tuple(status == control.expected_status for status in _STATUS_ORDER)
        if (
            observed.input_event_sha256s != tuple(event.event_sha256 for event in control.events)
            or observed.attempted_event_count != expected_attempted
            or observed.accepted_event_count != expected_accepted
            or observed.duplicate_count != expected_duplicates
            or observed.overlap_poison is not expected_overlap
            or observed.capacity_poison is not expected_capacity
            or observed.eos_status_spikes != expected_eos
            or observed.post_reset_active_count != 0
            or observed.case_reset_relay_tick != 3
            or observed.case_reset_arrival_tick != 4
            or observed.next_case_ready_initialization_tick != 5
        ):
            raise RuntimeError("compositor-control physical publication changed")
        _validate_complete_compositor_stage_trace(
            observed,
            expected_event_count=expected_attempted,
        )
        exact_rows.append(
            observed.status == control.expected_status
            and observed.factors == control.expected_factors
        )
    return (
        all(exact_rows)
        and result.exact_count == len(expected)
        and result.mismatch_count == 0
        and result.status == spec.COMPOSITION_EXECUTION_RECORD["pass_status"]
    )


class FrozenTeacherGatedFactorBridge:
    __slots__ = (
        "_causal_control_roots",
        "_checkpoint_sha256",
        "_excitatory_ticks",
        "_inhibitory_ticks",
        "_metadata",
        "atomic_result",
        "build_summary",
        "composition_result",
        "compositor_control_result",
        "electrical_preflight_sha256",
        "fit_replays",
    )

    def __init__(
        self,
        *,
        excitatory_ticks: np.ndarray,
        inhibitory_ticks: np.ndarray,
        metadata: dict[str, object],
        build_summary: BridgeBuildSummary,
        fit_replays: tuple[PlasticityReplay, ...],
        atomic_result: AtomicBatchResult,
        composition_result: CompositionBatchResult | None,
        compositor_control_result: CompositorControlBatchResult | None,
        causal_control_roots: dict[str, str],
    ) -> None:
        for name, values in (
            ("excitatory_ticks", excitatory_ticks),
            ("inhibitory_ticks", inhibitory_ticks),
        ):
            if values.shape != spec.FUTURE_STATE_SHAPE:
                raise ValueError(f"{name} must be uint8[24,512]")
            if values.dtype != TICK_DTYPE or np.any(values > spec.TICK_CAP):
                raise ValueError(f"{name} violates the frozen uint8 cap")
        self._excitatory_ticks = np.array(excitatory_ticks, copy=True, order="C")
        self._inhibitory_ticks = np.array(inhibitory_ticks, copy=True, order="C")
        self._excitatory_ticks.setflags(write=False)
        self._inhibitory_ticks.setflags(write=False)
        self._metadata = dict(metadata)
        if set(causal_control_roots) != {"teacher_off", "cell_off", "random_label"} or any(
            not _is_sha256(value) for value in causal_control_roots.values()
        ):
            raise ValueError("teacher-gated V5 causal-control roots changed")
        self._causal_control_roots = MappingProxyType(dict(causal_control_roots))
        self.build_summary = build_summary
        self.fit_replays = fit_replays
        self.atomic_result = atomic_result
        self.composition_result = composition_result
        self.compositor_control_result = compositor_control_result
        self.electrical_preflight_sha256 = str(self._metadata["electrical_preflight_sha256"])
        self._checkpoint_sha256 = sha256(self._checkpoint_bytes()).hexdigest()
        if self._checkpoint_sha256 != build_summary.checkpoint_sha256:
            raise RuntimeError("bridge received a different frozen checkpoint")
        self._authenticate_checkpoint()
        self._authenticate_atomic_result()
        if self.composition_result is not None:
            self._authenticate_composition_result()
        if self.compositor_control_result is not None:
            self._authenticate_control_result()

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
        _authenticate_state(
            checkpoint_sha256=self._checkpoint_sha256,
            metadata=self._metadata,
            excitatory_ticks=self._excitatory_ticks,
            inhibitory_ticks=self._inhibitory_ticks,
        )

    def _authenticate_protocol(self) -> None:
        _authenticate_protocol(spec.PROTOCOL)
        if (
            self._metadata.get("protocol_sha256") != spec.PROTOCOL.protocol_sha256
            or self._metadata.get("protocol_root_record") != spec.PROTOCOL.root_record()
        ):
            raise RuntimeError("frozen bridge protocol root changed")

    def _authenticate_atomic_result(self) -> None:
        expected = _namespaced_sha256(
            _ATOMIC_RESULT_NAMESPACE,
            _atomic_result_record(
                protocol_sha256=str(self._metadata["protocol_sha256"]),
                checkpoint_sha256=self._checkpoint_sha256,
                electrical_preflight_sha256=str(self._metadata["electrical_preflight_sha256"]),
                projection_panel_sha256=str(self._metadata["projection_panel_sha256"]),
                factor_registry_sha256=str(self._metadata["factor_registry_sha256"]),
                result=self.atomic_result,
                telemetry=dict(self._metadata["update_telemetry"]),
                causal_control_roots=dict(self._causal_control_roots),
            ),
        )
        if expected != self.atomic_result.sha256:
            raise RuntimeError("frozen teacher-gated V5 atomic result changed")

    def _authenticate_composition_result(self) -> None:
        if self.composition_result is None:
            raise RuntimeError("teacher-gated V5 composition result is absent")
        expected = _namespaced_sha256(
            _COMPOSITION_RESULT_NAMESPACE,
            _composition_result_record(
                protocol_sha256=str(self._metadata["protocol_sha256"]),
                checkpoint_sha256=self._checkpoint_sha256,
                atomic_result_sha256=self.atomic_result.sha256,
                result=self.composition_result,
            ),
        )
        if expected != self.composition_result.sha256:
            raise RuntimeError("frozen teacher-gated V5 composition result changed")

    def _authenticate_control_result(self) -> None:
        if self.composition_result is None or self.compositor_control_result is None:
            raise RuntimeError("teacher-gated V5 compositor-control result is absent")
        expected = _namespaced_sha256(
            _CONTROL_RESULT_NAMESPACE,
            _control_result_record(
                protocol_sha256=str(self._metadata["protocol_sha256"]),
                checkpoint_sha256=self._checkpoint_sha256,
                composition_result_sha256=self.composition_result.sha256,
                result=self.compositor_control_result,
            ),
        )
        if expected != self.compositor_control_result.sha256:
            raise RuntimeError("frozen teacher-gated V5 compositor-control result changed")

    def _authenticate_requested_batch(self, batch: object) -> AuthenticatedSpanBatch:
        if not isinstance(batch, AuthenticatedSpanBatch):
            raise TypeError("resolve requires AuthenticatedSpanBatch")
        record = {
            "protocol_sha256": batch.protocol_sha256,
            "spans": tuple(span.record() for span in batch.spans),
        }
        if (
            batch.protocol_sha256 != spec.PROTOCOL.protocol_sha256
            or _namespaced_sha256(_AUTHENTICATED_BATCH_NAMESPACE, record)
            != batch.authentication_sha256
        ):
            raise RuntimeError("requested teacher-gated V5 batch authentication failed")
        return batch

    def _resolve_authenticated_batch(
        self,
        batch: AuthenticatedSpanBatch,
    ) -> TeacherGatedFactorResolution:
        self._authenticate_checkpoint()
        raw_rows: list[tuple[int, ...]] = []
        source_ids: list[str] = []
        for span in batch.spans:
            _selector, raw = _selector_score_row(
                spec.PROTOCOL,
                tokens=span.tokens,
                source_sha256=span.source_sha256,
                excitatory_ticks=self._excitatory_ticks,
                inhibitory_ticks=self._inhibitory_ticks,
            )
            raw_rows.append(raw)
            source_ids.append(span.source_sha256)
        decisions = (
            _race_factor_batch(
                raw_rows=tuple(raw_rows),
                source_sha256s=tuple(source_ids),
            )
            if raw_rows
            else ()
        )
        event_rows: list[spec.ProtocolFactorEvent] = []
        relay_rows: list[tuple[int, ...]] = []
        for span, decision in zip(batch.spans, decisions, strict=True):
            relay_cells = decision.factor_relay_spike_cells
            if relay_cells:
                if len(relay_cells) != 1:
                    raise RuntimeError("requested factor relay was not one-hot")
                event_rows.append(
                    spec._event(
                        spec.FACTOR_REGISTRY[relay_cells[0]],
                        span.start_token,
                        span.stop_token,
                    )
                )
                relay_rows.append(relay_cells)
        events = tuple(event_rows)
        compositor = _execute_compositor(
            source_sha256=batch.authentication_sha256,
            events=events,
            factor_relay_spike_cells=tuple(relay_rows),
        )
        self._authenticate_checkpoint()
        semantic_record = {
            "protocol_sha256": batch.protocol_sha256,
            "batch_authentication_sha256": batch.authentication_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "span_decisions": tuple(decision.record() for decision in decisions),
            "events": tuple(event.record() for event in events),
            "compositor": compositor.record(),
            "checkpoint_authentications": 2,
        }
        return TeacherGatedFactorResolution(
            status=compositor.status,
            events=events,
            span_decisions=tuple(decisions),
            compositor=compositor,
            checkpoint_authentications=2,
            semantic_sha256=_namespaced_sha256(
                _RESOLUTION_NAMESPACE,
                semantic_record,
            ),
        )

    def resolve(self, batch: object) -> TeacherGatedFactorResolution:
        self._authenticate_protocol()
        self._authenticate_checkpoint()
        self._authenticate_atomic_result()
        atomic_pass = _atomic_gate_passes(spec.PROTOCOL, self.atomic_result)
        if not atomic_pass:
            if (
                self.composition_result is not None
                or self.compositor_control_result is not None
                or self.atomic_result.composition_attempt_count != 0
                or self.atomic_result.control_attempt_count != 0
                or self.build_summary.composition_attempt_count != 0
                or self.build_summary.duplicate_control_attempt_count != 0
                or self.build_summary.overlap_control_attempt_count != 0
                or self.build_summary.capacity_control_attempt_count != 0
            ):
                raise RuntimeError("atomic STOP downstream absence proof failed")
            raise RuntimeError(self.atomic_result.status)
        if self.composition_result is None:
            raise RuntimeError("atomic PASS is missing an authenticated composition result")
        self._authenticate_composition_result()
        composition_pass = _composition_gate_passes(
            spec.PROTOCOL,
            self.composition_result,
        )
        if not composition_pass:
            if (
                self.compositor_control_result is not None
                or self.composition_result.control_attempt_count != 0
                or self.build_summary.duplicate_control_attempt_count != 0
                or self.build_summary.overlap_control_attempt_count != 0
                or self.build_summary.capacity_control_attempt_count != 0
            ):
                raise RuntimeError("composition STOP downstream absence proof failed")
            raise RuntimeError(self.composition_result.status)
        if self.compositor_control_result is None:
            raise RuntimeError("composition PASS is missing a compositor-control result")
        self._authenticate_control_result()
        control_pass = _control_gate_passes(
            spec.PROTOCOL,
            self.compositor_control_result,
        )
        if not control_pass:
            raise RuntimeError(self.compositor_control_result.status)
        authenticated = self._authenticate_requested_batch(batch)
        return self._resolve_authenticated_batch(authenticated)


def build_frozen_teacher_gated_factor_bridge(
    protocol: spec.FrozenTeacherGatedFactorProtocol,
) -> FrozenTeacherGatedFactorBridge:
    _authenticate_protocol(protocol)
    electrical_record, electrical_sha, electrical_control_count = _execute_electrical_preflight(
        protocol
    )
    if _namespaced_sha256(_ELECTRICAL_NAMESPACE, electrical_record) != electrical_sha:
        raise RuntimeError("electrical preflight changed before state allocation")

    lesson_by_sha = {lesson.sha256: lesson for lesson in protocol.lessons}
    schedule = tuple(spec.LOCAL_RULE_RECORD["fit_schedule"]["lesson_sha256s"])
    if (
        len(schedule) != spec.FIT_COUNT
        or len(set(schedule)) != spec.FIT_COUNT
        or set(schedule) != set(protocol.fit_lesson_sha256s)
    ):
        raise RuntimeError("teacher-gated V5 one-pass FIT schedule changed")
    preliminary_selectors = tuple(
        _close_selector_preflight(
            _run_selector(
                protocol,
                lesson.tokens,
                source_sha256=lesson.sha256,
            )
        )
        for lesson in protocol.lessons
    )
    projection_panel_sha = _projection_panel_sha256(protocol, preliminary_selectors)
    factor_registry_sha = _namespaced_sha256(
        _FACTOR_REGISTRY_NAMESPACE,
        spec.FACTOR_REGISTRY,
    )
    identity_mapping = {factor: factor for factor in spec.TAUGHT_FACTORS}
    random_mapping = dict(spec.CONTROL_PANEL_RECORD["random_label_permutation"]["mapping"])
    teacher_off_record, teacher_off_root = _execute_teacher_training_control(
        protocol,
        schedule=schedule,
        teacher_mapping=identity_mapping,
        teacher_gain=0.0,
        name="teacher-off",
    )
    random_label_record, random_label_root = _execute_teacher_training_control(
        protocol,
        schedule=schedule,
        teacher_mapping=random_mapping,
        teacher_gain=1.0,
        name="random-label-permutation",
    )
    teacher_off_stop_count = sum(
        row["decision"] == "STOP" and not row["factor_relay_spike_cells"]
        for row in teacher_off_record["decisions"]
    )
    if (
        teacher_off_record["checkpoint_nonzero_bytes"] != 0
        or teacher_off_stop_count != spec.LESSON_COUNT
        or random_label_record["exact_count"] != spec.LESSON_COUNT
        or random_label_record["true_exact_count"] != 0
    ):
        raise RuntimeError("teacher-off/random-label physical control changed")

    excitatory = np.zeros(spec.FUTURE_STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(spec.FUTURE_STATE_SHAPE, dtype=TICK_DTYPE)
    fit_replays: list[PlasticityReplay] = []
    for lesson_sha256 in schedule:
        fit_replays.append(
            _execute_fit_lesson(
                protocol,
                lesson=lesson_by_sha[lesson_sha256],
                teacher_factor=lesson_by_sha[lesson_sha256].factor,
                excitatory_ticks=excitatory,
                inhibitory_ticks=inhibitory,
            )
        )
    fit_replay_tuple = tuple(fit_replays)
    telemetry = {
        "fit_pass_count": 1,
        "fit_lesson_attempt_count": len(fit_replay_tuple),
        "excitatory_update_events": sum(
            bool(replay.excitatory_update_spike_cells) for replay in fit_replay_tuple
        ),
        "inhibitory_update_events": sum(
            len(replay.inhibitory_update_spike_cells) for replay in fit_replay_tuple
        ),
        "committed_excitatory_ticks": sum(
            replay.committed_excitatory_ticks for replay in fit_replay_tuple
        ),
        "committed_inhibitory_ticks": sum(
            replay.committed_inhibitory_ticks for replay in fit_replay_tuple
        ),
        "excitatory_max_ticks": int(np.max(excitatory)),
        "inhibitory_max_ticks": int(np.max(inhibitory)),
    }
    expected_telemetry = dict(spec.ATOMIC_GATE_RECORD["expected_treatment_telemetry"])
    telemetry_exact = all(telemetry[key] == value for key, value in expected_telemetry.items())
    if np.any(excitatory > spec.TICK_CAP) or np.any(inhibitory > spec.TICK_CAP):
        raise RuntimeError("teacher-gated V5 learner exceeded uint8 tick cap")

    metadata: dict[str, object] = {
        "schema": "teacher-gated-factor-workspace-v5-checkpoint",
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_root_record": protocol.root_record(),
        "future_authentication_sha256": protocol.future_authentication_sha256,
        "electrical_preflight_contract_sha256": (protocol.electrical_preflight_contract_sha256),
        "electrical_preflight_sha256": electrical_sha,
        "electrical_control_count": electrical_control_count,
        "projection_panel_sha256": projection_panel_sha,
        "factor_registry_sha256": factor_registry_sha,
        "local_rule_sha256": protocol.local_rule_sha256,
        "teacher_scout_circuit_sha256": protocol.teacher_scout_circuit_sha256,
        "compositor_circuit_sha256": protocol.compositor_circuit_sha256,
        "fit_schedule_sha256s": schedule,
        "fit_pass_count": 1,
        "update_telemetry": telemetry,
        "telemetry_exact": telemetry_exact,
        "array_shape": spec.FUTURE_STATE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "array_bytes": int(excitatory.nbytes + inhibitory.nbytes),
        "tick_cap": spec.TICK_CAP,
        "composition_attempt_count_at_freeze": 0,
        "duplicate_control_attempt_count_at_freeze": 0,
        "overlap_control_attempt_count_at_freeze": 0,
        "capacity_control_attempt_count_at_freeze": 0,
        "pre_freeze_causal_control_roots": (
            ("teacher_off", teacher_off_root),
            ("random_label", random_label_root),
        ),
        "public_input_count": protocol.public_input_count,
        "canonical_input_count": protocol.canonical_input_count,
        "protected_input_count": protocol.protected_input_count,
    }
    if metadata["array_bytes"] != spec.FUTURE_STATE_BYTES:
        raise RuntimeError("teacher-gated V5 checkpoint byte count changed")
    checkpoint_sha = sha256(_checkpoint_payload(metadata, excitatory, inhibitory)).hexdigest()
    excitatory.setflags(write=False)
    inhibitory.setflags(write=False)
    _authenticate_state(
        checkpoint_sha256=checkpoint_sha,
        metadata=metadata,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
    )
    cell_off_record, cell_off_root = _execute_cell_off_control(
        protocol,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        checkpoint_sha256=checkpoint_sha,
    )
    if (
        cell_off_record["stop_count"] != spec.LESSON_COUNT
        or cell_off_record["factor_event_count"] != 0
    ):
        raise RuntimeError("cell-off physical control changed")
    causal_control_roots = {
        "teacher_off": teacher_off_root,
        "cell_off": cell_off_root,
        "random_label": random_label_root,
    }
    atomic_result, atomic_selectors, observed_projection_sha = _execute_atomic_batch(
        protocol,
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        checkpoint_sha256=checkpoint_sha,
        electrical_preflight_sha256=electrical_sha,
        factor_registry_sha256=factor_registry_sha,
        telemetry=telemetry,
        causal_control_roots=causal_control_roots,
    )
    if observed_projection_sha != projection_panel_sha or atomic_selectors != preliminary_selectors:
        raise RuntimeError("atomic selector projection changed across checkpoint freeze")
    if not telemetry_exact:
        atomic_result = replace(
            atomic_result,
            status=str(spec.COMPOSITION_EXECUTION_RECORD["atomic_failure_status"]),
            sha256="",
        )
        atomic_result = replace(
            atomic_result,
            sha256=_namespaced_sha256(
                _ATOMIC_RESULT_NAMESPACE,
                _atomic_result_record(
                    protocol_sha256=protocol.protocol_sha256,
                    checkpoint_sha256=checkpoint_sha,
                    electrical_preflight_sha256=electrical_sha,
                    projection_panel_sha256=projection_panel_sha,
                    factor_registry_sha256=factor_registry_sha,
                    result=atomic_result,
                    telemetry=telemetry,
                    causal_control_roots=causal_control_roots,
                ),
            ),
        )

    composition_result: CompositionBatchResult | None = None
    control_result: CompositorControlBatchResult | None = None
    authenticated_atomic_sha = _namespaced_sha256(
        _ATOMIC_RESULT_NAMESPACE,
        _atomic_result_record(
            protocol_sha256=protocol.protocol_sha256,
            checkpoint_sha256=checkpoint_sha,
            electrical_preflight_sha256=electrical_sha,
            projection_panel_sha256=projection_panel_sha,
            factor_registry_sha256=factor_registry_sha,
            result=atomic_result,
            telemetry=telemetry,
            causal_control_roots=causal_control_roots,
        ),
    )
    if authenticated_atomic_sha != atomic_result.sha256:
        raise RuntimeError("atomic publication changed before gate inspection")
    atomic_gate_pass = telemetry_exact and _atomic_gate_passes(protocol, atomic_result)
    if atomic_gate_pass:
        composition_result = _execute_composition_batch(
            protocol,
            excitatory_ticks=excitatory,
            inhibitory_ticks=inhibitory,
            metadata=metadata,
            checkpoint_sha256=checkpoint_sha,
            atomic_result_sha256=atomic_result.sha256,
        )
        authenticated_composition_sha = _namespaced_sha256(
            _COMPOSITION_RESULT_NAMESPACE,
            _composition_result_record(
                protocol_sha256=protocol.protocol_sha256,
                checkpoint_sha256=checkpoint_sha,
                atomic_result_sha256=atomic_result.sha256,
                result=composition_result,
            ),
        )
        if authenticated_composition_sha != composition_result.sha256:
            raise RuntimeError("composition publication changed before gate inspection")
        composition_gate_pass = _composition_gate_passes(protocol, composition_result)
        if composition_gate_pass:
            control_result = _execute_control_batch(
                protocol,
                excitatory_ticks=excitatory,
                inhibitory_ticks=inhibitory,
                metadata=metadata,
                checkpoint_sha256=checkpoint_sha,
                composition_result_sha256=composition_result.sha256,
            )
            authenticated_control_sha = _namespaced_sha256(
                _CONTROL_RESULT_NAMESPACE,
                _control_result_record(
                    protocol_sha256=protocol.protocol_sha256,
                    checkpoint_sha256=checkpoint_sha,
                    composition_result_sha256=composition_result.sha256,
                    result=control_result,
                ),
            )
            if authenticated_control_sha != control_result.sha256:
                raise RuntimeError("control publication changed before gate inspection")
            _control_gate_passes(protocol, control_result)
    silent_rows = np.asarray(
        [
            _FACTOR_INDEX[factor]
            for factor in spec.FACTOR_REGISTRY
            if factor not in spec.TAUGHT_FACTORS
        ],
        dtype=np.intp,
    )
    silent_nonzero = int(
        np.count_nonzero(excitatory[silent_rows]) + np.count_nonzero(inhibitory[silent_rows])
    )
    if control_result is not None:
        status = control_result.status
    elif composition_result is not None:
        status = composition_result.status
    else:
        status = atomic_result.status
    summary = BridgeBuildSummary(
        status=status,
        electrical_preflight_passed=True,
        electrical_control_count=electrical_control_count,
        electrical_pass_count=electrical_control_count,
        electrical_preflight_sha256=electrical_sha,
        projection_panel_sha256=projection_panel_sha,
        factor_registry_sha256=factor_registry_sha,
        fit_pass_count=1,
        excitatory_update_events=int(telemetry["excitatory_update_events"]),
        inhibitory_update_events=int(telemetry["inhibitory_update_events"]),
        committed_excitatory_ticks=int(telemetry["committed_excitatory_ticks"]),
        committed_inhibitory_ticks=int(telemetry["committed_inhibitory_ticks"]),
        excitatory_max_ticks=int(telemetry["excitatory_max_ticks"]),
        inhibitory_max_ticks=int(telemetry["inhibitory_max_ticks"]),
        teacher_off_stop_count=teacher_off_stop_count,
        cell_off_stop_count=int(cell_off_record["stop_count"]),
        random_label_permuted_exact_count=int(random_label_record["exact_count"]),
        random_label_true_exact_count=int(random_label_record["true_exact_count"]),
        silent_row_nonzero_bytes=silent_nonzero,
        atomic_pass_count=atomic_result.pass_count,
        atomic_mismatch_count=atomic_result.mismatch_count,
        composition_attempt_count=(
            composition_result.attempt_count if composition_result is not None else 0
        ),
        composition_exact_count=(
            composition_result.exact_count if composition_result is not None else 0
        ),
        duplicate_control_attempt_count=(
            control_result.duplicate_attempt_count if control_result is not None else 0
        ),
        overlap_control_attempt_count=(
            control_result.overlap_attempt_count if control_result is not None else 0
        ),
        capacity_control_attempt_count=(
            control_result.capacity_attempt_count if control_result is not None else 0
        ),
        control_exact_count=(control_result.exact_count if control_result is not None else 0),
        public_input_count=protocol.public_input_count,
        canonical_input_count=protocol.canonical_input_count,
        protected_input_count=protocol.protected_input_count,
        checkpoint_sha256=checkpoint_sha,
        atomic_result_sha256=atomic_result.sha256,
        composition_result_sha256=(
            composition_result.sha256 if composition_result is not None else None
        ),
        compositor_control_result_sha256=(
            control_result.sha256 if control_result is not None else None
        ),
    )
    bridge = FrozenTeacherGatedFactorBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        build_summary=summary,
        fit_replays=fit_replay_tuple,
        atomic_result=atomic_result,
        composition_result=composition_result,
        compositor_control_result=control_result,
        causal_control_roots=causal_control_roots,
    )
    if bridge.checkpoint_sha256 != checkpoint_sha:
        raise RuntimeError("teacher-gated V5 bridge froze a different checkpoint")
    return bridge


__all__ = [
    "AtomicBatchResult",
    "AuthenticatedSpanBatch",
    "BridgeBuildSummary",
    "CompositionBatchResult",
    "CompositionCaseReplay",
    "CompositorControlBatchResult",
    "CompositorReplay",
    "FactorRaceDecision",
    "FrozenTeacherGatedFactorBridge",
    "PlasticityReplay",
    "SelectorReplay",
    "TeacherGatedFactorResolution",
    "build_frozen_teacher_gated_factor_bridge",
]
