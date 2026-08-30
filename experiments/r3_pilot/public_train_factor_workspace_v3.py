"""Execute the frozen PUBLIC_TRAIN factor-set workspace V3 once.

The learner-visible input is only the V3 causal selector key.  Factor names are
teacher clamps during FIT and candidate relays after the physical races; lesson
IDs, profile hashes, and audit fields never enter selection or scoring.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4 as v4impl
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4spec
from experiments.r3_pilot import public_train_factor_workspace_v3_protocol as protocol_v3
from experiments.r3_pilot import teacher_gated_factor_workspace_v5 as v5impl
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5spec
from humanbrain.resting_glif_race import CompiledRestingGlifRace

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FACTOR_WORKSPACE_V3"
PASS_STATUS = "PASS_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3"
FIT_STOP_STATUS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_FIT_READINESS"
HELDOUT_STOP_STATUS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_UNTOUCHED_HELDOUT_REPLAY"

V3_PROTOCOL_SHA256 = "1722424f3cb540ed152d0035f916f85b9c7a8e9306242ecfe026b39b9685fdec"
V3_PROTOCOL_SOURCE_SHA256 = "30f9393169caa57313250cc9323fe064ae2f9b5a1ce1970df41f5177da2410e7"
V5_COMMIT = "5cb01cc"
V5_SOURCE_SHA256 = "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
V5_PROTOCOL_SOURCE_SHA256 = "d62fe76444ef48f0b58723f0cede1b6ca76e86c78623eb43c07f38be8e8c8746"

FACTOR_REGISTRY = protocol_v3.FACTOR_REGISTRY
ACTIVE_FACTORS = protocol_v3.ACTIVE_FACTORS
FACTOR_COUNT = len(FACTOR_REGISTRY)
EVENT_CAPACITY = protocol_v3.EVENT_CAPACITY
TICK_DTYPE = np.dtype(v5spec.TICK_DTYPE)
STATE_SHAPE = v5spec.FUTURE_STATE_SHAPE

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_COUNT = 0
PROGRAM_ARTIFACT_COUNT = 0
PROPERTY_ROW_COUNT = 0
FAMILY_IDENTIFIER_COUNT = 0
TASK_IDENTIFIER_COUNT = 0
FORBIDDEN_READ_COUNT = 0

_FACTOR_INDEX = MappingProxyType({factor: index for index, factor in enumerate(FACTOR_REGISTRY)})
_FULL_CURRENT_PA = float(v4spec.RELAY_COPY_CURRENT_PA)
_HALF_CURRENT_PA = _FULL_CURRENT_PA / 2.0
_CAUSAL_INPUT_NAMESPACE = b"public-train-factor-workspace-v3-causal-input\0"
_PREFLIGHT_NAMESPACE = b"public-train-factor-workspace-v3-electrical-preflight\0"
_CONTROL_NAMESPACE = b"public-train-factor-workspace-v3-causal-control\0"
_REPLAY_NAMESPACE = b"public-train-factor-workspace-v3-replay\0"
_RESULT_NAMESPACE = b"public-train-factor-workspace-v3-result\0"


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


def _authenticate_dependencies() -> None:
    protocol_path = Path(protocol_v3.__file__).resolve()
    v5_path = Path(v5impl.__file__).resolve()
    v5_protocol_path = Path(v5spec.__file__).resolve()
    rebuilt = protocol_v3._build_protocol()
    if (
        sha256(protocol_path.read_bytes()).hexdigest() != V3_PROTOCOL_SOURCE_SHA256
        or protocol_v3.PROTOCOL.protocol_sha256 != V3_PROTOCOL_SHA256
        or rebuilt != protocol_v3.PROTOCOL
        or rebuilt.root_record() != protocol_v3.PROTOCOL.root_record()
        or sha256(v5_path.read_bytes()).hexdigest() != V5_SOURCE_SHA256
        or sha256(v5_protocol_path.read_bytes()).hexdigest() != V5_PROTOCOL_SOURCE_SHA256
        or v5spec.PROTOCOL.protocol_sha256
        != "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"
        or STATE_SHAPE != (FACTOR_COUNT, protocol_v3.LEXICAL_CELL_COUNT)
        or EVENT_CAPACITY != 4
        or any(
            (
                protocol_v3.DEV_INPUT_COUNT,
                protocol_v3.PROTECTED_INPUT_COUNT,
                protocol_v3.DESIRED_OUTPUT_COUNT,
                protocol_v3.PROGRAM_ARTIFACT_COUNT,
                protocol_v3.PROPERTY_ROW_COUNT,
                protocol_v3.FAMILY_IDENTIFIER_COUNT,
                protocol_v3.TASK_IDENTIFIER_COUNT,
                protocol_v3.LEARNER_ARRAY_COUNT,
                protocol_v3.LEARNING_RUN_COUNT,
                protocol_v3.ELECTRICAL_EXECUTION_COUNT,
            )
        )
    ):
        raise RuntimeError("PUBLIC_TRAIN factor workspace V3 dependency changed")


def _causal_source_sha256(causal_input: protocol_v3.CausalSelectorInputKey) -> str:
    if not isinstance(causal_input, protocol_v3.CausalSelectorInputKey):
        raise TypeError("V3 source identity requires the causal selector key")
    return _namespaced_sha256(_CAUSAL_INPUT_NAMESPACE, causal_input.record())


def _read_latch(latch: object, size: int) -> np.ndarray:
    return np.asarray(
        latch.step(
            set_currents_pa=np.zeros(size),
            read_gate_currents_pa=np.full(size, _FULL_CURRENT_PA),
            direct_readout_currents_pa=np.zeros(size),
            reset_currents_pa=np.zeros(size),
        ),
        dtype=np.bool_,
    )


def _reset_latch(latch: object, size: int) -> None:
    latch.step(
        set_currents_pa=np.zeros(size),
        read_gate_currents_pa=np.zeros(size),
        direct_readout_currents_pa=np.zeros(size),
        reset_currents_pa=np.full(size, v4spec.BASKET_INHIBITION_CURRENT_PA),
    )
    if any(latch.storage_charge_pa):
        raise RuntimeError("persistent physical latch did not reset")


def _set_latch_cells(latch: object, size: int, cells: tuple[int, ...]) -> None:
    currents = np.zeros(size)
    currents[np.asarray(cells, dtype=np.intp)] = _FULL_CURRENT_PA
    latch.step(
        set_currents_pa=currents,
        read_gate_currents_pa=np.zeros(size),
        direct_readout_currents_pa=np.zeros(size),
        reset_currents_pa=np.zeros(size),
    )


def _selected_trace_spikes(
    causal_input: protocol_v3.CausalSelectorInputKey,
    *,
    lexical_gain: float = 1.0,
) -> tuple[np.ndarray, str]:
    source_sha = _causal_source_sha256(causal_input)
    selector, selected_latch = v5impl._run_selector_assemblies(
        v5spec.PROTOCOL,
        slot_assembly_cells=causal_input.slot_assembly_cells,
        slot_tokens=("slot-0", "slot-1", "slot-2", "slot-3"),
        source_sha256=source_sha,
        lexical_gain=lexical_gain,
    )
    selected = _read_latch(selected_latch, protocol_v3.LEXICAL_CELL_COUNT)
    _reset_latch(selected_latch, protocol_v3.LEXICAL_CELL_COUNT)
    expected_cells = causal_input.selected_assembly_cells if lexical_gain == 1.0 else ()
    observed_cells = tuple(int(cell) for cell in np.flatnonzero(selected))
    if lexical_gain == 1.0 and (
        selector.decision != "SELECTED_SLOT"
        or selector.selected_slot != causal_input.selected_slot
        or selector.selected_trace_cells != expected_cells
        or observed_cells != expected_cells
    ):
        raise RuntimeError("V3 physical selector disagreed with its causal key")
    if lexical_gain == 0.0 and observed_cells:
        raise RuntimeError("V3 cell-off selector emitted a selected trace")
    return selected, selector.trace_sha256


@dataclass(frozen=True, slots=True)
class FactorSetDecision:
    causal_source_sha256: str
    status: str
    ordered_factor_relays: tuple[str, ...]
    physical_occupancy: tuple[bool, ...]
    raw_factor_scores: tuple[int, ...]
    race_count: int
    candidate_exact: bool
    selector_trace_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_source_sha256": self.causal_source_sha256,
            "status": self.status,
            "ordered_factor_relays": self.ordered_factor_relays,
            "physical_occupancy": self.physical_occupancy,
            "raw_factor_scores": self.raw_factor_scores,
            "race_count": self.race_count,
            "candidate_exact": self.candidate_exact,
            "selector_trace_sha256": self.selector_trace_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ReplayBatchResult:
    name: str
    attempt_count: int
    exact_count: int
    mismatch_count: int
    status_counts: tuple[tuple[str, int], ...]
    mismatch_causal_source_sha256s: tuple[str, ...]
    checkpoint_authentication_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "attempt_count": self.attempt_count,
            "exact_count": self.exact_count,
            "mismatch_count": self.mismatch_count,
            "status_counts": self.status_counts,
            "mismatch_causal_source_sha256s": self.mismatch_causal_source_sha256s,
            "checkpoint_authentication_count": self.checkpoint_authentication_count,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class PublicTrainFactorWorkspaceV3Result:
    status: str
    protocol_sha256: str
    v5_commit: str
    electrical_preflight_sha256: str
    electrical_control_count: int
    arrays_allocated_after_preflight: bool
    fit_lesson_attempt_count: int
    fit_pass_count: int
    checkpoint_sha256: str
    learned_state_bytes: int
    excitatory_update_events: int
    inhibitory_update_events: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    untrained_exact_count: int
    untrained_stop_count: int
    label_off_exact_count: int
    label_off_stop_count: int
    shuffled_true_exact_count: int
    duplicate_control_attempt_count: int
    duplicate_control_stop_count: int
    fifth_factor_control_attempt_count: int
    fifth_factor_control_stop_count: int
    fit_replay: ReplayBatchResult
    heldout_replay: ReplayBatchResult | None
    heldout_attempt_count: int
    forbidden_read_count: int
    result_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "protocol_sha256": self.protocol_sha256,
            "v5_commit": self.v5_commit,
            "electrical_preflight_sha256": self.electrical_preflight_sha256,
            "electrical_control_count": self.electrical_control_count,
            "arrays_allocated_after_preflight": self.arrays_allocated_after_preflight,
            "fit_lesson_attempt_count": self.fit_lesson_attempt_count,
            "fit_pass_count": self.fit_pass_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "learned_state_bytes": self.learned_state_bytes,
            "excitatory_update_events": self.excitatory_update_events,
            "inhibitory_update_events": self.inhibitory_update_events,
            "committed_excitatory_ticks": self.committed_excitatory_ticks,
            "committed_inhibitory_ticks": self.committed_inhibitory_ticks,
            "excitatory_max_ticks": self.excitatory_max_ticks,
            "inhibitory_max_ticks": self.inhibitory_max_ticks,
            "untrained_exact_count": self.untrained_exact_count,
            "untrained_stop_count": self.untrained_stop_count,
            "label_off_exact_count": self.label_off_exact_count,
            "label_off_stop_count": self.label_off_stop_count,
            "shuffled_true_exact_count": self.shuffled_true_exact_count,
            "duplicate_control_attempt_count": self.duplicate_control_attempt_count,
            "duplicate_control_stop_count": self.duplicate_control_stop_count,
            "fifth_factor_control_attempt_count": self.fifth_factor_control_attempt_count,
            "fifth_factor_control_stop_count": self.fifth_factor_control_stop_count,
            "fit_replay": self.fit_replay.record(),
            "heldout_replay": (
                self.heldout_replay.record() if self.heldout_replay is not None else None
            ),
            "heldout_attempt_count": self.heldout_attempt_count,
            "forbidden_read_count": self.forbidden_read_count,
        }
        if include_sha256:
            record["result_sha256"] = self.result_sha256
        return record


class _PhysicalFactorSetCompositor:
    """Capacity-four physical occupancy/cardinality workspace."""

    __slots__ = (
        "_capacity_poison",
        "_cardinality",
        "_duplicate_poison",
        "_occupancy",
        "attempted_event_count",
        "ordered_factor_cells",
    )

    def __init__(self) -> None:
        self._occupancy = v5spec._NoLeakStorageReadoutPopulation(FACTOR_COUNT)
        self._cardinality = v5spec._NoLeakStorageReadoutPopulation(EVENT_CAPACITY + 1)
        self._duplicate_poison = v5spec._NoLeakStorageReadoutPopulation(1)
        self._capacity_poison = v5spec._NoLeakStorageReadoutPopulation(1)
        _set_latch_cells(self._cardinality, EVENT_CAPACITY + 1, (0,))
        self.attempted_event_count = 0
        self.ordered_factor_cells: list[int] = []

    def ingress(self, factor_cell: int) -> str:
        if not 0 <= factor_cell < FACTOR_COUNT:
            raise ValueError("factor ingress left the physical registry")
        self.attempted_event_count += 1
        occupancy = _read_latch(self._occupancy, FACTOR_COUNT)
        cardinality = _read_latch(self._cardinality, EVENT_CAPACITY + 1)
        if np.count_nonzero(cardinality) != 1:
            raise RuntimeError("factor-set cardinality ceased to be one-hot")

        duplicate_cells = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        )
        ingress = np.zeros(FACTOR_COUNT, dtype=np.bool_)
        ingress[factor_cell] = True
        duplicate = duplicate_cells.step(
            occupancy.astype(np.float64) * _HALF_CURRENT_PA
            + ingress.astype(np.float64) * _HALF_CURRENT_PA
        )
        if np.any(duplicate):
            _set_latch_cells(self._duplicate_poison, 1, (0,))
            return "STOP_DUPLICATE_FACTOR"

        capacity_detector = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        )
        capacity_spike = capacity_detector.step(
            np.asarray(
                (
                    (_HALF_CURRENT_PA if bool(cardinality[EVENT_CAPACITY]) else 0.0)
                    + _HALF_CURRENT_PA,
                )
            )
        )
        if bool(capacity_spike[0]):
            _set_latch_cells(self._capacity_poison, 1, (0,))
            return "STOP_EVENT_CAPACITY_4"

        old_count = int(np.flatnonzero(cardinality)[0])
        _set_latch_cells(self._occupancy, FACTOR_COUNT, (factor_cell,))
        reset = np.zeros(EVENT_CAPACITY + 1)
        reset[old_count] = v4spec.BASKET_INHIBITION_CURRENT_PA
        set_currents = np.zeros(EVENT_CAPACITY + 1)
        set_currents[old_count + 1] = _FULL_CURRENT_PA
        self._cardinality.step(
            set_currents_pa=set_currents,
            read_gate_currents_pa=np.zeros(EVENT_CAPACITY + 1),
            direct_readout_currents_pa=np.zeros(EVENT_CAPACITY + 1),
            reset_currents_pa=reset,
        )
        self.ordered_factor_cells.append(factor_cell)
        return "ACCEPTED"

    def finish(self, candidate_factors: tuple[str, ...]) -> tuple[str, tuple[bool, ...], bool]:
        occupancy = _read_latch(self._occupancy, FACTOR_COUNT)
        cardinality = _read_latch(self._cardinality, EVENT_CAPACITY + 1)
        duplicate = bool(_read_latch(self._duplicate_poison, 1)[0])
        capacity = bool(_read_latch(self._capacity_poison, 1)[0])
        if duplicate:
            return "STOP_DUPLICATE_FACTOR", tuple(bool(value) for value in occupancy), False
        if capacity:
            return "STOP_EVENT_CAPACITY_4", tuple(bool(value) for value in occupancy), False
        if not np.any(occupancy):
            return "STOP_ZERO", tuple(bool(value) for value in occupancy), False

        candidate = np.zeros(FACTOR_COUNT, dtype=np.bool_)
        candidate[np.asarray(tuple(_FACTOR_INDEX[factor] for factor in candidate_factors))] = True
        present = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            occupancy.astype(np.float64) * _HALF_CURRENT_PA
            + candidate.astype(np.float64) * _HALF_CURRENT_PA
        )
        absent = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            (~occupancy).astype(np.float64) * _HALF_CURRENT_PA
            + (~candidate).astype(np.float64) * _HALF_CURRENT_PA
        )
        equality = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            present.astype(np.float64) * _FULL_CURRENT_PA
            + absent.astype(np.float64) * _FULL_CURRENT_PA
        )
        all_match = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA * FACTOR_COUNT,
        ).step(np.asarray((np.count_nonzero(equality) * _FULL_CURRENT_PA,)))
        candidate_count = len(candidate_factors)
        cardinality_match = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            np.asarray(
                (
                    (_HALF_CURRENT_PA if bool(cardinality[candidate_count]) else 0.0)
                    + _HALF_CURRENT_PA,
                )
            )
        )
        admission = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            np.asarray(
                (
                    (_HALF_CURRENT_PA if bool(all_match[0]) else 0.0)
                    + (_HALF_CURRENT_PA if bool(cardinality_match[0]) else 0.0),
                )
            )
        )
        exact = bool(admission[0])
        return (
            "READY_FACTOR_SET" if exact else "STOP_FACTOR_SET_MISMATCH",
            tuple(bool(value) for value in occupancy),
            exact,
        )


def _run_principal_race(
    raw_scores: tuple[int, ...],
    mask_latch: object,
    *,
    source_sha256: str,
) -> v5impl.FactorRaceDecision:
    mask = _read_latch(mask_latch, FACTOR_COUNT)
    currents = v5impl._factor_currents(raw_scores)
    currents[:FACTOR_COUNT] += mask.astype(np.float64) * v4spec.BASKET_INHIBITION_CURRENT_PA
    race = CompiledRestingGlifRace.compile(step_count=v4spec.RACE_STEPS).race_many(
        currents[np.newaxis, :],
        require_unique=False,
    )[0]
    tier, _trace = v4impl._live_principal_tier(currents, race.spike_step)
    if len(tier) != race.simultaneous_spike_candidates:
        raise RuntimeError("masked factor race principal tier changed")
    decision, _relay = v5impl._relay_factor_decision(
        source_sha256=source_sha256,
        raw_scores=raw_scores,
        principal_step=race.spike_step,
        tier=tier,
    )
    return decision


def _infer_factor_set(
    *,
    causal_input: protocol_v3.CausalSelectorInputKey,
    candidate_factors: tuple[str, ...],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    lexical_gain: float = 1.0,
) -> FactorSetDecision:
    selected, selector_trace_sha = _selected_trace_spikes(
        causal_input,
        lexical_gain=lexical_gain,
    )
    selected_cells = tuple(int(cell) for cell in np.flatnonzero(selected))
    raw_scores = v5impl._raw_factor_scores(
        excitatory_ticks,
        inhibitory_ticks,
        selected_cells,
    )
    causal_source_sha = _causal_source_sha256(causal_input)
    mask_latch = v5spec._NoLeakStorageReadoutPopulation(FACTOR_COUNT)
    compositor = _PhysicalFactorSetCompositor()
    ordered: list[str] = []
    status = ""
    occupancy = (False,) * FACTOR_COUNT
    candidate_exact = False
    race_count = 0
    for masked_rank in range(EVENT_CAPACITY + 1):
        race_count += 1
        race_sha = _namespaced_sha256(
            _CAUSAL_INPUT_NAMESPACE + b"race\0",
            {"causal_source_sha256": causal_source_sha, "masked_rank": masked_rank},
        )
        race = _run_principal_race(
            raw_scores,
            mask_latch,
            source_sha256=race_sha,
        )
        if race.decision == "COLLISION":
            status = "COLLISION_STOP"
            break
        if race.decision == "STOP":
            status, occupancy, candidate_exact = compositor.finish(candidate_factors)
            break
        if race.decision != "FACTOR" or race.factor is None:
            raise RuntimeError("masked factor race emitted an unknown decision")
        factor_cell = _FACTOR_INDEX[race.factor]
        if bool(_read_latch(mask_latch, FACTOR_COUNT)[factor_cell]):
            raise RuntimeError("masked factor principal won a later race")
        _set_latch_cells(mask_latch, FACTOR_COUNT, (factor_cell,))
        ordered.append(race.factor)
        ingress_status = compositor.ingress(factor_cell)
        if ingress_status != "ACCEPTED":
            status, occupancy, candidate_exact = compositor.finish(candidate_factors)
            break
    if not status:
        raise RuntimeError("capacity-four factor workspace did not terminate")
    record = {
        "causal_source_sha256": causal_source_sha,
        "status": status,
        "ordered_factor_relays": tuple(ordered),
        "physical_occupancy": occupancy,
        "raw_factor_scores": raw_scores,
        "race_count": race_count,
        "candidate_exact": candidate_exact,
        "selector_trace_sha256": selector_trace_sha,
    }
    return FactorSetDecision(
        causal_source_sha256=causal_source_sha,
        status=status,
        ordered_factor_relays=tuple(ordered),
        physical_occupancy=occupancy,
        raw_factor_scores=raw_scores,
        race_count=race_count,
        candidate_exact=candidate_exact,
        selector_trace_sha256=selector_trace_sha,
        sha256=_namespaced_sha256(_REPLAY_NAMESPACE + b"decision\0", record),
    )


def _teacher_set_spikes(target_factors: tuple[str, ...], *, gain: float) -> np.ndarray:
    if gain not in (0.0, 1.0):
        raise ValueError("teacher-set gain changed")
    currents = np.zeros(FACTOR_COUNT)
    if gain == 1.0:
        currents[np.asarray(tuple(_FACTOR_INDEX[factor] for factor in target_factors))] = (
            v4spec.TEACHER_CLAMP_CURRENT_PA
        )
    return np.asarray(
        v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(currents),
        dtype=np.bool_,
    )


def _fit_lesson(
    *,
    causal_input: protocol_v3.CausalSelectorInputKey,
    teacher_factors: tuple[str, ...],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    teacher_gain: float,
) -> tuple[int, int, int, int]:
    selected, _selector_sha = _selected_trace_spikes(causal_input)
    selected_cells = tuple(int(cell) for cell in np.flatnonzero(selected))
    teacher_set = _teacher_set_spikes(teacher_factors, gain=teacher_gain)
    e_events = 0
    h_events = 0
    committed_e = 0
    committed_h = 0
    source_sha = _causal_source_sha256(causal_input)
    for teacher_factor in teacher_factors:
        raw_scores = v5impl._raw_factor_scores(
            excitatory_ticks,
            inhibitory_ticks,
            selected_cells,
        )
        _autonomous, autonomous_latch, scout_latch, _trace = v5impl._run_training_factor_circuit(
            source_sha256=source_sha,
            raw_scores=raw_scores,
        )
        autonomous_mask = _read_latch(autonomous_latch, FACTOR_COUNT)
        scout_mask = _read_latch(scout_latch, FACTOR_COUNT)
        teacher_mask = v5impl._teacher_spike_mask(teacher_factor, gain=teacher_gain)
        matches = v5impl._target_match_relays(
            teacher_spikes=teacher_mask,
            autonomous_readout_spikes=autonomous_mask,
        )
        e_updates, h_updates = v5impl._teacher_scout_update_relays(
            teacher_spikes=teacher_mask,
            scout_readout_spikes=scout_mask,
            target_match_spikes=matches,
        )
        gated_h = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            h_updates.astype(np.float64) * _FULL_CURRENT_PA
            + teacher_set.astype(np.float64) * v4spec.BASKET_INHIBITION_CURRENT_PA
        )
        delta_e, delta_h = v5impl._commit_local_plasticity(
            excitatory_ticks=excitatory_ticks,
            inhibitory_ticks=inhibitory_ticks,
            selected_trace_spikes=selected,
            excitatory_update_spikes=e_updates,
            inhibitory_update_spikes=np.asarray(gated_h, dtype=np.bool_),
        )
        e_events += int(np.any(e_updates))
        h_events += int(np.count_nonzero(gated_h))
        committed_e += delta_e
        committed_h += delta_h
        _reset_latch(autonomous_latch, FACTOR_COUNT)
        _reset_latch(scout_latch, FACTOR_COUNT)
    return e_events, h_events, committed_e, committed_h


def _fit_cohort(
    *,
    teacher_mapping: MappingProxyType[str, str],
    teacher_gain: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    excitatory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    telemetry = {
        "fit_lesson_attempt_count": 0,
        "excitatory_update_events": 0,
        "inhibitory_update_events": 0,
        "committed_excitatory_ticks": 0,
        "committed_inhibitory_ticks": 0,
    }
    fit_lessons = tuple(
        lesson for lesson in protocol_v3.PROTOCOL.lessons if lesson.split == protocol_v3.FIT
    )
    if len(fit_lessons) != protocol_v3.FIT_LESSON_COUNT:
        raise RuntimeError("V3 FIT schedule changed")
    for lesson in fit_lessons:
        mapped = tuple(teacher_mapping[factor] for factor in lesson.target_factors)
        teacher_factors = tuple(factor for factor in FACTOR_REGISTRY if factor in frozenset(mapped))
        e_events, h_events, delta_e, delta_h = _fit_lesson(
            causal_input=lesson.causal_input,
            teacher_factors=teacher_factors,
            excitatory_ticks=excitatory,
            inhibitory_ticks=inhibitory,
            teacher_gain=teacher_gain,
        )
        telemetry["fit_lesson_attempt_count"] += 1
        telemetry["excitatory_update_events"] += e_events
        telemetry["inhibitory_update_events"] += h_events
        telemetry["committed_excitatory_ticks"] += delta_e
        telemetry["committed_inhibitory_ticks"] += delta_h
    telemetry["excitatory_max_ticks"] = int(np.max(excitatory))
    telemetry["inhibitory_max_ticks"] = int(np.max(inhibitory))
    if np.any(excitatory > v5spec.TICK_CAP) or np.any(inhibitory > v5spec.TICK_CAP):
        raise RuntimeError("V3 learner exceeded the inherited uint8 tick cap")
    return excitatory, inhibitory, telemetry


class _FrozenCheckpoint:
    __slots__ = ("_checkpoint_sha256", "_excitatory", "_inhibitory", "_metadata")

    def __init__(
        self,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = dict(metadata)
        self._checkpoint_sha256 = sha256(
            v5impl._checkpoint_payload(
                self._metadata,
                self._excitatory,
                self._inhibitory,
            )
        ).hexdigest()
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def state_bytes(self) -> int:
        return self._excitatory.nbytes + self._inhibitory.nbytes

    def authenticate(self) -> None:
        v5impl._authenticate_state(
            checkpoint_sha256=self._checkpoint_sha256,
            metadata=self._metadata,
            excitatory_ticks=self._excitatory,
            inhibitory_ticks=self._inhibitory,
        )


def _replay(
    checkpoint: _FrozenCheckpoint,
    lessons: tuple[protocol_v3.FactorSetLesson, ...],
    *,
    name: str,
    lexical_gain: float = 1.0,
) -> ReplayBatchResult:
    checkpoint.authenticate()
    decisions = tuple(
        _infer_factor_set(
            causal_input=lesson.causal_input,
            candidate_factors=lesson.target_factors,
            excitatory_ticks=checkpoint._excitatory,
            inhibitory_ticks=checkpoint._inhibitory,
            lexical_gain=lexical_gain,
        )
        for lesson in lessons
    )
    checkpoint.authenticate()
    exact_count = sum(decision.candidate_exact for decision in decisions)
    mismatches = tuple(
        decision.causal_source_sha256 for decision in decisions if not decision.candidate_exact
    )
    status_counts = tuple(sorted(Counter(row.status for row in decisions).items()))
    record = {
        "name": name,
        "attempt_count": len(decisions),
        "exact_count": exact_count,
        "mismatch_count": len(decisions) - exact_count,
        "status_counts": status_counts,
        "mismatch_causal_source_sha256s": mismatches,
        "checkpoint_authentication_count": 2,
        "decision_sha256s": tuple(row.sha256 for row in decisions),
    }
    return ReplayBatchResult(
        name=name,
        attempt_count=len(decisions),
        exact_count=exact_count,
        mismatch_count=len(decisions) - exact_count,
        status_counts=status_counts,
        mismatch_causal_source_sha256s=mismatches,
        checkpoint_authentication_count=2,
        sha256=_namespaced_sha256(_REPLAY_NAMESPACE, record),
    )


def _checkpoint(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    *,
    electrical_preflight_sha256: str,
    telemetry: dict[str, int],
    name: str,
) -> _FrozenCheckpoint:
    metadata: dict[str, object] = {
        "schema": "public-train-factor-workspace-v3-checkpoint",
        "name": name,
        "protocol_sha256": V3_PROTOCOL_SHA256,
        "protocol_root_record": protocol_v3.PROTOCOL.root_record(),
        "v5_commit": V5_COMMIT,
        "v5_local_rule_sha256": v5spec.LOCAL_RULE_SHA256,
        "v3_teacher_rule_sha256": protocol_v3.TEACHER_RULE_SHA256,
        "electrical_preflight_sha256": electrical_preflight_sha256,
        "fit_causal_source_sha256s": tuple(
            _causal_source_sha256(lesson.causal_input)
            for lesson in protocol_v3.PROTOCOL.lessons
            if lesson.split == protocol_v3.FIT
        ),
        "fit_pass_count": 1,
        "telemetry": dict(telemetry),
        "array_shape": STATE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": v5spec.TICK_CAP,
        "dev_input_count": 0,
        "protected_input_count": 0,
        "desired_output_count": 0,
        "program_artifact_count": 0,
        "property_row_count": 0,
        "family_identifier_count": 0,
        "task_identifier_count": 0,
    }
    return _FrozenCheckpoint(excitatory, inhibitory, metadata)


def _run_masked_wta_control(control: object) -> dict[str, object]:
    mask_latch = v5spec._NoLeakStorageReadoutPopulation(FACTOR_COUNT)
    if control.name == "reset-clears-mask":
        _set_latch_cells(mask_latch, FACTOR_COUNT, (_FACTOR_INDEX["add"],))
        _reset_latch(mask_latch, FACTOR_COUNT)
    publications: list[str] = []
    status = ""
    for tier_index, tier in enumerate(control.principal_tiers):
        raw = [0] * FACTOR_COUNT
        for principal in tier:
            if principal != "STOP":
                raw[_FACTOR_INDEX[principal]] = v5spec.STOP_SCORE + 1
        decision = _run_principal_race(
            tuple(raw),
            mask_latch,
            source_sha256=_namespaced_sha256(
                _CONTROL_NAMESPACE + b"masked-race\0",
                {"control_sha256": control.sha256, "tier_index": tier_index},
            ),
        )
        if decision.decision == "COLLISION":
            status = "COLLISION_STOP"
            break
        if decision.decision == "STOP":
            status = "READY" if publications else "STOP"
            break
        if decision.factor is None:
            raise RuntimeError("masked control factor relay disappeared")
        factor_cell = _FACTOR_INDEX[decision.factor]
        _set_latch_cells(mask_latch, FACTOR_COUNT, (factor_cell,))
        publications.append(decision.factor)
        if len(publications) > EVENT_CAPACITY:
            status = "FIFTH_EVENT"
            break
    if control.expected_status == "READY_AFTER_RESET" and status == "READY":
        status = "READY_AFTER_RESET"
    masked = tuple(
        FACTOR_REGISTRY[cell] for cell in np.flatnonzero(_read_latch(mask_latch, FACTOR_COUNT))
    )
    if (
        status != control.expected_status
        or tuple(publications) != control.expected_factor_publication
        or frozenset(masked) != frozenset(control.expected_masked_factors)
    ):
        raise RuntimeError(f"masked WTA control changed: {control.name}")
    return {
        "control_sha256": control.sha256,
        "status": status,
        "publication": tuple(publications),
        "masked": masked,
    }


def _run_compositor_control(control: object) -> dict[str, object]:
    compositor = _PhysicalFactorSetCompositor()
    ingress_status = ""
    for event in control.events:
        ingress_status = compositor.ingress(_FACTOR_INDEX[event.factor])
        if ingress_status != "ACCEPTED":
            break
    candidate = (
        tuple(factor for factor in FACTOR_REGISTRY if factor in control.expected_public_factors)
        if control.expected_public_factors
        else ()
    )
    status, occupancy, exact = compositor.finish(candidate)
    public = tuple(FACTOR_REGISTRY[index] for index, active in enumerate(occupancy) if active)
    if (
        status != control.expected_status
        or (
            status == "READY_FACTOR_SET"
            and (not exact or frozenset(public) != frozenset(control.expected_public_factors))
        )
        or (
            status.startswith("STOP_")
            and status not in ("STOP_ZERO", "STOP_FACTOR_SET_MISMATCH")
            and exact
        )
    ):
        raise RuntimeError(f"factor-set compositor control changed: {control.name}")
    return {
        "control_sha256": control.sha256,
        "status": status,
        "public_factors": public if exact else (),
        "pre_poison_factors": public if not exact else (),
    }


def _execute_electrical_preflight() -> tuple[str, int, tuple[dict[str, object], ...]]:
    v5_record, v5_sha, v5_count = v5impl._execute_electrical_preflight(v5spec.PROTOCOL)
    if v5impl._namespaced_sha256(v5impl._ELECTRICAL_NAMESPACE, v5_record) != v5_sha:
        raise RuntimeError("inherited V5 electrical preflight root changed")
    masked = tuple(
        _run_masked_wta_control(control) for control in protocol_v3.PROTOCOL.masked_wta_controls
    )
    compositor = tuple(
        _run_compositor_control(control) for control in protocol_v3.PROTOCOL.compositor_controls
    )
    record = {
        "v3_protocol_sha256": V3_PROTOCOL_SHA256,
        "v5_commit": V5_COMMIT,
        "v5_electrical_preflight_sha256": v5_sha,
        "v5_electrical_control_count": v5_count,
        "masked_wta_topology_sha256": protocol_v3.v2.MASKED_WTA_TOPOLOGY_SHA256,
        "masked_wta_controls": masked,
        "compositor_topology_sha256": protocol_v3.v2.COMPOSITOR_TOPOLOGY_SHA256,
        "compositor_controls": compositor,
        "completed_before_treatment_array_allocation": True,
    }
    total = v5_count + len(masked) + len(compositor)
    return _namespaced_sha256(_PREFLIGHT_NAMESPACE, record), total, compositor


def _control_replay(
    *,
    name: str,
    teacher_mapping: MappingProxyType[str, str] | None,
    teacher_gain: float,
    electrical_preflight_sha256: str,
) -> ReplayBatchResult:
    if teacher_mapping is None:
        excitatory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
        inhibitory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
        telemetry = {
            "fit_lesson_attempt_count": 0,
            "excitatory_update_events": 0,
            "inhibitory_update_events": 0,
            "committed_excitatory_ticks": 0,
            "committed_inhibitory_ticks": 0,
            "excitatory_max_ticks": 0,
            "inhibitory_max_ticks": 0,
        }
    else:
        excitatory, inhibitory, telemetry = _fit_cohort(
            teacher_mapping=teacher_mapping,
            teacher_gain=teacher_gain,
        )
    checkpoint = _checkpoint(
        excitatory,
        inhibitory,
        electrical_preflight_sha256=electrical_preflight_sha256,
        telemetry=telemetry,
        name=name,
    )
    return _replay(
        checkpoint,
        protocol_v3.PROTOCOL.lessons,
        name=name,
    )


def _execute_once() -> PublicTrainFactorWorkspaceV3Result:
    _authenticate_dependencies()
    electrical_sha, electrical_count, compositor_controls = _execute_electrical_preflight()

    identity_mapping = MappingProxyType({factor: factor for factor in ACTIVE_FACTORS})
    shuffled_mapping = MappingProxyType(
        {
            factor: ACTIVE_FACTORS[(index + 1) % len(ACTIVE_FACTORS)]
            for index, factor in enumerate(ACTIVE_FACTORS)
        }
    )
    if any(shuffled_mapping[factor] == factor for factor in ACTIVE_FACTORS):
        raise RuntimeError("V3 shuffled-label control is not a derangement")

    untrained = _control_replay(
        name="untrained",
        teacher_mapping=None,
        teacher_gain=0.0,
        electrical_preflight_sha256=electrical_sha,
    )
    label_off = _control_replay(
        name="label-off",
        teacher_mapping=identity_mapping,
        teacher_gain=0.0,
        electrical_preflight_sha256=electrical_sha,
    )
    shuffled = _control_replay(
        name="shuffled-label",
        teacher_mapping=shuffled_mapping,
        teacher_gain=1.0,
        electrical_preflight_sha256=electrical_sha,
    )
    if untrained.exact_count != 0 or label_off.exact_count != 0 or shuffled.exact_count != 0:
        raise RuntimeError("V3 causal negative control became positive")

    excitatory, inhibitory, telemetry = _fit_cohort(
        teacher_mapping=identity_mapping,
        teacher_gain=1.0,
    )
    checkpoint = _checkpoint(
        excitatory,
        inhibitory,
        electrical_preflight_sha256=electrical_sha,
        telemetry=telemetry,
        name="treatment",
    )
    fit_lessons = tuple(
        lesson for lesson in protocol_v3.PROTOCOL.lessons if lesson.split == protocol_v3.FIT
    )
    fit_replay = _replay(checkpoint, fit_lessons, name="FIT")
    heldout_replay: ReplayBatchResult | None = None
    if fit_replay.exact_count != protocol_v3.FIT_LESSON_COUNT:
        status = FIT_STOP_STATUS
    else:
        heldout_lessons = tuple(
            lesson for lesson in protocol_v3.PROTOCOL.lessons if lesson.split == protocol_v3.HELDOUT
        )
        heldout_replay = _replay(checkpoint, heldout_lessons, name="untouched-heldout")
        status = (
            PASS_STATUS
            if heldout_replay.exact_count == protocol_v3.HELDOUT_LESSON_COUNT
            else HELDOUT_STOP_STATUS
        )

    duplicate_controls = tuple(
        row for row in compositor_controls if row["status"] == "STOP_DUPLICATE_FACTOR"
    )
    capacity_controls = tuple(
        row for row in compositor_controls if row["status"] == "STOP_EVENT_CAPACITY_4"
    )
    result = PublicTrainFactorWorkspaceV3Result(
        status=status,
        protocol_sha256=V3_PROTOCOL_SHA256,
        v5_commit=V5_COMMIT,
        electrical_preflight_sha256=electrical_sha,
        electrical_control_count=electrical_count,
        arrays_allocated_after_preflight=True,
        fit_lesson_attempt_count=int(telemetry["fit_lesson_attempt_count"]),
        fit_pass_count=1,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        learned_state_bytes=checkpoint.state_bytes,
        excitatory_update_events=int(telemetry["excitatory_update_events"]),
        inhibitory_update_events=int(telemetry["inhibitory_update_events"]),
        committed_excitatory_ticks=int(telemetry["committed_excitatory_ticks"]),
        committed_inhibitory_ticks=int(telemetry["committed_inhibitory_ticks"]),
        excitatory_max_ticks=int(telemetry["excitatory_max_ticks"]),
        inhibitory_max_ticks=int(telemetry["inhibitory_max_ticks"]),
        untrained_exact_count=untrained.exact_count,
        untrained_stop_count=dict(untrained.status_counts).get("STOP_ZERO", 0),
        label_off_exact_count=label_off.exact_count,
        label_off_stop_count=dict(label_off.status_counts).get("STOP_ZERO", 0),
        shuffled_true_exact_count=shuffled.exact_count,
        duplicate_control_attempt_count=len(duplicate_controls),
        duplicate_control_stop_count=len(duplicate_controls),
        fifth_factor_control_attempt_count=len(capacity_controls),
        fifth_factor_control_stop_count=len(capacity_controls),
        fit_replay=fit_replay,
        heldout_replay=heldout_replay,
        heldout_attempt_count=(heldout_replay.attempt_count if heldout_replay is not None else 0),
        forbidden_read_count=FORBIDDEN_READ_COUNT,
        result_sha256="",
    )
    return replace(
        result,
        result_sha256=_namespaced_sha256(
            _RESULT_NAMESPACE,
            result.record(include_sha256=False),
        ),
    )


RESULT = _execute_once()


__all__ = [
    "DESIRED_OUTPUT_COUNT",
    "DEV_INPUT_COUNT",
    "FIT_STOP_STATUS",
    "FORBIDDEN_READ_COUNT",
    "HELDOUT_STOP_STATUS",
    "PASS_STATUS",
    "PROGRAM_ARTIFACT_COUNT",
    "PROPERTY_ROW_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL_SCOPE",
    "RESULT",
    "TASK_IDENTIFIER_COUNT",
    "V3_PROTOCOL_SHA256",
    "V3_PROTOCOL_SOURCE_SHA256",
    "V5_COMMIT",
    "FactorSetDecision",
    "PublicTrainFactorWorkspaceV3Result",
    "ReplayBatchResult",
]
