"""Execute the frozen public-TRAIN role-bound factor workspace V4 once.

The learner receives only the physical selector input and its generated
content-by-role coincidence spikes.  Factor identities exist only as FIT
teacher clamps and as the registry names of physically active membership
relays.  The frozen V4 protocol is authenticated before its complete 13-row
electrical/control panel, and that panel must finish before the only two
persistent learner-state arrays are allocated.
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
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4base
from experiments.r3_pilot import public_train_factor_workspace_v4_protocol as protocol_v4
from experiments.r3_pilot import teacher_gated_factor_workspace_v5 as v5impl
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5spec

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FACTOR_WORKSPACE_V4"
PASS_STATUS = "PASS_PUBLIC_TRAIN_FACTOR_WORKSPACE_V4"
FIT_STOP_STATUS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V4_FIT_READINESS"
HELDOUT_STOP_STATUS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V4_UNTOUCHED_HELDOUT_REPLAY"
CONTROL_STOP_STATUS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V4_CAUSAL_CONTROLS"

V4_PROTOCOL_COMMIT = "c9da529"
V4_PROTOCOL_SHA256 = "83335e57fb896e9ba509a3f7f34e1bba39e7394d700c80813a702c4b34cc0f12"
V4_PROTOCOL_SOURCE_SHA256 = "9df77077bbce8ba206fc6dcf8a1df4d282005e983199020be94b5b979db533b3"
V4_PROTOCOL_TEST_SHA256 = "8980d6b8750a46fb9397dd29263dc513fa9e7a6e746f3afbf79e2a7fb896a1b4"
V5_COMMIT = "5cb01cc"
V5_SOURCE_SHA256 = "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
V5_PROTOCOL_SOURCE_SHA256 = "d62fe76444ef48f0b58723f0cede1b6ca76e86c78623eb43c07f38be8e8c8746"

FACTOR_REGISTRY = protocol_v4.FACTOR_REGISTRY
ACTIVE_FACTORS = protocol_v4.ACTIVE_FACTORS
FACTOR_COUNT = len(FACTOR_REGISTRY)
EVENT_CAPACITY = protocol_v4.EVENT_CAPACITY
STATE_SHAPE = protocol_v4.FUTURE_STATE_SHAPE
TICK_CAP = protocol_v4.TICK_CAP
TICK_DTYPE = np.dtype(protocol_v4.TICK_DTYPE)
REPRESENTATION_CELL_COUNT = protocol_v4.REPRESENTATION_CELL_COUNT

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_COUNT = 0
PROGRAM_ARTIFACT_COUNT = 0
PROPERTY_ROW_COUNT = 0
FAMILY_IDENTIFIER_COUNT = 0
TASK_IDENTIFIER_COUNT = 0
FORBIDDEN_READ_COUNT = 0

_FACTOR_INDEX = MappingProxyType({factor: index for index, factor in enumerate(FACTOR_REGISTRY)})
_FULL_CURRENT_PA = float(v4base.RELAY_COPY_CURRENT_PA)
_HALF_CURRENT_PA = _FULL_CURRENT_PA / 2.0
_CAUSAL_INPUT_NAMESPACE = b"public-train-factor-workspace-v4-causal-input\0"
_REPRESENTATION_NAMESPACE = b"public-train-factor-workspace-v4-representation\0"
_PREFLIGHT_NAMESPACE = b"public-train-factor-workspace-v4-electrical-preflight\0"
_CONTROL_NAMESPACE = b"public-train-factor-workspace-v4-executed-control\0"
_REPLAY_NAMESPACE = b"public-train-factor-workspace-v4-replay\0"
_RESULT_NAMESPACE = b"public-train-factor-workspace-v4-result\0"

_STATE_ARRAY_ALLOCATION_COUNT = 0


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


def _authenticate_dependencies() -> None:
    protocol_path = Path(protocol_v4.__file__).resolve()
    protocol_test_path = (
        protocol_path.parents[2] / "tests" / "test_public_train_factor_workspace_v4_protocol.py"
    )
    v5_path = Path(v5impl.__file__).resolve()
    v5_protocol_path = Path(v5spec.__file__).resolve()
    rebuilt = protocol_v4._build_protocol()
    if (
        sha256(protocol_path.read_bytes()).hexdigest() != V4_PROTOCOL_SOURCE_SHA256
        or sha256(protocol_test_path.read_bytes()).hexdigest() != V4_PROTOCOL_TEST_SHA256
        or protocol_v4.PROTOCOL.protocol_sha256 != V4_PROTOCOL_SHA256
        or rebuilt != protocol_v4.PROTOCOL
        or rebuilt.root_record() != protocol_v4.PROTOCOL.root_record()
        or sha256(v5_path.read_bytes()).hexdigest() != V5_SOURCE_SHA256
        or sha256(v5_protocol_path.read_bytes()).hexdigest() != V5_PROTOCOL_SOURCE_SHA256
        or v5spec.PROTOCOL.protocol_sha256
        != "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"
        or STATE_SHAPE != (FACTOR_COUNT, REPRESENTATION_CELL_COUNT)
        or STATE_SHAPE != (24, 5_632)
        or EVENT_CAPACITY != 4
        or TICK_CAP != 4
        or np.dtype("uint8") != TICK_DTYPE
        or any(
            (
                protocol_v4.DEV_INPUT_COUNT,
                protocol_v4.PROTECTED_INPUT_COUNT,
                protocol_v4.DESIRED_OUTPUT_COUNT,
                protocol_v4.PROGRAM_ARTIFACT_COUNT,
                protocol_v4.PROPERTY_ROW_COUNT,
                protocol_v4.FAMILY_IDENTIFIER_COUNT,
                protocol_v4.TASK_IDENTIFIER_COUNT,
                protocol_v4.LEARNER_ARRAY_COUNT,
                protocol_v4.LEARNING_RUN_COUNT,
                protocol_v4.ELECTRICAL_EXECUTION_COUNT,
                protocol_v4.HELDOUT_TREATMENT_RESULT_COUNT,
            )
        )
    ):
        raise RuntimeError("PUBLIC_TRAIN factor workspace V4 dependency changed")


def _causal_source_sha256(causal_input: protocol_v4.RoleBoundCausalInputKey) -> str:
    if not isinstance(causal_input, protocol_v4.RoleBoundCausalInputKey):
        raise TypeError("V4 source identity requires the role-bound causal input")
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
        reset_currents_pa=np.full(size, v4base.BASKET_INHIBITION_CURRENT_PA),
    )
    if any(latch.storage_charge_pa):
        raise RuntimeError("persistent physical latch did not reset")


def _set_latch_cells(latch: object, size: int, cells: tuple[int, ...]) -> None:
    currents = np.zeros(size)
    if cells:
        currents[np.asarray(cells, dtype=np.intp)] = _FULL_CURRENT_PA
    latch.step(
        set_currents_pa=currents,
        read_gate_currents_pa=np.zeros(size),
        direct_readout_currents_pa=np.zeros(size),
        reset_currents_pa=np.zeros(size),
    )


@dataclass(frozen=True, slots=True)
class PhysicalRepresentation:
    causal_source_sha256: str
    spike_cells: tuple[int, ...]
    selector_trace_sha256: str
    role_spike_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_source_sha256": self.causal_source_sha256,
            "spike_cells": self.spike_cells,
            "selector_trace_sha256": self.selector_trace_sha256,
            "role_spike_cells": self.role_spike_cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _physical_representation(
    causal_input: protocol_v4.RoleBoundCausalInputKey,
    *,
    cell_gain: float = 1.0,
    suppress_all_conjunctions: bool = False,
    suppress_directional_geometry: bool = False,
) -> tuple[np.ndarray, PhysicalRepresentation]:
    if cell_gain not in (0.0, 1.0):
        raise ValueError("V4 cell gain changed")
    causal_source_sha = _causal_source_sha256(causal_input)
    selector, selected_latch = v5impl._run_selector_assemblies(
        v5spec.PROTOCOL,
        slot_assembly_cells=causal_input.slot_assembly_cells,
        slot_tokens=("slot-0", "slot-1", "slot-2", "slot-3"),
        source_sha256=causal_source_sha,
        lexical_gain=cell_gain,
    )
    selected = _read_latch(selected_latch, protocol_v4.LEXICAL_CELL_COUNT)
    _reset_latch(selected_latch, protocol_v4.LEXICAL_CELL_COUNT)
    observed_selected = tuple(int(cell) for cell in np.flatnonzero(selected))
    if cell_gain == 1.0 and (
        selector.decision != "SELECTED_SLOT"
        or selector.selected_slot != causal_input.selected_slot
        or selector.selected_trace_cells != causal_input.carrier_assembly_cells
        or observed_selected != causal_input.carrier_assembly_cells
    ):
        raise RuntimeError("V4 physical selector disagreed with its causal input")
    if cell_gain == 0.0 and observed_selected:
        raise RuntimeError("V4 cell-off selector emitted content spikes")

    role_currents = np.zeros(protocol_v4.SELECTOR_ROLE_CELL_COUNT)
    if cell_gain == 1.0:
        role_currents[np.asarray(causal_input.selector_role_cells, dtype=np.intp)] = (
            _FULL_CURRENT_PA
        )
    role_spikes = np.asarray(
        v4impl._DiscreteNeuronPopulation(
            protocol_v4.SELECTOR_ROLE_CELL_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(role_currents),
        dtype=np.bool_,
    )
    if suppress_directional_geometry:
        role_spikes[: protocol_v4.SELECTOR_DIRECTION_CELL_COUNT] = False
        selected_role_start = (
            protocol_v4.SELECTOR_DIRECTION_CELL_COUNT + protocol_v4.SELECTOR_CUE_ROLE_CELL_COUNT
        )
        role_spikes[selected_role_start:] = False

    conjunction_spikes = np.zeros(protocol_v4.CONJUNCTION_CELL_COUNT, dtype=np.bool_)
    if not suppress_all_conjunctions:
        conjunction_currents = (
            selected.astype(np.float64)[np.newaxis, :] * _HALF_CURRENT_PA
            + role_spikes.astype(np.float64)[:, np.newaxis] * _HALF_CURRENT_PA
        ).reshape(-1)
        conjunction_spikes = np.asarray(
            v4impl._DiscreteNeuronPopulation(
                protocol_v4.CONJUNCTION_CELL_COUNT,
                threshold_pa=_FULL_CURRENT_PA,
            ).step(conjunction_currents),
            dtype=np.bool_,
        )
    representation = np.concatenate((selected, conjunction_spikes))
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(representation))
    role_cells = tuple(int(cell) for cell in np.flatnonzero(role_spikes))
    if (
        cell_gain == 1.0
        and not suppress_all_conjunctions
        and not suppress_directional_geometry
        and spike_cells != causal_input.representation_spike_cells
    ):
        raise RuntimeError("V4 physical role-bound projection changed")
    record = {
        "causal_source_sha256": causal_source_sha,
        "spike_cells": spike_cells,
        "selector_trace_sha256": selector.trace_sha256,
        "role_spike_cells": role_cells,
    }
    return representation, PhysicalRepresentation(
        causal_source_sha256=causal_source_sha,
        spike_cells=spike_cells,
        selector_trace_sha256=selector.trace_sha256,
        role_spike_cells=role_cells,
        sha256=_namespaced_sha256(_REPRESENTATION_NAMESPACE, record),
    )


@dataclass(frozen=True, slots=True)
class MembershipDecision:
    causal_source_sha256: str
    status: str
    physical_occupancy: tuple[bool, ...]
    factor_membership: tuple[str, ...]
    raw_scores: tuple[int, ...]
    membership_count: int
    stop_spike: bool
    candidate_exact: bool
    representation_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_source_sha256": self.causal_source_sha256,
            "status": self.status,
            "physical_occupancy": self.physical_occupancy,
            "factor_membership": self.factor_membership,
            "raw_scores": self.raw_scores,
            "membership_count": self.membership_count,
            "stop_spike": self.stop_spike,
            "candidate_exact": self.candidate_exact,
            "representation_sha256": self.representation_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


class _PhysicalUnorderedSetCollector:
    """Simultaneous occupancy/cardinality collector with absorbing poisons."""

    __slots__ = ("_capacity_poison", "_cardinality", "_duplicate_poison", "_occupancy")

    def __init__(self) -> None:
        self._occupancy = v5spec._NoLeakStorageReadoutPopulation(FACTOR_COUNT)
        self._cardinality = v5spec._NoLeakStorageReadoutPopulation(EVENT_CAPACITY + 1)
        self._duplicate_poison = v5spec._NoLeakStorageReadoutPopulation(1)
        self._capacity_poison = v5spec._NoLeakStorageReadoutPopulation(1)

    def resolve(
        self,
        event_counts: tuple[int, ...],
        candidate_factors: tuple[str, ...],
    ) -> tuple[str, tuple[bool, ...], int, bool]:
        if len(event_counts) != FACTOR_COUNT or any(
            type(count) is not int or count < 0 for count in event_counts
        ):
            raise ValueError("V4 factor-event multiplicities changed")
        events = np.asarray(event_counts, dtype=np.int64)
        present = events > 0
        present_cells = tuple(int(cell) for cell in np.flatnonzero(present))
        _set_latch_cells(self._occupancy, FACTOR_COUNT, present_cells)

        duplicate_spikes = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=2.0 * _FULL_CURRENT_PA,
        ).step(events.astype(np.float64) * _FULL_CURRENT_PA)
        if np.any(duplicate_spikes):
            _set_latch_cells(self._duplicate_poison, 1, (0,))
        distinct_count = len(present_cells)
        capacity_spike = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=(EVENT_CAPACITY + 1) * _FULL_CURRENT_PA,
        ).step(np.asarray((distinct_count * _FULL_CURRENT_PA,)))
        if bool(capacity_spike[0]):
            _set_latch_cells(self._capacity_poison, 1, (0,))
        if distinct_count <= EVENT_CAPACITY:
            _set_latch_cells(self._cardinality, EVENT_CAPACITY + 1, (distinct_count,))

        occupancy = _read_latch(self._occupancy, FACTOR_COUNT)
        duplicate = bool(_read_latch(self._duplicate_poison, 1)[0])
        capacity = bool(_read_latch(self._capacity_poison, 1)[0])
        if tuple(bool(value) for value in occupancy) != tuple(bool(value) for value in present):
            raise RuntimeError("V4 simultaneous physical occupancy changed")
        if duplicate:
            return (
                "STOP_DUPLICATE_FACTOR",
                tuple(bool(value) for value in occupancy),
                distinct_count,
                False,
            )
        if capacity:
            return (
                "STOP_EVENT_CAPACITY_4",
                tuple(bool(value) for value in occupancy),
                distinct_count,
                False,
            )
        if distinct_count == 0:
            return "STOP_ZERO", tuple(bool(value) for value in occupancy), 0, False

        candidate = np.zeros(FACTOR_COUNT, dtype=np.bool_)
        for factor in candidate_factors:
            candidate[_FACTOR_INDEX[factor]] = True
        if int(np.count_nonzero(candidate)) != len(candidate_factors):
            raise ValueError("V4 candidate factor set contains a duplicate")
        equal_cells = v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step((occupancy == candidate).astype(np.float64) * _FULL_CURRENT_PA)
        all_equal = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=FACTOR_COUNT * _FULL_CURRENT_PA,
        ).step(np.asarray((int(np.count_nonzero(equal_cells)) * _FULL_CURRENT_PA,)))
        cardinality = _read_latch(self._cardinality, EVENT_CAPACITY + 1)
        cardinality_match = bool(cardinality[len(candidate_factors)])
        admission = v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            np.asarray(
                (
                    (_HALF_CURRENT_PA if bool(all_equal[0]) else 0.0)
                    + (_HALF_CURRENT_PA if cardinality_match else 0.0),
                )
            )
        )
        exact = bool(admission[0])
        return (
            "READY_FACTOR_SET" if exact else "STOP_FACTOR_SET_MISMATCH",
            tuple(bool(value) for value in occupancy),
            distinct_count,
            exact,
        )


def _membership_spikes(raw_scores: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if len(raw_scores) != FACTOR_COUNT:
        raise ValueError("V4 membership scores left the factor registry")
    nonnegative = np.maximum(np.asarray(raw_scores, dtype=np.float64), 0.0)
    currents = nonnegative / float(protocol_v4.MEMBERSHIP_SCORE_THRESHOLD) * _FULL_CURRENT_PA
    membership = np.asarray(
        v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(currents),
        dtype=np.bool_,
    )
    any_factor = bool(
        v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(np.asarray((int(np.count_nonzero(membership)) * _FULL_CURRENT_PA,)))[0]
    )
    stop_current = _FULL_CURRENT_PA + (v4base.BASKET_INHIBITION_CURRENT_PA if any_factor else 0.0)
    stop_spike = bool(
        v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(np.asarray((stop_current,)))[0]
    )
    if stop_spike == any_factor:
        raise RuntimeError("V4 any-factor STOP inhibition changed")
    return membership, stop_spike


def _resolve_membership(
    *,
    raw_scores: tuple[int, ...],
    candidate_factors: tuple[str, ...],
    causal_source_sha256: str,
    representation_sha256: str,
    event_counts_override: tuple[int, ...] | None = None,
) -> MembershipDecision:
    membership_spikes, stop_spike = _membership_spikes(raw_scores)
    event_counts = (
        tuple(int(value) for value in membership_spikes)
        if event_counts_override is None
        else event_counts_override
    )
    if event_counts_override is not None and tuple(count > 0 for count in event_counts) != tuple(
        bool(value) for value in membership_spikes
    ):
        raise ValueError("V4 event injection disagreed with membership relays")
    collector = _PhysicalUnorderedSetCollector()
    status, occupancy, membership_count, exact = collector.resolve(
        event_counts,
        candidate_factors,
    )
    if stop_spike and status != "STOP_ZERO":
        raise RuntimeError("V4 STOP fired beside a factor membership")
    if not stop_spike and status == "STOP_ZERO":
        raise RuntimeError("V4 factor membership failed to inhibit STOP")
    factor_membership = tuple(
        FACTOR_REGISTRY[index] for index, active in enumerate(occupancy) if active
    )
    record = {
        "causal_source_sha256": causal_source_sha256,
        "status": status,
        "physical_occupancy": occupancy,
        "factor_membership": factor_membership,
        "raw_scores": raw_scores,
        "membership_count": membership_count,
        "stop_spike": stop_spike,
        "candidate_exact": exact,
        "representation_sha256": representation_sha256,
    }
    return MembershipDecision(
        **record,
        sha256=_namespaced_sha256(_REPLAY_NAMESPACE + b"decision\0", record),
    )


def _raw_scores(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    representation_spikes: np.ndarray,
) -> tuple[int, ...]:
    active = np.flatnonzero(representation_spikes)
    if active.size == 0:
        return (0,) * FACTOR_COUNT
    excitatory = np.sum(excitatory_ticks[:, active], axis=1, dtype=np.int64)
    inhibitory = np.sum(inhibitory_ticks[:, active], axis=1, dtype=np.int64)
    return tuple(int(value) for value in excitatory - inhibitory)


def _infer(
    *,
    causal_input: protocol_v4.RoleBoundCausalInputKey,
    candidate_factors: tuple[str, ...],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    cell_gain: float = 1.0,
) -> MembershipDecision:
    representation, publication = _physical_representation(
        causal_input,
        cell_gain=cell_gain,
    )
    return _resolve_membership(
        raw_scores=_raw_scores(excitatory_ticks, inhibitory_ticks, representation),
        candidate_factors=candidate_factors,
        causal_source_sha256=publication.causal_source_sha256,
        representation_sha256=publication.sha256,
    )


def _teacher_spikes(teacher_factors: tuple[str, ...], *, gain: float) -> np.ndarray:
    if gain not in (0.0, 1.0):
        raise ValueError("V4 teacher gain changed")
    currents = np.zeros(FACTOR_COUNT)
    if gain == 1.0:
        currents[np.asarray(tuple(_FACTOR_INDEX[factor] for factor in teacher_factors))] = (
            v4base.TEACHER_CLAMP_CURRENT_PA
        )
    return np.asarray(
        v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(currents),
        dtype=np.bool_,
    )


def _local_update_relays(
    representation_spikes: np.ndarray,
    teacher_factors: tuple[str, ...],
    *,
    teacher_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    teacher = _teacher_spikes(teacher_factors, gain=teacher_gain)
    event_spike = bool(
        v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(np.asarray((_FULL_CURRENT_PA if np.any(representation_spikes) else 0.0,)))[0]
    )
    any_teacher = bool(
        v4impl._DiscreteNeuronPopulation(
            1,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(np.asarray((int(np.count_nonzero(teacher)) * _FULL_CURRENT_PA,)))[0]
    )
    excitatory = np.asarray(
        v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(
            teacher.astype(np.float64) * _HALF_CURRENT_PA
            + (np.full(FACTOR_COUNT, _HALF_CURRENT_PA) if event_spike else 0.0)
        ),
        dtype=np.bool_,
    )
    scout_currents = np.zeros(FACTOR_COUNT)
    if event_spike and any_teacher:
        scout_currents.fill(_FULL_CURRENT_PA)
    scout_currents += teacher.astype(np.float64) * v4base.BASKET_INHIBITION_CURRENT_PA
    inhibitory = np.asarray(
        v4impl._DiscreteNeuronPopulation(
            FACTOR_COUNT,
            threshold_pa=_FULL_CURRENT_PA,
        ).step(scout_currents),
        dtype=np.bool_,
    )
    if np.any(excitatory & inhibitory):
        raise RuntimeError("V4 co-target inhibition ceased to be branch local")
    return excitatory, inhibitory


def _commit_local_ticks(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    representation_spikes: np.ndarray,
    excitatory_update_spikes: np.ndarray,
    inhibitory_update_spikes: np.ndarray,
) -> tuple[int, int]:
    active_cells = tuple(int(cell) for cell in np.flatnonzero(representation_spikes))
    committed_e = 0
    committed_h = 0
    for factor_cell in np.flatnonzero(excitatory_update_spikes):
        for representation_cell in active_cells:
            current = int(excitatory_ticks[factor_cell, representation_cell])
            if current < TICK_CAP:
                excitatory_ticks[factor_cell, representation_cell] = current + 1
                committed_e += 1
    for factor_cell in np.flatnonzero(inhibitory_update_spikes):
        for representation_cell in active_cells:
            current = int(inhibitory_ticks[factor_cell, representation_cell])
            if current < TICK_CAP:
                inhibitory_ticks[factor_cell, representation_cell] = current + 1
                committed_h += 1
    return committed_e, committed_h


def _array_frame(name: str, array: np.ndarray) -> bytes:
    if (
        array.shape != STATE_SHAPE
        or array.dtype != TICK_DTYPE
        or not array.flags.c_contiguous
        or array.nbytes != protocol_v4.FUTURE_STATE_BYTES // 2
    ):
        raise ValueError("V4 checkpoint array framing changed")
    header = _canonical_json(
        {
            "name": name,
            "shape": array.shape,
            "dtype": array.dtype.str,
            "nbytes": array.nbytes,
        }
    )
    body = memoryview(array).cast("B").tobytes()
    return len(header).to_bytes(8, "little") + header + body


def _checkpoint_payload(
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> bytes:
    metadata_frame = _canonical_json(metadata)
    return b"".join(
        (
            b"public-train-factor-workspace-v4-checkpoint\0",
            len(metadata_frame).to_bytes(8, "little"),
            metadata_frame,
            _array_frame("excitatory_ticks", excitatory_ticks),
            _array_frame("inhibitory_ticks", inhibitory_ticks),
        )
    )


@dataclass(frozen=True, slots=True)
class ExecutedControl:
    name: str
    status: str
    factor_membership_count: int
    exact: bool
    protocol_control_sha256: str
    evidence_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "factor_membership_count": self.factor_membership_count,
            "exact": self.exact,
            "protocol_control_sha256": self.protocol_control_sha256,
            "evidence_sha256": self.evidence_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ElectricalPreflight:
    protocol_sha256: str
    control_count: int
    pass_count: int
    arrays_allocated_before_completion: int
    controls: tuple[ExecutedControl, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "protocol_sha256": self.protocol_sha256,
            "control_count": self.control_count,
            "pass_count": self.pass_count,
            "arrays_allocated_before_completion": self.arrays_allocated_before_completion,
            "controls": tuple(row.record() for row in self.controls),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _control_membership(
    raw: tuple[int, ...],
    candidate: tuple[str, ...],
    *,
    event_counts: tuple[int, ...] | None = None,
) -> MembershipDecision:
    source_sha = _namespaced_sha256(
        _CONTROL_NAMESPACE + b"source\0",
        {"raw": raw, "candidate": candidate, "event_counts": event_counts},
    )
    return _resolve_membership(
        raw_scores=raw,
        candidate_factors=candidate,
        causal_source_sha256=source_sha,
        representation_sha256=source_sha,
        event_counts_override=event_counts,
    )


def _execute_control(control: protocol_v4.FutureCausalControl) -> ExecutedControl:
    raw = [0] * FACTOR_COUNT
    status = ""
    count = 0
    exact = False
    evidence: object
    if control.name == "zero-membership-stop":
        decision = _control_membership(tuple(raw), ())
        status, count, exact, evidence = (
            decision.status,
            decision.membership_count,
            decision.candidate_exact,
            decision.sha256,
        )
    elif control.name == "below-support-stop":
        raw[0] = protocol_v4.MEMBERSHIP_SCORE_THRESHOLD - 1
        decision = _control_membership(tuple(raw), ())
        status, count, exact, evidence = (
            decision.status,
            decision.membership_count,
            decision.candidate_exact,
            decision.sha256,
        )
    elif control.name in (
        "once-exposed-membership",
        "two-membership-unordered-set",
        "exact-four-membership",
        "fifth-membership-poison",
        "factor-set-permutation",
    ):
        membership_count = {
            "once-exposed-membership": 1,
            "two-membership-unordered-set": 2,
            "exact-four-membership": 4,
            "fifth-membership-poison": 5,
            "factor-set-permutation": 2,
        }[control.name]
        for index in range(membership_count):
            raw[index] = protocol_v4.MEMBERSHIP_SCORE_THRESHOLD
        candidate = tuple(FACTOR_REGISTRY[:membership_count])
        if control.name == "fifth-membership-poison":
            candidate = ()
        elif control.name == "factor-set-permutation":
            candidate = tuple(reversed(candidate))
        decision = _control_membership(tuple(raw), candidate)
        status, count, exact, evidence = (
            decision.status,
            decision.membership_count,
            decision.candidate_exact,
            decision.sha256,
        )
    elif control.name == "duplicate-relay-poison":
        raw[0] = protocol_v4.MEMBERSHIP_SCORE_THRESHOLD
        multiplicities = (2,) + (0,) * (FACTOR_COUNT - 1)
        decision = _control_membership(
            tuple(raw),
            (FACTOR_REGISTRY[0],),
            event_counts=multiplicities,
        )
        status, count, exact, evidence = (
            decision.status,
            decision.membership_count,
            decision.candidate_exact,
            decision.sha256,
        )
    elif control.name in ("role-binding-off", "direction-role-off"):
        left_class, right_class = protocol_v4.PROTOCOL.binding_classes[:2]
        template = protocol_v4.PROTOCOL.templates[0]
        left_input = protocol_v4._causal_input(left_class, template)
        right_input = protocol_v4._causal_input(right_class, template)
        if control.name == "role-binding-off":
            left, left_publication = _physical_representation(
                left_input,
                suppress_all_conjunctions=True,
            )
            right, right_publication = _physical_representation(
                right_input,
                suppress_all_conjunctions=True,
            )
            status = "STOP_CONTENT_ONLY_ALIAS" if np.array_equal(left, right) else "READY"
        else:
            left, left_publication = _physical_representation(
                left_input,
                suppress_directional_geometry=True,
            )
            right, right_publication = _physical_representation(
                right_input,
                suppress_directional_geometry=True,
            )
            status = "STOP_ROLE_CAUSAL_ALIAS" if np.array_equal(left, right) else "READY"
        count = 0
        exact = False
        evidence = (left_publication.sha256, right_publication.sha256)
    elif control.name == "teacher-off":
        lesson = next(row for row in protocol_v4.PROTOCOL.lessons if row.split == protocol_v4.FIT)
        representation, publication = _physical_representation(lesson.causal_input)
        e_updates, h_updates = _local_update_relays(
            representation,
            lesson.target_factors,
            teacher_gain=0.0,
        )
        status = "STOP_ZERO" if not np.any(e_updates) and not np.any(h_updates) else "READY"
        count = 0
        exact = False
        evidence = publication.sha256
    elif control.name == "cell-off":
        lesson = next(row for row in protocol_v4.PROTOCOL.lessons if row.split == protocol_v4.FIT)
        representation, publication = _physical_representation(
            lesson.causal_input,
            cell_gain=0.0,
        )
        e_updates, h_updates = _local_update_relays(
            representation,
            lesson.target_factors,
            teacher_gain=1.0,
        )
        decision = _control_membership((0,) * FACTOR_COUNT, ())
        status = (
            "STOP_ZERO"
            if not np.any(e_updates) and not np.any(h_updates) and decision.stop_spike
            else "READY"
        )
        count = 0
        exact = False
        evidence = publication.sha256
    elif control.name == "shuffled-teacher":
        mapping = _teacher_derangement()
        lesson = next(row for row in protocol_v4.PROTOCOL.lessons if row.split == protocol_v4.FIT)
        mapped = tuple(mapping[factor] for factor in lesson.target_factors)
        spikes = _teacher_spikes(mapped, gain=1.0)
        status = (
            "STOP_SHUFFLED_LABEL"
            if frozenset(mapped).isdisjoint(lesson.target_factors) and np.any(spikes)
            else "READY"
        )
        count = 0
        exact = False
        evidence = tuple(int(cell) for cell in np.flatnonzero(spikes))
    else:
        raise RuntimeError(f"unknown frozen V4 control: {control.name}")

    if (
        status != control.expected_status
        or count != control.expected_factor_membership_count
        or exact is not control.expected_exact
    ):
        raise RuntimeError(f"frozen V4 electrical control changed: {control.name}")
    evidence_sha = _namespaced_sha256(
        _CONTROL_NAMESPACE + b"evidence\0",
        {"control_sha256": control.sha256, "evidence": evidence},
    )
    record = {
        "name": control.name,
        "status": status,
        "factor_membership_count": count,
        "exact": exact,
        "protocol_control_sha256": control.sha256,
        "evidence_sha256": evidence_sha,
    }
    return ExecutedControl(
        **record,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE, record),
    )


def _execute_electrical_preflight() -> ElectricalPreflight:
    if _STATE_ARRAY_ALLOCATION_COUNT != 0:
        raise RuntimeError("V4 arrays existed before the electrical controls")
    controls = tuple(_execute_control(control) for control in protocol_v4.PROTOCOL.future_controls)
    record = {
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "control_count": len(controls),
        "pass_count": len(controls),
        "arrays_allocated_before_completion": _STATE_ARRAY_ALLOCATION_COUNT,
        "controls": tuple(row.record() for row in controls),
    }
    if len(controls) != 13:
        raise RuntimeError("V4 complete 13-control panel changed")
    return ElectricalPreflight(
        protocol_sha256=V4_PROTOCOL_SHA256,
        control_count=len(controls),
        pass_count=len(controls),
        arrays_allocated_before_completion=_STATE_ARRAY_ALLOCATION_COUNT,
        controls=controls,
        sha256=_namespaced_sha256(_PREFLIGHT_NAMESPACE, record),
    )


def _allocate_treatment_state(preflight: ElectricalPreflight) -> tuple[np.ndarray, np.ndarray]:
    global _STATE_ARRAY_ALLOCATION_COUNT
    if (
        preflight.control_count != 13
        or preflight.pass_count != 13
        or preflight.arrays_allocated_before_completion != 0
        or preflight != ELECTRICAL_PREFLIGHT
        or preflight.sha256
        != _namespaced_sha256(_PREFLIGHT_NAMESPACE, preflight.record(include_sha256=False))
        or _STATE_ARRAY_ALLOCATION_COUNT != 0
    ):
        raise RuntimeError("V4 treatment arrays requested before authenticated preflight")
    excitatory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    _STATE_ARRAY_ALLOCATION_COUNT += 1
    inhibitory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    _STATE_ARRAY_ALLOCATION_COUNT += 1
    return excitatory, inhibitory


def _fit_once(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> dict[str, int]:
    fit_lessons = tuple(
        lesson for lesson in protocol_v4.PROTOCOL.lessons if lesson.split == protocol_v4.FIT
    )
    if len(fit_lessons) != 48:
        raise RuntimeError("V4 FIT schedule changed")
    telemetry = {
        "fit_lesson_attempt_count": 0,
        "excitatory_update_events": 0,
        "inhibitory_update_events": 0,
        "committed_excitatory_ticks": 0,
        "committed_inhibitory_ticks": 0,
    }
    for lesson in fit_lessons:
        representation, _publication = _physical_representation(lesson.causal_input)
        e_updates, h_updates = _local_update_relays(
            representation,
            lesson.target_factors,
            teacher_gain=1.0,
        )
        delta_e, delta_h = _commit_local_ticks(
            excitatory,
            inhibitory,
            representation,
            e_updates,
            h_updates,
        )
        telemetry["fit_lesson_attempt_count"] += 1
        telemetry["excitatory_update_events"] += int(np.count_nonzero(e_updates))
        telemetry["inhibitory_update_events"] += int(np.count_nonzero(h_updates))
        telemetry["committed_excitatory_ticks"] += delta_e
        telemetry["committed_inhibitory_ticks"] += delta_h
    telemetry["excitatory_max_ticks"] = int(np.max(excitatory))
    telemetry["inhibitory_max_ticks"] = int(np.max(inhibitory))
    if (
        telemetry["fit_lesson_attempt_count"] != 48
        or np.any(excitatory > TICK_CAP)
        or np.any(inhibitory > TICK_CAP)
    ):
        raise RuntimeError("V4 one-pass local learner changed")
    return telemetry


class _FrozenCheckpoint:
    __slots__ = ("_checkpoint_sha256", "_excitatory", "_inhibitory", "_metadata")

    def __init__(
        self,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        if (
            _STATE_ARRAY_ALLOCATION_COUNT != 2
            or excitatory.shape != STATE_SHAPE
            or inhibitory.shape != STATE_SHAPE
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
        ):
            raise RuntimeError("V4 checkpoint state changed")
        excitatory.setflags(write=False)
        inhibitory.setflags(write=False)
        self._excitatory = excitatory
        self._inhibitory = inhibitory
        self._metadata = dict(metadata)
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(
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
        if self._excitatory.flags.writeable or self._inhibitory.flags.writeable:
            raise RuntimeError("V4 frozen checkpoint became writeable")
        observed = sha256(
            _checkpoint_payload(
                self._metadata,
                self._excitatory,
                self._inhibitory,
            )
        ).hexdigest()
        if observed != self._checkpoint_sha256:
            raise RuntimeError("V4 frozen checkpoint authentication failed")


def _freeze_checkpoint(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    telemetry: dict[str, int],
) -> _FrozenCheckpoint:
    metadata: dict[str, object] = {
        "schema": "public-train-factor-workspace-v4-checkpoint",
        "protocol_commit": V4_PROTOCOL_COMMIT,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "protocol_root_record": protocol_v4.PROTOCOL.root_record(),
        "electrical_preflight_sha256": ELECTRICAL_PREFLIGHT.sha256,
        "fit_causal_source_sha256s": tuple(
            _causal_source_sha256(lesson.causal_input)
            for lesson in protocol_v4.PROTOCOL.lessons
            if lesson.split == protocol_v4.FIT
        ),
        "fit_pass_count": 1,
        "telemetry": dict(telemetry),
        "array_shape": STATE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": TICK_CAP,
        "state_array_allocation_count": _STATE_ARRAY_ALLOCATION_COUNT,
        "dev_input_count": 0,
        "protected_input_count": 0,
        "desired_output_count": 0,
        "program_artifact_count": 0,
        "property_row_count": 0,
        "family_identifier_count": 0,
        "task_identifier_count": 0,
    }
    return _FrozenCheckpoint(excitatory, inhibitory, metadata)


@dataclass(frozen=True, slots=True)
class ReplayResult:
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


def _replay(
    checkpoint: _FrozenCheckpoint,
    lessons: tuple[protocol_v4.RoleBoundLesson, ...],
    *,
    name: str,
    cell_gain: float = 1.0,
    candidate_mapping: MappingProxyType[str, str] | None = None,
) -> ReplayResult:
    checkpoint.authenticate()
    decisions: list[MembershipDecision] = []
    for lesson in lessons:
        candidate = lesson.target_factors
        if candidate_mapping is not None:
            candidate = tuple(candidate_mapping[factor] for factor in candidate)
        decisions.append(
            _infer(
                causal_input=lesson.causal_input,
                candidate_factors=candidate,
                excitatory_ticks=checkpoint._excitatory,
                inhibitory_ticks=checkpoint._inhibitory,
                cell_gain=cell_gain,
            )
        )
    checkpoint.authenticate()
    exact_count = sum(row.candidate_exact for row in decisions)
    mismatches = tuple(row.causal_source_sha256 for row in decisions if not row.candidate_exact)
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
    return ReplayResult(
        name=name,
        attempt_count=len(decisions),
        exact_count=exact_count,
        mismatch_count=len(decisions) - exact_count,
        status_counts=status_counts,
        mismatch_causal_source_sha256s=mismatches,
        checkpoint_authentication_count=2,
        sha256=_namespaced_sha256(_REPLAY_NAMESPACE, record),
    )


def _teacher_derangement() -> MappingProxyType[str, str]:
    mapping = MappingProxyType(
        {
            factor: ACTIVE_FACTORS[(index + 2) % len(ACTIVE_FACTORS)]
            for index, factor in enumerate(ACTIVE_FACTORS)
        }
    )
    if any(mapping[factor] == factor for factor in ACTIVE_FACTORS) or any(
        not frozenset(mapping[factor] for factor in target).isdisjoint(target)
        for target in protocol_v4.FACTOR_SET_CLASSES
    ):
        raise RuntimeError("V4 shuffled teacher is not a derangement")
    return mapping


def _teacher_off_replay(
    heldout_lessons: tuple[protocol_v4.RoleBoundLesson, ...],
) -> tuple[int, int, int]:
    attempts = 0
    e_events = 0
    h_events = 0
    for lesson in heldout_lessons:
        representation, _publication = _physical_representation(lesson.causal_input)
        e_updates, h_updates = _local_update_relays(
            representation,
            lesson.target_factors,
            teacher_gain=0.0,
        )
        attempts += 1
        e_events += int(np.count_nonzero(e_updates))
        h_events += int(np.count_nonzero(h_updates))
    return attempts, e_events, h_events


@dataclass(frozen=True, slots=True)
class PublicTrainFactorWorkspaceV4Result:
    status: str
    protocol_commit: str
    protocol_sha256: str
    electrical_preflight_sha256: str
    electrical_control_count: int
    electrical_control_pass_count: int
    arrays_allocated_after_preflight: bool
    state_array_allocation_count: int
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
    fit_replay: ReplayResult
    heldout_replay: ReplayResult | None
    heldout_attempt_count: int
    teacher_off_attempt_count: int
    teacher_off_excitatory_update_events: int
    teacher_off_inhibitory_update_events: int
    cell_off_replay: ReplayResult | None
    shuffled_replay: ReplayResult | None
    duplicate_control_stop_count: int
    fifth_factor_control_stop_count: int
    forbidden_read_count: int
    result_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "protocol_commit": self.protocol_commit,
            "protocol_sha256": self.protocol_sha256,
            "electrical_preflight_sha256": self.electrical_preflight_sha256,
            "electrical_control_count": self.electrical_control_count,
            "electrical_control_pass_count": self.electrical_control_pass_count,
            "arrays_allocated_after_preflight": self.arrays_allocated_after_preflight,
            "state_array_allocation_count": self.state_array_allocation_count,
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
            "fit_replay": self.fit_replay.record(),
            "heldout_replay": (
                self.heldout_replay.record() if self.heldout_replay is not None else None
            ),
            "heldout_attempt_count": self.heldout_attempt_count,
            "teacher_off_attempt_count": self.teacher_off_attempt_count,
            "teacher_off_excitatory_update_events": (self.teacher_off_excitatory_update_events),
            "teacher_off_inhibitory_update_events": (self.teacher_off_inhibitory_update_events),
            "cell_off_replay": (
                self.cell_off_replay.record() if self.cell_off_replay is not None else None
            ),
            "shuffled_replay": (
                self.shuffled_replay.record() if self.shuffled_replay is not None else None
            ),
            "duplicate_control_stop_count": self.duplicate_control_stop_count,
            "fifth_factor_control_stop_count": self.fifth_factor_control_stop_count,
            "forbidden_read_count": self.forbidden_read_count,
        }
        if include_sha256:
            record["result_sha256"] = self.result_sha256
        return record


def _execute_once() -> PublicTrainFactorWorkspaceV4Result:
    _authenticate_dependencies()
    preflight = ELECTRICAL_PREFLIGHT
    excitatory, inhibitory = _allocate_treatment_state(preflight)
    telemetry = _fit_once(excitatory, inhibitory)
    checkpoint = _freeze_checkpoint(excitatory, inhibitory, telemetry)

    fit_lessons = tuple(
        lesson for lesson in protocol_v4.PROTOCOL.lessons if lesson.split == protocol_v4.FIT
    )
    fit_replay = _replay(checkpoint, fit_lessons, name="FIT")
    heldout_replay: ReplayResult | None = None
    cell_off_replay: ReplayResult | None = None
    shuffled_replay: ReplayResult | None = None
    teacher_off_attempts = 0
    teacher_off_e = 0
    teacher_off_h = 0
    if fit_replay.exact_count != protocol_v4.FIT_LESSON_COUNT:
        status = FIT_STOP_STATUS
    else:
        heldout_lessons = tuple(
            lesson for lesson in protocol_v4.PROTOCOL.lessons if lesson.split == protocol_v4.HELDOUT
        )
        heldout_replay = _replay(checkpoint, heldout_lessons, name="untouched-heldout")
        teacher_off_attempts, teacher_off_e, teacher_off_h = _teacher_off_replay(heldout_lessons)
        cell_off_replay = _replay(
            checkpoint,
            heldout_lessons,
            name="cell-off-heldout",
            cell_gain=0.0,
        )
        shuffled_replay = _replay(
            checkpoint,
            heldout_lessons,
            name="shuffled-teacher-heldout",
            candidate_mapping=_teacher_derangement(),
        )
        negative_controls_pass = (
            teacher_off_attempts == protocol_v4.HELDOUT_LESSON_COUNT
            and teacher_off_e == 0
            and teacher_off_h == 0
            and cell_off_replay.exact_count == 0
            and dict(cell_off_replay.status_counts).get("STOP_ZERO", 0)
            == protocol_v4.HELDOUT_LESSON_COUNT
            and shuffled_replay.exact_count == 0
        )
        if not negative_controls_pass:
            status = CONTROL_STOP_STATUS
        elif heldout_replay.exact_count != protocol_v4.HELDOUT_LESSON_COUNT:
            status = HELDOUT_STOP_STATUS
        else:
            status = PASS_STATUS

    duplicate_control_stop_count = sum(
        row.status == "STOP_DUPLICATE_FACTOR" for row in preflight.controls
    )
    fifth_factor_control_stop_count = sum(
        row.status == "STOP_EVENT_CAPACITY_4" for row in preflight.controls
    )
    result = PublicTrainFactorWorkspaceV4Result(
        status=status,
        protocol_commit=V4_PROTOCOL_COMMIT,
        protocol_sha256=V4_PROTOCOL_SHA256,
        electrical_preflight_sha256=preflight.sha256,
        electrical_control_count=preflight.control_count,
        electrical_control_pass_count=preflight.pass_count,
        arrays_allocated_after_preflight=True,
        state_array_allocation_count=_STATE_ARRAY_ALLOCATION_COUNT,
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
        fit_replay=fit_replay,
        heldout_replay=heldout_replay,
        heldout_attempt_count=(heldout_replay.attempt_count if heldout_replay else 0),
        teacher_off_attempt_count=teacher_off_attempts,
        teacher_off_excitatory_update_events=teacher_off_e,
        teacher_off_inhibitory_update_events=teacher_off_h,
        cell_off_replay=cell_off_replay,
        shuffled_replay=shuffled_replay,
        duplicate_control_stop_count=duplicate_control_stop_count,
        fifth_factor_control_stop_count=fifth_factor_control_stop_count,
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


_authenticate_dependencies()
ELECTRICAL_PREFLIGHT = _execute_electrical_preflight()
RESULT = _execute_once()


__all__ = [
    "CONTROL_STOP_STATUS",
    "DESIRED_OUTPUT_COUNT",
    "DEV_INPUT_COUNT",
    "ELECTRICAL_PREFLIGHT",
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
    "V4_PROTOCOL_COMMIT",
    "V4_PROTOCOL_SHA256",
    "V4_PROTOCOL_SOURCE_SHA256",
    "ExecutedControl",
    "MembershipDecision",
    "PhysicalRepresentation",
    "PublicTrainFactorWorkspaceV4Result",
    "ReplayResult",
]
