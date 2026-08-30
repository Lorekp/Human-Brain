"""Self-posed joint prompt-factor plus visible-I/O packed frontier.

The production seam has two operations.  ``build_frozen_joint_frontier``
first resolves an authenticated V5 prompt and builds a desired-free packed
typed frontier.  ``FrozenJointPromptFactorFrontier.select`` is the only place
that may inspect desired outputs or materialize a selected proof.

Prompt semantics never enter through factor names.  Ordered one-hot V5 relay
cells drive fixed neuronal lineage-coincidence populations.  Compiler-owned
generic family records contribute only their production lineage; they cannot
choose a literal, operand orientation, e-class, witness, or source program.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import teacher_gated_factor_workspace_v5 as factor_v5
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as factor_protocol
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from humanbrain.resting_glif_tier_beam import (
    RETAINED,
    STOP_LEADING_TIER,
    CompiledRestingGlifTierBeam,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

FRONTIER_READY = "JOINT_PROMPT_FACTOR_PACKED_FULL_TYPED_V1_READY"
STOP_FACTOR_ADMISSION = "STOP_JOINT_PROMPT_FACTOR_ADMISSION_V1"
STOP_ENGINE = "STOP_JOINT_PACKED_FULL_TYPED_ENGINE_V1"
SELECTED = "JOINT_PROMPT_FACTOR_PACKED_FULL_TYPED_V1_SELECTED"
STOP_NO_EXACT = "STOP_JOINT_PROMPT_FACTOR_PACKED_FULL_TYPED_V1_NO_EXACT"
STOP_AMBIGUOUS = "STOP_JOINT_PROMPT_FACTOR_PACKED_FULL_TYPED_V1_AMBIGUOUS"
STOP_INTEGRITY = "STOP_JOINT_PROMPT_FACTOR_PACKED_FULL_TYPED_V1_INTEGRITY"

PASS_MATRIX = "PASS_SELF_POSED_JOINT_PROMPT_FACTOR_VISIBLE_IO_V1"
STOP_MATRIX = "STOP_SELF_POSED_JOINT_PROMPT_FACTOR_VISIBLE_IO_V1"

V5_SOURCE_SHA256 = "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
V5_SOURCE_COMMIT = "5cb01cc"
V5_PROTOCOL_SHA256 = "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"
V5_FACTOR_REGISTRY_SHA256 = "34ba29c17861af6822702fd73455a3923ec16fd722d2b579e459022b3ec4a939"
V5_CHECKPOINT_SHA256 = "96e29d03e5b2d9f699281d62fa159bb047add7f0024cfe44ce24c11413024c0e"
V5_ATOMIC_RESULT_SHA256 = "9b75e031caf1c40729d223342a641eeb712a63457bc55ba3eef819e1239aeae8"
V5_COMPOSITION_RESULT_SHA256 = "f6d96fa61256ed1a0658ed67f432d0cb7aaa9b2556e77ff6f8376a307795abc7"
V5_CONTROL_RESULT_SHA256 = "c2290b393bad73c32b5247a3a8722a5930c05f93561191ccd2e3dba472735d29"
FULL_TYPED_SOURCE_SHA256 = "e488cc1a461b148b96c854a66ff7ffacc39384bcbe59f61787021b72eb85704f"
FULL_TYPED_SOURCE_COMMIT = "b0e95e8"

_NAMESPACE = b"joint-prompt-factor-packed-full-typed-v1\0"
_FACTOR_CELL_COUNT = 24
_MAX_FACTOR_EVENTS = 2
_LESS_THAN_RELAY_CELL = 5
_SUM_RELAY_CELL = 7
_COUNT_RELAY_CELL = 8
_SUPPORTED_GENERIC_LINEAGES = (
    (_SUM_RELAY_CELL,),
    (_COUNT_RELAY_CELL, _LESS_THAN_RELAY_CELL),
)

_RELAY_CURRENT_PA = 5_000.0
_MIN_CURRENT_PA = 600.0
_STOP_CURRENT_PA = 2_000.0
_MAX_CURRENT_PA = 3_400.0
_FEATURE_WEIGHT_PA = 2_800.0
_INHIBITION_CURRENT_PA = 3_000.0
_RACE_STEPS = 600


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


class _DiscreteNeuronPopulation:
    """Reset-on-spike current-integrating neurons for causal relay gates."""

    __slots__ = ("_membrane", "_threshold_pa")

    def __init__(self, size: int, *, threshold_pa: float) -> None:
        if type(size) is not int or size < 1 or threshold_pa <= 0.0:
            raise ValueError("joint discrete population requires a positive domain")
        self._membrane = np.zeros(size, dtype=np.float64)
        self._threshold_pa = float(threshold_pa)

    def step(self, currents_pa: np.ndarray) -> np.ndarray:
        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != self._membrane.shape or not np.all(np.isfinite(currents)):
            raise ValueError("joint discrete population current domain changed")
        self._membrane += currents
        spikes = self._membrane >= self._threshold_pa
        self._membrane[spikes] = 0.0
        return np.asarray(spikes, dtype=np.bool_)


class _Intervention(NamedTuple):
    name: str
    factor_projection: str
    visible_io_relay_gain: float


_TREATMENT = _Intervention("treatment", "identity", 1.0)
_FACTOR_OFF = _Intervention("factor-off", "off", 1.0)
_DERANGED = _Intervention("deranged-relays", "deranged", 1.0)
_VISIBLE_IO_OFF = _Intervention("visible-io-match-off", "identity", 0.0)


@dataclass(frozen=True, slots=True)
class PromptAdmissionAudit:
    batch_sha256: str
    resolution_sha256: str
    bridge_sha256: str
    raw_relay_cells: tuple[int, ...]
    projected_relay_cells: tuple[tuple[int, ...], tuple[int, ...]]
    projection_name: str
    projection_sha256: str
    supported_lineages: tuple[tuple[int, ...], ...]
    lineage_spikes: tuple[bool, ...]
    selected_lineage: tuple[int, ...] | None
    status: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "batch_sha256": self.batch_sha256,
            "resolution_sha256": self.resolution_sha256,
            "bridge_sha256": self.bridge_sha256,
            "raw_relay_cells": self.raw_relay_cells,
            "projected_relay_cells": self.projected_relay_cells,
            "projection_name": self.projection_name,
            "projection_sha256": self.projection_sha256,
            "supported_lineages": self.supported_lineages,
            "lineage_spikes": self.lineage_spikes,
            "selected_lineage": self.selected_lineage,
            "status": self.status,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RootAdmissionAudit:
    root_eclass_ids: tuple[str, ...]
    root_group_lineages: tuple[tuple[tuple[int, ...], ...], ...]
    group_lineage_spikes: tuple[tuple[bool, ...], ...]
    root_admission_spikes: tuple[bool, ...]
    family_lineage_sha256: str
    circuit_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "root_eclass_ids": self.root_eclass_ids,
            "root_group_lineages": self.root_group_lineages,
            "group_lineage_spikes": self.group_lineage_spikes,
            "root_admission_spikes": self.root_admission_spikes,
            "family_lineage_sha256": self.family_lineage_sha256,
            "circuit_sha256": self.circuit_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class JointPhysicalRaceAudit:
    status: str
    candidate_keys: tuple[str, ...]
    desired_sha256s: tuple[str, ...]
    detector_sha256s: tuple[str, ...]
    raw_match_spikes: tuple[tuple[bool, ...], ...]
    relayed_match_spikes: tuple[tuple[bool, ...], ...]
    admission_spikes: tuple[bool, ...]
    visible_io_relay_gain_hex: str
    currents_pa: tuple[float, ...]
    currents_sha256: str
    tier_sha256: str
    exact_admitted_keys: tuple[str, ...]
    winner_key: str | None
    sha256: str


@dataclass(frozen=True, slots=True)
class JointFrontierState:
    status: str
    construction_certificate_sha256: str
    prompt_admission_sha256: str
    projected_relay_sha256: str
    root_admission_sha256: str | None
    packed_frontier_sha256: str | None
    operator_checkpoint_sha256: str | None
    v5_source_commit: str
    full_typed_source_commit: str
    full_typed_source_sha256: str
    build_concrete_proof_witnesses: int
    visible_io_relay_gain_hex: str
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class JointSelection:
    status: str
    stop_reason: str | None
    frontier_sha256: str
    prompt_admission_sha256: str
    root_admission_sha256: str | None
    joint_race: JointPhysicalRaceAudit | None
    joint_winner_key: str | None
    packed_selection_sha256: str | None
    packed_race_sha256: str | None
    packed_winner_key: str | None
    packed_selected_eclass_id: str | None
    packed_concrete_proof_witnesses: int | None
    winner_identity_equal: bool | None
    selected_eclass_id: str | None
    witnesses: tuple[full_typed.PackedFullTypedWitness, ...]
    checkpoint_before_sha256: str | None
    checkpoint_after_sha256: str | None
    sha256: str


@dataclass(frozen=True, slots=True)
class MatrixRowResult:
    row_id: str
    intervention: str
    build_status: str
    selection_status: str
    source: str | None
    prompt_admission_sha256: str
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class SelfPosedJointMatrixResult:
    verdict: str
    rows: tuple[MatrixRowResult, ...]
    treatment_pass_count: int
    control_stop_count: int
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    sha256: str


class _JointBuildReceipt(NamedTuple):
    authority: object
    bridge: factor_v5.FrozenTeacherGatedFactorBridge
    prompt_admission: PromptAdmissionAudit
    root_admission: RootAdmissionAudit | None
    frontier: full_typed.PackedFullTypedFrontier | None
    intervention: _Intervention
    status: str
    projected_relay_sha256: str
    sha256: str


def _receipt_payload(receipt: _JointBuildReceipt) -> dict[str, object]:
    return {
        "prompt_admission_sha256": receipt.prompt_admission.sha256,
        "root_admission_sha256": (
            None if receipt.root_admission is None else receipt.root_admission.sha256
        ),
        "packed_frontier_sha256": (None if receipt.frontier is None else receipt.frontier.sha256),
        "intervention": tuple(receipt.intervention),
        "status": receipt.status,
        "projected_relay_sha256": receipt.projected_relay_sha256,
        "v5_source_commit": V5_SOURCE_COMMIT,
        "v5_source_sha256": V5_SOURCE_SHA256,
        "full_typed_source_commit": FULL_TYPED_SOURCE_COMMIT,
        "full_typed_source_sha256": FULL_TYPED_SOURCE_SHA256,
    }


def _make_receipt_authority() -> tuple[
    Callable[..., _JointBuildReceipt], Callable[[_JointBuildReceipt], bool]
]:
    authority = object()
    issued: dict[int, _JointBuildReceipt] = {}

    def issue(
        *,
        bridge: factor_v5.FrozenTeacherGatedFactorBridge,
        prompt_admission: PromptAdmissionAudit,
        root_admission: RootAdmissionAudit | None,
        frontier: full_typed.PackedFullTypedFrontier | None,
        intervention: _Intervention,
        status: str,
        projected_relays: np.ndarray,
    ) -> _JointBuildReceipt:
        projected = np.ascontiguousarray(projected_relays, dtype=np.bool_)
        if projected.shape != (_MAX_FACTOR_EVENTS, _FACTOR_CELL_COUNT):
            raise RuntimeError("joint construction projected-relay domain changed")
        projected_cells = tuple(
            tuple(int(cell) for cell in np.flatnonzero(row)) for row in projected
        )
        if projected_cells != prompt_admission.projected_relay_cells:
            raise RuntimeError("joint construction lost its executed relay receipt")
        projected_sha = sha256(memoryview(projected).cast("B")).hexdigest()
        partial = _JointBuildReceipt(
            authority=authority,
            bridge=bridge,
            prompt_admission=prompt_admission,
            root_admission=root_admission,
            frontier=frontier,
            intervention=intervention,
            status=status,
            projected_relay_sha256=projected_sha,
            sha256="",
        )
        receipt = partial._replace(
            sha256=_digest("joint-build-certificate", _receipt_payload(partial))
        )
        issued[id(receipt)] = receipt
        return receipt

    def authenticate(receipt: _JointBuildReceipt) -> bool:
        return bool(
            type(receipt) is _JointBuildReceipt
            and issued.get(id(receipt)) is receipt
            and receipt.authority is authority
            and receipt.sha256 == _digest("joint-build-certificate", _receipt_payload(receipt))
        )

    return issue, authenticate


_issue_joint_build_receipt, _authenticate_joint_build_receipt = _make_receipt_authority()


def _bridge_record(bridge: factor_v5.FrozenTeacherGatedFactorBridge) -> dict[str, object]:
    if not isinstance(bridge, factor_v5.FrozenTeacherGatedFactorBridge):
        raise TypeError("joint frontier requires a frozen V5 factor bridge")
    composition = bridge.composition_result
    controls = bridge.compositor_control_result
    summary = bridge.build_summary
    source_path = Path(cast(str, factor_v5.__file__)).resolve()
    if sha256(source_path.read_bytes()).hexdigest() != V5_SOURCE_SHA256:
        raise RuntimeError("joint frontier V5 learner source changed")
    if (
        factor_protocol.PROTOCOL.protocol_sha256 != V5_PROTOCOL_SHA256
        or summary.factor_registry_sha256 != V5_FACTOR_REGISTRY_SHA256
        or bridge.checkpoint_sha256 != V5_CHECKPOINT_SHA256
        or bridge.atomic_result.sha256 != V5_ATOMIC_RESULT_SHA256
        or composition is None
        or composition.sha256 != V5_COMPOSITION_RESULT_SHA256
        or controls is None
        or controls.sha256 != V5_CONTROL_RESULT_SHA256
        or (
            summary.public_input_count,
            summary.canonical_input_count,
            summary.protected_input_count,
        )
        != (0, 0, 0)
    ):
        raise RuntimeError("joint frontier V5 prerequisite authentication failed")
    return {
        "source_sha256": V5_SOURCE_SHA256,
        "source_commit": V5_SOURCE_COMMIT,
        "protocol_sha256": V5_PROTOCOL_SHA256,
        "factor_registry_sha256": V5_FACTOR_REGISTRY_SHA256,
        "checkpoint_sha256": V5_CHECKPOINT_SHA256,
        "atomic_result_sha256": V5_ATOMIC_RESULT_SHA256,
        "composition_result_sha256": V5_COMPOSITION_RESULT_SHA256,
        "control_result_sha256": V5_CONTROL_RESULT_SHA256,
        "public_input_count": 0,
        "canonical_input_count": 0,
        "protected_input_count": 0,
    }


def _authenticate_full_typed_source() -> None:
    source_path = Path(cast(str, full_typed.__file__)).resolve()
    if sha256(source_path.read_bytes()).hexdigest() != FULL_TYPED_SOURCE_SHA256:
        raise RuntimeError("joint frontier packed full-typed source changed")


def _factor_projection(name: str) -> tuple[np.ndarray, str]:
    projection = np.zeros((_FACTOR_CELL_COUNT, _FACTOR_CELL_COUNT), dtype=np.float64)
    if name == "identity":
        np.fill_diagonal(projection, 1.0)
    elif name == "deranged":
        for source in range(_FACTOR_CELL_COUNT):
            projection[source, (source + 1) % _FACTOR_CELL_COUNT] = 1.0
    elif name != "off":
        raise ValueError("unknown joint factor projection")
    projection = np.ascontiguousarray(projection, dtype="<f8")
    root = sha256(memoryview(projection).cast("B")).hexdigest()
    projection.setflags(write=False)
    return projection, root


def _relay_trace(
    resolution: factor_v5.TeacherGatedFactorResolution,
) -> tuple[int, ...]:
    cells: list[int] = []
    for decision in resolution.span_decisions:
        relays = decision.factor_relay_spike_cells
        if len(relays) > 1:
            raise RuntimeError("V5 factor relay was not one-hot")
        if relays:
            cell = relays[0]
            if (
                type(cell) is not int
                or not 0 <= cell < _FACTOR_CELL_COUNT
                or decision.unique_winner_cell != cell
            ):
                raise RuntimeError("V5 factor relay provenance changed")
            cells.append(cell)
    trace = tuple(cells)
    compositor = resolution.compositor
    if (
        len(trace) not in (1, 2)
        or len(resolution.events) != len(trace)
        or compositor.accepted_event_count != len(trace)
        or compositor.duplicate_count != 0
        or compositor.overlap_poison
        or compositor.capacity_poison
        or compositor.status != ("ONE_EVENT" if len(trace) == 1 else "TWO_EVENTS")
        or compositor.post_reset_active_count != 0
    ):
        raise RuntimeError("V5 prompt resolution did not close causally")
    return trace


def _project_relay_trace(raw_trace: tuple[int, ...], projection: np.ndarray) -> np.ndarray:
    raw = np.zeros((_MAX_FACTOR_EVENTS, _FACTOR_CELL_COUNT), dtype=np.float64)
    for ordinal, cell in enumerate(raw_trace):
        raw[ordinal, cell] = 1.0
    projected_currents = np.ascontiguousarray(raw @ projection * _RELAY_CURRENT_PA)
    relays: list[np.ndarray] = []
    for row in projected_currents:
        population = _DiscreteNeuronPopulation(_FACTOR_CELL_COUNT, threshold_pa=_RELAY_CURRENT_PA)
        relays.append(population.step(row))
    return np.ascontiguousarray(np.stack(relays), dtype=np.bool_)


def _lineage_spikes(
    projected_relays: np.ndarray,
    event_count: int,
    lineages: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    if (
        projected_relays.shape != (_MAX_FACTOR_EVENTS, _FACTOR_CELL_COUNT)
        or event_count not in (1, 2)
        or not lineages
    ):
        raise ValueError("joint lineage circuit domain changed")
    count_spikes = np.zeros(_MAX_FACTOR_EVENTS, dtype=np.float64)
    count_spikes[event_count - 1] = 1.0
    inputs = np.concatenate((projected_relays.astype(np.float64).ravel(), count_spikes))
    weights = np.zeros((len(lineages), inputs.size), dtype=np.float64)
    for row, lineage in enumerate(lineages):
        if len(lineage) not in (1, 2) or any(
            type(cell) is not int or not 0 <= cell < _FACTOR_CELL_COUNT for cell in lineage
        ):
            raise ValueError("generic family lineage left the relay domain")
        contribution = _RELAY_CURRENT_PA / (len(lineage) + 1)
        for ordinal, cell in enumerate(lineage):
            weights[row, ordinal * _FACTOR_CELL_COUNT + cell] = contribution
        weights[row, _MAX_FACTOR_EVENTS * _FACTOR_CELL_COUNT + len(lineage) - 1] = contribution
    currents = np.ascontiguousarray(weights @ inputs)
    population = _DiscreteNeuronPopulation(len(lineages), threshold_pa=_RELAY_CURRENT_PA)
    return population.step(currents)


def _resolve_prompt_admission(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
    batch: factor_v5.AuthenticatedSpanBatch,
    intervention: _Intervention,
) -> tuple[PromptAdmissionAudit, np.ndarray]:
    bridge_record = _bridge_record(bridge)
    bridge_sha = _digest("bridge-prerequisite", bridge_record)
    resolution = bridge.resolve(batch)
    raw_trace = _relay_trace(resolution)
    projection, projection_sha = _factor_projection(intervention.factor_projection)
    projected = _project_relay_trace(raw_trace, projection)
    spikes = _lineage_spikes(projected, len(raw_trace), _SUPPORTED_GENERIC_LINEAGES)
    selected = tuple(np.flatnonzero(spikes))
    selected_lineage = _SUPPORTED_GENERIC_LINEAGES[selected[0]] if len(selected) == 1 else None
    status = FRONTIER_READY if selected_lineage is not None else STOP_FACTOR_ADMISSION
    projected_cells = cast(
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple(tuple(int(cell) for cell in np.flatnonzero(row)) for row in projected),
    )
    payload = {
        "batch_sha256": batch.authentication_sha256,
        "resolution_sha256": resolution.semantic_sha256,
        "bridge_sha256": bridge_sha,
        "raw_relay_cells": raw_trace,
        "projected_relay_cells": projected_cells,
        "projection_name": intervention.factor_projection,
        "projection_sha256": projection_sha,
        "supported_lineages": _SUPPORTED_GENERIC_LINEAGES,
        "lineage_spikes": tuple(bool(value) for value in spikes),
        "selected_lineage": selected_lineage,
        "status": status,
    }
    audit = PromptAdmissionAudit(
        batch_sha256=batch.authentication_sha256,
        resolution_sha256=resolution.semantic_sha256,
        bridge_sha256=bridge_sha,
        raw_relay_cells=raw_trace,
        projected_relay_cells=projected_cells,
        projection_name=intervention.factor_projection,
        projection_sha256=projection_sha,
        supported_lineages=_SUPPORTED_GENERIC_LINEAGES,
        lineage_spikes=tuple(bool(value) for value in spikes),
        selected_lineage=selected_lineage,
        status=status,
        sha256=_digest("prompt-admission", payload),
    )
    return audit, projected


def _generic_family_lineage(
    family_id: str,
    family_index: dict[str, packed_v3.PackedFamily],
    memo: dict[str, tuple[int, ...]],
    active: set[str],
) -> tuple[int, ...]:
    cached = memo.get(family_id)
    if cached is not None:
        return cached
    if family_id in active:
        raise RuntimeError("generic typed family lineage became cyclic")
    family = family_index.get(family_id)
    if type(family) is not packed_v3.PackedFamily:
        raise RuntimeError("generic typed family lineage references an unknown family")
    active.add(family_id)
    production = family.production
    if production in {"collection-role", "item-role", "literal"}:
        lineage: tuple[int, ...] = ()
    elif production in {"less-than-item-literal", "less-than-literal-item"}:
        lineage = (_LESS_THAN_RELAY_CELL,)
    elif production == "predicate-binder":
        child_lines = tuple(
            _generic_family_lineage(child, family_index, memo, active)
            for child in family.child_family_ids
        )
        nonempty = tuple(item for item in child_lines if item)
        if len(nonempty) != 1:
            raise RuntimeError("generic predicate binder lineage changed")
        lineage = nonempty[0]
    elif production == "sum":
        lineage = (_SUM_RELAY_CELL,)
    elif production == "count":
        child_lines = tuple(
            _generic_family_lineage(child, family_index, memo, active)
            for child in family.child_family_ids
        )
        nonempty = tuple(item for item in child_lines if item)
        if len(nonempty) != 1:
            raise RuntimeError("generic count lineage changed")
        lineage = (_COUNT_RELAY_CELL, *nonempty[0])
    else:
        raise RuntimeError("joint slice encountered an unknown generic production")
    active.remove(family_id)
    memo[family_id] = lineage
    return lineage


def _root_admission(
    frontier: full_typed.PackedFullTypedFrontier,
    prompt: PromptAdmissionAudit,
    projected_relays: np.ndarray,
) -> RootAdmissionAudit:
    family_index = {family.family_id: family for family in frontier.families}
    if len(family_index) != len(frontier.families):
        raise RuntimeError("generic typed family identity collision")
    memo: dict[str, tuple[int, ...]] = {}
    nested_lineages: list[tuple[tuple[int, ...], ...]] = []
    nested_spikes: list[tuple[bool, ...]] = []
    admission_currents: list[float] = []
    for root in frontier.root_eclasses:
        lineages = tuple(
            _generic_family_lineage(group.family_id, family_index, memo, set())
            for group in root.group_refs
        )
        if not lineages or any(not lineage for lineage in lineages):
            raise RuntimeError("root e-class lost its generic factor lineage")
        spikes = _lineage_spikes(projected_relays, len(prompt.raw_relay_cells), lineages)
        nested_lineages.append(lineages)
        nested_spikes.append(tuple(bool(value) for value in spikes))
        admission_currents.append(_RELAY_CURRENT_PA * float(np.count_nonzero(spikes)))
    root_population = _DiscreteNeuronPopulation(
        len(admission_currents), threshold_pa=_RELAY_CURRENT_PA
    )
    root_spikes = root_population.step(np.asarray(admission_currents, dtype=np.float64))
    root_ids = tuple(root.eclass_id for root in frontier.root_eclasses)
    family_payload = tuple(sorted((key, value) for key, value in memo.items()))
    family_sha = _digest("generic-family-lineages", family_payload)
    circuit_payload = {
        "root_eclass_ids": root_ids,
        "root_group_lineages": tuple(nested_lineages),
        "prompt_projection_sha256": prompt.projection_sha256,
        "event_count": len(prompt.raw_relay_cells),
        "relay_current_pa_hex": _RELAY_CURRENT_PA.hex(),
    }
    circuit_sha = _digest("root-admission-circuit", circuit_payload)
    payload = {
        "root_eclass_ids": root_ids,
        "root_group_lineages": tuple(nested_lineages),
        "group_lineage_spikes": tuple(nested_spikes),
        "root_admission_spikes": tuple(bool(value) for value in root_spikes),
        "family_lineage_sha256": family_sha,
        "circuit_sha256": circuit_sha,
    }
    return RootAdmissionAudit(
        root_eclass_ids=root_ids,
        root_group_lineages=tuple(nested_lineages),
        group_lineage_spikes=tuple(nested_spikes),
        root_admission_spikes=tuple(bool(value) for value in root_spikes),
        family_lineage_sha256=family_sha,
        circuit_sha256=circuit_sha,
        sha256=_digest("root-admission", payload),
    )


def _joint_physical_race(
    frontier: full_typed.PackedFullTypedFrontier,
    admission: RootAdmissionAudit,
    desired_values: tuple[int, ...],
    *,
    visible_io_relay_gain: float,
) -> JointPhysicalRaceAudit:
    roots = frontier.root_eclasses
    keys = tuple(root.eclass_id for root in roots)
    if keys != admission.root_eclass_ids or not roots:
        raise RuntimeError("joint root admission no longer aligns with the packed bank")
    desired = tuple(
        output_cells.TypedOutputCellBank.encode(value, declared_type=output_cells.OutputType.INT)
        for value in desired_values
    )
    if len(desired) != len(frontier.state.case_arguments):
        raise ValueError("desired outputs must align with the typed cases")
    predictions_by_case = tuple(
        tuple(root.physical_outputs[case] for root in roots) for case in range(len(desired))
    )
    detectors = tuple(
        output_cells.FixedTypedCoincidenceDetector.for_target(target) for target in desired
    )
    results_by_case = tuple(
        detector.detect_many(predictions)
        for detector, predictions in zip(detectors, predictions_by_case, strict=True)
    )
    raw = np.asarray(
        [
            [bool(results_by_case[case][root].spiked) for case in range(len(desired))]
            for root in range(len(roots))
        ],
        dtype=np.bool_,
    )
    relayed_columns: list[np.ndarray] = []
    for case in range(len(desired)):
        population = _DiscreteNeuronPopulation(len(roots), threshold_pa=_RELAY_CURRENT_PA)
        currents = raw[:, case].astype(np.float64) * _RELAY_CURRENT_PA * visible_io_relay_gain
        relayed_columns.append(population.step(currents))
    relayed = np.ascontiguousarray(np.stack(relayed_columns, axis=1), dtype=np.bool_)
    admission_column = np.asarray(admission.root_admission_spikes, dtype=np.bool_)[:, None]
    features = np.concatenate((relayed, admission_column), axis=1).astype(np.float64)
    feature_count = features.shape[1]
    bias = _STOP_CURRENT_PA - (feature_count - 0.5) * _FEATURE_WEIGHT_PA
    candidate_currents = np.clip(
        np.sum(features, axis=1) * _FEATURE_WEIGHT_PA + bias,
        _MIN_CURRENT_PA,
        _MAX_CURRENT_PA,
    )
    currents = np.ascontiguousarray(
        np.concatenate((candidate_currents, np.asarray((_STOP_CURRENT_PA,))))
    )
    selector = CompiledRestingGlifTierBeam.compile(
        candidate_count=len(currents),
        beam_width=1,
        step_count=_RACE_STEPS,
        min_current_pa=_MIN_CURRENT_PA,
        max_current_pa=_MAX_CURRENT_PA,
        inhibition_current_pa=_INHIBITION_CURRENT_PA,
    )
    tier = selector.select(currents, stop_cell=len(keys))
    if len(tier.rounds) != 1:
        raise RuntimeError("joint semantic selector did not close in one tier")
    exact_cells = tier.retained_candidate_cells
    exact_keys = tuple(keys[cell] for cell in exact_cells)
    if tier.status == STOP_LEADING_TIER:
        status = STOP_NO_EXACT
        winner = None
    elif tier.status == RETAINED and len(exact_keys) == 1:
        cell = exact_cells[0]
        if not bool(admission_column[cell, 0]) or not bool(np.all(relayed[cell])):
            raise RuntimeError("joint non-exact candidate entered the physical leading tier")
        status = SELECTED
        winner = exact_keys[0]
    elif tier.status == RETAINED and len(exact_keys) > 1:
        status = STOP_AMBIGUOUS
        winner = None
    else:
        raise RuntimeError("joint semantic selector returned an unsupported state")
    currents_le = np.ascontiguousarray(currents, dtype="<f8")
    currents_sha = sha256(memoryview(currents_le).cast("B")).hexdigest()
    tier_sha = sha256(_canonical_json(asdict(tier))).hexdigest()
    payload = {
        "status": status,
        "candidate_keys": keys,
        "desired_sha256s": tuple(item.sha256 for item in desired),
        "detector_sha256s": tuple(item.sha256 for item in detectors),
        "raw_match_spikes": tuple(tuple(bool(value) for value in row) for row in raw),
        "relayed_match_spikes": tuple(tuple(bool(value) for value in row) for row in relayed),
        "admission_spikes": admission.root_admission_spikes,
        "visible_io_relay_gain_hex": visible_io_relay_gain.hex(),
        "currents_sha256": currents_sha,
        "tier_sha256": tier_sha,
        "exact_admitted_keys": exact_keys,
        "winner_key": winner,
    }
    return JointPhysicalRaceAudit(
        status=status,
        candidate_keys=keys,
        desired_sha256s=tuple(item.sha256 for item in desired),
        detector_sha256s=tuple(item.sha256 for item in detectors),
        raw_match_spikes=tuple(tuple(bool(value) for value in row) for row in raw),
        relayed_match_spikes=tuple(tuple(bool(value) for value in row) for row in relayed),
        admission_spikes=admission.root_admission_spikes,
        visible_io_relay_gain_hex=visible_io_relay_gain.hex(),
        currents_pa=tuple(float(value) for value in currents),
        currents_sha256=currents_sha,
        tier_sha256=tier_sha,
        exact_admitted_keys=exact_keys,
        winner_key=winner,
        sha256=_digest("joint-physical-race", payload),
    )


def _state_payload(state: JointFrontierState) -> dict[str, object]:
    return {
        "status": state.status,
        "construction_certificate_sha256": state.construction_certificate_sha256,
        "prompt_admission_sha256": state.prompt_admission_sha256,
        "projected_relay_sha256": state.projected_relay_sha256,
        "root_admission_sha256": state.root_admission_sha256,
        "packed_frontier_sha256": state.packed_frontier_sha256,
        "operator_checkpoint_sha256": state.operator_checkpoint_sha256,
        "v5_source_commit": state.v5_source_commit,
        "full_typed_source_commit": state.full_typed_source_commit,
        "full_typed_source_sha256": state.full_typed_source_sha256,
        "build_concrete_proof_witnesses": state.build_concrete_proof_witnesses,
        "visible_io_relay_gain_hex": state.visible_io_relay_gain_hex,
        "public_input_count": state.public_input_count,
        "canonical_input_count": state.canonical_input_count,
        "protected_input_count": state.protected_input_count,
    }


class FrozenJointPromptFactorFrontier:
    """Frozen prompt-gated packed frontier with one desired-output operation."""

    __slots__ = (
        "_bridge",
        "_construction_receipt",
        "_frontier",
        "_intervention",
        "_prompt_admission",
        "_root_admission",
        "_state",
    )

    def __init__(
        self,
        receipt: _JointBuildReceipt,
    ) -> None:
        if not _authenticate_joint_build_receipt(receipt):
            raise TypeError("joint frontier requires a private executed-build certificate")
        bridge = receipt.bridge
        prompt_admission = receipt.prompt_admission
        root_admission = receipt.root_admission
        frontier = receipt.frontier
        intervention = receipt.intervention
        status = receipt.status
        partial = JointFrontierState(
            status=status,
            construction_certificate_sha256=receipt.sha256,
            prompt_admission_sha256=prompt_admission.sha256,
            projected_relay_sha256=receipt.projected_relay_sha256,
            root_admission_sha256=(None if root_admission is None else root_admission.sha256),
            packed_frontier_sha256=None if frontier is None else frontier.sha256,
            operator_checkpoint_sha256=(
                None if frontier is None else frontier.state.operator_checkpoint_sha256
            ),
            v5_source_commit=V5_SOURCE_COMMIT,
            full_typed_source_commit=FULL_TYPED_SOURCE_COMMIT,
            full_typed_source_sha256=FULL_TYPED_SOURCE_SHA256,
            build_concrete_proof_witnesses=(
                0 if frontier is None else frontier.work.concrete_proof_witnesses
            ),
            visible_io_relay_gain_hex=intervention.visible_io_relay_gain.hex(),
            public_input_count=0,
            canonical_input_count=0,
            protected_input_count=0,
            sha256="",
        )
        self._state = JointFrontierState(
            **_state_payload(partial),
            sha256=_digest("joint-frontier", _state_payload(partial)),
        )
        self._bridge = bridge
        self._construction_receipt = receipt
        self._prompt_admission = prompt_admission
        self._root_admission = root_admission
        self._frontier = frontier
        self._intervention = intervention

    @property
    def state(self) -> JointFrontierState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def prompt_admission(self) -> PromptAdmissionAudit:
        return self._prompt_admission

    @property
    def root_admission(self) -> RootAdmissionAudit | None:
        return self._root_admission

    def _authenticate(self) -> full_typed.PackedFullTypedFrontier | None:
        _bridge_record(self._bridge)
        _authenticate_full_typed_source()
        receipt = self._construction_receipt
        if (
            not _authenticate_joint_build_receipt(receipt)
            or receipt.bridge is not self._bridge
            or receipt.prompt_admission is not self._prompt_admission
            or receipt.root_admission is not self._root_admission
            or receipt.frontier is not self._frontier
            or receipt.intervention != self._intervention
            or receipt.status != self._state.status
            or receipt.sha256 != self._state.construction_certificate_sha256
            or receipt.projected_relay_sha256 != self._state.projected_relay_sha256
            or self._prompt_admission.sha256
            != _digest(
                "prompt-admission",
                self._prompt_admission.record(include_sha256=False),
            )
            or self._state.sha256 != _digest("joint-frontier", _state_payload(self._state))
            or self._state.prompt_admission_sha256 != self._prompt_admission.sha256
            or self._state.v5_source_commit != V5_SOURCE_COMMIT
            or self._state.full_typed_source_commit != FULL_TYPED_SOURCE_COMMIT
            or self._state.full_typed_source_sha256 != FULL_TYPED_SOURCE_SHA256
            or self._state.build_concrete_proof_witnesses != 0
        ):
            raise RuntimeError("joint prompt/frontier authentication failed")
        if self._state.status != FRONTIER_READY:
            if self._frontier is not None or self._root_admission is not None:
                raise RuntimeError("joint STOP exposed an engine or root admission")
            return None
        frontier = self._frontier
        admission = self._root_admission
        if (
            frontier is None
            or admission is None
            or frontier.sha256 != self._state.packed_frontier_sha256
            or frontier.state.operator_checkpoint_sha256 != self._state.operator_checkpoint_sha256
            or admission.sha256 != self._state.root_admission_sha256
            or admission.sha256 != _digest("root-admission", admission.record(include_sha256=False))
            or not any(admission.root_admission_spikes)
        ):
            raise RuntimeError("joint packed/root admission authentication failed")
        return frontier

    def _selection(
        self,
        status: str,
        *,
        stop_reason: str | None,
        race: JointPhysicalRaceAudit | None = None,
        packed_selection: full_typed.PackedFullTypedSelection | None = None,
    ) -> JointSelection:
        witnesses = (
            () if packed_selection is None or status != SELECTED else packed_selection.witnesses
        )
        selected_eclass = (
            None
            if packed_selection is None or status != SELECTED
            else packed_selection.selected_eclass_id
        )
        before = None if packed_selection is None else packed_selection.checkpoint_before_sha256
        after = None if packed_selection is None else packed_selection.checkpoint_after_sha256
        packed_race = None if packed_selection is None else packed_selection.race
        joint_winner = None if race is None else race.winner_key
        packed_winner = None if packed_race is None else packed_race.winner_key
        packed_selected = None if packed_selection is None else packed_selection.selected_eclass_id
        packed_proof_witnesses = (
            None if packed_selection is None else packed_selection.work.concrete_proof_witnesses
        )
        winner_equal = (
            None
            if packed_selection is None
            else bool(
                race is not None
                and packed_race is not None
                and joint_winner is not None
                and joint_winner == packed_winner == packed_selected
            )
        )
        payload = {
            "status": status,
            "stop_reason": stop_reason,
            "frontier_sha256": self._state.sha256,
            "prompt_admission_sha256": self._prompt_admission.sha256,
            "root_admission_sha256": (
                None if self._root_admission is None else self._root_admission.sha256
            ),
            "joint_race_sha256": None if race is None else race.sha256,
            "joint_winner_key": joint_winner,
            "packed_selection_sha256": (
                None if packed_selection is None else packed_selection.sha256
            ),
            "packed_race_sha256": None if packed_race is None else packed_race.sha256,
            "packed_winner_key": packed_winner,
            "packed_selected_eclass_id": packed_selected,
            "packed_concrete_proof_witnesses": packed_proof_witnesses,
            "winner_identity_equal": winner_equal,
            "selected_eclass_id": selected_eclass,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
        }
        return JointSelection(
            status=status,
            stop_reason=stop_reason,
            frontier_sha256=self._state.sha256,
            prompt_admission_sha256=self._prompt_admission.sha256,
            root_admission_sha256=(
                None if self._root_admission is None else self._root_admission.sha256
            ),
            joint_race=race,
            joint_winner_key=joint_winner,
            packed_selection_sha256=(None if packed_selection is None else packed_selection.sha256),
            packed_race_sha256=None if packed_race is None else packed_race.sha256,
            packed_winner_key=packed_winner,
            packed_selected_eclass_id=packed_selected,
            packed_concrete_proof_witnesses=packed_proof_witnesses,
            winner_identity_equal=winner_equal,
            selected_eclass_id=selected_eclass,
            witnesses=witnesses,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
            sha256=_digest("joint-selection", payload),
        )

    def select(self, desired_outputs: Sequence[int]) -> JointSelection:
        frontier = self._authenticate()
        if frontier is None:
            return self._selection(
                self._state.status,
                stop_reason="prompt factor admission stopped before engine/desired access",
            )
        if isinstance(desired_outputs, (str, bytes)):
            raise TypeError("joint desired outputs must be exact integers")
        desired: list[int] = []
        for value in desired_outputs:
            if type(value) is not int:
                raise ValueError("joint desired outputs must be exact integers")
            desired.append(value)
        desired_tuple = tuple(desired)
        admission = cast(RootAdmissionAudit, self._root_admission)
        race = _joint_physical_race(
            frontier,
            admission,
            desired_tuple,
            visible_io_relay_gain=self._intervention.visible_io_relay_gain,
        )
        if race.status != SELECTED:
            return self._selection(
                race.status,
                stop_reason="no unique prompt-admitted visible-I/O root",
                race=race,
            )
        packed_selection = frontier.select(desired_tuple)
        if (
            packed_selection.status != full_typed.SELECTED
            or packed_selection.race is None
            or packed_selection.race.winner_key != race.winner_key
            or packed_selection.selected_eclass_id != race.winner_key
            or packed_selection.exact_proof_count != 1
            or len(packed_selection.witnesses) != 1
        ):
            return self._selection(
                STOP_INTEGRITY,
                stop_reason="packed selected proof disagreed with the joint physical winner",
                race=race,
                packed_selection=packed_selection,
            )
        family_index = {family.family_id: family for family in frontier.families}
        witness_lineage = _generic_family_lineage(
            packed_selection.witnesses[0].family_id,
            family_index,
            {},
            set(),
        )
        if witness_lineage != self._prompt_admission.selected_lineage:
            raise RuntimeError("selected proof escaped its physical prompt lineage")
        return self._selection(
            SELECTED,
            stop_reason=None,
            race=race,
            packed_selection=packed_selection,
        )


def _build_intervened_joint_frontier(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
    prompt_batch: factor_v5.AuthenticatedSpanBatch,
    *,
    signature: TypedSignature,
    case_arguments: Iterable[tuple[object, ...]],
    literals: Iterable[int],
    max_ast_cost: int,
    intervention: _Intervention,
) -> FrozenJointPromptFactorFrontier:
    prompt, projected = _resolve_prompt_admission(bridge, prompt_batch, intervention)
    if prompt.status != FRONTIER_READY:
        return FrozenJointPromptFactorFrontier(
            _issue_joint_build_receipt(
                bridge=bridge,
                prompt_admission=prompt,
                root_admission=None,
                frontier=None,
                intervention=intervention,
                status=STOP_FACTOR_ADMISSION,
                projected_relays=projected,
            )
        )
    _authenticate_full_typed_source()
    frontier = full_typed.PackedFullTypedFrontier.build(
        signature=signature,
        case_arguments=case_arguments,
        literals=literals,
        max_ast_cost=max_ast_cost,
    )
    if frontier.status != full_typed.FRONTIER_READY:
        return FrozenJointPromptFactorFrontier(
            _issue_joint_build_receipt(
                bridge=bridge,
                prompt_admission=prompt,
                root_admission=None,
                frontier=None,
                intervention=intervention,
                status=STOP_ENGINE,
                projected_relays=projected,
            )
        )
    admission = _root_admission(frontier, prompt, projected)
    if not any(admission.root_admission_spikes):
        return FrozenJointPromptFactorFrontier(
            _issue_joint_build_receipt(
                bridge=bridge,
                prompt_admission=prompt,
                root_admission=None,
                frontier=None,
                intervention=intervention,
                status=STOP_FACTOR_ADMISSION,
                projected_relays=projected,
            )
        )
    return FrozenJointPromptFactorFrontier(
        _issue_joint_build_receipt(
            bridge=bridge,
            prompt_admission=prompt,
            root_admission=admission,
            frontier=frontier,
            intervention=intervention,
            status=FRONTIER_READY,
            projected_relays=projected,
        )
    )


def build_frozen_joint_frontier(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
    prompt_batch: factor_v5.AuthenticatedSpanBatch,
    *,
    signature: TypedSignature,
    case_arguments: Iterable[tuple[object, ...]],
    literals: Iterable[int],
    max_ast_cost: int = 6,
) -> FrozenJointPromptFactorFrontier:
    """Resolve prompt factors and build a desired-free packed typed frontier."""

    return _build_intervened_joint_frontier(
        bridge,
        prompt_batch,
        signature=signature,
        case_arguments=case_arguments,
        literals=literals,
        max_ast_cost=max_ast_cost,
        intervention=_TREATMENT,
    )


_SELF_POSED_CASE_VALUES = (
    (-2, 1, 4),
    (2, 3),
    (-1, 2, 5, 7),
    (1,),
)
SELF_POSED_LITERALS = (0, 2, 4)
_SUM_DESIRED = (3, 5, 13, 1)
_COUNT_ITEM_LT_DESIRED = (2, 0, 1, 1)
_COUNT_LITERAL_LT_DESIRED = (1, 1, 2, 0)
_SUM_SOURCE = "def solve(values):\n    return sum(values)\n"
_COUNT_ITEM_LT_SOURCE = (
    "def solve(values):\n    return sum((1 for item_0 in values if item_0 < 2))\n"
)
_COUNT_LITERAL_LT_SOURCE = (
    "def solve(values):\n    return sum((1 for item_0 in values if 2 < item_0))\n"
)


def _self_posed_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("values", ProgramType.List(integer)),),
        integer,
    )


def _self_posed_cases() -> tuple[tuple[list[int]], ...]:
    return tuple((list(values),) for values in _SELF_POSED_CASE_VALUES)


def _self_posed_batch(
    case_index: int, *, single_left_span: bool = False
) -> factor_v5.AuthenticatedSpanBatch:
    case = factor_protocol.PROTOCOL.composition_cases[case_index]
    spans = (case.spans[0],) if single_left_span else case.spans
    return factor_v5.AuthenticatedSpanBatch(
        spans=spans,
        protocol_sha256=factor_protocol.PROTOCOL.protocol_sha256,
    )


def _matrix_row(
    row_id: str,
    intervention: _Intervention,
    frontier: FrozenJointPromptFactorFrontier,
    desired: tuple[int, ...],
) -> MatrixRowResult:
    selection = frontier.select(desired)
    source = selection.witnesses[0].source if selection.witnesses else None
    return MatrixRowResult(
        row_id=row_id,
        intervention=intervention.name,
        build_status=frontier.status,
        selection_status=selection.status,
        source=source,
        prompt_admission_sha256=frontier.prompt_admission.sha256,
        selection_sha256=selection.sha256,
    )


def _run_self_posed_matrix(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
) -> SelfPosedJointMatrixResult:
    """Execute the exact seven-row, zero-public-input mechanism matrix."""

    signature = _self_posed_signature()
    sum_batch = _self_posed_batch(8, single_left_span=True)
    count_batch = _self_posed_batch(5)
    reversed_batch = _self_posed_batch(3)

    def build(
        batch: factor_v5.AuthenticatedSpanBatch,
        intervention: _Intervention,
    ) -> FrozenJointPromptFactorFrontier:
        return _build_intervened_joint_frontier(
            bridge,
            batch,
            signature=signature,
            case_arguments=_self_posed_cases(),
            literals=SELF_POSED_LITERALS,
            max_ast_cost=6,
            intervention=intervention,
        )

    rows = (
        _matrix_row("treatment-sum", _TREATMENT, build(sum_batch, _TREATMENT), _SUM_DESIRED),
        _matrix_row(
            "treatment-count-item-lt",
            _TREATMENT,
            build(count_batch, _TREATMENT),
            _COUNT_ITEM_LT_DESIRED,
        ),
        _matrix_row(
            "treatment-count-literal-lt",
            _TREATMENT,
            build(count_batch, _TREATMENT),
            _COUNT_LITERAL_LT_DESIRED,
        ),
        _matrix_row(
            "control-factor-off",
            _FACTOR_OFF,
            build(count_batch, _FACTOR_OFF),
            _COUNT_ITEM_LT_DESIRED,
        ),
        _matrix_row(
            "control-deranged-relays",
            _DERANGED,
            build(count_batch, _DERANGED),
            _COUNT_ITEM_LT_DESIRED,
        ),
        _matrix_row(
            "control-event-order",
            _TREATMENT,
            build(reversed_batch, _TREATMENT),
            _COUNT_ITEM_LT_DESIRED,
        ),
        _matrix_row(
            "control-visible-io-match-off",
            _VISIBLE_IO_OFF,
            build(count_batch, _VISIBLE_IO_OFF),
            _COUNT_ITEM_LT_DESIRED,
        ),
    )
    expected = (
        (SELECTED, _SUM_SOURCE),
        (SELECTED, _COUNT_ITEM_LT_SOURCE),
        (SELECTED, _COUNT_LITERAL_LT_SOURCE),
        (STOP_FACTOR_ADMISSION, None),
        (STOP_FACTOR_ADMISSION, None),
        (STOP_FACTOR_ADMISSION, None),
        (STOP_NO_EXACT, None),
    )
    treatment_passes = sum(
        row.selection_status == status and row.source == source
        for row, (status, source) in zip(rows[:3], expected[:3], strict=True)
    )
    control_stops = sum(
        row.selection_status == status and row.source is None
        for row, (status, _source) in zip(rows[3:], expected[3:], strict=True)
    )
    verdict = PASS_MATRIX if (treatment_passes, control_stops) == (3, 4) else STOP_MATRIX
    payload = {
        "verdict": verdict,
        "rows": tuple(asdict(row) for row in rows),
        "treatment_pass_count": treatment_passes,
        "control_stop_count": control_stops,
        "public_input_count": 0,
        "canonical_input_count": 0,
        "protected_input_count": 0,
    }
    return SelfPosedJointMatrixResult(
        verdict=verdict,
        rows=rows,
        treatment_pass_count=treatment_passes,
        control_stop_count=control_stops,
        public_input_count=0,
        canonical_input_count=0,
        protected_input_count=0,
        sha256=_digest("self-posed-matrix", payload),
    )


SELF_POSED_MATRIX_PROTOCOL_SHA256 = _digest(
    "self-posed-matrix-protocol",
    {
        "row_order": (
            "treatment-sum",
            "treatment-count-item-lt",
            "treatment-count-literal-lt",
            "control-factor-off",
            "control-deranged-relays",
            "control-event-order",
            "control-visible-io-match-off",
        ),
        "case_values": _SELF_POSED_CASE_VALUES,
        "literals": SELF_POSED_LITERALS,
        "desired": (_SUM_DESIRED, _COUNT_ITEM_LT_DESIRED, _COUNT_LITERAL_LT_DESIRED),
        "supported_generic_lineages": _SUPPORTED_GENERIC_LINEAGES,
        "v5_source_commit": V5_SOURCE_COMMIT,
        "v5_protocol_sha256": V5_PROTOCOL_SHA256,
        "full_typed_source_commit": FULL_TYPED_SOURCE_COMMIT,
        "full_typed_source_sha256": FULL_TYPED_SOURCE_SHA256,
        "public_input_count": 0,
        "canonical_input_count": 0,
        "protected_input_count": 0,
    },
)


__all__ = [
    "FRONTIER_READY",
    "FULL_TYPED_SOURCE_COMMIT",
    "FULL_TYPED_SOURCE_SHA256",
    "PASS_MATRIX",
    "SELECTED",
    "SELF_POSED_LITERALS",
    "SELF_POSED_MATRIX_PROTOCOL_SHA256",
    "STOP_AMBIGUOUS",
    "STOP_ENGINE",
    "STOP_FACTOR_ADMISSION",
    "STOP_INTEGRITY",
    "STOP_MATRIX",
    "STOP_NO_EXACT",
    "V5_SOURCE_COMMIT",
    "FrozenJointPromptFactorFrontier",
    "JointFrontierState",
    "JointPhysicalRaceAudit",
    "JointSelection",
    "MatrixRowResult",
    "PromptAdmissionAudit",
    "RootAdmissionAudit",
    "SelfPosedJointMatrixResult",
    "build_frozen_joint_frontier",
]
