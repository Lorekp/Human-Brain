"""Learn an anonymous IDENTITY-vs-ABSOLUTE map/fold consequence.

This self-posed composition adds one two-cell transform population after the
banked occurrence-to-reducer bridge.  Three A/B/A-prime lessons expose only
neutral occurrence bundles and authenticated physical output consequences.
A physical exact semantic race supplies the teaching consequence; a uint8
2x2 E/H relay then maps the winning occurrence schema cell to one of the two
new transform cells.

Frozen build receives two neutral bundles, a typed signature, and visible
cases.  It reads only the exact occurrence schema-cell identities, requires
the banked physical SUM reducer receipt for the fold bundle, races the learned
transform population, and constructs exactly one desired-free singleton: raw
sum or the banked absolute-map-sum plan slice.  It receives no probe output,
desired program, operator/family/action name, target action, or schema label.

Bundle framing, carrier allocation, the fixed two-hypothesis intervention
roster, typed case framing, structural plan certificates, and compilation
remain explicit host scaffolds.  This is a narrow reusable map/fold relation,
not prompt parsing, arbitrary algorithms, or coding capability.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple, cast

import numpy as np
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_occurrence_executable_reducer_bridge_v1 as bridge
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_OCCURRENCE_ABSOLUTE_MAP_FOLD_V1"
READY = f"{SCHEMA}_READY"
STOP_TRANSFORM_OCCURRENCE = f"STOP_{SCHEMA}_TRANSFORM_OCCURRENCE"
STOP_FOLD_OCCURRENCE = f"STOP_{SCHEMA}_FOLD_OCCURRENCE"
STOP_FOLD_RELAY = f"STOP_{SCHEMA}_FOLD_RELAY"
STOP_REDUCER_ACTION = f"STOP_{SCHEMA}_REDUCER_ACTION"
STOP_REDUCER_NOT_SUM = f"STOP_{SCHEMA}_REDUCER_NOT_SUM"
STOP_TRANSFORM_ZERO = f"STOP_{SCHEMA}_TRANSFORM_ZERO"
STOP_TRANSFORM_AMBIGUOUS = f"STOP_{SCHEMA}_TRANSFORM_AMBIGUOUS"
STOP_CONSEQUENCE_ZERO = f"STOP_{SCHEMA}_CONSEQUENCE_ZERO"
STOP_CONSEQUENCE_AMBIGUOUS = f"STOP_{SCHEMA}_CONSEQUENCE_AMBIGUOUS"
STOP_CAUSAL_PAIR_CONFLICT = f"STOP_{SCHEMA}_CAUSAL_PAIR_CONFLICT"
STOP_PLAN_SLICE = f"STOP_{SCHEMA}_PLAN_SLICE"
STOP_WITNESS_NOT_SINGLETON = f"STOP_{SCHEMA}_WITNESS_NOT_SINGLETON"
STOP_WORK_BOUND = f"STOP_{SCHEMA}_WORK_BOUND"

CLAIM_BOUNDARY = (
    "one self-posed consequence-trained occurrence-schema transform relay "
    "selects raw sum versus absolute-map-sum and materializes one exact fresh "
    "witness; neutral bundle framing, carrier allocation, the fixed two-"
    "hypothesis roster, typed cases, structural certificates, and compilation "
    "remain explicit, so prompt parsing, arbitrary map/fold induction, public "
    "transfer, and coding capability remain unproven"
)

BRIDGE_SOURCE_COMMIT = "064dd8d"
ABSOLUTE_SOURCE_COMMIT = "ad6eb57"

SCHEMA_CAPACITY = occurrence.SCHEMA_CAPACITY
TRANSFORM_ACTION_CELL_COUNT = 2
TRANSFORM_ACTION_CELL_OFFSET = bridge.TOTAL_CELL_COUNT
IDENTITY_TRANSFORM_CELL = TRANSFORM_ACTION_CELL_OFFSET
ABSOLUTE_TRANSFORM_CELL = TRANSFORM_ACTION_CELL_OFFSET + 1
NEW_CELL_COUNT = TRANSFORM_ACTION_CELL_COUNT
TOTAL_CELL_COUNT = bridge.TOTAL_CELL_COUNT + NEW_CELL_COUNT

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
LEARNED_STATE_SHAPE = (SCHEMA_CAPACITY, TRANSFORM_ACTION_CELL_COUNT)
NEW_PERSISTENT_STATE_BYTES = (
    2 * int(np.prod(LEARNED_STATE_SHAPE)) * TICK_DTYPE.itemsize
)
COMPOSED_PERSISTENT_STATE_BYTES = (
    bridge.COMPOSED_PERSISTENT_STATE_BYTES + NEW_PERSISTENT_STATE_BYTES
)

TRAIN_LESSON_COUNT = 3
TRANSFORM_ACTION_RACE_STEPS = append.ACTION_RACE_STEPS
TRANSFORM_ACTION_RACE_CELL_STEPS = (
    TRANSFORM_ACTION_CELL_COUNT * TRANSFORM_ACTION_RACE_STEPS
)
CONSEQUENCE_RACE_STEPS = append.SEMANTIC_RACE_STEPS
CONSEQUENCE_RACE_CELL_STEPS = (
    TRANSFORM_ACTION_CELL_COUNT + 1
) * CONSEQUENCE_RACE_STEPS
BANKED_SUM_ACTION_CELL = append.REDUCER_ACTION_CELL_OFFSET + append._SUM_ACTION

DESIRED_READS_DURING_BUILD = 0
PROBE_OUTPUT_READS_DURING_BUILD = 0
OPERATOR_NAME_INPUTS_DURING_BUILD = 0
FAMILY_LABEL_INPUTS_DURING_BUILD = 0
TARGET_ACTION_INPUTS_DURING_BUILD = 0
SCHEMA_LABEL_INPUTS_DURING_BUILD = 0
DIRECT_OCCURRENCE_ACTION_READS = 0
DIRECT_APPEND_ACTION_READS = 0
SERIALIZED_SCHEMA_ACTION_READS = 0

PUBLIC_TRAIN_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

if (
    bridge.TOTAL_CELL_COUNT != 69_232
    or bridge.COMPOSED_PERSISTENT_STATE_BYTES != 768
    or SCHEMA_CAPACITY != 2
    or NEW_CELL_COUNT != 2
    or TOTAL_CELL_COUNT != 69_234
    or NEW_PERSISTENT_STATE_BYTES != 8
    or COMPOSED_PERSISTENT_STATE_BYTES != 776
    or TRANSFORM_ACTION_RACE_CELL_STEPS != 60
    or CONSEQUENCE_RACE_CELL_STEPS != 1_800
):
    raise RuntimeError("occurrence absolute map/fold composite identity changed")

_NAMESPACE = b"causal-occurrence-absolute-map-fold-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_TRAIN_WORK_NAMESPACE = _NAMESPACE + b"training-work\0"
_BUILD_WORK_NAMESPACE = _NAMESPACE + b"build-work\0"
_TRANSFORM_RACE_NAMESPACE = _NAMESPACE + b"transform-race\0"
_CERTIFICATE_NAMESPACE = _NAMESPACE + b"certificate\0"
_CONSTRUCTION_TOKEN = object()
_CERTIFICATE_AUTHORITY = object()


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    return b"\0".join(
        (
            name.encode("ascii"),
            value.dtype.str.encode("ascii"),
            _canonical_json(value.shape),
            value.tobytes(order="C"),
        )
    )


@dataclass(frozen=True, slots=True)
class CausalOccurrenceTransformLesson:
    """One neutral occurrence bundle paired with physical consequences only."""

    events: tuple[occurrence.PhysicalOccurrenceEvent, ...]
    observed_outputs: tuple[output_cells.TypedOutputEncoding, ...]


class TransformTrainingWork(NamedTuple):
    occurrence_physical_cell_steps: int
    consequence_race_count: int
    consequence_race_cell_steps: int
    packed_candidate_build_count: int
    packed_relation_completed_units: int
    packed_physical_predictions: int
    packed_prediction_cache_hits: int
    packed_physical_races: int
    occurrence_work_sha256s: tuple[str, ...]
    consequence_race_sha256s: tuple[str, ...]
    packed_work_sha256s: tuple[str, ...]
    total_physical_cell_steps: int
    sha256: str


class TransformTrainingSummary(NamedTuple):
    lesson_count: int
    occurrence_schema_cell_ids: tuple[int, ...]
    consequence_transform_cell_ids: tuple[int, ...]
    learned_cell_pairs: tuple[tuple[int, int], ...]
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    new_cell_count: int
    total_cell_count: int
    new_persistent_state_bytes: int
    composed_persistent_state_bytes: int
    work: TransformTrainingWork
    checkpoint_sha256: str


class TransformBuildWork(NamedTuple):
    fold_occurrence_physical_cell_steps: int
    transform_occurrence_physical_cell_steps: int
    fold_relay_race_cell_steps: int
    reducer_action_race_cell_steps: int
    transform_action_race_cell_steps: int
    stacked_cell_steps: int
    packed_build_count: int
    authorized_slice_count: int
    proof_materialization_count: int
    packed_relation_completed_units: int
    packed_physical_predictions: int
    packed_prediction_cache_hits: int
    packed_physical_races: int
    packed_work_sha256: str | None
    sha256: str


class _PackedMetrics(NamedTuple):
    relation_completed_units: int
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    sha256: str


class _CandidateRoster(NamedTuple):
    outputs: tuple[
        tuple[output_cells.TypedOutputEncoding, ...],
        tuple[output_cells.TypedOutputEncoding, ...],
    ]
    metrics: tuple[_PackedMetrics, _PackedMetrics]


class _TransformRaceDecision(NamedTuple):
    status: str
    transform_action_cell_id: int | None
    raw_scores: tuple[int, ...]
    spike_cells: tuple[int, ...]
    physical_cell_steps: int
    sha256: str


class _CertificateRecord(NamedTuple):
    authority: object
    checkpoint_sha256: str
    status: str
    stop_reason: str | None
    fold_occurrence_certificate_sha256: str | None
    transform_occurrence_certificate_sha256: str | None
    fold_schema_cell_id: int | None
    transform_schema_cell_id: int | None
    reducer_action_receipt_sha256: str | None
    reducer_action_cell_id: int | None
    transform_race_sha256: str | None
    transform_action_cell_id: int | None
    plan_slice_receipt_sha256: str | None
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    desired_reads_during_build: int
    probe_output_reads_during_build: int
    operator_name_inputs_during_build: int
    family_label_inputs_during_build: int
    target_action_inputs_during_build: int
    schema_label_inputs_during_build: int
    work: TransformBuildWork
    sha256: str


_ISSUED_CERTIFICATES: dict[int, OccurrenceAbsoluteMapFoldWitnessCertificate] = {}


class OccurrenceAbsoluteMapFoldWitnessCertificate:
    """Opaque exact-object authority containing zero or one exact witness."""

    __slots__ = ("_record",)

    def __init__(self, record: _CertificateRecord) -> None:
        if record.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("absolute map/fold witness requires private authority")
        self._record = record

    @property
    def status(self) -> str:
        return self._record.status

    @property
    def stop_reason(self) -> str | None:
        return self._record.stop_reason

    @property
    def fold_schema_cell_id(self) -> int | None:
        return self._record.fold_schema_cell_id

    @property
    def transform_schema_cell_id(self) -> int | None:
        return self._record.transform_schema_cell_id

    @property
    def reducer_action_cell_id(self) -> int | None:
        return self._record.reducer_action_cell_id

    @property
    def transform_action_cell_id(self) -> int | None:
        return self._record.transform_action_cell_id

    @property
    def exact_proof_count(self) -> int:
        return self._record.exact_proof_count

    @property
    def witness(self) -> breadth_v3.PackedGenericTypedWitness | None:
        return self._record.witness

    @property
    def work(self) -> TransformBuildWork:
        return self._record.work

    @property
    def sha256(self) -> str:
        return self._record.sha256


class OccurrenceAbsoluteMapFoldFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _validate_observed_outputs(
    values: Sequence[output_cells.TypedOutputEncoding],
    case_count: int,
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("observed outputs must be a physical encoding sequence")
    frozen = tuple(values)
    if len(frozen) != case_count or any(
        type(item) is not output_cells.TypedOutputEncoding for item in frozen
    ):
        raise ValueError("observed outputs must align exactly with visible cases")
    return tuple(output_cells.TypedOutputCellBank.validate(item) for item in frozen)


def _metrics(value: object) -> _PackedMetrics:
    work = value.work  # type: ignore[attr-defined]
    return _PackedMetrics(
        int(work.relation.completed_units),
        int(work.physical_predictions),
        int(work.prediction_cache_hits),
        int(work.physical_races),
        str(work.sha256),
    )


def _raw_sum_binding(
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
) -> tuple[
    plan_slice.PackedGenericTypedPromptPlanSliceV1,
    plan_slice._PlanSliceBinding,
]:
    delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
        signature,
        cases,
        absolute.FIXED_DUMMY_LITERALS,
        absolute.FIXED_MAX_COST,
        append.freeze_raw_sum_scaffold(),
    )
    receipt = delegate.plan_slice_receipt
    if (
        delegate.status != plan_slice.FRONTIER_READY
        or delegate.state.authorized_slice_count != 1
        or receipt is None
    ):
        raise OccurrenceAbsoluteMapFoldFitStop(
            STOP_PLAN_SLICE, "raw SUM did not authorize one physical plan slice"
        )
    binding = plan_slice._authenticate_plan_slice_receipt(delegate._frontier, receipt)
    return delegate, binding


def _candidate_roster(
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
) -> _CandidateRoster:
    raw_delegate, raw_binding = _raw_sum_binding(signature, cases)
    publication = absolute.PackedPhysicalAbsoluteMapFoldV1.build(signature, cases)
    absolute_receipt = publication.plan_slice_receipt
    if publication.status != absolute.READY or absolute_receipt is None:
        raise OccurrenceAbsoluteMapFoldFitStop(
            STOP_PLAN_SLICE, "absolute map/fold did not authorize one physical slice"
        )
    absolute_binding = plan_slice._authenticate_plan_slice_receipt(
        publication._frontier, absolute_receipt
    )
    outputs = (
        tuple(raw_binding.candidate.eclass.physical_outputs),
        tuple(absolute_binding.candidate.eclass.physical_outputs),
    )
    if outputs[0] == outputs[1]:
        raise OccurrenceAbsoluteMapFoldFitStop(
            STOP_CONSEQUENCE_AMBIGUOUS,
            "the two physical transform interventions have identical consequences",
        )
    return _CandidateRoster(outputs, (_metrics(raw_delegate), _metrics(publication)))


def _schema_ordinal(schema_cell_id: int) -> int:
    ordinal = schema_cell_id - occurrence.SCHEMA_CELL_OFFSET
    if not 0 <= ordinal < SCHEMA_CAPACITY:
        raise OccurrenceAbsoluteMapFoldFitStop(
            STOP_TRANSFORM_OCCURRENCE,
            "occurrence certificate did not expose one local schema cell",
        )
    return ordinal


def _learn_transform(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    source: int,
    action: int,
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for candidate in range(TRANSFORM_ACTION_CELL_COUNT):
        state = excitatory if candidate == action else inhibitory
        before = int(state[source, candidate])
        state[source, candidate] = min(TICK_CAP, before + 1)
        delta = int(state[source, candidate]) - before
        if candidate == action:
            e_ticks += delta
        else:
            h_ticks += delta
    return e_ticks, h_ticks


def _race_transform(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_cell_id: int,
    *,
    cell_gate: bool,
) -> _TransformRaceDecision:
    source = _schema_ordinal(schema_cell_id)
    raw_scores = tuple(
        int(excitatory[source, action]) - int(inhibitory[source, action])
        for action in range(TRANSFORM_ACTION_CELL_COUNT)
    )
    currents = np.asarray(
        [
            max(0, score) * append.OUTPUT_CURRENT_PA if cell_gate else 0.0
            for score in raw_scores
        ],
        dtype=np.float64,
    )
    spike_cells = tuple(
        int(index)
        for index in np.flatnonzero(append._ACTION_RACE.accumulated_spike_mask(currents))
    )
    if not spike_cells:
        status = STOP_TRANSFORM_ZERO
        action_cell = None
    elif len(spike_cells) != 1:
        status = STOP_TRANSFORM_AMBIGUOUS
        action_cell = None
    else:
        status = READY
        action_cell = TRANSFORM_ACTION_CELL_OFFSET + spike_cells[0]
    payload = {
        "schema_cell_id": schema_cell_id,
        "raw_scores": raw_scores,
        "cell_gate": cell_gate,
        "spike_cells": spike_cells,
        "status": status,
        "transform_action_cell_id": action_cell,
        "physical_cell_steps": TRANSFORM_ACTION_RACE_CELL_STEPS,
    }
    return _TransformRaceDecision(
        status,
        action_cell,
        raw_scores,
        spike_cells,
        TRANSFORM_ACTION_RACE_CELL_STEPS,
        _digest(_TRANSFORM_RACE_NAMESPACE, payload),
    )


def _checkpoint_sha256(
    bridge_checkpoint_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + bridge_checkpoint_sha256.encode("ascii")
        + b"\0"
        + _array_frame("transform-excitatory", excitatory)
        + _array_frame("transform-inhibitory", inhibitory)
        + _canonical_json(
            {
                "transform_action_cell_offset": TRANSFORM_ACTION_CELL_OFFSET,
                "total_cell_count": TOTAL_CELL_COUNT,
            }
        )
    ).hexdigest()


def _training_work_payload(value: TransformTrainingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def fit_causal_occurrence_absolute_map_fold_v1(
    bridge_checkpoint: bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1,
    lessons: tuple[CausalOccurrenceTransformLesson, ...],
    *,
    signature: TypedSignature,
    cases: Iterable[tuple[object, ...]],
    outcome_gate: bool = True,
) -> FrozenCausalOccurrenceAbsoluteMapFoldV1:
    """Fit one A/B/A-prime transform relay from physical consequences."""

    if (
        type(bridge_checkpoint)
        is not bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1
        or type(lessons) is not tuple
        or len(lessons) != TRAIN_LESSON_COUNT
        or any(type(item) is not CausalOccurrenceTransformLesson for item in lessons)
        or type(outcome_gate) is not bool
    ):
        raise TypeError("transform fit requires an exact bridge and three lessons")
    if not isinstance(signature, TypedSignature):
        raise TypeError("transform fit requires one typed signature")
    frozen_cases = tuple(cases)
    if not frozen_cases:
        raise ValueError("transform fit requires visible physical cases")
    bridge_checkpoint.authenticate()
    before = bridge_checkpoint.checkpoint_sha256
    roster = _candidate_roster(signature, frozen_cases)

    excitatory = np.zeros(LEARNED_STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    schema_cells: list[int] = []
    action_cells: list[int] = []
    occurrence_roots: list[str] = []
    consequence_roots: list[str] = []
    occurrence_steps = 0
    e_ticks = 0
    h_ticks = 0

    keys = tuple(
        f"transform-cell-{TRANSFORM_ACTION_CELL_OFFSET + index}"
        for index in range(TRANSFORM_ACTION_CELL_COUNT)
    )
    for lesson in lessons:
        certificate = bridge_checkpoint._occurrence_checkpoint.recognize(lesson.events)
        occurrence._authenticate_certificate(certificate)
        source_cell = certificate.schema_cell_id
        if source_cell is None:
            raise OccurrenceAbsoluteMapFoldFitStop(
                STOP_TRANSFORM_OCCURRENCE,
                certificate.stop_reason or certificate.status,
            )
        source = _schema_ordinal(source_cell)
        observed = _validate_observed_outputs(
            lesson.observed_outputs, len(frozen_cases)
        )
        consequence = semantic_race.select_exact_semantic_vector(
            keys, roster.outputs, observed
        )
        if consequence.status == semantic_race.STOP_NO_EXACT:
            raise OccurrenceAbsoluteMapFoldFitStop(
                STOP_CONSEQUENCE_ZERO,
                "neither physical transform intervention matched the consequence",
            )
        if (
            consequence.status != semantic_race.SELECTED
            or consequence.winner_key is None
        ):
            raise OccurrenceAbsoluteMapFoldFitStop(
                STOP_CONSEQUENCE_AMBIGUOUS,
                "multiple physical transform interventions matched the consequence",
            )
        action = keys.index(consequence.winner_key)
        if outcome_gate:
            delta_e, delta_h = _learn_transform(
                excitatory, inhibitory, source, action
            )
            e_ticks += delta_e
            h_ticks += delta_h
        schema_cells.append(source_cell)
        action_cells.append(TRANSFORM_ACTION_CELL_OFFSET + action)
        occurrence_roots.append(certificate.work.sha256)
        consequence_roots.append(consequence.sha256)
        occurrence_steps += certificate.work.total_physical_cell_steps

    if (
        tuple(schema_cells)
        != (schema_cells[0], schema_cells[1], schema_cells[0])
        or schema_cells[0] == schema_cells[1]
        or tuple(action_cells)
        != (IDENTITY_TRANSFORM_CELL, ABSOLUTE_TRANSFORM_CELL, IDENTITY_TRANSFORM_CELL)
    ):
        raise OccurrenceAbsoluteMapFoldFitStop(
            STOP_CAUSAL_PAIR_CONFLICT,
            "A/B/A-prime did not identify two reusable consequence laws",
        )

    checkpoint_sha = _checkpoint_sha256(before, excitatory, inhibitory)
    packed_relation_units = sum(item.relation_completed_units for item in roster.metrics)
    packed_predictions = sum(item.physical_predictions for item in roster.metrics)
    packed_hits = sum(item.prediction_cache_hits for item in roster.metrics)
    packed_races = sum(item.physical_races for item in roster.metrics)
    consequence_steps = TRAIN_LESSON_COUNT * CONSEQUENCE_RACE_CELL_STEPS
    partial_work = TransformTrainingWork(
        occurrence_steps,
        TRAIN_LESSON_COUNT,
        consequence_steps,
        TRANSFORM_ACTION_CELL_COUNT,
        packed_relation_units,
        packed_predictions,
        packed_hits,
        packed_races,
        tuple(occurrence_roots),
        tuple(consequence_roots),
        tuple(item.sha256 for item in roster.metrics),
        occurrence_steps + consequence_steps,
        "",
    )
    work = partial_work._replace(
        sha256=_digest(_TRAIN_WORK_NAMESPACE, _training_work_payload(partial_work))
    )
    learned_pairs = tuple(
        sorted(
            {
                (schema_cell, action_cell)
                for schema_cell, action_cell in zip(
                    schema_cells, action_cells, strict=True
                )
            }
        )
    )
    summary = TransformTrainingSummary(
        TRAIN_LESSON_COUNT,
        tuple(schema_cells),
        tuple(action_cells),
        learned_pairs,
        e_ticks,
        h_ticks,
        int(np.max(excitatory)),
        int(np.max(inhibitory)),
        NEW_CELL_COUNT,
        TOTAL_CELL_COUNT,
        NEW_PERSISTENT_STATE_BYTES,
        COMPOSED_PERSISTENT_STATE_BYTES,
        work,
        checkpoint_sha,
    )
    return FrozenCausalOccurrenceAbsoluteMapFoldV1(
        bridge_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _build_work_payload(value: TransformBuildWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _freeze_build_work(
    *,
    fold_occurrence_steps: int,
    transform_occurrence_steps: int,
    fold_relay_steps: int,
    reducer_action_steps: int,
    transform_action_steps: int,
    packed_build_count: int = 0,
    authorized_slice_count: int = 0,
    proof_materialization_count: int = 0,
    metrics: _PackedMetrics | None = None,
) -> TransformBuildWork:
    stacked = (
        fold_occurrence_steps
        + transform_occurrence_steps
        + fold_relay_steps
        + reducer_action_steps
        + transform_action_steps
    )
    partial = TransformBuildWork(
        fold_occurrence_steps,
        transform_occurrence_steps,
        fold_relay_steps,
        reducer_action_steps,
        transform_action_steps,
        stacked,
        packed_build_count,
        authorized_slice_count,
        proof_materialization_count,
        0 if metrics is None else metrics.relation_completed_units,
        0 if metrics is None else metrics.physical_predictions,
        0 if metrics is None else metrics.prediction_cache_hits,
        0 if metrics is None else metrics.physical_races,
        None if metrics is None else metrics.sha256,
        "",
    )
    return partial._replace(
        sha256=_digest(_BUILD_WORK_NAMESPACE, _build_work_payload(partial))
    )


def _certificate_payload(record: _CertificateRecord) -> dict[str, object]:
    return {
        field: (
            record.witness.sha256
            if field == "witness" and record.witness is not None
            else getattr(record, field)
        )
        for field in record._fields
        if field not in {"authority", "work", "sha256"}
    } | {"work_sha256": record.work.sha256}


def _issue_certificate(
    checkpoint: FrozenCausalOccurrenceAbsoluteMapFoldV1,
    *,
    status: str,
    stop_reason: str | None,
    fold_certificate_sha256: str | None,
    transform_certificate_sha256: str | None,
    fold_schema_cell_id: int | None,
    transform_schema_cell_id: int | None,
    action_receipt_sha256: str | None,
    reducer_action_cell_id: int | None,
    transform_race_sha256: str | None,
    transform_action_cell_id: int | None,
    plan_slice_receipt_sha256: str | None,
    exact_proof_count: int,
    witness: breadth_v3.PackedGenericTypedWitness | None,
    work: TransformBuildWork,
) -> OccurrenceAbsoluteMapFoldWitnessCertificate:
    partial = _CertificateRecord(
        _CERTIFICATE_AUTHORITY,
        checkpoint.checkpoint_sha256,
        status,
        stop_reason,
        fold_certificate_sha256,
        transform_certificate_sha256,
        fold_schema_cell_id,
        transform_schema_cell_id,
        action_receipt_sha256,
        reducer_action_cell_id,
        transform_race_sha256,
        transform_action_cell_id,
        plan_slice_receipt_sha256,
        exact_proof_count,
        witness,
        DESIRED_READS_DURING_BUILD,
        PROBE_OUTPUT_READS_DURING_BUILD,
        OPERATOR_NAME_INPUTS_DURING_BUILD,
        FAMILY_LABEL_INPUTS_DURING_BUILD,
        TARGET_ACTION_INPUTS_DURING_BUILD,
        SCHEMA_LABEL_INPUTS_DURING_BUILD,
        work,
        "",
    )
    record = partial._replace(
        sha256=_digest(_CERTIFICATE_NAMESPACE, _certificate_payload(partial))
    )
    certificate = OccurrenceAbsoluteMapFoldWitnessCertificate(record)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


class FrozenCausalOccurrenceAbsoluteMapFoldV1:
    """Immutable desired-free occurrence-to-map/fold transform checkpoint."""

    __slots__ = (
        "_bridge_checkpoint",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_summary",
    )

    def __init__(
        self,
        bridge_checkpoint: bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: TransformTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("transform checkpoints are builder-owned")
        if (
            type(bridge_checkpoint)
            is not bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1
            or excitatory.shape != LEARNED_STATE_SHAPE
            or inhibitory.shape != LEARNED_STATE_SHAPE
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("transform checkpoint geometry changed")
        bridge_checkpoint.authenticate()
        self._bridge_checkpoint = bridge_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            bridge_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("transform training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> TransformTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._bridge_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._bridge_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
        ):
            raise RuntimeError("transform checkpoint authentication failed")

    def build(
        self,
        fold_events: Sequence[occurrence.PhysicalOccurrenceEvent],
        transform_events: Sequence[occurrence.PhysicalOccurrenceEvent],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        fold_relation_cell_gate: bool = True,
        fold_relay_cell_gate: bool = True,
        reducer_action_cell_gate: bool = True,
        transform_relation_cell_gate: bool = True,
        transform_action_cell_gate: bool = True,
    ) -> OccurrenceAbsoluteMapFoldWitnessCertificate:
        """Materialize exactly one witness from two neutral physical bundles."""

        gates = (
            fold_relation_cell_gate,
            fold_relay_cell_gate,
            reducer_action_cell_gate,
            transform_relation_cell_gate,
            transform_action_cell_gate,
        )
        if any(type(value) is not bool for value in gates):
            raise TypeError("map/fold gates must be exact booleans")
        if not isinstance(signature, TypedSignature):
            raise TypeError("map/fold build requires one typed signature")
        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise ValueError("map/fold build requires visible physical cases")
        self.authenticate()

        fold_certificate = self._bridge_checkpoint._occurrence_checkpoint.recognize(
            fold_events,
            relation_cell_gate=fold_relation_cell_gate,
        )
        occurrence._authenticate_certificate(fold_certificate)
        fold_steps = fold_certificate.work.total_physical_cell_steps
        fold_cell = fold_certificate.schema_cell_id
        if fold_cell is None:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=0,
                fold_relay_steps=0,
                reducer_action_steps=0,
                transform_action_steps=0,
            )
            return _issue_certificate(
                self,
                status=STOP_FOLD_OCCURRENCE,
                stop_reason=fold_certificate.stop_reason or fold_certificate.status,
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=None,
                fold_schema_cell_id=None,
                transform_schema_cell_id=None,
                action_receipt_sha256=None,
                reducer_action_cell_id=None,
                transform_race_sha256=None,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        fold_relay = bridge._race_relay(
            self._bridge_checkpoint._excitatory,
            self._bridge_checkpoint._inhibitory,
            fold_cell,
            cell_gate=fold_relay_cell_gate,
        )
        if fold_relay.status != bridge.READY or fold_relay.static_schema_cell_id is None:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=0,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=0,
                transform_action_steps=0,
            )
            return _issue_certificate(
                self,
                status=STOP_FOLD_RELAY,
                stop_reason="banked occurrence-to-static relay abstained",
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=None,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=None,
                action_receipt_sha256=None,
                reducer_action_cell_id=None,
                transform_race_sha256=None,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        action_receipt = bridge._issue_action_receipt(
            self._bridge_checkpoint._content_checkpoint,
            fold_relay.static_schema_cell_id,
            cell_gate=reducer_action_cell_gate,
        )
        action_record = bridge._authenticate_action_receipt(
            action_receipt, self._bridge_checkpoint._content_checkpoint
        )
        if action_record.status != bridge.READY or action_record.reducer_action_cell_id is None:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=0,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
                transform_action_steps=0,
            )
            return _issue_certificate(
                self,
                status=STOP_REDUCER_ACTION,
                stop_reason=action_record.stop_reason or action_record.status,
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=None,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=None,
                action_receipt_sha256=action_receipt.sha256,
                reducer_action_cell_id=None,
                transform_race_sha256=None,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )
        if action_record.reducer_action_cell_id != BANKED_SUM_ACTION_CELL:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=0,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
                transform_action_steps=0,
            )
            return _issue_certificate(
                self,
                status=STOP_REDUCER_NOT_SUM,
                stop_reason="fold bundle did not issue the banked physical SUM receipt",
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=None,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=None,
                action_receipt_sha256=action_receipt.sha256,
                reducer_action_cell_id=action_record.reducer_action_cell_id,
                transform_race_sha256=None,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        transform_certificate = self._bridge_checkpoint._occurrence_checkpoint.recognize(
            transform_events,
            relation_cell_gate=transform_relation_cell_gate,
        )
        occurrence._authenticate_certificate(transform_certificate)
        transform_steps = transform_certificate.work.total_physical_cell_steps
        transform_cell = transform_certificate.schema_cell_id
        if transform_cell is None:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=transform_steps,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
                transform_action_steps=0,
            )
            return _issue_certificate(
                self,
                status=STOP_TRANSFORM_OCCURRENCE,
                stop_reason=(
                    transform_certificate.stop_reason or transform_certificate.status
                ),
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=transform_certificate.sha256,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=None,
                action_receipt_sha256=action_receipt.sha256,
                reducer_action_cell_id=action_record.reducer_action_cell_id,
                transform_race_sha256=None,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        transform_race = _race_transform(
            self._excitatory,
            self._inhibitory,
            transform_cell,
            cell_gate=transform_action_cell_gate,
        )
        if transform_race.status != READY or transform_race.transform_action_cell_id is None:
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=transform_steps,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
                transform_action_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
            )
            return _issue_certificate(
                self,
                status=transform_race.status,
                stop_reason="learned transform population abstained",
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=transform_certificate.sha256,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=transform_cell,
                action_receipt_sha256=action_receipt.sha256,
                reducer_action_cell_id=action_record.reducer_action_cell_id,
                transform_race_sha256=transform_race.sha256,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        try:
            if transform_race.transform_action_cell_id == IDENTITY_TRANSFORM_CELL:
                delegate, binding = _raw_sum_binding(signature, frozen_cases)
                metrics = _metrics(delegate)
                plan_receipt = delegate.plan_slice_receipt
                if plan_receipt is None:
                    raise OccurrenceAbsoluteMapFoldFitStop(
                        STOP_PLAN_SLICE, "raw SUM plan-slice receipt disappeared"
                    )
                try:
                    forest = plan_slice._materialize_authorized_forest(binding)
                except packed.PackedRelationWorkLimitExceeded as error:
                    raise OccurrenceAbsoluteMapFoldFitStop(
                        STOP_WORK_BOUND,
                        "raw SUM singleton materialization exceeded packed work",
                    ) from error
                proof_count = forest.exact_logical_recipe_count
                witnesses = forest.first_two_witnesses
                if proof_count != 1 or len(witnesses) != 1:
                    raise OccurrenceAbsoluteMapFoldFitStop(
                        STOP_WITNESS_NOT_SINGLETON,
                        "raw SUM plan did not materialize exactly one proof",
                    )
                route_witness = binding.route._render_witness(
                    witnesses[0], binding.component
                )
                witness = cast(
                    breadth_v3.PackedGenericTypedWitness,
                    delegate._frontier._compile_witness(route_witness),
                )
            elif transform_race.transform_action_cell_id == ABSOLUTE_TRANSFORM_CELL:
                publication = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
                    signature, frozen_cases
                )
                metrics = _metrics(publication)
                plan_receipt = publication.plan_slice_receipt
                if publication.status != absolute.READY or plan_receipt is None:
                    raise OccurrenceAbsoluteMapFoldFitStop(
                        STOP_PLAN_SLICE,
                        publication.state.stop_reason or publication.status,
                    )
                witness = absolute.require_absolute_map_fold_witness(publication)
                proof_count = publication.materialize().exact_proof_count
            else:
                raise RuntimeError("transform winner left the fixed two-cell range")
        except OccurrenceAbsoluteMapFoldFitStop as error:
            metrics_value = locals().get("metrics")
            metrics = metrics_value if isinstance(metrics_value, _PackedMetrics) else None
            plan_value = locals().get("plan_receipt")
            receipt_sha = None if plan_value is None else plan_value.sha256
            work = _freeze_build_work(
                fold_occurrence_steps=fold_steps,
                transform_occurrence_steps=transform_steps,
                fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
                reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
                transform_action_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
                packed_build_count=1,
                authorized_slice_count=int(receipt_sha is not None),
                proof_materialization_count=0,
                metrics=metrics,
            )
            return _issue_certificate(
                self,
                status=error.status,
                stop_reason=error.reason,
                fold_certificate_sha256=fold_certificate.sha256,
                transform_certificate_sha256=transform_certificate.sha256,
                fold_schema_cell_id=fold_cell,
                transform_schema_cell_id=transform_cell,
                action_receipt_sha256=action_receipt.sha256,
                reducer_action_cell_id=action_record.reducer_action_cell_id,
                transform_race_sha256=transform_race.sha256,
                transform_action_cell_id=transform_race.transform_action_cell_id,
                plan_slice_receipt_sha256=receipt_sha,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        work = _freeze_build_work(
            fold_occurrence_steps=fold_steps,
            transform_occurrence_steps=transform_steps,
            fold_relay_steps=bridge.RELAY_RACE_CELL_STEPS,
            reducer_action_steps=bridge.ACTION_RACE_CELL_STEPS,
            transform_action_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
            packed_build_count=1,
            authorized_slice_count=1,
            proof_materialization_count=1,
            metrics=metrics,
        )
        self.authenticate()
        return _issue_certificate(
            self,
            status=READY,
            stop_reason=None,
            fold_certificate_sha256=fold_certificate.sha256,
            transform_certificate_sha256=transform_certificate.sha256,
            fold_schema_cell_id=fold_cell,
            transform_schema_cell_id=transform_cell,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=action_record.reducer_action_cell_id,
            transform_race_sha256=transform_race.sha256,
            transform_action_cell_id=transform_race.transform_action_cell_id,
            plan_slice_receipt_sha256=plan_receipt.sha256,
            exact_proof_count=proof_count,
            witness=witness,
            work=work,
        )


def _authenticate_certificate(
    value: OccurrenceAbsoluteMapFoldWitnessCertificate,
) -> _CertificateRecord:
    if (
        type(value) is not OccurrenceAbsoluteMapFoldWitnessCertificate
        or _ISSUED_CERTIFICATES.get(id(value)) is not value
    ):
        raise TypeError("exact issued occurrence map/fold certificate required")
    record = value._record
    if (
        record.authority is not _CERTIFICATE_AUTHORITY
        or record.work.sha256
        != _digest(_BUILD_WORK_NAMESPACE, _build_work_payload(record.work))
        or record.sha256
        != _digest(_CERTIFICATE_NAMESPACE, _certificate_payload(record))
    ):
        raise RuntimeError("occurrence map/fold certificate authentication failed")
    return record


def require_occurrence_absolute_map_fold_witness(
    value: OccurrenceAbsoluteMapFoldWitnessCertificate,
) -> breadth_v3.PackedGenericTypedWitness:
    record = _authenticate_certificate(value)
    if record.status != READY or record.exact_proof_count != 1 or record.witness is None:
        raise ValueError(record.stop_reason or record.status)
    return record.witness


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "bridge_source_commit": BRIDGE_SOURCE_COMMIT,
        "absolute_source_commit": ABSOLUTE_SOURCE_COMMIT,
        "transform_action_cell_range": (
            TRANSFORM_ACTION_CELL_OFFSET,
            TOTAL_CELL_COUNT - 1,
        ),
        "learned_state_shape": LEARNED_STATE_SHAPE,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
        "composed_persistent_state_bytes": COMPOSED_PERSISTENT_STATE_BYTES,
        "consequence_race_cell_steps": CONSEQUENCE_RACE_CELL_STEPS,
        "transform_action_race_cell_steps": TRANSFORM_ACTION_RACE_CELL_STEPS,
        "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
        "probe_output_reads_during_build": PROBE_OUTPUT_READS_DURING_BUILD,
        "operator_name_inputs_during_build": OPERATOR_NAME_INPUTS_DURING_BUILD,
        "family_label_inputs_during_build": FAMILY_LABEL_INPUTS_DURING_BUILD,
        "target_action_inputs_during_build": TARGET_ACTION_INPUTS_DURING_BUILD,
        "schema_label_inputs_during_build": SCHEMA_LABEL_INPUTS_DURING_BUILD,
        "claim_boundary": CLAIM_BOUNDARY,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", dict(CONFIG_RECORD))


__all__ = [
    "ABSOLUTE_TRANSFORM_CELL",
    "BANKED_SUM_ACTION_CELL",
    "CLAIM_BOUNDARY",
    "COMPOSED_PERSISTENT_STATE_BYTES",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CONSEQUENCE_RACE_CELL_STEPS",
    "DESIRED_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "DIRECT_APPEND_ACTION_READS",
    "DIRECT_OCCURRENCE_ACTION_READS",
    "FAMILY_LABEL_INPUTS_DURING_BUILD",
    "G0_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "IDENTITY_TRANSFORM_CELL",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "OPERATOR_NAME_INPUTS_DURING_BUILD",
    "PATH2_INPUT_COUNT",
    "PROBE_OUTPUT_READS_DURING_BUILD",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "SCHEMA",
    "SCHEMA_LABEL_INPUTS_DURING_BUILD",
    "SERIALIZED_SCHEMA_ACTION_READS",
    "STOP_CAUSAL_PAIR_CONFLICT",
    "STOP_CONSEQUENCE_AMBIGUOUS",
    "STOP_CONSEQUENCE_ZERO",
    "STOP_FOLD_OCCURRENCE",
    "STOP_FOLD_RELAY",
    "STOP_PLAN_SLICE",
    "STOP_REDUCER_ACTION",
    "STOP_REDUCER_NOT_SUM",
    "STOP_TRANSFORM_AMBIGUOUS",
    "STOP_TRANSFORM_OCCURRENCE",
    "STOP_TRANSFORM_ZERO",
    "STOP_WITNESS_NOT_SINGLETON",
    "STOP_WORK_BOUND",
    "TARGET_ACTION_INPUTS_DURING_BUILD",
    "TOTAL_CELL_COUNT",
    "TRANSFORM_ACTION_CELL_OFFSET",
    "TRANSFORM_ACTION_RACE_CELL_STEPS",
    "CausalOccurrenceTransformLesson",
    "FrozenCausalOccurrenceAbsoluteMapFoldV1",
    "OccurrenceAbsoluteMapFoldFitStop",
    "OccurrenceAbsoluteMapFoldWitnessCertificate",
    "TransformBuildWork",
    "TransformTrainingSummary",
    "TransformTrainingWork",
    "fit_causal_occurrence_absolute_map_fold_v1",
    "require_occurrence_absolute_map_fold_witness",
]
