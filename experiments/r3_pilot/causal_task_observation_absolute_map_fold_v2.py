"""Example-grounded task observation to one raw/absolute map-fold witness.

This self-posed V2 removes V1's answer-bearing ``fold_events`` and
``transform_events`` interface.  Its sole cognitive input is a source-bound
neutral prompt carrier plus exact visible input/output examples.  Existing
physical operator cells independently project the raw-SUM and ABS-map-SUM
consequences without creating a plan receipt.  A physical exact-vector race
must select one anonymous example-effect cell; a learned uint8 E/H relay then
maps that cell to V1's existing IDENTITY or ABSOLUTE action cell.  Only that
winner may authorize and materialize one packed plan receipt.

Typed signature parsing and execution remain environment adapters.  The fixed
two-candidate roster, visible examples, packed structural certificates, and
compiler remain scaffolds.  Prompt bytes are authenticated provenance, not a
phrase decoder: the narrow claim is example-grounded two-task induction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple, cast

import numpy as np
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_occurrence_absolute_map_fold_v1 as map_fold
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_TASK_OBSERVATION_ABSOLUTE_MAP_FOLD_V2"
READY = f"{SCHEMA}_READY"
STOP_SOURCE_BINDING = f"STOP_{SCHEMA}_SOURCE_BINDING"
STOP_EXAMPLE_EFFECT_NOT_SINGLETON = (
    "STOP_CAUSAL_TASK_OBSERVATION_TRANSFORM_V2_EXAMPLE_EFFECT_NOT_SINGLETON"
)
STOP_TRANSFORM_ZERO = f"STOP_{SCHEMA}_TRANSFORM_ZERO"
STOP_TRANSFORM_AMBIGUOUS = f"STOP_{SCHEMA}_TRANSFORM_AMBIGUOUS"
STOP_CAUSAL_PAIR_CONFLICT = f"STOP_{SCHEMA}_CAUSAL_PAIR_CONFLICT"
STOP_PLAN_SLICE = f"STOP_{SCHEMA}_PLAN_SLICE"
STOP_WITNESS_NOT_SINGLETON = f"STOP_{SCHEMA}_WITNESS_NOT_SINGLETON"
STOP_WORK_BOUND = f"STOP_{SCHEMA}_WORK_BOUND"

CLAIM_BOUNDARY = (
    "one self-posed source-bound visible-example observation physically selects "
    "between raw sum and absolute-map-sum, then a learned two-by-two relay "
    "authorizes one exact executable witness; prompt bytes are provenance, while "
    "the fixed two-candidate roster, typed framing, structural certificates, "
    "compiler, and execution remain scaffolds, so free-text understanding, "
    "candidate discovery, public transfer, and general coding remain unproven"
)

SOURCE_COMMIT = "7954f29"
TRAIN_LESSON_COUNT = 3
EXAMPLE_EFFECT_CELL_COUNT = 2
EXAMPLE_EFFECT_CELL_OFFSET = map_fold.TOTAL_CELL_COUNT
RAW_EXAMPLE_EFFECT_CELL = EXAMPLE_EFFECT_CELL_OFFSET + 1
ABSOLUTE_EXAMPLE_EFFECT_CELL = EXAMPLE_EFFECT_CELL_OFFSET
_CANDIDATE_EFFECT_CELLS = (
    RAW_EXAMPLE_EFFECT_CELL,
    ABSOLUTE_EXAMPLE_EFFECT_CELL,
)
NEW_CELL_COUNT = EXAMPLE_EFFECT_CELL_COUNT
TOTAL_CELL_COUNT = map_fold.TOTAL_CELL_COUNT + NEW_CELL_COUNT

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
LEARNED_STATE_SHAPE = (EXAMPLE_EFFECT_CELL_COUNT, map_fold.TRANSFORM_ACTION_CELL_COUNT)
NEW_PERSISTENT_STATE_BYTES = (
    2 * int(np.prod(LEARNED_STATE_SHAPE)) * TICK_DTYPE.itemsize
)
COMPOSED_PERSISTENT_STATE_BYTES = (
    map_fold.COMPOSED_PERSISTENT_STATE_BYTES + NEW_PERSISTENT_STATE_BYTES
)
EXAMPLE_EFFECT_RACE_CELL_STEPS = map_fold.CONSEQUENCE_RACE_CELL_STEPS
TRANSFORM_ACTION_RACE_CELL_STEPS = map_fold.TRANSFORM_ACTION_RACE_CELL_STEPS

DESIRED_PROGRAM_READS_DURING_BUILD = 0
PROBE_OUTPUT_READS_DURING_BUILD = 0
ACTION_LABEL_INPUTS_DURING_BUILD = 0
FAMILY_LABEL_INPUTS_DURING_BUILD = 0
SCHEMA_LABEL_INPUTS_DURING_BUILD = 0
TOPOLOGY_INPUTS_DURING_BUILD = 0
DIRECT_V1_ACTION_READS = 0

PUBLIC_TRAIN_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

if (
    map_fold.TOTAL_CELL_COUNT != 69_234
    or map_fold.COMPOSED_PERSISTENT_STATE_BYTES != 776
    or map_fold.IDENTITY_TRANSFORM_CELL != 69_232
    or map_fold.ABSOLUTE_TRANSFORM_CELL != 69_233
    or NEW_CELL_COUNT != 2
    or TOTAL_CELL_COUNT != 69_236
    or NEW_PERSISTENT_STATE_BYTES != 8
    or COMPOSED_PERSISTENT_STATE_BYTES != 784
    or EXAMPLE_EFFECT_RACE_CELL_STEPS != 1_800
    or TRANSFORM_ACTION_RACE_CELL_STEPS != 60
):
    raise RuntimeError("task-observation V2 composite identity changed")

_NAMESPACE = b"causal-task-observation-absolute-map-fold-v2\0"
_SOURCE_NAMESPACE = _NAMESPACE + b"source\0"
_PROJECTION_NAMESPACE = _NAMESPACE + b"physical-projection\0"
_OBSERVATION_NAMESPACE = _NAMESPACE + b"effect-observation\0"
_TRANSFORM_NAMESPACE = _NAMESPACE + b"transform-race\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_TRAIN_WORK_NAMESPACE = _NAMESPACE + b"training-work\0"
_BUILD_WORK_NAMESPACE = _NAMESPACE + b"build-work\0"
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
class NeutralTaskObservation:
    """Neutral prompt provenance plus exact visible task examples."""

    prompt_bytes: bytes
    prompt_cells: tuple[int, ...]
    visible_inputs: tuple[tuple[object, ...], ...]
    visible_outputs: tuple[output_cells.TypedOutputEncoding, ...]


class PhysicalConsequenceProjection(NamedTuple):
    candidate_outputs: tuple[
        tuple[output_cells.TypedOutputEncoding, ...],
        tuple[output_cells.TypedOutputEncoding, ...],
    ]
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    operator_checkpoint_sha256: str
    sha256: str


class ExampleEffectObservation(NamedTuple):
    status: str
    stop_reason: str | None
    task_source_sha256: str
    example_effect_cell_id: int | None
    exact_effect_cell_ids: tuple[int, ...]
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    example_effect_race_cell_steps: int
    projection_sha256: str | None
    semantic_race_sha256: str | None
    sha256: str


class TaskObservationTrainingWork(NamedTuple):
    observation_count: int
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    example_effect_race_count: int
    example_effect_race_cell_steps: int
    observation_sha256s: tuple[str, ...]
    total_physical_cell_steps: int
    sha256: str


class TaskObservationTrainingSummary(NamedTuple):
    lesson_count: int
    example_effect_cell_ids: tuple[int, ...]
    transform_action_cell_ids: tuple[int, ...]
    learned_cell_pairs: tuple[tuple[int, int], ...]
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    new_cell_count: int
    total_cell_count: int
    new_persistent_state_bytes: int
    composed_persistent_state_bytes: int
    work: TaskObservationTrainingWork
    checkpoint_sha256: str


class TaskObservationBuildWork(NamedTuple):
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    example_effect_race_cell_steps: int
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
    task_source_sha256: str
    effect_observation_sha256: str
    example_effect_cell_id: int | None
    transform_race_sha256: str | None
    transform_action_cell_id: int | None
    plan_slice_receipt_sha256: str | None
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    desired_program_reads_during_build: int
    probe_output_reads_during_build: int
    action_label_inputs_during_build: int
    family_label_inputs_during_build: int
    schema_label_inputs_during_build: int
    topology_inputs_during_build: int
    work: TaskObservationBuildWork
    sha256: str


_ISSUED_CERTIFICATES: dict[int, TaskObservationAbsoluteMapFoldCertificate] = {}


class TaskObservationAbsoluteMapFoldCertificate:
    """Opaque exact-object authority containing zero or one executable witness."""

    __slots__ = ("_record",)

    def __init__(self, record: _CertificateRecord) -> None:
        if record.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("task-observation witness requires private authority")
        self._record = record

    @property
    def status(self) -> str:
        return self._record.status

    @property
    def stop_reason(self) -> str | None:
        return self._record.stop_reason

    @property
    def task_source_sha256(self) -> str:
        return self._record.task_source_sha256

    @property
    def example_effect_cell_id(self) -> int | None:
        return self._record.example_effect_cell_id

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
    def work(self) -> TaskObservationBuildWork:
        return self._record.work

    @property
    def sha256(self) -> str:
        return self._record.sha256


class TaskObservationAbsoluteMapFoldFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _freeze_visible_inputs(
    values: Sequence[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("visible inputs must be an exact row sequence")
    frozen = tuple(values)
    if not frozen:
        raise ValueError("task observation requires visible examples")
    for row in frozen:
        if type(row) is not tuple or len(row) != 1 or type(row[0]) is not list:
            raise ValueError("visible inputs must be exact List[Int] rows")
        items = cast(list[object], row[0])
        if not items or len(items) > output_cells.LIST_CAPACITY:
            raise ValueError("visible List[Int] rows exceed the physical capacity")
        if any(type(item) is not int for item in items):
            raise ValueError("visible List[Int] values must be exact integers")
    return frozen


def _validate_task_observation(
    value: NeutralTaskObservation,
    signature: TypedSignature,
) -> tuple[
    tuple[tuple[object, ...], ...],
    tuple[output_cells.TypedOutputEncoding, ...],
    str,
]:
    if type(value) is not NeutralTaskObservation:
        raise TypeError("build requires one exact neutral task observation")
    if not isinstance(signature, TypedSignature):
        raise TypeError("task observation requires one typed signature adapter")
    if type(value.prompt_bytes) is not bytes or not value.prompt_bytes:
        raise ValueError("neutral prompt bytes must be nonempty")
    if (
        type(value.prompt_cells) is not tuple
        or not value.prompt_cells
        or any(type(cell) is not int or cell < 0 for cell in value.prompt_cells)
        or len(set(value.prompt_cells)) != len(value.prompt_cells)
    ):
        raise ValueError("neutral prompt cells must be unique nonnegative identities")
    cases = _freeze_visible_inputs(value.visible_inputs)
    if (
        type(value.visible_outputs) is not tuple
        or len(value.visible_outputs) != len(cases)
        or any(type(item) is not output_cells.TypedOutputEncoding for item in value.visible_outputs)
    ):
        raise ValueError("visible outputs must align exactly with visible inputs")
    observed = tuple(
        output_cells.TypedOutputCellBank.validate(item) for item in value.visible_outputs
    )
    if not all(item.supported for item in observed):
        raise ValueError("visible outputs must be physically supported")
    source = {
        "prompt_bytes_hex": value.prompt_bytes.hex(),
        "prompt_cells": value.prompt_cells,
        "visible_inputs": cases,
        "visible_output_sha256s": tuple(item.sha256 for item in observed),
        "signature": repr(signature),
    }
    return cases, observed, _digest(_SOURCE_NAMESPACE, source)


def _physical_candidate_consequences(
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
) -> PhysicalConsequenceProjection:
    """Project both candidates through existing operator cells, with no receipt."""

    builder = absolute._AbsoluteListIntBuilder(
        signature,
        cases,
        absolute.FIXED_MAX_COST,
        absolute_cell_gate=True,
    )
    operator = builder.operator
    operator.clear_effect_memo()
    before = operator.live_checkpoint_sha256
    if before != operator.checkpoint_sha256:
        raise RuntimeError("typed operator checkpoint changed before projection")

    raw_outputs: list[output_cells.TypedOutputEncoding] = []
    absolute_outputs: list[output_cells.TypedOutputEncoding] = []
    for row in cases:
        values = cast(list[int], row[0])
        raw_accumulator = output_cells.TypedOutputCellBank.encode(
            0, declared_type=output_cells.OutputType.INT
        )
        absolute_accumulator = output_cells.TypedOutputCellBank.encode(
            0, declared_type=output_cells.OutputType.INT
        )
        for value in values:
            item = output_cells.TypedOutputCellBank.encode(
                value, declared_type=output_cells.OutputType.INT
            )
            raw_accumulator = builder._predict_effect(
                "add", raw_accumulator, item
            )
            mapped = builder._predict_absolute(item)
            absolute_accumulator = builder._predict_effect(
                "add", absolute_accumulator, mapped
            )
        raw_outputs.append(raw_accumulator)
        absolute_outputs.append(absolute_accumulator)

    after = operator.live_checkpoint_sha256
    if after != before:
        raise RuntimeError("typed operator checkpoint changed during projection")
    outputs = (tuple(raw_outputs), tuple(absolute_outputs))
    payload = {
        "candidate_output_sha256s": tuple(
            tuple(item.sha256 for item in candidate) for candidate in outputs
        ),
        "physical_predictions": builder.physical_predictions,
        "prediction_cache_hits": builder.prediction_cache_hits,
        "physical_races": builder.physical_races,
        "operator_checkpoint_sha256": before,
    }
    return PhysicalConsequenceProjection(
        outputs,
        builder.physical_predictions,
        builder.prediction_cache_hits,
        builder.physical_races,
        before,
        _digest(_PROJECTION_NAMESPACE, payload),
    )


def _observation_payload(value: ExampleEffectObservation) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _observe_example_effect(
    task_observation: NeutralTaskObservation,
    signature: TypedSignature,
    *,
    source_binding_cell_gate: bool,
) -> tuple[ExampleEffectObservation, tuple[tuple[object, ...], ...]]:
    if type(source_binding_cell_gate) is not bool:
        raise TypeError("source binding gate must be an exact boolean")
    cases, observed, source_sha = _validate_task_observation(task_observation, signature)
    if not source_binding_cell_gate:
        partial = ExampleEffectObservation(
            STOP_SOURCE_BINDING,
            "source-bound task observation cell was silent",
            source_sha,
            None,
            (),
            0,
            0,
            0,
            0,
            None,
            None,
            "",
        )
        return partial._replace(
            sha256=_digest(_OBSERVATION_NAMESPACE, _observation_payload(partial))
        ), cases

    projection = _physical_candidate_consequences(signature, cases)
    keys = tuple(f"example-effect-cell-{cell}" for cell in _CANDIDATE_EFFECT_CELLS)
    race = semantic_race.select_exact_semantic_vector(
        keys, projection.candidate_outputs, observed
    )
    exact_cells = tuple(
        int(key.rsplit("-", 1)[1]) for key in race.exact_winner_keys
    )
    if race.status == semantic_race.SELECTED and race.winner_key is not None:
        status = READY
        stop_reason = None
        winner = int(race.winner_key.rsplit("-", 1)[1])
    else:
        status = STOP_EXAMPLE_EFFECT_NOT_SINGLETON
        stop_reason = "visible examples selected zero or multiple exact effects"
        winner = None
    partial = ExampleEffectObservation(
        status,
        stop_reason,
        source_sha,
        winner,
        exact_cells,
        projection.physical_predictions,
        projection.prediction_cache_hits,
        projection.physical_races,
        EXAMPLE_EFFECT_RACE_CELL_STEPS,
        projection.sha256,
        race.sha256,
        "",
    )
    return partial._replace(
        sha256=_digest(_OBSERVATION_NAMESPACE, _observation_payload(partial))
    ), cases


def _effect_ordinal(effect_cell_id: int) -> int:
    ordinal = effect_cell_id - EXAMPLE_EFFECT_CELL_OFFSET
    if not 0 <= ordinal < EXAMPLE_EFFECT_CELL_COUNT:
        raise TaskObservationAbsoluteMapFoldFitStop(
            STOP_EXAMPLE_EFFECT_NOT_SINGLETON,
            "example effect lies outside the anonymous two-cell population",
        )
    return ordinal


def _candidate_action_cell(effect_cell_id: int) -> int:
    try:
        candidate = _CANDIDATE_EFFECT_CELLS.index(effect_cell_id)
    except ValueError as error:
        raise TaskObservationAbsoluteMapFoldFitStop(
            STOP_EXAMPLE_EFFECT_NOT_SINGLETON,
            "example effect is not an issued candidate consequence",
        ) from error
    return map_fold.TRANSFORM_ACTION_CELL_OFFSET + candidate


def _learn_transform(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    source: int,
    action: int,
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for candidate in range(map_fold.TRANSFORM_ACTION_CELL_COUNT):
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
    effect_cell_id: int,
    *,
    cell_gate: bool,
) -> _TransformRaceDecision:
    source = _effect_ordinal(effect_cell_id)
    raw_scores = tuple(
        int(excitatory[source, action]) - int(inhibitory[source, action])
        for action in range(map_fold.TRANSFORM_ACTION_CELL_COUNT)
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
        action_cell = map_fold.TRANSFORM_ACTION_CELL_OFFSET + spike_cells[0]
    payload = {
        "example_effect_cell_id": effect_cell_id,
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
        _digest(_TRANSFORM_NAMESPACE, payload),
    )


def _checkpoint_sha256(
    parent_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + parent_sha256.encode("ascii")
        + b"\0"
        + _array_frame("task-effect-excitatory", excitatory)
        + _array_frame("task-effect-inhibitory", inhibitory)
        + _canonical_json(
            {
                "example_effect_cell_offset": EXAMPLE_EFFECT_CELL_OFFSET,
                "transform_action_cell_offset": map_fold.TRANSFORM_ACTION_CELL_OFFSET,
                "total_cell_count": TOTAL_CELL_COUNT,
            }
        )
    ).hexdigest()


def _training_work_payload(value: TaskObservationTrainingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def fit_causal_task_observation_absolute_map_fold_v2(
    parent_checkpoint: map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1,
    lessons: tuple[NeutralTaskObservation, ...],
    *,
    signature: TypedSignature,
) -> FrozenCausalTaskObservationAbsoluteMapFoldV2:
    """Fit one A/B/A-prime example-effect-to-transform relay."""

    if (
        type(parent_checkpoint) is not map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1
        or type(lessons) is not tuple
        or len(lessons) != TRAIN_LESSON_COUNT
        or any(type(item) is not NeutralTaskObservation for item in lessons)
    ):
        raise TypeError("task-observation fit requires an exact parent and three lessons")
    if not isinstance(signature, TypedSignature):
        raise TypeError("task-observation fit requires one typed signature adapter")
    parent_checkpoint.authenticate()
    parent_sha = parent_checkpoint.checkpoint_sha256

    excitatory = np.zeros(LEARNED_STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    effect_cells: list[int] = []
    action_cells: list[int] = []
    observation_roots: list[str] = []
    predictions = 0
    cache_hits = 0
    physical_races = 0
    e_ticks = 0
    h_ticks = 0

    for lesson in lessons:
        observation, _cases = _observe_example_effect(
            lesson, signature, source_binding_cell_gate=True
        )
        effect_cell = observation.example_effect_cell_id
        if observation.status != READY or effect_cell is None:
            raise TaskObservationAbsoluteMapFoldFitStop(
                observation.status,
                observation.stop_reason or observation.status,
            )
        action_cell = _candidate_action_cell(effect_cell)
        source = _effect_ordinal(effect_cell)
        action = action_cell - map_fold.TRANSFORM_ACTION_CELL_OFFSET
        delta_e, delta_h = _learn_transform(
            excitatory, inhibitory, source, action
        )
        e_ticks += delta_e
        h_ticks += delta_h
        effect_cells.append(effect_cell)
        action_cells.append(action_cell)
        observation_roots.append(observation.sha256)
        predictions += observation.physical_predictions
        cache_hits += observation.prediction_cache_hits
        physical_races += observation.physical_races

    if (
        effect_cells[0] != effect_cells[2]
        or effect_cells[0] == effect_cells[1]
        or action_cells[0] != action_cells[2]
        or action_cells[0] == action_cells[1]
    ):
        raise TaskObservationAbsoluteMapFoldFitStop(
            STOP_CAUSAL_PAIR_CONFLICT,
            "A/B/A-prime did not identify two reusable consequence laws",
        )

    checkpoint_sha = _checkpoint_sha256(parent_sha, excitatory, inhibitory)
    effect_steps = TRAIN_LESSON_COUNT * EXAMPLE_EFFECT_RACE_CELL_STEPS
    partial_work = TaskObservationTrainingWork(
        TRAIN_LESSON_COUNT,
        predictions,
        cache_hits,
        physical_races,
        TRAIN_LESSON_COUNT,
        effect_steps,
        tuple(observation_roots),
        effect_steps,
        "",
    )
    work = partial_work._replace(
        sha256=_digest(_TRAIN_WORK_NAMESPACE, _training_work_payload(partial_work))
    )
    learned_pairs = tuple(
        sorted(set(zip(effect_cells, action_cells, strict=True)))
    )
    summary = TaskObservationTrainingSummary(
        TRAIN_LESSON_COUNT,
        tuple(effect_cells),
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
    return FrozenCausalTaskObservationAbsoluteMapFoldV2(
        parent_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _build_work_payload(value: TaskObservationBuildWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _freeze_build_work(
    observation: ExampleEffectObservation,
    *,
    transform_steps: int = 0,
    packed_build_count: int = 0,
    authorized_slice_count: int = 0,
    proof_materialization_count: int = 0,
    metrics: map_fold._PackedMetrics | None = None,
) -> TaskObservationBuildWork:
    stacked = observation.example_effect_race_cell_steps + transform_steps
    partial = TaskObservationBuildWork(
        observation.physical_predictions,
        observation.prediction_cache_hits,
        observation.physical_races,
        observation.example_effect_race_cell_steps,
        transform_steps,
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
    checkpoint: FrozenCausalTaskObservationAbsoluteMapFoldV2,
    observation: ExampleEffectObservation,
    *,
    status: str,
    stop_reason: str | None,
    transform_race_sha256: str | None,
    transform_action_cell_id: int | None,
    plan_slice_receipt_sha256: str | None,
    exact_proof_count: int,
    witness: breadth_v3.PackedGenericTypedWitness | None,
    work: TaskObservationBuildWork,
) -> TaskObservationAbsoluteMapFoldCertificate:
    partial = _CertificateRecord(
        _CERTIFICATE_AUTHORITY,
        checkpoint.checkpoint_sha256,
        status,
        stop_reason,
        observation.task_source_sha256,
        observation.sha256,
        observation.example_effect_cell_id,
        transform_race_sha256,
        transform_action_cell_id,
        plan_slice_receipt_sha256,
        exact_proof_count,
        witness,
        DESIRED_PROGRAM_READS_DURING_BUILD,
        PROBE_OUTPUT_READS_DURING_BUILD,
        ACTION_LABEL_INPUTS_DURING_BUILD,
        FAMILY_LABEL_INPUTS_DURING_BUILD,
        SCHEMA_LABEL_INPUTS_DURING_BUILD,
        TOPOLOGY_INPUTS_DURING_BUILD,
        work,
        "",
    )
    record = partial._replace(
        sha256=_digest(_CERTIFICATE_NAMESPACE, _certificate_payload(partial))
    )
    certificate = TaskObservationAbsoluteMapFoldCertificate(record)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


class FrozenCausalTaskObservationAbsoluteMapFoldV2:
    """Immutable task-observation-to-executable two-task checkpoint."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_parent_checkpoint",
        "_summary",
    )

    def __init__(
        self,
        parent_checkpoint: map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: TaskObservationTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("task-observation checkpoints are builder-owned")
        if (
            type(parent_checkpoint) is not map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1
            or excitatory.shape != LEARNED_STATE_SHAPE
            or inhibitory.shape != LEARNED_STATE_SHAPE
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("task-observation checkpoint geometry changed")
        parent_checkpoint.authenticate()
        self._parent_checkpoint = parent_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            parent_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("task-observation training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> TaskObservationTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._parent_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._parent_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
        ):
            raise RuntimeError("task-observation checkpoint authentication failed")

    def build(
        self,
        task_observation: NeutralTaskObservation,
        *,
        signature: TypedSignature,
        source_binding_cell_gate: bool = True,
        transform_action_cell_gate: bool = True,
    ) -> TaskObservationAbsoluteMapFoldCertificate:
        """Select by visible consequence, then materialize exactly one plan."""

        if type(transform_action_cell_gate) is not bool:
            raise TypeError("transform action gate must be an exact boolean")
        self.authenticate()
        observation, cases = _observe_example_effect(
            task_observation,
            signature,
            source_binding_cell_gate=source_binding_cell_gate,
        )
        effect_cell = observation.example_effect_cell_id
        if observation.status != READY or effect_cell is None:
            work = _freeze_build_work(observation)
            return _issue_certificate(
                self,
                observation,
                status=observation.status,
                stop_reason=observation.stop_reason,
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
            effect_cell,
            cell_gate=transform_action_cell_gate,
        )
        action_cell = transform_race.transform_action_cell_id
        if transform_race.status != READY or action_cell is None:
            work = _freeze_build_work(
                observation,
                transform_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
            )
            return _issue_certificate(
                self,
                observation,
                status=transform_race.status,
                stop_reason="learned transform population abstained",
                transform_race_sha256=transform_race.sha256,
                transform_action_cell_id=None,
                plan_slice_receipt_sha256=None,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        metrics: map_fold._PackedMetrics | None = None
        receipt_sha: str | None = None
        try:
            if action_cell == map_fold.IDENTITY_TRANSFORM_CELL:
                delegate, binding = map_fold._raw_sum_binding(signature, cases)
                metrics = map_fold._metrics(delegate)
                receipt = delegate.plan_slice_receipt
                if receipt is None:
                    raise TaskObservationAbsoluteMapFoldFitStop(
                        STOP_PLAN_SLICE, "raw SUM receipt disappeared"
                    )
                receipt_sha = receipt.sha256
                try:
                    forest = plan_slice._materialize_authorized_forest(binding)
                except packed.PackedRelationWorkLimitExceeded as error:
                    raise TaskObservationAbsoluteMapFoldFitStop(
                        STOP_WORK_BOUND,
                        "raw SUM singleton materialization exceeded packed work",
                    ) from error
                proof_count = forest.exact_logical_recipe_count
                witnesses = forest.first_two_witnesses
                if proof_count != 1 or len(witnesses) != 1:
                    raise TaskObservationAbsoluteMapFoldFitStop(
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
            elif action_cell == map_fold.ABSOLUTE_TRANSFORM_CELL:
                publication = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
                    signature, cases
                )
                metrics = map_fold._metrics(publication)
                receipt = publication.plan_slice_receipt
                if publication.status != absolute.READY or receipt is None:
                    raise TaskObservationAbsoluteMapFoldFitStop(
                        STOP_PLAN_SLICE,
                        publication.state.stop_reason or publication.status,
                    )
                receipt_sha = receipt.sha256
                materialization = publication.materialize()
                if (
                    materialization.status != absolute.READY
                    or materialization.exact_proof_count != 1
                    or materialization.witness is None
                ):
                    raise TaskObservationAbsoluteMapFoldFitStop(
                        STOP_WITNESS_NOT_SINGLETON,
                        materialization.stop_reason or materialization.status,
                    )
                proof_count = materialization.exact_proof_count
                witness = materialization.witness
            else:
                raise RuntimeError("transform winner left V1's two action cells")
        except (
            map_fold.OccurrenceAbsoluteMapFoldFitStop,
            TaskObservationAbsoluteMapFoldFitStop,
        ) as error:
            status = error.status
            reason = error.reason
            work = _freeze_build_work(
                observation,
                transform_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
                packed_build_count=1,
                authorized_slice_count=int(receipt_sha is not None),
                proof_materialization_count=0,
                metrics=metrics,
            )
            return _issue_certificate(
                self,
                observation,
                status=status,
                stop_reason=reason,
                transform_race_sha256=transform_race.sha256,
                transform_action_cell_id=action_cell,
                plan_slice_receipt_sha256=receipt_sha,
                exact_proof_count=0,
                witness=None,
                work=work,
            )

        work = _freeze_build_work(
            observation,
            transform_steps=TRANSFORM_ACTION_RACE_CELL_STEPS,
            packed_build_count=1,
            authorized_slice_count=1,
            proof_materialization_count=1,
            metrics=metrics,
        )
        self.authenticate()
        return _issue_certificate(
            self,
            observation,
            status=READY,
            stop_reason=None,
            transform_race_sha256=transform_race.sha256,
            transform_action_cell_id=action_cell,
            plan_slice_receipt_sha256=receipt_sha,
            exact_proof_count=proof_count,
            witness=witness,
            work=work,
        )


def _authenticate_certificate(
    value: TaskObservationAbsoluteMapFoldCertificate,
) -> _CertificateRecord:
    if (
        type(value) is not TaskObservationAbsoluteMapFoldCertificate
        or _ISSUED_CERTIFICATES.get(id(value)) is not value
    ):
        raise TypeError("exact issued task-observation certificate required")
    record = value._record
    if (
        record.authority is not _CERTIFICATE_AUTHORITY
        or record.work.sha256
        != _digest(_BUILD_WORK_NAMESPACE, _build_work_payload(record.work))
        or record.sha256
        != _digest(_CERTIFICATE_NAMESPACE, _certificate_payload(record))
    ):
        raise RuntimeError("task-observation certificate authentication failed")
    return record


def require_task_observation_absolute_map_fold_witness(
    value: TaskObservationAbsoluteMapFoldCertificate,
) -> breadth_v3.PackedGenericTypedWitness:
    record = _authenticate_certificate(value)
    if record.status != READY or record.exact_proof_count != 1 or record.witness is None:
        raise ValueError(record.stop_reason or record.status)
    return record.witness


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "example_effect_cell_range": (
            EXAMPLE_EFFECT_CELL_OFFSET,
            TOTAL_CELL_COUNT - 1,
        ),
        "reused_transform_action_cell_range": (
            map_fold.TRANSFORM_ACTION_CELL_OFFSET,
            map_fold.TRANSFORM_ACTION_CELL_OFFSET
            + map_fold.TRANSFORM_ACTION_CELL_COUNT
            - 1,
        ),
        "learned_state_shape": LEARNED_STATE_SHAPE,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
        "composed_persistent_state_bytes": COMPOSED_PERSISTENT_STATE_BYTES,
        "example_effect_race_cell_steps": EXAMPLE_EFFECT_RACE_CELL_STEPS,
        "transform_action_race_cell_steps": TRANSFORM_ACTION_RACE_CELL_STEPS,
        "desired_program_reads_during_build": DESIRED_PROGRAM_READS_DURING_BUILD,
        "probe_output_reads_during_build": PROBE_OUTPUT_READS_DURING_BUILD,
        "action_label_inputs_during_build": ACTION_LABEL_INPUTS_DURING_BUILD,
        "family_label_inputs_during_build": FAMILY_LABEL_INPUTS_DURING_BUILD,
        "schema_label_inputs_during_build": SCHEMA_LABEL_INPUTS_DURING_BUILD,
        "topology_inputs_during_build": TOPOLOGY_INPUTS_DURING_BUILD,
        "claim_boundary": CLAIM_BOUNDARY,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", dict(CONFIG_RECORD))


__all__ = [
    "ABSOLUTE_EXAMPLE_EFFECT_CELL",
    "ACTION_LABEL_INPUTS_DURING_BUILD",
    "CLAIM_BOUNDARY",
    "COMPOSED_PERSISTENT_STATE_BYTES",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_PROGRAM_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "DIRECT_V1_ACTION_READS",
    "EXAMPLE_EFFECT_CELL_OFFSET",
    "EXAMPLE_EFFECT_RACE_CELL_STEPS",
    "FAMILY_LABEL_INPUTS_DURING_BUILD",
    "G0_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "PATH2_INPUT_COUNT",
    "PROBE_OUTPUT_READS_DURING_BUILD",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "RAW_EXAMPLE_EFFECT_CELL",
    "READY",
    "SCHEMA",
    "SCHEMA_LABEL_INPUTS_DURING_BUILD",
    "SOURCE_COMMIT",
    "STOP_CAUSAL_PAIR_CONFLICT",
    "STOP_EXAMPLE_EFFECT_NOT_SINGLETON",
    "STOP_PLAN_SLICE",
    "STOP_SOURCE_BINDING",
    "STOP_TRANSFORM_AMBIGUOUS",
    "STOP_TRANSFORM_ZERO",
    "STOP_WITNESS_NOT_SINGLETON",
    "STOP_WORK_BOUND",
    "TOPOLOGY_INPUTS_DURING_BUILD",
    "TOTAL_CELL_COUNT",
    "TRANSFORM_ACTION_RACE_CELL_STEPS",
    "FrozenCausalTaskObservationAbsoluteMapFoldV2",
    "NeutralTaskObservation",
    "TaskObservationAbsoluteMapFoldCertificate",
    "TaskObservationAbsoluteMapFoldFitStop",
    "TaskObservationBuildWork",
    "TaskObservationTrainingSummary",
    "TaskObservationTrainingWork",
    "fit_causal_task_observation_absolute_map_fold_v2",
    "require_task_observation_absolute_map_fold_witness",
]
