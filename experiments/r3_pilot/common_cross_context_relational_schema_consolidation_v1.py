"""Cross-context constant-consequence schemas over the banked episodic memory.

One deep interface fits opaque subject/relation/object/source episodes and
freezes both views. Source-conditioned particulars are delegated unchanged to
V1. A new relation-keyed population accepts local votes only from physically
distinct subject+source contexts, then competes with an explicit tonic
abstention cell. The subject is projected at schema recall but is not answer
evidence, allowing the learned relation consequence to transfer to a novel
subject context.

This self-posed slice learns only relation-wide constant one-byte consequences.
It does not learn arbitrary rules, variables, syntax, or language.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as projection
from experiments.r3_pilot import common_episodic_semantic_consolidation_v1 as v1

SCOPE = "COMMON_CROSS_CONTEXT_RELATIONAL_SCHEMA_CONSOLIDATION_V1"
STATUS = "SELF_POSED_RELATION_WIDE_CONSTANT_CONSEQUENCE_MECHANISM_ONLY"
CLAIM_BOUNDARY = (
    "self-posed-relation-wide-constant-one-byte-consequence-transfer-only-"
    "no-arbitrary-rules-or-language-understanding"
)

V1_SOURCE_COMMIT = "3524da8f1a307fe23813231a94548d9814efacf1"
EXPECTED_V1_SOURCE_SHA256 = "cb80153bb5dc834bd554d85f53a5dbfa301615b294a95cc4aa34d8c7c73e726a"
EXPECTED_V1_CONFIG_SHA256 = "6e374cf9907d004186fdc1cb3ddb0ef32f6b6b3ba0b0302206600a44a7357494"

MAX_SCHEMA_NEURONS = 32
MAX_INDEPENDENT_CONTEXTS = 64
RELATION_CELLS = projection.TRACE_LANES
SUBJECT_CELLS = projection.TRACE_LANES
SOURCE_CELLS = projection.TRACE_LANES
OBJECT_CELLS = projection.TRACE_LANES
OUTPUT_SYNAPSES = projection.OUTPUT_DIGIT_LANES

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
POSITIVE_TICK = 1
NEGATIVE_EVIDENCE_TICK = 1
RELATION_STOP_SCORE = RELATION_CELLS
MAX_SCHEMA_SCORE = RELATION_CELLS * TICK_CAP

SCHEMA_CELL_OFFSET = len(v1.ROLE_NAMES) * projection.TOTAL_CELL_COUNT + v1.MAX_SEMANTIC_NEURONS
TONIC_ABSTENTION_CELL_ID = SCHEMA_CELL_OFFSET + MAX_SCHEMA_NEURONS
NEW_SCHEMA_CELL_COUNT = MAX_SCHEMA_NEURONS + 1
REUSED_COMMON_EVENT_CELL_COUNT = projection.TOTAL_CELL_COUNT
REUSED_EPISODIC_CELL_COUNT = associative.TOTAL_CELL_COUNT
ROLE_PROJECTION_CELL_ADDRESS_COUNT = len(v1.ROLE_NAMES) * projection.TOTAL_CELL_COUNT
TOTAL_DECLARED_CELL_COUNT = (
    REUSED_EPISODIC_CELL_COUNT
    + ROLE_PROJECTION_CELL_ADDRESS_COUNT
    + v1.MAX_SEMANTIC_NEURONS
    + 1
    + NEW_SCHEMA_CELL_COUNT
)

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_V1_SOURCE_PATH = _ROOT / "experiments/r3_pilot/common_episodic_semantic_consolidation_v1.py"
_CONFIG_NAMESPACE = b"common-cross-context-relational-schema-v1-config\0"
_CHECKPOINT_NAMESPACE = b"common-cross-context-relational-schema-v1-checkpoint\0"
_RECALL_NAMESPACE = b"common-cross-context-relational-schema-v1-recall\0"
_CONSTRUCTION_TOKEN = object()


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json({"name": name, "dtype": array.dtype.str, "shape": array.shape})
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return len(header).to_bytes(8, "little") + header + len(payload).to_bytes(8, "little") + payload


def _checkpoint_payload(
    metadata: bytes,
    episodic_semantic_v1_sha256: str,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(metadata).to_bytes(8, "little")
        + metadata
        + bytes.fromhex(episodic_semantic_v1_sha256)
        + b"".join(_array_frame(name, array) for name, array in arrays)
    )


def _full_match(stored: np.ndarray, active: tuple[int, ...]) -> bool:
    expected = np.asarray(active, dtype=stored.dtype)
    return bool(np.count_nonzero(stored == expected) == len(active))


def _authenticate_v1() -> None:
    if (
        sha256(_V1_SOURCE_PATH.read_bytes()).hexdigest() != EXPECTED_V1_SOURCE_SHA256
        or v1.SOURCE_SHA256 != EXPECTED_V1_SOURCE_SHA256
        or v1.CONFIG_SHA256 != EXPECTED_V1_CONFIG_SHA256
        or v1.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA != 0
        or any(
            (
                v1.GENERATOR_EXECUTION_COUNT,
                v1.CORPUS_INPUT_COUNT,
                v1.DEV_INPUT_COUNT,
                v1.M0_INPUT_COUNT,
                v1.PROTECTED_INPUT_COUNT,
            )
        )
    ):
        raise RuntimeError("banked episodic-semantic V1 identity changed")


@dataclass(frozen=True, slots=True)
class RelationalSchemaLesson:
    subject_event: bytes
    relation_event: bytes
    object_event: bytes
    source_event: bytes
    teacher_gate: bool = True

    def __post_init__(self) -> None:
        v1.EpisodicSemanticLesson(
            self.subject_event,
            self.relation_event,
            self.object_event,
            self.source_event,
            self.teacher_gate,
        )

    def as_v1(self) -> v1.EpisodicSemanticLesson:
        return v1.EpisodicSemanticLesson(
            self.subject_event,
            self.relation_event,
            self.object_event,
            self.source_event,
            self.teacher_gate,
        )


@dataclass(frozen=True, slots=True)
class SchemaConsolidationStep:
    lesson_ordinal: int
    status: str
    schema_cell_id: int | None
    physical_output_token: int | None
    negative_evidence_events: int


@dataclass(frozen=True, slots=True)
class RelationalSchemaTrainingSummary:
    lesson_count: int
    teacher_on_lesson_count: int
    teacher_off_lesson_count: int
    episodic_novel_episode_count: int
    independent_schema_context_count: int
    dependent_context_count: int
    schema_neuron_count: int
    excitatory_update_events: int
    negative_evidence_update_events: int
    consolidation_steps: tuple[SchemaConsolidationStep, ...]
    v1_episodic_checkpoint_sha256: str
    v1_checkpoint_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class CrossContextRecall:
    mode: str
    status: str
    output: projection.CommonOutputDecision
    schema_cell_id: int | None
    subject_context_cells: tuple[int, ...]
    relation_cells: tuple[int, ...]
    conditioned_source_cells: tuple[int, ...]
    candidate_raw_scores: tuple[int, ...]
    candidate_spike_step: int | None
    candidate_simultaneous_spike_count: int | None
    physical_cell_steps: int
    v1_checkpoint_sha256: str
    checkpoint_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _ProjectedLesson:
    subject_cells: tuple[int, ...]
    relation_cells: tuple[int, ...]
    object_cells: tuple[int, ...]
    source_cells: tuple[int, ...]
    output_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SchemaDecision:
    status: str
    schema_cell_id: int | None
    output: projection.CommonOutputDecision
    raw_scores: tuple[int, ...]
    spike_step: int
    simultaneous_spike_count: int
    physical_cell_steps: int


def _project_lesson(
    cells: projection.CommonEventCells,
    lesson: RelationalSchemaLesson,
) -> _ProjectedLesson:
    return _ProjectedLesson(
        subject_cells=v1._role_cells(cells, lesson.subject_event, 0),
        relation_cells=v1._role_cells(cells, lesson.relation_event, 1),
        object_cells=v1._role_cells(cells, lesson.object_event, 2),
        source_cells=v1._role_cells(cells, lesson.source_event, 3),
        output_cells=v1._output_cells(lesson.object_event),
    )


def _schema_decision(
    cells: projection.CommonEventCells,
    relation_cells: np.ndarray,
    output_cells: np.ndarray,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_count: int,
    active_relation_cells: tuple[int, ...],
) -> _SchemaDecision:
    active = np.asarray(active_relation_cells, dtype=np.int32)
    raw_scores = tuple(
        int(
            np.sum(
                np.where(
                    relation_cells[row] == active,
                    excitatory[row].astype(np.int16) - inhibitory[row].astype(np.int16),
                    0,
                ),
                dtype=np.int64,
            )
        )
        for row in range(schema_count)
    )
    race = v1._candidate_race(raw_scores, stop_score=RELATION_STOP_SCORE)
    if race.decision == "WINNER":
        winner = race.winner_index
        if winner is None:
            raise RuntimeError("relation schema winner disappeared")
        output = v1._publish_output(
            cells,
            tuple(int(cell) for cell in output_cells[winner]),
        )
        status = "RECALLED"
        schema_cell_id = SCHEMA_CELL_OFFSET + winner
    else:
        output = v1._publish_output(cells, None)
        schema_cell_id = None
        matching_best = tuple(
            row
            for row, score in enumerate(raw_scores)
            if _full_match(relation_cells[row], active_relation_cells)
            and score == max(raw_scores, default=0)
            and score >= RELATION_STOP_SCORE
        )
        if len(matching_best) >= 2:
            status = "ABSTAIN_CONTRADICTION"
        elif max(raw_scores, default=0) == 0:
            status = "ABSTAIN_UNKNOWN"
        else:
            status = "ABSTAIN_LOW_SUPPORT"
    return _SchemaDecision(
        status=status,
        schema_cell_id=schema_cell_id,
        output=output,
        raw_scores=raw_scores,
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_count,
        physical_cell_steps=race.physical_cell_steps + output.physical_cell_steps,
    )


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "v1_source_commit": V1_SOURCE_COMMIT,
    "v1_source_sha256": EXPECTED_V1_SOURCE_SHA256,
    "v1_config_sha256": EXPECTED_V1_CONFIG_SHA256,
    "common_event_geometry_sha256": projection.GEOMETRY_SHA256,
    "episodic_geometry_sha256": associative.GEOMETRY_SHA256,
    "relation_cells": RELATION_CELLS,
    "maximum_schema_neurons": MAX_SCHEMA_NEURONS,
    "maximum_independent_contexts": MAX_INDEPENDENT_CONTEXTS,
    "positive_tick": POSITIVE_TICK,
    "negative_evidence_tick": NEGATIVE_EVIDENCE_TICK,
    "tick_cap": TICK_CAP,
    "relation_stop_score": RELATION_STOP_SCORE,
    "schema_publication_minimum_independent_contexts": 2,
    "independence": "relation-local-distinct-physical-subject-and-source-cells",
    "competition": "banked-fitted-glif-first-spike-plus-explicit-tonic-abstention",
    "subject_used_as_schema_answer_evidence": False,
    "raw_event_name_or_hash_used_as_neural_evidence": False,
    "host_answer_ranking": False,
    "new_schema_cell_count": NEW_SCHEMA_CELL_COUNT,
    "total_declared_cell_count": TOTAL_DECLARED_CELL_COUNT,
    "domain_specific_computational_primitive_delta": 0,
    "generator_corpus_path2_dev_m0_protected_g0_inputs": (0, 0, 0, 0, 0, 0, 0),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


class FrozenCommonCrossContextRelationalSchemas:
    """Immutable V1 episodes plus relation-wide schema synapses."""

    __slots__ = (
        "_cells",
        "_checkpoint_sha256",
        "_context_relations",
        "_context_sources",
        "_context_subjects",
        "_excitatory",
        "_inhibitory",
        "_metadata",
        "_object_cells",
        "_output_cells",
        "_relation_cells",
        "_schema_count",
        "_summary",
        "_v1_memory",
    )

    def __init__(
        self,
        *,
        v1_memory: v1.FrozenCommonEpisodicSemanticMemory,
        relation_cells: np.ndarray,
        object_cells: np.ndarray,
        output_cells: np.ndarray,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        context_relations: np.ndarray,
        context_subjects: np.ndarray,
        context_sources: np.ndarray,
        schema_count: int,
        metadata: bytes,
        summary: RelationalSchemaTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("cross-context relational schemas are builder-owned")
        expected = (
            (relation_cells, (MAX_SCHEMA_NEURONS, RELATION_CELLS), np.dtype(np.int32)),
            (object_cells, (MAX_SCHEMA_NEURONS, OBJECT_CELLS), np.dtype(np.int32)),
            (output_cells, (MAX_SCHEMA_NEURONS, OUTPUT_SYNAPSES), np.dtype(np.int32)),
            (excitatory, (MAX_SCHEMA_NEURONS, RELATION_CELLS), TICK_DTYPE),
            (inhibitory, (MAX_SCHEMA_NEURONS, RELATION_CELLS), TICK_DTYPE),
            (
                context_relations,
                (MAX_INDEPENDENT_CONTEXTS, RELATION_CELLS),
                np.dtype(np.int32),
            ),
            (
                context_subjects,
                (MAX_INDEPENDENT_CONTEXTS, SUBJECT_CELLS),
                np.dtype(np.int32),
            ),
            (
                context_sources,
                (MAX_INDEPENDENT_CONTEXTS, SOURCE_CELLS),
                np.dtype(np.int32),
            ),
        )
        if any(
            array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous
            for array, shape, dtype in expected
        ):
            raise ValueError("cross-context schema synapse geometry changed")
        if not 0 <= schema_count <= MAX_SCHEMA_NEURONS:
            raise ValueError("schema neuron count changed")
        self._cells = projection.CommonEventCells()
        self._v1_memory = v1_memory
        self._schema_count = schema_count
        self._metadata = bytes(metadata)
        self._summary = summary
        for name, array in (
            ("_relation_cells", relation_cells),
            ("_object_cells", object_cells),
            ("_output_cells", output_cells),
            ("_excitatory", excitatory),
            ("_inhibitory", inhibitory),
            ("_context_relations", context_relations),
            ("_context_subjects", context_subjects),
            ("_context_sources", context_sources),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256()
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("cross-context schema checkpoint identity changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("schema-relation-cells", self._relation_cells),
            ("schema-object-cells", self._object_cells),
            ("schema-output-cells", self._output_cells),
            ("schema-excitatory", self._excitatory),
            ("schema-inhibitory", self._inhibitory),
            ("context-relations", self._context_relations),
            ("context-subjects", self._context_subjects),
            ("context-sources", self._context_sources),
        )

    def _calculate_checkpoint_sha256(self) -> str:
        return sha256(
            _checkpoint_payload(
                self._metadata,
                self._v1_memory.checkpoint_sha256,
                self._checkpoint_arrays(),
            )
        ).hexdigest()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RelationalSchemaTrainingSummary:
        self.authenticate()
        return self._summary

    def authenticate(self) -> None:
        self._v1_memory.authenticate()
        if self._calculate_checkpoint_sha256() != self._checkpoint_sha256:
            raise RuntimeError("cross-context schema checkpoint authentication failed")

    def recall(
        self,
        subject_event: bytes,
        relation_event: bytes,
        *,
        source_event: bytes | None = None,
        cell_gate: bool = True,
    ) -> CrossContextRecall:
        """Recall one episode when sourced, otherwise the relation-wide schema."""

        self.authenticate()
        if type(cell_gate) is not bool:
            raise ValueError("cell_gate must be an exact boolean")
        subject_cells = v1._role_cells(self._cells, subject_event, 0)
        relation_cells = v1._role_cells(self._cells, relation_event, 1)
        if source_event is not None:
            if not cell_gate:
                output = v1._publish_output(self._cells, None)
                status = "STOP_CELL_OFF"
                source_cells: tuple[int, ...] = ()
                physical_steps = output.physical_cell_steps
            else:
                recalled = self._v1_memory.recall_episode(
                    subject_event,
                    relation_event,
                    source_event,
                )
                output = recalled.output
                status = recalled.status
                source_cells = recalled.conditioned_source_cells
                physical_steps = output.physical_cell_steps
            record = {
                "mode": "EPISODIC",
                "status": status,
                "output": output.record(),
                "subject_context_cells": subject_cells,
                "relation_cells": relation_cells,
                "conditioned_source_cells": source_cells,
                "v1_checkpoint_sha256": self._v1_memory.checkpoint_sha256,
                "checkpoint_sha256": self._checkpoint_sha256,
            }
            return CrossContextRecall(
                mode="EPISODIC",
                status=status,
                output=output,
                schema_cell_id=None,
                subject_context_cells=subject_cells,
                relation_cells=relation_cells,
                conditioned_source_cells=source_cells,
                candidate_raw_scores=(),
                candidate_spike_step=None,
                candidate_simultaneous_spike_count=None,
                physical_cell_steps=physical_steps,
                v1_checkpoint_sha256=self._v1_memory.checkpoint_sha256,
                checkpoint_sha256=self._checkpoint_sha256,
                sha256=_digest(_RECALL_NAMESPACE, record),
            )

        active_relation = relation_cells if cell_gate else tuple(-1 for _ in range(RELATION_CELLS))
        decision = _schema_decision(
            self._cells,
            self._relation_cells,
            self._output_cells,
            self._excitatory,
            self._inhibitory,
            self._schema_count,
            active_relation,
        )
        record = {
            "mode": "SCHEMA",
            "status": decision.status,
            "output": decision.output.record(),
            "schema_cell_id": decision.schema_cell_id,
            "subject_context_cells": subject_cells,
            "relation_cells": active_relation,
            "candidate_raw_scores": decision.raw_scores,
            "candidate_spike_step": decision.spike_step,
            "candidate_simultaneous_spike_count": decision.simultaneous_spike_count,
            "v1_checkpoint_sha256": self._v1_memory.checkpoint_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        return CrossContextRecall(
            mode="SCHEMA",
            status=decision.status,
            output=decision.output,
            schema_cell_id=decision.schema_cell_id,
            subject_context_cells=subject_cells,
            relation_cells=active_relation,
            conditioned_source_cells=(),
            candidate_raw_scores=decision.raw_scores,
            candidate_spike_step=decision.spike_step,
            candidate_simultaneous_spike_count=decision.simultaneous_spike_count,
            physical_cell_steps=decision.physical_cell_steps,
            v1_checkpoint_sha256=self._v1_memory.checkpoint_sha256,
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def fit_common_cross_context_relational_schemas(
    lessons: tuple[RelationalSchemaLesson, ...],
) -> FrozenCommonCrossContextRelationalSchemas:
    """Fit V1 episodes and local relation schemas, then freeze both views."""

    if type(lessons) is not tuple or any(
        type(lesson) is not RelationalSchemaLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of RelationalSchemaLesson")
    _authenticate_v1()
    cells = projection.CommonEventCells()
    v1_memory = v1.fit_common_episodic_semantic_consolidation(
        tuple(lesson.as_v1() for lesson in lessons)
    )
    relation_cells = np.full((MAX_SCHEMA_NEURONS, RELATION_CELLS), -1, dtype=np.int32)
    object_cells = np.full((MAX_SCHEMA_NEURONS, OBJECT_CELLS), -1, dtype=np.int32)
    output_cells = np.full((MAX_SCHEMA_NEURONS, OUTPUT_SYNAPSES), -1, dtype=np.int32)
    excitatory = np.zeros((MAX_SCHEMA_NEURONS, RELATION_CELLS), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    context_relations = np.full((MAX_INDEPENDENT_CONTEXTS, RELATION_CELLS), -1, dtype=np.int32)
    context_subjects = np.full((MAX_INDEPENDENT_CONTEXTS, SUBJECT_CELLS), -1, dtype=np.int32)
    context_sources = np.full((MAX_INDEPENDENT_CONTEXTS, SOURCE_CELLS), -1, dtype=np.int32)
    schema_count = 0
    context_count = 0
    dependent_context_count = 0
    excitatory_events = 0
    negative_events = 0
    steps: list[SchemaConsolidationStep] = []

    for ordinal, lesson in enumerate(lessons):
        if not lesson.teacher_gate:
            steps.append(SchemaConsolidationStep(ordinal, "TEACHER_OFF", None, None, 0))
            continue
        projected = _project_lesson(cells, lesson)
        matching_contexts = tuple(
            row
            for row in range(context_count)
            if _full_match(context_relations[row], projected.relation_cells)
        )
        dependent = any(
            _full_match(context_subjects[row], projected.subject_cells)
            or _full_match(context_sources[row], projected.source_cells)
            for row in matching_contexts
        )
        if dependent:
            dependent_context_count += 1
            steps.append(
                SchemaConsolidationStep(
                    ordinal,
                    "DEPENDENT_CONTEXT_IGNORED",
                    None,
                    None,
                    0,
                )
            )
            continue
        if context_count == MAX_INDEPENDENT_CONTEXTS:
            raise RuntimeError("independent schema context capacity exhausted")
        context_relations[context_count] = np.asarray(projected.relation_cells, dtype=np.int32)
        context_subjects[context_count] = np.asarray(projected.subject_cells, dtype=np.int32)
        context_sources[context_count] = np.asarray(projected.source_cells, dtype=np.int32)
        context_count += 1

        matching_relation_rows = tuple(
            row
            for row in range(schema_count)
            if _full_match(relation_cells[row], projected.relation_cells)
        )
        matching_object_rows = tuple(
            row
            for row in matching_relation_rows
            if _full_match(object_cells[row], projected.object_cells)
        )
        if len(matching_object_rows) > 1:
            raise RuntimeError("relation schema private context detector duplicated")
        if matching_object_rows:
            target = matching_object_rows[0]
        else:
            if schema_count == MAX_SCHEMA_NEURONS:
                raise RuntimeError("relation schema neuron capacity exhausted")
            target = schema_count
            schema_count += 1
            relation_cells[target] = np.asarray(projected.relation_cells, dtype=np.int32)
            object_cells[target] = np.asarray(projected.object_cells, dtype=np.int32)
            output_cells[target] = np.asarray(projected.output_cells, dtype=np.int32)

        previous = excitatory[target].astype(np.uint16)
        updated = np.minimum(previous + POSITIVE_TICK, TICK_CAP)
        excitatory[target] = updated.astype(TICK_DTYPE)
        excitatory_events += int(np.any(updated > previous))

        lesson_negative_events = 0
        for rival in matching_relation_rows:
            if rival == target:
                continue
            previous = inhibitory[rival].astype(np.uint16)
            updated = np.minimum(previous + NEGATIVE_EVIDENCE_TICK, TICK_CAP)
            inhibitory[rival] = updated.astype(TICK_DTYPE)
            committed = int(np.any(updated > previous))
            lesson_negative_events += committed
            negative_events += committed

        decision = _schema_decision(
            cells,
            relation_cells,
            output_cells,
            excitatory,
            inhibitory,
            schema_count,
            projected.relation_cells,
        )
        if decision.status == "RECALLED":
            step_status = "REVISED" if lesson_negative_events else "CONSOLIDATED"
        elif decision.status == "ABSTAIN_CONTRADICTION":
            step_status = "CONTRADICTION_ABSTAIN"
        else:
            step_status = "SUPPORT_PENDING"
        steps.append(
            SchemaConsolidationStep(
                lesson_ordinal=ordinal,
                status=step_status,
                schema_cell_id=decision.schema_cell_id,
                physical_output_token=decision.output.token,
                negative_evidence_events=lesson_negative_events,
            )
        )

    metadata_record = {
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_sha256": CONFIG_SHA256,
        "lesson_count": len(lessons),
        "schema_count": schema_count,
        "independent_context_count": context_count,
        "dependent_context_count": dependent_context_count,
        "v1_checkpoint_sha256": v1_memory.checkpoint_sha256,
        "raw_event_retention_outside_v1_episodic_memory": False,
        "generator_corpus_path2_dev_m0_protected_g0_inputs": (0, 0, 0, 0, 0, 0, 0),
    }
    metadata = _canonical_json(metadata_record)
    arrays = (
        ("schema-relation-cells", relation_cells),
        ("schema-object-cells", object_cells),
        ("schema-output-cells", output_cells),
        ("schema-excitatory", excitatory),
        ("schema-inhibitory", inhibitory),
        ("context-relations", context_relations),
        ("context-subjects", context_subjects),
        ("context-sources", context_sources),
    )
    checkpoint_sha256 = sha256(
        _checkpoint_payload(metadata, v1_memory.checkpoint_sha256, arrays)
    ).hexdigest()
    v1_summary = v1_memory.training_summary
    summary = RelationalSchemaTrainingSummary(
        lesson_count=len(lessons),
        teacher_on_lesson_count=sum(lesson.teacher_gate for lesson in lessons),
        teacher_off_lesson_count=sum(not lesson.teacher_gate for lesson in lessons),
        episodic_novel_episode_count=v1_summary.novel_episode_count,
        independent_schema_context_count=context_count,
        dependent_context_count=dependent_context_count,
        schema_neuron_count=schema_count,
        excitatory_update_events=excitatory_events,
        negative_evidence_update_events=negative_events,
        consolidation_steps=tuple(steps),
        v1_episodic_checkpoint_sha256=v1_summary.episodic_checkpoint_sha256,
        v1_checkpoint_sha256=v1_memory.checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonCrossContextRelationalSchemas(
        v1_memory=v1_memory,
        relation_cells=relation_cells,
        object_cells=object_cells,
        output_cells=output_cells,
        excitatory=excitatory,
        inhibitory=inhibitory,
        context_relations=context_relations,
        context_subjects=context_subjects,
        context_sources=context_sources,
        schema_count=schema_count,
        metadata=metadata,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


_authenticate_v1()
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "MAX_INDEPENDENT_CONTEXTS",
    "MAX_SCHEMA_NEURONS",
    "NEW_SCHEMA_CELL_COUNT",
    "REUSED_COMMON_EVENT_CELL_COUNT",
    "REUSED_EPISODIC_CELL_COUNT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "TONIC_ABSTENTION_CELL_ID",
    "TOTAL_DECLARED_CELL_COUNT",
    "CrossContextRecall",
    "FrozenCommonCrossContextRelationalSchemas",
    "RelationalSchemaLesson",
    "RelationalSchemaTrainingSummary",
    "SchemaConsolidationStep",
    "fit_common_cross_context_relational_schemas",
]
