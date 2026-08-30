"""Self-posed episodic retention and semantic consolidation over common cells.

The module has one deep interface: fit opaque subject/relation/object/source
events, then recall either a source-conditioned episode or the consolidated
subject/relation consequence.  Exact episodes reuse the common ordered-event
associative memory.  Semantic hypotheses are private cells with only local
excitatory and negative-evidence ticks; their publication is decided by a
fitted-GLIF race with a tonic abstention cell.

Object events are deliberately one byte in this first slice so the existing
four-lane common output population can publish them without a host decoder.
No raw event, event name, digest, or Python lookup table is retained as learned
state or consulted during recall.
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
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "COMMON_EPISODIC_SEMANTIC_CONSOLIDATION_V1"
STATUS = "SELF_POSED_EPISODIC_SEMANTIC_MECHANISM_ONLY"
CLAIM_BOUNDARY = (
    "self-posed-one-byte-object-episodic-retention-and-semantic-revision-only"
)

MAX_SEMANTIC_NEURONS = 32
ROLE_NAMES = ("subject", "relation", "object", "source")
ROLE_CELL_OFFSETS = tuple(
    index * projection.TOTAL_CELL_COUNT for index in range(len(ROLE_NAMES))
)
EVENT_ACTIVE_CELLS = projection.TRACE_LANES
SEMANTIC_KEY_CELLS = 2 * EVENT_ACTIVE_CELLS
OBJECT_CELLS = EVENT_ACTIVE_CELLS
OUTPUT_SYNAPSES = projection.OUTPUT_DIGIT_LANES

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
POSITIVE_TICK = 1
NEGATIVE_EVIDENCE_TICK = 1
SEMANTIC_STOP_SCORE = SEMANTIC_KEY_CELLS
EPISODIC_CONTEXT_CELLS = 3 * EVENT_ACTIVE_CELLS
MAX_SEMANTIC_SCORE = SEMANTIC_KEY_CELLS * TICK_CAP

RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568

GENERATOR_EXECUTION_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
CORPUS_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_SOURCE_PATH = Path(__file__).resolve()
_CONFIG_NAMESPACE = b"common-episodic-semantic-consolidation-v1-config\0"
_CHECKPOINT_NAMESPACE = b"common-episodic-semantic-consolidation-v1-checkpoint\0"
_RECALL_NAMESPACE = b"common-episodic-semantic-consolidation-v1-recall\0"
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
    header = _canonical_json(
        {"name": name, "dtype": array.dtype.str, "shape": array.shape}
    )
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return (
        len(header).to_bytes(8, "little")
        + header
        + len(payload).to_bytes(8, "little")
        + payload
    )


def _checkpoint_payload(
    metadata: bytes,
    episodic_checkpoint_sha256: str,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(metadata).to_bytes(8, "little")
        + metadata
        + bytes.fromhex(episodic_checkpoint_sha256)
        + b"".join(_array_frame(name, array) for name, array in arrays)
    )


def _build_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(MAX_SEMANTIC_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("semantic fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("semantic fitted-GLIF calibration changed")
    return table, race


_CURRENT_TABLE, _CELL_RACE = _build_current_table()


@dataclass(frozen=True, slots=True)
class EpisodicSemanticLesson:
    subject_event: bytes
    relation_event: bytes
    object_event: bytes
    source_event: bytes
    teacher_gate: bool = True

    def __post_init__(self) -> None:
        for name in ("subject_event", "relation_event", "source_event"):
            event = getattr(self, name)
            if (
                type(event) is not bytes
                or not 1 <= len(event) <= projection.MAX_EVENT_BYTES
            ):
                raise ValueError(f"{name} must be one bounded opaque byte event")
        if type(self.object_event) is not bytes or len(self.object_event) != 1:
            raise ValueError("object_event must be exactly one opaque byte")
        if type(self.teacher_gate) is not bool:
            raise ValueError("teacher_gate must be an exact boolean")


@dataclass(frozen=True, slots=True)
class ConsolidationStep:
    lesson_ordinal: int
    status: str
    semantic_cell_id: int | None
    physical_output_token: int | None
    negative_evidence_events: int


@dataclass(frozen=True, slots=True)
class ConsolidationTrainingSummary:
    lesson_count: int
    teacher_on_lesson_count: int
    teacher_off_lesson_count: int
    novel_episode_count: int
    duplicate_source_count: int
    semantic_neuron_count: int
    excitatory_update_events: int
    negative_evidence_update_events: int
    consolidation_steps: tuple[ConsolidationStep, ...]
    episodic_checkpoint_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class SemanticRecall:
    status: str
    semantic_cell_id: int | None
    output: projection.CommonOutputDecision
    active_key_cells: tuple[int, ...]
    candidate_raw_scores: tuple[int, ...]
    candidate_spike_step: int
    candidate_simultaneous_spike_count: int
    physical_cell_steps: int
    checkpoint_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class EpisodicRecall:
    status: str
    output: projection.CommonOutputDecision
    conditioned_source_cells: tuple[int, ...]
    episodic_checkpoint_sha256: str
    checkpoint_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _ProjectedLesson:
    subject_cells: tuple[int, ...]
    relation_cells: tuple[int, ...]
    object_cells: tuple[int, ...]
    source_cells: tuple[int, ...]
    output_cells: tuple[int, ...]

    @property
    def key_cells(self) -> tuple[int, ...]:
        return (*self.subject_cells, *self.relation_cells)

    @property
    def episodic_context_cells(self) -> tuple[int, ...]:
        return (*self.subject_cells, *self.relation_cells, *self.source_cells)


@dataclass(frozen=True, slots=True)
class _CandidateRace:
    decision: str
    winner_index: int | None
    spike_step: int
    simultaneous_spike_count: int
    physical_cell_steps: int


@dataclass(frozen=True, slots=True)
class _SemanticDecision:
    status: str
    semantic_cell_id: int | None
    output: projection.CommonOutputDecision
    raw_scores: tuple[int, ...]
    race: _CandidateRace


def _role_cells(
    cells: projection.CommonEventCells,
    event: bytes,
    role_index: int,
) -> tuple[int, ...]:
    frame = cells.project(event)
    offset = ROLE_CELL_OFFSETS[role_index]
    return tuple(offset + cell for cell in frame.trace_cells)


def _output_cells(object_event: bytes) -> tuple[int, ...]:
    value = object_event[0]
    digits = [0] * projection.OUTPUT_DIGIT_LANES
    for lane in range(projection.OUTPUT_DIGIT_LANES - 1, -1, -1):
        digits[lane] = value % projection.OUTPUT_DIGIT_RADIX
        value //= projection.OUTPUT_DIGIT_RADIX
    return tuple(
        projection.OUTPUT_CELL_OFFSET
        + lane * projection.OUTPUT_DIGIT_RADIX
        + digit
        for lane, digit in enumerate(digits)
    )


def _project_lesson(
    cells: projection.CommonEventCells,
    lesson: EpisodicSemanticLesson,
) -> _ProjectedLesson:
    return _ProjectedLesson(
        subject_cells=_role_cells(cells, lesson.subject_event, 0),
        relation_cells=_role_cells(cells, lesson.relation_event, 1),
        object_cells=_role_cells(cells, lesson.object_event, 2),
        source_cells=_role_cells(cells, lesson.source_event, 3),
        output_cells=_output_cells(lesson.object_event),
    )


def _candidate_race(
    raw_scores: tuple[int, ...],
    *,
    stop_score: int,
) -> _CandidateRace:
    bounded = tuple(max(0, min(score, MAX_SEMANTIC_SCORE)) for score in raw_scores)
    currents = np.asarray(
        [[*(_CURRENT_TABLE[score] for score in bounded), _CURRENT_TABLE[stop_score]]],
        dtype=np.float64,
    )
    race = _CELL_RACE.race_many(currents, require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1:
        return _CandidateRace(
            decision="COLLISION",
            winner_index=None,
            spike_step=race.spike_step,
            simultaneous_spike_count=race.simultaneous_spike_candidates,
            physical_cell_steps=currents.shape[1] * RACE_STEPS,
        )
    if race.winner_cell == len(raw_scores):
        decision = "TONIC_STOP"
        winner_index = None
    else:
        decision = "WINNER"
        winner_index = race.winner_cell
    return _CandidateRace(
        decision=decision,
        winner_index=winner_index,
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        physical_cell_steps=currents.shape[1] * RACE_STEPS,
    )


def _publish_output(
    cells: projection.CommonEventCells,
    output_cells: tuple[int, ...] | None,
) -> projection.CommonOutputDecision:
    rows = [
        [0 for _digit in range(projection.OUTPUT_DIGIT_RADIX)]
        for _lane in range(projection.OUTPUT_DIGIT_LANES)
    ]
    if output_cells is not None:
        for lane, cell in enumerate(output_cells):
            local = cell - projection.OUTPUT_CELL_OFFSET
            observed_lane, digit = divmod(local, projection.OUTPUT_DIGIT_RADIX)
            if observed_lane != lane:
                raise RuntimeError("semantic output synapse topology changed")
            rows[lane][digit] = 2 * projection.STOP_SCORE
    return cells.race_output(tuple(tuple(row) for row in rows))


def _semantic_decision(
    cells: projection.CommonEventCells,
    key_cells: np.ndarray,
    output_cells: np.ndarray,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    semantic_count: int,
    active_key_cells: tuple[int, ...],
) -> _SemanticDecision:
    active = np.asarray(active_key_cells, dtype=np.int32)
    raw_scores = tuple(
        int(
            np.sum(
                np.where(
                    key_cells[row] == active,
                    excitatory[row].astype(np.int16)
                    - inhibitory[row].astype(np.int16),
                    0,
                ),
                dtype=np.int64,
            )
        )
        for row in range(semantic_count)
    )
    race = _candidate_race(raw_scores, stop_score=SEMANTIC_STOP_SCORE)
    if race.decision == "WINNER":
        winner = race.winner_index
        if winner is None:
            raise RuntimeError("semantic winner disappeared")
        output = _publish_output(
            cells,
            tuple(int(cell) for cell in output_cells[winner]),
        )
        status = "RECALLED"
        semantic_cell_id = len(ROLE_NAMES) * projection.TOTAL_CELL_COUNT + winner
    else:
        output = _publish_output(cells, None)
        semantic_cell_id = None
        if race.decision == "TONIC_STOP":
            status = "ABSTAIN_UNKNOWN"
        else:
            best = max(raw_scores, default=0)
            tied_candidates = sum(score == best for score in raw_scores)
            status = (
                "ABSTAIN_CONTRADICTION"
                if best >= SEMANTIC_STOP_SCORE and tied_candidates >= 2
                else "ABSTAIN_LOW_SUPPORT"
            )
    return _SemanticDecision(status, semantic_cell_id, output, raw_scores, race)


def _full_match(stored: np.ndarray, active: tuple[int, ...]) -> bool:
    return bool(np.count_nonzero(stored == np.asarray(active, dtype=stored.dtype)) == len(active))


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "common_projection_geometry": projection.GEOMETRY_SHA256,
    "common_associative_geometry": associative.GEOMETRY_SHA256,
    "roles": ROLE_NAMES,
    "event_active_cells": EVENT_ACTIVE_CELLS,
    "semantic_key_cells": SEMANTIC_KEY_CELLS,
    "maximum_semantic_neurons": MAX_SEMANTIC_NEURONS,
    "tick_dtype": TICK_DTYPE.str,
    "tick_cap": TICK_CAP,
    "positive_tick": POSITIVE_TICK,
    "negative_evidence_tick": NEGATIVE_EVIDENCE_TICK,
    "semantic_stop_score": SEMANTIC_STOP_SCORE,
    "semantic_competition": "fitted-glif-first-spike-with-tonic-abstention",
    "source_novelty": "private-exact-physical-context-detector",
    "raw_event_retention": False,
    "host_answer_ranking": False,
    "hash_or_name_as_neural_evidence": False,
    "domain_specific_computational_primitive_delta": 0,
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


class FrozenCommonEpisodicSemanticMemory:
    """Immutable episodic and semantic synapses behind two recall methods."""

    __slots__ = (
        "_cells",
        "_checkpoint_sha256",
        "_episodic",
        "_excitatory",
        "_inhibitory",
        "_key_cells",
        "_metadata",
        "_object_cells",
        "_output_cells",
        "_semantic_count",
        "_summary",
    )

    def __init__(
        self,
        *,
        episodic: associative.FrozenCommonEventAssociativeMemory,
        key_cells: np.ndarray,
        object_cells: np.ndarray,
        output_cells: np.ndarray,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        semantic_count: int,
        metadata: bytes,
        summary: ConsolidationTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("common episodic-semantic memory is builder-owned")
        expected = (
            (key_cells, (MAX_SEMANTIC_NEURONS, SEMANTIC_KEY_CELLS), np.dtype(np.int32)),
            (object_cells, (MAX_SEMANTIC_NEURONS, OBJECT_CELLS), np.dtype(np.int32)),
            (output_cells, (MAX_SEMANTIC_NEURONS, OUTPUT_SYNAPSES), np.dtype(np.int32)),
            (excitatory, (MAX_SEMANTIC_NEURONS, SEMANTIC_KEY_CELLS), TICK_DTYPE),
            (inhibitory, (MAX_SEMANTIC_NEURONS, SEMANTIC_KEY_CELLS), TICK_DTYPE),
        )
        if any(
            array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            for array, shape, dtype in expected
        ):
            raise ValueError("episodic-semantic synapse geometry changed")
        if not 0 <= semantic_count <= MAX_SEMANTIC_NEURONS:
            raise ValueError("semantic neuron count changed")
        self._cells = projection.CommonEventCells()
        self._episodic = episodic
        self._semantic_count = semantic_count
        self._metadata = bytes(metadata)
        self._summary = summary
        for name, array in (
            ("_key_cells", key_cells),
            ("_object_cells", object_cells),
            ("_output_cells", output_cells),
            ("_excitatory", excitatory),
            ("_inhibitory", inhibitory),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256()
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("episodic-semantic checkpoint identity changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("semantic-key-cells", self._key_cells),
            ("semantic-object-cells", self._object_cells),
            ("semantic-output-cells", self._output_cells),
            ("semantic-excitatory", self._excitatory),
            ("semantic-inhibitory", self._inhibitory),
        )

    def _calculate_checkpoint_sha256(self) -> str:
        return sha256(
            _checkpoint_payload(
                self._metadata,
                self._episodic.checkpoint_sha256,
                self._checkpoint_arrays(),
            )
        ).hexdigest()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> ConsolidationTrainingSummary:
        self.authenticate()
        return self._summary

    def authenticate(self) -> None:
        self._episodic.authenticate()
        if self._calculate_checkpoint_sha256() != self._checkpoint_sha256:
            raise RuntimeError("episodic-semantic checkpoint authentication failed")

    def _semantic_recall_cells(
        self,
        active_key_cells: tuple[int, ...],
    ) -> SemanticRecall:
        decision = _semantic_decision(
            self._cells,
            self._key_cells,
            self._output_cells,
            self._excitatory,
            self._inhibitory,
            self._semantic_count,
            active_key_cells,
        )
        record = {
            "status": decision.status,
            "semantic_cell_id": decision.semantic_cell_id,
            "output": decision.output.record(),
            "active_key_cells": active_key_cells,
            "candidate_raw_scores": decision.raw_scores,
            "candidate_spike_step": decision.race.spike_step,
            "candidate_simultaneous_spike_count": (
                decision.race.simultaneous_spike_count
            ),
            "physical_cell_steps": (
                decision.race.physical_cell_steps
                + decision.output.physical_cell_steps
            ),
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        return SemanticRecall(
            status=decision.status,
            semantic_cell_id=decision.semantic_cell_id,
            output=decision.output,
            active_key_cells=active_key_cells,
            candidate_raw_scores=decision.raw_scores,
            candidate_spike_step=decision.race.spike_step,
            candidate_simultaneous_spike_count=(
                decision.race.simultaneous_spike_count
            ),
            physical_cell_steps=(
                decision.race.physical_cell_steps
                + decision.output.physical_cell_steps
            ),
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )

    def recall_semantic(
        self,
        subject_event: bytes,
        relation_event: bytes,
        *,
        cell_gate: bool = True,
    ) -> SemanticRecall:
        """Recall the reusable relation, or physically abstain."""

        self.authenticate()
        if type(cell_gate) is not bool:
            raise ValueError("cell_gate must be an exact boolean")
        subject = _role_cells(self._cells, subject_event, 0)
        relation = _role_cells(self._cells, relation_event, 1)
        active = (
            (*subject, *relation)
            if cell_gate
            else tuple(-1 for _ in range(SEMANTIC_KEY_CELLS))
        )
        return self._semantic_recall_cells(active)

    def recall_episode(
        self,
        subject_event: bytes,
        relation_event: bytes,
        source_event: bytes,
    ) -> EpisodicRecall:
        """Recall one source-specific particular without changing semantic state."""

        self.authenticate()
        recalled = self._episodic.recall(
            (subject_event, relation_event, source_event)
        )
        source_cells = _role_cells(self._cells, source_event, 3)
        status = "RECALLED" if recalled.status == "RECALLED" else "STOP_UNKNOWN"
        record = {
            "status": status,
            "output": recalled.output.record(),
            "conditioned_source_cells": source_cells,
            "episodic_checkpoint_sha256": self._episodic.checkpoint_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        return EpisodicRecall(
            status=status,
            output=recalled.output,
            conditioned_source_cells=source_cells,
            episodic_checkpoint_sha256=self._episodic.checkpoint_sha256,
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def fit_common_episodic_semantic_consolidation(
    lessons: tuple[EpisodicSemanticLesson, ...],
) -> FrozenCommonEpisodicSemanticMemory:
    """Run one local self-posed fit and freeze episodic and semantic synapses."""

    if type(lessons) is not tuple or any(
        type(lesson) is not EpisodicSemanticLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of EpisodicSemanticLesson")
    cells = projection.CommonEventCells()
    key_cells = np.full(
        (MAX_SEMANTIC_NEURONS, SEMANTIC_KEY_CELLS), -1, dtype=np.int32
    )
    object_cells = np.full(
        (MAX_SEMANTIC_NEURONS, OBJECT_CELLS), -1, dtype=np.int32
    )
    output_cells = np.full(
        (MAX_SEMANTIC_NEURONS, OUTPUT_SYNAPSES), -1, dtype=np.int32
    )
    excitatory = np.zeros(
        (MAX_SEMANTIC_NEURONS, SEMANTIC_KEY_CELLS), dtype=TICK_DTYPE
    )
    inhibitory = np.zeros_like(excitatory)
    episodic_contexts: list[np.ndarray] = []
    episodic_lessons: list[associative.CommonAssociationLesson] = []
    steps: list[ConsolidationStep] = []
    semantic_count = 0
    duplicate_sources = 0
    excitatory_events = 0
    negative_events = 0

    for ordinal, lesson in enumerate(lessons):
        projected = _project_lesson(cells, lesson)
        if not lesson.teacher_gate:
            episodic_lessons.append(
                associative.CommonAssociationLesson(
                    observation_events=(
                        lesson.subject_event,
                        lesson.relation_event,
                        lesson.source_event,
                    ),
                    consequence_byte=lesson.object_event,
                    teacher_gate=False,
                )
            )
            steps.append(ConsolidationStep(ordinal, "TEACHER_OFF", None, None, 0))
            continue

        context = np.asarray(projected.episodic_context_cells, dtype=np.int32)
        if any(
            np.count_nonzero(stored == context) == EPISODIC_CONTEXT_CELLS
            for stored in episodic_contexts
        ):
            duplicate_sources += 1
            steps.append(
                ConsolidationStep(
                    ordinal,
                    "DUPLICATE_SOURCE_IGNORED",
                    None,
                    None,
                    0,
                )
            )
            continue
        episodic_contexts.append(context)
        episodic_lessons.append(
            associative.CommonAssociationLesson(
                observation_events=(
                    lesson.subject_event,
                    lesson.relation_event,
                    lesson.source_event,
                ),
                consequence_byte=lesson.object_event,
            )
        )

        matching_key_rows = tuple(
            row
            for row in range(semantic_count)
            if _full_match(key_cells[row], projected.key_cells)
        )
        matching_object_rows = tuple(
            row
            for row in matching_key_rows
            if _full_match(object_cells[row], projected.object_cells)
        )
        if len(matching_object_rows) > 1:
            raise RuntimeError("semantic private context detector duplicated")
        if matching_object_rows:
            target = matching_object_rows[0]
        else:
            if semantic_count == MAX_SEMANTIC_NEURONS:
                raise RuntimeError("semantic neuron capacity exhausted")
            target = semantic_count
            semantic_count += 1
            key_cells[target] = np.asarray(projected.key_cells, dtype=np.int32)
            object_cells[target] = np.asarray(projected.object_cells, dtype=np.int32)
            output_cells[target] = np.asarray(projected.output_cells, dtype=np.int32)

        previous = excitatory[target].astype(np.uint16)
        updated = np.minimum(previous + POSITIVE_TICK, TICK_CAP)
        excitatory[target] = updated.astype(TICK_DTYPE)
        excitatory_events += int(np.any(updated > previous))

        lesson_negative_events = 0
        for rival in matching_key_rows:
            if rival == target:
                continue
            previous = inhibitory[rival].astype(np.uint16)
            updated = np.minimum(previous + NEGATIVE_EVIDENCE_TICK, TICK_CAP)
            inhibitory[rival] = updated.astype(TICK_DTYPE)
            committed = int(np.any(updated > previous))
            lesson_negative_events += committed
            negative_events += committed

        decision = _semantic_decision(
            cells,
            key_cells=key_cells,
            output_cells=output_cells,
            excitatory=excitatory,
            inhibitory=inhibitory,
            semantic_count=semantic_count,
            active_key_cells=projected.key_cells,
        )
        if decision.status == "RECALLED":
            step_status = "REVISED" if lesson_negative_events else "CONSOLIDATED"
        elif decision.status == "ABSTAIN_CONTRADICTION":
            step_status = "CONTRADICTION_ABSTAIN"
        else:
            step_status = "SUPPORT_PENDING"
        steps.append(
            ConsolidationStep(
                lesson_ordinal=ordinal,
                status=step_status,
                semantic_cell_id=decision.semantic_cell_id,
                physical_output_token=decision.output.token,
                negative_evidence_events=lesson_negative_events,
            )
        )

    episodic = associative.fit_common_event_associations(tuple(episodic_lessons))
    metadata_record = {
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_sha256": CONFIG_SHA256,
        "lesson_count": len(lessons),
        "teacher_on_lesson_count": sum(lesson.teacher_gate for lesson in lessons),
        "teacher_off_lesson_count": sum(not lesson.teacher_gate for lesson in lessons),
        "novel_episode_count": len(episodic_contexts),
        "duplicate_source_count": duplicate_sources,
        "semantic_neuron_count": semantic_count,
        "episodic_checkpoint_sha256": episodic.checkpoint_sha256,
        "raw_event_retention": False,
        "generator_dev_m0_protected_corpus_inputs": (0, 0, 0, 0, 0),
    }
    metadata = _canonical_json(metadata_record)
    arrays = (
        ("semantic-key-cells", key_cells),
        ("semantic-object-cells", object_cells),
        ("semantic-output-cells", output_cells),
        ("semantic-excitatory", excitatory),
        ("semantic-inhibitory", inhibitory),
    )
    checkpoint_sha256 = sha256(
        _checkpoint_payload(metadata, episodic.checkpoint_sha256, arrays)
    ).hexdigest()
    summary = ConsolidationTrainingSummary(
        lesson_count=len(lessons),
        teacher_on_lesson_count=metadata_record["teacher_on_lesson_count"],
        teacher_off_lesson_count=metadata_record["teacher_off_lesson_count"],
        novel_episode_count=len(episodic_contexts),
        duplicate_source_count=duplicate_sources,
        semantic_neuron_count=semantic_count,
        excitatory_update_events=excitatory_events,
        negative_evidence_update_events=negative_events,
        consolidation_steps=tuple(steps),
        episodic_checkpoint_sha256=episodic.checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonEpisodicSemanticMemory(
        episodic=episodic,
        key_cells=key_cells,
        object_cells=object_cells,
        output_cells=output_cells,
        excitatory=excitatory,
        inhibitory=inhibitory,
        semantic_count=semantic_count,
        metadata=metadata,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "MAX_SEMANTIC_NEURONS",
    "ROLE_CELL_OFFSETS",
    "ROLE_NAMES",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "ConsolidationStep",
    "ConsolidationTrainingSummary",
    "EpisodicRecall",
    "EpisodicSemanticLesson",
    "FrozenCommonEpisodicSemanticMemory",
    "SemanticRecall",
    "fit_common_episodic_semantic_consolidation",
]
