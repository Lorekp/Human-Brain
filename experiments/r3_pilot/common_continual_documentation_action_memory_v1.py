"""Replay-free continual memory for documentation-conditioned opaque actions.

The module learns only lexical factors shared by an authenticated task projection
and an authenticated documentation-evidence projection.  Successful and failed
public outcomes add excitatory and inhibitory factor-to-action ticks.  Two
successful, independently rooted task/source observations are required before an
action can participate in recall.  Recall owns its action vocabulary and uses a
``CommonEventCells`` physical race; callers cannot order or rank candidates.

This is a mechanism module.  It acquires and reuses candidate-provided opaque
actions from lexical documentation factors.  It is not source synthesis,
semantic understanding, debugging, Rust execution, SWE evidence, or intelligence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from experiments.g0 import common_event_cells_v1 as common_events
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action

SCHEMA = "COMMON_CONTINUAL_DOCUMENTATION_ACTION_MEMORY_V1"
CLAIM_BOUNDARY = (
    "replay-free-candidate-provided-opaque-action-acquisition-and-reuse-from-"
    "exact-lexical-task-documentation-factor-overlap-with-two-independent-"
    "task-and-source-roots-physical-recall-and-rooted-public-feedback-not-"
    "source-synthesis-semantics-debugging-rust-execution-swe-or-intelligence"
)

RECALLED = "RECALLED"
STOP_CELL_OFF = "STOP_CONTINUAL_DOCUMENTATION_ACTION_CELL_OFF"
STOP_DOCUMENTATION_OFF = "STOP_CONTINUAL_DOCUMENTATION_ACTION_DOCUMENTATION_OFF"
STOP_NO_MATCHED_FACTOR = "STOP_CONTINUAL_DOCUMENTATION_ACTION_NO_MATCHED_FACTOR"
STOP_ACTION_UNPROMOTED = "STOP_CONTINUAL_DOCUMENTATION_ACTION_UNPROMOTED"
STOP_PHYSICAL_RACE = "STOP_CONTINUAL_DOCUMENTATION_ACTION_PHYSICAL_RACE"

MIN_PROMOTION_TASK_ROOTS = 2
MIN_PROMOTION_SOURCE_ROOTS = 2
MAX_ACTIONS = 4
MAX_FACTORS = 256
MAX_FACTORS_PER_EVIDENCE = 64
MAX_PROVENANCE_ROWS = 128
MAX_CASES = 32
MAX_VALUES_PER_CASE = 256
MAX_FACTOR_BYTES = 160
TICK_CAP = 255
TRACE_LANES = common_events.TRACE_LANES
OUTPUT_LANES = common_events.OUTPUT_DIGIT_LANES
OUTPUT_RADIX = common_events.OUTPUT_DIGIT_RADIX

CELL_DTYPE = np.dtype("<i4")
TICK_DTYPE = np.dtype(np.uint8)

_CELLS = common_events.CommonEventCells()
_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")

_EVIDENCE_NAMESPACE = b"continual-documentation-action-evidence-v1"
_LESSON_NAMESPACE = b"continual-documentation-action-lesson-v1"
_PROVENANCE_NAMESPACE = b"continual-documentation-action-provenance-v1"
_FACTOR_EVENT_NAMESPACE = b"continual-documentation-action-factor-v1"
_CHECKPOINT_NAMESPACE = b"continual-documentation-action-checkpoint-v1"
_CANDIDATE_SET_NAMESPACE = b"continual-documentation-action-candidate-set-v1"
_OUTPUT_NAMESPACE = b"continual-documentation-action-output-v1"
_RECALL_NAMESPACE = b"continual-documentation-action-recall-v1"


class ContinualDocumentationActionMemoryError(ValueError):
    """An evidence, lesson, checkpoint, or recall failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    frame = (
        array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json(array.shape)
        + b"\0"
        + array.tobytes(order="C")
    )
    return sha256(frame).hexdigest()


def _freeze_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _factor_event(factor: str) -> bytes:
    if type(factor) is not str or not factor or len(factor.encode("utf-8")) > MAX_FACTOR_BYTES:
        raise ContinualDocumentationActionMemoryError("documentation factor is invalid")
    value = _FACTOR_EVENT_NAMESPACE + b"\0" + factor.encode("utf-8")
    if len(value) > 256:
        raise ContinualDocumentationActionMemoryError("documentation factor event is too large")
    return value


def _factor_frame(factor: str) -> common_events.CommonEventFrame:
    return _CELLS.project(_factor_event(factor))


def _candidate_record(
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
) -> dict[str, str]:
    return {
        "candidate_id": candidate.candidate_id,
        "symbol": candidate.symbol,
        "source_sha256": candidate.source_sha256,
        "ast_sha256": candidate.ast_sha256,
        "sha256": candidate.sha256,
    }


def _validate_roots(values: object, *, label: str, nonempty: bool = True) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be one exact tuple")
    result = values
    if (
        (nonempty and not result)
        or result != tuple(sorted(set(result)))
        or any(not _valid_sha256(value) for value in result)
    ):
        raise ContinualDocumentationActionMemoryError(f"{label} is not canonical")
    return result


def _validate_outputs(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_CASES:
        raise TypeError(f"{label} must be one exact nonempty tuple")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not tuple or len(row) > MAX_VALUES_PER_CASE:
            raise TypeError(f"{label} rows must be exact bounded tuples")
        if any(type(item) is not int or not -(2**63) <= item < 2**63 for item in row):
            raise TypeError(f"{label} values must be exact signed 64-bit integers")
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class DocumentationActionEvidenceV1:
    factors: tuple[str, ...]
    source_sha256: str
    provenance_roots: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "factors": self.factors,
            "source_sha256": self.source_sha256,
            "provenance_roots": self.provenance_roots,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.factors) is not tuple:
            raise TypeError("documentation factors must be one exact tuple")
        if (
            not self.factors
            or len(self.factors) > MAX_FACTORS_PER_EVIDENCE
            or self.factors != tuple(sorted(set(self.factors)))
        ):
            raise ContinualDocumentationActionMemoryError("documentation factors are not canonical")
        for factor in self.factors:
            _factor_event(factor)
        roots = _validate_roots(
            self.provenance_roots,
            label="documentation provenance roots",
        )
        if (
            not _valid_sha256(self.source_sha256)
            or self.source_sha256 not in roots
            or self.sha256 != _digest(_EVIDENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualDocumentationActionMemoryError(
                "documentation evidence authentication failed"
            )


def make_documentation_action_evidence_v1(
    *,
    factors: tuple[str, ...],
    source_sha256: str,
    provenance_roots: tuple[str, ...],
) -> DocumentationActionEvidenceV1:
    if type(factors) is not tuple:
        raise TypeError("documentation factors must be one exact tuple")
    if type(provenance_roots) is not tuple:
        raise TypeError("documentation provenance roots must be one exact tuple")
    canonical_factors = tuple(sorted(set(factors)))
    canonical_roots = tuple(sorted({source_sha256, *provenance_roots}))
    partial = DocumentationActionEvidenceV1(
        factors=canonical_factors,
        source_sha256=source_sha256,
        provenance_roots=canonical_roots,
        sha256="",
    )
    result = DocumentationActionEvidenceV1(
        **{
            **partial.record(include_sha256=False),
            "sha256": _digest(
                _EVIDENCE_NAMESPACE,
                partial.record(include_sha256=False),
            ),
        }
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class ContinualDocumentationActionLessonV1:
    task: knowledge_action.KnowledgeActionTaskProjectionV1
    task_instance_sha256: str
    task_source_sha256: str
    evidence: DocumentationActionEvidenceV1
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    execution_sha256: str
    public_feedback_sha256: str
    observed_outputs: tuple[tuple[int, ...], ...]
    expected_outputs: tuple[tuple[int, ...], ...]
    passed_cell: int
    sha256: str

    @property
    def task_projection_sha256(self) -> str:
        return self.task.sha256

    @property
    def evidence_sha256(self) -> str:
        return self.evidence.sha256

    @property
    def documentation_source_sha256(self) -> str:
        return self.evidence.source_sha256

    @property
    def documentation_provenance_roots(self) -> tuple[str, ...]:
        return self.evidence.provenance_roots

    @property
    def matched_factors(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.task.factors).intersection(self.evidence.factors)))

    @property
    def observed_output_sha256(self) -> str:
        return _digest(_OUTPUT_NAMESPACE, self.observed_outputs)

    @property
    def expected_output_sha256(self) -> str:
        return _digest(_OUTPUT_NAMESPACE, self.expected_outputs)

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "task": {
                "factors": self.task.factors,
                "sha256": self.task.sha256,
            },
            "task_instance_sha256": self.task_instance_sha256,
            "task_source_sha256": self.task_source_sha256,
            "evidence": self.evidence.record(),
            "candidate": _candidate_record(self.candidate),
            "execution_sha256": self.execution_sha256,
            "public_feedback_sha256": self.public_feedback_sha256,
            "observed_outputs": self.observed_outputs,
            "expected_outputs": self.expected_outputs,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.task) is not knowledge_action.KnowledgeActionTaskProjectionV1:
            raise TypeError("lesson task must be one exact task projection")
        if type(self.evidence) is not DocumentationActionEvidenceV1:
            raise TypeError("lesson evidence must be one exact documentation evidence")
        if type(self.candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
            raise TypeError("lesson candidate must be one exact candidate projection")
        self.task.authenticate()
        self.evidence.authenticate()
        self.candidate.authenticate()
        observed = _validate_outputs(self.observed_outputs, label="observed outputs")
        expected = _validate_outputs(self.expected_outputs, label="expected outputs")
        if (
            not self.matched_factors
            or any(
                not _valid_sha256(value)
                for value in (
                    self.task_instance_sha256,
                    self.task_source_sha256,
                    self.execution_sha256,
                    self.public_feedback_sha256,
                )
            )
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.passed_cell != int(observed == expected)
            or self.sha256 != _digest(_LESSON_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualDocumentationActionMemoryError(
                "continual documentation action lesson authentication failed"
            )


def seal_continual_documentation_action_lesson_v1(
    *,
    task: knowledge_action.KnowledgeActionTaskProjectionV1,
    task_instance_sha256: str,
    task_source_sha256: str,
    evidence: DocumentationActionEvidenceV1,
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    execution_sha256: str,
    public_feedback_sha256: str,
    observed_outputs: tuple[tuple[int, ...], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
) -> ContinualDocumentationActionLessonV1:
    if type(task) is not knowledge_action.KnowledgeActionTaskProjectionV1:
        raise TypeError("lesson task must be one exact task projection")
    if type(evidence) is not DocumentationActionEvidenceV1:
        raise TypeError("lesson evidence must be one exact documentation evidence")
    if type(candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
        raise TypeError("lesson candidate must be one exact candidate projection")
    task.authenticate()
    evidence.authenticate()
    candidate.authenticate()
    observed = _validate_outputs(observed_outputs, label="observed outputs")
    expected = _validate_outputs(expected_outputs, label="expected outputs")
    matched = tuple(sorted(set(task.factors).intersection(evidence.factors)))
    if not matched:
        raise ContinualDocumentationActionMemoryError(
            "lesson has no task/documentation factor overlap"
        )
    partial = ContinualDocumentationActionLessonV1(
        task=task,
        task_instance_sha256=task_instance_sha256,
        task_source_sha256=task_source_sha256,
        evidence=evidence,
        candidate=candidate,
        execution_sha256=execution_sha256,
        public_feedback_sha256=public_feedback_sha256,
        observed_outputs=observed,
        expected_outputs=expected,
        passed_cell=int(observed == expected),
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_LESSON_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class ContinualDocumentationActionProvenanceV1:
    task_projection_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    evidence_sha256: str
    documentation_source_sha256: str
    matched_factors: tuple[str, ...]
    candidate_sha256: str
    execution_sha256: str
    public_feedback_sha256: str
    observed_output_sha256: str
    expected_output_sha256: str
    lesson_sha256: str
    passed_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return result

    def authenticate(self) -> None:
        values = self.record(include_sha256=False)
        if (
            type(self.matched_factors) is not tuple
            or not self.matched_factors
            or self.matched_factors != tuple(sorted(set(self.matched_factors)))
            or any(
                not _valid_sha256(value)
                for value in (
                    self.task_projection_sha256,
                    self.task_instance_sha256,
                    self.task_source_sha256,
                    self.evidence_sha256,
                    self.documentation_source_sha256,
                    self.candidate_sha256,
                    self.execution_sha256,
                    self.public_feedback_sha256,
                    self.observed_output_sha256,
                    self.expected_output_sha256,
                    self.lesson_sha256,
                )
            )
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.passed_cell != int(self.observed_output_sha256 == self.expected_output_sha256)
            or self.sha256 != _digest(_PROVENANCE_NAMESPACE, values)
        ):
            raise ContinualDocumentationActionMemoryError(
                "continual documentation provenance authentication failed"
            )
        for factor in self.matched_factors:
            _factor_event(factor)


@dataclass(frozen=True, slots=True)
class ContinualDocumentationActionTrainingSummaryV1:
    schema: str
    claim_boundary: str
    mode: str
    parent_checkpoint_sha256: str | None
    lesson_count: int
    new_lesson_count: int
    factor_count: int
    action_count: int
    promoted_action_count: int
    excitatory_update_count: int
    inhibitory_update_count: int
    learning_cell_steps: int
    cumulative_learning_cell_steps: int
    trainable_state_bytes: int
    checkpoint_array_bytes: int
    raw_task_text_count: int
    raw_output_count: int
    parent_lesson_replay_count: int
    parent_mutation_count: int
    plasticity_gate: bool
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ContinualDocumentationActionRecallWorkV1:
    query_projection_cell_steps: int
    factor_equality_cell_steps: int
    synaptic_reads: int
    output_race_cell_steps: int

    @property
    def total_cell_steps(self) -> int:
        return (
            self.query_projection_cell_steps
            + self.factor_equality_cell_steps
            + self.synaptic_reads
            + self.output_race_cell_steps
        )


def _physical_scores_from_candidate_scores(
    candidate_scores: tuple[tuple[int, str, int, int, int, int], ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], bool]:
    net_by_index = {row[0]: row[5] for row in candidate_scores}
    eligible_positive = tuple(row[0] for row in candidate_scores if row[2] == 1 and row[5] > 0)
    raw_scores = np.zeros((OUTPUT_LANES, OUTPUT_RADIX), dtype=np.int64)
    if eligible_positive:
        raw_scores[:-1, 0] = common_events.STOP_SCORE + 1
        support_levels = tuple(sorted({net_by_index[index] for index in eligible_positive}))
        for index in eligible_positive:
            support_rank = support_levels.index(net_by_index[index]) + 1
            raw_scores[-1, index] = common_events.STOP_SCORE + support_rank
    positive_scores = tuple(net_by_index[index] for index in eligible_positive)
    tied = bool(positive_scores and positive_scores.count(max(positive_scores)) > 1)
    return (
        tuple(tuple(int(value) for value in row) for row in raw_scores),
        eligible_positive,
        tied,
    )


@dataclass(frozen=True, slots=True)
class ContinualDocumentationActionRecallV1:
    status: str
    selected_action_index: int | None
    selected_candidate_id: str | None
    selected_candidate_sha256: str | None
    selected_source_sha256: str | None
    matched_factor_count: int
    unknown_factor_count: int
    candidates: tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...]
    cell_gate: bool
    documentation_gate: bool
    candidate_scores: tuple[tuple[int, str, int, int, int, int], ...]
    physical_raw_scores: tuple[tuple[int, ...], ...]
    physical_output_record: Mapping[str, object]
    work: ContinualDocumentationActionRecallWorkV1
    checkpoint_sha256: str
    task_sha256: str
    candidate_set_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "selected_action_index": self.selected_action_index,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "selected_source_sha256": self.selected_source_sha256,
            "matched_factor_count": self.matched_factor_count,
            "unknown_factor_count": self.unknown_factor_count,
            "candidates": tuple(_candidate_record(item) for item in self.candidates),
            "cell_gate": self.cell_gate,
            "documentation_gate": self.documentation_gate,
            "candidate_scores": self.candidate_scores,
            "physical_raw_scores": self.physical_raw_scores,
            "physical_output_record": dict(self.physical_output_record),
            "work": {
                **asdict(self.work),
                "total_cell_steps": self.work.total_cell_steps,
            },
            "checkpoint_sha256": self.checkpoint_sha256,
            "task_sha256": self.task_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        statuses = {
            RECALLED,
            STOP_CELL_OFF,
            STOP_DOCUMENTATION_OFF,
            STOP_NO_MATCHED_FACTOR,
            STOP_ACTION_UNPROMOTED,
            STOP_PHYSICAL_RACE,
        }
        if (
            self.status not in statuses
            or type(self.work) is not ContinualDocumentationActionRecallWorkV1
            or type(self.cell_gate) is not bool
            or type(self.documentation_gate) is not bool
            or type(self.candidates) is not tuple
            or not 1 <= len(self.candidates) <= MAX_ACTIONS
        ):
            raise ContinualDocumentationActionMemoryError("recall status or work changed")
        for candidate in self.candidates:
            if type(candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
                raise TypeError("recall candidates must be exact projections")
            candidate.authenticate()
        if (
            len({item.sha256 for item in self.candidates}) != len(self.candidates)
            or len({item.candidate_id for item in self.candidates}) != len(self.candidates)
            or len({item.symbol for item in self.candidates}) != len(self.candidates)
        ):
            raise ContinualDocumentationActionMemoryError("recall candidates changed")
        if any(type(value) is not int or value < 0 for value in asdict(self.work).values()):
            raise ContinualDocumentationActionMemoryError("recall work changed")
        if type(self.candidate_scores) is not tuple or len(self.candidate_scores) != len(
            self.candidates
        ):
            raise ContinualDocumentationActionMemoryError("recall candidate scores changed")
        for expected_index, (row, candidate) in enumerate(
            zip(self.candidate_scores, self.candidates, strict=True)
        ):
            if (
                type(row) is not tuple
                or len(row) != 6
                or type(row[0]) is not int
                or row[0] != expected_index
                or row[1] != candidate.sha256
                or type(row[2]) is not int
                or row[2] not in (0, 1)
                or any(type(value) is not int or value < 0 for value in row[3:])
                or row[5] != max(0, row[3] - row[4])
            ):
                raise ContinualDocumentationActionMemoryError("recall candidate scores changed")
        expected_raw_scores, eligible_positive, tied = _physical_scores_from_candidate_scores(
            self.candidate_scores
        )
        if (
            type(self.physical_raw_scores) is not tuple
            or len(self.physical_raw_scores) != OUTPUT_LANES
            or any(
                type(row) is not tuple or len(row) != OUTPUT_RADIX
                for row in self.physical_raw_scores
            )
            or self.physical_raw_scores != expected_raw_scores
        ):
            raise ContinualDocumentationActionMemoryError("recall physical scores changed")
        output = _CELLS.race_output(self.physical_raw_scores)
        if dict(output.record()) != dict(self.physical_output_record):
            raise ContinualDocumentationActionMemoryError("recall physical output changed")
        any_positive = any(row[5] > 0 for row in self.candidate_scores)
        if not self.cell_gate:
            expected_status = STOP_CELL_OFF
        elif not self.documentation_gate:
            expected_status = STOP_DOCUMENTATION_OFF
        elif self.matched_factor_count == 0:
            expected_status = STOP_NO_MATCHED_FACTOR
        elif any_positive and not eligible_positive:
            expected_status = STOP_ACTION_UNPROMOTED
        elif (
            not eligible_positive
            or tied
            or output.decision != "TOKEN"
            or output.token not in eligible_positive
        ):
            expected_status = STOP_PHYSICAL_RACE
        else:
            expected_status = RECALLED
        if self.status != expected_status:
            raise ContinualDocumentationActionMemoryError("recall status changed")
        selected = (
            self.selected_action_index,
            self.selected_candidate_id,
            self.selected_candidate_sha256,
            self.selected_source_sha256,
        )
        if self.status == RECALLED:
            selected_candidate = self.candidates[output.token]
            if (
                type(self.selected_action_index) is not int
                or output.decision != "TOKEN"
                or output.token != self.selected_action_index
                or selected
                != (
                    output.token,
                    selected_candidate.candidate_id,
                    selected_candidate.sha256,
                    selected_candidate.source_sha256,
                )
            ):
                raise ContinualDocumentationActionMemoryError("recalled selection changed")
        elif any(value is not None for value in selected) or output.decision != "STOP":
            raise ContinualDocumentationActionMemoryError("stopped recall selected an action")
        if (
            type(self.matched_factor_count) is not int
            or self.matched_factor_count < 0
            or type(self.unknown_factor_count) is not int
            or self.unknown_factor_count < 0
            or self.matched_factor_count + self.unknown_factor_count < 1
            or (
                (
                    not self.cell_gate
                    or not self.documentation_gate
                    or self.matched_factor_count == 0
                )
                and any(any(value != 0 for value in row[3:]) for row in self.candidate_scores)
            )
            or any(
                not _valid_sha256(value)
                for value in (
                    self.checkpoint_sha256,
                    self.task_sha256,
                    self.candidate_set_sha256,
                )
            )
            or self.candidate_set_sha256 != _candidate_set_sha256(self.candidates)
            or self.sha256 != _digest(_RECALL_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualDocumentationActionMemoryError("recall authentication failed")


def _provenance_from_lesson(
    lesson: ContinualDocumentationActionLessonV1,
) -> ContinualDocumentationActionProvenanceV1:
    payload = {
        "task_projection_sha256": lesson.task_projection_sha256,
        "task_instance_sha256": lesson.task_instance_sha256,
        "task_source_sha256": lesson.task_source_sha256,
        "evidence_sha256": lesson.evidence_sha256,
        "documentation_source_sha256": lesson.documentation_source_sha256,
        "matched_factors": lesson.matched_factors,
        "candidate_sha256": lesson.candidate.sha256,
        "execution_sha256": lesson.execution_sha256,
        "public_feedback_sha256": lesson.public_feedback_sha256,
        "observed_output_sha256": lesson.observed_output_sha256,
        "expected_output_sha256": lesson.expected_output_sha256,
        "lesson_sha256": lesson.sha256,
        "passed_cell": lesson.passed_cell,
    }
    result = ContinualDocumentationActionProvenanceV1(
        **payload,
        sha256=_digest(_PROVENANCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _candidate_set_sha256(
    candidates: tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
) -> str:
    return _digest(_CANDIDATE_SET_NAMESPACE, tuple(_candidate_record(item) for item in candidates))


def _promoted_indices(
    candidates: tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
    provenance: tuple[ContinualDocumentationActionProvenanceV1, ...],
) -> frozenset[int]:
    promoted: set[int] = set()
    for index, candidate in enumerate(candidates):
        successes = tuple(
            row
            for row in provenance
            if row.candidate_sha256 == candidate.sha256 and row.passed_cell == 1
        )
        if (
            len({row.task_projection_sha256 for row in successes}) >= MIN_PROMOTION_TASK_ROOTS
            and len({row.task_source_sha256 for row in successes}) >= MIN_PROMOTION_SOURCE_ROOTS
        ):
            promoted.add(index)
    return frozenset(promoted)


def _checkpoint_sha256(
    *,
    factor_names: tuple[str, ...],
    candidates: tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
    provenance: tuple[ContinualDocumentationActionProvenanceV1, ...],
    factor_cells: np.ndarray,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    summary: ContinualDocumentationActionTrainingSummaryV1,
) -> str:
    summary_record = asdict(summary)
    summary_record["checkpoint_sha256"] = ""
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "factor_names": factor_names,
            "candidates": tuple(_candidate_record(item) for item in candidates),
            "provenance": tuple(row.record() for row in provenance),
            "factor_cells_sha256": _array_sha256(factor_cells),
            "excitatory_sha256": _array_sha256(excitatory),
            "inhibitory_sha256": _array_sha256(inhibitory),
            "summary": summary_record,
        },
    )


class FrozenContinualDocumentationActionMemoryV1:
    """One flat immutable factor/action checkpoint with rooted outcomes."""

    __slots__ = (
        "_candidates",
        "_excitatory",
        "_factor_cells",
        "_factor_names",
        "_inhibitory",
        "_provenance",
        "_summary",
    )

    def __init__(
        self,
        *,
        factor_names: tuple[str, ...],
        candidates: tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
        provenance: tuple[ContinualDocumentationActionProvenanceV1, ...],
        factor_cells: np.ndarray,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: ContinualDocumentationActionTrainingSummaryV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("continual documentation memories are factory-owned")
        self._factor_names = factor_names
        self._candidates = candidates
        self._provenance = provenance
        self._factor_cells = _freeze_array(factor_cells, CELL_DTYPE)
        self._excitatory = _freeze_array(excitatory, TICK_DTYPE)
        self._inhibitory = _freeze_array(inhibitory, TICK_DTYPE)
        self._summary = summary
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._summary.checkpoint_sha256

    @property
    def training_summary(self) -> ContinualDocumentationActionTrainingSummaryV1:
        self.authenticate()
        return self._summary

    @property
    def candidate_projections(
        self,
    ) -> tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...]:
        self.authenticate()
        return self._candidates

    @property
    def provenance(self) -> tuple[ContinualDocumentationActionProvenanceV1, ...]:
        self.authenticate()
        return self._provenance

    @property
    def promoted_candidate_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        promoted = _promoted_indices(self._candidates, self._provenance)
        return tuple(self._candidates[index].sha256 for index in sorted(promoted))

    @property
    def checkpoint_array_sha256s(self) -> tuple[str, str, str]:
        self.authenticate()
        return tuple(_array_sha256(value) for value in self._arrays())

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._factor_cells, self._excitatory, self._inhibitory

    def authenticate(self) -> None:
        if (
            type(self._factor_names) is not tuple
            or not self._factor_names
            or len(self._factor_names) > MAX_FACTORS
            or len(set(self._factor_names)) != len(self._factor_names)
            or type(self._candidates) is not tuple
            or not 1 <= len(self._candidates) <= MAX_ACTIONS
            or len({item.sha256 for item in self._candidates}) != len(self._candidates)
            or len({item.candidate_id for item in self._candidates}) != len(self._candidates)
            or len({item.symbol for item in self._candidates}) != len(self._candidates)
            or type(self._provenance) is not tuple
            or not 1 <= len(self._provenance) <= MAX_PROVENANCE_ROWS
            or len({row.lesson_sha256 for row in self._provenance}) != len(self._provenance)
            or len({row.task_instance_sha256 for row in self._provenance}) != len(self._provenance)
        ):
            raise ContinualDocumentationActionMemoryError("checkpoint catalogue changed")
        for factor in self._factor_names:
            _factor_event(factor)
        for candidate in self._candidates:
            if type(candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
                raise TypeError("checkpoint candidates must be exact projections")
            candidate.authenticate()
        for row in self._provenance:
            if type(row) is not ContinualDocumentationActionProvenanceV1:
                raise TypeError("checkpoint provenance rows must be exact")
            row.authenticate()
        candidate_roots = {item.sha256 for item in self._candidates}
        if any(row.candidate_sha256 not in candidate_roots for row in self._provenance):
            raise ContinualDocumentationActionMemoryError("provenance action is unknown")
        factor_rows = {factor: index for index, factor in enumerate(self._factor_names)}
        candidate_rows = {
            candidate.sha256: index for index, candidate in enumerate(self._candidates)
        }
        expected_excitatory = np.zeros(
            (len(self._factor_names), len(self._candidates)), dtype=TICK_DTYPE
        )
        expected_inhibitory = np.zeros_like(expected_excitatory)
        for row in self._provenance:
            target = expected_excitatory if row.passed_cell == 1 else expected_inhibitory
            action_index = candidate_rows[row.candidate_sha256]
            for factor in row.matched_factors:
                factor_index = factor_rows.get(factor)
                if factor_index is None:
                    raise ContinualDocumentationActionMemoryError("provenance factor is unknown")
                target[factor_index, action_index] = np.uint8(
                    min(TICK_CAP, int(target[factor_index, action_index]) + 1)
                )
        expected_cells = np.asarray(
            tuple(_factor_frame(factor).trace_cells for factor in self._factor_names),
            dtype=CELL_DTYPE,
        )
        shape = (len(self._factor_names), len(self._candidates))
        arrays_valid = (
            self._factor_cells.dtype == CELL_DTYPE
            and self._factor_cells.shape == (len(self._factor_names), TRACE_LANES)
            and np.array_equal(self._factor_cells, expected_cells)
            and self._excitatory.dtype == TICK_DTYPE
            and self._excitatory.shape == shape
            and np.array_equal(self._excitatory, expected_excitatory)
            and self._inhibitory.dtype == TICK_DTYPE
            and self._inhibitory.shape == shape
            and np.array_equal(self._inhibitory, expected_inhibitory)
            and all(not value.flags.writeable for value in self._arrays())
        )
        promoted = _promoted_indices(self._candidates, self._provenance)
        expected_checkpoint = _checkpoint_sha256(
            factor_names=self._factor_names,
            candidates=self._candidates,
            provenance=self._provenance,
            factor_cells=self._factor_cells,
            excitatory=self._excitatory,
            inhibitory=self._inhibitory,
            summary=self._summary,
        )
        summary_valid = (
            type(self._summary) is ContinualDocumentationActionTrainingSummaryV1
            and self._summary.schema == SCHEMA
            and self._summary.claim_boundary == CLAIM_BOUNDARY
            and self._summary.lesson_count == len(self._provenance)
            and self._summary.factor_count == len(self._factor_names)
            and self._summary.action_count == len(self._candidates)
            and self._summary.promoted_action_count == len(promoted)
            and self._summary.trainable_state_bytes
            == self._excitatory.nbytes + self._inhibitory.nbytes
            and self._summary.checkpoint_array_bytes
            == sum(value.nbytes for value in self._arrays())
            and self._summary.raw_task_text_count == 0
            and self._summary.raw_output_count == 0
            and self._summary.parent_lesson_replay_count == 0
            and self._summary.parent_mutation_count == 0
            and self._summary.plasticity_gate is True
            and self._summary.checkpoint_sha256 == expected_checkpoint
        )
        if not arrays_valid or not summary_valid:
            raise ContinualDocumentationActionMemoryError("checkpoint authentication failed")

    def recall(
        self,
        task: knowledge_action.KnowledgeActionTaskProjectionV1,
        *,
        cell_gate: bool = True,
        documentation_gate: bool = True,
    ) -> ContinualDocumentationActionRecallV1:
        self.authenticate()
        if type(task) is not knowledge_action.KnowledgeActionTaskProjectionV1:
            raise TypeError("recall task must be one exact task projection")
        if type(cell_gate) is not bool or type(documentation_gate) is not bool:
            raise TypeError("recall gates must be exact booleans")
        task.authenticate()

        query_frames = tuple(_factor_frame(factor) for factor in task.factors)
        query_steps = sum(frame.work.total for frame in query_frames)
        matched: list[int] = []
        equality_steps = 0
        for frame in query_frames:
            equality_steps += len(self._factor_names) * TRACE_LANES
            equal = np.flatnonzero(
                np.all(
                    self._factor_cells == np.asarray(frame.trace_cells, dtype=CELL_DTYPE),
                    axis=1,
                )
            )
            if equal.size == 1:
                matched.append(int(equal[0]))
            elif equal.size > 1:
                raise ContinualDocumentationActionMemoryError("factor projection became ambiguous")
        matched = sorted(set(matched))
        promoted = _promoted_indices(self._candidates, self._provenance)
        excitatory = np.zeros((len(self._candidates),), dtype=np.int64)
        inhibitory = np.zeros((len(self._candidates),), dtype=np.int64)
        synaptic_reads = 0
        if cell_gate and documentation_gate and matched:
            synaptic_reads = len(matched) * len(self._candidates) * 2
            excitatory = np.sum(self._excitatory[matched, :], axis=0, dtype=np.int64)
            inhibitory = np.sum(self._inhibitory[matched, :], axis=0, dtype=np.int64)
        net = np.maximum(0, excitatory - inhibitory)

        candidate_scores = tuple(
            (
                index,
                candidate.sha256,
                int(index in promoted),
                int(excitatory[index]),
                int(inhibitory[index]),
                int(net[index]),
            )
            for index, candidate in enumerate(self._candidates)
        )
        physical_raw_scores, eligible_positive, tied = _physical_scores_from_candidate_scores(
            candidate_scores
        )
        output = _CELLS.race_output(physical_raw_scores)

        selected_index: int | None = None
        if not cell_gate:
            status = STOP_CELL_OFF
        elif not documentation_gate:
            status = STOP_DOCUMENTATION_OFF
        elif not matched:
            status = STOP_NO_MATCHED_FACTOR
        elif (
            any(int(net[index]) > 0 for index in range(len(self._candidates)))
            and not eligible_positive
        ):
            status = STOP_ACTION_UNPROMOTED
        elif not eligible_positive or (
            tied or output.decision != "TOKEN" or output.token not in eligible_positive
        ):
            status = STOP_PHYSICAL_RACE
        else:
            status = RECALLED
            selected_index = output.token

        selected = self._candidates[selected_index] if selected_index is not None else None
        work = ContinualDocumentationActionRecallWorkV1(
            query_projection_cell_steps=query_steps,
            factor_equality_cell_steps=equality_steps,
            synaptic_reads=synaptic_reads,
            output_race_cell_steps=output.physical_cell_steps,
        )
        partial = ContinualDocumentationActionRecallV1(
            status=status,
            selected_action_index=selected_index,
            selected_candidate_id=None if selected is None else selected.candidate_id,
            selected_candidate_sha256=None if selected is None else selected.sha256,
            selected_source_sha256=None if selected is None else selected.source_sha256,
            matched_factor_count=len(matched),
            unknown_factor_count=len(task.factors) - len(matched),
            candidates=self._candidates,
            cell_gate=cell_gate,
            documentation_gate=documentation_gate,
            candidate_scores=candidate_scores,
            physical_raw_scores=physical_raw_scores,
            physical_output_record=MappingProxyType(dict(output.record())),
            work=work,
            checkpoint_sha256=self._summary.checkpoint_sha256,
            task_sha256=task.sha256,
            candidate_set_sha256=_candidate_set_sha256(self._candidates),
            sha256="",
        )
        result = replace(
            partial,
            sha256=_digest(_RECALL_NAMESPACE, partial.record(include_sha256=False)),
        )
        result.authenticate()
        self.authenticate()
        return result


def _apply_lesson(
    parent: FrozenContinualDocumentationActionMemoryV1 | None,
    lesson: ContinualDocumentationActionLessonV1,
) -> tuple[
    tuple[str, ...],
    tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
    tuple[ContinualDocumentationActionProvenanceV1, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    lesson.authenticate()
    if parent is None:
        factor_names: list[str] = []
        candidates: list[knowledge_action.KnowledgeActionCandidateProjectionV1] = []
        provenance: list[ContinualDocumentationActionProvenanceV1] = []
        factor_cells = np.empty((0, TRACE_LANES), dtype=CELL_DTYPE)
        excitatory = np.empty((0, 0), dtype=TICK_DTYPE)
        inhibitory = np.empty((0, 0), dtype=TICK_DTYPE)
        cumulative_steps = 0
    else:
        parent.authenticate()
        factor_names = list(parent._factor_names)
        candidates = list(parent._candidates)
        provenance = list(parent._provenance)
        factor_cells = np.asarray(parent._factor_cells).copy()
        excitatory = np.asarray(parent._excitatory).copy()
        inhibitory = np.asarray(parent._inhibitory).copy()
        cumulative_steps = parent._summary.cumulative_learning_cell_steps
        if any(row.lesson_sha256 == lesson.sha256 for row in provenance):
            raise ContinualDocumentationActionMemoryError("lesson replay is forbidden")
        if any(row.task_instance_sha256 == lesson.task_instance_sha256 for row in provenance):
            raise ContinualDocumentationActionMemoryError("task instance replay is forbidden")

    action_index = next(
        (
            index
            for index, value in enumerate(candidates)
            if value.sha256 == lesson.candidate.sha256
        ),
        None,
    )
    if action_index is None:
        if any(
            value.candidate_id == lesson.candidate.candidate_id
            or value.symbol == lesson.candidate.symbol
            for value in candidates
        ):
            raise ContinualDocumentationActionMemoryError("candidate id or symbol collision")
        if lesson.passed_cell != 1:
            raise ContinualDocumentationActionMemoryError(
                "a failed lesson cannot introduce an action"
            )
        if len(candidates) >= MAX_ACTIONS:
            raise ContinualDocumentationActionMemoryError("action capacity exceeded")
        action_index = len(candidates)
        candidates.append(lesson.candidate)
        excitatory = np.pad(excitatory, ((0, 0), (0, 1)))
        inhibitory = np.pad(inhibitory, ((0, 0), (0, 1)))

    factor_index = {value: index for index, value in enumerate(factor_names)}
    new_frames: list[common_events.CommonEventFrame] = []
    for factor in lesson.matched_factors:
        if factor in factor_index:
            continue
        if len(factor_names) >= MAX_FACTORS:
            raise ContinualDocumentationActionMemoryError("factor capacity exceeded")
        frame = _factor_frame(factor)
        factor_index[factor] = len(factor_names)
        factor_names.append(factor)
        new_frames.append(frame)
        factor_cells = np.concatenate(
            (factor_cells, np.asarray(frame.trace_cells, dtype=CELL_DTYPE)[None, :]),
            axis=0,
        )
        excitatory = np.pad(excitatory, ((0, 1), (0, 0)))
        inhibitory = np.pad(inhibitory, ((0, 1), (0, 0)))

    target = excitatory if lesson.passed_cell == 1 else inhibitory
    updates = 0
    for factor in lesson.matched_factors:
        row = factor_index[factor]
        target[row, action_index] = np.uint8(min(TICK_CAP, int(target[row, action_index]) + 1))
        updates += 1
    provenance.append(_provenance_from_lesson(lesson))
    if len(provenance) > MAX_PROVENANCE_ROWS:
        raise ContinualDocumentationActionMemoryError("provenance capacity exceeded")
    learning_steps = sum(frame.work.total for frame in new_frames) + updates
    return (
        tuple(factor_names),
        tuple(candidates),
        tuple(provenance),
        factor_cells,
        excitatory,
        inhibitory,
        learning_steps,
        cumulative_steps + learning_steps,
        updates,
    )


def _freeze_memory(
    values: tuple[
        tuple[str, ...],
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, ...],
        tuple[ContinualDocumentationActionProvenanceV1, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        int,
    ],
    *,
    parent_checkpoint_sha256: str | None,
    mode: str,
    passed_cell: int,
) -> FrozenContinualDocumentationActionMemoryV1:
    (
        factor_names,
        candidates,
        provenance,
        factor_cells,
        excitatory,
        inhibitory,
        learning_steps,
        cumulative_steps,
        updates,
    ) = values
    factor_cells = _freeze_array(factor_cells, CELL_DTYPE)
    excitatory = _freeze_array(excitatory, TICK_DTYPE)
    inhibitory = _freeze_array(inhibitory, TICK_DTYPE)
    promoted = _promoted_indices(candidates, provenance)
    summary = ContinualDocumentationActionTrainingSummaryV1(
        schema=SCHEMA,
        claim_boundary=CLAIM_BOUNDARY,
        mode=mode,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        lesson_count=len(provenance),
        new_lesson_count=1,
        factor_count=len(factor_names),
        action_count=len(candidates),
        promoted_action_count=len(promoted),
        excitatory_update_count=updates if passed_cell == 1 else 0,
        inhibitory_update_count=updates if passed_cell == 0 else 0,
        learning_cell_steps=learning_steps,
        cumulative_learning_cell_steps=cumulative_steps,
        trainable_state_bytes=excitatory.nbytes + inhibitory.nbytes,
        checkpoint_array_bytes=factor_cells.nbytes + excitatory.nbytes + inhibitory.nbytes,
        raw_task_text_count=0,
        raw_output_count=0,
        parent_lesson_replay_count=0,
        parent_mutation_count=0,
        plasticity_gate=True,
        checkpoint_sha256="",
    )
    checkpoint = _checkpoint_sha256(
        factor_names=factor_names,
        candidates=candidates,
        provenance=provenance,
        factor_cells=factor_cells,
        excitatory=excitatory,
        inhibitory=inhibitory,
        summary=summary,
    )
    summary = ContinualDocumentationActionTrainingSummaryV1(
        **{**asdict(summary), "checkpoint_sha256": checkpoint}
    )
    return FrozenContinualDocumentationActionMemoryV1(
        factor_names=factor_names,
        candidates=candidates,
        provenance=provenance,
        factor_cells=factor_cells,
        excitatory=excitatory,
        inhibitory=inhibitory,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_continual_documentation_action_memory_v1(
    lesson: ContinualDocumentationActionLessonV1,
    *,
    plasticity_gate: bool = True,
) -> FrozenContinualDocumentationActionMemoryV1:
    if type(lesson) is not ContinualDocumentationActionLessonV1:
        raise TypeError("start requires one exact continual documentation lesson")
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    lesson.authenticate()
    if not plasticity_gate:
        raise ContinualDocumentationActionMemoryError(
            "an empty continual memory cannot start with plasticity disabled"
        )
    if lesson.passed_cell != 1:
        raise ContinualDocumentationActionMemoryError(
            "an empty continual memory must start from a successful lesson"
        )
    return _freeze_memory(
        _apply_lesson(None, lesson),
        parent_checkpoint_sha256=None,
        mode="INITIAL_CONTINUAL_DOCUMENTATION_ACTION_MEMORY",
        passed_cell=lesson.passed_cell,
    )


def advance_continual_documentation_action_memory_v1(
    parent: FrozenContinualDocumentationActionMemoryV1,
    lesson: ContinualDocumentationActionLessonV1,
    *,
    plasticity_gate: bool = True,
) -> FrozenContinualDocumentationActionMemoryV1:
    if type(parent) is not FrozenContinualDocumentationActionMemoryV1:
        raise TypeError("advance requires one exact continual documentation memory")
    if type(lesson) is not ContinualDocumentationActionLessonV1:
        raise TypeError("advance requires one exact continual documentation lesson")
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    parent.authenticate()
    lesson.authenticate()
    parent_root = parent.checkpoint_sha256
    parent_arrays = parent.checkpoint_array_sha256s
    if not plasticity_gate:
        return parent
    successor = _freeze_memory(
        _apply_lesson(parent, lesson),
        parent_checkpoint_sha256=parent_root,
        mode="REPLAY_FREE_CONTINUAL_DOCUMENTATION_ACTION_ADVANCE",
        passed_cell=lesson.passed_cell,
    )
    parent.authenticate()
    if parent.checkpoint_sha256 != parent_root or parent.checkpoint_array_sha256s != parent_arrays:
        raise RuntimeError("continual documentation advance mutated its parent")
    return successor


__all__ = [
    "CLAIM_BOUNDARY",
    "MAX_ACTIONS",
    "MAX_FACTORS",
    "MIN_PROMOTION_SOURCE_ROOTS",
    "MIN_PROMOTION_TASK_ROOTS",
    "RECALLED",
    "SCHEMA",
    "STOP_ACTION_UNPROMOTED",
    "STOP_CELL_OFF",
    "STOP_DOCUMENTATION_OFF",
    "STOP_NO_MATCHED_FACTOR",
    "STOP_PHYSICAL_RACE",
    "ContinualDocumentationActionLessonV1",
    "ContinualDocumentationActionMemoryError",
    "ContinualDocumentationActionProvenanceV1",
    "ContinualDocumentationActionRecallV1",
    "ContinualDocumentationActionRecallWorkV1",
    "ContinualDocumentationActionTrainingSummaryV1",
    "DocumentationActionEvidenceV1",
    "FrozenContinualDocumentationActionMemoryV1",
    "advance_continual_documentation_action_memory_v1",
    "make_documentation_action_evidence_v1",
    "seal_continual_documentation_action_lesson_v1",
    "start_continual_documentation_action_memory_v1",
]
