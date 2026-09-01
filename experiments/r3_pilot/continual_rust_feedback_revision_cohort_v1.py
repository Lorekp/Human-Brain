"""Public-development cohort for live Rust consequence revision.

Five selected-only Rust executions create a fresh continual lexical checkpoint.
One later task first recalls the wrong promoted action, learns from its real
authenticated failure, reraces the revised checkpoint, and executes the other
action successfully.  Two evaluation actions execute before a zero-argument
oracle callback is invoked once.

The actions are fixed candidate-provided sources.  This public-development
module demonstrates only authenticated consequence revision with exact lexical
factor reuse.  Its retained transcript is an external audit record and is never
fed back into learning.  It does not establish protected freshness, synthesize
Rust, learn source semantics, establish a hostile sandbox, solve an SWE
benchmark, or provide evidence of general intelligence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_promoted_rust_action_vault_v1 as promoted_vault
from experiments.r3_pilot import common_rust_action_intervention_v1 as intervention
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.continual-rust-feedback-revision-cohort.v1"
CLAIM_BOUNDARY = (
    "public-development-candidate-provided-actions-with-authenticated-consequence-"
    "revision-under-exact-lexical-factor-reuse-and-an-external-audit-transcript-"
    "with-two-selected-only-evaluations-complete-before-one-oracle-callback-"
    "invocation-not-protected-freshness-source-synthesis-semantic-understanding-"
    "debugging-hostile-sandboxing-swe-or-intelligence"
)

PASS = "PASS_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_V1"
FAIL_ORACLE_MISMATCH = "FAIL_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_V1_ORACLE_MISMATCH"
STOP_CELL_OFF = "STOP_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_CELL_OFF"
STOP_DOCUMENTATION_OFF = "STOP_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_DOCUMENTATION_OFF"

TRAIN = "TRAIN"
ADAPTATION_FAIL = "ADAPTATION_FAIL"
ADAPTATION_PASS = "ADAPTATION_PASS"
EVALUATION = "EVALUATION"

_SPEC_NAMESPACE = b"continual-rust-feedback-revision-spec-v1"
_OUTPUT_NAMESPACE = b"continual-rust-feedback-revision-output-v1"
_TASK_INSTANCE_NAMESPACE = b"continual-rust-feedback-revision-task-instance-v1"
_TASK_SOURCE_NAMESPACE = b"continual-rust-feedback-revision-task-source-v1"
_DOCUMENTATION_SOURCE_NAMESPACE = b"continual-rust-feedback-revision-documentation-source-v1"
_INTERVENTION_NAMESPACE = b"continual-rust-feedback-revision-intervention-v1"
_PUBLIC_FEEDBACK_NAMESPACE = b"continual-rust-feedback-revision-feedback-v1"
_ATTEMPT_NAMESPACE = b"continual-rust-feedback-revision-attempt-v1"
_WORK_NAMESPACE = b"continual-rust-feedback-revision-work-v1"
_ORACLE_NAMESPACE = b"continual-rust-feedback-revision-oracle-v1"
_RESULT_NAMESPACE = b"continual-rust-feedback-revision-result-v1"

_SHA256_HEX = frozenset("0123456789abcdef")


class ContinualRustFeedbackRevisionCohortError(ValueError):
    """The cohort specification, custody chain, or result failed authentication."""


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


def _validate_outputs(
    value: object,
    *,
    label: str,
    expected_cases: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be one exact nonempty tuple")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError(f"{label} rows must be exact tuples")
        if any(type(item) is not int or not -(2**63) <= item < 2**63 for item in row):
            raise TypeError(f"{label} values must be exact signed 64-bit integers")
        rows.append(row)
    result = tuple(rows)
    if expected_cases is not None and len(result) != expected_cases:
        raise ContinualRustFeedbackRevisionCohortError(f"{label} case count changed")
    return result


def feedback_revision_output_sha256_v1(
    outputs: tuple[tuple[int, ...], ...],
) -> str:
    """Return the cohort's public commitment for one exact output batch."""

    checked = _validate_outputs(outputs, label="public expected outputs")
    return _digest(_OUTPUT_NAMESPACE, checked)


@dataclass(frozen=True, slots=True)
class FeedbackRevisionCohortSpecV1:
    schema: str
    cohort_nonce_sha256: str
    training_batches: tuple[tuple[tuple[int, ...], ...], ...]
    training_expected_outputs: tuple[tuple[tuple[int, ...], ...], ...]
    adaptation_batch: tuple[tuple[int, ...], ...]
    adaptation_expected_outputs: tuple[tuple[int, ...], ...]
    evaluation_batches: tuple[tuple[tuple[int, ...], ...], ...]
    evaluation_expected_output_sha256s: tuple[str, str]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "cohort_nonce_sha256": self.cohort_nonce_sha256,
            "training_batches": self.training_batches,
            "training_expected_outputs": self.training_expected_outputs,
            "adaptation_batch": self.adaptation_batch,
            "adaptation_expected_outputs": self.adaptation_expected_outputs,
            "evaluation_batches": self.evaluation_batches,
            "evaluation_expected_output_sha256s": (self.evaluation_expected_output_sha256s),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            self.schema != SCHEMA
            or not _valid_sha256(self.cohort_nonce_sha256)
            or type(self.training_batches) is not tuple
            or len(self.training_batches) != 5
            or type(self.training_expected_outputs) is not tuple
            or len(self.training_expected_outputs) != 5
            or type(self.evaluation_batches) is not tuple
            or len(self.evaluation_batches) != 2
            or type(self.evaluation_expected_output_sha256s) is not tuple
            or len(self.evaluation_expected_output_sha256s) != 2
            or any(not _valid_sha256(value) for value in self.evaluation_expected_output_sha256s)
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision cohort specification changed"
            )
        for index, (batch, expected) in enumerate(
            zip(
                self.training_batches,
                self.training_expected_outputs,
                strict=True,
            )
        ):
            checked_batch = _validate_outputs(batch, label=f"training batch {index}")
            _validate_outputs(
                expected,
                label=f"training expected outputs {index}",
                expected_cases=len(checked_batch),
            )
        adaptation = _validate_outputs(
            self.adaptation_batch,
            label="adaptation batch",
        )
        _validate_outputs(
            self.adaptation_expected_outputs,
            label="adaptation expected outputs",
            expected_cases=len(adaptation),
        )
        for index, batch in enumerate(self.evaluation_batches):
            _validate_outputs(batch, label=f"evaluation batch {index}")
        if self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False)):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision cohort specification authentication failed"
            )


def make_feedback_revision_cohort_spec_v1(
    *,
    cohort_nonce_sha256: str,
    training_batches: tuple[tuple[tuple[int, ...], ...], ...],
    training_expected_outputs: tuple[tuple[tuple[int, ...], ...], ...],
    adaptation_batch: tuple[tuple[int, ...], ...],
    adaptation_expected_outputs: tuple[tuple[int, ...], ...],
    evaluation_batches: tuple[tuple[tuple[int, ...], ...], ...],
    evaluation_expected_outputs: tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ],
) -> FeedbackRevisionCohortSpecV1:
    """Seal replaceable public inputs around the frozen C0 cognition protocol."""

    if type(evaluation_expected_outputs) is not tuple or len(evaluation_expected_outputs) != 2:
        raise TypeError("evaluation expected outputs must contain two exact batches")
    payload = {
        "schema": SCHEMA,
        "cohort_nonce_sha256": cohort_nonce_sha256,
        "training_batches": training_batches,
        "training_expected_outputs": training_expected_outputs,
        "adaptation_batch": adaptation_batch,
        "adaptation_expected_outputs": adaptation_expected_outputs,
        "evaluation_batches": evaluation_batches,
        "evaluation_expected_output_sha256s": tuple(
            feedback_revision_output_sha256_v1(outputs) for outputs in evaluation_expected_outputs
        ),
    }
    result = FeedbackRevisionCohortSpecV1(
        **payload,
        sha256=_digest(_SPEC_NAMESPACE, payload),
    )
    result.authenticate()
    return result


ACTION_A_PROJECTION_SOURCE = "opaque_action.slot_a()"
ACTION_B_PROJECTION_SOURCE = "opaque_action.slot_b()"

ACTION_A_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "candidate_a",
    ACTION_A_PROJECTION_SOURCE,
)
ACTION_B_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "candidate_b",
    ACTION_B_PROJECTION_SOURCE,
)

ACTION_A_RUST_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().rev().copied().collect()
}
"""

ACTION_B_RUST_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    let mut output = values.to_vec();
    output.sort();
    output
}
"""

CANDIDATE_SOURCES = (
    (ACTION_A_CANDIDATE, ACTION_A_RUST_SOURCE),
    (ACTION_B_CANDIDATE, ACTION_B_RUST_SOURCE),
)

_PUBLIC_ORACLE_OUTPUTS = (
    ((2, 2, 7, 9), (-4, 0, 1, 3)),
    ((1, 8, -4), (5, -2, 0, 5)),
)

DEFAULT_PUBLIC_SPEC = make_feedback_revision_cohort_spec_v1(
    cohort_nonce_sha256=sha256(
        b"continual-rust-feedback-revision-public-development-v1"
    ).hexdigest(),
    training_batches=(
        ((1, 2, 3), (-2, 5)),
        ((4, 1, 3), (0, -1, 8, 2)),
        ((3, 1, 2), (5, -2, 5, 0)),
        ((9, 4, 7, 4), (0, -3, 2)),
        ((6, 1, -5), (2, 2, 8, 0)),
    ),
    training_expected_outputs=(
        ((3, 2, 1), (5, -2)),
        ((3, 1, 4), (2, 8, -1, 0)),
        ((1, 2, 3), (-2, 0, 5, 5)),
        ((4, 4, 7, 9), (-3, 0, 2)),
        ((-5, 1, 6), (0, 8, 2, 2)),
    ),
    adaptation_batch=((4, 1, 3, 2), (7, -1, 7, 0)),
    adaptation_expected_outputs=((1, 2, 3, 4), (-1, 0, 7, 7)),
    evaluation_batches=(
        ((9, 2, 7, 2), (3, -4, 1, 0)),
        ((-4, 8, 1), (5, 0, -2, 5)),
    ),
    evaluation_expected_outputs=_PUBLIC_ORACLE_OUTPUTS,
)

_CASE_LABELS = (
    "A1",
    "A2",
    "B1",
    "B2",
    "A3",
    "ADAPT_FAIL",
    "ADAPT_PASS",
    "R_PRIME",
    "U_PRIME",
)
_CASE_CUES: dict[str, tuple[str, ...]] = {
    "A1": ("kavun",),
    "A2": ("kavun", "pelto"),
    "B1": ("pelto",),
    "B2": ("pelto",),
    "A3": ("zimra",),
    "ADAPT": ("kavun", "pelto"),
    "R_PRIME": ("kavun", "pelto"),
    "U_PRIME": ("zimra",),
}
_TRAINING_CANDIDATES = (
    ACTION_A_CANDIDATE,
    ACTION_A_CANDIDATE,
    ACTION_B_CANDIDATE,
    ACTION_B_CANDIDATE,
    ACTION_A_CANDIDATE,
)


@dataclass(frozen=True, slots=True)
class FeedbackRevisionAttemptV1:
    schema: str
    ordinal: int
    phase: str
    case_label: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    selection_sha256: str
    selected_candidate_sha256: str
    candidate_scores: tuple[tuple[int, str, int, int, int, int], ...]
    expected_output_sha256: str
    observed_output_sha256: str
    public_feedback_sha256: str
    lesson_sha256: str | None
    passed: int
    memory_updated: int
    checkpoint_before_sha256: str | None
    checkpoint_after_sha256: str
    parent_checkpoint_after_sha256: str | None
    execution_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return result

    def authenticate(self) -> None:
        roots = (
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.selection_sha256,
            self.selected_candidate_sha256,
            self.expected_output_sha256,
            self.observed_output_sha256,
            self.public_feedback_sha256,
            self.checkpoint_after_sha256,
            self.execution_sha256,
        )
        if (
            self.schema != SCHEMA
            or type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 9
            or self.phase not in {TRAIN, ADAPTATION_FAIL, ADAPTATION_PASS, EVALUATION}
            or self.case_label not in _CASE_LABELS
            or any(not _valid_sha256(value) for value in roots)
            or type(self.passed) is not int
            or self.passed not in (0, 1)
            or self.passed != int(self.observed_output_sha256 == self.expected_output_sha256)
            or type(self.memory_updated) is not int
            or self.memory_updated not in (0, 1)
            or type(self.candidate_scores) is not tuple
        ):
            raise ContinualRustFeedbackRevisionCohortError("feedback-revision attempt changed")
        for index, row in enumerate(self.candidate_scores):
            if (
                type(row) is not tuple
                or len(row) != 6
                or row[0] != index
                or not _valid_sha256(row[1])
                or type(row[2]) is not int
                or row[2] not in (0, 1)
                or any(type(value) is not int or value < 0 for value in row[3:])
                or row[5] != max(0, row[3] - row[4])
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision attempt scores changed"
                )
        optional_roots = (
            self.lesson_sha256,
            self.checkpoint_before_sha256,
            self.parent_checkpoint_after_sha256,
        )
        if any(value is not None and not _valid_sha256(value) for value in optional_roots):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision attempt optional roots changed"
            )
        if self.memory_updated:
            if self.lesson_sha256 is None or (
                self.checkpoint_before_sha256 != self.parent_checkpoint_after_sha256
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision update custody changed"
                )
        elif (
            self.lesson_sha256 is not None
            or self.checkpoint_before_sha256 is None
            or self.checkpoint_before_sha256 != self.checkpoint_after_sha256
            or self.parent_checkpoint_after_sha256 != self.checkpoint_before_sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision evaluation mutated memory"
            )
        if self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False)):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class FeedbackRevisionWorkV1:
    source_release_count: int
    compile_count: int
    run_count: int
    unselected_execution_count: int
    memory_update_count: int
    evaluation_memory_update_count: int
    oracle_callback_count: int
    parent_mutation_count: int
    parent_lesson_replay_count: int
    intervention_issue_count: int
    vault_recall_count: int
    oracle_comparison_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return result

    def authenticate(self) -> None:
        counts = tuple(
            value
            for name, value in self.record(include_sha256=False).items()
            if name.endswith("_count")
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ContinualRustFeedbackRevisionCohortError("feedback-revision work counts changed")
        if self.sha256 != _digest(
            _WORK_NAMESPACE,
            self.record(include_sha256=False),
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision work authentication failed"
            )


@dataclass(frozen=True, slots=True)
class FeedbackRevisionCohortResultV1:
    schema: str
    status: str
    claim_boundary: str
    spec: FeedbackRevisionCohortSpecV1
    candidate_sha256s: tuple[str, str]
    initial_checkpoint_sha256: str | None
    trained_checkpoint_sha256: str | None
    failed_checkpoint_sha256: str | None
    final_checkpoint_sha256: str | None
    checkpoint_lineage_sha256s: tuple[str, ...]
    attempts: tuple[FeedbackRevisionAttemptV1, ...]
    intervention_authority: intervention.RustActionInterventionAuthorityV1 | None
    intervention_catalogue: intervention.RustActionInterventionCatalogueV1 | None
    intervention_selections: tuple[intervention.RustActionInterventionSelectionV1, ...]
    promoted_vault_receipts: tuple[promoted_vault.PromotedRustActionVaultReceiptV1, ...]
    promoted_vault_authorities: tuple[promoted_vault.PromotedRustActionVaultAuthorityV1, ...]
    continual_recall_selections: tuple[continual.ContinualDocumentationActionRecallV1, ...]
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...]
    lessons: tuple[continual.ContinualDocumentationActionLessonV1, ...]
    failure_update_off_recall: continual.ContinualDocumentationActionRecallV1 | None
    work: FeedbackRevisionWorkV1
    evaluation_match_cells: tuple[int, ...]
    oracle_observation_sha256: str | None
    sha256: str

    @property
    def adaptation_attempts(self) -> tuple[FeedbackRevisionAttemptV1, ...]:
        return tuple(
            attempt
            for attempt in self.attempts
            if attempt.phase in {ADAPTATION_FAIL, ADAPTATION_PASS}
        )

    @property
    def evaluation_attempts(self) -> tuple[FeedbackRevisionAttemptV1, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.phase == EVALUATION)

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "spec": self.spec.record(),
            "candidate_sha256s": self.candidate_sha256s,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "trained_checkpoint_sha256": self.trained_checkpoint_sha256,
            "failed_checkpoint_sha256": self.failed_checkpoint_sha256,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "checkpoint_lineage_sha256s": self.checkpoint_lineage_sha256s,
            "attempts": tuple(attempt.record() for attempt in self.attempts),
            "intervention_authority": (
                None if self.intervention_authority is None else asdict(self.intervention_authority)
            ),
            "intervention_catalogue": (
                None if self.intervention_catalogue is None else asdict(self.intervention_catalogue)
            ),
            "intervention_selections": tuple(
                asdict(selection) for selection in self.intervention_selections
            ),
            "promoted_vault_receipts": tuple(
                asdict(receipt) for receipt in self.promoted_vault_receipts
            ),
            "promoted_vault_authorities": tuple(
                asdict(authority) for authority in self.promoted_vault_authorities
            ),
            "continual_recall_selections": tuple(
                selection.record() for selection in self.continual_recall_selections
            ),
            "executions": tuple(asdict(execution) for execution in self.executions),
            "lessons": tuple(lesson.record() for lesson in self.lessons),
            "failure_update_off_recall": (
                None
                if self.failure_update_off_recall is None
                else self.failure_update_off_recall.record()
            ),
            "work": self.work.record(),
            "evaluation_match_cells": self.evaluation_match_cells,
            "oracle_observation_sha256": self.oracle_observation_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or type(self.spec) is not FeedbackRevisionCohortSpecV1
            or self.candidate_sha256s != (ACTION_A_CANDIDATE.sha256, ACTION_B_CANDIDATE.sha256)
            or type(self.attempts) is not tuple
            or type(self.intervention_selections) is not tuple
            or type(self.promoted_vault_receipts) is not tuple
            or type(self.promoted_vault_authorities) is not tuple
            or type(self.continual_recall_selections) is not tuple
            or type(self.executions) is not tuple
            or type(self.lessons) is not tuple
            or type(self.checkpoint_lineage_sha256s) is not tuple
            or type(self.evaluation_match_cells) is not tuple
            or type(self.work) is not FeedbackRevisionWorkV1
        ):
            raise ContinualRustFeedbackRevisionCohortError("feedback-revision result changed")
        self.spec.authenticate()

        stopped = self.status in {STOP_CELL_OFF, STOP_DOCUMENTATION_OFF}
        if stopped:
            if (
                any(
                    value is not None
                    for value in (
                        self.initial_checkpoint_sha256,
                        self.trained_checkpoint_sha256,
                        self.failed_checkpoint_sha256,
                        self.final_checkpoint_sha256,
                        self.oracle_observation_sha256,
                    )
                )
                or self.checkpoint_lineage_sha256s
                or self.attempts
                or self.intervention_authority is not None
                or self.intervention_catalogue is not None
                or self.intervention_selections
                or self.promoted_vault_receipts
                or self.promoted_vault_authorities
                or self.continual_recall_selections
                or self.executions
                or self.lessons
                or self.failure_update_off_recall is not None
                or self.evaluation_match_cells
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "stopped feedback-revision cohort performed work"
                )
            self.work.authenticate()
        else:
            if self.status not in {PASS, FAIL_ORACLE_MISMATCH}:
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision terminal status changed"
                )
            roots = (
                self.initial_checkpoint_sha256,
                self.trained_checkpoint_sha256,
                self.failed_checkpoint_sha256,
                self.final_checkpoint_sha256,
                self.oracle_observation_sha256,
                *self.checkpoint_lineage_sha256s,
            )
            if (
                len(self.checkpoint_lineage_sha256s) != 7
                or len(set(self.checkpoint_lineage_sha256s)) != 7
                or any(not _valid_sha256(value) for value in roots)
                or self.initial_checkpoint_sha256 != self.checkpoint_lineage_sha256s[0]
                or self.trained_checkpoint_sha256 != self.checkpoint_lineage_sha256s[4]
                or self.failed_checkpoint_sha256 != self.checkpoint_lineage_sha256s[5]
                or self.final_checkpoint_sha256 != self.checkpoint_lineage_sha256s[6]
                or len(self.attempts) != 9
                or self.evaluation_match_cells not in {(1, 1), (0, 1), (1, 0), (0, 0)}
                or self.status
                != (PASS if self.evaluation_match_cells == (1, 1) else FAIL_ORACLE_MISMATCH)
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision result lineage changed"
                )
            for ordinal, attempt in enumerate(self.attempts, start=1):
                if type(attempt) is not FeedbackRevisionAttemptV1:
                    raise TypeError("result attempts must be exact")
                attempt.authenticate()
                if attempt.ordinal != ordinal or attempt.case_label != _CASE_LABELS[ordinal - 1]:
                    raise ContinualRustFeedbackRevisionCohortError(
                        "feedback-revision attempt order changed"
                    )

            expected_phases = (
                TRAIN,
                TRAIN,
                TRAIN,
                TRAIN,
                TRAIN,
                ADAPTATION_FAIL,
                ADAPTATION_PASS,
                EVALUATION,
                EVALUATION,
            )
            expected_candidates = (
                self.candidate_sha256s[0],
                self.candidate_sha256s[0],
                self.candidate_sha256s[1],
                self.candidate_sha256s[1],
                self.candidate_sha256s[0],
                self.candidate_sha256s[0],
                self.candidate_sha256s[1],
                self.candidate_sha256s[1],
                self.candidate_sha256s[0],
            )
            if (
                tuple(attempt.phase for attempt in self.attempts) != expected_phases
                or tuple(attempt.selected_candidate_sha256 for attempt in self.attempts)
                != expected_candidates
                or tuple(attempt.passed for attempt in self.attempts)
                != (1, 1, 1, 1, 1, 0, 1, *self.evaluation_match_cells)
                or tuple(attempt.memory_updated for attempt in self.attempts)
                != (1, 1, 1, 1, 1, 1, 1, 0, 0)
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision causal sequence changed"
                )

            a_root, b_root = self.candidate_sha256s
            expected_scores = (
                (
                    (0, a_root, 1, 3, 0, 3),
                    (1, b_root, 1, 2, 0, 2),
                ),
                (
                    (0, a_root, 1, 3, 2, 1),
                    (1, b_root, 1, 2, 0, 2),
                ),
                (
                    (0, a_root, 1, 3, 2, 1),
                    (1, b_root, 1, 4, 0, 4),
                ),
                (
                    (0, a_root, 1, 1, 0, 1),
                    (1, b_root, 1, 0, 0, 0),
                ),
            )
            if tuple(attempt.candidate_scores for attempt in self.attempts[5:]) != expected_scores:
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision score transition changed"
                )

            lineage = self.checkpoint_lineage_sha256s
            expected_before = (None, *lineage[:4], lineage[4], lineage[5], lineage[6], lineage[6])
            expected_after = (*lineage, lineage[6], lineage[6])
            if (
                tuple(attempt.checkpoint_before_sha256 for attempt in self.attempts)
                != expected_before
                or tuple(attempt.checkpoint_after_sha256 for attempt in self.attempts)
                != expected_after
                or any(
                    attempt.parent_checkpoint_after_sha256 != attempt.checkpoint_before_sha256
                    for attempt in self.attempts
                )
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "feedback-revision parent immutability changed"
                )
            _authenticate_result_transcript(self)
            self.work.authenticate()

        derived_work = _derive_work(
            intervention_selections=self.intervention_selections,
            continual_recall_selections=self.continual_recall_selections,
            executions=self.executions,
            lessons=self.lessons,
            attempts=self.attempts,
            evaluation_match_cells=self.evaluation_match_cells,
            oracle_observation_sha256=self.oracle_observation_sha256,
        )
        if self.work != derived_work:
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision work differs from the audit transcript"
            )
        if not stopped and tuple(
            value
            for name, value in self.work.record(include_sha256=False).items()
            if name.endswith("_count")
        ) != (9, 9, 9, 0, 7, 0, 1, 0, 0, 5, 4, 2):
            raise ContinualRustFeedbackRevisionCohortError("feedback-revision audited work changed")

        if self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False)):
            raise ContinualRustFeedbackRevisionCohortError(
                "feedback-revision result authentication failed"
            )


@dataclass(frozen=True, slots=True)
class _TaskMaterialV1:
    task: knowledge_action.KnowledgeActionTaskProjectionV1
    task_instance_sha256: str
    task_source_sha256: str
    evidence: continual.DocumentationActionEvidenceV1


def _task_material(spec: FeedbackRevisionCohortSpecV1, case_label: str) -> _TaskMaterialV1:
    cues = _CASE_CUES[case_label]
    seed = {"spec_sha256": spec.sha256, "case_label": case_label}
    task_instance_sha256 = _digest(_TASK_INSTANCE_NAMESPACE, seed)
    task_source_sha256 = _digest(_TASK_SOURCE_NAMESPACE, seed)
    documentation_source_sha256 = _digest(_DOCUMENTATION_SOURCE_NAMESPACE, seed)
    opaque_instance = f"instance_{task_instance_sha256[:20]}"
    task = knowledge_action.project_knowledge_action_task_v1(" ".join((*cues, opaque_instance)))
    cue_projection = knowledge_action.project_knowledge_action_task_v1(" ".join(cues))
    evidence = continual.make_documentation_action_evidence_v1(
        factors=cue_projection.factors,
        source_sha256=documentation_source_sha256,
        provenance_roots=(spec.sha256, task_source_sha256),
    )
    if tuple(sorted(set(task.factors).intersection(evidence.factors))) != tuple(
        f"tok:{cue}" for cue in cues
    ):
        raise ContinualRustFeedbackRevisionCohortError("opaque cue projection changed")
    return _TaskMaterialV1(
        task=task,
        task_instance_sha256=task_instance_sha256,
        task_source_sha256=task_source_sha256,
        evidence=evidence,
    )


def _public_feedback_sha256(
    *,
    spec: FeedbackRevisionCohortSpecV1,
    case_label: str,
    candidate_sha256: str,
    execution: rust_executor.RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
) -> str:
    execution.authenticate_structure()
    observed_root = feedback_revision_output_sha256_v1(execution.outputs)
    expected_root = feedback_revision_output_sha256_v1(expected_outputs)
    return _digest(
        _PUBLIC_FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "case_label": case_label,
            "candidate_sha256": candidate_sha256,
            "execution_sha256": execution.sha256,
            "observed_output_sha256": observed_root,
            "expected_output_sha256": expected_root,
            "passed": int(execution.outputs == expected_outputs),
        },
    )


def _seal_lesson(
    *,
    spec: FeedbackRevisionCohortSpecV1,
    case_label: str,
    material: _TaskMaterialV1,
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    selection_sha256: str,
    execution: rust_executor.RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
) -> tuple[continual.ContinualDocumentationActionLessonV1, str]:
    execution.authenticate_structure()
    checked_expected = _validate_outputs(
        expected_outputs,
        label=f"{case_label} public expected outputs",
        expected_cases=len(execution.outputs),
    )
    if (
        execution.selected_source.selection_sha256 != selection_sha256
        or execution.selected_source.action_handle != candidate.candidate_id
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "selected execution differs from the opaque candidate"
        )
    public_feedback_sha256 = _public_feedback_sha256(
        spec=spec,
        case_label=case_label,
        candidate_sha256=candidate.sha256,
        execution=execution,
        expected_outputs=checked_expected,
    )
    lesson = continual.seal_continual_documentation_action_lesson_v1(
        task=material.task,
        task_instance_sha256=material.task_instance_sha256,
        task_source_sha256=material.task_source_sha256,
        evidence=material.evidence,
        candidate=candidate,
        execution_sha256=execution.sha256,
        public_feedback_sha256=public_feedback_sha256,
        observed_outputs=execution.outputs,
        expected_outputs=checked_expected,
    )
    lesson.authenticate()
    return lesson, public_feedback_sha256


def _advance_memory(
    parent: continual.FrozenContinualDocumentationActionMemoryV1 | None,
    lesson: continual.ContinualDocumentationActionLessonV1,
) -> tuple[
    continual.FrozenContinualDocumentationActionMemoryV1,
    str | None,
    str | None,
]:
    if parent is None:
        successor = continual.start_continual_documentation_action_memory_v1(lesson)
        return successor, None, None
    parent_root = parent.checkpoint_sha256
    parent_arrays = parent.checkpoint_array_sha256s
    successor = continual.advance_continual_documentation_action_memory_v1(
        parent,
        lesson,
    )
    parent.authenticate()
    if parent.checkpoint_sha256 != parent_root or parent.checkpoint_array_sha256s != parent_arrays:
        raise ContinualRustFeedbackRevisionCohortError("continual advance mutated its parent")
    return successor, parent_root, parent.checkpoint_sha256


def _make_attempt(
    *,
    ordinal: int,
    phase: str,
    case_label: str,
    material: _TaskMaterialV1,
    selection_sha256: str,
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    candidate_scores: tuple[tuple[int, str, int, int, int, int], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
    execution: rust_executor.RustSelectedActionExecutionV2,
    public_feedback_sha256: str,
    lesson_sha256: str | None,
    memory_updated: int,
    checkpoint_before_sha256: str | None,
    checkpoint_after_sha256: str,
    parent_checkpoint_after_sha256: str | None,
) -> FeedbackRevisionAttemptV1:
    partial = FeedbackRevisionAttemptV1(
        schema=SCHEMA,
        ordinal=ordinal,
        phase=phase,
        case_label=case_label,
        task_sha256=material.task.sha256,
        task_instance_sha256=material.task_instance_sha256,
        task_source_sha256=material.task_source_sha256,
        selection_sha256=selection_sha256,
        selected_candidate_sha256=candidate.sha256,
        candidate_scores=candidate_scores,
        expected_output_sha256=feedback_revision_output_sha256_v1(expected_outputs),
        observed_output_sha256=feedback_revision_output_sha256_v1(execution.outputs),
        public_feedback_sha256=public_feedback_sha256,
        lesson_sha256=lesson_sha256,
        passed=int(execution.outputs == expected_outputs),
        memory_updated=memory_updated,
        checkpoint_before_sha256=checkpoint_before_sha256,
        checkpoint_after_sha256=checkpoint_after_sha256,
        parent_checkpoint_after_sha256=parent_checkpoint_after_sha256,
        execution_sha256=execution.sha256,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _derive_work(
    *,
    intervention_selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    continual_recall_selections: tuple[continual.ContinualDocumentationActionRecallV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    lessons: tuple[continual.ContinualDocumentationActionLessonV1, ...],
    attempts: tuple[FeedbackRevisionAttemptV1, ...],
    evaluation_match_cells: tuple[int, ...],
    oracle_observation_sha256: str | None,
) -> FeedbackRevisionWorkV1:
    selection_roots = {
        *(selection.sha256 for selection in intervention_selections),
        *(selection.sha256 for selection in continual_recall_selections),
    }
    values = {
        "source_release_count": len(executions),
        "compile_count": sum(
            execution.compile_receipt.compile_invocations for execution in executions
        ),
        "run_count": sum(execution.run_receipt.run_invocations for execution in executions),
        "unselected_execution_count": sum(
            execution.selected_source.selection_sha256 not in selection_roots
            for execution in executions
        ),
        "memory_update_count": sum(attempt.memory_updated for attempt in attempts),
        "evaluation_memory_update_count": sum(
            attempt.memory_updated for attempt in attempts if attempt.phase == EVALUATION
        ),
        "oracle_callback_count": int(oracle_observation_sha256 is not None),
        "parent_mutation_count": sum(
            attempt.parent_checkpoint_after_sha256 != attempt.checkpoint_before_sha256
            for attempt in attempts
        ),
        "parent_lesson_replay_count": len(lessons) - len({lesson.sha256 for lesson in lessons}),
        "intervention_issue_count": len(intervention_selections),
        "vault_recall_count": len(continual_recall_selections),
        "oracle_comparison_count": len(evaluation_match_cells),
    }
    partial = FeedbackRevisionWorkV1(**values, sha256="")
    result = replace(
        partial,
        sha256=_digest(_WORK_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _executor_batch_commitments(
    batch: tuple[tuple[int, ...], ...],
) -> tuple[str, str, int]:
    cases = rust_executor._validate_batch(batch)
    stdin = rust_executor._encode_batch(cases)
    return (
        rust_executor._digest(rust_executor._V2_BATCH_NAMESPACE, cases),
        sha256(stdin).hexdigest(),
        len(cases),
    )


def _authenticate_result_transcript(result: FeedbackRevisionCohortResultV1) -> None:
    if (
        type(result.intervention_authority) is not intervention.RustActionInterventionAuthorityV1
        or type(result.intervention_catalogue) is not intervention.RustActionInterventionCatalogueV1
        or len(result.intervention_selections) != 5
        or len(result.promoted_vault_receipts) != 4
        or len(result.promoted_vault_authorities) != 4
        or len(result.continual_recall_selections) != 4
        or len(result.executions) != 9
        or len(result.lessons) != 7
        or type(result.failure_update_off_recall)
        is not continual.ContinualDocumentationActionRecallV1
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "feedback-revision audit transcript shape changed"
        )
    if any(
        type(execution) is not rust_executor.RustSelectedActionExecutionV2
        for execution in result.executions
    ):
        raise TypeError("audit executions must be exact V2 receipts")
    if any(
        type(lesson) is not continual.ContinualDocumentationActionLessonV1
        for lesson in result.lessons
    ):
        raise TypeError("audit lessons must be exact continual lessons")

    expected_candidates = (
        ACTION_A_CANDIDATE,
        ACTION_A_CANDIDATE,
        ACTION_B_CANDIDATE,
        ACTION_B_CANDIDATE,
        ACTION_A_CANDIDATE,
        ACTION_A_CANDIDATE,
        ACTION_B_CANDIDATE,
        ACTION_B_CANDIDATE,
        ACTION_A_CANDIDATE,
    )
    material_labels = (
        "A1",
        "A2",
        "B1",
        "B2",
        "A3",
        "ADAPT",
        "ADAPT",
        "R_PRIME",
        "U_PRIME",
    )
    expected_materials = tuple(_task_material(result.spec, label) for label in material_labels)
    expected_batches = (
        *result.spec.training_batches,
        result.spec.adaptation_batch,
        result.spec.adaptation_batch,
        *result.spec.evaluation_batches,
    )
    adaptation_expected_sha256 = feedback_revision_output_sha256_v1(
        result.spec.adaptation_expected_outputs
    )
    expected_output_sha256s = (
        *tuple(
            feedback_revision_output_sha256_v1(outputs)
            for outputs in result.spec.training_expected_outputs
        ),
        adaptation_expected_sha256,
        adaptation_expected_sha256,
        *result.spec.evaluation_expected_output_sha256s,
    )
    expected_lesson_outputs = (
        *result.spec.training_expected_outputs,
        result.spec.adaptation_expected_outputs,
        result.spec.adaptation_expected_outputs,
    )
    for attempt, material, expected_output_sha256 in zip(
        result.attempts,
        expected_materials,
        expected_output_sha256s,
        strict=True,
    ):
        if (
            attempt.task_sha256 != material.task.sha256
            or attempt.task_instance_sha256 != material.task_instance_sha256
            or attempt.task_source_sha256 != material.task_source_sha256
            or attempt.expected_output_sha256 != expected_output_sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "attempt task material or expected consequence differs from the retained spec"
            )

    rust_sources_by_candidate_sha256 = {
        candidate.sha256: source for candidate, source in CANDIDATE_SOURCES
    }
    canonical_candidate_sources = tuple(sorted(CANDIDATE_SOURCES, key=lambda pair: pair[0].sha256))
    intervention_bindings = tuple(
        intervention._make_binding(candidate, source)
        for candidate, source in canonical_candidate_sources
    )
    intervention_catalogue = result.intervention_catalogue
    intervention_authority = result.intervention_authority
    intervention_catalogue.authenticate_structure()
    intervention_authority.authenticate_structure()
    if (
        intervention_catalogue.candidate_sha256s
        != tuple(candidate.sha256 for candidate, _source in canonical_candidate_sources)
        or intervention_catalogue.rust_source_sha256s
        != tuple(
            sha256(source.encode("ascii", errors="strict")).hexdigest()
            for _candidate, source in canonical_candidate_sources
        )
        or intervention_catalogue.binding_sha256s
        != tuple(binding.sha256 for binding in intervention_bindings)
        or intervention_catalogue.action_count != len(CANDIDATE_SOURCES)
        or intervention_authority.catalogue_sha256 != intervention_catalogue.sha256
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "intervention catalogue or authority differs from frozen candidate sources"
        )

    expected_intervention_sha256s = tuple(
        _digest(
            _INTERVENTION_NAMESPACE,
            {
                "spec_sha256": result.spec.sha256,
                "case_label": case_label,
                "task_sha256": material.task.sha256,
                "candidate_sha256": candidate.sha256,
                "batch": batch,
                "expected_output_sha256": feedback_revision_output_sha256_v1(expected_outputs),
            },
        )
        for case_label, material, candidate, batch, expected_outputs in zip(
            _CASE_LABELS[:5],
            expected_materials[:5],
            expected_candidates[:5],
            result.spec.training_batches,
            result.spec.training_expected_outputs,
            strict=True,
        )
    )
    for selection, expected_intervention_sha256 in zip(
        result.intervention_selections,
        expected_intervention_sha256s,
        strict=True,
    ):
        if type(selection) is not intervention.RustActionInterventionSelectionV1:
            raise TypeError("intervention transcript selections must be exact")
        selection.authenticate_structure()
        if (
            selection.authority_sha256 != intervention_authority.sha256
            or selection.catalogue_sha256 != intervention_catalogue.sha256
            or selection.intervention_sha256 != expected_intervention_sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "intervention selection differs from its exact spec-rooted intervention"
            )
    if tuple(selection.candidate_sha256 for selection in result.intervention_selections) != tuple(
        candidate.sha256 for candidate in expected_candidates[:5]
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "intervention transcript candidate custody changed"
        )

    released_recalls = result.continual_recall_selections
    for offset, selection in enumerate(released_recalls, start=5):
        if type(selection) is not continual.ContinualDocumentationActionRecallV1:
            raise TypeError("continual transcript selections must be exact")
        selection.authenticate()
        attempt = result.attempts[offset]
        candidate = expected_candidates[offset]
        if (
            selection.status != continual.RECALLED
            or selection.candidates != (ACTION_A_CANDIDATE, ACTION_B_CANDIDATE)
            or not selection.cell_gate
            or not selection.documentation_gate
            or selection.selected_candidate_sha256 != candidate.sha256
            or selection.selected_candidate_id != candidate.candidate_id
            or selection.selected_source_sha256 != candidate.source_sha256
            or selection.sha256 != attempt.selection_sha256
            or selection.task_sha256 != attempt.task_sha256
            or selection.checkpoint_sha256 != attempt.checkpoint_before_sha256
            or selection.candidate_scores != attempt.candidate_scores
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "continual recall transcript differs from its attempt"
            )

    expected_vault_checkpoint_sha256s = (
        result.trained_checkpoint_sha256,
        result.failed_checkpoint_sha256,
        result.final_checkpoint_sha256,
        result.final_checkpoint_sha256,
    )
    lesson_provenance = tuple(
        continual._provenance_from_lesson(lesson) for lesson in result.lessons
    )
    lesson_provenance_sha256s = tuple(row.sha256 for row in lesson_provenance)
    expected_vault_provenance_sha256s = (
        lesson_provenance_sha256s[:5],
        lesson_provenance_sha256s[:6],
        lesson_provenance_sha256s[:7],
        lesson_provenance_sha256s[:7],
    )
    expected_rust_source_sha256s = tuple(
        sha256(source.encode("ascii", errors="strict")).hexdigest()
        for _candidate, source in CANDIDATE_SOURCES
    )
    promoted_bindings = tuple(
        promoted_vault._make_binding(candidate, source) for candidate, source in CANDIDATE_SOURCES
    )
    for offset, (receipt, authority, selection, execution) in enumerate(
        zip(
            result.promoted_vault_receipts,
            result.promoted_vault_authorities,
            released_recalls,
            result.executions[5:],
            strict=True,
        )
    ):
        if type(receipt) is not promoted_vault.PromotedRustActionVaultReceiptV1:
            raise TypeError("promoted-vault transcript receipts must be exact")
        if type(authority) is not promoted_vault.PromotedRustActionVaultAuthorityV1:
            raise TypeError("promoted-vault transcript authorities must be exact")
        receipt.authenticate_structure()
        authority.authenticate_structure()
        selected_binding = promoted_bindings[selection.selected_action_index]
        promotion_provenance_sha256s = tuple(
            row.sha256
            for row in lesson_provenance[: (5, 6, 7, 7)[offset]]
            if row.candidate_sha256 == selection.selected_candidate_sha256 and row.passed_cell == 1
        )
        expected_release = promoted_vault._make_release(
            authority_sha256=authority.sha256,
            selection=selection,
            binding=selected_binding,
            promotion_provenance_sha256s=promotion_provenance_sha256s,
        )
        if (
            receipt.memory_checkpoint_sha256 != expected_vault_checkpoint_sha256s[offset]
            or receipt.memory_checkpoint_sha256 != selection.checkpoint_sha256
            or receipt.memory_checkpoint_sha256
            != result.attempts[offset + 5].checkpoint_before_sha256
            or receipt.candidate_sha256s != (ACTION_A_CANDIDATE.sha256, ACTION_B_CANDIDATE.sha256)
            or receipt.promoted_candidate_sha256s
            != (ACTION_A_CANDIDATE.sha256, ACTION_B_CANDIDATE.sha256)
            or receipt.rust_source_sha256s != expected_rust_source_sha256s
            or receipt.binding_sha256s != tuple(binding.sha256 for binding in promoted_bindings)
            or receipt.provenance_sha256s != expected_vault_provenance_sha256s[offset]
            or authority.vault_receipt_sha256 != receipt.sha256
            or execution.selected_source.authority_sha256 != authority.sha256
            or execution.selected_source.release_sha256 != expected_release.sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "promoted-vault receipt, authority, recall, and execution custody changed"
            )
    if len({authority.sha256 for authority in result.promoted_vault_authorities}) != 4:
        raise ContinualRustFeedbackRevisionCohortError(
            "promoted-vault executions did not use four distinct authorities"
        )

    all_selection_roots = tuple(
        selection.sha256 for selection in result.intervention_selections
    ) + tuple(selection.sha256 for selection in released_recalls)
    if len(set(all_selection_roots)) != 9:
        raise ContinualRustFeedbackRevisionCohortError("released selection roots are not one-shot")
    for index, (attempt, execution, selection_sha256, candidate, expected_batch) in enumerate(
        zip(
            result.attempts,
            result.executions,
            all_selection_roots,
            expected_candidates,
            expected_batches,
            strict=True,
        )
    ):
        if type(execution) is not rust_executor.RustSelectedActionExecutionV2:
            raise TypeError("audit executions must be exact V2 receipts")
        execution.authenticate_structure()
        rust_source = rust_sources_by_candidate_sha256[candidate.sha256]
        batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(expected_batch)
        if (
            selection_sha256 != attempt.selection_sha256
            or execution.selected_source.selection_sha256 != selection_sha256
            or execution.selected_source.action_handle != candidate.candidate_id
            or execution.selected_source.source != rust_source
            or execution.selected_source.source_sha256
            != sha256(rust_source.encode("ascii", errors="strict")).hexdigest()
            or execution.preparation.batch_sha256 != batch_sha256
            or execution.preparation.stdin_sha256 != stdin_sha256
            or execution.preparation.case_count != case_count
            or execution.sha256 != attempt.execution_sha256
            or feedback_revision_output_sha256_v1(execution.outputs)
            != attempt.observed_output_sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "selection, execution, candidate, and attempt custody changed"
            )
        if index < 5:
            intervention_selection = result.intervention_selections[index]
            intervention_binding = next(
                binding
                for binding in intervention_bindings
                if binding.candidate.sha256 == candidate.sha256
            )
            expected_release = intervention._make_release(
                intervention_selection,
                intervention_binding,
            )
            if (
                execution.selected_source.authority_sha256 != intervention_authority.sha256
                or execution.selected_source.authority_sha256
                != intervention_selection.authority_sha256
                or execution.selected_source.release_sha256 != expected_release.sha256
            ):
                raise ContinualRustFeedbackRevisionCohortError(
                    "intervention execution belongs to a different authority or release"
                )
        feedback_payload = {
            "spec_sha256": result.spec.sha256,
            "case_label": attempt.case_label,
            "candidate_sha256": candidate.sha256,
            "execution_sha256": execution.sha256,
            "observed_output_sha256": attempt.observed_output_sha256,
            "expected_output_sha256": attempt.expected_output_sha256,
            "passed": attempt.passed,
        }
        if attempt.phase == EVALUATION:
            feedback_payload["oracle_observation_sha256"] = result.oracle_observation_sha256
        if attempt.public_feedback_sha256 != _digest(
            _PUBLIC_FEEDBACK_NAMESPACE,
            feedback_payload,
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "attempt feedback is not rooted in its exact execution consequence"
            )

    for attempt, execution, lesson, candidate, material, expected_outputs in zip(
        result.attempts[:7],
        result.executions[:7],
        result.lessons,
        expected_candidates[:7],
        expected_materials[:7],
        expected_lesson_outputs,
        strict=True,
    ):
        if type(lesson) is not continual.ContinualDocumentationActionLessonV1:
            raise TypeError("audit lessons must be exact continual lessons")
        lesson.authenticate()
        if (
            lesson.candidate != candidate
            or lesson.task != material.task
            or lesson.task.sha256 != attempt.task_sha256
            or lesson.task_instance_sha256 != material.task_instance_sha256
            or lesson.task_instance_sha256 != attempt.task_instance_sha256
            or lesson.task_source_sha256 != material.task_source_sha256
            or lesson.task_source_sha256 != attempt.task_source_sha256
            or lesson.evidence != material.evidence
            or lesson.execution_sha256 != execution.sha256
            or lesson.execution_sha256 != attempt.execution_sha256
            or lesson.public_feedback_sha256 != attempt.public_feedback_sha256
            or lesson.observed_outputs != execution.outputs
            or lesson.expected_outputs != expected_outputs
            or feedback_revision_output_sha256_v1(lesson.observed_outputs)
            != attempt.observed_output_sha256
            or feedback_revision_output_sha256_v1(lesson.expected_outputs)
            != attempt.expected_output_sha256
            or lesson.passed_cell != attempt.passed
            or lesson.sha256 != attempt.lesson_sha256
        ):
            raise ContinualRustFeedbackRevisionCohortError(
                "lesson, execution, feedback, outputs, and attempt custody changed"
            )

    failure_attempt, pass_attempt = result.attempts[5:7]
    failure_execution, pass_execution = result.executions[5:7]
    failure_lesson, pass_lesson = result.lessons[5:7]
    if (
        failure_attempt.task_sha256 != pass_attempt.task_sha256
        or failure_attempt.task_instance_sha256 != pass_attempt.task_instance_sha256
        or failure_attempt.task_source_sha256 != pass_attempt.task_source_sha256
        or failure_lesson.task != pass_lesson.task
        or failure_lesson.task_instance_sha256 != pass_lesson.task_instance_sha256
        or failure_lesson.task_source_sha256 != pass_lesson.task_source_sha256
        or failure_lesson.evidence != pass_lesson.evidence
        or failure_execution.preparation.batch_sha256 != pass_execution.preparation.batch_sha256
        or failure_execution.preparation.stdin_sha256 != pass_execution.preparation.stdin_sha256
        or len(
            {
                failure_attempt.selection_sha256,
                pass_attempt.selection_sha256,
            }
        )
        != 2
        or len(
            {
                failure_attempt.execution_sha256,
                pass_attempt.execution_sha256,
            }
        )
        != 2
        or len(
            {
                failure_attempt.public_feedback_sha256,
                pass_attempt.public_feedback_sha256,
            }
        )
        != 2
        or len({failure_attempt.lesson_sha256, pass_attempt.lesson_sha256}) != 2
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "adaptation failure and rerace did not reuse one exact task and batch"
        )

    control = result.failure_update_off_recall
    control.authenticate()
    failure_selection = released_recalls[0]
    if (
        control.status != continual.RECALLED
        or control.candidates != (ACTION_A_CANDIDATE, ACTION_B_CANDIDATE)
        or control.selected_candidate_sha256 != ACTION_A_CANDIDATE.sha256
        or control.task_sha256 != failure_attempt.task_sha256
        or control.checkpoint_sha256 != result.trained_checkpoint_sha256
        or control.candidate_scores != failure_selection.candidate_scores
        or control.sha256 != failure_selection.sha256
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "plasticity-off causal recall changed the pre-update winner"
        )

    if (
        any(attempt.lesson_sha256 is not None for attempt in result.attempts[7:])
        or any(attempt.memory_updated for attempt in result.attempts[7:])
        or any(
            selection.checkpoint_sha256 != result.final_checkpoint_sha256
            for selection in released_recalls[2:]
        )
        or any(
            attempt.checkpoint_before_sha256 != result.final_checkpoint_sha256
            or attempt.checkpoint_after_sha256 != result.final_checkpoint_sha256
            for attempt in result.attempts[7:]
        )
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "evaluation altered the authenticated learning transcript"
        )
    if result.oracle_observation_sha256 != _digest(
        _ORACLE_NAMESPACE,
        {
            "spec_sha256": result.spec.sha256,
            "execution_sha256s": tuple(execution.sha256 for execution in result.executions[7:]),
            "expected_output_sha256s": tuple(
                attempt.expected_output_sha256 for attempt in result.attempts[7:]
            ),
        },
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "oracle observation differs from completed evaluation receipts"
        )


def _seal_result(
    *,
    status: str,
    spec: FeedbackRevisionCohortSpecV1,
    initial_checkpoint_sha256: str | None,
    trained_checkpoint_sha256: str | None,
    failed_checkpoint_sha256: str | None,
    final_checkpoint_sha256: str | None,
    checkpoint_lineage_sha256s: tuple[str, ...],
    attempts: tuple[FeedbackRevisionAttemptV1, ...],
    intervention_authority: intervention.RustActionInterventionAuthorityV1 | None,
    intervention_catalogue: intervention.RustActionInterventionCatalogueV1 | None,
    intervention_selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    promoted_vault_receipts: tuple[promoted_vault.PromotedRustActionVaultReceiptV1, ...],
    promoted_vault_authorities: tuple[promoted_vault.PromotedRustActionVaultAuthorityV1, ...],
    continual_recall_selections: tuple[continual.ContinualDocumentationActionRecallV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    lessons: tuple[continual.ContinualDocumentationActionLessonV1, ...],
    failure_update_off_recall: continual.ContinualDocumentationActionRecallV1 | None,
    evaluation_match_cells: tuple[int, ...],
    oracle_observation_sha256: str | None,
) -> FeedbackRevisionCohortResultV1:
    work = _derive_work(
        intervention_selections=intervention_selections,
        continual_recall_selections=continual_recall_selections,
        executions=executions,
        lessons=lessons,
        attempts=attempts,
        evaluation_match_cells=evaluation_match_cells,
        oracle_observation_sha256=oracle_observation_sha256,
    )
    partial = FeedbackRevisionCohortResultV1(
        schema=SCHEMA,
        status=status,
        claim_boundary=CLAIM_BOUNDARY,
        spec=spec,
        candidate_sha256s=(ACTION_A_CANDIDATE.sha256, ACTION_B_CANDIDATE.sha256),
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        trained_checkpoint_sha256=trained_checkpoint_sha256,
        failed_checkpoint_sha256=failed_checkpoint_sha256,
        final_checkpoint_sha256=final_checkpoint_sha256,
        checkpoint_lineage_sha256s=checkpoint_lineage_sha256s,
        attempts=attempts,
        intervention_authority=intervention_authority,
        intervention_catalogue=intervention_catalogue,
        intervention_selections=intervention_selections,
        promoted_vault_receipts=promoted_vault_receipts,
        promoted_vault_authorities=promoted_vault_authorities,
        continual_recall_selections=continual_recall_selections,
        executions=executions,
        lessons=lessons,
        failure_update_off_recall=failure_update_off_recall,
        work=work,
        evaluation_match_cells=evaluation_match_cells,
        oracle_observation_sha256=oracle_observation_sha256,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _stopped_result(
    spec: FeedbackRevisionCohortSpecV1,
    *,
    status: str,
) -> FeedbackRevisionCohortResultV1:
    return _seal_result(
        status=status,
        spec=spec,
        initial_checkpoint_sha256=None,
        trained_checkpoint_sha256=None,
        failed_checkpoint_sha256=None,
        final_checkpoint_sha256=None,
        checkpoint_lineage_sha256s=(),
        attempts=(),
        intervention_authority=None,
        intervention_catalogue=None,
        intervention_selections=(),
        promoted_vault_receipts=(),
        promoted_vault_authorities=(),
        continual_recall_selections=(),
        executions=(),
        lessons=(),
        failure_update_off_recall=None,
        evaluation_match_cells=(),
        oracle_observation_sha256=None,
    )


def _execute_vault_selection(
    memory: continual.FrozenContinualDocumentationActionMemoryV1,
    *,
    material: _TaskMaterialV1,
    batch: tuple[tuple[int, ...], ...],
    cell_gate: bool,
    documentation_gate: bool,
) -> tuple[
    continual.ContinualDocumentationActionRecallV1,
    rust_executor.RustSelectedActionExecutionV2,
    promoted_vault.PromotedRustActionVaultReceiptV1,
    promoted_vault.PromotedRustActionVaultAuthorityV1,
]:
    vault = promoted_vault.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=CANDIDATE_SOURCES,
    )
    selection = vault.recall(
        material.task,
        cell_gate=cell_gate,
        documentation_gate=documentation_gate,
    )
    selection.authenticate()
    if selection.status != continual.RECALLED:
        raise ContinualRustFeedbackRevisionCohortError(selection.status)
    executor = rust_executor.prepare_rust_selected_action_executor_v2(vault, batch)
    execution = executor.execute_selected_once(selection)
    execution.authenticate_structure()
    return selection, execution, vault.receipt, vault.authority


def run_cohort_v1(
    spec: FeedbackRevisionCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
    *,
    oracle_callback: Callable[
        [],
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
        ],
    ],
    cell_gate: bool = True,
    documentation_gate: bool = True,
) -> FeedbackRevisionCohortResultV1:
    """Run one exact selected-only public feedback/revision cohort."""

    if type(spec) is not FeedbackRevisionCohortSpecV1:
        raise TypeError("run_cohort_v1 requires one exact cohort specification")
    if not callable(oracle_callback):
        raise TypeError("oracle_callback must be one zero-argument callable")
    if type(cell_gate) is not bool or type(documentation_gate) is not bool:
        raise TypeError("recall gates must be exact booleans")
    spec.authenticate()
    if not cell_gate:
        return _stopped_result(spec, status=STOP_CELL_OFF)
    if not documentation_gate:
        return _stopped_result(spec, status=STOP_DOCUMENTATION_OFF)

    prepared_intervention = intervention.prepare_rust_action_intervention_v1(CANDIDATE_SOURCES)
    if prepared_intervention.candidate_sha256s != tuple(
        sorted((ACTION_A_CANDIDATE.sha256, ACTION_B_CANDIDATE.sha256))
    ):
        raise ContinualRustFeedbackRevisionCohortError("intervention candidate catalogue changed")

    attempts: list[FeedbackRevisionAttemptV1] = []
    intervention_selections: list[intervention.RustActionInterventionSelectionV1] = []
    promoted_vault_receipts: list[promoted_vault.PromotedRustActionVaultReceiptV1] = []
    promoted_vault_authorities: list[promoted_vault.PromotedRustActionVaultAuthorityV1] = []
    continual_recall_selections: list[continual.ContinualDocumentationActionRecallV1] = []
    executions: list[rust_executor.RustSelectedActionExecutionV2] = []
    lessons: list[continual.ContinualDocumentationActionLessonV1] = []
    checkpoint_lineage: list[str] = []
    memory: continual.FrozenContinualDocumentationActionMemoryV1 | None = None

    for index, (case_label, candidate, batch, expected_outputs) in enumerate(
        zip(
            _CASE_LABELS[:5],
            _TRAINING_CANDIDATES,
            spec.training_batches,
            spec.training_expected_outputs,
            strict=True,
        ),
        start=1,
    ):
        material = _task_material(spec, case_label)
        intervention_sha256 = _digest(
            _INTERVENTION_NAMESPACE,
            {
                "spec_sha256": spec.sha256,
                "case_label": case_label,
                "task_sha256": material.task.sha256,
                "candidate_sha256": candidate.sha256,
                "batch": batch,
                "expected_output_sha256": feedback_revision_output_sha256_v1(expected_outputs),
            },
        )
        selection = prepared_intervention.issue(
            candidate.sha256,
            intervention_sha256,
        )
        intervention_selections.append(selection)
        execution = prepared_intervention.execute_selected(selection, batch)
        execution.authenticate_structure()
        executions.append(execution)
        lesson, feedback_root = _seal_lesson(
            spec=spec,
            case_label=case_label,
            material=material,
            candidate=candidate,
            selection_sha256=selection.sha256,
            execution=execution,
            expected_outputs=expected_outputs,
        )
        lessons.append(lesson)
        if lesson.passed_cell != 1:
            raise ContinualRustFeedbackRevisionCohortError(
                f"{case_label} training execution did not pass"
            )
        successor, before_root, parent_after_root = _advance_memory(memory, lesson)
        memory = successor
        checkpoint_lineage.append(memory.checkpoint_sha256)
        attempts.append(
            _make_attempt(
                ordinal=index,
                phase=TRAIN,
                case_label=case_label,
                material=material,
                selection_sha256=selection.sha256,
                candidate=candidate,
                candidate_scores=(),
                expected_outputs=expected_outputs,
                execution=execution,
                public_feedback_sha256=feedback_root,
                lesson_sha256=lesson.sha256,
                memory_updated=1,
                checkpoint_before_sha256=before_root,
                checkpoint_after_sha256=memory.checkpoint_sha256,
                parent_checkpoint_after_sha256=parent_after_root,
            )
        )

    assert memory is not None
    memory.authenticate()
    trained_checkpoint_sha256 = memory.checkpoint_sha256
    if memory.promoted_candidate_sha256s != (
        ACTION_A_CANDIDATE.sha256,
        ACTION_B_CANDIDATE.sha256,
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "training did not promote both opaque actions"
        )

    adaptation_material = _task_material(spec, "ADAPT")
    (
        failure_selection,
        failure_execution,
        failure_vault_receipt,
        failure_vault_authority,
    ) = _execute_vault_selection(
        memory,
        material=adaptation_material,
        batch=spec.adaptation_batch,
        cell_gate=cell_gate,
        documentation_gate=documentation_gate,
    )
    promoted_vault_receipts.append(failure_vault_receipt)
    promoted_vault_authorities.append(failure_vault_authority)
    continual_recall_selections.append(failure_selection)
    executions.append(failure_execution)
    if (
        failure_selection.selected_candidate_sha256 != ACTION_A_CANDIDATE.sha256
        or failure_selection.candidate_scores
        != (
            (0, ACTION_A_CANDIDATE.sha256, 1, 3, 0, 3),
            (1, ACTION_B_CANDIDATE.sha256, 1, 2, 0, 2),
        )
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "adaptation did not begin from the frozen A3-versus-B2 score"
        )
    failure_lesson, failure_feedback_root = _seal_lesson(
        spec=spec,
        case_label="ADAPT_FAIL",
        material=adaptation_material,
        candidate=ACTION_A_CANDIDATE,
        selection_sha256=failure_selection.sha256,
        execution=failure_execution,
        expected_outputs=spec.adaptation_expected_outputs,
    )
    lessons.append(failure_lesson)
    if failure_lesson.passed_cell != 0:
        raise ContinualRustFeedbackRevisionCohortError(
            "adaptation's first selected action did not fail"
        )
    failure_update_off_memory = continual.advance_continual_documentation_action_memory_v1(
        memory,
        failure_lesson,
        plasticity_gate=False,
    )
    if failure_update_off_memory is not memory:
        raise ContinualRustFeedbackRevisionCohortError(
            "plasticity-off failure update changed the parent checkpoint"
        )
    failure_update_off_recall = failure_update_off_memory.recall(
        adaptation_material.task,
        cell_gate=cell_gate,
        documentation_gate=documentation_gate,
    )
    failure_update_off_recall.authenticate()
    if (
        failure_update_off_recall.status != continual.RECALLED
        or failure_update_off_recall.selected_candidate_sha256 != ACTION_A_CANDIDATE.sha256
        or failure_update_off_recall.candidate_scores != failure_selection.candidate_scores
        or failure_update_off_recall.sha256 != failure_selection.sha256
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "plasticity-off failure control did not preserve A as winner"
        )

    failed_memory, before_root, parent_after_root = _advance_memory(memory, failure_lesson)
    checkpoint_lineage.append(failed_memory.checkpoint_sha256)
    attempts.append(
        _make_attempt(
            ordinal=6,
            phase=ADAPTATION_FAIL,
            case_label="ADAPT_FAIL",
            material=adaptation_material,
            selection_sha256=failure_selection.sha256,
            candidate=ACTION_A_CANDIDATE,
            candidate_scores=failure_selection.candidate_scores,
            expected_outputs=spec.adaptation_expected_outputs,
            execution=failure_execution,
            public_feedback_sha256=failure_feedback_root,
            lesson_sha256=failure_lesson.sha256,
            memory_updated=1,
            checkpoint_before_sha256=before_root,
            checkpoint_after_sha256=failed_memory.checkpoint_sha256,
            parent_checkpoint_after_sha256=parent_after_root,
        )
    )
    failed_checkpoint_sha256 = failed_memory.checkpoint_sha256

    (
        pass_selection,
        pass_execution,
        pass_vault_receipt,
        pass_vault_authority,
    ) = _execute_vault_selection(
        failed_memory,
        material=adaptation_material,
        batch=spec.adaptation_batch,
        cell_gate=cell_gate,
        documentation_gate=documentation_gate,
    )
    promoted_vault_receipts.append(pass_vault_receipt)
    promoted_vault_authorities.append(pass_vault_authority)
    continual_recall_selections.append(pass_selection)
    executions.append(pass_execution)
    if (
        pass_selection.selected_candidate_sha256 != ACTION_B_CANDIDATE.sha256
        or pass_selection.candidate_scores
        != (
            (0, ACTION_A_CANDIDATE.sha256, 1, 3, 2, 1),
            (1, ACTION_B_CANDIDATE.sha256, 1, 2, 0, 2),
        )
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "inhibitory revision did not rerace B2 over A1"
        )
    pass_lesson, pass_feedback_root = _seal_lesson(
        spec=spec,
        case_label="ADAPT_PASS",
        material=adaptation_material,
        candidate=ACTION_B_CANDIDATE,
        selection_sha256=pass_selection.sha256,
        execution=pass_execution,
        expected_outputs=spec.adaptation_expected_outputs,
    )
    lessons.append(pass_lesson)
    if pass_lesson.passed_cell != 1:
        raise ContinualRustFeedbackRevisionCohortError("adaptation rerace did not pass")
    final_memory, before_root, parent_after_root = _advance_memory(
        failed_memory,
        pass_lesson,
    )
    checkpoint_lineage.append(final_memory.checkpoint_sha256)
    attempts.append(
        _make_attempt(
            ordinal=7,
            phase=ADAPTATION_PASS,
            case_label="ADAPT_PASS",
            material=adaptation_material,
            selection_sha256=pass_selection.sha256,
            candidate=ACTION_B_CANDIDATE,
            candidate_scores=pass_selection.candidate_scores,
            expected_outputs=spec.adaptation_expected_outputs,
            execution=pass_execution,
            public_feedback_sha256=pass_feedback_root,
            lesson_sha256=pass_lesson.sha256,
            memory_updated=1,
            checkpoint_before_sha256=before_root,
            checkpoint_after_sha256=final_memory.checkpoint_sha256,
            parent_checkpoint_after_sha256=parent_after_root,
        )
    )

    evaluation_observations: list[
        tuple[
            str,
            _TaskMaterialV1,
            continual.ContinualDocumentationActionRecallV1,
            rust_executor.RustSelectedActionExecutionV2,
        ]
    ] = []
    for case_label, batch in zip(
        _CASE_LABELS[7:],
        spec.evaluation_batches,
        strict=True,
    ):
        material = _task_material(spec, case_label)
        selection, execution, vault_receipt, vault_authority = _execute_vault_selection(
            final_memory,
            material=material,
            batch=batch,
            cell_gate=cell_gate,
            documentation_gate=documentation_gate,
        )
        promoted_vault_receipts.append(vault_receipt)
        promoted_vault_authorities.append(vault_authority)
        continual_recall_selections.append(selection)
        executions.append(execution)
        evaluation_observations.append((case_label, material, selection, execution))

    expected_evaluation_candidates = (
        ACTION_B_CANDIDATE.sha256,
        ACTION_A_CANDIDATE.sha256,
    )
    if (
        tuple(
            selection.selected_candidate_sha256
            for _label, _material, selection, _execution in evaluation_observations
        )
        != expected_evaluation_candidates
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "fresh evaluation selected unexpected opaque actions"
        )

    # Both selected actions have compiled and run before this callback is invoked.
    oracle_outputs = oracle_callback()
    if type(oracle_outputs) is not tuple or len(oracle_outputs) != 2:
        raise TypeError("oracle callback must return two exact output batches")
    checked_oracle = tuple(
        _validate_outputs(
            output,
            label=f"oracle output {index}",
            expected_cases=len(spec.evaluation_batches[index]),
        )
        for index, output in enumerate(oracle_outputs)
    )
    if (
        tuple(feedback_revision_output_sha256_v1(output) for output in checked_oracle)
        != spec.evaluation_expected_output_sha256s
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "oracle outputs differ from the rooted public commitments"
        )
    oracle_observation_sha256 = _digest(
        _ORACLE_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "execution_sha256s": tuple(
                execution.sha256
                for _label, _material, _selection, execution in evaluation_observations
            ),
            "expected_output_sha256s": spec.evaluation_expected_output_sha256s,
        },
    )

    evaluation_match_cells: list[int] = []
    for offset, (
        case_label,
        material,
        selection,
        execution,
    ) in enumerate(evaluation_observations):
        expected_outputs = checked_oracle[offset]
        candidate = ACTION_B_CANDIDATE if offset == 0 else ACTION_A_CANDIDATE
        passed_cell = int(execution.outputs == expected_outputs)
        evaluation_match_cells.append(passed_cell)
        feedback_root = _digest(
            _PUBLIC_FEEDBACK_NAMESPACE,
            {
                "spec_sha256": spec.sha256,
                "case_label": case_label,
                "candidate_sha256": candidate.sha256,
                "execution_sha256": execution.sha256,
                "observed_output_sha256": feedback_revision_output_sha256_v1(execution.outputs),
                "expected_output_sha256": feedback_revision_output_sha256_v1(expected_outputs),
                "oracle_observation_sha256": oracle_observation_sha256,
                "passed": passed_cell,
            },
        )
        attempts.append(
            _make_attempt(
                ordinal=8 + offset,
                phase=EVALUATION,
                case_label=case_label,
                material=material,
                selection_sha256=selection.sha256,
                candidate=candidate,
                candidate_scores=selection.candidate_scores,
                expected_outputs=expected_outputs,
                execution=execution,
                public_feedback_sha256=feedback_root,
                lesson_sha256=None,
                memory_updated=0,
                checkpoint_before_sha256=final_memory.checkpoint_sha256,
                checkpoint_after_sha256=final_memory.checkpoint_sha256,
                parent_checkpoint_after_sha256=final_memory.checkpoint_sha256,
            )
        )

    for execution in executions:
        execution.authenticate_structure()
    if (
        len(executions) != 9
        or sum(item.compile_receipt.compile_invocations for item in executions) != 9
        or sum(item.run_receipt.run_invocations for item in executions) != 9
    ):
        raise ContinualRustFeedbackRevisionCohortError("selected-only execution work changed")
    intervention_work = prepared_intervention.work_counters
    if asdict(intervention_work) != {
        "issued_selection_count": 5,
        "spent_selection_count": 5,
        "source_release_count": 5,
        "execution_attempt_count": 5,
        "successful_execution_count": 5,
    }:
        raise ContinualRustFeedbackRevisionCohortError("training intervention work changed")
    final_memory.authenticate()
    if (
        final_memory.checkpoint_sha256 != checkpoint_lineage[-1]
        or final_memory.training_summary.lesson_count != 7
        or final_memory.training_summary.excitatory_update_count != 2
        or final_memory.training_summary.inhibitory_update_count != 0
        or any(
            attempt.checkpoint_after_sha256 != final_memory.checkpoint_sha256
            for attempt in attempts[7:]
        )
    ):
        raise ContinualRustFeedbackRevisionCohortError(
            "final continual checkpoint changed during evaluation"
        )

    match_cells = tuple(evaluation_match_cells)
    return _seal_result(
        status=PASS if match_cells == (1, 1) else FAIL_ORACLE_MISMATCH,
        spec=spec,
        initial_checkpoint_sha256=checkpoint_lineage[0],
        trained_checkpoint_sha256=trained_checkpoint_sha256,
        failed_checkpoint_sha256=failed_checkpoint_sha256,
        final_checkpoint_sha256=final_memory.checkpoint_sha256,
        checkpoint_lineage_sha256s=tuple(checkpoint_lineage),
        attempts=tuple(attempts),
        intervention_authority=prepared_intervention.authority,
        intervention_catalogue=prepared_intervention.catalogue,
        intervention_selections=tuple(intervention_selections),
        promoted_vault_receipts=tuple(promoted_vault_receipts),
        promoted_vault_authorities=tuple(promoted_vault_authorities),
        continual_recall_selections=tuple(continual_recall_selections),
        executions=tuple(executions),
        lessons=tuple(lessons),
        failure_update_off_recall=failure_update_off_recall,
        evaluation_match_cells=match_cells,
        oracle_observation_sha256=oracle_observation_sha256,
    )


__all__ = [
    "ACTION_A_CANDIDATE",
    "ACTION_B_CANDIDATE",
    "ADAPTATION_FAIL",
    "ADAPTATION_PASS",
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "EVALUATION",
    "FAIL_ORACLE_MISMATCH",
    "PASS",
    "SCHEMA",
    "STOP_CELL_OFF",
    "STOP_DOCUMENTATION_OFF",
    "TRAIN",
    "ContinualRustFeedbackRevisionCohortError",
    "FeedbackRevisionAttemptV1",
    "FeedbackRevisionCohortResultV1",
    "FeedbackRevisionCohortSpecV1",
    "FeedbackRevisionWorkV1",
    "feedback_revision_output_sha256_v1",
    "make_feedback_revision_cohort_spec_v1",
    "run_cohort_v1",
]
