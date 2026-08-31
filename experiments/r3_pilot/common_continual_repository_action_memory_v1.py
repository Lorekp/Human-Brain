"""Replay-free repository-action memory over visible source contexts.

The module is deliberately below the repository controller.  It owns stable
ordered-relation action addresses, source projection, authenticated physical
lessons, immutable flat successors, learned context recall, and the two-stage
GLIF context/action competition.  It does not import the controller and never
receives desired outputs, repository paths, test commands, or raw E/H arrays.

The context claim is narrow: exact recurring visible-source projections can
retrieve an action preference learned from real controller consequences.  The
projection is not language understanding and the environment still owns source
attestation, compilation, filesystem/process I/O, tests, and rollback.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.g0 import common_event_cells_v1 as common_events
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from humanbrain.resting_glif_tier_beam import RETAINED, CompiledRestingGlifTierBeam

SCHEMA = "COMMON_CONTINUAL_REPOSITORY_ACTION_MEMORY_V1"
RECALLED = f"{SCHEMA}_RECALLED"
STOP_RECALL_GATE = f"STOP_{SCHEMA}_RECALL_GATE"
STOP_CONTEXT_GATE = f"STOP_{SCHEMA}_CONTEXT_GATE"
STOP_CONTEXT_UNKNOWN = f"STOP_{SCHEMA}_CONTEXT_UNKNOWN"
STOP_CONTEXT_AMBIGUOUS = f"STOP_{SCHEMA}_CONTEXT_AMBIGUOUS"
STOP_CONTEXT_COLLISION = f"STOP_{SCHEMA}_CONTEXT_COLLISION"
STOP_CANDIDATE_GATE = f"STOP_{SCHEMA}_CANDIDATE_GATE"
STOP_CANDIDATE_UNKNOWN = f"STOP_{SCHEMA}_CANDIDATE_UNKNOWN"
STOP_CANDIDATE_AMBIGUOUS = f"STOP_{SCHEMA}_CANDIDATE_AMBIGUOUS"
STOP_ACTION_BANK = f"STOP_{SCHEMA}_ACTION_BANK"

OUTCOME_COMMIT = "COMMIT"
OUTCOME_REVISE = "REVISE"

MAX_CONTEXTS = 2
ACTION_COUNT = action_bank.CANDIDATE_COUNT
MAX_POLICY_BANKS = 2
MAX_GLOBAL_ACTIONS = MAX_POLICY_BANKS * ACTION_COUNT
MAX_PROVENANCE_ROWS = 8
TRACE_LANES = 12
EVIDENCE_TICKS = 4
TICK_CAP = 16

RACE_TICKS = 600
CONTEXT_STOP_CELL = MAX_CONTEXTS
CANDIDATE_STOP_CELL = ACTION_COUNT
CONTEXT_CURRENT_BASE_PA = 600.0
CONTEXT_CURRENT_GAIN_PA = 160.0
CONTEXT_CURRENT_MAX_PA = 3_000.0
CANDIDATE_CURRENT_BASE_PA = 600.0
CANDIDATE_CURRENT_GAIN_PA = 400.0
CANDIDATE_CURRENT_MAX_PA = 7_000.0
INHIBITION_CURRENT_PA = 7_000.0

FIXED_CONTROLLER_CELL_COUNT = 6
MAX_ADDED_CELL_COUNT = 12
MAX_TRAINABLE_STATE_BYTES = 16
MAX_CHECKPOINT_ARRAY_BYTES = 1_544
MAX_CONTROLLER_CELL_STEPS = 500_000

PUBLIC_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0

CLAIM_BOUNDARY = (
    "non-adversarial integration-authenticated replay-free exact recurring "
    "visible-source projection retrieval across two policy-bound ordered-relation "
    "repository action banks, with provenance "
    "and revision from authenticated controller consequences; not language "
    "understanding, broad coding, or protected benchmark evidence"
)

_SOURCE_CONTEXT_NAMESPACE = b"common-repository-source-context-v1"
_CONTROLLER_RESULT_NAMESPACE = b"common-repository-controller-result-v1"
_LESSON_BODY_NAMESPACE = b"common-repository-action-lesson-body-v1"
_LESSON_NAMESPACE = b"common-repository-action-lesson-v1"
_CANDIDATE_NAMESPACE = b"common-repository-action-candidate-v1"
_RECALL_NAMESPACE = b"common-repository-action-recall-v1"
_CHECKPOINT_NAMESPACE = b"common-continual-repository-action-checkpoint-v1"
_PHYSICAL_TIER_NAMESPACE = b"common-repository-action-physical-tier-v1"

_SOURCE_CELLS = common_events.CommonEventCells()
_CONTEXT_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=MAX_CONTEXTS + 1,
    step_count=RACE_TICKS,
    min_current_pa=CONTEXT_CURRENT_BASE_PA,
    max_current_pa=CONTEXT_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_CANDIDATE_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=ACTION_COUNT + 1,
    step_count=RACE_TICKS,
    min_current_pa=CANDIDATE_CURRENT_BASE_PA,
    max_current_pa=CANDIDATE_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_CONSTRUCTION_TOKEN = object()

_CHECKPOINT_ARRAY_NAMES = (
    "policy-roots",
    "action-roots",
    "action-policy-indices",
    "context-cues",
    "context-frame-roots",
    "excitatory",
    "inhibitory",
    "provenance-context-indices",
    "provenance-action-indices",
    "provenance-passed",
    "provenance-ordinals",
    "provenance-result-roots",
    "provenance-lesson-roots",
    "provenance-program-roots",
    "provenance-consequence-roots",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(value.tobytes(order="C")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return action_bank.valid_sha256(value)


class RepositorySourceContextV1(NamedTuple):
    source_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.frame_sha256)
            or type(self.trace_cells) is not tuple
            or len(self.trace_cells) != TRACE_LANES
            or any(type(cell) is not int for cell in self.trace_cells)
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps < 0
        ):
            raise ValueError("repository source context shape changed")
        expected = _digest(
            _SOURCE_CONTEXT_NAMESPACE,
            {
                "source": self.source_sha256,
                "frame": self.frame_sha256,
                "trace": self.trace_cells,
                "projection_cell_steps": self.projection_cell_steps,
            },
        )
        if self.sha256 != expected:
            raise ValueError("repository source context authentication failed")


def project_attested_repository_source_v1(source: bytes) -> RepositorySourceContextV1:
    """Project one already-attested visible source; retain no raw bytes."""

    if type(source) is not bytes:
        raise TypeError("repository source must be exact bytes")
    frame = _SOURCE_CELLS.project(source)
    result = RepositorySourceContextV1(
        source_sha256=_bytes_sha256(source),
        frame_sha256=frame.sha256,
        trace_cells=frame.trace_cells,
        projection_cell_steps=frame.work.total,
        sha256=_digest(
            _SOURCE_CONTEXT_NAMESPACE,
            {
                "source": _bytes_sha256(source),
                "frame": frame.sha256,
                "trace": frame.trace_cells,
                "projection_cell_steps": frame.work.total,
            },
        ),
    )
    result.authenticate()
    return result


class _RepositoryActionConsequenceV1(NamedTuple):
    candidate_sha256: str
    applied_relation_receipt_sha256s: tuple[str, ...]
    program_sha256: str
    attempt_sha256: str
    consequence_sha256: str
    passed_cell: int
    outcome_key: str
    rolled_back: bool
    committed: bool

    def authenticate(self) -> None:
        action_bank.canonical_repository_action_receipts_v1(
            self.applied_relation_receipt_sha256s
        )
        expected_outcome = OUTCOME_COMMIT if self.passed_cell == 1 else OUTCOME_REVISE
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.candidate_sha256,
                    self.program_sha256,
                    self.attempt_sha256,
                    self.consequence_sha256,
                )
            )
            or self.passed_cell not in (0, 1)
            or self.outcome_key != expected_outcome
            or self.rolled_back != (self.passed_cell == 0)
            or self.committed != (self.passed_cell == 1)
        ):
            raise ValueError("repository action consequence shape changed")


class PhysicalRepositoryActionLessonV1(NamedTuple):
    policy_checkpoint_sha256: str
    controller_trajectory_sha256: str
    controller_result_sha256: str
    controller_lesson_body_sha256s: tuple[str, ...]
    controller_attempt_sha256s: tuple[str, ...]
    controller_status: str
    controller_committed: bool
    controller_final_passed_cell: int
    frontier_sha256: str
    action_bank_sha256: str
    action_bank_sha256s: tuple[str, ...]
    candidate_sha256: str
    candidate_action_sha256: str
    applied_relation_receipt_sha256s: tuple[str, ...]
    program_sha256: str
    source_context_sha256: str
    source_sha256: str
    source_context_frame_sha256: str
    source_context_trace_cells: tuple[int, ...]
    source_projection_cell_steps: int
    attempt_ordinal: int
    attempt_count: int
    attempt_sha256: str
    consequence_sha256: str
    passed_cell: int
    outcome_key: str
    rolled_back: bool
    committed: bool
    lesson_body_sha256: str
    sha256: str

    def authenticate(self) -> None:
        receipts = action_bank.canonical_repository_action_receipts_v1(
            self.applied_relation_receipt_sha256s
        )
        bank = tuple(sorted(self.action_bank_sha256s))
        bodies = self.controller_lesson_body_sha256s
        attempts = self.controller_attempt_sha256s
        RepositorySourceContextV1(
            source_sha256=self.source_sha256,
            frame_sha256=self.source_context_frame_sha256,
            trace_cells=self.source_context_trace_cells,
            projection_cell_steps=self.source_projection_cell_steps,
            sha256=self.source_context_sha256,
        ).authenticate()
        _RepositoryActionConsequenceV1(
            candidate_sha256=self.candidate_sha256,
            applied_relation_receipt_sha256s=receipts,
            program_sha256=self.program_sha256,
            attempt_sha256=self.attempt_sha256,
            consequence_sha256=self.consequence_sha256,
            passed_cell=self.passed_cell,
            outcome_key=self.outcome_key,
            rolled_back=self.rolled_back,
            committed=self.committed,
        ).authenticate()
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.policy_checkpoint_sha256,
                    self.controller_trajectory_sha256,
                    self.controller_result_sha256,
                    self.frontier_sha256,
                    self.action_bank_sha256,
                    self.candidate_action_sha256,
                    self.lesson_body_sha256,
                )
            )
            or type(self.controller_status) is not str
            or not self.controller_status
            or self.controller_committed is not True
            or self.controller_final_passed_cell != 1
            or len(bank) != ACTION_COUNT
            or len(set(bank)) != ACTION_COUNT
            or bank != self.action_bank_sha256s
            or any(not _valid_sha256(value) for value in bank)
            or self.candidate_action_sha256 not in bank
            or self.candidate_action_sha256
            != action_bank.repository_action_sha256_v1(
                self.policy_checkpoint_sha256,
                receipts,
            )
            or type(bodies) is not tuple
            or len(bodies) != self.attempt_count
            or len(set(bodies)) != self.attempt_count
            or any(not _valid_sha256(value) for value in bodies)
            or type(attempts) is not tuple
            or len(attempts) != self.attempt_count
            or len(set(attempts)) != self.attempt_count
            or any(not _valid_sha256(value) for value in attempts)
            or not 1 <= self.attempt_ordinal <= self.attempt_count <= 2
            or bodies[self.attempt_ordinal - 1] != self.lesson_body_sha256
            or attempts[self.attempt_ordinal - 1] != self.attempt_sha256
            or self.lesson_body_sha256 != _lesson_body_sha256(self._asdict())
            or self.controller_result_sha256
            != controller_result_sha256_v1(
                self.controller_trajectory_sha256,
                bodies,
            )
            or self.sha256 != _lesson_sha256(self._asdict())
        ):
            raise ValueError("repository action lesson authentication failed")


def _lesson_body_sha256(record: dict[str, object]) -> str:
    payload = dict(record)
    for key in (
        "controller_result_sha256",
        "controller_lesson_body_sha256s",
        "lesson_body_sha256",
        "sha256",
    ):
        payload.pop(key, None)
    return _digest(_LESSON_BODY_NAMESPACE, payload)


def _lesson_sha256(record: dict[str, object]) -> str:
    payload = dict(record)
    payload.pop("sha256", None)
    return _digest(_LESSON_NAMESPACE, payload)


def controller_result_sha256_v1(
    controller_trajectory_sha256: str,
    lesson_body_sha256s: Sequence[str],
) -> str:
    bodies = tuple(lesson_body_sha256s)
    if (
        not _valid_sha256(controller_trajectory_sha256)
        or not 1 <= len(bodies) <= 2
        or len(set(bodies)) != len(bodies)
        or any(not _valid_sha256(value) for value in bodies)
    ):
        raise ValueError("controller result requires one ordered lesson-body cohort")
    return _digest(
        _CONTROLLER_RESULT_NAMESPACE,
        {
            "trajectory": controller_trajectory_sha256,
            "lesson_bodies": bodies,
        },
    )


def _seal_physical_repository_action_lessons_v1(
    *,
    controller_trajectory_sha256: str,
    controller_status: str,
    action_bank: action_bank.FrozenRepositoryActionBankV1,
    source_context: RepositorySourceContextV1,
    consequences: tuple[_RepositoryActionConsequenceV1, ...],
) -> tuple[str, tuple[PhysicalRepositoryActionLessonV1, ...]]:
    """Controller-internal seam: seal one actual trajectory without a cycle."""

    action_bank.authenticate()
    source_context.authenticate()
    if type(consequences) is not tuple or not 1 <= len(consequences) <= 2:
        raise TypeError("controller consequences must be one exact nonempty tuple")
    for consequence in consequences:
        if type(consequence) is not _RepositoryActionConsequenceV1:
            raise TypeError("exact repository action consequences required")
        consequence.authenticate()
    if (
        consequences[-1].passed_cell != 1
        or consequences[-1].committed is not True
        or consequences[-1].rolled_back is not False
        or any(
            consequence.passed_cell != 0
            or consequence.rolled_back is not True
            or consequence.committed is not False
            for consequence in consequences[:-1]
        )
    ):
        raise ValueError("only one complete committed controller trajectory is learnable")
    policy_checkpoint_sha256 = action_bank.policy_checkpoint_sha256
    frontier_sha256 = action_bank.frontier_sha256
    bank = tuple(sorted(action_bank.action_sha256s))
    candidates = {item.sha256: item for item in action_bank.records}
    for consequence in consequences:
        candidate = candidates.get(consequence.candidate_sha256)
        if (
            candidate is None
            or candidate.program_sha256 != consequence.program_sha256
            or candidate.applied_relation_receipt_sha256s
            != consequence.applied_relation_receipt_sha256s
        ):
            raise ValueError("controller consequence is outside the authenticated bank")
    attempt_count = len(consequences)
    attempt_roots = tuple(item.attempt_sha256 for item in consequences)
    body_payloads: list[dict[str, object]] = []
    body_roots: list[str] = []
    for offset, consequence in enumerate(consequences):
        receipts = action_bank.canonical_repository_action_receipts_v1(
            consequence.applied_relation_receipt_sha256s
        )
        payload: dict[str, object] = {
            "policy_checkpoint_sha256": policy_checkpoint_sha256,
            "controller_trajectory_sha256": controller_trajectory_sha256,
            "controller_status": controller_status,
            "controller_committed": True,
            "controller_final_passed_cell": 1,
            "controller_attempt_sha256s": attempt_roots,
            "frontier_sha256": frontier_sha256,
            "action_bank_sha256": action_bank.sha256,
            "action_bank_sha256s": bank,
            "candidate_sha256": consequence.candidate_sha256,
            "candidate_action_sha256": action_bank.repository_action_sha256_v1(
                policy_checkpoint_sha256,
                receipts,
            ),
            "applied_relation_receipt_sha256s": receipts,
            "program_sha256": consequence.program_sha256,
            "source_context_sha256": source_context.sha256,
            "source_sha256": source_context.source_sha256,
            "source_context_frame_sha256": source_context.frame_sha256,
            "source_context_trace_cells": source_context.trace_cells,
            "source_projection_cell_steps": source_context.projection_cell_steps,
            "attempt_ordinal": offset + 1,
            "attempt_count": attempt_count,
            "attempt_sha256": consequence.attempt_sha256,
            "consequence_sha256": consequence.consequence_sha256,
            "passed_cell": consequence.passed_cell,
            "outcome_key": consequence.outcome_key,
            "rolled_back": consequence.rolled_back,
            "committed": consequence.committed,
        }
        body_payloads.append(payload)
        body_roots.append(_digest(_LESSON_BODY_NAMESPACE, payload))
    bodies = tuple(body_roots)
    controller_result = controller_result_sha256_v1(
        controller_trajectory_sha256,
        bodies,
    )
    lessons: list[PhysicalRepositoryActionLessonV1] = []
    for payload, body in zip(body_payloads, bodies, strict=True):
        partial = PhysicalRepositoryActionLessonV1(
            **payload,
            controller_result_sha256=controller_result,
            controller_lesson_body_sha256s=bodies,
            lesson_body_sha256=body,
            sha256="",
        )
        lesson = partial._replace(sha256=_lesson_sha256(partial._asdict()))
        lesson.authenticate()
        lessons.append(lesson)
    return controller_result, tuple(lessons)


class RepositoryActionCandidateV1(NamedTuple):
    candidate_sha256: str
    candidate_action_sha256: str
    program_sha256: str
    cell_address: int
    excitatory_ticks: int
    inhibitory_ticks: int
    sha256: str

    def authenticate(self) -> None:
        payload = {
            "candidate": self.candidate_sha256,
            "action": self.candidate_action_sha256,
            "program": self.program_sha256,
            "address": self.cell_address,
            "excitatory": self.excitatory_ticks,
            "inhibitory": self.inhibitory_ticks,
        }
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.candidate_sha256,
                    self.candidate_action_sha256,
                    self.program_sha256,
                )
            )
            or type(self.cell_address) is not int
            or not 0 <= self.cell_address < 2**32
            or not 0 <= self.excitatory_ticks <= TICK_CAP
            or not 0 <= self.inhibitory_ticks <= TICK_CAP
            or self.sha256 != _digest(_CANDIDATE_NAMESPACE, payload)
        ):
            raise ValueError("repository action candidate authentication failed")


class RepositoryActionRecallWorkV1(NamedTuple):
    source_projection_cell_steps: int
    context_equality_reads: int
    context_equality_arrivals: int
    context_race_cell_steps: int
    candidate_race_cell_steps: int

    @property
    def total_cell_steps(self) -> int:
        return (
            self.source_projection_cell_steps
            + self.context_race_cell_steps
            + self.candidate_race_cell_steps
        )


class RepositoryActionRecallV1(NamedTuple):
    status: str
    memory_checkpoint_sha256: str
    policy_checkpoint_sha256: str
    source_sha256: str
    source_context_frame_sha256: str
    source_context_trace_cells: tuple[int, ...]
    frontier_sha256: str
    action_bank_sha256s: tuple[str, ...]
    candidates: tuple[RepositoryActionCandidateV1, ...]
    selected_candidate_action_sha256: str | None
    context_index: int | None
    context_tier_sha256: str
    candidate_tier_sha256: str | None
    work: RepositoryActionRecallWorkV1
    sha256: str

    def authenticate(self) -> None:
        for candidate in self.candidates:
            candidate.authenticate()
        action_keys = tuple(item.candidate_action_sha256 for item in self.candidates)
        candidate_stage = self.status in {
            RECALLED,
            STOP_CANDIDATE_GATE,
            STOP_CANDIDATE_UNKNOWN,
            STOP_CANDIDATE_AMBIGUOUS,
        }
        if (
            not _valid_sha256(self.memory_checkpoint_sha256)
            or not _valid_sha256(self.policy_checkpoint_sha256)
            or not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.source_context_frame_sha256)
            or not _valid_sha256(self.frontier_sha256)
            or self.action_bank_sha256s != tuple(sorted(self.action_bank_sha256s))
            or len(self.action_bank_sha256s) != ACTION_COUNT
            or len(set(self.action_bank_sha256s)) != ACTION_COUNT
            or (
                candidate_stage
                and (
                    tuple(sorted(action_keys)) != self.action_bank_sha256s
                    or len(set(action_keys)) != ACTION_COUNT
                )
            )
            or (not candidate_stage and self.candidates != ())
            or (
                self.status == RECALLED
                and self.selected_candidate_action_sha256 not in action_keys
            )
            or (
                self.status != RECALLED
                and self.selected_candidate_action_sha256 is not None
            )
        ):
            raise ValueError("repository action recall shape changed")
        payload = _recall_payload(self._asdict())
        if self.sha256 != _digest(_RECALL_NAMESPACE, payload):
            raise ValueError("repository action recall authentication failed")


def _recall_payload(record: dict[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload.pop("sha256", None)
    payload["candidates"] = tuple(item.sha256 for item in record["candidates"])
    work = record["work"]
    if type(work) is RepositoryActionRecallWorkV1:
        payload["work"] = work._asdict()
    return payload


class RepositoryActionTrainingSummaryV1(NamedTuple):
    mode: str
    policy_checkpoint_sha256s: tuple[str, ...]
    parent_checkpoint_sha256: str | None
    policy_count: int
    context_count: int
    action_count: int
    provenance_count: int
    new_lesson_count: int
    parent_lesson_replay_count: int
    retained_parent_count: int
    raw_lesson_retention_count: int
    parent_mutation_count: int
    added_cell_count: int
    trainable_state_bytes: int
    provenance_state_bytes: int
    checkpoint_array_bytes: int
    learning_cell_steps: int
    checkpoint_sha256: str


class RepositoryActionProvenanceV1(NamedTuple):
    context_index: int
    action_index: int
    passed_cell: int
    attempt_ordinal: int
    controller_result_sha256: str
    lesson_sha256: str
    program_sha256: str
    consequence_sha256: str


def _freeze_array(value: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _physical_tier_sha256(value: object) -> str:
    return _digest(
        _PHYSICAL_TIER_NAMESPACE,
        {
            "status": value.status,
            "retained_cells": value.retained_cells,
            "retained_candidates": value.retained_candidate_cells,
            "stop_cell": value.stop_cell,
            "rounds": tuple(
                {
                    "tier": item.tier_index,
                    "winner": item.race.winner_cell,
                    "spike": item.race.spike_step,
                    "simultaneous": item.race.simultaneous_spike_candidates,
                    "cells": item.tier_cells,
                    "leading_stop": item.leading_stop,
                }
                for item in value.rounds
            ),
            "projection_before": value.inhibition_projection_before_sha256,
            "projection_after": value.inhibition_projection_after_sha256,
        },
    )


def _checkpoint_sha256(
    summary: RepositoryActionTrainingSummaryV1,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    payload = summary._asdict()
    payload.pop("checkpoint_sha256")
    payload["arrays"] = tuple(
        (name, array.shape, array.dtype.str, _array_sha256(array))
        for name, array in arrays
    )
    return _digest(_CHECKPOINT_NAMESPACE, payload)


class FrozenContinualRepositoryActionMemoryV1:
    """One flat immutable context/action checkpoint; no parent or raw lesson."""

    __slots__ = (
        "_policy_roots",
        "_action_roots",
        "_action_policy_indices",
        "_checkpoint_sha256",
        "_context_cues",
        "_context_frame_roots",
        "_excitatory",
        "_inhibitory",
        "_provenance_action_indices",
        "_provenance_consequence_roots",
        "_provenance_context_indices",
        "_provenance_lesson_roots",
        "_provenance_ordinals",
        "_provenance_passed",
        "_provenance_program_roots",
        "_provenance_result_roots",
        "_summary",
    )

    def __init__(
        self,
        *,
        policy_roots: np.ndarray,
        action_roots: np.ndarray,
        action_policy_indices: np.ndarray,
        context_cues: np.ndarray,
        context_frame_roots: np.ndarray,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        provenance_context_indices: np.ndarray,
        provenance_action_indices: np.ndarray,
        provenance_passed: np.ndarray,
        provenance_ordinals: np.ndarray,
        provenance_result_roots: np.ndarray,
        provenance_lesson_roots: np.ndarray,
        provenance_program_roots: np.ndarray,
        provenance_consequence_roots: np.ndarray,
        summary: RepositoryActionTrainingSummaryV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("continual repository action memories are builder-owned")
        self._policy_roots = _freeze_array(policy_roots, np.uint8)
        self._action_roots = _freeze_array(action_roots, np.uint8)
        self._action_policy_indices = _freeze_array(
            action_policy_indices, np.uint8
        )
        self._context_cues = _freeze_array(context_cues, "<u4")
        self._context_frame_roots = _freeze_array(context_frame_roots, np.uint8)
        self._excitatory = _freeze_array(excitatory, np.uint8)
        self._inhibitory = _freeze_array(inhibitory, np.uint8)
        self._provenance_context_indices = _freeze_array(
            provenance_context_indices, np.uint8
        )
        self._provenance_action_indices = _freeze_array(
            provenance_action_indices, np.uint8
        )
        self._provenance_passed = _freeze_array(provenance_passed, np.uint8)
        self._provenance_ordinals = _freeze_array(provenance_ordinals, np.uint8)
        self._provenance_result_roots = _freeze_array(
            provenance_result_roots, np.uint8
        )
        self._provenance_lesson_roots = _freeze_array(
            provenance_lesson_roots, np.uint8
        )
        self._provenance_program_roots = _freeze_array(
            provenance_program_roots, np.uint8
        )
        self._provenance_consequence_roots = _freeze_array(
            provenance_consequence_roots, np.uint8
        )
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(summary, self._arrays())
        self.authenticate()

    def _arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        arrays = (
            self._policy_roots,
            self._action_roots,
            self._action_policy_indices,
            self._context_cues,
            self._context_frame_roots,
            self._excitatory,
            self._inhibitory,
            self._provenance_context_indices,
            self._provenance_action_indices,
            self._provenance_passed,
            self._provenance_ordinals,
            self._provenance_result_roots,
            self._provenance_lesson_roots,
            self._provenance_program_roots,
            self._provenance_consequence_roots,
        )
        return tuple(zip(_CHECKPOINT_ARRAY_NAMES, arrays, strict=True))

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RepositoryActionTrainingSummaryV1:
        self.authenticate()
        return self._summary

    @property
    def provenance(self) -> tuple[RepositoryActionProvenanceV1, ...]:
        self.authenticate()
        return tuple(
            RepositoryActionProvenanceV1(
                context_index=int(self._provenance_context_indices[row]),
                action_index=int(self._provenance_action_indices[row]),
                passed_cell=int(self._provenance_passed[row]),
                attempt_ordinal=int(self._provenance_ordinals[row]),
                controller_result_sha256=bytes(
                    self._provenance_result_roots[row]
                ).hex(),
                lesson_sha256=bytes(self._provenance_lesson_roots[row]).hex(),
                program_sha256=bytes(self._provenance_program_roots[row]).hex(),
                consequence_sha256=bytes(
                    self._provenance_consequence_roots[row]
                ).hex(),
            )
            for row in range(self._provenance_passed.shape[0])
        )

    def authenticate(self) -> None:
        arrays = self._arrays()
        if any(array.flags.writeable for _, array in arrays):
            raise RuntimeError("continual repository action checkpoint became mutable")
        policies = self._policy_roots.shape[0]
        actions = self._action_roots.shape[0]
        contexts = self._context_cues.shape[0]
        rows = self._provenance_passed.shape[0]
        expected_shapes = (
            (policies, 32),
            (actions, 32),
            (actions,),
            (contexts, TRACE_LANES),
            (contexts, 32),
            (contexts, actions),
            (contexts, actions),
            (rows,),
            (rows,),
            (rows,),
            (rows,),
            (rows, 32),
            (rows, 32),
            (rows, 32),
            (rows, 32),
        )
        if tuple(array.shape for _, array in arrays) != expected_shapes:
            raise RuntimeError("continual repository action checkpoint shape changed")
        if (
            not 1 <= policies <= MAX_POLICY_BANKS
            or actions != policies * ACTION_COUNT
            or actions > MAX_GLOBAL_ACTIONS
            or not 1 <= contexts <= MAX_CONTEXTS
            or not 1 <= rows <= MAX_PROVENANCE_ROWS
            or len({bytes(row) for row in self._policy_roots}) != policies
            or len({bytes(row) for row in self._action_roots}) != actions
            or tuple(int(value) for value in self._action_policy_indices)
            != tuple(
                policy_index
                for policy_index in range(policies)
                for _ in range(ACTION_COUNT)
            )
            or len({tuple(row) for row in self._context_cues.tolist()}) != contexts
            or np.any(self._excitatory > TICK_CAP)
            or np.any(self._inhibitory > TICK_CAP)
            or np.any(self._provenance_context_indices >= contexts)
            or np.any(self._provenance_action_indices >= actions)
            or np.any(self._provenance_passed > 1)
            or np.any(self._provenance_ordinals < 1)
            or np.any(self._provenance_ordinals > 2)
            or len({bytes(row) for row in self._provenance_lesson_roots}) != rows
        ):
            raise RuntimeError("continual repository action checkpoint values changed")
        trainable = int(self._excitatory.nbytes + self._inhibitory.nbytes)
        checkpoint_bytes = sum(array.nbytes for _, array in arrays)
        provenance = sum(array.nbytes for _, array in arrays[7:])
        summary = self._summary
        policy_roots = tuple(bytes(row).hex() for row in self._policy_roots)
        if (
            type(summary.mode) is not str
            or not summary.mode
            or summary.policy_checkpoint_sha256s != policy_roots
            or any(not _valid_sha256(value) for value in policy_roots)
            or (
                summary.parent_checkpoint_sha256 is not None
                and not _valid_sha256(summary.parent_checkpoint_sha256)
            )
            or summary.policy_count != policies
            or summary.context_count != contexts
            or summary.action_count != actions
            or summary.provenance_count != rows
            or not 1 <= summary.new_lesson_count <= 2
            or summary.parent_lesson_replay_count != 0
            or summary.retained_parent_count != 0
            or summary.raw_lesson_retention_count != 0
            or summary.parent_mutation_count != 0
            or summary.added_cell_count
            != FIXED_CONTROLLER_CELL_COUNT + contexts + actions
            or summary.added_cell_count > MAX_ADDED_CELL_COUNT
            or summary.trainable_state_bytes != trainable
            or summary.provenance_state_bytes != provenance
            or summary.checkpoint_array_bytes != checkpoint_bytes
            or trainable > MAX_TRAINABLE_STATE_BYTES
            or checkpoint_bytes > MAX_CHECKPOINT_ARRAY_BYTES
            or summary.learning_cell_steps > MAX_CONTROLLER_CELL_STEPS
            or summary.checkpoint_sha256 != self._checkpoint_sha256
            or _checkpoint_sha256(summary, arrays) != self._checkpoint_sha256
        ):
            raise RuntimeError("continual repository action checkpoint authentication failed")

    def recall_attested_context(
        self,
        context: RepositorySourceContextV1,
        bank: action_bank.FrozenRepositoryActionBankV1,
        *,
        candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
        | None = None,
        recall_gate: bool = True,
        context_gate: bool = True,
        candidate_gate: bool = True,
    ) -> RepositoryActionRecallV1:
        self.authenticate()
        if type(context) is not RepositorySourceContextV1:
            raise TypeError("repository action recall requires one exact source context")
        if type(bank) is not action_bank.FrozenRepositoryActionBankV1:
            raise TypeError("repository action recall requires one exact action bank")
        context.authenticate()
        bank.authenticate()
        if any(
            type(gate) is not bool
            for gate in (recall_gate, context_gate, candidate_gate)
        ):
            raise TypeError("repository action recall gates must be exact booleans")
        authentic = bank.records
        supplied = authentic if candidate_records is None else tuple(candidate_records)
        canonical = {item.sha256: item for item in authentic}
        records_authentic = (
            len(supplied) == ACTION_COUNT
            and len(canonical) == ACTION_COUNT
            and all(
                type(item) is action_bank.RepositoryActionCandidateRecordV1
                for item in supplied
            )
            and len({item.sha256 for item in supplied}) == ACTION_COUNT
            and all(
                canonical.get(item.sha256) == item
                for item in supplied
            )
        )
        policy = bank.policy_checkpoint_sha256
        candidate_rows: list[
            tuple[str, action_bank.RepositoryActionCandidateRecordV1]
        ] = []
        if records_authentic:
            candidate_rows = sorted(
                (
                    bank.action_sha256_for(item),
                    item,
                )
                for item in supplied
            )
        policy_matches = tuple(
            index
            for index, row in enumerate(self._policy_roots)
            if bytes(row).hex() == policy
        )
        supplied_actions = tuple(row[0] for row in candidate_rows)
        memory_action_indices = {
            bytes(row).hex(): index for index, row in enumerate(self._action_roots)
        }
        policy_index = policy_matches[0] if len(policy_matches) == 1 else None
        bank_valid = (
            records_authentic
            and policy_index is not None
            and tuple(sorted(supplied_actions)) == tuple(sorted(bank.action_sha256s))
            and len(set(supplied_actions)) == ACTION_COUNT
            and all(action in memory_action_indices for action in supplied_actions)
            and all(
                int(self._action_policy_indices[memory_action_indices[action]])
                == policy_index
                for action in supplied_actions
            )
        )

        arrivals = np.count_nonzero(
            self._context_cues
            == np.asarray(context.trace_cells, dtype="<u4"),
            axis=1,
        ).astype(np.int16)
        context_currents = np.full(
            MAX_CONTEXTS + 1,
            CONTEXT_CURRENT_BASE_PA,
            dtype=np.float64,
        )
        if recall_gate and context_gate:
            context_currents[: self._context_cues.shape[0]] += (
                CONTEXT_CURRENT_GAIN_PA * arrivals
            )
        context_currents[CONTEXT_STOP_CELL] += CONTEXT_CURRENT_GAIN_PA * (
            TRACE_LANES - 1
        )
        context_physical = _CONTEXT_SELECTOR.select(
            context_currents,
            stop_cell=CONTEXT_STOP_CELL,
        )
        context_tier = _physical_tier_sha256(context_physical)
        context_index: int | None = None
        candidate_tier: str | None = None
        selected_action: str | None = None
        candidates: tuple[RepositoryActionCandidateV1, ...] = ()
        candidate_steps = 0

        if not recall_gate:
            status = STOP_RECALL_GATE
        elif not bank_valid:
            status = STOP_ACTION_BANK
        elif not context_gate:
            status = STOP_CONTEXT_GATE
        elif context_physical.status != RETAINED:
            status = STOP_CONTEXT_AMBIGUOUS
        elif len(context_physical.retained_candidate_cells) != 1:
            status = STOP_CONTEXT_AMBIGUOUS
        elif context_physical.retained_candidate_cells[0] == CONTEXT_STOP_CELL:
            status = STOP_CONTEXT_UNKNOWN
        elif not 0 <= context_physical.retained_candidate_cells[0] < self._context_cues.shape[0]:
            status = STOP_CONTEXT_UNKNOWN
        elif (
            bytes(
                self._context_frame_roots[
                    context_physical.retained_candidate_cells[0]
                ]
            ).hex()
            != context.frame_sha256
        ):
            status = STOP_CONTEXT_COLLISION
        else:
            context_index = context_physical.retained_candidate_cells[0]
            candidates_list: list[RepositoryActionCandidateV1] = []
            for action_sha256, record in candidate_rows:
                action_index = memory_action_indices[action_sha256]
                excitatory = int(self._excitatory[context_index, action_index])
                inhibitory = int(self._inhibitory[context_index, action_index])
                address = int.from_bytes(bytes.fromhex(action_sha256)[:4], "little")
                payload = {
                    "candidate": record.sha256,
                    "action": action_sha256,
                    "program": record.program_sha256,
                    "address": address,
                    "excitatory": excitatory,
                    "inhibitory": inhibitory,
                }
                candidates_list.append(
                    RepositoryActionCandidateV1(
                        candidate_sha256=record.sha256,
                        candidate_action_sha256=action_sha256,
                        program_sha256=record.program_sha256,
                        cell_address=address,
                        excitatory_ticks=excitatory,
                        inhibitory_ticks=inhibitory,
                        sha256=_digest(_CANDIDATE_NAMESPACE, payload),
                    )
                )
            candidates = tuple(candidates_list)
            candidate_currents = np.full(
                ACTION_COUNT + 1,
                CANDIDATE_CURRENT_BASE_PA,
                dtype=np.float64,
            )
            if candidate_gate:
                for index, candidate in enumerate(candidates):
                    drive = candidate.excitatory_ticks - candidate.inhibitory_ticks
                    candidate_currents[index] += CANDIDATE_CURRENT_GAIN_PA * max(
                        0, drive
                    )
            else:
                candidate_currents[CANDIDATE_STOP_CELL] += CANDIDATE_CURRENT_GAIN_PA
            candidate_physical = _CANDIDATE_SELECTOR.select(
                candidate_currents,
                stop_cell=CANDIDATE_STOP_CELL,
            )
            candidate_tier = _physical_tier_sha256(candidate_physical)
            candidate_steps = (ACTION_COUNT + 1) * RACE_TICKS
            if not candidate_gate:
                status = STOP_CANDIDATE_GATE
            elif candidate_physical.status != RETAINED:
                status = STOP_CANDIDATE_AMBIGUOUS
            elif len(candidate_physical.retained_candidate_cells) != 1:
                status = STOP_CANDIDATE_AMBIGUOUS
            elif candidate_physical.retained_candidate_cells[0] == CANDIDATE_STOP_CELL:
                status = STOP_CANDIDATE_UNKNOWN
            elif not 0 <= candidate_physical.retained_candidate_cells[0] < ACTION_COUNT:
                status = STOP_CANDIDATE_UNKNOWN
            else:
                status = RECALLED
                selected_action = candidates[
                    candidate_physical.retained_candidate_cells[0]
                ].candidate_action_sha256

        work = RepositoryActionRecallWorkV1(
            source_projection_cell_steps=context.projection_cell_steps,
            context_equality_reads=self._context_cues.shape[0] * TRACE_LANES,
            context_equality_arrivals=int(np.sum(arrivals, dtype=np.int64)),
            context_race_cell_steps=(MAX_CONTEXTS + 1) * RACE_TICKS,
            candidate_race_cell_steps=candidate_steps,
        )
        partial = RepositoryActionRecallV1(
            status=status,
            memory_checkpoint_sha256=self._checkpoint_sha256,
            policy_checkpoint_sha256=policy,
            source_sha256=context.source_sha256,
            source_context_frame_sha256=context.frame_sha256,
            source_context_trace_cells=context.trace_cells,
            frontier_sha256=bank.frontier_sha256,
            action_bank_sha256s=tuple(sorted(bank.action_sha256s)),
            candidates=candidates,
            selected_candidate_action_sha256=selected_action,
            context_index=context_index,
            context_tier_sha256=context_tier,
            candidate_tier_sha256=candidate_tier,
            work=work,
            sha256="",
        )
        result = partial._replace(
            sha256=_digest(_RECALL_NAMESPACE, _recall_payload(partial._asdict()))
        )
        result.authenticate()
        self.authenticate()
        return result


def _validate_lessons(lessons: tuple[PhysicalRepositoryActionLessonV1, ...]) -> None:
    if type(lessons) is not tuple or not lessons:
        raise TypeError("repository action lessons must be one nonempty exact tuple")
    if len(lessons) > 2:
        raise ValueError("one repository contributes at most two action lessons")
    for lesson in lessons:
        if type(lesson) is not PhysicalRepositoryActionLessonV1:
            raise TypeError("exact physical repository action lessons required")
        lesson.authenticate()
    if len({lesson.lesson_body_sha256 for lesson in lessons}) != len(lessons):
        raise ValueError("repository action lesson replay is forbidden")
    first = lessons[0]
    shared = (
        first.policy_checkpoint_sha256,
        first.controller_trajectory_sha256,
        first.controller_result_sha256,
        first.controller_lesson_body_sha256s,
        first.controller_attempt_sha256s,
        first.controller_status,
        first.frontier_sha256,
        first.action_bank_sha256,
        first.action_bank_sha256s,
        first.source_context_sha256,
        first.attempt_count,
    )
    if any(
        (
            lesson.policy_checkpoint_sha256,
            lesson.controller_trajectory_sha256,
            lesson.controller_result_sha256,
            lesson.controller_lesson_body_sha256s,
            lesson.controller_attempt_sha256s,
            lesson.controller_status,
            lesson.frontier_sha256,
            lesson.action_bank_sha256,
            lesson.action_bank_sha256s,
            lesson.source_context_sha256,
            lesson.attempt_count,
        )
        != shared
        for lesson in lessons
    ):
        raise ValueError("one advance may consume only one controller result cohort")
    result_ordinals = tuple(
        (lesson.controller_result_sha256, lesson.attempt_ordinal)
        for lesson in lessons
    )
    if (
        len(set(result_ordinals)) != len(result_ordinals)
        or tuple(lesson.attempt_ordinal for lesson in lessons)
        != tuple(sorted(lesson.attempt_ordinal for lesson in lessons))
    ):
        raise ValueError("repository action lessons must preserve unique consequence order")


def _mutable_arrays(
    parent: FrozenContinualRepositoryActionMemoryV1 | None,
    lessons: tuple[PhysicalRepositoryActionLessonV1, ...],
) -> tuple[list[np.ndarray], int]:
    policy = lessons[0].policy_checkpoint_sha256
    bank = tuple(sorted(lessons[0].action_bank_sha256s))
    if any(
        lesson.policy_checkpoint_sha256 != policy
        or lesson.action_bank_sha256 != lessons[0].action_bank_sha256
        or tuple(sorted(lesson.action_bank_sha256s)) != bank
        for lesson in lessons
    ):
        raise ValueError("repository action lesson policy/bank changed")
    if parent is None:
        arrays = [
            np.asarray([tuple(bytes.fromhex(policy))], dtype=np.uint8),
            np.asarray(
                [tuple(bytes.fromhex(value)) for value in bank],
                dtype=np.uint8,
            ),
            np.zeros((ACTION_COUNT,), dtype=np.uint8),
            np.empty((0, TRACE_LANES), dtype="<u4"),
            np.empty((0, 32), dtype=np.uint8),
            np.empty((0, ACTION_COUNT), dtype=np.uint8),
            np.empty((0, ACTION_COUNT), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 32), dtype=np.uint8),
            np.empty((0, 32), dtype=np.uint8),
            np.empty((0, 32), dtype=np.uint8),
            np.empty((0, 32), dtype=np.uint8),
        ]
        prior_steps = 0
    else:
        arrays = [array.copy() for _, array in parent._arrays()]
        prior_steps = parent.training_summary.learning_cell_steps
        policy_roots = tuple(bytes(row).hex() for row in arrays[0])
        if policy in policy_roots:
            policy_index = policy_roots.index(policy)
            existing_bank = tuple(
                bytes(arrays[1][index]).hex()
                for index, owner in enumerate(arrays[2])
                if int(owner) == policy_index
            )
            if existing_bank != bank:
                raise ValueError("repository action memory policy bank changed")
        else:
            if len(policy_roots) >= MAX_POLICY_BANKS:
                raise ValueError("repository action policy capacity exceeded")
            existing_actions = {bytes(row).hex() for row in arrays[1]}
            if existing_actions.intersection(bank):
                raise ValueError("repository action roots collided across policies")
            policy_index = len(policy_roots)
            arrays[0] = np.concatenate(
                (
                    arrays[0],
                    np.frombuffer(bytes.fromhex(policy), dtype=np.uint8)[None, :],
                ),
                axis=0,
            )
            arrays[1] = np.concatenate(
                (
                    arrays[1],
                    np.asarray(
                        [tuple(bytes.fromhex(value)) for value in bank],
                        dtype=np.uint8,
                    ),
                ),
                axis=0,
            )
            arrays[2] = np.concatenate(
                (
                    arrays[2],
                    np.full((ACTION_COUNT,), policy_index, dtype=np.uint8),
                )
            )
            arrays[5] = np.concatenate(
                (
                    arrays[5],
                    np.zeros((arrays[5].shape[0], ACTION_COUNT), dtype=np.uint8),
                ),
                axis=1,
            )
            arrays[6] = np.concatenate(
                (
                    arrays[6],
                    np.zeros((arrays[6].shape[0], ACTION_COUNT), dtype=np.uint8),
                ),
                axis=1,
            )
    return arrays, prior_steps


def _apply_lessons(
    parent: FrozenContinualRepositoryActionMemoryV1 | None,
    lessons: tuple[PhysicalRepositoryActionLessonV1, ...],
) -> tuple[list[np.ndarray], int]:
    arrays, learning_steps = _mutable_arrays(parent, lessons)
    (
        policy_roots,
        action_roots,
        action_policy_indices,
        context_cues,
        context_frame_roots,
        excitatory,
        inhibitory,
        provenance_context_indices,
        provenance_action_indices,
        provenance_passed,
        provenance_ordinals,
        provenance_result_roots,
        provenance_lesson_roots,
        provenance_program_roots,
        provenance_consequence_roots,
    ) = arrays
    seen_lessons = {bytes(row).hex() for row in provenance_lesson_roots}
    action_indices = {
        bytes(row).hex(): index for index, row in enumerate(action_roots)
    }
    for lesson in lessons:
        if lesson.lesson_body_sha256 in seen_lessons:
            raise ValueError("prior repository action lesson replay is forbidden")
        duplicate_ordinal = tuple(
            row
            for row in range(provenance_ordinals.shape[0])
            if bytes(provenance_result_roots[row]).hex()
            == lesson.controller_result_sha256
            and int(provenance_ordinals[row]) == lesson.attempt_ordinal
        )
        if duplicate_ordinal:
            raise ValueError("repository result ordinal replay is forbidden")
        if lesson.attempt_ordinal > 1:
            previous = tuple(
                row
                for row in range(provenance_ordinals.shape[0])
                if bytes(provenance_result_roots[row]).hex()
                == lesson.controller_result_sha256
                and int(provenance_ordinals[row]) == lesson.attempt_ordinal - 1
                and bytes(provenance_lesson_roots[row]).hex()
                == lesson.controller_lesson_body_sha256s[
                    lesson.attempt_ordinal - 2
                ]
            )
            if len(previous) != 1:
                raise ValueError("repository consequence prefix is missing")
        trace = np.asarray(lesson.source_context_trace_cells, dtype="<u4")
        arrivals = (
            np.count_nonzero(context_cues == trace, axis=1)
            if context_cues.shape[0]
            else np.empty((0,), dtype=np.int64)
        )
        exact = tuple(
            index for index, value in enumerate(arrivals) if int(value) == TRACE_LANES
        )
        learning_steps += (MAX_CONTEXTS + 1) * RACE_TICKS
        if len(exact) == 1:
            context_index = exact[0]
            if (
                bytes(context_frame_roots[context_index]).hex()
                != lesson.source_context_frame_sha256
            ):
                raise ValueError("repository source projection collision")
        elif exact:
            raise RuntimeError("repository context cue became ambiguous")
        else:
            if context_cues.shape[0] >= MAX_CONTEXTS:
                raise ValueError("repository context capacity exceeded")
            context_index = context_cues.shape[0]
            context_cues = np.concatenate((context_cues, trace[None, :]), axis=0)
            context_frame_roots = np.concatenate(
                (
                    context_frame_roots,
                    np.frombuffer(
                        bytes.fromhex(lesson.source_context_frame_sha256),
                        dtype=np.uint8,
                    )[None, :],
                ),
                axis=0,
            )
            excitatory = np.concatenate(
                (
                    excitatory,
                    np.zeros((1, action_roots.shape[0]), dtype=np.uint8),
                ),
                axis=0,
            )
            inhibitory = np.concatenate(
                (
                    inhibitory,
                    np.zeros((1, action_roots.shape[0]), dtype=np.uint8),
                ),
                axis=0,
            )
        action_index = action_indices.get(lesson.candidate_action_sha256)
        if action_index is None:
            raise ValueError("repository action lesson is outside the frozen bank")
        target = excitatory if lesson.passed_cell else inhibitory
        target[context_index, action_index] = np.uint8(
            min(TICK_CAP, int(target[context_index, action_index]) + EVIDENCE_TICKS)
        )
        provenance_context_indices = np.append(
            provenance_context_indices, np.uint8(context_index)
        )
        provenance_action_indices = np.append(
            provenance_action_indices, np.uint8(action_index)
        )
        provenance_passed = np.append(
            provenance_passed, np.uint8(lesson.passed_cell)
        )
        provenance_ordinals = np.append(
            provenance_ordinals, np.uint8(lesson.attempt_ordinal)
        )
        provenance_result_roots = np.concatenate(
            (
                provenance_result_roots,
                np.frombuffer(
                    bytes.fromhex(lesson.controller_result_sha256), dtype=np.uint8
                )[None, :],
            ),
            axis=0,
        )
        provenance_lesson_roots = np.concatenate(
            (
                provenance_lesson_roots,
                np.frombuffer(
                    bytes.fromhex(lesson.lesson_body_sha256), dtype=np.uint8
                )[None, :],
            ),
            axis=0,
        )
        provenance_program_roots = np.concatenate(
            (
                provenance_program_roots,
                np.frombuffer(
                    bytes.fromhex(lesson.program_sha256), dtype=np.uint8
                )[None, :],
            ),
            axis=0,
        )
        provenance_consequence_roots = np.concatenate(
            (
                provenance_consequence_roots,
                np.frombuffer(
                    bytes.fromhex(lesson.consequence_sha256), dtype=np.uint8
                )[None, :],
            ),
            axis=0,
        )
        seen_lessons.add(lesson.lesson_body_sha256)
    if provenance_passed.shape[0] > MAX_PROVENANCE_ROWS:
        raise ValueError("repository action provenance capacity exceeded")
    return [
        policy_roots,
        action_roots,
        action_policy_indices,
        context_cues,
        context_frame_roots,
        excitatory,
        inhibitory,
        provenance_context_indices,
        provenance_action_indices,
        provenance_passed,
        provenance_ordinals,
        provenance_result_roots,
        provenance_lesson_roots,
        provenance_program_roots,
        provenance_consequence_roots,
    ], learning_steps


def _freeze_memory(
    arrays: list[np.ndarray],
    *,
    parent_checkpoint_sha256: str | None,
    mode: str,
    new_lesson_count: int,
    learning_cell_steps: int,
) -> FrozenContinualRepositoryActionMemoryV1:
    policies = arrays[0].shape[0]
    actions = arrays[1].shape[0]
    contexts = arrays[3].shape[0]
    rows = arrays[9].shape[0]
    trainable = int(arrays[5].nbytes + arrays[6].nbytes)
    provenance = sum(array.nbytes for array in arrays[7:])
    checkpoint_bytes = sum(array.nbytes for array in arrays)
    partial = RepositoryActionTrainingSummaryV1(
        mode=mode,
        policy_checkpoint_sha256s=tuple(bytes(row).hex() for row in arrays[0]),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        policy_count=policies,
        context_count=contexts,
        action_count=actions,
        provenance_count=rows,
        new_lesson_count=new_lesson_count,
        parent_lesson_replay_count=0,
        retained_parent_count=0,
        raw_lesson_retention_count=0,
        parent_mutation_count=0,
        added_cell_count=FIXED_CONTROLLER_CELL_COUNT + contexts + actions,
        trainable_state_bytes=trainable,
        provenance_state_bytes=provenance,
        checkpoint_array_bytes=checkpoint_bytes,
        learning_cell_steps=learning_cell_steps,
        checkpoint_sha256="",
    )
    frozen_arrays = tuple(
        (name, _freeze_array(array, array.dtype))
        for name, array in zip(_CHECKPOINT_ARRAY_NAMES, arrays, strict=True)
    )
    checkpoint = _checkpoint_sha256(partial, frozen_arrays)
    summary = partial._replace(checkpoint_sha256=checkpoint)
    result = FrozenContinualRepositoryActionMemoryV1(
        policy_roots=arrays[0],
        action_roots=arrays[1],
        action_policy_indices=arrays[2],
        context_cues=arrays[3],
        context_frame_roots=arrays[4],
        excitatory=arrays[5],
        inhibitory=arrays[6],
        provenance_context_indices=arrays[7],
        provenance_action_indices=arrays[8],
        provenance_passed=arrays[9],
        provenance_ordinals=arrays[10],
        provenance_result_roots=arrays[11],
        provenance_lesson_roots=arrays[12],
        provenance_program_roots=arrays[13],
        provenance_consequence_roots=arrays[14],
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )
    return result


def start_continual_repository_action_memory_v1(
    lessons: tuple[PhysicalRepositoryActionLessonV1, ...],
    *,
    plasticity_gate: bool = True,
) -> FrozenContinualRepositoryActionMemoryV1:
    _validate_lessons(lessons)
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    if not plasticity_gate:
        raise ValueError("an empty memory cannot start with plasticity disabled")
    arrays, learning_steps = _apply_lessons(None, lessons)
    return _freeze_memory(
        arrays,
        parent_checkpoint_sha256=None,
        mode="INITIAL_REPOSITORY_ACTION_MEMORY",
        new_lesson_count=len(lessons),
        learning_cell_steps=learning_steps,
    )


def advance_continual_repository_action_memory_v1(
    parent: FrozenContinualRepositoryActionMemoryV1,
    new_lessons: tuple[PhysicalRepositoryActionLessonV1, ...],
    *,
    plasticity_gate: bool = True,
) -> FrozenContinualRepositoryActionMemoryV1:
    if type(parent) is not FrozenContinualRepositoryActionMemoryV1:
        raise TypeError("parent must be one exact frozen repository action memory")
    _validate_lessons(new_lessons)
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    parent_array_roots = tuple(_array_sha256(array) for _, array in parent._arrays())
    if not plasticity_gate:
        parent.authenticate()
        return parent
    arrays, learning_steps = _apply_lessons(parent, new_lessons)
    successor = _freeze_memory(
        arrays,
        parent_checkpoint_sha256=parent_root,
        mode="REPLAY_FREE_REPOSITORY_ACTION_ADVANCE",
        new_lesson_count=len(new_lessons),
        learning_cell_steps=learning_steps,
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or tuple(_array_sha256(array) for _, array in parent._arrays())
        != parent_array_roots
    ):
        raise RuntimeError("repository action advance mutated its parent")
    return successor


__all__ = [
    "ACTION_COUNT",
    "CLAIM_BOUNDARY",
    "EVIDENCE_TICKS",
    "MAX_ADDED_CELL_COUNT",
    "MAX_CHECKPOINT_ARRAY_BYTES",
    "MAX_CONTROLLER_CELL_STEPS",
    "MAX_PROVENANCE_ROWS",
    "MAX_TRAINABLE_STATE_BYTES",
    "OUTCOME_COMMIT",
    "OUTCOME_REVISE",
    "RECALLED",
    "SCHEMA",
    "STOP_ACTION_BANK",
    "STOP_CANDIDATE_AMBIGUOUS",
    "STOP_CANDIDATE_GATE",
    "STOP_CANDIDATE_UNKNOWN",
    "STOP_CONTEXT_AMBIGUOUS",
    "STOP_CONTEXT_COLLISION",
    "STOP_CONTEXT_GATE",
    "STOP_CONTEXT_UNKNOWN",
    "STOP_RECALL_GATE",
    "FrozenContinualRepositoryActionMemoryV1",
    "PhysicalRepositoryActionLessonV1",
    "RepositoryActionCandidateV1",
    "RepositoryActionProvenanceV1",
    "RepositoryActionRecallV1",
    "RepositoryActionRecallWorkV1",
    "RepositoryActionTrainingSummaryV1",
    "RepositorySourceContextV1",
    "advance_continual_repository_action_memory_v1",
    "controller_result_sha256_v1",
    "project_attested_repository_source_v1",
    "start_continual_repository_action_memory_v1",
]
