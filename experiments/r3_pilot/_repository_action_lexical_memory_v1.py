"""Replay-free exact-plus-lexical repository-action memory.

The exact child remains authoritative.  A Python lexical factor tier is reached
only after an exact unknown, learns only from committed controller consequence
cohorts, and selects only stable actions already present in the neutral bank.
Frozen checkpoints retain no source bytes, token lexemes, lessons, experiences,
or parent object.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import common_continual_repository_action_memory_v1 as continual
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from humanbrain.resting_glif_tier_beam import RETAINED, CompiledRestingGlifTierBeam

SCHEMA = "COMMON_FACTORIZED_REPOSITORY_ACTION_MEMORY_V1"

EXACT_EPISODIC = "EXACT_EPISODIC"
LEXICAL_FACTOR = "LEXICAL_FACTOR"
STOP_FACTOR_GATE = f"STOP_{SCHEMA}_FACTOR_GATE"
STOP_FACTOR_UNKNOWN = f"STOP_{SCHEMA}_FACTOR_UNKNOWN"
STOP_FACTOR_AMBIGUOUS = f"STOP_{SCHEMA}_FACTOR_AMBIGUOUS"
STOP_FACTOR_CONFLICT = f"STOP_{SCHEMA}_FACTOR_CONFLICT"

ACTION_COUNT = action_bank.CANDIDATE_COUNT
TRACE_LANES = source_views.TRACE_LANES
MAX_POLICY_BANKS = 2
MAX_FACTOR_ROWS = 24
MAX_ADMITTED_FACTORS = MAX_FACTOR_ROWS
MAX_PROVENANCE_SOURCE_ROWS = 8
MIN_INDEPENDENT_SOURCE_ROOTS = 2
TICK_CAP = continual.TICK_CAP
EVIDENCE_TICKS = continual.EVIDENCE_TICKS

RACE_TICKS = 600
CURRENT_BASE_PA = 600.0
CURRENT_GAIN_PA = 400.0
MATCH_CURRENT_GAIN_PA = 160.0
ACTION_CURRENT_MAX_PA = 7_000.0
MATCH_CURRENT_MAX_PA = 3_000.0
INHIBITION_CURRENT_PA = 7_000.0
SUPPORT_STOP_CELL = 1
ACTION_STOP_CELL = ACTION_COUNT
MATCH_STOP_CELL = source_views.MAX_SOURCE_FACTORS

_EXPERIENCE_NAMESPACE = b"common-factorized-repository-action-experience-v1"
_CHECKPOINT_NAMESPACE = b"common-factorized-repository-action-checkpoint-v1"
_RECALL_NAMESPACE = b"common-factorized-repository-action-recall-v1"
_PHYSICAL_TIER_NAMESPACE = b"common-factorized-repository-physical-tier-v1"
_CANDIDATE_NAMESPACE = b"common-repository-action-candidate-v1"

_SUPPORT_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=2,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=ACTION_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_ACTION_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=ACTION_COUNT + 1,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=ACTION_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_MATCH_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=source_views.MAX_SOURCE_FACTORS + 1,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=MATCH_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)

_CONSTRUCTION_TOKEN = object()
_ARRAY_NAMES = (
    "policy-roots",
    "policy-action-roots",
    "factor-policy-indices",
    "factor-semantic-roots",
    "factor-frame-roots",
    "factor-cues",
    "factor-excitatory",
    "factor-inhibitory",
    "admitted-actions",
    "source-roots",
    "experience-roots",
    "source-factor-presence",
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


def _array_sha256(value: np.ndarray) -> str:
    return sha256(value.tobytes(order="C")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return action_bank.valid_sha256(value)


def _root_row(value: str) -> np.ndarray:
    return np.asarray([tuple(bytes.fromhex(value))], dtype=np.uint8)


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


def _support_is_admitted(distinct_source_count: int) -> bool:
    currents = np.full(2, CURRENT_BASE_PA, dtype=np.float64)
    currents[0] += CURRENT_GAIN_PA * min(distinct_source_count, 2)
    currents[SUPPORT_STOP_CELL] += CURRENT_GAIN_PA
    physical = _SUPPORT_SELECTOR.select(currents, stop_cell=SUPPORT_STOP_CELL)
    return (
        physical.status == RETAINED
        and physical.retained_candidate_cells == (0,)
        and distinct_source_count >= MIN_INDEPENDENT_SOURCE_ROOTS
    )


def _action_physical(
    excitatory: Sequence[int],
    inhibitory: Sequence[int],
) -> object:
    currents = np.full(ACTION_COUNT + 1, CURRENT_BASE_PA, dtype=np.float64)
    for index in range(ACTION_COUNT):
        drive = max(0, int(excitatory[index]) - int(inhibitory[index]))
        currents[index] += CURRENT_GAIN_PA * drive
    currents[ACTION_STOP_CELL] += CURRENT_GAIN_PA
    return _ACTION_SELECTOR.select(currents, stop_cell=ACTION_STOP_CELL)


def _unique_action(
    excitatory: Sequence[int],
    inhibitory: Sequence[int],
) -> tuple[int | None, object]:
    physical = _action_physical(excitatory, inhibitory)
    positive_actions = tuple(
        index for index in range(ACTION_COUNT) if int(excitatory[index]) > 0
    )
    if (
        len(positive_actions) == 1
        and physical.status == RETAINED
        and len(physical.retained_candidate_cells) == 1
        and 0 <= physical.retained_candidate_cells[0] < ACTION_COUNT
        and physical.retained_candidate_cells[0] == positive_actions[0]
    ):
        return physical.retained_candidate_cells[0], physical
    return None, physical


class FactorizedRepositoryActionExperienceV1(NamedTuple):
    source_views: source_views.RepositorySourceViewsV1
    action_lessons: tuple[continual.PhysicalRepositoryActionLessonV1, ...]
    sha256: str

    def authenticate(self) -> None:
        if type(self.source_views) is not source_views.RepositorySourceViewsV1:
            raise TypeError("exact repository source views required")
        if type(self.action_lessons) is not tuple or not self.action_lessons:
            raise TypeError("one nonempty exact action lesson cohort required")
        self.source_views.authenticate()
        for lesson in self.action_lessons:
            if type(lesson) is not continual.PhysicalRepositoryActionLessonV1:
                raise TypeError("exact physical repository action lessons required")
            lesson.authenticate()
        lessons = self.action_lessons
        first = lessons[0]
        exact = self.source_views.exact_context
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
        source_shape = (
            exact.source_sha256,
            exact.frame_sha256,
            exact.trace_cells,
            exact.projection_cell_steps,
            exact.sha256,
        )
        payload = {
            "source_views": self.source_views.sha256,
            "lessons": tuple(lesson.sha256 for lesson in lessons),
        }
        if (
            len(lessons) != first.attempt_count
            or not 1 <= len(lessons) <= 2
            or tuple(lesson.attempt_ordinal for lesson in lessons)
            != tuple(range(1, len(lessons) + 1))
            or len({lesson.sha256 for lesson in lessons}) != len(lessons)
            or any(
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
            )
            or any(
                (
                    lesson.source_sha256,
                    lesson.source_context_frame_sha256,
                    lesson.source_context_trace_cells,
                    lesson.source_projection_cell_steps,
                    lesson.source_context_sha256,
                )
                != source_shape
                for lesson in lessons
            )
            or first.controller_lesson_body_sha256s
            != tuple(lesson.lesson_body_sha256 for lesson in lessons)
            or first.controller_attempt_sha256s
            != tuple(lesson.attempt_sha256 for lesson in lessons)
            or any(
                lesson.passed_cell != 0
                or lesson.rolled_back is not True
                or lesson.committed is not False
                for lesson in lessons[:-1]
            )
            or lessons[-1].passed_cell != 1
            or lessons[-1].rolled_back is not False
            or lessons[-1].committed is not True
            or self.sha256 != _digest(_EXPERIENCE_NAMESPACE, payload)
        ):
            raise ValueError("factorized repository action experience changed")


def make_factorized_repository_action_experience_v1(
    source_observation: source_views.RepositorySourceViewsV1,
    action_lessons: tuple[continual.PhysicalRepositoryActionLessonV1, ...],
) -> FactorizedRepositoryActionExperienceV1:
    if type(source_observation) is not source_views.RepositorySourceViewsV1:
        raise TypeError("exact repository source views required")
    if type(action_lessons) is not tuple:
        raise TypeError("repository action lesson cohort must be one exact tuple")
    partial = FactorizedRepositoryActionExperienceV1(
        source_views=source_observation,
        action_lessons=action_lessons,
        sha256="",
    )
    result = partial._replace(
        sha256=_digest(
            _EXPERIENCE_NAMESPACE,
            {
                "source_views": source_observation.sha256,
                "lessons": tuple(lesson.sha256 for lesson in action_lessons),
            },
        )
    )
    result.authenticate()
    return result


class FactorizedRepositoryActionTrainingSummaryV1(NamedTuple):
    mode: str
    parent_checkpoint_sha256: str | None
    exact_memory_checkpoint_sha256: str
    policy_count: int
    factor_count: int
    admitted_factor_count: int
    source_count: int
    new_lesson_count: int
    raw_experience_retention_count: int
    parent_mutation_count: int
    trainable_state_bytes: int
    checkpoint_array_bytes: int
    factor_learning_cell_steps: int
    learning_cell_steps: int
    checkpoint_sha256: str


class FactorizedRepositoryFactorAdmissionV1(NamedTuple):
    policy_checkpoint_sha256: str
    semantic_sha256: str
    selected_candidate_action_sha256: str | None
    distinct_source_count: int
    excitatory_ticks: tuple[int, ...]
    inhibitory_ticks: tuple[int, ...]


class FactorizedRepositorySourceProvenanceV1(NamedTuple):
    source_sha256: str
    experience_sha256: str
    factor_count: int


class FactorizedRepositoryActionRecallWorkV1(NamedTuple):
    source_projection_cell_steps: int
    exact_context_equality_reads: int
    exact_context_equality_arrivals: int
    exact_context_race_cell_steps: int
    exact_candidate_race_cell_steps: int
    lexical_projection_cell_steps: int
    factor_equality_reads: int
    factor_equality_arrivals: int
    factor_match_cell_steps: int
    factor_action_race_cell_steps: int
    factor_consensus_cell_steps: int

    @property
    def total_cell_steps(self) -> int:
        return (
            self.source_projection_cell_steps
            + self.exact_context_race_cell_steps
            + self.exact_candidate_race_cell_steps
            + self.lexical_projection_cell_steps
            + self.factor_match_cell_steps
            + self.factor_action_race_cell_steps
            + self.factor_consensus_cell_steps
        )

    def authenticate(self) -> None:
        if any(type(value) is not int or value < 0 for value in self):
            raise ValueError("factorized repository recall work changed")


class FactorizedRepositoryActionRecallV1(NamedTuple):
    status: str
    tier: str
    memory_checkpoint_sha256: str
    exact_memory_checkpoint_sha256: str
    policy_checkpoint_sha256: str
    source_sha256: str
    source_context_frame_sha256: str
    source_context_trace_cells: tuple[int, ...]
    frontier_sha256: str
    action_bank_sha256s: tuple[str, ...]
    candidates: tuple[continual.RepositoryActionCandidateV1, ...]
    selected_candidate_action_sha256: str | None
    exact_recall_sha256: str
    lexical_view_sha256: str | None
    matched_factor_semantic_sha256s: tuple[str, ...]
    factor_match_tier_sha256s: tuple[str, ...]
    factor_action_tier_sha256s: tuple[str, ...]
    factor_consensus_tier_sha256: str | None
    work: FactorizedRepositoryActionRecallWorkV1
    sha256: str

    def authenticate(self) -> None:
        if type(self.candidates) is not tuple:
            raise TypeError("factorized recall candidates must be one exact tuple")
        for candidate in self.candidates:
            if type(candidate) is not continual.RepositoryActionCandidateV1:
                raise TypeError("exact repository action recall candidates required")
            candidate.authenticate()
        if type(self.work) is not FactorizedRepositoryActionRecallWorkV1:
            raise TypeError("exact factorized repository recall work required")
        self.work.authenticate()
        action_keys = tuple(candidate.candidate_action_sha256 for candidate in self.candidates)
        payload = _recall_payload(self._asdict())
        if (
            self.status
            not in {
                continual.RECALLED,
                continual.STOP_RECALL_GATE,
                continual.STOP_CONTEXT_GATE,
                continual.STOP_CONTEXT_UNKNOWN,
                continual.STOP_CONTEXT_AMBIGUOUS,
                continual.STOP_CONTEXT_COLLISION,
                continual.STOP_CANDIDATE_GATE,
                continual.STOP_CANDIDATE_UNKNOWN,
                continual.STOP_CANDIDATE_AMBIGUOUS,
                continual.STOP_ACTION_BANK,
                STOP_FACTOR_GATE,
                STOP_FACTOR_UNKNOWN,
                STOP_FACTOR_AMBIGUOUS,
                STOP_FACTOR_CONFLICT,
            }
            or self.tier not in (EXACT_EPISODIC, LEXICAL_FACTOR)
            or any(
                not _valid_sha256(value)
                for value in (
                    self.memory_checkpoint_sha256,
                    self.exact_memory_checkpoint_sha256,
                    self.policy_checkpoint_sha256,
                    self.source_sha256,
                    self.source_context_frame_sha256,
                    self.frontier_sha256,
                    self.exact_recall_sha256,
                )
            )
            or type(self.source_context_trace_cells) is not tuple
            or len(self.source_context_trace_cells) != TRACE_LANES
            or any(type(cell) is not int for cell in self.source_context_trace_cells)
            or self.action_bank_sha256s != tuple(sorted(self.action_bank_sha256s))
            or len(self.action_bank_sha256s) != ACTION_COUNT
            or len(set(self.action_bank_sha256s)) != ACTION_COUNT
            or len(self.candidates) not in (0, ACTION_COUNT)
            or (
                self.candidates
                and (
                    tuple(sorted(action_keys)) != self.action_bank_sha256s
                    or len(set(action_keys)) != ACTION_COUNT
                )
            )
            or (
                self.status == continual.RECALLED
                and self.selected_candidate_action_sha256 not in action_keys
            )
            or (
                self.status != continual.RECALLED
                and self.selected_candidate_action_sha256 is not None
            )
            or (self.tier == EXACT_EPISODIC and self.lexical_view_sha256 is not None)
            or (
                self.tier == LEXICAL_FACTOR
                and self.status != STOP_FACTOR_GATE
                and not _valid_sha256(self.lexical_view_sha256)
            )
            or any(
                not _valid_sha256(value)
                for values in (
                    self.matched_factor_semantic_sha256s,
                    self.factor_match_tier_sha256s,
                    self.factor_action_tier_sha256s,
                )
                for value in values
            )
            or (
                self.factor_consensus_tier_sha256 is not None
                and not _valid_sha256(self.factor_consensus_tier_sha256)
            )
            or self.sha256 != _digest(_RECALL_NAMESPACE, payload)
        ):
            raise ValueError("factorized repository action recall authentication failed")


def _recall_payload(record: dict[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload.pop("sha256", None)
    payload["candidates"] = tuple(item.sha256 for item in record["candidates"])
    work = record["work"]
    if type(work) is FactorizedRepositoryActionRecallWorkV1:
        payload["work"] = work._asdict()
    return payload


def _checkpoint_sha256(
    summary: FactorizedRepositoryActionTrainingSummaryV1,
    exact_checkpoint_sha256: str,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    summary_payload = summary._asdict()
    summary_payload.pop("checkpoint_sha256", None)
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "summary": summary_payload,
            "exact": exact_checkpoint_sha256,
            "arrays": tuple(
                (name, str(array.dtype), array.shape, _array_sha256(array))
                for name, array in arrays
            ),
        },
    )


class FrozenFactorizedRepositoryActionMemoryV1:
    """One immutable composite checkpoint with no replay or parent retention."""

    __slots__ = (
        "_exact_memory",
        "_policy_roots",
        "_policy_action_roots",
        "_factor_policy_indices",
        "_factor_semantic_roots",
        "_factor_frame_roots",
        "_factor_cues",
        "_factor_excitatory",
        "_factor_inhibitory",
        "_admitted_actions",
        "_source_roots",
        "_experience_roots",
        "_source_factor_presence",
        "_summary",
        "_checkpoint_sha256",
    )

    def __init__(
        self,
        *,
        exact_memory: continual.FrozenContinualRepositoryActionMemoryV1,
        arrays: list[np.ndarray],
        summary: FactorizedRepositoryActionTrainingSummaryV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("factorized repository checkpoints require their builder")
        if type(exact_memory) is not continual.FrozenContinualRepositoryActionMemoryV1:
            raise TypeError("one exact continual repository memory child is required")
        if len(arrays) != len(_ARRAY_NAMES):
            raise ValueError("factorized repository checkpoint arrays changed")
        self._exact_memory = exact_memory
        (
            self._policy_roots,
            self._policy_action_roots,
            self._factor_policy_indices,
            self._factor_semantic_roots,
            self._factor_frame_roots,
            self._factor_cues,
            self._factor_excitatory,
            self._factor_inhibitory,
            self._admitted_actions,
            self._source_roots,
            self._experience_roots,
            self._source_factor_presence,
        ) = tuple(
            _freeze_array(array, array.dtype)
            for array in arrays
        )
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            summary,
            exact_memory.checkpoint_sha256,
            self._arrays(),
        )
        self.authenticate()

    def _arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        arrays = (
            self._policy_roots,
            self._policy_action_roots,
            self._factor_policy_indices,
            self._factor_semantic_roots,
            self._factor_frame_roots,
            self._factor_cues,
            self._factor_excitatory,
            self._factor_inhibitory,
            self._admitted_actions,
            self._source_roots,
            self._experience_roots,
            self._source_factor_presence,
        )
        return tuple(zip(_ARRAY_NAMES, arrays, strict=True))

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def exact_memory_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._exact_memory.checkpoint_sha256

    @property
    def training_summary(self) -> FactorizedRepositoryActionTrainingSummaryV1:
        self.authenticate()
        return self._summary

    @property
    def factor_admissions(self) -> tuple[FactorizedRepositoryFactorAdmissionV1, ...]:
        self.authenticate()
        result: list[FactorizedRepositoryFactorAdmissionV1] = []
        for row in range(self._factor_policy_indices.shape[0]):
            policy_index = int(self._factor_policy_indices[row])
            admitted = int(self._admitted_actions[row])
            result.append(
                FactorizedRepositoryFactorAdmissionV1(
                    policy_checkpoint_sha256=bytes(
                        self._policy_roots[policy_index]
                    ).hex(),
                    semantic_sha256=bytes(self._factor_semantic_roots[row]).hex(),
                    selected_candidate_action_sha256=(
                        None
                        if admitted < 0
                        else bytes(
                            self._policy_action_roots[policy_index, admitted]
                        ).hex()
                    ),
                    distinct_source_count=int(
                        np.sum(self._source_factor_presence[:, row], dtype=np.int64)
                    ),
                    excitatory_ticks=tuple(
                        int(value) for value in self._factor_excitatory[row]
                    ),
                    inhibitory_ticks=tuple(
                        int(value) for value in self._factor_inhibitory[row]
                    ),
                )
            )
        return tuple(result)

    @property
    def source_provenance(self) -> tuple[FactorizedRepositorySourceProvenanceV1, ...]:
        self.authenticate()
        return tuple(
            FactorizedRepositorySourceProvenanceV1(
                source_sha256=bytes(self._source_roots[row]).hex(),
                experience_sha256=bytes(self._experience_roots[row]).hex(),
                factor_count=int(
                    np.sum(self._source_factor_presence[row], dtype=np.int64)
                ),
            )
            for row in range(self._source_roots.shape[0])
        )

    def authenticate(self) -> None:
        self._exact_memory.authenticate()
        exact_summary = self._exact_memory.training_summary
        arrays = self._arrays()
        if any(array.flags.writeable for _, array in arrays):
            raise RuntimeError("factorized repository action checkpoint became mutable")
        policies = self._policy_roots.shape[0]
        factors = self._factor_policy_indices.shape[0]
        sources = self._source_roots.shape[0]
        expected_shapes = (
            (policies, 32),
            (policies, ACTION_COUNT, 32),
            (factors,),
            (factors, 32),
            (factors, 32),
            (factors, TRACE_LANES),
            (factors, ACTION_COUNT),
            (factors, ACTION_COUNT),
            (factors,),
            (sources, 32),
            (sources, 32),
            (sources, factors),
        )
        if tuple(array.shape for _, array in arrays) != expected_shapes:
            raise RuntimeError("factorized repository action checkpoint shape changed")
        policy_values = tuple(bytes(row).hex() for row in self._policy_roots)
        action_values = tuple(
            bytes(row).hex()
            for policy in self._policy_action_roots
            for row in policy
        )
        keys = tuple(
            (
                int(self._factor_policy_indices[row]),
                bytes(self._factor_semantic_roots[row]),
            )
            for row in range(factors)
        )
        if (
            not 1 <= policies <= MAX_POLICY_BANKS
            or not 0 <= factors <= MAX_FACTOR_ROWS
            or not 1 <= sources <= MAX_PROVENANCE_SOURCE_ROWS
            or len(set(policy_values)) != policies
            or any(
                tuple(sorted(bytes(row).hex() for row in policy))
                != tuple(bytes(row).hex() for row in policy)
                or len({bytes(row) for row in policy}) != ACTION_COUNT
                for policy in self._policy_action_roots
            )
            or len(set(keys)) != factors
            or np.any(self._factor_policy_indices >= policies)
            or len({bytes(row) for row in self._source_roots}) != sources
            or len({bytes(row) for row in self._experience_roots}) != sources
            or np.any(self._source_factor_presence > 1)
            or np.any(self._factor_excitatory > TICK_CAP)
            or np.any(self._factor_inhibitory > TICK_CAP)
            or np.any(self._admitted_actions < -1)
            or np.any(self._admitted_actions >= ACTION_COUNT)
            or self._exact_memory.policy_checkpoint_sha256s != policy_values
            or self._exact_memory.action_sha256s != action_values
        ):
            raise RuntimeError("factorized repository action checkpoint values changed")
        admitted_count = int(np.count_nonzero(self._admitted_actions >= 0))
        if admitted_count > MAX_ADMITTED_FACTORS:
            raise RuntimeError("factorized repository factor admission changed")
        trainable = int(
            exact_summary.trainable_state_bytes
            + self._factor_excitatory.nbytes
            + self._factor_inhibitory.nbytes
            + self._admitted_actions.nbytes
        )
        checkpoint_bytes = exact_summary.checkpoint_array_bytes + sum(
            array.nbytes for _, array in arrays
        )
        summary = self._summary
        if (
            summary.policy_count != policies
            or summary.factor_count != factors
            or summary.admitted_factor_count != admitted_count
            or summary.source_count != sources
            or summary.exact_memory_checkpoint_sha256
            != self._exact_memory.checkpoint_sha256
            or summary.raw_experience_retention_count != 0
            or summary.parent_mutation_count != 0
            or summary.trainable_state_bytes != trainable
            or summary.checkpoint_array_bytes != checkpoint_bytes
            or summary.learning_cell_steps
            != (
                exact_summary.learning_cell_steps
                + summary.factor_learning_cell_steps
            )
            or summary.checkpoint_sha256 != self._checkpoint_sha256
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                summary,
                self._exact_memory.checkpoint_sha256,
                arrays,
            )
        ):
            raise RuntimeError("factorized repository action summary changed")

    def recall_attested_source(
        self,
        source: bytes,
        context: source_views.RepositorySourceContextV1,
        bank: action_bank.FrozenRepositoryActionBankV1,
        *,
        candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
        | None = None,
        recall_gate: bool = True,
        context_gate: bool = True,
        candidate_gate: bool = True,
        factor_gate: bool = True,
    ) -> FactorizedRepositoryActionRecallV1:
        if type(source) is not bytes:
            raise TypeError("repository source must be exact bytes")
        if len(source) > source_views.MAX_SOURCE_BYTES:
            raise source_views.RepositorySourceCapacityError(
                f"repository source exceeds {source_views.MAX_SOURCE_BYTES} bytes"
            )
        self.authenticate()
        if type(factor_gate) is not bool:
            raise TypeError("repository factor gate must be one exact boolean")
        exact = self._exact_memory.recall_attested_source(
            source,
            context,
            bank,
            candidate_records=candidate_records,
            recall_gate=recall_gate,
            context_gate=context_gate,
            candidate_gate=candidate_gate,
            factor_gate=factor_gate,
        )
        exact.authenticate()
        if exact.status not in {
            continual.STOP_CONTEXT_UNKNOWN,
            continual.STOP_CANDIDATE_UNKNOWN,
        }:
            return self._finish_recall(
                exact=exact,
                tier=EXACT_EPISODIC,
                status=exact.status,
                candidates=exact.candidates,
                selected_action=exact.selected_candidate_action_sha256,
                lexical=None,
                matched=(),
                match_tiers=(),
                action_tiers=(),
                consensus_tier=None,
                factor_reads=0,
                factor_arrivals=0,
                factor_match_steps=0,
                factor_action_steps=0,
                factor_consensus_steps=0,
            )
        if not factor_gate:
            return self._finish_recall(
                exact=exact,
                tier=LEXICAL_FACTOR,
                status=STOP_FACTOR_GATE,
                candidates=(),
                selected_action=None,
                lexical=None,
                matched=(),
                match_tiers=(),
                action_tiers=(),
                consensus_tier=None,
                factor_reads=0,
                factor_arrivals=0,
                factor_match_steps=0,
                factor_action_steps=0,
                factor_consensus_steps=0,
            )

        lexical = source_views.project_python_repository_lexical_view_v1(source)
        policy_matches = tuple(
            index
            for index, row in enumerate(self._policy_roots)
            if bytes(row).hex() == bank.policy_checkpoint_sha256
        )
        if len(policy_matches) != 1:
            raise RuntimeError("exact child admitted a bank outside factor state")
        policy_index = policy_matches[0]
        admitted_rows = tuple(
            row
            for row in range(self._factor_policy_indices.shape[0])
            if int(self._factor_policy_indices[row]) == policy_index
            and int(self._admitted_actions[row]) >= 0
        )
        factor_reads = 0
        factor_arrivals = 0
        match_steps = 0
        action_steps = 0
        consensus_steps = 0
        matched: list[str] = []
        matched_rows: list[int] = []
        votes: list[int] = []
        match_tiers: list[str] = []
        action_tiers: list[str] = []
        ambiguous = False
        observed = lexical.factors

        for row in admitted_rows:
            arrivals = np.count_nonzero(
                np.asarray(tuple(factor.trace_cells for factor in observed), dtype="<u4")
                == self._factor_cues[row],
                axis=1,
            ).astype(np.int16) if observed else np.empty((0,), dtype=np.int16)
            factor_reads += len(observed) * TRACE_LANES
            factor_arrivals += int(np.sum(arrivals, dtype=np.int64))
            currents = np.full(
                source_views.MAX_SOURCE_FACTORS + 1,
                CURRENT_BASE_PA,
                dtype=np.float64,
            )
            if observed:
                currents[: len(observed)] += MATCH_CURRENT_GAIN_PA * arrivals
            currents[MATCH_STOP_CELL] += MATCH_CURRENT_GAIN_PA * (TRACE_LANES - 1)
            physical = _MATCH_SELECTOR.select(currents, stop_cell=MATCH_STOP_CELL)
            match_tiers.append(_physical_tier_sha256(physical))
            match_steps += (source_views.MAX_SOURCE_FACTORS + 1) * RACE_TICKS
            if physical.status != RETAINED:
                continue
            if len(physical.retained_candidate_cells) != 1:
                ambiguous = True
                break
            observed_index = physical.retained_candidate_cells[0]
            if not 0 <= observed_index < len(observed):
                ambiguous = True
                break
            factor = observed[observed_index]
            if (
                bytes(self._factor_semantic_roots[row]).hex()
                != factor.semantic_sha256
                or bytes(self._factor_frame_roots[row]).hex() != factor.frame_sha256
            ):
                ambiguous = True
                break
            action_index, action_physical = _unique_action(
                self._factor_excitatory[row],
                self._factor_inhibitory[row],
            )
            action_tiers.append(_physical_tier_sha256(action_physical))
            action_steps += (ACTION_COUNT + 1) * RACE_TICKS
            if action_index is None or action_index != int(self._admitted_actions[row]):
                ambiguous = True
                break
            matched.append(factor.semantic_sha256)
            matched_rows.append(row)
            votes.append(action_index)

        consensus_tier: str | None = None
        candidates: tuple[continual.RepositoryActionCandidateV1, ...] = ()
        selected_action: str | None = None
        if ambiguous:
            status = STOP_FACTOR_AMBIGUOUS
        elif not matched_rows:
            status = STOP_FACTOR_UNKNOWN
        elif len(set(votes)) != 1:
            status = STOP_FACTOR_CONFLICT
            candidates = self._factor_candidates(
                policy_index,
                matched_rows,
                bank,
                candidate_records,
            )
        else:
            consensus_currents = np.full(
                ACTION_COUNT + 1,
                CURRENT_BASE_PA,
                dtype=np.float64,
            )
            for vote in votes:
                consensus_currents[vote] = min(
                    ACTION_CURRENT_MAX_PA,
                    consensus_currents[vote] + CURRENT_GAIN_PA,
                )
            consensus = _ACTION_SELECTOR.select(
                consensus_currents,
                stop_cell=ACTION_STOP_CELL,
            )
            consensus_tier = _physical_tier_sha256(consensus)
            consensus_steps = (ACTION_COUNT + 1) * RACE_TICKS
            candidates = self._factor_candidates(
                policy_index,
                matched_rows,
                bank,
                candidate_records,
            )
            if (
                consensus.status != RETAINED
                or consensus.retained_candidate_cells != (votes[0],)
            ):
                status = STOP_FACTOR_AMBIGUOUS
            else:
                status = continual.RECALLED
                selected_action = bytes(
                    self._policy_action_roots[policy_index, votes[0]]
                ).hex()

        return self._finish_recall(
            exact=exact,
            tier=LEXICAL_FACTOR,
            status=status,
            candidates=candidates,
            selected_action=selected_action,
            lexical=lexical,
            matched=tuple(matched),
            match_tiers=tuple(match_tiers),
            action_tiers=tuple(action_tiers),
            consensus_tier=consensus_tier,
            factor_reads=factor_reads,
            factor_arrivals=factor_arrivals,
            factor_match_steps=match_steps,
            factor_action_steps=action_steps,
            factor_consensus_steps=consensus_steps,
        )

    def _factor_candidates(
        self,
        policy_index: int,
        matched_rows: Sequence[int],
        bank: action_bank.FrozenRepositoryActionBankV1,
        candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
        | None,
    ) -> tuple[continual.RepositoryActionCandidateV1, ...]:
        supplied = bank.records if candidate_records is None else tuple(candidate_records)
        canonical = {record.sha256: record for record in bank.records}
        if (
            len(supplied) != ACTION_COUNT
            or len({record.sha256 for record in supplied}) != ACTION_COUNT
            or any(canonical.get(record.sha256) != record for record in supplied)
        ):
            raise RuntimeError("exact child admitted unauthenticated factor candidates")
        records_by_action = sorted(
            (bank.action_sha256_for(record), record) for record in supplied
        )
        expected_actions = tuple(
            bytes(row).hex() for row in self._policy_action_roots[policy_index]
        )
        if tuple(action for action, _ in records_by_action) != expected_actions:
            raise RuntimeError("factor action bank changed after exact recall")
        result: list[continual.RepositoryActionCandidateV1] = []
        for action_index, (action_sha256, record) in enumerate(records_by_action):
            excitatory = min(
                TICK_CAP,
                sum(int(self._factor_excitatory[row, action_index]) for row in matched_rows),
            )
            inhibitory = min(
                TICK_CAP,
                sum(int(self._factor_inhibitory[row, action_index]) for row in matched_rows),
            )
            address = int.from_bytes(bytes.fromhex(action_sha256)[:4], "little")
            payload = {
                "candidate": record.sha256,
                "action": action_sha256,
                "program": record.program_sha256,
                "address": address,
                "excitatory": excitatory,
                "inhibitory": inhibitory,
            }
            candidate = continual.RepositoryActionCandidateV1(
                candidate_sha256=record.sha256,
                candidate_action_sha256=action_sha256,
                program_sha256=record.program_sha256,
                cell_address=address,
                excitatory_ticks=excitatory,
                inhibitory_ticks=inhibitory,
                sha256=_digest(_CANDIDATE_NAMESPACE, payload),
            )
            candidate.authenticate()
            result.append(candidate)
        return tuple(result)

    def _finish_recall(
        self,
        *,
        exact: continual.RepositoryActionRecallV1,
        tier: str,
        status: str,
        candidates: tuple[continual.RepositoryActionCandidateV1, ...],
        selected_action: str | None,
        lexical: source_views.PythonRepositoryLexicalViewV1 | None,
        matched: tuple[str, ...],
        match_tiers: tuple[str, ...],
        action_tiers: tuple[str, ...],
        consensus_tier: str | None,
        factor_reads: int,
        factor_arrivals: int,
        factor_match_steps: int,
        factor_action_steps: int,
        factor_consensus_steps: int,
    ) -> FactorizedRepositoryActionRecallV1:
        work = FactorizedRepositoryActionRecallWorkV1(
            source_projection_cell_steps=exact.work.source_projection_cell_steps,
            exact_context_equality_reads=exact.work.context_equality_reads,
            exact_context_equality_arrivals=exact.work.context_equality_arrivals,
            exact_context_race_cell_steps=exact.work.context_race_cell_steps,
            exact_candidate_race_cell_steps=exact.work.candidate_race_cell_steps,
            lexical_projection_cell_steps=(
                0 if lexical is None else lexical.projection_cell_steps
            ),
            factor_equality_reads=factor_reads,
            factor_equality_arrivals=factor_arrivals,
            factor_match_cell_steps=factor_match_steps,
            factor_action_race_cell_steps=factor_action_steps,
            factor_consensus_cell_steps=factor_consensus_steps,
        )
        partial = FactorizedRepositoryActionRecallV1(
            status=status,
            tier=tier,
            memory_checkpoint_sha256=self._checkpoint_sha256,
            exact_memory_checkpoint_sha256=self._exact_memory.checkpoint_sha256,
            policy_checkpoint_sha256=exact.policy_checkpoint_sha256,
            source_sha256=exact.source_sha256,
            source_context_frame_sha256=exact.source_context_frame_sha256,
            source_context_trace_cells=exact.source_context_trace_cells,
            frontier_sha256=exact.frontier_sha256,
            action_bank_sha256s=exact.action_bank_sha256s,
            candidates=candidates,
            selected_candidate_action_sha256=selected_action,
            exact_recall_sha256=exact.sha256,
            lexical_view_sha256=None if lexical is None else lexical.sha256,
            matched_factor_semantic_sha256s=matched,
            factor_match_tier_sha256s=match_tiers,
            factor_action_tier_sha256s=action_tiers,
            factor_consensus_tier_sha256=consensus_tier,
            work=work,
            sha256="",
        )
        result = partial._replace(
            sha256=_digest(_RECALL_NAMESPACE, _recall_payload(partial._asdict()))
        )
        result.authenticate()
        self.authenticate()
        return result


def _empty_arrays() -> list[np.ndarray]:
    return [
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, ACTION_COUNT, 32), dtype=np.uint8),
        np.empty((0,), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, TRACE_LANES), dtype="<u4"),
        np.empty((0, ACTION_COUNT), dtype=np.uint8),
        np.empty((0, ACTION_COUNT), dtype=np.uint8),
        np.empty((0,), dtype=np.int8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 0), dtype=np.uint8),
    ]


def _copy_arrays(
    parent: FrozenFactorizedRepositoryActionMemoryV1 | None,
) -> list[np.ndarray]:
    if parent is None:
        return _empty_arrays()
    return [array.copy() for _, array in parent._arrays()]


def _ensure_not_duplicate(
    parent: FrozenFactorizedRepositoryActionMemoryV1,
    experience: FactorizedRepositoryActionExperienceV1,
) -> None:
    source_root = experience.source_views.exact_context.source_sha256
    if source_root in {bytes(row).hex() for row in parent._source_roots}:
        raise ValueError("repository source provenance replay is forbidden")
    if experience.sha256 in {bytes(row).hex() for row in parent._experience_roots}:
        raise ValueError("repository factor experience replay is forbidden")


def _apply_experience(
    parent: FrozenFactorizedRepositoryActionMemoryV1 | None,
    experience: FactorizedRepositoryActionExperienceV1,
) -> tuple[list[np.ndarray], int]:
    if parent is not None:
        _ensure_not_duplicate(parent, experience)
    arrays = _copy_arrays(parent)
    lessons = experience.action_lessons
    policy = lessons[0].policy_checkpoint_sha256
    actions = tuple(sorted(lessons[0].action_bank_sha256s))
    policy_values = tuple(bytes(row).hex() for row in arrays[0])
    if policy in policy_values:
        policy_index = policy_values.index(policy)
        stored_actions = tuple(bytes(row).hex() for row in arrays[1][policy_index])
        if stored_actions != actions:
            raise ValueError("factor policy action bank changed")
    else:
        if len(policy_values) >= MAX_POLICY_BANKS:
            raise ValueError("factor policy capacity exceeded")
        policy_index = len(policy_values)
        arrays[0] = np.vstack((arrays[0], _root_row(policy)))
        action_rows = np.asarray(
            [[tuple(bytes.fromhex(action)) for action in actions]],
            dtype=np.uint8,
        )
        arrays[1] = np.vstack((arrays[1], action_rows))

    factor_keys = {
        (int(arrays[2][row]), bytes(arrays[3][row]).hex()): row
        for row in range(arrays[2].shape[0])
    }
    lexical_factors = tuple(
        sorted(
            experience.source_views.lexical_view.factors,
            key=lambda factor: factor.semantic_sha256,
        )
    )
    new_factors = tuple(
        factor
        for factor in lexical_factors
        if (policy_index, factor.semantic_sha256) not in factor_keys
    )
    if arrays[2].shape[0] + len(new_factors) > MAX_FACTOR_ROWS:
        raise ValueError("repository lexical factor row capacity exceeded")
    old_factor_count = arrays[2].shape[0]
    if new_factors:
        arrays[2] = np.concatenate(
            (
                arrays[2],
                np.full(len(new_factors), policy_index, dtype=np.uint8),
            )
        )
        arrays[3] = np.vstack(
            (
                arrays[3],
                np.asarray(
                    [tuple(bytes.fromhex(factor.semantic_sha256)) for factor in new_factors],
                    dtype=np.uint8,
                ),
            )
        )
        arrays[4] = np.vstack(
            (
                arrays[4],
                np.asarray(
                    [tuple(bytes.fromhex(factor.frame_sha256)) for factor in new_factors],
                    dtype=np.uint8,
                ),
            )
        )
        arrays[5] = np.vstack(
            (
                arrays[5],
                np.asarray(
                    [factor.trace_cells for factor in new_factors],
                    dtype="<u4",
                ),
            )
        )
        zeros = np.zeros((len(new_factors), ACTION_COUNT), dtype=np.uint8)
        arrays[6] = np.vstack((arrays[6], zeros))
        arrays[7] = np.vstack((arrays[7], zeros.copy()))
        arrays[8] = np.concatenate(
            (arrays[8], np.full(len(new_factors), -1, dtype=np.int8))
        )
        arrays[11] = np.pad(
            arrays[11],
            ((0, 0), (0, len(new_factors))),
            mode="constant",
        )
        factor_keys.update(
            {
                (policy_index, factor.semantic_sha256): old_factor_count + offset
                for offset, factor in enumerate(new_factors)
            }
        )

    factor_indices: list[int] = []
    for factor in lexical_factors:
        row = factor_keys[(policy_index, factor.semantic_sha256)]
        if (
            bytes(arrays[4][row]).hex() != factor.frame_sha256
            or tuple(int(value) for value in arrays[5][row]) != factor.trace_cells
        ):
            raise ValueError("repository lexical factor projection collision")
        factor_indices.append(row)

    if arrays[9].shape[0] >= MAX_PROVENANCE_SOURCE_ROWS:
        raise ValueError("repository source provenance capacity exceeded")
    arrays[9] = np.vstack(
        (arrays[9], _root_row(experience.source_views.exact_context.source_sha256))
    )
    arrays[10] = np.vstack((arrays[10], _root_row(experience.sha256)))
    presence = np.zeros((1, arrays[2].shape[0]), dtype=np.uint8)
    if factor_indices:
        presence[0, factor_indices] = 1
    arrays[11] = np.vstack((arrays[11], presence))

    action_indices = {action: index for index, action in enumerate(actions)}
    for lesson in lessons:
        action_index = action_indices[lesson.candidate_action_sha256]
        target = arrays[6] if lesson.passed_cell == 1 else arrays[7]
        for row in factor_indices:
            target[row, action_index] = min(
                TICK_CAP,
                int(target[row, action_index]) + EVIDENCE_TICKS,
            )

    learning_steps = 0
    admitted: list[int] = []
    for row in range(arrays[2].shape[0]):
        support = int(np.sum(arrays[11][:, row], dtype=np.int64))
        supported = _support_is_admitted(support)
        learning_steps += 2 * RACE_TICKS
        action: int | None = None
        if supported:
            action, _ = _unique_action(arrays[6][row], arrays[7][row])
            learning_steps += (ACTION_COUNT + 1) * RACE_TICKS
        admitted.append(-1 if action is None else action)
    if sum(value >= 0 for value in admitted) > MAX_ADMITTED_FACTORS:
        raise ValueError("admitted repository lexical factor capacity exceeded")
    arrays[8] = np.asarray(admitted, dtype=np.int8)
    return arrays, learning_steps


def _freeze_memory(
    exact_memory: continual.FrozenContinualRepositoryActionMemoryV1,
    arrays: list[np.ndarray],
    *,
    parent_checkpoint_sha256: str | None,
    mode: str,
    new_lesson_count: int,
    factor_learning_cell_steps: int,
) -> FrozenFactorizedRepositoryActionMemoryV1:
    policies = arrays[0].shape[0]
    factors = arrays[2].shape[0]
    sources = arrays[9].shape[0]
    exact_summary = exact_memory.training_summary
    trainable = int(
        exact_summary.trainable_state_bytes
        + arrays[6].nbytes
        + arrays[7].nbytes
        + arrays[8].nbytes
    )
    checkpoint_bytes = exact_summary.checkpoint_array_bytes + sum(
        array.nbytes for array in arrays
    )
    exact_learning_steps = exact_summary.learning_cell_steps
    partial = FactorizedRepositoryActionTrainingSummaryV1(
        mode=mode,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        exact_memory_checkpoint_sha256=exact_memory.checkpoint_sha256,
        policy_count=policies,
        factor_count=factors,
        admitted_factor_count=int(np.count_nonzero(arrays[8] >= 0)),
        source_count=sources,
        new_lesson_count=new_lesson_count,
        raw_experience_retention_count=0,
        parent_mutation_count=0,
        trainable_state_bytes=trainable,
        checkpoint_array_bytes=checkpoint_bytes,
        factor_learning_cell_steps=factor_learning_cell_steps,
        learning_cell_steps=exact_learning_steps + factor_learning_cell_steps,
        checkpoint_sha256="",
    )
    frozen_arrays = tuple(
        (name, _freeze_array(array, array.dtype))
        for name, array in zip(_ARRAY_NAMES, arrays, strict=True)
    )
    checkpoint = _checkpoint_sha256(
        partial,
        exact_memory.checkpoint_sha256,
        frozen_arrays,
    )
    summary = partial._replace(checkpoint_sha256=checkpoint)
    return FrozenFactorizedRepositoryActionMemoryV1(
        exact_memory=exact_memory,
        arrays=arrays,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_factorized_repository_action_memory_v1(
    experience: FactorizedRepositoryActionExperienceV1,
    *,
    plasticity_gate: bool = True,
) -> FrozenFactorizedRepositoryActionMemoryV1:
    if type(experience) is not FactorizedRepositoryActionExperienceV1:
        raise TypeError("one exact factorized repository action experience is required")
    experience.authenticate()
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    if not plasticity_gate:
        raise ValueError("an empty factorized memory cannot start without plasticity")
    arrays, factor_steps = _apply_experience(None, experience)
    exact = continual.start_continual_repository_action_memory_v1(
        experience.action_lessons,
        plasticity_gate=True,
    )
    return _freeze_memory(
        exact,
        arrays,
        parent_checkpoint_sha256=None,
        mode="INITIAL_FACTORIZED_REPOSITORY_ACTION_MEMORY",
        new_lesson_count=len(experience.action_lessons),
        factor_learning_cell_steps=factor_steps,
    )


def advance_factorized_repository_action_memory_v1(
    parent: FrozenFactorizedRepositoryActionMemoryV1,
    experience: FactorizedRepositoryActionExperienceV1,
    *,
    plasticity_gate: bool = True,
) -> FrozenFactorizedRepositoryActionMemoryV1:
    if type(parent) is not FrozenFactorizedRepositoryActionMemoryV1:
        raise TypeError("parent must be one exact frozen factorized memory")
    if type(experience) is not FactorizedRepositoryActionExperienceV1:
        raise TypeError("one exact factorized repository action experience is required")
    experience.authenticate()
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    parent.authenticate()
    _ensure_not_duplicate(parent, experience)
    if not plasticity_gate:
        parent.authenticate()
        return parent
    parent_root = parent.checkpoint_sha256
    parent_exact_root = parent._exact_memory.checkpoint_sha256
    parent_array_roots = tuple(_array_sha256(array) for _, array in parent._arrays())
    arrays, factor_steps = _apply_experience(parent, experience)
    exact = continual.advance_continual_repository_action_memory_v1(
        parent._exact_memory,
        experience.action_lessons,
        plasticity_gate=True,
    )
    successor = _freeze_memory(
        exact,
        arrays,
        parent_checkpoint_sha256=parent_root,
        mode="REPLAY_FREE_FACTORIZED_REPOSITORY_ACTION_ADVANCE",
        new_lesson_count=len(experience.action_lessons),
        factor_learning_cell_steps=(
            parent.training_summary.factor_learning_cell_steps + factor_steps
        ),
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent._exact_memory.checkpoint_sha256 != parent_exact_root
        or tuple(_array_sha256(array) for _, array in parent._arrays())
        != parent_array_roots
    ):
        raise RuntimeError("factorized repository action advance mutated its parent")
    return successor
