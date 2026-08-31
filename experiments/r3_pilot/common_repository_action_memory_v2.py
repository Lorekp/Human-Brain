"""Authenticated exact, structural-role, and lexical repository-action memory.

The public façade consumes one already-normalized observation.  It never imports
an environment adapter and never receives source bytes.  Exact episodic recall
is authoritative; a two-cell physical applicability race admits the generic
structural-role tier; only a physical role STOP may reach the lexical tier.

Frozen checkpoints retain bounded numeric arrays and authenticated roots, not
source text, graphs, observations, lessons, experiences, or parent objects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import NamedTuple

import numpy as np

from experiments.r3_pilot import common_continual_repository_action_memory_v1 as continual
from experiments.r3_pilot import _repository_action_lexical_memory_v1 as lexical_memory
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph
from experiments.r3_pilot import common_structured_role_path_projection_v1 as role_projection
from humanbrain.resting_glif_tier_beam import (
    RETAINED,
    STOP_LEADING_TIER,
    CompiledRestingGlifTierBeam,
)

SCHEMA = "COMMON_REPOSITORY_ACTION_MEMORY_V2"

RECALLED = continual.RECALLED
EXACT_EPISODIC = "EXACT_EPISODIC"
STRUCTURAL_ROLE = "STRUCTURAL_ROLE"
LEXICAL_FACTOR = "LEXICAL_FACTOR"

STOP_RECALL_GATE = f"STOP_{SCHEMA}_RECALL_GATE"
STOP_ACTION_BANK = f"STOP_{SCHEMA}_ACTION_BANK"
STOP_EXACT_CONTEXT_GATE = f"STOP_{SCHEMA}_EXACT_CONTEXT_GATE"
STOP_EXACT_CONTEXT_AMBIGUOUS = f"STOP_{SCHEMA}_EXACT_CONTEXT_AMBIGUOUS"
STOP_EXACT_CONTEXT_COLLISION = f"STOP_{SCHEMA}_EXACT_CONTEXT_COLLISION"
STOP_EXACT_CANDIDATE_GATE = f"STOP_{SCHEMA}_EXACT_CANDIDATE_GATE"
STOP_EXACT_CANDIDATE_AMBIGUOUS = f"STOP_{SCHEMA}_EXACT_CANDIDATE_AMBIGUOUS"
STOP_STRUCTURAL_ROLE_GATE = f"STOP_{SCHEMA}_STRUCTURAL_ROLE_GATE"
STOP_STRUCTURAL_ROLE_UNKNOWN = f"STOP_{SCHEMA}_STRUCTURAL_ROLE_UNKNOWN"
STOP_STRUCTURAL_ROLE_AMBIGUOUS = f"STOP_{SCHEMA}_STRUCTURAL_ROLE_AMBIGUOUS"
STOP_STRUCTURAL_ROLE_COLLISION = f"STOP_{SCHEMA}_STRUCTURAL_ROLE_COLLISION"
STOP_STRUCTURAL_ROLE_CONFLICT = f"STOP_{SCHEMA}_STRUCTURAL_ROLE_CONFLICT"
STOP_LEXICAL_FACTOR_GATE = f"STOP_{SCHEMA}_LEXICAL_FACTOR_GATE"
STOP_LEXICAL_FACTOR_UNKNOWN = f"STOP_{SCHEMA}_LEXICAL_FACTOR_UNKNOWN"
STOP_LEXICAL_FACTOR_AMBIGUOUS = f"STOP_{SCHEMA}_LEXICAL_FACTOR_AMBIGUOUS"
STOP_LEXICAL_FACTOR_COLLISION = f"STOP_{SCHEMA}_LEXICAL_FACTOR_COLLISION"
STOP_LEXICAL_FACTOR_CONFLICT = f"STOP_{SCHEMA}_LEXICAL_FACTOR_CONFLICT"

_EXACT_WORK = (False, False, False)
_ROLE_APPLICABILITY_WORK = (True, False, False)
_ROLE_DETAILED_WORK = (True, True, False)
_LEXICAL_DETAILED_WORK = (True, False, True)
_STATUS_TIERS = {
    RECALLED: frozenset((EXACT_EPISODIC, STRUCTURAL_ROLE, LEXICAL_FACTOR)),
    STOP_RECALL_GATE: frozenset((EXACT_EPISODIC,)),
    STOP_ACTION_BANK: frozenset((EXACT_EPISODIC,)),
    STOP_EXACT_CONTEXT_GATE: frozenset((EXACT_EPISODIC,)),
    STOP_EXACT_CONTEXT_AMBIGUOUS: frozenset((EXACT_EPISODIC,)),
    STOP_EXACT_CONTEXT_COLLISION: frozenset((EXACT_EPISODIC,)),
    STOP_EXACT_CANDIDATE_GATE: frozenset((EXACT_EPISODIC,)),
    STOP_EXACT_CANDIDATE_AMBIGUOUS: frozenset((EXACT_EPISODIC,)),
    STOP_STRUCTURAL_ROLE_GATE: frozenset((STRUCTURAL_ROLE,)),
    STOP_STRUCTURAL_ROLE_UNKNOWN: frozenset((STRUCTURAL_ROLE,)),
    STOP_STRUCTURAL_ROLE_AMBIGUOUS: frozenset((STRUCTURAL_ROLE,)),
    STOP_STRUCTURAL_ROLE_COLLISION: frozenset((STRUCTURAL_ROLE,)),
    STOP_STRUCTURAL_ROLE_CONFLICT: frozenset((STRUCTURAL_ROLE,)),
    STOP_LEXICAL_FACTOR_GATE: frozenset((LEXICAL_FACTOR,)),
    STOP_LEXICAL_FACTOR_UNKNOWN: frozenset((LEXICAL_FACTOR,)),
    STOP_LEXICAL_FACTOR_AMBIGUOUS: frozenset((LEXICAL_FACTOR,)),
    STOP_LEXICAL_FACTOR_COLLISION: frozenset((LEXICAL_FACTOR,)),
    STOP_LEXICAL_FACTOR_CONFLICT: frozenset((LEXICAL_FACTOR,)),
}
_TIER_WORK_SHAPES = {
    EXACT_EPISODIC: frozenset((_EXACT_WORK,)),
    STRUCTURAL_ROLE: frozenset(
        (_EXACT_WORK, _ROLE_APPLICABILITY_WORK, _ROLE_DETAILED_WORK)
    ),
    LEXICAL_FACTOR: frozenset(
        (_ROLE_APPLICABILITY_WORK, _LEXICAL_DETAILED_WORK)
    ),
}
_STATUS_WORK_SHAPES = {
    RECALLED: frozenset((_EXACT_WORK, _ROLE_DETAILED_WORK, _LEXICAL_DETAILED_WORK)),
    STOP_RECALL_GATE: frozenset((_EXACT_WORK,)),
    STOP_ACTION_BANK: frozenset((_EXACT_WORK,)),
    STOP_EXACT_CONTEXT_GATE: frozenset((_EXACT_WORK,)),
    STOP_EXACT_CONTEXT_AMBIGUOUS: frozenset((_EXACT_WORK,)),
    STOP_EXACT_CONTEXT_COLLISION: frozenset((_EXACT_WORK,)),
    STOP_EXACT_CANDIDATE_GATE: frozenset((_EXACT_WORK,)),
    STOP_EXACT_CANDIDATE_AMBIGUOUS: frozenset((_EXACT_WORK,)),
    STOP_STRUCTURAL_ROLE_GATE: frozenset((_EXACT_WORK,)),
    STOP_STRUCTURAL_ROLE_UNKNOWN: frozenset((_ROLE_APPLICABILITY_WORK,)),
    STOP_STRUCTURAL_ROLE_AMBIGUOUS: frozenset(
        (_ROLE_APPLICABILITY_WORK, _ROLE_DETAILED_WORK)
    ),
    STOP_STRUCTURAL_ROLE_COLLISION: frozenset((_ROLE_DETAILED_WORK,)),
    STOP_STRUCTURAL_ROLE_CONFLICT: frozenset((_ROLE_DETAILED_WORK,)),
    STOP_LEXICAL_FACTOR_GATE: frozenset((_ROLE_APPLICABILITY_WORK,)),
    STOP_LEXICAL_FACTOR_UNKNOWN: frozenset(
        (_ROLE_APPLICABILITY_WORK, _LEXICAL_DETAILED_WORK)
    ),
    STOP_LEXICAL_FACTOR_AMBIGUOUS: frozenset((_LEXICAL_DETAILED_WORK,)),
    STOP_LEXICAL_FACTOR_COLLISION: frozenset((_LEXICAL_DETAILED_WORK,)),
    STOP_LEXICAL_FACTOR_CONFLICT: frozenset((_LEXICAL_DETAILED_WORK,)),
}
_STATUS_TIER_WORK_SHAPES = {
    (RECALLED, EXACT_EPISODIC): _EXACT_WORK,
    (RECALLED, STRUCTURAL_ROLE): _ROLE_DETAILED_WORK,
    (RECALLED, LEXICAL_FACTOR): _LEXICAL_DETAILED_WORK,
}

ACTION_COUNT = action_bank.CANDIDATE_COUNT
TRACE_LANES = source_views.TRACE_LANES
ROLE_CUE_CELLS = role_projection.FACTOR_CUE_CELLS
MAX_ROLE_ROWS = role_projection.MAX_FACTORS
MAX_SOURCE_ROWS = lexical_memory.MAX_PROVENANCE_SOURCE_ROWS
MIN_INDEPENDENT_SOURCE_ROOTS = lexical_memory.MIN_INDEPENDENT_SOURCE_ROOTS
TICK_CAP = continual.TICK_CAP
EVIDENCE_TICKS = continual.EVIDENCE_TICKS

RACE_TICKS = 600
CURRENT_BASE_PA = 600.0
CURRENT_GAIN_PA = 400.0
MATCH_CURRENT_GAIN_PA = 160.0
ACTION_CURRENT_MAX_PA = 7_000.0
MATCH_CURRENT_MAX_PA = 5_000.0
INHIBITION_CURRENT_PA = 7_000.0
SUPPORT_STOP_CELL = 1
ACTION_STOP_CELL = ACTION_COUNT
ROLE_APPLICABILITY_STOP_CELL = 1
ROLE_MATCH_STOP_CELL = MAX_ROLE_ROWS
LEXICAL_MATCH_STOP_CELL = source_views.MAX_SOURCE_FACTORS

MAX_RECALL_CELL_STEPS = 512_000
MAX_RECALL_ROOT_READS = 8_192
MAX_CHECKPOINT_ARRAY_BYTES = 7_352
MAX_TRAINABLE_STATE_BYTES = 192
MAX_TRANSIENT_NUMPY_ARRAY_BYTES = 384 * 1024

_OBSERVATION_NAMESPACE = b"common-repository-action-observation-v2"
_EXPERIENCE_NAMESPACE = b"common-repository-action-experience-v2"
_CHECKPOINT_NAMESPACE = b"common-repository-action-checkpoint-v2"
_RECALL_NAMESPACE = b"common-repository-action-recall-v2"

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
_ROLE_APPLICABILITY_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=2,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=ACTION_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_ROLE_MATCH_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=MAX_ROLE_ROWS + 1,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=MATCH_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)
_LEXICAL_MATCH_SELECTOR = CompiledRestingGlifTierBeam.compile(
    candidate_count=source_views.MAX_SOURCE_FACTORS + 1,
    step_count=RACE_TICKS,
    min_current_pa=CURRENT_BASE_PA,
    max_current_pa=MATCH_CURRENT_MAX_PA,
    inhibition_current_pa=INHIBITION_CURRENT_PA,
    beam_width=1,
)

_CONSTRUCTION_TOKEN = object()
_ROLE_ARRAY_NAMES = (
    "role-policy-indices",
    "role-event-semantic-roots",
    "role-path-semantic-roots",
    "role-factor-roots",
    "role-cues",
    "role-excitatory",
    "role-inhibitory",
    "role-admitted-actions",
    "v2-experience-roots",
    "source-role-presence",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(value.tobytes(order="C")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return action_bank.valid_sha256(value)


def _root_row(value: str) -> np.ndarray:
    return np.asarray([tuple(bytes.fromhex(value))], dtype=np.uint8)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=value.dtype).copy()
    result.setflags(write=False)
    return result


class RepositoryActionObservationV2(NamedTuple):
    """One authenticated normalized observation supplied outside cognition."""

    source_sha256: str
    graph_sha256: str
    producer_receipt: structured_graph.StructuredEventGraphProducerReceiptV1
    source_views: source_views.RepositorySourceViewsV1
    structural_roles: role_projection.RootedStructuredRoleProjectionV1
    sha256: str

    @property
    def exact_context(self) -> source_views.RepositorySourceContextV1:
        return self.source_views.exact_context

    @property
    def producer_receipt_sha256(self) -> str:
        return self.producer_receipt.sha256

    def authenticate(self) -> None:
        if (
            type(self.producer_receipt)
            is not structured_graph.StructuredEventGraphProducerReceiptV1
        ):
            raise TypeError("observation requires one exact graph producer receipt")
        if type(self.source_views) is not source_views.RepositorySourceViewsV1:
            raise TypeError("observation requires exact repository source views")
        if (
            type(self.structural_roles)
            is not role_projection.RootedStructuredRoleProjectionV1
        ):
            raise TypeError("observation requires exact rooted structural roles")
        self.producer_receipt.authenticate()
        self.source_views.authenticate()
        self.structural_roles.authenticate()
        payload = {
            "source": self.source_sha256,
            "graph": self.graph_sha256,
            "producer_receipt": self.producer_receipt.sha256,
            "source_views": self.source_views.sha256,
            "structural_roles": self.structural_roles.sha256,
        }
        if (
            not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.graph_sha256)
            or self.source_sha256 != self.producer_receipt.source_sha256
            or self.graph_sha256 != self.producer_receipt.graph_sha256
            or self.source_sha256 != self.source_views.exact_context.source_sha256
            or self.source_sha256 != self.source_views.lexical_view.source_sha256
            or self.graph_sha256 != self.structural_roles.graph_sha256
            or self.sha256 != _digest(_OBSERVATION_NAMESPACE, payload)
        ):
            raise ValueError("repository action observation authentication failed")


def make_repository_action_observation_v2(
    normalized_source_views: source_views.RepositorySourceViewsV1,
    structural_roles: role_projection.RootedStructuredRoleProjectionV1,
    *,
    producer_receipt: structured_graph.StructuredEventGraphProducerReceiptV1,
) -> RepositoryActionObservationV2:
    if type(normalized_source_views) is not source_views.RepositorySourceViewsV1:
        raise TypeError("exact repository source views required")
    if type(structural_roles) is not role_projection.RootedStructuredRoleProjectionV1:
        raise TypeError("exact rooted structural roles required")
    if (
        type(producer_receipt)
        is not structured_graph.StructuredEventGraphProducerReceiptV1
    ):
        raise TypeError("one exact graph producer receipt is required")
    producer_receipt.authenticate()
    payload = {
        "source": producer_receipt.source_sha256,
        "graph": producer_receipt.graph_sha256,
        "producer_receipt": producer_receipt.sha256,
        "source_views": normalized_source_views.sha256,
        "structural_roles": structural_roles.sha256,
    }
    result = RepositoryActionObservationV2(
        source_sha256=producer_receipt.source_sha256,
        graph_sha256=producer_receipt.graph_sha256,
        producer_receipt=producer_receipt,
        source_views=normalized_source_views,
        structural_roles=structural_roles,
        sha256=_digest(_OBSERVATION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class RepositoryActionExperienceV2(NamedTuple):
    observation: RepositoryActionObservationV2
    action_lessons: tuple[continual.PhysicalRepositoryActionLessonV1, ...]
    sha256: str

    def authenticate(self) -> None:
        if type(self.observation) is not RepositoryActionObservationV2:
            raise TypeError("experience requires one exact V2 observation")
        if type(self.action_lessons) is not tuple:
            raise TypeError("experience lessons must be one exact tuple")
        self.observation.authenticate()
        lexical_experience = (
            lexical_memory.make_factorized_repository_action_experience_v1(
                self.observation.source_views,
                self.action_lessons,
            )
        )
        payload = {
            "observation": self.observation.sha256,
            "lexical_experience": lexical_experience.sha256,
            "lessons": tuple(lesson.sha256 for lesson in self.action_lessons),
        }
        if self.sha256 != _digest(_EXPERIENCE_NAMESPACE, payload):
            raise ValueError("repository action experience authentication failed")


def make_repository_action_experience_v2(
    observation: RepositoryActionObservationV2,
    action_lessons: tuple[continual.PhysicalRepositoryActionLessonV1, ...],
) -> RepositoryActionExperienceV2:
    if type(observation) is not RepositoryActionObservationV2:
        raise TypeError("one exact V2 observation is required")
    if type(action_lessons) is not tuple:
        raise TypeError("action lesson cohort must be one exact tuple")
    observation.authenticate()
    lexical_experience = lexical_memory.make_factorized_repository_action_experience_v1(
        observation.source_views,
        action_lessons,
    )
    payload = {
        "observation": observation.sha256,
        "lexical_experience": lexical_experience.sha256,
        "lessons": tuple(lesson.sha256 for lesson in action_lessons),
    }
    result = RepositoryActionExperienceV2(
        observation=observation,
        action_lessons=action_lessons,
        sha256=_digest(_EXPERIENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class RepositoryActionTrainingSummaryV2(NamedTuple):
    mode: str
    parent_checkpoint_sha256: str | None
    lexical_memory_checkpoint_sha256: str
    policy_count: int
    lexical_factor_count: int
    admitted_lexical_factor_count: int
    structural_role_factor_count: int
    admitted_structural_role_factor_count: int
    source_count: int
    new_lesson_count: int
    raw_experience_retention_count: int
    parent_mutation_count: int
    trainable_state_bytes: int
    checkpoint_array_bytes: int
    structural_role_learning_cell_steps: int
    learning_cell_steps: int
    transient_numpy_array_bound_bytes: int
    checkpoint_sha256: str


class RepositoryLexicalAdmissionV2(NamedTuple):
    policy_checkpoint_sha256: str
    semantic_sha256: str
    selected_candidate_action_sha256: str | None
    distinct_source_count: int
    excitatory_ticks: tuple[int, ...]
    inhibitory_ticks: tuple[int, ...]


class RepositoryStructuralRoleAdmissionV2(NamedTuple):
    policy_checkpoint_sha256: str
    event_semantic_sha256: str
    role_semantic_sha256: str
    factor_sha256: str
    selected_candidate_action_sha256: str | None
    distinct_source_count: int
    excitatory_ticks: tuple[int, ...]
    inhibitory_ticks: tuple[int, ...]


class RepositoryActionSourceProvenanceV2(NamedTuple):
    source_sha256: str
    experience_sha256: str
    lexical_experience_sha256: str
    lexical_factor_count: int
    structural_role_factor_count: int


class RepositoryActionRecallWorkV2(NamedTuple):
    source_projection_cell_steps: int
    lexical_projection_cell_steps: int
    role_boundary_seed_cell_steps: int
    role_relay_cell_steps: int
    role_path_projection_cell_steps: int
    role_projection_synaptic_reads: int
    role_projection_cue_reads: int
    exact_context_equality_reads: int
    exact_context_equality_arrivals: int
    exact_context_race_cell_steps: int
    exact_candidate_race_cell_steps: int
    role_applicability_equality_reads: int
    role_applicability_equality_arrivals: int
    role_applicability_race_cell_steps: int
    role_factor_equality_reads: int
    role_factor_equality_arrivals: int
    role_factor_match_cell_steps: int
    role_action_race_cell_steps: int
    role_consensus_cell_steps: int
    lexical_factor_equality_reads: int
    lexical_factor_equality_arrivals: int
    lexical_factor_match_cell_steps: int
    lexical_action_race_cell_steps: int
    lexical_consensus_cell_steps: int

    @property
    def total_cell_steps(self) -> int:
        return (
            self.source_projection_cell_steps
            + self.lexical_projection_cell_steps
            + self.role_boundary_seed_cell_steps
            + self.role_relay_cell_steps
            + self.role_path_projection_cell_steps
            + self.exact_context_race_cell_steps
            + self.exact_candidate_race_cell_steps
            + self.role_applicability_race_cell_steps
            + self.role_factor_match_cell_steps
            + self.role_action_race_cell_steps
            + self.role_consensus_cell_steps
            + self.lexical_factor_match_cell_steps
            + self.lexical_action_race_cell_steps
            + self.lexical_consensus_cell_steps
        )

    @property
    def total_reads(self) -> int:
        return (
            self.role_projection_synaptic_reads
            + self.role_projection_cue_reads
            + self.exact_context_equality_reads
            + self.role_applicability_equality_reads
            + self.role_factor_equality_reads
            + self.lexical_factor_equality_reads
        )

    def authenticate(self) -> None:
        if any(type(value) is not int or value < 0 for value in self):
            raise ValueError("repository action recall work changed")
        role_detailed = (
            self.role_factor_equality_reads
            + self.role_factor_match_cell_steps
            + self.role_action_race_cell_steps
            + self.role_consensus_cell_steps
        )
        lexical_detailed = (
            self.lexical_factor_equality_reads
            + self.lexical_factor_match_cell_steps
            + self.lexical_action_race_cell_steps
            + self.lexical_consensus_cell_steps
        )
        if (
            role_detailed > 0
            and lexical_detailed > 0
            or self.total_cell_steps > MAX_RECALL_CELL_STEPS
            or self.total_reads > MAX_RECALL_ROOT_READS
        ):
            raise ValueError("repository action recall work cap changed")


def _recall_work_shape(
    work: RepositoryActionRecallWorkV2,
) -> tuple[bool, bool, bool]:
    role_applicability = any(
        (
            work.role_applicability_equality_reads,
            work.role_applicability_equality_arrivals,
            work.role_applicability_race_cell_steps,
        )
    )
    role_detailed = any(
        (
            work.role_factor_equality_reads,
            work.role_factor_equality_arrivals,
            work.role_factor_match_cell_steps,
            work.role_action_race_cell_steps,
            work.role_consensus_cell_steps,
        )
    )
    lexical_detailed = any(
        (
            work.lexical_factor_equality_reads,
            work.lexical_factor_equality_arrivals,
            work.lexical_factor_match_cell_steps,
            work.lexical_action_race_cell_steps,
            work.lexical_consensus_cell_steps,
        )
    )
    return role_applicability, role_detailed, lexical_detailed


class RepositoryActionRecallV2(NamedTuple):
    status: str
    tier: str
    memory_checkpoint_sha256: str
    policy_checkpoint_sha256: str
    observation_sha256: str
    source_sha256: str
    source_context_frame_sha256: str
    source_context_trace_cells: tuple[int, ...]
    frontier_sha256: str
    action_bank_sha256s: tuple[str, ...]
    candidates: tuple[action_bank.RepositoryActionCandidateRecordV1, ...]
    selected_candidate_action_sha256: str | None
    work: RepositoryActionRecallWorkV2
    sha256: str

    def authenticate(self) -> None:
        if type(self.candidates) is not tuple:
            raise TypeError("recall candidates must be one exact tuple")
        for candidate in self.candidates:
            if type(candidate) is not action_bank.RepositoryActionCandidateRecordV1:
                raise TypeError("exact repository candidate records required")
            candidate.authenticate()
        candidate_actions = tuple(
            sorted(
                action_bank.repository_action_sha256_v1(
                    self.policy_checkpoint_sha256,
                    candidate.applied_relation_receipt_sha256s,
                )
                for candidate in self.candidates
            )
        )
        if type(self.work) is not RepositoryActionRecallWorkV2:
            raise TypeError("exact repository action recall work required")
        self.work.authenticate()
        work_shape = _recall_work_shape(self.work)
        payload = _recall_payload(self._asdict())
        if (
            self.status not in _STATUS_TIERS
            or self.tier not in _STATUS_TIERS.get(self.status, ())
            or work_shape not in _TIER_WORK_SHAPES.get(self.tier, ())
            or work_shape not in _STATUS_WORK_SHAPES.get(self.status, ())
            or (
                self.status == RECALLED
                and work_shape
                != _STATUS_TIER_WORK_SHAPES.get((self.status, self.tier))
            )
            or any(
                not _valid_sha256(value)
                for value in (
                    self.memory_checkpoint_sha256,
                    self.policy_checkpoint_sha256,
                    self.observation_sha256,
                    self.source_sha256,
                    self.source_context_frame_sha256,
                    self.frontier_sha256,
                )
            )
            or type(self.source_context_trace_cells) is not tuple
            or len(self.source_context_trace_cells) != TRACE_LANES
            or any(type(cell) is not int for cell in self.source_context_trace_cells)
            or self.action_bank_sha256s != tuple(sorted(self.action_bank_sha256s))
            or len(self.action_bank_sha256s) != ACTION_COUNT
            or len(set(self.action_bank_sha256s)) != ACTION_COUNT
            or len(self.candidates) not in (0, ACTION_COUNT)
            or len({candidate.sha256 for candidate in self.candidates})
            != len(self.candidates)
            or (
                self.candidates
                and candidate_actions != self.action_bank_sha256s
            )
            or (
                self.status == RECALLED
                and (
                    len(self.candidates) != ACTION_COUNT
                    or self.selected_candidate_action_sha256
                    not in self.action_bank_sha256s
                )
            )
            or (
                self.status != RECALLED
                and self.selected_candidate_action_sha256 is not None
            )
            or self.sha256 != _digest(_RECALL_NAMESPACE, payload)
        ):
            raise ValueError("repository action recall authentication failed")


def _recall_payload(record: dict[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload.pop("sha256", None)
    payload["candidates"] = tuple(item.sha256 for item in record["candidates"])
    work = record["work"]
    if type(work) is RepositoryActionRecallWorkV2:
        payload["work"] = work._asdict()
    return payload


def _checkpoint_sha256(
    summary: RepositoryActionTrainingSummaryV2,
    lexical_checkpoint_sha256: str,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    summary_payload = summary._asdict()
    summary_payload.pop("checkpoint_sha256", None)
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "summary": summary_payload,
            "lexical_memory": lexical_checkpoint_sha256,
            "arrays": tuple(
                (name, str(array.dtype), array.shape, _array_sha256(array))
                for name, array in arrays
            ),
        },
    )


class FrozenRepositoryActionMemoryV2:
    """One immutable deep façade over exact, role, and lexical memory."""

    __slots__ = (
        "_lexical_memory",
        "_role_policy_indices",
        "_role_event_semantic_roots",
        "_role_path_semantic_roots",
        "_role_factor_roots",
        "_role_cues",
        "_role_excitatory",
        "_role_inhibitory",
        "_role_admitted_actions",
        "_v2_experience_roots",
        "_source_role_presence",
        "_summary",
        "_checkpoint_sha256",
    )

    def __init__(
        self,
        *,
        lexical: lexical_memory.FrozenFactorizedRepositoryActionMemoryV1,
        arrays: list[np.ndarray],
        summary: RepositoryActionTrainingSummaryV2,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("V2 repository checkpoints require their builder")
        if type(lexical) is not lexical_memory.FrozenFactorizedRepositoryActionMemoryV1:
            raise TypeError("one exact factorized repository memory child is required")
        if len(arrays) != len(_ROLE_ARRAY_NAMES):
            raise ValueError("V2 repository role arrays changed")
        self._lexical_memory = lexical
        (
            self._role_policy_indices,
            self._role_event_semantic_roots,
            self._role_path_semantic_roots,
            self._role_factor_roots,
            self._role_cues,
            self._role_excitatory,
            self._role_inhibitory,
            self._role_admitted_actions,
            self._v2_experience_roots,
            self._source_role_presence,
        ) = tuple(_freeze_array(array) for array in arrays)
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            summary, lexical.checkpoint_sha256, self._arrays()
        )
        self.authenticate()

    def _arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        values = (
            self._role_policy_indices,
            self._role_event_semantic_roots,
            self._role_path_semantic_roots,
            self._role_factor_roots,
            self._role_cues,
            self._role_excitatory,
            self._role_inhibitory,
            self._role_admitted_actions,
            self._v2_experience_roots,
            self._source_role_presence,
        )
        return tuple(zip(_ROLE_ARRAY_NAMES, values, strict=True))

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RepositoryActionTrainingSummaryV2:
        self.authenticate()
        return self._summary

    @property
    def lexical_admissions(self) -> tuple[RepositoryLexicalAdmissionV2, ...]:
        self.authenticate()
        return tuple(
            RepositoryLexicalAdmissionV2(*admission)
            for admission in self._lexical_memory.factor_admissions
        )

    @property
    def structural_role_admissions(
        self,
    ) -> tuple[RepositoryStructuralRoleAdmissionV2, ...]:
        self.authenticate()
        result: list[RepositoryStructuralRoleAdmissionV2] = []
        for row in range(self._role_policy_indices.shape[0]):
            policy_index = int(self._role_policy_indices[row])
            admitted = int(self._role_admitted_actions[row])
            result.append(
                RepositoryStructuralRoleAdmissionV2(
                    policy_checkpoint_sha256=bytes(
                        self._lexical_memory._policy_roots[policy_index]
                    ).hex(),
                    event_semantic_sha256=bytes(
                        self._role_event_semantic_roots[row]
                    ).hex(),
                    role_semantic_sha256=bytes(
                        self._role_path_semantic_roots[row]
                    ).hex(),
                    factor_sha256=bytes(self._role_factor_roots[row]).hex(),
                    selected_candidate_action_sha256=(
                        None
                        if admitted < 0
                        else bytes(
                            self._lexical_memory._policy_action_roots[
                                policy_index, admitted
                            ]
                        ).hex()
                    ),
                    distinct_source_count=int(
                        np.sum(self._source_role_presence[:, row], dtype=np.int64)
                    ),
                    excitatory_ticks=tuple(
                        int(value) for value in self._role_excitatory[row]
                    ),
                    inhibitory_ticks=tuple(
                        int(value) for value in self._role_inhibitory[row]
                    ),
                )
            )
        return tuple(result)

    @property
    def source_provenance(self) -> tuple[RepositoryActionSourceProvenanceV2, ...]:
        self.authenticate()
        return tuple(
            RepositoryActionSourceProvenanceV2(
                source_sha256=bytes(self._lexical_memory._source_roots[row]).hex(),
                experience_sha256=bytes(self._v2_experience_roots[row]).hex(),
                lexical_experience_sha256=bytes(
                    self._lexical_memory._experience_roots[row]
                ).hex(),
                lexical_factor_count=int(
                    np.sum(
                        self._lexical_memory._source_factor_presence[row],
                        dtype=np.int64,
                    )
                ),
                structural_role_factor_count=int(
                    np.sum(self._source_role_presence[row], dtype=np.int64)
                ),
            )
            for row in range(self._source_role_presence.shape[0])
        )

    def authenticate(self) -> None:
        self._lexical_memory.authenticate()
        arrays = self._arrays()
        if any(array.flags.writeable for _, array in arrays):
            raise RuntimeError("V2 repository checkpoint became mutable")
        role_count = self._role_policy_indices.shape[0]
        source_count = self._lexical_memory._source_roots.shape[0]
        expected_shapes = (
            (role_count,),
            (role_count, 32),
            (role_count, 32),
            (role_count, 32),
            (role_count, ROLE_CUE_CELLS),
            (role_count, ACTION_COUNT),
            (role_count, ACTION_COUNT),
            (role_count,),
            (source_count, 32),
            (source_count, role_count),
        )
        if tuple(array.shape for _, array in arrays) != expected_shapes:
            raise RuntimeError("V2 repository checkpoint shape changed")
        policies = self._lexical_memory._policy_roots.shape[0]
        keys = tuple(
            (
                int(self._role_policy_indices[row]),
                bytes(self._role_event_semantic_roots[row]),
                bytes(self._role_path_semantic_roots[row]),
            )
            for row in range(role_count)
        )
        factor_bindings = tuple(
            (
                int(self._role_policy_indices[row]),
                bytes(self._role_factor_roots[row]),
            )
            for row in range(role_count)
        )
        if (
            not 0 <= role_count <= MAX_ROLE_ROWS
            or not 1 <= source_count <= MAX_SOURCE_ROWS
            or len(set(keys)) != role_count
            or np.any(self._role_policy_indices >= policies)
            or len(set(factor_bindings)) != role_count
            or len({bytes(row) for row in self._v2_experience_roots})
            != source_count
            or np.any(self._source_role_presence > 1)
            or np.any(self._role_excitatory > TICK_CAP)
            or np.any(self._role_inhibitory > TICK_CAP)
            or np.any(self._role_admitted_actions < -1)
            or np.any(self._role_admitted_actions >= ACTION_COUNT)
        ):
            raise RuntimeError("V2 repository checkpoint values changed")
        lexical_summary = self._lexical_memory.training_summary
        admitted_count = int(np.count_nonzero(self._role_admitted_actions >= 0))
        trainable = int(
            lexical_summary.trainable_state_bytes
            + self._role_excitatory.nbytes
            + self._role_inhibitory.nbytes
            + self._role_admitted_actions.nbytes
        )
        checkpoint_bytes = lexical_summary.checkpoint_array_bytes + sum(
            array.nbytes for _, array in arrays
        )
        summary = self._summary
        if (
            summary.lexical_memory_checkpoint_sha256
            != self._lexical_memory.checkpoint_sha256
            or summary.policy_count != policies
            or summary.lexical_factor_count != lexical_summary.factor_count
            or summary.admitted_lexical_factor_count
            != lexical_summary.admitted_factor_count
            or summary.structural_role_factor_count != role_count
            or summary.admitted_structural_role_factor_count != admitted_count
            or summary.source_count != source_count
            or summary.raw_experience_retention_count != 0
            or summary.parent_mutation_count != 0
            or summary.trainable_state_bytes != trainable
            or summary.checkpoint_array_bytes != checkpoint_bytes
            or summary.learning_cell_steps
            != (
                lexical_summary.learning_cell_steps
                + summary.structural_role_learning_cell_steps
            )
            or summary.transient_numpy_array_bound_bytes
            != MAX_TRANSIENT_NUMPY_ARRAY_BYTES
            or trainable > MAX_TRAINABLE_STATE_BYTES
            or checkpoint_bytes > MAX_CHECKPOINT_ARRAY_BYTES
            or summary.checkpoint_sha256 != self._checkpoint_sha256
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                summary, self._lexical_memory.checkpoint_sha256, arrays
            )
        ):
            raise RuntimeError("V2 repository training summary changed")

    def recall_attested_observation(
        self,
        observation: RepositoryActionObservationV2,
        bank: action_bank.FrozenRepositoryActionBankV1,
        *,
        candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
        | None = None,
        recall_gate: bool = True,
        exact_context_gate: bool = True,
        exact_candidate_gate: bool = True,
        structural_role_gate: bool = True,
        lexical_factor_gate: bool = True,
    ) -> RepositoryActionRecallV2:
        self.authenticate()
        if type(observation) is not RepositoryActionObservationV2:
            raise TypeError("recall requires one exact V2 observation")
        if type(bank) is not action_bank.FrozenRepositoryActionBankV1:
            raise TypeError("recall requires one exact repository action bank")
        observation.authenticate()
        bank.authenticate()
        if any(
            type(gate) is not bool
            for gate in (
                recall_gate,
                exact_context_gate,
                exact_candidate_gate,
                structural_role_gate,
                lexical_factor_gate,
            )
        ):
            raise TypeError("repository action recall gates must be exact booleans")

        supplied, records_authentic = _candidate_records(bank, candidate_records)
        exact = self._lexical_memory._exact_memory.recall_attested_context(
            observation.exact_context,
            bank,
            candidate_records=candidate_records,
            recall_gate=recall_gate,
            context_gate=exact_context_gate,
            candidate_gate=exact_candidate_gate,
        )
        exact.authenticate()
        work = _base_recall_work(observation, exact)
        if exact.status not in {
            continual.STOP_CONTEXT_UNKNOWN,
            continual.STOP_CANDIDATE_UNKNOWN,
        }:
            status = _map_exact_status(exact.status)
            return self._finish_recall(
                observation,
                bank,
                status=status,
                tier=EXACT_EPISODIC,
                candidates=supplied if records_authentic else (),
                selected=exact.selected_candidate_action_sha256,
                work=work,
            )
        if not structural_role_gate:
            return self._finish_recall(
                observation,
                bank,
                status=STOP_STRUCTURAL_ROLE_GATE,
                tier=STRUCTURAL_ROLE,
                candidates=supplied if records_authentic else (),
                selected=None,
                work=work,
            )

        policy_index = self._policy_index(bank)
        admitted_role_rows = tuple(
            row
            for row in range(self._role_policy_indices.shape[0])
            if int(self._role_policy_indices[row]) == policy_index
            and int(self._role_admitted_actions[row]) >= 0
        )
        observed_roles = observation.structural_roles.factors
        pair_reads = len(admitted_role_rows) * len(observed_roles) * 2
        pair_arrivals = 0
        pair_matches = 0
        for row in admitted_role_rows:
            stored_event = bytes(self._role_event_semantic_roots[row]).hex()
            stored_role = bytes(self._role_path_semantic_roots[row]).hex()
            for factor in observed_roles:
                event_equal = stored_event == factor.event_root.semantic_sha256
                role_equal = stored_role == factor.role_path.semantic_sha256
                pair_arrivals += int(event_equal) + int(role_equal)
                pair_matches += int(event_equal and role_equal)
        applicability_currents = np.full((1, 2), CURRENT_BASE_PA, dtype=np.float64)
        if pair_matches:
            applicability_currents[0, 0] += CURRENT_GAIN_PA * 2
        applicability_currents[0, ROLE_APPLICABILITY_STOP_CELL] += CURRENT_GAIN_PA
        applicability = _ROLE_APPLICABILITY_SELECTOR.select_many(
            applicability_currents, stop_cell=ROLE_APPLICABILITY_STOP_CELL
        )[0]
        work = work._replace(
            role_applicability_equality_reads=pair_reads,
            role_applicability_equality_arrivals=pair_arrivals,
            role_applicability_race_cell_steps=2 * RACE_TICKS,
        )
        if (
            applicability.status == RETAINED
            and applicability.retained_cells == (0,)
            and applicability.retained_candidate_cells == (0,)
        ):
            status, selected, role_work = self._recall_structural_role(
                policy_index, admitted_role_rows, observed_roles
            )
            work = work._replace(**role_work)
            return self._finish_recall(
                observation,
                bank,
                status=status,
                tier=STRUCTURAL_ROLE,
                candidates=supplied if records_authentic else (),
                selected=selected,
                work=work,
            )
        if not (
            applicability.status == STOP_LEADING_TIER
            and applicability.retained_cells == (ROLE_APPLICABILITY_STOP_CELL,)
            and applicability.retained_candidate_cells == ()
        ):
            return self._finish_recall(
                observation,
                bank,
                status=STOP_STRUCTURAL_ROLE_AMBIGUOUS,
                tier=STRUCTURAL_ROLE,
                candidates=supplied if records_authentic else (),
                selected=None,
                work=work,
            )
        if not lexical_factor_gate:
            return self._finish_recall(
                observation,
                bank,
                status=STOP_LEXICAL_FACTOR_GATE,
                tier=LEXICAL_FACTOR,
                candidates=supplied if records_authentic else (),
                selected=None,
                work=work,
            )
        status, selected, lexical_work = self._recall_lexical(
            policy_index, observation.source_views.lexical_view.factors
        )
        work = work._replace(**lexical_work)
        return self._finish_recall(
            observation,
            bank,
            status=status,
            tier=LEXICAL_FACTOR,
            candidates=supplied if records_authentic else (),
            selected=selected,
            work=work,
        )

    def _policy_index(self, bank: action_bank.FrozenRepositoryActionBankV1) -> int:
        matches = tuple(
            row
            for row, root in enumerate(self._lexical_memory._policy_roots)
            if bytes(root).hex() == bank.policy_checkpoint_sha256
        )
        if len(matches) != 1:
            raise RuntimeError("exact child admitted a bank outside V2 state")
        return matches[0]

    def _recall_structural_role(
        self,
        policy_index: int,
        admitted_rows: tuple[int, ...],
        observed: tuple[role_projection.StructuredEventRoleFactorV1, ...],
    ) -> tuple[str, str | None, dict[str, int]]:
        cues = (
            np.asarray([factor.cue_cells for factor in observed], dtype="<u4")
            if observed
            else np.empty((0, ROLE_CUE_CELLS), dtype="<u4")
        )
        currents = np.full(
            (len(admitted_rows), MAX_ROLE_ROWS + 1),
            CURRENT_BASE_PA,
            dtype=np.float64,
        )
        arrivals_rows: list[np.ndarray] = []
        for local, row in enumerate(admitted_rows):
            arrivals = (
                np.count_nonzero(cues == self._role_cues[row], axis=1).astype(np.int16)
                if observed
                else np.empty((0,), dtype=np.int16)
            )
            arrivals_rows.append(arrivals)
            if observed:
                currents[local, : len(observed)] += MATCH_CURRENT_GAIN_PA * arrivals
            currents[local, ROLE_MATCH_STOP_CELL] += MATCH_CURRENT_GAIN_PA * (
                ROLE_CUE_CELLS - 1
            )
        physicals = (
            _ROLE_MATCH_SELECTOR.select_many(currents, stop_cell=ROLE_MATCH_STOP_CELL)
            if admitted_rows
            else ()
        )
        reads = len(admitted_rows) * len(observed) * ROLE_CUE_CELLS
        arrivals_total = sum(
            int(np.sum(arrivals, dtype=np.int64)) for arrivals in arrivals_rows
        )
        matched_rows: list[int] = []
        collision = False
        ambiguous = False
        for row, physical in zip(admitted_rows, physicals, strict=True):
            stored_pair = (
                bytes(self._role_event_semantic_roots[row]).hex(),
                bytes(self._role_path_semantic_roots[row]).hex(),
            )
            semantic_indices = tuple(
                index
                for index, factor in enumerate(observed)
                if stored_pair
                == (
                    factor.event_root.semantic_sha256,
                    factor.role_path.semantic_sha256,
                )
            )
            if physical.status != RETAINED:
                collision = collision or bool(semantic_indices)
                continue
            if len(physical.retained_candidate_cells) != 1:
                ambiguous = True
                break
            observed_index = physical.retained_candidate_cells[0]
            if not 0 <= observed_index < len(observed):
                collision = collision or bool(semantic_indices)
                continue
            factor = observed[observed_index]
            if observed_index not in semantic_indices:
                collision = True
                break
            if (
                bytes(self._role_factor_roots[row]).hex() != factor.sha256
                or tuple(int(value) for value in self._role_cues[row])
                != factor.cue_cells
            ):
                collision = True
                break
            matched_rows.append(row)
        match_steps = len(admitted_rows) * (MAX_ROLE_ROWS + 1) * RACE_TICKS
        base_work = {
            "role_factor_equality_reads": reads,
            "role_factor_equality_arrivals": arrivals_total,
            "role_factor_match_cell_steps": match_steps,
            "role_action_race_cell_steps": 0,
            "role_consensus_cell_steps": 0,
        }
        if ambiguous:
            return STOP_STRUCTURAL_ROLE_AMBIGUOUS, None, base_work
        if collision or not matched_rows:
            return STOP_STRUCTURAL_ROLE_COLLISION, None, base_work
        actions, action_ambiguous = self._row_action_votes(
            matched_rows,
            self._role_excitatory,
            self._role_inhibitory,
            self._role_admitted_actions,
        )
        base_work["role_action_race_cell_steps"] = (
            len(matched_rows) * (ACTION_COUNT + 1) * RACE_TICKS
        )
        if action_ambiguous:
            return STOP_STRUCTURAL_ROLE_AMBIGUOUS, None, base_work
        if len(set(actions)) != 1:
            return STOP_STRUCTURAL_ROLE_CONFLICT, None, base_work
        selected, consensus_ok = self._consensus_action(actions)
        base_work["role_consensus_cell_steps"] = (ACTION_COUNT + 1) * RACE_TICKS
        if not consensus_ok:
            return STOP_STRUCTURAL_ROLE_AMBIGUOUS, None, base_work
        return (
            RECALLED,
            bytes(self._lexical_memory._policy_action_roots[policy_index, selected]).hex(),
            base_work,
        )

    def _recall_lexical(
        self,
        policy_index: int,
        observed: tuple[source_views.RepositoryLexicalFactorV1, ...],
    ) -> tuple[str, str | None, dict[str, int]]:
        admitted_rows = tuple(
            row
            for row in range(self._lexical_memory._factor_policy_indices.shape[0])
            if int(self._lexical_memory._factor_policy_indices[row]) == policy_index
            and int(self._lexical_memory._admitted_actions[row]) >= 0
        )
        cues = (
            np.asarray([factor.trace_cells for factor in observed], dtype="<u4")
            if observed
            else np.empty((0, TRACE_LANES), dtype="<u4")
        )
        currents = np.full(
            (len(admitted_rows), source_views.MAX_SOURCE_FACTORS + 1),
            CURRENT_BASE_PA,
            dtype=np.float64,
        )
        arrivals_rows: list[np.ndarray] = []
        for local, row in enumerate(admitted_rows):
            arrivals = (
                np.count_nonzero(
                    cues == self._lexical_memory._factor_cues[row], axis=1
                ).astype(np.int16)
                if observed
                else np.empty((0,), dtype=np.int16)
            )
            arrivals_rows.append(arrivals)
            if observed:
                currents[local, : len(observed)] += MATCH_CURRENT_GAIN_PA * arrivals
            currents[local, LEXICAL_MATCH_STOP_CELL] += MATCH_CURRENT_GAIN_PA * (
                TRACE_LANES - 1
            )
        physicals = (
            _LEXICAL_MATCH_SELECTOR.select_many(
                currents, stop_cell=LEXICAL_MATCH_STOP_CELL
            )
            if admitted_rows
            else ()
        )
        reads = len(admitted_rows) * len(observed) * TRACE_LANES
        arrivals_total = sum(
            int(np.sum(arrivals, dtype=np.int64)) for arrivals in arrivals_rows
        )
        matched_rows: list[int] = []
        collision = False
        ambiguous = False
        for row, physical in zip(admitted_rows, physicals, strict=True):
            stored_semantic = bytes(
                self._lexical_memory._factor_semantic_roots[row]
            ).hex()
            semantic_indices = tuple(
                index
                for index, factor in enumerate(observed)
                if factor.semantic_sha256 == stored_semantic
            )
            if physical.status != RETAINED:
                collision = collision or bool(semantic_indices)
                continue
            if len(physical.retained_candidate_cells) != 1:
                ambiguous = True
                break
            observed_index = physical.retained_candidate_cells[0]
            if not 0 <= observed_index < len(observed):
                collision = collision or bool(semantic_indices)
                continue
            factor = observed[observed_index]
            if observed_index not in semantic_indices:
                collision = True
                break
            if (
                bytes(self._lexical_memory._factor_frame_roots[row]).hex()
                != factor.frame_sha256
                or tuple(
                    int(value) for value in self._lexical_memory._factor_cues[row]
                )
                != factor.trace_cells
            ):
                collision = True
                break
            matched_rows.append(row)
        match_steps = (
            len(admitted_rows)
            * (source_views.MAX_SOURCE_FACTORS + 1)
            * RACE_TICKS
        )
        base_work = {
            "lexical_factor_equality_reads": reads,
            "lexical_factor_equality_arrivals": arrivals_total,
            "lexical_factor_match_cell_steps": match_steps,
            "lexical_action_race_cell_steps": 0,
            "lexical_consensus_cell_steps": 0,
        }
        if ambiguous:
            return STOP_LEXICAL_FACTOR_AMBIGUOUS, None, base_work
        if collision:
            return STOP_LEXICAL_FACTOR_COLLISION, None, base_work
        if not matched_rows:
            return STOP_LEXICAL_FACTOR_UNKNOWN, None, base_work
        actions, action_ambiguous = self._row_action_votes(
            matched_rows,
            self._lexical_memory._factor_excitatory,
            self._lexical_memory._factor_inhibitory,
            self._lexical_memory._admitted_actions,
        )
        base_work["lexical_action_race_cell_steps"] = (
            len(matched_rows) * (ACTION_COUNT + 1) * RACE_TICKS
        )
        if action_ambiguous:
            return STOP_LEXICAL_FACTOR_AMBIGUOUS, None, base_work
        if len(set(actions)) != 1:
            return STOP_LEXICAL_FACTOR_CONFLICT, None, base_work
        selected, consensus_ok = self._consensus_action(actions)
        base_work["lexical_consensus_cell_steps"] = (ACTION_COUNT + 1) * RACE_TICKS
        if not consensus_ok:
            return STOP_LEXICAL_FACTOR_AMBIGUOUS, None, base_work
        return (
            RECALLED,
            bytes(self._lexical_memory._policy_action_roots[policy_index, selected]).hex(),
            base_work,
        )

    @staticmethod
    def _row_action_votes(
        rows: Sequence[int],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        admitted: np.ndarray,
    ) -> tuple[tuple[int, ...], bool]:
        currents = np.full(
            (len(rows), ACTION_COUNT + 1), CURRENT_BASE_PA, dtype=np.float64
        )
        for local, row in enumerate(rows):
            for action in range(ACTION_COUNT):
                drive = max(
                    0,
                    int(excitatory[row, action])
                    - int(inhibitory[row, action]),
                )
                currents[local, action] += CURRENT_GAIN_PA * drive
            currents[local, ACTION_STOP_CELL] += CURRENT_GAIN_PA
        physicals = _ACTION_SELECTOR.select_many(currents, stop_cell=ACTION_STOP_CELL)
        votes: list[int] = []
        for row, physical in zip(rows, physicals, strict=True):
            positives = tuple(
                action
                for action in range(ACTION_COUNT)
                if int(excitatory[row, action]) > 0
            )
            if (
                len(positives) != 1
                or physical.status != RETAINED
                or physical.retained_candidate_cells != positives
                or int(admitted[row]) != positives[0]
            ):
                return (), True
            votes.append(positives[0])
        return tuple(votes), False

    @staticmethod
    def _consensus_action(votes: Sequence[int]) -> tuple[int, bool]:
        currents = np.full((1, ACTION_COUNT + 1), CURRENT_BASE_PA, dtype=np.float64)
        for vote in votes:
            currents[0, vote] = min(
                ACTION_CURRENT_MAX_PA, currents[0, vote] + CURRENT_GAIN_PA
            )
        physical = _ACTION_SELECTOR.select_many(
            currents, stop_cell=ACTION_STOP_CELL
        )[0]
        selected = votes[0]
        return selected, (
            physical.status == RETAINED
            and physical.retained_candidate_cells == (selected,)
        )

    def _finish_recall(
        self,
        observation: RepositoryActionObservationV2,
        bank: action_bank.FrozenRepositoryActionBankV1,
        *,
        status: str,
        tier: str,
        candidates: tuple[action_bank.RepositoryActionCandidateRecordV1, ...],
        selected: str | None,
        work: RepositoryActionRecallWorkV2,
    ) -> RepositoryActionRecallV2:
        work.authenticate()
        partial = RepositoryActionRecallV2(
            status=status,
            tier=tier,
            memory_checkpoint_sha256=self._checkpoint_sha256,
            policy_checkpoint_sha256=bank.policy_checkpoint_sha256,
            observation_sha256=observation.sha256,
            source_sha256=observation.source_sha256,
            source_context_frame_sha256=observation.exact_context.frame_sha256,
            source_context_trace_cells=observation.exact_context.trace_cells,
            frontier_sha256=bank.frontier_sha256,
            action_bank_sha256s=tuple(sorted(bank.action_sha256s)),
            candidates=candidates,
            selected_candidate_action_sha256=selected,
            work=work,
            sha256="",
        )
        result = partial._replace(
            sha256=_digest(_RECALL_NAMESPACE, _recall_payload(partial._asdict()))
        )
        result.authenticate()
        self.authenticate()
        return result


def _candidate_records(
    bank: action_bank.FrozenRepositoryActionBankV1,
    candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
    | None,
) -> tuple[tuple[action_bank.RepositoryActionCandidateRecordV1, ...], bool]:
    supplied = bank.records if candidate_records is None else tuple(candidate_records)
    canonical = {record.sha256: record for record in bank.records}
    authentic = (
        len(supplied) == ACTION_COUNT
        and all(
            type(record) is action_bank.RepositoryActionCandidateRecordV1
            for record in supplied
        )
        and len({record.sha256 for record in supplied}) == ACTION_COUNT
        and all(canonical.get(record.sha256) == record for record in supplied)
    )
    return supplied, authentic


def _map_exact_status(status: str) -> str:
    mapping = {
        continual.RECALLED: RECALLED,
        continual.STOP_RECALL_GATE: STOP_RECALL_GATE,
        continual.STOP_ACTION_BANK: STOP_ACTION_BANK,
        continual.STOP_CONTEXT_GATE: STOP_EXACT_CONTEXT_GATE,
        continual.STOP_CONTEXT_AMBIGUOUS: STOP_EXACT_CONTEXT_AMBIGUOUS,
        continual.STOP_CONTEXT_COLLISION: STOP_EXACT_CONTEXT_COLLISION,
        continual.STOP_CANDIDATE_GATE: STOP_EXACT_CANDIDATE_GATE,
        continual.STOP_CANDIDATE_AMBIGUOUS: STOP_EXACT_CANDIDATE_AMBIGUOUS,
    }
    try:
        return mapping[status]
    except KeyError as exc:
        raise RuntimeError("unmapped exact repository recall status") from exc


def _base_recall_work(
    observation: RepositoryActionObservationV2,
    exact: continual.RepositoryActionRecallV1,
) -> RepositoryActionRecallWorkV2:
    role_work = observation.structural_roles.work
    return RepositoryActionRecallWorkV2(
        source_projection_cell_steps=observation.exact_context.projection_cell_steps,
        lexical_projection_cell_steps=(
            observation.source_views.lexical_view.projection_cell_steps
        ),
        role_boundary_seed_cell_steps=role_work.boundary_seed_cell_steps,
        role_relay_cell_steps=role_work.relay_cell_steps,
        role_path_projection_cell_steps=role_work.role_projection_cell_steps,
        role_projection_synaptic_reads=role_work.reverse_synaptic_reads,
        role_projection_cue_reads=role_work.cue_cell_reads,
        exact_context_equality_reads=exact.work.context_equality_reads,
        exact_context_equality_arrivals=exact.work.context_equality_arrivals,
        exact_context_race_cell_steps=exact.work.context_race_cell_steps,
        exact_candidate_race_cell_steps=exact.work.candidate_race_cell_steps,
        role_applicability_equality_reads=0,
        role_applicability_equality_arrivals=0,
        role_applicability_race_cell_steps=0,
        role_factor_equality_reads=0,
        role_factor_equality_arrivals=0,
        role_factor_match_cell_steps=0,
        role_action_race_cell_steps=0,
        role_consensus_cell_steps=0,
        lexical_factor_equality_reads=0,
        lexical_factor_equality_arrivals=0,
        lexical_factor_match_cell_steps=0,
        lexical_action_race_cell_steps=0,
        lexical_consensus_cell_steps=0,
    )


def _empty_role_arrays() -> list[np.ndarray]:
    return [
        np.empty((0,), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, ROLE_CUE_CELLS), dtype="<u4"),
        np.empty((0, ACTION_COUNT), dtype=np.uint8),
        np.empty((0, ACTION_COUNT), dtype=np.uint8),
        np.empty((0,), dtype=np.int8),
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 0), dtype=np.uint8),
    ]


def _copy_role_arrays(parent: FrozenRepositoryActionMemoryV2 | None) -> list[np.ndarray]:
    if parent is None:
        return _empty_role_arrays()
    return [array.copy() for _, array in parent._arrays()]


def _ensure_not_duplicate(
    parent: FrozenRepositoryActionMemoryV2,
    experience: RepositoryActionExperienceV2,
) -> None:
    source_root = experience.observation.source_sha256
    if source_root in {
        bytes(row).hex() for row in parent._lexical_memory._source_roots
    }:
        raise ValueError("repository source provenance replay is forbidden")


def _apply_structural_role_experience(
    parent: FrozenRepositoryActionMemoryV2 | None,
    experience: RepositoryActionExperienceV2,
    lexical: lexical_memory.FrozenFactorizedRepositoryActionMemoryV1,
) -> tuple[list[np.ndarray], int]:
    if parent is not None:
        _ensure_not_duplicate(parent, experience)
    arrays = _copy_role_arrays(parent)
    policy = experience.action_lessons[0].policy_checkpoint_sha256
    policy_values = tuple(bytes(row).hex() for row in lexical._policy_roots)
    if policy not in policy_values:
        raise RuntimeError("lexical child omitted the role policy")
    policy_index = policy_values.index(policy)
    keys = {
        (
            int(arrays[0][row]),
            bytes(arrays[1][row]).hex(),
            bytes(arrays[2][row]).hex(),
        ): row
        for row in range(arrays[0].shape[0])
    }
    factors = tuple(
        sorted(
            experience.observation.structural_roles.factors,
            key=lambda factor: (
                factor.event_root.semantic_sha256,
                factor.role_path.semantic_sha256,
            ),
        )
    )
    new_factors = tuple(
        factor
        for factor in factors
        if (
            policy_index,
            factor.event_root.semantic_sha256,
            factor.role_path.semantic_sha256,
        )
        not in keys
    )
    if arrays[0].shape[0] + len(new_factors) > MAX_ROLE_ROWS:
        raise ValueError("repository structural-role row capacity exceeded")
    old_count = arrays[0].shape[0]
    if new_factors:
        arrays[0] = np.concatenate(
            (arrays[0], np.full(len(new_factors), policy_index, dtype=np.uint8))
        )
        arrays[1] = np.vstack(
            (
                arrays[1],
                np.asarray(
                    [
                        tuple(bytes.fromhex(factor.event_root.semantic_sha256))
                        for factor in new_factors
                    ],
                    dtype=np.uint8,
                ),
            )
        )
        arrays[2] = np.vstack(
            (
                arrays[2],
                np.asarray(
                    [
                        tuple(bytes.fromhex(factor.role_path.semantic_sha256))
                        for factor in new_factors
                    ],
                    dtype=np.uint8,
                ),
            )
        )
        arrays[3] = np.vstack(
            (
                arrays[3],
                np.asarray(
                    [tuple(bytes.fromhex(factor.sha256)) for factor in new_factors],
                    dtype=np.uint8,
                ),
            )
        )
        arrays[4] = np.vstack(
            (
                arrays[4],
                np.asarray([factor.cue_cells for factor in new_factors], dtype="<u4"),
            )
        )
        zeros = np.zeros((len(new_factors), ACTION_COUNT), dtype=np.uint8)
        arrays[5] = np.vstack((arrays[5], zeros))
        arrays[6] = np.vstack((arrays[6], zeros.copy()))
        arrays[7] = np.concatenate(
            (arrays[7], np.full(len(new_factors), -1, dtype=np.int8))
        )
        arrays[9] = np.pad(
            arrays[9], ((0, 0), (0, len(new_factors))), mode="constant"
        )
        keys.update(
            {
                (
                    policy_index,
                    factor.event_root.semantic_sha256,
                    factor.role_path.semantic_sha256,
                ): old_count + offset
                for offset, factor in enumerate(new_factors)
            }
        )
    factor_rows: list[int] = []
    for factor in factors:
        key = (
            policy_index,
            factor.event_root.semantic_sha256,
            factor.role_path.semantic_sha256,
        )
        row = keys[key]
        if (
            bytes(arrays[3][row]).hex() != factor.sha256
            or tuple(int(value) for value in arrays[4][row]) != factor.cue_cells
        ):
            raise ValueError("repository structural-role projection collision")
        factor_rows.append(row)

    expected_sources = lexical._source_roots.shape[0]
    if (
        arrays[8].shape[0] + 1 != expected_sources
        or arrays[9].shape[0] + 1 != expected_sources
    ):
        raise RuntimeError("role provenance lost lexical source alignment")
    if bytes(lexical._source_roots[-1]).hex() != experience.observation.source_sha256:
        raise RuntimeError("role provenance source root changed")
    arrays[8] = np.vstack((arrays[8], _root_row(experience.sha256)))
    presence = np.zeros((1, arrays[0].shape[0]), dtype=np.uint8)
    if factor_rows:
        presence[0, factor_rows] = 1
    arrays[9] = np.vstack((arrays[9], presence))

    action_roots = tuple(
        bytes(row).hex() for row in lexical._policy_action_roots[policy_index]
    )
    action_indices = {root: index for index, root in enumerate(action_roots)}
    for lesson in experience.action_lessons:
        action_index = action_indices[lesson.candidate_action_sha256]
        target = arrays[5] if lesson.passed_cell == 1 else arrays[6]
        for row in factor_rows:
            target[row, action_index] = min(
                TICK_CAP, int(target[row, action_index]) + EVIDENCE_TICKS
            )

    role_count = arrays[0].shape[0]
    learning_steps = 0
    admitted = np.full(role_count, -1, dtype=np.int8)
    if role_count:
        supports = np.sum(arrays[9], axis=0, dtype=np.int64)
        support_currents = np.full(
            (role_count, 2), CURRENT_BASE_PA, dtype=np.float64
        )
        support_currents[:, 0] += CURRENT_GAIN_PA * np.minimum(supports, 2)
        support_currents[:, SUPPORT_STOP_CELL] += CURRENT_GAIN_PA
        support_physicals = _SUPPORT_SELECTOR.select_many(
            support_currents, stop_cell=SUPPORT_STOP_CELL
        )
        supported_rows = tuple(
            row
            for row, physical in enumerate(support_physicals)
            if int(supports[row]) >= MIN_INDEPENDENT_SOURCE_ROOTS
            and physical.status == RETAINED
            and physical.retained_candidate_cells == (0,)
        )
        learning_steps += role_count * 2 * RACE_TICKS
        if supported_rows:
            action_currents = np.full(
                (len(supported_rows), ACTION_COUNT + 1),
                CURRENT_BASE_PA,
                dtype=np.float64,
            )
            for local, row in enumerate(supported_rows):
                for action in range(ACTION_COUNT):
                    drive = max(
                        0,
                        int(arrays[5][row, action])
                        - int(arrays[6][row, action]),
                    )
                    action_currents[local, action] += CURRENT_GAIN_PA * drive
                action_currents[local, ACTION_STOP_CELL] += CURRENT_GAIN_PA
            action_physicals = _ACTION_SELECTOR.select_many(
                action_currents, stop_cell=ACTION_STOP_CELL
            )
            learning_steps += len(supported_rows) * (ACTION_COUNT + 1) * RACE_TICKS
            for row, physical in zip(
                supported_rows, action_physicals, strict=True
            ):
                positives = tuple(
                    action
                    for action in range(ACTION_COUNT)
                    if int(arrays[5][row, action]) > 0
                )
                if (
                    len(positives) == 1
                    and physical.status == RETAINED
                    and physical.retained_candidate_cells == positives
                ):
                    admitted[row] = positives[0]
    arrays[7] = admitted
    return arrays, learning_steps


def _freeze_memory(
    lexical: lexical_memory.FrozenFactorizedRepositoryActionMemoryV1,
    arrays: list[np.ndarray],
    *,
    parent_checkpoint_sha256: str | None,
    mode: str,
    new_lesson_count: int,
    structural_role_learning_cell_steps: int,
) -> FrozenRepositoryActionMemoryV2:
    lexical_summary = lexical.training_summary
    trainable = int(
        lexical_summary.trainable_state_bytes
        + arrays[5].nbytes
        + arrays[6].nbytes
        + arrays[7].nbytes
    )
    checkpoint_bytes = lexical_summary.checkpoint_array_bytes + sum(
        array.nbytes for array in arrays
    )
    partial = RepositoryActionTrainingSummaryV2(
        mode=mode,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        lexical_memory_checkpoint_sha256=lexical.checkpoint_sha256,
        policy_count=lexical_summary.policy_count,
        lexical_factor_count=lexical_summary.factor_count,
        admitted_lexical_factor_count=lexical_summary.admitted_factor_count,
        structural_role_factor_count=arrays[0].shape[0],
        admitted_structural_role_factor_count=int(np.count_nonzero(arrays[7] >= 0)),
        source_count=lexical_summary.source_count,
        new_lesson_count=new_lesson_count,
        raw_experience_retention_count=0,
        parent_mutation_count=0,
        trainable_state_bytes=trainable,
        checkpoint_array_bytes=checkpoint_bytes,
        structural_role_learning_cell_steps=structural_role_learning_cell_steps,
        learning_cell_steps=(
            lexical_summary.learning_cell_steps + structural_role_learning_cell_steps
        ),
        transient_numpy_array_bound_bytes=MAX_TRANSIENT_NUMPY_ARRAY_BYTES,
        checkpoint_sha256="",
    )
    frozen_arrays = tuple(
        (name, _freeze_array(array))
        for name, array in zip(_ROLE_ARRAY_NAMES, arrays, strict=True)
    )
    checkpoint = _checkpoint_sha256(
        partial, lexical.checkpoint_sha256, frozen_arrays
    )
    return FrozenRepositoryActionMemoryV2(
        lexical=lexical,
        arrays=arrays,
        summary=partial._replace(checkpoint_sha256=checkpoint),
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_repository_action_memory_v2(
    experience: RepositoryActionExperienceV2,
    *,
    plasticity_gate: bool = True,
) -> FrozenRepositoryActionMemoryV2:
    if type(experience) is not RepositoryActionExperienceV2:
        raise TypeError("one exact V2 repository action experience is required")
    experience.authenticate()
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    if not plasticity_gate:
        raise ValueError("an empty V2 memory cannot start without plasticity")
    lexical_experience = lexical_memory.make_factorized_repository_action_experience_v1(
        experience.observation.source_views, experience.action_lessons
    )
    lexical = lexical_memory.start_factorized_repository_action_memory_v1(
        lexical_experience, plasticity_gate=True
    )
    arrays, role_steps = _apply_structural_role_experience(
        None, experience, lexical
    )
    return _freeze_memory(
        lexical,
        arrays,
        parent_checkpoint_sha256=None,
        mode="INITIAL_REPOSITORY_ACTION_MEMORY_V2",
        new_lesson_count=len(experience.action_lessons),
        structural_role_learning_cell_steps=role_steps,
    )


def advance_repository_action_memory_v2(
    parent: FrozenRepositoryActionMemoryV2,
    experience: RepositoryActionExperienceV2,
    *,
    plasticity_gate: bool = True,
) -> FrozenRepositoryActionMemoryV2:
    if type(parent) is not FrozenRepositoryActionMemoryV2:
        raise TypeError("parent must be one exact frozen V2 memory")
    if type(experience) is not RepositoryActionExperienceV2:
        raise TypeError("one exact V2 repository action experience is required")
    experience.authenticate()
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity gate must be one exact boolean")
    parent.authenticate()
    _ensure_not_duplicate(parent, experience)
    if not plasticity_gate:
        parent.authenticate()
        return parent
    parent_root = parent.checkpoint_sha256
    parent_child_root = parent._lexical_memory.checkpoint_sha256
    parent_array_roots = tuple(_array_sha256(array) for _, array in parent._arrays())
    lexical_experience = lexical_memory.make_factorized_repository_action_experience_v1(
        experience.observation.source_views, experience.action_lessons
    )
    lexical = lexical_memory.advance_factorized_repository_action_memory_v1(
        parent._lexical_memory, lexical_experience, plasticity_gate=True
    )
    arrays, role_steps = _apply_structural_role_experience(
        parent, experience, lexical
    )
    successor = _freeze_memory(
        lexical,
        arrays,
        parent_checkpoint_sha256=parent_root,
        mode="REPLAY_FREE_REPOSITORY_ACTION_MEMORY_V2_ADVANCE",
        new_lesson_count=len(experience.action_lessons),
        structural_role_learning_cell_steps=(
            parent.training_summary.structural_role_learning_cell_steps + role_steps
        ),
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent._lexical_memory.checkpoint_sha256 != parent_child_root
        or tuple(_array_sha256(array) for _, array in parent._arrays())
        != parent_array_roots
    ):
        raise RuntimeError("V2 repository action advance mutated its parent")
    return successor


__all__ = [
    "ACTION_COUNT",
    "EVIDENCE_TICKS",
    "EXACT_EPISODIC",
    "LEXICAL_FACTOR",
    "MAX_CHECKPOINT_ARRAY_BYTES",
    "MAX_RECALL_CELL_STEPS",
    "MAX_RECALL_ROOT_READS",
    "MAX_ROLE_ROWS",
    "MAX_TRAINABLE_STATE_BYTES",
    "MAX_TRANSIENT_NUMPY_ARRAY_BYTES",
    "MIN_INDEPENDENT_SOURCE_ROOTS",
    "RECALLED",
    "SCHEMA",
    "STOP_ACTION_BANK",
    "STOP_EXACT_CANDIDATE_AMBIGUOUS",
    "STOP_EXACT_CANDIDATE_GATE",
    "STOP_EXACT_CONTEXT_AMBIGUOUS",
    "STOP_EXACT_CONTEXT_COLLISION",
    "STOP_EXACT_CONTEXT_GATE",
    "STOP_LEXICAL_FACTOR_AMBIGUOUS",
    "STOP_LEXICAL_FACTOR_COLLISION",
    "STOP_LEXICAL_FACTOR_CONFLICT",
    "STOP_LEXICAL_FACTOR_GATE",
    "STOP_LEXICAL_FACTOR_UNKNOWN",
    "STOP_RECALL_GATE",
    "STOP_STRUCTURAL_ROLE_AMBIGUOUS",
    "STOP_STRUCTURAL_ROLE_COLLISION",
    "STOP_STRUCTURAL_ROLE_CONFLICT",
    "STOP_STRUCTURAL_ROLE_GATE",
    "STOP_STRUCTURAL_ROLE_UNKNOWN",
    "STRUCTURAL_ROLE",
    "FrozenRepositoryActionMemoryV2",
    "RepositoryActionExperienceV2",
    "RepositoryActionObservationV2",
    "RepositoryActionRecallV2",
    "RepositoryActionRecallWorkV2",
    "RepositoryActionSourceProvenanceV2",
    "RepositoryActionTrainingSummaryV2",
    "RepositoryLexicalAdmissionV2",
    "RepositoryStructuralRoleAdmissionV2",
    "advance_repository_action_memory_v2",
    "make_repository_action_experience_v2",
    "make_repository_action_observation_v2",
    "start_repository_action_memory_v2",
]
