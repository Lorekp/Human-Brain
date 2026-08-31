"""Dual-view episodic retention and reusable one-byte transform consolidation.

This self-posed R2 slice composes two already banked cell mechanisms.  Exact
source-conditioned episodes live in ``FrozenCommonEventAssociativeMemory``;
lane-local semantic transforms live in ``FrozenCommonOneByteCopyRelationV2``.
The composite never decodes an authenticated filler frame, inspects a digit,
or selects between child outputs.  Each recall delegates to exactly one frozen
cell checkpoint.

Semantic extension is replay-free.  The exact associative mechanism has no
continual-extension API, so an extension explicitly refits that view from a
caller-supplied cumulative episode tuple.  This experiment therefore claims
retention, abstraction, conflict, and revision, not replay-free episodic
continual learning.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms

SCOPE = "COMMON_EPISODIC_TRANSFORM_CONSOLIDATION_V1"
STATUS = "SELF_POSED_DUAL_VIEW_EPISODIC_TRANSFORM_CONSOLIDATION"
CLAIM_BOUNDARY = (
    "self-posed-one-byte-exact-episodes-plus-lane-local-transform-"
    "consolidation-conflict-and-revision-only-no-language-coding-corpus-"
    "protected-transfer-or-replay-free-episodic-extension-evidence"
)

STRUCTURAL_ROUTE_CHOICE_COUNT = transforms.INPUT_FILLER_CELLS
WHOLE_BYTE_MAPPING_COUNT = transforms.RADIX**transforms.LANES
FULL_SEMANTIC_SYNAPTIC_STATE_BYTES = 1_120
FULL_SEMANTIC_CHECKPOINT_ARRAY_BYTES = 8_744
FULL_EPISODIC_CHECKPOINT_ARRAY_BYTES = associative.CHECKPOINT_ARRAY_BYTES

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_SOURCE_PATH = Path(__file__).resolve()
_CONFIG_NAMESPACE = b"common-episodic-transform-consolidation-v1-config\0"
_CHECKPOINT_NAMESPACE = b"common-episodic-transform-consolidation-v1-checkpoint\0"
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


def _validate_episode_lessons(
    lessons: tuple[associative.CommonAssociationLesson, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not associative.CommonAssociationLesson
        for lesson in lessons
    ):
        raise TypeError(
            "episodic lessons must be an exact tuple of CommonAssociationLesson values"
        )


def _validate_semantic_lessons(
    lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not transforms.RelationGatedOneByteLesson
        for lesson in lessons
    ):
        raise TypeError(
            "semantic lessons must be an exact tuple of RelationGatedOneByteLesson values"
        )


@dataclass(frozen=True, slots=True)
class EpisodicTransformTrainingSummary:
    mode: str
    parent_composite_checkpoint_sha256: str | None
    episodic_lesson_count: int
    cumulative_episodic_refit_lesson_count: int
    semantic_new_lesson_count: int
    semantic_parent_lesson_replay_count: int
    independent_context_count: int
    relation_count: int
    provenance_count: int
    structural_route_choice_count: int
    whole_byte_mapping_count: int
    semantic_synaptic_state_bytes: int
    semantic_checkpoint_array_bytes: int
    episodic_checkpoint_array_bytes: int
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    semantic_extension_status: str | None
    checkpoint_sha256: str


class FrozenCommonEpisodicTransformMemoryV1:
    """Immutable exact-episode and semantic-transform checkpoints."""

    __slots__ = (
        "_checkpoint_sha256",
        "_episodic",
        "_semantic",
        "_summary",
    )

    def __init__(
        self,
        *,
        episodic: associative.FrozenCommonEventAssociativeMemory,
        semantic: transforms.FrozenCommonOneByteCopyRelationV2,
        summary: EpisodicTransformTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("episodic-transform memories are builder-owned")
        if type(episodic) is not associative.FrozenCommonEventAssociativeMemory:
            raise TypeError("episodic checkpoint type changed")
        if type(semantic) is not transforms.FrozenCommonOneByteCopyRelationV2:
            raise TypeError("semantic checkpoint type changed")
        self._episodic = episodic
        self._semantic = semantic
        self._summary = summary
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256(summary)
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("composite episodic-transform checkpoint identity changed")
        self.authenticate()

    def _calculate_checkpoint_sha256(
        self,
        summary: EpisodicTransformTrainingSummary,
    ) -> str:
        record = {
            **dataclasses.asdict(summary),
            "checkpoint_sha256": None,
            "episodic_checkpoint_sha256": self._episodic.checkpoint_sha256,
            "semantic_checkpoint_sha256": self._semantic.checkpoint_sha256,
        }
        return _digest(_CHECKPOINT_NAMESPACE, record)

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> EpisodicTransformTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def episodic_training_summary(self) -> associative.AssociationTrainingSummary:
        self.authenticate()
        return self._episodic.training_summary

    @property
    def semantic_training_summary(self) -> transforms.RelationGatedTrainingSummary:
        self.authenticate()
        return self._semantic.training_summary

    @property
    def semantic_extension_summary(
        self,
    ) -> transforms.RelationGatedExtensionSummary | None:
        self.authenticate()
        return self._semantic.extension_summary

    def authenticate(self) -> None:
        self._episodic.authenticate()
        self._semantic.authenticate()
        if self._episodic.checkpoint_sha256 != self._summary.episodic_checkpoint_sha256:
            raise RuntimeError("composite episodic checkpoint authentication failed")
        if self._semantic.checkpoint_sha256 != self._summary.semantic_checkpoint_sha256:
            raise RuntimeError("composite semantic checkpoint authentication failed")
        if self._calculate_checkpoint_sha256(self._summary) != self._checkpoint_sha256:
            raise RuntimeError("composite episodic-transform authentication failed")

    def recall_episode(
        self,
        observation_events: tuple[bytes, ...],
    ) -> associative.CommonAssociationRecall:
        """Delegate one exact cue directly to the frozen episodic checkpoint."""

        self.authenticate()
        return self._episodic.recall(observation_events)

    def recall_semantic(
        self,
        input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        cell_gate: bool = True,
        relation_gate: bool = True,
    ) -> transforms.RelationGatedCopyRecall:
        """Delegate one physical frame directly to the frozen relation checkpoint."""

        self.authenticate()
        return self._semantic.recall(
            input_frame,
            relation_cells,
            cell_gate=cell_gate,
            relation_gate=relation_gate,
        )

    def relation_state_sha256(self, relation_cells: tuple[int, ...]) -> str | None:
        self.authenticate()
        return self._semantic.relation_state_sha256(relation_cells)


def _freeze(
    *,
    episodic: associative.FrozenCommonEventAssociativeMemory,
    semantic: transforms.FrozenCommonOneByteCopyRelationV2,
    mode: str,
    parent_checkpoint_sha256: str | None,
    episodic_lesson_count: int,
    semantic_new_lesson_count: int,
) -> FrozenCommonEpisodicTransformMemoryV1:
    episodic_summary = episodic.training_summary
    semantic_summary = semantic.training_summary
    extension = semantic.extension_summary
    summary_record = {
        "mode": mode,
        "parent_composite_checkpoint_sha256": parent_checkpoint_sha256,
        "episodic_lesson_count": episodic_lesson_count,
        "cumulative_episodic_refit_lesson_count": episodic_lesson_count,
        "semantic_new_lesson_count": semantic_new_lesson_count,
        "semantic_parent_lesson_replay_count": 0,
        "independent_context_count": semantic_summary.independent_context_count,
        "relation_count": semantic_summary.relation_count,
        "provenance_count": len(semantic_summary.provenance),
        "structural_route_choice_count": STRUCTURAL_ROUTE_CHOICE_COUNT,
        "whole_byte_mapping_count": WHOLE_BYTE_MAPPING_COUNT,
        "semantic_synaptic_state_bytes": semantic_summary.synaptic_state_bytes,
        "semantic_checkpoint_array_bytes": semantic_summary.checkpoint_array_bytes,
        "episodic_checkpoint_array_bytes": episodic_summary.checkpoint_array_bytes,
        "episodic_checkpoint_sha256": episodic.checkpoint_sha256,
        "semantic_checkpoint_sha256": semantic.checkpoint_sha256,
        "semantic_extension_status": None if extension is None else extension.status,
    }
    checkpoint_sha256 = _digest(
        _CHECKPOINT_NAMESPACE,
        {**summary_record, "checkpoint_sha256": None},
    )
    summary = EpisodicTransformTrainingSummary(
        **summary_record,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonEpisodicTransformMemoryV1(
        episodic=episodic,
        semantic=semantic,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_common_episodic_transform_consolidation_v1(
    episodic_lessons: tuple[associative.CommonAssociationLesson, ...],
    semantic_lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> FrozenCommonEpisodicTransformMemoryV1:
    """Fit both physical views once without interpreting either representation."""

    _validate_episode_lessons(episodic_lessons)
    _validate_semantic_lessons(semantic_lessons)
    episodic = associative.fit_common_event_associations(episodic_lessons)
    semantic = transforms.fit_common_one_byte_copy_relation_v2(semantic_lessons)
    return _freeze(
        episodic=episodic,
        semantic=semantic,
        mode="INITIAL_FIT",
        parent_checkpoint_sha256=None,
        episodic_lesson_count=len(episodic_lessons),
        semantic_new_lesson_count=len(semantic_lessons),
    )


def extend_common_episodic_transform_consolidation_v1(
    parent: FrozenCommonEpisodicTransformMemoryV1,
    cumulative_episodic_lessons: tuple[associative.CommonAssociationLesson, ...],
    new_semantic_lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> FrozenCommonEpisodicTransformMemoryV1:
    """Refit cumulative episodes while extending semantic routes without replay."""

    if type(parent) is not FrozenCommonEpisodicTransformMemoryV1:
        raise TypeError("parent must be an exact frozen composite checkpoint")
    _validate_episode_lessons(cumulative_episodic_lessons)
    _validate_semantic_lessons(new_semantic_lessons)
    parent.authenticate()
    parent_checkpoint_sha256 = parent.checkpoint_sha256
    episodic = associative.fit_common_event_associations(cumulative_episodic_lessons)
    semantic = transforms.extend_common_one_byte_copy_relation_v2(
        parent._semantic,
        new_semantic_lessons,
    )
    successor = _freeze(
        episodic=episodic,
        semantic=semantic,
        mode="SEMANTIC_EXTENSION_WITH_CUMULATIVE_EPISODIC_REFIT",
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        episodic_lesson_count=len(cumulative_episodic_lessons),
        semantic_new_lesson_count=len(new_semantic_lessons),
    )
    parent.authenticate()
    if parent.checkpoint_sha256 != parent_checkpoint_sha256:
        raise RuntimeError("parent composite changed during extension")
    return successor


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "episodic_geometry_sha256": associative.GEOMETRY_SHA256,
    "semantic_config_sha256": transforms.CONFIG_SHA256,
    "structural_route_choice_count": STRUCTURAL_ROUTE_CHOICE_COUNT,
    "whole_byte_mapping_count": WHOLE_BYTE_MAPPING_COUNT,
    "full_semantic_synaptic_state_bytes": FULL_SEMANTIC_SYNAPTIC_STATE_BYTES,
    "full_semantic_checkpoint_array_bytes": FULL_SEMANTIC_CHECKPOINT_ARRAY_BYTES,
    "full_episodic_checkpoint_array_bytes": FULL_EPISODIC_CHECKPOINT_ARRAY_BYTES,
    "semantic_parent_lesson_replay_count": 0,
    "host_output_selection_count": 0,
    "raw_byte_or_digit_decode_after_projection_count": 0,
    "schema_per_context_growth": False,
    "generator_corpus_path2_dev_m0_protected_g0_coding_inputs": (
        GENERATOR_EXECUTION_COUNT,
        CORPUS_INPUT_COUNT,
        PATH2_INPUT_COUNT,
        DEV_INPUT_COUNT,
        M0_INPUT_COUNT,
        PROTECTED_INPUT_COUNT,
        G0_INPUT_COUNT,
        CODING_INPUT_COUNT,
    ),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()
