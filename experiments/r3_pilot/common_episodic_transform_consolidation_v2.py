"""Replay-free dual-view episodic and one-byte transform consolidation.

The exact view is one authenticated V1 common associative base plus the flat
append-only delta matrices of ``FrozenCommonEventAssociativeMemoryV2``.  The
semantic view is the independently frozen lane-local relation transform of
``FrozenCommonOneByteCopyRelationV2``.  Advancing the composite accepts only
new physical lessons: it neither retains a composite parent nor receives or
replays an old lesson tuple.

Recall delegates to exactly one child checkpoint.  This module never decodes a
filler frame, compares child answers, or selects an output on the host.  Exact
delta storage and semantic synaptic/checkpoint storage are reported separately;
no combined compression claim is made.
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
from experiments.r3_pilot import common_event_associative_memory_v2 as exact_memory
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms

SCOPE = "COMMON_EPISODIC_TRANSFORM_CONSOLIDATION_V2"
STATUS = "SELF_POSED_REPLAY_FREE_DUAL_VIEW_CONSOLIDATION"
CLAIM_BOUNDARY = (
    "self-posed-one-byte-source-conditioned-exact-delta-retention-plus-"
    "lane-local-transform-consolidation-conflict-and-revision-only-no-"
    "language-coding-corpus-protected-transfer-or-broad-compression-evidence"
)

EXACT_DELTA_ARRAY_BYTES_PER_ROW = exact_memory.DELTA_ARRAY_BYTES_PER_EPISODE
STRUCTURAL_ROUTE_CHOICE_COUNT = transforms.INPUT_FILLER_CELLS
WHOLE_BYTE_MAPPING_COUNT = transforms.RADIX**transforms.LANES

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
_CONFIG_NAMESPACE = b"common-episodic-transform-consolidation-v2-config\0"
_CHECKPOINT_NAMESPACE = b"common-episodic-transform-consolidation-v2-checkpoint\0"
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


def _validate_initial_exact_lessons(
    lessons: tuple[associative.CommonAssociationLesson, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not associative.CommonAssociationLesson
        for lesson in lessons
    ):
        raise TypeError(
            "initial exact lessons must be an exact tuple of "
            "CommonAssociationLesson values"
        )


def _validate_new_exact_lessons(
    lessons: tuple[exact_memory.PhysicalCommonAssociationLessonV2, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not exact_memory.PhysicalCommonAssociationLessonV2
        for lesson in lessons
    ):
        raise TypeError(
            "new exact lessons must be an exact tuple of "
            "PhysicalCommonAssociationLessonV2 values"
        )


def _validate_semantic_lessons(
    lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not transforms.RelationGatedOneByteLesson
        for lesson in lessons
    ):
        raise TypeError(
            "semantic lessons must be an exact tuple of "
            "RelationGatedOneByteLesson values"
        )


@dataclass(frozen=True, slots=True)
class EpisodicTransformTrainingSummaryV2:
    mode: str
    parent_composite_checkpoint_sha256: str | None
    initial_exact_v1_lesson_count: int
    exact_new_lesson_count: int
    exact_accepted_delta_row_count: int
    exact_rejected_lesson_count: int
    exact_parent_lesson_replay_count: int
    exact_delta_episode_count: int
    exact_delta_array_bytes_per_row: int
    exact_delta_array_bytes_added: int
    exact_delta_checkpoint_array_bytes: int
    inherited_exact_v1_checkpoint_array_bytes: int
    semantic_new_lesson_count: int
    semantic_parent_lesson_replay_count: int
    semantic_accepted_new_context_count: int
    semantic_committed_update_ticks: int
    independent_context_count: int
    relation_count: int
    provenance_count: int
    structural_route_choice_count: int
    whole_byte_mapping_count: int
    semantic_synaptic_state_bytes: int
    semantic_checkpoint_array_bytes: int
    episodic_base_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    semantic_extension_status: str | None
    checkpoint_sha256: str


class FrozenCommonEpisodicTransformMemoryV2:
    """One immutable exact child and one immutable semantic child."""

    __slots__ = (
        "_checkpoint_sha256",
        "_episodic",
        "_semantic",
        "_summary",
    )

    def __init__(
        self,
        *,
        episodic: exact_memory.FrozenCommonEventAssociativeMemoryV2,
        semantic: transforms.FrozenCommonOneByteCopyRelationV2,
        summary: EpisodicTransformTrainingSummaryV2,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("V2 episodic-transform memories are builder-owned")
        if type(episodic) is not exact_memory.FrozenCommonEventAssociativeMemoryV2:
            raise TypeError("exact episodic checkpoint type changed")
        if type(semantic) is not transforms.FrozenCommonOneByteCopyRelationV2:
            raise TypeError("semantic checkpoint type changed")
        if type(summary) is not EpisodicTransformTrainingSummaryV2:
            raise TypeError("composite summary type changed")
        self._episodic = episodic
        self._semantic = semantic
        self._summary = summary
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256(summary)
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("V2 composite checkpoint identity changed")
        self.authenticate()

    def _calculate_checkpoint_sha256(
        self,
        summary: EpisodicTransformTrainingSummaryV2,
    ) -> str:
        record = {
            **dataclasses.asdict(summary),
            "checkpoint_sha256": None,
            "episodic_base_checkpoint_sha256": (
                self._episodic.base_checkpoint_sha256
            ),
            "episodic_checkpoint_sha256": self._episodic.checkpoint_sha256,
            "semantic_checkpoint_sha256": self._semantic.checkpoint_sha256,
        }
        return _digest(_CHECKPOINT_NAMESPACE, record)

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def episodic_base_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._episodic.base_checkpoint_sha256

    @property
    def episodic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._episodic.checkpoint_sha256

    @property
    def semantic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._semantic.checkpoint_sha256

    @property
    def training_summary(self) -> EpisodicTransformTrainingSummaryV2:
        self.authenticate()
        return self._summary

    @property
    def episodic_training_summary(
        self,
    ) -> exact_memory.CommonAssociationTrainingSummaryV2:
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
        exact_summary = self._episodic.training_summary
        semantic_summary = self._semantic.training_summary
        if (
            self._episodic.base_checkpoint_sha256
            != self._summary.episodic_base_checkpoint_sha256
            or self._episodic.checkpoint_sha256
            != self._summary.episodic_checkpoint_sha256
            or self._semantic.checkpoint_sha256
            != self._summary.semantic_checkpoint_sha256
            or exact_summary.delta_episode_count
            != self._summary.exact_delta_episode_count
            or exact_summary.delta_checkpoint_array_bytes
            != self._summary.exact_delta_checkpoint_array_bytes
            or exact_summary.inherited_v1_checkpoint_array_bytes
            != self._summary.inherited_exact_v1_checkpoint_array_bytes
            or semantic_summary.synaptic_state_bytes
            != self._summary.semantic_synaptic_state_bytes
            or semantic_summary.checkpoint_array_bytes
            != self._summary.semantic_checkpoint_array_bytes
            or self._calculate_checkpoint_sha256(self._summary)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("V2 episodic-transform authentication failed")

    def recall_episode(
        self,
        observation_events: tuple[bytes, ...],
        *,
        cell_gate: bool = True,
    ) -> exact_memory.CommonAssociationRecallV2:
        """Delegate one exact cue directly to the exact-memory child."""

        self.authenticate()
        return self._episodic.recall(observation_events, cell_gate=cell_gate)

    def recall_semantic(
        self,
        input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        cell_gate: bool = True,
        relation_gate: bool = True,
    ) -> transforms.RelationGatedCopyRecall:
        """Delegate one filler frame directly to the relation-transform child."""

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


def _semantic_committed_ticks(
    summary: transforms.RelationGatedTrainingSummary,
) -> int:
    return (
        summary.local_physical_update_ticks
        + summary.relation_excitatory_update_ticks
        + summary.route_excitatory_update_ticks
        + summary.route_inhibitory_update_ticks
    )


def _freeze(
    *,
    episodic: exact_memory.FrozenCommonEventAssociativeMemoryV2,
    semantic: transforms.FrozenCommonOneByteCopyRelationV2,
    mode: str,
    parent_checkpoint_sha256: str | None,
    initial_exact_v1_lesson_count: int,
    exact_new_lesson_count: int,
    exact_accepted_delta_row_count: int,
    exact_delta_array_bytes_added: int,
    semantic_new_lesson_count: int,
) -> FrozenCommonEpisodicTransformMemoryV2:
    exact_summary = episodic.training_summary
    semantic_summary = semantic.training_summary
    extension = semantic.extension_summary
    if exact_delta_array_bytes_added != (
        exact_accepted_delta_row_count * EXACT_DELTA_ARRAY_BYTES_PER_ROW
    ):
        raise RuntimeError("exact delta byte accounting changed")
    if exact_summary.retained_v2_parent_count != 0:
        raise RuntimeError("exact child retained a V2 parent")
    summary_record = {
        "mode": mode,
        "parent_composite_checkpoint_sha256": parent_checkpoint_sha256,
        "initial_exact_v1_lesson_count": initial_exact_v1_lesson_count,
        "exact_new_lesson_count": exact_new_lesson_count,
        "exact_accepted_delta_row_count": exact_accepted_delta_row_count,
        "exact_rejected_lesson_count": (
            exact_new_lesson_count - exact_accepted_delta_row_count
        ),
        "exact_parent_lesson_replay_count": 0,
        "exact_delta_episode_count": exact_summary.delta_episode_count,
        "exact_delta_array_bytes_per_row": EXACT_DELTA_ARRAY_BYTES_PER_ROW,
        "exact_delta_array_bytes_added": exact_delta_array_bytes_added,
        "exact_delta_checkpoint_array_bytes": (
            exact_summary.delta_checkpoint_array_bytes
        ),
        "inherited_exact_v1_checkpoint_array_bytes": (
            exact_summary.inherited_v1_checkpoint_array_bytes
        ),
        "semantic_new_lesson_count": semantic_new_lesson_count,
        "semantic_parent_lesson_replay_count": 0,
        "semantic_accepted_new_context_count": (
            semantic_summary.independent_context_count
            if mode == "INITIAL_FIT"
            else (
                0
                if semantic_new_lesson_count == 0 or extension is None
                else extension.accepted_new_context_count
            )
        ),
        "semantic_committed_update_ticks": (
            _semantic_committed_ticks(semantic_summary)
            if mode == "INITIAL_FIT"
            else (
                0
                if semantic_new_lesson_count == 0 or extension is None
                else extension.committed_update_ticks
            )
        ),
        "independent_context_count": semantic_summary.independent_context_count,
        "relation_count": semantic_summary.relation_count,
        "provenance_count": len(semantic_summary.provenance),
        "structural_route_choice_count": STRUCTURAL_ROUTE_CHOICE_COUNT,
        "whole_byte_mapping_count": WHOLE_BYTE_MAPPING_COUNT,
        "semantic_synaptic_state_bytes": semantic_summary.synaptic_state_bytes,
        "semantic_checkpoint_array_bytes": semantic_summary.checkpoint_array_bytes,
        "episodic_base_checkpoint_sha256": episodic.base_checkpoint_sha256,
        "episodic_checkpoint_sha256": episodic.checkpoint_sha256,
        "semantic_checkpoint_sha256": semantic.checkpoint_sha256,
        "semantic_extension_status": (
            None
            if semantic_new_lesson_count == 0 or extension is None
            else extension.status
        ),
    }
    checkpoint_sha256 = _digest(
        _CHECKPOINT_NAMESPACE,
        {**summary_record, "checkpoint_sha256": None},
    )
    summary = EpisodicTransformTrainingSummaryV2(
        **summary_record,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonEpisodicTransformMemoryV2(
        episodic=episodic,
        semantic=semantic,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_common_episodic_transform_consolidation_v2(
    initial_exact_lessons: tuple[associative.CommonAssociationLesson, ...],
    initial_semantic_lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> FrozenCommonEpisodicTransformMemoryV2:
    """Fit one exact legacy base, upgrade it, and fit the semantic view."""

    _validate_initial_exact_lessons(initial_exact_lessons)
    _validate_semantic_lessons(initial_semantic_lessons)
    base = associative.fit_common_event_associations(initial_exact_lessons)
    episodic = exact_memory.upgrade_common_event_associative_memory_v2(base)
    semantic = transforms.fit_common_one_byte_copy_relation_v2(
        initial_semantic_lessons
    )
    return _freeze(
        episodic=episodic,
        semantic=semantic,
        mode="INITIAL_FIT",
        parent_checkpoint_sha256=None,
        initial_exact_v1_lesson_count=len(initial_exact_lessons),
        exact_new_lesson_count=0,
        exact_accepted_delta_row_count=0,
        exact_delta_array_bytes_added=0,
        semantic_new_lesson_count=len(initial_semantic_lessons),
    )


def advance_common_episodic_transform_consolidation_v2(
    parent: FrozenCommonEpisodicTransformMemoryV2,
    new_exact_lessons: tuple[
        exact_memory.PhysicalCommonAssociationLessonV2,
        ...,
    ],
    new_semantic_lessons: tuple[transforms.RelationGatedOneByteLesson, ...],
) -> FrozenCommonEpisodicTransformMemoryV2:
    """Advance both views from new physical lessons only, with zero replay."""

    if type(parent) is not FrozenCommonEpisodicTransformMemoryV2:
        raise TypeError("parent must be an exact frozen V2 composite")
    _validate_new_exact_lessons(new_exact_lessons)
    _validate_semantic_lessons(new_semantic_lessons)
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    parent_exact_root = parent.episodic_checkpoint_sha256
    parent_semantic_root = parent.semantic_checkpoint_sha256

    episodic = parent._episodic
    accepted_rows = 0
    exact_bytes_added = 0
    for lesson in new_exact_lessons:
        transition = exact_memory.advance_common_event_association_v2(
            episodic,
            lesson,
        )
        if transition.parent_lesson_replay_count != 0:
            raise RuntimeError("exact child replayed a parent lesson")
        if transition.accepted:
            row_bytes = (
                transition.cue_addresses_added * 4
                + transition.output_addresses_added * 4
                + transition.provenance_bytes_added
            )
            if (
                row_bytes != EXACT_DELTA_ARRAY_BYTES_PER_ROW
                or transition.match_cells_added != 1
            ):
                raise RuntimeError("exact delta row geometry changed")
            accepted_rows += 1
            exact_bytes_added += row_bytes
        episodic = transition.successor

    if new_semantic_lessons:
        semantic = transforms.extend_common_one_byte_copy_relation_v2(
            parent._semantic,
            new_semantic_lessons,
        )
        extension = semantic.extension_summary
        if (
            extension is None
            or extension.parent_checkpoint_sha256 != parent_semantic_root
        ):
            raise RuntimeError("semantic extension parent authority changed")
    else:
        semantic = parent._semantic

    successor = _freeze(
        episodic=episodic,
        semantic=semantic,
        mode="REPLAY_FREE_NEW_LESSON_ADVANCE",
        parent_checkpoint_sha256=parent_root,
        initial_exact_v1_lesson_count=(
            parent.training_summary.initial_exact_v1_lesson_count
        ),
        exact_new_lesson_count=len(new_exact_lessons),
        exact_accepted_delta_row_count=accepted_rows,
        exact_delta_array_bytes_added=exact_bytes_added,
        semantic_new_lesson_count=len(new_semantic_lessons),
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent.episodic_checkpoint_sha256 != parent_exact_root
        or parent.semantic_checkpoint_sha256 != parent_semantic_root
    ):
        raise RuntimeError("parent composite changed during replay-free advance")
    return successor


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "exact_memory_config_sha256": exact_memory.CONFIG_SHA256,
    "semantic_config_sha256": transforms.CONFIG_SHA256,
    "initial_legacy_base_count": 1,
    "exact_delta_array_bytes_per_row": EXACT_DELTA_ARRAY_BYTES_PER_ROW,
    "structural_route_choice_count": STRUCTURAL_ROUTE_CHOICE_COUNT,
    "whole_byte_mapping_count": WHOLE_BYTE_MAPPING_COUNT,
    "retained_v2_parent_count": 0,
    "composite_parent_object_count": 0,
    "exact_old_lesson_replay_count": 0,
    "semantic_old_lesson_replay_count": 0,
    "host_output_selection_count": 0,
    "semantic_decode_count": 0,
    "raw_byte_or_digit_decode_after_projection_count": 0,
    "schema_per_context_growth": False,
    "combined_compression_claim": False,
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
