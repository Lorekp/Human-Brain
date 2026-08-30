"""Frozen local-span V3 prompt-factor pre-gate.

This separately named successor does not reuse the V2 learned checkpoint.  It
keeps the banked three-bank anatomy, 24 independent fitted-cell heads, race
constants, four-epoch local rule, and uint16 E/H state, but fits a new
self-posed curriculum whose teacher labels point at local prefix spans.

The frozen curriculum fails its atomic readiness gate.  Consequently the
multi-span composition surface is deliberately sealed: no composition prompt
is inspected and ``compose`` fails closed before creating an attempt.  The
negative result is still useful because it pins the exact generative
curriculum and rules out silently resolving multi-head outputs in host code.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as v2,
)


BRIDGE_SCOPE = "SELF_POSED_PROMPT_FACTOR_MULTISPAN_EVENT_BRIDGE_V3"
STOP_STATUS = "STOP_SELF_POSED_MULTISPAN_EVENT_BRIDGE_V3_ATOMIC_READINESS"
CLAIM_BOUNDARY = (
    "Self-posed nine-factor local-span atomic-readiness STOP only; zero "
    "composition attempts and no public, canonical, protected, desired-output, "
    "prompt-only coding, family-transfer, DEV/eval, M0, Path-2, C1, SWE-bench, "
    "Gemma, or Opus evidence."
)

FIT = "FIT"
HELDOUT = "HELDOUT"
GROUP_COUNT = 18
LESSON_COUNT = 54
FIT_COUNT = 36
HELDOUT_COUNT = 18
TAUGHT_FACTOR_COUNT = 9
SILENT_HEAD_COUNT = 15
EXPECTED_ATOMIC_PASS_COUNT = 40
EXPECTED_ATOMIC_MISMATCH_COUNT = 14
EXPECTED_EXCITATORY_HEAD_UPDATE_EVENTS = 31
EXPECTED_INHIBITORY_HEAD_UPDATE_EVENTS = 3
EXPECTED_COMMITTED_CELL_TICKS = 3_266
EXPECTED_EXCITATORY_MAX_TICKS = 4
EXPECTED_INHIBITORY_MAX_TICKS = 1

_SUFFIX_START_TOKEN = 7
_TRIADS = (
    ("add", "map", "sum"),
    ("less-than", "count", "absolute"),
    ("minimum", "maximum", "subtract"),
)
_TRIAD_ROWS = (
    (FIT, "rather", False),
    (FIT, "instead", False),
    (FIT, "never", False),
    (FIT, "rather", True),
    (HELDOUT, "instead", True),
    (HELDOUT, "never", True),
)
_TAUGHT_FACTORS = tuple(factor for triad in _TRIADS for factor in triad)
_SILENT_HEADS = tuple(
    index for index, factor in enumerate(v2.FACTOR_REGISTRY)
    if factor not in _TAUGHT_FACTORS
)
_FACTOR_INDEX = MappingProxyType(
    {factor: index for index, factor in enumerate(v2.FACTOR_REGISTRY)}
)

_INPUT_NAMESPACE = b"prompt-factor-multispan-event-v3-inputs\0"
_CURRICULUM_NAMESPACE = b"prompt-factor-multispan-event-v3-curriculum\0"
_SPLIT_NAMESPACE = b"prompt-factor-multispan-event-v3-split\0"
_EXTRACTION_NAMESPACE = b"prompt-factor-multispan-event-v3-extraction\0"
_LESSON_NAMESPACE = b"prompt-factor-multispan-event-v3-lesson\0"
_CHECKPOINT_NAMESPACE = b"prompt-factor-multispan-event-v3-checkpoint\0"
_RESULT_NAMESPACE = b"prompt-factor-multispan-event-v3-result\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


EXTRACTION_SHA256 = _namespaced_sha256(
    _EXTRACTION_NAMESPACE,
    {
        "triads": _TRIADS,
        "rows": _TRIAD_ROWS,
        "semantic_surface_width": 4,
        "teacher_semantic_prefix_width": 3,
        "less_than_raw_surface": ("less", "than"),
        "ordinary_teacher_raw_width": 3,
        "less_than_teacher_raw_width": 4,
        "training_epochs": v2.TRAINING_EPOCHS,
        "winner_margin_tie_break": False,
    },
)


@dataclass(frozen=True, slots=True)
class LocalSpanLesson:
    """One authenticated four-semantic-token contrast with one local teacher."""

    group_index: int
    row_index: int
    semantic_tokens: tuple[str, str, str, str]
    raw_source_tokens: tuple[str, ...]
    teacher_span: v2.CompositionalPromptInput
    present_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.group_index) is not int or not 0 <= self.group_index < GROUP_COUNT:
            raise ValueError("group_index lies outside the frozen curriculum")
        if type(self.row_index) is not int or not 0 <= self.row_index < len(_TRIAD_ROWS):
            raise ValueError("row_index lies outside the frozen row schedule")
        if (
            type(self.semantic_tokens) is not tuple
            or len(self.semantic_tokens) != 4
            or any(type(token) is not str or not token for token in self.semantic_tokens)
        ):
            raise ValueError("semantic contrast surfaces require four tokens")
        if (
            type(self.raw_source_tokens) is not tuple
            or len(self.raw_source_tokens) not in (4, 5)
            or any(type(token) is not str or not token for token in self.raw_source_tokens)
        ):
            raise ValueError("raw contrast source requires four or five tokens")
        if not isinstance(self.teacher_span, v2.CompositionalPromptInput):
            raise TypeError("teacher_span must be a CompositionalPromptInput")
        if (
            type(self.present_factors) is not tuple
            or len(self.present_factors) != 1
            or self.present_factors[0] not in _TAUGHT_FACTORS
        ):
            raise ValueError("each local teacher must carry one taught factor")

    @property
    def factor(self) -> str:
        return self.present_factors[0]

    @property
    def split(self) -> str:
        return _TRIAD_ROWS[self.row_index][0]

    def authentication_record(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "row_index": self.row_index,
            "semantic_tokens": self.semantic_tokens,
            "raw_source_tokens": self.raw_source_tokens,
            "teacher_start_token": self.teacher_span.start_token,
            "teacher_stop_token": self.teacher_span.stop_token,
            "teacher_tokens": self.teacher_span.canonical_tokens,
            "teacher_whole_span_sha256": self.teacher_span.whole_span_sha256,
            "teacher_audit_prompt_sha256": self.teacher_span.audit_prompt_sha256,
            "present_factors": self.present_factors,
            "split": self.split,
        }

    @property
    def record_sha256(self) -> str:
        return _namespaced_sha256(_LESSON_NAMESPACE, self.authentication_record())


@dataclass(frozen=True, slots=True)
class LocalSpanContrastGroup:
    lessons: tuple[LocalSpanLesson, ...]

    def __post_init__(self) -> None:
        if (
            type(self.lessons) is not tuple
            or len(self.lessons) != 3
            or not all(isinstance(lesson, LocalSpanLesson) for lesson in self.lessons)
        ):
            raise ValueError("each frozen contrast group requires three lessons")


@dataclass(frozen=True, slots=True)
class LocalSpanSplit:
    fit_record_sha256s: tuple[str, ...]
    heldout_record_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("fit_record_sha256s", self.fit_record_sha256s),
            ("heldout_record_sha256s", self.heldout_record_sha256s),
        ):
            if type(values) is not tuple or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique identities")
            if any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in values
            ):
                raise ValueError(f"{name} must contain lowercase SHA-256 values")
        if set(self.fit_record_sha256s) & set(self.heldout_record_sha256s):
            raise ValueError("FIT and heldout identities must be disjoint")

    @property
    def sha256(self) -> str:
        return _namespaced_sha256(
            _SPLIT_NAMESPACE,
            {
                "fit": tuple(sorted(self.fit_record_sha256s)),
                "heldout": tuple(sorted(self.heldout_record_sha256s)),
            },
        )


@dataclass(frozen=True, slots=True)
class AtomicSpanMismatch:
    semantic_tokens: tuple[str, str, str, str]
    teacher_tokens: tuple[str, ...]
    split: str
    expected_factors: tuple[str, ...]
    observed_factors: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "semantic_tokens": self.semantic_tokens,
            "teacher_tokens": self.teacher_tokens,
            "split": self.split,
            "expected_factors": self.expected_factors,
            "observed_factors": self.observed_factors,
        }


@dataclass(frozen=True, slots=True)
class MultiSpanBuildSummary:
    status: str
    group_count: int
    lesson_count: int
    fit_count: int
    heldout_count: int
    atomic_pass_count: int
    atomic_mismatch_count: int
    taught_factor_count: int
    silent_head_count: int
    epochs: int
    excitatory_head_update_events: int
    inhibitory_head_update_events: int
    committed_cell_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    composition_attempt_count: int
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int


@dataclass(frozen=True, slots=True)
class FactorProfileResult:
    """Reserved result type; the frozen atomic STOP never constructs one."""

    status: str
    events: tuple[object, ...]
    semantic_sha256: str


def _array_frame(name: str, values: np.ndarray) -> bytes:
    header = _canonical_json(
        {"name": name, "dtype": values.dtype.str, "shape": values.shape}
    )
    raw = np.ascontiguousarray(values, dtype=v2.TICK_DTYPE).tobytes(order="C")
    return (
        len(header).to_bytes(8, "little")
        + header
        + len(raw).to_bytes(8, "little")
        + raw
    )


def _checkpoint_payload(
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> bytes:
    metadata_bytes = _canonical_json(metadata)
    return b"".join(
        (
            _CHECKPOINT_NAMESPACE,
            len(metadata_bytes).to_bytes(8, "little"),
            metadata_bytes,
            _array_frame("excitatory_ticks", excitatory_ticks),
            _array_frame("inhibitory_ticks", inhibitory_ticks),
        )
    )


class FrozenMultiSpanBridge:
    """Immutable failed pre-gate with a deliberately sealed composition seam."""

    __slots__ = (
        "_excitatory_ticks",
        "_inhibitory_ticks",
        "_metadata",
        "_checkpoint_sha256",
        "_result_sha256",
        "build_summary",
        "mismatches",
        "input_sha256",
        "curriculum_sha256",
        "split_sha256",
    )

    def __init__(
        self,
        *,
        excitatory_ticks: np.ndarray,
        inhibitory_ticks: np.ndarray,
        metadata: dict[str, object],
        build_summary: MultiSpanBuildSummary,
        mismatches: tuple[AtomicSpanMismatch, ...],
        input_sha256: str,
        curriculum_sha256: str,
        split_sha256: str,
    ) -> None:
        v2._validate_tick_arrays(excitatory_ticks, inhibitory_ticks)
        self._excitatory_ticks = np.array(
            excitatory_ticks, dtype=v2.TICK_DTYPE, order="C", copy=True
        )
        self._inhibitory_ticks = np.array(
            inhibitory_ticks, dtype=v2.TICK_DTYPE, order="C", copy=True
        )
        self._excitatory_ticks.setflags(write=False)
        self._inhibitory_ticks.setflags(write=False)
        self._metadata = dict(metadata)
        self.build_summary = build_summary
        self.mismatches = mismatches
        self.input_sha256 = input_sha256
        self.curriculum_sha256 = curriculum_sha256
        self.split_sha256 = split_sha256
        self._checkpoint_sha256 = sha256(self.checkpoint_bytes()).hexdigest()
        self._result_sha256 = _namespaced_sha256(
            _RESULT_NAMESPACE,
            self.result_record(include_result_sha256=False),
        )

    @property
    def excitatory_ticks(self) -> np.ndarray:
        values = self._excitatory_ticks.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        values = self._inhibitory_ticks.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def learned_state_bytes(self) -> int:
        return self._excitatory_ticks.nbytes + self._inhibitory_ticks.nbytes

    @property
    def checkpoint_sha256(self) -> str:
        self._authenticate_checkpoint()
        return self._checkpoint_sha256

    @property
    def result_sha256(self) -> str:
        self._authenticate_checkpoint()
        return self._result_sha256

    def checkpoint_bytes(self) -> bytes:
        return _checkpoint_payload(
            self._metadata,
            self._excitatory_ticks,
            self._inhibitory_ticks,
        )

    def _authenticate_checkpoint(self) -> None:
        if sha256(self.checkpoint_bytes()).hexdigest() != self._checkpoint_sha256:
            raise RuntimeError("frozen multi-span V3 checkpoint changed")

    def result_record(self, *, include_result_sha256: bool = True) -> dict[str, object]:
        self._authenticate_checkpoint()
        record: dict[str, object] = {
            "status": self.build_summary.status,
            "claim_boundary": CLAIM_BOUNDARY,
            "group_count": self.build_summary.group_count,
            "lesson_count": self.build_summary.lesson_count,
            "fit_count": self.build_summary.fit_count,
            "heldout_count": self.build_summary.heldout_count,
            "atomic_pass_count": self.build_summary.atomic_pass_count,
            "atomic_mismatch_count": self.build_summary.atomic_mismatch_count,
            "taught_factor_count": self.build_summary.taught_factor_count,
            "silent_head_count": self.build_summary.silent_head_count,
            "training_epochs": self.build_summary.epochs,
            "excitatory_head_update_events": (
                self.build_summary.excitatory_head_update_events
            ),
            "inhibitory_head_update_events": (
                self.build_summary.inhibitory_head_update_events
            ),
            "committed_cell_ticks": self.build_summary.committed_cell_ticks,
            "excitatory_max_ticks": self.build_summary.excitatory_max_ticks,
            "inhibitory_max_ticks": self.build_summary.inhibitory_max_ticks,
            "composition_attempt_count": self.build_summary.composition_attempt_count,
            "public_input_count": self.build_summary.public_input_count,
            "canonical_input_count": self.build_summary.canonical_input_count,
            "protected_input_count": self.build_summary.protected_input_count,
            "mismatches": tuple(mismatch.to_record() for mismatch in self.mismatches),
            "input_sha256": self.input_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "split_sha256": self.split_sha256,
            "extraction_sha256": EXTRACTION_SHA256,
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        if include_result_sha256:
            record["result_sha256"] = self._result_sha256
        return record

    def compose(self, prompt: object) -> FactorProfileResult:
        """Fail closed before inspecting ``prompt`` or recording an attempt."""

        del prompt
        self._authenticate_checkpoint()
        raise RuntimeError(
            "multi-span composition is sealed: the frozen atomic span gate stopped"
        )


def _factor_surface(factor: str) -> tuple[str, ...]:
    return ("less", "than") if factor == "less-than" else (factor,)


def _build_canonical_curriculum() -> tuple[
    tuple[LocalSpanContrastGroup, ...],
    LocalSpanSplit,
]:
    groups: list[LocalSpanContrastGroup] = []
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    group_index = 0
    for triad in _TRIADS:
        for row_index, (membership, connector, swap_distractors) in enumerate(
            _TRIAD_ROWS
        ):
            lessons: list[LocalSpanLesson] = []
            for factor_index, factor in enumerate(triad):
                distractors = [
                    triad[(factor_index + 1) % 3],
                    triad[(factor_index + 2) % 3],
                ]
                if swap_distractors:
                    distractors.reverse()
                semantic_tokens = (
                    factor,
                    connector,
                    distractors[0],
                    distractors[1],
                )
                raw_factor = _factor_surface(factor)
                raw_source = raw_factor + (
                    connector,
                    distractors[0],
                    distractors[1],
                )
                teacher_tokens = raw_factor + (connector, distractors[0])
                teacher = v2._make_prompt_input(
                    " ".join(teacher_tokens),
                    start_token=_SUFFIX_START_TOKEN,
                    audit_prompt=(
                        f"self-posed multispan event bridge v3 group {group_index}: "
                        + " ".join(raw_source)
                    ),
                )
                lesson = LocalSpanLesson(
                    group_index=group_index,
                    row_index=row_index,
                    semantic_tokens=semantic_tokens,
                    raw_source_tokens=raw_source,
                    teacher_span=teacher,
                    present_factors=(factor,),
                )
                lessons.append(lesson)
                destination = fit_ids if membership == FIT else heldout_ids
                destination.append(lesson.record_sha256)
            groups.append(LocalSpanContrastGroup(tuple(lessons)))
            group_index += 1
    return (
        tuple(groups),
        LocalSpanSplit(tuple(fit_ids), tuple(heldout_ids)),
    )


def self_posed_local_span_curriculum() -> tuple[
    tuple[LocalSpanContrastGroup, ...],
    LocalSpanSplit,
]:
    """Return the frozen self-posed inputs; no public corpus is consulted."""

    return _build_canonical_curriculum()


def _flatten(
    groups: tuple[LocalSpanContrastGroup, ...],
) -> tuple[LocalSpanLesson, ...]:
    return tuple(lesson for group in groups for lesson in group.lessons)


def _canonical_lesson_records(
    lessons: tuple[LocalSpanLesson, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        lesson.authentication_record()
        for lesson in sorted(
            lessons,
            key=lambda item: (
                item.semantic_tokens,
                item.teacher_span.canonical_tokens,
            ),
        )
    )


def _validate_curriculum(
    curriculum: Sequence[LocalSpanContrastGroup],
    split: LocalSpanSplit,
) -> tuple[
    tuple[LocalSpanContrastGroup, ...],
    tuple[LocalSpanLesson, ...],
    tuple[LocalSpanLesson, ...],
    tuple[LocalSpanLesson, ...],
]:
    if isinstance(curriculum, (str, bytes)):
        raise TypeError("curriculum must be a sequence of local-span groups")
    groups = tuple(curriculum)
    if not all(isinstance(group, LocalSpanContrastGroup) for group in groups):
        raise TypeError("curriculum contains a non-local-span group")
    if not isinstance(split, LocalSpanSplit):
        raise TypeError("split must be LocalSpanSplit")
    lessons = _flatten(groups)
    identities = tuple(lesson.record_sha256 for lesson in lessons)
    fit_ids = set(split.fit_record_sha256s)
    heldout_ids = set(split.heldout_record_sha256s)
    if (
        len(groups) != GROUP_COUNT
        or len(lessons) != LESSON_COUNT
        or len(identities) != len(set(identities))
        or len(fit_ids) != FIT_COUNT
        or len(heldout_ids) != HELDOUT_COUNT
        or set(identities) != fit_ids | heldout_ids
    ):
        raise ValueError("curriculum does not match the frozen 18/54/36/18 shape")

    canonical_groups, canonical_split = _build_canonical_curriculum()
    canonical_lessons = _flatten(canonical_groups)
    if _canonical_lesson_records(lessons) != _canonical_lesson_records(
        canonical_lessons
    ):
        raise ValueError("local-span curriculum or extraction schedule changed")
    if split.sha256 != canonical_split.sha256:
        raise ValueError("local-span split changed")

    for group in groups:
        semantic_multisets = {
            tuple(sorted(lesson.semantic_tokens)) for lesson in group.lessons
        }
        semantic_widths = {len(lesson.semantic_tokens) for lesson in group.lessons}
        factors = tuple(lesson.factor for lesson in group.lessons)
        if (
            len(semantic_multisets) != 1
            or semantic_widths != {4}
            or len(set(factors)) != 3
        ):
            raise ValueError("semantic contrast matching changed")
        memberships = {
            FIT if lesson.record_sha256 in fit_ids else HELDOUT
            for lesson in group.lessons
        }
        if len(memberships) != 1:
            raise ValueError("a contrast group cannot cross the split")

    fit = tuple(lesson for lesson in lessons if lesson.record_sha256 in fit_ids)
    heldout = tuple(
        lesson for lesson in lessons if lesson.record_sha256 in heldout_ids
    )
    for factor in _TAUGHT_FACTORS:
        if (
            sum(lesson.factor == factor for lesson in fit) != 4
            or sum(lesson.factor == factor for lesson in heldout) != 2
        ):
            raise ValueError("each factor requires exactly four FIT and two heldout")
    if any(
        lesson.teacher_span.canonical_tokens
        in {fit_lesson.teacher_span.canonical_tokens for fit_lesson in fit}
        for lesson in heldout
    ):
        raise ValueError("heldout local spans must be exact-input-disjoint from FIT")
    return groups, lessons, fit, heldout


def _input_sha256(lessons: tuple[LocalSpanLesson, ...]) -> str:
    return _namespaced_sha256(
        _INPUT_NAMESPACE,
        tuple(
            {
                "raw_source_tokens": lesson.raw_source_tokens,
                "teacher_start_token": lesson.teacher_span.start_token,
                "teacher_stop_token": lesson.teacher_span.stop_token,
                "teacher_tokens": lesson.teacher_span.canonical_tokens,
                "teacher_whole_span_sha256": lesson.teacher_span.whole_span_sha256,
                "teacher_audit_prompt_sha256": lesson.teacher_span.audit_prompt_sha256,
            }
            for lesson in sorted(
                lessons,
                key=lambda item: (
                    item.semantic_tokens,
                    item.teacher_span.canonical_tokens,
                ),
            )
        ),
    )


def _curriculum_sha256(lessons: tuple[LocalSpanLesson, ...]) -> str:
    return _namespaced_sha256(
        _CURRICULUM_NAMESPACE,
        _canonical_lesson_records(lessons),
    )


def freeze_multispan_bridge(
    curriculum: Sequence[LocalSpanContrastGroup],
    split: LocalSpanSplit,
) -> FrozenMultiSpanBridge:
    """Spend the fixed atomic pre-gate once and return its immutable STOP."""

    groups, lessons, fit, heldout = _validate_curriculum(curriculum, split)
    del groups, heldout
    excitatory = np.zeros(
        (v2.HEAD_COUNT, v2.INPUT_CELL_COUNT), dtype=v2.TICK_DTYPE
    )
    inhibitory = np.zeros_like(excitatory)
    ordered_fit = tuple(
        sorted(
            fit,
            key=lambda lesson: (
                lesson.teacher_span.canonical_tokens,
                lesson.present_factors,
            ),
        )
    )
    excitatory_updates = 0
    inhibitory_updates = 0
    committed_ticks = 0
    for _epoch in range(v2.TRAINING_EPOCHS):
        for lesson in ordered_fit:
            learned = v2._learn_sequence(
                excitatory,
                inhibitory,
                lesson.teacher_span,
                lesson.present_factors,
            )
            excitatory_updates += len(learned.excitatory_head_updates)
            inhibitory_updates += len(learned.inhibitory_head_updates)
            committed_ticks += learned.committed_cell_ticks

    mismatches: list[AtomicSpanMismatch] = []
    for lesson in sorted(
        lessons,
        key=lambda item: (item.semantic_tokens, item.teacher_span.canonical_tokens),
    ):
        races = v2._race_heads(
            excitatory,
            inhibitory,
            v2._project_prompt(lesson.teacher_span),
        )
        observed = tuple(
            race.factor for race in races if race.decision == v2.PRESENT
        )
        if observed != lesson.present_factors:
            mismatches.append(
                AtomicSpanMismatch(
                    semantic_tokens=lesson.semantic_tokens,
                    teacher_tokens=lesson.teacher_span.canonical_tokens,
                    split=lesson.split,
                    expected_factors=lesson.present_factors,
                    observed_factors=observed,
                )
            )

    mismatch_tuple = tuple(mismatches)
    e_max = int(excitatory.max())
    h_max = int(inhibitory.max())
    actual = (
        LESSON_COUNT - len(mismatch_tuple),
        len(mismatch_tuple),
        excitatory_updates,
        inhibitory_updates,
        committed_ticks,
        e_max,
        h_max,
    )
    expected = (
        EXPECTED_ATOMIC_PASS_COUNT,
        EXPECTED_ATOMIC_MISMATCH_COUNT,
        EXPECTED_EXCITATORY_HEAD_UPDATE_EVENTS,
        EXPECTED_INHIBITORY_HEAD_UPDATE_EVENTS,
        EXPECTED_COMMITTED_CELL_TICKS,
        EXPECTED_EXCITATORY_MAX_TICKS,
        EXPECTED_INHIBITORY_MAX_TICKS,
    )
    if actual != expected:
        raise RuntimeError("frozen V3 atomic-readiness telemetry changed")
    silent_rows = np.asarray(_SILENT_HEADS, dtype=np.intp)
    if np.any(excitatory[silent_rows]) or np.any(inhibitory[silent_rows]):
        raise RuntimeError("untaught V3 factor heads changed")

    input_sha = _input_sha256(lessons)
    curriculum_sha = _curriculum_sha256(lessons)
    summary = MultiSpanBuildSummary(
        status=STOP_STATUS,
        group_count=GROUP_COUNT,
        lesson_count=LESSON_COUNT,
        fit_count=FIT_COUNT,
        heldout_count=HELDOUT_COUNT,
        atomic_pass_count=EXPECTED_ATOMIC_PASS_COUNT,
        atomic_mismatch_count=EXPECTED_ATOMIC_MISMATCH_COUNT,
        taught_factor_count=TAUGHT_FACTOR_COUNT,
        silent_head_count=SILENT_HEAD_COUNT,
        epochs=v2.TRAINING_EPOCHS,
        excitatory_head_update_events=excitatory_updates,
        inhibitory_head_update_events=inhibitory_updates,
        committed_cell_ticks=committed_ticks,
        excitatory_max_ticks=e_max,
        inhibitory_max_ticks=h_max,
        composition_attempt_count=0,
        public_input_count=0,
        canonical_input_count=0,
        protected_input_count=0,
    )
    metadata = {
        "schema": "prompt-factor-multispan-event-bridge-v3-checkpoint",
        "scope": BRIDGE_SCOPE,
        "status": STOP_STATUS,
        "v2_mechanism_scope": v2.BRIDGE_SCOPE,
        "factor_registry_sha256": v2.FACTOR_REGISTRY_SHA256,
        "extraction_sha256": EXTRACTION_SHA256,
        "input_sha256": input_sha,
        "curriculum_sha256": curriculum_sha,
        "split_sha256": split.sha256,
        "head_count": v2.HEAD_COUNT,
        "bank_count": 3,
        "bank_cell_count": v2.BANK_CELL_COUNT,
        "input_cell_count": v2.INPUT_CELL_COUNT,
        "tick_dtype": v2.TICK_DTYPE.str,
        "tick_cap": v2.TICK_CAP,
        "tick_current_pa": v2.TICK_CURRENT_PA,
        "present_base_current_pa": v2.PRESENT_BASE_CURRENT_PA,
        "stop_current_pa": v2.STOP_CURRENT_PA,
        "present_max_current_pa": v2.PRESENT_MAX_CURRENT_PA,
        "race_steps": v2.RACE_STEPS,
        "training_epochs": v2.TRAINING_EPOCHS,
        "array_shape": (v2.HEAD_COUNT, v2.INPUT_CELL_COUNT),
        "learned_state_bytes": v2.LEARNED_STATE_BYTES,
        "winner_margin_tie_break": False,
        "composition_attempt_count": 0,
    }
    return FrozenMultiSpanBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        build_summary=summary,
        mismatches=mismatch_tuple,
        input_sha256=input_sha,
        curriculum_sha256=curriculum_sha,
        split_sha256=split.sha256,
    )


__all__ = [
    "BRIDGE_SCOPE",
    "CLAIM_BOUNDARY",
    "EXTRACTION_SHA256",
    "FIT",
    "FIT_COUNT",
    "GROUP_COUNT",
    "HELDOUT",
    "HELDOUT_COUNT",
    "LESSON_COUNT",
    "SILENT_HEAD_COUNT",
    "STOP_STATUS",
    "TAUGHT_FACTOR_COUNT",
    "AtomicSpanMismatch",
    "FactorProfileResult",
    "FrozenMultiSpanBridge",
    "LocalSpanContrastGroup",
    "LocalSpanLesson",
    "LocalSpanSplit",
    "MultiSpanBuildSummary",
    "freeze_multispan_bridge",
    "self_posed_local_span_curriculum",
]
