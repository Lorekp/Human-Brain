"""Self-posed terminal probe for the frozen core prompt compositions.

The probe extends the banked 54-lesson atomic curriculum only inside this
module.  It rebuilds the same three-bank, 24-head checkpoint from zero through
the bridge's local sequence-end rule, then records the preregistered terminal
STOP.  It never calls or widens the public bridge builder and reads no public,
canonical, or protected task material.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as bridge,
)


PROBE_SCOPE = "SELF_POSED_CORE_PROMPT_COMPOSITION_V1"
STOP_STATUS = "STOP_SELF_POSED_CORE_PROMPT_COMPOSITION_V1"
CLAIM_BOUNDARY = (
    "Self-posed nine-factor core prompt-composition capacity STOP only; no "
    "public or canonical task inputs, protected inputs, desired outputs, "
    "prompt-only coding, family transfer, DEV/eval, M0, Path-2, or C1 evidence."
)

FIT = "FIT"
HELDOUT = "HELDOUT"
GROUP_COUNT = 42
LESSON_COUNT = 108
FIT_COUNT = 72
HELDOUT_COUNT = 36
TAUGHT_FACTOR_COUNT = 9
SILENT_HEAD_COUNT = 15

EXPECTED_EXCITATORY_HEAD_UPDATE_EVENTS = 77
EXPECTED_INHIBITORY_HEAD_UPDATE_EVENTS = 43
EXPECTED_COMMITTED_CELL_TICKS = 15_282
EXPECTED_EXCITATORY_MAX_TICKS = 14
EXPECTED_INHIBITORY_MAX_TICKS = 10

EXPECTED_INPUT_SHA256 = (
    "516e401c4dbac54ab20f2183fa39e9ac11abe4756155c9739f9722f72b97ece9"
)
EXPECTED_CURRICULUM_SHA256 = (
    "deb4b8e1cb79d8dc1bbff51a2234eac06eda3dcfe03e722ed6e9a6464f139080"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "ad4585ad76d184f36cf0edd0a52883d68fb13ed73735e4f20dcd73eb60892c47"
)
EXPECTED_RESULT_SHA256 = (
    "1b229f5e091cc38b46de3bee78b89fc34778f72fd1066327b2fbfc8dfe0406b1"
)

_AUDIT_PROMPT = "self-posed core prompt composition probe v1"
_PAIR_ROWS = (
    (FIT, "rather", "than"),
    (FIT, "instead", "of"),
    (FIT, "never", "over"),
    (FIT, "rather", "of"),
    (HELDOUT, "instead", "than"),
    (HELDOUT, "never", "of"),
)
_TRIAD_ROWS = (
    (FIT, "rather", False),
    (FIT, "instead", False),
    (FIT, "never", False),
    (FIT, "rather", True),
    (HELDOUT, "instead", True),
    (HELDOUT, "never", True),
)
_ATOMIC_PAIR_FAMILIES = (
    ("add", "subtract"),
    ("minimum", "maximum"),
    ("absolute", "less-than"),
)
_ATOMIC_TRIAD = ("map", "sum", "count")
_COMPOSITION_TRIADS = (
    ("add", "map", "sum"),
    ("less-than", "count", "map"),
    ("minimum", "maximum", "sum"),
)
_TAUGHT_FACTORS = bridge.FACTOR_REGISTRY[:TAUGHT_FACTOR_COUNT]


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


def _registry_ordered_pair(first: str, second: str) -> tuple[str, str]:
    ordered = tuple(
        factor
        for factor in bridge.FACTOR_REGISTRY
        if factor in (first, second)
    )
    if len(ordered) != 2:
        raise RuntimeError("composition pair does not contain two registry factors")
    return ordered  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _LessonSpec:
    group_index: int
    canonical_tokens: tuple[str, str, str, str]
    present_factors: tuple[str, ...]
    split: str


def _build_frozen_specs() -> tuple[_LessonSpec, ...]:
    specs: list[_LessonSpec] = []
    group_index = 0
    for left, right in _ATOMIC_PAIR_FAMILIES:
        for split, first_connector, second_connector in _PAIR_ROWS:
            specs.extend(
                (
                    _LessonSpec(
                        group_index,
                        (left, first_connector, second_connector, right),
                        (left,),
                        split,
                    ),
                    _LessonSpec(
                        group_index,
                        (right, first_connector, second_connector, left),
                        (right,),
                        split,
                    ),
                )
            )
            group_index += 1

    for split, connector, swap_distractors in _TRIAD_ROWS:
        for index, factor in enumerate(_ATOMIC_TRIAD):
            distractors = (
                _ATOMIC_TRIAD[(index + 1) % 3],
                _ATOMIC_TRIAD[(index + 2) % 3],
            )
            if swap_distractors:
                distractors = tuple(reversed(distractors))
            specs.append(
                _LessonSpec(
                    group_index,
                    (factor, connector, distractors[0], distractors[1]),
                    (factor,),
                    split,
                )
            )
        group_index += 1

    for family in _COMPOSITION_TRIADS:
        for split, connector, swap_distractors in _TRIAD_ROWS:
            for index, factor in enumerate(family):
                distractors = (
                    family[(index + 1) % 3],
                    family[(index + 2) % 3],
                )
                if swap_distractors:
                    distractors = tuple(reversed(distractors))
                companion = family[(index + 1) % 3]
                specs.append(
                    _LessonSpec(
                        group_index,
                        (factor, connector, distractors[0], distractors[1]),
                        _registry_ordered_pair(factor, companion),
                        split,
                    )
                )
            group_index += 1
    return tuple(specs)


_FROZEN_SPECS = _build_frozen_specs()


@dataclass(frozen=True, slots=True)
class _ProbeFixture:
    groups: tuple[bridge.CompositionalContrastGroup, ...]
    fit_span_sha256s: tuple[str, ...]
    heldout_span_sha256s: tuple[str, ...]


def _canonical_fixture() -> _ProbeFixture:
    grouped: list[list[bridge.CompositionalLesson]] = [
        [] for _ in range(GROUP_COUNT)
    ]
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    for spec in _FROZEN_SPECS:
        prompt_input = bridge._make_prompt_input(
            " ".join(spec.canonical_tokens),
            start_token=7,
            audit_prompt=_AUDIT_PROMPT,
        )
        lesson = bridge.CompositionalLesson(
            prompt_input,
            spec.present_factors,
        )
        grouped[spec.group_index].append(lesson)
        membership = fit_ids if spec.split == FIT else heldout_ids
        membership.append(prompt_input.whole_span_sha256)
    return _ProbeFixture(
        groups=tuple(
            bridge.CompositionalContrastGroup(tuple(lessons))
            for lessons in grouped
        ),
        fit_span_sha256s=tuple(fit_ids),
        heldout_span_sha256s=tuple(heldout_ids),
    )


def _controlled_fixture(*, relabel_authentication: bool) -> _ProbeFixture:
    canonical = _canonical_fixture()
    canonical_fit = set(canonical.fit_span_sha256s)
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    groups: list[bridge.CompositionalContrastGroup] = []
    relabel_index = 0
    for group in reversed(canonical.groups):
        lessons: list[bridge.CompositionalLesson] = []
        for lesson in reversed(group.lessons):
            prompt_input = lesson.prompt_input
            if relabel_authentication:
                relabel_index += 1
                prompt_input = replace(
                    prompt_input,
                    whole_span_sha256=f"{10_000 + relabel_index:064x}",
                    audit_prompt_sha256=f"{20_000 + relabel_index:064x}",
                )
            lessons.append(replace(lesson, prompt_input=prompt_input))
            membership = (
                fit_ids
                if lesson.prompt_input.whole_span_sha256 in canonical_fit
                else heldout_ids
            )
            membership.append(prompt_input.whole_span_sha256)
        groups.append(bridge.CompositionalContrastGroup(tuple(lessons)))
    return _ProbeFixture(tuple(groups), tuple(fit_ids), tuple(heldout_ids))


def _flatten(
    fixture: _ProbeFixture,
) -> tuple[bridge.CompositionalLesson, ...]:
    return tuple(
        lesson for group in fixture.groups for lesson in group.lessons
    )


def _validate_bridge_contract() -> None:
    actual = (
        bridge.FACTOR_REGISTRY,
        bridge.HEAD_COUNT,
        len(bridge._ANATOMY.banks),
        tuple(bank.cell_count for bank in bridge._ANATOMY.banks),
        tuple(bank.global_offset for bank in bridge._ANATOMY.banks),
        bridge.INPUT_CELL_COUNT,
        bridge.LEARNED_STATE_BYTES,
        bridge.TICK_DTYPE.str,
        bridge.TICK_CAP,
        bridge.TICK_CURRENT_PA,
        bridge.PRESENT_BASE_CURRENT_PA,
        bridge.STOP_CURRENT_PA,
        bridge.PRESENT_MAX_CURRENT_PA,
        bridge.RACE_STEPS,
        bridge.TRAINING_EPOCHS,
    )
    expected = (
        bridge.FACTOR_REGISTRY,
        24,
        3,
        (512, 512, 512),
        (0, 512, 1024),
        1536,
        147_456,
        "<u2",
        50,
        32.0,
        600.0,
        650.0,
        2_200.0,
        600,
        4,
    )
    if actual != expected:
        raise RuntimeError("banked compositional bridge contract changed")


def _validate_fixture(
    fixture: _ProbeFixture,
) -> tuple[bridge.CompositionalLesson, ...]:
    lessons = _flatten(fixture)
    fit_ids = set(fixture.fit_span_sha256s)
    heldout_ids = set(fixture.heldout_span_sha256s)
    identities = tuple(
        lesson.prompt_input.whole_span_sha256 for lesson in lessons
    )
    exact_inputs = tuple(
        lesson.prompt_input.canonical_tokens for lesson in lessons
    )
    if (
        len(fixture.groups) != GROUP_COUNT
        or len(lessons) != LESSON_COUNT
        or len(fit_ids) != FIT_COUNT
        or len(heldout_ids) != HELDOUT_COUNT
        or len(set(identities)) != LESSON_COUNT
        or len(set(exact_inputs)) != LESSON_COUNT
        or fit_ids & heldout_ids
        or set(identities) != fit_ids | heldout_ids
    ):
        raise ValueError("probe fixture does not match the frozen 108/72/36 split")

    expected_semantics = {
        spec.canonical_tokens: (spec.present_factors, spec.split)
        for spec in _FROZEN_SPECS
    }
    actual_semantics = {
        lesson.prompt_input.canonical_tokens: (
            lesson.present_factors,
            FIT
            if lesson.prompt_input.whole_span_sha256 in fit_ids
            else HELDOUT,
        )
        for lesson in lessons
    }
    if actual_semantics != expected_semantics:
        raise ValueError("probe fixture semantic table changed")

    expected_groups = {
        frozenset(
            spec.canonical_tokens
            for spec in _FROZEN_SPECS
            if spec.group_index == group_index
        )
        for group_index in range(GROUP_COUNT)
    }
    actual_groups = {
        frozenset(
            lesson.prompt_input.canonical_tokens for lesson in group.lessons
        )
        for group in fixture.groups
    }
    if actual_groups != expected_groups:
        raise ValueError("probe matched-group boundaries changed")

    fit = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in fit_ids
    )
    heldout = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in heldout_ids
    )
    for factor in _TAUGHT_FACTORS:
        fit_features = tuple(set() for _ in range(3))
        for lesson in fit:
            if factor not in lesson.present_factors:
                continue
            for destination, source in zip(
                fit_features,
                bridge._feature_keys(lesson.prompt_input.canonical_tokens),
                strict=True,
            ):
                destination.update(source)
        for lesson in heldout:
            if factor not in lesson.present_factors:
                continue
            token_keys, slot_keys, transition_keys = bridge._feature_keys(
                lesson.prompt_input.canonical_tokens
            )
            if (
                not token_keys <= fit_features[0]
                or not slot_keys <= fit_features[1]
                or len(transition_keys & fit_features[2]) < 2
            ):
                raise ValueError(
                    "heldout factor features violate the frozen support gate"
                )
    bridge._validate_feature_geometry(
        fixture.groups,
        lessons,
        _TAUGHT_FACTORS,
    )
    return lessons


@dataclass(frozen=True, slots=True)
class ProbeMismatch:
    canonical_tokens: tuple[str, ...]
    split: str
    expected_factors: tuple[str, ...]
    observed_factors: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "canonical_tokens": self.canonical_tokens,
            "split": self.split,
            "expected_factors": self.expected_factors,
            "observed_factors": self.observed_factors,
        }


EXPECTED_MISMATCHES = (
    ProbeMismatch(
        ("count", "rather", "less-than", "map"),
        FIT,
        ("map", "count"),
        ("count",),
    ),
    ProbeMismatch(
        ("count", "rather", "map", "less-than"),
        FIT,
        ("map", "count"),
        ("count",),
    ),
    ProbeMismatch(
        ("less-than", "instead", "map", "count"),
        HELDOUT,
        ("less-than", "count"),
        ("less-than",),
    ),
    ProbeMismatch(
        ("map", "instead", "add", "sum"),
        HELDOUT,
        ("map", "sum"),
        ("add", "map", "sum"),
    ),
    ProbeMismatch(
        ("map", "rather", "count", "sum"),
        FIT,
        ("map",),
        ("map", "sum"),
    ),
    ProbeMismatch(
        ("maximum", "instead", "than", "minimum"),
        HELDOUT,
        ("maximum",),
        ("maximum", "sum"),
    ),
    ProbeMismatch(
        ("maximum", "rather", "than", "minimum"),
        FIT,
        ("maximum",),
        ("maximum", "sum"),
    ),
    ProbeMismatch(
        ("minimum", "instead", "sum", "maximum"),
        HELDOUT,
        ("minimum", "maximum"),
        ("minimum", "maximum", "sum"),
    ),
    ProbeMismatch(
        ("minimum", "never", "sum", "maximum"),
        HELDOUT,
        ("minimum", "maximum"),
        ("minimum", "maximum", "sum"),
    ),
    ProbeMismatch(
        ("minimum", "rather", "sum", "maximum"),
        FIT,
        ("minimum", "maximum"),
        ("minimum", "maximum", "sum"),
    ),
    ProbeMismatch(
        ("sum", "never", "map", "count"),
        HELDOUT,
        ("sum",),
        (),
    ),
)


@dataclass(frozen=True, slots=True)
class CompositionProbeResult:
    status: str
    claim_boundary: str
    group_count: int
    lesson_count: int
    fit_count: int
    heldout_count: int
    taught_factor_count: int
    silent_head_count: int
    training_epochs: int
    excitatory_head_update_events: int
    inhibitory_head_update_events: int
    committed_cell_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    mismatches: tuple[ProbeMismatch, ...]
    self_posed_input_count: int
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    checkpoint_immutable: bool
    input_sha256: str
    curriculum_sha256: str
    checkpoint_sha256: str
    result_sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            **_result_payload(self),
            "result_sha256": self.result_sha256,
        }


def _result_payload(result: CompositionProbeResult) -> dict[str, object]:
    return {
        "status": result.status,
        "claim_boundary": result.claim_boundary,
        "group_count": result.group_count,
        "lesson_count": result.lesson_count,
        "fit_count": result.fit_count,
        "heldout_count": result.heldout_count,
        "taught_factor_count": result.taught_factor_count,
        "silent_head_count": result.silent_head_count,
        "training_epochs": result.training_epochs,
        "excitatory_head_update_events": result.excitatory_head_update_events,
        "inhibitory_head_update_events": result.inhibitory_head_update_events,
        "committed_cell_ticks": result.committed_cell_ticks,
        "excitatory_max_ticks": result.excitatory_max_ticks,
        "inhibitory_max_ticks": result.inhibitory_max_ticks,
        "mismatches": tuple(
            mismatch.to_record() for mismatch in result.mismatches
        ),
        "self_posed_input_count": result.self_posed_input_count,
        "public_input_count": result.public_input_count,
        "canonical_input_count": result.canonical_input_count,
        "protected_input_count": result.protected_input_count,
        "checkpoint_immutable": result.checkpoint_immutable,
        "input_sha256": result.input_sha256,
        "curriculum_sha256": result.curriculum_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
    }


@dataclass(frozen=True, slots=True)
class _ProbeExecution:
    result: CompositionProbeResult
    checkpoint: bridge.FrozenCompositionalBridge
    prompt_inputs: tuple[bridge.CompositionalPromptInput, ...]
    semantic_races: tuple[
        tuple[tuple[str, ...], tuple[bridge.FactorHeadRace, ...]], ...
    ]


def _input_sha256(
    lessons: tuple[bridge.CompositionalLesson, ...],
) -> str:
    return _namespaced_sha256(
        b"self-posed-core-prompt-composition-v1-inputs\0",
        tuple(
            {
                "start_token": lesson.prompt_input.start_token,
                "stop_token": lesson.prompt_input.stop_token,
                "canonical_tokens": lesson.prompt_input.canonical_tokens,
                "whole_span_sha256": lesson.prompt_input.whole_span_sha256,
                "audit_prompt_sha256": lesson.prompt_input.audit_prompt_sha256,
            }
            for lesson in sorted(
                lessons,
                key=lambda item: item.prompt_input.canonical_tokens,
            )
        ),
    )


def _curriculum_sha256(
    lessons: tuple[bridge.CompositionalLesson, ...],
    fit_ids: set[str],
) -> str:
    return _namespaced_sha256(
        b"self-posed-core-prompt-composition-v1-curriculum\0",
        tuple(
            {
                "canonical_tokens": lesson.prompt_input.canonical_tokens,
                "present_factors": lesson.present_factors,
                "split": (
                    FIT
                    if lesson.prompt_input.whole_span_sha256 in fit_ids
                    else HELDOUT
                ),
            }
            for lesson in sorted(
                lessons,
                key=lambda item: item.prompt_input.canonical_tokens,
            )
        ),
    )


def _fit_checkpoint(
    lessons: tuple[bridge.CompositionalLesson, ...],
    fit_ids: set[str],
) -> bridge.FrozenCompositionalBridge:
    excitatory = np.zeros(
        (bridge.HEAD_COUNT, bridge.INPUT_CELL_COUNT),
        dtype=bridge.TICK_DTYPE,
    )
    inhibitory = np.zeros_like(excitatory)
    fit = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in fit_ids
    )
    ordered_fit = tuple(
        sorted(
            fit,
            key=lambda lesson: (
                lesson.prompt_input.canonical_tokens,
                lesson.present_factors,
            ),
        )
    )
    excitatory_updates = 0
    inhibitory_updates = 0
    committed_ticks = 0
    for _epoch in range(bridge.TRAINING_EPOCHS):
        for lesson in ordered_fit:
            learned = bridge._learn_sequence(
                excitatory,
                inhibitory,
                lesson.prompt_input,
                lesson.present_factors,
            )
            excitatory_updates += len(learned.excitatory_head_updates)
            inhibitory_updates += len(learned.inhibitory_head_updates)
            committed_ticks += learned.committed_cell_ticks

    summary = bridge.BridgeBuildSummary(
        epochs=bridge.TRAINING_EPOCHS,
        fit_sequence_count=FIT_COUNT,
        generalization_sequence_count=HELDOUT_COUNT,
        matched_contrast_group_count=GROUP_COUNT,
        excitatory_head_update_events=excitatory_updates,
        inhibitory_head_update_events=inhibitory_updates,
        committed_cell_ticks=committed_ticks,
    )
    return bridge.FrozenCompositionalBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata={
            "schema": "self-posed-core-prompt-composition-checkpoint-v1",
            "scope": PROBE_SCOPE,
            "bridge_scope": bridge.BRIDGE_SCOPE,
            "factor_registry_sha256": bridge.FACTOR_REGISTRY_SHA256,
            "head_count": bridge.HEAD_COUNT,
            "bank_cell_count": bridge.BANK_CELL_COUNT,
            "input_cell_count": bridge.INPUT_CELL_COUNT,
            "training_epochs": bridge.TRAINING_EPOCHS,
            "race_steps": bridge.RACE_STEPS,
            "array_shape": (bridge.HEAD_COUNT, bridge.INPUT_CELL_COUNT),
            "learned_state_bytes": bridge.LEARNED_STATE_BYTES,
        },
        build_summary=summary,
    )


def _execute_fixture(fixture: _ProbeFixture) -> _ProbeExecution:
    _validate_bridge_contract()
    lessons = _validate_fixture(fixture)
    fit_ids = set(fixture.fit_span_sha256s)
    checkpoint = _fit_checkpoint(lessons, fit_ids)
    excitatory = checkpoint.excitatory_ticks
    inhibitory = checkpoint.inhibitory_ticks
    e_max = int(excitatory.max())
    h_max = int(inhibitory.max())
    summary = checkpoint.build_summary
    if (
        summary.excitatory_head_update_events
        != EXPECTED_EXCITATORY_HEAD_UPDATE_EVENTS
        or summary.inhibitory_head_update_events
        != EXPECTED_INHIBITORY_HEAD_UPDATE_EVENTS
        or summary.committed_cell_ticks != EXPECTED_COMMITTED_CELL_TICKS
        or e_max != EXPECTED_EXCITATORY_MAX_TICKS
        or h_max != EXPECTED_INHIBITORY_MAX_TICKS
        or np.any(excitatory[TAUGHT_FACTOR_COUNT:])
        or np.any(inhibitory[TAUGHT_FACTOR_COUNT:])
    ):
        raise RuntimeError("unchanged-anatomy composition telemetry changed")

    checkpoint_before = checkpoint.checkpoint_bytes()
    heldout_ids = set(fixture.heldout_span_sha256s)
    mismatches: list[ProbeMismatch] = []
    races: list[tuple[tuple[str, ...], tuple[bridge.FactorHeadRace, ...]]] = []
    ordered_lessons = tuple(
        sorted(lessons, key=lambda item: item.prompt_input.canonical_tokens)
    )
    for lesson in ordered_lessons:
        inference = checkpoint.infer(lesson.prompt_input)
        races.append(
            (lesson.prompt_input.canonical_tokens, inference.head_races)
        )
        if inference.present_factors != lesson.present_factors:
            mismatches.append(
                ProbeMismatch(
                    lesson.prompt_input.canonical_tokens,
                    HELDOUT
                    if lesson.prompt_input.whole_span_sha256 in heldout_ids
                    else FIT,
                    lesson.present_factors,
                    inference.present_factors,
                )
            )
    mismatch_tuple = tuple(mismatches)
    if mismatch_tuple != EXPECTED_MISMATCHES:
        raise RuntimeError("canonical 11-mismatch composition STOP changed")
    checkpoint_immutable = checkpoint.checkpoint_bytes() == checkpoint_before
    if not checkpoint_immutable:
        raise RuntimeError("probe inference mutated the frozen checkpoint")

    input_sha = _input_sha256(lessons)
    curriculum_sha = _curriculum_sha256(lessons, fit_ids)
    provisional = CompositionProbeResult(
        status=STOP_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        group_count=GROUP_COUNT,
        lesson_count=LESSON_COUNT,
        fit_count=FIT_COUNT,
        heldout_count=HELDOUT_COUNT,
        taught_factor_count=TAUGHT_FACTOR_COUNT,
        silent_head_count=SILENT_HEAD_COUNT,
        training_epochs=bridge.TRAINING_EPOCHS,
        excitatory_head_update_events=(
            summary.excitatory_head_update_events
        ),
        inhibitory_head_update_events=(
            summary.inhibitory_head_update_events
        ),
        committed_cell_ticks=summary.committed_cell_ticks,
        excitatory_max_ticks=e_max,
        inhibitory_max_ticks=h_max,
        mismatches=mismatch_tuple,
        self_posed_input_count=LESSON_COUNT,
        public_input_count=0,
        canonical_input_count=0,
        protected_input_count=0,
        checkpoint_immutable=True,
        input_sha256=input_sha,
        curriculum_sha256=curriculum_sha,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        result_sha256="",
    )
    result_sha = _namespaced_sha256(
        b"self-posed-core-prompt-composition-v1-result\0",
        _result_payload(provisional),
    )
    result = replace(provisional, result_sha256=result_sha)
    return _ProbeExecution(
        result=result,
        checkpoint=checkpoint,
        prompt_inputs=tuple(
            lesson.prompt_input for lesson in ordered_lessons
        ),
        semantic_races=tuple(races),
    )


def run_probe() -> CompositionProbeResult:
    """Return the authenticated deterministic terminal composition STOP."""

    result = _execute_fixture(_canonical_fixture()).result
    expected_hashes = (
        EXPECTED_INPUT_SHA256,
        EXPECTED_CURRICULUM_SHA256,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_RESULT_SHA256,
    )
    observed_hashes = (
        result.input_sha256,
        result.curriculum_sha256,
        result.checkpoint_sha256,
        result.result_sha256,
    )
    if observed_hashes != expected_hashes:
        raise RuntimeError("canonical core-composition probe hashes changed")
    return result


__all__ = [
    "CLAIM_BOUNDARY",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_COMMITTED_CELL_TICKS",
    "EXPECTED_CURRICULUM_SHA256",
    "EXPECTED_EXCITATORY_HEAD_UPDATE_EVENTS",
    "EXPECTED_EXCITATORY_MAX_TICKS",
    "EXPECTED_INHIBITORY_HEAD_UPDATE_EVENTS",
    "EXPECTED_INHIBITORY_MAX_TICKS",
    "EXPECTED_INPUT_SHA256",
    "EXPECTED_MISMATCHES",
    "EXPECTED_RESULT_SHA256",
    "FIT_COUNT",
    "GROUP_COUNT",
    "HELDOUT_COUNT",
    "LESSON_COUNT",
    "PROBE_SCOPE",
    "SILENT_HEAD_COUNT",
    "STOP_STATUS",
    "TAUGHT_FACTOR_COUNT",
    "CompositionProbeResult",
    "ProbeMismatch",
    "run_probe",
]
