from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from experiments.r3_pilot import prompt_factor_multispan_event_bridge_v3 as v3


EXPECTED_EXTRACTION_SHA256 = (
    "809a2ed889765d1bb531a7ccbccd21878f1a171f2ba12d065e1cc8ef94a459a8"
)
EXPECTED_INPUT_SHA256 = (
    "127f148edf67f0bb8746c317b6aac7f36f36338f5d13a4873026153284575ecf"
)
EXPECTED_CURRICULUM_SHA256 = (
    "e3d9f12017b27c8d492e026d61a8140f11555ce8f237453448bfb6aca81d0578"
)
EXPECTED_SPLIT_SHA256 = (
    "438c62f56d45667bd3a52316048101ce3536f37b8cf7cbe956f77af0f8d87a03"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "244429c7061e0e2e85a9088997417e9dbb5e3840c2572132dbc4bb655af1ab0c"
)
EXPECTED_RESULT_SHA256 = (
    "cf623dafde5c02b9c226a60e010348eb9b9a6c44633f29fcd82653dfa5940fd4"
)


@pytest.fixture(scope="module")
def canonical() -> tuple[
    tuple[v3.LocalSpanContrastGroup, ...],
    v3.LocalSpanSplit,
    v3.FrozenMultiSpanBridge,
]:
    curriculum, split = v3.self_posed_local_span_curriculum()
    return curriculum, split, v3.freeze_multispan_bridge(curriculum, split)


def test_frozen_18_group_local_prefix_curriculum_is_exact() -> None:
    curriculum, split = v3.self_posed_local_span_curriculum()
    lessons = tuple(lesson for group in curriculum for lesson in group.lessons)
    fit_ids = set(split.fit_record_sha256s)
    heldout_ids = set(split.heldout_record_sha256s)

    assert (len(curriculum), len(lessons), len(fit_ids), len(heldout_ids)) == (
        18,
        54,
        36,
        18,
    )
    assert len({lesson.record_sha256 for lesson in lessons}) == 54
    assert fit_ids.isdisjoint(heldout_ids)
    assert fit_ids | heldout_ids == {lesson.record_sha256 for lesson in lessons}
    assert v3.EXTRACTION_SHA256 == EXPECTED_EXTRACTION_SHA256

    for group in curriculum:
        assert len(group.lessons) == 3
        assert len({tuple(sorted(row.semantic_tokens)) for row in group.lessons}) == 1
        assert {len(row.semantic_tokens) for row in group.lessons} == {4}
        assert len({row.factor for row in group.lessons}) == 3
    assert {
        len(lesson.teacher_span.canonical_tokens)
        for lesson in lessons
        if lesson.factor == "less-than"
    } == {4}
    assert {
        len(lesson.teacher_span.canonical_tokens)
        for lesson in lessons
        if lesson.factor != "less-than"
    } == {3}

    fit_inputs = {
        lesson.teacher_span.canonical_tokens
        for lesson in lessons
        if lesson.record_sha256 in fit_ids
    }
    heldout_inputs = {
        lesson.teacher_span.canonical_tokens
        for lesson in lessons
        if lesson.record_sha256 in heldout_ids
    }
    assert fit_inputs.isdisjoint(heldout_inputs)
    for factor in v3._TAUGHT_FACTORS:
        assert sum(
            lesson.factor == factor and lesson.record_sha256 in fit_ids
            for lesson in lessons
        ) == 4
        assert sum(
            lesson.factor == factor and lesson.record_sha256 in heldout_ids
            for lesson in lessons
        ) == 2


def test_atomic_pre_gate_banks_exact_terminal_stop_and_hashes(
    canonical: tuple[
        tuple[v3.LocalSpanContrastGroup, ...],
        v3.LocalSpanSplit,
        v3.FrozenMultiSpanBridge,
    ],
) -> None:
    _curriculum, _split, frozen = canonical
    summary = frozen.build_summary
    assert summary == v3.MultiSpanBuildSummary(
        status=v3.STOP_STATUS,
        group_count=18,
        lesson_count=54,
        fit_count=36,
        heldout_count=18,
        atomic_pass_count=40,
        atomic_mismatch_count=14,
        taught_factor_count=9,
        silent_head_count=15,
        epochs=4,
        excitatory_head_update_events=31,
        inhibitory_head_update_events=3,
        committed_cell_ticks=3266,
        excitatory_max_ticks=4,
        inhibitory_max_ticks=1,
        composition_attempt_count=0,
        public_input_count=0,
        canonical_input_count=0,
        protected_input_count=0,
    )
    assert frozen.learned_state_bytes == 147_456
    assert frozen.excitatory_ticks.shape == (24, 1536)
    assert frozen.inhibitory_ticks.shape == (24, 1536)
    assert frozen.excitatory_ticks.dtype == np.dtype("<u2")
    assert frozen.inhibitory_ticks.dtype == np.dtype("<u2")

    assert frozen.input_sha256 == EXPECTED_INPUT_SHA256
    assert frozen.curriculum_sha256 == EXPECTED_CURRICULUM_SHA256
    assert frozen.split_sha256 == EXPECTED_SPLIT_SHA256
    assert frozen.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert frozen.result_sha256 == EXPECTED_RESULT_SHA256
    assert frozen.result_record()["claim_boundary"] == v3.CLAIM_BOUNDARY


def test_exact_fourteen_heldout_mismatches_are_empty_cell_profiles(
    canonical: tuple[
        tuple[v3.LocalSpanContrastGroup, ...],
        v3.LocalSpanSplit,
        v3.FrozenMultiSpanBridge,
    ],
) -> None:
    frozen = canonical[2]
    expected = (
        ("absolute", "instead", "count", "less-than"),
        ("absolute", "never", "count", "less-than"),
        ("add", "instead", "sum", "map"),
        ("add", "never", "sum", "map"),
        ("map", "instead", "add", "sum"),
        ("map", "never", "add", "sum"),
        ("maximum", "instead", "minimum", "subtract"),
        ("maximum", "never", "minimum", "subtract"),
        ("minimum", "instead", "subtract", "maximum"),
        ("minimum", "never", "subtract", "maximum"),
        ("subtract", "instead", "maximum", "minimum"),
        ("subtract", "never", "maximum", "minimum"),
        ("sum", "instead", "map", "add"),
        ("sum", "never", "map", "add"),
    )
    assert tuple(row.semantic_tokens for row in frozen.mismatches) == expected
    assert {row.split for row in frozen.mismatches} == {v3.HELDOUT}
    assert {row.observed_factors for row in frozen.mismatches} == {()}
    assert tuple(row.expected_factors[0] for row in frozen.mismatches) == tuple(
        row[0] for row in expected
    )


def test_reordering_is_noncausal_and_semantic_or_split_tamper_refuses(
    canonical: tuple[
        tuple[v3.LocalSpanContrastGroup, ...],
        v3.LocalSpanSplit,
        v3.FrozenMultiSpanBridge,
    ],
) -> None:
    curriculum, split, frozen = canonical
    reordered_split = v3.LocalSpanSplit(
        tuple(reversed(split.fit_record_sha256s)),
        tuple(reversed(split.heldout_record_sha256s)),
    )
    rebuilt = v3.freeze_multispan_bridge(
        tuple(reversed(curriculum)),
        reordered_split,
    )
    assert rebuilt.checkpoint_sha256 == frozen.checkpoint_sha256
    assert rebuilt.result_sha256 == frozen.result_sha256
    np.testing.assert_array_equal(rebuilt.excitatory_ticks, frozen.excitatory_ticks)
    np.testing.assert_array_equal(rebuilt.inhibitory_ticks, frozen.inhibitory_ticks)

    first = curriculum[0].lessons[0]
    forged = dataclasses.replace(first, present_factors=("sum",))
    forged_group = dataclasses.replace(
        curriculum[0], lessons=(forged,) + curriculum[0].lessons[1:]
    )
    with pytest.raises(ValueError, match="curriculum|split|shape"):
        v3.freeze_multispan_bridge((forged_group,) + curriculum[1:], split)

    with pytest.raises(ValueError, match="split|shape"):
        v3.freeze_multispan_bridge(
            curriculum,
            v3.LocalSpanSplit(
                split.fit_record_sha256s[:-1],
                split.heldout_record_sha256s,
            ),
        )


def test_checkpoint_is_detached_and_compose_fails_before_inspecting_prompt(
    canonical: tuple[
        tuple[v3.LocalSpanContrastGroup, ...],
        v3.LocalSpanSplit,
        v3.FrozenMultiSpanBridge,
    ],
) -> None:
    frozen = canonical[2]
    before_checkpoint = frozen.checkpoint_bytes()
    before_result = frozen.result_record()
    exposed = frozen.excitatory_ticks
    assert not exposed.flags.writeable
    exposed.setflags(write=True)
    exposed[:] = np.uint16(50)
    assert frozen.checkpoint_bytes() == before_checkpoint

    class UninspectablePrompt:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"sealed composition inspected {name}")

    with pytest.raises(RuntimeError, match="atomic span gate stopped"):
        frozen.compose(UninspectablePrompt())
    assert frozen.checkpoint_bytes() == before_checkpoint
    assert frozen.result_record() == before_result
    assert frozen.build_summary.composition_attempt_count == 0

    assert tuple(inspect.signature(v3.freeze_multispan_bridge).parameters) == (
        "curriculum",
        "split",
    )
    assert tuple(inspect.signature(v3.FrozenMultiSpanBridge.compose).parameters) == (
        "self",
        "prompt",
    )
    forbidden = {"retry", "tuning", "current_scale", "epoch_override"}
    assert forbidden.isdisjoint(v3.MultiSpanBuildSummary.__dataclass_fields__)
    assert forbidden.isdisjoint(frozen.result_record())
