from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as bridge,
)


_PAIR_ROWS = (
    (True, "rather", "than"),
    (True, "instead", "of"),
    (True, "never", "over"),
    (True, "rather", "of"),
    (False, "instead", "than"),
    (False, "never", "of"),
)
_TRIAD_ROWS = (
    (True, "rather", False),
    (True, "instead", False),
    (True, "never", False),
    (True, "rather", True),
    (False, "instead", True),
    (False, "never", True),
)
_CORE_LAYERS = (
    ("add", "subtract"),
    ("minimum", "maximum"),
    ("absolute", "less-than"),
    ("map", "sum", "count"),
)


def _flatten(
    curriculum: tuple[bridge.CompositionalContrastGroup, ...],
) -> tuple[bridge.CompositionalLesson, ...]:
    return tuple(lesson for group in curriculum for lesson in group.lessons)


def _lesson(surface: str, factor: str) -> bridge.CompositionalLesson:
    return bridge.CompositionalLesson(
        bridge._make_prompt_input(
            surface,
            start_token=7,
            audit_prompt="self-posed cumulative compositional curriculum",
        ),
        (factor,),
    )


def _pair_groups(
    factors: tuple[str, str],
) -> tuple[tuple[bridge.CompositionalContrastGroup, bool], ...]:
    left, right = factors
    return tuple(
        (
            bridge.CompositionalContrastGroup(
                (
                    _lesson(f"{left} {first} {second} {right}", left),
                    _lesson(f"{right} {first} {second} {left}", right),
                )
            ),
            is_fit,
        )
        for is_fit, first, second in _PAIR_ROWS
    )


def _triad_groups(
    factors: tuple[str, str, str],
) -> tuple[tuple[bridge.CompositionalContrastGroup, bool], ...]:
    rows: list[tuple[bridge.CompositionalContrastGroup, bool]] = []
    for is_fit, connector, swap_distractors in _TRIAD_ROWS:
        lessons: list[bridge.CompositionalLesson] = []
        for index, factor in enumerate(factors):
            distractors = (factors[(index + 1) % 3], factors[(index + 2) % 3])
            if swap_distractors:
                distractors = tuple(reversed(distractors))
            lessons.append(
                _lesson(
                    f"{factor} {connector} {distractors[0]} {distractors[1]}",
                    factor,
                )
            )
        rows.append((bridge.CompositionalContrastGroup(tuple(lessons)), is_fit))
    return tuple(rows)


def _self_posed_curriculum(*, max_layer: int = 3) -> tuple[
    tuple[bridge.CompositionalContrastGroup, ...],
    bridge.CompositionalSplit,
]:
    if not 0 <= max_layer < len(_CORE_LAYERS):
        raise ValueError("max_layer lies outside the frozen core curriculum")
    groups: list[bridge.CompositionalContrastGroup] = []
    fit: list[str] = []
    heldout: list[str] = []
    for factors in _CORE_LAYERS[: max_layer + 1]:
        grouped = (
            _pair_groups(factors)
            if len(factors) == 2
            else _triad_groups(factors)
        )
        for group, is_fit in grouped:
            groups.append(group)
            membership = fit if is_fit else heldout
            membership.extend(
                lesson.prompt_input.whole_span_sha256 for lesson in group.lessons
            )
    return (
        tuple(groups),
        bridge.CompositionalSplit(tuple(fit), tuple(heldout)),
    )


def _replace_group(
    curriculum: tuple[bridge.CompositionalContrastGroup, ...],
    split: bridge.CompositionalSplit,
    group_index: int,
    replacement: bridge.CompositionalContrastGroup,
) -> tuple[
    tuple[bridge.CompositionalContrastGroup, ...],
    bridge.CompositionalSplit,
]:
    original = curriculum[group_index]
    if len(original.lessons) != len(replacement.lessons):
        raise ValueError("replacement group width changed")
    relabel = {
        before.prompt_input.whole_span_sha256: after.prompt_input.whole_span_sha256
        for before, after in zip(
            original.lessons,
            replacement.lessons,
            strict=True,
        )
    }
    groups = list(curriculum)
    groups[group_index] = replacement
    return (
        tuple(groups),
        bridge.CompositionalSplit(
            tuple(relabel.get(value, value) for value in split.fit_span_sha256s),
            tuple(
                relabel.get(value, value)
                for value in split.generalization_span_sha256s
            ),
        ),
    )


@pytest.fixture(scope="module")
def frozen_fixture() -> tuple[
    tuple[bridge.CompositionalContrastGroup, ...],
    bridge.CompositionalSplit,
    bridge.FrozenCompositionalBridge,
]:
    curriculum, split = _self_posed_curriculum()
    return curriculum, split, bridge.build_frozen_bridge(curriculum, split)


def test_private_projector_drives_three_disjoint_fitted_glif_banks() -> None:
    prompt_input = bridge._make_prompt_input(
        "add rather than subtract",
        start_token=11,
        audit_prompt="a self-posed physical-drive sentence",
    )
    projected = bridge._project_prompt(prompt_input)

    assert len(bridge._ANATOMY.banks) == 3
    assert tuple(bank.global_offset for bank in bridge._ANATOMY.banks) == (
        0,
        512,
        1024,
    )
    assert tuple(bank.cell_count for bank in bridge._ANATOMY.banks) == (
        512,
        512,
        512,
    )
    assert tuple(map(len, (
        projected.token_assemblies,
        projected.slot_assemblies,
        projected.transition_assemblies,
    ))) == (4, 4, 3)
    for bank_index, assemblies in enumerate(
        (
            projected.token_assemblies,
            projected.slot_assemblies,
            projected.transition_assemblies,
        )
    ):
        lower = bank_index * bridge.BANK_CELL_COUNT
        upper = lower + bridge.BANK_CELL_COUNT
        assert all(
            assembly.bank_index == bank_index
            and len(assembly.local_cells)
            == bridge.ASSEMBLY_ACTIVE_CELL_COUNT
            and len(set(assembly.local_cells))
            == bridge.ASSEMBLY_ACTIVE_CELL_COUNT
            and all(lower <= cell < upper for cell in assembly.global_cells)
            for assembly in assemblies
        )
    assert len(projected.active_global_cells) == projected.active_cell_count
    assert all(
        drive.spike_local_cells == drive.active_local_cells
        and drive.active_local_cells
        and drive.first_spike_step >= 0
        for drive in projected.bank_drives
    )


def test_local_false_negative_and_false_positive_rule_has_two_update_ladder() -> None:
    prompt_input = bridge._make_prompt_input("add rather than subtract")
    projected = bridge._project_prompt(prompt_input)
    excitatory = np.zeros(
        (bridge.HEAD_COUNT, bridge.INPUT_CELL_COUNT),
        dtype=bridge.TICK_DTYPE,
    )
    inhibitory = np.zeros_like(excitatory)
    add_index = bridge.FACTOR_REGISTRY.index("add")
    active = np.asarray(projected.active_global_cells, dtype=np.intp)

    initial = bridge._race_heads(excitatory, inhibitory, projected)[add_index]
    assert initial.decision == bridge.STOP
    assert initial.difference_ticks == 0
    assert initial.present_current_pa == 600.0
    assert initial.spike_step == 213

    first_error = bridge._learn_sequence(
        excitatory, inhibitory, prompt_input, ("add",)
    )
    assert first_error.excitatory_head_updates == (add_index,)
    assert first_error.inhibitory_head_updates == ()
    np.testing.assert_array_equal(excitatory[add_index, active], 1)
    after_one = bridge._race_heads(excitatory, inhibitory, projected)[add_index]
    assert after_one.decision == bridge.STOP
    assert after_one.difference_ticks == projected.active_cell_count
    assert after_one.present_current_pa == 632.0
    assert after_one.winner_cell == bridge.STOP_CELL
    assert after_one.spike_step == 213

    second_error = bridge._learn_sequence(
        excitatory, inhibitory, prompt_input, ("add",)
    )
    assert second_error.excitatory_head_updates == (add_index,)
    acquired = bridge._race_heads(excitatory, inhibitory, projected)[add_index]
    assert acquired.decision == bridge.PRESENT
    assert acquired.difference_ticks == 2 * projected.active_cell_count
    assert acquired.present_current_pa == 664.0
    assert acquired.winner_cell == bridge.PRESENT_CELL
    assert acquired.spike_step == 208

    false_positive = bridge._learn_sequence(
        excitatory, inhibitory, prompt_input, ()
    )
    assert false_positive.excitatory_head_updates == ()
    assert false_positive.inhibitory_head_updates == (add_index,)
    np.testing.assert_array_equal(inhibitory[add_index, active], 1)
    inhibited = bridge._race_heads(excitatory, inhibitory, projected)[add_index]
    assert inhibited.decision == bridge.STOP
    assert inhibited.present_current_pa == 632.0


def test_builder_materializes_all_heads_and_exact_persistent_state(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalContrastGroup, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    _curriculum, _split, frozen = frozen_fixture

    assert len(bridge.FACTOR_REGISTRY) == bridge.HEAD_COUNT == 24
    assert frozen.excitatory_ticks.shape == (24, 1536)
    assert frozen.inhibitory_ticks.shape == (24, 1536)
    assert frozen.excitatory_ticks.dtype == np.dtype("uint16")
    assert frozen.inhibitory_ticks.dtype == np.dtype("uint16")
    assert frozen.learned_state_bytes == bridge.LEARNED_STATE_BYTES == 147_456
    assert not frozen.excitatory_ticks.flags.writeable
    assert not frozen.inhibitory_ticks.flags.writeable
    np.testing.assert_array_equal(frozen.excitatory_ticks[9:], 0)
    np.testing.assert_array_equal(frozen.inhibitory_ticks[9:], 0)
    assert all(np.any(frozen.excitatory_ticks[index]) for index in range(9))
    assert int(frozen.excitatory_ticks.max()) == 4
    assert int(frozen.inhibitory_ticks.max()) == 1
    assert frozen.build_summary == bridge.BridgeBuildSummary(
        epochs=4,
        fit_sequence_count=36,
        generalization_sequence_count=18,
        matched_contrast_group_count=24,
        excitatory_head_update_events=35,
        inhibitory_head_update_events=8,
        committed_cell_ticks=5479,
    )
    with pytest.raises(ValueError, match="read-only"):
        frozen.excitatory_ticks[0, 0] = 1


def test_unseen_exact_spans_recombine_order_features_and_silence_other_heads(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalContrastGroup, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    curriculum, split, frozen = frozen_fixture
    lessons = _flatten(curriculum)
    fit_ids = set(split.fit_span_sha256s)
    fit = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in fit_ids
    )
    generalization = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256
        in set(split.generalization_span_sha256s)
    )
    fit_sequences = {lesson.prompt_input.canonical_tokens for lesson in fit}
    assert len(curriculum) == 24
    assert len(lessons) == 54
    assert len(fit) == 36
    assert len(generalization) == 18
    assert all(
        lesson.prompt_input.canonical_tokens not in fit_sequences
        for lesson in generalization
    )
    assert all(
        len(group.lessons) in (2, 3)
        and len(
            {
                tuple(sorted(lesson.prompt_input.canonical_tokens))
                for lesson in group.lessons
            }
        )
        == 1
        for group in curriculum
    )

    for lesson in generalization:
        inference = frozen.infer(lesson.prompt_input)
        assert inference.present_factors == lesson.present_factors
        assert len(inference.events) == 1
        assert inference.events[0].start_token == 7
        assert inference.events[0].stop_token == 11
        assert all(
            race.decision == bridge.STOP
            for race in inference.head_races
            if race.factor != lesson.present_factors[0]
        )
        assert all(count > 0 for count in inference.bank_active_cell_counts)
        assert all(step >= 0 for step in inference.bank_first_spike_steps)


def test_full_core_curriculum_does_not_regress_prior_layers(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalContrastGroup, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    _, _, full = frozen_fixture
    for max_layer in range(3):
        curriculum, _split = _self_posed_curriculum(max_layer=max_layer)
        assert all(
            full.infer(lesson.prompt_input).present_factors
            == lesson.present_factors
            for lesson in _flatten(curriculum)
        )


@pytest.mark.parametrize("max_layer", range(3))
def test_public_builder_rejects_curriculum_prefixes(max_layer: int) -> None:
    curriculum, split = _self_posed_curriculum(max_layer=max_layer)

    with pytest.raises(ValueError, match="exactly layers 0-3"):
        bridge.build_frozen_bridge(curriculum, split)


def test_validator_stops_on_nonmatched_multiset() -> None:
    curriculum, split = _self_posed_curriculum()
    original = curriculum[0]
    replacement = bridge.CompositionalContrastGroup(
        (
            _lesson("add rather than maximum", "add"),
            original.lessons[1],
        )
    )
    changed_curriculum, changed_split = _replace_group(
        curriculum,
        split,
        0,
        replacement,
    )

    with pytest.raises(ValueError, match="equal-width token multisets"):
        bridge._validate_curriculum(changed_curriculum, changed_split)


def test_validator_stops_on_unseen_heldout_factor_features() -> None:
    curriculum, split = _self_posed_curriculum()
    group_index = len(curriculum) - 1
    original = curriculum[group_index]
    replacement = bridge.CompositionalContrastGroup(
        tuple(
            _lesson(
                " ".join(
                    (
                        lesson.prompt_input.canonical_tokens[0],
                        "seldom",
                        *lesson.prompt_input.canonical_tokens[2:],
                    )
                ),
                lesson.present_factors[0],
            )
            for lesson in original.lessons
        )
    )
    changed_curriculum, changed_split = _replace_group(
        curriculum,
        split,
        group_index,
        replacement,
    )

    with pytest.raises(ValueError, match="heldout token/slot features"):
        bridge._validate_curriculum(changed_curriculum, changed_split)


def test_projection_geometry_stops_on_collisions_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curriculum, split = _self_posed_curriculum()

    monkeypatch.setattr(
        bridge,
        "_assembly_cells",
        lambda _bank_index, _key: tuple(range(bridge.ASSEMBLY_ACTIVE_CELL_COUNT)),
    )
    with pytest.raises(ValueError, match="distinct projector keys"):
        bridge._validate_curriculum(curriculum, split)

    overlapping_indices: dict[tuple[int, tuple[object, ...]], int] = {}

    def overlapping_cells(
        bank_index: int,
        key: tuple[object, ...],
    ) -> tuple[int, ...]:
        identity = (bank_index, key)
        index = overlapping_indices.setdefault(identity, len(overlapping_indices))
        return (*range(bridge.ASSEMBLY_ACTIVE_CELL_COUNT - 1), 32 + index)

    monkeypatch.setattr(bridge, "_assembly_cells", overlapping_cells)
    with pytest.raises(ValueError, match="factor cue assemblies"):
        bridge._validate_curriculum(curriculum, split)

    separated_indices: dict[int, dict[tuple[object, ...], int]] = {}

    def collapsed_order_cells(
        bank_index: int,
        key: tuple[object, ...],
    ) -> tuple[int, ...]:
        bank_indices = separated_indices.setdefault(bank_index, {})
        index = bank_indices.setdefault(key, len(bank_indices))
        if bank_index == bridge._TOKEN_BANK:
            start = index * bridge.ASSEMBLY_ACTIVE_CELL_COUNT
            return tuple(
                range(start, start + bridge.ASSEMBLY_ACTIVE_CELL_COUNT)
            )
        return (*range(bridge.ASSEMBLY_ACTIVE_CELL_COUNT - 1), 32 + index)

    monkeypatch.setattr(bridge, "_assembly_cells", collapsed_order_cells)
    with pytest.raises(ValueError, match="order features"):
        bridge._validate_curriculum(curriculum, split)


def test_authentication_hashes_are_noncausal_and_inference_is_byte_frozen(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalContrastGroup, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    curriculum, split, frozen = frozen_fixture
    lessons = _flatten(curriculum)
    prompt_input = next(
        lesson.prompt_input
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256
        in set(split.generalization_span_sha256s)
    )
    changed_authentication = dataclasses.replace(
        prompt_input,
        whole_span_sha256="0" * 64,
        audit_prompt_sha256="f" * 64,
    )
    checkpoint_before = frozen.checkpoint_bytes()
    checkpoint_sha_before = frozen.checkpoint_sha256

    first = frozen.infer(prompt_input)
    intervened = frozen.infer(changed_authentication)
    repeated = frozen.infer(prompt_input)

    assert intervened == first == repeated
    assert frozen.checkpoint_bytes() == checkpoint_before
    assert frozen.checkpoint_sha256 == checkpoint_sha_before

    relabeled_groups: list[bridge.CompositionalContrastGroup] = []
    relabeled_fit: list[str] = []
    relabeled_generalization: list[str] = []
    fit_ids = set(split.fit_span_sha256s)
    index = 0
    for group in reversed(curriculum):
        relabeled_lessons: list[bridge.CompositionalLesson] = []
        for lesson in reversed(group.lessons):
            index += 1
            relabeled_sha = f"{index:064x}"
            relabeled_input = dataclasses.replace(
                lesson.prompt_input,
                whole_span_sha256=relabeled_sha,
                audit_prompt_sha256=f"{index + len(lessons):064x}",
            )
            relabeled_lessons.append(
                dataclasses.replace(lesson, prompt_input=relabeled_input)
            )
            membership = (
                relabeled_fit
                if lesson.prompt_input.whole_span_sha256 in fit_ids
                else relabeled_generalization
            )
            membership.append(relabeled_sha)
        relabeled_groups.append(
            bridge.CompositionalContrastGroup(tuple(relabeled_lessons))
        )
    relabeled = bridge.build_frozen_bridge(
        tuple(relabeled_groups),
        bridge.CompositionalSplit(
            tuple(relabeled_fit),
            tuple(relabeled_generalization),
        ),
    )

    assert np.array_equal(relabeled.excitatory_ticks, frozen.excitatory_ticks)
    assert np.array_equal(relabeled.inhibitory_ticks, frozen.inhibitory_ticks)
    assert {
        lesson.prompt_input.canonical_tokens: relabeled.infer(
            lesson.prompt_input
        ).head_races
        for lesson in _flatten(tuple(relabeled_groups))
    } == {
        lesson.prompt_input.canonical_tokens: frozen.infer(
            lesson.prompt_input
        ).head_races
        for lesson in lessons
    }


def test_exposed_tick_arrays_are_detached_from_the_frozen_checkpoint(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalContrastGroup, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    _, _, frozen = frozen_fixture
    checkpoint_before = frozen.checkpoint_bytes()
    excitatory = frozen.excitatory_ticks
    inhibitory = frozen.inhibitory_ticks

    excitatory.setflags(write=True)
    inhibitory.setflags(write=True)
    excitatory[0, 0] = (int(excitatory[0, 0]) + 1) % (bridge.TICK_CAP + 1)
    inhibitory[1, 1] = (int(inhibitory[1, 1]) + 1) % (bridge.TICK_CAP + 1)

    assert frozen.checkpoint_bytes() == checkpoint_before
    assert not np.shares_memory(excitatory, frozen.excitatory_ticks)
    assert not np.shares_memory(inhibitory, frozen.inhibitory_ticks)


def test_constructor_copies_caller_owned_tick_arrays() -> None:
    excitatory = np.zeros(
        (bridge.HEAD_COUNT, bridge.INPUT_CELL_COUNT), dtype=bridge.TICK_DTYPE
    )
    inhibitory = np.zeros_like(excitatory)
    frozen = bridge.FrozenCompositionalBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata={"schema": "self-posed-constructor-alias-control"},
        build_summary=bridge.BridgeBuildSummary(
            epochs=0,
            fit_sequence_count=0,
            generalization_sequence_count=0,
            matched_contrast_group_count=0,
            excitatory_head_update_events=0,
            inhibitory_head_update_events=0,
            committed_cell_ticks=0,
        ),
    )
    checkpoint_before = frozen.checkpoint_bytes()

    excitatory[0, 0] = 1
    inhibitory[1, 1] = 1

    assert frozen.checkpoint_bytes() == checkpoint_before
    assert not np.shares_memory(excitatory, frozen.excitatory_ticks)
    assert not np.shares_memory(inhibitory, frozen.inhibitory_ticks)


def test_public_seam_has_no_benchmark_adapter_or_semantic_shortcut() -> None:
    assert tuple(inspect.signature(bridge.build_frozen_bridge).parameters) == (
        "curriculum",
        "split",
    )
    assert tuple(
        inspect.signature(bridge.FrozenCompositionalBridge.infer).parameters
    ) == ("self", "prompt_input")
    assert tuple(
        field.name for field in dataclasses.fields(bridge.CompositionalPromptInput)
    ) == (
        "start_token",
        "stop_token",
        "canonical_tokens",
        "whole_span_sha256",
        "audit_prompt_sha256",
    )
    assert tuple(
        field.name for field in dataclasses.fields(bridge.CompositionalContrastGroup)
    ) == ("lessons",)
    assert set(bridge.__all__).isdisjoint(
        {"_ANATOMY", "_project_prompt", "_make_prompt_input"}
    )
