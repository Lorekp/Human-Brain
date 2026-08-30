from __future__ import annotations

import dataclasses
import inspect
from collections import Counter

import numpy as np
import pytest

from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as bridge,
)


_CONNECTOR_PAIRS = (
    ("rather", "than"),
    ("instead", "of"),
    ("never", "over"),
    ("rather", "of"),
    ("instead", "than"),
    ("never", "of"),
)


def _self_posed_curriculum() -> tuple[
    tuple[bridge.CompositionalLesson, ...],
    bridge.CompositionalSplit,
]:
    lessons: list[bridge.CompositionalLesson] = []
    fit: list[str] = []
    generalization: list[str] = []
    for group, (first_connector, second_connector) in enumerate(
        _CONNECTOR_PAIRS
    ):
        for factor, left, right in (
            ("add", "add", "subtract"),
            ("subtract", "subtract", "add"),
        ):
            prompt_input = bridge._make_prompt_input(
                f"{left} {first_connector} {second_connector} {right}",
                start_token=7,
                audit_prompt=f"self-posed contrast {group}: {factor}",
            )
            lessons.append(
                bridge.CompositionalLesson(prompt_input, (factor,))
            )
            membership = fit if group < 4 else generalization
            membership.append(prompt_input.whole_span_sha256)
    return (
        tuple(lessons),
        bridge.CompositionalSplit(tuple(fit), tuple(generalization)),
    )


@pytest.fixture(scope="module")
def frozen_fixture() -> tuple[
    tuple[bridge.CompositionalLesson, ...],
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
        tuple[bridge.CompositionalLesson, ...],
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
    np.testing.assert_array_equal(frozen.excitatory_ticks[2:], 0)
    np.testing.assert_array_equal(frozen.inhibitory_ticks[2:], 0)
    assert frozen.build_summary == bridge.BridgeBuildSummary(
        epochs=4,
        fit_sequence_count=8,
        generalization_sequence_count=4,
        matched_contrast_group_count=6,
        excitatory_head_update_events=8,
        inhibitory_head_update_events=2,
        committed_cell_ticks=1269,
    )
    with pytest.raises(ValueError, match="read-only"):
        frozen.excitatory_ticks[0, 0] = 1


def test_unseen_exact_spans_recombine_order_features_and_silence_other_heads(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalLesson, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    curriculum, split, frozen = frozen_fixture
    fit_ids = set(split.fit_span_sha256s)
    fit = tuple(
        lesson
        for lesson in curriculum
        if lesson.prompt_input.whole_span_sha256 in fit_ids
    )
    generalization = tuple(
        lesson
        for lesson in curriculum
        if lesson.prompt_input.whole_span_sha256
        in set(split.generalization_span_sha256s)
    )
    fit_sequences = {lesson.prompt_input.canonical_tokens for lesson in fit}
    assert len(fit) == 8
    assert len(generalization) == 4
    assert all(
        lesson.prompt_input.canonical_tokens not in fit_sequences
        for lesson in generalization
    )

    for pair_start in (0, 2):
        add_lesson, subtract_lesson = generalization[pair_start : pair_start + 2]
        assert add_lesson.present_factors == ("add",)
        assert subtract_lesson.present_factors == ("subtract",)
        assert Counter(add_lesson.prompt_input.canonical_tokens) == Counter(
            subtract_lesson.prompt_input.canonical_tokens
        )
        assert (
            add_lesson.prompt_input.canonical_tokens
            != subtract_lesson.prompt_input.canonical_tokens
        )

    for lesson in generalization:
        inference = frozen.infer(lesson.prompt_input)
        assert inference.present_factors == lesson.present_factors
        assert len(inference.events) == 1
        assert inference.events[0].start_token == 7
        assert inference.events[0].stop_token == 11
        assert all(
            race.decision == bridge.STOP for race in inference.head_races[2:]
        )
        assert all(count > 0 for count in inference.bank_active_cell_counts)
        assert all(step >= 0 for step in inference.bank_first_spike_steps)


def test_authentication_hashes_are_noncausal_and_inference_is_byte_frozen(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalLesson, ...],
        bridge.CompositionalSplit,
        bridge.FrozenCompositionalBridge,
    ],
) -> None:
    curriculum, split, frozen = frozen_fixture
    prompt_input = next(
        lesson.prompt_input
        for lesson in curriculum
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

    relabeled_lessons: list[bridge.CompositionalLesson] = []
    relabeled_fit: list[str] = []
    relabeled_generalization: list[str] = []
    fit_ids = set(split.fit_span_sha256s)
    for index, lesson in enumerate(reversed(curriculum), start=1):
        relabeled_sha = f"{index:064x}"
        relabeled_input = dataclasses.replace(
            lesson.prompt_input,
            whole_span_sha256=relabeled_sha,
            audit_prompt_sha256=f"{index + len(curriculum):064x}",
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
    relabeled = bridge.build_frozen_bridge(
        tuple(relabeled_lessons),
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
        ).present_factors
        for lesson in relabeled_lessons
    } == {
        lesson.prompt_input.canonical_tokens: frozen.infer(
            lesson.prompt_input
        ).present_factors
        for lesson in curriculum
    }


def test_exposed_tick_arrays_are_detached_from_the_frozen_checkpoint(
    frozen_fixture: tuple[
        tuple[bridge.CompositionalLesson, ...],
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
    assert set(bridge.__all__).isdisjoint(
        {"_ANATOMY", "_project_prompt", "_make_prompt_input"}
    )
