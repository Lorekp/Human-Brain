from __future__ import annotations

import inspect

from experiments.g0 import common_event_cells_v1 as common_cells
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as learner
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import common_one_byte_relation_eventwise_v1 as eventwise
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

FINAL_CASES = (
    ([],),
    ([-20, -10, 0, 10],),
    ([-5, 1, 9],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
TRAIN_PROBE = (([-4, 2],), ([-4, 2, 5],), ([-4, 2, -8],))
COUNT_PROBE = (([3, 0],), ([3, 0, 6],), ([3, 0, -12],))
SUM_PROBE = (([2, 0],), ([2, 0, -10],), ([2, 0, 8],))
RELABEL_PROBE = (([0, -20],), ([0, -20, 5],), ([0, -20, -30],))
SIGNATURE_ROLE = {
    "parameters": [
        {
            "role": "ARG0",
            "type": {
                "constructor": "List",
                "arguments": [{"constructor": "Int", "arguments": []}],
            },
        }
    ],
    "result": {"constructor": "Int", "arguments": []},
    "role_binding": "reverse-declaration-depth",
    "entrypoint_spelling_present": False,
    "parameter_spelling_present": False,
}


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", ProgramType.List(integer)),),
        integer,
    )


def _spans(seed: int, shift: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "start_token": shift,
            "stop_token": shift + 8,
            "first_spike_step": 25,
            "assembly_cells": tuple(range(seed, seed + 12)),
            "spike_cells": tuple(range(seed, seed + 12)),
        },
        {
            "start_token": shift + 2,
            "stop_token": shift + 12,
            "first_spike_step": 26,
            "assembly_cells": tuple(range(seed + 16, seed + 28)),
            "spike_cells": tuple(range(seed + 16, seed + 28)),
        },
    )


def _cells(value: int) -> tuple[int, ...]:
    return output_cells.TypedOutputCellBank.encode(
        value, declared_type=output_cells.OutputType.INT
    ).active_global_cell_indices()


def _outputs(*values: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value, declared_type=output_cells.OutputType.INT
        )
        for value in values
    )


def _literals(
    repeated: int,
    residual_value: int,
    positions: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(
            positions, (repeated, residual_value, repeated), strict=True
        )
    )


def _count_desired(
    hole: int,
    offset: int,
    operand_action: int,
    cases: tuple[tuple[list[int]], ...] = FINAL_CASES,
) -> tuple[int, ...]:
    if operand_action == ordered.REPEATED_TO_OPERAND_ZERO:
        return tuple(sum(hole < item + offset for item in values) for (values,) in cases)
    return tuple(sum(item + offset < hole for item in values) for (values,) in cases)


def _sum_desired(
    cases: tuple[tuple[list[int]], ...] = FINAL_CASES,
) -> tuple[int, ...]:
    return tuple(sum(values) for (values,) in cases)


def _banked_residual_checkpoint(
    signature: TypedSignature,
) -> residual.FrozenCausalResidualLiteralRouteV1:
    literals_checkpoint = literal_role.fit_causal_relative_literal_role_v1(
        (
            literal_role.CausalLiteralEpisode(
                _spans(10, 0),
                _literals(-14, -8, (2, 4, 7)),
                SIGNATURE_ROLE,
                _count_desired(-14, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            literal_role.CausalLiteralEpisode(
                _spans(70, 10),
                _literals(-12, -6, (13, 15, 19)),
                SIGNATURE_ROLE,
                _count_desired(-12, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
        ),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-15, -14, -12, -8, -6, 0),
        max_cost=8,
    )
    ordered_checkpoint = ordered.fit_causal_relative_ordered_operand_v1(
        literals_checkpoint,
        (
            ordered.CausalOrderedOperandEpisode(
                _spans(150, 30),
                _literals(-9, 5, (32, 39, 34)),
                SIGNATURE_ROLE,
                _count_desired(-9, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            ordered.CausalOrderedOperandEpisode(
                _spans(220, 50),
                _literals(-7, 13, (55, 52, 58)),
                SIGNATURE_ROLE,
                _count_desired(-7, -15, ordered.REPEATED_TO_OPERAND_ONE),
            ),
        ),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-15, -9, -7, 0),
        max_cost=8,
    )
    return residual.fit_causal_residual_literal_route_v1(
        ordered_checkpoint,
        (
            residual.CausalResidualLiteralEpisode(
                _spans(290, 80),
                _literals(-5, 2, (82, 90, 85)),
                SIGNATURE_ROLE,
                _count_desired(-5, 2, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            residual.CausalResidualLiteralEpisode(
                _spans(350, 100),
                _literals(-1, -3, (105, 102, 108)),
                SIGNATURE_ROLE,
                _count_desired(-1, -3, ordered.REPEATED_TO_OPERAND_ONE),
            ),
        ),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-15, -5, -3, -1, 0, 2),
        max_cost=8,
    )


def _append_checkpoint(
    signature: TypedSignature,
    episodes: tuple[append.CausalAppendEffectEpisode, ...],
) -> append.FrozenCausalAppendEffectReducerV1:
    return append.fit_causal_append_effect_reducer_v1(
        _banked_residual_checkpoint(signature),
        episodes[:2],
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-15, -5, -3, -1, 0, 2, 3, 4),
        max_cost=8,
    )


def _build_and_select(
    checkpoint: learner.FrozenSegmentedAggregateV1,
    signature: TypedSignature,
    spans: tuple[dict[str, object], ...],
    literals: tuple[dict[str, object], ...],
    probe_cases: tuple[tuple[list[int]], ...],
    probe_outputs: tuple[output_cells.TypedOutputEncoding, ...],
    desired: tuple[int, ...],
    *,
    compilation_literals: tuple[int, ...],
    event_order: tuple[int, ...] = learner.CURRICULUM_VALUES,
) -> tuple[learner.SegmentedAggregateFrontierV1, learner.SegmentedAggregateSelection]:
    frontier = checkpoint.build(
        spans,
        literals,
        SIGNATURE_ROLE,
        probe_cases,
        probe_outputs,
        signature=signature,
        cases=FINAL_CASES,
        literals=compilation_literals,
        max_cost=8,
        event_order=event_order,
    )
    return frontier, frontier.select(desired)


def test_segmented_anonymous_aggregate_composes_banked_relations() -> None:
    signature = _signature()
    episodes = (
        append.CausalAppendEffectEpisode(
            _spans(30, 130),
            _literals(-5, 2, (132, 140, 135)),
            SIGNATURE_ROLE,
            _outputs(2, 3, 2),
        ),
        append.CausalAppendEffectEpisode(
            _spans(90, 150),
            _literals(-1, -3, (155, 152, 158)),
            SIGNATURE_ROLE,
            _outputs(-2, 3, -10),
        ),
        append.CausalAppendEffectEpisode(
            _spans(150, 170),
            _literals(4, 3, (175, 172, 178)),
            SIGNATURE_ROLE,
            _outputs(1, 1, 2),
        ),
    )
    append_checkpoint = _append_checkpoint(signature, episodes)
    append_root = append_checkpoint.checkpoint_sha256
    relation_cells = common_cells.CommonEventCells().project(
        b"segmented-anonymous-aggregate/relation"
    ).trace_cells
    initial_segmenter = segmenter.start_predictive_segmenter_v1(
        eventwise.start_eventwise_common_one_byte_relation_v1(
            relation_v2.fit_common_one_byte_copy_relation_v2(())
        )
    )
    initial_segmenter_root = initial_segmenter.checkpoint_sha256

    checkpoint = learner.fit_causal_segmented_aggregate_v1(
        append_checkpoint,
        initial_segmenter,
        episodes,
        probe_cases=TRAIN_PROBE,
        relation_cells=relation_cells,
    )
    summary = checkpoint.training_summary
    assert append_checkpoint.checkpoint_sha256 == append_root
    assert initial_segmenter.checkpoint_sha256 == initial_segmenter_root
    assert summary.episode_count == 3
    assert summary.effect_relations == (
        append.PREDICATE_EFFECT_RELATION,
        append.PAYLOAD_EFFECT_RELATION,
        append.PREDICATE_EFFECT_RELATION,
    )
    assert summary.append_binding_cell_steps == 3 * 4776
    assert summary.effect_projection_cell_steps == 3 * 8 * 12_000
    assert summary.segmenter_cell_steps == 3 * 8 * 37_200
    assert summary.router_outcome_race_count == 2
    assert summary.router_outcome_race_cell_steps == 2 * 60
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 2
    assert summary.trainable_state_bytes == learner.TRAINABLE_STATE_BYTES == 8
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert {
        (audit.decision_relation, audit.pass_action)
        for audit in summary.router_audits
    } == {
        (learner.NEW_RELATION, learner.OPEN_ACTION),
        (learner.CONTINUE_RELATION, learner.EXTEND_ACTION),
    }

    count_spans = _spans(410, 200)
    count_literals = _literals(-7, 3, (202, 210, 205))
    count_frontier, count_selection = _build_and_select(
        checkpoint,
        signature,
        count_spans,
        count_literals,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        _count_desired(-7, 3, ordered.REPEATED_TO_OPERAND_ZERO),
        compilation_literals=(-7, 0, 3),
    )
    sum_frontier, sum_selection = _build_and_select(
        checkpoint,
        signature,
        _spans(470, 230),
        _literals(-11, -6, (235, 232, 238)),
        SUM_PROBE,
        _outputs(2, -8, 10),
        _sum_desired(),
        compilation_literals=(-11, -6, 0),
    )
    relabel_order = (
        0x1E,
        0xB4,
        0x63,
        0xC9,
        0x78,
        0x27,
        0xD2,
        0x8D,
    )
    relabel_frontier, relabel_selection = _build_and_select(
        checkpoint,
        signature,
        tuple(reversed(_spans(380, 260))),
        tuple(reversed(_literals(17, 31, (262, 270, 265)))),
        RELABEL_PROBE,
        _outputs(1, 2, 1),
        _count_desired(17, 31, ordered.REPEATED_TO_OPERAND_ZERO),
        compilation_literals=(0, 17, 31),
        event_order=relabel_order,
    )
    flip_frontier, flip_selection = _build_and_select(
        checkpoint,
        signature,
        count_spans,
        count_literals,
        COUNT_PROBE,
        _outputs(3, 9, -9),
        _sum_desired(),
        compilation_literals=(-7, 0, 3),
    )

    successful = (
        (count_frontier, count_selection),
        (sum_frontier, sum_selection),
        (relabel_frontier, relabel_selection),
        (flip_frontier, flip_selection),
    )
    for frontier, selection in successful:
        state = frontier.state
        assert frontier.status == plan_slice.FRONTIER_READY
        assert state.packed_build_count == 1
        assert state.desired_reads_during_build == 0
        assert state.router_actions == (learner.OPEN_ACTION,) + (
            learner.EXTEND_ACTION,
        ) * 7
        assert len(state.work.transition_sha256s) == 8
        assert len(state.work.router_race_sha256s) == 8
        assert state.work.append_binding_cell_steps == 4776
        assert state.work.effect_projection_cell_steps == 8 * 12_000
        assert state.work.segmenter_cell_steps == 8 * 37_200
        assert state.work.router_race_cell_steps == 8 * 60
        assert state.work.total_physical_cell_steps == 398_856
        assert state.work.input_projection_work > 0
        assert selection.status == plan_slice.SELECTED
        assert selection.delegated is not None
        assert selection.delegated.exact_proof_count == 1

    assert count_selection.delegated is not None
    assert sum_selection.delegated is not None
    assert relabel_selection.delegated is not None
    assert flip_selection.delegated is not None
    assert count_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -7 < item_0 + 3))\n"
    )
    assert sum_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum(numbers)\n"
    )
    assert relabel_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if 17 < item_0 + 31))\n"
    )
    assert flip_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum(numbers)\n"
    )
    assert count_frontier.state.effect_relation == append.PREDICATE_EFFECT_RELATION
    assert relabel_frontier.state.effect_relation == append.PREDICATE_EFFECT_RELATION
    assert sum_frontier.state.effect_relation == append.PAYLOAD_EFFECT_RELATION
    assert flip_frontier.state.effect_relation == append.PAYLOAD_EFFECT_RELATION
    assert count_frontier.state.selected_action == relabel_frontier.state.selected_action
    assert sum_frontier.state.selected_action == flip_frontier.state.selected_action
    assert count_frontier.state.selected_action != flip_frontier.state.selected_action

    cue_off = checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
        cue_gate=False,
    )
    router_off = checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
        router_cell_gate=False,
    )
    append_effect_off = checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
        effect_gate=False,
    )
    assert cue_off.status == "ABSTAIN_SEGMENT_AMBIGUITY"
    assert cue_off.state.packed_build_count == 0
    assert router_off.status == learner.STOP_ROUTER_ZERO
    assert router_off.state.packed_build_count == 0
    assert append_effect_off.status == append.STOP_EFFECT_GATE
    assert append_effect_off.state.packed_build_count == 0

    assert "desired" not in inspect.signature(checkpoint.build).parameters
    source = inspect.getsource(learner)
    assert all(
        forbidden not in source
        for forbidden in (
            "public_train_fit_prompt_record_source_event_producer_v1",
            "FIT_SOURCE_BUNDLES",
            "protected_panel",
        )
    )
    assert learner.TOTAL_CELL_COUNT == append.TOTAL_CELL_COUNT + segmenter.TOTAL_CELL_COUNT + 2
    assert learner.TOTAL_CELL_COUNT == 69_208
    assert learner.PUBLIC_TRAIN_INPUT_COUNT == learner.CANONICAL_PANEL_SPEND_COUNT == 0
    assert learner.DEV_INPUT_COUNT == learner.M0_INPUT_COUNT == 0
    assert learner.C1_INPUT_COUNT == learner.G0_INPUT_COUNT == 0
    assert learner.PATH2_INPUT_COUNT == learner.PROTECTED_INPUT_COUNT == 0
