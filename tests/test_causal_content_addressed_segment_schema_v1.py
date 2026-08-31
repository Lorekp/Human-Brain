from __future__ import annotations

import inspect

from experiments.g0 import common_event_cells_v1 as common_cells
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_content_addressed_segment_schema_v1 as learner
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
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
    literal_checkpoint = literal_role.fit_causal_relative_literal_role_v1(
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
        literal_checkpoint,
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


def _segmented_checkpoint(
    signature: TypedSignature,
    episodes: tuple[append.CausalAppendEffectEpisode, ...],
) -> segmented.FrozenSegmentedAggregateV1:
    relation_cells = common_cells.CommonEventCells().project(
        b"content-addressed-segment-schema/relation"
    ).trace_cells
    initial_segmenter = segmenter.start_predictive_segmenter_v1(
        eventwise.start_eventwise_common_one_byte_relation_v1(
            relation_v2.fit_common_one_byte_copy_relation_v2(())
        )
    )
    return segmented.fit_causal_segmented_aggregate_v1(
        _append_checkpoint(signature, episodes),
        initial_segmenter,
        episodes,
        probe_cases=TRAIN_PROBE,
        relation_cells=relation_cells,
    )


def _build_and_select(
    checkpoint: learner.FrozenCausalContentAddressedSegmentSchemaV1,
    signature: TypedSignature,
    spans: tuple[dict[str, object], ...],
    literals: tuple[dict[str, object], ...],
    probe_cases: tuple[tuple[list[int]], ...],
    probe_outputs: tuple[output_cells.TypedOutputEncoding, ...],
    desired: tuple[int, ...],
    *,
    compilation_literals: tuple[int, ...],
    event_order: tuple[int, ...] = segmented.CURRICULUM_VALUES,
) -> tuple[
    learner.ContentAddressedSchemaFrontierV1,
    learner.ContentAddressedSchemaSelection,
]:
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


def test_two_content_addressed_schemas_reuse_causal_laws() -> None:
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
    dependency = _segmented_checkpoint(signature, episodes)
    dependency_root = dependency.checkpoint_sha256
    checkpoint = learner.fit_causal_content_addressed_segment_schema_v1(
        dependency,
        episodes,
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-15, -5, -3, -1, 0, 2, 3, 4),
        max_cost=8,
    )
    summary = checkpoint.training_summary

    assert dependency.checkpoint_sha256 == dependency_root
    assert summary.persistent_schema_count == learner.SCHEMA_CAPACITY == 2
    assert summary.episodic_segment_count == 3
    assert summary.schema_assignments == (0, 1, 0)
    assert summary.effect_relations == (
        append.PREDICATE_EFFECT_RELATION,
        append.PAYLOAD_EFFECT_RELATION,
        append.PREDICATE_EFFECT_RELATION,
    )
    assert tuple(audit.disposition for audit in summary.fit_audits) == (
        "NOVEL",
        "NOVEL",
        "REUSED",
    )
    assert summary.prototype_excitatory_ticks == 2 * segmenter.ROUTES
    assert summary.prototype_inhibitory_ticks == 2 * segmenter.ROUTES * 3
    assert summary.action_excitatory_ticks == summary.action_inhibitory_ticks == 2
    assert summary.completeness_cell_steps == 3
    assert summary.effect_race_cell_steps == 3 * append.SEMANTIC_RACE_CELL_STEPS
    assert summary.effect_projection_cell_steps == 3 * 8 * 12_000
    assert summary.schema_race_cell_steps == 3 * learner.SCHEMA_RACE_CELL_STEPS
    assert summary.packed_authorization_count == 6
    assert summary.outcome_race_count == 3
    assert summary.direct_append_action_reads == learner.DIRECT_APPEND_ACTION_READS == 0
    assert summary.new_persistent_state_bytes == learner.NEW_PERSISTENT_STATE_BYTES == 274
    assert learner.NEW_CELL_COUNT == 5
    assert learner.TOTAL_CELL_COUNT == 69_213
    assert checkpoint.learned_state_shapes == (
        (2, 16, 4),
        (2, 16, 4),
        (2, 2),
        (2, 2),
        (2,),
        (2,),
    )

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
    relabel_frontier, relabel_selection = _build_and_select(
        checkpoint,
        signature,
        tuple(reversed(_spans(380, 260))),
        tuple(reversed(_literals(17, 31, (262, 270, 265)))),
        RELABEL_PROBE,
        _outputs(1, 2, 1),
        _count_desired(17, 31, ordered.REPEATED_TO_OPERAND_ZERO),
        compilation_literals=(0, 17, 31),
        event_order=(0x1E, 0xB4, 0x63, 0xC9, 0x78, 0x27, 0xD2, 0x8D),
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
        assert state.probe_packed_build_count == state.final_packed_build_count == 1
        assert state.desired_reads_during_build == 0
        assert state.direct_append_action_reads == 0
        assert state.schema_action == state.outcome_action
        assert state.router_actions == (segmented.OPEN_ACTION,) + (
            segmented.EXTEND_ACTION,
        ) * 7
        assert len(state.work.transition_sha256s) == 8
        assert len(state.work.router_race_sha256s) == 8
        assert state.work.completeness_cell_steps == 1
        assert state.work.effect_projection_cell_steps == 8 * 12_000
        assert state.work.segmenter_cell_steps == 8 * 37_200
        assert state.work.router_race_cell_steps == 8 * 60
        assert state.work.schema_race_cell_steps == learner.SCHEMA_RACE_CELL_STEPS
        assert state.work.action_race_cell_steps == learner.ACTION_RACE_CELL_STEPS
        assert state.work.packed_authorization_count == 2
        assert state.work.outcome_race_count == 1
        assert state.work.total_physical_cell_steps == 401_256
        assert state.work.input_projection_work > 0
        assert selection.status == plan_slice.SELECTED
        assert selection.delegated is not None
        assert selection.delegated.exact_proof_count == 1

    assert (*summary.schema_assignments, sum_frontier.state.schema_index) == (0, 1, 0, 1)
    assert count_frontier.state.schema_index == relabel_frontier.state.schema_index == 0
    assert sum_frontier.state.schema_index == flip_frontier.state.schema_index == 1
    assert count_frontier.state.schema_action != sum_frontier.state.schema_action
    assert count_frontier.state.effect_relation == append.PREDICATE_EFFECT_RELATION
    assert relabel_frontier.state.effect_relation == append.PREDICATE_EFFECT_RELATION
    assert sum_frontier.state.effect_relation == append.PAYLOAD_EFFECT_RELATION
    assert flip_frontier.state.effect_relation == append.PAYLOAD_EFFECT_RELATION

    reversed_episodes = (
        episodes[1],
        episodes[0],
        append.CausalAppendEffectEpisode(
            _spans(330, 290),
            _literals(-6, -2, (295, 292, 298)),
            SIGNATURE_ROLE,
            _outputs(-2, 3, -10),
        ),
    )
    reversed_checkpoint = learner.fit_causal_content_addressed_segment_schema_v1(
        dependency,
        reversed_episodes,
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-6, -5, -3, -2, -1, 0, 2),
        max_cost=8,
    )
    reversed_count = reversed_checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
    )
    assert reversed_checkpoint.training_summary.effect_relations == (
        append.PAYLOAD_EFFECT_RELATION,
        append.PREDICATE_EFFECT_RELATION,
        append.PAYLOAD_EFFECT_RELATION,
    )
    assert reversed_checkpoint.training_summary.schema_assignments == (0, 1, 0)
    assert reversed_count.status == plan_slice.FRONTIER_READY
    assert reversed_count.state.schema_index == 1
    assert reversed_count.state.schema_index != count_frontier.state.schema_index

    prototype_off = learner.freeze_prototype_off_control(checkpoint).build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
    )
    schema_cell_off = checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
        schema_cell_gate=False,
    )
    schema_comparator_off = checkpoint.build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
        schema_comparator_gate=False,
    )
    action_deranged = learner.freeze_deranged_action_control(checkpoint).build(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
    )

    for stopped in (prototype_off, schema_cell_off, schema_comparator_off):
        assert stopped.status == learner.STOP_SCHEMA_TONIC
        assert stopped.state.probe_packed_build_count == 0
        assert stopped.state.final_packed_build_count == 0
        assert stopped.select(_sum_desired()).delegated is None
    assert action_deranged.status == learner.STOP_ACTION_CONSEQUENCE
    assert action_deranged.state.probe_packed_build_count == 1
    assert action_deranged.state.final_packed_build_count == 0
    assert action_deranged.select(_sum_desired()).delegated is None

    assert "desired" not in inspect.signature(checkpoint.build).parameters
    source = inspect.getsource(learner)
    assert ".selected_action" not in source
    assert learner.DIRECT_APPEND_ACTION_READS == 0
    assert all(
        forbidden not in source
        for forbidden in (
            "FIT_SOURCE_BUNDLES",
            "protected_panel",
            "public_train_fit_prompt_record_source_event_producer_v1",
        )
    )
    assert learner.PUBLIC_TRAIN_INPUT_COUNT == learner.CANONICAL_PANEL_SPEND_COUNT == 0
    assert learner.DEV_INPUT_COUNT == learner.M0_INPUT_COUNT == 0
    assert learner.C1_INPUT_COUNT == learner.G0_INPUT_COUNT == 0
    assert learner.PATH2_INPUT_COUNT == learner.PROTECTED_INPUT_COUNT == 0
