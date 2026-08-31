from __future__ import annotations

import inspect
from itertools import pairwise

from experiments.g0 import common_event_cells_v1 as common_cells
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_content_addressed_segment_schema_v1 as content
from experiments.r3_pilot import causal_occurrence_executable_reducer_bridge_v1 as bridge
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import common_one_byte_relation_eventwise_v1 as eventwise
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

FINAL_CASES = (
    ([],),
    ([-20, -9, 0, 14],),
    ([-5, 7],),
    ([3, 3, 3],),
)
DEPENDENCY_CASES = (
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


def _groups(*indices: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(4 * index, 4 * index + 4)) for index in indices)


def _relabel(
    nodes: tuple[tuple[int, ...], ...], *, multiplier: int, offset: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted((multiplier * cell + offset) % 64 for cell in node))
        for node in nodes
    )


def _chain(
    nodes: tuple[tuple[int, ...], ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        occurrence.PhysicalOccurrenceEvent(
            occurrence.OPEN_CUE_CELL, nodes[0], nodes[0]
        ),
        *(
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, source, target
            )
            for source, target in pairwise(nodes)
        ),
    )


def _fanout(
    nodes: tuple[tuple[int, ...], ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        occurrence.PhysicalOccurrenceEvent(
            occurrence.OPEN_CUE_CELL, nodes[0], nodes[0]
        ),
        *(
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, nodes[0], target
            )
            for target in nodes[1:]
        ),
    )


def _closed(
    events: tuple[occurrence.PhysicalOccurrenceEvent, ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        *events,
        occurrence.PhysicalOccurrenceEvent(occurrence.EOS_CUE_CELL, (), ()),
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


def _count_values(
    hole: int,
    offset: int,
    operand_action: int,
    cases: tuple[tuple[list[int]], ...],
) -> tuple[int, ...]:
    if operand_action == ordered.REPEATED_TO_OPERAND_ZERO:
        return tuple(sum(hole < item + offset for item in values) for (values,) in cases)
    return tuple(sum(item + offset < hole for item in values) for (values,) in cases)


def _sum_values(cases: tuple[tuple[list[int]], ...]) -> tuple[int, ...]:
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
                _count_values(
                    -14,
                    -15,
                    ordered.REPEATED_TO_OPERAND_ZERO,
                    DEPENDENCY_CASES,
                ),
            ),
            literal_role.CausalLiteralEpisode(
                _spans(70, 10),
                _literals(-12, -6, (13, 15, 19)),
                SIGNATURE_ROLE,
                _count_values(
                    -12,
                    -15,
                    ordered.REPEATED_TO_OPERAND_ZERO,
                    DEPENDENCY_CASES,
                ),
            ),
        ),
        signature=signature,
        cases=DEPENDENCY_CASES,
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
                _count_values(
                    -9,
                    -15,
                    ordered.REPEATED_TO_OPERAND_ZERO,
                    DEPENDENCY_CASES,
                ),
            ),
            ordered.CausalOrderedOperandEpisode(
                _spans(220, 50),
                _literals(-7, 13, (55, 52, 58)),
                SIGNATURE_ROLE,
                _count_values(
                    -7,
                    -15,
                    ordered.REPEATED_TO_OPERAND_ONE,
                    DEPENDENCY_CASES,
                ),
            ),
        ),
        signature=signature,
        cases=DEPENDENCY_CASES,
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
                _count_values(
                    -5,
                    2,
                    ordered.REPEATED_TO_OPERAND_ZERO,
                    DEPENDENCY_CASES,
                ),
            ),
            residual.CausalResidualLiteralEpisode(
                _spans(350, 100),
                _literals(-1, -3, (105, 102, 108)),
                SIGNATURE_ROLE,
                _count_values(
                    -1,
                    -3,
                    ordered.REPEATED_TO_OPERAND_ONE,
                    DEPENDENCY_CASES,
                ),
            ),
        ),
        signature=signature,
        cases=DEPENDENCY_CASES,
        literals=(-15, -5, -3, -1, 0, 2),
        max_cost=8,
    )


def _segmented_checkpoint(
    signature: TypedSignature,
    episodes: tuple[append.CausalAppendEffectEpisode, ...],
) -> segmented.FrozenSegmentedAggregateV1:
    append_checkpoint = append.fit_causal_append_effect_reducer_v1(
        _banked_residual_checkpoint(signature),
        episodes[:2],
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-15, -5, -3, -1, 0, 2, 3, 4),
        max_cost=8,
    )
    relation_cells = common_cells.CommonEventCells().project(
        b"occurrence-executable-reducer-bridge/relation"
    ).trace_cells
    initial_segmenter = segmenter.start_predictive_segmenter_v1(
        eventwise.start_eventwise_common_one_byte_relation_v1(
            relation_v2.fit_common_one_byte_copy_relation_v2(())
        )
    )
    return segmented.fit_causal_segmented_aggregate_v1(
        append_checkpoint,
        initial_segmenter,
        episodes,
        probe_cases=TRAIN_PROBE,
        relation_cells=relation_cells,
    )


def _execute(source: str) -> tuple[int, ...]:
    namespace: dict[str, object] = {}
    exec(source, {}, namespace)
    aggregate = namespace["aggregate"]
    return tuple(aggregate(list(values)) for (values,) in FINAL_CASES)  # type: ignore[operator]


def test_occurrence_bundle_materializes_two_desired_free_executable_witnesses() -> None:
    signature = _signature()
    base_episodes = (
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
    dependency = _segmented_checkpoint(signature, base_episodes)
    reversed_content_episodes = (
        base_episodes[1],
        base_episodes[0],
        append.CausalAppendEffectEpisode(
            _spans(330, 290),
            _literals(-6, -2, (295, 292, 298)),
            SIGNATURE_ROLE,
            _outputs(-2, 3, -10),
        ),
    )
    content_checkpoint = content.fit_causal_content_addressed_segment_schema_v1(
        dependency,
        reversed_content_episodes,
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-6, -5, -3, -2, -1, 0, 2),
        max_cost=8,
    )
    assert content_checkpoint.training_summary.effect_relations == (
        append.PAYLOAD_EFFECT_RELATION,
        append.PREDICATE_EFFECT_RELATION,
        append.PAYLOAD_EFFECT_RELATION,
    )

    a3 = _chain(_groups(0, 1, 2))
    b3 = _fanout(_groups(3, 4, 5))
    a4 = _chain(_groups(6, 7, 8, 9))
    occurrence_checkpoint = occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1(
        a3 + b3 + _closed(a4)
    )
    lessons = (
        bridge.CausalOccurrenceExecutableLesson(
            _closed(_chain(_relabel(_groups(0, 1, 2), multiplier=3, offset=1))),
            _spans(410, 200),
            _literals(-7, 3, (202, 210, 205)),
            SIGNATURE_ROLE,
            COUNT_PROBE,
            _outputs(2, 3, 2),
        ),
        bridge.CausalOccurrenceExecutableLesson(
            _closed(_fanout(_relabel(_groups(3, 4, 5), multiplier=5, offset=9))),
            _spans(470, 230),
            _literals(-11, -6, (235, 232, 238)),
            SIGNATURE_ROLE,
            SUM_PROBE,
            _outputs(2, -8, 10),
        ),
        bridge.CausalOccurrenceExecutableLesson(
            _closed(_chain(_relabel(_groups(6, 7, 8, 9), multiplier=7, offset=3))),
            tuple(reversed(_spans(380, 260))),
            tuple(reversed(_literals(17, 31, (262, 270, 265)))),
            SIGNATURE_ROLE,
            RELABEL_PROBE,
            _outputs(1, 2, 1),
            (0x1E, 0xB4, 0x63, 0xC9, 0x78, 0x27, 0xD2, 0x8D),
        ),
    )
    checkpoint = bridge.fit_causal_occurrence_executable_reducer_bridge_v1(
        occurrence_checkpoint, content_checkpoint, lessons
    )
    frozen_root = checkpoint.checkpoint_sha256
    summary = checkpoint.training_summary

    assert summary.occurrence_schema_cell_ids == (
        occurrence.SCHEMA_CELL_OFFSET,
        occurrence.SCHEMA_CELL_OFFSET + 1,
        occurrence.SCHEMA_CELL_OFFSET,
    )
    assert summary.static_schema_cell_ids == (
        bridge.REBASED_CONTENT_SCHEMA_CELL_OFFSET + 1,
        bridge.REBASED_CONTENT_SCHEMA_CELL_OFFSET,
        bridge.REBASED_CONTENT_SCHEMA_CELL_OFFSET + 1,
    )
    assert summary.learned_cell_pairs == (
        (
            occurrence.SCHEMA_CELL_OFFSET,
            bridge.REBASED_CONTENT_SCHEMA_CELL_OFFSET + 1,
        ),
        (
            occurrence.SCHEMA_CELL_OFFSET + 1,
            bridge.REBASED_CONTENT_SCHEMA_CELL_OFFSET,
        ),
    )
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 3
    assert summary.maximum_excitatory_tick == summary.maximum_inhibitory_tick == 2
    assert summary.new_persistent_state_bytes == bridge.NEW_PERSISTENT_STATE_BYTES == 8
    assert summary.composed_persistent_state_bytes == 768
    assert summary.work.relay_race_cell_steps == 5_400
    assert summary.work.action_race_cell_steps == 180
    assert summary.work.bridge_physical_cell_steps == 5_580
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert bridge.NEW_CELL_COUNT == 0
    assert bridge.TOTAL_CELL_COUNT == 69_232

    prompt_spans = _spans(420, 310)
    prompt_literals = _literals(-13, 4, (312, 320, 315))
    c5_events = _closed(
        _chain(_relabel(_groups(0, 1, 2, 3, 4), multiplier=3, offset=1))
    )
    f5_events = _closed(
        _fanout(_relabel(_groups(9, 10, 11, 12, 13), multiplier=7, offset=3))
    )

    def build(
        events: tuple[occurrence.PhysicalOccurrenceEvent, ...],
        spans: tuple[dict[str, object], ...] = prompt_spans,
        literals: tuple[dict[str, object], ...] = prompt_literals,
        **gates: bool,
    ) -> bridge.OccurrenceExecutableWitnessCertificate:
        return checkpoint.build(
            events,
            spans,
            literals,
            SIGNATURE_ROLE,
            signature=signature,
            cases=FINAL_CASES,
            literals=(-13, 0, 4),
            max_cost=8,
            **gates,
        )

    c5 = build(c5_events)
    f4 = build(
        _closed(
            _fanout(_relabel(_groups(5, 6, 7, 8), multiplier=5, offset=9))
        ),
        _spans(450, 340),
        _literals(-11, -6, (342, 350, 345)),
    )
    same_prompt_flip = build(f5_events)
    relabeled = build(
        _closed(
            _chain(_relabel(_groups(6, 7, 8, 9, 10), multiplier=11, offset=5))
        ),
        tuple(reversed(prompt_spans)),
        tuple(reversed(prompt_literals)),
    )

    count_source = (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -13 < item_0 + 4))\n"
    )
    sum_source = "def aggregate(numbers):\n    return sum(numbers)\n"
    ready = (c5, f4, same_prompt_flip, relabeled)
    for result in ready:
        assert result.status == bridge.READY
        assert result.exact_proof_count == 1
        assert result.work.relay_race_cell_steps == 1_800
        assert result.work.action_race_cell_steps == 60
        assert result.work.bridge_physical_cell_steps == 1_860
        assert result.work.packed_frontier_build_count == 1
        assert result.work.authorized_slice_count == 1
        assert result.work.proof_materialization_count == 1
        assert bridge.require_occurrence_executable_witness(result) is result.witness
    assert c5.witness is not None and c5.witness.source == count_source
    assert relabeled.witness is not None and relabeled.witness.source == count_source
    assert f4.witness is not None and f4.witness.source == sum_source
    assert same_prompt_flip.witness is not None
    assert same_prompt_flip.witness.source == sum_source
    assert c5.static_schema_cell_id == relabeled.static_schema_cell_id
    assert f4.static_schema_cell_id == same_prompt_flip.static_schema_cell_id
    assert c5.static_schema_cell_id != f4.static_schema_cell_id
    assert c5.reducer_action_cell_id != f4.reducer_action_cell_id
    assert _execute(c5.witness.source) == (0, 3, 2, 3)
    assert _execute(relabeled.witness.source) == (0, 3, 2, 3)
    assert _execute(f4.witness.source) == (0, -15, 2, 9)
    assert _execute(same_prompt_flip.witness.source) == (0, -15, 2, 9)
    assert checkpoint.checkpoint_sha256 == frozen_root

    relation_off = build(c5_events, occurrence_relation_cell_gate=False)
    relay_off = build(c5_events, relay_cell_gate=False)
    action_off = build(c5_events, reducer_action_cell_gate=False)
    residual_off = build(c5_events, residual_cell_gate=False)
    assert relation_off.status == bridge.STOP_OCCURRENCE_SCHEMA
    assert relay_off.status == bridge.STOP_RELAY_TONIC
    assert action_off.status == bridge.STOP_ACTION_ZERO
    assert residual_off.status == bridge.STOP_RESIDUAL_BINDING
    for stopped in (relation_off, relay_off, action_off, residual_off):
        assert stopped.exact_proof_count == 0
        assert stopped.witness is None
        assert stopped.work.packed_frontier_build_count == 0

    hidden_build_inputs = {
        "probe_cases",
        "probe_outputs",
        "desired",
        "target_action",
        "family",
        "cohort",
        "schema_index",
    }
    assert hidden_build_inputs.isdisjoint(
        inspect.signature(
            bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1.build
        ).parameters
    )
    inference_source = inspect.getsource(bridge._build_witness)
    assert ".selected_actions" not in inference_source
    assert ".selected_action" not in inference_source
    assert ".state.schema_action" not in inference_source
    assert "probe_outputs" not in inference_source
    assert "desired" not in inspect.signature(
        bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1.build
    ).parameters
    assert all(
        value == 0
        for value in (
            bridge.DESIRED_READS_DURING_BUILD,
            bridge.PROBE_OUTPUT_READS_DURING_BUILD,
            bridge.DIRECT_OCCURRENCE_ACTION_READS,
            bridge.DIRECT_APPEND_ACTION_READS,
            bridge.SERIALIZED_SCHEMA_ACTION_READS,
            bridge.PUBLIC_TRAIN_INPUT_COUNT,
            bridge.CANONICAL_PANEL_SPEND_COUNT,
            bridge.DEV_INPUT_COUNT,
            bridge.HELDOUT_INPUT_COUNT,
            bridge.M0_INPUT_COUNT,
            bridge.C1_INPUT_COUNT,
            bridge.G0_INPUT_COUNT,
            bridge.PATH2_INPUT_COUNT,
            bridge.PROTECTED_INPUT_COUNT,
        )
    )
    assert "OccurrenceFactorStaticCertificate" not in inspect.getsource(bridge)
