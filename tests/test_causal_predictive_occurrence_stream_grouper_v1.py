from __future__ import annotations

import dataclasses
from itertools import pairwise

from experiments.g0 import common_event_cells_v1 as common_cells
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_predictive_occurrence_stream_grouper_v1 as grouper
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
    ([-20, -10, 0, 10],),
    ([-5, 1, 9],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
TRAIN_PROBE = (([-4, 2],), ([-4, 2, 5],), ([-4, 2, -8],))
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
) -> tuple[int, ...]:
    if operand_action == ordered.REPEATED_TO_OPERAND_ZERO:
        return tuple(
            sum(hole < item + offset for item in values) for (values,) in FINAL_CASES
        )
    return tuple(
        sum(item + offset < hole for item in values) for (values,) in FINAL_CASES
    )


def _residual_checkpoint(
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


def _segmented_checkpoint() -> segmented.FrozenSegmentedAggregateV1:
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
    append_checkpoint = append.fit_causal_append_effect_reducer_v1(
        _residual_checkpoint(signature),
        episodes[:2],
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-15, -5, -3, -1, 0, 2, 3, 4),
        max_cost=8,
    )
    relation_cells = common_cells.CommonEventCells().project(
        b"predictive-occurrence-grouper/relation"
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


def _groups(*indices: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(4 * index, 4 * index + 4)) for index in indices)


def _relabel(
    nodes: tuple[tuple[int, ...], ...], *, multiplier: int, offset: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted((multiplier * cell + offset) % 64 for cell in node))
        for node in nodes
    )


def _physical_chain(
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


def _physical_fanout(
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


def _occurrence_checkpoint() -> occurrence.FrozenAnonymousOccurrenceBundleSchemaV1:
    a3 = _physical_chain(_groups(0, 1, 2))
    b3 = _physical_fanout(_groups(3, 4, 5))
    a4 = _physical_chain(_groups(6, 7, 8, 9))
    return occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1(
        a3 + b3 + _closed(a4)
    )


def _raw_chain(
    nodes: tuple[tuple[int, ...], ...], interval_cell_id: int
) -> tuple[grouper.RawNeutralOccurrenceEvent, ...]:
    return tuple(
        grouper.RawNeutralOccurrenceEvent(source, target, interval_cell_id)
        for source, target in pairwise(nodes)
    )


def _raw_fanout(
    nodes: tuple[tuple[int, ...], ...], interval_cell_id: int
) -> tuple[grouper.RawNeutralOccurrenceEvent, ...]:
    return tuple(
        grouper.RawNeutralOccurrenceEvent(nodes[0], target, interval_cell_id)
        for target in nodes[1:]
    )


def test_raw_neutral_stream_grouping_preserves_the_predictive_falsification() -> None:
    segmented_checkpoint = _segmented_checkpoint()
    occurrence_checkpoint = _occurrence_checkpoint()
    cue_excitatory = segmented_checkpoint._segmenter.cue_excitatory
    cue_inhibitory = segmented_checkpoint._segmenter.cue_inhibitory
    continue_cues = tuple(
        index
        for index in range(segmenter.INTERVAL_CELL_COUNT)
        if (
            cue_excitatory[index][segmenter.CONTINUE_DECISION]
            - cue_inhibitory[index][segmenter.CONTINUE_DECISION]
        )
        > (
            cue_excitatory[index][segmenter.NEW_DECISION]
            - cue_inhibitory[index][segmenter.NEW_DECISION]
        )
    )
    assert len(continue_cues) == 1
    neutral_interval = segmenter.INTERVAL_CELL_IDS[continue_cues[0]]

    assert tuple(
        field.name for field in dataclasses.fields(grouper.RawNeutralOccurrenceEvent)
    ) == ("source_cells", "target_cells", "interval_cell_id")
    assert sorted(cue_excitatory) == [(0, 2), (4, 0)]
    assert sorted(cue_inhibitory) == [(0, 4), (2, 0)]
    assert grouper.NEW_CELL_COUNT == grouper.TRAINABLE_STATE_BYTES == 0
    assert grouper.RUNTIME_STATE_BYTES == 60
    assert grouper.TOTAL_CELL_COUNT == occurrence.TOTAL_CELL_COUNT == 69_227
    assert grouper.BOUNDARY_BOOLEAN_INPUT_COUNT == 0
    assert grouper.OPEN_EXTEND_EOS_INPUT_COUNT == 0
    assert grouper.HOST_GROUP_LENGTH_INPUT_COUNT == 0

    chain_nodes = _relabel(_groups(0, 1, 2, 3, 4), multiplier=3, offset=1)
    fanout_nodes = _relabel(_groups(6, 7, 8, 9, 10), multiplier=5, offset=7)
    fanout_nodes = (
        fanout_nodes[0],
        fanout_nodes[3],
        fanout_nodes[1],
        fanout_nodes[4],
        fanout_nodes[2],
    )
    chain_raw = _raw_chain(chain_nodes, neutral_interval)
    fanout_raw = _raw_fanout(fanout_nodes, neutral_interval)
    assert {event.interval_cell_id for event in chain_raw + fanout_raw} == {
        neutral_interval
    }

    chain = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint, occurrence_checkpoint, chain_raw
    )
    fanout = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint, occurrence_checkpoint, fanout_raw
    )
    chain_binding = occurrence.require_occurrence_graph_schema(chain.certificates[0])
    fanout_binding = occurrence.require_occurrence_graph_schema(
        fanout.certificates[0]
    )
    assert chain.status == fanout.status == grouper.READY
    assert chain_binding.schema_index == 0
    assert fanout_binding.schema_index == 1
    assert chain.candidate_certificate is chain.certificates[0]
    assert fanout.candidate_certificate is fanout.certificates[0]
    assert chain.work.trigger_replay_count == fanout.work.trigger_replay_count == 1
    assert chain.work.accepted_raw_event_count == len(chain_raw)
    assert fanout.work.accepted_raw_event_count == len(fanout_raw)

    first = _raw_chain(_relabel(_groups(0, 1, 2), multiplier=7, offset=3), neutral_interval)
    second = _raw_fanout(
        _relabel(_groups(3, 4, 5, 6), multiplier=9, offset=5),
        neutral_interval,
    )
    shifted_first = _raw_chain(
        _relabel(_groups(7, 8, 9, 10, 11), multiplier=11, offset=1),
        neutral_interval,
    )
    shifted_second = _raw_fanout(
        _relabel(_groups(12, 13, 14), multiplier=13, offset=3),
        neutral_interval,
    )
    continuous_a = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint, occurrence_checkpoint, first + second
    )
    continuous_b = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint, occurrence_checkpoint, shifted_first + shifted_second
    )
    assert continuous_a.status == continuous_b.status == (
        grouper.STOP_SEGMENT_AMBIGUITY
    )
    assert continuous_a.stop_reason == continuous_b.stop_reason == (
        "banked predictive race abstained with COLLISION"
    )
    assert continuous_a.stop_event_index == len(first)
    assert continuous_b.stop_event_index == len(shifted_first)
    assert continuous_a.accepted_raw_event_count == len(first)
    assert continuous_b.accepted_raw_event_count == len(shifted_first)
    assert continuous_a.candidate_certificate is not None
    assert continuous_b.candidate_certificate is not None
    occurrence.require_occurrence_graph_schema(continuous_a.candidate_certificate)
    occurrence.require_occurrence_graph_schema(continuous_b.candidate_certificate)
    assert not continuous_a.certificates and not continuous_b.certificates
    assert {
        event.interval_cell_id
        for event in first + second + shifted_first + shifted_second
    } == {neutral_interval}

    segment_off = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        chain_raw,
        segmenter_cue_gate=False,
    )
    recurrence_off = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        chain_raw,
        recurrence_cell_gate=False,
    )
    router_off = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        chain_raw,
        router_cell_gate=False,
    )
    completion_off = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        chain_raw,
        completion_cell_gate=False,
    )
    replay_off = grouper.group_causal_predictive_occurrence_stream_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        chain_raw,
        replay_cell_gate=False,
    )
    assert segment_off.status == recurrence_off.status == (
        grouper.STOP_SEGMENT_AMBIGUITY
    )
    assert segment_off.accepted_raw_event_count == 1
    assert recurrence_off.accepted_raw_event_count == 2
    assert router_off.status == grouper.STOP_ROUTER
    assert router_off.accepted_raw_event_count == 0
    assert completion_off.status == grouper.STOP_OCCURRENCE
    assert completion_off.candidate_certificate is not None
    assert completion_off.candidate_certificate.status == occurrence.STOP_INCOMPLETE
    assert replay_off.status == grouper.STOP_REPLAY
    assert replay_off.accepted_raw_event_count == 0
    assert replay_off.work.trigger_replay_count == 1
    assert all(
        value == 0
        for value in (
            grouper.PUBLIC_TRAIN_INPUT_COUNT,
            grouper.DEV_INPUT_COUNT,
            grouper.M0_INPUT_COUNT,
            grouper.C1_INPUT_COUNT,
            grouper.G0_INPUT_COUNT,
            grouper.PATH2_INPUT_COUNT,
            grouper.PROTECTED_INPUT_COUNT,
        )
    )
