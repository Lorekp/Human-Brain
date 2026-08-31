from __future__ import annotations

from inspect import signature
from itertools import pairwise

from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_learned_occurrence_boundary_gate_v1 as learner
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter

from tests import test_causal_predictive_occurrence_stream_grouper_v1 as banked_fixture


def _opaque_pool(
    *, multiplier: int, offset: int
) -> tuple[tuple[int, ...], ...]:
    base = banked_fixture._groups(*range(16))
    relabeled = banked_fixture._relabel(
        base, multiplier=multiplier, offset=offset
    )
    order = (7, 1, 12, 4, 15, 2, 10, 6, 13, 0, 9, 3, 14, 5, 11, 8)
    return tuple(relabeled[index] for index in order)


def _training_transports(
    interval_cell_id: int,
) -> tuple[tuple[learner.RawNeutralOccurrenceEvent, ...], ...]:
    nodes = _opaque_pool(multiplier=3, offset=1)
    return (
        banked_fixture._raw_chain(nodes[0:3], interval_cell_id),
        banked_fixture._raw_fanout(nodes[3:6], interval_cell_id),
        banked_fixture._raw_chain(nodes[6:10], interval_cell_id),
        banked_fixture._raw_fanout(
            (nodes[10], nodes[13], nodes[11], nodes[12]), interval_cell_id
        ),
        banked_fixture._raw_chain(
            (nodes[14], nodes[2], nodes[5], nodes[8], nodes[15]),
            interval_cell_id,
        ),
    )


def _fresh_streams(
    interval_cell_id: int,
) -> tuple[
    tuple[learner.RawNeutralOccurrenceEvent, ...],
    tuple[learner.RawNeutralOccurrenceEvent, ...],
]:
    first = _opaque_pool(multiplier=5, offset=7)
    first_chain = banked_fixture._raw_chain(first[0:5], interval_cell_id)
    first_fanout = banked_fixture._raw_fanout(
        (first[5], first[7], first[6]), interval_cell_id
    )
    second_chain = banked_fixture._raw_chain(first[8:14], interval_cell_id)

    second = _opaque_pool(multiplier=11, offset=3)
    second_fanout = banked_fixture._raw_fanout(
        (second[0], second[3], second[1], second[2]), interval_cell_id
    )
    second_tail = banked_fixture._raw_chain(second[4:9], interval_cell_id)
    return (
        first_chain + first_fanout + second_chain,
        second_fanout + second_tail,
    )


def test_physical_completeness_recurrence_gate_learns_continuous_boundaries() -> None:
    segmented_checkpoint = banked_fixture._segmented_checkpoint()
    occurrence_checkpoint = banked_fixture._occurrence_checkpoint()
    neutral_interval = segmenter.INTERVAL_CELL_IDS[0]
    transports = _training_transports(neutral_interval)
    assert {
        event.interval_cell_id for bundle in transports for event in bundle
    } == {neutral_interval}
    for prior, following in pairwise(transports):
        assert following[0].source_cells not in (
            prior[0].source_cells,
            prior[-1].target_cells,
        )

    checkpoint = learner.fit_causal_learned_occurrence_boundary_gate_v1(
        segmented_checkpoint,
        occurrence_checkpoint,
        transports,
    )
    summary = checkpoint.training_summary
    assert summary.transport_count == 5
    assert summary.feature_exposures == (5, 0, 4, 4)
    assert summary.action_intervention_count == 26
    assert summary.consequence_race_count == 13
    assert summary.excitatory_update_ticks == 12
    assert summary.inhibitory_update_ticks == 12
    assert summary.maximum_excitatory_tick == 4
    assert summary.maximum_inhibitory_tick == 4
    assert summary.trainable_state_bytes == 16
    assert checkpoint.excitatory == (
        (4, 0),
        (0, 0),
        (4, 0),
        (0, 4),
    )
    assert checkpoint.inhibitory == (
        (0, 4),
        (0, 0),
        (0, 4),
        (4, 0),
    )
    assert learner.NEW_CELL_COUNT == 4
    assert learner.TOTAL_CELL_COUNT == 69_231
    assert learner.TRAINABLE_STATE_BYTES == learner.PERSISTENT_STATE_BYTES == 16
    assert learner.RUNTIME_STATE_BYTES == 60
    assert learner.INTERVAL_GATE_INPUT_COUNT == 0
    assert learner.BOUNDARY_BOOLEAN_INPUT_COUNT == 0
    assert learner.OPEN_EXTEND_EOS_INPUT_COUNT == 0
    assert learner.HOST_GROUP_LENGTH_PLASTICITY_INPUT_COUNT == 0
    assert learner.SERIALIZED_ACTION_INPUT_COUNT == 0
    assert {
        "boundary",
        "boundary_boolean",
        "length",
        "desired",
        "action",
    }.isdisjoint(
        signature(
            learner.fit_causal_learned_occurrence_boundary_gate_v1
        ).parameters
    )

    stream_a, stream_b = _fresh_streams(neutral_interval)
    assert {event.interval_cell_id for event in stream_a + stream_b} == {
        neutral_interval
    }
    outcome_a = checkpoint.group(stream_a)
    outcome_b = checkpoint.group(stream_b)
    assert outcome_a.status == outcome_b.status == learner.READY
    assert outcome_a.work.boundary_event_indices == (4, 6)
    assert outcome_b.work.boundary_event_indices == (3,)
    assert outcome_a.accepted_raw_event_count == len(stream_a)
    assert outcome_b.accepted_raw_event_count == len(stream_b)
    assert outcome_a.work.trigger_replay_count == 3
    assert outcome_b.work.trigger_replay_count == 2
    assert tuple(
        occurrence.require_occurrence_graph_schema(value).schema_index
        for value in outcome_a.certificates
    ) == (0, 1, 0)
    assert tuple(
        occurrence.require_occurrence_graph_schema(value).schema_index
        for value in outcome_b.certificates
    ) == (1, 0)
    assert outcome_a.candidate_certificate is outcome_a.certificates[-1]
    assert outcome_b.candidate_certificate is outcome_b.certificates[-1]

    completion_off = checkpoint.group(stream_a, completion_cell_gate=False)
    recurrence_off = checkpoint.group(stream_a, recurrence_cell_gate=False)
    gate_off = checkpoint.group(stream_a, gate_cell_gate=False)
    router_off = checkpoint.group(stream_a, router_cell_gate=False)
    replay_off = checkpoint.group(stream_a, replay_cell_gate=False)
    assert completion_off.status == learner.STOP_GATE_ABSTAIN
    assert completion_off.stop_event_index == 4
    assert recurrence_off.status == learner.STOP_GATE_ABSTAIN
    assert recurrence_off.stop_event_index == 1
    assert gate_off.status == learner.STOP_FEATURE_ZERO
    assert gate_off.stop_event_index == 1
    assert router_off.status == learner.STOP_ROUTER
    assert router_off.accepted_raw_event_count == 0
    assert replay_off.status == learner.STOP_REPLAY
    assert replay_off.accepted_raw_event_count == 0
    assert replay_off.work.trigger_replay_count == 1
    assert all(
        value == 0
        for value in (
            learner.PUBLIC_TRAIN_INPUT_COUNT,
            learner.DEV_INPUT_COUNT,
            learner.M0_INPUT_COUNT,
            learner.C1_INPUT_COUNT,
            learner.G0_INPUT_COUNT,
            learner.PATH2_INPUT_COUNT,
            learner.PROTECTED_INPUT_COUNT,
        )
    )
