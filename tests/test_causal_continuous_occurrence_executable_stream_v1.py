from __future__ import annotations

import inspect

from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_content_addressed_segment_schema_v1 as content
from experiments.r3_pilot import causal_continuous_occurrence_executable_stream_v1 as stream
from experiments.r3_pilot import causal_learned_occurrence_boundary_gate_v1 as boundary_gate
from experiments.r3_pilot import causal_occurrence_executable_reducer_bridge_v1 as bridge
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter

from tests import test_causal_learned_occurrence_boundary_gate_v1 as gate_fixture
from tests import test_causal_occurrence_executable_reducer_bridge_v1 as bridge_fixture
from tests import test_causal_predictive_occurrence_stream_grouper_v1 as occurrence_fixture


def test_continuous_neutral_stream_publishes_ordered_exact_witnesses() -> None:
    signature = bridge_fixture._signature()
    base_episodes = (
        append.CausalAppendEffectEpisode(
            bridge_fixture._spans(30, 130),
            bridge_fixture._literals(-5, 2, (132, 140, 135)),
            bridge_fixture.SIGNATURE_ROLE,
            bridge_fixture._outputs(2, 3, 2),
        ),
        append.CausalAppendEffectEpisode(
            bridge_fixture._spans(90, 150),
            bridge_fixture._literals(-1, -3, (155, 152, 158)),
            bridge_fixture.SIGNATURE_ROLE,
            bridge_fixture._outputs(-2, 3, -10),
        ),
        append.CausalAppendEffectEpisode(
            bridge_fixture._spans(150, 170),
            bridge_fixture._literals(4, 3, (175, 172, 178)),
            bridge_fixture.SIGNATURE_ROLE,
            bridge_fixture._outputs(1, 1, 2),
        ),
    )
    shared_segmented = occurrence_fixture._segmented_checkpoint()
    content_checkpoint = content.fit_causal_content_addressed_segment_schema_v1(
        shared_segmented,
        (
            base_episodes[1],
            base_episodes[0],
            append.CausalAppendEffectEpisode(
                bridge_fixture._spans(330, 290),
                bridge_fixture._literals(-6, -2, (295, 292, 298)),
                bridge_fixture.SIGNATURE_ROLE,
                bridge_fixture._outputs(-2, 3, -10),
            ),
        ),
        signature=signature,
        probe_cases=bridge_fixture.TRAIN_PROBE,
        literals=(-6, -5, -3, -2, -1, 0, 2),
        max_cost=8,
    )
    occurrence_checkpoint = occurrence_fixture._occurrence_checkpoint()
    bridge_checkpoint = bridge.fit_causal_occurrence_executable_reducer_bridge_v1(
        occurrence_checkpoint,
        content_checkpoint,
        (
            bridge.CausalOccurrenceExecutableLesson(
                bridge_fixture._closed(
                    bridge_fixture._chain(
                        bridge_fixture._relabel(
                            bridge_fixture._groups(0, 1, 2),
                            multiplier=3,
                            offset=1,
                        )
                    )
                ),
                bridge_fixture._spans(410, 200),
                bridge_fixture._literals(-7, 3, (202, 210, 205)),
                bridge_fixture.SIGNATURE_ROLE,
                bridge_fixture.COUNT_PROBE,
                bridge_fixture._outputs(2, 3, 2),
            ),
            bridge.CausalOccurrenceExecutableLesson(
                bridge_fixture._closed(
                    bridge_fixture._fanout(
                        bridge_fixture._relabel(
                            bridge_fixture._groups(3, 4, 5),
                            multiplier=5,
                            offset=9,
                        )
                    )
                ),
                bridge_fixture._spans(470, 230),
                bridge_fixture._literals(-11, -6, (235, 232, 238)),
                bridge_fixture.SIGNATURE_ROLE,
                bridge_fixture.SUM_PROBE,
                bridge_fixture._outputs(2, -8, 10),
            ),
            bridge.CausalOccurrenceExecutableLesson(
                bridge_fixture._closed(
                    bridge_fixture._chain(
                        bridge_fixture._relabel(
                            bridge_fixture._groups(6, 7, 8, 9),
                            multiplier=7,
                            offset=3,
                        )
                    )
                ),
                tuple(reversed(bridge_fixture._spans(380, 260))),
                tuple(
                    reversed(bridge_fixture._literals(17, 31, (262, 270, 265)))
                ),
                bridge_fixture.SIGNATURE_ROLE,
                bridge_fixture.RELABEL_PROBE,
                bridge_fixture._outputs(1, 2, 1),
                (0x1E, 0xB4, 0x63, 0xC9, 0x78, 0x27, 0xD2, 0x8D),
            ),
        ),
    )
    neutral_interval = segmenter.INTERVAL_CELL_IDS[0]
    gate_checkpoint = boundary_gate.fit_causal_learned_occurrence_boundary_gate_v1(
        shared_segmented,
        occurrence_checkpoint,
        gate_fixture._training_transports(neutral_interval),
    )
    checkpoint = stream.compose_causal_continuous_occurrence_executable_stream_v1(
        gate_checkpoint, bridge_checkpoint
    )
    frozen_root = checkpoint.checkpoint_sha256

    first_raw, second_raw = gate_fixture._fresh_streams(neutral_interval)
    prompt_spans = tuple(reversed(bridge_fixture._spans(420, 310)))
    prompt_literals = tuple(
        reversed(bridge_fixture._literals(-13, 4, (312, 320, 315)))
    )

    def build(
        raw_events: tuple[boundary_gate.RawNeutralOccurrenceEvent, ...],
        **gates: bool,
    ) -> stream.ContinuousOccurrenceExecutableOutcome:
        return checkpoint.build(
            raw_events,
            prompt_spans,
            prompt_literals,
            bridge_fixture.SIGNATURE_ROLE,
            signature=signature,
            cases=bridge_fixture.FINAL_CASES,
            literals=(-13, 0, 4),
            max_cost=8,
            **gates,
        )

    first = build(first_raw)
    second = build(second_raw)
    assert first.status == second.status == stream.READY
    assert first.grouping.work.boundary_event_indices == (4, 6)
    assert second.grouping.work.boundary_event_indices == (3,)
    assert first.work.raw_event_count == len(first_raw)
    assert second.work.raw_event_count == len(second_raw)
    assert first.work.occurrence_certificate_count == first.work.bridge_build_count == 3
    assert second.work.occurrence_certificate_count == second.work.bridge_build_count == 2
    assert first.work.exact_proof_count == 3
    assert second.work.exact_proof_count == 2

    first_schema = tuple(
        occurrence.require_occurrence_graph_schema(value).schema_index
        for value in first.occurrence_certificates
    )
    second_schema = tuple(
        occurrence.require_occurrence_graph_schema(value).schema_index
        for value in second.occurrence_certificates
    )
    assert first_schema == (0, 1, 0)
    assert second_schema == (1, 0)
    count_source = (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -13 < item_0 + 4))\n"
    )
    sum_source = "def aggregate(numbers):\n    return sum(numbers)\n"
    first_sources = tuple(
        bridge.require_occurrence_executable_witness(value).source
        for value in first.witness_certificates
    )
    second_sources = tuple(
        bridge.require_occurrence_executable_witness(value).source
        for value in second.witness_certificates
    )
    assert first_sources == (count_source, sum_source, count_source)
    assert second_sources == (sum_source, count_source)
    assert bridge_fixture._execute(first_sources[0]) == (0, 3, 2, 3)
    assert bridge_fixture._execute(first_sources[1]) == (0, -15, 2, 9)

    for outcome in (first, second):
        assert outcome.work.packed_frontier_build_count == len(
            outcome.witness_certificates
        )
        assert outcome.work.authorized_slice_count == len(
            outcome.witness_certificates
        )
        assert outcome.work.proof_materialization_count == len(
            outcome.witness_certificates
        )
        assert outcome.work.total_executed_physical_cell_steps == (
            outcome.work.grouping_physical_cell_steps
            + outcome.work.residual_binding_cell_steps
            + outcome.work.bridge_physical_cell_steps
        )
        assert outcome.work.certificate_provenance_occurrence_cell_steps > 0

    gate_off = build(first_raw, grouping_cell_gate=False)
    bridge_off = build(first_raw, bridge_relay_cell_gate=False)
    assert gate_off.status == boundary_gate.STOP_FEATURE_ZERO
    assert gate_off.work.bridge_build_count == gate_off.work.exact_proof_count == 0
    assert gate_off.witness_certificates == ()
    assert bridge_off.grouping.status == boundary_gate.READY
    assert bridge_off.status == bridge.STOP_RELAY_TONIC
    assert bridge_off.work.bridge_build_count == 1
    assert bridge_off.work.exact_proof_count == 0
    assert bridge_off.work.packed_frontier_build_count == 0
    assert bridge_off.witness_certificates[0].witness is None

    assert stream.NEW_CELL_COUNT == 0
    assert stream.NEW_TRAINABLE_STATE_BYTES == 0
    assert stream.NEW_PERSISTENT_STATE_BYTES == 0
    assert stream.TOTAL_CELL_COUNT == 69_236
    assert stream.COMPOSED_PERSISTENT_STATE_BYTES == 784
    assert stream.RUNTIME_STATE_BYTES == 60
    assert checkpoint.checkpoint_sha256 == frozen_root
    hidden_inputs = {
        "boundary",
        "group_length",
        "selected_action",
        "desired",
        "desired_program",
        "operator",
        "operator_name",
    }
    assert hidden_inputs.isdisjoint(
        inspect.signature(
            stream.FrozenContinuousOccurrenceExecutableStreamV1.build
        ).parameters
    )
    inference_source = inspect.getsource(
        stream.FrozenContinuousOccurrenceExecutableStreamV1.build
    )
    assert ".build_from_occurrence_certificate(" in inference_source
    assert "._bridge.build(" not in inference_source
    assert ".selected_actions" not in inference_source
    assert all(
        value == 0
        for value in (
            stream.BOUNDARY_BOOLEAN_INPUT_COUNT,
            stream.GROUP_LENGTH_INPUT_COUNT,
            stream.SELECTED_ACTION_INPUT_COUNT,
            stream.DESIRED_PROGRAM_INPUT_COUNT,
            stream.OPERATOR_NAME_INPUT_COUNT,
            stream.PUBLIC_TRAIN_INPUT_COUNT,
            stream.DEV_INPUT_COUNT,
            stream.M0_INPUT_COUNT,
            stream.C1_INPUT_COUNT,
            stream.G0_INPUT_COUNT,
            stream.PATH2_INPUT_COUNT,
            stream.PROTECTED_INPUT_COUNT,
        )
    )
