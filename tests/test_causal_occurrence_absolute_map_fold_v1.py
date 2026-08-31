from __future__ import annotations

import inspect

from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_content_addressed_segment_schema_v1 as content
from experiments.r3_pilot import causal_occurrence_absolute_map_fold_v1 as map_fold
from experiments.r3_pilot import causal_occurrence_executable_reducer_bridge_v1 as bridge
from experiments.r3_pilot import public_typed_output_cells as output_cells

from tests import test_causal_occurrence_executable_reducer_bridge_v1 as banked

FIT_CASES = (
    ([-9, 4, -2],),
    ([-3, -1],),
    ([6, -5, 1, -4],),
)
FRESH_B4_CASES = (
    ([-11, 4, -2, 9],),
    ([-8],),
    ([3, -5, 7],),
    ([6, 0],),
)
LAW_FLIP_CASES = (
    ([-7, 2, -4, 3],),
    ([5, -6],),
    ([-2, -3, -1],),
    ([4],),
)
RELABEL_CASES = (
    ([-5, 3, -2, 4],),
    ([7, -8, 1],),
    ([-6],),
)


def _outputs(
    cases: tuple[tuple[list[int]], ...], *, absolute_values: bool
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    values = tuple(
        sum(abs(item) for item in row) if absolute_values else sum(row)
        for (row,) in cases
    )
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value, declared_type=output_cells.OutputType.INT
        )
        for value in values
    )


def _execute(
    source: str, cases: tuple[tuple[list[int]], ...]
) -> tuple[int, ...]:
    namespace: dict[str, object] = {}
    exec(source, {}, namespace)
    aggregate = namespace["aggregate"]
    return tuple(aggregate(list(values)) for (values,) in cases)  # type: ignore[operator]


def _banked_bridge() -> tuple[
    object, bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1
]:
    signature = banked._signature()
    base_episodes = (
        append.CausalAppendEffectEpisode(
            banked._spans(30, 130),
            banked._literals(-5, 2, (132, 140, 135)),
            banked.SIGNATURE_ROLE,
            banked._outputs(2, 3, 2),
        ),
        append.CausalAppendEffectEpisode(
            banked._spans(90, 150),
            banked._literals(-1, -3, (155, 152, 158)),
            banked.SIGNATURE_ROLE,
            banked._outputs(-2, 3, -10),
        ),
        append.CausalAppendEffectEpisode(
            banked._spans(150, 170),
            banked._literals(4, 3, (175, 172, 178)),
            banked.SIGNATURE_ROLE,
            banked._outputs(1, 1, 2),
        ),
    )
    dependency = banked._segmented_checkpoint(signature, base_episodes)
    content_checkpoint = content.fit_causal_content_addressed_segment_schema_v1(
        dependency,
        (
            base_episodes[1],
            base_episodes[0],
            append.CausalAppendEffectEpisode(
                banked._spans(330, 290),
                banked._literals(-6, -2, (295, 292, 298)),
                banked.SIGNATURE_ROLE,
                banked._outputs(-2, 3, -10),
            ),
        ),
        signature=signature,
        probe_cases=banked.TRAIN_PROBE,
        literals=(-6, -5, -3, -2, -1, 0, 2),
        max_cost=8,
    )
    occurrence_checkpoint = occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1(
        banked._chain(banked._groups(0, 1, 2))
        + banked._fanout(banked._groups(3, 4, 5))
        + banked._closed(banked._chain(banked._groups(6, 7, 8, 9)))
    )
    lessons = (
        bridge.CausalOccurrenceExecutableLesson(
            banked._closed(
                banked._chain(
                    banked._relabel(
                        banked._groups(0, 1, 2), multiplier=3, offset=1
                    )
                )
            ),
            banked._spans(410, 200),
            banked._literals(-7, 3, (202, 210, 205)),
            banked.SIGNATURE_ROLE,
            banked.COUNT_PROBE,
            banked._outputs(2, 3, 2),
        ),
        bridge.CausalOccurrenceExecutableLesson(
            banked._closed(
                banked._fanout(
                    banked._relabel(
                        banked._groups(3, 4, 5), multiplier=5, offset=9
                    )
                )
            ),
            banked._spans(470, 230),
            banked._literals(-11, -6, (235, 232, 238)),
            banked.SIGNATURE_ROLE,
            banked.SUM_PROBE,
            banked._outputs(2, -8, 10),
        ),
        bridge.CausalOccurrenceExecutableLesson(
            banked._closed(
                banked._chain(
                    banked._relabel(
                        banked._groups(6, 7, 8, 9), multiplier=7, offset=3
                    )
                )
            ),
            tuple(reversed(banked._spans(380, 260))),
            tuple(reversed(banked._literals(17, 31, (262, 270, 265)))),
            banked.SIGNATURE_ROLE,
            banked.RELABEL_PROBE,
            banked._outputs(1, 2, 1),
            (0x1E, 0xB4, 0x63, 0xC9, 0x78, 0x27, 0xD2, 0x8D),
        ),
    )
    checkpoint = bridge.fit_causal_occurrence_executable_reducer_bridge_v1(
        occurrence_checkpoint, content_checkpoint, lessons
    )
    return signature, checkpoint


def _closed_chain(
    *indices: int, multiplier: int, offset: int
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    nodes = banked._relabel(
        banked._groups(*indices), multiplier=multiplier, offset=offset
    )
    return banked._closed(banked._chain(nodes))


def _closed_fanout(
    *indices: int, multiplier: int, offset: int
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    nodes = banked._relabel(
        banked._groups(*indices), multiplier=multiplier, offset=offset
    )
    return banked._closed(banked._fanout(nodes))


def _permuted_reversed_fanout() -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    nodes = tuple(
        reversed(
            banked._relabel(
                banked._groups(10, 11, 12, 13, 14),
                multiplier=13,
                offset=11,
            )
        )
    )
    opened = banked._fanout(nodes)
    permuted = (opened[0], opened[3], opened[1], opened[4], opened[2])
    return banked._closed(permuted)


def test_occurrence_transform_learns_identity_vs_absolute_map_fold() -> None:
    signature, bridge_checkpoint = _banked_bridge()
    bridge_root = bridge_checkpoint.checkpoint_sha256
    fit_lessons = (
        map_fold.CausalOccurrenceTransformLesson(
            _closed_chain(0, 1, 2, multiplier=3, offset=1),
            _outputs(FIT_CASES, absolute_values=False),
        ),
        map_fold.CausalOccurrenceTransformLesson(
            _closed_fanout(3, 4, 5, multiplier=5, offset=9),
            _outputs(FIT_CASES, absolute_values=True),
        ),
        map_fold.CausalOccurrenceTransformLesson(
            _closed_chain(6, 7, 8, 9, multiplier=7, offset=3),
            _outputs(FIT_CASES, absolute_values=False),
        ),
    )
    checkpoint = map_fold.fit_causal_occurrence_absolute_map_fold_v1(
        bridge_checkpoint,
        fit_lessons,
        signature=signature,
        cases=FIT_CASES,
    )
    frozen_root = checkpoint.checkpoint_sha256
    summary = checkpoint.training_summary

    assert summary.occurrence_schema_cell_ids == (
        occurrence.SCHEMA_CELL_OFFSET,
        occurrence.SCHEMA_CELL_OFFSET + 1,
        occurrence.SCHEMA_CELL_OFFSET,
    )
    assert summary.consequence_transform_cell_ids == (
        map_fold.IDENTITY_TRANSFORM_CELL,
        map_fold.ABSOLUTE_TRANSFORM_CELL,
        map_fold.IDENTITY_TRANSFORM_CELL,
    )
    assert summary.learned_cell_pairs == (
        (occurrence.SCHEMA_CELL_OFFSET, map_fold.IDENTITY_TRANSFORM_CELL),
        (occurrence.SCHEMA_CELL_OFFSET + 1, map_fold.ABSOLUTE_TRANSFORM_CELL),
    )
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 3
    assert summary.maximum_excitatory_tick == summary.maximum_inhibitory_tick == 2
    assert summary.work.consequence_race_count == 3
    assert summary.work.consequence_race_cell_steps == 5_400
    assert summary.work.packed_candidate_build_count == 2
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert map_fold.NEW_CELL_COUNT == 2
    assert map_fold.TOTAL_CELL_COUNT == 69_234
    assert map_fold.NEW_PERSISTENT_STATE_BYTES == 8
    assert map_fold.COMPOSED_PERSISTENT_STATE_BYTES == 776

    fold_sum = _closed_fanout(0, 1, 2, 3, 4, multiplier=11, offset=5)
    transform_b4 = _closed_fanout(4, 5, 6, 7, multiplier=9, offset=7)
    transform_b5 = _closed_fanout(5, 6, 7, 8, 9, multiplier=7, offset=3)
    transform_a5 = _closed_chain(0, 1, 2, 3, 4, multiplier=3, offset=1)
    transform_relabel = _permuted_reversed_fanout()

    b4 = checkpoint.build(
        fold_sum,
        transform_b4,
        signature=signature,
        cases=FRESH_B4_CASES,
    )
    b5 = checkpoint.build(
        fold_sum,
        transform_b5,
        signature=signature,
        cases=LAW_FLIP_CASES,
    )
    a5_law_flip = checkpoint.build(
        fold_sum,
        transform_a5,
        signature=signature,
        cases=LAW_FLIP_CASES,
    )
    relabeled_permuted_reversed = checkpoint.build(
        fold_sum,
        transform_relabel,
        signature=signature,
        cases=RELABEL_CASES,
    )

    ready = (b4, b5, a5_law_flip, relabeled_permuted_reversed)
    for result in ready:
        assert result.status == map_fold.READY
        assert result.reducer_action_cell_id == map_fold.BANKED_SUM_ACTION_CELL
        assert result.exact_proof_count == 1
        assert result.work.fold_relay_race_cell_steps == 1_800
        assert result.work.reducer_action_race_cell_steps == 60
        assert result.work.transform_action_race_cell_steps == 60
        assert result.work.packed_build_count == 1
        assert result.work.authorized_slice_count == 1
        assert result.work.proof_materialization_count == 1
        assert (
            map_fold.require_occurrence_absolute_map_fold_witness(result)
            is result.witness
        )

    raw_source = "def aggregate(numbers):\n    return sum(numbers)\n"
    absolute_source = (
        "def aggregate(numbers):\n"
        "    return sum([abs(item_0) for item_0 in numbers])\n"
    )
    assert a5_law_flip.transform_action_cell_id == map_fold.IDENTITY_TRANSFORM_CELL
    assert a5_law_flip.witness is not None
    assert a5_law_flip.witness.source == raw_source
    assert _execute(a5_law_flip.witness.source, LAW_FLIP_CASES) == tuple(
        sum(values) for (values,) in LAW_FLIP_CASES
    )
    for result, cases in (
        (b4, FRESH_B4_CASES),
        (b5, LAW_FLIP_CASES),
        (relabeled_permuted_reversed, RELABEL_CASES),
    ):
        assert result.transform_action_cell_id == map_fold.ABSOLUTE_TRANSFORM_CELL
        assert result.witness is not None
        assert result.witness.source == absolute_source
        assert _execute(result.witness.source, cases) == tuple(
            sum(abs(item) for item in values) for (values,) in cases
        )
    assert a5_law_flip.fold_schema_cell_id == b5.fold_schema_cell_id
    assert a5_law_flip.transform_schema_cell_id != b5.transform_schema_cell_id
    assert a5_law_flip.witness.source != b5.witness.source  # type: ignore[union-attr]
    assert checkpoint.checkpoint_sha256 == frozen_root
    assert bridge_checkpoint.checkpoint_sha256 == bridge_root

    outcome_off_checkpoint = map_fold.fit_causal_occurrence_absolute_map_fold_v1(
        bridge_checkpoint,
        fit_lessons,
        signature=signature,
        cases=FIT_CASES,
        outcome_gate=False,
    )
    outcome_off = outcome_off_checkpoint.build(
        fold_sum,
        transform_b5,
        signature=signature,
        cases=LAW_FLIP_CASES,
    )
    occurrence_off = checkpoint.build(
        fold_sum,
        transform_b5,
        signature=signature,
        cases=LAW_FLIP_CASES,
        transform_relation_cell_gate=False,
    )
    transform_off = checkpoint.build(
        fold_sum,
        transform_b5,
        signature=signature,
        cases=LAW_FLIP_CASES,
        transform_action_cell_gate=False,
    )
    reducer_off = checkpoint.build(
        fold_sum,
        transform_b5,
        signature=signature,
        cases=LAW_FLIP_CASES,
        reducer_action_cell_gate=False,
    )
    assert outcome_off_checkpoint.training_summary.excitatory_update_ticks == 0
    assert outcome_off_checkpoint.training_summary.inhibitory_update_ticks == 0
    assert outcome_off.status == map_fold.STOP_TRANSFORM_ZERO
    assert occurrence_off.status == map_fold.STOP_TRANSFORM_OCCURRENCE
    assert transform_off.status == map_fold.STOP_TRANSFORM_ZERO
    assert reducer_off.status == map_fold.STOP_REDUCER_ACTION
    for stopped in (outcome_off, occurrence_off, transform_off, reducer_off):
        assert stopped.witness is None
        assert stopped.exact_proof_count == 0
        assert stopped.work.packed_build_count == 0

    hidden_build_inputs = {
        "probe_cases",
        "probe_outputs",
        "observed_outputs",
        "desired",
        "target_action",
        "operator",
        "family",
        "cohort",
        "schema",
        "schema_index",
    }
    build_parameters = inspect.signature(
        map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1.build
    ).parameters
    assert hidden_build_inputs.isdisjoint(build_parameters)
    inference_source = inspect.getsource(
        map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1.build
    )
    assert ".selected_actions" not in inference_source
    assert ".selected_action" not in inference_source
    assert ".state.schema_action" not in inference_source
    assert "probe_outputs" not in inference_source
    assert all(
        value == 0
        for value in (
            map_fold.DESIRED_READS_DURING_BUILD,
            map_fold.PROBE_OUTPUT_READS_DURING_BUILD,
            map_fold.OPERATOR_NAME_INPUTS_DURING_BUILD,
            map_fold.FAMILY_LABEL_INPUTS_DURING_BUILD,
            map_fold.TARGET_ACTION_INPUTS_DURING_BUILD,
            map_fold.SCHEMA_LABEL_INPUTS_DURING_BUILD,
            map_fold.DIRECT_OCCURRENCE_ACTION_READS,
            map_fold.DIRECT_APPEND_ACTION_READS,
            map_fold.SERIALIZED_SCHEMA_ACTION_READS,
            map_fold.PUBLIC_TRAIN_INPUT_COUNT,
            map_fold.DEV_INPUT_COUNT,
            map_fold.HELDOUT_INPUT_COUNT,
            map_fold.M0_INPUT_COUNT,
            map_fold.C1_INPUT_COUNT,
            map_fold.G0_INPUT_COUNT,
            map_fold.PATH2_INPUT_COUNT,
            map_fold.PROTECTED_INPUT_COUNT,
        )
    )
