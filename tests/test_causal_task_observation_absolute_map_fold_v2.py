from __future__ import annotations

import inspect
import json

from experiments.r3_pilot import causal_occurrence_absolute_map_fold_v1 as map_fold
from experiments.r3_pilot import causal_task_observation_absolute_map_fold_v2 as task_v2
from experiments.r3_pilot import public_typed_output_cells as output_cells

from tests import test_causal_occurrence_absolute_map_fold_v1 as banked_v1

FIT_CASES = (
    ([-8, 3, -2],),
    ([-5, 1],),
    ([6, -4, 2, -3],),
)
FIT_A_PRIME_CASES = (
    ([-7],),
    ([2, -9, 4],),
    ([-3, 5, -1, 2],),
)
RAW_VISIBLE = (
    ([-12, 5],),
    ([4, -9, 3],),
    ([-2, -1, 7, -4],),
)
ABS_VISIBLE = (
    ([-11],),
    ([6, -2, 5],),
    ([-8, 1, -3, 4],),
)
FLIP_VISIBLE = (
    ([-10, 3, -1],),
    ([7, -6],),
    ([-4, -2, 5, 1],),
)
AMBIGUOUS_VISIBLE = (
    ([1, 4],),
    ([2, 0, 7],),
    ([3, 6, 5, 8],),
)
FRESH_EXECUTION = (
    ([-13, 2],),
    ([5, -8, 1],),
    ([-3, -6, 4, 2],),
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


def _observation(
    prompt_bytes: bytes,
    prompt_cells: tuple[int, ...],
    cases: tuple[tuple[list[int]], ...],
    *,
    absolute_values: bool,
) -> task_v2.NeutralTaskObservation:
    return task_v2.NeutralTaskObservation(
        prompt_bytes,
        prompt_cells,
        cases,
        _outputs(cases, absolute_values=absolute_values),
    )


def _execute(source: str) -> tuple[int, ...]:
    namespace: dict[str, object] = {}
    exec(source, {}, namespace)
    aggregate = namespace["aggregate"]
    return tuple(
        aggregate(list(values))  # type: ignore[operator]
        for (values,) in FRESH_EXECUTION
    )


def _parent_checkpoint() -> tuple[object, map_fold.FrozenCausalOccurrenceAbsoluteMapFoldV1]:
    signature, bridge_checkpoint = banked_v1._banked_bridge()
    lessons = (
        map_fold.CausalOccurrenceTransformLesson(
            banked_v1._closed_chain(0, 1, 2, multiplier=3, offset=1),
            banked_v1._outputs(banked_v1.FIT_CASES, absolute_values=False),
        ),
        map_fold.CausalOccurrenceTransformLesson(
            banked_v1._closed_fanout(3, 4, 5, multiplier=5, offset=9),
            banked_v1._outputs(banked_v1.FIT_CASES, absolute_values=True),
        ),
        map_fold.CausalOccurrenceTransformLesson(
            banked_v1._closed_chain(6, 7, 8, 9, multiplier=7, offset=3),
            banked_v1._outputs(banked_v1.FIT_CASES, absolute_values=False),
        ),
    )
    checkpoint = map_fold.fit_causal_occurrence_absolute_map_fold_v1(
        bridge_checkpoint,
        lessons,
        signature=signature,
        cases=banked_v1.FIT_CASES,
    )
    return signature, checkpoint


def test_task_observation_examples_select_one_learned_map_fold_witness() -> None:
    signature, parent_checkpoint = _parent_checkpoint()
    parent_root = parent_checkpoint.checkpoint_sha256
    fit_lessons = (
        _observation(
            b"\x91\x05\xd4\x2a",
            (701, 113, 907, 331),
            FIT_CASES,
            absolute_values=False,
        ),
        _observation(
            b"\x3c\xe8\x77\x10\xa6",
            (229, 881, 443, 157, 991),
            FIT_CASES,
            absolute_values=True,
        ),
        _observation(
            b"\xbf\x21\x64\xfa\x08\x93",
            (587, 269, 953, 419, 743, 101),
            FIT_A_PRIME_CASES,
            absolute_values=False,
        ),
    )
    checkpoint = task_v2.fit_causal_task_observation_absolute_map_fold_v2(
        parent_checkpoint,
        fit_lessons,
        signature=signature,
    )
    frozen_root = checkpoint.checkpoint_sha256
    summary = checkpoint.training_summary

    assert summary.example_effect_cell_ids == (
        task_v2.RAW_EXAMPLE_EFFECT_CELL,
        task_v2.ABSOLUTE_EXAMPLE_EFFECT_CELL,
        task_v2.RAW_EXAMPLE_EFFECT_CELL,
    )
    assert summary.transform_action_cell_ids == (
        map_fold.IDENTITY_TRANSFORM_CELL,
        map_fold.ABSOLUTE_TRANSFORM_CELL,
        map_fold.IDENTITY_TRANSFORM_CELL,
    )
    assert summary.learned_cell_pairs == (
        (task_v2.ABSOLUTE_EXAMPLE_EFFECT_CELL, map_fold.ABSOLUTE_TRANSFORM_CELL),
        (task_v2.RAW_EXAMPLE_EFFECT_CELL, map_fold.IDENTITY_TRANSFORM_CELL),
    )
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 3
    assert summary.maximum_excitatory_tick == summary.maximum_inhibitory_tick == 2
    assert summary.work.observation_count == 3
    assert summary.work.example_effect_race_count == 3
    assert summary.work.example_effect_race_cell_steps == 5_400
    assert summary.work.total_physical_cell_steps == 5_400
    assert summary.work.physical_predictions > 0
    assert summary.work.physical_races > 0
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert task_v2.NEW_CELL_COUNT == 2
    assert task_v2.TOTAL_CELL_COUNT == 69_236
    assert task_v2.NEW_PERSISTENT_STATE_BYTES == 8
    assert task_v2.COMPOSED_PERSISTENT_STATE_BYTES == 784

    raw_observation = _observation(
        b"\x02\xf1\x88\x4d\xb7",
        (1201, 409, 1667, 853, 1423),
        RAW_VISIBLE,
        absolute_values=False,
    )
    absolute_observation = _observation(
        bytes(reversed(b"\x19\xc4\x6e\xa1\x35\xf8")),
        tuple(reversed((1307, 461, 1789, 967, 1543, 211))),
        ABS_VISIBLE,
        absolute_values=True,
    )
    flip_prompt = b"\xed\x31\x9a\x04\xc7\x58"
    flip_cells = (1871, 577, 1601, 293, 1217, 941)
    flip_raw_observation = _observation(
        flip_prompt,
        flip_cells,
        FLIP_VISIBLE,
        absolute_values=False,
    )
    flip_absolute_observation = _observation(
        flip_prompt,
        flip_cells,
        FLIP_VISIBLE,
        absolute_values=True,
    )

    raw = checkpoint.build(raw_observation, signature=signature)
    absolute_result = checkpoint.build(absolute_observation, signature=signature)
    flip_raw = checkpoint.build(flip_raw_observation, signature=signature)
    flip_absolute = checkpoint.build(
        flip_absolute_observation, signature=signature
    )
    ready = (raw, absolute_result, flip_raw, flip_absolute)

    for result in ready:
        assert result.status == task_v2.READY
        assert result.exact_proof_count == 1
        assert result.work.example_effect_race_cell_steps == 1_800
        assert result.work.transform_action_race_cell_steps == 60
        assert result.work.stacked_cell_steps == 1_860
        assert result.work.physical_predictions > 0
        assert result.work.physical_races > 0
        assert result.work.packed_build_count == 1
        assert result.work.authorized_slice_count == 1
        assert result.work.proof_materialization_count == 1
        assert (
            task_v2.require_task_observation_absolute_map_fold_witness(result)
            is result.witness
        )

    raw_source = "def aggregate(numbers):\n    return sum(numbers)\n"
    absolute_source = (
        "def aggregate(numbers):\n"
        "    return sum([abs(item_0) for item_0 in numbers])\n"
    )
    for result in (raw, flip_raw):
        assert result.example_effect_cell_id == task_v2.RAW_EXAMPLE_EFFECT_CELL
        assert result.transform_action_cell_id == map_fold.IDENTITY_TRANSFORM_CELL
        assert result.witness is not None
        assert result.witness.source == raw_source
        assert _execute(result.witness.source) == tuple(
            sum(values) for (values,) in FRESH_EXECUTION
        )
    for result in (absolute_result, flip_absolute):
        assert (
            result.example_effect_cell_id
            == task_v2.ABSOLUTE_EXAMPLE_EFFECT_CELL
        )
        assert result.transform_action_cell_id == map_fold.ABSOLUTE_TRANSFORM_CELL
        assert result.witness is not None
        assert result.witness.source == absolute_source
        assert _execute(result.witness.source) == tuple(
            sum(abs(item) for item in values) for (values,) in FRESH_EXECUTION
        )
    assert flip_raw.task_source_sha256 != flip_absolute.task_source_sha256
    assert flip_raw.transform_action_cell_id != flip_absolute.transform_action_cell_id
    assert flip_raw.witness.source != flip_absolute.witness.source  # type: ignore[union-attr]
    assert len({item.task_source_sha256 for item in ready}) == len(ready)

    ambiguous = checkpoint.build(
        _observation(
            b"\xa4\x17\xd9\x62",
            (1999, 617, 1451, 337),
            AMBIGUOUS_VISIBLE,
            absolute_values=False,
        ),
        signature=signature,
    )
    source_off = checkpoint.build(
        absolute_observation,
        signature=signature,
        source_binding_cell_gate=False,
    )
    transform_off = checkpoint.build(
        absolute_observation,
        signature=signature,
        transform_action_cell_gate=False,
    )
    assert ambiguous.status == task_v2.STOP_EXAMPLE_EFFECT_NOT_SINGLETON
    assert ambiguous.example_effect_cell_id is None
    assert ambiguous.work.example_effect_race_cell_steps == 1_800
    assert source_off.status == task_v2.STOP_SOURCE_BINDING
    assert source_off.work.example_effect_race_cell_steps == 0
    assert source_off.work.physical_predictions == 0
    assert transform_off.status == task_v2.STOP_TRANSFORM_ZERO
    assert transform_off.example_effect_cell_id == task_v2.ABSOLUTE_EXAMPLE_EFFECT_CELL
    assert transform_off.work.example_effect_race_cell_steps == 1_800
    assert transform_off.work.transform_action_race_cell_steps == 60
    for stopped in (ambiguous, source_off, transform_off):
        assert stopped.witness is None
        assert stopped.exact_proof_count == 0
        assert stopped.transform_action_cell_id is None
        assert stopped.work.packed_build_count == 0
        assert stopped.work.authorized_slice_count == 0
        assert stopped.work.proof_materialization_count == 0

    assert checkpoint.checkpoint_sha256 == frozen_root
    assert parent_checkpoint.checkpoint_sha256 == parent_root

    hidden_build_inputs = {
        "fold_events",
        "transform_events",
        "probe_cases",
        "probe_outputs",
        "desired",
        "desired_program",
        "target_action",
        "operator",
        "family",
        "cohort",
        "schema",
        "schema_index",
        "topology",
    }
    build_parameters = inspect.signature(
        task_v2.FrozenCausalTaskObservationAbsoluteMapFoldV2.build
    ).parameters
    assert hidden_build_inputs.isdisjoint(build_parameters)
    inference_source = inspect.getsource(
        task_v2.FrozenCausalTaskObservationAbsoluteMapFoldV2.build
    )
    assert ".selected_actions" not in inference_source
    assert ".selected_action" not in inference_source
    assert ".state.schema_action" not in inference_source
    assert "fold_events" not in inference_source
    assert "transform_events" not in inference_source
    assert all(
        value == 0
        for value in (
            task_v2.DESIRED_PROGRAM_READS_DURING_BUILD,
            task_v2.PROBE_OUTPUT_READS_DURING_BUILD,
            task_v2.ACTION_LABEL_INPUTS_DURING_BUILD,
            task_v2.FAMILY_LABEL_INPUTS_DURING_BUILD,
            task_v2.SCHEMA_LABEL_INPUTS_DURING_BUILD,
            task_v2.TOPOLOGY_INPUTS_DURING_BUILD,
            task_v2.DIRECT_V1_ACTION_READS,
            task_v2.PUBLIC_TRAIN_INPUT_COUNT,
            task_v2.DEV_INPUT_COUNT,
            task_v2.HELDOUT_INPUT_COUNT,
            task_v2.M0_INPUT_COUNT,
            task_v2.C1_INPUT_COUNT,
            task_v2.G0_INPUT_COUNT,
            task_v2.PATH2_INPUT_COUNT,
            task_v2.PROTECTED_INPUT_COUNT,
        )
    )
    print(
        "TASK_OBSERVATION_V2_LEDGER="
        + json.dumps(
            {
                "training_predictions": summary.work.physical_predictions,
                "training_cache_hits": summary.work.prediction_cache_hits,
                "training_physical_races": summary.work.physical_races,
                "training_effect_cell_steps": summary.work.total_physical_cell_steps,
                "raw_build_predictions": raw.work.physical_predictions,
                "raw_build_physical_races": raw.work.physical_races,
                "absolute_build_predictions": absolute_result.work.physical_predictions,
                "absolute_build_physical_races": absolute_result.work.physical_races,
                "ready_stacked_cell_steps_each": raw.work.stacked_cell_steps,
                "ambiguous_stacked_cell_steps": ambiguous.work.stacked_cell_steps,
            },
            sort_keys=True,
        )
    )
