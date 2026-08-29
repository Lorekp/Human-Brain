from __future__ import annotations

import inspect

import pytest

from experiments.r3_pilot import public_typed_list_map_behavior as behavior
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.typed_program import FactorAction


MAP_IDENTITY = FactorAction(
    (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
)
MAP_ABSOLUTE_BODY = FactorAction(
    (0, 0), "APPLY1", "absolute", "SELECTED", "NONE", None
)
MAP_SUBTRACT_TWO_BODY = FactorAction(
    (0, 0), "APPLY2", "subtract", "SELECTED", "LITERAL", 2
)


@pytest.fixture(scope="module")
def model() -> behavior.FrozenListMapBehavior:
    return behavior.build_public_list_map_behavior()


def _lists(*values: list[int]) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.LIST_INT,
        )
        for value in values
    )


def _expected(*values: list[int]) -> tuple[str, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.LIST_INT,
        ).sha256
        for value in values
    )


def _assert_physical_and_frozen(
    model: behavior.FrozenListMapBehavior,
    prediction: behavior.ListMapBehaviorPrediction,
) -> None:
    assert prediction.checkpoint_before_sha256 == (
        prediction.checkpoint_after_sha256
    ) == model.checkpoint_sha256 == model.live_checkpoint_sha256
    assert prediction.races
    assert all(race.simultaneous_spike_candidates == 1 for race in prediction.races)
    assert all(not case.output.cells.flags.writeable for case in prediction.cases)
    assert all(not case.output.active_factor_mask.flags.writeable for case in prediction.cases)


def test_map_absolute_fans_out_and_reassembles_bounded_list_factors(
    model: behavior.FrozenListMapBehavior,
) -> None:
    prediction = model.predict(
        _lists([-3, 0, 4], [2, -1], []),
        MAP_IDENTITY,
        MAP_ABSOLUTE_BODY,
    )

    assert prediction.supported
    assert prediction.status == behavior.SUPPORTED
    assert tuple(output.sha256 for output in prediction.outputs) == _expected(
        [3, 0, 4],
        [2, 1],
        [],
    )
    assert tuple(len(case.element_predictions) for case in prediction.cases) == (
        3,
        2,
        0,
    )
    empty = prediction.cases[2]
    assert empty.output.list_length_cells[0] == 1.0
    assert not empty.output.list_item_cells.any()
    assert tuple(race.bank for race in empty.reassembly_races) == (
        "list-map-reassemble-type-2",
        "list-map-reassemble-length-2",
    )
    _assert_physical_and_frozen(model, prediction)


def test_map_subtract_two_preserves_element_operand_orientation(
    model: behavior.FrozenListMapBehavior,
) -> None:
    prediction = model.predict(
        _lists([-3, 0, 4], [2, -1], []),
        MAP_IDENTITY,
        MAP_SUBTRACT_TWO_BODY,
    )

    assert prediction.supported
    assert tuple(output.sha256 for output in prediction.outputs) == _expected(
        [-5, -2, 2],
        [0, -3],
        [],
    )
    assert all(
        item.action == MAP_SUBTRACT_TWO_BODY
        for case in prediction.cases
        for item in case.element_predictions
    )
    assert all(
        output.output_type is output_cells.OutputType.LIST_INT
        for output in prediction.outputs
    )
    _assert_physical_and_frozen(model, prediction)


@pytest.mark.parametrize(
    "first,second",
    (
        (
            FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
            MAP_ABSOLUTE_BODY,
        ),
        (
            MAP_IDENTITY,
            FactorAction(
                (0, 0),
                "APPLY2",
                "less-than",
                "SELECTED",
                "LITERAL",
                2,
            ),
        ),
        (
            MAP_IDENTITY,
            FactorAction(
                (1,), "APPLY1", "absolute", "SELECTED", "NONE", None
            ),
        ),
    ),
)
def test_every_unsupported_sequence_returns_unique_physical_unknown(
    model: behavior.FrozenListMapBehavior,
    first: FactorAction,
    second: FactorAction,
) -> None:
    prediction = model.predict(_lists([-3, 0, 4], []), first, second)

    assert not prediction.supported
    assert prediction.status == behavior.UNKNOWN
    assert prediction.sequence_race.unknown_won
    assert all(not case.output.supported for case in prediction.cases)
    assert all(case.output.unknown_active for case in prediction.cases)
    assert all(case.map_prediction is None for case in prediction.cases)
    _assert_physical_and_frozen(model, prediction)


def test_unknown_list_and_scalar_overflow_propagate_to_physical_unknown(
    model: behavior.FrozenListMapBehavior,
) -> None:
    unknown_list = output_cells.TypedOutputCellBank.encode(
        object(),
        declared_type=output_cells.OutputType.LIST_INT,
    )
    unknown = model.predict(
        (unknown_list,),
        MAP_IDENTITY,
        MAP_ABSOLUTE_BODY,
    )
    overflow = model.predict(
        _lists([20]),
        MAP_IDENTITY,
        FactorAction(
            (0, 0), "APPLY2", "add", "SELECTED", "LITERAL", 2
        ),
    )

    assert not unknown.supported
    assert unknown.outputs[0].unknown_active
    assert not overflow.supported
    assert overflow.outputs[0].list_length_cells[1] == 1.0
    assert overflow.outputs[0].list_item_cells[0, output_cells.INT_UNKNOWN_CELL] == 1.0
    _assert_physical_and_frozen(model, unknown)
    _assert_physical_and_frozen(model, overflow)


def test_inference_has_no_program_planner_target_or_actuator_seam() -> None:
    source = inspect.getsource(behavior.FrozenListMapBehavior)

    assert "operator_successor.predict" in source
    for forbidden in (
        "Program(",
        "factor_action_candidates",
        "apply_factor_action",
        "replace_subtree",
        "compile(",
        "exec(",
        "desired",
        "prompt",
        "target",
        "Horizon2",
    ):
        assert forbidden not in source
