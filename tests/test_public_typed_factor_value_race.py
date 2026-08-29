from __future__ import annotations

import inspect

import pytest

from experiments.r3_pilot import public_typed_factor_value_race as value_race
from experiments.r3_pilot import public_typed_output_cells as outputs
from humanbrain.resting_glif_race import RestingGlifRaceFailure
from humanbrain.typed_program import FactorAction


DONE = FactorAction((), "DONE", "NONE", "NONE", "NONE", None)
ADD2 = FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2)
ABS = FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)


def _int(value: object):
    return outputs.TypedOutputCellBank.encode(value, declared_type=outputs.OutputType.INT)


def test_exact_typed_predictions_drive_the_complete_candidate_winner() -> None:
    projection = value_race.FixedTypedFactorValueProjection.build(
        (DONE, ADD2, ABS),
        tuple(_int(value) for value in (3, 5, 8)),
    )
    predictions = (
        tuple(_int(value) for value in (1, 3, 6)),
        tuple(_int(value) for value in (3, 5, 8)),
        tuple(_int(value) for value in (1, 3, 6)),
    )

    decision = projection.select_unique(predictions)

    assert decision.status == "SELECTED"
    assert decision.selected_candidate_cell == 1
    assert decision.selected_action == ADD2
    assert decision.simultaneous_spike_candidates == 1
    assert decision.currents_pa[1] > max(decision.currents_pa[0], decision.currents_pa[2])


def test_physical_unknown_is_inhibited_below_stop() -> None:
    projection = value_race.FixedTypedFactorValueProjection.build(
        (DONE, ADD2),
        (_int(3),),
    )
    unknown = _int(object())

    decision = projection.select_unique(((unknown,), (unknown,)))

    assert decision.status == "STOP_ALL_UNKNOWN"
    assert decision.selected_candidate_cell is None
    assert decision.winner_cell == 2
    assert decision.simultaneous_spike_candidates == 1


def test_unsupported_string_is_inhibited_below_stop() -> None:
    projection = value_race.FixedTypedFactorValueProjection.build(
        (DONE,),
        (outputs.TypedOutputCellBank.encode("ok"),),
    )
    unsupported = outputs.TypedOutputCellBank.encode("A\n")

    decision = projection.select_unique(((unsupported,),))

    assert not unsupported.supported
    assert unsupported.unknown_active
    assert decision.status == "STOP_ALL_UNKNOWN"
    assert decision.selected_candidate_cell is None
    assert decision.currents_pa == (-9_300.0, 650.0)


def test_tier_currents_preserve_semantic_ties_and_strict_motor_stops() -> None:
    projection = value_race.FixedTypedFactorValueProjection.build(
        (DONE, ADD2, ABS),
        (_int(3), _int(5)),
    )
    exact = (_int(3), _int(5))
    wrong = (_int(2), _int(4))
    features = projection.features((exact, exact, wrong))
    tier = projection.tier_currents(features)

    assert tier[0] == tier[1] > tier[2] > tier[3]
    with pytest.raises(RestingGlifRaceFailure, match="2 simultaneous"):
        projection.select_unique((exact, exact, wrong))


def test_projection_is_readonly_and_inference_has_no_host_candidate_ranking() -> None:
    projection = value_race.FixedTypedFactorValueProjection.build(
        (DONE, ADD2),
        (_int(3),),
    )
    assert not projection.weights_pa.flags.writeable
    assert not projection.bias_pa.flags.writeable

    source = "\n".join(
        inspect.getsource(function)
        for function in (
            value_race.FixedTypedFactorValueProjection.features,
            value_race.FixedTypedFactorValueProjection.tier_currents,
            value_race.FixedTypedFactorValueProjection.select_unique,
        )
    )
    for forbidden in (
        "sorted(",
        ".sort(",
        "argmax",
        "argpartition",
        "apply_factor_action",
        "replace_subtree",
        "compile(",
        "exec(",
    ):
        assert forbidden not in source
