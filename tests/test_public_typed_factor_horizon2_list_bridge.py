from __future__ import annotations

import pytest

from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_list_map_behavior as list_behavior
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


CASES = (([-3, 0, 4],), ([],), ([2, -1],))
DESIRED = ([-6, -3, 1], [], [-1, -4])
PROMPT = "Subtract three from every value."
MAP_IDENTITY = tp.FactorAction(
    (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
)
SUBTRACT_THREE_BODY = tp.FactorAction(
    (0, 0), "APPLY2", "subtract", "SELECTED", "LITERAL", 3
)
DONE = tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    list_int = ProgramType.List(integer)
    return TypedSignature(
        "solve",
        (ProgramParameter("values", list_int),),
        list_int,
    )


def _expected_sha256() -> tuple[str, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.LIST_INT,
        ).sha256
        for value in DESIRED
    )


@pytest.fixture(scope="module")
def model() -> h2.FrozenFactorizedHorizon2Int:
    return h2.build_public_factorized_horizon2_int(literals=(3,))


def _assert_list_terminal(
    terminal: h2.FactorTerminalSequencePrediction | None,
) -> None:
    assert terminal is not None
    assert terminal.supported
    assert tuple(output.sha256 for output in terminal.outputs) == _expected_sha256()
    assert all(not output.cells.flags.writeable for output in terminal.outputs)
    assert all(
        not output.active_factor_mask.flags.writeable
        for output in terminal.outputs
    )
    list_races = tuple(
        race
        for race in terminal.races
        if isinstance(race, list_behavior.ListFactorRace)
    )
    assert list_races
    assert all(race.simultaneous_spike_candidates == 1 for race in list_races)
    banks = {race.bank for race in list_races}
    assert "list-map-fanout-0-0" in banks
    assert "list-map-reassemble-item-0-0" in banks
    assert "list-map-reassemble-type-1" in banks
    assert "list-map-reassemble-length-1" in banks
    assert "list-map-fanout-1-0" not in banks
    assert terminal.checkpoint_before_sha256 == terminal.checkpoint_after_sha256


def test_factorized_h2_maps_then_replans_one_body_edit(
    model: h2.FrozenFactorizedHorizon2Int,
) -> None:
    incumbent = tp.Program(("variable-0",), _signature())
    observation = model.observe(incumbent, CASES)
    planner_before = model.live_checkpoint_sha256
    terminal_before = model.terminal_behavior.live_checkpoint_sha256
    list_before = model.terminal_behavior.list_map.live_checkpoint_sha256

    first = model.plan(observation, PROMPT, DESIRED)

    assert first.status == h2.STATUS_SELECTED
    assert first.selected_sequence == h2.FactorH2Sequence(
        MAP_IDENTITY,
        SUBTRACT_THREE_BODY,
    )
    assert first.selected_sequence_cell is not None
    assert all(
        desired.output_type is output_cells.OutputType.LIST_INT
        for desired in first.desired
    )
    _assert_list_terminal(
        first.predictions[first.selected_sequence_cell].terminal_behavior
    )

    mapped = model.actuate_and_reobserve(incumbent, first)
    assert mapped.actuator_calls == 1
    assert mapped.program.tokens == (
        "map",
        "bind",
        "variable-0",
        "variable-0",
    )
    assert mapped.structure_parity
    assert mapped.action_site_parity
    assert mapped.root_output_parity

    second = model.plan(mapped.observation, PROMPT, DESIRED)
    assert second.status == h2.STATUS_SELECTED
    assert second.selected_sequence == h2.FactorH2Sequence(
        SUBTRACT_THREE_BODY,
        DONE,
    )
    assert second.selected_sequence_cell is not None
    _assert_list_terminal(
        second.predictions[second.selected_sequence_cell].terminal_behavior
    )

    final = model.actuate_and_reobserve(mapped.program, second)
    assert final.actuator_calls == 1
    assert final.program.tokens == (
        "map",
        "bind",
        "subtract",
        "variable-0",
        "integer",
        "+",
        "3",
        "end",
        "variable-0",
    )
    assert tuple(
        output.sha256 for output in final.observation.root_outputs
    ) == _expected_sha256()
    assert final.observation.root_outputs[1].list_length_cells[0] == 1.0
    assert not final.observation.root_outputs[1].list_item_cells.any()
    assert model.plan(final.observation, PROMPT, DESIRED).status == h2.STATUS_DONE

    assert model.live_checkpoint_sha256 == planner_before == model.checkpoint_sha256
    assert (
        model.terminal_behavior.live_checkpoint_sha256
        == terminal_before
        == model.terminal_behavior.checkpoint_sha256
    )
    assert (
        model.terminal_behavior.list_map.live_checkpoint_sha256
        == list_before
        == model.terminal_behavior.list_map.checkpoint_sha256
    )
