from __future__ import annotations

from types import SimpleNamespace

import pytest
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_list_map_behavior as list_behavior
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
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


class _PromptSourceState:
    def __init__(self, program: tp.Program) -> None:
        self.sites = tuple(
            SimpleNamespace(site=site) for site in tp.factor_action_sites(program)
        )

    def site(self, path: tuple[int, ...]) -> SimpleNamespace:
        return next(item for item in self.sites if item.site.path == path)


def test_prompt_selected_source_subtracts_map_binder_depth() -> None:
    integer = ProgramType.Int()
    list_int = ProgramType.List(integer)
    signature = TypedSignature(
        "solve",
        (
            ProgramParameter("values", list_int),
            ProgramParameter("offset", integer),
        ),
        list_int,
    )
    program = tp.Program(
        ("map", "bind", "variable-1", "variable-1"),
        signature,
    )
    state = _PromptSourceState(program)
    body = next(
        item.site
        for item in state.sites
        if item.site.selected_scope_index == 1
        and any(
            parent.site.is_binder
            and len(item.site.path) > len(parent.site.path)
            and item.site.path[: len(parent.site.path)] == parent.site.path
            for parent in state.sites
        )
    )
    collection = next(
        item.site
        for item in state.sites
        if item.site.path != body.path
        and item.site.selected_scope_index == 1
    )

    body_action = tp.FactorAction(
        body.path, "APPLY1", "absolute", "SELECTED", "NONE", None
    )
    collection_action = tp.FactorAction(
        collection.path, "MAP_BIND", "map", "BOUND0", "SELECTED", None
    )
    assert h2._prompt_selected_source(state, body_action) == "SCOPE0"
    assert h2._prompt_selected_source(state, collection_action) == "SCOPE1"


@pytest.mark.parametrize(
    ("operator", "literal", "prompt", "source_operator"),
    (
        (
            "add",
            4,
            "Return a new list containing every item plus 4.",
            "+",
        ),
        (
            "subtract",
            2,
            "Return a new list with two subtracted from every value.",
            "-",
        ),
    ),
)
def test_materialized_map_body_then_done_projects_from_container_prompt(
    operator: str,
    literal: int,
    prompt: str,
    source_operator: str,
) -> None:
    signature = _signature()
    identity = tp.Program(("variable-0",), signature)
    map_action = next(
        action
        for action in tp.factor_action_candidates(identity, literals=(literal,)).actions
        if action.schema == "MAP_BIND"
        and action.left_source == "BOUND0"
        and action.right_source == "SELECTED"
    )
    mapped = tp.apply_factor_action(identity, map_action)
    body_action = next(
        action
        for action in tp.factor_action_candidates(mapped, literals=(literal,)).actions
        if action.path == (0, 0)
        and action.schema == "APPLY2"
        and action.operator == operator
        and action.left_source == "SELECTED"
        and action.right_source == "LITERAL"
        and action.literal == literal
    )
    completed = tp.apply_factor_action(mapped, body_action)
    done = next(
        action
        for action in tp.factor_action_candidates(completed, literals=(literal,)).actions
        if action.schema == "DONE"
    )
    evidence = prompt_factors.PromptSequenceEvidence(
        first_action=body_action,
        first_selected_source=h2._prompt_selected_source(
            _PromptSourceState(mapped), body_action
        ),
        second_action=done,
        second_selected_source=None,
    )
    observation = prompt_factors.observe_prompt_compositions(
        prompt, signature
    )

    projection = prompt_factors.FixedPromptSequenceProjection.build((evidence,))
    decision = projection.project(prompt, signature)

    assert tp.compile(completed).source == (
        f"def solve(values):\n"
        f"    return [item_0 {source_operator} {literal} for item_0 in values]\n"
    )
    assert evidence.first_selected_source == "BOUND0"
    assert observation.supported
    assert observation.pattern is not None
    assert observation.pattern.container_schema == "MAP_BIND"
    assert decision.status == prompt_factors.PROJECTION_STATUS_PROJECTED
    assert decision.matching_sequence_cells == (0,)


def test_scalar_path_coincidence_is_not_a_materialized_map_body() -> None:
    signature = TypedSignature(
        "solve_scalar",
        (ProgramParameter("value", ProgramType.Int()),),
        ProgramType.Int(),
    )
    scalar = tp.Program(("absolute", "absolute", "variable-0"), signature)
    body_action = next(
        action
        for action in tp.factor_action_candidates(scalar, literals=(4,)).actions
        if action.path == (0, 0)
        and action.schema == "APPLY2"
        and action.operator == "add"
        and action.left_source == "SELECTED"
        and action.right_source == "LITERAL"
        and action.literal == 4
    )
    completed = tp.apply_factor_action(scalar, body_action)
    done = next(
        action
        for action in tp.factor_action_candidates(completed, literals=(4,)).actions
        if action.schema == "DONE"
    )
    evidence = prompt_factors.PromptSequenceEvidence(
        first_action=body_action,
        first_selected_source=h2._prompt_selected_source(
            _PromptSourceState(scalar), body_action
        ),
        second_action=done,
        second_selected_source=None,
    )

    projection = prompt_factors.FixedPromptSequenceProjection.build((evidence,))

    assert evidence.first_selected_source == "SCOPE0"
    assert projection.patterns == ()


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
    assert h2._prompt_selected_source(observation, MAP_IDENTITY) == "SCOPE0"
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
    assert mapped.applied_action == MAP_IDENTITY
    assert mapped.program.tokens == (
        "map",
        "bind",
        "variable-0",
        "variable-0",
    )
    assert mapped.structure_parity
    assert mapped.action_site_parity
    assert mapped.root_output_parity
    assert h2._prompt_selected_source(
        mapped.observation, SUBTRACT_THREE_BODY
    ) == "BOUND0"

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
    assert final.applied_action == SUBTRACT_THREE_BODY
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
