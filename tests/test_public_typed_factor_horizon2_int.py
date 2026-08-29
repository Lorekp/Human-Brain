from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


def _one_int_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("x", integer),),
        integer,
    )


def _two_int_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        integer,
    )


@pytest.fixture(scope="module")
def model() -> h2.FrozenFactorizedHorizon2Int:
    return h2.build_public_factorized_horizon2_int(literals=(1,))


def test_compiler_site_seam_matches_incumbent_and_map_body_contract() -> None:
    integer = ProgramType.Int()
    list_int = ProgramType.List(integer)
    signature = TypedSignature(
        "solve",
        (ProgramParameter("values", list_int),),
        list_int,
    )
    identity = tp.Program(("variable-0",), signature)
    mapped = tp.apply_factor_action(
        identity,
        tp.FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None),
    )
    sites = tp.factor_action_sites(mapped)
    actions = tp.factor_actions_for_sites(sites, literals=(2,))

    assert actions == tp.factor_action_candidates(mapped, literals=(2,)).actions
    body = next(site for site in sites if site.path == (0, 0))
    assert (
        body.result_type_label,
        body.head_production,
        body.scope_binding_type_labels,
        body.selected_scope_index,
    ) == ("Int", "variable-0", ("Int", "List[Int]"), 0)
    assert tp.FactorAction(
        (0, 0), "APPLY1", "absolute", "SELECTED", "NONE", None
    ) in actions
    assert tp.FactorAction(
        (0, 0), "APPLY2", "subtract", "SELECTED", "LITERAL", 2
    ) in actions
    assert not any(
        action.path == (0, 0)
        and action.left_source in {"SCOPE0", "BOUND0"}
        for action in actions
    )


def test_factorized_h2_predicts_two_plies_and_replans_after_one_actuation(
    model: h2.FrozenFactorizedHorizon2Int,
) -> None:
    cases = ((7, 2), (-3, 4), (5, 5), (0, -2))
    desired = (5, 7, 0, 2)
    prompt = "Return the absolute value of a minus b."
    incumbent = tp.Program(("absolute", "variable-0"), _two_int_signature())
    observation = model.observe(incumbent, cases)
    before = model.live_checkpoint_sha256

    decision = model.plan(observation, prompt, desired)

    expected_first = tp.FactorAction(
        (), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None
    )
    expected_second = tp.FactorAction(
        (), "APPLY1", "absolute", "SELECTED", "NONE", None
    )
    assert decision.status == h2.STATUS_SELECTED
    assert decision.candidate_actions == tp.factor_action_candidates(
        incumbent,
        literals=(1,),
    ).actions
    assert decision.selected_sequence == h2.FactorH2Sequence(
        expected_first,
        expected_second,
    )
    assert decision.selected_sequence.second.schema != "DONE"
    assert decision.unique_first_expansions == h2.BEAM_WIDTH
    assert decision.terminal_attempts[-1].exact_leading_sequence_cells == (
        decision.selected_sequence_cell,
    )

    # The learned first transition creates compiler-representable child sites;
    # their second-ply actions exist before any actuator touches the incumbent.
    hypothetical_site_changes = tuple(
        prediction
        for prediction in decision.predictions
        if prediction.sequence.first.path == ()
        and prediction.sequence.second.path in {(0,), (1,)}
    )
    assert hypothetical_site_changes
    assert all(
        prediction.sequence.second
        in model.world.candidate_actions(prediction.first_transition.predicted)
        for prediction in hypothetical_site_changes
    )

    actuation = model.actuate_and_reobserve(incumbent, decision)
    assert actuation.actuator_calls == 1
    assert actuation.program.tokens == (
        "subtract",
        "variable-1",
        "variable-0",
    )
    assert actuation.structure_parity
    assert actuation.action_site_parity
    assert actuation.root_output_parity

    replanned = model.plan(actuation.observation, prompt, desired)
    assert replanned.status == h2.STATUS_SELECTED
    assert replanned.selected_sequence is not None
    assert replanned.selected_sequence.first == expected_second
    assert replanned.selected_sequence.second.schema == "DONE"
    final = model.actuate_and_reobserve(actuation.program, replanned)
    assert final.actuator_calls == 1
    assert final.program.tokens == (
        "absolute",
        "subtract",
        "variable-1",
        "variable-0",
    )
    assert model.plan(final.observation, prompt, desired).status == h2.STATUS_DONE
    assert model.live_checkpoint_sha256 == before == model.checkpoint_sha256

    plan_source = inspect.getsource(h2.FrozenFactorizedHorizon2Int.plan)
    world_source = inspect.getsource(h2.FrozenFactorH2WorldModel.predict_transition)
    forbidden = ("apply_factor_action", "actuate_selected", "tp.Program(", "tp.compile(")
    assert not any(token in plan_source for token in forbidden)
    assert not any(token in world_source for token in forbidden)


def test_unsupported_int_ancestor_propagates_physical_unknown(
    model: h2.FrozenFactorizedHorizon2Int,
) -> None:
    program = tp.Program(
        (
            "if",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
            "absolute",
            "variable-0",
        ),
        _one_int_signature(),
    )
    observation = model.observe(program, ((-2,), (0,), (3,)))
    action = tp.FactorAction(
        (1,), "APPLY1", "absolute", "SELECTED", "NONE", None
    )
    before = model.live_checkpoint_sha256

    prediction = model.world.behavior.predict_root_many(observation, (action,))[0]

    assert not prediction.supported
    assert "outside compiler-representable Int reassembly" in str(
        prediction.unsupported_reason
    )
    assert all(not output.supported for output in prediction.root_outputs)
    assert prediction.races
    assert all(race.simultaneous_spike_candidates == 1 for race in prediction.races)
    assert model.live_checkpoint_sha256 == before == model.checkpoint_sha256
