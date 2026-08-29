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


def test_factorized_h2_reports_the_new_direct_child_orientation_ambiguity(
    model: h2.FrozenFactorizedHorizon2Int,
) -> None:
    cases = ((7, 2), (-3, 4), (5, 5), (0, -2))
    desired = (5, 7, 0, 2)
    prompt = "Return the absolute value of a minus b."
    incumbent = tp.Program(("absolute", "variable-0"), _two_int_signature())
    observation = model.observe(incumbent, cases)
    before = model.live_checkpoint_sha256

    decision = model.plan(observation, prompt, desired)

    expected_sequences = {
        h2.FactorH2Sequence(
            tp.FactorAction(
                (0,), "APPLY2", "subtract", "SELECTED", "SCOPE1", None
            ),
            tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
        ),
        h2.FactorH2Sequence(
            tp.FactorAction(
                (0,), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
            ),
            tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
        ),
    }
    assert decision.status == h2.STATUS_STOP_PROMPT
    assert decision.stop_reason == (
        "prompt evidence did not uniquely resolve the leading structure tier"
    )
    assert decision.candidate_actions == tp.factor_action_candidates(
        incumbent,
        literals=(1,),
    ).actions
    assert decision.unique_first_expansions == h2.BEAM_WIDTH
    assert decision.selected_sequence is None
    assert decision.selected_sequence_cell is None
    assert decision.motor_decision is None

    attempt = decision.terminal_attempts[-1]
    assert set(
        decision.predictions[cell].sequence
        for cell in attempt.structure_leading_sequence_cells
    ) == expected_sequences
    assert attempt.exact_leading_sequence_cells == (
        attempt.structure_leading_sequence_cells
    )
    assert all(
        decision.predictions[cell].first_transition.supported
        for cell in attempt.structure_leading_sequence_cells
    )
    assert attempt.prompt_race is not None
    assert attempt.prompt_race.simultaneous_spike_candidates == 1
    assert attempt.prompt_race.winner_cell == len(
        attempt.structure_leading_sequence_cells
    )
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
