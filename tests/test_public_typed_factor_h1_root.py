import inspect
from functools import cache

from experiments.r3_pilot import public_typed_factor_h1_root as planner
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


@cache
def _planner() -> planner.FrozenTypedFactorH1Root:
    return planner.build_public_typed_factor_h1_root()


def _two_int_incumbent() -> tp.Program:
    integer = ProgramType.Int()
    signature = TypedSignature(
        "solve",
        (
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        integer,
    )
    return tp.Program(("absolute", "variable-0"), signature)


def test_visible_a_minus_b_banks_the_exact_semantic_alias_tie() -> None:
    model = _planner()
    decision = model.plan(
        _two_int_incumbent(),
        planner.PUBLIC_TWO_INT_CASES,
        (5, -7, 0),
    )
    selected_alias = tp.FactorAction(
        (), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
    )
    scope_binding = tp.FactorAction(
        (), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None
    )
    alias_cell = decision.candidate_bank.actions.index(selected_alias)
    binding_cell = decision.candidate_bank.actions.index(scope_binding)
    top_current = decision.diagnostic_currents_pa[alias_cell]
    tied = tuple(
        action
        for action, current in zip(
            decision.candidate_bank.actions,
            decision.diagnostic_currents_pa[:-1],
            strict=True,
        )
        if current == top_current
    )

    assert decision.status == planner.STATUS_STOP_VALUE_TIE
    assert decision.stop_reason == "race row 0 produced 2 simultaneous first spikes"
    assert tied == (selected_alias, scope_binding)
    assert decision.diagnostic_currents_pa[binding_cell] == top_current == 4_500.0
    assert decision.diagnostic_currents_pa[-1] == 650.0
    assert len(decision.candidate_bank.actions) == 212
    assert decision.value_decision is None
    assert decision.motor_projection is None and decision.motor_decision is None
    assert decision.selected_structure is None and decision.selected_program is None
    assert decision.final_observation is None and decision.final_structure is None
    assert decision.structure_prediction_calls == 0
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256
    assert decision.checkpoint_after_sha256 == model.checkpoint_sha256
    assert model.live_checkpoint_sha256 == model.checkpoint_sha256
    assert not decision.value_projection.weights_pa.flags.writeable
    assert not decision.value_projection.bias_pa.flags.writeable
    assert len(decision.candidate_predictions) == len(decision.candidate_bank.actions)
    assert all(
        race.simultaneous_spike_candidates == 1
        for candidate in decision.candidate_predictions
        for prediction in candidate.operator_predictions
        for race in prediction.races
    )
    assert all(
        output.unknown_active
        for candidate in decision.candidate_predictions
        if not candidate.root_site_supported
        for output in candidate.scored_outputs
    )


def test_inference_has_no_hypothetical_program_or_candidate_structure_loop() -> None:
    source = inspect.getsource(planner.FrozenTypedFactorH1Root.plan)

    for forbidden in (
        "tp.Program(",
        "tp.apply_factor_action",
        "replace_subtree",
        "tp.compile",
        "exec(",
        "sorted(",
        ".sort(",
        "argmax",
        "np.max",
    ):
        assert forbidden not in source
    assert "factor_action_candidates" in source
    assert "operator_successor.predict_many" in source
    assert "select_unique" in source
    assert "motor_projection.resolve" in source
    assert source.count("structure_successor.predict(") == 1
    assert source.count("actuate_selected_factor_action(") == 1
    assert source.index("select_unique") < source.index("structure_successor.predict(")
    assert source.index("structure_successor.predict(") < source.index(
        "actuate_selected_factor_action("
    )
    assert planner.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
