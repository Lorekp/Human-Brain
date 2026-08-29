import ast
import inspect
from functools import cache

import numpy as np

from experiments.r3_pilot import public_typed_structure_successor as structure


@cache
def _trained() -> tuple[
    structure.FrozenStructureSuccessor,
    structure.StructureTrainingSummary,
]:
    return structure.train_public_structure_successor()


def _int_identity() -> structure.tp.Program:
    signature = structure.TypedSignature(
        "int_identity",
        (structure.ProgramParameter("value", structure.ProgramType.Int()),),
        structure.ProgramType.Int(),
    )
    return structure.tp.Program(("variable-0",), signature)


def _list_identity() -> structure.tp.Program:
    list_int = structure.ProgramType.List(structure.ProgramType.Int())
    signature = structure.TypedSignature(
        "list_identity",
        (structure.ProgramParameter("items", list_int),),
        list_int,
    )
    return structure.tp.Program(("variable-0",), signature)


def _assert_unique_and_frozen(
    model: structure.FrozenStructureSuccessor,
    prediction: structure.StructurePrediction,
) -> None:
    assert all(race.simultaneous_spike_candidates == 1 for race in prediction.races)
    assert prediction.checkpoint_before_sha256 == prediction.checkpoint_after_sha256
    assert prediction.checkpoint_after_sha256 == model.checkpoint_sha256
    assert model.live_checkpoint_sha256 == model.checkpoint_sha256


def test_root_wraps_compose_with_an_unseen_selected_subtree() -> None:
    model, summary = _trained()
    identity = _int_identity()
    observation = structure.observe_program_structure(identity)
    unary = model.predict(
        observation,
        structure.tp.FactorAction(
            (), "APPLY1", "absolute", "SELECTED", "NONE", None
        ),
    )
    binary = model.predict(
        observation,
        structure.tp.FactorAction(
            (), "APPLY2", "add", "SELECTED", "LITERAL", 2
        ),
    )
    binary_three = model.predict(
        observation,
        structure.tp.FactorAction(
            (), "APPLY2", "add", "SELECTED", "LITERAL", 3
        ),
    )

    unseen_add = structure.tp.Program(
        ("add", "variable-0", "integer", "+", "2", "end"),
        identity.signature,
    )
    composed = model.predict(
        structure.observe_program_structure(unseen_add),
        structure.tp.FactorAction(
            (), "APPLY1", "absolute", "SELECTED", "NONE", None
        ),
    )
    expected = structure.observe_program_structure(
        structure.tp.apply_factor_action(
            unseen_add,
            structure.tp.FactorAction(
                (), "APPLY1", "absolute", "SELECTED", "NONE", None
            ),
        )
    )

    assert summary.observations == 14
    assert summary.transition_rows == 11
    assert summary.local_rule_rows > 0
    assert unary.supported and binary.supported
    assert unary.predicted.site(()).production == "absolute"
    assert unary.predicted.site((0,)).production == "variable-0"
    assert unary.predicted.site((1,)).presence == structure.ABSENT
    assert binary.predicted.site(()).production == "add"
    assert binary.predicted.site((0,)).production == "variable-0"
    assert binary.predicted.site((1,)).production == "integer"
    assert binary_three.supported
    assert binary_three.predicted.paths == binary.predicted.paths
    assert composed.supported
    assert composed.predicted.paths == expected.paths
    assert composed.predicted.site(()).production == "absolute"
    assert composed.predicted.site((0,)).production == "add"
    assert composed.predicted.site((0, 0)).production == "variable-0"
    assert composed.predicted.site((0, 1)).production == "integer"
    _assert_unique_and_frozen(model, unary)
    _assert_unique_and_frozen(model, binary)
    _assert_unique_and_frozen(model, binary_three)
    _assert_unique_and_frozen(model, composed)


def test_two_scope_and_comparison_operand_orientations_are_structural() -> None:
    model, _ = _trained()
    two_int_signature = structure.TypedSignature(
        "two_scope",
        (
            structure.ProgramParameter("left", structure.ProgramType.Int()),
            structure.ProgramParameter("right", structure.ProgramType.Int()),
        ),
        structure.ProgramType.Int(),
    )
    two_int = structure.tp.Program(("variable-0",), two_int_signature)
    two_int_observation = structure.observe_program_structure(two_int)
    scope1_first_action = structure.tp.FactorAction(
        (), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None
    )
    scope0_first_action = structure.tp.FactorAction(
        (), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None
    )
    scope1_first = model.predict(two_int_observation, scope1_first_action)
    scope0_first = model.predict(two_int_observation, scope0_first_action)

    for prediction, action in (
        (scope1_first, scope1_first_action),
        (scope0_first, scope0_first_action),
    ):
        expected = structure.observe_program_structure(
            structure.tp.apply_factor_action(two_int, action)
        )
        assert prediction.supported
        assert prediction.predicted.paths == expected.paths
        _assert_unique_and_frozen(model, prediction)
    assert scope1_first.predicted.site((0,)).production == "variable-1"
    assert scope1_first.predicted.site((1,)).production == "variable-0"
    assert scope0_first.predicted.site((0,)).production == "variable-0"
    assert scope0_first.predicted.site((1,)).production == "variable-1"

    bool_signature = structure.TypedSignature(
        "literal_first_comparison",
        (structure.ProgramParameter("value", structure.ProgramType.Int()),),
        structure.ProgramType.Bool(),
    )
    comparison = structure.tp.Program(
        ("less-than", "variable-0", "integer", "+", "0", "end"),
        bool_signature,
    )
    literal_first_action = structure.tp.FactorAction(
        (), "APPLY2", "less-than", "LITERAL", "SCOPE0", 3
    )
    literal_first = model.predict(
        structure.observe_program_structure(comparison),
        literal_first_action,
    )
    literal_first_expected = structure.observe_program_structure(
        structure.tp.apply_factor_action(comparison, literal_first_action)
    )
    assert literal_first.supported
    assert literal_first.predicted.paths == literal_first_expected.paths
    assert literal_first.predicted.site((0,)).production == "integer"
    assert literal_first.predicted.site((1,)).production == "variable-0"
    _assert_unique_and_frozen(model, literal_first)


def test_map_bind_projects_binder_body_collection_and_scope_metadata() -> None:
    model, _ = _trained()
    identity = _list_identity()
    observation = structure.observe_program_structure(identity)
    map_action = structure.tp.FactorAction(
        (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
    )
    prediction = model.predict(
        observation,
        map_action,
    )

    root = prediction.predicted.site(())
    binder = prediction.predicted.site((0,))
    body = prediction.predicted.site((0, 0))
    collection = prediction.predicted.site((1,))
    assert (root.production, root.result_type, root.path_role) == (
        "map",
        "List[Int]",
        "root",
    )
    assert (binder.production, binder.result_type, binder.binder_marker) == (
        "bind",
        "internal-function",
        "binder",
    )
    assert (
        body.production,
        body.result_type,
        body.scope0_type,
        body.scope1_type,
        body.path_role,
    ) == ("variable-0", "Int", "Int", "List[Int]", "child-0")
    assert (
        collection.production,
        collection.result_type,
        collection.path_role,
    ) == ("variable-0", "List[Int]", "child-1")
    assert prediction.predicted.site((0, 1)).presence == structure.ABSENT
    assert prediction.supported
    assert not model.contacts.flags.writeable
    assert not model.contact_counts.flags.writeable
    assert all(not weights.flags.writeable for weights in model.output_weights)
    assert all(not bank.contacts.flags.writeable for bank in model.rule_banks)
    assert all(not bank.contact_counts.flags.writeable for bank in model.rule_banks)
    assert all(not bank.projection.flags.writeable for bank in model.rule_banks)
    _assert_unique_and_frozen(model, prediction)

    mapped = structure.tp.apply_factor_action(identity, map_action)
    selected_body_action = structure.tp.FactorAction(
        (0, 0), "APPLY1", "absolute", "SELECTED", "NONE", None
    )
    selected_body = model.predict(
        structure.observe_program_structure(mapped),
        selected_body_action,
    )
    selected_body_expected = structure.observe_program_structure(
        structure.tp.apply_factor_action(mapped, selected_body_action)
    )
    assert selected_body.supported
    assert selected_body.predicted.paths == selected_body_expected.paths
    _assert_unique_and_frozen(model, selected_body)

    body_action = structure.tp.FactorAction(
        (0, 0), "APPLY2", "add", "BOUND0", "LITERAL", 2
    )
    body_prediction = model.predict(
        structure.observe_program_structure(mapped),
        body_action,
    )
    body_expected = structure.observe_program_structure(
        structure.tp.apply_factor_action(mapped, body_action)
    )
    assert body_prediction.supported
    assert body_prediction.predicted.paths == body_expected.paths
    assert body_prediction.predicted.site(()).production == "map"
    assert body_prediction.predicted.site((0,)).production == "bind"
    assert body_prediction.predicted.site((0, 0)).production == "add"
    _assert_unique_and_frozen(model, body_prediction)


def test_unsupported_row_and_unknown_input_physically_propagate_unknown() -> None:
    model, _ = _trained()
    observation = structure.observe_program_structure(_int_identity())
    unsupported = model.predict(
        observation,
        structure.tp.FactorAction(
            (), "APPLY2", "add", "SCOPE0", "SCOPE0", None
        ),
    )

    assert not unsupported.supported
    assert unsupported.transition_race.winner == structure.UNKNOWN
    assert all(site.presence == structure.UNKNOWN for site in unsupported.predicted.paths)
    assert all(race.winner == structure.UNKNOWN for race in unsupported.races)
    _assert_unique_and_frozen(model, unsupported)

    propagated = model.predict(
        unsupported.predicted,
        structure.tp.FactorAction(
            (), "APPLY1", "absolute", "SELECTED", "NONE", None
        ),
    )
    assert not propagated.supported
    assert all(race.winner == structure.UNKNOWN for race in propagated.races)
    _assert_unique_and_frozen(model, propagated)


def test_prediction_cannot_reach_compiler_or_candidate_actuator(monkeypatch) -> None:
    model, _ = _trained()
    observation = structure.observe_program_structure(_int_identity())
    action = structure.tp.FactorAction(
        (), "APPLY1", "absolute", "SELECTED", "NONE", None
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inference reached the compiler-owned actuator")

    monkeypatch.setattr(structure.tp, "apply_factor_action", forbidden)
    monkeypatch.setattr(structure.tp, "program_edit_sites", forbidden)
    monkeypatch.setattr(structure.tp, "compile", forbidden)
    monkeypatch.setattr(structure.tp, "Program", forbidden)
    prediction = model.predict(observation, action)

    source = inspect.getsource(structure.FrozenStructureSuccessor.predict)
    ast.parse(inspect.cleandoc(source))
    for forbidden_name in (
        "apply_factor_action",
        "program_edit_sites",
        "replace_subtree",
        "tp.Program",
        "tp.compile",
    ):
        assert forbidden_name not in source
    assert prediction.supported
    assert np.count_nonzero(model.contacts) > 0
