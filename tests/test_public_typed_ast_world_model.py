import ast
import inspect
from functools import cache
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_world_model as world


@cache
def _trained() -> tuple[
    world.FrozenAstWorldModel,
    world.AstWorldTrainingSummary,
]:
    return world.train_public_ast_world_model()


def _identity() -> world.tp.Program:
    return world.total.banked.banked._identity_program()


def test_root_wrap_prediction_matches_selected_actuator() -> None:
    model, summary = _trained()
    arguments = (-8, -1, 4, 11)
    incumbent = _identity()
    observation = world.observe_current_program(incumbent, arguments)

    predicted = model.predict(observation, (), "WRAP_ABS")
    selected = world.apply_selected_wrap(incumbent, (), "WRAP_ABS")
    actual = world.observe_current_program(selected, arguments).site(()).outputs

    assert summary.target_free_observations == 410
    assert predicted.predicted_root_outputs == actual == (8, 1, 4, 11)
    assert predicted.ancestor_actions == ()
    assert predicted.supported
    assert all(race.simultaneous_spike_candidates == 1 for race in predicted.races)
    assert model.live_checkpoint_sha256 == model.checkpoint_sha256


def test_child_path_recomposes_literal_and_absolute_ancestors() -> None:
    model, _ = _trained()
    arguments = (-8, -1, 4, 9)
    incumbent = world.tp.Program(
        (
            "absolute",
            "add",
            "variable-0",
            "integer",
            "+",
            "2",
            "end",
        ),
        _identity().signature,
    )
    observation = world.observe_current_program(incumbent, arguments)

    predicted = model.predict(observation, (0, 0), "WRAP_SUB[1]")
    selected = world.apply_selected_wrap(incumbent, (0, 0), "WRAP_SUB[1]")
    actual = world.observe_current_program(selected, arguments).site(()).outputs

    assert predicted.ancestor_actions == ("WRAP_ADD[2]", "WRAP_ABS")
    assert predicted.predicted_root_outputs == actual == (7, 0, 5, 10)
    assert predicted.supported
    assert len(predicted.races) == len(arguments) * 3


def test_unseen_transition_and_unary_irreducible_branch_return_unknown() -> None:
    model, _ = _trained()
    identity_observation = world.observe_current_program(_identity(), (21,))

    unseen = model.predict(identity_observation, (), "WRAP_ABS")

    assert unseen.predicted_root_outputs == (world.UNKNOWN,)
    assert unseen.races[0].winner == "UNKNOWN"
    assert not unseen.supported

    subtract = world.tp.Program(
        (
            "subtract",
            "variable-0",
            "integer",
            "+",
            "2",
            "end",
        ),
        _identity().signature,
    )
    observation = world.observe_current_program(subtract, (-4, 5))
    unsupported = model.predict(observation, (1,), "WRAP_ABS")

    assert unsupported.predicted_root_outputs == (
        world.UNKNOWN,
        world.UNKNOWN,
    )
    assert unsupported.ancestor_actions == (None,)
    assert all(race.winner == "UNKNOWN" for race in unsupported.ancestors[0].races)
    assert not unsupported.supported
    assert "right-child subtraction" in str(unsupported.unsupported_reason)


def test_prediction_has_no_candidate_construction_or_execution() -> None:
    source = Path(world.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    predict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    predict_source = ast.get_source_segment(source, predict)
    assert predict_source is not None

    for forbidden in (
        "replace_subtree",
        "_apply_action",
        "_execute",
        "tp.Program(",
        "tp.compile",
    ):
        assert forbidden not in predict_source
    assert "program_edit_sites" in inspect.getsource(world.observe_current_program)
    observation_source = inspect.getsource(world._observe_tokens)
    assert "_execute" in observation_source
    assert "_evaluate_expression" not in source
    actuator_source = inspect.getsource(world.apply_selected_wrap)
    assert actuator_source.count("replace_subtree") == 1
    assert "replace_subtree" not in inspect.getsource(world.FrozenAstWorldModel)
    assert "run_fresh" not in source
