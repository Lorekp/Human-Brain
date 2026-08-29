import ast
import inspect
import json
from functools import cache
from pathlib import Path

import numpy as np

from experiments.r3_pilot import public_typed_horizon2_successor_planner as planner


ARTIFACT = Path(planner.__file__).with_name(
    "public_typed_horizon2_successor_visible_probe.json"
)


@cache
def _trained() -> tuple[planner.FrozenSuccessorModel, planner.SuccessorTrainingSummary]:
    return planner.train_public_successor()


def test_exact_value_projection_is_physical_one_hot() -> None:
    for value in (-32, -7, 0, 13, 32):
        spikes = planner.ExactValueCellEncoder.encode(value)
        assert spikes.shape == (65,)
        assert np.flatnonzero(spikes).tolist() == [value - planner.VALUE_MIN]
    assert len(planner.ExactValueCellEncoder.projection_sha256()) == 64


def test_target_free_local_successor_replays_unseen_composition_primitives() -> None:
    successor, summary = _trained()

    assert summary.observations == summary.local_updates == 410
    assert summary.learned_synapses == 410
    assert successor.weights_pa.flags.writeable is False
    assert successor.live_checkpoint_sha256 == successor.checkpoint_sha256
    for value, action, expected in (
        (-7, "WRAP_ADD[2]", -5),
        (-7, "WRAP_ABS", 7),
        (12, "WRAP_SUB[3]", 9),
        (0, "DONE", 0),
    ):
        predicted, race, _ = successor.predict(value, action)
        assert predicted == expected
        assert race.simultaneous_spike_candidates == 1
    assert successor.live_checkpoint_sha256 == successor.checkpoint_sha256


def test_value_is_all_case_coincidence_and_pair_spike_drives_motor_cells() -> None:
    two, _ = planner._coincidence_features((1, 2, 3), (1, 2, 4))
    three, _ = planner._coincidence_features((1, 2, 3), (1, 2, 3))
    assert np.count_nonzero(two) == 2
    assert np.count_nonzero(three) == 3

    projection = planner.FixedPairValueProjection.create(6)
    features = np.zeros(projection.weights_pa.shape, dtype=np.float64)
    done_done = planner.PAIR_LABELS.index("DONE->DONE")
    abs_done = planner.PAIR_LABELS.index("WRAP_ABS->DONE")
    abs_add_two = planner.PAIR_LABELS.index("WRAP_ABS->WRAP_ADD[2]")
    features[done_done, :, :, 0] = 1.0
    features[abs_done, :, :, 0] = 1.0
    features[abs_add_two, 1, :, 0] = 1.0
    value_currents = projection.currents(features)
    best_five_match = (
        planner._PAIR_BASE_CURRENT_PA
        + 5 * planner._TERMINAL_MATCH_PA
        + 6 * planner._INTERMEDIATE_MATCH_PA
    )
    assert value_currents[abs_add_two] > best_five_match
    value_race = planner._physical_race(
        value_currents, planner.PAIR_LABELS, bank="target-free-value-calibration"
    )
    assert value_race.winner == "DONE->DONE"

    currents = np.full(100, -1_000.0, dtype=np.float64)
    pair_cell = planner.PAIR_LABELS.index("WRAP_ABS->WRAP_ADD[2]")
    currents[pair_cell] = planner._PAIR_BASE_CURRENT_PA
    pair_race = planner._physical_race(currents, planner.PAIR_LABELS, bank="test-pair")
    motor_race = planner._motor_race_from_pair(pair_race)
    assert motor_race.winner == "WRAP_ABS"
    assert motor_race.simultaneous_spike_candidates == 1


def test_inference_planner_has_no_host_program_rollout_or_ranking() -> None:
    source = Path(planner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_horizon2"
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None

    assert "_apply_action" not in function_source
    assert "_execute" not in function_source
    assert "argmin" not in function_source
    assert "argmax" not in function_source
    assert "argsort" not in function_source
    assert "np.linalg" not in source
    assert "_coincidence_features(" in function_source
    assert "FixedPairValueProjection.create(" in function_source
    assert "_motor_race_from_pair(" in function_source
    assert len(planner.PAIR_BANK) == len(set(planner.PAIR_LABELS)) == 100
    assert planner.VISIBLE_GRIDS == (
        (-12, -7, -1, 2, 6, 11),
        (-10, -5, 0, 3, 7, 12),
    )
    assert not hasattr(planner, "FRESH_BOUNDARY")
    assert not hasattr(planner, "run_fresh_boundary")

    inference_helpers = (
        planner.plan_horizon2,
        planner.FrozenSuccessorModel.predict,
        planner.FrozenSuccessorModel.predict_many,
        planner._coincidence_features,
        planner.FixedPairValueProjection.currents,
        planner._motor_race_from_pair,
    )
    helper_source = "\n".join(inspect.getsource(helper) for helper in inference_helpers)
    for forbidden in ("_apply_action", "_execute", "argmin", "argmax", "argsort"):
        assert forbidden not in helper_source


def test_visible_artifact_banks_all_mixed_order_arms() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    tasks = result["tasks"]

    assert result["schema"] == "public-typed-horizon2-successor-visible-v1"
    assert result["verdict"] == "PASS_HORIZON2_SUCCESSOR_VISIBLE"
    assert result["passed_tasks"] == result["total_tasks"] == len(tasks) == 24
    assert result["cell_pair_predictions"] == 7_200
    assert result["successor_cell_races"] == 47_520
    assert result["selected_real_executions"] == 72
    assert all(task["passed"] for task in tasks)
    assert all(task["unique_races"] for task in tasks)
    assert all(task["successor_checkpoint_immutable"] for task in tasks)
    assert all(task["selected_actions"] == task["expected_actions"] for task in tasks)
    assert all(task["final_outputs"] == task["desired_outputs"] for task in tasks)
    assert all(task["final_tokens"] == task["target_tokens"] for task in tasks)
    assert {task["name"].split("@", 1)[0] for task in tasks} == {
        f"abs(x){sign}{magnitude}"
        for magnitude in (1, 2, 4)
        for sign in ("+", "-")
    } | {
        f"abs(x{sign}{magnitude})"
        for magnitude in (1, 2, 4)
        for sign in ("+", "-")
    }
