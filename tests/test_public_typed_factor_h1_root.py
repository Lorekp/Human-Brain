import inspect
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_factor_h1_root as planner


def test_banked_visible_a_minus_b_stop_ledger_is_exact() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_factor_h1_root_stop.json"
    )
    artifact = json.loads(path.read_bytes())

    assert artifact["verdict"] == "STOP_TYPED_FACTOR_H1_ROOT_VALUE_ALIAS"
    assert artifact["attempted_tasks"] == 1
    assert artifact["remaining_tasks_unrun"] == 3
    assert artifact["candidate_count"] == 212
    assert artifact["operator_predictions"] == 636
    assert artifact["physical_operator_races"] == 1257
    assert artifact["operator_races_unique"]
    assert tuple(item["candidate_cell"] for item in artifact["value_tie"]) == (73, 74)
    assert all(item["current_pa"] == 4_500.0 for item in artifact["value_tie"])
    assert artifact["stop_current_pa"] == 650.0
    assert not artifact["motor_ran"] and not artifact["actuation_ran"]
    assert artifact["structure_prediction_calls"] == 0


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
    assert "operator_successor.predict_actions_many" in source
    assert source.index("root_actions = tuple(") < source.index(
        "operator_successor.predict_actions_many"
    )
    assert source.index("operator_successor.predict_actions_many") < source.index(
        "if root_supported:"
    )
    assert "effects = ()" in source
    assert "select_unique" in source
    assert "motor_projection.resolve" in source
    assert source.count("structure_successor.predict(") == 1
    assert source.count("actuate_selected_factor_action(") == 1
    assert source.index("select_unique") < source.index("structure_successor.predict(")
    assert source.index("structure_successor.predict(") < source.index(
        "actuate_selected_factor_action("
    )
    assert planner.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
