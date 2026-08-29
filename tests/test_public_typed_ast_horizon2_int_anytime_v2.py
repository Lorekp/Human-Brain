from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime as v1
from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime_v2 as v2


EXHAUSTIVE_SEQUENCE_COUNT = 2_439
V2_RECORDED_SECONDS = 41.1713905


def _artifact_ledger(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        task["name"]: {
            "decisions": tuple(
                (tuple(path), action) for path, action in task["actual_decisions"]
            ),
            "tokens": tuple(task["final_tokens"]),
            "outputs": tuple(task["final_outputs"]),
        }
        for task in artifact["tasks"]
    }


def test_anytime_v2_uses_stop_frontier_and_physical_terminal_confidence() -> None:
    assert v2.BEAM_WIDTH == v1.BEAM_WIDTH == 4
    assert v2._FIRST_PLY_MIN_CURRENT_PA == v1._FIRST_PLY_MIN_CURRENT_PA
    assert v2._FIRST_PLY_MAX_CURRENT_PA == v1._FIRST_PLY_MAX_CURRENT_PA
    assert (
        v2._FIRST_PLY_INHIBITION_CURRENT_PA
        == v1._FIRST_PLY_INHIBITION_CURRENT_PA
    )

    source = Path(v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_sequence_count",
        "_enumerate_first_ply_through_stop",
        "plan_site_action_h2_anytime_v2",
    }
    inference_source = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in inference_names
    )
    for forbidden in (
        "sort(",
        "sorted(",
        "argsort",
        "argpartition",
        "argmax",
        "replace_subtree",
        "apply_selected_wrap",
        "observe_current_program",
        "tp.Program",
        "tp.compile",
        "exec(",
        "array_equal",
        "predicted_final_outputs ==",
    ):
        assert forbidden not in inference_source
    assert "CompiledStopBoundedRestingGlifTiers.compile" in inference_source
    assert "enumerator.enumerate(\n        scored_currents," in inference_source
    assert "stop_cell=len(predictions)" in inference_source
    assert "v1._expand_new_tier" in inference_source
    assert "v1._score_terminal_attempt" in inference_source


def test_banked_anytime_v2_pass_ledger_matches_exhaustive_decisions() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_path = (
        root
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_anytime_v2_pass.json"
    )
    exhaustive_path = (
        root
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_visible_probe.json"
    )
    banked = json.loads(artifact_path.read_bytes())
    exhaustive = _artifact_ledger(json.loads(exhaustive_path.read_bytes()))

    assert banked["verdict"] == "PASS_TYPED_AST_HORIZON2_INT_ANYTIME_V2"
    assert banked["passed_tasks"] == banked["total_tasks"] == 3
    assert banked["first_predictions_scored"] == 252
    assert banked["admissible_first_candidates"] == 198
    assert banked["excluded_first_candidates"] == 54
    assert banked["first_ply_tiers_enumerated"] == 41
    assert banked["first_ply_tiers_consumed"] == 22
    assert banked["first_candidates_expanded"] == 81
    assert banked["second_ply_transition_expansions"] == 730
    assert banked["unique_terminal_sequences"] == 738
    assert banked["terminal_sequence_evaluations"] == 828
    assert banked["terminal_race_attempts"] == 11
    assert banked["wall_time_seconds"] == V2_RECORDED_SECONDS
    assert banked["terminal_sequence_evaluations"] < EXHAUSTIVE_SEQUENCE_COUNT
    for task in banked["tasks"]:
        decisions = tuple(
            (tuple(path), action) for path, action in task["actual_decisions"]
        )
        assert decisions == exhaustive[task["name"]]["decisions"]
        assert all(plan["attempts"] >= 1 for plan in task["plans"])
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "v2.run_public_matrix()" not in test_source
    module_source = Path(v2.__file__).read_text(encoding="utf-8")
    runner = next(
        node
        for node in ast.walk(ast.parse(module_source))
        if isinstance(node, ast.FunctionDef) and node.name == "run_public_task"
    )
    runner_source = ast.get_source_segment(module_source, runner) or ""
    assert runner_source.index("plan_site_action_h2_anytime_v2(") < runner_source.index(
        "h2._validate_public_target(task)"
    )
