from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime as anytime
from experiments.r3_pilot import public_typed_ast_horizon2_int_tier_beam as tier


ANYTIME_RECORDED_SECONDS = 30.418630200001644


def test_anytime_inference_uses_physical_uncertainty_not_host_ranking() -> None:
    assert anytime.BEAM_WIDTH == tier.BEAM_WIDTH == 4
    assert anytime._FIRST_PLY_MIN_CURRENT_PA == tier._FIRST_PLY_MIN_CURRENT_PA
    assert anytime._FIRST_PLY_MAX_CURRENT_PA == tier._FIRST_PLY_MAX_CURRENT_PA
    assert (
        anytime._FIRST_PLY_INHIBITION_CURRENT_PA
        == tier._FIRST_PLY_INHIBITION_CURRENT_PA
    )

    source = Path(anytime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_enumerate_first_ply_tiers",
        "_cached_sequence_bank",
        "_expand_new_tier",
        "_score_terminal_attempt",
        "plan_site_action_h2_anytime",
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
    assert "beam_width=len(predictions)" in inference_source
    assert "selector.select(\n            scored_currents," in inference_source
    assert "stop_cell=len(predictions)" in inference_source
    assert "np.count_nonzero(coincidences" in inference_source
    assert "all(spike_count == 1" in inference_source


def test_banked_anytime_stop_ledger_is_exact_and_does_not_rerun_matrix() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_anytime_stop.json"
    )
    artifact = json.loads(artifact_path.read_bytes())

    assert artifact["verdict"] == "STOP_TYPED_AST_HORIZON2_INT_ANYTIME"
    assert artifact["passed_tasks"] == 0
    assert artifact["total_tasks"] == 3
    assert artifact["first_predictions_scored"] == 160
    assert artifact["first_candidates_expanded"] == 58
    assert artifact["second_ply_transition_expansions"] == 540
    assert artifact["unique_terminal_sequences"] == 544
    assert artifact["terminal_sequence_evaluations"] == 634
    assert artifact["terminal_race_attempts"] == 6
    assert artifact["wall_time_seconds"] == ANYTIME_RECORDED_SECONDS
    assert tuple(task["name"] for task in artifact["tasks"]) == (
        "absolute-then-add",
        "add-then-absolute",
        "child-then-root-replan",
    )
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_public_matrix"
        for call in ast.walk(ast.parse(test_source))
        if isinstance(call, ast.Call)
    )
    module_source = Path(anytime.__file__).read_text(encoding="utf-8")
    runner = next(
        node
        for node in ast.walk(ast.parse(module_source))
        if isinstance(node, ast.FunctionDef) and node.name == "run_public_task"
    )
    runner_source = ast.get_source_segment(module_source, runner) or ""
    assert runner_source.index("plan_site_action_h2_anytime(") < runner_source.index(
        "h2._validate_public_target(task)"
    )
