from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_horizon2_int_beam as strict
from experiments.r3_pilot import public_typed_ast_horizon2_int_tier_beam as tier


def _artifact_ledger(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        task["name"]: {
            "decisions": tuple(
                (tuple(path), action) for path, action in task["actual_decisions"]
            ),
            "tokens": tuple(task["final_tokens"]),
            "outputs": tuple(task["final_outputs"]),
            "sequence_counts": tuple(
                decision["sequence_count"] for decision in task["decisions"]
            ),
        }
        for task in artifact["tasks"]
    }


def test_tier_variant_preserves_frozen_currents_and_has_no_host_top_k() -> None:
    assert tier.BEAM_WIDTH == strict.BEAM_WIDTH == 4
    assert tier._FIRST_PLY_MIN_CURRENT_PA == strict._FIRST_PLY_MIN_CURRENT_PA
    assert tier._FIRST_PLY_MAX_CURRENT_PA == strict._FIRST_PLY_MAX_CURRENT_PA
    assert (
        tier._FIRST_PLY_INHIBITION_CURRENT_PA
        == strict._FIRST_PLY_INHIBITION_CURRENT_PA
    )

    source = Path(tier.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_score_and_filter_first_ply",
        "plan_site_action_h2_tier_beam",
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
        "_physical_race",
    ):
        assert forbidden not in inference_source
    assert "selector.select(\n            scored_currents," in inference_source
    assert "stop_cell=len(predictions)" in inference_source


def test_public_tier_b4_matrix_once_against_committed_exhaustive_ledger() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_visible_probe.json"
    )
    artifact = json.loads(artifact_path.read_bytes())
    exhaustive = _artifact_ledger(artifact)
    exhaustive_sequence_count = sum(
        sum(record["sequence_counts"]) for record in exhaustive.values()
    )

    result = tier.run_public_matrix()
    summary = {
        "verdict": result.verdict,
        "passed_tasks": result.passed_tasks,
        "total_tasks": result.total_tasks,
        "first_predictions_scored": result.first_predictions_scored,
        "retained_first_candidates": result.retained_first_candidates,
        "first_ply_tier_count": result.first_ply_tier_count,
        "largest_first_ply_tier": result.largest_first_ply_tier,
        "second_ply_transition_expansions": result.second_ply_transition_expansions,
        "terminal_sequence_count": result.terminal_sequence_count,
        "exhaustive_sequence_count": exhaustive_sequence_count,
        "wall_time_seconds": result.wall_time_seconds,
        "exhaustive_recorded_wall_time_seconds": 114.0,
        "tasks": tuple(
            {
                "name": task.name,
                "passed": task.passed,
                "error": task.error,
                "decisions": task.actual_decisions,
                "plans": tuple(
                    {
                        "status": plan.status,
                        "error": plan.error,
                        "first": plan.first_prediction_count,
                        "retained": plan.retained_first_count,
                        "tiers": plan.tier_count,
                        "leading_tier": plan.leading_tier_size,
                        "expanded": plan.second_ply_transition_expansions,
                        "sequences": plan.terminal_sequence_count,
                    }
                    for plan in task.plans
                ),
            }
            for task in result.tasks
        ),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))

    assert result.total_tasks == 3
    assert result.first_predictions_scored > 0
    assert result.largest_first_ply_tier > 1
    assert all(task.checkpoint_immutable for task in result.tasks)
    if result.verdict == "PASS_TYPED_AST_HORIZON2_INT_TIER_BEAM":
        assert result.passed_tasks == 3
        assert 0 < result.terminal_sequence_count <= exhaustive_sequence_count
        assert result.second_ply_transition_expansions > 0
        for task in result.tasks:
            expected = exhaustive[task.name]
            assert task.actual_decisions == expected["decisions"]
            assert task.final_tokens == expected["tokens"]
            assert task.final_outputs == expected["outputs"]
    else:
        # This is a one-run structural successor.  A miss is banked here and
        # is not followed by B/current/weight/order or fixture repair.
        assert result.verdict == "STOP_TYPED_AST_HORIZON2_INT_TIER_BEAM"
        assert result.passed_tasks < 3
