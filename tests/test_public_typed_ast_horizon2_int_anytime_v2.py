from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime as v1
from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime_v2 as v2


EXHAUSTIVE_SEQUENCE_COUNT = 2_439
EXHAUSTIVE_RECORDED_SECONDS = 114.0
TIER_STOP_SEQUENCE_COUNT = 506
TIER_STOP_RECORDED_SECONDS = 26.508549900005164
ANYTIME_V1_SEQUENCE_COUNT = 634
ANYTIME_V1_RECORDED_SECONDS = 30.418630200001644


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


def test_public_anytime_v2_matrix_once_against_committed_exhaustive_ledger() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_visible_probe.json"
    )
    artifact = json.loads(artifact_path.read_bytes())
    exhaustive = _artifact_ledger(artifact)

    result = v2.run_public_matrix()
    summary = {
        "verdict": result.verdict,
        "passed_tasks": result.passed_tasks,
        "total_tasks": result.total_tasks,
        "first_predictions_scored": result.first_predictions_scored,
        "admissible_first_candidates": result.admissible_first_candidates,
        "excluded_first_candidates": result.excluded_first_candidates,
        "first_ply_tiers_enumerated": result.first_ply_tiers_enumerated,
        "first_ply_tiers_consumed": result.first_ply_tiers_consumed,
        "first_candidates_expanded": result.first_candidates_expanded,
        "second_ply_transition_expansions": result.second_ply_transition_expansions,
        "unique_terminal_sequences": result.unique_terminal_sequences,
        "terminal_sequence_evaluations": result.terminal_sequence_evaluations,
        "terminal_race_attempts": result.terminal_race_attempts,
        "wall_time_seconds": result.wall_time_seconds,
        "baselines": {
            "exhaustive_sequences": EXHAUSTIVE_SEQUENCE_COUNT,
            "exhaustive_seconds": EXHAUSTIVE_RECORDED_SECONDS,
            "tier_stop_sequences": TIER_STOP_SEQUENCE_COUNT,
            "tier_stop_seconds": TIER_STOP_RECORDED_SECONDS,
            "anytime_v1_sequences": ANYTIME_V1_SEQUENCE_COUNT,
            "anytime_v1_seconds": ANYTIME_V1_RECORDED_SECONDS,
        },
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
                        "admissible": plan.admissible_first_candidates,
                        "excluded": plan.excluded_first_candidates,
                        "tiers_total": plan.total_tier_count,
                        "tiers_used": plan.consumed_tier_count,
                        "expanded_first": plan.expanded_first_candidates,
                        "expanded_second": plan.second_ply_transition_expansions,
                        "unique_sequences": plan.unique_terminal_sequences,
                        "admissible_sequences": plan.admissible_sequence_count,
                        "exhaustive_sequences": plan.exhaustive_sequence_count,
                        "sequence_evaluations": plan.terminal_sequence_evaluations,
                        "attempts": plan.terminal_attempt_count,
                        "all_exact": plan.selected_all_cases_exact,
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
    assert result.first_ply_tiers_enumerated >= result.first_ply_tiers_consumed
    assert all(task.checkpoint_immutable for task in result.tasks)
    if result.verdict == "PASS_TYPED_AST_HORIZON2_INT_ANYTIME_V2":
        assert result.passed_tasks == 3
        assert result.excluded_first_candidates > 0
        assert result.terminal_sequence_evaluations < EXHAUSTIVE_SEQUENCE_COUNT
        assert result.unique_terminal_sequences <= result.terminal_sequence_evaluations
        retry_plans = tuple(
            plan
            for task in result.tasks
            for plan in task.plans
            if plan.terminal_attempt_count > 1
        )
        assert retry_plans
        assert all(
            plan.terminal_sequence_evaluations > plan.unique_terminal_sequences
            and plan.second_ply_transition_expansions
            <= plan.unique_terminal_sequences
            for plan in retry_plans
        )
        for task in result.tasks:
            expected = exhaustive[task.name]
            assert task.actual_decisions == expected["decisions"]
            assert task.final_tokens == expected["tokens"]
            assert task.final_outputs == expected["outputs"]
            assert all(plan.selected_all_cases_exact for plan in task.plans)
    else:
        assert result.verdict == "STOP_TYPED_AST_HORIZON2_INT_ANYTIME_V2"
        assert result.passed_tasks < 3
        exhausted = tuple(
            plan
            for task in result.tasks
            for plan in task.plans
            if plan.status == v2.STOP_ADMISSIBLE_EXHAUSTED
        )
        for plan in exhausted:
            assert (
                plan.expanded_first_candidates == plan.admissible_first_candidates
            )
            assert plan.unique_terminal_sequences == plan.admissible_sequence_count
