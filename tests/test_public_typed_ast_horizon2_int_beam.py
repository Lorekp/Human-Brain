from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_ast_horizon2_int_beam as beam


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


def test_beam_inference_has_no_host_top_k_or_candidate_actuation() -> None:
    source = Path(beam.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_score_and_filter_first_ply",
        "_expand_retained_second_ply",
        "_score_terminal_sequences",
        "plan_site_action_h2_beam",
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
    ):
        assert forbidden not in inference_source


def test_public_b4_matrix_gate_against_committed_exhaustive_ledger() -> None:
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

    result = beam.run_public_matrix()
    summary = {
        "verdict": result.verdict,
        "passed_tasks": result.passed_tasks,
        "total_tasks": result.total_tasks,
        "first_predictions_scored": result.first_predictions_scored,
        "retained_first_candidates": result.retained_first_candidates,
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

    if result.verdict == "PASS_TYPED_AST_HORIZON2_INT_BEAM":
        assert result.passed_tasks == result.total_tasks == 3
        assert result.terminal_sequence_count * 2 < exhaustive_sequence_count
        assert result.second_ply_transition_expansions < exhaustive_sequence_count
        for task in result.tasks:
            expected = exhaustive[task.name]
            assert task.actual_decisions == expected["decisions"]
            assert task.final_tokens == expected["tokens"]
            assert task.final_outputs == expected["outputs"]
            assert task.checkpoint_immutable
    else:
        # A strict physical stop is the banked outcome; never repair the B,
        # currents, weights, or candidate order after this gate opens.
        assert result.verdict == "STOP_TYPED_AST_HORIZON2_INT_BEAM"
        assert result.passed_tasks < result.total_tasks
        stopped_plans = tuple(
            plan
            for task in result.tasks
            for plan in task.plans
            if plan.status in {
                beam.STOP_FIRST_PLY_BEAM_RACE,
                beam.STOP_TERMINAL_SEQUENCE_RACE,
            }
        )
        assert stopped_plans
        assert all(plan.error and "simultaneous" in plan.error for plan in stopped_plans)
        assert all(task.checkpoint_immutable for task in result.tasks)
