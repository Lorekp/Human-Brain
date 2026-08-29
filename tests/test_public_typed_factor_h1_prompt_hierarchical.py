from __future__ import annotations

import ast
import inspect
import json
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import (
    public_typed_factor_h1_prompt_hierarchical as hierarchical,
)


def _canonical_lf_sha256(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_four_public_tasks_are_literal_and_order_frozen() -> None:
    tasks = hierarchical.PUBLIC_TASKS

    assert hierarchical.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
    assert tuple(task.prompt for task in tasks) == (
        "Return a minus b.",
        "Return b minus a.",
        "Return whether x is less than 2.",
        "Return whether 2 is less than x.",
    )
    assert tuple(task.cases for task in tasks) == (
        hierarchical.PUBLIC_TWO_INT_CASES,
        hierarchical.PUBLIC_TWO_INT_CASES,
        hierarchical.PUBLIC_BOOL_CASES,
        hierarchical.PUBLIC_BOOL_CASES,
    )
    assert tuple(task.incumbent_tokens for task in tasks) == (
        ("absolute", "variable-0"),
        ("absolute", "variable-0"),
        ("less-than", "variable-0", "variable-0"),
        ("less-than", "variable-0", "variable-0"),
    )
    assert tuple(task.desired_outputs for task in tasks) == (
        (5, -7, 0),
        (-5, 7, 0),
        (True, True, False, False),
        (False, False, False, True),
    )


def test_inference_is_value_tier_then_prompt_then_one_real_actuation() -> None:
    source = inspect.getsource(hierarchical.FrozenHierarchicalPromptH1.plan)

    for forbidden in (
        "tp.Program(",
        "tp.compile",
        "exec(",
        "expected_tokens",
        "PUBLIC_TARGETS",
        "sorted(",
        ".sort(",
        "argmax",
        "argpartition",
    ):
        assert forbidden not in source
    assert "factor_action_candidates" in source
    assert "predict_actions_many" in source
    assert "effects = ()" in source
    assert "CompiledRestingGlifTierBeam.compile" in source
    assert "beam_width=1" in source
    assert source.count("structure_successor.predict(") == 1
    assert source.count("actuate_selected_factor_action(") == 1
    assert source.index("selector.select(") < source.index(
        "FixedPromptBindingProjection.build("
    ) < source.index("_PROMPT_ACTION_RACE.race_many(") < source.index(
        "motor_projection.resolve("
    ) < source.index("structure_successor.predict(") < source.index(
        "actuate_selected_factor_action("
    )


def test_validation_target_is_read_only_after_inference() -> None:
    source = inspect.getsource(hierarchical.run_public_task)

    assert source.index("model.plan(") < source.index("PUBLIC_TARGETS")
    assert source.index("PUBLIC_TARGETS") < source.index("expected_tokens")


def test_spend_marker_is_exclusive_and_commits_all_four_tasks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "fresh.spend.json"
    monkeypatch.setattr(hierarchical, "_SPEND_MARKER_PATH", marker)
    commitment = hierarchical._fresh_commitment()

    assert hierarchical._acquire_spend_marker(commitment)
    assert not hierarchical._acquire_spend_marker(commitment)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert len(payload["commitment"]["tasks"]) == 4
    assert payload["commitment"]["literal_bank"] == list(range(-4, 5))
    assert payload["commitment_sha256"] == sha256(
        hierarchical._canonical_json(commitment)
    ).hexdigest()


def test_existing_artifact_guard_never_runs_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExistingArtifact:
        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def read_text(*, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps({"verdict": "BANKED"})

        def __str__(self) -> str:
            return "banked.json"

    monkeypatch.setattr(hierarchical, "_ARTIFACT_PATH", ExistingArtifact())
    monkeypatch.setattr(
        hierarchical,
        "run_public_matrix",
        lambda: (_ for _ in ()).throw(AssertionError("guard reran matrix")),
    )

    assert hierarchical.main() == 0


def test_banked_artifact_records_the_sole_four_task_run() -> None:
    artifact_raw = hierarchical._ARTIFACT_PATH.read_bytes()
    marker_raw = hierarchical._SPEND_MARKER_PATH.read_bytes()
    payload = json.loads(artifact_raw)
    marker = json.loads(marker_raw)

    assert _canonical_lf_sha256(hierarchical._ARTIFACT_PATH) == (
        "4ee553a90a9c54baaa92d65d731f7b8daec853dad0cad212bbd45c597c8d719a"
    )
    assert _canonical_lf_sha256(hierarchical._SPEND_MARKER_PATH) == (
        "8c2ba4791f4ad9dcb7feb05a0773a5ce9d63215529cb3bae58b23e13d91969ad"
    )
    assert payload["schema"] == hierarchical.SCHEMA
    assert payload["verdict"] == hierarchical.PASS_VERDICT
    assert payload["passed_tasks"] == payload["total_tasks"] == 4
    assert len(payload["tasks"]) == 4
    assert [task["name"] for task in payload["tasks"]] == [
        task.name for task in hierarchical.PUBLIC_TASKS
    ]
    assert [len(task["leading_value_tier_cells"]) for task in payload["tasks"]] == [
        2,
        2,
        2,
        3,
    ]
    assert [task["prompt_matching_tier_cells"] for task in payload["tasks"]] == [
        [1],
        [1],
        [1],
        [0],
    ]
    assert all(
        task["passed"]
        and task["status"] == hierarchical.STATUS_SELECTED
        and task["structure_prediction_calls"] == 1
        and task["actuator_calls"] == 1
        and task["operator_races_unique"]
        and task["prompt_race_unique"]
        and task["motor_races_unique"]
        and task["structure_races_unique"]
        and task["checkpoint_immutable"]
        and not task["value_stop_in_leading_tier"]
        and task["final_outputs"] == task["desired_outputs"]
        for task in payload["tasks"]
    )
    assert marker["commitment_sha256"] == payload["commitment_sha256"]
    assert marker["commitment"]["planner_source_sha256"] == sha256(
        Path(hierarchical.__file__).read_bytes()
    ).hexdigest()


def test_tests_do_not_call_the_spent_matrix() -> None:
    source = inspect.getsource(inspect.getmodule(test_tests_do_not_call_the_spent_matrix))
    tree = ast.parse(source)
    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))

    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_public_matrix"
        for call in calls
    )
