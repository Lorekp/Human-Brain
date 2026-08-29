from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate as gate,
)
from humanbrain import typed_program as tp


def test_created_child_public_boundary_is_exactly_predeclared() -> None:
    assert gate.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
    assert tuple(
        (
            task.name,
            task.prompt,
            task.cases,
            task.incumbent_tokens,
            task.desired_outputs,
        )
        for task in gate.PUBLIC_TASKS
    ) == (
        (
            "left-created-child",
            "Subtract beta from the absolute value of alpha.",
            ((-7, 2), (7, 2), (-7, -3), (7, -3)),
            ("variable-1",),
            (5, 5, 10, 10),
        ),
        (
            "right-created-child",
            "Subtract the absolute value of beta from alpha.",
            ((2, -7), (2, 7), (-3, -7), (-3, 7)),
            ("variable-0",),
            (-5, -5, -10, -10),
        ),
    )
    assert tuple(
        (
            target.name,
            target.created_path,
            target.expected_first,
            target.expected_second,
            target.expected_tokens,
        )
        for target in gate.PUBLIC_TARGETS
    ) == (
        (
            "left-created-child",
            (0,),
            tp.FactorAction(
                (), "APPLY2", "subtract", "SELECTED", "SCOPE0", None
            ),
            tp.FactorAction(
                (0,), "APPLY1", "absolute", "SELECTED", "NONE", None
            ),
            ("subtract", "absolute", "variable-1", "variable-0"),
        ),
        (
            "right-created-child",
            (1,),
            tp.FactorAction(
                (), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
            ),
            tp.FactorAction(
                (1,), "APPLY1", "absolute", "SELECTED", "NONE", None
            ),
            ("subtract", "variable-1", "absolute", "variable-0"),
        ),
    )


def test_existing_artifact_guard_never_runs_public_matrix(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "banked.json"
    artifact.write_text(
        json.dumps({"verdict": gate.PASS_VERDICT}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", artifact)

    def forbidden_run():
        raise AssertionError("guarded public matrix reran")

    monkeypatch.setattr(gate, "run_public_matrix", forbidden_run)
    assert gate.main() == 0


def test_banked_created_child_stop_artifact_contract() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_factor_horizon2_int_created_child_gate.json"
    )
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)

    assert sha256(raw).hexdigest() == (
        "4f4164d031990770ac9e8573250ae91bd8fb22ed3bd3b93eadb0a9a116ef2c5e"
    )
    assert artifact["schema"] == gate.SCHEMA
    assert artifact["verdict"] == gate.STOP_VERDICT
    assert artifact["passed_tasks"] == 0
    assert artifact["attempted_tasks"] == 1
    assert artifact["total_tasks"] == 2
    assert artifact["literal_bank"] == list(range(-4, 5))
    assert artifact["model_checkpoint_sha256"] == (
        "e27a877ed192aa5e2767e10ac0eb7d84b2fd0230593ad22e4000e6f937b27d88"
    )
    assert artifact["world_checkpoint_sha256"] == (
        "43fe57aee76dce1bbdacb5d9104bbfdd40dd9da0609c481ed3632069b39912eb"
    )
    assert artifact["runner_source_lf_sha256"] == (
        "1a7ec5743430b2c1368f2b0c4c3509f248150bb00626e296826d250c9e662d15"
    )

    task = artifact["tasks"][0]
    assert task["name"] == "left-created-child"
    assert task["passed"] is False
    assert task["no_h1_all_case_solution"] is True
    assert task["h1_exact_actions"] == []
    assert task["initial_paths"] == [[]]
    assert task["created_path"] == [0]
    assert task["created_path_absent_initially"] is True
    assert task["first_actuation"] is None
    assert task["second_decision"] is None
    assert task["actual_trajectory"] == []
    assert task["final_tokens"] == ["variable-1"]
    assert task["checkpoint_immutable"] is True
    assert task["error"] == (
        "first H=2 plan stopped: all exact terminal sequences produced "
        "structural UNKNOWN"
    )

    decision = task["first_decision"]
    assert (
        decision["status"],
        decision["candidate_count"],
        decision["first_admissible_count"],
        decision["first_excluded_count"],
        decision["first_tier_count"],
        decision["expanded_first_count"],
        decision["unique_sequence_predictions"],
        decision["cumulative_sequence_evaluations"],
        decision["terminal_attempt_sequence_counts"],
    ) == (
        "STOP_TERMINAL_STRUCTURE",
        83,
        83,
        0,
        4,
        83,
        6_414,
        7_790,
        [4, 1_372, 6_414],
    )
    assert decision["checkpoint_immutable"] is True
