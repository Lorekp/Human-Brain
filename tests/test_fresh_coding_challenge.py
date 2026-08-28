from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from experiments.r3_pilot import fresh_coding_challenge as challenge

ROOT = Path(__file__).resolve().parents[1]


def test_tasks_are_uniform_and_absent_from_frozen_corpus() -> None:
    tasks = challenge.task_specs()
    challenge._validate_task_specs(tasks)

    corpus = challenge.checkpoint_spec.load_corpus(challenge.CORPUS_PATH)
    corpus_tasks = {example["task"] for example in corpus["examples"]}
    assert all(task.text not in corpus_tasks for task in tasks)
    assert all(len(case.args) == 3 for task in tasks for case in task.visible_cases)

def test_oracles_cover_ties_state_and_negative_rotation() -> None:
    assert challenge._oracle_nearest_prefix([8, 3, 5, 4], 4, 3) == 3
    assert challenge._oracle_running_clamped_maxima([1, 7, 3, 12], 0, 9) == [
        1,
        7,
        7,
        9,
    ]
    assert challenge._oracle_cyclic_rotation_prefix([1, 2, 3, 4], -1, 6) == [
        4,
        1,
        2,
        3,
        4,
        1,
    ]


def test_prompts_expose_only_task_candidate_and_raw_feedback() -> None:
    task = challenge.task_specs()[0]
    direct = challenge._direct_prompt(task)
    raw = '{"cases":[{"actual":0,"args":[[],2,3],"expected":[],"passed":false}],"passed":false}'
    source = "def solve(arg0, arg1, arg2):\n    return arg0\n"
    repair = challenge._repair_prompt(task, source, raw)

    assert direct.endswith(f"{challenge.checkpoint_spec.SOURCE_OPEN}\n")
    assert challenge.checkpoint_spec.FEEDBACK_OPEN not in direct
    assert raw in repair
    assert "parseable" not in repair
    assert "MAP" not in repair and "BUILD" not in repair
    assert challenge._extract_source(source + challenge.checkpoint_spec.SOURCE_END) == (
        source,
        True,
    )
    malformed = f"{challenge.checkpoint_spec.SOURCE_OPEN}\n{source} trailing"
    assert challenge._extract_source(malformed) == (malformed, False)


def _fake_result(cases: Any, passed_count: int) -> dict[str, Any]:
    rows = []
    for index, case in enumerate(cases):
        passed = index < passed_count
        rows.append(
            {
                "args": list(case.args),
                "expected": case.expected,
                "actual": case.expected if passed else None,
                "passed": passed,
            }
        )
    return {"cases": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def test_arm_keeps_last_candidate_instead_of_best_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = challenge.task_specs()[0]
    sources = [
        f"def solve(arg0, arg1, arg2):\n    return {index}\n" for index in range(3)
    ]
    passed_counts = iter((0, 2, 1))

    def fake_candidate(**kwargs: Any) -> tuple[str, None, bool]:
        return sources[kwargs["attempt"]], None, False

    def fake_execute(
        source: str, cases: Any, worker_bytes: bytes, executable: Path
    ) -> dict[str, Any]:
        del source, worker_bytes, executable
        return _fake_result(cases, next(passed_counts))

    monkeypatch.setattr(challenge, "_candidate_for_arm", fake_candidate)
    monkeypatch.setattr(challenge, "_execute_cases", fake_execute)
    result = challenge._run_arm(
        arm="cell",
        task=task,
        model=object(),  # type: ignore[arg-type]
        inventory=(),
        worker_bytes=b"worker",
        executable=Path("python"),
    )

    assert result["final_attempt"] == 2
    assert result["final_source"] == sources[2]
    assert result["attempts"][1]["visible_passed_cases"] == 2
    assert result["attempts"][2]["visible_passed_cases"] == 1
    assert result["selection_rule"] == "last_attempted_candidate_not_best_of"


def test_fail_fast_and_hidden_gate_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [
        {
            "task_id": "fresh_01_affine_list",
            "arms": {"cell": {"productive_visible_signal": False}},
        },
        {
            "task_id": "fresh_02_nearest_prefix",
            "arms": {"cell": {"productive_visible_signal": False}},
        },
    ]
    assert challenge._fail_fast_status(records)["triggered"] is True
    records[1]["arms"]["cell"]["productive_visible_signal"] = True
    assert challenge._fail_fast_status(records)["triggered"] is False

    missing_seal = tmp_path / "missing-seal.json"
    monkeypatch.setattr(challenge, "SEAL_PATH", missing_seal)

    def hidden_must_not_open(task_id: str) -> tuple[challenge.Case, ...]:
        raise AssertionError(f"hidden cases opened before seal for {task_id}")

    monkeypatch.setattr(challenge, "_hidden_cases", hidden_must_not_open)
    with pytest.raises(RuntimeError, match="durable completion seal"):
        challenge._score_hidden(
            task=challenge.task_specs()[0],
            task_record={"arms": {}},
            worker_bytes=b"worker",
            executable=Path("python"),
            expected_seal_sha256="0" * 64,
            expected_seal_payload={},
        )


def test_worker_schema_is_strict_and_source_does_not_import_protected_tasks() -> None:
    case = challenge._case(([1], 2, 3), [5])
    result = {
        "cases": [
            {"actual": [5], "args": [[1], 2, 3], "expected": [5], "passed": True}
        ],
        "passed": True,
    }
    encoded = challenge._canonical_json(result).encode()
    assert challenge._decode_worker_result(encoded, (case,)) == result
    with pytest.raises(RuntimeError, match="envelope"):
        challenge._decode_worker_result(b'{"cases":[],"extra":1,"passed":false}', ())

    challenge._validate_source_policy(
        "def solve(arg0, arg1, arg2):\n    return arg0\n"
    )
    with pytest.raises(ValueError, match="Import"):
        challenge._validate_source_policy(
            "def solve(arg0, arg1, arg2):\n    import os\n    return arg0\n"
        )
    with pytest.raises(challenge.CompletionPreflightError, match="signature_error"):
        challenge._validate_source_policy("def solve(arg0):\n    return arg0\n")

    tree = ast.parse(
        (ROOT / "experiments" / "r3_pilot" / "fresh_coding_challenge.py").read_text(
            encoding="utf-8"
        )
    )
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert "humanbrain.simple_swe.tasks" not in imported
    assert "humanbrain.simple_swe.taskset" not in imported
    assert "humanbrain.simple_swe.evaluator" not in imported
