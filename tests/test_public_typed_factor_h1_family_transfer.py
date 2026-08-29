from __future__ import annotations

import ast
import inspect
import json
from hashlib import sha256

import pytest

from experiments.r3_pilot import public_typed_factor_h1_family_transfer as transfer


def _lf_sha256(raw: bytes) -> str:
    return sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def test_transfer_cohort_is_exactly_frozen() -> None:
    tasks = transfer.PUBLIC_TASKS

    assert transfer.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
    assert tuple(task.prompt for task in tasks) == (
        "Subtract beta from alpha.",
        "Subtract alpha from beta.",
        "Return whether datum < -2.",
        "Return whether -2 < datum.",
    )
    assert tuple(task.cases for task in tasks) == (
        transfer.PUBLIC_TWO_INT_CASES,
        transfer.PUBLIC_TWO_INT_CASES,
        transfer.PUBLIC_BOOL_CASES,
        transfer.PUBLIC_BOOL_CASES,
    )
    assert tuple(parameter.name for parameter in tasks[0].signature.parameters) == (
        "alpha",
        "beta",
    )
    assert tasks[2].signature.parameters[0].name == "datum"
    assert tuple(task.incumbent_tokens for task in tasks) == (
        ("add", "variable-0", "integer", "+", "0", "end"),
        ("add", "variable-0", "integer", "+", "0", "end"),
        (
            "less-than",
            "absolute",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
        ),
        (
            "less-than",
            "absolute",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
        ),
    )
    assert tuple(task.desired_outputs for task in tasks) == (
        (13, -9, 0, 1),
        (-13, 9, 0, -1),
        (True, False, False, False),
        (False, False, True, True),
    )
    assert tuple(len(target.expected_leading_actions) for target in transfer.PUBLIC_TARGETS) == (
        2,
        2,
        3,
        2,
    )


def test_prior_hierarchical_pass_is_authenticated_with_canonical_lf_hashes() -> None:
    prior = transfer.authenticate_prior_pass()

    assert prior.artifact_lf_sha256 == transfer.PRIOR_ARTIFACT_LF_SHA256
    assert prior.spend_marker_lf_sha256 == transfer.PRIOR_SPEND_MARKER_LF_SHA256
    assert prior.planner_source_lf_sha256 == transfer.PRIOR_PLANNER_SOURCE_LF_SHA256
    assert prior.model_checkpoint_sha256 == transfer.PRIOR_MODEL_CHECKPOINT_SHA256
    assert prior.verdict == transfer.hierarchical.PASS_VERDICT


def test_transfer_runner_is_a_thin_post_inference_validation_boundary() -> None:
    source = inspect.getsource(transfer.run_public_task)

    assert source.count("model.plan(") == 1
    assert source.index("model.plan(") < source.index("PUBLIC_TARGETS")
    assert source.index("PUBLIC_TARGETS") < source.index("expected_tokens")
    for forbidden in (
        "factor_action_candidates",
        "apply_factor_action",
        "actuate_selected_factor_action",
        "compile(",
        "exec(",
        "sorted(",
        ".sort(",
        "argmax",
        "argpartition",
    ):
        assert forbidden not in source


def test_spend_marker_is_exclusive_and_commits_prior_pass_and_cohort(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "family-transfer.spend.json"
    monkeypatch.setattr(transfer, "_SPEND_MARKER_PATH", marker)
    prior = transfer.authenticate_prior_pass()
    commitment = transfer._fresh_commitment(prior)

    assert transfer._acquire_spend_marker(commitment)
    assert not transfer._acquire_spend_marker(commitment)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert len(payload["commitment"]["tasks"]) == 4
    assert payload["commitment"]["literal_bank"] == list(range(-4, 5))
    assert payload["commitment"]["prior_pass"]["artifact_lf_sha256"] == (
        transfer.PRIOR_ARTIFACT_LF_SHA256
    )
    assert payload["commitment_sha256"] == sha256(
        transfer._canonical_json(commitment)
    ).hexdigest()


def test_existing_artifact_guard_never_runs_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(transfer, "_ARTIFACT_PATH", ExistingArtifact())
    monkeypatch.setattr(
        transfer,
        "run_public_matrix",
        lambda *_: (_ for _ in ()).throw(AssertionError("guard reran cohort")),
    )

    assert transfer.main() == 0


@pytest.mark.skipif(
    not transfer._ARTIFACT_PATH.exists(),
    reason="family-transfer cohort has not been spent yet",
)
def test_banked_artifact_records_the_sole_transfer_cohort() -> None:
    artifact_raw = transfer._ARTIFACT_PATH.read_bytes()
    marker_raw = transfer._SPEND_MARKER_PATH.read_bytes()
    payload = json.loads(artifact_raw)
    marker = json.loads(marker_raw)

    assert _lf_sha256(artifact_raw) == (
        "7e82d29168620eaa5ec0d3015b97e4e6e1986a2a49a346ee4bffee75afe25556"
    )
    assert _lf_sha256(marker_raw) == (
        "a96389d93d2825ddf7ed61d9d2a7907d124a0799e02095aaae6ae518852ad1a7"
    )
    assert payload["schema"] == transfer.SCHEMA
    assert payload["verdict"] in {transfer.PASS_VERDICT, transfer.STOP_VERDICT}
    assert payload["total_tasks"] == 4
    assert payload["prior_pass"]["artifact_lf_sha256"] == (
        transfer.PRIOR_ARTIFACT_LF_SHA256
    )
    assert payload["commitment_sha256"] == marker["commitment_sha256"]
    assert marker["commitment"]["runner_source_lf_sha256"] == transfer._lf_sha256(
        transfer.Path(transfer.__file__)
    )
    if payload["verdict"] == transfer.PASS_VERDICT:
        assert payload["passed_tasks"] == 4
        assert [len(task["leading_value_tier_cells"]) for task in payload["tasks"]] == [
            2,
            2,
            3,
            2,
        ]
        assert all(
            task["passed"]
            and task["status"] == transfer.hierarchical.STATUS_SELECTED
            and task["leading_value_tier_actions"]
            == task["expected_leading_value_tier_actions"]
            and task["selected_action"] == task["expected_selected_action"]
            and task["final_tokens"] == task["expected_tokens"]
            and task["final_outputs"] == task["desired_outputs"]
            and task["structure_prediction_calls"] == 1
            and task["actuator_calls"] == 1
            and task["operator_races_unique"]
            and task["prompt_race_unique"]
            and task["motor_races_unique"]
            and task["structure_races_unique"]
            and task["checkpoint_immutable"]
            and not task["value_stop_in_leading_tier"]
            for task in payload["tasks"]
        )


def test_tests_never_call_the_spent_transfer_matrix() -> None:
    module = inspect.getmodule(test_tests_never_call_the_spent_transfer_matrix)
    tree = ast.parse(inspect.getsource(module))
    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))

    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_public_matrix"
        for call in calls
    )
