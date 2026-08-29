from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate as v1,
)
from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate_v2 as gate,
)
from experiments.r3_pilot import public_typed_program_observation as root_observation
from humanbrain import typed_program as tp


def test_v2_reuses_the_exact_v1_public_protocol() -> None:
    assert gate.MECHANISM_COMMIT == (
        "a31f64751b6025d99b06087ed4ec2c36bb8b4718"
    )
    assert gate.PUBLIC_LITERAL_BANK == v1.PUBLIC_LITERAL_BANK == tuple(range(-4, 5))
    assert gate.PUBLIC_TASKS is v1.PUBLIC_TASKS
    assert gate.PUBLIC_TARGETS is v1.PUBLIC_TARGETS
    assert tuple(task.name for task in gate.PUBLIC_TASKS) == (
        "left-created-child",
        "right-created-child",
    )
    assert gate.public_protocol_sha256() == (
        "8391878e95b6da435308654ed4998f7e3b89af681728f9ecfbe1bbd36c34adfd"
    )


def test_declared_targets_follow_reversed_debruijn_scope_order() -> None:
    # Signature parameters enter the expression environment in reverse order:
    # variable-0/SCOPE0 is beta and variable-1/SCOPE1 is alpha.
    for task, target in zip(gate.PUBLIC_TASKS, gate.PUBLIC_TARGETS, strict=True):
        program = tp.Program(task.incumbent_tokens, task.signature)
        post_first = tp.apply_factor_action(program, target.expected_first)
        final = tp.apply_factor_action(post_first, target.expected_second)
        outputs = tuple(
            case.raw_output
            for case in root_observation.observe_current_program_root(
                final,
                task.cases,
            ).cases
        )
        assert final.tokens == target.expected_tokens
        assert outputs == task.desired_outputs


def test_existing_v2_artifact_guard_never_runs_public_matrix(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "banked.json"
    spend = tmp_path / "banked.spend.json"
    artifact.write_text(
        json.dumps({"verdict": gate.PASS_VERDICT}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", artifact)
    monkeypatch.setattr(gate, "_SPEND_MARKER_PATH", spend)

    def forbidden_run(*, spend_marker_lf_sha256: str):
        raise AssertionError(spend_marker_lf_sha256)

    monkeypatch.setattr(gate, "run_public_matrix", forbidden_run)
    assert gate.main() == 0
    assert not spend.exists()


def test_spent_v2_without_artifact_refuses_same_name_retry(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "missing.json"
    spend = tmp_path / "banked.spend.json"
    spend.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", artifact)
    monkeypatch.setattr(gate, "_SPEND_MARKER_PATH", spend)

    with pytest.raises(RuntimeError, match="same-name rerun refused"):
        gate.main()


def test_banked_created_child_v2_stop_artifact_contract() -> None:
    experiment_dir = (
        Path(__file__).resolve().parents[1] / "experiments" / "r3_pilot"
    )
    artifact_path = experiment_dir / (
        "public_typed_factor_horizon2_int_created_child_gate_v2.json"
    )
    spend_path = experiment_dir / (
        "public_typed_factor_horizon2_int_created_child_gate_v2.spend.json"
    )
    runner_path = Path(gate.__file__)
    raw = artifact_path.read_bytes()
    spend_raw = spend_path.read_bytes()
    artifact = json.loads(raw)

    assert sha256(raw).hexdigest() == (
        "8ae8dbc87965a0860567b31edcc4ff95f465818cb59b1490fc92a0e99d586c61"
    )
    assert sha256(spend_raw).hexdigest() == (
        "80552d5b33905e74bc2a56db74ede5f0f8f874484c603eadabb642c9fff8ab25"
    )
    assert artifact["schema"] == gate.SCHEMA
    assert artifact["verdict"] == gate.STOP_VERDICT
    assert artifact["passed_tasks"] == 0
    assert artifact["attempted_tasks"] == 1
    assert artifact["total_tasks"] == 2
    assert artifact["literal_bank"] == list(range(-4, 5))
    assert artifact["mechanism_commit"] == gate.MECHANISM_COMMIT
    assert artifact["public_protocol_sha256"] == gate.public_protocol_sha256()
    assert artifact["model_checkpoint_sha256"] == (
        "1d15cec6f8449937d44ce293ba1d01467a6e245518870c40cad4ee2bf1ab50a5"
    )
    assert artifact["world_checkpoint_sha256"] == (
        "37b5e25bbb532f7398f984defeb52b937a6230535fad76069aefc359a5f5b666"
    )
    assert artifact["structure_successor_checkpoint_sha256"] == (
        "95a35644cb0689ddd30dc5994ac4c9f15b61ae019ce377681051845c68d96c8e"
    )
    assert artifact["runner_source_lf_sha256"] == sha256(
        runner_path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert artifact["structure_successor_source_lf_sha256"] == (
        "d1e7670e8f218aa719f8ccd715f404a75a3200ae7f79744dd46a02005b2909bf"
    )
    assert artifact["spend_marker_lf_sha256"] == sha256(
        spend_raw.replace(b"\r\n", b"\n")
    ).hexdigest()

    task = artifact["tasks"][0]
    assert task["name"] == "left-created-child"
    assert task["passed"] is False
    assert task["no_h1_all_case_solution"] is True
    assert task["h1_exact_actions"] == []
    assert task["first_actuation"] is None
    assert task["second_decision"] is None
    assert task["done_decision"] is None
    assert task["checkpoint_immutable"] is True
    assert task["error"] == (
        "first H=2 plan stopped: prompt evidence did not uniquely resolve "
        "the leading structure tier"
    )
    decision = task["first_decision"]
    assert (
        decision["status"],
        decision["candidate_count"],
        decision["expanded_first_count"],
        decision["unique_sequence_predictions"],
        decision["cumulative_sequence_evaluations"],
        decision["terminal_attempt_sequence_counts"],
    ) == ("STOP_PROMPT", 83, 4, 1_317, 1_317, [1_317])
