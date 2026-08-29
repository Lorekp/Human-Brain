from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate as v1,
)
from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate_v2 as v2,
)
from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate_v3 as gate,
)
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2


def test_v3_freezes_the_exact_v1_v2_public_protocol_and_new_checkpoint() -> None:
    assert gate.MECHANISM_COMMIT == (
        "3f2a3cbe97f4cd935137e4f97970957271b9384a"
    )
    assert gate.PUBLIC_LITERAL_BANK == v1.PUBLIC_LITERAL_BANK == v2.PUBLIC_LITERAL_BANK
    assert gate.PUBLIC_TASKS is v1.PUBLIC_TASKS is v2.PUBLIC_TASKS
    assert gate.PUBLIC_TARGETS is v1.PUBLIC_TARGETS is v2.PUBLIC_TARGETS
    assert tuple(task.name for task in gate.PUBLIC_TASKS) == (
        "left-created-child",
        "right-created-child",
    )
    assert gate.public_protocol_sha256() == v2.public_protocol_sha256() == (
        "8391878e95b6da435308654ed4998f7e3b89af681728f9ecfbe1bbd36c34adfd"
    )

    model = h2.build_public_factorized_horizon2_int(
        literals=gate.PUBLIC_LITERAL_BANK
    )
    gate._validate_predeclared_boundary(model)


def test_v1_v2_banked_artifacts_remain_byte_exact() -> None:
    experiment_dir = Path(gate.__file__).parent
    expected = {
        "public_typed_factor_horizon2_int_created_child_gate.json": (
            "4f4164d031990770ac9e8573250ae91bd8fb22ed3bd3b93eadb0a9a116ef2c5e"
        ),
        "public_typed_factor_horizon2_int_created_child_gate_v2.json": (
            "8ae8dbc87965a0860567b31edcc4ff95f465818cb59b1490fc92a0e99d586c61"
        ),
        "public_typed_factor_horizon2_int_created_child_gate_v2.spend.json": (
            "80552d5b33905e74bc2a56db74ede5f0f8f874484c603eadabb642c9fff8ab25"
        ),
    }
    for name, expected_sha256 in expected.items():
        assert sha256((experiment_dir / name).read_bytes()).hexdigest() == expected_sha256


def test_existing_v3_artifact_guard_never_builds_or_runs(
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

    def forbidden_build(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(h2, "build_public_factorized_horizon2_int", forbidden_build)
    assert gate.main() == 0
    assert not spend.exists()


def test_spent_v3_without_artifact_refuses_same_name_retry(
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


def test_banked_created_child_v3_pass_artifact_contract() -> None:
    experiment_dir = Path(gate.__file__).parent
    artifact_path = experiment_dir / (
        "public_typed_factor_horizon2_int_created_child_gate_v3.json"
    )
    spend_path = experiment_dir / (
        "public_typed_factor_horizon2_int_created_child_gate_v3.spend.json"
    )
    raw = artifact_path.read_bytes()
    spend_raw = spend_path.read_bytes()
    artifact = json.loads(raw)

    assert sha256(raw).hexdigest() == (
        "432872d76a9f628305f2a921843586957f587d033112bde03d28f6a442b10104"
    )
    assert sha256(spend_raw).hexdigest() == (
        "0284a7b6a9e0fb74e02ffcd4c0fa4aea0646d81a01d864ca9f0a76510e7d16c6"
    )
    assert artifact["schema"] == gate.SCHEMA
    assert artifact["verdict"] == gate.PASS_VERDICT
    assert artifact["passed_tasks"] == artifact["attempted_tasks"] == 2
    assert artifact["total_tasks"] == 2
    assert artifact["mechanism_commit"] == gate.MECHANISM_COMMIT
    assert artifact["public_protocol_sha256"] == gate.public_protocol_sha256()
    assert artifact["model_checkpoint_sha256"] == (
        "7820c258c0d7d243008c5ec95b7ed19ceb91b77729396877b0d474af5a272f85"
    )
    assert artifact["prompt_factors_source_lf_sha256"] == (
        "d740f14731affe5561801e8d0eea3bd37cdf1ddc088e57839d857b5340bcdd5d"
    )
    assert artifact["spend_marker_lf_sha256"] == sha256(
        spend_raw.replace(b"\r\n", b"\n")
    ).hexdigest()

    expected_prompt_checkpoints = (
        "923ae4b49e2487b5db544877c01f8c42b874f278d231436e535621fa7c827951",
        "8e90ebc00b84e1edfc8dcce11f8261db425a04827316e0ebe4c7eddb392eeb3a",
    )
    for trace, expected in zip(
        artifact["prompt_sequence_traces"],
        expected_prompt_checkpoints,
        strict=True,
    ):
        assert trace["statuses"] == ["PROJECTED"]
        assert trace["projection_sha256s"] == [expected]
        assert trace["checkpoint_before_sha256s"] == [expected]
        assert trace["checkpoint_after_sha256s"] == [expected]
        assert trace["immutable"] is True

    for task in artifact["tasks"]:
        assert task["passed"] is True
        assert task["error"] is None
        assert task["no_h1_all_case_solution"] is True
        assert task["h1_exact_actions"] == []
        assert task["actual_trajectory"] == task["expected_trajectory"]
        assert task["final_tokens"] == task["expected_tokens"]
        assert task["final_outputs"] == task["desired_outputs"]
        assert task["checkpoint_immutable"] is True
        assert task["first_decision"]["status"] == "SELECTED"
        assert task["second_decision"]["status"] == "SELECTED"
        assert task["done_decision"]["status"] == "DONE"
        assert task["first_actuation"]["actuator_calls"] == 1
        assert task["second_actuation"]["actuator_calls"] == 1
        for actuation_name in ("first_actuation", "second_actuation"):
            actuation = task[actuation_name]
            assert actuation["structure_parity"] is True
            assert actuation["action_site_parity"] is True
            assert actuation["root_output_parity"] is True
