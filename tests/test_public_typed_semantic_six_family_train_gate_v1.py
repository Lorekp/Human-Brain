from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.r3_pilot import (
    public_typed_semantic_six_family_train_gate_v1 as gate,
)
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


def _fake_task(family: str, tier: int, ordinal: int) -> SimpleNamespace:
    parameter_name, parameter_type, result_type = (
        gate._EXPECTED_FAMILY_SIGNATURE_SHAPES[family]
    )
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter(name=parameter_name, type=parameter_type),),
        result=result_type,
    )
    return SimpleNamespace(
        public=SimpleNamespace(
            task_id=f"simple-swe-v1/train/tier-{tier}/{family}/{ordinal:03d}",
            partition="train",
            family=family,
            ordinal=ordinal,
            signature=signature,
        ),
        private=SimpleNamespace(
            regression_cases=(None,) * gate.VISIBLE_REGRESSION_CASES,
            property_cases=(None,) * gate.UNEXPOSED_PROPERTY_CASES,
        ),
    )


def test_frozen_metadata_preflight_does_not_generate_tasks(monkeypatch) -> None:
    protocol = load_protocol(gate._PROTOCOL_PATH)
    monkeypatch.setattr(
        gate,
        "generate_visible_partition",
        lambda *_args, **_kwargs: pytest.fail("preflight generated sealed rows"),
    )

    record = gate._protocol_preflight(protocol)

    assert record["verdict"] == gate.PREFLIGHT_VERDICT
    assert record["sealed_rows_materialized"] == 0
    assert record["declared_tasks"] == 60
    assert tuple(record["declared_task_ids"]) == gate.DECLARED_TASK_IDS
    assert record["declared_task_ids_sha256"] == (
        "e1d3276b46a32e7811053851ba4776d3c5cddeed2292d6a10761b938a7334dd2"
    )
    assert gate.PUBLIC_FAMILIES == (
        "t0_scalar_clamp",
        "t0_scalar_nonnegative",
        "t1_string_filter_class",
        "t2_clamp_then_sum",
        "t2_map_then_count",
        "t2_filter_then_reverse",
    )
    assert gate.MAX_AST_COST == 9
    assert gate.PUBLIC_LITERAL_BANK == tuple(range(-9, 10))


def test_selection_uses_the_complete_frozen_round_robin_without_real_rows(
    monkeypatch,
) -> None:
    protocol = load_protocol(gate._PROTOCOL_PATH)
    synthetic = tuple(
        _fake_task(family, tier, ordinal)
        for ordinal in range(gate.FAMILY_QUOTA)
        for family, tier in zip(
            gate.PUBLIC_FAMILIES,
            gate.PUBLIC_FAMILY_TIERS,
            strict=True,
        )
    )
    monkeypatch.setattr(
        gate,
        "generate_visible_partition",
        lambda observed, partition: (
            synthetic
            if observed is protocol and partition == "train"
            else pytest.fail("unexpected selection call")
        ),
    )

    selected = gate.select_public_train_tasks(protocol)

    assert tuple(task.public.task_id for task in selected) == gate.DECLARED_TASK_IDS
    assert len(selected) == 60


def test_exact_prerequisites_and_exposure_registry_authenticate() -> None:
    prerequisites = gate._require_prior_bridge_boundary()
    exposure = gate._load_exposure_registry()

    assert prerequisites["three_family_gate"]["passed_tasks"] == 30
    assert prerequisites["compact_gate"]["comparison_gates"] == 44
    assert exposure["prompt_exposed_tasks"] == 60
    assert exposure["prior_r3_artifact_presence_tasks"] == 24
    assert (
        exposure["payload_sha256"]
        == gate._EXPECTED_EXPOSURE_REGISTRY_PAYLOAD_SHA256
    )
    assert exposure["prior_r3_artifact_presence_by_family"] == {
        "t0_scalar_clamp": 8,
        "t0_scalar_nonnegative": 6,
        "t1_string_filter_class": 0,
        "t2_clamp_then_sum": 7,
        "t2_map_then_count": 3,
        "t2_filter_then_reverse": 0,
    }


def test_work_bound_stop_is_a_serialized_task_failure() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter(name="n", type=integer),),
        result=integer,
    )
    cases = tuple(
        SimpleNamespace(args=(value,), expected=value)
        for value in (-2, -1, 0, 1)
    )
    task = SimpleNamespace(
        public=SimpleNamespace(
            task_id="synthetic/train/t0/000",
            family="synthetic",
            ordinal=0,
            task_sha256="1" * 64,
            hidden_test_sha256="2" * 64,
            signature=signature,
        ),
        private=SimpleNamespace(
            regression_cases=cases,
            property_cases=(),
            test_sha256="3" * 64,
        ),
    )
    work_bound = semantic.SemanticFrontierWorkBoundAudit(
        bound_name="operator-predictions",
        limit=30_000,
        completed=30_000,
        attempted=30_001,
        ast_cost=3,
        environment_type_labels=("Int",),
        result_type_label="Int",
        context_shape=("root", 4),
        completed_declared_candidates=100,
        completed_supported_root_candidates=90,
        completed_operator_predictions=30_000,
        completed_physical_races=70_000,
        completed_retained_representatives=100,
        completed_operator_races_sha256="4" * 64,
    )
    decision = semantic.TypedSemanticSynthesisDecision(
        status=semantic.STATUS_WORK_BOUND_STOP,
        stop_reason="semantic frontier work bound reached: operator-predictions",
        candidate_count=90,
        observed_semantic_classes=50,
        tier_attempts=(),
        selected_candidate_id=None,
        program=None,
        source=None,
        operator_predictions=30_000,
        operator_races=70_000,
        operator_races_sha256="4" * 64,
        checkpoint_before_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        checkpoint_after_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        work_bound=work_bound,
    )

    class FrozenStopModel:
        checkpoint_sha256 = gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        live_checkpoint_sha256 = gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256

        def synthesize(self, observed_signature, arguments, desired):
            assert observed_signature is signature
            assert arguments == ((-2,), (-1,), (0,), (1,))
            assert desired == (-2, -1, 0, 1)
            return decision

    runtime = gate.SandboxRuntime(
        executable=Path("unused"),
        identity={"executable_sha256": "5" * 64},
        worker_source=b"",
        worker_sha256="6" * 64,
    )

    record = gate._task_record(
        FrozenStopModel(),
        task,
        load_protocol(gate._PROTOCOL_PATH),
        runtime,
    )

    assert record["semantic_status"] == semantic.STATUS_WORK_BOUND_STOP
    assert record["passed"] is False
    assert record["sandbox"] is None
    assert record["work_bound"]["completed_operator_predictions"] == 30_000
    with pytest.raises(RuntimeError, match="omitted its completed-prefix audit"):
        gate._validate_work_bound_decision(replace(decision, work_bound=None))
    with pytest.raises(RuntimeError, match="counters differ"):
        gate._validate_work_bound_decision(
            replace(decision, operator_predictions=29_999)
        )


def test_spend_marker_is_exclusive_and_hashed(
    monkeypatch,
    tmp_path,
) -> None:
    marker = tmp_path / "gate.spent.json"
    artifact = tmp_path / "gate.json"
    source_freeze = {"files": {"runner.py": "a" * 64}, "bundle_sha256": "b" * 64}
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "_SPEND_MARKER_PATH", marker)
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", artifact)
    monkeypatch.setattr(gate, "_mechanism_source_freeze", lambda: source_freeze)
    host_runtime = {"numpy_version": "fixture"}
    git_identity = {"commit": "e" * 40}
    monkeypatch.setattr(gate, "_host_runtime_identity", lambda: host_runtime)
    monkeypatch.setattr(gate, "_git_identity", lambda: git_identity)

    model = SimpleNamespace(
        checkpoint_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        live_checkpoint_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        operator=SimpleNamespace(
            checkpoint_sha256=gate._EXPECTED_OPERATOR_CHECKPOINT_SHA256
        ),
    )
    runtime = gate.SandboxRuntime(
        executable=Path("unused"),
        identity={"executable_sha256": "c" * 64},
        worker_source=b"",
        worker_sha256="d" * 64,
    )
    kwargs = {
        "preflight": {"declared_tasks": 60},
        "prerequisites": {"exact": True},
        "source_freeze": source_freeze,
        "git_identity": git_identity,
        "exposure_registry": {"raw_sha256": "f" * 64},
        "runtime": runtime,
        "host_runtime": host_runtime,
        "model": model,
        "exclusive_host_attested": True,
    }

    with pytest.raises(RuntimeError, match="runtime changed"):
        gate._claim_spend(
            **{
                **kwargs,
                "host_runtime": {"numpy_version": "changed"},
            }
        )
    assert not marker.exists()
    drifted_model = SimpleNamespace(
        checkpoint_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        live_checkpoint_sha256="0" * 64,
        operator=model.operator,
    )
    with pytest.raises(RuntimeError, match="semantic checkpoint differs"):
        gate._claim_spend(**{**kwargs, "model": drifted_model})
    assert not marker.exists()
    monkeypatch.setattr(gate, "_git_identity", lambda: {"commit": "1" * 40})
    with pytest.raises(RuntimeError, match="Git tag/commit/tree identity changed"):
        gate._claim_spend(**kwargs)
    assert not marker.exists()
    monkeypatch.setattr(gate, "_git_identity", lambda: git_identity)

    observed = gate._claim_spend(**kwargs)
    raw = marker.read_bytes()
    payload = json.loads(raw)

    assert observed["status"] == "CLAIMED_BEFORE_TASK_MATERIALIZATION"
    assert observed["raw_sha256"] == sha256(raw).hexdigest()
    assert payload["status"] == observed["status"]
    without_hash = dict(payload)
    expected = without_hash.pop("payload_sha256")
    assert expected == sha256(gate._canonical_json(without_hash)).hexdigest()
    with pytest.raises(RuntimeError, match="already exists"):
        gate._claim_spend(**kwargs)


def test_execute_claims_marker_before_selecting_any_task_rows(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    secret = "SENTINEL_VISIBLE_ROW_SECRET_DO_NOT_PUBLISH"
    marker = tmp_path / "gate.spent.json"
    source_freeze = {"files": {"runner.py": "a" * 64}, "bundle_sha256": "b" * 64}
    host_runtime = {"numpy_version": "fixture"}
    git_identity: dict[str, object] = {}
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(gate, "_SPEND_MARKER_PATH", marker)
    monkeypatch.setattr(gate, "_mechanism_source_freeze", lambda: source_freeze)
    monkeypatch.setattr(gate, "_host_runtime_identity", lambda: host_runtime)
    monkeypatch.setattr(gate, "_git_identity", lambda: git_identity)

    model = SimpleNamespace(
        checkpoint_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        live_checkpoint_sha256=gate._EXPECTED_SEMANTIC_CHECKPOINT_SHA256,
        operator=SimpleNamespace(
            checkpoint_sha256=gate._EXPECTED_OPERATOR_CHECKPOINT_SHA256
        ),
    )
    runtime = gate.SandboxRuntime(
        executable=Path("unused"),
        identity={"executable_sha256": "c" * 64},
        worker_source=b"",
        worker_sha256="d" * 64,
    )
    claim_spend = gate._claim_spend

    def claim(**kwargs):
        observed = claim_spend(**kwargs)
        events.append("claim")
        return observed

    def select(_protocol):
        marker_payload = json.loads(marker.read_bytes())
        assert marker_payload["status"] == "CLAIMED_BEFORE_TASK_MATERIALIZATION"
        events.append("generate")
        return ()

    def publish(_payload, _path):
        events.append("publish")

    def fail_after_generation(*_args):
        raise gate._CohortExecutionError(
            RuntimeError(secret),
            completed_results=(
                {
                    "task_id": "synthetic/train/t0/000",
                    "terminal_program_sha256": "7" * 64,
                    "terminal_source_sha256": "8" * 64,
                },
            ),
            attempted_tasks=1,
            failing_task_id="synthetic/train/t0/000",
        )

    monkeypatch.setattr(gate, "_claim_spend", claim)
    monkeypatch.setattr(gate, "select_public_train_tasks", select)
    monkeypatch.setattr(gate, "require_public_visible_support", lambda _tasks: {})
    monkeypatch.setattr(
        gate,
        "run_public_train_cohort",
        fail_after_generation,
    )
    monkeypatch.setattr(gate, "write_public_artifact", publish)

    payload, exit_code = gate._execute_once(
        protocol=load_protocol(gate._PROTOCOL_PATH),
        preflight={},
        prerequisite={},
        exposure_registry={},
        source_freeze=source_freeze,
        git_identity=git_identity,
        runtime=runtime,
        host_runtime=host_runtime,
        model=model,
        training_summary=object(),
        exclusive_host_attested=True,
    )

    assert events == ["claim", "generate", "publish"]
    assert payload["verdict"] == gate.INVALID_VERDICT
    assert payload["error_detail_redacted"] is True
    assert payload["error_detail_sha256"] == sha256(secret.encode()).hexdigest()
    assert secret not in json.dumps(payload)
    assert payload["failure_stage"] == "cohort"
    assert payload["attempted_tasks"] == 1
    assert payload["completed_tasks"] == 1
    assert payload["partial_progress_available"] is True
    assert payload["completed_task_records"][0]["terminal_program_sha256"] == "7" * 64
    assert exit_code == 2


def test_spent_plain_preflight_blocks_before_any_runtime_or_task_work(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    marker = tmp_path / "gate.spent.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gate, "_SPEND_MARKER_PATH", marker)
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(
        gate,
        "_parse_args",
        lambda: SimpleNamespace(
            execute=False,
            attest_exclusive_host=False,
            python_executable=None,
        ),
    )
    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda *_args: pytest.fail("blocked preflight opened the protocol/runtime path"),
    )

    assert gate.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["verdict"] == gate.BLOCKED_PREFLIGHT_VERDICT
    assert record["sealed_rows_materialized"] == 0


def test_host_synthesis_runtime_matches_frozen_python_and_numpy_build() -> None:
    identity = gate._host_runtime_identity()

    for key, value in gate._EXPECTED_HOST_RUNTIME.items():
        assert identity[key] == value


def test_artifact_publication_is_hashed_and_exclusive(tmp_path) -> None:
    path = tmp_path / "result.json"
    gate.write_public_artifact({"schema": "fixture", "verdict": "PASS"}, path)
    payload = json.loads(path.read_bytes())
    expected = payload.pop("payload_sha256")

    assert expected == sha256(gate._canonical_json(payload)).hexdigest()
    with pytest.raises(RuntimeError, match="already exists"):
        gate.write_public_artifact({"schema": "fixture", "verdict": "PASS"}, path)
