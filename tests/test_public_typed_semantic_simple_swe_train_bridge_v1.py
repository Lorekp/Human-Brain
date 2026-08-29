from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.r3_pilot import (
    public_typed_semantic_simple_swe_train_bridge_v1 as bridge,
)
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import ExecutionResult


def test_banked_typed_semantic_train_bridge_pass_is_exact() -> None:
    raw = bridge._ARTIFACT_PATH.read_bytes()
    assert sha256(raw).hexdigest() == (
        "09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767"
    )
    payload = json.loads(raw)

    assert payload["verdict"] == bridge.PASS_VERDICT
    assert payload["declared_tasks"] == 30
    assert payload["attempted_tasks"] == 30
    assert payload["passed_tasks"] == 30
    assert payload["stop_family"] is None
    assert payload["stop_reason"] is None
    assert payload["protocol"] == {
        "families": list(bridge.PUBLIC_FAMILIES),
        "partition": "train",
        "protocol_id": "human-cell-simple-swe-v1",
        "sha256": "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2",
    }
    assert payload["visibility_contract"] == {
        "prompt_visible_to_synthesizer": False,
        "property_cases_sandboxed_per_task": 32,
        "property_cases_visible_per_task": 0,
        "regression_arguments_visible_per_task": 4,
        "regression_outputs_visible_per_task": 4,
        "task_or_family_identity_visible_to_synthesizer": False,
        "typed_signature_visible": True,
    }
    assert payload["synthesizer"]["checkpoint_immutable"] is True
    assert payload["synthesizer"]["checkpoint_sha256"] == (
        "d28aba8651348646f153461fb306c00f873330968042a43b81475f52cfe8b74f"
    )
    assert payload["synthesizer"]["operator_checkpoint_sha256"] == (
        "62100b5296fe37eb13b3a7a1d9f92f032de494af12c95e20c3d46c4409cca8d0"
    )
    assert payload["synthesizer"]["source_freeze"]["bundle_sha256"] == (
        "fc734e4a3e8fbc2934701d6786978b6175843b91c4b919827851d356e7872841"
    )

    results = payload["results"]
    assert len(results) == 30
    assert len({result["task_id"] for result in results}) == 30
    assert Counter(result["family"] for result in results) == {
        family: 10 for family in bridge.PUBLIC_FAMILIES
    }
    assert Counter(
        next(
            attempt["ast_cost"]
            for attempt in result["tier_attempts"]
            if attempt["race"] is not None
            and attempt["race"]["status"] == "SELECTED"
        )
        for result in results
    ) == {3: 10, 5: 10, 6: 10}
    assert all(result["passed"] for result in results)
    assert all(result["checkpoint_immutable"] for result in results)
    assert all(result["sandbox"]["passed"] for result in results)
    assert all(
        result["property_cases_exposed_to_synthesizer"] == 0
        and result["property_cases_scored_in_sandbox"] == 32
        for result in results
    )


def test_typed_semantic_train_bridge_static_protocol_and_boundary(tmp_path) -> None:
    protocol = load_protocol(bridge._PROTOCOL_PATH)
    tasks = bridge.select_public_train_tasks(protocol)

    assert len(tasks) == 30
    assert Counter(task.public.family for task in tasks) == {
        family: 10 for family in bridge.PUBLIC_FAMILIES
    }
    assert tuple(
        (task.public.family, task.public.ordinal) for task in tasks
    ) == tuple(
        (family, ordinal)
        for ordinal in range(bridge.FAMILY_QUOTA)
        for family in bridge.PUBLIC_FAMILIES
    )
    assert bridge.PUBLIC_LITERAL_BANK == tuple(range(-9, 10))
    assert bridge.MAX_AST_COST == 6
    assert bridge.FAMILY_PASS_THRESHOLD == 8
    assert bridge.FAMILY_FAILURE_STOP == 3
    assert all(len(task.private.regression_cases) == 4 for task in tasks)
    assert all(len(task.private.property_cases) == 32 for task in tasks)

    assert bridge.require_public_visible_support(tasks) == {
        "tasks": 30,
        "cases": 120,
        "encoded_values": 240,
        "int_min": -64,
        "int_max": 64,
        "list_capacity": 4,
    }
    boundary = bridge._require_prior_bridge_boundary(
        bridge._PRIOR_BRIDGE_ARTIFACT_PATH
    )
    assert boundary == {
        "schema": "public-factorized-simple-swe-train-bridge-v1",
        "verdict": "STOP_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1",
        "artifact_sha256": (
            "f748e50cc98ddc987bcdbbc5804b4203e04b0c0376509d66f374f822781fae32"
        ),
        "declared_tasks": 30,
        "attempted_tasks": 7,
        "passed_tasks": 2,
        "stop_family": "t0_scalar_offset",
    }

    freeze = bridge._mechanism_source_freeze()
    assert tuple(freeze["files"]) == bridge._MECHANISM_SOURCE_PATHS
    assert len(freeze["bundle_sha256"]) == 64
    assert freeze["files"][
        "experiments/r3_pilot/public_typed_semantic_synthesizer.py"
    ]

    tampered = tmp_path / "tampered-prior.json"
    payload = json.loads(
        bridge._PRIOR_BRIDGE_ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    payload["passed_tasks"] = 3
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact broad STOP"):
        bridge._require_prior_bridge_boundary(tampered)


def test_task_selection_seam_excludes_identity_and_main_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = load_protocol(bridge._PROTOCOL_PATH)
    task = bridge.select_public_train_tasks(protocol)[0]
    calls: list[dict[str, object]] = []

    class RecordingSynthesizer:
        checkpoint_sha256 = "frozen-checkpoint"

        @property
        def live_checkpoint_sha256(self) -> str:
            return self.checkpoint_sha256

        def synthesize(self, signature, arguments, desired):
            calls.append(
                {
                    "signature": signature,
                    "arguments": arguments,
                    "desired": desired,
                }
            )
            return SimpleNamespace(
                status=bridge.semantic.STATUS_DONE,
                stop_reason=None,
                candidate_count=7,
                observed_semantic_classes=5,
                tier_attempts=(),
                selected_candidate_id="candidate",
                program=SimpleNamespace(sha256="program-sha256"),
                source="return value",
                operator_predictions=3,
                operator_races=3,
                operator_races_sha256="races-sha256",
                checkpoint_before_sha256=self.checkpoint_sha256,
                checkpoint_after_sha256=self.checkpoint_sha256,
            )

    sandbox_calls: list[dict[str, object]] = []

    def fake_execute(source, public, private, limits, **kwargs):
        sandbox_calls.append(
            {
                "source": source,
                "public": public,
                "private": private,
                "limits": limits,
                "kwargs": kwargs,
            }
        )
        return ExecutionResult("pass", True, 0.0, "ok", True)

    monkeypatch.setattr(bridge, "execute_completion", fake_execute)
    runtime = bridge.SandboxRuntime(
        executable=Path("python"),
        identity={"executable_sha256": "python-sha256"},
        worker_source=b"worker",
        worker_sha256="worker-sha256",
    )
    record = bridge._task_record(
        RecordingSynthesizer(), task, protocol, runtime
    )

    signature, expected_arguments, expected_desired = bridge._visible_problem(task)
    assert calls == [
        {
            "signature": signature,
            "arguments": expected_arguments,
            "desired": expected_desired,
        }
    ]
    assert set(calls[0]) == {"signature", "arguments", "desired"}
    assert sandbox_calls[0]["source"] == "return value"
    assert sandbox_calls[0]["public"] is task.public
    assert sandbox_calls[0]["private"] is task.private
    assert record["passed"] is True
    assert record["property_cases_exposed_to_synthesizer"] == 0

    setup_calls: list[str] = []
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        bridge,
        "_parse_args",
        lambda: argparse.Namespace(
            protocol=bridge._PROTOCOL_PATH,
            json=output,
            python_executable=None,
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_PRIOR_BRIDGE_ARTIFACT_PATH",
        tmp_path / "missing-prior.json",
    )
    monkeypatch.setattr(
        bridge,
        "_prepare_sandbox",
        lambda *_args, **_kwargs: setup_calls.append("sandbox"),
    )
    monkeypatch.setattr(
        bridge.semantic,
        "build_public_typed_semantic_synthesizer",
        lambda **_kwargs: setup_calls.append("model"),
    )
    with pytest.raises(RuntimeError, match="boundary is absent"):
        bridge.main()
    assert setup_calls == []

    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact already exists"):
        bridge.main()
    assert output.read_text(encoding="utf-8") == "occupied"
