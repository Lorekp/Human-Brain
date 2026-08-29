from __future__ import annotations

import argparse
import json
from collections import Counter

import pytest

from experiments.r3_pilot import public_factorized_simple_swe_train_bridge_v1 as bridge
from humanbrain.simple_swe.protocol import load_protocol


def test_public_train_identity_roots_are_physically_supported() -> None:
    protocol = load_protocol(bridge._PROTOCOL_PATH)
    support = bridge.require_public_visible_support(
        bridge.select_public_train_tasks(protocol)
    )

    assert support == {
        "tasks": 30,
        "cases": 120,
        "int_min": -64,
        "int_max": 64,
        "list_capacity": 4,
    }


def test_public_train_bridge_static_protocol_and_recovery_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
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
    assert bridge.MAX_EDITS == 2
    assert bridge.FAMILY_PASS_THRESHOLD == 8
    assert bridge.FAMILY_FAILURE_STOP == 3
    assert bridge.PUBLIC_LITERAL_BANK == tuple(range(-9, 10))

    for task in tasks:
        visible = bridge._visible_cases(task)
        assert len(visible) == 4
        assert len(task.private.property_cases) == 32
        assert tuple(case.arguments for case in visible) == tuple(
            bridge._snapshot(case.args) for case in task.private.regression_cases
        )
        assert tuple(case.expected for case in visible) == tuple(
            bridge._snapshot(case.expected) for case in task.private.regression_cases
        )

    boundary = bridge._require_recovery_boundary(bridge._RECOVERY_ARTIFACT_PATH)
    assert boundary["verdict"] == bridge.recovery_v1.STOP_VERDICT
    assert boundary["passed_tasks"] == 1
    assert boundary["failed_task"] == "list-map-add-four"
    assert boundary["first_applied_action"]["schema"] == "MAP_BIND"
    repaired = bridge._require_repaired_mechanism()
    assert repaired["verified"] is True
    assert tuple(repaired["module_sha256"]) == (
        "public_typed_prompt_factors.py",
    )

    recovery_path = tmp_path / "tampered-recovery.json"
    recovery_payload = json.loads(bridge._RECOVERY_ARTIFACT_PATH.read_text("utf-8"))
    recovery_payload["passed_tasks"] = 2
    recovery_path.write_text(json.dumps(recovery_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact MAP_BIND STOP boundary"):
        bridge._require_recovery_boundary(recovery_path)

    output_path = tmp_path / "bridge.json"
    missing_recovery = tmp_path / "missing-recovery.json"
    setup_calls: list[str] = []

    monkeypatch.setattr(
        bridge,
        "_parse_args",
        lambda: argparse.Namespace(
            protocol=bridge._PROTOCOL_PATH,
            json=output_path,
            python_executable=None,
        ),
    )
    monkeypatch.setattr(bridge, "_RECOVERY_ARTIFACT_PATH", missing_recovery)
    monkeypatch.setattr(
        bridge,
        "_prepare_sandbox",
        lambda *_args, **_kwargs: setup_calls.append("sandbox"),
    )
    monkeypatch.setattr(
        bridge.h2,
        "build_public_factorized_horizon2_int",
        lambda **_kwargs: setup_calls.append("model"),
    )
    with pytest.raises(RuntimeError, match="recovery boundary is absent"):
        bridge.main()
    assert setup_calls == []

    output_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact already exists"):
        bridge.main()
    assert output_path.read_text(encoding="utf-8") == "occupied"
