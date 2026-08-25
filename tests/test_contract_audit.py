"""Tests for humanbrain.contract_audit contract tripwires."""

import sys
import types

import pytest

from humanbrain.contract_audit import (
    ContractViolation,
    assert_no_forbidden_trainable_components,
    audit,
    state_dict_digest,
    weight_snapshot_compare,
)

TORCHISH = ("torch", "torch.nn", "torch.optim", "autograd")


@pytest.fixture
def fake_torch(monkeypatch):
    """Inject a fake torch stack into sys.modules to simulate a violation."""
    for name in TORCHISH:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_violation_trips_on_fake_torch(fake_torch):
    with pytest.raises(ContractViolation) as excinfo:
        assert_no_forbidden_trainable_components()
    message = str(excinfo.value)
    assert "torch" in message
    assert "autograd" in message


def test_clean_environment_passes(monkeypatch):
    for name in TORCHISH:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert assert_no_forbidden_trainable_components() is None


def test_decorator_trips_before_body_runs(fake_torch):
    calls = []

    @audit
    def scored_run():
        calls.append(1)
        return {"score": 0.5}

    with pytest.raises(ContractViolation):
        scored_run()
    assert calls == []  # pre-check fires before the body executes


def test_decorator_clean_passthrough(monkeypatch):
    for name in list(sys.modules):
        if name == "torch" or name.startswith(("torch.", "autograd")):
            monkeypatch.delitem(sys.modules, name, raising=False)

    @audit(weight_source=lambda: {"w": [1.0]})
    def scored_run():
        return 42

    assert scored_run() == 42


def test_state_dict_digest_deterministic_and_sensitive():
    state = {"layer.weight": [1.0, 2.0, 3.0], "layer.bias": 0.5}
    assert state_dict_digest(state) == state_dict_digest(dict(state))
    reordered = {"layer.bias": 0.5, "layer.weight": [1.0, 2.0, 3.0]}
    assert state_dict_digest(state) == state_dict_digest(reordered)
    assert state_dict_digest(state) != state_dict_digest(
        {**state, "layer.bias": 0.500001}
    )


def test_weight_snapshot_compare_clean_and_drift():
    before = {"w": [1.0, 2.0], "b": 0.0}
    report = weight_snapshot_compare(before, {"b": 0.0, "w": [1.0, 2.0]})
    assert report["identical"] is True
    assert report["changed"] == []

    drifted = {"w": [1.0, 2.5], "b": 0.0, "extra": 1}
    with pytest.raises(ContractViolation) as excinfo:
        weight_snapshot_compare(before, drifted)
    message = str(excinfo.value)
    assert "changed" in message and "'w'" in message
    assert "'added'" in message and "'extra'" in message


def test_audit_weight_source_detects_mutation():
    state = {"w": [1.0]}

    @audit(weight_source=lambda: dict(state))
    def sneaky_mutation():
        state["w"][0] = 99.0
        return "ok"

    with pytest.raises(ContractViolation):
        sneaky_mutation()
