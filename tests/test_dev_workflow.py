"""Exercise the developer runner's exit/receipt contract with real pytest children."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dev(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "doctor", lambda: {"checkout": str(tmp_path)})
    monkeypatch.setattr(sys, "argv", [str(path), "qa", "smoke"])
    # main configures the process environment; restore it after each test.
    for name in ("HUMANBRAIN_NATIVE_CACHE", "HUMANBRAIN_DEVICE", "RUFF_CACHE_DIR", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        monkeypatch.delenv(name, raising=False)
    return module


@pytest.mark.parametrize("passes, expected", [(True, 0), (False, 1)])
def test_runner_preserves_real_pytest_exit_and_debug_receipts(dev, passes, expected, monkeypatch):
    # Ambient flags must not turn a passing lane into collection-only or load
    # arbitrary plugins. A failing child must still execute and report failure.
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_ambient_plugin")
    fixture = dev.ROOT / "test_fixture.py"
    fixture.write_text(f"def test_result():\n    assert {passes}\n", encoding="utf-8")
    dev.QA["smoke"] = [str(fixture)]

    assert dev.main() == expected

    receipt_path, = (dev.ROOT / ".pytest-tmp" / "dev").glob("*/receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["exit_code"] == expected
    assert str(fixture) in receipt["pytest_command"]
    assert (receipt_path.parent / "junit.xml").is_file()
    assert ("1 passed" if passes else "1 failed") in (
        receipt_path.parent / "output.log"
    ).read_text(encoding="utf-8")


def test_failed_preflight_never_starts_pytest(dev, monkeypatch):
    def wrong_checkout():
        raise RuntimeError("Wrong editable installation")

    monkeypatch.setattr(dev, "doctor", wrong_checkout)
    assert dev.main() == 1
    receipt_path, = (dev.ROOT / ".pytest-tmp" / "dev").glob("*/receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["error"] == "Wrong editable installation"
    assert "pytest_command" not in receipt
    assert not (receipt_path.parent / "output.log").exists()
