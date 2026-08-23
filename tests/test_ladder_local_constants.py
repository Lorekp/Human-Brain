"""Red tests: local-rule constant injection knobs for ladder_experiments.

Contract for the preregistered local_rule_constant_screen_v1
(docs/research/local-rule-constant-derivation.md §5). Red against harness
HEAD 081a492 (constants hardcoded at shipped 64.0/64.0); green only when
run_experiment/_grow_circuit/CLI plumb RecurrentCircuitConfig's
local_learning_rate_pa and local_competitive_margin_pa.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("ladder_experiments", HARNESS)
SPEC.loader.exec_module(HARNESS)


def test_run_experiment_accepts_local_rule_constants() -> None:
    params = inspect.signature(HARNESS.run_experiment).parameters
    assert params["local_learning_rate_pa"].default == pytest.approx(64.0)
    assert params["local_competitive_margin_pa"].default == pytest.approx(64.0)


def test_grow_circuit_forwards_constants_into_config() -> None:
    src = inspect.getsource(HARNESS._grow_circuit)
    assert "local_learning_rate_pa=local_learning_rate_pa" in src
    assert "local_competitive_margin_pa=local_competitive_margin_pa" in src


def test_cli_exposes_local_constant_flags(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        HARNESS.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--local-lr-pa" in out
    assert "--local-margin-pa" in out


def test_nonpositive_local_constants_rejected_up_front() -> None:
    with pytest.raises(ValueError):
        HARNESS.run_experiment(
            experiment_id="knob-validation-probe",
            arm="invalid",
            n_cells=128,
            seed=7,
            readout_rule="three_factor",
            schedule="massed",
            data_scale=42,
            supervision="all_transition",
            local_learning_rate_pa=0.0,
            local_competitive_margin_pa=363.0,
        )
