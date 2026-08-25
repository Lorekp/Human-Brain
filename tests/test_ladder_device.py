"""Red tests for the ladder-harness ``--device`` knob (codebook scale-up).

The codebook_compare matrix ran at N <= 1024 under the pinned-CPU python
circuit. Scaling to N >= 4096 requires the CUDA full-touch path (the same
``HUMANBRAIN_DEVICE=cuda`` policy ``benchmarks/scale_opt_gpu_bench.py``
uses), which the harness previously refused at import time and at argument
validation (``--N`` capped at the python-circuit limit of 4096).

Covers:

- import-time device gate accepts ``HUMANBRAIN_DEVICE`` in {unset, cpu,
  cuda} and still refuses unknown devices;
- ``--N`` above the python-circuit limit stays rejected on ``--device cpu``
  (the RAM projection guard is about the python circuit, not dogma);
- ``--device cuda`` legitimately lifts the python-circuit N cap but is
  bounded by the HOST-side ridge-statistics budget (dense (N+1)^2 float64
  normal matrix, doubled during consolidation) — N=16384 must validate,
  N=65536 must be refused with an honest reason;
- every assembly must still fit the reservoir, and ``--codebook explicit``
  still requires ``--assembly-k 16`` on either device.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_harness_under_env(env_value: str | None) -> subprocess.CompletedProcess:
    """Import the harness in a fresh interpreter under one device env value."""
    code = textwrap.dedent(
        """
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "ladder_experiments_env_probe", r"{module}"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("ladder_experiments_env_probe", module)
        spec.loader.exec_module(module)
        print("DEVICE_GATE_OK")
        """
    ).format(module=str(MODULE_PATH))
    env = dict(os.environ)
    if env_value is None:
        env.pop("HUMANBRAIN_DEVICE", None)
    else:
        env["HUMANBRAIN_DEVICE"] = env_value
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _arguments(**overrides) -> argparse.Namespace:
    values = dict(
        experiment_id="codebook_scale",
        arm="probe",
        N=512,
        readout_rule="ridge",
        schedule="massed",
        data_scale=42,
        supervision="all_transition",
        codebook="similarity",
        assembly_k=16,
        device="cpu",
        # Keep in step with every knob the harness validates (added by
        # sibling knob authors; update here when a new validated knob lands).
        inhibitory_fraction=0.2,
        observatory_samples=128,
        observatory_state_stride=20,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ladder_experiments_tests", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ladder_experiments_tests", module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Import-time device gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_value", [None, "", "cpu", "cuda"])
def test_device_gate_accepts_known_devices(env_value: str | None) -> None:
    proc = _load_harness_under_env(env_value)
    assert proc.returncode == 0, proc.stderr
    assert "DEVICE_GATE_OK" in proc.stdout


def test_device_gate_still_refuses_unknown_devices() -> None:
    proc = _load_harness_under_env("tpu")
    assert proc.returncode != 0
    assert "HUMANBRAIN_DEVICE" in (proc.stderr + proc.stdout)


# ---------------------------------------------------------------------------
# 2. Validation: python-circuit cap on cpu, host-budget cap on cuda
# ---------------------------------------------------------------------------


def test_cpu_device_keeps_the_python_circuit_n_cap() -> None:
    harness = _harness()
    with pytest.raises(SystemExit, match="python circuit limit"):
        harness._validate_arguments(_arguments(N=16384, device="cpu"))
    # The historical boundary stays exactly where it was.
    harness._validate_arguments(_arguments(N=4096, device="cpu"))


def test_cuda_device_validates_16k_and_refuses_64k_on_host_budget() -> None:
    harness = _harness()
    # Primary mission point must pass validation on cuda.
    harness._validate_arguments(_arguments(N=16384, device="cuda"))
    # 64k needs a dense (65537^2) float64 ridge normal matrix (~34.4 GiB,
    # doubled during consolidation) — larger than this box's physical RAM.
    # The refusal must say so honestly instead of silently attempting it.
    with pytest.raises(SystemExit, match="ridge"):
        harness._validate_arguments(_arguments(N=65536, device="cuda"))


def test_cuda_device_budget_boundary_is_consistent() -> None:
    harness = _harness()
    limit = harness._MAX_CUDA_HOST_RIDGE_CELLS
    harness._validate_arguments(_arguments(N=limit, device="cuda"))
    with pytest.raises(SystemExit):
        harness._validate_arguments(_arguments(N=limit + 1, device="cuda"))


def test_assembly_k_must_fit_the_reservoir_on_both_devices() -> None:
    harness = _harness()
    with pytest.raises(SystemExit, match="assembly-k"):
        harness._validate_arguments(
            _arguments(N=16384, assembly_k=99999, device="cuda")
        )
    harness._validate_arguments(_arguments(N=16384, assembly_k=32, device="cuda"))


def test_explicit_arm_still_requires_default_k_on_cuda() -> None:
    harness = _harness()
    with pytest.raises(SystemExit, match="assembly-k 16"):
        harness._validate_arguments(
            _arguments(codebook="explicit", assembly_k=32, device="cuda")
        )


def test_resolve_device_prefers_flag_over_default_env() -> None:
    harness = _harness()
    assert harness._resolve_device(_arguments(device="cuda")) == "cuda"
    assert harness._resolve_device(_arguments(device="cpu")) == "cpu"
    assert harness._resolve_device(_arguments(device="auto")) in {"cpu", "cuda"}
