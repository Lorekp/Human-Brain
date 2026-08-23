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
  bounded by the HOST-side ridge-statistics budget (dense F^2 float64
  normal matrix, doubled during consolidation, F = K*N + 1 under the R0
  multi-trace feature space) — the budget binds at N = 4406; larger N
  (e.g. the pre-R0 mission point 16384) must be refused with an honest
  reason, and the CPU single-process budget likewise moved (4096 -> 1669);
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
    # R0 multi-trace arithmetic (protocol §5): the ridge feature space is
    # F = K*N + 1 with K = 8 traces, so the 1.5 GiB single-process peak-array
    # budget would bind at N = 1669. Superseded by R0-b protocol §6: dense-
    # ridge statistics on the cpu path are gated by the preregistered
    # 20 GiB host budget (family-aware geometry), which is what lets the
    # frozen voltage_quantiles / combo arms run at N=1024.
    limit = max(
        n
        for n in range(1, harness._MAX_N + 1)
        if harness._projected_peak_bytes(n, "ridge") <= harness._RAM_BUDGET_BYTES
    )
    assert limit == 1669  # the legacy 1.5 GiB projection arithmetic is intact
    ridge_limit = max(
        n
        for n in range(1, harness._MAX_N + 1)
        if harness._projected_peak_bytes(n, "ridge")
        <= harness._HOST_RIDGE_BUDGET_BYTES
    )
    # Within the python-circuit N cap the 20 GiB §6 budget never binds
    # (_MAX_CUDA_HOST_RIDGE_CELLS=4406 exceeds _MAX_N=4096).
    assert ridge_limit == harness._MAX_N
    assert harness._MAX_CUDA_HOST_RIDGE_CELLS > harness._MAX_N
    harness._validate_arguments(_arguments(N=limit, device="cpu"))
    harness._validate_arguments(_arguments(N=ridge_limit, device="cpu"))
    with pytest.raises(SystemExit, match="python circuit limit"):
        harness._validate_arguments(_arguments(N=harness._MAX_N + 1, device="cpu"))


def test_cuda_device_refuses_stale_16k_mission_and_pins_the_new_boundary() -> None:
    harness = _harness()
    # Pre-R0, N=16384 was the codebook scale-up mission point. Multi-trace
    # §1.2 grows the ridge feature space to F = K*N + 1 (K = 8), so its host
    # statistics at N=16384 would need ~275 GB — validation must refuse that
    # honestly and pin the new largest-supported N (4406 within the 20 GB
    # host budget; §5 cost arithmetic).
    assert harness._MAX_CUDA_HOST_RIDGE_CELLS == 4406
    with pytest.raises(
        SystemExit,
        match=r"HOST ridge statistics.*largest supported N on this box is 4406",
    ):
        harness._validate_arguments(_arguments(N=16384, device="cuda"))
    # The moved boundary itself must validate cleanly on cuda.
    harness._validate_arguments(
        _arguments(N=harness._MAX_CUDA_HOST_RIDGE_CELLS, device="cuda")
    )


def test_cuda_device_budget_boundary_is_consistent() -> None:
    harness = _harness()
    limit = harness._MAX_CUDA_HOST_RIDGE_CELLS
    harness._validate_arguments(_arguments(N=limit, device="cuda"))
    with pytest.raises(SystemExit):
        harness._validate_arguments(_arguments(N=limit + 1, device="cuda"))


def test_assembly_k_must_fit_the_reservoir_on_both_devices() -> None:
    harness = _harness()
    # Knob-domain errors fire BEFORE resource projections: an assembly that
    # cannot fit the reservoir is reported as an assembly-k error even when
    # the same N would also breach a device budget.
    with pytest.raises(SystemExit, match="assembly-k"):
        harness._validate_arguments(
            _arguments(N=16384, assembly_k=99999, device="cuda")
        )
    # Feasible multi-trace mission point: k=32 fits the N=4406 reservoir and
    # the whole argument set validates on cuda.
    harness._validate_arguments(
        _arguments(
            N=harness._MAX_CUDA_HOST_RIDGE_CELLS, assembly_k=32, device="cuda"
        )
    )


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
