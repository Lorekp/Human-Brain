"""Red tests: --inhibitory-fraction knob for ladder_experiments (V3a EI sweep).

Contract for the preregistered ``ei_sweep``
(docs/research/a013-cell-construction-variants.md V3a as corrected by review
522551d; experiments/ei_sweep/PREREGISTRATION.md). Red against harness HEAD
c7c08ed (knob absent from run_experiment/_grow_circuit/CLI); green when the
harness plumbs RecurrentCircuitConfig.inhibitory_fraction and exposes the
sign-pattern metrics helper.

Covers: knob determinism (identical args -> identical records; default 0.2
reproduces the status quo), fraction bounds validation (run_experiment and
CLI gates), and sign-hash stability (same seed+fraction -> same canonical
sign hash; different fraction -> different hash with topology untouched).
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
# Register before exec: Python dataclasses resolve annotations through
# sys.modules, which importlib file-location modules do not populate.
sys.modules.setdefault("ladder_experiments", HARNESS)
SPEC.loader.exec_module(HARNESS)

N_PROBE = 128


def _base_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "experiment_id": "ei-knob-probe",
        "arm": "probe",
        "n_cells": N_PROBE,
        "seed": 7,
        "readout_rule": "three_factor",
        "schedule": "massed",
        "data_scale": 21,
        "supervision": "program_only",
    }
    kwargs.update(overrides)
    return kwargs


def _record_without_wall(record: dict[str, object]) -> bytes:
    """Canonical bytes minus run-identity fields (wall_s, arm label)."""
    return HARNESS._canonical_json(
        {k: v for k, v in record.items() if k not in ("wall_s", "arm")}
    )


# ---------------------------------------------------------------------------
# 1. Knob plumbing + determinism
# ---------------------------------------------------------------------------


def test_run_experiment_accepts_inhibitory_fraction() -> None:
    params = inspect.signature(HARNESS.run_experiment).parameters
    assert "inhibitory_fraction" in params
    assert params["inhibitory_fraction"].default == pytest.approx(0.2)


def test_grow_circuit_forwards_inhibitory_fraction_into_config() -> None:
    src = inspect.getsource(HARNESS._grow_circuit)
    assert "inhibitory_fraction=inhibitory_fraction" in src


def test_cli_exposes_inhibitory_fraction_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        HARNESS.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--inhibitory-fraction" in out


def test_knob_determinism_and_default_is_status_quo() -> None:
    """Same args -> byte-identical records; explicit 0.2 == shipped default."""
    default = HARNESS.run_experiment(**_base_kwargs(arm="default"))
    explicit = HARNESS.run_experiment(
        **_base_kwargs(arm="explicit02", inhibitory_fraction=0.2)
    )
    assert _record_without_wall(default) == _record_without_wall(explicit)

    repeat = HARNESS.run_experiment(
        **_base_kwargs(arm="repeat", inhibitory_fraction=0.35)
    )
    other = HARNESS.run_experiment(
        **_base_kwargs(arm="other", inhibitory_fraction=0.35)
    )
    assert _record_without_wall(repeat) == _record_without_wall(other)

    # A different fraction shares the seed's curriculum but must be able to
    # move dynamics: the full record may not be a no-op clone.
    zero = HARNESS.run_experiment(**_base_kwargs(arm="zero", inhibitory_fraction=0.0))
    assert default["curriculum_sha256"] == zero["curriculum_sha256"]
    assert _record_without_wall(zero) != _record_without_wall(default)


def test_config_dump_records_effective_fraction() -> None:
    record = HARNESS.run_experiment(
        **_base_kwargs(arm="provenance", inhibitory_fraction=0.35)
    )
    dump = record["config_dump"]
    assert dump["inhibitory_fraction"] == pytest.approx(0.35)
    assert dump["circuit_config"]["inhibitory_fraction"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# 2. Fraction bounds validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.25, float("nan"), float("inf")])
def test_out_of_bounds_fraction_rejected_up_front(bad: float) -> None:
    with pytest.raises(ValueError):
        HARNESS.run_experiment(
            **_base_kwargs(arm="invalid", inhibitory_fraction=bad)
        )
    assert not math.isfinite(bad) or bad < 0.0 or bad >= 1.0


_BASE_ARGS = [
    "--experiment",
    "ei-gate",
    "--arm",
    "gate",
    "--N",
    str(N_PROBE),
    "--seed",
    "7",
]


@pytest.mark.parametrize("bad", ["1.0", "-0.1", "nan", "inf"])
def test_cli_rejects_out_of_bounds_fraction(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SystemExit):
        HARNESS.main(
            [*_BASE_ARGS, "--out", str(tmp_path), "--inhibitory-fraction", bad]
        )


# ---------------------------------------------------------------------------
# 3. Sign-hash stability + mechanism engagement
# ---------------------------------------------------------------------------


def _circuit(fraction: float, seed: int = 7):
    config = HARNESS.RecurrentCircuitConfig(
        reservoir_cells=N_PROBE,
        input_fanout=HARNESS._INPUT_FANOUT,
        recurrent_fanout=HARNESS._RECURRENT_FANOUT,
        random_seed=seed,
        readout_learning_rule="ridge",
        inhibitory_fraction=fraction,
    )
    return HARNESS.RecurrentCellCircuit(HARNESS._TOKEN_CELLS, config=config)


def test_sign_hash_stable_within_arms_varies_across_arms() -> None:
    baseline = HARNESS.recurrent_sign_metrics(_circuit(0.2))
    again = HARNESS.recurrent_sign_metrics(_circuit(0.2))
    assert baseline["sign_sha256"] == again["sign_sha256"]

    for fraction in (0.0, 0.1, 0.35):
        other = HARNESS.recurrent_sign_metrics(_circuit(fraction))
        assert other["sign_sha256"] != baseline["sign_sha256"]

    # Independent recomputation from the weight matrix must agree.
    circuit = _circuit(0.2)
    weights = circuit._recurrent_weights_pa[:, 0]
    expected = __import__("hashlib").sha256(
        np.sign(weights).astype(np.int8).tobytes()
    ).hexdigest()
    assert baseline["sign_sha256"] == expected


def test_sign_counts_and_magnitudes_track_fraction() -> None:
    active = N_PROBE
    current = HARNESS.RecurrentCircuitConfig(
        reservoir_cells=N_PROBE,
        input_fanout=HARNESS._INPUT_FANOUT,
        recurrent_fanout=HARNESS._RECURRENT_FANOUT,
    ).recurrent_current_pa
    for fraction in (0.0, 0.1, 0.2, 0.35):
        metrics = HARNESS.recurrent_sign_metrics(_circuit(fraction))
        assert metrics["excitatory_cells"] == active - round(fraction * active)
        assert metrics["inhibitory_cells"] == round(fraction * active)
        assert metrics["zero_sign_cells"] == 0
        if metrics["inhibitory_cells"]:
            assert metrics["mean_abs_weight_inhibitory_pa"] == pytest.approx(current)
        assert metrics["mean_abs_weight_excitatory_pa"] == pytest.approx(current)

    zero = HARNESS.recurrent_sign_metrics(_circuit(0.0))
    assert zero["inhibitory_cells"] == 0
    assert zero["mean_abs_weight_inhibitory_pa"] is None


def test_fraction_leaves_afferent_and_recurrent_topology_untouched() -> None:
    """Sign draws consume the RNG after targets: structure is fraction-invariant."""
    reference = _circuit(0.0)
    for fraction in (0.1, 0.2, 0.35):
        shifted = _circuit(fraction)
        assert np.array_equal(shifted._afferent_targets, reference._afferent_targets)
        assert np.array_equal(shifted._recurrent_targets, reference._recurrent_targets)
