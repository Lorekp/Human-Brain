"""Tests for the CUDA-scale ridge acceleration (scipy dsyrk consolidation).

The accelerated path must compute the SAME regularized least-squares fit as
the numpy path up to floating-point reordering: rank-k statistics accumulated
into one symmetric triangle, mirrored before the ridge solve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.linalg.blas")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments_ridge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("ladder_experiments_ridge", HARNESS)
SPEC.loader.exec_module(HARNESS)


def _numpy_reference(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = features.T @ features
    regularized = normal.copy()
    diagonal = np.diag_indices_from(regularized)
    regularized[diagonal] += 1.0
    regularized[-1, -1] -= 0.99
    return normal, regularized


def test_syrk_triangle_accumulation_matches_numpy_dot() -> None:
    rng = np.random.default_rng(7)
    dimension = 96
    normal = np.zeros((dimension, dimension), dtype=np.float64)
    for _ in range(4):
        features = rng.standard_normal((37, dimension))
        # Exactly what install_scipy_ridge's flush does per batch:
        from scipy.linalg.blas import dsyrk

        normal = dsyrk(
            alpha=1.0, a=features, trans=1, beta=1.0, c=normal, overwrite_c=1
        )
    reference = _numpy_reference(np.vstack([rng.standard_normal((37, dimension))]))[0]
    # Recompute the same batches deterministically instead: rebuild inputs.
    rng = np.random.default_rng(7)
    batches = [rng.standard_normal((37, dimension)) for _ in range(4)]
    stacked = np.vstack(batches)
    expected = _numpy_reference(stacked)[0]
    upper = np.triu(normal)
    full = upper + upper.T - np.diag(np.diag(upper))
    assert np.allclose(full, expected, rtol=1e-12, atol=1e-9)


def test_scipy_consolidation_solution_matches_numpy_path() -> None:
    rng = np.random.default_rng(11)
    rows, dimension = 512, 128
    stacked = rng.standard_normal((rows, dimension))
    _, regularized = _numpy_reference(stacked)
    moments = rng.standard_normal((8, dimension))

    expected = np.linalg.solve(regularized, moments.T).T

    from scipy.linalg import solve as scipy_solve

    got = scipy_solve(regularized, moments.T, assume_a="pos").T
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-7)


def test_install_scipy_ridge_binds_and_reports_label() -> None:
    class _FakeCircuit:  # attribute surface used by the installer only
        pass

    holder = _FakeCircuit()
    label = HARNESS.install_scipy_ridge(holder)  # type: ignore[arg-type]
    assert label == "scipy-openblas-dsyrk"
    assert callable(holder._flush_growth_statistics)
    assert callable(holder._consolidate_readout)
