"""Drafted tests: multi-timescale trace REUSE in the qa path (overnight wave).

The R0 falsifier's named successors — spike-count windows (family A),
voltage quantiles (family B), assembly overlap (family C) — are
multi-timescale trace banks maintained at every ``observe()`` boundary and
consumed through ``feature_state_vector()`` by strengthening, readout, and the
QA-v1 path (``benchmarks.qa_experiments`` grows/completes
``RecurrentCellLanguageModel`` over ``qa_prompt`` texts with
``RecurrentCircuitConfig(feature_family=...)``, exactly the r0b gate G2
contract).

These drafts pin the reuse contract:

1. the family-A window-count rows are the single source of truth: an
   independent hand-computed sliding-window sum matches the bank bit-exactly;
2. a qa ``complete()`` drives those banks through the standard boundary
   updates — the captured per-token feature trajectory is reproducible
   bit-exactly across repeated completions of the same prompt (no parallel
   recomputation path, no drift);
3. family-B quantile rows are CACHED between consecutive readout queries
   (``_quantile_stale`` latch): repeated ``feature_state_vector()`` calls do
   not recompute, and the next observe invalidates the cache;
4. combo families flow through the qa experiment configuration unchanged.

Scope note: these are drafted during the Bible-QA corpus wave (branch
codex/a013-cognition-design) as TDD anchors for the qa-path trace-reuse
requirement; where behavior is already implemented they are green pins.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from humanbrain.qa_corpus import build_curriculum, growth_text, prompt_for_task
from humanbrain.recurrent import (
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
    parse_feature_family,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "qa_experiments", ROOT / "benchmarks" / "qa_experiments.py"
)
QA = importlib.util.module_from_spec(_SPEC)
# qa_experiments imports only ``humanbrain.*``; registering the module keeps
# repeat loads cheap without packaging benchmarks/ as a package.
sys.modules.setdefault("qa_experiments", QA)
_SPEC.loader.exec_module(QA)


def _family_circuit(family: str, **overrides: object) -> RecurrentCellCircuit:
    config_kwargs = dict(
        reservoir_cells=24,
        active_reservoir_cells=18,
        input_fanout=4,
        recurrent_fanout=2,
        token_interval_ms=2.0,
        input_pulse_ms=1.0,
        input_current_pa=8_000.0,
        recurrent_current_pa=325.0,
        random_seed=41,
        feature_family=family,
    )
    config_kwargs.update(overrides)
    return RecurrentCellCircuit(
        token_cells=32,
        config=RecurrentCircuitConfig(**config_kwargs),
    )


# --------------------------------------------------------------------------
# 1. Family-A rows equal an independent hand-computed multi-window sum
# --------------------------------------------------------------------------


def test_qa_trace_reuse_A_window_rows_match_hand_computed_sums() -> None:
    circuit = _family_circuit("spike_counts")
    widths = [int(w) for w in circuit._feature_window_tokens]
    assert widths == sorted(widths) and all(w > 0 for w in widths)

    rng = np.random.default_rng(7)
    stream = [int(t) for t in rng.integers(0, circuit.token_cells, size=40)]
    snaps: list[np.ndarray] = []
    for step, token in enumerate(stream, start=1):
        circuit.observe(token)
        # snapshot AFTER observe returns == the vector pushed this boundary
        snaps.append(circuit._token_spike_counts.astype(np.int64).copy())
        rows = circuit._window_counts.reshape(len(widths), -1)
        for k, w in enumerate(widths):
            expected = np.zeros(rows.shape[1], dtype=np.int64)
            for snap in snaps[-w:]:
                expected += snap
            np.testing.assert_array_equal(
                rows[k], expected,
                err_msg=f"step {step}: window {w} row diverged from hand sum",
            )


# --------------------------------------------------------------------------
# 2. qa complete() reuses the banks: reproducible per-token trajectory
# --------------------------------------------------------------------------


def _spy_trajectory(model, prompt: str, max_new_tokens: int) -> list[np.ndarray]:
    circuit = model._circuit  # noqa: SLF001 - benchmark/test seam
    captured: list[np.ndarray] = []
    original_observe = circuit.observe

    def spy(token_id: int) -> object:
        result = original_observe(token_id)
        captured.append(circuit.feature_state_vector().copy())
        return result

    circuit.observe = spy  # type: ignore[method-assign]
    try:
        model.complete(prompt, max_new_tokens=max_new_tokens)
    finally:
        del circuit.observe  # restore the bound method
    return captured


def test_qa_trace_reuse_complete_trajectory_reproducible_bit_exact() -> None:
    from humanbrain.recurrent_language import (
        RecurrentCellLanguageModel,
        RecurrentModelConfig,
    )

    curriculum = build_curriculum(seed=7)
    train_tasks = curriculum["train_tasks"][:3]
    model = RecurrentCellLanguageModel.grow(
        [growth_text(task) for task in train_tasks],
        RecurrentModelConfig(
            growth_epochs=1,
            max_phrases=64,
            min_phrase_frequency=1,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=64,
                active_reservoir_cells=48,
                input_fanout=8,
                recurrent_fanout=4,
                random_seed=11,
                feature_family="spike_counts",
            ),
        ),
    )
    prompt = prompt_for_task(curriculum["calibration_tasks"][0])
    traj_a = _spy_trajectory(model, prompt, max_new_tokens=6)
    traj_b = _spy_trajectory(model, prompt, max_new_tokens=6)
    assert len(traj_a) == len(traj_b) > 0
    evolved = any(not np.array_equal(traj_a[0], vec) for vec in traj_a[1:])
    assert evolved, "multi-timescale rows must advance across qa completion"
    for a, b in zip(traj_a, traj_b):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# 3. Family-B quantile rows are cached between consecutive readout queries
# --------------------------------------------------------------------------


def test_qa_trace_reuse_B_quantile_cache_latch() -> None:
    circuit = _family_circuit("voltage_quantiles")
    for token in (3, 7, 11):
        circuit.observe(token)
    first = circuit.feature_state_vector().copy()
    assert circuit._quantile_stale is False, "first readout refreshes the cache"
    second = circuit.feature_state_vector()
    np.testing.assert_array_equal(first, second)
    assert circuit._quantile_stale is False, "second readout must be a cache hit"
    circuit.observe(5)
    assert circuit._quantile_stale is True, "observe invalidates the cache"
    third = circuit.feature_state_vector()
    assert circuit._quantile_stale is False
    assert third.shape == first.shape


# --------------------------------------------------------------------------
# 4. Combo families flow through the qa experiment configuration unchanged
# --------------------------------------------------------------------------


def test_qa_trace_reuse_combo_family_accepted_by_qa_experiment_config() -> None:
    members = parse_feature_family("combo:spike_counts+voltage_quantiles")
    assert members == ("spike_counts", "voltage_quantiles")
    cfg = QA.ExperimentConfig(
        seed=7,
        max_train_tasks=2,
        growth_epochs=1,
        circuit=RecurrentCircuitConfig(
            reservoir_cells=48,
            active_reservoir_cells=36,
            input_fanout=6,
            recurrent_fanout=3,
            random_seed=13,
            feature_family="combo:spike_counts+voltage_quantiles",
        ),
        eval_limit=1,
    )
    assert cfg.circuit.feature_family == "combo:spike_counts+voltage_quantiles"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
