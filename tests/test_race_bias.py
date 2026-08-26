"""Probe P-B tests: predictive-state depolarizing bias into the first-spike race.

Gate order mirrors the preregistration
(experiments/race_bias_probes/preregistration.json): the FIRST tests are the
flag-off bit-parity gates — a cortex built with the bias machinery present but
flagged OFF must be byte-identical to the incumbent race (records and grown
branch state), or nothing else may run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from humanbrain.cortex import DendriticCortex
from humanbrain.race_bias import (
    BIAS_GAIN_PA_DEFAULT,
    CONTACT_CAP,
    DECREMENT_FRACTION_DEFAULT,
    PREDICTED_SEGMENT_DECREMENT_FRACTION,
    THETA_CONNECTED_DEFAULT,
    PredictiveBiasConfig,
    adjust_branch_strength,
    apply_predictive_bias,
    connected_contact_count,
    grow_with_segment_decrement,
)

REPO = Path(__file__).resolve().parents[1]


def _stream(rng: np.random.Generator, length: int, alphabet: int) -> list[int]:
    return [int(x) for x in rng.integers(0, alphabet, size=length)]


def _state_hash(cortex: DendriticCortex) -> str:
    rows = [
        [json.dumps(list(context)), int(target), repr(float(strength))]
        for context, target, strength in cortex.iter_branches()
    ]
    rows.sort()
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _race_records(cortex: DendriticCortex, contexts: list[list[int]]) -> list[dict]:
    records = []
    for context in contexts:
        race = cortex.race(context)
        records.append(
            {
                "winner": int(race.winner),
                "spike_time_ms": repr(float(race.spike_time_ms)),
                "peak_voltage_mv": repr(float(race.peak_voltage_mv)),
                "active_cells": int(race.active_cells),
                "simulated_steps": int(race.simulated_steps),
            }
        )
    return records


def _paired_off_cortex(cell_count: int = 32, seed: int = 7) -> tuple:
    """Incumbent cortex and flag-off probe cortex, identical wiring."""
    baseline = DendriticCortex(cell_count=cell_count, max_order=4, seed=seed)
    flagged = DendriticCortex(
        cell_count=cell_count,
        max_order=4,
        seed=seed,
        predictive_bias=PredictiveBiasConfig(mode="off"),
    )
    return baseline, flagged


def _theta_connected_targets(cortex, context):
    out = {}
    for _order, target, strength in cortex.matching_segments(context):
        if connected_contact_count(strength) >= THETA_CONNECTED_DEFAULT:
            out.setdefault(target, 0)
            out[target] += 1
    return out


# ---------------------------------------------------------------------------
# GATE 1 — flag-off bit parity (must pass before anything else matters)
# ---------------------------------------------------------------------------


def test_flag_off_race_records_byte_identical():
    baseline, flagged = _paired_off_cortex()
    rng = np.random.default_rng(3)
    stream = _stream(rng, 400, 12)
    for cortex in (baseline, flagged):
        cortex.grow_sequence(stream)
    contexts = [stream[:k] for k in range(1, len(stream) + 1, 7)] + [
        _stream(np.random.default_rng(99), 6, 12)
    ]
    assert _race_records(flagged, contexts) == _race_records(baseline, contexts)


def test_flag_off_branch_state_hash_identical():
    baseline, flagged = _paired_off_cortex()
    rng = np.random.default_rng(5)
    stream = _stream(rng, 300, 10)
    for cortex in (baseline, flagged):
        cortex.grow_sequence(stream)
    assert _state_hash(flagged) == _state_hash(baseline)
    # And against the exact incumbent constructor path (no kwarg at all).
    plain = DendriticCortex(cell_count=32, max_order=4, seed=7)
    plain.grow_sequence(stream)
    assert _state_hash(plain) == _state_hash(baseline)


def test_flag_off_parity_with_decrement_machinery_idle():
    """The decrement replay helper at fraction 0 must mutate nothing."""
    baseline, flagged = _paired_off_cortex()
    for cortex in (baseline, flagged):
        cortex.grow_sequence(_stream(np.random.default_rng(8), 200, 10))
    probe_stream = _stream(np.random.default_rng(21), 60, 10)
    grow_with_segment_decrement(flagged, probe_stream, decrement_fraction=0.0)
    grow_with_segment_decrement(baseline, probe_stream, decrement_fraction=0.0)
    assert _state_hash(flagged) == _state_hash(baseline)


# ---------------------------------------------------------------------------
# Constants and configuration contract
# ---------------------------------------------------------------------------


def test_frozen_constants_match_preregistration():
    prereg = json.loads(
        (REPO / "experiments" / "race_bias_probes" / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    constants = prereg["constants"]
    assert THETA_CONNECTED_DEFAULT == constants["theta_connected"] == 13
    assert CONTACT_CAP == constants["contact_cap"] == 20
    assert float(BIAS_GAIN_PA_DEFAULT) == pytest.approx(
        constants["bias_gain_pa_per_matched_segment"]
    )
    assert DECREMENT_FRACTION_DEFAULT == pytest.approx(
        constants["predicted_segment_decrement_fraction"]
    )
    # Alias kept so callers can use the HTM-native name.
    assert PREDICTED_SEGMENT_DECREMENT_FRACTION == DECREMENT_FRACTION_DEFAULT


def test_predictive_bias_config_validation():
    with pytest.raises(ValueError):
        PredictiveBiasConfig(mode="teleport")
    with pytest.raises(ValueError):
        PredictiveBiasConfig(mode="proportional", bias_gain_pa=0.0)
    with pytest.raises(ValueError):
        PredictiveBiasConfig(mode="proportional", theta_connected=0)
    with pytest.raises(ValueError):
        PredictiveBiasConfig(mode="proportional", contact_cap=-3)
    assert PredictiveBiasConfig(mode="off").mode == "off"


def test_default_cortex_has_no_bias_object():
    assert DendriticCortex(cell_count=16, seed=3).predictive_bias is None


# ---------------------------------------------------------------------------
# Segment matching: lane-native instantiation of theta ~= 13 of ~20
# ---------------------------------------------------------------------------


def _seeded_cortex() -> DendriticCortex:
    return DendriticCortex(cell_count=64, max_order=4, seed=11)


def test_connected_contact_count_mapping():
    assert connected_contact_count(0.0) == 0
    assert connected_contact_count(1.0) == 1
    assert connected_contact_count(12.999) == 12  # floor below theta
    assert connected_contact_count(13.0) == 13  # exactly theta connects
    assert connected_contact_count(57.0) == CONTACT_CAP  # budget cap
    assert connected_contact_count(-4.0) == 0  # fully decremented segment


def test_matching_segments_orders_and_determinism():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2, 3, 4] * 6)
    segments = cortex.matching_segments([3, 4])
    orders = [order for order, _target, _strength in segments]
    targets = [target for _order, target, _strength in segments]
    strengths = [strength for _order, _target, strength in segments]
    assert orders == sorted(orders)  # deterministic ascending order
    assert set(targets) == {1}  # (3,4) is always followed by 1 here
    # The first 1 of the stream has no predecessors, so every (3,4)->1
    # transition occurs 5 times, not 6.
    assert all(strength == 5.0 for strength in strengths)
    assert cortex.matching_segments([3, 4]) == segments
    deep = cortex.matching_segments([1, 2, 3, 4])
    assert {order for order, _t, _s in deep} >= {1, 2, 3, 4}


def test_theta_gate_excludes_weak_segments():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2, 3])  # every branch strength 1 << theta
    segments = cortex.matching_segments([1, 2])
    assert segments, "raw matching must still see the weak branch"
    assert _theta_connected_targets(cortex, [1, 2]) == {}
    weak = [t for _o, t, s in segments if connected_contact_count(s) >= 1]
    assert weak.count(3) == len(segments)  # single continuation token


# ---------------------------------------------------------------------------
# Bias application arithmetic
# ---------------------------------------------------------------------------


def test_apply_predictive_bias_adds_exactly_gain_times_count():
    cortex = _seeded_cortex()
    cortex.grow_sequence([5, 9, 6] * 15 + [7, 9, 8] * 20)
    context = [5, 9]
    active, drive = cortex._dendritic_drive(context)
    cfg = PredictiveBiasConfig(mode="proportional", bias_gain_pa=30.0)
    biased = apply_predictive_bias(cortex, active, drive.copy(), context, cfg)
    counts = _theta_connected_targets(cortex, context)
    lookup = {int(cell): i for i, cell in enumerate(active)}
    expected_offsets = {
        cell: cfg.bias_gain_pa * count for cell, count in counts.items() if cell in lookup
    }
    assert expected_offsets, "fixture must produce at least one biased active cell"
    for index, cell in enumerate(active):
        offset = expected_offsets.get(int(cell), 0.0)
        assert biased[index] == pytest.approx(drive[index] + offset, rel=0, abs=1e-12)


def test_bias_zero_when_no_theta_connected_segment():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2, 3])  # all branches strength 1 < theta
    context = [2, 3]
    active, drive = cortex._dendritic_drive(context)
    cfg = PredictiveBiasConfig(mode="proportional")
    biased = apply_predictive_bias(cortex, active, drive.copy(), context, cfg)
    assert np.array_equal(biased, drive)


def test_proportional_scales_with_matched_segment_count():
    cortex = _seeded_cortex()
    # Cell 1 matches context (1,2,3,4) at every order 1..4; counts multiply.
    cortex.grow_sequence([1, 2, 3, 4] * 15)
    cfg = PredictiveBiasConfig(mode="proportional", bias_gain_pa=10.0)
    context = [1, 2, 3, 4]
    active, drive = cortex._dendritic_drive(context)
    biased = apply_predictive_bias(cortex, active, drive.copy(), context, cfg)
    lookup = {int(cell): i for i, cell in enumerate(active)}
    count_1 = _theta_connected_targets(cortex, context)[1]
    assert count_1 >= 2, "fixture relies on multi-order matching for cell 1"
    assert biased[lookup[1]] == pytest.approx(
        drive[lookup[1]] + cfg.bias_gain_pa * count_1, rel=0, abs=1e-12
    )


# ---------------------------------------------------------------------------
# Shuffled-bias control
# ---------------------------------------------------------------------------


def _unequal_counts_cortex() -> DendriticCortex:
    """Cell 3 matches ctx (8,1,2) at orders 1-2, cell 7 at orders 1-3, plus
    eight weak theta-failing targets -> active sheet of 10 cells with unequal
    per-cell bias magnitudes (60 and 90 drive units at gain 30)."""
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2, 3] * 20)
    cortex.grow_sequence([8, 1, 2, 7] * 30)
    for weak in range(50, 58):
        cortex.grow_sequence([1, 2, weak])
    return cortex


def test_shuffled_preserves_magnitude_multiset_and_detaches_assignment():
    def one_draw():
        cortex = _unequal_counts_cortex()
        context = [8, 1, 2]
        active, drive = cortex._dendritic_drive(context)
        prop = apply_predictive_bias(
            cortex,
            active,
            drive.copy(),
            context,
            PredictiveBiasConfig(mode="proportional"),
        )
        shuf = apply_predictive_bias(
            cortex,
            active,
            drive.copy(),
            context,
            PredictiveBiasConfig(mode="shuffled", control_seed=13),
        )
        return prop - drive, shuf - drive

    counts = _theta_connected_targets(_unequal_counts_cortex(), [8, 1, 2])
    assert counts == {3: 2, 7: 3}, "fixture drifted"
    draws = [one_draw() for _ in range(5)]
    for d_prop, d_shuf in draws:
        # Same magnitude multiset ...
        assert np.allclose(np.sort(d_prop), np.sort(d_shuf), rtol=0, atol=1e-12)
    # ... attached differently somewhere (identity reassignment coincides with
    # probability ~ (1/C(10,2))*(1/2) per draw; over 5 fresh draws ~ 1e-8).
    assert any(not np.allclose(p, s) for p, s in draws)


def test_shuffled_control_is_deterministic_given_seed():
    def run_once():
        cortex = _seeded_cortex()
        cortex.grow_sequence([5, 9, 6] * 15 + [7, 9, 8] * 20)
        context = [5, 9]
        active, drive = cortex._dendritic_drive(context)
        cfg = PredictiveBiasConfig(mode="shuffled", control_seed=29)
        out = []
        for _ in range(6):
            biased = apply_predictive_bias(cortex, active, drive.copy(), context, cfg)
            out.append((biased - drive).copy())
        return out

    first, second = run_once(), run_once()
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# predictedSegmentDecrement analog (+/- on converged-but-lost segments)
# ---------------------------------------------------------------------------


def _strength_map(cortex):
    return {
        (json.dumps(list(ctx)), tgt): float(s) for ctx, tgt, s in cortex.iter_branches()
    }


def test_replay_without_converged_context_never_decays():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2] * 20)
    before = _strength_map(cortex)
    # Replay [1]: only position t=0 exists (empty history) -> grow only.
    grow_with_segment_decrement(cortex, [1], decrement_fraction=0.04)
    assert _strength_map(cortex) == before
    # Replay [3, 1]: at t=1 the history (3,) matches NOTHING -> no converged
    # segment, so no decay; the new branch (3,)->1 appears from growth.
    grow_with_segment_decrement(cortex, [3, 1], decrement_fraction=0.04)
    after = _strength_map(cortex)
    for key, s_before in before.items():
        assert after[key] == pytest.approx(s_before, abs=1e-15)


def test_decrement_decays_only_the_converged_loser_by_exact_amount():
    cortex = _seeded_cortex()
    # ((1,))->2 strength 20, ((1,))->4 strength 25 -> race at (1,) picks 4;
    # the theta-connected loser segment ((1,))->2 must lose exactly 0.04.
    cortex.grow_sequence([1, 2] * 20 + [1, 4] * 25)
    winner = int(cortex.race([1]).winner)
    assert winner == 4
    before = _strength_map(cortex)
    grow_with_segment_decrement(cortex, [1, 2], decrement_fraction=0.04)
    after = _strength_map(cortex)
    key_loser = (json.dumps([1]), 2)
    key_winner = (json.dumps([1]), 4)
    # Loser: decayed 0.04 by the converged-lost event, then strengthened 1.0
    # by the replayed ground-truth continuation 2.
    assert after[key_loser] == pytest.approx(before[key_loser] + 1.0 - 0.04)
    # Winner segment: untouched by decrement; not the replayed target.
    assert after[key_winner] == pytest.approx(before[key_winner])


def test_decrement_minus_strengthens_losers_diagnostic():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2] * 20 + [1, 4] * 25)
    winner = int(cortex.race([1]).winner)
    assert winner == 4
    before = _strength_map(cortex)
    grow_with_segment_decrement(cortex, [1, 2], sign=-1)
    after = _strength_map(cortex)
    assert after[(json.dumps([1]), 2)] == pytest.approx(
        before[(json.dumps([1]), 2)] + 1.0 + 0.04
    )


def test_adjust_branch_strength_floors_at_zero_and_deletes():
    cortex = _seeded_cortex()
    cortex.grow_sequence([1, 2] * 20)
    assert adjust_branch_strength(cortex, (1,), 2, -1e6) is True
    remaining = [
        (list(ctx), tgt) for ctx, tgt, _s in cortex.iter_branches()
    ]
    assert ([1], 2) not in remaining
    assert adjust_branch_strength(cortex, (1,), 99, 1.0) is False
    assert adjust_branch_strength(cortex, (404,), 2, 1.0) is False


# ---------------------------------------------------------------------------
# End-to-end arm smoke through the public race path
# ---------------------------------------------------------------------------


def test_proportional_arm_end_to_end_runs_and_differs_when_bias_fires():
    baseline = _seeded_cortex()
    armed = _seeded_cortex()
    armed.predictive_bias = PredictiveBiasConfig(mode="proportional")
    stream = [1, 2, 3, 4, 5] * 40
    for cortex in (baseline, armed):
        cortex.grow_sequence(stream)
    contexts = [stream[:k] for k in range(1, 21)]
    base_records = _race_records(baseline, contexts)
    armed_records = _race_records(armed, contexts)
    finite = all(
        np.isfinite(float(r["spike_time_ms"]))
        and np.isfinite(float(r["peak_voltage_mv"]))
        for r in armed_records
    )
    assert finite
    assert armed_records != base_records  # bias actually entered some races
