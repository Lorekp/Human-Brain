"""T-ROUTE week-1 red tests: determinism / Hebbian identity / scaling normalization.

Pins the representation spec frozen in docs/research/trained-inter-space-
preregistration.md BEFORE implementation of src/humanbrain/hebbian_projection.py:

- sparse W projection (seeded-random init, same projector-keyed convention as
  sdr_union.py), one-shot outer-product accumulation dW = pre x post x M,
- homeostatic IDF-like row scaling + row L2 normalization,
- kWinner route selection (exact-k, unique, deterministic tie-break),
- shuffled-W value permutation preserving support and value multiset,
- episode-level determinism: identical inputs -> bit-identical matrices and
  routed winner sets.

Written RED first (module absent); turned green by the D2-3 implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.hebbian_projection import (
    DEFAULT_FAN_IN,
    DEFAULT_K_WINNERS,
    DEFAULT_N_BITS,
    DEFAULT_W_SEED,
    HebbianConfig,
    SparseProjection,
    kwinner_indices,
)
from humanbrain.sdr_union import SDRProjector


TOY = HebbianConfig(n_bits=64, w_ones=8, fan_in=6, w_seed=7, k_winners=12)


def _toy_events(projector: SDRProjector) -> list[tuple[int, int]]:
    """Deterministic (pre_token, post_token) pairing events on a toy vocab."""
    return [(0, 10), (0, 11), (1, 10), (2, 12), (3, 13), (3, 13)]


# ------------------------------------------------------------------- topology


def test_topology_seeded_and_reproducible():
    a = SparseProjection(TOY)
    b = SparseProjection(TOY)
    assert np.array_equal(a.support(), b.support())
    assert np.array_equal(a.values(), b.values())
    # Same projector-keyed convention as sdr_union: keyed by (seed, n, k, post).
    assert a.support().shape == (TOY.n_bits, TOY.fan_in)


def test_topology_partner_uniqueness_per_post_bit():
    proj = SparseProjection(TOY)
    sup = proj.support()
    for post in range(TOY.n_bits):
        partners = sup[post]
        assert len(set(partners.tolist())) == TOY.fan_in  # no replacement


# ----------------------------------------------------------- hebbian identity


def test_hebbian_accumulation_identity_outer_products():
    """dW equals the support-masked sum of pre(x)post M outer products.

    The freeze pins training ONLY at projection synapses (strategy: "trained
    only at projection synapses"), so the dense expectation is each event's
    outer product masked to W's sparse support.
    """
    projector = SDRProjector(n_bits=TOY.n_bits, w_ones=TOY.w_ones, seed=TOY.w_seed)
    events = _toy_events(projector)

    trained = SparseProjection(TOY)
    trained.accumulate_events(
        [(projector.indices(p).tolist(), projector.indices(q).tolist()) for p, q in events]
    )

    expected = SparseProjection(TOY)  # same init, no training
    support_mask = np.zeros((TOY.n_bits, TOY.n_bits), dtype=bool)
    sup = expected.support()
    support_mask[np.repeat(np.arange(TOY.n_bits), TOY.fan_in), sup.ravel()] = True
    delta_expected = np.zeros((TOY.n_bits, TOY.n_bits), dtype=np.float64)
    for p, q in events:
        # W convention: rows = post (passage) bits, cols = pre (question) bits.
        delta_expected += support_mask * np.outer(projector.vector(q), projector.vector(p))
    assert np.allclose(trained.to_dense() - expected.to_dense(), delta_expected,
                       rtol=0, atol=1e-12)
    # Repeated identical events accumulate multiplicities (additivity).
    n_block_entries = len(events) * TOY.w_ones * TOY.w_ones
    masked_entries = int((delta_expected > 0).sum())
    assert masked_entries <= n_block_entries


def test_one_shot_no_epoch_amplification():
    """A second accumulate pass with zero events changes nothing (one-shot rule)."""
    projector = SDRProjector(n_bits=TOY.n_bits, w_ones=TOY.w_ones, seed=TOY.w_seed)
    events = [
        (projector.indices(p).tolist(), projector.indices(q).tolist())
        for p, q in _toy_events(projector)
    ]
    once = SparseProjection(TOY)
    once.accumulate_events(events)
    snapshot = once.to_dense().copy()
    once.accumulate_events([])  # empty pass: no-op
    assert np.array_equal(once.to_dense(), snapshot)


def test_apply_dense_delta_matches_eventwise():
    """Aggregated token-pair counts land identical mass as event-wise updates."""
    projector = SDRProjector(n_bits=TOY.n_bits, w_ones=TOY.w_ones, seed=TOY.w_seed)
    events = _toy_events(projector)
    ev_bits = [
        (projector.indices(p).tolist(), projector.indices(q).tolist())
        for p, q in events
    ]
    eventwise = SparseProjection(TOY)
    eventwise.accumulate_events(ev_bits)

    counts: dict[tuple[int, int], float] = {}
    for p, q in events:
        counts[(p, q)] = counts.get((p, q), 0.0) + 1.0
    dense = np.zeros((TOY.n_bits, TOY.n_bits), dtype=np.float64)
    for (p, q), m in counts.items():
        dense[np.ix_(projector.indices(q), projector.indices(p))] += m

    fast = SparseProjection(TOY)
    fast.apply_dense_delta(dense)
    assert np.allclose(fast.to_dense(), eventwise.to_dense(), rtol=0, atol=1e-12)


# ------------------------------------------------------- scaling normalization


def test_homeostatic_row_normalization():
    """Change Note 1 formula: rows unit-normalized, then scaled by idf(b).

    Nonzero rows end with L2 norm == idf(df_b); IDF ordering is preserved as a
    per-bit gain; structurally zero rows stay zero.
    """
    projector = SDRProjector(n_bits=TOY.n_bits, w_ones=TOY.w_ones, seed=TOY.w_seed)
    events = [
        (projector.indices(p).tolist(), projector.indices(q).tolist())
        for p, q in _toy_events(projector)
    ]
    proj = SparseProjection(TOY)
    proj.accumulate_events(events)
    df = np.zeros(TOY.n_bits, dtype=np.int64)
    for t in (10, 11, 12, 13):
        df[projector.indices(t)] += 1
    view = proj.apply_homeostasis(df, n_passages=4)
    scaled = view.to_dense()
    expected_scales = np.array([TOY.idf_scale(int(d), 4) for d in df])
    row_norms = np.linalg.norm(scaled, axis=1)
    touched = np.linalg.norm(proj.to_dense(), axis=1) > 0  # all rows here
    assert np.allclose(row_norms[touched], expected_scales[touched], atol=1e-12)
    # Direction preserved: cosine between raw and scaled rows is 1 on touched.
    raw = proj.to_dense()
    dots = (raw * scaled).sum(axis=1)
    assert np.allclose(dots[touched], row_norms[touched] * np.linalg.norm(raw, axis=1)[touched])


def test_idf_scale_monotone_in_document_frequency():
    """Rarer post-bits get larger-or-equal scale factors (IDF-like ordering)."""
    cfg = HebbianConfig(n_bits=64, w_ones=8, fan_in=6, w_seed=7, k_winners=12)
    scale_rare = cfg.idf_scale(df_bit=1, n_passages=100)
    scale_common = cfg.idf_scale(df_bit=90, n_passages=100)
    assert scale_rare > scale_common > 0.0


# ------------------------------------------------------------------- routing


def test_kwinner_exact_k_unique_deterministic_ties():
    drive = np.zeros(32)
    drive[[5, 3, 3]] = [1.0, 2.0, 2.0]  # duplicate index ignored via set semantics
    winners = kwinner_indices(drive, k=4)
    assert len(winners) == 4 and len(set(winners.tolist())) == 4
    again = kwinner_indices(drive.copy(), k=4)
    assert np.array_equal(winners, again)
    # Tie between equal-valued indices resolves by ascending bit index:
    tie = np.zeros(8)
    tie[[2, 4, 6]] = 0.5
    assert kwinner_indices(tie, k=2).tolist() == [2, 4]


def test_route_determinism_end_to_end():
    """Identical episodes -> bit-identical W and winner sets (freeze §1)."""
    projector = SDRProjector(n_bits=TOY.n_bits, w_ones=TOY.w_ones, seed=TOY.w_seed)
    events = [
        (projector.indices(p).tolist(), projector.indices(q).tolist())
        for p, q in _toy_events(projector)
    ]
    q_pop = np.maximum.reduce([projector.vector(t) for t in (0, 1, 2)])

    runs = []
    for _ in range(2):
        proj = SparseProjection(TOY)
        proj.accumulate_events(events)
        df = np.zeros(TOY.n_bits, dtype=np.int64)
        for t in (10, 11, 12, 13):
            df[projector.indices(t)] += 1
        scaled = proj.apply_homeostasis(df, n_passages=4)
        runs.append((scaled.to_dense().copy(), scaled.route(q_pop)))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])
    assert len(runs[0][1]) == TOY.k_winners


# ---------------------------------------------------------------- shuffled-W


def test_shuffle_values_preserves_support_and_multiset():
    proj = SparseProjection(TOY)
    sup_before = proj.support().copy()
    vals_before = np.sort(proj.values(), axis=None)
    shuf = proj.shuffled_values_copy(seed=11)
    assert np.array_equal(shuf.support(), sup_before)
    assert np.array_equal(np.sort(shuf.values(), axis=None), vals_before)
    # And it is actually permuted (astronomically unlikely to be identity).
    assert not np.array_equal(shuf.values(), proj.values()) or TOY.fan_in * TOY.n_bits < 64


def test_public_constants_match_freeze():
    """Freeze pins: n=2048, w implied by projector reuse, fan-in 32, K=128, seed 7."""
    assert DEFAULT_N_BITS == 2048
    assert DEFAULT_FAN_IN == 32
    assert DEFAULT_K_WINNERS == 128
    assert DEFAULT_W_SEED == 7
