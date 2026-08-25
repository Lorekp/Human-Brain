"""U-STATE week-1 red tests: determinism / capacity / decay identities.

Pins the representation spec frozen in docs/research/ustate-preregistration.md
§1 BEFORE/AFTER implementation of src/humanbrain/sdr_union.py:
seeded token->SDR projection (n=2048 default, w=32), union state
U_t = max(lam*U_{t-1}, s_t), count-union C_t = lam*C_{t-1} + s_t, encode/decode
helpers, and the separation diagnostics consumed by the D3 logging harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.sdr_union import (
    DEFAULT_LAMBDA,
    DEFAULT_N_BITS,
    DEFAULT_SEED,
    DEFAULT_W_ONES,
    SDRProjector,
    UnionEncoder,
    UnionState,
    cosine_separation,
    decode_tokens,
    token_sdr_indices,
)


# ------------------------------------------------------------------ projection


def test_projection_deterministic_across_instances():
    a = SDRProjector()
    b = SDRProjector()
    for tok in (0, 7, 65, 255, 12345):
        assert np.array_equal(a.indices(tok), b.indices(tok))
        assert np.array_equal(a.vector(tok), b.vector(tok))
        assert np.array_equal(token_sdr_indices(tok), a.indices(tok))


def test_projection_shape_weight_sorted():
    p = SDRProjector()
    for tok in (0, 1, 255):
        idx = p.indices(tok)
        assert idx.shape == (DEFAULT_W_ONES,)
        assert idx.dtype == np.int64
        assert np.all(np.diff(idx) > 0), "active indices must be sorted unique"
        assert idx.min() >= 0 and idx.max() < DEFAULT_N_BITS
        vec = p.vector(tok)
        assert vec.shape == (DEFAULT_N_BITS,)
        assert vec.sum() == DEFAULT_W_ONES
        assert set(np.unique(vec)) == {0.0, 1.0}


def test_projection_differs_across_tokens_and_seeds():
    p7 = SDRProjector(seed=7)
    p8 = SDRProjector(seed=8)
    v_a = p7.vector(42)
    v_b = p7.vector(43)
    v_c = p8.vector(42)
    assert ((v_a > 0) & (v_b > 0)).sum() <= 2  # w^2/n = 0.5 expected; tiny tail
    assert not np.array_equal(v_a, v_c)


def test_projection_validation():
    p = SDRProjector()
    with pytest.raises(ValueError):
        p.indices(-1)
    with pytest.raises(ValueError):
        SDRProjector(n_bits=16, w_ones=17)
    with pytest.raises(ValueError):
        UnionEncoder(lam=1.5)


# -------------------------------------------------------------- union dynamics


def _stream(n, lo=0, hi=256, seed=123):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n).tolist()


def test_union_fresh_write_and_decay_values():
    p = SDRProjector()
    u = UnionState(p, lam=0.90)
    u.step(11)
    assert np.array_equal(u.value, p.vector(11))
    u.step(12)
    # fresh token bits == 1; surviving bits of token 11 scaled by lam unless shared
    fresh = p.vector(12) == 1.0
    assert np.all(u.value[fresh] == 1.0)
    old_only = (p.vector(11) == 1.0) & ~fresh
    if old_only.any():
        assert np.all(u.value[old_only] == pytest.approx(0.90))


def test_decay_identity_geometric():
    """k no-input decay steps must scale both states by lam^k exactly."""
    p = SDRProjector(seed=7)
    u = UnionState(p, lam=0.97)
    for tok in _stream(40, seed=5):
        u.step(int(tok))
    union_before = u.snapshot().copy()
    count_before = u.count_snapshot().copy()
    k = 6
    for _ in range(k):
        u.decay()
    assert np.allclose(u.value, union_before * (0.97**k), rtol=0, atol=1e-15)
    assert np.allclose(u.count_value, count_before * (0.97**k), rtol=0, atol=1e-15)


def test_count_union_linear_recurrence():
    p = SDRProjector(seed=7)
    lam = 0.97
    enc = UnionEncoder(lam=lam)
    stream = [3, 3, 9, 3, 200]
    c = np.zeros(DEFAULT_N_BITS)
    for tok in stream:
        c = lam * c + p.vector(tok)
    out = enc.encode(stream)
    assert np.allclose(out.count_union, c, rtol=0, atol=1e-15)


def test_reset_replays_bit_identical():
    enc_a = UnionEncoder(lam=0.97)
    enc_b = UnionEncoder(lam=0.97)
    stream = _stream(150)
    a1 = enc_a.encode(stream)
    a2 = enc_a.encode(stream)  # encode() resets internally
    b = enc_b.encode(stream)
    assert np.array_equal(a1.union, a2.union)
    assert np.array_equal(a1.union, b.union)
    assert np.array_equal(a1.last_sdr, b.last_sdr)


def test_encode_incremental_equivalence():
    p = SDRProjector()
    incremental = UnionState(p, lam=0.99)
    enc = UnionEncoder(lam=0.99)
    stream = _stream(80, seed=9)
    for tok in stream:
        incremental.step(int(tok))
    out = enc.encode(stream)
    assert np.array_equal(incremental.value, out.union)


def test_lambda_sweep_arms_pinned():
    for name, lam in (("lam090", 0.90), ("lam097", 0.97), ("lam099", 0.99)):
        enc = UnionEncoder(lam=lam)
        assert enc.lam == lam
    assert DEFAULT_LAMBDA == 0.97
    assert (DEFAULT_N_BITS, DEFAULT_W_ONES, DEFAULT_SEED) == (2048, 32, 7)


# -------------------------------------------------------------------- capacity


def test_capacity_pairwise_overlap_matches_w2_over_n():
    """Mean pairwise SDR overlap over many token pairs ~= w^2/n = 0.5 bit."""
    p = SDRProjector()
    rng = np.random.default_rng(DEFAULT_SEED)
    toks = rng.choice(100000, size=600, replace=False).tolist()
    vecs = np.stack([p.vector(t) for t in toks])
    gram = vecs @ vecs.T
    iu = np.triu_indices(len(toks), k=1)
    overlaps = gram[iu]
    assert abs(overlaps.mean() - DEFAULT_W_ONES**2 / DEFAULT_N_BITS) < 0.05
    assert np.median(overlaps) == 0.0


def test_capacity_fresh_union_support_analytic():
    """Support of a max-union of L distinct tokens ~ n(1-(1-w/n)^L) within 10%."""
    p = SDRProjector()
    enc = UnionEncoder(lam=1.0)  # lam=1: pure fresh union, no decay shrinkage
    rng = np.random.default_rng(DEFAULT_SEED)
    toks = rng.choice(100000, size=50, replace=False).tolist()
    out = enc.encode(toks)
    analytic = DEFAULT_N_BITS * (1.0 - (1.0 - DEFAULT_W_ONES / DEFAULT_N_BITS) ** 50)
    measured = float((out.union > 0).sum())
    assert abs(measured - analytic) / analytic < 0.10


def test_union_support_monotone_under_updates():
    p = SDRProjector()
    u = UnionState(p, lam=0.99)
    supports = []
    for tok in _stream(60, seed=4):
        u.step(int(tok))
        supports.append(int((u.value > 0).sum()))
    assert all(b >= a for a, b in zip(supports, supports[1:]))


def test_structural_separation_disjoint_alphabets():
    """Unions over disjoint byte alphabets are near-disjoint BY CONSTRUCTION.

    Collision math: supports of ~128 bits each over n=2048 share ~8 accidental
    bits in expectation, so 4-token unions land near separation ~0.9 — an
    order of magnitude above the G-U1 bar (0.10). Single-token unions collide
    on <=1 bit, giving separation >= 0.97.
    """
    enc = UnionEncoder(lam=0.97)
    left = enc.encode([65, 66, 67, 68] * 20)
    right = enc.encode([112, 113, 114, 115] * 20)
    sep = cosine_separation(np.stack([left.union, right.union]))
    assert sep["pooled_median_one_minus_cosine"] >= 0.85
    one_a = enc.encode([65])
    one_b = enc.encode([115])
    sep_single = cosine_separation(np.stack([one_a.union, one_b.union]))
    assert sep_single["pooled_median_one_minus_cosine"] >= 0.97
    same = enc.encode([65, 66, 67, 68] * 20)
    sep_same = cosine_separation(np.stack([left.union, same.union]))
    assert sep_same["pooled_median_one_minus_cosine"] == pytest.approx(0.0)


# --------------------------------------------------------------------- decode


def test_decode_ranks_recent_token_first():
    enc = UnionEncoder(lam=0.97)
    stream = [7] * 5 + [250]
    out = enc.encode(stream)
    ranking = decode_tokens(out.union, enc.projector, top_k=8)
    assert ranking[0] == 250
    assert 7 in ranking[:4]


def test_decode_helper_validation():
    p = SDRProjector()
    with pytest.raises(ValueError):
        decode_tokens(np.zeros(17), p, top_k=4)
