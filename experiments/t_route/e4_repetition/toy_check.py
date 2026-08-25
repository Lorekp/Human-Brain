#!/usr/bin/env python3
"""Pre-spend mechanics validation for E4 (Change Note 5 §8 discipline).

Validates BEFORE any corpus encoding; writes only throwaway temp files:
  T1  multiplicative arm math: values == init*(1+b)^C exactly; CAP_EXP clamp bites;
      round_update increments fire-counts ONLY on (winner post bit x existing
      support synapse) pairs, aggregating within-round multiplicities exactly;
  T2  gated-additive equivalence: with k_winners == n_bits (every post bit wins,
      gating vacuous) the sequential add_gated encoder lands BIT-IDENTICAL weights
      to the E1-proven batch apply_dense_delta path over the same episode stream;
  T3  end-to-end scorer: synthetic full-schema logs with hand-set per-stratum
      overlaps reproduce all three frozen verdict scenarios (REPEATED-only PASS =>
      regime confirmed; nothing PASSes => pivot closes; both PASS => E3 undermined).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

E4_DIR = Path(__file__).resolve().parent
REPO = E4_DIR.parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.hebbian_projection import HebbianConfig, SparseProjection
from humanbrain.sdr_union import SDRProjector

spec_run = importlib.util.spec_from_file_location("run_e4_mod", E4_DIR / "run_e4.py")
run_e4 = importlib.util.module_from_spec(spec_run)
spec_run.loader.exec_module(run_e4)

spec_score = importlib.util.spec_from_file_location("score_e4_mod", E4_DIR / "score_e4.py")
score_e4 = importlib.util.module_from_spec(spec_score)
spec_score.loader.exec_module(score_e4)

TOY_ARMS = ("mult_b0.05", "mult_b0.25", "mult_b0.5", "add_gated", "add_free")


def t1_mult_math() -> None:
    cfg = HebbianConfig(n_bits=64, w_ones=4, fan_in=4, w_seed=7, k_winners=64)
    proj = SDRProjector(n_bits=64, w_ones=4, seed=7)
    enc = run_e4.GateEncoder(cfg, proj)
    m = run_e4.MultProjection(enc, 0.5)

    # values == init * (1+beta)^counts, exact
    m.counts[3, 1] = 5
    assert m.values()[3, 1] == enc._init[3, 1] * 1.5 ** 5
    # clamp bites: cap_exp=3 => value freezes at (1.5)^3 despite count 9
    m.cap_exp = 3
    m.counts[3, 1] = 9
    assert m.values()[3, 1] == enc._init[3, 1] * 1.5 ** 3

    # round_update gating: k_winners==n_bits so ALL post bits win; expect exact
    # fire-count increments on existing support synapses of expanded event blocks.
    m2 = run_e4.MultProjection(enc, 0.05)
    q_ids = [10, 11]
    p_ids = [10, 12, 13, 10]  # token 10 anchors at j=0 and j=3 (O_MAX=8)
    tok_counts = run_e4.episode_pair_counts(q_ids, p_ids)
    assert tok_counts, "toy episode produced no pairing events"
    q_pop = np.zeros(64)
    q_pop[proj.indices(10)] = 1.0
    q_pop[proj.indices(11)] = 1.0
    before = m2.counts.copy()
    m2.round_update(q_pop, tok_counts)
    delta_counts = m2.counts - before
    # independent expectation: full w x w blocks, gated by lut >= 0 (all bits win)
    expect = np.zeros_like(before)
    for (tq, tp), mult in tok_counts.items():
        for b in proj.indices(tp):
            for a in proj.indices(tq):
                col = enc._lut[b, a]
                if col >= 0:
                    expect[b, col] += mult
    assert np.array_equal(delta_counts, expect), "fire-count mismatch vs expectation"
    # multiplicity compounding visible somewhere (>1 on some synapse given 2 anchors)
    assert (expect > 1).any(), f"toy expected multiplicity compounding, got {expect.max()}"
    print("T1 mult math: OK")


def t2_gated_additive_equivalence() -> None:
    cfg = HebbianConfig(n_bits=64, w_ones=4, fan_in=4, w_seed=7, k_winners=64)
    proj = SDRProjector(n_bits=64, w_ones=4, seed=7)
    enc = run_e4.GateEncoder(cfg, proj)
    gated = run_e4.AddGatedProjection(enc)

    episodes = [
        ([10, 11], [10, 12, 13, 10]),
        ([11, 14], [14, 15, 10]),
        ([12], [12, 13, 14, 15, 12]),
    ]
    global_counts: Counter = Counter()
    for q_ids, p_ids in episodes:
        qc = run_e4.episode_pair_counts(q_ids, p_ids)
        q_pop = np.zeros(64)
        for t in q_ids:
            q_pop[proj.indices(t)] = 1.0
        gated.round_update(q_pop, qc)  # all bits win -> gating vacuous
        global_counts.update(qc)
    batch = SparseProjection(cfg)
    batch.apply_dense_delta(run_e4.dense_delta_from_pairs(global_counts, proj))
    # integer-valued float additions commute exactly
    assert np.array_equal(gated.values(), batch.values()), "gated != batch"
    print("T2 gated-additive == batch-additive under vacuous gating: OK")


def _write_synthetic_log(
    tmp: Path, spec: dict[str, dict[str, float]], ctrl_levels: dict[str, float]
) -> Path:
    """Full-schema E4 log; spec maps routing_set -> {stratum: target overlap}."""
    n_items, k, n_bits = 40, 32, 64
    strata = np.array(["REPEATED"] * 20 + ["SINGLE"] * 20)
    gold = np.zeros((n_items, n_bits), dtype=bool)
    for i in range(n_items):
        lo = (i * 3) % (n_bits - k)
        gold[i, lo:lo + k] = True

    def winners(levels: dict[str, float], seed: int) -> np.ndarray:
        out = np.zeros((n_items, k), dtype=np.int16)
        g = np.random.default_rng(seed)
        for i in range(n_items):
            lvl = int(round(levels[strata[i]]))
            gbits = np.flatnonzero(gold[i])
            fillers = np.flatnonzero(~gold[i])
            chosen = list(g.choice(gbits, size=lvl, replace=False))
            chosen += list(g.choice(fillers, size=k - lvl, replace=False))
            out[i] = np.array(sorted(chosen), dtype=np.int16)
        return out

    sets, rows = [], []
    for rset, lv in spec.items():
        sets.append(rset)
        rows.append(winners(lv, 1000 + len(sets)))
    sets.append("randomized_W")
    rows.append(winners(ctrl_levels, 999))
    arrays = {
        "winner_sets": np.concatenate(rows),
        "routing_set": np.concatenate([np.full(n_items, s, dtype="U20") for s in sets]),
        "item_index": np.tile(np.arange(n_items), len(sets)),
        "gold_mask": gold,
        "gold_sizes": gold.sum(axis=1).astype(np.int64),
        "matched": np.ones(n_items, dtype=bool),
        "stratum": strata.astype("U8"),
        "pair_count": np.where(strata == "REPEATED", 27, 1).astype(np.int64),
        "shuffle_perm": np.arange(n_items),
        "episode_ids": np.array([f"syn-{i:06d}" for i in range(n_items)], dtype="U24"),
        "answers": np.array(["X"] * n_items, dtype="U24"),
    }
    npz = tmp / "e4_logged_winner_sets.npz"
    np.savez_compressed(npz, **arrays)
    hashes = {
        name: {"sha256": hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest(),
               "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }
    manifest = {
        "schema_version": "toy",
        "config": {"betas": [0.05]},
        "integrity": {},
        "max_synapses_at_cap_by_arm": {},
        "array_sha256": hashes,
    }
    (tmp / "e4_manifest.json").write_text(json.dumps(manifest))
    return tmp


def _run_scenario(
    name: str, spec: dict[str, dict[str, float]], expect: str,
    ctrl_levels: dict[str, float],
) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = _write_synthetic_log(Path(td), spec, ctrl_levels)
        score_e4.NPZ_PATH = tmp / "e4_logged_winner_sets.npz"
        score_e4.MANIFEST_PATH = tmp / "e4_manifest.json"
        score_e4.RESULTS_PATH = tmp / "e4_results.json"
        score_e4.main()
        res = json.loads(score_e4.RESULTS_PATH.read_text())
        assert res["verdict"] == expect, (name, res["verdict"], expect)
        tab = res["gates"]["table"]
        if name == "regime-confirmed":
            for arm in TOY_ARMS:
                assert tab[arm]["REPEATED"]["base_med"] == 20.0, tab[arm]["REPEATED"]
                assert tab[arm]["REPEATED"]["PASS"] and not tab[arm]["SINGLE"]["PASS"]
    print(f"T3 scenario '{name}' -> {expect}: OK")


def t3_end_to_end() -> None:
    hi, lo = {"REPEATED": 20.0, "SINGLE": 4.0}, {"REPEATED": 4.0, "SINGLE": 4.0}
    confirmed = {}
    for a in TOY_ARMS:
        confirmed[f"base_{a}"] = hi          # beats controls on REPEATED only
        confirmed[f"shufq_{a}"] = lo
    _run_scenario("regime-confirmed", confirmed,
                  "REPETITION_REGIME_CONFIRMED__PIVOT_VIABLE_ONLY_THERE", lo)

    flat = {}
    for a in TOY_ARMS:
        flat[f"base_{a}"] = lo
        flat[f"shufq_{a}"] = lo
    _run_scenario("pivot-closes", flat,
                  "MULT_KCAP_ADDS_NOTHING_OVER_E2__ASSEMBLY_PIVOT_CLOSES", lo)

    undermined = {}
    hi_both = {"REPEATED": 20.0, "SINGLE": 20.0}
    for a in TOY_ARMS:
        undermined[f"base_{a}"] = hi_both       # beats controls on BOTH strata
        undermined[f"shufq_{a}"] = lo
    _run_scenario("e3-undermined", undermined,
                  "SINGLE_EXPOSURE_ROUTES_QUERY_DEPENDENTLY__"
                  "E3_UNDERMINED_NEMO_INSTALLATION_SPIKE", lo)


if __name__ == "__main__":
    t1_mult_math()
    t2_gated_additive_equivalence()
    t3_end_to_end()
    print("ALL TOY CHECKS GREEN")
