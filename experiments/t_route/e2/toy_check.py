#!/usr/bin/env python3
"""Pre-spend mechanics validation for E2 (Change Note 3 C6 discipline).

Validates BEFORE any corpus encoding, mirroring the E1 D4 practice:
  T1  topk_row_sparsify: correct keeps, ties broken by ASCENDING column index,
      zeroing elsewhere, k >= fan_in is the exact identity, clamping;
  T2  identity transform at k=fan_in routes BIT-IDENTICALLY to the raw
      accumulator on a trained toy projection (synthetic dense delta);
  T3  full score_e2 pipeline on SYNTHETIC full-size logs with independently
      recomputed expectations -- passing-scenario verdict
      ROUTING_RECOVERED_WITHOUT_HOMEOSTAT with per-config pass/fail split;
  T4  failing-scenario verdict RECOVERY_FAILS_SPARSITY_INSUFFICIENT.

Writes only throwaway temp files; produces no experiment artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

E2_DIR = Path(__file__).resolve().parent
REPO = E2_DIR.parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.hebbian_projection import (  # noqa: E402
    HebbianConfig,
    HomeostaticProjection,
    SparseProjection,
)

spec_run = importlib.util.spec_from_file_location("run_e2_mod", E2_DIR / "run_e2.py")
run_e2 = importlib.util.module_from_spec(spec_run)
spec_run.loader.exec_module(run_e2)

spec_score = importlib.util.spec_from_file_location("score_e2_mod", E2_DIR / "score_e2.py")
score_e2 = importlib.util.module_from_spec(spec_score)
spec_score.loader.exec_module(score_e2)


def t1_topk_unit() -> None:
    support = np.array([[3, 1, 2, 0], [7, 5, 6, 4]], dtype=np.int64)
    values = np.array([[1.0, 3.0, 3.0, 2.0], [9.0, 8.0, 7.0, 6.0]])
    out = run_e2.topk_row_sparsify(support, values, 2)
    # Row 0: tie 3.0 between col1/col2 -> BOTH kept above 2.0/1.0 (tie order
    # irrelevant here because both survive); Row 1: cols {0,1}.
    assert out[0].tolist() == [0.0, 3.0, 3.0, 0.0], out[0]
    assert out[1].tolist() == [9.0, 8.0, 0.0, 0.0], out[1]
    # Tie-break visibility: keep only ONE of the two tied columns -> must be
    # the LOWER column index (col1), never col2.
    out1 = run_e2.topk_row_sparsify(support, values, 1)
    assert out1[0].tolist() == [0.0, 3.0, 0.0, 0.0], out1[0]
    # Identity + clamp.
    ident = run_e2.topk_row_sparsify(support, values, 4)
    assert np.array_equal(ident, values)
    assert np.array_equal(run_e2.topk_row_sparsify(support, values, 99), values)
    print("T1 topk_row_sparsify unit: OK")


def t2_identity_routing() -> None:
    cfg = HebbianConfig(n_bits=64, w_ones=4, fan_in=8, w_seed=3, k_winners=16)
    proj = SparseProjection(cfg)
    rng = np.random.default_rng(0)
    delta = np.zeros((64, 64))
    pairs = [(int(a), int(b)) for a, b in zip(rng.integers(0, 64, 200), rng.integers(0, 64, 200))]
    for a, b in pairs:
        delta[a, b] += 1.0
    proj.apply_dense_delta(delta)
    q = (np.random.default_rng(1).random(64) < 0.25).astype(np.float64)

    ident_vals = run_e2.topk_row_sparsify(proj.support(), proj.values(), cfg.fan_in)
    raw_vals = proj.values()
    assert np.array_equal(ident_vals, raw_vals), "k=fan_in must be the exact identity"
    mat = HomeostaticProjection(cfg, proj.support(), ident_vals)
    for _ in range(5):
        q = (np.random.default_rng(int(rng.integers(1 << 30))).random(64) < 0.25).astype(np.float64)
        assert np.array_equal(mat.route(q, 16), proj.route(q, 16)), "routing must be bit-identical"

    sparser = run_e2.topk_row_sparsify(proj.support(), proj.values(), 3)
    assert int((sparser > 0).sum()) == 64 * 3, "exactly k entries kept per row"
    print("T2 identity routing at k=fan_in + exact-k keeping: OK")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_log(npz_path: Path, manifest_path: Path, arrays: dict, extra: dict) -> None:
    if npz_path.exists():
        npz_path.unlink()
    np.savez_compressed(npz_path, **arrays)
    hashes = {
        name: {"sha256": _sha(np.ascontiguousarray(arrays[name]).tobytes()),
               "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in arrays
    }
    manifest_path.write_text(json.dumps({"array_sha256": hashes, **extra}), encoding="utf-8")


def _synth_scenario(tmp: Path, *, arm8_ov: int, other_ov: int, shuf_ov: int,
                    rand_ov: int, armb_ov: int, tag: str) -> dict:
    """Build synthetic E1+E2 logs whose overlaps are known by construction."""
    n_bits, K = 2048, 128
    sizes = {"train": 6, "val": 4}
    fams = {"train": ["who_said", "who_said", "address_lookup", "address_lookup",
                      "verse_continuation", "verse_continuation"],
            "val": ["who_said", "who_said", "address_lookup", "verse_continuation"]}
    rng = np.random.default_rng(42)
    z1: dict[str, np.ndarray] = {}
    gold: dict[str, np.ndarray] = {}
    for s, n in sizes.items():
        masks = np.zeros((n, n_bits), dtype=bool)
        for i in range(n):
            masks[i, rng.choice(n_bits, size=60, replace=False)] = True
        gold[s] = masks
        z1[f"gold_mask_{s}"] = masks
        z1[f"gold_sizes_{s}"] = masks.sum(axis=1).astype(np.int64)
        z1[f"matched_{s}"] = np.ones(n, dtype=bool)
        z1[f"families_{s}"] = np.array(fams[s], dtype="U24")
        z1[f"sigma_{s}"] = np.arange(n, dtype=np.int64)

    def winners_for(goldbits: set[int], ovr: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        non_gold = [b for b in range(n_bits) if b not in goldbits]
        w = list(r.choice(sorted(goldbits), size=ovr, replace=False))
        w += list(r.choice(non_gold, size=K - ovr, replace=False))
        return np.array(w, dtype=np.int16)

    rows2: list[tuple[str, str, int, np.ndarray]] = []
    rows1: list[tuple[str, str, int, np.ndarray]] = []
    for s, n in sizes.items():
        for i in range(n):
            gb = set(np.flatnonzero(gold[s][i]).tolist())
            rows1.append(("armB", s, i, winners_for(gb, armb_ov, 1000 + hash((tag, "b", s, i)) % 10**6)))
            rows2.append(("randomized_W", s, i, winners_for(gb, rand_ov, 2000 + hash((tag, "r", s, i)) % 10**6)))
            for k, ovv in ((8, arm8_ov), (16, other_ov), (32, other_ov)):
                rows2.append((f"arm_k{k}", s, i, winners_for(gb, ovv, 3000 + hash((tag, f"a{k}", s, i)) % 10**6)))
                rows2.append((f"shufq_k{k}", s, i, winners_for(gb, shuf_ov, 4000 + hash((tag, f"s{k}", s, i)) % 10**6)))

    def to_arrays(rows):
        return {
            "winner_sets": np.stack([r[3] for r in rows]),
            "routing_set": np.array([r[0] for r in rows], dtype="U24"),
            "split": np.array([r[1] for r in rows], dtype="U5"),
            "item_index": np.array([r[2] for r in rows], dtype=np.int64),
        }

    d = tmp / tag
    d.mkdir(parents=True, exist_ok=True)
    a1 = {**z1, **to_arrays(rows1)}
    a2 = to_arrays(rows2)
    _write_log(d / "e1.npz", d / "e1_manifest.json", a1, {})
    _write_log(d / "e2.npz", d / "e2_manifest.json", a2, {
        "integrity": {"freeze_candidate_raw_sha256": "x", "train_raw_sha256": "x",
                      "val_raw_sha256": "x", "train_count": 6, "val_count": 4},
        "config": {"k_sweep": [8, 16, 32]},
        "pairing_events_total": 0, "unique_token_pairs": 0,
        "reproduction_check": {"all_six_e1_routing_sets_bit_exact": True},
    })

    # Independent expectation (plain python, no shared code with the scorer).
    exp_cfg_pass = {}
    for k, ovv in ((8, arm8_ov), (16, other_ov), (32, other_ov)):
        conj = []
        for s in sizes:
            bm, shm, rm = float(ovv), float(shuf_ov), float(rand_ov)
            conj.append(bm > 0 and bm >= 2 * shm and bm >= 1.5 * rm)
        exp_cfg_pass[f"arm_k{k}"] = len(conj) == 2 and all(conj)
    exp_verdict = ("ROUTING_RECOVERED_WITHOUT_HOMEOSTAT" if any(exp_cfg_pass.values())
                   else "RECOVERY_FAILS_SPARSITY_INSUFFICIENT")

    # Point the scorer at the synthetic logs and run it end-to-end.
    score_e2.E2_NPZ_PATH = d / "e2.npz"
    score_e2.E2_MANIFEST_PATH = d / "e2_manifest.json"
    score_e2.E1_NPZ_PATH = d / "e1.npz"
    score_e2.E1_MANIFEST_PATH = d / "e1_manifest.json"
    score_e2.RESULTS_PATH = d / "e2_results.json"
    score_e2.main()
    res = json.loads((d / "e2_results.json").read_text(encoding="utf-8"))

    assert res["verdict"] == exp_verdict, (res["verdict"], exp_verdict)
    for cfg_name, ok in exp_cfg_pass.items():
        got = res["gt1"]["per_config_pass"][cfg_name]
        assert got == ok, (cfg_name, got, ok)
        # Overlap medians must equal the constructed overlap counts exactly.
        for s in sizes:
            row = res["gt1"]["table"][cfg_name]["who_said"][s]
            assert row["base_med"] == float({"arm_k8": arm8_ov}.get(cfg_name, other_ov)), row
            assert row["shuf_med"] == float(shuf_ov), row
            assert row["rand_med"] == float(rand_ov), row
            assert row["armB_med"] == float(armb_ov), row
            assert row["eligible_n"] == sum(1 for f in fams[s] if f == "who_said"), row
    print(f"{tag}: verdict={exp_verdict} per_config_pass={exp_cfg_pass} -> OK")


if __name__ == "__main__":
    t1_topk_unit()
    t2_identity_routing()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # T3: k=8 passes all conjuncts on both splits; k=16/32 deliberately fail
        # their 2x_shuf conjunct -> mixed outcome, any-config disjunction fires.
        _synth_scenario(tmp, arm8_ov=50, other_ov=3, shuf_ov=2, rand_ov=10, armb_ov=45, tag="pass_mixed")
        # T4: nothing passes (bases below 1.5x rand).
        _synth_scenario(tmp, arm8_ov=5, other_ov=4, shuf_ov=3, rand_ov=10, armb_ov=45, tag="fail_all")
    print("ALL E2 MECHANICS CHECKS OK (pre-spend)")
