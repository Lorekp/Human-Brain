#!/usr/bin/env python3
"""T-ROUTE E4 scorer (Change Note 5 §5-§7).

Reads ONLY the hash-verified E4 log + manifest. No re-routing, no recomputation of
gold, no tuning hooks.

Per arm x stratum on eligible (matched) items:
    base_med > 0 AND base_med >= 2*shuf_med AND base_med >= 1.5*rand_med   (n >= 20)

Verdict matrix (frozen):
    R = any arm PASSes REPEATED; S = any arm PASSes SINGLE
    no R                  -> MULT_KCAP_ADDS_NOTHING_OVER_E2__ASSEMBLY_PIVOT_CLOSES
    R and not S           -> REPETITION_REGIME_CONFIRMED__PIVOT_VIABLE_ONLY_THERE
    S (with or without R) -> SINGLE_EXPOSURE_ROUTES_QUERY_DEPENDENTLY__E3_UNDERMINED_NEMO_INSTALLATION_SPIKE

Descriptive (not gated): enrichment ratios, gold headroom, cross-question winner
Jaccard within strata, MID-band table, per-arm max |weight| context from manifest.

Output: experiments/t_route/e4_repetition/e4_results.json (+ printed summary).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

E4_DIR = Path(__file__).resolve().parent
NPZ_PATH = E4_DIR / "e4_logged_winner_sets.npz"
MANIFEST_PATH = E4_DIR / "e4_manifest.json"
RESULTS_PATH = E4_DIR / "e4_results.json"

N_BITS = 2048
K_WINNERS = 128
STRATA = ("REPEATED", "SINGLE", "MID")
GATE_STRATA = ("REPEATED", "SINGLE")
ARMS = ("mult_b0.05", "mult_b0.25", "mult_b0.5", "add_gated", "add_free")
MIN_GATE_N = 20
JACCARD_SAMPLE = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(E4_DIR.parents[2]), text=True,
        ).strip()
    except Exception:
        return "unknown"


def verify_npz(npz_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as z:
        bad = []
        for name, meta in manifest["array_sha256"].items():
            arr = z[name]
            ok = (
                str(arr.dtype) == meta["dtype"]
                and list(arr.shape) == meta["shape"]
                and sha256_bytes(np.ascontiguousarray(arr).tobytes()) == meta["sha256"]
            )
            if not ok:
                bad.append(name)
        if bad:
            raise SystemExit(f"STOP: array hash mismatch for {bad}")
        return {name: np.array(z[name]) for name in z.files}


def routed_matrix(z: dict, rset: str, n_items: int) -> np.ndarray:
    sel = z["routing_set"] == rset
    idx = z["item_index"][sel]
    order = np.argsort(idx)
    assert len(idx) == n_items, f"missing rows for {rset}"
    assert np.array_equal(idx[order], np.arange(n_items)), f"row order broken for {rset}"
    return z["winner_sets"][sel][order].astype(np.int64)


def overlaps_own_gold(W: np.ndarray, gold_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(W.shape[0], dtype=np.int64)
    for i in range(W.shape[0]):
        out[i] = len(set(W[i].tolist()) & set(np.flatnonzero(gold_mask[i]).tolist()))
    return out


def median(xs) -> float | None:
    xs = [float(x) for x in xs]
    return float(np.median(xs)) if xs else None


def cross_question_jaccard(W: np.ndarray, idx: np.ndarray) -> float | None:
    """Mean pairwise Jaccard of winner sets over a capped sample (descriptive)."""
    idx = idx[:JACCARD_SAMPLE]
    if len(idx) < 2:
        return None
    sets = [frozenset(W[i].tolist()) for i in idx]
    num = den = 0
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            num += len(sets[a] & sets[b])
            den += len(sets[a] | sets[b])
    return (num / den) if den else None


def main() -> None:
    log("loading + verifying E4 log")
    e4_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    z = verify_npz(NPZ_PATH, MANIFEST_PATH)
    n_items = len(z["stratum"])
    strata = z["stratum"]
    matched = z["matched"].astype(bool)
    gold_masks = z["gold_mask"].astype(bool)
    gold_sizes = z["gold_sizes"].astype(np.int64)
    log(f"items={n_items}; strata census="
        f"{ {s: int((strata == s).sum()) for s in STRATA} }; "
        f"matched={int(matched.sum())}")

    ov: dict[tuple[str, str], np.ndarray] = {}
    for arm in ARMS:
        ov[(f"base_{arm}", "ALL")] = overlaps_own_gold(
            routed_matrix(z, f"base_{arm}", n_items), gold_masks)
        ov[(f"shufq_{arm}", "ALL")] = overlaps_own_gold(
            routed_matrix(z, f"shufq_{arm}", n_items), gold_masks)
    ov[("randomized_W", "ALL")] = overlaps_own_gold(
        routed_matrix(z, "randomized_W", n_items), gold_masks)

    gate_table: dict[str, dict[str, dict]] = {}
    pass_repeated, pass_single = {}, {}
    jaccard: dict[str, dict[str, float | None]] = {}
    for arm in ARMS:
        gate_table[arm], jaccard[arm] = {}, {}
        for strat in STRATA:
            elig = (strata == strat) & matched
            row: dict = {"eligible_n": int(elig.sum())}
            bm = median(ov[(f"base_{arm}", "ALL")][elig])
            shm = median(ov[(f"shufq_{arm}", "ALL")][elig])
            rm = median(ov[("randomized_W", "ALL")][elig])
            row.update(base_med=bm, shuf_med=shm, rand_med=rm)
            row["enrich_vs_shuf"] = (bm / shm) if (bm is not None and shm not in (None, 0.0)) else None
            row["enrich_vs_rand"] = (bm / rm) if (bm is not None and rm not in (None, 0.0)) else None
            c0 = bm is not None and bm > 0
            c1 = c0 and shm is not None and bm >= 2 * shm
            c2 = c0 and rm is not None and bm >= 1.5 * rm
            row["gate_conjuncts"] = {
                "base_med>0": bool(c0),
                "base>=2x_shuf": bool(c1),
                "base>=1.5x_rand": bool(c2),
            }
            row["gate_eligible"] = bool(elig.sum() >= MIN_GATE_N)
            row["PASS"] = bool(row["gate_eligible"] and c0 and c1 and c2)
            sizes = gold_sizes[elig]
            row["median_gold_bits"] = median(sizes.tolist()) if sizes.size else None
            if strat in GATE_STRATA:
                jaccard[arm][strat] = cross_question_jaccard(
                    routed_matrix(z, f"base_{arm}", n_items), np.flatnonzero(elig))
            gate_table[arm][strat] = row
            log(f"  [{arm}/{strat}] n={row['eligible_n']} base={bm} shuf={shm} "
                f"rand={rm} PASS={row['PASS']}")

    pass_repeated = {a: gate_table[a]["REPEATED"]["PASS"] for a in ARMS}
    pass_single = {a: gate_table[a]["SINGLE"]["PASS"] for a in ARMS}
    R, S = any(pass_repeated.values()), any(pass_single.values())
    if not R:
        verdict = ("SINGLE_EXPOSURE_ROUTES_QUERY_DEPENDENTLY__"
                   "E3_UNDERMINED_NEMO_INSTALLATION_SPIKE" if S else
                   "MULT_KCAP_ADDS_NOTHING_OVER_E2__ASSEMBLY_PIVOT_CLOSES")
    else:
        verdict = ("REPETITION_REGIME_CONFIRMED__PIVOT_VIABLE_ONLY_THERE"
                   if not S else
                   "SINGLE_EXPOSURE_ROUTES_QUERY_DEPENDENTLY__"
                   "E3_UNDERMINED_NEMO_INSTALLATION_SPIKE")

    # Descriptive MID table + headroom already inside gate_table; assemble results.
    results = {
        "schema_version": "humanbrain/t-route-e4/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "change_note": "experiments/t_route/e4_repetition/CHANGE_NOTE_5.md",
        "config_echo": e4_manifest["config"],
        "integrity_echo": e4_manifest["integrity"],
        "max_synapses_at_cap_by_arm": e4_manifest.get("max_synapses_at_cap_by_arm"),
        "gates": {
            "rule": "per arm x stratum on eligible items (matched): "
                    "base_med>0 AND base_med>=2*shuf_med AND base_med>=1.5*rand_med; "
                    "stratum gate-eligible iff eligible_n>=20",
            "table": gate_table,
            "per_arm_pass_REPEATED": pass_repeated,
            "per_arm_pass_SINGLE": pass_single,
        },
        "cross_question_jaccard_descriptive": jaccard,
        "verdict": verdict,
        "verdict_matrix": {"any_REPEATED_pass": bool(R), "any_SINGLE_pass": bool(S)},
        "verdict_note": (
            "Rows of CN5 §7; branch mapping: repeated-fail=>pivot closes; "
            "repeated-pass+single-fail=>regime confirmed; single-pass=>E3 undermined."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {RESULTS_PATH.name}")
    print(json.dumps({
        "verdict": verdict,
        "per_arm_pass_REPEATED": pass_repeated,
        "per_arm_pass_SINGLE": pass_single,
    }, indent=1))


if __name__ == "__main__":
    main()
