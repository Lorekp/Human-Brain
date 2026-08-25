#!/usr/bin/env python3
"""T-ROUTE E2 scorer (Change Note 3 C5/C6).

Reads ONLY logged sets: the hash-verified E2 log (this directory) plus the
hash-verified E1 log (../) for gold regions / matched flags / families /
sigma and the frozen arm-B dense baseline. No re-routing, no recomputation of
gold, no tuning hooks.

Gates (frozen): per config k in {8,16,32}, G-T1 conjuncts on the who_said
gating stratum on BOTH splits:
    base_med > 0 AND base_med >= 2*shuf_med AND base_med >= 1.5*rand_med
Config PASS iff both splits satisfy all three conjuncts. Mechanical verdict:
ANY config passing => ROUTING_RECOVERED_WITHOUT_HOMEOSTAT, else
RECOVERY_FAILS_SPARSITY_INSUFFICIENT. Descriptive (not gated): per-split
contrast of every config's base_med against the LOGGED arm-B medians, plus
gold-region-size headroom context.

Output: experiments/t_route/e2/e2_results.json (+ printed summary).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))  # noqa: E402 (unused import guard symmetry)

E2_DIR = Path(__file__).resolve().parent
E1_DIR = Path(__file__).resolve().parents[1]
E2_NPZ_PATH = E2_DIR / "e2_logged_winner_sets.npz"
E2_MANIFEST_PATH = E2_DIR / "e2_manifest.json"
E1_NPZ_PATH = E1_DIR / "logged_winner_sets.npz"
E1_MANIFEST_PATH = E1_DIR / "e1_manifest.json"
RESULTS_PATH = E2_DIR / "e2_results.json"

N_BITS = 2048
K_WINNERS = 128
FAMILIES = ("who_said", "address_lookup", "verse_continuation")
GATING_STRATUM = "who_said"
SPLITS = ("train", "val")
CONFIGS = (8, 16, 32)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "unknown"


def median(xs) -> float | None:
    xs = [float(x) for x in xs]
    return float(np.median(xs)) if xs else None


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
            raise SystemExit(f"STOP: array hash mismatch for {bad} in {npz_path.name}")
        return {name: np.array(z[name]) for name in z.files}


def winners_matrix(z: dict, rset: str, split: str, n_items: int) -> np.ndarray:
    sel = (z["split"] == split) & (z["routing_set"] == rset)
    idx = z["item_index"][sel]
    order = np.argsort(idx)
    assert len(idx) == n_items, f"missing rows for {rset}/{split}"
    assert np.array_equal(idx[order], np.arange(n_items)), f"row order broken for {rset}/{split}"
    return z["winner_sets"][sel][order].astype(np.int64)


def overlaps_own_gold(W: np.ndarray, gold_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(W.shape[0], dtype=np.int64)
    for i in range(W.shape[0]):
        out[i] = len(set(W[i].tolist()) & set(np.flatnonzero(gold_mask[i]).tolist()))
    return out


def main() -> None:
    log("loading + verifying E2 and E1 logs")
    e2_manifest = json.loads(E2_MANIFEST_PATH.read_text(encoding="utf-8"))
    z2 = verify_npz(E2_NPZ_PATH, E2_MANIFEST_PATH)
    z1 = verify_npz(E1_NPZ_PATH, E1_MANIFEST_PATH)

    families = {s: z1[f"families_{s}"] for s in SPLITS}
    matched = {s: z1[f"matched_{s}"].astype(bool) for s in SPLITS}
    gold_masks = {s: z1[f"gold_mask_{s}"].astype(bool) for s in SPLITS}
    gold_sizes = {s: z1[f"gold_sizes_{s}"].astype(np.int64) for s in SPLITS}

    def smask(split: str, family: str) -> np.ndarray:
        return families[split] == family

    # Overlap vectors: every E2 set vs own gold; armB from the verified E1 log.
    ov: dict[tuple[str, str], np.ndarray] = {}
    for k in CONFIGS:
        for label in (f"arm_k{k}", f"shufq_k{k}"):
            for s in SPLITS:
                Wm = winners_matrix(z2, label, s, len(families[s]))
                ov[(label, s)] = overlaps_own_gold(Wm, gold_masks[s])
    for s in SPLITS:
        Wm_rand = winners_matrix(z2, "randomized_W", s, len(families[s]))
        ov[("randomized_W", s)] = overlaps_own_gold(Wm_rand, gold_masks[s])
        Wm_b = winners_matrix(z1, "armB", s, len(families[s]))
        ov[("armB", s)] = overlaps_own_gold(Wm_b, gold_masks[s])

    # ---------------------------------------------------------- G-T1 per config
    log("evaluating G-T1 conjuncts per config from logged sets")
    gt1_table: dict[str, dict[str, dict]] = {}
    config_pass: dict[str, bool] = {}
    for k in CONFIGS:
        cfg = f"arm_k{k}"
        gt1_table[cfg] = {}
        conjuncts_all: list[bool] = []
        for fam in FAMILIES:
            gt1_table[cfg][fam] = {}
            for s in SPLITS:
                elig = smask(s, fam) & matched[s]
                row: dict = {"eligible_n": int(elig.sum())}
                row["base_med"] = median(ov[(f"arm_k{k}", s)][elig])
                row["shuf_med"] = median(ov[(f"shufq_k{k}", s)][elig])
                row["rand_med"] = median(ov[("randomized_W", s)][elig])
                row["armB_med"] = median(ov[("armB", s)][elig])
                bm, shm, rm = row["base_med"], row["shuf_med"], row["rand_med"]
                row["enrich_vs_shuf"] = (bm / shm) if (bm is not None and shm not in (None, 0.0)) else None
                row["enrich_vs_rand"] = (bm / rm) if (bm is not None and rm not in (None, 0.0)) else None
                if fam == GATING_STRATUM:
                    c0 = bm is not None and bm > 0
                    c1 = c0 and shm is not None and bm >= 2 * shm
                    c2 = c0 and rm is not None and bm >= 1.5 * rm
                    row["gate_conjuncts"] = {
                        "base_med>0": bool(c0),
                        "base>=2x_shuf": bool(c1),
                        "base>=1.5x_rand": bool(c2),
                    }
                    conjuncts_all.append(bool(c0 and c1 and c2))
                gt1_table[cfg][fam][s] = row
        config_pass[cfg] = bool(len(conjuncts_all) == 2 and all(conjuncts_all))
        log(f"  {cfg}: G-T1 {'PASS' if config_pass[cfg] else 'FAIL'} "
            f"(who_said conjuncts train,val={conjuncts_all})")
        for fam in FAMILIES:
            for s in SPLITS:
                r = gt1_table[cfg][fam][s]
                log(f"    [{fam}/{s}] base={r['base_med']} shuf={r['shuf_med']} rand={r['rand_med']} armB={r['armB_med']} (n={r['eligible_n']})")

    # ---------------------------------------------------------- arm-B contrast + headroom
    beats_armB: dict[str, dict[str, bool]] = {}
    for k in CONFIGS:
        beats_armB[f"arm_k{k}"] = {}
        for s in SPLITS:
            elig = smask(s, GATING_STRATUM) & matched[s]
            bm = median(ov[(f"arm_k{k}", s)][elig])
            ab = median(ov[("armB", s)][elig])
            beats_armB[f"arm_k{k}"][s] = bool(bm is not None and ab is not None and bm > ab)

    headroom = {}
    for fam in FAMILIES:
        headroom[fam] = {}
        for s in SPLITS:
            elig = smask(s, fam) & matched[s]
            sizes = gold_sizes[s][elig]
            headroom[fam][s] = {
                "n": int(elig.sum()),
                "median_gold_bits": median(sizes.tolist()),
                "p25_gold_bits": float(np.percentile(sizes, 25)) if sizes.size else None,
                "p75_gold_bits": float(np.percentile(sizes, 75)) if sizes.size else None,
            }

    any_pass = any(config_pass.values())
    verdict = "ROUTING_RECOVERED_WITHOUT_HOMEOSTAT" if any_pass else "RECOVERY_FAILS_SPARSITY_INSUFFICIENT"
    stop_rule = (
        "C5: ANY config passing G-T1 on both splits => ROUTING_RECOVERED_WITHOUT_HOMEOSTAT; "
        "no config => RECOVERY_FAILS_SPARSITY_INSUFFICIENT."
    )

    results = {
        "schema_version": "humanbrain/t-route-e2/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "amendment": "experiments/t_route/e2/CHANGE_NOTE_3.md",
        "reproduction_check": e2_manifest.get("reproduction_check", {}).get(
            "all_six_e1_routing_sets_bit_exact"
        ),
        "config_echo": e2_manifest["config"],
        "integrity_echo": {
            "freeze_candidate_raw_sha256": e2_manifest["integrity"]["freeze_candidate_raw_sha256"],
            "train_raw_sha256": e2_manifest["integrity"]["train_raw_sha256"],
            "val_raw_sha256": e2_manifest["integrity"]["val_raw_sha256"],
            "counts": {
                "train": e2_manifest["integrity"]["train_count"],
                "val": e2_manifest["integrity"]["val_count"],
            },
            "pairing_events_total": e2_manifest["pairing_events_total"],
            "unique_token_pairs": e2_manifest["unique_token_pairs"],
        },
        "gt1": {
            "gating_stratum": GATING_STRATUM,
            "rule": "per config on BOTH splits: base_med>0 AND base_med>=2*shuf_med AND base_med>=1.5*rand_med (who_said)",
            "table": gt1_table,
            "per_config_pass": config_pass,
            "pass_any_config": any_pass,
        },
        "armB_contrast": {
            "note": "descriptive, not gated; armB medians recomputed from the hash-verified E1 log",
            "beats_armB_by_split": beats_armB,
        },
        "headroom": headroom,
        "verdict": verdict,
        "stop_rule_applied": stop_rule,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {RESULTS_PATH.name}")
    print(json.dumps({
        "verdict": verdict,
        "per_config_gT1_pass": config_pass,
        "beats_armB": beats_armB,
    }, indent=1))


if __name__ == "__main__":
    main()
