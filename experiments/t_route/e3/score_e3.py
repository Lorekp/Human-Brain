#!/usr/bin/env python3
"""T-ROUTE E3 scorer (Change Note 4 C5/C6).

Reads ONLY logged sets: the hash-verified E3 log (this directory), plus the
hash-verified E2 log (../e2/) for randomized_W and the arm_k16 baseline, plus
the hash-verified E1 log (../) for gold regions / matched flags / families /
sigma and the arm-B dense baseline. No re-routing, no recomputation of gold,
no tuning hooks.

Gates (frozen pre-spend): on the who_said gating stratum, per split in
{TRAIN-self, VAL}, with medians over eligible items
    base_med  = median(arm_q16)      shuf_med   = median(shufq_q16)
    rand_med  = median(randomized_W) rsp_med    = median(rs16)
    rshuf_med = median(shufq_rs16)
G-T1' conjuncts (both splits required): base_med > 0 AND base_med >= 2*shuf_med
AND base_med >= 1.5*rand_med. Specificity ratios spec_q = base_med/shuf_med,
spec_rs = rsp_med/rshuf_med (0 denominator + positive numerator => +inf).

Mechanical three-way verdict:
    QUOTE_SPAN_ROUTING_RECOVERED       iff G-T1' both splits AND spec_q > spec_rs
                                       both splits;
    GATING_ARTIFACT_NO_QUOTE_SPECIFICITY iff G-T1' both splits but not
                                       spec_q > spec_rs on both;
    QUOTE_SPAN_GATE_FAILS              otherwise (program-grade conclusion per
                                       docs/research/t-route-e1e2-verdict.md).

Output: experiments/t_route/e3/e3_results.json (+ printed summary).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))  # noqa: E402 (symmetry with harness)

E3_DIR = Path(__file__).resolve().parent
E2_DIR = Path(__file__).resolve().parents[1] / "e2"
E1_DIR = Path(__file__).resolve().parents[1]
E3_NPZ_PATH = E3_DIR / "e3_logged_winner_sets.npz"
E3_MANIFEST_PATH = E3_DIR / "e3_manifest.json"
E2_NPZ_PATH = E2_DIR / "e2_logged_winner_sets.npz"
E2_MANIFEST_PATH = E2_DIR / "e2_manifest.json"
E1_NPZ_PATH = E1_DIR / "logged_winner_sets.npz"
E1_MANIFEST_PATH = E1_DIR / "e1_manifest.json"
RESULTS_PATH = E3_DIR / "e3_results.json"

K_WINNERS = 128
FAMILIES = ("who_said", "address_lookup", "verse_continuation")
GATING_STRATUM = "who_said"
SPLITS = ("train", "val")


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


def specificity(num: float | None, den: float | None) -> float | None:
    """Ratio with frozen inf-guard: den==0 and num>0 => +inf; num==0 => ratio 0."""
    if num is None or den is None:
        return None
    if den == 0.0:
        return math.inf if num > 0.0 else 0.0
    return float(num / den)


def evaluate_verdict(
    gt1_both_splits: bool,
    spec_q_train: float | None,
    spec_q_val: float | None,
    spec_rs_train: float | None,
    spec_rs_val: float | None,
) -> tuple[str, dict]:
    """Frozen C5 three-way verdict (pure function; unit-tested in toy_check.py)."""
    spec_gt = (
        spec_q_train is not None
        and spec_rs_train is not None
        and spec_q_train > spec_rs_train
        and spec_q_val is not None
        and spec_rs_val is not None
        and spec_q_val > spec_rs_val
    )
    detail = {
        "gt1_both_splits": bool(gt1_both_splits),
        "spec_q_gt_spec_rs_both_splits": bool(spec_gt),
        "spec_q": {"train": spec_q_train, "val": spec_q_val},
        "spec_rs": {"train": spec_rs_train, "val": spec_rs_val},
    }
    if gt1_both_splits and spec_gt:
        return "QUOTE_SPAN_ROUTING_RECOVERED", detail
    if gt1_both_splits:
        return "GATING_ARTIFACT_NO_QUOTE_SPECIFICITY", detail
    return "QUOTE_SPAN_GATE_FAILS", detail


def main() -> None:
    log("loading + verifying E3, E2 and E1 logs")
    e3_manifest = json.loads(E3_MANIFEST_PATH.read_text(encoding="utf-8"))
    z3 = verify_npz(E3_NPZ_PATH, E3_MANIFEST_PATH)
    z2 = verify_npz(E2_NPZ_PATH, E2_MANIFEST_PATH)
    z1 = verify_npz(E1_NPZ_PATH, E1_MANIFEST_PATH)

    families = {s: z1[f"families_{s}"] for s in SPLITS}
    matched = {s: z1[f"matched_{s}"].astype(bool) for s in SPLITS}
    gold_masks = {s: z1[f"gold_mask_{s}"].astype(bool) for s in SPLITS}
    gold_sizes = {s: z1[f"gold_sizes_{s}"].astype(np.int64) for s in SPLITS}

    def smask(split: str, family: str) -> np.ndarray:
        return families[split] == family

    # Overlap vectors vs own gold: E3 sets + baselines from verified logs.
    sources = {
        "arm_q16": (z3, "arm_q16"),
        "shufq_q16": (z3, "shufq_q16"),
        "rs16": (z3, "rs16"),
        "shufq_rs16": (z3, "shufq_rs16"),
        "randomized_W": (z2, "randomized_W"),
        "arm_k16_e2": (z2, "arm_k16"),
        "armB_e1": (z1, "armB"),
    }
    ov: dict[tuple[str, str], np.ndarray] = {}
    for label, (z, rset) in sources.items():
        for s in SPLITS:
            Wm = winners_matrix(z, rset, s, len(families[s]))
            ov[(label, s)] = overlaps_own_gold(Wm, gold_masks[s])

    # ---------------------------------------------------------- medians per stratum/split
    log("computing verbatim medians per stratum/split")
    medians_table: dict[str, dict[str, dict]] = {}
    for fam in FAMILIES:
        medians_table[fam] = {}
        for s in SPLITS:
            elig = smask(s, fam) & matched[s]
            row = {"eligible_n": int(elig.sum())}
            for label in sources:
                row[label] = median(ov[(label, s)][elig])
            bm, shm, rm = row["arm_q16"], row["shufq_q16"], row["randomized_W"]
            row["enrich_vs_shuf"] = specificity(bm, shm)
            row["enrich_vs_rand"] = specificity(bm, rm)
            if fam == GATING_STRATUM:
                c0 = bm is not None and bm > 0
                c1 = c0 and shm is not None and bm >= 2 * shm
                c2 = c0 and rm is not None and bm >= 1.5 * rm
                row["gate_conjuncts"] = {
                    "base_med>0": bool(c0),
                    "base>=2x_shuf": bool(c1),
                    "base>=1.5x_rand": bool(c2),
                }
            medians_table[fam][s] = row
            pretty = {k: v for k, v in row.items() if k != "gate_conjuncts"}
            log(f"  [{fam}/{s}] {pretty}")

    # ---------------------------------------------------------- G-T1' + verdict
    gt1_per_split: dict[str, bool] = {}
    spec_q: dict[str, float | None] = {}
    spec_rs: dict[str, float | None] = {}
    for s in SPLITS:
        r = medians_table[GATING_STRATUM][s]
        conj = r.get("gate_conjuncts")
        gt1_per_split[s] = bool(conj and all(conj.values()))
        spec_q[s] = specificity(r["arm_q16"], r["shufq_q16"])
        spec_rs[s] = specificity(r["rs16"], r["shufq_rs16"])
    verdict, verdict_detail = evaluate_verdict(
        all(gt1_per_split.values()),
        spec_q["train"], spec_q["val"],
        spec_rs["train"], spec_rs["val"],
    )
    log(f"G-T1' per split: {gt1_per_split}; spec_q={spec_q}; spec_rs={spec_rs}")
    log(f"VERDICT: {verdict}")

    # ---------------------------------------------------------- diagnostics (non-gated)
    diag: dict[str, object] = {}
    for s in SPLITS:
        n_items = len(families[s])
        distinct: dict[str, int] = {}
        for label in ("arm_q16", "rs16", "arm_k16_e2"):
            z, rset = sources[label]
            Wm = winners_matrix(z, rset, s, n_items)
            distinct[label] = int(len(set(Wm.ravel().tolist())))
        # cross-question winner Jaccard vs each matrix's OWN shuffled-question rows
        jacc_pairs = (
            ("arm_q16", "shufq_q16"),
            ("rs16", "shufq_rs16"),
        )
        jacc: dict[str, list[float]] = {b: [] for b, _ in jacc_pairs}
        for base_label, shuf_label in jacc_pairs:
            zb, rb = sources[base_label]
            zs, rs_ = sources[shuf_label]
            Wb = winners_matrix(zb, rb, s, n_items)
            Ws = winners_matrix(zs, rs_, s, n_items)
            for i in range(n_items):
                a, b = set(Wb[i].tolist()), set(Ws[i].tolist())
                u = len(a | b)
                jacc[base_label].append((len(a & b) / u) if u else 0.0)
        # E2 reference: arm_k16 vs its own shuffled-question control (both in E2 log)
        Wa = winners_matrix(z2, "arm_k16", s, n_items)
        Wb2 = winners_matrix(z2, "shufq_k16", s, n_items)
        jacc_armk16 = []
        for i in range(n_items):
            a, b = set(Wa[i].tolist()), set(Wb2[i].tolist())
            u = len(a | b)
            jacc_armk16.append((len(a & b) / u) if u else 0.0)
        entry = {
            "distinct_bits": distinct,
            "median_cross_question_jaccard": {
                **{k: float(np.median(v)) for k, v in jacc.items()},
                "arm_k16_e2": float(np.median(jacc_armk16)),
            },
        }
        diag[s] = entry
        log(f"  diagnostics[{s}]: {entry}")

    # ---------------------------------------------------------- headroom
    headroom = {}
    for fam in FAMILIES:
        headroom[fam] = {}
        for s in SPLITS:
            elig = smask(s, fam) & matched[s]
            sizes = gold_sizes[s][elig]
            headroom[fam][s] = {
                "n": int(elig.sum()),
                "median_gold_bits": median(sizes.tolist()) if sizes.size else None,
                "p25_gold_bits": float(np.percentile(sizes, 25)) if sizes.size else None,
                "p75_gold_bits": float(np.percentile(sizes, 75)) if sizes.size else None,
            }

    stop_rule = (
        "C5: QUOTE_SPAN_ROUTING_RECOVERED iff G-T1' holds on BOTH splits AND spec_q > spec_rs "
        "on BOTH splits; G-T1' without that => GATING_ARTIFACT_NO_QUOTE_SPECIFICITY; otherwise "
        "QUOTE_SPAN_GATE_FAILS (program-grade conclusion per t-route-e1e2-verdict.md)."
    )

    results = {
        "schema_version": "humanbrain/t-route-e3/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "amendment": "experiments/t_route/e3/CHANGE_NOTE_4.md",
        "reproduction_check": {
            "all_six_e1_routing_sets_bit_exact": e3_manifest["reproduction_check"][
                "all_six_e1_routing_sets_bit_exact"
            ],
            "all_seven_e2_logged_sets_bit_exact": e3_manifest["reproduction_check"][
                "all_seven_e2_logged_sets_bit_exact"
            ],
            "ungated_counter_equals_e1_e2_aggregate": e3_manifest["reproduction_check"][
                "ungated_counter_equals_e1_e2_aggregate"
            ],
        },
        "config_echo": e3_manifest["config"],
        "gating_census_echo": e3_manifest["gating_census"],
        "integrity_echo": {
            "freeze_candidate_raw_sha256": e3_manifest["integrity"]["freeze_candidate_raw_sha256"],
            "train_raw_sha256": e3_manifest["integrity"]["train_raw_sha256"],
            "val_raw_sha256": e3_manifest["integrity"]["val_raw_sha256"],
            "counts": {
                "train": e3_manifest["integrity"]["train_count"],
                "val": e3_manifest["integrity"]["val_count"],
            },
        },
        "medians": medians_table,
        "gt1_prime": {
            "gating_stratum": GATING_STRATUM,
            "rule": "per split: base_med>0 AND base_med>=2*shuf_med AND base_med>=1.5*rand_med (who_said); pass requires BOTH splits",
            "per_split_pass": gt1_per_split,
        },
        "specificity": {
            "definition": "spec_q = base_med/shuf_med; spec_rs = rsp_med/rshuf_med; den==0 & num>0 => +inf",
            **verdict_detail,
        },
        "diagnostics_non_gated": diag,
        "headroom": headroom,
        "verdict": verdict,
        "stop_rule_applied": stop_rule,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {RESULTS_PATH.name}")
    print(json.dumps({
        "verdict": verdict,
        "gt1_prime_per_split": gt1_per_split,
        "spec_q": spec_q,
        "spec_rs": spec_rs,
        "who_said_medians": {
            s: {k: medians_table[GATING_STRATUM][s][k]
                for k in ("eligible_n", "arm_q16", "shufq_q16", "randomized_W", "rs16", "shufq_rs16", "arm_k16_e2", "armB_e1")}
            for s in SPLITS
        },
    }, indent=1))


if __name__ == "__main__":
    main()
