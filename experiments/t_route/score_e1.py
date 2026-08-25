#!/usr/bin/env python3
"""T-ROUTE E1 scorer + mechanical D5 verdict (prereg §5-§7).

Consumes ONLY the logged winner sets (logged_winner_sets.npz + e1_manifest.json
produced by run_e1.py) for every gate evaluation — prereg §6 step 1: "evaluate
G-T3 then G-T1 from the logged sets only".

Gate order and stop-on-failure (prereg §6):
  G-T3 (anti-leak) evaluated first; then G-T1 (routing enrichment);
  T1 FAIL  -> verdict NEGATIVE_CORE_WOUND; the G-T2 pipeline leg is NOT spent;
  T1 PASS ^ T3 FAIL -> pipeline run ONCE as DIAGNOSTIC ONLY ->
                       NEGATIVE_WITH_LOCALIZATION_V3_LEAK;
  T1 PASS ^ T3 PASS -> G-T2 pipeline leg runs ONCE; verdict per §7
                       (>= 3/639 => PROCEED_TO_INSTALL, else
                       NEGATIVE_WITH_LOCALIZATION_V3).

The pipeline re-derives token streams deterministically (frozen fit recipe),
takes hop-1 winner sets FROM THE LOGGED NPZ, and feeds rankings to the
UNCHANGED experiments/r0c_probes/probes.py SpanScorer (639 VAL answer-word
positions enforced on the official leg).

Output: experiments/t_route/e1_results.json (+ printed verdict summary).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.sdr_union import SDRProjector  # noqa: E402
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig  # noqa: E402

OUT_DIR = Path(os.environ.get("T_ROUTE_OUT_DIR", str(Path(__file__).resolve().parent)))
NPZ_PATH = OUT_DIR / "logged_winner_sets.npz"
MANIFEST_PATH = OUT_DIR / "e1_manifest.json"
RESULTS_PATH = OUT_DIR / "e1_results.json"

N_BITS = 2048
K_WINNERS = 128
H2 = 8  # prereg §5 G-T2 hop-2 radius
FAMILIES = ("who_said", "address_lookup", "verse_continuation")
GATING_STRATUM = "who_said"
SPAN_PREDICTIONS = 639

# Verbatim comparator anchors (UNCHANGED; experiments/qa_baselines_bible_v1/*.json)
UNIGRAM_SPAN_ACC = 0.04538341158059468  # 29/639
MAJORITY_SPAN_ACC = 0.0406885758998435  # 26/639
RANDOM_SPAN_ACC = 0.001564945226917058  # 1/639


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


def load_probes_module():
    spec = importlib.util.spec_from_file_location(
        "r0c_probes", REPO / "experiments" / "r0c_probes" / "probes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def median(xs) -> float | None:
    xs = [float(x) for x in xs]
    return float(np.median(xs)) if xs else None


def verify_npz(manifest: dict) -> dict:
    """Load + hash-verify every array; returns plain dict (archive closed)."""
    with np.load(NPZ_PATH, allow_pickle=False) as z:
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


def winners_matrix(z, rset: str, split: str, n_items: int) -> np.ndarray:
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


def overlaps_vs_targets(W: np.ndarray, gold_mask: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.zeros(W.shape[0], dtype=np.int64)
    for i, t in enumerate(targets):
        gbits = set(np.flatnonzero(gold_mask[t]).tolist())
        out[i] = len(set(W[i].tolist()) & gbits)
    return out


def main() -> None:
    log("loading + verifying logged winner sets")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    z = verify_npz(manifest)

    splits = ["train", "val"]
    routing_sets = ("armA", "armB", "shuffled_question", "randomized_W", "wrong_passage", "shuffled_W")

    families = {s: z[f"families_{s}"] for s in splits}
    matched = {s: z[f"matched_{s}"].astype(bool) for s in splits}
    gold_masks = {s: z[f"gold_mask_{s}"].astype(bool) for s in splits}
    gold_sizes = {s: z[f"gold_sizes_{s}"].astype(np.int64) for s in splits}
    sigma = {s: z[f"sigma_{s}"].astype(np.int64) for s in splits}

    Wm = {
        (rset, s): winners_matrix(z, rset, s, len(families[s]))
        for rset in routing_sets
        for s in splits
    }

    ov = {
        (rset, s): overlaps_own_gold(Wm[(rset, s)], gold_masks[s])
        for rset in ("armA", "armB", "shuffled_question", "randomized_W", "shuffled_W")
        for s in splits
    }
    ov_wp = {s: overlaps_vs_targets(Wm[("wrong_passage", s)], gold_masks[s], sigma[s]) for s in splits}

    def smask(split: str, family: str) -> np.ndarray:
        return families[split] == family

    # ---------------------------------------------------------- G-T3 (first)
    log("evaluating G-T3 (anti-leak) from logged sets")
    wp_obs, wp_exp, wp_rows = [], [], []
    for s in splits:
        m = smask(s, GATING_STRATUM) & matched[s][sigma[s]]
        for i in np.flatnonzero(m):
            obs = int(ov_wp[s][i])
            exp = K_WINNERS * float(gold_sizes[s][sigma[s][i]]) / N_BITS
            wp_obs.append(obs)
            wp_exp.append(exp)
            wp_rows.append({
                "split": s, "item_index": int(i), "observed": obs,
                "expected_chance": exp, "gold_target_size": int(gold_sizes[s][sigma[s][i]]),
            })
    gt3a_median_obs = median(wp_obs)
    gt3a_median_exp = median(wp_exp)
    gt3a_pass = (
        gt3a_median_obs is not None
        and gt3a_median_exp is not None
        and gt3a_median_obs <= gt3a_median_exp
    )

    rand_pool, shufw_pool = [], []
    for s in splits:
        m = smask(s, GATING_STRATUM) & matched[s]
        rand_pool.extend(ov[("randomized_W", s)][m].tolist())
        shufw_pool.extend(ov[("shuffled_W", s)][m].tolist())
    gt3b_rand_med = median(rand_pool)
    gt3b_shufw_med = median(shufw_pool)
    band_lo = None if gt3b_rand_med is None else gt3b_rand_med / 1.5
    band_hi = None if gt3b_rand_med is None else gt3b_rand_med * 1.5
    gt3b_pass = (
        gt3b_rand_med is not None
        and gt3b_shufw_med is not None
        and band_lo <= gt3b_shufw_med <= band_hi
    )
    gt3_pass = bool(gt3a_pass and gt3b_pass)
    log(f"  G-T3a wrong-passage: median_obs={gt3a_median_obs} vs median_chance={gt3a_median_exp} "
        f"-> {'PASS' if gt3a_pass else 'FAIL'} (n={len(wp_obs)})")
    log(f"  G-T3b shuffled-W: [{band_lo}, {band_hi}] contains shufw_med={gt3b_shufw_med} "
        f"-> {'PASS' if gt3b_pass else 'FAIL'} (rand_med={gt3b_rand_med})")

    # ---------------------------------------------------------- G-T1 (second)
    log("evaluating G-T1 (routing enrichment) from logged sets")
    gt1_table: dict[str, dict[str, dict]] = {}
    gate_conjuncts = []
    for fam in FAMILIES:
        gt1_table[fam] = {}
        for s in splits:
            elig = smask(s, fam) & matched[s]
            row: dict = {"eligible_n": int(elig.sum())}
            for label, rset in (("base_med", "armA"), ("shuf_med", "shuffled_question"), ("rand_med", "randomized_W")):
                row[label] = median(ov[(rset, s)][elig])
            bm, shm, rm = row["base_med"], row["shuf_med"], row["rand_med"]
            row["enrich_vs_shuf"] = (bm / shm) if (bm is not None and shm not in (None, 0.0)) else None
            row["enrich_vs_rand"] = (bm / rm) if (bm is not None and rm not in (None, 0.0)) else None
            if fam == GATING_STRATUM:
                c0 = bm is not None and bm > 0
                c1 = c0 and shm is not None and bm >= 2 * shm
                c2 = c0 and rm is not None and bm >= 1.5 * rm
                row["gate_conjuncts"] = {"base_med>0": bool(c0), "base>=2x_shuf": bool(c1), "base>=1.5x_rand": bool(c2)}
                gate_conjuncts.append(bool(c0 and c1 and c2))
            gt1_table[fam][s] = row
    gt1_pass = bool(gate_conjuncts) and all(gate_conjuncts)
    log(f"  G-T1 gating stratum={GATING_STRATUM}: conjuncts(train,val)={gate_conjuncts} -> {'PASS' if gt1_pass else 'FAIL'}")
    for fam in FAMILIES:
        for s in splits:
            r = gt1_table[fam][s]
            log(f"    [{fam}/{s}] base={r['base_med']} shuf={r['shuf_med']} rand={r['rand_med']} (n={r['eligible_n']})")

    # ---------------------------------------------------------- stop rules / G-T2
    t2_block: dict | None = None
    pipeline_spent = False
    if not gt1_pass:
        verdict = "NEGATIVE_CORE_WOUND"
        stop_rule = "§6.2: G-T1 FAIL => NEGATIVE_CORE_WOUND; pipeline leg NOT spent."
    elif not gt3_pass:
        pipeline_spent = True  # diagnostic only; cannot produce a pass path
        t2_block = run_pipeline(z, diagnostics_only=True)
        verdict = "NEGATIVE_WITH_LOCALIZATION_V3_LEAK"
        stop_rule = "§6.3: G-T1 PASS ∧ G-T3 FAIL => pipeline once as DIAGNOSTIC ONLY."
    else:
        pipeline_spent = True
        t2_block = run_pipeline(z, diagnostics_only=False)
        val_correct = t2_block["armA_val"]["correct"]
        t2_block["threshold"] = "VAL exact >= 3/639"
        t2_block["pass"] = bool(val_correct >= 3)
        verdict = "PROCEED_TO_INSTALL" if t2_block["pass"] else "NEGATIVE_WITH_LOCALIZATION_V3"
        stop_rule = "§6.4/§7: G-T1 ∧ G-T3 PASS => G-T2 leg run once."

    results = {
        "schema_version": "humanbrain/t-route-e1/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "integrity_echo": {
            "freeze_candidate_raw_sha256": manifest["integrity"]["freeze_candidate_raw_sha256"],
            "train_raw_sha256": manifest["integrity"]["train_raw_sha256"],
            "train_canonical_sha256": manifest["integrity"]["train_embedded_canonical"],
            "val_raw_sha256": manifest["integrity"]["val_raw_sha256"],
            "val_canonical_sha256": manifest["integrity"]["val_embedded_canonical"],
            "counts": {"train": manifest["integrity"]["train_count"], "val": manifest["integrity"]["val_count"]},
            "pairing_events_total": manifest["pairing_events_total"],
            "unique_token_pairs": manifest["unique_token_pairs"],
        },
        "config_echo": manifest["config"],
        "gold_unmatched_census": manifest["gold_unmatched_census"],
        "gt3": {
            "gating_stratum": GATING_STRATUM,
            "pooled_splits": True,
            "a_wrong_passage": {
                "median_overlap": gt3a_median_obs,
                "median_expected_chance": gt3a_median_exp,
                "n_eligible": len(wp_obs),
                "rule": "median wrong-passage overlap <= median per-item analytic chance K*|gold_sigma(i)|/n",
                "pass": bool(gt3a_pass),
            },
            "b_shuffled_w": {
                "rand_med": gt3b_rand_med,
                "shufw_med": gt3b_shufw_med,
                "band_lo": band_lo,
                "band_hi": band_hi,
                "rule": "rand_med/1.5 <= shufw_med <= rand_med*1.5 (who_said, pooled splits)",
                "pass": bool(gt3b_pass),
            },
            "per_item_wrong_passage": wp_rows,
            "pass": gt3_pass,
        },
        "gt1": {
            "gating_stratum": GATING_STRATUM,
            "table": gt1_table,
            "rule": "on BOTH splits: base_med>0 AND base_med>=2*shuf_med AND base_med>=1.5*rand_med (who_said)",
            "pass": gt1_pass,
        },
        "gt2": t2_block,
        "pipeline_spent": pipeline_spent,
        "comparators_unchanged": {
            "unigram_span_acc": UNIGRAM_SPAN_ACC,
            "majority_prior_span_acc": MAJORITY_SPAN_ACC,
            "random_ranking_seed11": RANDOM_SPAN_ACC,
            "source": "experiments/qa_baselines_bible_v1/*.json anchors (29/639, 26/639, 1/639)",
        },
        "stop_rule_applied": stop_rule,
        "verdict": verdict,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {RESULTS_PATH.name}")
    print(json.dumps({
        "verdict": verdict,
        "gt1_pass": gt1_pass,
        "gt3_pass": gt3_pass,
        "gt2": None if t2_block is None else {
            "armA_val_correct": t2_block["armA_val"]["correct"],
            "pass": t2_block.get("pass"),
        },
    }, indent=1))


# ------------------------------------------------------------------ G-T2 leg
def run_pipeline(z, *, diagnostics_only: bool) -> dict:
    log(f"G-T2 two-hop pipeline ({'DIAGNOSTIC ONLY' if diagnostics_only else 'official'}) - run once")
    probes = load_probes_module()

    freeze_dir = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        / "Temp" / "hb_bible_freeze" / "evidence"
    )
    train_items = json.loads((freeze_dir / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads((freeze_dir / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]

    tok = BytePhraseTokenizer.fit(
        [it["serialized_prompt_answer"] for it in train_items],
        TokenizerConfig(max_phrases=0, min_frequency=3),
    )
    assert tok.vocab_size == 256
    proj = SDRProjector(n_bits=N_BITS, w_ones=32, seed=7)

    streams = {}
    for split, items in (("train", train_items), ("val", val_items)):
        streams[split] = {
            "q": [tok.encode(it["question"]) for it in items],
            "p": [tok.encode("\n".join(it["context"])) for it in items],
        }
    n_train, n_val = len(train_items), len(val_items)

    classes = probes.answer_classes(train_items)
    majority_order = probes.majority_ranking(train_items, classes)
    pos_of_word = {w: k for k, w in enumerate(majority_order)}

    # Extraction candidates: distinct TRAIN answer strings (frozen), fixed order.
    candidates = sorted({it["answer"] for it in train_items})
    cand_majpos = [min(pos_of_word[w] for w in probes.words(c)) for c in candidates]
    B = np.zeros((len(candidates), N_BITS), dtype=bool)
    cand_sizes = np.zeros(len(candidates), dtype=np.int64)
    for k, c in enumerate(candidates):
        ids = tok.encode(c)
        for t in ids:
            B[k][proj.indices(t)] = True
        cand_sizes[k] = int(B[k].sum())
    log(f"  extraction: {len(candidates)} distinct TRAIN answer candidates")

    def two_hop(rset: str, split: str, scorer_items, expected) -> dict:
        Wrows = Wm_cache[(rset, split)]
        p_streams = streams[split]["p"]
        rankings, centers = [], []
        for i, p_seq in enumerate(p_streams):
            s1_set = set(int(b) for b in Wrows[i])
            s1_mask = np.zeros(N_BITS, dtype=bool)
            s1_mask[Wrows[i]] = True
            writes = np.zeros(len(p_seq), dtype=np.int64)
            for j, t in enumerate(p_seq):
                if bool(s1_mask[proj.indices(t)].any()):
                    writes[j] = 1
            csum = np.concatenate([[0], np.cumsum(writes)])
            pos_score = np.array([
                csum[min(len(p_seq), j + H2 + 1)] - csum[max(0, j - H2)]
                for j in range(len(p_seq))
            ], dtype=np.int64)
            center = int(np.argmax(pos_score)) if pos_score.size else 0
            centers.append(center)
            lo, hi = max(0, center - H2), min(len(p_seq), center + H2 + 1)
            R = np.zeros(N_BITS, dtype=bool)
            for j in range(lo, hi):
                R[proj.indices(p_seq[j])] = True
            inter = (B & R).sum(axis=1).astype(np.float64)
            scores = np.divide(inter, cand_sizes.astype(np.float64),
                               out=np.zeros_like(inter), where=cand_sizes > 0)
            order = sorted(range(len(candidates)),
                           key=lambda k: (-scores[k], cand_majpos[k], candidates[k]))
            rankings.append([candidates[k] for k in order])
        scorer = probes.SpanScorer(scorer_items, expected)
        scored = scorer.score(rankings)
        return {
            "scored": scored,
            "centers_first10": centers[:10],
        }

    Wm_cache = {(r, s): winners_matrix(z, r, s, n_train if s == "train" else n_val)
                for r in ("armA", "armB") for s in ("train", "val")}

    armA_val = two_hop("armA", "val", val_items, SPAN_PREDICTIONS)
    out: dict = {
        "armA_val": {
            "correct": armA_val["scored"]["correct"],
            "predictions": armA_val["scored"]["predictions"],
            "accuracy": armA_val["scored"]["accuracy"],
            "distinct_predicted_answers": armA_val["scored"]["distinct_predicted_answers"],
            "diagnostics_only": bool(diagnostics_only),
        },
        "armB_val_diagnostic": {
            "correct": two_hop("armB", "val", val_items, SPAN_PREDICTIONS)["scored"]["correct"],
            "note": "diagnostic only",
        },
        "armA_train_self_diagnostic": {
            "scored": two_hop("armA", "train", train_items, None)["scored"],
            "note": "early-warning diagnostic",
        },
    }
    return out


if __name__ == "__main__":
    main()
