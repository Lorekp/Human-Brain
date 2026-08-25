#!/usr/bin/env python3
"""T-ROUTE E2 one-shot logging harness (Change Note 3, PRE-SPEND committed).

Executes the E2 amendment (experiments/t_route/e2/CHANGE_NOTE_3.md) EXACTLY
once. The delta vs the E1 harness (../run_e1.py) is ONLY the post-training
W transform: instead of the Change-Note-1 homeostat, each arm config keeps the
top-k largest entries per row of the raw trained accumulator (init + delta on
the fixed fan-in-32 support), k in the frozen sweep {8, 16, 32}. NO homeostat,
NO normalization, no new random sources anywhere.

Pipeline:
  step 0  integrity: corpus/manifest digests + vocab + counts (parent pins);
          any mismatch => STOP;
  step 1  tokenizer fit + token streams (frozen recipe, verbatim from E1);
  step 2  deterministic rebuild of every E1 substrate (accumulator -> arm A
          homeostat view, arm B dense correlation, randomized-W seed 13,
          shuffled-W seed 11);
  step 3  REPRODUCTION CHECK (Change Note 3 C4): hash-verify the E1 log
          (logged_winner_sets.npz against e1_manifest.json), re-route ALL six
          E1 routing sets with the rebuilt substrates, require BIT-EXACT
          agreement; any mismatch => STOP before anything E2 is logged;
  step 4  route + log E2 configs arm_k{k} and their per-config
          shuffled-question controls, plus the shared randomized-W control,
          into e2_logged_winner_sets.npz (+ e2_manifest.json array hashes).
          Gold/matched/families/sigma arrays are NOT duplicated: the scorer
          reads them from the hash-verified E1 log (same pairing data).

Pure numpy, CPU-only, deterministic. No reruns: this script is spent by its
first successful completion over the corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.hebbian_projection import (  # noqa: E402
    HebbianConfig,
    HomeostaticProjection,
    SparseProjection,
    kwinner_indices,
    passage_union_with_decay,
    population_union,
    shuffled_values_copy,
)
from humanbrain.sdr_union import SDRProjector  # noqa: E402
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig  # noqa: E402

# ------------------------------------------------------------------ freeze pins
FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)
EV = FREEZE_DIR / "evidence"

PIN_FREEZE_CANDIDATE_SHA = "361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c"
PIN_TRAIN_RAW_SHA = "d82fe615dced145d4a6d10d19e90583d1eca4bb6be79e638a96bebc511b0a0e8"
PIN_TRAIN_CANONICAL = "d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b"
PIN_VAL_RAW_SHA = "4f9be9b725c630cbc0e94f8cea4a96413bc026895a3d1b5312882e1980e37dec"
PIN_VAL_CANONICAL = "2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec"

N_BITS, W_ONES, PROJECTOR_SEED = 2048, 32, 7
FAN_IN, W_SEED, K_WINNERS = 32, 7, 128
RAND_W_SEED, SHUFFLE_W_SEED, SHUFFLED_Q_SEED = 13, 11, 11
H_WIN, O_MAX, LAM_P = 12, 8, 0.90
EXPECTED_TRAIN, EXPECTED_VAL = 200, 60

E2_DIR = Path(__file__).resolve().parent
E1_DIR = Path(__file__).resolve().parents[1]
E1_NPZ_PATH = E1_DIR / "logged_winner_sets.npz"
E1_MANIFEST_PATH = E1_DIR / "e1_manifest.json"
NPZ_PATH = E2_DIR / "e2_logged_winner_sets.npz"
MANIFEST_PATH = E2_DIR / "e2_manifest.json"

K_SWEEP = (8, 16, 32)
E1_ROUTING_SETS = (
    "armA",
    "armB",
    "shuffled_question",
    "randomized_W",
    "wrong_passage",
    "shuffled_W",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: object) -> bytes:
    """The builder's own canonicalization (tools/build_bible_qa_corpus.py)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ step 0
def integrity() -> tuple[dict, dict]:
    log("step 0: integrity pins")
    checks: dict[str, object] = {}

    raw_freeze = (FREEZE_DIR / "FREEZE-CANDIDATE-v3-BIBLE.json").read_bytes()
    checks["freeze_candidate_raw_sha256"] = sha256_bytes(raw_freeze)
    if checks["freeze_candidate_raw_sha256"] != PIN_FREEZE_CANDIDATE_SHA:
        raise SystemExit("STOP: freeze-candidate digest mismatch")

    manifests: dict[str, dict] = {}
    for split, raw_pin, canon_pin in (
        ("train", PIN_TRAIN_RAW_SHA, PIN_TRAIN_CANONICAL),
        ("val", PIN_VAL_RAW_SHA, PIN_VAL_CANONICAL),
    ):
        raw = (EV / f"qa_manifest_{split}.json").read_bytes()
        digest = sha256_bytes(raw)
        manifest = json.loads(raw.decode("utf-8"))
        embedded = manifest.get("sha256_canonical")
        recomputed = sha256_bytes(compact_json({k: v for k, v in manifest.items() if k != "sha256_canonical"}))
        checks[f"{split}_raw_sha256"] = digest
        checks[f"{split}_embedded_canonical"] = embedded
        checks[f"{split}_recomputed_canonical"] = recomputed
        if digest != raw_pin:
            raise SystemExit(f"STOP: {split} manifest raw-bytes digest mismatch")
        if embedded != canon_pin or recomputed != canon_pin:
            raise SystemExit(f"STOP: {split} manifest canonical digest mismatch")
        n = len(manifest["items"])
        expected = EXPECTED_TRAIN if split == "train" else EXPECTED_VAL
        if n != expected:
            raise SystemExit(f"STOP: {split} item count {n} != {expected}")
        checks[f"{split}_count"] = n
        manifests[split] = manifest

    families_train = Counter(it["family"] for it in manifests["train"]["items"])
    families_val = Counter(it["family"] for it in manifests["val"]["items"])
    checks["families"] = {"train": dict(families_train), "val": dict(families_val)}
    log(f"  integrity OK; families train={dict(families_train)} val={dict(families_val)}")
    return checks, manifests


# ------------------------------------------------------------------ step 1/2 helpers (verbatim E1 recipes)
def aggregate_pair_counts(train_items: list[dict], tokenizer: BytePhraseTokenizer) -> Counter:
    counts: Counter = Counter()
    for item in train_items:  # canonical manifest order
        p_ids = tokenizer.encode("\n".join(item["context"]))
        q_ids = tokenizer.encode(item["question"])
        occurrences: dict[int, list[int]] = {}
        for j, t in enumerate(p_ids):
            occurrences.setdefault(t, []).append(j)
        for tq in dict.fromkeys(q_ids):
            anchors = occurrences.get(tq)
            if not anchors:
                continue
            for j in anchors[:O_MAX]:
                lo, hi = max(0, j - H_WIN), min(len(p_ids), j + H_WIN + 1)
                for jp in range(lo, hi):
                    counts[(tq, p_ids[jp])] += 1
    return counts


def dense_delta_from_pairs(counts: Counter, projector: SDRProjector) -> np.ndarray:
    n = int(projector.n_bits)
    dense = np.zeros((n, n), dtype=np.float64)
    if not counts:
        return dense
    triplets = [(int(tq), int(tp), int(c)) for (tq, tp), c in counts.items()]
    w = int(projector.w_ones)
    rows = np.concatenate([np.repeat(projector.indices(tp), w) for _, tp, _ in triplets])
    cols = np.concatenate([np.tile(projector.indices(tq), w) for tq, _, _ in triplets])
    vals = np.repeat(np.array([c for _, _, c in triplets], dtype=np.float64), w * w)
    np.add.at(dense, (rows, cols), vals)
    return dense


def document_frequencies(p_streams: list[list[int]], projector: SDRProjector) -> np.ndarray:
    df = np.zeros(N_BITS, dtype=np.int64)
    for p_ids in p_streams:
        support = np.zeros(N_BITS, dtype=bool)
        for t in set(p_ids):
            support[projector.indices(t)] = True
        df += support
    return df


# ------------------------------------------------------------------ step 4 helper
def topk_row_sparsify(support: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Change Note 3 C2: keep the k largest-valued entries per row, zero the rest.

    Ties broken by ASCENDING column index (stable argsort on negated values);
    rows with fewer than k nonzeros keep all of them. Deterministic; no RNG.
    """
    values = np.asarray(values, dtype=np.float64)
    fan_in = support.shape[1]
    kk = int(min(int(k), fan_in))
    order = np.argsort(-values, axis=1, kind="stable")[:, :kk]
    out = np.zeros_like(values)
    rows = np.repeat(np.arange(values.shape[0]), kk)
    out[rows, order.ravel()] = values[rows, order.ravel()]
    return out


class WinnerSink:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, int]] = []
        self.winners: list[np.ndarray] = []

    def record(self, routing_set: str, split: str, index: int, winners: np.ndarray) -> None:
        self.rows.append((routing_set, split, int(index)))
        self.winners.append(np.asarray(winners, dtype=np.int16))

    def arrays(self) -> dict[str, np.ndarray]:
        stacked = np.stack(self.winners) if self.winners else np.zeros((0, K_WINNERS), np.int16)
        sets = np.array([r[0] for r in self.rows], dtype="U24")
        splits = np.array([r[1] for r in self.rows], dtype="U5")
        idx = np.array([r[2] for r in self.rows], dtype=np.int64)
        return {
            "winner_sets": stacked,
            "routing_set": sets,
            "split": splits,
            "item_index": idx,
        }


# ------------------------------------------------------------------ step 3
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


def logged_winners(z: dict, rset: str, split: str, n_items: int) -> np.ndarray:
    sel = (z["split"] == split) & (z["routing_set"] == rset)
    idx = z["item_index"][sel]
    order = np.argsort(idx)
    assert len(idx) == n_items, f"missing rows for {rset}/{split}"
    assert np.array_equal(idx[order], np.arange(n_items)), f"row order broken for {rset}/{split}"
    return z["winner_sets"][sel][order].astype(np.int64)


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.perf_counter()
    integrity_checks, manifests = integrity()

    log("step 1: tokenizer fit + streams (frozen recipe)")
    train_items = manifests["train"]["items"]
    val_items = manifests["val"]["items"]
    tokenizer = BytePhraseTokenizer.fit(
        [it["serialized_prompt_answer"] for it in train_items],
        TokenizerConfig(max_phrases=0, min_frequency=3),
    )
    if tokenizer.vocab_size != 256:
        raise SystemExit(f"STOP: tokenizer.vocab_size {tokenizer.vocab_size} != 256")

    streams: dict[str, dict[str, list[list[int]]]] = {}
    for split, items in (("train", train_items), ("val", val_items)):
        qs, ps = [], []
        for item in items:
            qs.append(tokenizer.encode(item["question"]))
            ps.append(tokenizer.encode("\n".join(item["context"])))
        streams[split] = {"q": qs, "p": ps}

    projector = SDRProjector(n_bits=N_BITS, w_ones=W_ONES, seed=PROJECTOR_SEED)

    log("step 2: deterministic rebuild of every E1 substrate")
    t0 = time.perf_counter()
    pair_counts = aggregate_pair_counts(train_items, tokenizer)
    log(f"  pairing events total={sum(pair_counts.values())} unique pairs={len(pair_counts)}")

    config = HebbianConfig(
        n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN, w_seed=W_SEED, k_winners=K_WINNERS
    )
    accumulator = SparseProjection(config)
    accumulator.apply_dense_delta(dense_delta_from_pairs(pair_counts, projector))

    df = document_frequencies(streams["train"]["p"], projector)
    n_pass = len(train_items)
    armA: HomeostaticProjection = accumulator.apply_homeostasis(df, n_passages=n_pass)

    armB = np.zeros((N_BITS, N_BITS), dtype=np.float64)
    for i in range(n_pass):
        q_pop = population_union(projector, streams["train"]["q"][i])
        p_pop = passage_union_with_decay(projector, streams["train"]["p"][i], LAM_P)
        armB += np.outer(p_pop, q_pop)

    randW = SparseProjection(
        HebbianConfig(n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN, w_seed=RAND_W_SEED, k_winners=K_WINNERS)
    )
    shufW = shuffled_values_copy(accumulator, SHUFFLE_W_SEED)
    log(f"  substrates rebuilt ({time.perf_counter()-t0:.1f}s)")

    e1_matrices: dict[str, object] = {
        "armA": armA,
        "shuffled_question": armA,
        "randomized_W": randW,
        "wrong_passage": armA,
        "shuffled_W": shufW,
    }

    log("step 3: reproduction check vs hash-verified E1 log (Change Note 3 C4)")
    e1_z = verify_npz(E1_NPZ_PATH, E1_MANIFEST_PATH)
    max_diff = 0
    mismatches: list[str] = []
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        perm_rng = np.random.default_rng(SHUFFLED_Q_SEED)
        pi = perm_rng.permutation(n_items)
        for rset in E1_ROUTING_SETS:
            rebuilt = np.empty((n_items, K_WINNERS), dtype=np.int64)
            for i in range(n_items):
                q_idx = pi[i] if rset == "shuffled_question" else i
                q_pop = population_union(projector, streams[split]["q"][q_idx])
                if rset == "armB":
                    rebuilt[i] = kwinner_indices(armB @ q_pop, K_WINNERS)
                else:
                    rebuilt[i] = e1_matrices[rset].route(q_pop, K_WINNERS)  # type: ignore[attr-defined]
            logged = logged_winners(e1_z, rset, split, n_items)
            d = int(np.abs(rebuilt.astype(np.int64) - logged.astype(np.int64)).max()) if n_items else 0
            identical = bool(np.array_equal(rebuilt.astype(np.int64), logged.astype(np.int64)))
            max_diff = max(max_diff, d)
            if not identical:
                mismatches.append(f"{rset}/{split}")
            log(f"  {rset}/{split}: {'IDENTICAL' if identical else f'MISMATCH maxdiff={d}'}")
    if mismatches:
        raise SystemExit(f"STOP: reproduction check FAILED for {mismatches}; nothing E2 logged")
    log("  reproduction check PASSED: all six E1 routing sets bit-exact")

    # Gold/matched/families/sigma come FROM the verified E1 log (C4): same pairing data.
    gold_echo = {
        name: {"sha256": sha256_bytes(e1_z[name].tobytes()), "dtype": str(e1_z[name].dtype)}
        for name in sorted(e1_z)
        if name.startswith(("gold_", "matched_", "families_", "sigma_", "shuffle_perm_", "qa_ids_"))
    }

    log("step 4: route + log E2 configs (one-shot)")
    sink = WinnerSink()
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        pi = np.random.default_rng(SHUFFLED_Q_SEED).permutation(n_items)
        q_pops = [population_union(projector, streams[split]["q"][i]) for i in range(n_items)]
        q_pops_shuf = [q_pops[int(pi[i])] for i in range(n_items)]

        # Shared randomized-W control (identical machinery, untrained seed 13).
        for i in range(n_items):
            sink.record("randomized_W", split, i, randW.route(q_pops[i], K_WINNERS))

        for k in K_SWEEP:
            sparsified = topk_row_sparsify(accumulator.support(), accumulator.values(), k)
            mat_k = HomeostaticProjection(config, accumulator.support(), sparsified)
            if k == FAN_IN:
                # Identity-transform guard: k=fan_in must equal the raw
                # accumulator's own routing exactly (no homeostat, nothing else).
                for i in range(0, n_items, max(1, n_items // 20)):
                    assert np.array_equal(
                        mat_k.route(q_pops[i], K_WINNERS),
                        accumulator.route(q_pops[i], K_WINNERS),
                    ), f"identity transform broken at k={fan_in}"
            nnz_kept = float((sparsified > 0).mean())
            for i in range(n_items):
                sink.record(f"arm_k{k}", split, i, mat_k.route(q_pops[i], K_WINNERS))
                sink.record(f"shufq_k{k}", split, i, mat_k.route(q_pops_shuf[i], K_WINNERS))
            log(f"  arm_k{k}/{split}: routed (kept nnz fraction={nnz_kept:.3f})")

    arrays = sink.arrays()
    if NPZ_PATH.exists():
        raise SystemExit(f"STOP: {NPZ_PATH.name} already exists (one-shot discipline)")
    np.savez_compressed(NPZ_PATH, **arrays)

    array_hashes = {
        name: {"sha256": sha256_bytes(arrays[name].tobytes()), "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }
    manifest = {
        "schema_version": "humanbrain/t-route-e2-winner-sets/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "amendment": "experiments/t_route/e2/CHANGE_NOTE_3.md (committed pre-spend)",
        "npz_file": NPZ_PATH.name,
        "npz_sha256": sha256_bytes(NPZ_PATH.read_bytes()),
        "array_sha256": array_hashes,
        "reproduction_check": {
            "e1_npz_file": "../" + E1_NPZ_PATH.name,
            "e1_npz_sha256": sha256_bytes(E1_NPZ_PATH.read_bytes()),
            "all_six_e1_routing_sets_bit_exact": True,
            "gold_array_echo_from_e1_log": gold_echo,
        },
        "integrity": integrity_checks,
        "config": {
            "n_bits": N_BITS, "w_ones": W_ONES, "projector_seed": PROJECTOR_SEED,
            "fan_in": FAN_IN, "w_seed": W_SEED, "k_winners": K_WINNERS,
            "randomized_w_seed": RAND_W_SEED, "shuffle_w_seed": SHUFFLE_W_SEED,
            "shuffled_question_seed": SHUFFLED_Q_SEED,
            "h_win": H_WIN, "o_max": O_MAX, "lam_P": LAM_P,
            "k_sweep": list(K_SWEEP),
            "transform": "per-row top-k by value, ties ascending column index; NO homeostat, NO normalization",
        },
        "pairing_events_total": int(sum(pair_counts.values())),
        "unique_token_pairs": int(len(pair_counts)),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "note": "Binary npz stays local; array-level sha256 committed here.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"E2 logging complete: {NPZ_PATH.name} ({NPZ_PATH.stat().st_size} bytes), {MANIFEST_PATH.name}")
    log(f"total elapsed {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
