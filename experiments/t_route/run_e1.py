#!/usr/bin/env python3
"""T-ROUTE E1 one-shot logging harness (D4).

Executes the frozen protocol of
docs/research/trained-inter-space-preregistration.md EXACTLY once:

  step 0  integrity: corpus/manifest digests (raw bytes + embedded canonical
          recomputed with the builder's own compact_json recipe), vocab==256,
          counts 200/60 — any mismatch => STOP, nothing runs;
  step 1  tokenizer fit + token streams (question / passage / answer);
  step 2  arm A substrate: sparse W (seed 7) + ONE-SHOT aggregated
          token-pair dW via apply_dense_delta (equivalence-tested), then the
          Change-Note-1 homeostat; arm B: dense unnormalized correlation;
          controls randomized-W (seed 13) and shuffled-W (values permuted,
          seed 11, topology fixed);
  step 3  routed winner sets for ALL arms/controls logged through an
          observation-sink into logged_winner_sets.npz (+ e1_manifest.json
          with array-level sha256; the binary npz stays local).

Storage convention (all arms identical): rows = postsynaptic/passage bits,
columns = presynaptic/question bits, so prereg section 4's uniform readout
``drive = M_arm @ q_pop`` holds verbatim for every matrix. The correlation
content of arm B and of the pairing outer products is exactly the frozen
outer(q, p); only the memory layout follows the readout convention.

Gates are NOT evaluated here — score_e1.py reads the logged sets only.

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.hebbian_projection import (  # noqa: E402
    HebbianConfig,
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
H_WIN, O_MAX, ADJ, LAM_P = 12, 8, 2, 0.90
EXPECTED_TRAIN, EXPECTED_VAL = 200, 60

OUT_DIR = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "logged_winner_sets.npz"
MANIFEST_PATH = OUT_DIR / "e1_manifest.json"

ROUTING_SETS = (
    "armA",              # trained sparse W + homeostat (prereg arm A)
    "armB",              # dense unnormalized correlation (prereg arm B)
    "shuffled_question", # pi = default_rng(11).permutation per split; arm-A routing
    "randomized_W",      # untrained seed-13 sparse matrix
    "wrong_passage",     # real question, arm-A routing, scored vs gold(sigma(i))
    "shuffled_W",        # trained-W values permuted on fixed support, seed 11
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
        # Recompute with the builder's own recipe (manifest sans digest field).
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


# ------------------------------------------------------------------ step 2
def aggregate_pair_counts(
    train_items: list[dict], tokenizer: BytePhraseTokenizer
) -> Counter:
    """Prereg section 2 pairing events, aggregated to token-pair counts.

    For each question token id tq: first O_MAX anchor positions j with
    t_p[j]==tq; every j' with |j'-j| <= H_WIN contributes one event (tq,
    t_p[j']) with M=1 (binary third factor folded into multiplicity).
    """
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
    """Aggregated dW blocks: dense_delta[post_bit, pre_bit] += event count."""
    n = int(projector.n_bits)
    dense = np.zeros((n, n), dtype=np.float64)
    if not counts:
        return dense
    triplets = [(int(tq), int(tp), int(c)) for (tq, tp), c in counts.items()]
    w = int(projector.w_ones)
    # Full w x w outer-product block per token pair: post bits repeated across
    # columns, pre bits tiled down rows, event count on every combination.
    rows = np.concatenate([np.repeat(projector.indices(tp), w) for _, tp, _ in triplets])
    cols = np.concatenate([np.tile(projector.indices(tq), w) for tq, _, _ in triplets])
    vals = np.repeat(
        np.array([c for _, _, c in triplets], dtype=np.float64), w * w
    )
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


class WinnerSink:
    """Observation sink: collects routed winner sets during routing only."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, int]] = []
        self.winners: list[np.ndarray] = []

    def record(self, routing_set: str, split: str, index: int, winners: np.ndarray) -> None:
        self.rows.append((routing_set, split, int(index)))
        self.winners.append(np.asarray(winners, dtype=np.int16))

    def arrays(self) -> dict[str, np.ndarray]:
        stacked = np.stack(self.winners) if self.winners else np.zeros((0, K_WINNERS), np.int16)
        sets = np.array([r[0] for r in self.rows], dtype="U24")  # longest name: "shuffled_question" (17)
        splits = np.array([r[1] for r in self.rows], dtype="U5")
        idx = np.array([r[2] for r in self.rows], dtype=np.int64)
        return {
            "winner_sets": stacked,
            "routing_set": sets,
            "split": splits,
            "item_index": idx,
        }


# ------------------------------------------------------------------ main
def main() -> None:
    integrity_checks, manifests = integrity()

    log("step 1: tokenizer fit + streams")
    train_items = manifests["train"]["items"]
    val_items = manifests["val"]["items"]
    tokenizer = BytePhraseTokenizer.fit(
        [it["serialized_prompt_answer"] for it in train_items],
        TokenizerConfig(max_phrases=0, min_frequency=3),
    )
    if tokenizer.vocab_size != 256:
        raise SystemExit(f"STOP: tokenizer.vocab_size {tokenizer.vocab_size} != 256")

    streams: dict[str, list[list[int]]] = {}
    for split, items in (("train", train_items), ("val", val_items)):
        qs, ps, ans = [], [], []
        for item in items:
            qs.append(tokenizer.encode(item["question"]))
            ps.append(tokenizer.encode("\n".join(item["context"])))
            ans.append(tokenizer.encode(item["answer"]))
        streams[split] = {"q": qs, "p": ps, "a": ans}
    log(
        f"  vocab={tokenizer.vocab_size}; mean |q|="
        f"{np.mean([len(s) for s in streams['train']['q']]):.1f}/"
        f"{np.mean([len(s) for s in streams['val']['q']]):.1f}, "
        f"mean |p|={np.mean([len(s) for s in streams['train']['p']]):.1f}"
    )

    projector = SDRProjector(n_bits=N_BITS, w_ones=W_ONES, seed=PROJECTOR_SEED)

    log("step 2: arms — one-shot accumulation (TRAIN only)")
    t0 = time.perf_counter()
    pair_counts = aggregate_pair_counts(train_items, tokenizer)
    log(f"  pairing events total={sum(pair_counts.values())} unique pairs={len(pair_counts)}")

    config = HebbianConfig(
        n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN, w_seed=W_SEED, k_winners=K_WINNERS
    )
    accumulator = SparseProjection(config)
    accumulator.apply_dense_delta(dense_delta_from_pairs(pair_counts, projector))
    log(f"  arm A delta applied ({time.perf_counter()-t0:.1f}s)")

    df = document_frequencies(streams["train"]["p"], projector)
    n_pass = len(train_items)
    armA = accumulator.apply_homeostasis(df, n_passages=n_pass)
    touched = int((np.linalg.norm(accumulator.values(), axis=1) > 0).sum())
    log(f"  homeostat: nonzero rows={touched}/{N_BITS} df[min/max]={df.min()}/{df.max()}")

    # Arm B: C[passage_bit, question_bit] += P_pop * Q_pop (undecayed Q, lam_P=0.90 P).
    armB = np.zeros((N_BITS, N_BITS), dtype=np.float64)
    for i in range(n_pass):
        q_pop = population_union(projector, streams["train"]["q"][i])
        p_pop = passage_union_with_decay(projector, streams["train"]["p"][i], LAM_P)
        armB += np.outer(p_pop, q_pop)
    log(f"  arm B accumulated: nnz={int(np.count_nonzero(armB))} max={armB.max():.4g}")

    randW = SparseProjection(
        HebbianConfig(n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN, w_seed=RAND_W_SEED, k_winners=K_WINNERS)
    )
    shufW = shuffled_values_copy(accumulator, SHUFFLE_W_SEED)

    matrices: dict[str, object] = {
        "armA": armA,
        "armB": None,  # dense path handled inline below
        "shuffled_question": armA,
        "randomized_W": randW,
        "wrong_passage": armA,
        "shuffled_W": shufW,
    }

    log("step 3: gold regions")
    gold_masks: dict[str, np.ndarray] = {}
    gold_sizes: dict[str, np.ndarray] = {}
    matched: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        masks = np.zeros((n_items, N_BITS), dtype=bool)
        hits = np.zeros(n_items, dtype=np.int64)
        for i in range(n_items):
            a_seq = streams[split]["a"][i]
            p_seq = streams[split]["p"][i]
            regions = contiguous_match_regions(a_seq, p_seq)
            hits[i] = len(regions)
            support = np.zeros(N_BITS, dtype=bool)
            for lo, hi in regions:
                for j in range(lo, hi):
                    support[projector.indices(p_seq[j])] = True
            masks[i] = support
        gold_masks[split] = masks
        gold_sizes[split] = masks.sum(axis=1).astype(np.int64)
        matched[split] = hits > 0
        log(
            f"  {split}: GOLD_UNMATCHED={int((~matched[split]).sum())}/{n_items} "
            f"(multi-region items={int((hits > 1).sum())})"
        )

    log("step 3b: routing all arms/controls through the winner sink")
    sink = WinnerSink()
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        perm_rng = np.random.default_rng(SHUFFLED_Q_SEED)
        pi = perm_rng.permutation(n_items)
        sigma = [(i + n_items // 2) % n_items for i in range(n_items)]
        for rset in ROUTING_SETS:
            mat = matrices[rset]
            for i in range(n_items):
                q_idx = pi[i] if rset == "shuffled_question" else i
                q_pop = population_union(projector, streams[split]["q"][q_idx])
                if rset == "armB":
                    winners = kwinner_indices(armB @ q_pop, K_WINNERS)
                else:
                    winners = mat.route(q_pop, K_WINNERS)  # type: ignore[attr-defined]
                sink.record(rset, split, i, winners)
    log(f"  routed rows={len(sink.rows)} x K={K_WINNERS}")

    log("step 3c: writing npz + manifest")
    arrays = sink.arrays()
    for split in ("train", "val"):
        arrays[f"gold_mask_{split}"] = gold_masks[split]
        arrays[f"gold_sizes_{split}"] = gold_sizes[split]
        arrays[f"matched_{split}"] = matched[split]
        arrays[f"shuffle_perm_{split}"] = np.random.default_rng(SHUFFLED_Q_SEED).permutation(
            len(streams[split]["q"])
        ).astype(np.int64)
        arrays[f"sigma_{split}"] = np.array(
            [(i + len(streams[split]["q"]) // 2) % len(streams[split]["q"]) for i in range(len(streams[split]["q"]))],
            dtype=np.int64,
        )
    for split in ("train", "val"):
        arrays[f"qa_ids_{split}"] = np.array([it["qa_id"] for it in (train_items if split == "train" else val_items)], dtype="U24")
        arrays[f"families_{split}"] = np.array([it["family"] for it in (train_items if split == "train" else val_items)], dtype="U24")

    if NPZ_PATH.exists():
        raise SystemExit(f"STOP: {NPZ_PATH.name} already exists (one-shot discipline)")
    np.savez_compressed(NPZ_PATH, **arrays)

    array_hashes = {
        name: {"sha256": sha256_bytes(arrays[name].tobytes()), "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }
    manifest = {
        "schema_version": "humanbrain/t-route-e1-winner-sets/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "npz_file": NPZ_PATH.name,
        "npz_sha256": sha256_bytes(NPZ_PATH.read_bytes()),
        "array_sha256": array_hashes,
        "integrity": integrity_checks,
        "config": {
            "n_bits": N_BITS, "w_ones": W_ONES, "projector_seed": PROJECTOR_SEED,
            "fan_in": FAN_IN, "w_seed": W_SEED, "k_winners": K_WINNERS,
            "randomized_w_seed": RAND_W_SEED, "shuffle_w_seed": SHUFFLE_W_SEED,
            "shuffled_question_seed": SHUFFLED_Q_SEED,
            "h_win": H_WIN, "o_max": O_MAX, "adj": ADJ, "lam_P": LAM_P,
            "n_passages_train": n_pass,
        },
        "pairing_events_total": int(sum(pair_counts.values())),
        "unique_token_pairs": int(len(pair_counts)),
        "gold_unmatched_census": {
            split: {
                "count": int((~matched[split]).sum()),
                "total": int(matched[split].size),
                "by_family": {
                    fam: int(sum(1 for k, it in enumerate(items_split) if not matched[split][k] and it["family"] == fam))
                    for fam in sorted({it["family"] for it in items_split})
                },
            }
            for split, items_split in (("train", train_items), ("val", val_items))
        },
        "note": "Binary npz stays local; array-level sha256 committed here.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"D4 logging complete: {NPZ_PATH.name} ({NPZ_PATH.stat().st_size} bytes), {MANIFEST_PATH.name}")


def contiguous_match_regions(a_seq: list[int], p_seq: list[int]) -> list[tuple[int, int]]:
    """All exact contiguous matches of the answer token sequence, extended by ADJ."""
    regions: list[tuple[int, int]] = []
    la, lp = len(a_seq), len(p_seq)
    if la == 0 or lp < la:
        return regions
    for j in range(lp - la + 1):
        if p_seq[j : j + la] == a_seq:
            regions.append((max(0, j - ADJ), min(lp, j + la + ADJ)))
    return regions


if __name__ == "__main__":
    main()
