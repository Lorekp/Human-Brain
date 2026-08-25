#!/usr/bin/env python3
"""T-ROUTE E3 one-shot logging harness (Change Note 4, PRE-SPEND committed).

Executes the E3 amendment (experiments/t_route/e3/CHANGE_NOTE_4.md) EXACTLY
once. Delta vs the E2 harness (../e2/run_e2.py): the aggregated TRAIN delta is
QUOTE-SPAN-GATED — a pair (tq, p_ids[jp]) contributes iff the POSTSYNAPTIC
passage position jp lies inside an in-passage occurrence of the episode's
verbatim quote (dW_ij *= 1[token_j in quote_span(episode)]). Post-training
transform: E2's exact per-row top-k sparsifier with k frozen at 16 (E2-best;
sweep was flat). NO homeostat, NO normalization, no new RNG anywhere except
the frozen random-span control stream (default_rng(17)).

Pipeline:
  step 0   integrity: corpus/manifest digests + vocab + counts (parent pins);
  step 1   tokenizer fit + token streams (frozen recipe, verbatim from E1);
  step 2   deterministic rebuild of every E1 substrate (accumulator -> arm A
           homeostat view, arm B dense correlation, randomized-W seed 13,
           shuffled-W seed 11) + quote-span census and the TWO gated pair
           counters (quote spans; random spans, same inclusion set/lengths,
           default_rng(17) contiguous placement);
  step 3a  REPRODUCTION CHECK C4(1): hash-verify the E1 log, re-route ALL six
           E1 routing sets with the rebuilt substrates, require BIT-EXACT
           agreement (as E2 C4);
  step 3b  REPRODUCTION CHECK C4(2)+(3): hash-verify the E2 log, re-route ALL
           SEVEN E2 logged sets with the verbatim E2 sparsifier, require
           BIT-EXACT agreement; assert the independently recomputed ungated
           pair Counter equals the E1/E2 aggregate exactly;
  step 4   build the two gated accumulators (quote / random span), apply the
           k=16 transform, route + log ONLY {arm_q16, shufq_q16, rs16,
           shufq_rs16} x {train, val} into e3_logged_winner_sets.npz
           (+ e3_manifest.json array hashes). Overwrite-refusing one-shot.

Pure numpy, CPU-only, deterministic. This script is spent by its first
successful completion over the corpus.
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

K_FIXED = 16                 # Change Note 4 C2: single config, E2-best k
RANDOM_SPAN_SEED = 17        # Change Note 4 C3b: random-span control stream

E3_DIR = Path(__file__).resolve().parent
E2_DIR = Path(__file__).resolve().parents[1] / "e2"
E1_DIR = Path(__file__).resolve().parents[1]
E1_NPZ_PATH = E1_DIR / "logged_winner_sets.npz"
E1_MANIFEST_PATH = E1_DIR / "e1_manifest.json"
E2_NPZ_PATH = E2_DIR / "e2_logged_winner_sets.npz"
E2_MANIFEST_PATH = E2_DIR / "e2_manifest.json"
NPZ_PATH = E3_DIR / "e3_logged_winner_sets.npz"
MANIFEST_PATH = E3_DIR / "e3_manifest.json"

E2_K_SWEEP = (8, 16, 32)
E1_ROUTING_SETS = (
    "armA",
    "armB",
    "shuffled_question",
    "randomized_W",
    "wrong_passage",
    "shuffled_W",
)
E2_ROUTING_SETS = tuple(
    [f"arm_k{k}" for k in E2_K_SWEEP]
    + [f"shufq_k{k}" for k in E2_K_SWEEP]
    + ["randomized_W"]
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


# ------------------------------------------------- step 1/2 helpers (verbatim recipes)
def aggregate_pair_counts(train_items: list[dict], tokenizer: BytePhraseTokenizer) -> Counter:
    """VERBATIM E1/E2 recipe (experiments/t_route/run_e1.py)."""
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


def episode_pair_occurrences(
    question_ids: list[int], passage_ids: list[int]
) -> list[tuple[int, int]]:
    """Same §2 pairing rule, kept POSITIONAL: returns (tq, jp) event pairs.

    Aggregating Counter((tq, passage_ids[jp]) ...) over this list reproduces
    aggregate_pair_counts exactly (asserted per item at run time).
    """
    passage = [int(t) for t in passage_ids]
    n_p = len(passage)
    occurrences: dict[int, list[int]] = {}
    for j, t in enumerate(passage):
        occurrences.setdefault(t, []).append(j)
    occ: list[tuple[int, int]] = []
    for tq in dict.fromkeys(int(t) for t in question_ids):
        anchors = occurrences.get(tq)
        if not anchors:
            continue
        for j in anchors[:O_MAX]:
            lo, hi = max(0, j - H_WIN), min(n_p, j + H_WIN + 1)
            for jp in range(lo, hi):
                occ.append((tq, jp))
    return occ


def quote_token_ids(question: str, tokenizer: BytePhraseTokenizer) -> list[int] | None:
    """Change Note 4 C2: text between the FIRST '"' and the LAST '"'.

    Returns None (empty-mask episode) if fewer than two quotes or an empty
    span; otherwise the token ids of the quoted substring.
    """
    qb = question.encode("utf-8")
    i0, i1 = qb.find(b'"'), qb.rfind(b'"')
    if i0 == -1 or i1 <= i0:
        return None
    return tokenizer.encode(qb[i0 + 1 : i1].decode("utf-8"))


def contiguous_subsequence_matches(haystack: list[int], needle: list[int]) -> list[int]:
    """All (overlapping) start positions of needle in haystack."""
    if not needle or len(needle) > len(haystack):
        return []
    starts: list[int] = []
    first = needle[0]
    for k in range(len(haystack) - len(needle) + 1):
        if haystack[k] == first and haystack[k : k + len(needle)] == needle:
            starts.append(k)
    return starts


def quote_span_mask(
    question: str, passage_ids: list[int], tokenizer: BytePhraseTokenizer
) -> tuple[np.ndarray | None, int, int]:
    """Change Note 4 C2 span localization: union of ALL contiguous matches.

    Returns (mask|None, n_matches, span_len). mask is None iff the episode has
    no quoted substring (contributes ZERO plasticity to gated arms).
    """
    q_ids = quote_token_ids(question, tokenizer)
    if q_ids is None:
        return None, 0, 0
    matches = contiguous_subsequence_matches([int(t) for t in passage_ids], q_ids)
    mask = np.zeros(len(passage_ids), dtype=bool)
    for k in matches:
        mask[k : k + len(q_ids)] = True
    return mask, len(matches), len(q_ids)


def random_span_start(rng: np.random.Generator, n_p: int, span_len: int) -> int:
    """One uniform integer start in [0, n_p - span_len] (inclusive bounds)."""
    return int(rng.integers(0, n_p - span_len + 1))


def dense_delta_from_pairs(counts: Counter, projector: SDRProjector) -> np.ndarray:
    """VERBATIM E1/E2 recipe: rows = PASSAGE bits (postsynaptic), cols = question."""
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
    """VERBATIM E2 recipe (Change Note 3 C2): keep k largest per row, ties ASC column."""
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


# ------------------------------------------------------------------ log verification
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
    log(f"  ungated pairing events total={sum(pair_counts.values())} unique pairs={len(pair_counts)}")

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

    log("step 2b: quote-span census + gated pair counters (Change Note 4 C2/C3b)")
    t0 = time.perf_counter()
    gated_counts: Counter = Counter()
    rspan_counts: Counter = Counter()
    rng_rs = np.random.default_rng(RANDOM_SPAN_SEED)
    rs_draw_echo = hashlib.sha256()
    census = {
        "gated_episodes": 0,
        "excluded_no_quote_or_unmatched": 0,
        "total_matches": 0,
        "multi_match_episodes": 0,
        "span_byte_lengths": [],
    }
    for t_i, item in enumerate(train_items):  # canonical TRAIN manifest order
        p_ids = streams["train"]["p"][t_i]
        occ = episode_pair_occurrences(streams["train"]["q"][t_i], p_ids)
        ungated_item = Counter((tq, p_ids[jp]) for tq, jp in occ)
        expect_item = aggregate_pair_counts([item], tokenizer)
        if ungated_item != expect_item:
            raise SystemExit(f"STOP: positional/unaggregated counter mismatch for {item['qa_id']}")
        mask, n_match, span_len = quote_span_mask(item["question"], p_ids, tokenizer)
        if mask is None or not mask.any():
            census["excluded_no_quote_or_unmatched"] += 1
            continue  # Change Note 4 C2: ZERO plasticity from this episode
        census["gated_episodes"] += 1
        census["total_matches"] += n_match
        census["multi_match_episodes"] += int(n_match > 1)
        census["span_byte_lengths"].append(span_len)
        for tq, jp in occ:
            if mask[jp]:
                gated_counts[(tq, p_ids[jp])] += 1
        # Random-span control: same inclusion set, same length, random placement.
        start = random_span_start(rng_rs, len(p_ids), span_len)
        rs_draw_echo.update(f"{item['qa_id']}:{start}:{len(p_ids)}:{span_len};".encode("ascii"))
        rmask = np.zeros(len(p_ids), dtype=bool)
        rmask[start : start + span_len] = True
        for tq, jp in occ:
            if rmask[jp]:
                rspan_counts[(tq, p_ids[jp])] += 1
    if census["gated_episodes"] != 60:
        raise SystemExit(
            f"STOP: gated episode census {census['gated_episodes']} != 60 "
            "(design-time feasibility figure frozen in Change Note 4)"
        )
    su, sg, sr = sum(pair_counts.values()), sum(gated_counts.values()), sum(rspan_counts.values())
    if sr == 0 or sg == 0:
        raise SystemExit(f"STOP: gated mass is empty (quote={sg}, random={sr}, ungated={su})")
    # NOTE: pair-EVENT masses sg != sr in general BY DESIGN: equal per-episode
    # masked token counts (span lengths) do NOT imply equal event counts, since
    # events depend on anchor-window coverage of the masked POSITIONS — the very
    # quantity the random-span control deliberately varies. Only span lengths and
    # the inclusion set are matched (Change Note 4 C3b).
    log(
        f"  gated episodes={census['gated_episodes']} excluded={census['excluded_no_quote_or_unmatched']} "
        f"multi-match={census['multi_match_episodes']} matches={census['total_matches']}"
    )
    log(
        f"  event mass: ungated={su} gated_quote={sg} ({sg/su:.4f} of ungated) "
        f"gated_random={sr} | unique pairs {len(pair_counts)}/{len(gated_counts)}/{len(rspan_counts)}"
    )
    census["span_byte_lengths"] = {
        "mean": float(np.mean(census["span_byte_lengths"])),
        "median": float(np.median(census["span_byte_lengths"])),
        "min": int(np.min(census["span_byte_lengths"])),
        "max": int(np.max(census["span_byte_lengths"])),
    }
    log(f"  counters built ({time.perf_counter()-t0:.1f}s)")

    e1_z = verify_npz(E1_NPZ_PATH, E1_MANIFEST_PATH)
    e2_z = verify_npz(E2_NPZ_PATH, E2_MANIFEST_PATH)

    log("step 3a: reproduction check vs hash-verified E1 log (C4 precondition 1)")
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
            identical = bool(np.array_equal(rebuilt.astype(np.int64), logged.astype(np.int64)))
            if not identical:
                d = int(np.abs(rebuilt.astype(np.int64) - logged.astype(np.int64)).max())
                max_diff = max(max_diff, d)
                mismatches.append(f"E1:{rset}/{split}")
                log(f"  E1 {rset}/{split}: MISMATCH maxdiff={d}")
    if mismatches:
        raise SystemExit(f"STOP: E1 reproduction check FAILED for {mismatches}; nothing E3 logged")
    log("  E1 reproduction check PASSED: all six routing sets bit-exact")

    log("step 3b: reproduction check vs hash-verified E2 log (C4 precondition 2)")
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        pi = np.random.default_rng(SHUFFLED_Q_SEED).permutation(n_items)
        q_pops = [population_union(projector, streams[split]["q"][i]) for i in range(n_items)]
        q_pops_shuf = [q_pops[int(pi[i])] for i in range(n_items)]
        for k in E2_K_SWEEP:
            sparsified = topk_row_sparsify(accumulator.support(), accumulator.values(), k)
            mat_k = HomeostaticProjection(config, accumulator.support(), sparsified)
            for label, pops in ((f"arm_k{k}", q_pops), (f"shufq_k{k}", q_pops_shuf)):
                rebuilt = np.stack([mat_k.route(pops[i], K_WINNERS) for i in range(n_items)])
                logged = logged_winners(e2_z, label, split, n_items)
                if not np.array_equal(rebuilt.astype(np.int64), logged.astype(np.int64)):
                    mismatches.append(f"E2:{label}/{split}")
        rebuilt_rand = np.stack([randW.route(q_pops[i], K_WINNERS) for i in range(n_items)])
        logged_rand = logged_winners(e2_z, "randomized_W", split, n_items)
        if not np.array_equal(rebuilt_rand.astype(np.int64), logged_rand.astype(np.int64)):
            mismatches.append(f"E2:randomized_W/{split}")
    if mismatches:
        raise SystemExit(f"STOP: reproduction checks FAILED for {mismatches}; nothing E3 logged")
    log("  E2 reproduction check PASSED: all seven logged sets bit-exact on both splits")

    log("step 4: build gated accumulators, route + log E3 configs (one-shot)")
    t0 = time.perf_counter()
    acc_q = SparseProjection(config)
    acc_q.apply_dense_delta(dense_delta_from_pairs(gated_counts, projector))
    acc_r = SparseProjection(config)
    acc_r.apply_dense_delta(dense_delta_from_pairs(rspan_counts, projector))

    def mk_mat(acc: SparseProjection) -> HomeostaticProjection:
        return HomeostaticProjection(
            config, acc.support(), topk_row_sparsify(acc.support(), acc.values(), K_FIXED)
        )

    mat_q16 = mk_mat(acc_q)
    mat_r16 = mk_mat(acc_r)

    sink = WinnerSink()
    for split in ("train", "val"):
        n_items = len(streams[split]["q"])
        pi = np.random.default_rng(SHUFFLED_Q_SEED).permutation(n_items)
        q_pops = [population_union(projector, streams[split]["q"][i]) for i in range(n_items)]
        q_pops_shuf = [q_pops[int(pi[i])] for i in range(n_items)]
        for i in range(n_items):
            sink.record("arm_q16", split, i, mat_q16.route(q_pops[i], K_WINNERS))
            sink.record("shufq_q16", split, i, mat_q16.route(q_pops_shuf[i], K_WINNERS))
            sink.record("rs16", split, i, mat_r16.route(q_pops[i], K_WINNERS))
            sink.record("shufq_rs16", split, i, mat_r16.route(q_pops_shuf[i], K_WINNERS))
        log(f"  {split}: routed 4 new sets x {n_items}")

    arrays = sink.arrays()
    if NPZ_PATH.exists():
        raise SystemExit(f"STOP: {NPZ_PATH.name} already exists (one-shot discipline)")
    np.savez_compressed(NPZ_PATH, **arrays)

    array_hashes = {
        name: {"sha256": sha256_bytes(arrays[name].tobytes()), "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }
    gold_echo = {
        name: {"sha256": sha256_bytes(e1_z[name].tobytes()), "dtype": str(e1_z[name].dtype)}
        for name in sorted(e1_z)
        if name.startswith(("gold_", "matched_", "families_", "sigma_", "shuffle_perm_", "qa_ids_"))
    }
    manifest = {
        "schema_version": "humanbrain/t-route-e3-winner-sets/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "amendment": "experiments/t_route/e3/CHANGE_NOTE_4.md (committed pre-spend)",
        "npz_file": NPZ_PATH.name,
        "npz_sha256": sha256_bytes(NPZ_PATH.read_bytes()),
        "array_sha256": array_hashes,
        "reproduction_check": {
            "e1_npz_file": "../" + E1_NPZ_PATH.name,
            "e1_npz_sha256": sha256_bytes(E1_NPZ_PATH.read_bytes()),
            "e2_npz_file": "../e2/" + E2_NPZ_PATH.name,
            "e2_npz_sha256": sha256_bytes(E2_NPZ_PATH.read_bytes()),
            "all_six_e1_routing_sets_bit_exact": True,
            "all_seven_e2_logged_sets_bit_exact": True,
            "ungated_counter_equals_e1_e2_aggregate": True,
            "gold_array_echo_from_e1_log": gold_echo,
        },
        "integrity": integrity_checks,
        "config": {
            "n_bits": N_BITS, "w_ones": W_ONES, "projector_seed": PROJECTOR_SEED,
            "fan_in": FAN_IN, "w_seed": W_SEED, "k_winners": K_WINNERS,
            "randomized_w_seed": RAND_W_SEED, "shuffle_w_seed": SHUFFLE_W_SEED,
            "shuffled_question_seed": SHUFFLED_Q_SEED,
            "h_win": H_WIN, "o_max": O_MAX, "lam_P": LAM_P,
            "gate": "dW_ij *= 1[token_j in quote_span(episode)]; quote = text between first/last '\"'; union of ALL byte-subsequence matches; empty-mask episodes contribute zero plasticity",
            "transform": f"per-row top-k by value k={K_FIXED}, ties ascending column index; NO homeostat, NO normalization",
            "k_fixed": K_FIXED,
            "random_span_seed": RANDOM_SPAN_SEED,
        },
        "gating_census": {
            **census,
            "ungated_event_total": su,
            "gated_event_total_quote": sg,
            "gated_event_total_random": sr,
            "ungated_unique_pairs": len(pair_counts),
            "gated_unique_pairs_quote": len(gated_counts),
            "gated_unique_pairs_random": len(rspan_counts),
            "random_span_draw_echo_sha256": rs_draw_echo.hexdigest(),
            "val_episodes_touched_w": 0,
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "note": "Binary npz stays local; array-level sha256 committed here.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"E3 logging complete: {NPZ_PATH.name} ({NPZ_PATH.stat().st_size} bytes), {MANIFEST_PATH.name}")
    log(f"gating/routing elapsed {time.perf_counter()-t0:.1f}s; total {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
