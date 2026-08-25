#!/usr/bin/env python3
"""T-ROUTE E4 repetition-regime logging harness (Change Note 5, PRE-SPEND committed).

Executes the preregistered path-3 probe (path-decision-synthesis.md @ e16d22c §Path 3,
DD-ASP §Cheap falsification) EXACTLY once, per experiments/t_route/e4_repetition/
CHANGE_NOTE_5.md:

  step 0  integrity: episodes_train.jsonl raw-bytes AND LF-normalized sha256 pins +
          pair-repetition census (1 pair >=10 reps / 27 eps, 1593 singletons,
          17 mid-band pairs / 36 eps) — any mismatch => STOP;
  step 1  tokenizer fit (E2 recipe on this corpus) + question/passage streams;
  step 2  encode five arms through the SAME 1,656-round stream (k-cap active during
          encoding): mult_b{0.05,0.25,0.5} multiplicative (1+BETA)^m winner-gated,
          add_gated additive winner-gated, add_free batch-additive ungated (E2 style);
  step 3  route + log base_<arm>, shufq_<arm> (pi seed 11) and randomized_W (seed 13)
          into e4_logged_winner_sets.npz (+ e4_manifest.json array hashes).

Gold/matched/stratum arrays are logged alongside so score_e4.py reads logs only.
Pure numpy, CPU-only, deterministic. No reruns: spent by first successful completion.
"""

from __future__ import annotations

import hashlib
import json
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
    SparseProjection,
    kwinner_indices,
    population_union,
)
from humanbrain.sdr_union import SDRProjector  # noqa: E402
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig  # noqa: E402

# ------------------------------------------------------------------ data pins (CN5 §2)
EPISODES_PATH = (
    REPO / "experiments" / "t_route" / "synthetic_episodes" / "episodes_train.jsonl"
)
PIN_EPISODES_LF_SHA = "a94a8c22cedace1b533351b3a811829b68f62a374daaceff683b052b1b359e5a"
PIN_EPISODES_RAW_SHA = "fb759b86d7c796967e4bdb74558f835ffc92af3762add434ba68d4f264b6b293"
EXPECTED_EPISODES = 1656
# Frozen census (Change Note 5 §2): measured before freeze; asserted here so any data
# drift STOPs the harness instead of silently re-stratifying.
PIN_PAIRS_GE10, PIN_EPISODES_GE10 = 1, 27
PIN_PAIRS_EQ1, PIN_EPISODES_EQ1 = 1593, 1593
PIN_PAIRS_MID, PIN_EPISODES_MID = 17, 36

# ------------------------------------------------------------------ config pins (CN5 §3)
N_BITS, W_ONES, PROJECTOR_SEED = 2048, 32, 7
FAN_IN, W_SEED, K_WINNERS = 32, 7, 128  # K_WINNERS is the k-cap, encode AND retrieve
H_WIN, O_MAX, ADJ = 12, 8, 2
RAND_W_SEED, SHUFFLED_Q_SEED = 13, 11
BETAS = (0.05, 0.25, 0.5)
CAP_EXP = {0.05: 11790, 0.25: 2577, 0.5: 1418}  # floor(log_{1+b}(1e250/1.5)), CN5 §3
CLIP_WEIGHT_CEILING = 1e250
MULT_ARMS = tuple(f"mult_b{b}" for b in BETAS)
ARMS = MULT_ARMS + ("add_gated", "add_free")
STRATA = ("REPEATED", "SINGLE", "MID")

E4_DIR = Path(__file__).resolve().parent
NPZ_PATH = E4_DIR / "e4_logged_winner_sets.npz"
MANIFEST_PATH = E4_DIR / "e4_manifest.json"


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


# ------------------------------------------------------------------ step 0
def load_episodes() -> tuple[list[dict], dict]:
    """Load + verify pinned episodes; returns (episodes, integrity_record)."""
    log("step 0: integrity pins")
    raw = EPISODES_PATH.read_bytes()
    digest_raw = sha256_bytes(raw)
    norm = raw.replace(b"\r\n", b"\n")
    digest_lf = sha256_bytes(norm)
    if digest_raw != PIN_EPISODES_RAW_SHA:
        raise SystemExit("STOP: episodes raw-bytes sha256 mismatch")
    if digest_lf != PIN_EPISODES_LF_SHA:
        raise SystemExit("STOP: episodes LF-normalized sha256 mismatch")

    episodes = [json.loads(l) for l in norm.decode("utf-8").splitlines() if l.strip()]
    if len(episodes) != EXPECTED_EPISODES:
        raise SystemExit(f"STOP: episode count {len(episodes)} != {EXPECTED_EPISODES}")

    counts = Counter((e["meta"]["quote"], e["answer"]) for e in episodes)
    census = {
        "unique_pairs": len(counts),
        "pairs_ge10": sum(1 for c in counts.values() if c >= 10),
        "episodes_ge10": sum(c for c in counts.values() if c >= 10),
        "pairs_eq1": sum(1 for c in counts.values() if c == 1),
        "episodes_eq1": sum(c for c in counts.values() if c == 1),
        "pairs_mid": sum(1 for c in counts.values() if 2 <= c <= 9),
        "episodes_mid": sum(c for c in counts.values() if 2 <= c <= 9),
    }
    expect = {
        "unique_pairs": PIN_PAIRS_GE10 + PIN_PAIRS_EQ1 + PIN_PAIRS_MID,
        "pairs_ge10": PIN_PAIRS_GE10, "episodes_ge10": PIN_EPISODES_GE10,
        "pairs_eq1": PIN_PAIRS_EQ1, "episodes_eq1": PIN_EPISODES_EQ1,
        "pairs_mid": PIN_PAIRS_MID, "episodes_mid": PIN_EPISODES_MID,
    }
    if census != expect:
        raise SystemExit(f"STOP: pair census drifted {census} != {expect}")
    log(f"  integrity OK; census={census}")
    integrity_record = {
        "episodes_raw_sha256": digest_raw,
        "episodes_lf_canonical_sha256": digest_lf,
        "episode_count": len(episodes),
        "pair_census": expect,
    }
    return episodes, integrity_record


def strata_of(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stratum labels + per-item pair counts (CN5 §2)."""
    counts = Counter((e["meta"]["quote"], e["answer"]) for e in episodes)
    pc = np.array([counts[(e["meta"]["quote"], e["answer"])] for e in episodes], dtype=np.int64)
    lab = np.where(pc >= 10, "REPEATED", np.where(pc == 1, "SINGLE", "MID"))
    return lab.astype("U8"), pc


# ------------------------------------------------------------------ pairing (E1/E2 recipe)
def episode_pair_counts(q_ids: list[int], p_ids: list[int]) -> dict[tuple[int, int], int]:
    """Token-pair multiplicities within ONE episode (prereg §2 rule)."""
    occurrences: dict[int, list[int]] = {}
    for j, t in enumerate(p_ids):
        occurrences.setdefault(t, []).append(j)
    counts: Counter = Counter()
    for tq in dict.fromkeys(q_ids):
        anchors = occurrences.get(tq)
        if not anchors:
            continue
        for j in anchors[:O_MAX]:
            lo, hi = max(0, j - H_WIN), min(len(p_ids), j + H_WIN + 1)
            for jp in range(lo, hi):
                counts[(tq, p_ids[jp])] += 1
    return dict(counts)


class GateEncoder:
    """Shared k-cap-gated sequential encoder (CN5 §3 steps 1-4).

    Holds the fixed fan-in support, its init values, and a (post, col) LUT. One
    `round()` call applies ONE episode: drive -> k-cap winners -> gated update.
    """

    def __init__(self, config: HebbianConfig, projector: SDRProjector):
        base = SparseProjection(config)  # delta == 0 here, so values() == init
        self.config = config
        self._support = base.support()
        self._init = base.values()
        self._lut = np.full((config.n_bits, config.n_bits), -1, dtype=np.int64)
        rows = np.repeat(np.arange(config.n_bits, dtype=np.int64), config.fan_in)
        self._lut[rows, self._support.ravel()] = np.tile(
            np.arange(config.fan_in, dtype=np.int64), config.n_bits
        )
        self.projector = projector
        self.w_ones = int(projector.w_ones)

    def _expand(self, tok_counts: dict[tuple[int, int], int]):
        triplets = [(tq, tp, m) for (tq, tp), m in tok_counts.items()]
        if not triplets:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        w = self.w_ones
        cols = np.concatenate([np.tile(self.projector.indices(tq), w) for tq, _, _ in triplets])
        rows = np.concatenate([np.repeat(self.projector.indices(tp), w) for _, tp, _ in triplets])
        mult = np.repeat(np.array([m for _, _, m in triplets], dtype=np.int64), w * w)
        return rows, cols, mult

    def round_update(
        self,
        q_pop: np.ndarray,
        tok_counts: dict[tuple[int, int], int],
        values_now: np.ndarray,
        apply_fn,
    ) -> None:
        """drive -> winners -> apply_fn(post_rows, fan_in_cols, mults)."""
        gathered = q_pop[self._support]
        drive = np.zeros(self.config.n_bits, dtype=np.float64)
        np.einsum("ij,ij->i", gathered, values_now, out=drive)
        winners = kwinner_indices(drive, self.config.k_winners)
        win_mask = np.zeros(self.config.n_bits, dtype=bool)
        win_mask[winners] = True
        rows, cols, mult = self._expand(tok_counts)
        lut_cols = self._lut[rows, cols]
        valid = win_mask[rows] & (lut_cols >= 0) & (mult > 0)
        if not valid.any():
            return
        rows, lut_cols, mult = rows[valid], lut_cols[valid], mult[valid]
        key = rows * self.config.n_bits + lut_cols
        uniq, inv = np.unique(key, return_inverse=True)
        agg = np.zeros(uniq.size, dtype=np.int64)
        np.add.at(agg, inv, mult)
        apply_fn((uniq // self.config.n_bits).astype(np.int64),
                 (uniq % self.config.n_bits).astype(np.int64), agg)


class MultProjection:
    """Multiplicative arm: w = init * (1+beta)^min(C, CAP_EXP) (CN5 §3)."""

    def __init__(self, enc: GateEncoder, beta: float):
        self.enc = enc
        self.beta = float(beta)
        self.cap_exp = CAP_EXP[beta]
        self.counts = np.zeros_like(enc._init, dtype=np.int64)

    def values(self) -> np.ndarray:
        capped = np.minimum(self.counts, self.cap_exp)
        return self.enc._init * np.power(1.0 + self.beta, capped.astype(np.float64))

    def round_update(self, q_pop, tok_counts) -> None:
        self.enc.round_update(
            q_pop, tok_counts, self.values(),
            lambda r, c, m: self.counts.__setitem__((r, c), self.counts[r, c] + m),
        )

    def route(self, q_pop: np.ndarray, k: int) -> np.ndarray:
        gathered = q_pop[self.enc._support]
        drive = np.zeros(self.enc.config.n_bits, dtype=np.float64)
        np.einsum("ij,ij->i", gathered, self.values(), out=drive)
        return kwinner_indices(drive, k)


class AddGatedProjection:
    """Additive arm with the SAME k-cap gating: w = init + delta (gated)."""

    def __init__(self, enc: GateEncoder):
        self.enc = enc
        self.delta = np.zeros_like(enc._init)

    def values(self) -> np.ndarray:
        return self.enc._init + self.delta

    def round_update(self, q_pop, tok_counts) -> None:
        self.enc.round_update(
            q_pop, tok_counts, self.values(),
            lambda r, c, m: self.delta.__setitem__((r, c), self.delta[r, c] + m),
        )

    def route(self, q_pop: np.ndarray, k: int) -> np.ndarray:
        gathered = q_pop[self.enc._support]
        drive = np.zeros(self.enc.config.n_bits, dtype=np.float64)
        np.einsum("ij,ij->i", gathered, self.values(), out=drive)
        return kwinner_indices(drive, k)


def dense_delta_from_pairs(counts: Counter, projector: SDRProjector) -> np.ndarray:
    """Aggregated dW blocks for the add_free arm (E1/E2 recipe verbatim)."""
    n = int(projector.n_bits)
    dense = np.zeros((n, n), dtype=np.float64)
    triplets = [(int(tq), int(tp), int(c)) for (tq, tp), c in counts.items()]
    w = int(projector.w_ones)
    rows = np.concatenate([np.repeat(projector.indices(tp), w) for _, tp, _ in triplets])
    cols = np.concatenate([np.tile(projector.indices(tq), w) for tq, _, _ in triplets])
    vals = np.repeat(np.array([c for _, _, c in triplets], dtype=np.float64), w * w)
    np.add.at(dense, (rows, cols), vals)
    return dense


# ------------------------------------------------------------------ gold regions (E1)
def contiguous_match_regions(a_seq: list[int], p_seq: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    la, lp = len(a_seq), len(p_seq)
    if la == 0 or lp < la:
        return regions
    for j in range(lp - la + 1):
        if p_seq[j : j + la] == a_seq:
            regions.append((max(0, j - ADJ), min(lp, j + la + ADJ)))
    return regions


class WinnerSink:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int]] = []
        self.winners: list[np.ndarray] = []

    def record(self, routing_set: str, index: int, winners: np.ndarray) -> None:
        self.rows.append((routing_set, int(index)))
        self.winners.append(np.asarray(winners, dtype=np.int16))

    def arrays(self) -> dict[str, np.ndarray]:
        stacked = np.stack(self.winners)
        sets = np.array([r[0] for r in self.rows], dtype="U20")
        idx = np.array([r[1] for r in self.rows], dtype=np.int64)
        return {"winner_sets": stacked, "routing_set": sets, "item_index": idx}


def routed_matrix(z: dict, rset: str, n_items: int) -> np.ndarray:
    sel = z["routing_set"] == rset
    idx = z["item_index"][sel]
    order = np.argsort(idx)
    assert len(idx) == n_items and np.array_equal(idx[order], np.arange(n_items))
    return z["winner_sets"][sel][order].astype(np.int64)


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.perf_counter()
    if NPZ_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit("STOP: E4 outputs already exist (one-shot discipline)")
    episodes, integrity_record = load_episodes()
    strata, pair_counts = strata_of(episodes)

    log("step 1: tokenizer fit + streams (E2 recipe on pinned episodes)")
    tokenizer = BytePhraseTokenizer.fit(
        [e["serialized_prompt_answer"] for e in episodes],
        TokenizerConfig(max_phrases=0, min_frequency=3),
    )
    if tokenizer.vocab_size != 256:
        raise SystemExit(f"STOP: tokenizer.vocab_size {tokenizer.vocab_size} != 256")
    q_streams = [tokenizer.encode(e["question"]) for e in episodes]
    p_streams = [tokenizer.encode("\n".join(e["context"])) for e in episodes]
    a_streams = [tokenizer.encode(e["answer"]) for e in episodes]
    projector = SDRProjector(n_bits=N_BITS, w_ones=W_ONES, seed=PROJECTOR_SEED)
    q_pops = [population_union(projector, qs) for qs in q_streams]
    tok_counts = [episode_pair_counts(qs, ps) for qs, ps in zip(q_streams, p_streams)]
    n_events = int(sum(sum(c.values()) for c in tok_counts))
    log(f"  vocab=256; within-episode token-pair events total={n_events}")

    log("step 2: encoding 1,656 rounds x 5 arms (k-cap active during encoding)")
    config = HebbianConfig(
        n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN, w_seed=W_SEED, k_winners=K_WINNERS
    )
    enc = GateEncoder(config, projector)
    arms: dict[str, object] = {}
    for b in BETAS:
        arms[f"mult_b{b}"] = MultProjection(enc, b)
    arms["add_gated"] = AddGatedProjection(enc)

    diag_clip, t0 = {f"mult_b{b}": 0 for b in BETAS}, time.perf_counter()
    for i in range(len(episodes)):
        qc = tok_counts[i]
        for name in MULT_ARMS:
            arms[name].round_update(q_pops[i], qc)
        arms["add_gated"].round_update(q_pops[i], qc)
        for name in MULT_ARMS:
            cl = int((arms[name].counts >= arms[name].cap_exp).sum())
            diag_clip[name] = max(diag_clip[name], cl)
        if (i + 1) % 400 == 0:
            log(f"  round {i+1}/{len(episodes)} ({time.perf_counter()-t0:.0f}s)")

    add_free = SparseProjection(config)
    global_counts: Counter = Counter()
    for c in tok_counts:
        global_counts.update(c)
    add_free.apply_dense_delta(dense_delta_from_pairs(global_counts, projector))
    arms["add_free"] = add_free
    log("  add_free (batch additive, ungated) accumulated")

    log("step 3: gold regions")
    n_items = len(episodes)
    gold_masks = np.zeros((n_items, N_BITS), dtype=bool)
    hits = np.zeros(n_items, dtype=np.int64)
    for i in range(n_items):
        regions = contiguous_match_regions(a_streams[i], p_streams[i])
        hits[i] = len(regions)
        for lo, hi in regions:
            for j in range(lo, hi):
                gold_masks[i][projector.indices(p_streams[i][j])] = True
    matched = hits > 0
    log(f"  GOLD_UNMATCHED={int((~matched).sum())}/{n_items}")

    log("step 4: routing all sets through the winner sink")
    sink = WinnerSink()
    pi = np.random.default_rng(SHUFFLED_Q_SEED).permutation(n_items)
    randW = SparseProjection(
        HebbianConfig(n_bits=N_BITS, w_ones=W_ONES, fan_in=FAN_IN,
                      w_seed=RAND_W_SEED, k_winners=K_WINNERS)
    )
    for i in range(n_items):
        sink.record("randomized_W", i, randW.route(q_pops[i], K_WINNERS))
    for name in ARMS:
        mat = arms[name]
        for i in range(n_items):
            sink.record(f"base_{name}", i, mat.route(q_pops[i], K_WINNERS))
            sink.record(f"shufq_{name}", i, mat.route(q_pops[int(pi[i])], K_WINNERS))
        log(f"  {name}: routed base+shufq")

    arrays = sink.arrays()
    arrays["gold_mask"] = gold_masks
    arrays["gold_sizes"] = gold_masks.sum(axis=1).astype(np.int64)
    arrays["matched"] = matched
    arrays["stratum"] = strata
    arrays["pair_count"] = pair_counts
    arrays["shuffle_perm"] = pi.astype(np.int64)
    arrays["episode_ids"] = np.array([e["episode_id"] for e in episodes], dtype="U24")
    arrays["answers"] = np.array([e["answer"] for e in episodes], dtype="U24")

    np.savez_compressed(NPZ_PATH, **arrays)
    array_hashes = {
        name: {"sha256": sha256_bytes(arrays[name].tobytes()),
               "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }
    manifest = {
        "schema_version": "humanbrain/t-route-e4-winner-sets/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "amendment": "experiments/t_route/e4_repetition/CHANGE_NOTE_5.md (committed pre-spend)",
        "npz_file": NPZ_PATH.name,
        "npz_sha256": sha256_bytes(NPZ_PATH.read_bytes()),
        "array_sha256": array_hashes,
        "integrity": integrity_record,
        "config": {
            "n_bits": N_BITS, "w_ones": W_ONES, "projector_seed": PROJECTOR_SEED,
            "fan_in": FAN_IN, "w_seed": W_SEED, "k_winners": K_WINNERS,
            "h_win": H_WIN, "o_max": O_MAX, "adj": ADJ,
            "randomized_w_seed": RAND_W_SEED, "shuffled_question_seed": SHUFFLED_Q_SEED,
            "betas": list(BETAS), "cap_exp": {str(b): CAP_EXP[b] for b in BETAS},
            "arms": list(ARMS),
            "encoding": "1656 sequential rounds, file order; k-cap top-128 winners "
                        "per round; potentiation gated to winner post bits on fixed "
                        "fan-in-32 support; multiplicative (1+b)^m via integer fire "
                        "counts, additive arms as noted",
        },
        "within_episode_token_pair_events_total": n_events,
        "max_synapses_at_cap_by_arm": diag_clip,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "note": "Binary npz stays local; array-level sha256 committed here.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"E4 logging complete: {NPZ_PATH.name} ({NPZ_PATH.stat().st_size} bytes)")
    log(f"total elapsed {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
