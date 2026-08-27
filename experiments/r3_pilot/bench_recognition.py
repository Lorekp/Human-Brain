"""Historical bench + bit-exactness rig for a Python tracker prototype.

This prototype is not eligible for the physical R3/C1 path.  That path uses
the native ``TemporalStore.recognize_runtime_context`` contact-cell index.
These measurements close an abandoned handoff artifact; they are evidence
only about ``SegmentGrowthTracker`` and must not be used for a C1 cap decision.

Addresses C1_READINESS.md Blocker 2 (mandatory pre-launch decision (b),
@ 10453e0): the reference ``SegmentGrowthTracker.step`` recognition pass is a
GLOBAL scan -- every live segment is visited each step and tested with
``np.isin(presyn, W)`` against the arriving winner set, costing ~25 ms/step at
1k live segments and ~278 ms/step at 8k (measured, N=131072, |W|=54).  The
program already proved the cure once: Seam-B replaced procedural.rs linear
membership scans with a gid-keyed ``BTreeMap<u64, u32>`` index, bit-exactly
(docs/research/cpu-seams-spec.md section B; 38x/23x EP/FT).

This file prototypes the same pattern for the tracker WITHOUT touching
``segment_growth.py``:

1. ``IndexedSegmentGrowthTracker`` -- drop-in subclass maintaining the
   inverted index ``presynaptic cell id -> {packed segment keys}``.  A segment
   can only reach ``theta_a = 13`` connected-active synapses if >= theta_a of
   its contacts lie in ``W_{t-1}``, so the per-step candidate set is exactly
   the segments sharing >= theta_a contacts with W -- bounded by
   ``sum_{c in W} deg(c)``, NOT by total live segments.
2. Bit-exactness rig -- identical step sequences against the reference
   tracker with per-step ``StepResult`` equality, final deep-state equality,
   and an every-step shadow-rebuild check of the maintained index.
3. Profile decomposition -- what inside ``step`` is O(segments).
4. A/B wall-clock on the requested {1k, 2k, 4k, 8k, 16k, 32k} grid under
   three contact distributions.  The historical closure run is intentionally
   capped at 8k; 16k/32k belong to the native physical-path benchmark.

Run:  C:/Users/laure/Desktop/DeepSim/.venv/Scripts/python.exe \
          experiments/r3_pilot/bench_recognition.py [--quick] [--json PATH]

Determinism: seeded; same invocation => identical tables and verdicts.
This bench is measurement-only: no prereg constants, runner, or tracker source
is modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for _p in (str(SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from humanbrain.segment_growth import (  # noqa: E402
    SegmentGrowthTracker,
    StepResult,
    _Segment,
)

# ---------------------------------------------------------------------------
# Indexed prototype (Seam-B pattern applied to the recognition pass)
# ---------------------------------------------------------------------------


class IndexedSegmentGrowthTracker(SegmentGrowthTracker):
    """Bit-exact drop-in replacement: indexed candidate narrowing.

    Maintains the Seam-B-style inverted index

        ``self._inv : Dict[int, Set[int]]``
            presynaptic lane cell id -> set of packed segment keys

    where ``key = (cell << KEY_BITS) | segment_index``.  Packing into one int
    makes key sort order IDENTICAL to the reference scan's visitation order
    (ascending cell id, then ascending segment index), which is what makes the
    hits/misses ordering -- and therefore every StepResult field --
    element-wise identical (see plan doc, Lemma 2).

    Index maintenance covers every site that mutates ``seg.presyn``:
    miss-path purge, hit-path reinforce+purge, election-time learn+purge/fill,
    and burst allocation + fill.  Nothing else writes presyn.
    """

    KEY_BITS = 16  # MAX_SEGMENTS_PER_CELL = 255 < 2**16; cell ids < 2**40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inv: Dict[int, Set[int]] = {}
        super().__init__(*args, **kwargs)
        self.index_rebuilds = 0

    # -- key packing -------------------------------------------------------

    @staticmethod
    def pack(cell: int, si: int) -> int:
        return (int(cell) << IndexedSegmentGrowthTracker.KEY_BITS) | int(si)

    @staticmethod
    def unpack(key: int) -> Tuple[int, int]:
        return (
            key >> IndexedSegmentGrowthTracker.KEY_BITS,
            key & ((1 << IndexedSegmentGrowthTracker.KEY_BITS) - 1),
        )

    # -- index maintenance -------------------------------------------------

    def _add_key(self, ids: Iterable[int], key: int) -> None:
        inv = self._inv
        for c in ids:
            bucket = inv.get(c)
            if bucket is None:
                inv[c] = {key}
            else:
                bucket.add(key)

    def _drop_key(self, ids: Iterable[int], key: int) -> None:
        inv = self._inv
        for c in ids:
            bucket = inv.get(c)
            if bucket is not None:  # tolerate repeated ids defensively
                bucket.discard(key)
                if not bucket:
                    del inv[c]

    def _sync(self, cell: int, si: int, old_presyn: List[int], new_presyn: List[int]) -> None:
        """Reconcile the index after presyn changed from old to new."""
        if old_presyn is new_presyn:
            return
        old_set = set(old_presyn)
        new_set = set(new_presyn)
        key = self.pack(cell, si)
        removed = old_set - new_set
        added = new_set - old_set
        if removed:
            self._drop_key(removed, key)
        if added:
            self._add_key(added, key)

    def rebuild_index(self) -> None:
        """Full rebuild from live state (used by injection + shadow checks)."""
        inv: Dict[int, Set[int]] = {}
        pack = self.pack
        for cell, segs in self._segments.items():
            for si, seg in enumerate(segs):
                key = pack(cell, si)
                for c in seg.presyn:
                    bucket = inv.get(c)
                    if bucket is None:
                        inv[c] = {key}
                    else:
                        bucket.add(key)
        self._inv = inv
        self.index_rebuilds += 1

    def verify_index(self) -> bool:
        """Shadow rebuild == maintained index? (verification mode only)"""
        shadow: Dict[int, Set[int]] = {}
        pack = self.pack
        for cell, segs in self._segments.items():
            for si, seg in enumerate(segs):
                key = pack(cell, si)
                for c in seg.presyn:
                    shadow.setdefault(c, set()).add(key)
        return shadow == self._inv

    # -- overridden mutation hooks ------------------------------------------

    def _allocate_segment(self, cell: int, w: np.ndarray) -> None:
        super()._allocate_segment(cell, w)
        segs = self._segments[cell]
        si = len(segs) - 1  # fill_toward ran inside super(); register final contacts
        self._add_key(segs[si].presyn, self.pack(cell, si))

    def _learn_on_segment(self, cell: int, segment_index: int, w: np.ndarray) -> None:
        old = list(self._segments[cell][segment_index].presyn)
        super()._learn_on_segment(cell, segment_index, w)
        self._sync(
            cell,
            segment_index,
            old,
            self._segments[cell][segment_index].presyn,
        )

    # -- the replacement recognition pass ------------------------------------

    def step(self, pool_id: Any, winner_set: Sequence[int]) -> StepResult:
        """Identical mechanics to ``SegmentGrowthTracker.step`` except that the
        global scan is replaced by inverted-index candidate generation.

        Contract preserved verbatim: same w normalization, same hit/miss
        predicate expressions on the same arrays, same visitation ORDER, same
        punish/confirm loops, same election call, same StepResult fields.
        """
        if pool_id not in self._pool_cells:
            raise ValueError(f"unknown pool {pool_id!r}")
        pool_cells = self._pool_cells[pool_id]
        pool_set = frozenset(pool_cells)
        w = np.unique(np.asarray([int(x) for x in winner_set], dtype=np.int64))

        # --- INDEXED recognition pass (replaces: for cell in sorted(...) ---
        # A segment can only reach theta_a connected-active synapses if at
        # least theta_a of its contacts lie in W (connected-active implies
        # active implies presyn in W).  counts[key] accumulates, for every
        # segment sharing >= 1 contact with W, its exact |presyn ^ W|; only
        # keys with counts >= theta_a can possibly classify.
        counts: Dict[int, int] = {}
        inv_get = self._inv.get
        for c in w.tolist():
            bucket = inv_get(c)
            if bucket:
                for key in bucket:
                    counts[key] = counts.get(key, 0) + 1
        theta_a = self._theta_active
        candidates = [k for k, n in counts.items() if n >= theta_a]
        candidates.sort()  # == reference visitation order (Lemma 2)

        hits: List[Tuple[int, int, np.ndarray]] = []
        misses: List[Tuple[int, int, np.ndarray]] = []
        for key in candidates:
            cell, si = self.unpack(key)
            seg = self._segments[cell][si]
            presyn = np.asarray(seg.presyn, dtype=np.int64)
            perm = np.asarray(seg.perm, dtype=np.float64)
            active = np.isin(presyn, w)
            then_connected_active = active & (perm >= self._connected_permanence)
            if int(then_connected_active.sum()) >= theta_a:
                idx = np.flatnonzero(then_connected_active)
                (hits if cell in pool_set else misses).append((cell, si, idx))

        removed_total = 0
        punished: List[Tuple[int, int]] = []
        for cell, si, idx in misses:  # unconfirmed converged predictors
            seg = self._segments[cell][si]
            old_presyn = seg.presyn
            perm = np.asarray(seg.perm, dtype=np.float64)
            perm[idx] -= self.DELTA_NEGATIVE
            presyn_arr = np.asarray(seg.presyn, dtype=np.int64)
            seg.presyn, seg.perm, removed = self._purge(presyn_arr, perm)
            removed_total += removed
            punished.append((cell, si))
            self._sync(cell, si, old_presyn, seg.presyn)
        for cell, si, _idx in hits:  # predicted-path confirm
            seg = self._segments[cell][si]
            old_presyn = seg.presyn
            presyn_arr = np.asarray(seg.presyn, dtype=np.int64)
            perm = np.asarray(seg.perm, dtype=np.float64)
            active = np.isin(presyn_arr, w)
            perm[active] += self._permanence_increment
            perm[~active] -= self._permanence_increment
            np.clip(perm, 0.0, 1.0, out=perm)
            seg.presyn, seg.perm, removed = self._purge(presyn_arr, perm)
            removed_total += removed
            self._sync(cell, si, old_presyn, seg.presyn)

        burst = not hits
        elected: int | None = None
        allocated = False
        if burst:
            elected, allocated = self._elect(pool_cells, w)
        self._step_index += 1

        return StepResult(
            pool_id=pool_id,
            step_index=self._step_index - 1,
            burst=burst,
            elected_winner=elected,
            fired_cells=tuple(cell for cell, _, _ in hits),
            punished_segments=tuple(punished),
            allocated_segment=allocated,
            removed_contacts=removed_total,
        )


# ---------------------------------------------------------------------------
# Synthetic-but-faithful segment tables + context streams
# ---------------------------------------------------------------------------

LANE_UNIVERSE = 131_072  # lane gid space (C1 lane size); pools sit above it
POOL_K = 32  # frozen cellsPerPool
N_POOLS = 512  # arriving token identities for the election path
W_SIZE = 54  # measured mean lane winner-set size (C1_READINESS basis)


def zipf_weights(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    wts = ranks ** (-alpha)
    return wts / wts.sum()


class ContextModel:
    """Template-based context streams mimicking bigram replay structure.

    'zipf'    -- recurring contexts drawn from a Zipf-weighted vocabulary;
                 realistic concentration of contacts on hot context cells.
    'uniform' -- contexts uniform over the whole lane universe; near-zero
                 overlap between any two draws (index best case).
    'hot'     -- stress bound: one hub cell belongs to EVERY segment and to
                 every W; index fanout degrades to O(total segments) but with
                 dict-increment constants instead of numpy-call constants.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        regime: str,
        *,
        vocab: int = 8_192,
        n_templates: int = 512,
        alpha: float = 1.2,
        reuse_p: float = 0.7,
        max_mutation: int = 5,
    ) -> None:
        self.rng = rng
        self.regime = regime
        self.vocab = vocab
        self.reuse_p = float(reuse_p)
        self.max_mutation = int(max_mutation)
        vocab_probs = zipf_weights(vocab, alpha) if regime in ("zipf", "hot") else None
        self.vocab_probs = vocab_probs
        t_probs = zipf_weights(n_templates, 1.1) if regime != "uniform" else None
        self.template_probs = t_probs
        self.hot_cell = 0 if regime == "hot" else None
        self.templates: List[np.ndarray] = []
        for _ in range(n_templates):
            if regime == "uniform":
                cells = rng.choice(LANE_UNIVERSE, size=W_SIZE, replace=False)
            else:
                cells = rng.choice(vocab, size=W_SIZE, replace=False, p=vocab_probs)
                if regime == "hot":
                    cells = np.unique(np.concatenate(([self.hot_cell], cells)))[:W_SIZE]
            self.templates.append(np.sort(cells))

    def draw_w(self) -> List[int]:
        """One step's winner set W_{t-1}: mostly recurring templates with small
        mutations, sometimes a fresh draw (novel context)."""
        rng = self.rng
        if self.regime == "uniform" and rng.random() > self.reuse_p * 0.25:
            # uniform regime stays mostly novel; occasional template reuse
            base = rng.choice(LANE_UNIVERSE, size=W_SIZE, replace=False)
        elif rng.random() < self.reuse_p:
            idx = rng.choice(len(self.templates), p=self.template_probs)
            base = self.templates[idx].copy()
        else:
            if self.regime == "uniform":
                base = rng.choice(LANE_UNIVERSE, size=W_SIZE, replace=False)
            else:
                base = rng.choice(self.vocab, size=W_SIZE, replace=False, p=self.vocab_probs)
                if self.regime == "hot":
                    base = np.unique(np.concatenate(([self.hot_cell], base)))[:W_SIZE]
        m = int(rng.integers(0, self.max_mutation + 1))
        if m:
            pos = rng.choice(W_SIZE, size=m, replace=False)
            fresh = rng.choice(
                LANE_UNIVERSE if self.regime == "uniform" else self.vocab,
                size=m,
                replace=False,
            )
            base[pos] = fresh
            base = np.unique(base)
            if base.size < W_SIZE:  # mutation collapsed duplicates: top up
                extra = rng.choice(self.vocab if self.regime != "uniform" else LANE_UNIVERSE,
                                   size=W_SIZE - base.size, replace=False)
                base = np.unique(np.concatenate((base, extra)))
        if self.regime == "hot":
            base = np.unique(np.concatenate(([self.hot_cell], base)))
        return [int(x) for x in base[: max(W_SIZE, 13)]]

    def sample_contacts(self, size: int = 20) -> Tuple[List[int], bool]:
        """Contacts for one injected segment (subset of one template)."""
        rng = self.rng
        idx = rng.choice(len(self.templates), p=self.template_probs)
        tmpl = self.templates[idx]
        contacts = np.sort(rng.choice(tmpl, size=size, replace=False)).astype(np.int64)
        forced_hot = False
        if self.regime == "hot":
            contacts[0] = self.hot_cell
            contacts = np.sort(np.unique(contacts))
            forced_hot = True
        return [int(x) for x in contacts], forced_hot


def sample_permanences(rng: np.random.Generator, n: int, converged: bool) -> List[float]:
    """Permanence mixture: converged segments model repeatedly confirmed
    detectors (mostly connected); young ones model fresh allocations."""
    if converged:
        kinds = rng.random(n) < 0.85
        perms = np.where(kinds, rng.uniform(0.75, 1.0, n), rng.uniform(0.01, 0.41, n))
    else:
        r = rng.random(n)
        perms = np.where(
            r < 0.35,
            0.21,  # initialPermanence
            np.where(r < 0.80, rng.uniform(0.51, 0.91, n), rng.uniform(0.01, 0.41, n)),
        )
    return [float(x) for x in perms]


def build_pools() -> Dict[str, Tuple[int, ...]]:
    pools: Dict[str, Tuple[int, ...]] = {}
    for j in range(N_POOLS):
        base = LANE_UNIVERSE + j * POOL_K
        pools[f"t{j}"] = tuple(base + i for i in range(POOL_K))
    return pools


def inject_state(tracker: SegmentGrowthTracker, specs: Dict[int, List[Tuple[List[int], List[float]]]]) -> None:
    """Write synthetic segment state directly (identically into both trackers)."""
    for cell, seg_specs in specs.items():
        lst = tracker._segments.setdefault(int(cell), [])
        for presyn, perm in seg_specs:
            assert len(set(presyn)) == len(presyn), "contact uniqueness invariant"
            seg = _Segment()
            seg.presyn = list(presyn)
            seg.perm = list(perm)
            lst.append(seg)
    if isinstance(tracker, IndexedSegmentGrowthTracker):
        tracker.rebuild_index()


def synthesize_tables(
    ctx: ContextModel,
    n_segments: int,
    seed_offset: int,
) -> Dict[int, List[Tuple[List[int], List[float]]]]:
    """n_segments distributed over pool cells with mild Zipf popularity,
    capped at the frozen MAX_SEGMENTS_PER_CELL = 255."""
    rng = np.random.default_rng(9_000_000 + seed_offset)
    pool_probs = zipf_weights(N_POOLS, 0.8)
    within = zipf_weights(POOL_K, 0.5)
    cap = SegmentGrowthTracker.MAX_SEGMENTS_PER_CELL
    counts: Dict[int, int] = {}
    specs: Dict[int, List[Tuple[List[int], List[float]]]] = {}
    made = 0
    while made < n_segments:
        pj = int(rng.choice(N_POOLS, p=pool_probs))
        off = int(rng.choice(POOL_K, p=within))
        cell = LANE_UNIVERSE + pj * POOL_K + off
        if counts.get(cell, 0) >= cap:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        contacts, _ = ctx.sample_contacts(20)
        perms = sample_permanences(rng, len(contacts), converged=bool(rng.random() < 0.4))
        specs.setdefault(cell, []).append((contacts, perms))
        made += 1
    return specs


def make_pair(specs: Dict[int, List[Tuple[List[int], List[float]]]], growth_on: bool = True):
    pools = build_pools()
    ref = SegmentGrowthTracker(pools, growth_on=growth_on, construction_seed=17)
    idx = IndexedSegmentGrowthTracker(pools, growth_on=growth_on, construction_seed=17)
    inject_state(ref, specs)
    inject_state(idx, specs)
    return ref, idx


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------


def snapshot(tracker: SegmentGrowthTracker) -> Dict[str, Any]:
    return {
        "segments": {
            cell: [(list(s.presyn), [repr(p) for p in s.perm]) for s in segs]
            for cell, segs in sorted(tracker._segments.items())
        },
        "wins": dict(sorted(tracker._wins.items())),
        "steps": tracker.num_steps,
        "contacts": tracker.total_contacts(),
    }


def run_equivalence(n_segments: int, steps: int, seed: int, label: str) -> Dict[str, Any]:
    """Identical scripted traces through both trackers; assert per-step
    StepResult equality, every-step index integrity, final deep-state equality."""
    ctx = ContextModel(
        np.random.default_rng(seed), "zipf", vocab=1_024, n_templates=64,
        reuse_p=0.75, max_mutation=8,
    )
    specs = synthesize_tables(ctx, n_segments, seed_offset=seed % 1000)
    ref, idx = make_pair(specs)
    assert snapshot(ref) == snapshot(idx), f"[{label}] pre-run state mismatch"
    assert idx.verify_index(), f"[{label}] initial index broken"

    w_rng = np.random.default_rng(seed + 777)
    pool_ids = list(build_pools().keys())
    pool_probs = zipf_weights(len(pool_ids), 0.9)
    mismatches: List[str] = []
    hits_seen = misses_seen = bursts = allocs = 0
    for t in range(steps):
        if w_rng.random() < 0.03:
            w: List[int] = []  # exercise empty-W edge
        else:
            w = ctx.draw_w()
        pool_id = pool_ids[int(w_rng.choice(len(pool_ids), p=pool_probs))]
        r1 = ref.step(pool_id, w)
        r2 = idx.step(pool_id, w)
        if r1 != r2:
            mismatches.append(f"step {t}: {r1!r} != {r2!r}")
            break
        if not idx.verify_index():
            mismatches.append(f"step {t}: shadow-rebuild mismatch")
            break
        hits_seen += len(r1.fired_cells)
        misses_seen += len(r1.punished_segments)
        bursts += int(r1.burst)
        allocs += int(r1.allocated_segment)
    ok = not mismatches and snapshot(ref) == snapshot(idx)
    result = {
        "label": label,
        "n_segments_injected": n_segments,
        "steps": steps,
        "seed": seed,
        "step_results_identical": not mismatches,
        "final_deep_state_identical": snapshot(ref) == snapshot(idx),
        "index_shadow_verified_every_step": not mismatches,
        "hits": hits_seen,
        "misses": misses_seen,
        "bursts": bursts,
        "allocations": allocs,
        "mismatch": mismatches[:1],
    }
    print(
        f"  [equivalence {label}] N={n_segments} steps={steps}: "
        f"results={'OK' if result['step_results_identical'] else 'FAIL'} "
        f"state={'OK' if result['final_deep_state_identical'] else 'FAIL'} "
        f"(hits={hits_seen} misses={misses_seen} bursts={bursts} allocs={allocs})",
        flush=True,
    )
    if not ok:
        raise SystemExit(f"EQUIVALENCE FAILURE: {result}")
    return result


# ---------------------------------------------------------------------------
# Profiling the reference scan
# ---------------------------------------------------------------------------


def global_scan_readonly(tracker: SegmentGrowthTracker, w: np.ndarray):
    """The reference recognition loop verbatim (read-only copy) for timing."""
    hits: List[Tuple[int, int, np.ndarray]] = []
    misses: List[Tuple[int, int, np.ndarray]] = []
    theta_a = tracker._theta_active
    cp = tracker._connected_permanence
    for cell in sorted(tracker._segments):
        for si, seg in enumerate(tracker._segments[cell]):
            presyn = np.asarray(seg.presyn, dtype=np.int64)
            if presyn.size == 0:
                continue
            perm = np.asarray(seg.perm, dtype=np.float64)
            active = np.isin(presyn, w)
            then_connected_active = active & (perm >= cp)
            if int(then_connected_active.sum()) >= theta_a:
                idx = np.flatnonzero(then_connected_active)
                hits.append((cell, si, idx))
    return hits, misses


def indexed_scan_readonly(tracker: IndexedSegmentGrowthTracker, w: np.ndarray):
    """The replacement recognition pass verbatim (read-only copy) for timing;
    also returns (total incidences, distinct touched segments)."""
    counts: Dict[int, int] = {}
    inv_get = tracker._inv.get
    for c in w.tolist():
        bucket = inv_get(c)
        if bucket:
            for key in bucket:
                counts[key] = counts.get(key, 0) + 1
    theta_a = tracker._theta_active
    candidates = [k for k, n in counts.items() if n >= theta_a]
    candidates.sort()
    cp = tracker._connected_permanence
    hits: List[Tuple[int, int]] = []
    for key in candidates:
        cell, si = tracker.unpack(key)
        seg = tracker._segments[cell][si]
        presyn = np.asarray(seg.presyn, dtype=np.int64)
        perm = np.asarray(seg.perm, dtype=np.float64)
        active = np.isin(presyn, w)
        then_connected_active = active & (perm >= cp)
        if int(then_connected_active.sum()) >= theta_a:
            hits.append((cell, si))
    return hits, sum(counts.values()), len(counts)


def profile_reference(n_segments: int, regime: str, seed: int) -> Dict[str, Any]:
    ctx = ContextModel(np.random.default_rng(seed + 31), regime)
    specs = synthesize_tables(ctx, n_segments, seed_offset=n_segments)
    ref, idx = make_pair(specs)
    w_list = ctx.draw_w()
    w = np.unique(np.asarray(w_list, dtype=np.int64))

    reps = max(3, min(15, int(30_000 / max(n_segments, 1))))
    t_scan: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        global_scan_readonly(ref, w)
        t_scan.append(time.perf_counter_ns() - t0)

    t_sorted: List[float] = []
    for _ in range(max(20, reps)):
        t0 = time.perf_counter_ns()
        sorted(ref._segments)
        t_sorted.append(time.perf_counter_ns() - t0)

    # microbench: np.isin call overhead on a realistic 20-contact array
    seg0 = next(iter(specs.values()))[0]
    arr20 = np.asarray(seg0[0], dtype=np.int64)
    t_isin: List[float] = []
    for _ in range(4_000):
        t0 = time.perf_counter_ns()
        np.isin(arr20, w)
        t_isin.append(time.perf_counter_ns() - t0)

    # indexed-side work profile over a fresh W stream
    incs: List[int] = []
    cands: List[int] = []
    w_rng = np.random.default_rng(seed + 99)
    ctx2 = ContextModel(np.random.default_rng(seed + 31), regime)
    for _ in range(50):
        wl = ctx2.draw_w()
        ww = np.unique(np.asarray(wl, dtype=np.int64))
        counts: Dict[int, int] = {}
        for c in ww.tolist():
            for key in idx._inv.get(c, ()):
                counts[key] = counts.get(key, 0) + 1
        incs.append(sum(counts.values()))
        cands.append(sum(1 for v in counts.values() if v >= idx._theta_active))

    scan_ms = statistics.median(t_scan) / 1e6
    return {
        "n_segments": n_segments,
        "regime": regime,
        "scan_ms_median": round(scan_ms, 3),
        "sorted_keys_ms": round(statistics.median(t_sorted) / 1e6, 4),
        "per_segment_us": round(scan_ms * 1000.0 / n_segments, 2),
        "np_isin_us_median": round(statistics.median(t_isin) / 1e3, 2),
        "isin_calls_per_step": n_segments,
        "indexed_w_incidents_mean": round(statistics.mean(incs), 1),
        "indexed_candidates_mean": round(statistics.mean(cands), 1),
        "w_size": len(w_list),
    }


# ---------------------------------------------------------------------------
# A/B timing
# ---------------------------------------------------------------------------

REQUESTED_TARGETS = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
TARGETS = REQUESTED_TARGETS[:4]
STEPS_BY_N = {
    1_000: 120,
    2_000: 100,
    4_000: 80,
    8_000: 60,
    16_000: 40,
    32_000: 30,
}
ROUNDS_BY_N = {
    1_000: 3,
    2_000: 3,
    4_000: 3,
    8_000: 2,
    16_000: 2,
    32_000: 2,
}


def pregen_stream(ctx: ContextModel, pool_seq: List[str], probs: np.ndarray,
                  w_rng: np.random.Generator, steps: int) -> List[Tuple[str, List[int]]]:
    """One identical scripted (pool_id, W) stream for BOTH variants."""
    out: List[Tuple[str, List[int]]] = []
    for _ in range(steps):
        w = ctx.draw_w()
        pool_id = pool_seq[int(w_rng.choice(len(pool_seq), p=probs))]
        out.append((pool_id, w))
    return out


def run_stream(tracker: SegmentGrowthTracker, stream: Sequence[Tuple[str, List[int]]]) -> List[float]:
    samples: List[float] = []
    step = tracker.step
    for pool_id, w in stream:
        t0 = time.perf_counter_ns()
        step(pool_id, w)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def ab_bench(n_segments: int, regime: str, rounds: int, steps: int, seed: int) -> Dict[str, Any]:
    ctx_spec = ContextModel(np.random.default_rng(seed + 500 + n_segments), regime)
    specs = synthesize_tables(ctx_spec, n_segments, seed_offset=n_segments * 3 + 1)
    ref, idx = make_pair(specs)
    pool_seq = list(build_pools().keys())
    probs = zipf_weights(len(pool_seq), 0.9)
    w_rng = np.random.default_rng(seed + 1_000 + n_segments)

    # identical streams: warmup (discarded), then alternating timed rounds
    warm_stream = pregen_stream(ctx_spec, pool_seq, probs, w_rng, 5)
    run_stream(ref, warm_stream)
    run_stream(idx, warm_stream)

    ref_all: List[float] = []
    idx_all: List[float] = []
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for r in range(rounds):
            stream = pregen_stream(ctx_spec, pool_seq, probs, w_rng, steps)
            # alternate order to average out machine drift
            if r % 2 == 0:
                ref_all += run_stream(ref, stream)
                idx_all += run_stream(idx, stream)
            else:
                idx_all += run_stream(idx, stream)
                ref_all += run_stream(ref, stream)
            gc.collect()
    finally:
        if gc_was_enabled:
            gc.enable()

    ref_ms = [x / 1e6 for x in ref_all]
    idx_ms = [x / 1e6 for x in idx_all]
    ref_med = statistics.median(ref_ms)
    idx_med = statistics.median(idx_ms)

    # scan-seam isolation: time ONLY the recognition pass, read-only, on the
    # same post-run state (mirrors the Seam-B "indexed lookup vs linear scan"
    # comparison; excludes punish/confirm/elect mechanics common to both)
    w_rng_scan = np.random.default_rng(seed + 7_000 + n_segments)
    scan_stream = pregen_stream(ctx_spec, pool_seq, probs, w_rng_scan, min(60, max(20, steps)))
    ref_scan: List[float] = []
    idx_scan: List[float] = []
    incidences: List[int] = []
    touched: List[int] = []
    gc_was_enabled2 = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for pool_id, wl in scan_stream:
            ww = np.unique(np.asarray(wl, dtype=np.int64))
            t0 = time.perf_counter_ns()
            global_scan_readonly(ref, ww)
            ref_scan.append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            _h, inc, tch = indexed_scan_readonly(idx, ww)
            idx_scan.append(time.perf_counter_ns() - t0)
            incidences.append(inc)
            touched.append(tch)
        reps = 3
        for pool_id, wl in scan_stream[: max(5, len(scan_stream) // 3)]:
            ww = np.unique(np.asarray(wl, dtype=np.int64))
            for _ in range(reps):
                t0 = time.perf_counter_ns()
                global_scan_readonly(ref, ww)
                ref_scan.append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()
                _h, inc, tch = indexed_scan_readonly(idx, ww)
                idx_scan.append(time.perf_counter_ns() - t0)
                incidences.append(inc)
                touched.append(tch)
    finally:
        if gc_was_enabled2:
            gc.enable()

    row = {
        "n_segments_start": n_segments,
        "regime": regime,
        "rounds": rounds,
        "steps_per_round": steps,
        "ref_ms_median": round(ref_med, 3),
        "ref_ms_mean": round(statistics.fmean(ref_ms), 3),
        "ref_ms_p95": round(sorted(ref_ms)[int(0.95 * len(ref_ms)) - 1], 3),
        "idx_ms_median": round(idx_med, 4),
        "idx_ms_mean": round(statistics.fmean(idx_ms), 4),
        "idx_ms_p95": round(sorted(idx_ms)[max(0, int(0.95 * len(idx_ms)) - 1)], 4),
        "speedup_median": round(ref_med / idx_med, 1),
        "ref_scan_ms_median": round(statistics.median(ref_scan) / 1e6, 3),
        "idx_scan_ms_median": round(statistics.median(idx_scan) / 1e6, 4),
        "scan_speedup_median": round(
            statistics.median(ref_scan) / statistics.median(idx_scan), 1
        ),
        "w_incidents_mean": round(statistics.mean(incidences), 1),
        "w_touched_segments_mean": round(statistics.mean(touched), 1),
        "segments_end_ref": ref.total_segments(),
        "segments_end_idx": idx.total_segments(),
        "state_equal_after_timing": snapshot(ref) == snapshot(idx),
        "index_ok_after_timing": idx.verify_index(),
    }
    if not row["state_equal_after_timing"]:
        raise SystemExit(f"A/B STATE DIVERGENCE at N={n_segments} regime={regime}: {row}")
    print(
        f"  [A/B N={n_segments:>6} {regime:>7}] ref {row['ref_ms_median']:>8.3f} ms/step"
        f"  idx {row['idx_ms_median']:>7.4f} ms/step  speedup {row['speedup_median']:>7,.0f}x"
        f" | scan-only {row['ref_scan_ms_median']:>8.3f} -> {row['idx_scan_ms_median']:>8.4f}"
        f" ({row['scan_speedup_median']:>8,.0f}x;"
        f" incidences/step {row['w_incidents_mean']},"
        f" touched {row['w_touched_segments_mean']})"
        f"  (segs {row['segments_end_ref']}, state_equal=True)",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="smoke: N=1000 only, few steps")
    ap.add_argument("--json", type=str, default="", help="write full results JSON to PATH")
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args(argv)

    targets = [1_000] if args.quick else list(TARGETS)
    regimes = ["uniform", "zipf", "hot"]
    print(f"bench_recognition: targets={targets} regimes={regimes} seed={args.seed}", flush=True)
    out: Dict[str, Any] = {
        "schema_version": "r3-recognition-python-prototype-bench-v1",
        "scope": "historical-python-prototype-only; not C1 wall-clock evidence",
        "requested_targets": list(REQUESTED_TARGETS),
        "targets": targets,
        "not_run_targets": [n for n in REQUESTED_TARGETS if n not in targets],
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "seed": args.seed,
        }
    }

    # 1) correctness first -- fail fast before any timing claim
    print("[1/4] bit-exactness equivalence ...", flush=True)
    if args.quick:
        eq_runs = [("small-zipf-mutation", 600, 120)]
    else:
        eq_runs = [
            ("small-zipf-mutation", 600, 300),
            ("mid-zipf", 4_000, 150),
        ]
    out["correctness"] = [
        run_equivalence(n, s, args.seed + i, label)
        for i, (label, n, s) in enumerate(eq_runs)
    ]

    # 2) profile decomposition of the reference scan
    print("[2/4] reference scan profile decomposition ...", flush=True)
    out["profile"] = [profile_reference(n, "zipf", args.seed) for n in targets]
    for row in out["profile"]:
        print(
            f"  [profile N={row['n_segments']:>6}] full-scan {row['scan_ms_median']:>8.3f} ms"
            f"  ({row['per_segment_us']} us/segment; np.isin median {row['np_isin_us_median']} us"
            f" x{row['isin_calls_per_step']} calls/step; sorted() {row['sorted_keys_ms']} ms;"
            f" indexed incidences/step {row['indexed_w_incidents_mean']},"
            f" candidates/step {row['indexed_candidates_mean']})",
            flush=True,
        )

    # 3) A/B benches
    print("[3/4] A/B reference vs indexed ...", flush=True)
    rows: List[Dict[str, Any]] = []
    for n in targets:
        steps = 40 if args.quick else STEPS_BY_N[n]
        rounds = 1 if args.quick else ROUNDS_BY_N[n]
        for regime in regimes:
            rows.append(ab_bench(n, regime, rounds, steps, args.seed))
    out["ab"] = rows

    # 4) calibration vs C1_READINESS measured rows (|W| = 54 basis)
    if not args.quick:
        print("[4/4] calibration vs C1_READINESS rows ...", flush=True)
        cal = []
        for n in (125, 500, 1_000, 2_000):
            cal.append(ab_bench(n, "zipf", 3, 200, args.seed))
        out["calibration"] = cal

    print("\n=== summary (median ms/step; end-to-end and scan-seam-only) ===")
    for r in rows:
        print(
            f"  N={r['n_segments_start']:>6} {r['regime']:>7}: "
            f"{r['ref_ms_median']:>9.3f} -> {r['idx_ms_median']:>8.4f}  "
            f"({r['speedup_median']:>8,.0f}x)   "
            f"scan: {r['ref_scan_ms_median']:>9.3f} -> {r['idx_scan_ms_median']:>8.4f}  "
            f"({r['scan_speedup_median']:>9,.0f}x)"
        )
    if "calibration" in out:
        print("  calibration (C1_READINESS measured ref column: 3.1/12.2/24.9/50.3 ms):")
        for r in out["calibration"]:
            print(
                f"    N={r['n_segments_start']:>5}: ref {r['ref_ms_median']:>7.3f} ms/step"
                f"  idx {r['idx_ms_median']:>7.4f} ms/step"
            )
    out["prototype_verdict"] = {
        "bit_exact": all(
            row["step_results_identical"]
            and row["final_deep_state_identical"]
            and row["index_shadow_verified_every_step"]
            for row in out["correctness"]
        )
        and all(
            row["state_equal_after_timing"] and row["index_ok_after_timing"]
            for row in rows
        ),
        "c1_eligible": False,
        "reason": (
            "The physical path uses native TemporalStore recognition; this bench "
            "measures only the legacy Python SegmentGrowthTracker prototype."
        ),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nresults JSON written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
