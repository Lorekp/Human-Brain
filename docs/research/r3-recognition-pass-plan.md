# R3 historical Python recognition prototype — indexed candidate narrowing

**Status: HISTORICAL PROTOTYPE; INELIGIBLE FOR C1.** This document closes an
abandoned handoff artifact for the legacy Python `SegmentGrowthTracker`. The
physical R3 path already uses the native
`TemporalStore::recognize_runtime_context` contact-cell index. Nothing measured
here is current physical-path throughput or evidence for the C1 wall-clock cap,
and the prototype must not be integrated into the C1 runner.

The original dispatch was pre-launch decision (b) of the superseded
`C1_READINESS.md` state at `10453e0`. `segment_growth.py` remains unmodified;
the prototype lives only in `experiments/r3_pilot/bench_recognition.py`.

Companion bench: `experiments/r3_pilot/bench_recognition.py` (seeded,
deterministic; `--quick`, `--json PATH` flags). Every number below was measured
on 2026-08-28 on Windows 11, Python 3.11.15, NumPy 2.4.6, seed 20260826. The
machine-readable result is `experiments/r3_pilot/recognition_bench.json`
(SHA-256 `3763443a685314244aa7ee77f24deb30c5602ae9663493a6e7983c88e2038ad5`).
Ratios describe this Python prototype only.

---

## 1. Profile: what exactly is O(segments)

`SegmentGrowthTracker.step` (src/humanbrain/segment_growth.py:184) runs, on
EVERY step, before any mutation:

```python
for cell in sorted(self._segments):          # every cell owning >= 1 segment
    for si, seg in enumerate(self._segments[cell]):   # every live segment
        presyn = np.asarray(seg.presyn, dtype=np.int64)
        if presyn.size == 0: continue
        perm  = np.asarray(seg.perm, dtype=np.float64)
        active = np.isin(presyn, w)                       # <-- per-segment numpy call
        then_connected_active = active & (perm >= self._connected_permanence)
        if int(then_connected_active.sum()) >= self._theta_active:
            (hits if cell in pool_set else misses).append((cell, si, idx))
```

Candidate generation is therefore **all live segments × one `np.isin` against
W_{t-1} each**, regardless of whether the segment could possibly reach
theta_a = 13. Decomposition measured at N=1000 (zipf table, |W|=54):

| component | measured |
|---|---|
| full scan (read-only copy of the loop above) | 29.769 ms/step |
| `sorted(self._segments)` alone | 0.0201 ms |
| `np.isin(20-elem presyn, 54-elem w)` call overhead, median | 38.8 µs/call |
| calls of exactly that kind per step | = live segments |
| ⇒ observed full-scan cost per segment | 29.77 µs |

The scan is ~72% of the 1k Zipf end-to-end median; the remaining shared mechanics
(punish/confirm loops, election over the arriving pool's ≤ 32 cells × their
segments) are O(candidates + pool segments), not O(table). Scaling is linear
in live segments up to ~2k and mildly superlinear beyond (allocator/cache),
matching C1_READINESS's 24.9 ms @1k → 278.5 ms @8k session. The theta_m=10
potential-active check inside `_elect` is NOT the problem: it touches only the
arriving pool's cells.

## 2. Structural fact the index exploits

A segment is classified (hit or miss) iff

```
#{i : presyn_i ∈ W_{t-1}  AND  perm_i >= 0.50}  >=  theta_a = 13.
```

Connected-active ⊆ active ⊆ {contacts whose presynaptic cell ∈ W}. So a
necessary condition for classification is:

> **≥ theta_a of the segment's ≤ s=20 contacts lie in W_{t-1}.**

Segments sharing < theta_a contacts with W cannot pass, no permanence
configuration notwithstanding. Contacts only ever enter toward past winner
sets (`_fill_toward`) and leave by purge, so the relation "segment S lists
presynaptic cell c" changes only at four code sites (§4). This makes an
**inverted index** exact:

```
inv : presynaptic cell id  ->  set of packed segment keys
key = (cell << 16) | segment_index        # MAX_SEGMENTS_PER_CELL = 255 < 2^16
```

Per step: candidates = segments with `|presyn ∩ W| >= theta_a`, obtained by
walking `inv[c]` for c ∈ W and counting incidences. Cost is bounded by

```
incidences(W) = sum_{c in W} deg(c)      -- NOT O(total segments)
```

i.e. by the winner-set mass of actual contact overlap, which is bounded by
min(|W| · max_fanout, total contacts) and is tiny when W lands on cold cells.
This is the same move as Seam-B (cpu-seams-spec.md §B): replace an O(table)
procedural membership walk with an id-keyed index whose cost tracks the
*working set*, bit-exactly. (Seam-B used `BTreeMap<u64,u32>`; here a Python
dict-of-sets is the right constant factor for ~20-element contact lists.)

## 3. The prototype

`IndexedSegmentGrowthTracker(SegmentGrowthTracker)` in
bench_recognition.py overrides:

- **`step`** — byte-for-byte the reference body EXCEPT the scan block:
  count incidences over `inv[c]`, c∈W; keep keys with count ≥ theta_a;
  `candidates.sort()` (packed-key order == reference visitation order, Lemma 2);
  run the IDENTICAL predicate expressions on the identical arrays; identical
  punish/confirm loops plus `_sync(...)` index reconciliation after each
  presyn mutation; identical election call and StepResult.
- **`_allocate_segment` / `_learn_on_segment`** — register/diff-sync new or
  purged/filled contacts (election-time growth paths).
- **`rebuild_index` / `verify_index`** — full reconstruction from live state +
  shadow check (verification mode; also used after state injection).

## 4. Index invariant and maintenance sites

Invariant: `inv[c] == {(cell,si) : c ∈ _segments[cell][si].presyn}` at every
step boundary. `seg.presyn` is written at exactly four sites, each covered:

1. miss-path decay+purge (`step`, misses loop) → sync old→new list;
2. hit-path reinforce+purge (`step`, hits loop) → sync;
3. election-time learn (`_learn_on_segment`: reinforce+purge, then
   `_fill_toward` may add contacts into freed budget) → diff old→new;
4. burst allocation (`_allocate_segment`: append + `_fill_toward`) → register
   final contact list.

No other code writes `presyn` (verified by grep; `_purge`/`_fill_toward` only
build/return lists). Contact ids within a segment are unique (fill skips
existing; purge keeps subsets), so multiset ambiguity never arises. Empty-presyn
segments (fully decayed) stay allocated but carry no index entries — matching
the reference's explicit `presyn.size == 0` skip.

## 5. Bit-exactness argument (proof sketch)

**Lemma 1 — candidate-set equivalence (sound + complete).**
Reference classifies segment S iff `conn(S,W) ≥ θ_a` where
`conn(S,W)=|{i: presyn_i∈W ∧ perm_i≥cp}|`. Since conn ≤ |presyn∩W| = counts[S],
any classified S has counts[S] ≥ θ_a and IS generated by the index walk.
Conversely any key with counts[S] < θ_a has conn ≤ counts < θ_a and the
reference provably does not classify it. For retained keys both
implementations evaluate the same expressions (`np.isin`, `>=` cp, `.sum()`,
`np.flatnonzero`) on the same arrays → identical idx arrays. Segments absent
from inv have zero W-incidences → conn = 0 < θ_a = 13, excluded identically.

**Lemma 2 — visitation-order equivalence.**
The reference visits `(cell, si)` in strictly ascending lexicographic order
(`sorted(dict)` gives ascending cell ids; `enumerate` gives ascending si along
the append-only per-cell list). Packed keys sort identically to their
`(cell, si)` lexicographic order because si occupies the low 16 bits and
si < 255. Hence hits/misses receive identical tuples in identical order →
`fired_cells` and `punished_segments` are element-wise identical, and
`removed_total` (a sum) matches.

**Lemma 3 — transition identity ⇒ trajectory identity.**
Post-scan mutations are per-segment-local functions of (state, w, idx) applied
in identical order to bitwise-identical inputs using the same float ops
(`+= -=` clip IEEE-754 deterministic); election reads only pool-local state
and w. By induction on steps: identical `_segments`, `_wins`, counters, and
StepResult sequence forever. The bench verifies this empirically (per-step
StepResult equality, final deep-state equality incl. `repr()`-exact
permanences, wins map, contact totals).

**Index maintenance** is covered by induction over the four sites (§4) plus the
run-time shadow-rebuild check executed EVERY step of the equivalence runs.

Edge cases verified in-run: empty W (no candidates ↔ reference finds nothing);
W disjoint from all contacts (same); fully-decayed empty segments; hits vs
misses split by arriving-pool membership; MAX_SEGMENTS_PERCell stop still
raises through the inherited allocator.

## 6. Bench results (2026-08-28)

Method: seeded synthetic tables structurally faithful to replay (pools of k=32
above the lane gid space; segments' 20 contacts drawn toward recurring
Zipf-weighted context templates; 40% "converged" segments with connected-heavy
permanences; |W|=54 as in the C1_READINESS measurement basis; election path
live so bursts allocate during timing). Three regimes bracket reality:
`uniform` (contexts uniform over 131,072 lane cells — near-zero overlap),
`zipf` (recurring template contexts, heavy overlap — realistic-to-pessimistic),
`hot` (one hub cell in every segment AND every W — fanout stress bound).
Median ms/step over interleaved identical streams, GC off. Dedicated
equivalence traces covered 300 steps from 600 injected segments and 150 steps
from 4,000 injected segments. Every `StepResult`, final deep state, and
every-step shadow-index check matched. All 12 timed rows also ended with equal
state and a valid index.

The requested grid was 1k/2k/4k/8k/16k/32k. The run was deliberately stopped
at 8k once the path was confirmed historical; 16k/32k are recorded as NOT RUN
because the native physical benchmark owns those measurements.

### 6.1 Scan seam only (recognition pass in isolation)

| start segments | regime | reference ms | indexed ms | speedup | incidences/step |
|---:|---|---:|---:|---:|---:|
| 1,000 | uniform | 41.192 | 0.0256 | 1,609.1× | 14.5 |
| 1,000 | zipf | 36.967 | 0.6549 | 56.5× | 7,128.9 |
| 1,000 | hot | 39.972 | 0.7637 | 52.3× | 7,225.2 |
| 2,000 | uniform | 89.433 | 0.0296 | 3,021.4× | 36.3 |
| 2,000 | zipf | 67.065 | 1.1732 | 57.2× | 11,749.0 |
| 2,000 | hot | 83.638 | 1.2619 | 66.3× | 11,897.8 |
| 4,000 | uniform | 127.893 | 0.0333 | 3,840.6× | 46.6 |
| 4,000 | zipf | 127.821 | 4.2517 | 30.1× | 25,800.8 |
| 4,000 | hot | 126.685 | 4.2995 | 29.5× | 25,866.3 |
| 8,000 | uniform | 250.602 | 0.0415 | 6,038.6× | 112.0 |
| 8,000 | zipf | 249.702 | 6.1270 | 40.8× | 47,294.7 |
| 8,000 | hot | 249.549 | 5.4585 | 45.7× | 47,701.5 |

### 6.2 End-to-end `step()` (what the runner experiences)

| start segments | regime | reference ms | indexed ms | speedup |
|---:|---|---:|---:|---:|
| 1,000 | uniform | 37.808 | 0.4338 | 87.1× |
| 1,000 | zipf | 41.384 | 1.6026 | 25.8× |
| 1,000 | hot | 34.453 | 1.3468 | 25.6× |
| 2,000 | uniform | 71.496 | 0.4396 | 162.6× |
| 2,000 | zipf | 69.313 | 2.6951 | 25.7× |
| 2,000 | hot | 66.764 | 2.6211 | 25.5× |
| 4,000 | uniform | 127.793 | 0.7525 | 169.8× |
| 4,000 | zipf | 124.987 | 4.0671 | 30.7× |
| 4,000 | hot | 148.741 | 4.3091 | 34.5× |
| 8,000 | uniform | 258.472 | 1.5494 | 166.8× |
| 8,000 | zipf | 249.869 | 10.8310 | 23.1× |
| 8,000 | hot | 253.491 | 10.3041 | 24.6× |

Reading: the indexed variant's residual cost is dominated by shared mechanics
the reference ALSO pays (miss-decay churn, election np.isin over pool
segments) — under zipf stress ~13 segments/step take negative evidence, which
is real work, not scan waste. Under `hot`, incidence mass degrades linearly
with total segments but at dict-increment constants (~0.15 µs) vs the
reference's numpy-call constants (~35–45 µs) — still orders ahead, and the
degradation bound is `sum deg(c)`, not a second global scan.

### 6.3 Scaling verdict

The isolated reference scan exponent is 1.021 over 1k–8k (29.77–31.19
µs/segment), confirming a linear full-table scan. End-to-end reference
exponents are 0.86–0.98. The indexed prototype follows incidence mass:
uniform overlap grows 3.57× while table size grows 8×; Zipf and hot grow
6.76× and 7.65× as their incidence mass grows 6.63× and 6.60×. The minimum
observed end-to-end median speedup is 23.1×.

## 7. Disposition

**Prototype verdict: PASS for the legacy Python algorithm, NO-GO for C1
integration.** The inverted index is bit-exact on the exercised traces and
changes candidate-generation cost from whole-table size to winner-contact
incidence mass. That is useful historical algorithm evidence.

It is not a current implementation proposal. The physical R3 path already has
native contact-cell indexing in `TemporalStore::recognize_runtime_context`.
Adding this Python tracker to the runner would duplicate the wrong layer and
would not measure or improve the native path.

## 8. Wall-clock conclusion

There is deliberately no C1 projection from these numbers. Python synthetic
tables, the legacy tracker, and its mutation loop are not the frozen physical
execution path. The current native physical curve and integrated persistence /
physics / recognition benchmark are the only eligible wall-clock evidence.

## 9. Limits

- Synthetic tables are structurally useful but not corpus-derived physical
  state.
- The parity traces cover the exercised state transitions, not a proof for an
  unbounded Python implementation.
- 16k and 32k were intentionally not run after the architectural eligibility
  check; absence of those rows is not a performance failure.
- No C1 protocol, runner, native temporal source, or production tracker was
  changed or executed.
