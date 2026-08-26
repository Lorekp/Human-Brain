# R3 A3 recognition index decision

**Status: CURRENT PRE-OUTCOME DESIGN/COST INFERENCE.** This note records an
engineering decision before any R3-F0 outcome-bearing run. It is not a measured
throughput result, capability result, threshold selection, protocol amendment,
or authorization to launch C1.

## Decision

1. Keep the current intrusive per-source posting index as the exact generic
   reference implementation, but classify it **NO-GO for final-density A3
   production recognition**. Its final TRAIN schedule requires hundreds of
   millions of serial posting visits per second before physics or mutation.
2. Remove the whole-touched-set sort from diagnostic recognition. In a future
   production transaction, accumulate recognition in arbitrary posting order
   and preserve the Python mutation contract by ordering only active/predicted
   segment IDs.
3. Prototype a private, lossless index keyed only by the exact physical
   11-source input assembly. It is conditional on the invariant and A/B gates
   below. Non-code or uncertifiable contexts take an exact full-segment fallback.
4. Treat per-source Roaring bitmaps as a contingency generic design only if the
   assembly index is scientifically rejected or fails the full-density gate.
   Do not add a production Roaring dependency yet.
5. Do not build a threshold-specific subset index until the threshold is frozen.
   This note leaves all threshold arms unchanged.

Recognition optimization alone cannot settle C1 readiness. The integrated
physics, active-segment mutation, checkpoint, and recognition path must still
fit the wall-clock cap.

## Provenance and sealed scope

The inspection was performed at repository HEAD
`9621e09566e59604d946ce6a5426b84390126e74` with these executable sources:

| Source | Last source commit | SHA-256 |
| --- | --- | --- |
| `src/humanbrain/_native_src/src/temporal.rs` | `3dd2c4fa9eefb7ae0d99da2d5363053ba540a838` | `e9ecb27710ec6ca27ac7f32a6ae9b534ffdca0c3208cb48c1eb858474bae7ef9` |
| `src/humanbrain/sparse_temporal_cortex.py` | `c67c6c4b960a26ca8227f78d80da306076d1a931` | `22e5ab6bb9e3b1bc7010e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8` |

Structural sizing inputs were read from
`experiments/r3_pilot/calibrate_relay_pool_capacity.py` at SHA-256
`a5951db0537dc4201bd7cb2be900e5f31cb6087fd879535395bd78df674f2750`
and `experiments/r3_pilot/relay_pool_capacity.json` at SHA-256
`4d0117ed27cb16763747e0559580a34f0add110a9dbdec031d23b2e5da825d94`.
The JSON remains an ideal structural sizing artifact, not capability or
physical-race evidence; a later capacity refreeze may supersede it without
changing the arithmetic recorded here.

Canonical TRAIN inputs and counts:

| Item | Value |
| --- | --- |
| Corpus SHA-256 | `d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa` |
| Tokenizer SHA-256 | `130f42fc92e5c1f42c53d8904aeefbc4db21c189dc5b9fae98157dc06c9218a2` |
| TRAIN stream SHA-256 | `f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5` |
| K=11 codebook SHA-256 | `1f3385590a09aa6c1e27eca59f52dcd900fe0e96547b965813176dfb8fc96e00` |
| TRAIN books / tokens per pass | 49 / 2,117,412 |
| Non-boundary transitions per pass | 2,117,363 |
| Novel contexts / observed token IDs | 2,116,946 / 522 |
| Sparse vocabulary codes | 768 (`0..255`, `1024..1535`) |
| Held-out books | 17, role-skipped before decode |
| Held-out semantic decodes / outcome accesses | 0 / 0 |

The canonical whole-corpus binary was hashed for provenance. No held-out span
was decoded and no held-out or C1 outcome was inspected.

## Fixed size and notation

For final R3-F0 A3 sizing:

- segments `S = 23,286,406`;
- live pair-slot capacity `P = 10S = 232,864,060`;
- logical physical-source domain `D = 4096`;
- query width `q = 11` logical source/relay pairs;
- physical match thresholds `10, 16, 18, 20`, hence logical thresholds
  `r = 5, 8, 9, 10`;
- `V` is posting memberships visited, `T` is distinct touched segments, `M`
  is matching segments, `A` is active segments, and `O` is touched owners.

Each logical pair contributes two physical contacts with one shared permanence
tick. All calculations in this note use logical pair membership and convert
thresholds by `ceil(theta / 2)`.

## Correction chain: 625 thousand, 8.825 million, and the schedule

### The 625,367 figure is a hypothetical uniform control

If all `P` pair memberships were independently and uniformly distributed over
the `D` sources, an 11-source query would visit

```text
E[V_uniform] = qP / D
             = 11 * 232,864,060 / 4096
             = 625,367.349 postings/query.
```

The matching uniform touched expectation is

```text
E[T_uniform] = S * (1 - C(4085, 10) / C(4096, 10))
             = 618,535.329 segments/query.
```

This is useful only as a synthetic control. It is not representative of A3,
because A3 places every segment's ten sources inside one correlated eleven-bit
token code.

### Actual A3 code correlation

Let `C_u` be the number of novel contexts whose previous physical input code is
token code `u`, and let `o(t,u)` be the intersection of query code `t` and code
`u`. A segment omits one uniformly chosen source from its eleven-source parent
code, giving

```text
E[V_t] = 10 * sum_u(o(t,u) * C_u).
```

The complete 768-code table has this measured pairwise overlap distribution:

| Distinct-code overlap | Code pairs |
| ---: | ---: |
| 0 | 285,981 |
| 1 | 8,446 |
| 2 | 101 |

The maximum distinct-code overlap is 2, below every candidate `r >= 5`.
Therefore a full/live segment can qualify only for its own physical code.

The earlier final-density `8,825,340.84` value used all 2,117,412 TRAIN token
IDs as the query-frequency proxy. Exact scheduled recognition uses the previous
token at each of 2,117,363 non-boundary transitions and gives
`8,825,489.9615` expected visits/query. The difference is 0.00169%; the latter
is the governing scheduled value.

Final-density structural stress points are:

| Workload | Visits `V` | Touched `T` | Own-code/full-live qualifiers |
| --- | ---: | ---: | ---: |
| Maximum-visits query (token 101) | 21,793,860 | 3,539,448 | 2,028,268 |
| Maximum-touched query (token 1164) | 4,819,940 | 4,798,061 | 2,431 |
| Maximum-qualifier query (token 32) | 20,713,940 | 2,173,517 | 2,060,047 |
| Scheduled final-density mean | 8,825,489.9615 | approximately 1.447 million | 819,720.0195 |

Visits and cross-code touches are structural expectations over the omitted
source position. They are not measured native instructions or wall time.

### Exact growth schedule replaces the stale 56.06-trillion envelope

Multiplying a final-density mean by all three passes produced the older
approximately 56.06-trillion estimate. That intentionally conservative
envelope incorrectly charged first-pass queries as if the complete index
already existed. It is retired for cap decisions.

A TRAIN-order replay that grows mappings on pass one and reuses them on passes
two and three gives:

| Schedule slice | Expected intrusive visits |
| --- | ---: |
| First growing pass | 9,344,152,197,710 |
| Each full-density replay pass | 18,686,765,901,440 |
| Exact three-pass integration | **46,717,684,000,590** |

Spending the complete 15-hour cap on recognition alone would require
`865,142,296` posting visits/second. This excludes the fitted GLIF3 physics,
race, ordered plasticity, checkpointing, and all other runner work. A serial
intrusive pointer chain is therefore not a credible final A3 mainline without
contradictory full-density measurement.

## Current intrusive index

The native store holds a doubly linked per-source membership list through
`next_pair`, `prev_pair`, and source head/tail metadata. Recognition traverses
the eleven lists, uses generation-stamped segment counters, sorts every touched
segment, reduces matching/active owners, and finally sorts owners.

### Whole-touched sorting is not semantic

`touched_segments.sort_unstable()` can be removed from diagnostic `prepare`
without changing any public output:

- candidate, matching, active, and maximum counts are commutative integers;
- duplicate physical sources are rejected;
- matching and active best-segment comparators explicitly compare primary
  score, secondary score, then lower segment ID;
- owner rows remain canonical through `touched_owners.sort_unstable()`;
- the private touched-vector order is not exported or retained across calls.

Generation rollover and output-capacity retry add no hidden ordering
dependency.

For future production, collect a segment only when its potential count crosses
`theta_match`, scan those matching qualifiers for final owner scores, compact
active IDs in place, and order only the active IDs before the Python-equivalent
reinforce/punish loop. Matching-only reinforcement already occurs in frozen
token-column order. This gives

```text
diagnostic: O(V + M + O log O)
production: O(V + M + A log A + O log O) + mutation
worst case: V=P and T=M=A=S.
```

It is a worthwhile exact cleanup, but it does not change the dominant `V` term.

Current A3 requested element payload, excluding allocator metadata, the
separately measured 432-byte store header, physics, and Python diagnostics:

| Allocation | Bytes |
| --- | ---: |
| Learned segment SoA | 1,490,461,056 |
| Intrusive source index `8P + 12D` | 1,862,961,632 |
| Source generation stamps | 16,384 |
| Segment stamps/counters/touched scratch `10S` | 232,864,060 |
| Owner stamps/bests/output-order scratch | 2,097,152 |
| **Native total** | **3,588,400,284** |

The Python diagnostic output adds 2,097,152 bytes.

## Alternative exact indexes

### Sorted postings

An immutable CSR floor is

```text
4P + 4(D + 1) = 931,472,628 bytes.
```

An eleven-way document-at-a-time threshold merge yields ascending IDs with
`O(V log q)` work and `O(q)` iterator scratch. This is not a simple mutable
replacement: packed CSR insertion shifts the tail, sorted intrusive insertion
searches an average 56,851.6-entry source list, and hot lists reach millions.
Trees or mutable sorted chunks reproduce the problem with more metadata.

### Raw and Roaring bitmaps

A raw bitmap needs `ceil(S/64) = 363,851` words/source, or
`11,922,669,568` bytes for all 4096 sources. An eleven-input bit-sliced query
reads 4,002,361 words and needs 11,643,232 bytes for four count planes before
enumerating qualifiers. It is exact and mutable but rejected on memory.

Canonical Roaring payload is bounded by `2P = 465,728,120` bytes because array
containers hold at most 4096 u16 values and denser containers use an 8192-byte
bitmap. There can be as many as 1,458,176 containers, so directories, headers,
capacity, and allocator overhead remain material and must be measured. A
noncanonical mutable implementation that retains bitmap containers after heavy
deletion can approach the raw envelope.

Roaring still needs a custom exact k-of-11 container threshold operation and
then segment-slot inspection for connected counts and best scores. Mutation
causes container insertion/removal and possible allocation or representation
conversion, complicating atomic fail-stop. The current crate has no Roaring
dependency. It is the generic contingency, not the next production slice.

### Pair and threshold-subset indexes

The following are optimistic u32 segment-ID floors. They exclude subset keys,
bucket dictionaries, collision verification, deduplication, allocator state,
and the links needed for pair-atomic mutation.

| Logical threshold `r` | Stored subsets/segment | Query subsets | ID bytes | Hits/full-overlap segment |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 252 | 462 | 23,472,697,248 | 252 |
| 8 | 45 | 165 | 4,191,553,080 | 45 |
| 9 | 10 | 55 | 931,456,240 | 10 |
| 10 | 1 | 11 | 93,145,624 | 1 |

A pair-of-sources index also has 45 memberships/segment and is unattractive.
Reject `r=5` and `r=8`; benchmark `r=9` only after threshold selection; `r=10`
is compelling only if it becomes the frozen matching threshold. Maintaining all
four indexes is prohibited.

## Conditional physical-assembly routing

This is a compiled synaptic-routing index, not a semantic lookup.

Let frozen physical source codes be `C_i`, with `|C_i|=11`, maximum distinct
intersection 2, and `r>=5`. A routed live segment `s` carries only a derived
runtime route asserting `L_s subset C_i`, where `L_s` is its live physical
source set.

For a query exactly equal to physical code `C_j`:

- if `i != j`, `|L_s intersection C_j| <= 2 < r`, so `s` cannot match;
- if `i == j`, potential is exactly twice the live-pair count and connected is
  twice the connected-pair count;
- if `|L_s| < r`, the segment cannot match any query and is inert.

The recognizer can therefore scan only the certified `C_j` bucket. A query that
is not exactly one physical code, a codebook with `max_overlap >= r`, or any
uncertifiable store state takes the exact ascending full-segment scan.

### Mutation invariant

- **Allocate:** sample all sources from the exact physical input code.
- **Reinforce:** retain a subset of current sources and regrow only from that
  same physical code.
- **Cross-code exclusion:** a routed segment cannot reach match threshold under
  another exact code, so it cannot be reinforced there.
- **Punish:** only decrement or remove sources, preserving the subset.
- **Ticks:** source/relay twins continue to share one tick and live/connected
  totals update in the same atomic transaction.
- **Non-code mutation:** use the generic exact path and recertify after rewrite;
  if uniqueness cannot be proven, disable the fast path rather than guess.

Allocations append increasing segment IDs to the route. Valid reinforce and
punish operations never change that route. Bucket iteration is consequently in
ascending segment-ID order and already preserves the mutation contract.

### Checkpoint rebuild

The derived route, buckets, and live/connected totals are not serialized.
Restore scans physical segments in ascending ID order and:

1. validates source uniqueness, ranges, source/relay tick identity, and live
   ticks;
2. recomputes live and connected pair counts;
3. marks live-count `<r` segments inert;
4. finds the unique physical code containing every remaining live source;
5. appends the segment ID to that route, or marks the store generic if there is
   no route;
6. STOPs if more than one route exists, because that contradicts the certified
   `max_overlap < r` codebook invariant.

This makes resume identity depend only on physical learned state, segment IDs,
the frozen encoder, and configuration—not a serialized acceleration cache.

### Forbidden shortcuts

The following would cross the cell-only line and are forbidden:

- caller-supplied token identity or previous-token identity;
- a `(context, target)` key;
- a context-to-owner, context-to-winner, or prediction table;
- a corpus transition table or frequency-conditioned route;
- a fast path that omits the exact fallback or skips invariant validation;
- serialized derived routing treated as learned evidence.

The only legal key is the canonical set of currently active physical source
GIDs, matched against the already-frozen physical encoder wiring.

### Memory and schedule work

The compact routing payload is:

```text
bucket segment IDs     4S
runtime route ID       2S
live-pair count        1S
connected-pair count   1S
                       --
total                  8S = 186,291,248 bytes
```

A concrete fixed 256-ID block arena needs at most 91,730 blocks including one
partial block per route. With 8-byte block metadata, 8-byte head/tail metadata
per route, and the 768 x 11 u16 physical codebook, its exact proposed ceiling is
187,834,024 bytes. Combined with the learned SoA and owner scratch, the proposed
native recognition-store payload is 1,680,392,232 bytes (about 1.565 GiB), plus
2,097,152 bytes when materializing Python diagnostics.

Exact TRAIN-order bucket records scanned under the structural allocation model:

| Schedule slice | Bucket records |
| --- | ---: |
| First growing pass | 867,893,735,775 |
| Each full-density replay pass | 1,735,644,839,643 |
| Three-pass integration | **4,339,183,415,061** |

Using the entire 15-hour cap would require `80,355,248` bucket records/second.
This is approximately one order of magnitude below the intrusive visit rate and
the records are chunk-sequential, but it is still an inference. Active mutation
can require visiting many of the same segments and may remain the real lower
bound.

## Implementation order

1. Remove only the diagnostic touched sort and freeze its order-independence
   proof in focused tests.
2. Add generic threshold-crossing accumulation and in-place active compaction;
   retain only the active-ordering seam required by mutation.
3. Add a synthetic full-density benchmark fixture using exact TRAIN/codebook
   histograms but no outcomes.
4. Prototype physical-assembly routing behind the existing deep native
   recognition interface. Do not expose route classification to the host.
5. Run the A/B gate against the optimized intrusive reference.
6. Prototype Roaring only if assembly routing is rejected, invalidated, or misses
   the gate. It is also the candidate for future arbitrary A1-style contexts.
7. Reconsider an `r=9` or `r=10` subset index only after threshold freeze.

## Correctness gate

Require zero mismatches against the dense Python/native oracle across:

- every threshold arm;
- all 768 physical codewords;
- at least 10,000 seeded arbitrary, mixed, and input-order-permuted contexts;
- equal-score ties requiring the lower segment ID;
- reinforce, punish, drop, regrow, inert-live-count, and recertification paths;
- generation rollover and retryable output-capacity failure;
- full ordered mutation traces and post-step snapshot hashes;
- checkpoint rebuild with no serialized derived routing.

Corrupt route certification, pair-twin ticks, codebook hash, or threshold/overlap
preconditions must STOP or take the exact generic path before producing output.

## Performance and selection gate

Run at exact `S` and `P`, not an extrapolated 1k--32k store:

1. uniform control (`V` approximately 625k, `T` approximately 619k);
2. token 101 maximum-visits profile;
3. token 1164 false-heavy maximum-touched profile;
4. token 32 maximum-qualifier profile;
5. exact scheduled TRAIN weighting;
6. concentrated arbitrary-context adversary `V=P`, `T=S`.

Measure cold and warm median/p95 latency, visits or records/second, qualifier and
ordering time, mutation p95, post-create allocations, committed RSS, and
checkpoint rebuild time. The optimized intrusive path is the generic control.

The assembly index is selectable only if it has exact parity, no outcome
dependence, bounded allocation/fail-stop behavior, material p95 improvement
(target at least 1.5x where it adds module complexity), and a measured
integrated 95% upper bound below the 15-hour cap after adding physics, ordered
mutation, runner, and checkpoint costs. Recognition-only cap fit is
insufficient.

If assembly routing fails, Roaring becomes the next generic A/B candidate. If
both fail, the correct verdict is that final-density A3 is not runnable under
the current semantics; a transition lookup is not an acceptable escape hatch.
