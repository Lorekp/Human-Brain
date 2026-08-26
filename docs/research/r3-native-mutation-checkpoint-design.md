# R3 native mutation transaction and canonical checkpoint design

**Status: CURRENT PRE-IMPLEMENTATION DESIGN.** This note specifies the next
native mutation and checkpoint slice for `TemporalCircuitRuntime`. It records
no executable implementation, protocol amendment, threshold or policy
selection, outcome, held-out access, capability evidence, wall-clock result, or
authorization to launch C1.

Prediction quorum and the A3 source policy remain explicit, configurable, and
**unfrozen**. In particular, this design does not choose between bound-winner
sources and generic afferent sources.

## Decision

1. Keep `TemporalCircuitRuntime` as the deep module. Its external interface
   accepts token IDs and a read-only or learning mode; callers must never inject
   winners, recognized classes, currents, reserve plans, or mutations.
2. Make one token a transaction with a complete preflight and one
   allocation-free, no-fail commit point. A failing token may latch a terminal
   STOP, but it must not partially advance learned state, recruited cohorts,
   contexts, or counters.
3. Keep `TemporalStore` authoritative for segment rows, postings, and owner
   loads. Mutation reaches it through a purpose-named crate-private façade that
   consumes a prevalidated plan; there is no second learned-load cache and no
   public mutation FFI.
4. Checkpoint only canonical logical state. Rebuild column CSR, owner loads,
   postings, stamps, and scratch into a new runtime, validate the rebuilt state,
   and publish the new handle only after the entire import succeeds.
5. Preserve exact Python-oracle pair, permanence, ordering, punishment, reserve,
   and Random123 semantics. Optimization may change representation and derived
   indexes, but not those semantics.

## Provenance and claim boundary

This design was completed against committed repository HEAD
`292ac10cebba93d9f586cb86d4d975ec97ac4312` (`docs(r3): decide A3 recognition
index`). The governing committed implementation inputs include:

| Input | Governing commit or state |
| --- | --- |
| Canonical pair/reserve/Philox Python oracle | `fc2c4e3` and later committed source at HEAD |
| Intrusive mutable native temporal store | `3dd2c4f` and later committed source at HEAD |
| Fitted native GLIF3 temporal physics | `612b2f5` and later committed source at HEAD |
| A3 recognition decision and final-density arithmetic | HEAD `292ac10` |

The inspection also read a live, uncommitted read-only runtime under active
development. These hashes identify only the observed working-tree snapshot;
they are not a source freeze and must not be treated as committed evidence:

| Inspected source | Observed SHA-256 |
| --- | --- |
| `src/humanbrain/_native_src/src/temporal.rs` | `d9fb2c833b18a7587e5011f787c5a4906af7cbe3fbd9604d5d62b2a98763dedb` |
| `src/humanbrain/_native_src/src/temporal_runtime.rs` | `0c7aec0f6272ad1deb0b37aa658ded23ae4b08e97206295a0609fd1a1c49df74` |
| `src/humanbrain/_native_src/src/random123.rs` | `0a229d96f549ae12f4dc8d7b6765a129fdd3cd0834a4aca0403e80d3a2397435` |
| `src/humanbrain/sparse_temporal_cortex.py` | `22e5ab6bb9e3b1bc701e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8` |
| `tests/test_sparse_temporal_cortex.py` | `b39c63462fc41f02e3f4ebdd259605f8a4c02a4c958cc3ba0b0df3cb65f22349` |

At that snapshot the live runtime already had an explicit `segment_capacity`
separate from preload length and used `TemporalStore::runtime_owner_load` as
the single learned-load authority. Those are required properties, not settled
committed facts. Every incompatibility in this note must be rechecked against
the eventual read-only-runtime commit before mutation implementation begins.

No outcome or held-out data were used. This is an implementation design under
the existing protocol, not a Change Note.

## Deep module and seam

The external seam remains deliberately small:

```text
advance_block(token_ids, mode, reset_before?, output)
    -> completed_prefix + ordered step results or STOP
```

`mode` distinguishes `READ_ONLY` from `LEARN`. An immutable plasticity policy
distinguishes enabled learning from a growth-off circuit. Prediction quorum and
context-source policy are immutable circuit configuration supplied at
construction or verified checkpoint import.

Everything below token decoding belongs behind this interface:

- context-source selection;
- segment recognition and owner classification;
- proximal, relay, pyramidal, and basket physics;
- reserve-cohort planning;
- winner election;
- reinforcement, punishment, regrowth, and allocation;
- counter and context advancement;
- capacity and scientific STOP classification;
- canonical checkpoint export/import.

`PreparedLearnStep` is a private implementation detail, not a caller-visible
interface. Tests may exercise private store primitives inside the module, but
the authoritative behavioral test surface is the runtime interface.

## One-token transaction

### Persistent logical state

The transaction protects all of the following as one logical state:

- recruited full reserve cohorts and `next_reserve_cell`;
- segment count, owner, normalized sources, and permanence ticks;
- owner segment loads;
- `learning_step`;
- previous policy-selected context;
- previous winner context;
- `event_count`;
- the terminal poison latch.

Recognition generations, stamps, touched lists, counters, diagnostics, race
scratch, candidate scratch, and alternate CSR build buffers are nonlogical
scratch. They may change during a failed preflight because they are not
observable after a terminal latch and are never checkpointed.

### Block-level preflight

Before reset, recognition, or physics:

1. validate mode, pointers, lengths, and mutually consistent arguments;
2. validate output capacity for the complete requested block or define an
   explicit bounded-prefix interface;
3. reject an already-poisoned runtime;
4. validate the first token lookup needed to begin work.

External output capacity must never be discovered after a token has advanced.
An explicit reset operation is its own transaction. If a block interface uses
`reset_before`, use empty effective context while staging the first token and
commit the reset with that token; failure must not leave a half-applied reset.

### Token preflight order

For each token, perform the following exact order.

#### 1. Token and effective context

- Reject an unknown token as retryable and non-poisoning.
- Resolve the token to its eleven sorted unique columns.
- Select the effective prior context, including a staged reset boundary when
  applicable.
- Validate the context's source/relay pairing and the configured source domain.

#### 2. Recognition

- Call native store recognition on the eleven logical sources in the first half
  of the 22-cell context.
- Preserve access to all touched segment IDs and their potential/connected
  counts for this recognition generation.
- Keep owner best-active and best-matching diagnostics for race classification.

Punishment requires **every active segment**, not merely the best active
segment reported for each owner. The store must provide an internal sorted
active iterator/view or perform the active-segment mutation pass itself.

#### 3. Proximal physics

- Stage all eleven proximal spikes.
- Stage all eleven proximal-relay spikes and checked absolute relay ticks.
- Do not modify contexts or counters.

Any physical no-spike, multiple-spike, missing-inhibition, invalid latency, or
tick-overflow result goes directly to the scientific STOP path.

#### 4. Global reserve-cohort plan

This stage runs only for `LEARN` with plasticity enabled and nonempty prior
context.

For each arriving column in frozen token-code order:

1. inspect recognition among its currently committed cells;
2. if no owner is active or matching and every current cell has reached the
   configured owner segment cap, request one complete `novel_race_cap` cohort;
3. assign sequential physical GIDs from the staged reserve cursor;
4. give every planned cell authoritative staged load zero.

Preflight the aggregate request across all eleven columns. The full plan must
fit below `max_pyramidal_cells`; a partial cohort is forbidden. Rebuild the
prospective column CSR once into a construction-time-reserved alternate buffer.
No membership is published yet.

#### 5. Classification and physical races

For each arriving column:

- reject more than one active owner;
- otherwise reject more than one matching owner;
- reject any categorized candidate set that violates the frozen race cap;
- construct the candidate set from committed cells plus any staged cohort;
- use authoritative store load for committed cells and load zero for staged
  cells;
- run the coupled fitted-cell/basket race and stage winner, basket,
  inhibition-onset, and simulated-tick results.

After all column races, stage all winner-relay spikes and both 22-cell next
contexts. Preflight `event_count + 1` with checked arithmetic. Prediction is a
pure staged result based on the explicit, still-unfrozen `prediction_quorum`.

#### 6. Exact learning plan

This stage runs only for `LEARN` with plasticity enabled. Use the Python logical
order, not posting traversal order:

1. Iterate all active segment IDs in ascending order.
2. Reinforce a segment when it is the confirmed active segment of a current
   winner.
3. Punish an unconfirmed active segment only if its owner is not any current
   winner. Other active segments owned by a correct winner are not punished.
4. Then process the eleven token columns in column order.
5. If that winner had no active segment but has a best matching segment,
   reinforce that matching segment.
6. Otherwise allocate a segment for the winner only when prior context is
   nonempty.

No segment may receive two mutations in one token. Precompute a compact set of
confirmed IDs and winner owners so this is checked before commit.

#### 7. Capacity, arithmetic, and RNG preflight

Before the first mutation:

- count the exact number of allocations;
- check `segment_count + allocations <= segment_capacity`;
- project every affected owner count and check both configured
  `max_segments_per_cell` and the physical `u8` ceiling of 255;
- validate every segment ID, owner, source row, and tick row referenced by the
  plan;
- precompute the common initial-allocation selection mask when any allocation
  occurs;
- precompute each reinforced segment's regrowth mask;
- reject Philox rejection-address exhaustion;
- preflight `learning_step + 1`;
- prove all store vectors, fixed contexts, mutation scratch, and alternate CSR
  buffers have sufficient construction-time capacity.

The old `learning_step` is the address for every allocation and regrowth draw
in this token. The increment is not visible until commit.

### The single no-fail commit

After all preflight stages succeed, enter one allocation-free commit that
cannot return an error:

1. swap in the prospective CSR and advance `next_reserve_cell`, if cohorts were
   planned;
2. apply active-segment reinforcement and punishment in ascending segment-ID
   order;
3. apply matching reinforcement and segment allocation in token-column order;
4. update store-authoritative owner loads for successful appends only;
5. set `learning_step = old_learning_step + 1`;
6. publish both contexts and `event_count`;
7. return the fully staged physical result.

Everything after entry returns `()` and performs no allocation, capacity check,
fallible conversion, hash, I/O, or caller callback. Bounds and capacity facts
are established by construction and preflight. A violated internal assertion
there is a process-level implementation defect; returning a partially mutated
runtime is not an allowed recovery path.

An exclusive Rust `&mut` borrow and the adapter's runtime lock prevent another
step or checkpoint from observing mid-commit state.

### Block failure semantics

Successful earlier tokens in a block remain committed. The failing token makes
no logical advance except a terminal poison latch when its STOP class requires
one. Return the completed-prefix count and the first STOP. Subsequent calls on a
latched runtime return `POISONED` without work.

## Canonical segment mutation

### Logical rows

Every segment is exactly one fixed-width canonical row containing zero through
ten live logical source/relay pairs:

- live entries occupy a prefix;
- normalized sources are strictly increasing and unique;
- each live tick is in `1..=500`;
- every unused suffix entry is `source = u32::MAX`, `tick = 0`;
- zero is never a live permanence tick;
- one shared tick controls both physical contacts in the implicit source/relay
  pair.

Bound-winner rows normalize physical winner IDs from GID base zero. Generic
afferent rows normalize proximal IDs by subtracting `proximal_offset`. The two
policies are different configurations of the same logical row; neither is
selected here.

Segments with zero live pairs remain allocated. They continue to consume one
global segment slot and one owner load, matching the Python oracle.

### Reinforcement

For each old live pair:

- if its normalized source is in active prior context, set the tick to
  `min(500, old + permanence_increment_ticks)`;
- otherwise set it to
  `max(0, old - permanence_increment_ticks)`;
- remove the pair if the result is zero.

Then fill missing positions, up to ten, from sorted
`active_context - retained_sources` using the exact regrowth sampler. New pairs
receive `initial_permanence_ticks`. Sort the final sources, carrying their
paired ticks, before rewrite. The sample size is
`min(missing_slots, candidate_count)`; run the frozen sampler for that exact
size even when it selects the entire candidate population.

### Punishment

Punishment affects a pair only when:

1. its source is in active prior context; and
2. its **pre-mutation** tick is at least
   `connected_permanence_ticks`.

Subtract `negative_decrement_ticks`, clamp at zero, remove zero pairs, retain
all others exactly, and never regrow during punishment. Sort the surviving row
canonically.

### Allocation

Allocation occurs only for a winner with neither an active nor matching
segment and only when prior context is nonempty. Select ten positions from the
eleven logical context sources. Every allocation in the same token uses the
same selected positions and writes `initial_permanence_ticks` for all ten
pairs.

### Exact Random123 addresses

Use the existing Philox2x64-10 numeric word order and exact Lemire rejection.

Initial allocation:

```text
INIT_DOMAIN = 0x52334630494e4954
key          = construction_seed XOR INIT_DOMAIN
counter_0    = preincrement learning_step
counter_1    = (rejection_attempt << 32) | draw
word         = philox2x64_10(counter_0, counter_1, key).0
```

Regrowth derives a segment-specific sampler key:

```text
RGRW_DOMAIN = 0x5233463052475257
key = philox2x64_10(
    owner,
    segment_id,
    construction_seed XOR RGRW_DOMAIN,
).0
```

The sampler then uses the same counter scheme and the same preincrement
`learning_step`. Map each word into the remaining population with exact Lemire
low-tail rejection, perform partial Fisher-Yates without replacement, and sort
the selected positions. Exhaustion of the 32-bit rejection-attempt address is
a scientific/integrity STOP, never a modulo fallback.

`learning_step` increments exactly once after every successful `LEARN` token
with plasticity enabled, including a boundary/empty-context token and a token
whose plan contains no segment edits. It does not increment in `READ_ONLY` or
growth-off mode. Transient reset does not reset it.

## Store mutation façade

`TemporalStore::rewrite_segment_pairs_canonical` is already the correct final
unlink/rewrite/relink mechanism if, and only if, its caller has prevalidated:

- the segment ID;
- equal source/tick lengths in `0..=10`;
- normalized in-domain, strictly increasing sources;
- live ticks in `1..=500`;
- every backing-array capacity and posting invariant.

The primitive unlinks existing live slots, rewrites the fixed ten-slot row, and
appends new postings using preallocated arrays. It should remain private behind
a purpose-named crate-private no-fail store façade. The façade also needs a
prevalidated append operation that supports 0–10 live pairs; the existing
constructor/import allocation path requiring exactly ten is insufficient.

No clone-sized rollback is necessary:

- punishment recomputes one replacement row at a time in stack
  `[u32; 10]`/`[u16; 10]` storage while walking the frozen sorted active list;
- reinforcement/regrowth needs at most eleven small staged records;
- allocation needs at most eleven owner records and one common initial mask;
- rewriting postings does not invalidate the already-materialized touched
  segment IDs or pre-mutation recognition counters.

If the A3 recognition optimization removes the whole-touched-set sort, it must
still materialize active segment IDs in ascending order for this mutation
contract. Raw posting order is never semantic.

## STOP taxonomy and state advancement

### Retryable, non-poisoning

The failing token or operation makes no state change:

- unknown token;
- invalid external mode, pointer, length, or argument combination;
- insufficient external step-output or checkpoint-output capacity;
- checkpoint destination collision or transient I/O failure;
- checkpoint magic, version, length, hash, identity, or structural rejection;
- allocation failure while constructing a replacement runtime before publish.

Retrying the same valid operation against the same runtime must produce the
same result.

### Scientific or integrity poison

Only the stable poison latch changes for the failing token:

- multiple active owners in an arriving column;
- multiple matching owners in an arriving column;
- categorized candidates exceeding the configured physical race cap;
- malformed internal paired context or policy-source violation;
- proximal, proximal-relay, column-race, or winner-relay no-spike,
  multiple-spike, or missing-basket-inhibition result;
- latency, relay-tick, event-counter, or learning-counter overflow;
- store, posting, canonical-row, or prepared-plan invariant conflict;
- Philox rejection-address exhaustion.

### Capacity poison

Only the stable poison latch changes for the failing token:

- a complete reserve cohort cannot fit;
- global segment capacity cannot fit the exact allocation plan;
- an owner would exceed configured capacity or 255 segments;
- any construction-time-preallocated runtime/store scratch or vector is
  unexpectedly too small.

The last class should be unreachable after correct construction, but must fail
closed if detected during preflight.

### Advancement matrix

| Operation result | Contexts / `event_count` | Cohorts / segments / owner loads | `learning_step` | Poison latch |
| --- | --- | --- | --- | --- |
| Successful `READ_ONLY` | advance | unchanged | unchanged | unchanged |
| Successful growth-off `LEARN` | advance | unchanged | unchanged | unchanged |
| Successful plastic `LEARN` | advance | commit plan | increment once | unchanged |
| Retryable failure | unchanged | unchanged | unchanged | unchanged |
| Scientific/capacity failure | unchanged | unchanged | unchanged | latch stable code |
| Already poisoned | unchanged | unchanged | unchanged | unchanged |
| Explicit transient reset | clear / zero | unchanged | unchanged | unchanged |

Store recognition generations and scratch are intentionally absent from this
table because they are nonlogical and nonserialized.

## Canonical logical checkpoint

### Format rules

Use one versioned, fixed-order, little-endian binary format. Version 1 has no
backward-compatibility reader and no optional extension records. All reserved
bytes must be zero. Integer arithmetic used to compute lengths must be checked
before allocation or pointer formation.

The fixed header is exactly 256 bytes and is encoded in this order:

| Field | Type | Rule |
| --- | --- | --- |
| magic | `[u8; 8]` | ASCII `R3NTCP1` followed by one zero byte |
| format version | `u16` | exactly 1 |
| header bytes | `u16` | exactly 256 |
| reserved flags | `u32` | exactly zero |
| total bytes | `u64` | exact header plus body |
| body bytes | `u64` | exact formula below |
| protocol identity | `[u8; 32]` | SHA-256 of governing protocol inputs |
| source-bundle identity | `[u8; 32]` | Gate-3 executable-source bundle SHA-256 |
| circuit identity | `[u8; 32]` | SHA-256 of canonical immutable behavior inputs |
| body digest | `[u8; 32]` | SHA-256 of body bytes |
| semantic index digest | `[u8; 32]` | derived logical posting identity |
| segment count | `u32` | `S <= segment_capacity` |
| cohort count | `u32` | complete cohorts only |
| next reserve cell | `u32` | must equal the value derived from cohorts |
| event count | `u64` | logical transient counter |
| learning step | `u64` | persistent RNG address counter |
| prior-context length | `u8` | only 0 or 22 |
| winner-context length | `u8` | only 0 or 22 |
| poison code | `u16` | stable code; zero means unpoisoned |
| reserved tail | `[u8; 32]` | exactly zero; completes byte 256 |

All integers are unsigned little-endian. Hash fields are their 32 digest bytes
in SHA-256 output order. The format has no host-alignment padding.

The stable `poison_code` values are:

| Code | Latched class |
| ---: | --- |
| 0 | unpoisoned |
| 1 | multiple active owners |
| 2 | multiple matching owners |
| 3 | categorized candidates exceed race cap |
| 4 | malformed paired context or source policy |
| 5 | proximal physics |
| 6 | proximal-relay physics |
| 7 | column-race physics |
| 8 | winner-relay physics |
| 9 | tick or counter overflow |
| 10 | store, posting, row, or prepared-plan invariant |
| 11 | Philox rejection-address exhaustion |
| 12 | reserve-cell capacity |
| 13 | global segment capacity |
| 14 | owner segment capacity |
| 15 | preallocated runtime/store capacity invariant |

The initial rich STOP remains the caller-visible diagnostic. Only this stable
terminal class persists because all later calls expose `POISONED`.

The body is the concatenation of these canonical logical arrays:

1. `cohort_columns[u32; cohort_count]` in global reserve-allocation order;
2. `previous_context[u32; prior_context_length]`, order preserved;
3. `previous_winner_context[u32; winner_context_length]`, order preserved;
4. `segment_owner[u32; S]` in implicit segment-ID order;
5. `segment_source[u32; S*10]` in row-major slot order;
6. `segment_ticks[u16; S*10]` in row-major slot order.

Thus:

```text
body_bytes =
    4 * cohort_count
  + 4 * prior_context_length
  + 4 * winner_context_length
  + 64 * segment_count
```

There is no live-count array. The canonical sentinel suffix makes live count
derivable and independently validates row integrity.

### Identity hashes

The circuit identity covers every immutable behavior-affecting input in a
canonical fixed order, including:

- all population dimensions, capacities, offsets, thresholds, permanence
  constants, construction seed, and physical current/timing constants;
- exact source policy, plasticity policy, and prediction quorum;
- the sorted token codebook and every eleven-column code;
- every fitted threshold-offset `f64` as its exact IEEE-754 bits;
- any exact codebook-overlap certificate required by a selected derived
  recognition path.

The runtime checkpoint deliberately does not know TRAIN corpus position. The
outer R3 runner envelope must separately bind TRAIN identity, book/pass/arm and
stream position, cumulative elapsed time, protocol/source identities, and the
ordered winner-stream prefix digest.

The semantic index digest is SHA-256 over live logical posting triples sorted
lexicographically by `(normalized_source, segment_id, slot)`, each encoded as
three little-endian `u32` values. It is derived from canonical rows, not from
intrusive traversal order.

### Never serialized

Do not serialize:

- source heads, tails, live counts, `next_pair`, or `prev_pair`;
- owner segment-count caches;
- column offsets or column-cell CSR;
- source, segment, or owner stamps and recognition generation;
- touched segments, touched owners, potential/connected counts, best-segment
  diagnostics, race candidates, physics scratch, or alternate CSR scratch;
- immutable config, codebook, threshold arrays, or derived route buckets.

Intrusive posting order changes when a rewrite unlinks and appends a pair at a
source-list tail. It is history-dependent but behaviorally irrelevant because
recognition uses integer counts and explicit segment-ID tie/order rules. Raw
posting bytes must therefore never become checkpoint identity.

### Import validation order

Import is a build-and-publish transaction:

1. Validate magic, version, header size, exact total/body lengths, reserved
   zeros, and checked length arithmetic.
2. Compare protocol, source-bundle, and circuit identities before allocating a
   multi-gigabyte store.
3. Verify the body SHA-256 by streaming.
4. Validate segment/cohort counts, cursor bounds, context lengths, counters, and
   poison code.
5. Validate every cohort column, reconstruct deterministic base cells plus each
   complete `novel_race_cap` cohort, and require
   `next_reserve_cell = base_total + cohort_count * novel_race_cap <=
   max_pyramidal_cells`.
6. Validate every segment owner is committed, count owners with checked
   arithmetic, and reject any owner above configured cap or 255.
7. Validate every row's sentinel suffix, strictly increasing normalized live
   sources, source domain, and ticks in `1..=500`.
8. Validate contexts are both empty or both exactly 22. Verify unique first
   halves, exact source/relay pairing, policy-correct domains, committed winner
   cells, and that both contexts describe the same eleven codebook columns.
9. Construct a brand-new runtime and preallocated store off to the side. Rebuild
   column CSR, owner counts, and postings in segment-ID/slot order.
10. Compute and compare the semantic index digest, then canonically re-hash or
    re-export logical state.
11. Return or publish the new handle only after every check succeeds.

There is no in-place restore. Invalid input publishes nothing and cannot change
an existing runtime.

### Export and publish

Checkpoint export holds the runtime lock and streams canonical rows directly to
an exclusive new temporary file or caller-provided seekable sink while hashing.
It must not clone the learned store into another 1.5 GB buffer. After body and
header hashes are complete, flush and fsync the file and its directory, then
atomically rename without overwriting an existing destination. Destination
collision or I/O failure is retryable and does not poison or roll back neuronal
state.

A poisoned runtime may be exported; the stable latch code must be preserved so
restore cannot clear a terminal STOP. Rich one-shot diagnostic payloads need
not be persistent because subsequent calls expose only `POISONED`.

## Current layout incompatibilities to resolve

These observations are provisional because the read-only runtime was
uncommitted while this note was written. Re-audit them after it lands.

1. The live `segment_capacity` field is correctly independent of preload
   length. Preserve it through construction, checkpoint identity, and import.
2. The live runtime correctly removed a duplicate owner-load vector and reads
   `TemporalStore::runtime_owner_load`. Keep the store authoritative; mutation
   must not reintroduce a parallel load cache.
3. `owner_segment_counts: Vec<u8>` imposes a hard 255-segment owner ceiling.
   Add explicit `max_segments_per_cell <= 255` configuration and preflight the
   configured cap rather than relying only on overflow.
4. Runtime configuration still needs initial, reinforcement, and punishment
   tick constants; immutable plasticity policy; `learning_step`; and reserve
   cursor state.
5. `TemporalSegmentInit` and the current runtime import path require exactly ten
   pairs. Checkpoint restore and learned mutation need canonical 0–10 rows.
6. `allocate_runtime_segment` accepts physical GIDs and normalizes them, whereas
   checkpoint rows are already normalized. Use distinct purpose-named
   construction and normalized-import façades to prevent double normalization.
7. Best-owner diagnostics cannot express the all-active punishment pass.
   Preserve an internal all-active view or move that pass into the store.
8. `lemire_candidate`, bounded Philox selection, and
   `sample_without_replacement` are currently test-only. Promote their exact,
   allocation-free implementations into production before learning.
9. `previous_context` and `previous_winner_context` are `Vec<u32>` and can
   allocate at the current commit point. Replace them with fixed `[u32; 22]`
   storage plus explicit presence/length.
10. Column-drive and physical-race paths allocate candidate/result vectors.
    Replace them with fixed top-eight or construction-time-reserved scratch and
    exact zero-drive elision before the integrated cost gate.
11. Existing `allocate_segment` can poison and returns `Result`; it cannot be
    used after commit begins. Add a prevalidated infallible append façade over
    already-reserved vectors.
12. `TemporalStore::prepare` advances scratch generation before it can report
    output-capacity failure. The runtime's owner-sized scratch makes that error
    unreachable; any later external output path must preflight capacity before
    recognition.
13. Runtime and store currently have separate poison mechanisms. Runtime-owned
    learning must use non-poisoning prevalidated store primitives and retain one
    stable runtime latch as the transaction authority.
14. Column CSR cannot insert reserve cohorts in place without shifting or
    allocation. Reserve one alternate CSR buffer at construction, rebuild once
    per recruiting token, and swap only at commit.
15. Raw intrusive posting order is not stable across uninterrupted and rebuilt
    executions. Only logical rows and the sorted semantic index digest may be
    parity surfaces.

## Memory and cost implications

At the current final-density A3 segment count `S = 23,286,406`, the checkpoint
segment body omits derived owner loads and indexes:

```text
owners       4S =   93,145,624 bytes
sources     40S =  931,456,240 bytes
ticks       20S =  465,728,120 bytes
                 -------------------
logical SoA     = 1,490,329,984 bytes
```

The final reserve population has
`(114,400 - 32,768) / 8 = 10,204` cohorts, so its column log is only 40,816
bytes. Two populated 22-cell contexts add at most 176 bytes plus the fixed
header. The checkpoint is therefore approximately 1.49037 GB at full segment
capacity.

For comparison, the requested current native intrusive-store payload is about
3.590 GB for A1 and 3.588 GB for A3 before physics and adapter diagnostics. The
checkpoint must never coexist with a clone-sized rollback buffer during normal
mutation. A single alternate column CSR buffer costs roughly another 0.5 MB at
C1 dimensions; both CSR copies remain around 1 MB.

Mutation-plan memory is O(11), apart from the recognition scratch already
required to identify active segments. Punishment is linear only in recognized
active segments and allocates nothing. Rebuilding CSR once per recruiting token
is linear in committed cells plus columns; cohort recruitment is bounded and
all cohorts for a token share that one rebuild.

Checkpoint export is O(S*10) streaming I/O. Import is O(S*10 + source domain +
owner domain) and rewrites every live posting. Both must be benchmarked at exact
final density. Ordinary export needs only the live store plus a small I/O
buffer; import creates one new store and publishes it after validation rather
than cloning an existing runtime.

## Implementation sequence

1. Land and independently verify the read-only `TemporalCircuitRuntime`.
2. Re-audit every incompatibility above against its committed source hash.
3. Add red interface-level differential tests for LEARN semantics and atomic
   STOP behavior.
4. Promote the exact Random123 sampler and remove all possible post-construction
   allocations from contexts, races, mutation scratch, and commit paths.
5. Add explicit plasticity/mutation configuration, `learning_step`, reserve
   cursor, store-authoritative active iteration, pair views, and no-fail
   rewrite/append façade.
6. Implement exact mutation first without reserve recruitment and prove parity.
7. Add aggregate reserve planning and the alternate-CSR commit.
8. Add the canonical checkpoint codec, streaming export, new-handle import, and
   semantic index digest.
9. Add the block adapter and Python seam; remove obsolete shallow mutation or
   restore paths rather than retaining compatibility layers.
10. Refresh Gate-3/source binding, then run q=4, q=255, and R3-F0 fixtures.
11. Only after correctness, resume parity, post-create allocation, and
    final-density cost gates pass should any later Change Note consider C1.

## Decisive tests

### Mutation parity

- literal INIT and RGRW Philox vectors, Lemire rejection attempts, and sorted
  without-replacement positions;
- proof that all allocations in one token share initial positions and every
  regrowth uses the old `learning_step` with its segment-specific key;
- tiny exact allocation, reinforcement, punishment, regrowth, and mixed-arm
  traces against the Python oracle;
- exhaustive 0–10 live-pair rows and randomized canonical rewrite comparison
  against a dense oracle, including no stale postings;
- active non-best segment punishment and the correct-winner-owner exemption;
- empty-context success and one-and-only-one learning-step increment;
- learned-state hash invariance for `READ_ONLY` and growth-off.

### Atomicity and STOPs

- inject every scientific and capacity failure at its preflight point;
- compare canonical logical bytes before and after: identical except for the
  expected stable poison code;
- verify retryable failures leave even the latch unchanged and the identical
  retry succeeds;
- verify block completed-prefix behavior;
- prove insufficient external output capacity advances zero tokens;
- prove no vector pointer or capacity changes after runtime construction;
- prove no allocation-capable function is reachable from the commit phase.

### Checkpoint parity and corruption

- uninterrupted versus resumed execution across reset, cohort recruitment,
  allocation, reinforcement, punishment, drop-to-zero, and regrowth;
- run every source policy, plasticity policy, and prediction-quorum fixture
  without selecting a scientific winner among them;
- require identical emitted steps, contexts, counters, learning addresses,
  logical checkpoint bytes, and semantic index digests;
- corrupt truncation, trailing bytes, reserved bits, body hash, protocol/source/
  circuit identity, partial cohort, cursor, uncommitted owner, 256th owner
  segment, sentinel gap, zero live tick, nonzero empty tick, out-of-domain,
  unsorted or duplicate source, malformed pair, and malformed context;
- assert every corrupt import publishes no handle and cannot modify an existing
  runtime;
- restore poisoned checkpoints and prove the latch cannot be cleared.

### Cost gates

- benchmark mutation and recognition together at exact A1 and A3 production
  shapes using frozen non-outcome fixtures;
- measure cold/warm median and p95, active-segment mutation cost, post-create
  allocations, RSS, checkpoint streaming throughput, and index rebuild time;
- include fresh and resumed runs and exact final-density segment count;
- retain the optimized exact generic path as the reference for any conditional
  recognition index.

These tests establish implementation correctness and cost only. They do not
spend C1, select prediction quorum or A3 source policy, amend the protocol, or
produce capability evidence.
