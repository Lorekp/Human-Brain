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

This design was originally completed at `292ac10`, first audited after the
deep read-only runtime landed at `bd6a036`, and re-audited against committed
repository HEAD `e5a775ef30000d4b408ecac28cfdb0df702e846a`
(`refactor(native): make temporal read-only allocation-free`). The governing
committed implementation inputs are:

| Input | Governing commit or state |
| --- | --- |
| Canonical pair/reserve/Philox Python oracle | `fc2c4e3` and later committed source at HEAD |
| Intrusive mutable native temporal store | `3dd2c4f` and later committed source at HEAD |
| Fitted native GLIF3 temporal physics | `612b2f5` and later committed source at HEAD |
| Deep read-only physical runtime | `bd6a036` and prerequisite fixes through HEAD `e5a775e` |
| A3 recognition decision and final-density arithmetic | `292ac10` |

The audit through `e5a775e` establishes committed facts that were provisional
in the first version of this note: `segment_capacity` is independent of preload
length, `TemporalStore::runtime_owner_load` is the sole learned-load authority,
runtime construction accepts canonical rows with zero through ten live pairs,
and `GenericAfferents` rejects `base_cells_per_column < 2`. It also establishes
the production Random123 sampler, fixed-size context storage, construction-time
race scratch, typed categorized-cap/tick-overflow STOPs, and the stable compact
`TemporalPoisonCode` ABI. The numbered audit near the end records every
remaining gap and supersedes both the old working-tree hash inventory and the
intermediate `bd6a036` status assessment.

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
prevalidated append operation that supports 0–10 live pairs. Committed runtime
construction now supports that width, but it remains fallible and poison-capable
and therefore is not the commit-phase façade.

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

## Cross-language integer contract

The Rust module, block adapter, Python adapter, and checkpoint codec use one
numeric domain. Python must require `type(value) is int` (therefore rejecting
`bool`) and range-check before conversion. Rust may use `usize` only for a
checked local allocation index after decoding a fixed-width value; `usize` is
never an identity, serialized field, counter, or cross-language argument.
There is no truncation, saturation, wrapping conversion, negative sentinel, or
host-width-dependent value at the seam.

| Logical value | Canonical domain | Additional rule |
| --- | --- | --- |
| token ID, column ID, owner/pyramidal GID, physical GID, segment ID, normalized source, posting pair slot | `u32` | `u32::MAX` is permitted only as the internal empty-slot sentinel, never as a live ID |
| column count, maximum pyramidal cells, segment capacity, source count, codebook count, cohort count, next reserve cell, physical offsets/count | `u32` | all derived products and sums are checked in `u64` before conversion; segment capacity also satisfies `10 * capacity <= u32::MAX` |
| base cells per column, novel race cap, prediction quorum, match/active thresholds, context lengths, policy tags | `u8` | context lengths are only 0 or 22; thresholds are physical-contact counts in `1..=20` |
| owner load and configured owner-segment cap | `u8` | cap is in `1..=255`; a prospective load is computed wider, checked against the configured cap and 255, then converted |
| permanence ticks, spike/relay/simulated ticks, race-tick limit | `u16` | live permanence is `1..=500`, empty is exactly 0; tick arithmetic is checked before conversion |
| construction seed | `u32` | zero is valid |
| event count and learning step | `u64` | checked increment only; no wrap and no reset of `learning_step` |
| checkpoint total/body byte lengths | `u64` | checked against platform addressability before allocation or pointer formation |
| poison code and checkpoint version/header size | `u16` | only the enumerated values below are valid |
| exact `f64` identity words | `u64` | IEEE-754 `to_bits()` payload, little-endian; behavior inputs must be finite |

The current Rust `segment_capacity: usize` is therefore an internal-layout gap,
not the v1 interface: implementation must accept/encode a checked `u32` and
derive `usize` only after proving addressability. `column_offsets: Vec<usize>`
remains a derived, nonserialized implementation detail.

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
- stage-typed physics `InvalidInput`, no-spike, column multiple-spike, or column
  missing-basket-inhibition result;
- latency, relay-tick, event-counter, or learning-counter overflow;
- store, posting, canonical-row, or prepared-plan invariant conflict;
- Philox rejection-address exhaustion.

### Capacity poison

Only the stable poison latch changes for the failing token:

- a complete reserve cohort cannot fit;
- global segment capacity cannot fit the exact allocation plan;
- an owner would exceed configured capacity or 255 segments;
- any construction-time-preallocated runtime/store scratch or vector is
  unexpectedly too small;
- checked capacity/cursor/backing-length arithmetic overflow.

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

The fixed header is exactly 256 bytes and is encoded in this order. Offsets are
zero-based, inclusive byte ranges:

| Bytes | Field | Type | Rule |
| --- | --- | --- | --- |
| 0..7 | magic | `[u8; 8]` | ASCII `R3NTCP1` followed by one zero byte |
| 8..9 | format version | `u16` | exactly 1 |
| 10..11 | header bytes | `u16` | exactly 256 |
| 12..15 | reserved flags | `u32` | exactly zero |
| 16..23 | total bytes | `u64` | exact header plus body |
| 24..31 | body bytes | `u64` | exact formula below |
| 32..63 | protocol identity | `[u8; 32]` | exact identity contract below |
| 64..95 | source-bundle identity | `[u8; 32]` | exact identity contract below |
| 96..127 | circuit identity | `[u8; 32]` | exact identity contract below |
| 128..159 | body digest | `[u8; 32]` | SHA-256 of body bytes |
| 160..191 | semantic index digest | `[u8; 32]` | derived logical posting identity |
| 192..195 | segment count | `u32` | `S <= segment_capacity` |
| 196..199 | cohort count | `u32` | complete cohorts only |
| 200..203 | next reserve cell | `u32` | must equal the value derived from cohorts |
| 204..211 | event count | `u64` | logical transient counter |
| 212..219 | learning step | `u64` | persistent RNG address counter |
| 220 | prior-context length | `u8` | only 0 or 22 |
| 221 | winner-context length | `u8` | only 0 or 22 |
| 222..223 | poison code | `u16` | stable mapping below; zero means unpoisoned |
| 224..255 | reserved tail | `[u8; 32]` | exactly zero |

The size audit is
`8 + 2 + 2 + 4 + 8 + 8 + 5*32 + 4 + 4 + 4 + 8 + 8 + 1 + 1 + 2 + 32 = 256`.
No field moved and the body still begins at byte 256.

All integers are unsigned little-endian. Hash fields are their 32 digest bytes
in SHA-256 output order. The format has no host-alignment padding.

The stable `poison_code` mapping is the compact `#[repr(u16)]`
`TemporalPoisonCode` ABI committed at `e5a775e`. It is closed and exhaustive
for checkpoint v1:

| Code | Stable name | Exact terminal class |
| ---: | --- | --- |
| 0 | `Unpoisoned` | no terminal poison |
| 1 | `MultipleActiveOwners` | more than one active owner in one arriving column |
| 2 | `MultipleMatchingOwners` | more than one matching owner in one arriving column |
| 3 | `CategorizedRaceCapacity` | union of active and matching owners exceeds `novel_race_cap` before novel fill |
| 4 | `MalformedContextOrPolicy` | malformed paired context, unreachable slot order, or runtime source-policy violation |
| 5 | `ProximalPhysics` | any `TemporalPhysicsStop` from proximal singleton physics |
| 6 | `ProximalRelayPhysics` | any `TemporalPhysicsStop` from proximal-relay singleton physics |
| 7 | `ColumnRacePhysics` | any `TemporalPhysicsStop` from the coupled column race |
| 8 | `WinnerRelayPhysics` | any `TemporalPhysicsStop` from winner-relay singleton physics |
| 9 | `TickOrCounterOverflow` | proximal-relay tick, winner-relay tick, event-counter, or learning-counter overflow |
| 10 | `StoreOrPlanInvariant` | recognition, owner, row, pair, posting, canonical-suffix, or prepared-plan invariant |
| 11 | `PhiloxAddressExhaustion` | all `2^32` rejection-attempt addresses for one bounded draw are exhausted |
| 12 | `ReserveCellCapacity` | aggregate complete reserve cohorts do not fit below `max_pyramidal_cells` |
| 13 | `GlobalSegmentCapacity` | exact allocation plan does not fit global segment capacity |
| 14 | `OwnerSegmentCapacity` | prospective owner load exceeds the configured cap or the physical `u8` ceiling of 255 |
| 15 | `PreallocatedCapacityInvariant` | runtime/store scratch, cursor, backing length, or checked capacity arithmetic contradicts construction-time bounds |

Codes `1..=11` are scientific/integrity poisons; `12..=15` are capacity
poisons. Checkpoint import rejects every other `u16` value. `UnknownToken`,
external argument/output failures, and checkpoint validation/I/O failures are
retryable and therefore never receive a poison code.

The stable code deliberately retains less detail than the first caller-visible
STOP. The mapping is exact:

- `TemporalCircuitStop::Physics { stage, source }` maps only by the exhaustive
  `PhysicalStage`: proximal/proximal-relay/column-race/winner-relay become
  codes 5/6/7/8. This includes `TemporalPhysicsStop::InvalidInput(_)`,
  `NoPyramidalSpike`, `MultiplePyramidalSpikes`, and
  `BasketInhibitionMissing`; code selection never parses diagnostic text.
- `TickOrCounterOverflow` maps every exhaustive `TemporalOverflowStage` to 9.
  Mutation adds `LearningCounter` to that rich stage enum without changing the
  stable class. Relay-tick and event overflow are already typed this way at
  `e5a775e`, rather than being disguised as physics `InvalidInput`.
- `StoreRecognition` and `StoreOrPlanInvariant` map to 10. Learned row/posting
  conflicts and prepared-plan conflicts join the latter rich variant; no store
  numeric status is persisted directly.
- Philox, reserve, global-segment, owner, and preallocated-capacity failures map
  to 11, 12, 13, 14, and 15 respectively. Compute a prospective owner load in
  a wider integer first; both configured-cap failure and `>255` failure have
  stable code 14, while the rich first STOP distinguishes them if needed.

The `TemporalCircuitStop` match and its `PhysicalStage` submatch are exhaustive
with no catch-all arm, so a new top-level terminal class or physical stage
cannot compile until its stable mapping is chosen. Nested physics reasons
intentionally collapse to their physical-stage class, and all rich overflow
stages intentionally collapse to code 9; adding either kind of reason does not
version the checkpoint. The token-preflight order fixes which code wins when a
malformed plan violates more than one condition, independent of traversal
order.

The initial rich STOP remains the caller-visible diagnostic. Only this stable
terminal code persists because all later calls expose `POISONED`.

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

All three header identities are raw 32-byte SHA-256 outputs, not 64-byte hex
text. Import compares them to identities supplied by the validated runtime
factory before allocating the store; it never accepts checkpoint-provided
identities as the expected values.

#### Protocol identity

For R3 calibration v3, the protocol identity source is
`experiments/r3_pilot/calibration_v3_protocol.py`. Its exact preimage is:

```text
json.dumps(
    build_protocol(),
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
```

There is no BOM, indentation, trailing newline, or alternate JSON
canonicalizer. The identity is `SHA256(preimage).digest()`, exactly the raw
bytes represented by `protocol_sha256()`. A production run or checkpoint
export must first validate the same object with
`validate_protocol(value, require_runnable=True)`. The current preregistration
has `source_binding.runnable = false`; defining this identity does not make it
runnable, amend it, or authorize an artifact. Any later refreeze changes the
protocol identity and intentionally makes the old checkpoint incompatible.

#### Source-bundle identity

The source-bundle identity is the raw digest represented by the existing
`humanbrain.native.procedural_runtime_source_bundle_sha256()` result. It uses
the established `_framed_sha256` algorithm with namespace
`humanbrain-procedural-runtime-source-v1`:

```text
u64be(len(namespace_utf8)) || namespace_utf8
for each ordered (path, raw_payload):
    u64be(len(path_utf8)) || path_utf8
  || u64be(len(raw_payload)) || raw_payload
```

The ordered entries are exactly `_procedural_runtime_source_paths()`: the
fixed `_PROCEDURAL_RUNTIME_SOURCE_PATHS` tuple first, then the byte-stable
POSIX-path-sorted `_native_build_input_paths()` enumeration, with first
occurrence retained when a path repeats. Paths are UTF-8 repository-relative
POSIX strings. Payloads are the bounded raw file bytes; unlike the protocol's
per-file source pins, this digest performs no CRLF normalization. Symlinks in
the discovered native build inputs, missing fixed inputs or native build trees,
and per-file or aggregate size-limit violations fail before an identity exists.
The 64-character helper result is hex-decoded into the 32-byte header field.

#### Circuit-identity framing

The circuit identity uses the same unambiguous framing under the distinct
namespace `humanbrain-r3-temporal-circuit-v1`. Entry-name and payload lengths
are `u64` big-endian exactly as above. Entry names are the literal ASCII names
in the following table. Numeric payloads are fixed-width little-endian; no
payload contains alignment padding. The entry order is semantic and fixed:

| # | Entry name | Payload |
| ---: | --- | --- |
| 1 | `active_columns` | `u8`, exactly 11 |
| 2 | `context_cells` | `u8`, exactly 22 |
| 3 | `pairs_per_segment` | `u8`, exactly 10 |
| 4 | `physical_contacts_per_pair` | `u8`, exactly 2 |
| 5 | `context_policy` | `u8`: 1 = `BoundWinners`, 2 = `GenericAfferents`; 0 invalid |
| 6 | `plasticity_policy` | `u8`: 0 = growth-off, 1 = plastic; all other values invalid |
| 7 | `recognition_route` | `u8`: 0 = exact full scan, 1 = certified physical-source buckets |
| 8 | `column_count` | `u32` |
| 9 | `base_cells_per_column` | `u8` |
| 10 | `max_pyramidal_cells` | `u32` |
| 11 | `segment_capacity` | `u32` |
| 12 | `novel_race_cap` | `u8` |
| 13 | `max_segments_per_cell` | `u8` |
| 14 | `prediction_quorum` | `u8` |
| 15 | `construction_seed` | `u32` |
| 16 | `connected_permanence_ticks` | `u16` |
| 17 | `theta_match` | `u8`, physical-contact count |
| 18 | `theta_active` | `u8`, physical-contact count |
| 19 | `permanence_max_ticks` | `u16`, exactly 500 |
| 20 | `initial_permanence_ticks` | `u16` |
| 21 | `permanence_increment_ticks` | `u16` |
| 22 | `negative_decrement_ticks` | `u16` |
| 23 | `relay_offset` | `u32`, exactly `max_pyramidal_cells` |
| 24 | `basket_offset` | `u32`, exactly `2 * max_pyramidal_cells` |
| 25 | `proximal_offset` | `u32`, exactly `basket_offset + column_count` |
| 26 | `proximal_relay_offset` | `u32`, exactly `proximal_offset + column_count` |
| 27 | `physical_cell_count` | `u32`, exactly `proximal_relay_offset + column_count` |
| 28 | `source_gid_base` | `u32`, 0 for bound winners or `proximal_offset` for generic afferents |
| 29 | `source_count` | `u32`, `max_pyramidal_cells` for bound winners or `column_count` for generic afferents |
| 30 | `integration_step_ms` | exact finite `f64` bits as `u64`, currently `0x3fa999999999999a` (0.05 ms) |
| 31 | `race_ticks` | `u16`, currently 600 |
| 32 | `physics_f64` | fourteen exact `u64` words in the order below |
| 33 | `token_codebook` | exact variable payload below |
| 34 | `threshold_offsets_mv` | exact variable payload below |
| 35 | `recognition_certificate` | empty for route 0; one raw 32-byte digest for route 1 |

`physics_f64` contains no count and has this exact order:

| # | Quantity | Exact `f64` bits |
| ---: | --- | --- |
| 1 | leak nS | `0x4020f0a7d90553e6` |
| 2 | voltage-step scale | `0x3f33ea95a3d1e76b` |
| 3 | relative threshold mV | `0x4039a18e521fe42e` |
| 4 | NMDA decay | `0x3feff44e351ef7e1` |
| 5 | GABAA decay | `0x3fefccf5acbfae90` |
| 6 | novel base pA | `0x4081d00000000000` (570) |
| 7 | novel rank decrement pA | `0x4047800000000000` (47) |
| 8 | matching pA | `0x4083600000000000` (620) |
| 9 | active pA | `0x4084f00000000000` (670) |
| 10 | proximal pA | `0x4081d00000000000` (570) |
| 11 | relay pA | `0x40d3880000000000` (20,000) |
| 12 | homeostatic numerator pA | `0x4034000000000000` (20) |
| 13 | basket current pA | `0x41086a0000000000` (200,000) |
| 14 | basket inhibition pA | `0x407f400000000000` (500) |

Each word is emitted little-endian even though the table prints the usual
numeric hexadecimal form.

The `token_codebook` payload is `code_count:u32`, then `code_count` fixed
records sorted by ascending `token_id`; each record is `token_id:u32` followed
by exactly eleven `column_id:u32` values in ascending order. Counts and all IDs
are little-endian. Duplicate token IDs, duplicate eleven-column codes,
non-increasing columns, or an out-of-range column make construction fail before
hashing.

The `threshold_offsets_mv` payload is `physical_cell_count:u32` followed by one
exact finite `f64.to_bits():u64` word per physical GID in ascending GID order.
The count must equal entry 27. Signed zero is preserved because it is part of
the immutable input bits; NaN and infinity are invalid.

Initial column membership is canonical rather than another identity field:
column `c` initially owns exactly the ascending physical GIDs
`c * base_cells_per_column .. (c + 1) * base_cells_per_column`. The validated
factory rejects any supplied `column_members` that differs from this layout.
Later complete reserve cohorts are checkpoint body state and reconstruct from
`cohort_columns`; permitting arbitrary initial membership would require a new
circuit-identity field and therefore a new checkpoint version.

For recognition route 0, `recognition_certificate` has length zero. Route 1
uses this raw digest:

```text
SHA256(frame(
    "humanbrain-r3-physical-route-certificate-v1",
    ("logical_match_pairs", u8(ceil(theta_match / 2))),
    ("max_distinct_code_overlap", u8(recomputed_max_overlap)),
    ("token_codebook", exact_token_codebook_payload),
)).digest()
```

`frame` is the same `u64be` namespace/name/payload framing above. Import
recomputes the maximum intersection across every pair of distinct eleven-column
codes and requires `max_overlap < ceil(theta_match / 2)` before enabling route
1. Route 1 is valid only with `GenericAfferents`; `BoundWinners` requires route
0. Otherwise import rejects the circuit identity or uses a separately
constructed route-0 circuit; it never trusts a claimed certificate and never
serializes derived buckets. Defining both tags does not select an A3 policy or
prediction quorum.

Finally:

```text
circuit_identity = SHA256(circuit_preimage).digest()
```

The circuit preimage contains immutable behavior only. Recruited cohorts,
segments, contexts, counters, and the poison latch remain checkpoint body/header
state, not circuit identity.

#### Literal circuit-identity vector

The following mechanically specified route-0 fixture is independent of any
repository helper:

- entries 1--4 use their fixed values;
- policies are `BoundWinners` (1), plastic (1), and full scan (0);
- `column_count=11`, `base_cells_per_column=2`,
  `max_pyramidal_cells=22`, `segment_capacity=1`, `novel_race_cap=2`,
  `max_segments_per_cell=1`, `prediction_quorum=1`, and
  `construction_seed=0x01020304`;
- connected/initial/increment/decrement ticks are 250/255/50/2,
  `theta_match=18`, `theta_active=20`, and the maximum is 500;
- derived offsets/counts are 22, 44, 55, 66, and 77; source base/count are
  0 and 22;
- timing and `physics_f64` use the exact table above;
- the codebook is one record: token 7 maps to columns 0 through 10;
- threshold offsets are exactly 77 positive-zero `f64` words;
- the recognition certificate is empty.

This standalone CPython 3.11 program is the exact digest-construction
provenance. It imports no Human Brain module and contains every emitted field:

```python
from hashlib import sha256
from struct import pack


u8 = lambda value: pack("<B", value)
u16 = lambda value: pack("<H", value)
u32 = lambda value: pack("<I", value)
u64le = lambda value: pack("<Q", value)
u64be = lambda value: pack(">Q", value)

physics = [
    0x4020F0A7D90553E6, 0x3F33EA95A3D1E76B,
    0x4039A18E521FE42E, 0x3FEFF44E351EF7E1,
    0x3FEFCCF5ACBFAE90, 0x4081D00000000000,
    0x4047800000000000, 0x4083600000000000,
    0x4084F00000000000, 0x4081D00000000000,
    0x40D3880000000000, 0x4034000000000000,
    0x41086A0000000000, 0x407F400000000000,
]
codebook = u32(1) + u32(7) + b"".join(u32(i) for i in range(11))
thresholds = u32(77) + b"".join(u64le(0) for _ in range(77))
entries = [
    ("active_columns", u8(11)),
    ("context_cells", u8(22)),
    ("pairs_per_segment", u8(10)),
    ("physical_contacts_per_pair", u8(2)),
    ("context_policy", u8(1)),
    ("plasticity_policy", u8(1)),
    ("recognition_route", u8(0)),
    ("column_count", u32(11)),
    ("base_cells_per_column", u8(2)),
    ("max_pyramidal_cells", u32(22)),
    ("segment_capacity", u32(1)),
    ("novel_race_cap", u8(2)),
    ("max_segments_per_cell", u8(1)),
    ("prediction_quorum", u8(1)),
    ("construction_seed", u32(0x01020304)),
    ("connected_permanence_ticks", u16(250)),
    ("theta_match", u8(18)),
    ("theta_active", u8(20)),
    ("permanence_max_ticks", u16(500)),
    ("initial_permanence_ticks", u16(255)),
    ("permanence_increment_ticks", u16(50)),
    ("negative_decrement_ticks", u16(2)),
    ("relay_offset", u32(22)),
    ("basket_offset", u32(44)),
    ("proximal_offset", u32(55)),
    ("proximal_relay_offset", u32(66)),
    ("physical_cell_count", u32(77)),
    ("source_gid_base", u32(0)),
    ("source_count", u32(22)),
    ("integration_step_ms", u64le(0x3FA999999999999A)),
    ("race_ticks", u16(600)),
    ("physics_f64", b"".join(u64le(word) for word in physics)),
    ("token_codebook", codebook),
    ("threshold_offsets_mv", thresholds),
    ("recognition_certificate", b""),
]

namespace = b"humanbrain-r3-temporal-circuit-v1"
preimage = u64be(len(namespace)) + namespace
for name, payload in entries:
    name = name.encode("ascii")
    preimage += u64be(len(name)) + name + u64be(len(payload)) + payload

print(len(preimage), sha256(preimage).hexdigest())
print(preimage[:64].hex())
print(preimage[-64:].hex())
```

The framed preimage is exactly 2,064 bytes. Its first 64 bytes are:

```text
000000000000002168756d616e627261696e2d72332d74656d706f72616c2d63
6972637569742d7631000000000000000e6163746976655f636f6c756d6e7300
```

Its final 64 bytes are:

```text
0000000000000000000000000000000000000000000000000000000000000000
177265636f676e6974696f6e5f63657274696669636174650000000000000000
```

The expected SHA-256 is
`1cb6b08d2cc0c5c02e9c33f60aa42a12e941b3973f30e5241c957f1f8179ff9e`.
On 2026-08-27 this was reproduced independently by (1) a standalone CPython
3.11.15 `struct`/`hashlib` encoder and (2) a PowerShell/.NET
`BitConverter`/`SHA256.HashData` encoder. Neither imported Human Brain code;
both emitted the same 2,064-byte length, digest, first 64 bytes, and final 64
bytes shown here. The implementation test must rebuild this vector from the
literal fields, not paste the expected preimage.

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
8. Validate contexts against the exact runtime-reachable grammar below. Both
   are empty, or both are exactly 22; set equality or a permutation is never
   sufficient.
9. Construct a brand-new runtime and preallocated store off to the side. Rebuild
   column CSR, owner counts, and postings in segment-ID/slot order.
10. Compute and compare the semantic index digest, then canonically re-hash or
    re-export logical state.
11. Return or publish the new handle only after every check succeeds.

There is no in-place restore. Invalid input publishes nothing and cannot change
an existing runtime.

#### Runtime-reachable context grammar

For a nonempty checkpoint, find exactly one canonical codebook record whose
ordered columns are `C[0]..C[10]`. Runtime construction already guarantees
that no two token IDs share one code. Validate slots in index order:

For `BoundWinners`:

- `previous_context` must equal `previous_winner_context` byte-for-byte;
- for each `i in 0..11`, slot `i` is one committed pyramidal winner belonging
  to column `C[i]`;
- slot `11+i` is exactly `relay_offset + slot[i]`.

For `GenericAfferents`:

- `previous_context[i]` is exactly `proximal_offset + C[i]`;
- `previous_context[11+i]` is exactly `proximal_relay_offset + C[i]`;
- `previous_winner_context[i]` is one committed pyramidal winner belonging to
  column `C[i]`;
- `previous_winner_context[11+i]` is exactly
  `relay_offset + previous_winner_context[i]`.

Thus both stored contexts describe the same token code in its canonical sorted
column order, while only the generic policy permits the two contexts to differ.
The importer rejects reversed halves, interleaved source/relay pairs, a valid
set in the wrong slot order, relay twins paired to another slot, a winner from
another column, mixed token codes, or a bound-winner context differing from the
winner context. This is precisely what `e5a775e` can emit: first all eleven
sources in token-code order, then their eleven relays in the same order.

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

## Current layout audit against `e5a775e`

This is the current re-audit of all fifteen original
incompatibilities. `RESOLVED` means the committed read-only runtime now has the
required property; it does not claim mutation/checkpoint implementation.

1. **RESOLVED — independent segment capacity.**
   `TemporalCircuitConfig.segment_capacity` is independent of preload length,
   rejects zero and a preload larger than capacity, and is forwarded to
   `TemporalStoreConfig`. Preserve it through construction, circuit identity,
   and import. The separate `usize`-to-`u32` interface-width correction is
   frozen in the integer contract above.
2. **RESOLVED — store-authoritative owner load.** The runtime has no duplicate
   load vector and reads `TemporalStore::runtime_owner_load`. Mutation must not
   reintroduce a parallel cache.
3. **OPEN — explicit owner cap.** `owner_segment_counts: Vec<u8>` still imposes
   a hard 255 ceiling, while `TemporalCircuitConfig` has no
   `max_segments_per_cell`. Add the configured `u8` cap and preflight both
   rich failure reasons into stable poison code 14.
4. **OPEN — mutation state/config.** Initial, reinforcement, and punishment
   ticks; immutable plasticity policy; `learning_step`; reserve cursor; and
   reserve-cohort log do not exist in the runtime.
5. **RESOLVED — zero through ten live pairs at runtime construction.**
   `TemporalSegmentInit` holds variable vectors and
   `allocate_runtime_segment` accepts equal lengths in `0..=10`; focused tests
   cover empty and partial rows. The public diagnostic FFI still intentionally
   requires exactly ten and is not the checkpoint seam.
6. **OPEN — normalized checkpoint import façade.**
   `allocate_runtime_segment` accepts physical GIDs and normalizes them;
   checkpoint rows are already normalized. Add distinct purpose-named
   construction and normalized-import façades so restore cannot normalize
   twice.
7. **OPEN — all-active mutation view.** Best-owner diagnostics still cannot
   express the ascending all-active punishment pass. Preserve an internal
   all-active view or move that pass into the store.
8. **RESOLVED — production Random123 sampler.** `lemire_candidate`, bounded
   Philox selection, and allocation-free `sample_without_replacement` are
   production crate functions with exact known-answer and Python-oracle tests.
9. **RESOLVED — fixed contexts.** `previous_context` and
   `previous_winner_context` are fixed `[u32; 22]` arrays with explicit `u8`
   lengths; success and reset retain their addresses and perform no allocation.
10. **PARTIALLY RESOLVED — allocation-free races and typed failure staging.**
    Column-drive, novel-candidate, and coupled-race storage is reserved at
    construction; measured success, first-STOP, and already-poisoned paths
    allocate zero times. Categorized-cap and relay/event overflow are typed
    runtime STOPs with the stable mapping above. The runtime intentionally
    retains exact zero-drive candidates for parity; sparse zero-drive elision
    remains a separately proved performance optimization before the integrated
    cost gate, not a mutation correctness prerequisite.
11. **OPEN — no-fail store commit façade.** Existing `allocate_segment` can
    poison and returns `Result`; it cannot be used after commit begins. Add a
    prevalidated infallible append/rewrite façade over already-reserved vectors.
12. **PARTIALLY RESOLVED — recognition output capacity.** The runtime owns an
    owner-sized diagnostic buffer, so `TemporalStore::prepare` output-capacity
    failure is unreachable through `recognize`. `prepare` still advances
    scratch generation before checking its raw output capacity, and the future
    block output/checkpoint buffers do not exist. Preflight every external
    capacity before recognition or logical advance.
13. **PARTIALLY RESOLVED — one poison authority.** The runtime now retains one
    allocation-free stable `TemporalPoisonCode` latch and maps all committed
    rich runtime STOPs to the v1 ABI. The store still has its older poison
    mechanism. Runtime-owned learning must use nonpoisoning prevalidated store
    primitives so only the runtime latch remains authoritative after publish.
14. **OPEN — reserve CSR transaction.** Column CSR still cannot insert cohorts
    without shifting/allocation. Reserve an alternate CSR buffer at
    construction, rebuild once per recruiting token, and swap only at commit.
15. **CONFIRMED DESIGN CONSTRAINT — derived posting order.** Raw intrusive
    posting order remains history-dependent across rewrite and rebuild. Only
    logical rows and the sorted semantic index digest may be parity surfaces;
    checkpoint implementation and resume tests remain open.

One additional construction guard requested by the original audit is now
**RESOLVED**: `GenericAfferents` rejects `base_cells_per_column < 2` at
`e5a775e`, and committed fixtures exercise the exact base-width-two source and
context layout.

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

1. **DONE through `e5a775e`:** land and independently verify the read-only
   `TemporalCircuitRuntime`, production Random123 sampler, fixed contexts,
   allocation-free race path, typed STOP staging, and stable poison ABI.
2. **DONE in this revision:** re-audit every incompatibility above against the
   exact committed source hash.
3. Add red interface-level differential tests for LEARN semantics and atomic
   STOP behavior.
4. Build on the committed sampler/context/race prerequisites; preallocate the
   remaining mutation, checkpoint, block-output, and commit scratch. Treat
   zero-drive elision as a separate bit-exact performance change.
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
- exhaustively map every top-level rich STOP to stable codes 1--15, including
  all four stage-class physics `InvalidInput` paths, categorized-cap, both
  relay-tick overflows, both counters, and store/prepared-plan invariants; make
  a new top-level STOP or physical stage without a mapping a compile failure;
- compare canonical logical bytes before and after: identical except for the
  expected stable poison code;
- verify retryable failures leave even the latch unchanged and the identical
  retry succeeds;
- verify block completed-prefix behavior;
- prove insufficient external output capacity advances zero tokens;
- prove no vector pointer or capacity changes after runtime construction;
- prove no allocation-capable function is reachable from the commit phase.

### Checkpoint parity and corruption

- reproduce the 2,064-byte literal circuit-identity vector and independently
  test protocol/source framing boundaries, raw-vs-hex digest bytes, codebook
  order, threshold `f64` bits, and route-certificate recomputation;
- uninterrupted versus resumed execution across reset, cohort recruitment,
  allocation, reinforcement, punishment, drop-to-zero, and regrowth;
- run every source policy, plasticity policy, and prediction-quorum fixture
  without selecting a scientific winner among them;
- require identical emitted steps, contexts, counters, learning addresses,
  logical checkpoint bytes, and semantic index digests;
- corrupt truncation, trailing bytes, reserved bits, body hash, protocol/source/
  circuit identity, partial cohort, cursor, uncommitted owner, 256th owner
  segment, sentinel gap, zero live tick, nonzero empty tick, out-of-domain,
  unsorted or duplicate source, malformed pair, unknown poison code, and every
  forbidden context permutation/policy mixture in the runtime-reachable grammar;
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
