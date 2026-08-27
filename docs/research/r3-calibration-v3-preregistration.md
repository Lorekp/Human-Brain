# R3 electrical, temporal-code, and capacity calibration v3

**Date:** 2026-08-26
**Status:** **NOT RUNNABLE — PRE-OUTCOME PREREGISTRATION**
**Machine-readable authority:**
[`experiments/r3_pilot/calibration_v3_protocol.py`](../../experiments/r3_pilot/calibration_v3_protocol.py)

This document freezes the repair protocol discovered after the reserve audit of
the v2 temporal-code calibration. It authorizes no outcome-bearing replay and
does not select a threshold. The committed v1/v2 electrical and temporal JSON
artifacts remain historical evidence bound to `df26cc2`; they do not justify a
C1-scale threshold after the cohort allocator and exact-pair changes.

Schema 4 is deliberately fail-closed. It pins the explicit-threshold Python
oracle, the schema-7 TRAIN-only capacity artifact, and the governing nested
Gate-3 manifest at
`benchmarks/procedural_runtime_gate3_slice_protocol_v1.json`, but reports
`runnable=false`. Native source closure is governed transitively by that nested
manifest; schema 4 does not duplicate its native-file digests. Its exact
blocker is: "Official calibration-v3 screen execution, checkpoint/resume
orchestration, and immutable result publication remain disabled behind
`_execute_frozen_plan`." `--check-runnable` must therefore continue to fail.

Every source digest uses one cross-checkout rule: SHA-256 of raw bytes after
CRLF-to-LF normalization; a lone CR is invalid. `--check-draft` resolves every
literal `commit:path` with Git and verifies that canonical blob digest while
tolerating live working-tree drift. `--check-runnable` must additionally match
the live canonical bytes. Thus an `autocrlf` checkout cannot invalidate a
sound binding, and a dirty checkout cannot become runnable.

## 1. Candidate selection and seed separation

Candidate arms, in immutable preference order, are:

1. `(theta_match=10, theta_active=13)`
2. `(16, 18)`
3. `(18, 20)`
4. `(20, 20)`

The 20/20 arm is included a priori because source/relay contacts change in
pairs: 19/20 adds no distinct collision boundary. Failure of 20/20 means a
threshold-only repair is unavailable.

- Electrical/q=4/q=255 development seeds: `3, 7, 17, 29`.
- Electrical/q=4/q=255 independent confirmation seeds: `43, 59, 83, 101`.
- R3-F0 uses only the governing C1 construction seed `17`; multi-seed C1
  confirmation remains explicitly out of scope.
- Run electrical v3 first, q=4 for every candidate arm, both q=255 regimes for
  every q=4-passing arm, and TRAIN-order R3-F0 for fixture-passing arms in the
  listed order.
- Select the first arm passing every development gate.
- Run only that arm's q=4/q=255 screens on independent confirmation seeds.
- On their pass, run the selected arm's second fresh seed-17 R3-F0 replay plus
  resumes from the frozen 25/50/75% positions. Any confirmation failure rejects
  v3 without reopening another arm.
- CN-2 may write the selected literals only after the confirmed PASS.

### 1.1 Immutable artifact namespace

All possible destinations are expanded literally in the machine-readable
manifest under `experiments/r3_pilot/calibration_v3_artifacts`: 126 JSON paths
and 12 binary checkpoints, 138 unique paths total. They occupy 39 direct parent
directories and 54 directories in the full ancestor closure through the
artifact root. The finite path grammar is:

```text
electrical-v3/{development|independent-confirmation}/seed-{seed:010d}.json
q4/{development|independent-confirmation}/{arm}/seed-{seed:010d}.json
q255/{development|independent-confirmation}/{regime}/{arm}/seed-{seed:010d}.json
r3-f0/development/{arm}/construction-seed-0000000017.json
r3-f0/deterministic-confirmation/{arm}/construction-seed-0000000017.json
r3-f0/deterministic-confirmation/{arm}/resume/transition-{event:010d}.r3ntcp1
r3-f0/deterministic-confirmation/{arm}/resume/transition-{event:010d}.resume.json
selection/development-selection.json
selection/final-verdict.json
```

Here `arm` is one of `theta-m-10_theta-a-13`, `theta-m-16_theta-a-18`,
`theta-m-18_theta-a-20`, or `theta-m-20_theta-a-20`; `regime` is
`production-overlap-ordered-pairs` or
`target-disjoint-isolation-control`; and seeds come only from the phase set
above. Resume events are exactly `529340`, `1058681`, and `1588022`. The 24
resume paths are ordered by the four frozen arms, then those three events, then
`.r3ntcp1` before `.resume.json`. Canonical JSON of that ordered list uses
`sort_keys=True`, compact separators, `ensure_ascii=True`, and
`allow_nan=False`; it is exactly 3,409 bytes with SHA-256
`82581f614b7ef2766f0c8264d7ea44f59818bc2baca8661ea241d24400b09bb8`.

The large `.r3ntcp1` checkpoint is streamed exclusively through
`humanbrain.native_temporal_runtime.NativeTemporalCircuit.publish_checkpoint`.
JSON manifests and results, each bounded to at most 1 MiB, are published through
`humanbrain.native_temporal_runtime.publish_immutable_bytes`. A pre-existing
target is a STOP. Both paths forbid overwrite, retry, fallback, force options,
and publisher-side directory creation. The resume manifest is the transaction
commit point: a durably published checkpoint without its matching manifest is
an orphan requiring manual STOP, not an input eligible for automatic recovery.
The development-selection file is written once after all development gates;
the final-verdict file is written once after all confirmation gates. Paths
reserved for unselected arms remain absent.

## 2. Electrical v3 — reserve-valid fitted-cell fixture

The old sweep covered the 32,768 base pyramidal IDs. Reserve cohorts can use
any of 131,072 pyramidal IDs and can attach to any of 4,096 baskets, so v3 must
cover the full configured domains on all eight seeds.

For each seed, assign rank 0 to the globally maximum pyramidal threshold, rank
1 to the minimum, and map descending remaining currents to ascending remaining
thresholds. Pair this with the globally worst basket threshold. The three
novel/matching/active boundary scenarios must still emit the intended winner
and exactly one post-inhibition pyramidal spike. If spike-latency monotonicity
is not proved for the frozen cell equations, replay every actual recruited
cohort mapping instead of relying on the extrema argument.

Also require:

- a 20,000-pA singleton winner-relay check over the full 131,072-relay domain;
- 32,768 generic proximal/relay trials (`4096 columns × 8 seeds`) with
  proximal GIDs `[266240,270336)`, relay GIDs `[270336,274432)`, and currents
  570/20,000 pA respectively;
- exact per-seed hashes and the verdict
  `PASS_RESERVE_VALID_ELECTRICAL_V3`.

The three literal eight-current boundary rows are:

```text
active_vs_matching  = [650,640,590,543,496,449,402,355]
matching_vs_novel   = [600,590,543,496,449,402,355,308]
novel_rank_boundary = [550,543,496,449,402,355,308,261]
```

At the 0.05-ms integration step, same-tick candidates are ordered by descending
voltage-minus-threshold margin and then ascending physical GID. The extrema
shortcut is eligible only if, at fixed current and rest state, first-spike tick
is nondecreasing in threshold offset over the complete configured domain;
otherwise every recruited cohort mapping must be replayed. Each boundary
control must emit two pyramidal spikes with natural latency strictly greater
than 100 µs; with basket current 200,000 pA and GABA current 500 pA, the
inhibited race must emit only the intended winner, with winner-to-basket and
basket-to-GABA delays each exactly one tick.

The relay claim is only *orchestrated source-then-relay latency ordering*.
Without source-off synaptic ablation it is not evidence of synaptic causality.

## 3. q=4 compressed reserve fixture

Use `C=4096`, `K=11`, base/cohort width 8, `q=4`, target token 2041, and
129 target exposures. Contexts are the literal first 129 target-disjoint IDs
in ascending search order: `0..24`, `26..55`, `57..62`, `64..93`, and
`96..133`. Their canonical JSON SHA-256 is
`8ad1663264ca08bfa18830f285d65457fdc499393babefbe098f49223c75feef`.
Every episode is:

```text
reset_transient
recognize(context)       # read-only prelude
diagnose_context
learn(target)            # the only plastic step
reset_transient
```

The recognize-only prelude guarantees exactly one possible target allocation
per active target column and keeps RNG addressing unambiguous.

| Exposure | Required target-column state | Reserve cursor |
| ---: | --- | ---: |
| 32 | base cohort full, depth 8 | 32,768 |
| 33 | reserve cohort 1 recruited, depth 16 | 32,856 |
| 64 | cohort 1 full | 32,856 |
| 65 | cohort 2 recruited, depth 24 | 32,944 |
| 96 | cohort 2 full | 32,944 |
| 97 | cohort 3 recruited, depth 32 | 33,032 |
| 128 | cohort 3 full | 33,032 |
| 129 | cohort 4 sentinel recruited, depth 40 | 33,120 |

Pass requires complete atomic eight-cell cohorts, three full reserve
recruit/fill cycles, no early next-cohort use, at most eight driven novel
candidates, exactly 11 allocated target segments per novel exposure (1,419
total), zero pre-exposure matching/active/predicted columns, and identical
fresh versus milestone-resumed logical snapshots.

This is a compressed synthetic allocator/race fixture, not production-load
collision evidence.

## 4. q=255 full-cohort fixtures

Both regimes use 2,041 target exposures with one plastic step per episode.
Exposure 2,040 must leave all eight base owners at load 255. Exposure 2,041
must recruit eleven complete eight-cell cohorts atomically, allocate exactly
11 new segments, reach target-column depth 16, and advance the reserve cursor
to 32,856. Total target segments are 22,451.

### 4.1 Production-overlap regime

Use the actual 768-ID C1 codebook, `0..255` plus `1024..1535`, with target 32
excluded from the 767-token prelude vocabulary `V`. Its frozen input SDRs are
unique with maximum column overlap 2. The codebook digest preimage is canonical
JSON `[[token_id, sorted_column_ids], ...]` in ascending frozen token-ID order;
its SHA-256 is
`388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c`.
For exposure `n = j×767 + r`:

```text
a = V[r]
b = V[(r + j + 1) mod 767]
reset → recognize(a) → recognize(b) → diagnose → learn(32) → reset
```

The 2,041-pair SHA-256 is
`8ef9805b578c5f55485473f62ff577f0bcf319565f134788ea43902c4dfa7edf`.
This regime retains real codebook overlap and recurrent leakage opportunities.

### 4.2 Target-disjoint isolation control

Use target 2041 and the first 2,041 ascending token IDs whose 11-column SDR is
disjoint from the target. The final learned ID is 2101. The next 128 qualifying
IDs, ending at 2233, are the frozen novel probes. The complete learned/novel ID
manifest SHA-256 is
`495e9fdf8d501617bc392cfa21ee8c31538fef700aaeef6cd5c446e268d399e3`.
All 2,169 rows are unique; learned pairwise column overlap is at most 3,
novel-versus-learned overlap is at most 2, and target overlap is exactly zero.
The exact episode sequence is:

```text
reset → recognize(context_id) → diagnose → learn(2041) → reset
```

This regime localizes allocator/code failures; disjointness is not required
for the exposure schedule. It may use an exact generator only after complete
q=4 parity and a generated-snapshot physical replay across exposures
2,039–2,041.

### 4.3 q=255 falsifiers

Each regime must show:

- 2,041 unique source contexts and 2,041 unique target winner tuples;
- maximum winner overlap strictly below `ceil(theta_match/2)` and physical
  winner/relay overlap exactly twice the winner overlap;
- zero unintended pre-exposure matching, active segments, or predicted columns;
- exactly 11 allocations per novel exposure;
- direct learned-source recurrence selects all 11 stored owners;
- all 128 novel probes have zero false match/prediction;
- complete cohort/load/cursor invariants;
- identical fresh/resumed logical and recognition-index hashes.

These are synchronous fixture screens. Because all eleven target columns
recruit together, they cannot bound collisions when production columns recruit
asynchronously.

## 5. TRAIN-order R3-F0 — sole C1-scale identity gate

Before CN-2, replay the frozen C1 TRAIN allocation order only. Held-out loaders,
scores, and outcome files are forbidden. Inputs are:

- TRAIN stream SHA-256
  `f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5`;
- 2,117,412 tokens in 49 canonical TRAIN books;
- 2,117,363 transition events and exactly 49 transient resets per replay;
- 2,116,946 distinct temporal contexts;
- 23,286,406 segments.

One replay is exactly one TRAIN pass at construction seed 17. For each book in
canonical order: `reset_transient`; `learn(first_token)` once as the
zero-allocation context seed; then `learn(token)` exactly once for every
remaining token in order. Learned state persists across book boundaries, but
membrane state and previous context do not. The 49 first-token calls increment
the learning counter but allocate no transition segment; the remaining calls
are the 2,117,363 transition events.

Use the `c67c6c4` Python semantics or a native generator proven bit-exact to it.
For every distinct temporal key, the winner tuple must be injective; repeating
the key must reproduce the same tuple. A stored source may legitimately predict
multiple next tokens, but a segment whose stored source differs from `W_prev`
must never match. Repeated observed transitions must select their stored owners;
novel transitions must allocate one segment in every active column.

The overlap rejection bounds for theta-match 10/16/18/20 are 5/8/9/10 winner
pairs. An exact quadratic comparison is unnecessary: different production
tokens share at most two columns, so bucket tuples by current token and reject
duplicate `r`-subsets within each bucket.

Development executes one fresh seed-17 pass for each fixture-passing arm in
candidate order. After the first passing arm is selected and its independent
q=4/q=255 confirmation passes, deterministic R3-F0 confirmation executes that
arm's second fresh seed-17 pass and three resumed continuations checkpointed
after transition events 529,340, 1,058,681, and 1,588,022. These are
`floor(total_transition_events × n / 4)` for `n=1,2,3`.

The checkpoint cursors are literal, not recomputed at run time:

| Transition event | Book (zero-based ordinal) | Last/next token index | Event and learning count | Remaining transitions | Future resets |
| ---: | --- | --- | ---: | ---: | ---: |
| 529,340 | Joshua (5) | 42,123 / 42,124 | 529,346 | 1,588,023 | 43 |
| 1,058,681 | 2 Chronicles (13) | 82,661 / 82,662 | 1,058,695 | 1,058,682 | 35 |
| 1,588,022 | Jeremiah (23) | 33,073 / 33,074 | 1,588,046 | 529,341 | 25 |

For all three, `reset_before_next_token=false`.

The final state must have complete eight-cell cohorts, no load above 255,
114,400 committed pyramidal cells, 1,286 recruited columns, and maximum depth
1,128. Snapshot hashes include source/protocol identities, TRAIN position,
previous context, cohorts, owner loads, segments, exact permanence ticks,
counters, recognition-index/rebuild digest, and the ordered winner stream.

Only `PASS_R3_F0_TRAIN_STRUCTURE` is C1-scale identity/collision evidence. It
is still TRAIN-only structural evidence, not generalization or capability.

### 5.1 Immutable envelopes

Both resume manifests and results use the exact outer keys `artifact_id`,
`payload`, `payload_sha256`, and `schema_version`. The resume artifact ID is
`r3-f0-resume-manifest-v1`; its payload keys are `arm_slug`, `artifact_path`,
`checkpoint`, `checkpoint_publication`, `circuit_contract`, `claim_boundary`,
`construction_seed`, `cursor`, `phase`, `previous_state`, `protocol_sha256`,
`reference_trajectory_id`, `screen_id`, `source_binding`,
`source_bundle_identity_sha256`, `stream_prefixes`, `thresholds`, `timing`, and
`train_input`.

The result artifact ID is `r3-calibration-v3-result-envelope-v1`; its common
payload keys are `artifact_path`, `claim_boundary`, `construction_seeds`,
`phase`, `protocol_sha256`, `result`, `screen_id`, `source_binding`,
`source_bundle_identity_sha256`, and `thresholds`. An R3-F0 `result` has exactly
`arm_slug`, `comparisons`, `kind`, `reference`, `resumes`,
`structural_metrics`, `train_input`, and `verdict`.

### 5.2 Adopted active-wall contract

Schema 4 adopts timing contract `r3-cumulative-active-wall-v1` with the exact
cap `54,000,000,000,000 ns` while leaving execution disabled. CPython 3.11
`time.perf_counter_ns` measures one closed active interval without internal
pauses; scheduler delay, sleep, locks, garbage collection, I/O, asynchronous
waits, and incidental error handling remain charged.

A fresh session starts before the first of production circuit construction,
production TRAIN-source access, or frozen output-parent creation. A resumed
session starts immediately before opening the exact frozen resume manifest, so
manifest validation, checkpoint read/import, derived-index rebuild, and parity
checks are charged. A clean checkpoint stop occurs only after all scientific
work, durable checkpoint publication, fresh reopen, full import validation,
derived-index reconstruction, and identity/parity checks. Only the bounded
resume-manifest or terminal-result construction/publication/reopen is outside
the interval.

Every clean record stores exact non-negative integer `prior_cumulative_ns`,
`session_active_ns`, and `cumulative_active_ns`, with first prior zero and the
prior added exactly once. The conservative decision input is
`max(uninterrupted_cumulative_ns,resumed_cumulative_ns)`. Check the cap within
books and immediately before and after every atomic persistence operation. A
crossing finishes any non-cancellable persistence operation and cleanup,
records the exact overshoot, and emits STOP. An unclean or open session is
`TIMING_INDETERMINATE`.

## 6. Capacity schema 7 TRAIN-only bound

For column demand `D_c`:

```text
depth_c = max(8, 8 * ceil(D_c / (8 * 255)))
```

The frozen TRAIN counts produce 114,400 required pyramidal cells versus the
131,072 ceiling, leaving 16,672. Cohort rounding adds 5,184 cells over the
109,216 ideal; 81,632 cells are recruited beyond the base, maximum depth is
1,128, and 1,286 columns recruit. The structure retains 23,286,406 segments,
232,864,060 logical pair slots, and 465,728,120 physical contacts.

The bound artifact deliberately retains scientific artifact ID
`r3-relay-pool-capacity-v6-cohort-rounded` while its envelope schema is 7.
Schema 7 changed provenance only: the canonical scientific payload excluding
`schema_version` and `scope` remains SHA-256
`72665d2fc7360cc1dd32367211d9477e64a61e14e059714b0d1d1274297eb7d1`.
Its live TRAIN authority is
`29304cf71cce83ab34ee8253117043aec8413806ba0047ff850bb3c93a31d966`,
covering 3,656,777 bytes and stream SHA-256
`f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5`.
`held_out_bytes_live_verified=false`; the historical full-corpus digest is
recorded explicitly as not live verified. Capacity is conditional on R3-F0.
Arithmetic fit does not prove code injectivity, runtime, or capability.

## 7. Claim boundary and unlock

No held-out data is accessed. No threshold is selected. No result JSON is
created. No capability or transfer claim is made. Electrical v3, q=4, and
q=255 are fixture evidence; R3-F0 is TRAIN-only structural evidence.

To make this protocol runnable, a later pre-outcome commit must:

1. keep the governing nested Gate-3 source closure green after final changes;
2. implement the frozen checkpoint/resume, timing, and immutable result
   contracts behind `_execute_frozen_plan` without changing this protocol;
3. prove clean fresh/resumed arithmetic, STOP, and indeterminate-session
   behavior before enabling official screens;
4. flip `source_binding.runnable` only after all source-byte validation passes;
5. rerun fixture tests before any calibration outcome is opened.

Until then, `python experiments/r3_pilot/calibration_v3_protocol.py
--check-runnable` must exit nonzero.
