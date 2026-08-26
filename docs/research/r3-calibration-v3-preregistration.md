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

The manifest is deliberately fail-closed. It pins the explicit-threshold
Python oracle at `c67c6c4`, Random123 at `15963da`, the intrusive temporal
store and wrapper at `3dd2c4f`, the fitted electrical race module and native
entry point at `612b2f5`, and the Gate-3 refreeze at `1a262b6`, but reports
`runnable=false`. Those layers are committed and byte-bound. The remaining
blockers are the absent native mutation/integration runner and disabled
outcome-bearing runners. `--check-runnable` must fail until a later pre-outcome
commit binds that final implementation and removes both blockers.

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
manifest (114 unique JSON paths) under
`experiments/r3_pilot/calibration_v3_artifacts`. The finite path grammar is:

```text
electrical-v3/{development|independent-confirmation}/seed-{seed:010d}.json
q4/{development|independent-confirmation}/{arm}/seed-{seed:010d}.json
q255/{development|independent-confirmation}/{regime}/{arm}/seed-{seed:010d}.json
r3-f0/development/{arm}/construction-seed-0000000017.json
r3-f0/deterministic-confirmation/{arm}/construction-seed-0000000017.json
selection/development-selection.json
selection/final-verdict.json
```

Here `arm` is one of `theta-m-10_theta-a-13`, `theta-m-16_theta-a-18`,
`theta-m-18_theta-a-20`, or `theta-m-20_theta-a-20`; `regime` is
`production-overlap-ordered-pairs` or
`target-disjoint-isolation-control`; and seeds come only from the phase set
above. Every result uses exclusive creation (`xb`). A pre-existing target is a
STOP, and no force/overwrite option exists. The development-selection file is
written once after all development gates; the final-verdict file is written
once after all confirmation gates. Paths reserved for unselected arms remain
absent.

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

The final state must have complete eight-cell cohorts, no load above 255,
114,400 committed pyramidal cells, 1,286 recruited columns, and maximum depth
1,128. Snapshot hashes include source/protocol identities, TRAIN position,
previous context, cohorts, owner loads, segments, exact permanence ticks,
counters, recognition-index/rebuild digest, and the ordered winner stream.

Only `PASS_R3_F0_TRAIN_STRUCTURE` is C1-scale identity/collision evidence. It
is still TRAIN-only structural evidence, not generalization or capability.

## 6. Capacity-v6 static bound

For column demand `D_c`:

```text
depth_c = max(8, 8 * ceil(D_c / (8 * 255)))
```

The frozen TRAIN counts produce 114,400 required pyramidal cells versus the
131,072 ceiling, leaving 16,672. Cohort rounding adds 5,184 cells over the
109,216 ideal; maximum depth is 1,128 and 1,286 columns recruit.

Configured and materialized Python physics remains 274,432 cells. The 241,088
number is only a hypothetical lazy-commit projection and is not implemented.

At `3dd2c4f`, intrusive variable-live-pair storage is implemented. Its requested
payload is 3,590,431,900 bytes for A1 (1,490,461,056 learned SoA,
1,864,485,344 index, 524,288 source scratch, 232,864,060 segment scratch, and
2,097,152 owner scratch) and 3,588,400,284 bytes for A3 (the same learned,
segment, and owner arrays plus 1,862,961,632 index and 16,384 source scratch).
Each number excludes the 432-byte store header, allocator metadata, fitted-cell
physics, and 2,097,152 bytes of Python diagnostics. Storage existence is not a
whole mutable-runtime claim: mutation transaction, checkpoint, and production
runtime integration remain unimplemented.

Capacity-v6 is conditional on R3-F0. Arithmetic fit does not prove code
injectivity, runtime, or capability.

## 7. Claim boundary and unlock

No held-out data is accessed. No threshold is selected. No result JSON is
created. No capability or transfer claim is made. Electrical v3, q=4, and
q=255 are fixture evidence; R3-F0 is TRAIN-only structural evidence.

To make this protocol runnable, a later pre-outcome commit must:

1. implement and bind the final native mutation/integration runner to an
   immutable commit and SHA-256 values; the dedicated fitted-race
   `src/humanbrain/_native_src/src/temporal_physics.rs` module is already bound;
2. keep the Gate-3 source pin green after those changes;
3. implement outcome runners that consume this manifest without changing it;
4. flip `source_binding.runnable` only after source-byte validation passes;
5. rerun fixture tests before any calibration outcome is opened.

Until then, `python experiments/r3_pilot/calibration_v3_protocol.py
--check-runnable` must exit nonzero.
