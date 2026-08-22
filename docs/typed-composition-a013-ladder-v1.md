# Typed-composition A013 scale-ladder v1 decision and run contract

**Status:** drafted preregistration; arm sizes UNFINALIZED pending the measured
resident-cost curve; material execution of every arm is on hold until this
protocol is frozen, hash-pinned, reviewed, and separately authorized

**Draft base commit:** `49eb043bd3efdaeb3b76508b9a7fc82954734017`
(branch `codex/typed-ladder-a013`, worktree
`.worktrees/human-brain-typed-ladder-a013`)

**Inherits:** `docs/typed-composition-a012-v1.md` (framing, eligibility, and
execution discipline) and its implementation base
`a33d2ffd52e3365fb9e4f93c052453cd55ec8189`.

**Sealed prior result being extended:**
`docs/typed-composition-a012-v1-results.md` — a012 is spent evidence and is
never rerun, retuned, or reinterpreted into a pass.

**Raw protocol SHA-256:** TBD at freeze; computed over
`benchmarks/typed_composition_ladder_protocol_v1.json` exactly as committed.

## Decision

The smallest next increment is a preregistered dose-response ladder over
configured reservoir size N on the same 42/18 task curriculum, same three
corpus passes, and same one-shot fail-closed machinery as a012 — with one
mandatory learner change and no other architecture change:

| Arm | Lane | Configured cells N | Readout | Status |
| --- | --- | --- | --- | --- |
| L1 | A2 | 128 (bridge) | local sparse three-factor | sole capability candidate |
| L2 | A2 | 10^6 `{PARAM}` | local sparse three-factor | sole capability candidate |
| L3 | A2 | 10^7 `{PARAM}` | local sparse three-factor | sole capability candidate |
| L4 | A2 | 10^8 `{PARAM}` | local sparse three-factor | sole capability candidate |
| L5 | A2 | N_MAX = largest feasible cohort, target >= 10^9 `{PARAM}` | local sparse three-factor | sole capability candidate |
| L6 | A1 | 10^7 `{PARAM}` (matched to L3) | local sparse three-factor | diagnostic only |

A0 is not rerun anywhere in this ladder. The framing axis stays represented by
the matched pair L6 (A1@10^7) vs L3 (A2@10^7): identical observation stream,
only `strengthen` calls differ, isolating target filtering at matched scale.
The A2 series L1->L5 isolates the capacity axis. Every arm observes the same
token IDs in the same order; arms differ only in reservoir size, declared
activity density, and (for L6) supervision scope.

`{PARAM}` marks are load-bearing: the five scaled arm sizes are **not guessed
here**. Each is finalized exactly once, from measured data, before protocol
freeze (see "Arm-size finalization"), then becomes an immutable constant of
the frozen protocol. No arm size may be changed after freeze for any reason,
including a failed arm.

## Why the readout must change before any scaling arm runs

Ridge regression over reservoir activations is arithmetically excluded at
ladder scales. The normal matrix is O(N^2) memory and the solve O(N^3): it was
already 288 MB at 6,144 cells (`development_scale_6144_v2.json`) and about
72 GiB near 40k cells. Every arm above a few thousand cells is impossible
with ridge. This is not a tuning choice; it is a feasibility constraint.

The ladder therefore adopts the already-proven local sparse three-factor
readout (main `sparse-readout` work; `docs/sparse-cell-readout.md`;
`sparse_cuda_engineering_v1_*.json`; worktree
`.worktrees/human-brain-sparse-readout`):

- fan-in capacity K = 64 learned synapses per output cell, candidates C = 4K;
- resident readout bytes ~ 64 * V * K plus small per-output terms —
  **independent of N** (measured 1,052,672 bytes versus 409,600,000 dense at
  N=200,000; host memory 428 -> 35 MiB);
- local updates only; no global normal matrix ever exists;
- measured CUDA throughput at 200k cells rose 127.265 -> 158.177 tokens/s.

V (output/token cells) stays fixed at the a012 value (257), so the readout is
size-invariant across the whole ladder; only the reservoir scales.

**Honest comparability flag.** Switching ridge to local sparse three-factor
changes learning dynamics. Sealed a012 numbers are ridge numbers; they are NOT
directly comparable to any ladder arm. Arm L1 (A2@N=128 with the local
readout) exists precisely to re-measure the a012 operating point under the new
learner so the ladder is internally comparable end-to-end. L1 vs the sealed
a012 result isolates the readout-rule change at known scale; L1 vs L2..L5 then
isolates scale alone.

## Fresh cohort, never byte-reused

The curriculum structure is identical to a012 (42 TRAIN tasks: four ten-task
factors plus two identities; 18 calibration tasks; three public cases each;
three corpus passes; identical literal sets, templates, identity prompts, and
template-parity rule), but the cohort is regenerated under NEW domains:

- identifier domain: `humanbrain/typed-composition/a013/v1/entry-point`;
- task-order domains: `humanbrain/typed-composition/a013/v1/task-order`
  (with `/train` and `/calibration` suffixes as before).

Every opaque entry-point identifier, canonical signature byte sequence, task
ID, sort key, and presentation order therefore differs from a012's sealed
cohort. Reusing, importing, or regenerating-with-old-domains any a012 task
bytes is a red-test failure. Signature lengths remain exactly 151 bytes
(scalar) and 225 bytes (list) by construction, so token-level accounting is
unchanged from a012's exact constants.

## Capacity and generalization gates, reported separately

Per arm, all-or-nothing as before, but the two failure modes a012 conflated
are now separate gate families, both reported in every terminal summary:

- **Capacity family (fits?):** TRAIN label replay 42/42 AND TRAIN cases
  126/126.
- **Generalization family (transfers?):** calibration label replay 18/18 AND
  calibration cases 54/54.

An arm is eligible only if BOTH families pass together, plus the shared
integrity/margin/collision gates. The ladder reads dose-responsely:

| Pattern across arms | Meaning | Consequence |
| --- | --- | --- |
| capacity fails below some N, passes above, generalization passes above | pure capacity defect, resolves with scale | continue scaling per ladder; no architecture change |
| capacity passes but generalization fails at every N including N_MAX | fits-but-does-not-generalize; scale does not fix it | stop scaling; new hypothesis/protocol, fresh cohort |
| neither family passes anywhere, margins healthy | learner/state defect, not capacity | inspect readout dynamics and state separation; do not resize again |
| integrity or accounting gate fails | implementation defect | fix code, new protocol version; cohort spent |

## Diagnostics carried forward, bounded for scale

All a012 diagnostic definitions carry over with one scale adaptation:

- **Prompt-trace retention** (non-veto): unchanged formula
  `exp(-(distance*5ms)/80ms)`; distances stay A0-free (no A0 lane), A1/A2 = 1
  token = 5 ms. a012 confirmed the framing fix (~7e-5 -> 0.9394); this ladder
  monitors it does not regress at scale.
- **State separation** (non-veto): all unordered TRAIN pairs, same result type,
  different first teacher token; distance `1 - cosine(e_i,e_j)`; computed by
  streaming accumulation over full vectors; only count/min/p10/median/mean per
  type stratum are persisted. Preregistered exploratory question: does
  separation improve with N (it did not from A0/A1 to prompt-last at 128)?
- **Exact state-collision check** (veto): key is SHA-256 over the COMPLETE
  eligibility byte vector (all N little-endian float64 cells, C-order) of the
  decision, grouped by `(allowed_cell_ids, digest)`; two decisions with the
  same allowed set and digest but different expected targets fail the arm.
  Computing the digest requires touching every exact byte, preserving the
  a012 exact-byte semantics; additionally the first and last 1,024 bytes of
  the vector are sealed as excerpts so the digest can be spot-reverified.
  Storage per decision is O(1) (32-byte digest + excerpts), not O(N).
- **Target margins** (veto): every competitive TRAIN position in sealed
  exact-replay completions; margin `I(expected) - max(I(other allowed))` in pA;
  count/min/p10/median/mean persisted; all finite and strictly positive.

Sealing remains seal-everything-before-any-scoring, per arm.

## Readout-capacity and feasibility gate family (new)

Each A2 arm additionally asserts, fail-closed:

- every output cell holds at most K = 64 learned synapses at every sealed
  decision (fan-in budget invariant);
- all readout weights finite and within the preregistered magnitude bound
  `{PARAM: w_bound}` finalized from sparse-readout evidence at freeze;
- sparse structural clocks positive (existing config validation);
- resident readout byte count matches the closed-form `64*V*K + V*C + small`
  accounting within recorded tolerance;
- no dense O(N^2) or O(V*N) float array is ever allocated: the runner refuses
  any layout outside the preregistered sparse state schema;
- pre-execution feasibility check from the measured curve: projected resident
  bytes and projected wall time for the arm's declared activity density fit
  the frozen envelope; an arm that fails feasibility is refused BEFORE its
  claim marker is created (and is then finalized out of the ladder with
  cause recorded — it does not silently run smaller).

## Cost model, residency arithmetic, and activity density

Said plainly: at the GLIF5 state floor of 60 B/cell, 100 billion configured
cells require >= 6 TB of live state, and the 170-billion-neuron target is
roughly 10 TB — full-touch simulation is out of reach on one workstation.
Sparse, event-driven residency where per-token cost tracks RESIDENT (active)
cells rather than configured N is the only path, and this ladder is designed
around that path rather than around dense simulation.

Verified anchors folded into the cost model (estimates, never gates):

- GLIF5 cell state floor 60 B/cell; measured total ~181 B/cell including
  sparse readout at 200k cells (`sparse_cuda_engineering_v1_baseline.json`);
- native frontier throughput 843 tok/s at ~98k cells at 2.5% activity falling
  to 11.7 tok/s at 100% activity; frontier scans beat dense SIMD at low
  density and lose at high density at 1M cells
  (`native_activity_density_v1.json`: 131,072 cells 1.8 ms/tok @1% ->
  85.4 ms/tok @100%; 1,048,576 cells 18.8 ms/tok @1% -> 711.4 ms/tok @100%);
- token cost therefore depends on ACTIVITY FRACTION: every arm declares an
  assumed activity density used ONLY for planning estimates; the declared
  value never enters any eligibility gate and the observed density is
  recorded as telemetry;
- readout residency is O(V*K), independent of N (see above).

Because the curve above is measured only up to 10^6 configured cells, all
scaled arm sizes stay `{PARAM}` until the parallel benchmark extends the
curve; planning estimates for each arm are computed mechanically from that
measured curve at finalization time and recorded in the protocol as
estimates-not-gates.

## Arm-size finalization (before freeze, once)

1. The parallel resident-cost benchmark publishes measured resident-bytes and
   tok/s versus configured N and activity density, with artifact digests.
2. From those measurements exactly one N_MAX is selected: the largest
   configured cohort whose projected resident memory and wall time fit the
   frozen envelope, target >= 10^9 configured cells. If measurements prove
   10^9 infeasible, N_MAX is set to the measured maximum and the shortfall is
   recorded in the protocol text at freeze — never discovered after a run.
3. Intermediate sizes 10^6, 10^7, 10^8 are confirmed against curve breakpoints
   (adjusted only if the measured curve shows a feasibility boundary below a
   nominal decade point).
4. All resolved values replace their `{PARAM}` markers; the complete JSON is
   hashed once; that hash is the pinned protocol forever. Post-freeze edits,
   including arm-size edits, invalidate the hash and refuse execution.

## Execution contract

Identical one-shot discipline as a012, extended per arm:

- plan-by-default CLI; planning validates the protocol, the source bundle
  (a012 closure inherited by reference plus this increment's new files,
  fully enumerated at freeze), and per-arm feasibility projections.
- Material execution requires `--execute`, a metadata `--run-id`, the exact
  raw protocol SHA-256, and the exact reviewed source-bundle SHA-256.
- Arms execute once each, in fixed ascending order L1, L2, L3+L6, L4, L5,
  inside one authorized window. Each arm creates its own exclusive spent
  marker `artifacts/.typed_composition_a013_ladder_v1_lN.claimed` (plus
  `.claiming` staging, `.invalid` poison, `.complete` authority, Windows
  MoveFileExW semantics, symlink/reparse refusals — all as a012).
- An arm failing any gate is terminal for THAT arm only; remaining arms still
  run. No retry, resume, resize, or reordering exists for a claimed arm.
- No arm's outcome may alter any later arm's parameters; the whole ladder's
  parameters were fixed before the first run by construction.
- Per-arm `999-terminal-summary.json` conjoins that arm's capacity family,
  generalization family, collision/margin/integrity gates, and diagnostics;
  a ladder-level roll-up record lists per-arm eligibility and is the only
  place the word "ladder" may appear in an eligibility statement.
- Checkpoint schema v5 carries explicit `cell_count` and sparse readout
  state; checkpoint v4 and v3 have no loader here. Immutability checks
  (raw bytes, cell/synapse state) repeat per arm.

The run hold stands: nothing here authorizes any material TRAIN, calibration,
or future-cohort execution. Authorization must be granted separately, after
freeze and review, and names the exact protocol and bundle hashes.

## Future regioned variant (placeholder — NOT part of this protocol)

Placeholder for a follow-up protocol family (candidate name
`typed-composition-regioned-v1`): a circuit split into a distinct encoder
region (prompt/signature ingestion, possibly heterogeneous timescales) and a
processor region (program-token competition and readout), with region-local
state, explicit inter-region projection budgets, and its own preregistered
accounting. Nothing in this file authorizes, designs, or schedules it; it has
no lanes, no arms, no cohort, and no runner path. It is recorded here only so
the ladder's outcome feeds a named next step instead of an improvised one.

## Implementation and test seams (to be built test-first)

- `benchmarks/typed_composition_ladder_protocol_v1.json`: raw public
  commitments, arms array, per-arm expected accounting, run hold.
- New runner module extending `src/humanbrain/typed_composition_runner.py`
  with per-arm exclusive claims, feasibility checks, and ladder roll-up.
- `src/humanbrain/typed_program.py` gains a variable-N circuit config and the
  local sparse three-factor lane wiring; ridge is unreachable from the ladder
  runner.
- Focused tests: see the numbered red tests below; every one must exist and
  fail before the corresponding implementation lands.

## Red tests (test-first; each numbered test is written before its feature)

1. Protocol loader rejects a raw SHA-256 mismatch, oversized (>32 KiB), or
   non-canonical ladder protocol file, fail-closed, before any parsing use.
2. Plan mode refuses `--execute` while any arm size is unresolved (`{PARAM}`
   / null): an unfinalized ladder cannot spend anything.
3. Finalization provenance: resolving arm sizes requires the recorded
   measured-curve artifact digests; a resolution without matching digests is
   rejected; post-freeze size mutation changes the pinned hash and refuses.
4. Fresh-cohort disjointness: every a013 entry-point identifier set is
   disjoint from a012's (recomputed from a012's public frozen domain), and no
   a013 task's (prompt, signature) byte pair equals any a012 task's.
5. Curriculum accounting exactness: mechanically recomputed 42/18 tasks,
   126/54 cases, 272 program targets/pass, 8,484 framing tokens/pass
   (= 7,156 signature + 1,244 prompt + 84 delimiters), 8,714 observations/
   pass, 816 A2 and 26,142 A0/A1-pattern strengthen calls over 3 passes;
   any mismatch fails the load.
6. Token-accounting N-invariance: the token-level accounting block is
   identical across all six arms; only scale-dependent fields vary with N.
7. Bridge-arm isolation: L1 differs from sealed a012 A2 conditions in exactly
   {readout rule, cohort identity}; framing order, supervision scope,
   literals, templates, passes asserted equal by test.
8. Ridge exclusion: the ladder runner cannot construct a ridge readout
   (config rejection), and every arm's config records
   `readout_learning_rule="sparse_three_factor"`.
9. Fan-in invariant: at every sealed decision of every arm, each output cell
   has <= 64 learned synapses; a seeded 65th-synapse fixture fails the arm.
10. Weight bounds: all readout weights finite and within the frozen
   magnitude bound; a seeded out-of-bound fixture is a terminal arm failure.
11. Structural clocks: nonpositive sparse structural clock values are
   rejected at config time.
12. Readout residency accounting: persisted resident readout bytes match the
   closed form `64*V*K + V*C + small per-output terms` within tolerance.
13. Dense-layout refusal: allocating any (N,N) or (V,N)-shaped float state in
   the ladder path raises; static layout allowlist enforced.
14. Collision-gate fidelity: the collision key is the SHA-256 of the complete
   N-cell eligibility vector; a planted fixture with identical digest group
   and differing expected targets fails the arm; a single flipped low-index
   byte changes the digest (exact-byte semantics, not excerpt-only).
15. Excerpt verification: sealed first/last 1,024-byte excerpts reverify
   against the stored digest; tampering either excerpt invalidates the record.
16. Split reporting: terminal summaries report capacity-family and
    generalization-family results as separate fields; a summary omitting
    either is invalid.
17. All-or-nothing per arm: any single failed constituent zeroes that arm's
    eligibility; partial passes never round up; ladder continues to later
    arms regardless.
18. Exclusive per-arm claims: `.claimed` creation is atomic and exclusive
    per arm; interruption, marker collision, or alias (symlink/reparse)
    permanently spends only that arm; sibling arms remain runnable.
19. Cross-arm immutability: no code path can mutate a later arm's parameters
    from an earlier arm's results; mid-ladder protocol byte change aborts
    with `.invalid` on the current arm only.
20. Activity-density declaration: every arm carries a declared assumed
    density; tests assert it is absent from every eligibility computation and
    present only in estimates/telemetry.
21. Feasibility pre-check: an arm whose projected resident bytes or wall time
    exceed the frozen envelope is refused before its claim marker exists,
    with cause recorded; it is never silently downsized.
22. Checkpoint v5: stores explicit cell_count and sparse readout state;
    loading v4 or v3 fails closed; raw-byte and cell/synapse immutability
    verified per arm.
23. Seal-before-score per arm: any attempt to read labels/cases before all of
    that arm's completions are sealed raises; completion-attempt counts match
    60 per arm exactly.
24. Terminal authority: per-arm summaries and the ladder roll-up rederive
    every gate from rebound evidence digests; exit status derives only from
    the verified roll-up; the public seam cannot mint eligibility.
25. Regioned placeholder inert: no runner, CLI, or loader path references the
    regioned variant block; its presence or absence never affects execution.

No material command has been run for this stage and none is authorized by
this draft.
