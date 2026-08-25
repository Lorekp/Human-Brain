# Race/output pivot protocol — winner-feedback broadcast loop + margin abstention seed

Status: **PREREGISTERED DESIGN — gates, arms, thresholds, and the falsifier clause are
frozen BEFORE implementation.** No arm of this protocol has executed. One shot means one
shot: single unmanipulated gate execution per leg; no alternate seeds; no arm additions;
nothing tuned on failure; failure is a publishable result (R0 §3 / R0-b §5 discipline,
`docs/research/r0-multitrace-protocol.md`, `docs/research/r0b-state-feature-protocol.md`).

Exclusive file: this document (plus its future red-test file `tests/test_race_pivot.py`
and gate runner `tools/race_pivot_gate.py`). All other paths READ-ONLY for this protocol.

Design anchor: branch `codex/a013-cognition-design` @ `47e77bd`. Line numbers below are
verified at that HEAD. Known drift warning: older docs cite stale line numbers
(ASSESSMENT-v4-a012.md cites `race` at recurrent.py:833; r0c-strategy-decision.md cites
`output_current_pa()` at recurrent.py:858-869) — those shifted to :1383/:1368 by later
commits; re-verify before execution and abort the anchor preflight on mismatch.

---

## 1. Activation status and provenance (why this protocol is live NOW)

This protocol is the **preregistered fallback promoted to primary**. Chain of custody:

1. Six feature families falsified under ridge+linear readout (R0 + R0-b verdict tables,
   `docs/research/r0c-strategy-decision.md` §Verdict table): multi-taus eligibility
   (sep 6.77e-05), spike-count windows (1.10e-04), voltage quantiles (7.59e-04),
   assembly-overlap similarity (0.0978), assembly-overlap random (**0.1019 G1-pass but
   12/42 replay** — capacity-regressive), combo spike+voltage (5.41e-04). Root cause
   measured as cosine collapse 0.9989 → every prompt collapses to ONE point in feature
   space (`roadmap-reading-to-coding.md:16-18`); degeneracy probe shows modal
   `[variable-0]` answers everywhere (`experiment-level-ladder.md:63`).
2. R0-b §7 (:600-631) froze THAT the pivot happens and its first mechanism:
   "the race/output layer … winner-feedback broadcast loop … seeded by the HOT-2
   hypothesis already queued at roadmap :119-120". The pivot protocol owns its own
   gates — this document IS that protocol.
3. Strategy report ranked it runner-up: "(a) winner-feedback broadcast + HOT-2 margins —
   parallel red tests only (shared injection seam)" (`r0c-strategy-decision.md:35-36`).
4. The mainline's Week-1 decision checkpoint returned **NEGATIVE-WITH-LOCALIZATION**
   (2026-08-24, `experiments/r0c_probes/FINDINGS.md`): best learned decoder span-acc
   0.03286 < unigram 0.04538; P2 memory route fails its gate in every arm×N; P3 shows
   identity is ALREADY ABSENT at the answer:-boundary state. Thread-1 D6-9 integration
   does NOT start; the runner-up thread (this protocol) becomes live.

## 2. Measured substrate being attacked (all numbers from committed records)

| Fact | Value | Source |
|---|---|---|
| Race semantics | pure argmax-current, first-spike; ties resolve to LOWEST id | `src/humanbrain/recurrent.py:1440` (`np.argmax` first-occurrence) over candidates sorted by `np.unique` at :1430 |
| Race purity | `_race_output_candidates` is non-mutating (fitted cell dynamics over a copied current vector; no circuit state written) | recurrent.py:1432-1499 |
| Winner consumed by | token stream only (`complete()`: race :128 → append :129 → observe(winner) :130) | `src/humanbrain/recurrent_language.py:117-131` |
| Margins exist | median 1482.73 / 1593.58 / 389.88 / 530.92 / 1598.95 pA across the five R0-b arms (~1500 pA typical in ridge arms) | `experiments/r0b_state_features/gate_results.json`, `diagnostics_never_gate.margin_median_pa` |
| Margin formula | target current minus best competitor over the allowed set | `benchmarks/ladder_experiments.py:797`; collection :781-808 |
| Margins never re-enter computation | HOT-2 ABSENT-as-deployed: "only external gates consume them (typed_composition_runner.py:1072); no component inside the system monitors its own decision quality" | `docs/indicator-assessment/ASSESSMENT-v4-a012.md`:36, :90-93; training-side enforcement only at `src/humanbrain/native.py:1422-1438` |
| Broadcast half missing | GWT-2 PARTIAL: "first-spike race IS a bottleneck competition … no workspace buffer reuses it"; roadmap: "missing half is BROADCAST — winner re-enters the network as input" | ASSESSMENT-v4-a012.md:32; `roadmap-reading-to-coding.md:62-65` |
| State differences available | ~1e-4 measurable (median 1−cosine 1.10e-04 spike windows; 7.59e-04 voltage quantiles) | R0-b gate records (§1) |
| Afferent event path | `observe(token)` → native `humanbrain_observe_token` with `self._afferent_targets[token_id]` → family boundary update → observation sink | recurrent.py:1189-1212 (:1194 advance, :1200 boundary, :1201-1206 sink); `src/humanbrain/native.py:1289-1307` |
| Observation sink contract | property+setter :1126-1143; fires AFTER boundary update; callback must NOT drive the circuit | recurrent.py:1126-1143, :1201-1206; `tests/test_recurrent_observation_sink.py` |

## 3. Mechanism (frozen before any execution)

### 3.1 Winner-feedback broadcast loop

The race itself is UNMODIFIED in every arm — argmax-current semantics, tie-to-lowest-id,
and `_race_output_candidates` byte-preservation are invariants, not defaults. The entire
mechanism lives in a host-side caller loop around the two existing primitives
`race()` (non-mutating probe) and `observe(token)` (afferent event). New method
`RecurrentCellLanguageModel.complete_broadcast(...)` beside `complete()` 
(`recurrent_language.py:117-131`), plus an equivalent circuit-level driver so
`evaluate()` can opt in.

Per output position (B = winner-feedback injections committed at this position;
sweep **B ∈ {1, 2, 4}**):

```
w0 = circuit.race().winner              # unchanged argmax-current race
circuit.observe(w0)                     # injection #1 == the normal commit observe
w = w0
for round in range(MAX_SETTLE_ROUNDS): # MAX_SETTLE_ROUNDS = 4, frozen
    for _ in range(B - 1):
        circuit.observe(w)              # broadcast: re-inject winner's afferent event
    w_probe = circuit.race().winner     # non-mutating stability probe
    if w_probe == w:
        STABLE; break
    if not CHASE_TO_SETTLE:             # arm P-S: probe only, never chase
        UNSTABLE_AT_POSITION; break
    w = w_probe                         # arm P-R: attractor moved — chase it
    circuit.observe(w)                  # inject the new claimant, re-probe next round
else:
    UNCONVERGED  # recorded honestly as broadcast_unstable=True; diagnostic-only
emit_and_log(w, settle_profile)
```

Two preregistered variants, both gated:

- **P-S (context shaping, PRIMARY)**: emitted token is ALWAYS the initial race winner
  `w0`. Broadcast/settle shapes only the state carried into subsequent positions.
  Chasing disabled. *Key property: B=1 performs ZERO extra injections (injection #1 is
  the ordinary commit observe of current `complete()`), so P-S/B=1 is bit-identical to
  deployed behavior while still measuring stability for free (probe races are pure).*
- **P-R (settled answer, SECONDARY)**: chasing enabled; emitted token is the SETTLED
  winner after the loop terminates (stable or MAX_SETTLE_ROUNDS exhausted). This arm can
  flip emitted tokens; `flip_rate` (positions where settled ≠ initial) is a mandatory
  logged diagnostic. B=1 here still adds chase injections when unstable, so parity is
  anchored on P-S/B=1 alone.

Stability definition (frozen): position-level stable iff the probe race after the B
injections returns the same winner. Sequence-level diagnostics: fraction stable,
settle-round histogram, flip_rate. Stability metrics NEVER gate pass/fail (diagnostics-
never-gate rule); they feed the falsifier read (§7).

Determinism: broadcast introduces NO new stochasticity anywhere — argmax race +
deterministic injections; seed 7 throughout, as R0/R0-b/R0-c.

### 3.2 Margin consumers (HOT-2 seed)

Margin at a decision point, generalizing the ladder formula
(`benchmarks/ladder_experiments.py:797`) to free-run: over the SAME allowed set the race
used (all token cells when unrestricted),

```
m = currents[committed_winner] − max(currents[c] for c in allowed_set − {committed_winner})
```

computed host-side from the public `output_current_pa()` (`recurrent.py:1368-1381`) —
no kernel involvement.

**Calibration freeze (BEFORE any held-out scoring):** pool margins over every TRAIN
answer-position decision of the fixture corpus (and, at the N=1024 leg, the Bible TRAIN
manifest), seed 7, per arm. Freeze `τ_low = 5th percentile`, `τ_high = 95th percentile`
of that pooled distribution INTO the gate JSON's pre-registration block before any
CALIBRATION/VAL free-run is scored. Post-freeze recalibration attempts fail the anchor
check (hash mismatch ⇒ ABORT, cohort NOT spent — r0b_gate.py :42-58 pattern).

- **P-M1 (abstention arm)**: no kernel or dynamics change. At each held-out decision
  point: if `m < τ_low` → emit ABSTAIN sentinel (position excluded from the selective-
  accuracy denominator, counted in coverage; abstained positions can never score as
  correct). If `m > τ_high` → log high-confidence flag (diagnostic). Frozen success gate
  (N=1024 leg only; fixture leg reports diagnostics): selective span-accuracy ≥
  **0.061851478566709396** (the frozen two-sigma threshold computed from unigram
  29/639, `experiments/r0c_probes/FINDINGS.md`) AND coverage ≥ **0.30** (abstention
  that hides everything is vacuous — minimum-coverage bound frozen now).
- **P-M2 (margin-as-input-feature arm)**: one extra feature channel = previous-step
  competitive margin (pA), standardized by the FROZEN TRAIN mean/std, appended to the
  flat feature buffer (F → F+1) via the existing feature-family seam
  (`_advance_feature_boundary`, recurrent.py:586; width guards ride the existing
  `feature_cell_count` metadata keys :855-864 and restore shape checks :891-893 — no new
  checkpoint format version). Ridge refit on TRAIN; G3 replay gate re-checked in-arm.
  Backend scope: HOST scipy/numpy path ONLY (native K-bank mirrors are untouched per
  §6; native-backend parity is explicitly NOT claimed for P-M2 — recorded limitation).

## 4. Why this can succeed where linear feature readouts failed

- **Iterated nonlinear dynamics amplify small state differences into attractor
  selection.** The one-shot requirement that killed six families was LINEAR separability
  of a single boundary state under ridge. The broadcast loop replaces that with a
  different demand: the map `state → inject(winner) → state′ → argmax` iterated
  B×MAX_SETTLE_ROUNDS times must separate contexts into distinct settled attractors.
  GLIF cells contribute threshold crossing, refractory effects, and after-spike currents
  (nonlinear, history-dependent); differences measured at ~1e-4 in feature space need
  only be amplified ACROSS ITERATIONS, not linearly decoded AT ONE STEP. Attractor
  selection is the canonical computation that fixed random dynamical systems can
  perform without any trained linear map deciding.
- **The missing workspace half exists as plumbing today.** ASSESSMENT GWT-2 is PARTIAL
  precisely because the race is a genuine winner-take-all bottleneck whose winner is
  thrown away after decoding (:32). Closing the loop costs three lines of caller code
  (§3.1) because the afferent event path already accepts arbitrary token ids at
  arbitrary times (`observe`, recurrent.py:1189).
- **Margins are large, real, and currently orphaned.** Median competitive margins of
  ~1500 pA in ridge arms mean winners win DECISIVELY — there is a well-conditioned
  confidence signal sitting unused (HOT-2 ABSENT, :36). Routing it back gives the
  system its first internal monitor: abstention (P-M1) converts margin into selective
  accuracy; the margin channel (P-M2) lets step t−1 decisiveness modulate step t —
  neither requires the features that failed.

### Honest counterarguments (stated so they can be wrong in public)

1. **Amplifying noise.** P1/P2/P3 localization says item identity is ALREADY ABSENT at
   the boundary state (best decoder loses to the majority prior; P3 floor even at ≤1
   chapter distance). If the ~1e-4 differences are mostly noise, iteration amplifies
   noise identically: the settled attractor may be a deterministic function of the
   unigram prior alone. This protocol MEASURES that outcome rather than assuming it
   away — the G1-style separation read on post-settle states (§7) is exactly the test.
2. **Broadcasting cannot create information.** Re-injecting the winner adds at most
   log₂(vocab) bits that any downstream consumer could already read off the race. The
   only new quantity is nonlinear mixing of winner identity with reservoir state — which
   helps ONLY IF the reservoir retains item-specific signal above its own noise floor.
   Current evidence says it does not. The bet, falsifiable within one fixture leg: that
   ITERATION, not representation quality, was the binding constraint. If P-S/P-R leave
   separation < 0.10 AND answers modal-degenerate, that bet is dead and takes the whole
   architecture family with it (§7) — by design.
3. **Degeneracy trap.** Under modal `[variable-0]` collapse the loop will faithfully
   broadcast variable-0: a constant workspace. The battery therefore keeps the
   de-degeneration guard (distinct ≥ 3) as a PASSING precondition — broadcasting nothing
   must register as failure, not as a stable attractor success.
4. **Chase pathology (P-R).** Chasing a moving argmax can orbit without converging;
   MAX_SETTLE_ROUNDS=4 bounds cost and the UNCONVERGED outcome is recorded, never
   silently truncated.

## 5. Arms and execution order

Shared config everywhere: fixture QA corpus at committed defaults, seed 7,
96-cell circuit (`RecurrentCircuitConfig(reservoir_cells=96, ...)`,
`r0b-state-feature-protocol.md` §fixture config); N=1024 leg uses the r0b gate config
(N_CELLS=1024, `tools/r0b_gate.py:58`). Curriculum anchors pinned: typed-program
fixture anchor `48219eda110f09f735525a7de00e56596ffac729746654e809c3c5eb525ccae6`;
Bible legs (N=1024) pin FREEZE-CANDIDATE-v3-BIBLE.json
(`candidate_sha256 361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c`),
TRAIN manifest `d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b`,
VAL manifest `2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec`.

| Arm | Config | Role |
|---|---|---|
| P-S1 | broadcast B=1, no chase | parity anchor + stability readout (confirmatory) |
| P-S2 | broadcast B=2, no chase | context-shaping intervention (confirmatory) |
| P-S4 | broadcast B=4, no chase | context-shaping intervention (confirmatory) |
| P-R4 | broadcast B=4, chase-to-settle, MAX_SETTLE_ROUNDS=4 | settled-answer intervention (confirmatory) |
| P-M1 | margin abstention, τ frozen from TRAIN | HOT-2 consumer, selective accuracy (confirmatory) |
| P-M2 | margin-as-input-feature, F→F+1, ridge refit, host backend | HOT-2 consumer, modulation (confirmatory) |

Execution order, mechanical: **N=96 fixture leg FIRST** — all six arms run the full
battery (§6) at fixture scale; only arms passing ALL gates proceed to the N=1024 leg
(same battery + separation read + P-M1 selective-accuracy gate). No arm may skip the
fixture leg; a fixture-leg failure is terminal for that arm. Combo selection IF ≥2 arms
pass both legs: prefer P-S family over P-R over P-M by frozen order above; ties broken
by higher fixture exact matches, then lower wall_s. Nothing else may influence selection.

## 6. Mechanical gates (identical structure to R0/R0-b)

Gate battery per arm, evaluated mechanically by `tools/race_pivot_gate.py` (structure of
`tools/r0b_gate.py`: anchor preflight → per-arm runs → aggregate):

- **G-P1 (answers, primary)**: `distinct_answers ≥ 3` on the bundled QA fixture corpus
  held-out CALIBRATION free-run (`GUARD_DISTINCT_MIN = 3`,
  `benchmarks/qa_experiments.py:55`).
- **G-P2 (exactness, primary — deliberate tightening vs R0-b)**:
  `free_run_exact_matches ≥ 1`. R0-b recorded exact matches as diagnostic only and got
  six-distinct-but-zero-exact everywhere ("diverse-but-wrong priors",
  `r0c-strategy-decision.md` finding 4). Broadcast must produce at least one exactly
  correct answer to prove it does more than de-collapse.
- **G-P3 (capacity non-regression, secondary)**: `train_replay_tasks ≥ 39/42`
  (baseline 41/42 single-trace seed-7 ridge; floor 39), train_cases_passed reported.
- **G-P4 (post-settle state separation)**: pooled median 1−cosine > **0.10** across the
  18 CALIBRATION tasks (44+4 pairs) computed on POST-SETTLE states captured via
  `observation_sink` (post-boundary reads only). Full battery at BOTH legs; the
  architecture-falsifier read (§7) uses the N=1024 value; the N=96 value is recorded
  alongside.
- Diagnostics (never gate): settle profiles, stability fraction, flip_rate, τ values,
  margin distributions, coverage, wall_s, backend strings.

Exit codes, fixed in advance (`tools/race_pivot_gate.py`): **0** = all executed gates
PASS; **2** = valid terminal INELIGIBILITY (a scientific result — preserved, NEVER
retried silently); anything else = partials + terminal-failure evidence preserved, cohort
spent. Attempt directories keyed by protocol/cohort identity (not caller-supplied ids);
refuse overwrite; preterminal records structurally incapable of carrying pass flags;
eligibility recomputed at the terminal summary from raw metrics — stored booleans never
trusted. Anchor preflight verifies: git HEAD == design anchor lineage, curriculum
sha256 match, frozen τ block present and hash-sealed, BEFORE any arm executes.

## 7. Red tests T-P1..T-P8 (all must be green BEFORE the first gate leg)

Written to `tests/test_race_pivot.py`; they run on synthetic small circuits
(test_recurrent_observation_sink.py pattern: token_cells=16, reservoir_cells=32, seed 7)
unless stated. No red test consumes the cohort.

- **T-P1 broadcast determinism**: identical seed/prompt/config with B=2 → two runs give
  identical token streams, identical settle-profile hashes, bit-identical final feature
  vectors.
- **T-P2 stability convergence**: constructed fixed-point case terminates STABLE within
  MAX_SETTLE_ROUNDS with correct settle counters; constructed oscillating case flags
  UNCONVERGED at exactly MAX_SETTLE_ROUNDS and returns (never hangs); P-R flip bookkeeping
  consistent (emitted==settled, initial recorded separately).
- **T-P3 B=1 parity**: P-S/B=1 produces token stream, per-step
  `output_current_pa()` sha256s, and final feature vectors BIT-IDENTICAL to current
  `complete()` on the same circuit; event-count equality asserted via a counting sink
  (zero extra observe events).
- **T-P4 flag-off bit-identity**: broadcast disabled → `complete()` outputs and event
  counts byte-identical to HEAD `47e77bd` behavior (mirrors the strategy doc's T-A1
  flag-off discipline).
- **T-P5 margin capture correctness**: logged margins equal independently recomputed
  values from `output_current_pa()` to 0.0 tolerance; sign convention
  (target − best competitor) asserted; single-candidate sets yield m = 0 handled
  explicitly (no max over empty set).
- **T-P6 margin calibration reproducibility**: τ computed twice from the frozen TRAIN
  manifest agrees to machine precision; τ block hash-sealed into the gate JSON; any
  post-freeze mutation of calibration inputs fails the preflight (simulated in-test).
- **T-P7 abstention correctness**: crafted margin vectors straddling τ_low produce
  exactly the specified abstain/answer labels; coverage denominators verified on
  synthetic records; an abstained position engineered to be "correct" still never counts
  toward selective accuracy.
- **T-P8 logging discipline**: settle profiles and margins reach logs through the
  `observation_sink` contract (post-boundary reads only; a callback that calls
  observe/race/reset raises); logging disabled → results identical and zero sink calls.

"Honest implementation" for §8 means: T-P1..T-P8 green at the execution HEAD, anchor
preflight passed, single unmanipulated execution, no arm added/removed/re-tuned,
diagnostics never gating.

## 8. Cost and blast radius (host-side only — no new native kernels)

- **P-S1/S2/S4, P-R4, P-M1**: ZERO changes to `recurrent.py` race/dynamics paths and
  ZERO changes to Rust/CUDA kernels. The mechanism is a host-side Python loop calling
  existing `race()` / `observe()` (re-injection rides the EXISTING afferent event path,
  `humanbrain_observe_token` + `_afferent_targets`). New code: one method beside
  `complete()` (`recurrent_language.py`), a settle-record dataclass, gate tooling.
- **P-M2**: host-side feature-buffer extension (+1 channel) riding existing metadata
  guards (recurrent.py:855-864, :891-893); scipy/numpy backend only; native mirrors
  untouched (limitation recorded, §3.2).
- Checkpoint format: UNCHANGED (v6 carries `feature_cell_count` metadata already).
- Wall-clock estimate (NOT a gate — r0b §6 discipline): P-S arms multiply observe()
  calls per output position by ≈B (≤ ×4) plus ≤ MAX_SETTLE_ROUNDS extra probes (probes
  are pure and cheap relative to observe); P-R worst case ≈ ×(B·(MAX_SETTLE_ROUNDS+1))
  ≈ ×20 observe-equivalents at fixture scale where walls are seconds-to-minutes
  (r0b N=1024 walls 44-536 s). Records carry measured wall_s per arm.

## 9. Architecture-family falsifier clause (frozen now, not negotiated later)

If, after HONEST implementation (§7 definition) of ALL confirmatory arms in a single
unmanipulated execution:

- **Condition A**: post-settle state separation at N=1024 remains **< 0.10**
  (pooled median 1−cosine, G-P4 machinery) for EVERY broadcast/margin arm, AND
- **Condition B**: answers remain modal-degenerate on the fixture battery —
  `distinct_answers ≤ 2` OR `modal_share ≥ 0.50` — for EVERY arm,

then: **the reservoir-computing architecture family — fixed random GLIF reservoir with
trained output readout, iterated or not — is FALSIFIED for this task class**, and the
conclusion is stronger than R0-b §7's: not only do linear readouts fail on these states,
but closing the workspace loop (broadcast) and routing the confidence signal back
(margins) — the two remaining mechanisms that do NOT require linear separability — also
fail to de-degenerate answers or differentiate states. Iteration was the last free lunch.

Boundary distinctions preserved from R0-b §7 (:622-631), verbatim in spirit:

- ∃ arm with separation ≥ 0.10 but ¬G-P1/G-P2 ⇒ **NOT** family falsification — dynamics
  differentiate; answers stay degenerate downstream. Disposition label:
  "readout/race blocker"; margin arms become primary suspects; fresh v2 protocol
  required for follow-ups.
- Capacity-only failures (¬G-P3 alone) ⇒ broadcast exonerated as a mechanism (the base
  circuit cannot replay even basally); banked; fresh v2 protocols per family.
- Any arm passing the full battery ⇒ falsifier CANNOT fire regardless of other arms;
  integrate behind a config flag (D6-9 pattern) with capacity non-regression asserted.

**Pivot on falsification (user-sanctioned: "custom architecture different than GLIF
allowed"):** the program pivots to a custom architecture outside the GLIF-reservoir
family. Named shortlist, frozen now:

1. **Integrate-and-fire arrays with local STDP + global-workspace broadcast** —
   engineered plasticity instead of fitted-cell dynamics; broadcast retained as a
   first-class architectural element (this protocol's GWT reading transfers directly);
   event-driven, sparse, cheap at scale.
2. **HTM-style sparse distributed representations** — SDR key properties (semantic
   overlap, union robustness) attack exactly the measured failure: context identity
   dying in a collapsed state; union property is designed to carry multiple contexts
   simultaneously without linear separability demands.

Shortlist discipline: each entry gets its OWN one-shot preregistered protocol (gates,
cohort, falsifier) before any implementation; this document fixes only THAT the pivot
happens and the two named candidates, mirroring how R0-b §7 fixed this pivot.

## 10. Honesty notes

- Every figure labeled ESTIMATE (§8 walls) is arithmetic on measured anchors, not a
  measurement; the records will carry measured wall_s.
- The ~1500 pA margin figure summarizes R0-b RIDGE arms (1482.73/1593.58/1598.95);
  assembly arms were 389.88/530.92 — margins are arm-dependent; P-M calibration freezes
  per-arm τ precisely so the headline number never substitutes for the distribution.
- The mechanism bet (§4) is stated so it can be wrong in public; §9 is what "wrong"
  means, agreed before the first arm runs.
- Fixture-scale results evidence de-degeneration and plumbing correctness, not reading
  skill; the N=1024 Bible leg owns the real question (r0b honesty note preserved).
- P-M2's host-backend-only scope means its N=1024 numbers are not backend-comparable to
  native-accelerated arms; recorded per-run, never silently pooled.
- This protocol spends its cohort ONCE. Retired arms are never rerun "to check".
