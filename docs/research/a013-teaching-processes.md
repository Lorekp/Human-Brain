# A013 teaching-process experiment matrix

Status: design input — no protocol executed. Companion to
`a013-representation-choice.md` (owns the encoding axis; NOT YET WRITTEN on any
branch) and `docs/typed-composition-a013-ladder-v1.md` (owns scale arms; lives
on sibling branch `codex/typed-ladder-a013`, worktree
`.worktrees/human-brain-typed-ladder-a013`). Preregistration discipline: every
experiment declares gates before running; no tuning after failure.

**Verification note (this revision):** every code claim below was re-checked
against source at main `3059cbd` / branch base of this doc; line numbers are
identical on both checkouts. Sealed numbers were cross-checked against
`docs/typed-composition-a012-v1-results.md`. Corrections to the previous draft
are listed first.

## Corrections to the previous draft

1. **T4 citation was wrong.** The draft said the grow loop ":1218-1228 switches
   scope per epoch." It does not: supervision scope is a fixed per-lane
   property (`_LANE_COMMITMENTS`, typed_program.py:344-348 — A0/A1
   all-transition, A2 program-only), checked identically on every transition of
   every epoch (:1224-1227). Per-epoch scope scheduling does not exist and is
   precisely the NEW mechanism T4 proposes (now specified in row T4).
2. **T5 cost was overstated.** "One flag flip" is true only for the
   error-GATED variant. Errorless teaching (no competitor depression) is NOT
   reachable through existing parameters: the native kernel hard-codes
   `update_contacts(target, 1.0); update_contacts(winner, -1.0)`
   (lib.rs:1037-1038), and `winner_post_gate` only scales the winner's
   apposition-trace addition and pacemaker delta (lib.rs:1001-1012) — setting
   it to 0 still depresses nonzero decayed winner traces via :1038. T5 is split
   into T5a (error-gated, Python-only) and T5b (errorless, small native change).
3. **T1 citation spanned two functions.** `_flush_growth_statistics` is
   recurrent.py:764-772; :774-792 is `_consolidate_readout` (the ridge solve at
   freeze). Fixed.
4. **T6 hook sharpened.** The development controller REQUIRES the dense
   `three_factor` readout (development.py:450-451), so its stage hooks cannot
   host replay for the ridge sealed path or the sparse ladder path directly.
   Primary hook for the typed-program path is pre-freeze inside `grow()`
   (typed_program.py:1218-1228). Naming hazard flagged: `_consolidate_readout`
   (recurrent.py:774) is the ridge least-squares fit, NOT biological
   consolidation.
5. **Companion pointers could dangle.** The ladder contract exists only on the
   sibling branch named above; `a013-representation-choice.md` does not exist
   yet anywhere. Marked inline so readers do not hunt dead files.
6. **Baseline block was incomplete.** TRAIN cases 120/126, calibration cases
   7/54, min margin +56 pA over 246 competitive positions added below.

Verified-correct in the prior draft (re-checked, no change): feature vector is
exactly `[eligibility; bias]` with `feature_count = reservoir_cells + 1`
(recurrent.py:231, filled at :738-740); 3 ordered corpus passes with
per-example resets (typed_program.py:333, :1218-1221); lane supervision split
(:344-348); eligibility decay exp(-Δt/80 ms) with Δt = 5 ms/token
(recurrent.py:95, :100, precomputed per-step at :293-295, passed as
`trace_decay` at :705-707); sparse three-factor strengthens target AND depresses
current winner, with early return and NO depression when the race is won
(lib.rs:931, winner==target return :1006).

## Baseline being improved

Sealed a012 (run `hb-a012-once-20260822`, main `2b8c6a5`, exit 2, cohort
permanently spent): 3 identical epochs of observe->strengthen over 42 TRAIN
tasks — one growth example per task (`training_examples()` feeding `grow`,
typed_composition_runner.py:2065-2068) — supervision per lane (all-transition
A0/A1, program-only A2). Result at N=128 (reservoir_cells=128,
input_fanout=16, seed=17, ridge rule; typed_program.py:327-332):

| Metric | Gate | Observed |
| --- | --- | --- |
| TRAIN label replay | 42/42 | 40/42 |
| TRAIN cases | 126/126 | 120/126 |
| Calibration label replay | 18/18 | 0/18 |
| Calibration cases | 54/54 | 7/54 |
| Competitive margins | >0 finite | 246 positions, min +56 pA, median 726 pA |

State separation 1−cosine medians ~1e-3 in every lane stratum (A0 Int
0.000954 / List 0.001147; A1=A2 Int 0.001170 / List 0.001103): fresh prompts do
not separate task states at N=128. Accounting pinned by the ladder red tests:
8,714 observations/pass; A0/A1-pattern strengthen calls 26,142 total vs A2 816
(272 program targets/pass ×3 passes).

Feature-space fact (verified, recurrent.py:231): the ridge feature vector is
exactly `[eligibility vector; bias]`. There are NO token-identity features in
the readout itself; token specificity enters only through which cells the fixed
random afferent fanout excites (recurrent.py:243-251). Consequence: the
readout's capacity ceiling is set by how linearly separable different tasks are
within one shared eligibility geometry — under explicit token identities with
~1e-3 state separation, tasks collide in feature space. Adding data cannot help
once features collide; changing the code geometry can. This sharpens T1/T2.

## Axis matrix

| # | Axis | Mechanism hypothesis (code-cited) | Capacity prediction | Generalization prediction | Cost | Decisive experiment + preregistered gates |
|---|---|---|---|---|---|---|
| T1 | Data-scale dose-response | Ridge moments are plain sums A += ffᵀ, Y += f per presentation (recurrent.py:738-744, flush :764-772); accumulation saturates when new examples reuse existing feature directions; binding constraint = distinct-task count, not pass count. Corpus cap is 256 examples (typed_program.py:330, enforced :1182-1186); one example/task ⇒ up to 256 tasks without protocol change | monotone TRAIN-replay rise to a knee, then flat | flat ≈0 at every cohort size under explicit codes (features collide regardless of data volume) | S | Cohorts 21/42/84/168 tasks, nested subsets (21⊂42⊂84⊂168), fixed learner = sparse three-factor @ N=128 (L1-comparable). Gates: knee := smallest cohort with TRAIN replay = cohort size AND cases = 3×cohort; preregistered falsifier: no knee by 168 tasks, OR knee then decline, OR calibration >1/18 anywhere (would contradict the collision thesis) |
| T2 | Kind diversity — **CONDITIONAL ON REPRESENTATION** | Feature geometry decides whether a new kind adds independent directions (interference in shared 64-synapse fan-in budgets, recurrent.py:106, lib.rs fan-in loops) or fills shared structure (overlap). Under explicit identities + ~1e-3 separation, kinds collide; under distributed overlapping codes they share substrate (mechanism detail owned by representation doc) | ± depends on codes (negative under explicit, neutral-to-positive under distributed) | THE transfer lever, but ONLY with distributed/similarity codes | S per cell once codebook exists; M before that (codebook unbuilt) | Diversity-x-representation interaction table below (each cell carries a numeric falsifiable prediction) |
| T3 | Repetition schedule: spaced vs massed vs interleave | Eligibility decays exp(-Δt/80 ms), Δt=5 ms/token (recurrent.py:95,:100,:293-295,:705-707). SHARPENING: ridge sums are permutation-invariant — reordering identical presentations changes NOTHING in A,Y; schedule acts only through trace CARRYOVER changing the content of f (massed within 80 ms re-adds a near-duplicate direction — trace already saturated; spacing lets traces decay so re-presentation re-fires on fresh context). The sparse path adds genuine timing physics: per-teach trace_decay (:705-707), activity EMA (:993-999 region), structural turnover every 16 presentations (:712-723) | spaced ≥ massed at equal presentation count (massed wastes presentations on duplicate rank-1 updates); effect expected mainly in the sparse-rule variant | interleaved ≥ blocked slightly (less catastrophic overwrite of shared directions) | S | Equal-presentation-count spaced vs massed vs interleaved, sparse three-factor @ N=128 (ridge arm reported separately as order-insensitive control). Gates: capacity family (TRAIN replay + cases) and margin distribution (min/p10/median) reported per arm; preregistered direction: spaced ≥ massed on TRAIN replay; falsifier: massed ≥ spaced by ≥2 tasks |
| T4 | Supervision-density mixing | All-transition spends 26,142 strengthens on framing+program transitions; program-only spends 816 but starves framing-token structure that shapes shared eligibility geometry. Scheduled mix {all, all, program} could get both. Requires NEW mechanism: per-epoch scope override — hoist the fixed scope check (:1224-1227) into an epoch-indexed schedule (correction #1) | ++ vs program-only alone (framing supervision builds the shared geometry program decisions ride on) | + vs all-transition alone (program-token pressure preserved where it matters) | S (≈10-line scheduler + protocol plumbing) | Epoch-schedule arms {all,all,all}(=A0 replica), {program×3}(=A2 replica), {all,all,program}, {all,program,program} on one fresh cohort, sparse @ N=128. Gates: capacity family AND generalization family per arm (ladder red-test-16 pattern); preregistered: mixed ≥ both replicas on the COMBINED families; falsifier: every mixed arm strictly worse than its best replica on either family |
| T5a | Error-comparison: error-GATED updates (teach only when the race loses) | Current rule always-on teacher forcing: strengthen fires teach() on every supervised transition (recurrent.py:700-711) regardless of race outcome; gating concentrates plasticity on confusable transitions only. Python-level change: skip `self._sparse_readout.teach(...)` when `race.winner == target` | neutral-to-negative (won-but-weak transitions stop training → thinner margins) | positive mechanism candidate: discrimination pressure where the model is actually wrong → calibration cases improve | S (skip-call; no native change) | Three-arm comparison on one cohort @ N=128: always-on (= L1 rule) vs error-gated vs errorless(T5b). Gates: calibration cases family primary; margin min/p10 secondary; preregistered: error-gated > always-on on calibration cases AND error-gated < always-on on margin min; falsifier stated per arm |
| T5b | Error-comparison: ERRORLESS variant (target-only, no competitor depression) | Removes the −1.0 winner depression (lib.rs:1038) to test whether contrast is necessary for the race readout's calibration. NOT parameter-reachable (correction #2): needs native change — parameterize modulation sign or skip winner branch — plus binding rebuild; runner already pins native library identity per run (typed_composition_runner.py:2046-2051, :2069-2074) so versioning is auditable | neutral | predicted WORSE than always-on (no discrimination pressure → collisions unbroken); mirrors errorless-learning literature expectation | S-M (native diff + rebuild + identity bump) | Same three-arm design as T5a; preregistered: always-on ≥ errorless on calibration cases; falsifier: errorless strictly better |
| T6 | Replay consolidation pass | Biology: offline error-weighted reactivation consolidates tails (a013-biology-learning-fusion.md §3; Wilson & McNaughton 1994; PER à la Schaul et al. 2016). Code: grow() is strictly forward, freezes forever (typed_program.py:1218-1228; freeze :794-801); NO store/replay mechanism exists anywhere today. Replay = store teacher-forced sequences, pre-freeze error-prioritized re-strengthen. Ridge instantiation: importance-weighted moment reweighting (rows of `_target_moments`/`_normal_matrix`; sums commute so weighting ≡ replay). Sparse instantiation: genuine second teach() passes over stored sequences. Hooks verified: typed-program path pre-freeze in grow() (:1218-1228); development-controller rest phase between close_stage (:486) and prune/freeze (:487-488) — dense three_factor ONLY (:450-451); ladder path needs a new pre-freeze pass in grow() (M) | DIRECT: recovers the exact sealed tail gap (40/42 → 42/42, 120/126 → 126/126) | flat alone — replay re-imbues task-specific traces unless paired with T5/codes; this null IS the sealed interpretation's prediction | M (store + scheduler + pass; must be a preregistered protocol element — the a012 cohort is spent) | One replay pass before freeze on fresh a013 cohort, sparse @ N=128, priority = race-violating transitions. Gates: TRAIN 42/42 AND cases 126/126 required to count as recovery; calibration families REPORTED with preregistered null ≤1/18; falsifiers: (a) capacity not recovered ⇒ tail-underweighting interpretation wrong; (b) calibration moves materially ⇒ task-specific-trace interpretation wrong |

## The interaction row (first-class)

Concrete preregistered predictions per cell (numbers are gate proposals for the
protocol freeze, all at N=128 sparse unless a cell says otherwise; "TRAIN" =
label replay / cases):

| Teaching \ Codes | Explicit identities (current) | Random distributed k-of-N | Similarity-projected top-k |
|---|---|---|---|
| Single-kind curriculum | replicates sealed/L1 pattern: TRAIN ≤41/42, calibration ≤1/18. Falsifier: calibration ≥4/18 here breaks the task-specific-trace reading of a012 | memorize, no extra: calibration ≤2/18 (one family has nothing to share) | memorize, mild structure: calibration ≤2/18 |
| Diverse kinds, interleaved | interference: disjoint feature sets compete for 64-synapse fan-in budgets → TRAIN DROPS ≥2 tasks below single-kind (≤39/42), margins stay positive, calibration ≤1/18 | partial sharing: TRAIN recovers to ≥41/42, calibration 1–3/18 | shared assemblies across families: TRAIN ≥41/42, calibration ≥4/18 — transfer predicted HERE first |
| Spaced + diverse + replay | capacity fully recovers (TRAIN 42/42 AND 126/126), transfer still ~0: calibration ≤1/18 | partial transfer: calibration 2–5/18 | full prediction: transfer appears: calibration ≥8/18 |

Prediction to preregister: diversity benefit materializes only in the right
columns. Falsification clause: if diverse+spaced+replay moves calibration above
1/18 under EXPLICIT codes, the representation-dependence thesis is falsified
and representation work is deprioritized in favor of teaching-schedule work.

## Ranking by information gain per compute-minute

Stated assumptions:

- A1. The local sparse three-factor learner is implemented, native-tested, and
  CUDA-measured (lib.rs:931+; `docs/sparse-cell-readout.md`) ⇒ learner-side
  variants (T5) and schedule variants (T3/T4) are small diffs, not new math.
- A2. Compute-minutes are dominated by grow passes over tiny corpora at N=128
  (minutes per run); each experiment costs O(1) runs plus one-time plumbing;
  T5b uniquely costs a native rebuild + library-identity bump.
- A3. Every experiment uses a FRESH a013-domain cohort (a012 is permanently
  spent; the ladder forbids byte reuse) — cohort generation is cheap relative
  to runs.
- A4. Information gain = expected movement on the two sealed deficits,
  capacity-family and generalization-family pass probability; experiments
  targeting the binding constraint rank first.
- A5. The T2 axis is BLOCKED on the codebook/similarity projection being built
  (exists nowhere today) ⇒ its cells carry implementation cost M before any
  data arrives, despite highest ultimate value.

Ranking:

1. **T6 replay** — aims at the exact sealed gap (2 tasks / 6 cases); clean
   interpretation either way because BOTH outcomes are informative (gates
   above); assumption: tail-underweighting reading of uniform epochs.
2. **T3 spacing** — pure schedule change isolating trace mechanics; NOTE the
   permutation-invariance correction makes ridge-side spacing nearly vacuous,
   so the decisive variant is sparse-rule (still cheap).
3. **T5a error-gating** — Python-only skip-call against the dormant always-on
   default; direct calibration-family probe.
4. **T4 supervision mixing** — tiny scheduler, but two replica arms must also
   run for the contrast, tripling run count.
5. **T5b errorless** — same info as T5a plus a native change; sequenced after
   T5a's result.
6. **T1 dose-response** — cheap but largely confirmatory (saturation expected);
   retained because the measured knee feeds ladder-v2 sizing exactly like the
   ladder's own arm-size finalization procedure.
7. **T2 interaction cells** — highest ultimate value; executes AFTER the
   representation choice lands and the codebook exists.

## Protocol wiring

Frozen-ladder immunity: the ladder contract fixes all arm parameters before
the first run and forbids any outcome from altering later arms. Therefore NONE
of T1-T6 may modify ladder-v1 arms. Each experiment wires one of two ways:

- **Standalone dev-gate now**: fresh a013 cohort, N=128, sparse three-factor
  (L1-comparable unless stated), plan-by-default CLI with `--execute`
  authorization, exclusive spent marker per configuration (ladder claim-marker
  semantics), seal-before-score, capacity/generalization families reported
  separately, one-shot, no tuning after failure.
- **Ladder-v2 registered modifier**: recorded now by name; wired only after
  ladder v1 completes, through the same finalize-before-freeze discipline used
  for arm sizes.

| Experiment | Wiring | Red tests (written FIRST; each must fail before its feature lands) |
|---|---|---|
| T6 replay | standalone dev-gate now; ladder-v2 capacity modifier if recovered | R-T6-1 replay disabled ⇒ grow() output bytes identical to current implementation on a seeded fixture (regression guard). R-T6-2 store round-trips sequences hash-exact. R-T6-3 seeded fixture with known race violations ⇒ replay touches exactly the violating transitions (ridge: reweight set equals violation set). R-T6-4 execution refuses while replay fraction/priority constants unresolved ({PARAM} pattern) |
| T3 spacing | standalone dev-gate now | R-T3-1 schedule generator emits exactly equal presentation counts across arms; mismatch fails load. R-T3-2 fixed seed ⇒ byte-identical presentation order. R-T3-3 token-accounting block identical across schedule arms except declared schedule fields (ladder red-test-6 pattern) |
| T5a/b | standalone dev-gate now (three-arm, one cohort) | R-T5-1 gated mode: teach() call counter zero on won races, one on lost. R-T5-2 errorless mode: seeded-array native unit test shows winner contact weights unchanged after an incorrect-race update while target weights increase (test written before the native diff). R-T5-3 native library identity recorded before/after lanes; mismatch refuses production (existing runner checks reused) |
| T4 mixing | standalone dev-gate now | R-T4-1 seeded {all,program} schedule produces exactly the scheduled strengthen-target counts per epoch (accounting constants 26,142 / 816 reused as fixtures). R-T4-2 malformed schedule string or out-of-range epoch index fails closed at protocol load |
| T1 dose-response | standalone dev-gate now; knee feeds ladder-v2 sizing | R-T1-1 fixed seed ⇒ identical nested subsets (21⊂42⊂84⊂168). R-T1-2 subset/calibration disjointness asserted per cohort (fresh-cohort red-test-4 pattern). R-T3-style accounting per subset. R-T1-3 cohort exceeding 256 examples refused with cause (typed_program.py:1183) |
| T2 interaction | BLOCKED on representation choice landing; then standalone 2x2/3x3 dev-gates | inherit representation-doc red tests for the codebook; plus R-T2-1 interaction harness asserts identical cohorts/streams across code columns, differing ONLY in afferent/code assignment |

Common gates (every experiment above): fixed seeds declared in the protocol;
preregistered gates frozen before any run; capacity family (TRAIN replay +
cases) and generalization family (calibration replay + cases) reported as
separate fields — a summary omitting either is invalid; one-shot execution per
configuration with permanent spent markers; outcomes may not alter any other
experiment's parameters.
