# A013 cognitive-properties roadmap

Status: design input for preregistered protocols — nothing here has been executed.
Scope: map 11 candidate architecture properties onto THIS simulator's actual
substrate, with existing-code seeds, minimal instantiations, dependency edges
in both directions, cost tiers, and a build order. Functional framing only; no
claims about subjective experience. Companion instrument:
`docs/indicator-assessment/ASSESSMENT-v4-a012.md` (14-indicator scorecard);
a reconciliation between this roadmap and its indicator families is a section
below. Every code citation in this file was re-verified against source on
2026-08-23.

## Corrections applied in this revision

Errors found in the prior draft while verifying against source:

1. **Trace-horizon table was wrong for the voltage and ASC rows.** The prior
   draft listed voltage τ = 20 ms (~15 s horizon) and a single ASC τ = 200 ms
   (~150 s). The cell this circuit actually runs (`recurrent.py:218` defaults
   to `AllenGlif5Parameters`, cells.py:106) has membrane τ = R·C ≈
   298.635 MΩ × 263.165 pF ≈ 78.6 ms (cells.py:119–120,
   `membrane_tau_ms` cells.py:161–163) and after-spike taus (100, 33.3) ms
   (cells.py:127). The 20 ms figure matches the GLIF3 checkpoint cell of the
   older language-model path (cells.py:33–34; consumed at artifacts.py:85–88),
   not this circuit; 200 ms matches neither parameter set. The table below is
   recomputed; the qualitative conclusion survives unchanged.
2. **Broadcast-interval bound updated accordingly**: re-injection must occur
   within the ~59 s voltage horizon (eligibility ~60 s), not "~15 s".
3. **"8 mutable fields" citation corrected** to `procedural_runtime.py:7–9`,
   which lists them authoritatively (voltage, two after-spike currents, spike
   and voltage threshold components, refractory counter, recurrent synaptic
   current, eligibility). cells.py's `AdaptiveGlifState` (cells.py:287–294)
   holds the first six; synaptic current and eligibility live in the
   recurrent-circuit state.
4. **Row 4 retention numbers clarified**: the sealed-run contrast
   (7.03e-05 vs 0.9394 of prompt trace retained at answer time) is
   framing-dependent (prompt-first vs signature-first). Persistence is real;
   its measured benefit currently depends on externally chosen framing.
5. **Row 11 homeostat location clarified**: the turnover activity-EMA homeostat
   (lib.rs:1063) belongs to the sparse-readout structural path, dormant under
   the ridge default (`readout_learning_rule="ridge"`, recurrent.py:105); the
   always-on equivalent is developmental homeostatic scaling
   (development.py:34).
6. **Row 2 now records the sealed state-separation null**, so no row can be
   read as claiming integrated states already exist (scorecard RPT-2 PARTIAL
   with that diagnostic NOT SUPPORTED).

Everything else verified clean; citations stand as written.

## Substrate recap (verified anchors)

- Fitted GLIF5 point cells — `AllenGlif5Parameters` (cells.py:106), dt = 0.05 ms
  (cells.py:118); exactly 8 mutable per-cell fields
  (procedural_runtime.py:7–9).
- Fixed sparse recurrent fanout 8 (recurrent.py:94) + eligibility trace
  τ = 80 ms (recurrent.py:100); static random topology (recurrent.py:116);
  recurrent synaptic-current decay τ = 8 ms (recurrent.py:99).
- First-spike race readout over token-cells: `_compile_output_race`
  (recurrent.py:332), `race` (recurrent.py:833), whole-context currents from
  `output_current_pa` (recurrent.py:820, readout synapses × eligibility at
  :830) — winner-take-all competition among output hypotheses, one winner per
  step.
- Teacher-forced grow passes (`grow_sequence`, recurrent.py:803); strengthen =
  ridge in the sealed run (`readout_learning_rule="ridge"` recurrent.py:105;
  sealed config typed_program.py:327–332 leaves the default) or sparse
  three-factor as dormant alternative (teach path recurrent.py:677–723;
  kernel `humanbrain_sparse_three_factor_update` lib.rs:931).
- Staged recruitment/development controller (`DevelopmentalGrowthController`,
  development.py:120; frozen schedule `DevelopmentalGrowthConfig`
  development.py:26).
- Native event-driven runtime: configured address space is metadata
  (procedural_runtime.py:3), cost tracks the resident nonzero-state working
  set (evictable only when all eight fields are canonical zero,
  procedural_runtime.py:9–12); scale language is strictly qualified as
  addressable/resident/updated (scale_accounting.py:30) and this slice claims
  no bounded steady state (procedural_runtime.py:12–14); the 1e11-neuron
  address space is the architecture target (100-billion-cell-architecture.md),
  not a property of the current sealed subject.

## Property-by-property mapping

Columns: existing seed (with citations) / minimal instantiation in THIS
substrate / depends-on and unblocks (both directions) / cost S·M·L / impact on
the two sealed deficits (TRAIN-replay capacity gap 40/42; zero calibration
transfer, replay 0/18 cases 7/54) or an honest "none".

| # | Property | Existing seed (citations) | Minimal instantiation in this substrate | Depends on | Unblocks | Cost | Impact on sealed deficits |
|---|---|---|---|---|---|---|---|
| 1 | Strong recurrent processing | Fanout 8 + eligibility τ=80 ms carry prior-token state (recurrent.py:94,:100); topology static/random, never shaped by learning (recurrent.py:116). Scorecard RPT-1 PARTIAL. | Let structural scores shape edges inside the existing recruitment controller: `DevelopmentalGrowthController` already owns recruit/prune/turnover (development.py:120); feed within-run performance signals into its pruning rule instead of activity-only. | 11 (bounded reshaping), 4 (traces to score edges) | 2, 3a; longer-range binding generally | M | capacity++ indirectly: longer dependencies give multi-step programs more substrate |
| 2 | Integrated representations | Afferent+recurrent convergence on shared units; race reads whole-context currents (recurrent.py:820,:882). Sealed null: state-separation diagnostic NOT SUPPORTED (1−cosine medians ~0.001) — RPT-2 PARTIAL, distinct wholes not yet distinguishable. | Distributed overlapping output codes: replace one-hot token-cell identity with similarity-projected readout weight vectors (the readout is already a per-cell vector over eligibility features, recurrent.py:830) so similarity becomes physical current overlap at race time. | 1 (shaped recurrence stabilizes shared structure) | 3b/3c content worth broadcasting; 6; 7; quality-space readings (HOT-4) | M | generalization++ — the core fix for zero transfer (0/18, 7/54) |
| 3a | Specialized parallel modules | None: one monolithic reservoir handles all neural processing (recurrent.py:190; sealed config 128 cells, typed_program.py:327–332). Scorecard GWT-1 ABSENT. | Two populations with distinct regimes — E accumulating consolidated readout structure, P racing — wired by learned projection; staged via a `DevelopmentalGrowthConfig` extension (development.py:26) rather than new machinery. | 2 (needs codes both sides can share) | full form of 3b/3c as workspace+broadcast *between* subsystems | L | indirect via compositional codes only |
| 3b | Limited-capacity attentional workspace | The first-spike race IS winner-take-all competition among output hypotheses, one winner per step (recurrent.py:833; compiled response recurrent.py:332; decisiveness margin native.py:1382). Scorecard GWT-2 PARTIAL. | Race winners become inputs to a consumer population (workspace content) instead of terminating at the token stream/audit records. Minimal version needs only wiring (see 3c); full version needs 3a's consumer subsystems. | minimal: nothing new; full: 3a, 2 | 3c; consumption sites for 5, 7 | S–M | calibration of what gets processed per step |
| 3c | Global broadcast | Winner recorded by harness only (`_race_record`, typed_composition_runner.py:940, audit fields :945–953); no internal consumer reuses winning content. Scorecard GWT-3 ABSENT. | Re-inject the winner as an event batch into consumer populations on the next token interval — same afferent event path tokens already use (`observe`, recurrent.py:652). Bound: re-entry must land inside the ~59 s voltage / ~60 s eligibility horizons (table below). | 3b | 7 (self-model inputs); feedback pathway relevant to HOT-1; 10 partial | M | compositionality: later steps can condition on earlier decisions |
| 4 | Persistent internal state | Eligibility traces across tokens (recurrent.py:634, τ :100); fitted cell state persists within a sequence (`reset_sequence` marks the boundary, recurrent.py:645); synapse-state checkpoints (recurrent_state.py:17); 8 mutable fields total (procedural_runtime.py:7–9). Already present. | Keep; add slow weights as a consolidation target — one error-prioritized pre-freeze pass through stored sequences before `freeze_learning` (seam at recurrent.py:795–801). Fusion #2 in the biology survey. | nothing (present) | 8; scoring signal for 1 | already present | retention proven but framing-dependent: prompt-first retains ~7e-05 of prompt trace, signature-first ~0.94 (sealed results doc) — the property exists, its exploitation currently lives in the external framing |
| 5 | Metacognitive monitoring | Margins computed and stored (typed_composition_runner.py:117–118; sealed run: 246 competitive positions, min +56 pA, median 726 pA) but consumed ONLY by external gates (`_a2_capability_gates_pass` typed_composition_runner.py:1062–1075, margin checks :1072–1074; eligibility decided by harness, :152, :1078–1087). Scorecard HOT-2/HOT-3 ABSENT. | Feed each race's competitive margin back as an input feature — a confidence-cell population reading the just-completed race (margins/currents/spike times already in the audit record, runner:951). No new computation, only a consumer. | 3b/3c for a natural consumption site (direct margin input also possible) | 9 (margin progress as its goal signal); 11 (regulated quantity); calibrated abstention | S | direct: abstention/calibration gates; makes the strong-margins-no-transfer pathology visible to the system itself |
| 6 | Attention model (state-dependent selection) | None in-system: candidate masks exist (`_normalize_race_candidates`, recurrent.py:866) but allowed sets come from host-side program grammars/protocol. Scorecard GWT-4 ABSENT. | Candidate masks computed in-circuit: a gating population whose races emit allowed sets from reservoir state + margin history, replacing host grammars on the identical seam. | 2 (state must carry decision-relevant structure); 3c (broadcast supplies the state) | 9, 10 | M | generalization+ — state-dependent selection instead of template-position lookup |
| 7 | Self-model | None. Persistent state exists (eligibility recurrent.py:634; checkpoints recurrent_state.py:17) but nothing represents the system's own selection process. Scorecard AST-1 ABSENT. | Broadcast consumer trained to predict its own next winner (auxiliary head over own race outcomes); explicitly distinct from eligibility traces, which are synaptic state, not a self-model. | 3c (input content), 5 (margins as training signal) | deeper metacognitive influence (HOT-3 shape) | L | honest none for fixed-task prediction/SWE today |
| 8 | Long-term autobiographical memory | Checkpoint files are external to runs (recurrent_state.py:17 envelope); no in-run replay; consolidation/freeze seam exists (recurrent.py:795–801). | Replay store + one error-prioritized pre-freeze consolidation pass (biology survey fusion #2, survey §3). | 4 (traces/state to replay from) | stability of skills across tasks; persistence test for 1 | M | capacity — targets the TRAIN-replay gap (40/42) directly |
| 9 | Goal-directed agency | None: tasks/targets/schedules all external (`CompositionTask` typed_composition.py:97; `CompositionProtocol` :169; frozen development schedule development.py:26). Scorecard AE-1 ABSENT. | Self-generated subgoals scored by internal margin progress: a goal population whose races nominate next targets, accepted when predicted margin improves (signal from 5). | 5, 6 | 10 full loop | L | honest none for the sealed-deficit set today; out of scope for fixed-task prediction |
| 10 | Embodiment/closed loop | Outputs go to audit records; inputs teacher-supplied (scorecard AE-2 ABSENT); strengthen acts only after the fact on external targets (recurrent.py:667; PP-1 PARTIAL). | Action→observation loop against an executor: emitted tokens become sandbox actions, observations return as next inputs. The environment seam already exists in-tree (`src/humanbrain/simple_swe/sandbox.py`). | 3c (re-entry path), 6 (state-dependent action selection) | PP-1 full form (self-generated predictions shaping processing); AE-2 | L | direct for the SWE-bench rung |
| 11 | Homeostatic drives/valence | Two homeostats exist: turnover activity-EMA in the sparse structural path (`humanbrain_sparse_structural_turnover` lib.rs:1063, `output_activity_ema` :1080; decay config recurrent.py:111; called from strengthen :713–723) — dormant under ridge default — and developmental homeostatic scaling in the active recruitment path (development.py:34, recorded per stage :80–82). | Margin-distribution homeostat + valence scalar gating plasticity rate (fusion #3): post-consolidation multiplicative normalization to a target margin quantile, ordering-safe because winner latency is monotone in synaptic current (recurrent.py:890–897). | 5 (margins as regulated quantity) | keeps 1–10 stable during growth | S | prevents margin-saturation pathology (strong margins coexisting with zero transfer, as sealed) |

## Dependency-ordered build sequence (what unblocks what)

Explicit edges; "requires" lists hard prerequisites only.

| Step | Build item | Requires | Unblocks | Cost |
|---|---|---|---|---|
| 0 | (present) race + margins + traces | — | everything below reads these | — |
| 1 | Error-gated local learning + replay (fusions #1/#2 → properties 8, part of 11) | step 0 only | capacity test (TRAIN recovery with calibration flat ⇒ statistics-limited vs representation-limited); consolidation target for 8 | S–M |
| 2 | Distributed overlapping codes (property 2) | step 0 (readout recoding only); co-designed with 1 | broadcast content worth sending (3b/3c), in-circuit gating (6), self-model inputs (7), quality-space structure | M |
| 3 | Workspace completion (3b→3c) | step 2 for useful content; the wiring itself needs only step 0 | consumption site for 5; inputs for 7; feedback pathway (HOT-1 shape); partial 10 | M |
| 4 | Metacognition (5) | margins (step 0) + a consumption site (step 3, or direct margin input) | goal signal for 9; regulated quantity for 11; calibrated abstention for honest SWE scoring | S |
| 5 | Regions (3a) | step 2 | full workspace+broadcast form between subsystems (completes GWT shape) | L |
| 6 | Agentic stack (6→7→9/10) | steps 3+4 (and 2) | closed action→observation loop once the simple_swe sandbox becomes the environment | L |

Notes. Two independent entry points need nothing beyond the present
substrate: step 1 (plasticity rule swap + replay pass) and step 2 (readout
recoding). The critical path to the embodied rung is 1→2→3→4→6. Items 7 and 9
are deliberately last: they have no effect on either sealed deficit and
should not gate anything.

Out of scope for token-prediction/SWE tasks today: full self-model (7),
autonomous goal generation (9). Load-bearing even now: 1, 2, 3b/c, 4, 5, 8, 11.

## Why these matter functionally (no consciousness claims)

Recurrence binds tokens over time; distributed codes make similarity physical
so learning transfers; a workspace bottleneck forces compositionality instead
of lookup; metacognition enables calibrated abstention; consolidation makes
skills durable; homeostasis prevents the strong-margins-no-transfer pathology
observed in the sealed run. Each is an engineering property with a test, not a
philosophical stance.

## Why regions matter functionally: trace-horizon arithmetic (corrected)

Horizon = time for a field decaying exponentially from order-1 magnitude to
the float64 smallest subnormal (≈4.94e-324), i.e. τ × ln(1/4.94e-324) ≈
744 × τ, at dt = 0.05 ms. Taus from the fitted GLIF5 parameters this circuit
runs (cells.py:106, defaults via recurrent.py:218):

| field | tau | horizon |
|---|---|---|
| membrane voltage (R·C, cells.py:119–120,161–163) | ≈78.6 ms | ≈59 s |
| ASC1 (cells.py:127) | 100 ms | ≈74 s |
| ASC2 (cells.py:127) | 33.3 ms | ≈25 s |
| threshold spike component (decay 92.15/s, cells.py:131) | ≈10.9 ms | ≈8.1 s |
| eligibility (recurrent.py:100) | 80 ms | ≈60 s |

(The prior draft's 20 ms/~15 s row matches the GLIF3 checkpoint cell of the
older language-model path — R·C ≈ 19.4 ms, cells.py:33–34 — which is not the
cell in this circuit; the 200 ms ASC value matched nothing in cells.py.)

Consequence (unchanged in direction, corrected in magnitude): ALL fast traces
die within tens of seconds of biological time — fastest ≈8 s, longest ≈74 s.
Any skill requiring longer-range binding (multi-step tasks, cross-task
structure) cannot live in cell state; it must live in SYNAPTIC weights (grown
structure) or in externally refreshed re-entry. This is the physical argument
for Region E forming persistent assembly codebooks (weights) rather than
relying on transient activation, and it bounds workspace broadcast intervals:
re-injection must occur within the ~59 s voltage / ~60 s eligibility horizons
to bind. Note these are simulated-biological-time bounds; they constrain
scheduling of re-entry relative to the token stream, not engineering caching.

## Reconciliation with the indicator scorecard (Butlin-style families)

The companion scorecard (`docs/indicator-assessment/ASSESSMENT-v4-a012.md`)
scores 14 indicators across six families (RPT recurrent processing, GWT
global workspace theory, HOT higher-order thought, AST attention schema, PP
predictive processing, AE agency/embodiment). This roadmap is organized by
engineering property, not by family; the mapping is:

| Family / indicators | Roadmap properties | Current level | What movement would require |
|---|---|---|---|
| RPT-1 | 1 (+4 as its evidence base) | PARTIAL | task-shaped recurrent topology via the existing development controller |
| RPT-2 | 2 | PARTIAL (state-separation null on record) | distributed codes making distinct tasks yield separable states; re-run the failed diagnostic as the resolution test |
| GWT-1 | 3a | ABSENT | region split (E/P) |
| GWT-2 | 3b | PARTIAL (race = wta seed) | a workspace buffer holding winners, readable by ≥2 subsystems |
| GWT-3 | 3c | ABSENT | winner re-injection as input |
| GWT-4 | 6 | ABSENT | internally generated candidate masks replacing host grammars |
| HOT-1 | 3c (feedback pathway) + 2 | ABSENT | any later-to-earlier pass during inference |
| HOT-2 / HOT-3 | 5 (+11 for influence) | ABSENT | route the already-computed margins to an in-system consumer; quantities exist, consumer missing |
| HOT-4 | 2 | ABSENT | similarity-structured representational space (flat token-id population today) |
| AST-1 | 7 | ABSENT | a model of the system's own selection process; property 4's persistence does NOT count (scorecard trap already rejected) |
| PP-1 | 10 + error-gated learning (fusion #1) | PARTIAL | self-generated predictions replacing teacher forcing, errors modulating online processing |
| AE-1 | 9 | ABSENT | internally generated goals |
| AE-2 | 10 | ABSENT | an environment loop where outputs causally determine subsequent inputs |

Discipline: levels change only through sealed runs scored by the rubric — a
property appearing in this roadmap is design intent, not indicator movement.
Property 4 (persistent state) must not be read as moving AST-1, and 3b's race
seed must not be read as moving GWT-2 past PARTIAL; both constraints are
already encoded in the scorecard's evidence lines and are preserved here. No
numeric scores are implied by any ordering, cost tier, or arrow in this file.

## Near-term experiments (gates preregistered elsewhere)

- Codes experiment: explicit vs random-distributed vs similarity-projected at
  matched N/teaching; gate = calibration-transfer separation (representation
  study doc owns details).
- Replay experiment: one error-prioritized pre-freeze pass; gate = TRAIN 42/42
  recovery with calibration flat ⇒ statistics-limited vs representation-limited.
- Workspace experiment: broadcast-on/off at matched N; gate = multi-step task
  accuracy delta.

## Open questions

- Broadcast consumers: does the minimal version add one auxiliary population
  now, or wait for the E/P split so consumers are regions from the start?
- Should margin-as-feature (5) read pre-mask or post-mask margins? Post-mask
  margins are decision-relative; pre-mask margins are comparable across steps.
- Does the ~59 s binding window ever bind in practice, given wall-clock speed
  far exceeds biological time — i.e., is re-entry timing a real constraint or
  only a scheduling discipline?
- Which pruning/performance signal shapes recurrence in step 1 without
  leaking evaluation information into growth (protocol-clean reward only)?
