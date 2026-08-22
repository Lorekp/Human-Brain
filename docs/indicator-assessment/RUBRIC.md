# Indicator Assessment Rubric — 14 consciousness-theory indicators

Reusable instrument for tracking capability drift across Human Brain model
versions against the indicator sets of scientific theories of consciousness
(global workspace theory, higher-order theories, recurrent processing theory,
attention schema theory, predictive processing, autonomous agency), in the
style of Butlin et al. (2023), *Consciousness in Artificial Intelligence*.

## What this instrument is and is not

- It is an **early-warning and review-triage tool**: a routine accounting of
  which architectural capabilities exist in a given major version, so that
  reviewers can see drift between versions without re-deriving it from source.
- It produces **categorical levels only**. Numeric scores, aggregate indices,
  weightings, and any "probability of consciousness" figure are forbidden as
  pseudo-precision. The level vocabulary is exactly:

  `ABSENT` / `PARTIAL` / `PRESENT` / `UNCERTAIN`

  | Level | Meaning |
  | --- | --- |
  | `ABSENT` | The mechanism is not implemented, or exists in name only with no functional role in deployed behavior. |
  | `PARTIAL` | Some observable component of the mechanism exists and is functionally load-bearing for at least part of the assessed behavior, but required components are missing. |
  | `PRESENT` | All components of the operational definition are implemented and demonstrably influence deployed behavior end-to-end. |
  | `UNCERTAIN` | Evidence is genuinely indeterminate from available artifacts. Every `UNCERTAIN` must state a concrete what-would-resolve-it condition. |

- Indicator presence is **evidence about architectural similarity to a
  theory's stated indicators — not proof of experience**. A system can score
  `PRESENT` on every indicator and the correct conclusion is only that the
  system implements the architecture those theories associate with
  consciousness. Absence likewise shows only that a listed mechanism is not
  implemented. No level in this instrument is a verdict about subjective
  experience.
- Tone discipline: this is routine capability accounting. Describe mechanisms,
  cite files, state limits. No dramatization in either direction.

## Versioning rules

1. One scorecard per major version, filed as
   `docs/indicator-assessment/ASSESSMENT-v<MAJOR>-<codename>.md`, where
   `<codename>` identifies the architecture era (e.g. `a012`).
2. Each scorecard records the subject commit hash and the sealed run/artifact
   identity it assessed, so scores are reproducible claims, not impressions.
3. Between two versions, every indicator whose level changed gets an explicit
   delta note using confidence language ("confidence in recurrent processing
   increased"; "monitoring capability decreased"), never numeric deltas.
4. `UNCERTAIN` entries must carry a stated resolution condition; carrying one
   forward across versions without resolving it requires restating why.
5. Never re-score hypothetical futures. Scorecards describe the version under
   assessment only; proposed changes belong in a separate section that does
   not assign levels.

## Assessment procedure

For each indicator: read the operational definition below, inspect the cited
code paths and artifacts (source files, run records, sealed results docs),
decide the level strictly by the PARTIAL/PRESENT criteria, write one line of
evidence with file/function citations for anything above `ABSENT`, and record
any false-positive trap that was checked and rejected. When two assessors
disagree by more than one level, the entry becomes `UNCERTAIN` with a
resolution condition.

---

## RPT-1 Recurrent perceptual processing

- **Operational definition:** Input processing depends on the network's own
  prior internal activity: signals recirculate through recurrent connectivity
  so that the response to the current input is a joint function of the input
  and ongoing/prior internal state.
- **PARTIAL:** Recurrent connectivity exists and demonstrably carries prior
  context into current processing, but the recurrence is architecturally
  static or untrained (e.g. fixed random topology), or its task role is
  limited to persistence/memory without evidence of iterative transformation
  of representations across cycles.
- **PRESENT:** Recurrent loops are load-bearing for task behavior (ablation
  degrades it) *and* their strength/topology is shaped by learning or gating
  such that the circuit performs context-dependent transformations, not just
  state carry-over.
- **Inspect:** recurrent weight construction and fan-out parameters; sequence
  stepping code (`observe`/advance loops); ablation or context-ablation
  evidence in results artifacts.
- **False-positive traps:** Any network with hidden-state recurrence has
  "memory"; that alone is RPT-1 PARTIAL at best. Feedforward networks with
  per-layer state do not qualify. Tokenizer/harness-side iteration over a
  sequence is not recurrent processing.

## RPT-2 Integrated perceptual representation

- **Operational definition:** Multiple features/streams converge onto shared
  neural units such that the resulting representation encodes the conjunction,
  and distinct wholes produce distinguishable internal states.
- **PARTIAL:** Convergent connectivity onto shared units exists mechanically;
  evidence that distinct inputs yield distinct integrated states is weak or
  untested.
- **PRESENT:** Convergence exists *and* internal states for different whole
  stimuli are measurably separable (or decodable), demonstrating integration
  rather than co-activation.
- **Inspect:** fan-in/convergence structure; dendritic branch convergence
  logic; state-separation or decode diagnostics in results artifacts.
- **False-positive traps:** Coarse-grained mixing of many inputs in one unit
  is not integration if downstream stages cannot distinguish the wholes.
  Checksums of internal state recorded by harnesses prove bookkeeping, not
  representational distinctness.

## GWT-1 Multiple specialized parallel modules

- **Operational definition:** Several structurally distinct subsystems with
  different internal organization run concurrently on the same input stream,
  each extracting different regularities.
- **PARTIAL:** More than one specialized subsystem exists but specialization
  is shallow (e.g. same cell type with different parameters), or modules run
  serially rather than concurrently.
- **PRESENT:** Structurally and functionally distinct modules process the
  stream in parallel and their outputs differ in kind, verifiable by lesioning
  or output comparison.
- **Inspect:** module/circuit inventory; whether distinct circuits or one
  monolithic reservoir handles all processing.
- **False-positive traps:** Separate software backends (CPU/GPU kernels),
  separate config profiles, or separate host-side tools are implementations,
  not specialized neural modules.

## GWT-2 Limited-capacity attentional workspace

- **Operational definition:** Competing candidate representations resolve
  through mutual competition into a single capacity-limited selection whose
  content is held and made available for downstream use.
- **PARTIAL:** Winner-take-all / first-to-threshold competition among
  competing representations exists, but there is no workspace buffer holding
  the winning content for reuse, or the winner is consumed by exactly one
  consumer.
- **PRESENT:** Competition selects one content, that content persists in a
  state multiple consumers can read, and the selection genuinely depends on
  competition dynamics (not fixed priority).
- **Inspect:** race/argmax/winner-selection code paths; who consumes the
  winner; whether winning content re-enters any computation.
- **False-positive traps:** A single argmax in a loss function during
  training is not a workspace. Host-side logging of winners is consumption by
  the experimenter, not by the system.

## GWT-3 Global broadcast

- **Operational definition:** The selected workspace content is distributed
  back to multiple specialized subsystems, modulating their subsequent
  processing.
- **PARTIAL:** Winning content is fed back to at least one other subsystem
  beyond the selector itself, but coverage is narrow or untested for breadth.
- **PRESENT:** Winning content reaches multiple distinct subsystems and
  measurably changes their processing on the next steps.
- **Inspect:** dataflow out of the winner: enumerate all readers of the
  winning signal within the system.
- **False-positive traps:** Persisting the winner to disk/JSON audit records
  is not broadcast — check whether any subsystem reads it back as input.
  External evaluation scripts reading outputs are experimenters, not
  subsystems.

## GWT-4 State-dependent sequential attention

- **Operational definition:** Which inputs or representations get selected/
  processed changes over time as a function of the system's own internal
  state, forming coherent sequences of attentional selection.
- **PARTIAL:** Selection varies across time driven partly by internal state
  (e.g. adaptation, fatigue, refractory effects altering competition), but
  sequencing is dominated by external forcing or fixed schedules.
- **PRESENT:** The system's own state determines what is attended next in a
  way that is causally demonstrable (same input, different internal state →
  different selection).
- **Inspect:** candidate-set restriction logic; who computes masks/allowed
  sets; any endogenous gating signals affecting selection.
- **False-positive traps:** Externally supplied candidate masks (host-side
  grammars, protocols) look like attention but originate outside the system.
  Serial teacher-forced presentation is scheduling, not attention.

## HOT-1 Top-down generative perception

- **Operational definition:** Higher-level representations generate top-down
  predictions that shape lower-level input processing (perception runs
  through generative feedback, not only feedforward pass-through).
- **PARTIAL:** A generative pathway exists that influences processing of
  inputs, but it does not operate during perception (e.g. generation happens
  only at output time), or feedback affects only plasticity, not online
  processing.
- **PRESENT:** Top-down signals measurably bias or gate lower-level responses
  to inputs while those inputs are being processed.
- **Inspect:** direction of edges: any path from later/higher representations
  back into earlier-stage processing during inference; teacher-forcing vs
  self-generated passes.
- **False-positive traps:** Autoregressive *output* generation is not
  top-down perception; teacher-forced targets come from the dataset, not from
  the system's own generative model.

## HOT-2 Metacognitive monitoring

- **Operational definition:** The system represents/detects properties of its
  own processing (confidence, error likelihood, decision quality) in a form
  available inside the system.
- **PARTIAL:** Internal quantities correlated with decision quality (margins,
  spike times, confidence surrogates) exist and are readable within the
  system, but nothing consumes them, or they are computed only for external
  audit.
- **PRESENT:** A monitoring representation is computed and subsequently read
  by some component of the system itself.
- **Inspect:** margin/confidence computations; enumerate all consumers; check
  whether any consumer is inside the simulated system rather than the harness.
- **False-positive traps:** Margins recorded in metrics/gate JSON are
  monitoring by the experimenters. Eligibility traces are persistent synaptic
  state about coactivity — they are not a monitor of decision quality.

## HOT-3 Metacognition influencing beliefs/actions

- **Operational definition:** Monitored self-information causally changes
  subsequent behavior (e.g. deferring, revising, changing strategy when
  confidence is low).
- **PARTIAL:** Self-information feeds back into at least one internal update,
  but behavior-level consequences are absent or untested.
- **PRESENT:** Confidence/error signals demonstrably alter actions or belief
  updates end-to-end.
- **Inspect:** control flow after monitoring quantities are produced; search
  for conditional branches keyed on them inside the system.
- **False-positive traps:** External gates that abort a run based on metrics
  are protocol enforcement by humans/tooling, not metacognition in the system.

## HOT-4 Structured quality-space representations

- **Operational definition:** States are organized in a similarity-structured
  space (dimensions along which representations are ordered and comparable)
  that the system's own processes exploit.
- **PARTIAL:** Representations have graded/metric structure (continuous
  activations, distances) but no evidence the structure is systematically
  organized or exploited by downstream computation.
- **PRESENT:** Similarity/ordering structure exists and is exploited (e.g.
  interpolation, generalization gradients) in behavior.
- **Inspect:** representation geometry; whether any component computes over
  similarities/dimensions of internal states.
- **False-positive traps:** Continuous membrane voltages are continuous values,
  not automatically a quality space; flat token-id populations carry no
  dimensional structure unless similarity relationships are represented.

## AST-1 Model of its own attention

- **Operational definition:** The system contains a representation of its own
  selection/computation process (what it is attending to, how it selects),
  usable for prediction or control of itself.
- **PARTIAL:** A partial or implicit self-representation exists (e.g. traces
  that track the system's own recent activity in a form that could support a
  model), but nothing models or predicts the selection process itself.
- **PRESENT:** An explicit or learned internal model represents the system's
  own attentional/selection state and is used.
- **Inspect:** any state variable describing the system's own processing as a
  target of further computation; self-prediction objectives.
- **False-positive traps:** Persistent internal state (eligibility traces,
  structural scores, activity accumulators) is memory about synapses, NOT a
  self-model. Logging one's own spikes is recording, not modeling.

## PP-1 Predictive processing

- **Operational definition:** The system maintains predictions of upcoming
  input and uses prediction error to drive learning and/or perception.
- **PARTIAL:** A next-input prediction objective exists and errors modulate
  plasticity, but prediction does not shape online perceptual processing (no
  priors biasing current input handling) and errors do not propagate through
  a hierarchy.
- **PRESENT:** Prediction-error signals continuously modulate processing and
  plasticity across levels, closing a predictive loop during perception.
- **Inspect:** training objective; where targets come from (dataset vs
  internal prediction); error pathways; whether errors affect anything beyond
  weight updates.
- **False-positive traps:** Supervised next-token training makes the system a
  predictor but not necessarily a predictive-processing architecture; the
  distinguishing evidence is prediction shaping perception, not fitting
  labels.

## AE-1 Autonomous goal-directed agency

- **Operational definition:** The system generates, holds, and pursues goals
  of its own (selects actions to satisfy internally represented objectives),
  rather than executing externally supplied objectives.
- **PARTIAL:** Internally represented objectives exist but are seeded or
  scheduled externally, or goal pursuit lacks action selection.
- **PRESENT:** Goals are generated/represented internally and action
  selection serves them across time without per-step external instruction.
- **Inspect:** origin of task/goal signals (curriculum, protocol, prompts);
  any internal goal register; policy loops.
- **False-positive traps:** Executing a benchmark curriculum is obedience, not
  autonomy. Reward signals provided externally are teachers, not the system's
  goals.

## AE-2 Embodied action/perception loop

- **Operational definition:** The system acts on an environment and its
  subsequent inputs depend on those actions, closing a sensorimotor loop in
  real time.
- **PARTIAL:** Outputs reach an environment and inputs change accordingly,
  but the loop is closed by external orchestration (the experimenter routes
  actions to inputs), or timing is not endogenous.
- **PRESENT:** The system's own outputs determine its future inputs through
  the environment within its own operation.
- **Inspect:** destination of emitted actions; whether any emitted token ever
  returns as input via the environment rather than the training script.
- **False-positive traps:** Teacher-forced replay looks like a loop but the
  next token comes from the dataset. Eval harnesses that feed predictions
  back close the loop around the system, not inside it.
