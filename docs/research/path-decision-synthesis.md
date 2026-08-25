# Path decision synthesis — the post-T-ROUTE fork after the six external deep dives

**Date:** 2026-08-25
**Status:** decision input for the escalation fork declared in
[t-route-program-conclusion.md](t-route-program-conclusion.md); consolidates the
six primary-source deep dives into one adopt/adapt/avoid ledger, a sharpened
per-path fork decision, two new cheap probes, and a recommended execution order
under the single-GPU + $70 vast.ai envelope.

**Inputs (abbreviations used throughout):**

| Abbr | Document |
| --- | --- |
| [DD-ASP] | [deep-dive-assembly-calculus.md](deep-dive-assembly-calculus.md) |
| [DD-DCL] | [deep-dive-dendricl-ctp.md](deep-dive-dendricl-ctp.md) |
| [DD-HTM] | [deep-dive-htm-tm.md](deep-dive-htm-tm.md) |
| [DD-NEST] | [deep-dive-nest-eprop-scale.md](deep-dive-nest-eprop-scale.md) |
| [DD-SGS] | [deep-dive-spikegpt-neuronspark.md](deep-dive-spikegpt-neuronspark.md) |
| [DD-SRL] | [deep-dive-spiking-reservoir-lm.md](deep-dive-spiking-reservoir-lm.md) |

Committed-result anchors: `t-route-program-conclusion.md` (E1/E2/E3),
`t-route-e1e2-verdict.md`, `apical-and-verdict.md` (G-A1, best median CI 1.0290
vs gate ≥ 1.5), `experiments/ustate_probes/FINDINGS.md` (NEGATIVE_WITH_LOCALIZATION_V2;
decode 2.19% < unigram 4.54%; TRAIN 38.3% → VAL 0%), `race-pivot-gate-verdict.md`
(TERMINAL INELIGIBLE at N=96), `roadmap-reading-to-coding.md` (feature-starvation
cosine 0.9989; dendritic byte prediction 86.42% vs Witten-Bell 86.40%; 1M-cell
CUDA ≈ 15 h for full KJV on this box), [similar-projects-landscape.md](similar-projects-landscape.md)
(+ dated addendum correcting CTP-Hybrid/DendriCL rows [DD-DCL]).

## Bottom line

1. **Nothing found reopens the T-ROUTE termination.** Three independent external
   angles corroborate it: the Allen-Zhu & Li ~2 bits/trained-parameter capacity
   umbrella under which our empirical chain sits [DD-SRL §2 N1]; the
   gate-collapses/estimator-succeeds dissociation on DendriCL's own baselines,
   replicating the direction of the APICAL-AND falsification [DD-DCL §2]; and
   the quantified cost of identity-bit channels (Heaviside-RWKV +0.202 BPC)
   matching our "identity bits cannot express position" statement [DD-SGS §1.3].
2. **The deep dives change the fork by re-pricing it, not reopening it.**
   Path 2 (trained-by-construction connectivity) is strengthened by an existence
   proof plus a validated credit-assignment template; path 3 (task pivot) is
   narrowed to a measurable repetition regime; path 1 (non-local optimization)
   gets a complete recipe and a calibrated price tag that consumes most of the
   $70 envelope for one honest shot.
3. **Two new cheap probes ride existing machinery and inform every path:**
   B′ = APICAL-LMS Stage-0 (plus a zero-GPU retro-probe of the already-saved
   apical cohort states) and the HTM predictive-state depolarizing bias into the
   first-spike race. Both have frozen gates, controls, and falsifiers below.
4. **Recommended order:** zero-GPU retro-probe → race-bias probe → B′ Stage-0
   → path-3 repetition probe (parallel, near-free) → path-2 mainline pilot;
   path-1's rented-GPU run last, only on explicit user ruling. Everything except
   the path-1 shot fits on the local box; the $70 exists to buy exactly the
   path-1 replication attempt (or a larger-than-local cohort later).

---

## 1. The fork, sharpened per path

### Path 1 — non-local optimization (relax local-learning-only)

- **Prospects change [DD-SGS]:** NeuronSpark proves a pure-PLIF backbone trains
  stably at ~0.9B parameters from random init (pretraining loss 3.6 after 1.4B
  tokens; fluent SFT dialogue) and publishes the complete stabilization recipe
  (surrogate-gradient BPTT, 10× LR on neuron parameters, natural-gradient
  compensation in β-space, residual centering, lateral-inhibition normalization
  = RMSNorm, structured multi-timescale init, fused scans). Under a contract
  amendment, our **Allen-fitted GLIF3/GLIF5 parameters become a strictly better
  initialization** than their hand-built statistical proxies, and the
  first-spike latency-race decoder survives path 1 intact and remains ours alone
  (v2.0 CTP-Hybrid decodes via logits — no public precedent for our output race
  [DD-DCL §3.3]). But the price is now measured: best-case spiking tax
  **+0.082 BPC (~6.8% relative)** at fully matched settings, persisting at 216M
  against GPT-2-class dense models; naive binarization costs +0.202 (~17%);
  scaling probes show ordinary diminishing returns, not gap closure. NeuronSpark
  itself demonstrates **no competitiveness**: zero benchmarks by its own
  limitation list, arithmetic 0/8, logic via keyword repetition. Zero
  code-generation results exist anywhere in the cluster, so gemma4-ladder
  M4–M6 remain undemonstrated in principle for spiking backbones. Training
  throughput asymmetry is ~3 orders of magnitude vs dense at equal size
  (~960 tok/s on 8×RTX 4090 at 0.9B).
- **Concrete first experiment:** after a formal contract amendment — train a
  45M-class stack of our fitted-cell blocks (Allen-init) with the NeuronSpark
  stabilization recipe on the frozen code-language v3 corpus, keeping the
  first-spike race decoder, against a dense RWKV-equivalent control at matched
  settings. Gate: land inside the published tax band (**within +0.08–0.10 test
  BPC of the dense control**), i.e., reproduce SpikeGPT's engineered-best-case
  relationship with physiology doing the work their hand-tuned LIF did.
- **Cost estimate:** SpikeGPT's 46M run was 12 h on 4×V100 [DD-SGS §1.2] ≈
  50–60 single-4090-class hours; at typical vast.ai rates (~$0.35–0.45/h) that
  is ~$20–30 per shot; ×3 seeds plus fine-tune ≈ **$60–90 — i.e., essentially
  the entire $70 budget for one preregistered attempt**. This is the only item
  in the program that requires the rental at all.
- **Falsifier:** if the fitted-cell-initialized system lands outside the band —
  in Heaviside-RWKV territory (> +0.15 BPC vs its dense control) — then "fitted
  physiology as initialization" buys nothing over hand-designed dynamics, and
  path 1 degenerates into re-running someone else's architecture. That outcome
  ends path 1 for this program, not just this run. Secondary tripwires adopted
  regardless: silent-path audit and rate-health checks (§2.1).

### Path 2 — trained-by-construction connectivity (R3 episodic-store mainline)

- **Prospects change:** strengthened on three independent axes.
  (a) HTM Temporal Memory is an **existence proof that surprise-gated growth of
  connectivity toward recent winners is biologically attested and compatible
  with local rules** [DD-HTM §Q3, §Relation]: it supplies concrete constants
  (θ ≈ 9–20 active connected synapses, s ≈ 15–25 grown contacts, contact cap
  ≈ 20, negative-evidence decay ≈ 4% × increment ON) and names the exact
  mechanism whose absence killed G-U2 — context-conditioned re-representation,
  making (input, context) explicit at encode time. Direction-A-style cumulative
  union states stay dead in any encoding; what transfers is structural.
  (b) e-prop supplies the **validated three-factor credit-assignment template**
  (trace × third-factor signal), demonstrated to ≈ BPTT on TIMIT at 800–2400
  neurons and scaled to 2M plastic neurons event-driven; the supervised third
  factor substitutes cleanly with a local novelty/consolidation factor computed
  from race outcomes/replay statistics, and human data enters as
  assimilation (drive/growth biases validated on held-out probes), never as
  token-level supervision [DD-NEST §4].
  (c) Assembly Calculus contributes position-as-place (dedicated populations
  per role so binding is cross-area co-activation), k-cap competitive selection
  during encoding, and repetition-aware multiplicative potentiation [DD-ASP].
  The capacity umbrella re-frames what growth must buy: count stored
  *transitions*, not bits of union support [DD-HTM §1.5].
- **Concrete first experiment:** winner-cell election + segment-growth pilot on
  the existing GLIF3 dendritic lane: grow distal segments toward recent winner
  cells during passage replay using the adopted HTM constants, elect one
  persistent winner cell per new (context, token) conjunction on burst
  (least-used/best-match tie-break), negative-evidence term ON. Measure
  transition recognition (predicted-path fraction above burst baseline) on
  train books vs held-out books at 128k–1M cells.
- **Cost estimate:** local box only. Anchor: 1M-cell CUDA full-KJV pass ≈ 15 h
  on this box (`roadmap-reading-to-coding.md`); 128k replay cohorts are
  hours. **Rental cost $0**; the $70 stays reserved for path 1 or a later
  above-local cohort.
- **Falsifier:** the recurring program failure signature — TRAIN-book
  transition recognition improves while held-out books sit at burst baseline
  (U-STATE P3: 38.3% TRAIN → 0% VAL) — means growth memorizes without
  transferable structure; kill this pilot and fall back to the probe-defined
  path 3 boundary. If instead held-out recognition moves, R3 episodic-store
  proceeds with measured mechanics behind it.

### Path 3 — task pivot (tasks identity-bit substrates can express)

- **Prospects change:** narrowed to a defined, testable regime. The Assembly
  Calculus deep dive shows the nearest-fit pivot shape is **pairwise associative
  binding (quote-assembly ↔ speaker-assembly by repeated co-firing, retrieval by
  pattern completion)** — the one task class where local Hebbian genuinely
  excels — but only under repetition-rich, co-present, well-separated exposure
  [DD-ASP §Q2]. Their own limits bound pivot size far below Bible-QA who-said:
  20–40 chained assemblies (published sequence limit), ~10³ low-overlap
  assemblies per LEX-class area vs ~3×10⁴ verses and hundreds of overlapping
  quotes at single exposure. Reservoir-lm adds a caution from the trained-readout
  side: reallocating or adding small trained capacity does not escape capacity
  competition (N6/N7: two remediation attempts, zero movement, regression on
  previously-fixed content) [DD-SRL §2].
- **Concrete first experiment:** the preregistered probe specified in
  [DD-ASP §Cheap falsification]: rerun the E2 configuration with (i)
  multiplicative `*=(1+β)` updates and (ii) k-cap winner selection active
  *during* encoding, on the synthetic-episode synthesizer (2000+ episodes ready,
  see `t-route-program-conclusion.md` §What survives) with ≥ 10 repetitions per
  quote/speaker pair; score shuffled-question enrichment stratified into
  repeated-pair vs held-out single-exposure items.
- **Cost estimate:** CPU-class hours on the existing E2 harness; **~$0**, can
  run in parallel with anything.
- **Falsifier:** prediction is enrichment recovers **only on the repeated-pair
  stratum** and collapses on single-exposure VAL items — confirming the pivot is
  viable exclusively inside repetition-regime task shapes (an honest boundary,
  now measured rather than asserted). If even the repeated stratum fails to
  enrich, multiplicative+k-cap adds nothing over E2 and the assembly-flavored
  pivot closes entirely. If single-exposure VAL items route query-dependently,
  the E3 conclusion itself is undermined and NEMO deserves an installation spike
  [DD-ASP].

---

## 2. Master adopt/adapt/avoid, grouped by path

Consolidates the per-deep-dive tables; every row cites its source deep dive.
"Adopt" = lift into workflow within the current contract; "adapt" = transferable
with rework; "avoid" = incompatible or already falsified.

### 2.1 Cross-cutting (bind under every path)

| Disposition | Item | Source | Why / anchor |
| --- | --- | --- | --- |
| Adopt | Continuous-carrier/local-events design law: inter-module messaging may carry graded state; spikes are events, not the communication alphabet | [DD-SGS] | Both gradient-funded systems converged on this independently as necessary for function |
| Adopt | Linear-probe verification before scaling: freeze checkpoint, ridge-probe internal state against a reference algorithm trajectory, gate BEFORE further spend | [DD-DCL] | Would have caught G-A2 state collapse earlier and cheaper; applies retroactively to saved npz states |
| Adopt | Seed-stability reporting: per-seed distributions and failure modes; treat grokking/bimodality as an architectural warning sign | [DD-DCL] | Their trimodal Transformer σ=0.479 vs our one-shot N=1024 discipline — complementary protections |
| Adopt | Mechanical contract tripwires: assert-no-forbidden-trainable-components; clone-and-compare weight snapshots at every checkpoint | [DD-SRL] A3 | Cell-only contract currently enforced socially; make violations fail loudly |
| Adopt | Spike-rate health pre-flight (analytic steady-state check; abort if population rate < ~0.1% or saturated) before long cohorts | [DD-SRL] A2 | Their near-silent reservoir bug caught only mid-project; pairs with our fixture-first discipline |
| Adopt | Seeded multi-candidate substrate selection: k networks differing only in construction seed, cheapest existing probe decides, spread recorded | [DD-SRL] A1 | Replaces arbitrary seed choice; minutes of cost |
| Adopt | Exposure-per-parameter budget arithmetic stated before every run (their ~60 bytes/param rule) so "undertrained" vs "falsified" is separable | [DD-SRL] A6 | Makes every future negative interpretable |
| Adopt | Any specialization run carries a general-domain replay slice + forgetting monitor at every checkpoint | [DD-SRL] A7 | Required by A013 teaching-process design; cheap and preregistrable |
| Adopt | Spiking-tax calibration numbers (+0.082/+0.202 BPC, matched settings) as the citation-grade prior in any dense-vs-cell comparison prose | [DD-SGS] Tab. 5 | Prevents optimistic drift in roadmap claims |
| Adopt | Burst fraction as a free anomaly/surprise metric in all replay logs | [DD-HTM] | Zero-cost instrumentation with TM semantics |
| Adapt | Windowed decoder probes (readout over a bounded window of recent states) as an offline diagnostic tier above single-timestep logistic/MLP probes | [DD-SRL] A4 | Separates "state lacks information" from "decoder lacks capacity"; sharpens the next localization |
| Adapt | Soft-spike graded-membrane feature φ(mem)=Φ((mem−θ)/σ) as one arm in windowed-probe sweeps, diagnostics only | [DD-SRL] A5 | Reading membrane voltage reads genuine cell state; not race-compatible; voltage-quantile features were capacity-regressive (r0b) |
| Adapt | Structured multi-timescale initialization for unfitted/synthetic cells: β logit-spaced [0.80, 0.99], firing-rate targets 25%→8%, variance-calibrated thresholds | [DD-SGS] | Published prior where Allen fits don't exist |
| Avoid | Neuromorphic op/energy credits (e.g., 32.2× at R̂=0.15 on asynchronous hardware) in fairness-rule-3 accounting | [DD-SGS] | Our comparisons are wall-clock + parameters on this box |
| Avoid | Benchmark-free fluency as success evidence | [DD-SGS], [DD-SRL] N8 | Their 0/~320 coherent generations; our free-running 17.51% gate failure |
| Avoid | Self-consistency/majority-vote decode tricks | [DD-SRL] N5 | Measured negative; reinforces mode of peaked distributions |

### 2.2 Path 1 items (inert until the user relaxes local-learning-only)

| Disposition | Item | Source | Why / anchor |
| --- | --- | --- | --- |
| Template (on ruling) | Full surrogate-gradient BPTT recipe: 10× neuron-param LR, natural-gradient compensation in β-space + per-layer gradient-norm equalization, residual centering, RMSNorm-as-lateral-inhibition, PonderNet adaptive K, fused single-pass scan kernels | [DD-SGS] §4 | Proven stable at 0.9B from random init; wholesale adoption over fitted-cell initialization once legal |
| Template (on ruling) | Pair binarized activations with a recurrent/linear-attention state carrying temporal structure the spikes cannot | [DD-SGS] | SpikeGPT token-mixer lesson; Heaviside-RWKV quantifies the alternative (+0.202 BPC) |
| Keep (ours, differentiating) | Allen GLIF3/GLIF5 fitted params as initialization; soft reset; E/I balance + cell-type heterogeneity; dendritic branches as connectivity prior; first-spike latency-race decoding | [DD-SGS] §4 | Race decode survives path 1 intact and has no public precedent [DD-DCL] |
| Keep (calibration) | Expectations: ~7% best-case tax, no demonstrated closure with scale, M4–M6 undemonstrated cluster-wide | [DD-SGS] §3 | Sharpens the gemma4 power analysis; no milestone edits warranted |
| Note | Under path 1 the [DD-DCL] "avoid importing BPTT meta-training" row lapses for new builds — but hand-set (α, γ) inside stability bounds remains preferable wherever it suffices (fewer degrees of freedom, cheaper ablations) | [DD-DCL] App. J | Architecture forces LMS structure; BPTT only tunes constants there anyway |

### 2.3 Path 2 items (mainline)

| Disposition | Item | Source | Why / anchor |
| --- | --- | --- | --- |
| Adopt | Context-conditioned re-representation as a design law: every future representation makes (input, context) explicit at encode time | [DD-HTM] | Frontiers ABCD/XBCY argument explains G-U2 failure mechanistically |
| Adopt | Sub-sampled binary coincidence constants for dendritic branches: θ ≈ 9–20 active connected synapses, s ≈ 15–25 grown contacts, contact budget cap ≈ 20, one-pattern-per-new-context allocation | [DD-HTM] | htm.core defaults + Ahmad–Hawkins error tables (fp ≤ 1e-11 regime) |
| Adopt | Negative-evidence term: decay branches that converged but were not confirmed, ≈ 4% × increment, ON by default | [DD-HTM] | Frontiers eq. (7); htm.core default-off is a known weakness not worth copying |
| Adopt | Momentary bounded predicted-unions only (membership tested by incoming input); never a cumulative state, never at n≈2048 | [DD-HTM] | arXiv:1503.07469 safe regime; our n=2048 unions sat in the flagged mix-and-match zone |
| Adopt | Three-factor skeleton dE/dW = Σ L·ē as universal update shape, with the supervised L replaced by a local novelty/consolidation factor computed from race outcomes/replay statistics | [DD-NEST] | Validated decomposition; substitution preserves locality and the cell-only contract |
| Adopt | Per-synapse eligibility formula e = ψ·z̄ with the ALIF ε_a recursion for adaptation-bearing cells | [DD-NEST] | Exact, cheap, close to our GLIF5 trace lane; ε_a is how adaptation enters credit |
| Adopt | Event-driven synapse storage + time-driven neuron archives + flush-event retrieval; learning memory bounded by touched cells | [DD-NEST] | Maps 1:1 onto resident-frontier/event-queue runtime |
| Adopt | Transmission-delay handling decoupled from solver step; left-open-interval accumulation repair | [DD-NEST] | Required discipline for any future trace rule in heterogeneous-delay topology |
| Adopt | Population-parameterized connectivity (no spatial coordinates, statistically equivalent neurons) — keep as invariant | [DD-NEST] | HuMAM/DTB confirm procedural topology at 3.47M–20B scale; `configured_ids_scanned = 0` stands |
| Adopt | Ultra-sparse long-range fan-out prior (~1.73% density at 20B neurons) for 100B-address-space planning | [DD-NEST] | DTB parameter anchor |
| Adopt | Assimilation-not-supervision human-data pattern: passage statistics enter as drive statistics and growth biases, validated against held-out QA/replay probes | [DD-NEST] | DTB Vw-dHMDA method; keeps cell-only output contract while staying data-constrained |
| Adapt | Firing-rate regularization repurposed as the stabilization homeostat scheduled for R3+ | [DD-NEST] | Matches dormant activity-EMA homeostat plan |
| Adapt | Reward-based e-prop shape (actor-critic + scalar broadcast) as template for win/lose race-outcome third factor | [DD-NEST] | Closest published diffuse-third-factor form compatible with race outcomes |
| Adapt | Winner-cell election: persist a dedicated cell per recurring context (least-used/best-match tie-break) so repeated contexts converge instead of re-bursting | [DD-HTM] | Named design gap worth one preregistered probe; maps onto dormant recruitment + three-factor machinery |
| Adapt | Two-tier branch scoring: recall-competent (connected ≥ θ_a) vs learnable (potential ≥ θ_m) for different decisions | [DD-HTM] | Cleaner than single-threshold branch lookup |
| Adapt | Position-as-place: dedicate populations to roles (speaker-area, quote-span-area, attribution-slot) so binding is cross-area co-activation | [DD-ASP] | How to express "where" without token-position bits; converges with R3 episodic reasoning |
| Adapt | Multiplicative `(1+β)` potentiation replacing `+=` — only paired with repetition-bearing episodes | [DD-ASP] | Attractor lock-in requires recurrence of events; synthetic-episode synthesizer produces them |
| Adapt | Similarity-increases-with-scale law reframed behaviorally: predict capacity/transfer improvement along the R4 ladder as residents grow | [DD-NEST] | Citable precedent for expecting fidelity gains from resident growth pre-learning |
| Avoid | Cumulative union-of-token-identities as network state, byte or concept encoding | [DD-HTM] | NEGATIVE_WITH_LOCALIZATION_V2 + flat λ sweep: conjunctions structurally unwritable |
| Avoid | Offline learned heads (logistic/MLP) as arbiter of spike-state content in future probes | [DD-HTM] | Every program failure with offline heads (r0b/r0c/U-STATE/APICAL-AND/T-ROUTE) consistent with missing state function; HTM resolves by competition |
| Avoid | Porting the algorithmic HTM simulator literally (binary permanences, pseudo-states) | [DD-HTM] | Contract requires fitted GLIF dynamics; adopt principles/constants, not codebase |
| Avoid | Supervised per-neuron gradients (L_j^t from a differentiable loss) in the production loop | [DD-NEST] | Contract violation; offline-diagnostic use only |
| Avoid | Dense per-step archiving of continuous variables for all neurons; population-wide scans during build/update | [DD-NEST] | Unbounded at our scales; peers partition-and-parameter too |
| Avoid | Tabula-rasa random connectivity + local Hebbian scaling to corpus-scale episodic QA | [DD-ASP] | No counterexample anywhere in that literature; capacity/separation math points the other way |
| Avoid | Counting NEMO-style grounding as unsupervised learning | [DD-ASP] | Planted assemblies, index-block words, scripted fibers are teacher signals |

### 2.4 Path 3 items (bounded fallback)

| Disposition | Item | Source | Why / anchor |
| --- | --- | --- | --- |
| Adopt | Pivot-task design law: reformulate targets into pairwise associative binding over repeated, co-present, well-separated pairs — never many-way discrimination between near-duplicate spans at single exposure | [DD-ASP] | The one regime separation-based theory covers; who-said verbatim sits outside it |
| Adopt | Lazy-support simulation trick (track ever-fired support; sample newcomers above the (n−k)/n quantile) for any large-k pilot | [DD-ASP] | Independently validates staged-membership engineering at 10⁵–10⁶ scale on a laptop |
| Adopt | Union actuarial math as a bounding tool: how many parallel predictions/branches may converge before mix-and-match errors matter, evaluated at true cell-space n | [DD-HTM] | Turns `ustate-capacity-study.md` + Table 4 into a design-time calculator |
| Adapt | Inter-area mutual inhibition for serial role arbitration ("one speaker candidate wins per step") in QA readout | [DD-ASP] | Clean primitive, cheaper than global WTA hacks |
| Keep as input encoders | Seeded SDR projections (deterministic, order-independent) feed branch growth; observation-sink logging/replay harness carries forward | [DD-HTM] | Salvage list from Direction A — non-state-shaped survivors |
| Avoid | Assuming pivot tasks escape capacity competition merely by being smaller | [DD-SRL] N6/N7 | Two independent remediation attempts moved nothing at ~1.5M trained params; capacity math still governs |

---

## 3. The two new cheap probes (preregistration sketches)

Both ride machinery that is bit-exact and live; neither requires the rental;
both inherit the program disciplines (fixture-first, stop-on-failure, cohort
spent once, gates frozen before implementation, mechanical tripwires per §2.1).

### Probe P-A — B′ = APICAL-LMS Stage-0 (with Stage-minus-one retro-probe)

Motivation: APICAL-AND is dead as a *gate* (best median CI 1.0290 vs ≥ 1.5,
recruitment-limited regime) — and independently corroborated dead by Active
Dendrites collapsing to chance while estimator-style architectures succeed on
the identical benchmark [DD-DCL §2]. What was never tested is Direction B's
other half: a persistent apical state hosting a *learned transformation*
(estimator), with the soma demoted to readout.

- **Step −1 (zero GPU, do first):** linear-probe the *already saved*
  `experiments/apical_probes/logged_states_apical_cohort.npz` — continuous
  subthreshold voltages, not just trace banks/spike counts — against
  decision-point variables. Falsifier: if even continuous-state linear probes
  sit inside the shuffled-label band, the problem is upstream state
  construction; corpus-level B′ dies before any build.
- **Stage 0 (days, local box):** Garg-2022-style regression task driven through
  the existing `TwoCompartmentGLIF5` substrate with a hand-implemented local
  apical update `Δu_A = α·u_A + γ·e_t·x_t` riding `observe()` (e_t =
  next-token prediction residual, available in replay; gated OFF at decision
  points; u_A never reset by somatic spikes). Hand-set α ≈ 0.99-class and γ
  inside the stability bound γ < 2/(d_eff+2); declare d_apical ≤ 384 jointly
  with γ at freeze (width cliff: d_apical ≥ 512 diverges [DD-DCL §1.3]).
  Flag-off parity must remain unchanged.
- **Gate / falsifier:** linear-probe R² of the apical trajectory vs a reference
  online-LMS trajectory ≥ pre-set threshold (benchmark shape: 0.85–0.93
  [DD-DCL §1.3]). Below threshold → a fitted GLIF5 apical membrane cannot host
  the recursion — the single biggest unknown, since DendriCL's compartment is
  an abstract linear recurrence, not fitted physiology — and B′ dies here.
- **Only on pass:** Stage 1 = decision-point decoding from u_A on corpus replay
  against the same G-A3 controls (majority prior, shuffled-label band) that
  killed APICAL-AND; Stage 2 = integrate u_A as an additional *input* to the
  output-cell race, never as a threshold gate. Stop-on-failure at every stage.

### Probe P-B — HTM predictive-state depolarizing bias into the first-spike race

Motivation: TM resolves ambiguity by dynamics — a depolarized cell spikes
first — and this race-head-start form is TM-native, untested here, and
mechanistically distinct from the falsified multiplicative plateau gate
(APICAL-AND killed gate-*as-amplifier* in the recruitment-limited regime, not
proportional bias entering a race) [DD-HTM §Q3 mapping table].

- **Mechanism:** small sustained somatic bias (or threshold offset),
  proportional to matched-segment count, on cells whose distal branch matched
  the recent winner history, consumed by the existing basket-cell first-spike
  race. Constants from §2.3 (θ ≈ 13 connected of ~20 grown to start).
- **Arms:** (i) proportional bias; (ii) shuffled-bias control (same magnitude
  distribution, random cell assignment); (iii) ± predictedSegmentDecrement
  analog (≈ 4% × increment) on branches that converged but lost.
- **Scoring:** race-resolved readouts only — no offline heads (§2.3 avoid row);
  inherit the APICAL-AND early-warning diagnostic (TRAIN self-store vs
  VAL-generalization split reported at every stage).
- **Gates:** N=96 fixture first (stop-on-failure), then one N=1024 cohort;
  pass = held-out-book span accuracy above the shuffled-bias control AND above
  the frozen comparators (unigram 4.54%, majority-prior 4.07%).
- **Cost:** hours–days on the local box; instrumentation (burst fraction) comes
  free per §2.1.
- **Falsifier:** depolarized arm ≤ shuffled-control arm, or movement confined
  to TRAIN books (the 38.3%-to-0% signature) → predictive-state-as-race-head-
start joins the dead list; TM's residual value reduces to representation
discipline and constants, and path 2 loses its cheapest mechanism candidate.

---

## 4. Recommended execution order under single-GPU + $70 vast.ai

Ordering principle: cheapest-most-informative first; nothing blocks the user's
fork ruling except the path-1 rental; every step ends in a committed verdict
doc that feeds the next decision. Rental anchors: SpikeGPT 46M = 12 h on
4×V100, 216M = 48 h on 4×V100 [DD-SGS]; NeuronSpark sustained ~960 tok/s on
8×RTX 4090 at 0.9B [DD-SGS]; local anchor: full KJV at 1M CUDA cells ≈ 15 h on
this box (`roadmap-reading-to-coding.md`). Vast.ai prices are market-typical
ranges, not commitments.

| # | Step | Compute | Rental cost | Informs |
| --- | --- | --- | --- | --- |
| 0 | P-A step −1 retro-probe of saved apical states (linear probes, CPU) | hours | $0 | Kills or licenses corpus-level B′ before any spend |
| 1 | Probe P-B race-bias (fixture N=96 → cohort N=1024, stop-on-failure) | local, 1–2 days | $0 | Cheapest untested mechanism candidate; composes with any path outcome |
| 2 | P-A Stage 0 (Garg regression through TwoCompartmentGLIF5) | local, days | $0 | Direction-B revival vs permanent death of dendritic estimation |
| 3 | Path-3 repetition-regime probe (E2 + `*=(1+β)` + k-cap on synthetic episodes) | CPU, hours | $0 | Measures the exact boundary of any task pivot; runs parallel to 1–2 |
| 4 | Path-2 mainline pilot: winner-cell election + segment-growth constants on 128k–1M Bible-QA replay | local, ≤ 15 h/cohort | $0 | R3 episodic-store goes from design to measured mechanics |
| 5 | **Decision point:** with 0–4 committed, present the sharpened fork to the user with measured gates attached | — | — | Fork ruling now lands on evidence, not prospects |
| 6 | Only on explicit path-1 ruling: contract amendment + 45M fitted-cell BPTT replication vs dense RWKV control, band +0.08–0.10 BPC | rented 4090-class, 3 seeds | ~$60–90 (≈ the whole $70 for one shot) | Tests the only untested global hypothesis: fitted physiology as gradient-training initialization |

If the $70 must be spent before step 5 for scheduling reasons, the only
defensible rental target is step 6 shrunk to a single seed (~$20–30); spending
it on larger cohorts of steps 1–4 is dominated by the local box, which already
demonstrated the required scale end-to-end.

## Traceability notes

- Every adopt/adapt/avoid row cites its source deep dive; deep dives in turn
  cite primary texts read in full this session (verification ledgers therein).
- Committed-result numbers quoted above come from the verdict/indexed documents
  listed in the header; no number is introduced here that does not appear in a
  deep dive or a committed result/artifact.
- The [DD-DCL]-owed corrections to `similar-projects-landscape.md` are already
  applied as its dated addendum, committed alongside this document.
- Cost figures for vast.ai are estimates anchored on published run sizes and
  market-typical rates; they are planning inputs, not measurements, and are
  labeled as such.
