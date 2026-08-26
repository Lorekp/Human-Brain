# Research document index

Catalogue of every document in `docs/research/`. The README links here; from
this page every research document is reachable. Statuses use a fixed
vocabulary:

- **current** — still-governing design, protocol, or decision input.
- **superseded** — replaced or overtaken by a newer document (link given).
- **falsified-negative** — the tested hypothesis failed; kept as an honest
  record with its mechanism localization.
- **historical** — frozen/spent one-shot records retained for provenance.

Outcome artifacts referenced below live under `experiments/` beside their
harness scripts, per the repo's committed-evidence convention.

## Program strategy and roadmaps

| Document | One-line description | Status |
| --- | --- | --- |
| [roadmap-reading-to-coding.md](roadmap-reading-to-coding.md) | Living roadmap: read/remember text first (Bible), coding as north star; phase gates R0→R5 and the immediate queue | current |
| [experiment-level-ladder.md](experiment-level-ladder.md) | Survey + dependency graph from single-cell to brain-level experiments with box-aware parallelization plan | current |
| [custom-architecture-directions.md](custom-architecture-directions.md) | Post-reservoir-falsification directions; spawned U-STATE (A), APICAL-AND (B), race-pivot, T-ROUTE | current |
| [r0c-strategy-decision.md](r0c-strategy-decision.md) | Adopted post-R0/R0-b mainline: associative-memory-through-race; P1/P2/P3 probes gating D5 | superseded — mainline advanced through U-STATE/APICAL-AND/T-ROUTE chain ([t-route-program-conclusion.md](t-route-program-conclusion.md)) |
| [t-route-strategy.md](t-route-strategy.md) | Adopted mainline cycle 2: trained question-space↔passage-space mapping (T-ROUTE) with G-T1..G-T3 gates | superseded — trained-mapping family falsified ([t-route-program-conclusion.md](t-route-program-conclusion.md)) |
| [gemma4-comparison-design.md](gemma4-comparison-design.md) | Fixes the Gemma-4-12B comparison target, Wilson-CI decision rules, and M4-M6 ladder before any tuning | current |
| [similar-projects-landscape.md](similar-projects-landscape.md) | Primary-source landscape of cell-based language/program-learning projects (SNN LMs, HTM/HTM-dendritic, Assembly Calculus/NEMO, Spaun, HuMAM/DTB/NEST); Human Brain's combination is unusual, not unique | current |
| [hermes-ox-alpha-workforce.md](hermes-ox-alpha-workforce.md) | First-party audit + smoke tests for using Hermes Agent / Ox Alpha as parallel research workforce on public work only | current |

## External project deep dives and path decision (2026-08-25 wave)

Primary-source full-text reads of the six closest external systems, consolidated into one fork-decision document. All six corroborate or re-price the post-T-ROUTE fork; none reopens it.

| Document | One-line description | Status |
| --- | --- | --- |
| [path-decision-synthesis.md](path-decision-synthesis.md) | Consolidated fork decision after the six deep dives: master adopt/adapt/avoid grouped by path, sharpened path-1/2/3 prospects + first experiments + costs + falsifiers, two new cheap probes (B′ APICAL-LMS Stage-0; HTM predictive-state race bias), recommended order under single-GPU + $70 vast.ai | current (active decision input) |
| [deep-dive-assembly-calculus.md](deep-dive-assembly-calculus.md) | NEMO/Assembly Calculus: repetition-rich regime avoids (not escapes) the E3 trap; position-as-place, k-cap, mutual inhibition; preregistered repetition-regime probe for path 3 | current |
| [deep-dive-dendricl-ctp.md](deep-dive-dendricl-ctp.md) | DendriCL + CTP-Hybrid: estimator-not-gate resurrects Direction B as B′ = APICAL-LMS; CTP v2.0 has no output first-spike race (landscape corrections) | current |
| [deep-dive-htm-tm.md](deep-dive-htm-tm.md) | HTM Temporal Memory: why TM sequences work where U-STATE decode failed; growth constants, context-conditioned re-representation, predictive-state race-bias probe; path 2 strengthened | current |
| [deep-dive-nest-eprop-scale.md](deep-dive-nest-eprop-scale.md) | NEST e-prop (validated three-factor rule to 2M plastic neurons) + HuMAM/DTB scale platforms that learn nothing; credit-assignment template and scale anchors for path 2 | current |
| [deep-dive-spikegpt-neuronspark.md](deep-dive-spikegpt-neuronspark.md) | SpikeGPT + NeuronSpark: ceiling of gradient-trained spiking (~7% best-case tax); full path-1 recipe template; M4–M6 undemonstrated cluster-wide | current |
| [deep-dive-spiking-reservoir-lm.md](deep-dive-spiking-reservoir-lm.md) | spiking-reservoir-lm: independent capacity-law corroboration of our termination chain; cheap methodology transfers (seed selection, pre-flights, tripwires, windowed probes) | current |

## Path-2 R3 campaign

| Document | One-line description | Status |
| --- | --- | --- |
| [r3-c1-prelaunch-audit.md](r3-c1-prelaunch-audit.md) | Independent C1 verification: current GLIF5+Python-tracker runner is a hard NO-GO because it violates the GLIF3 plateau/race contract, leaks held-out learning, and cannot meet the cap as claimed; freezes the repair gates | current — governing C1 readiness decision |
| [r3-calibration-v3-preregistration.md](r3-calibration-v3-preregistration.md) | Pre-outcome electrical/reserve/collision refreeze: q=4 and q=255 remain fixture screens; TRAIN-order R3-F0 is the sole C1-scale identity gate; capacity-v6 is conditional | current protocol — NOT RUNNABLE; no outcome or threshold selection |
| [r3-segment-growth-preregistration.md](r3-segment-growth-preregistration.md) | Frozen Path-2 C1/C2 winner-cell and distal-segment protocol with G-S0..G-S3 and stop-on-failure rules | current protocol, blocked pending CN-2 repair |
| [r3-tokenizer-freeze.md](r3-tokenizer-freeze.md) | Freezes the concept tokenizer, pool mapping, and corpus-derived vocabulary; its required pre-outcome diagnostic is committed as `experiments/r3_pilot/token_overlap_calibration.json` | current |
| [r3-drive-calibration.md](r3-drive-calibration.md) | CN-1 measures the scale-dependent ignition transition and records the CUDA observation-sink mirror defect; its ~50% active regime is too dense for C1 coincidence coding | current rheobase/telemetry evidence; C1 drive choice superseded by [prelaunch audit](r3-c1-prelaunch-audit.md) |
| [r3-learning-rule-spec.md](r3-learning-rule-spec.md) | Path-2 local learning-rule design following the segment-growth pilot | current design input; gated by valid C1 |
| [P-B race-bias findings](../../experiments/race_bias_probes/FINDINGS.md) | Predictive-state proportional bias passed the frozen shuffled/unigram/majority controls; smaller gain over the incumbent off arm remains a stated caveat | current mechanism evidence |

## Falsified-negative verdicts

| Document | One-line description | Status |
| --- | --- | --- |
| [race-pivot-gate-verdict.md](race-pivot-gate-verdict.md) | Race/output pivot gate TERMINAL INELIGIBLE at N=96; N=1024 leg not spent (stop-on-failure) | falsified-negative |
| [apical-and-verdict.md](apical-and-verdict.md) | APICAL-AND week-1 NEGATIVE-FALSIFIED at gate G-A1, cohort spent exactly once at N=1024 | falsified-negative |
| [t-route-e1e2-verdict.md](t-route-e1e2-verdict.md) | E1 NEGATIVE_CORE_WOUND (homeostat floors common bytes); E2 recovery fails sparsity (query-independent winners, shuffled enrichment 1.00–1.06) | falsified-negative |
| [t-route-program-conclusion.md](t-route-program-conclusion.md) | E3 quote-span gating fails (random-span control reproduces gated arm); local-learning escalation ladder exhausted in all three third-factor forms — program-grade termination of Bible-QA for fixed-local-connectivity; three remaining paths need user direction | falsified-negative (active decision point) |

## Frozen preregistrations and protocols (spent or awaiting runner)

| Document | One-line description | Status |
| --- | --- | --- |
| [race-pivot-protocol.md](race-pivot-protocol.md) | Winner-feedback broadcast loop + margin abstention seed; gates/falsifier frozen before implementation | historical — spent once; outcome in [race-pivot-gate-verdict.md](race-pivot-gate-verdict.md) |
| [apical-and-preregistration.md](apical-and-preregistration.md) | APICAL-AND week-1 protocol, frozen before any TwoCompartmentGLIF5 code existed | historical — spent once; outcome in [apical-and-verdict.md](apical-and-verdict.md) |
| [trained-inter-space-preregistration.md](trained-inter-space-preregistration.md) | T-ROUTE week-1 protocol (`hebbian_projection`, C4 reproduction preconditions), frozen before implementation | historical — E1/E2/E3 executed; outcomes in [t-route-e1e2-verdict.md](t-route-e1e2-verdict.md) and [t-route-program-conclusion.md](t-route-program-conclusion.md) |
| [ustate-preregistration.md](ustate-preregistration.md) | U-STATE week-1 protocol (`sdr_union` memory, G-U1/G-U2/G-U3 gates), frozen before implementation | historical — probes executed once; outcome NEGATIVE_WITH_LOCALIZATION_V2 in `experiments/ustate_probes/FINDINGS.md` |
| [ustate-change-note-1.md](ustate-change-note-1.md) | Sanctioned pre-spend amendment adding the concept-arm to U-STATE | historical — filed before any probe verdict; governs the executed arms |
| [ustate-capacity-study.md](ustate-capacity-study.md) | Measured w²/n union-capacity validation across the full λ×n×w grid, run before probe scoring | current — capacity precondition evidence for any future union design |
| [r0-multitrace-protocol.md](r0-multitrace-protocol.md) | R0 keystone protocol: multi-timescale eligibility traces, gates frozen pre-implementation | falsified-negative — arms executed on `codex/bible-qa-r2-pilot`; transfer separation ~1e-4 vs required >0.10 (`experiments/r0_multitrace/gate_results.json`) |
| [r0b-state-feature-protocol.md](r0b-state-feature-protocol.md) | R0-b successor state-feature families (spike windows, voltage quantiles, assembly overlaps) + combination arm | falsified-negative — capacity-regressive; `experiments/r0b_state_features/gate_results.json` |
| [a013-schedule-experiment-spec.md](a013-schedule-experiment-spec.md) | Spacing-vs-massing mechanical preregistration with precomputed permutation schedules | current — spec awaiting harness schedule knobs |
| [simple-swe-opening-preregistration.md](simple-swe-opening-preregistration.md) | Simple-SWE v1 opening: baselines, freeze/seed ceremony, stack verification (feeds MILESTONES.md) | current |

## A013 cognition design inputs (nothing here has been executed)

| Document | One-line description | Status |
| --- | --- | --- |
| [a013-biology-learning-fusion.md](a013-biology-learning-fusion.md) | Maps real plasticity mechanisms onto implemented equations; ranked proposals each with a decisive experiment | current |
| [a013-regioned-architecture.md](a013-regioned-architecture.md) | Region E as distributed-representation mechanism, Region P as processor | current (design proposal) |
| [a013-representation-choice.md](a013-representation-choice.md) | Encoding-axis choice for A013 (byte/concept/word representations vs cost) | current (design) |
| [a013-teaching-processes.md](a013-teaching-processes.md) | Teaching-process experiment matrix companion to representation choice | current (design) |
| [a013-cell-construction-variants.md](a013-cell-construction-variants.md) | Cell-construction variants researched against primary sources | current (design) |
| [a013-cognitive-properties.md](a013-cognitive-properties.md) | 11 candidate architecture properties mapped onto this substrate with costs and dependency edges | current (design) |
| [local-rule-constant-derivation.md](local-rule-constant-derivation.md) | Derives principled three-factor/sparse-three-factor constants pre-tuning + minimal constant screen | current |

## Corpus and tokenizer

| Document | One-line description | Status |
| --- | --- | --- |
| [cell-lm-corpus-bible.md](cell-lm-corpus-bible.md) | KJV Bible corpus pivot: freeze-candidate v3 (200 train / 60 val QA triples), BPE replay accounting | current — active corpus plan |
| [a013-word-prediction-protocol.md](a013-word-prediction-protocol.md) | Cell-LM v1 first word-prediction protocol (fiction corpus era) | superseded — corpus pivoted to KJV ([cell-lm-corpus-bible.md](cell-lm-corpus-bible.md)) |

## Scale, runtime, and hardware engineering

| Document | One-line description | Status |
| --- | --- | --- |
| [cpu-seams-spec.md](cpu-seams-spec.md) | Implementation-grade spec for `read_cells` FFI (Seam A) + indexed staged membership (Seam B); both landed, EP now strict default ≥1% activity | current (implemented) |
| [heterogeneous-placement.md](heterogeneous-placement.md) | Preregistered design for splitting large circuits across GPU, pinned RAM, virtual identity | current |
| [hybrid-partitioning-audit.md](hybrid-partitioning-audit.md) | Decision-grade CPU/GPU hybrid partitioning feasibility audit | current — operative decision: DO NOT BUILD NOW |
| [hardware-capacity-envelope.md](hardware-capacity-envelope.md) | Measured-bytes arithmetic: does 10^10 configured fit on this box? (~1.9e8 residents max) | current |
| [100-billion-cell-architecture.md](100-billion-cell-architecture.md) | Falsifiable path to a 100-billion-neuron address space | current (architecture research; partially realized by the procedural slice) |

## Cell biology and representation background

| Document | One-line description | Status |
| --- | --- | --- |
| [glif3-as-glif5-expressibility.md](glif3-as-glif5-expressibility.md) | GLIF3 is exactly expressible as GLIF5; V1 mixture cost verified with red tests | current |
| [human-ei-cell-types-and-balanced-dynamics.md](human-ei-cell-types-and-balanced-dynamics.md) | Human E/I cell families and balanced recurrent dynamics; primary-source plan | current |
| [sparse-cell-output-readout.md](sparse-cell-output-readout.md) | Evidence review for vocabulary-sized sparse output cells with fixed contact budgets | current |

## Typed programs and software tasks

| Document | One-line description | Status |
| --- | --- | --- |
| [cell-program-decoding.md](cell-program-decoding.md) | Design study: cell-only program decoding for the simple coding gate | current |
| [typed-composition-primary-sources.md](typed-composition-primary-sources.md) | Primary-source design picking the smallest informative capability increment (led to A012/A013) | current |

## Session memory

| Document | One-line description | Status |
| --- | --- | --- |
| [session-memory-addendum-v2-20260824.md](session-memory-addendum-v2-20260824.md) | Campaign state post-R0-b: durable facts, environment quirks, open decisions | current |
| [session-memory-addendum-20260824.md](session-memory-addendum-20260824.md) | Earlier Aug-24 campaign snapshot | superseded by [session-memory-addendum-v2-20260824.md](session-memory-addendum-v2-20260824.md) |
