# A013 cognitive-properties roadmap

Status: design input for preregistered protocols — nothing here has been executed.
Scope: map 11 candidate architecture properties onto THIS simulator's actual
substrate, with existing-code seeds, minimal instantiations, dependency edges,
cost tiers, and a build order. Functional framing only; no claims about
subjective experience. Companion scorecard instrument lives in
`docs/indicator-assessment/`.

## Substrate recap (verified anchors)

- Fitted GLIF5 point cells (`cells.py`, 8 mutable fields), dt=0.05 ms.
- Fixed sparse recurrent fanout + eligibility trace tau=80 ms (`recurrent.py`).
- First-spike race readout over token-cells (`recurrent.py:_compile_output_race`,
  `race`) — winner-take-all competition among output hypotheses.
- Teacher-forced grow passes; strengthen = ridge (sealed run) or sparse
  three-factor (dormant, `lib.rs` ~:931).
- Staged recruitment/development controller (`development.py`).
- Native event-driven procedural runtime: O(resident) cost, 1e9+ addressable.

## Property-by-property mapping

| # | Property | Existing seed | Minimal instantiation | Enables | Cost | Impact on sealed deficits |
|---|---|---|---|---|---|---|
| 1 | Strong recurrent processing | fanout+eligibility (recurrent.py:94) | learned recurrent weights (edge-resolved eligibility) | 4, integration | M | capacity++ (longer dependencies) |
| 2 | Integrated representations | afferent+recurrent convergence; race reads whole-context currents (recurrent.py:882) | distributed overlapping codes (see representation study) — similarity becomes physical overlap | workspace properties, transfer | M | generalization++ (the core fix) |
| 3a | Specialized parallel modules | none (monolithic reservoir) | Region E / Region P split (regioned doc) | 3b, 3c | L | indirect via compositional codes |
| 3b | Limited-capacity attentional workspace | first-spike race IS wta competition (recurrent.py:833) | race winners become Region-P inputs (workspace content) | broadcast | S–M | calibration of what gets processed |
| 3c | Global broadcast | winner recorded by harness only (runner:940) | winner spikes re-injected as event batch into E/P populations | self-model inputs, agency | M | compositionality |
| 4 | Persistent internal state | eligibility traces between tokens | keep; add slow weights (consolidation target) | memory properties | already present | retention proven (0.00007→0.939) |
| 5 | Metacognitive monitoring | margins computed, consumed ONLY by external gates (runner:117,:1072) | feed race margin back as an input feature (confidence cell population) | selective answering | S | direct: abstention/calibration gates |
| 6 | Attention model | host-side grammars pick candidates (not the system) | state-dependent candidate masks computed in-circuit | agency | M | generalization+ |
| 7 | Self-model | none | broadcast consumer that predicts own next winner (auxiliary output head) | metacognition depth | L | none directly for SWE tasks |
| 8 | Long-term autobiographical memory | checkpoint files (external) | replay store + consolidation pass (biology survey #2) | stability of skills | M | capacity++ (the 40/42 gap) |
| 9 | Goal-directed agency | all tasks/targets external | self-generated subgoals scored by internal margin progress | embodiment | L | out of scope for fixed-task prediction |
| 10 | Embodiment/closed loop | outputs go to records; inputs teacher-supplied | action->observation loop against an executor (SWE sandbox!) | agency, PP-1 full form | L | direct for the SWE-bench rung |
| 11 | Homeostatic drives/valence | turnover activity-EMA (lib.rs :1060+) | margin-distribution homeostat + valence scalar gating plasticity rate | stability of 1-10 | S | prevents margin saturation pathology |

## Dependency order (build sequence)

1. **Distributed codes (2)** — prerequisite for most of the rest; without shared
   structure there is nothing for modules/workspace/broadcast to organize.
2. **Error-gated local learning + replay (11, 8)** — biology fusions #1/#2;
   cheap, targets both deficits directly.
3. **Workspace completion (3b→3c)** — race winner re-injected as input;
   converts an existing primitive into a broadcast.
4. **Metacognition (5)** — margins as features; enables calibrated abstention
   (needed later for honest SWE-bench scoring).
5. **Regions (3a)** — after codes exist; E distributes, P processes.
6. **Attention model (6) → self-model (7) → agency (9)/embodiment (10)** —
   the agentic stack; embodiment lands naturally when the SWE sandbox becomes
   the environment.

Out of scope for token-prediction/SWE tasks today: full self-model, autonomous
goal generation. Load-bearing even now: 1, 2, 3b/c, 4, 5, 8, 11.

## Why these matter functionally (no consciousness claims)

Recurrence binds tokens over time; distributed codes make similarity physical
so learning transfers; a workspace bottleneck forces compositionality instead
of lookup; metacognition enables calibrated abstention; consolidation makes
skills durable; homeostasis prevents the strong-margins-no-transfer pathology
observed in the sealed run. Each is an engineering property with a test, not a
philosophical stance.

## Decisive experiments (preregistrable)

- Codes experiment: explicit vs random-distributed vs similarity-projected at
  matched N/teaching; gate = calibration-transfer separation (representation
  doc owns details).
- Replay experiment: one error-prioritized pre-freeze pass; gate = TRAIN 42/42
  recovery with calibration flat => statistics-limited vs representation-limited.
- Workspace experiment: broadcast-on/off at matched N; gate = multi-step task
  accuracy delta.
