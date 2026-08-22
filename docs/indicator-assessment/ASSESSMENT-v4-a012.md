# Indicator scorecard — Human Brain v4 (a012-era architecture)

**Assessed:** 2026-08-22. Subject commit `2f0e4e3` (worktree
`human-brain-a013-cognition-design`, branch `codex/a013-cognition-design`).
Sealed result reference: `docs/typed-composition-a012-v1-results.md`
(run `hb-a012-once-20260822`, N=128, `a2_eligible=false`).
Rubric: `docs/indicator-assessment/RUBRIC.md`. Levels are exactly
`ABSENT` / `PARTIAL` / `PRESENT` / `UNCERTAIN`. This is routine capability
accounting for version-to-version drift review; indicator presence is
evidence about architectural similarity to consciousness theories, not
evidence of experience, and no numeric or probabilistic reading is implied
or permitted.

**Subject system (v4 = a012-era):** monolithic fitted-GLIF5 reservoir with
fixed sparse recurrent fan-out and eligibility traces
(`src/humanbrain/recurrent.py`, `src/humanbrain/native.py`); first-spike
race readout over an output-cell population (ridge default, sparse
three-factor alternative) (`recurrent.py:833` `race`, `recurrent.py:105`
`readout_learning_rule`); teacher-forced grow passes
(`recurrent.py:803` `grow_sequence`); staged development controller
(`src/humanbrain/development.py`); native event-driven sparse runtime
(`src/humanbrain/native.py`). No regions, no replay, no workspace, no
top-down feedback pass, no self-model, no autonomous goals, no closed loop.

## Scorecard

| # | Indicator | Level | One-line evidence |
| --- | --- | --- | --- |
| 1 | RPT-1 recurrent perceptual processing | PARTIAL | Fixed sparse recurrent fan-out (recurrent.py:94, fan-out 8) carries prior-token state into current processing (load-bearing: prompt-trace retention drives replay in the sealed run), but topology is static/random and unshaped by learning. |
| 2 | RPT-2 integrated perceptual representation | PARTIAL | Afferent + recurrent currents converge on shared reservoir units and race over whole-context currents (recurrent.py:882), but the state-separation diagnostic in the sealed run found near-identical states across tasks (1−cosine medians ~0.001), so distinct integrated states are not demonstrated. |
| 3 | GWT-1 multiple specialized parallel modules | ABSENT | One monolithic GLIF5 reservoir handles all neural processing (recurrent.py:190 `RecurrentCellCircuit`); no distinct specialized subsystems exist. |
| 4 | GWT-2 limited-capacity attentional workspace | PARTIAL | First-spike race is genuine winner-take-all competition among output hypotheses, one winner per step (recurrent.py:833 `race`, native.py:1382 margin enforcement), but the winner is consumed only by the token stream/audit records — no workspace buffer reuses it. |
| 5 | GWT-3 global broadcast | ABSENT | The race winner is recorded for the harness (typed_composition_runner.py:940 `_race_record`); no subsystem reads winning content back as input. |
| 6 | GWT-4 state-dependent sequential attention | ABSENT | Candidate-set restrictions come from host-side program grammars/protocol, not from the system's own state; token order is teacher-forced. |
| 7 | HOT-1 top-down generative perception | ABSENT | All processing is feedforward observe→race (recurrent.py:652, :833); grow passes are teacher-forced (recurrent.py:803); no pathway from later representations back into input processing. |
| 8 | HOT-2 metacognitive monitoring | ABSENT | Competitive margins are computed (typed_composition_runner.py:117-118; sealed run: 246 positions, min +56 pA, median 726 pA) but only external gates consume them (typed_composition_runner.py:1072); no component inside the system monitors its own decision quality. |
| 9 | HOT-3 metacognition influencing beliefs/actions | ABSENT | Nothing downstream of a race reads margins or confidence; the `a2_eligible` gate decision is made by the external protocol harness, not the system. |
| 10 | HOT-4 structured quality-space representations | ABSENT | Output space is a flat token-id population; no similarity-ordered representational dimensions are represented or exploited. |
| 11 | AST-1 model of its own attention | ABSENT | Persistent internal state exists (eligibility traces, recurrent.py:634; synapse state checkpoints, recurrent_state.py:17) but nothing represents or models the system's own selection process. |
| 12 | PP-1 predictive processing | PARTIAL | Next-token prediction is the training objective and race-vs-target mismatch drives plasticity (recurrent.py:667 `strengthen` adds teaching current only on loss), but targets are externally supplied and errors never shape online perceptual processing or propagate through levels. |
| 13 | AE-1 autonomous goal-directed agency | ABSENT | All tasks, targets, and schedules come from the external curriculum/protocol (typed_composition.py:97 `CompositionTask`); no internal goal representation or generation. |
| 14 | AE-2 embodied action/perception loop | ABSENT | Emitted tokens go to audit records; subsequent inputs are teacher-supplied dataset tokens, never consequences of the system's own outputs. |

Totals by level: 4 PARTIAL, 10 ABSENT, 0 PRESENT, 0 UNCERTAIN. Counts are
bookkeeping only — they are not scores.

## Per-indicator notes (citations and trap checks)

### RPT-1 — PARTIAL
`RecurrentCircuitConfig.recurrent_fanout = 8` (recurrent.py:94) with
`recurrent_current_pa`/`recurrent_tau_ms` drives fixed random recurrent
edges; `reset_sequence` (recurrent.py:645) clears per-sequence state,
confirming within-sequence persistence. The sealed run shows context
retention is load-bearing: signature-first/prompt-last framing retains
0.9394 of prompt trace at answer time versus 7.03e-05 for prompt-first
(typed-composition-a012-v1-results.md). Not PRESENT because topology is
static and never shaped by learning — recurrence carries state, it is not a
learned transformation loop. Trap rejected: eligibility traces are
persistence, counted here only as state carry-over, not as anything
higher-order.

### RPT-2 — PARTIAL
Convergence is mechanical: each reservoir unit receives afferent plus
recurrent current, and the race integrates whole-context synaptic current
(recurrent.py:820 `output_current_pa`, :882 `_race_output_candidates`).
Not PRESENT because the sealed run's state-separation hypothesis was NOT
SUPPORTED: different-task reservoir states were nearly identical
(1−cosine medians 0.00095–0.00117 across strata). Distinct wholes do not
currently produce distinguishable states, so integration beyond
co-activation is not demonstrated. Trap rejected: the output current vector
is continuous, but continuity alone is not a quality space (see HOT-4).

### GWT-2 — PARTIAL
Competition is real and load-bearing: `race` (recurrent.py:833) runs all
output cells through one GLIF core and the first crossing wins;
`competitive_margin_pa` (native.py:1382) enforces decisiveness of winners.
Exactly one token is selected per step — capacity-limited selection exists.
Not PRESENT because there is no workspace: the winner becomes the emitted
token and an audit record (`_race_record`), and no internal consumer reuses
the winning content. Trap rejected: harness logging of winners
(typed_composition_runner.py:945-953) is experimenter-side consumption.

### GWT-4 — ABSENT
`allowed_token_ids` masking (recurrent.py:866 `_normalize_race_candidates`)
restricts races, but allowed sets are computed by host-side program
grammars and the typed-composition protocol, never from the system's
internal state. Serial token order is imposed by `grow_sequence`/the
runner. Trap rejected: external masks are not attention.

### HOT-2 — ABSENT as deployed
Margins, spike times, and per-race currents are all computed and stored —
the raw material for monitoring exists. But every consumer found
(`competitive_train_margin_count`, `minimum_train_target_margin_pa`,
typed_composition_runner.py:117-118, gate check :1072-1074) is external
protocol machinery deciding `a2_eligible` (:152, :2974). No simulated
component reads its own decision quality. This is the clearest
"available but not integrated" case in the architecture; it is scored
ABSENT because the rubric requires a consumer inside the system.

### PP-1 — PARTIAL
`strengthen` (recurrent.py:667) applies teaching current only when the race
loses to the target — an error signal — and the ridge/three-factor rules
fit readout synapses to targets; `predict_next` (recurrent.py:1016)
generates continuations from context. Not PRESENT because targets are
externally supplied (teacher forcing), errors act only on plasticity after
the fact, and no prediction biases processing of the current input.

### AE-1 / AE-2 — ABSENT
Goals and targets originate in `CompositionProtocol`/`CompositionTask`
(typed_composition.py:169, :97) and the development schedule is a frozen
configuration (development.py:26 `DevelopmentalGrowthConfig`). Emitted
races terminate in records; no emitted token returns as input through an
environment. Trap rejected: the eval harness closing a loop around the
system is not embodiment of the system.

## Changes that would move indicators (design pointers, not scores)

Tied to the sibling a013 cognition-design work (regions / workspace /
metacognition) and `docs/research/a013-biology-learning-fusion.md`.
Listed as architectural changes only; none is scored here.

- **RPT-1 → PRESENT:** learned or developmentally gated recurrent topology
  (the staged controller in development.py already owns a recruitment
  schedule; letting structural scores shape edges would make recurrence
  task-shaped).
- **RPT-2 → PRESENT:** capacity and diversity sufficient that distinct
  tasks yield separable states — the sealed run's failed state-separation
  diagnostic is the direct resolution test to re-run.
- **GWT-2 → PRESENT:** a workspace buffer that holds race winners and makes
  them readable by other subsystems (the a013 workspace design).
- **GWT-3 / GWT-4:** broadcast of workspace content to multiple regions, and
  internally generated candidate masks or gating (replacing host-side
  grammars) — the a013 regions design.
- **HOT-1:** any feedback pass from later-stage representations into input
  processing during inference.
- **HOT-2 / HOT-3:** routing the already-computed margins back into the
  system — e.g. a monitor component reading race margins and modulating
  subsequent races (metacognition design). The quantities exist; only the
  consumer is missing.
- **AST-1:** a component that models the system's own selection process
  (distinct from eligibility traces, which are synaptic state, not a
  self-model).
- **PP-1 → PRESENT:** self-generated predictions replacing teacher-forced
  targets, with errors modulating online processing.
- **AE-1 / AE-2:** internally generated goals, and an environment loop
  where emitted tokens causally determine subsequent inputs.

## Delta instructions for the next scorecard

Compare each indicator against this file, cite the new subject commit and
sealed run, and write one delta line per changed indicator in
confidence language (e.g. "confidence in GWT-2 increased: workspace buffer
added, winners now read by two subsystems"). Use
`render_scorecard.py --delta` with the machine-readable block below
extracted to JSON to render the comparison table.

## Machine-readable block

```json
{
  "RPT-1": {"level": "PARTIAL", "one_line_evidence": "Fixed sparse recurrent fan-out (recurrent.py:94) carries prior-token state into current processing (prompt-trace retention load-bearing in sealed run); topology static and unlearned."},
  "RPT-2": {"level": "PARTIAL", "one_line_evidence": "Afferent+recurrent convergence onto shared units, race over whole-context currents (recurrent.py:882); sealed state-separation diagnostic null (1-cosine medians ~0.001)."},
  "GWT-1": {"level": "ABSENT", "one_line_evidence": "Single monolithic GLIF5 reservoir (recurrent.py:190); no specialized subsystems."},
  "GWT-2": {"level": "PARTIAL", "one_line_evidence": "First-spike race winner-take-all over output hypotheses (recurrent.py:833; native.py:1382 margin enforcement); winner consumed only by token stream/audit, no workspace reuse."},
  "GWT-3": {"level": "ABSENT", "one_line_evidence": "Race winner recorded by harness (typed_composition_runner.py:940); no subsystem reads it back as input."},
  "GWT-4": {"level": "ABSENT", "one_line_evidence": "Candidate masks from host-side program grammars/protocol; token order teacher-forced; no state-dependent selection."},
  "HOT-1": {"level": "ABSENT", "one_line_evidence": "Feedforward observe->race only (recurrent.py:652, :833); grow passes teacher-forced (recurrent.py:803); no top-down pathway."},
  "HOT-2": {"level": "ABSENT", "one_line_evidence": "Margins computed (typed_composition_runner.py:117) but consumed only by external gates (:1072); no in-system monitor."},
  "HOT-3": {"level": "ABSENT", "one_line_evidence": "No component reads margins/confidence; a2_eligible decided by external harness."},
  "HOT-4": {"level": "ABSENT", "one_line_evidence": "Flat token-id output population; no similarity-structured representational space represented or exploited."},
  "AST-1": {"level": "ABSENT", "one_line_evidence": "Persistent synaptic state (recurrent.py:634 eligibility; recurrent_state.py:17) but no model of the system's own selection process."},
  "PP-1": {"level": "PARTIAL", "one_line_evidence": "Next-token prediction objective with error-driven plasticity (recurrent.py:667 teaching current on loss); targets external; no online perceptual role."},
  "AE-1": {"level": "ABSENT", "one_line_evidence": "Tasks/targets/schedules all external (typed_composition.py:97; development.py:26); no internal goals."},
  "AE-2": {"level": "ABSENT", "one_line_evidence": "Emitted tokens go to records; inputs are teacher-supplied; no environment loop."}
}
```
