# Typed composition A012 v1 — sealed result

**Executed:** 2026-08-22, run metadata id `hb-a012-once-20260822`, main `2b8c6a5`.
**Outcome:** exit code 2 — valid terminal summary asserting `a2_eligible=false`. Cohort permanently spent; evidence envelope `artifacts/typed_composition_a012_v1/` (9 records, digest-manifested in `999-terminal-summary.json`).

## Gates

All 16 authenticated production execution gates PASSED (`production_execution_gates.*` all true).
The ineligibility is purely scientific: capability gates failed.

## Capability metrics (frozen gates vs observed)

| Gate | Required | Observed | Verdict |
| --- | --- | --- | --- |
| TRAIN replay | 42/42 | 40/42 | FAIL |
| TRAIN cases | 126/126 | 120/126 | FAIL |
| Calibration replay | 18/18 | 0/18 | FAIL |
| Calibration cases | 54/54 | 7/54 | FAIL |
| Decision collisions | 0 | 0 | pass |
| Competitive margins | >0, finite, positive min | 246, min +56.0 pA, median 726 pA | pass |

## Diagnostics (non-veto, recorded)

- **Prompt-trace retention (the A1/A2 framing hypothesis): CONFIRMED.**
  A0 retains ~7.03e-05 (Int) / ~6.89e-07 (List[Int]) of the prompt trace at answer time;
  A1/A2 retain 0.9394 in both strata — ~13,000× improvement. The DEV-v1
  forgetting failure mode is eliminated by signature-first/prompt-last framing.
- **State separation hypothesis: NOT SUPPORTED at 128 cells.** 1−cosine medians:
  A0 Int 0.000954 / List 0.001147; A1=A2 Int 0.001170 / List 0.001103.
  Fresh prompts alone did not make different-task states meaningfully more separable.
- Where the circuit committed to teacher targets it did so decisively
  (246 competitive positions, all finite-positive).

## Interpretation

Two separable deficits at N=128:

1. **Capacity.** Even TRAIN tasks do not fully replay (2 tasks / 6 cases short):
   the fitted-GLIF5 128-cell reservoir plus ridge readout cannot represent the
   complete three-pass supervision of the curriculum.
2. **Generalization.** Zero calibration replay with strong train margins means
   grown synapses carry task-specific traces rather than transferable program
   structure at this scale.

Both deficits motivate the preregistered scale ladder (typed-composition-a013
family): identical observation/framing discipline, N escalated across
independent cohorts, capacity and calibration tracked separately so a future
run distinguishes "too small to fit" from "fits but does not generalize".

## Dispositions

- Future teacher-free cohort: NOT generated (gate requires A2 eligible).
- No rerun, resume, seed change, repair, or tuning is possible or attempted;
  this protocol/cohort identity is spent permanently.
- Procedural-runtime CPU profiling remains sequenced after this run (no overlap occurred).
