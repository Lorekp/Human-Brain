# local_rule_constant_screen_v1 — findings

Preregistered screen executed exactly as frozen in
`docs/research/local-rule-constant-derivation.md` §4 (committed `5be40ad`),
after harness support landed (`0e51cdd`: `--local-lr-pa` / `--local-margin-pa`
plumbing + red-test suite, gates green). One shot per arm on
2026-08-23; the harness refuses metric-file overwrites, so no arm was rerun,
repaired, or retuned. **Experiment outputs (this directory) are uncommitted by
design** — only harness + tests are committed.

## Verdict

**Family falsified at this feature scale** (decision rule 2): no combo passes
the primary widen-gate, so nothing widens and the derived-constant family
(lr\* = 0.1 pA, m ∈ {363, 726} pA with ±×4 lr brackets) does not revive the
dense three_factor rule for ladder integration under massed / data_scale 42 /
all_transition at N = 512.

## Fixed configuration (all arms)

rule = `three_factor`, schedule = `massed`, data_scale = 42,
supervision = `all_transition`, N = 512, seed = 7, codebook = explicit,
harness defaults otherwise; runner `benchmarks/ladder_experiments.py`;
Python 3.11.15 venv, `HUMANBRAIN_DEVICE=cpu`; metrics recorded at git head
`1345825c87df2b900747f92142cdb42b0f8cd04a`. Injected constants verified in
every record via `config_dump.circuit_config`.

## Primary widen-gate (mechanical, recomputed from each metrics JSON)

Gate: `margin_count > 0 AND isfinite(margin_min_pa) AND margin_min_pa > 0 AND
train_replay_tasks >= 21`.

| Arm | margin_count | margin_min_pa | train_replay_tasks (/42) | Gate |
| --- | --- | --- | --- | --- |
| tf_lr0.025_m363_N512_s7 | 148 | −0.972 | 0 | FAIL |
| tf_lr0.1_m363_N512_s7 | 42 | −20.382 | 1 | FAIL |
| tf_lr0.4_m363_N512_s7 | 49 | −116.667 | 0 | FAIL |
| tf_lr0.025_m726_N512_s7 | 148 | −0.972 | 0 | FAIL |
| tf_lr0.1_m726_N512_s7 | 42 | −20.382 | 1 | FAIL |
| tf_lr0.4_m726_N512_s7 | 49 | −116.667 | 0 | FAIL |

Every arm fails BOTH positive-margin and replay clauses. Secondary metrics
(reported, not gated):

| Arm | train_cases (/126) | calib_cases (/54) | margin_median_pa |
| --- | --- | --- | --- |
| tf_lr0.025_m363_N512_s7 | 0 | 0 | −0.790 |
| tf_lr0.1_m363_N512_s7 | 15 | 10 | −16.881 |
| tf_lr0.4_m363_N512_s7 | 12 | 7 | −24.789 |
| tf_lr0.025_m726_N512_s7 | 0 | 0 | −0.790 |
| tf_lr0.1_m726_N512_s7 | 15 | 10 | −16.881 |
| tf_lr0.4_m726_N512_s7 | 12 | 7 | −24.789 |

Degeneracy probe: N/A — preregistered as a precondition for *crediting a
passing arm*; no arm passed, nothing was credited.

## Preregistered prediction scorecard

- **P1** ("lr=0.1, m=363 passes all primary gates") — **FALSE**: replay 1/42,
  margin_min −20.38 pA.
- **P2** ("lr=0.4 fails via margin degeneration or replay < 21") — nominally
  TRUE (min −116.7 pA, replay 0), but not discriminating: every arm failed,
  including the derived center.
- **P3** ("lr=0.025 fails via replay < 21") — nominally TRUE (replay 0), also
  not discriminating for the same reason.
- Family falsifier ("any combo passes while P1's combo fails") — **not
  triggered**: zero combos passed. Closure comes from decision rule 2 instead.

## Observations (diagnostic, non-gated)

1. **The margin knob was inert across its whole axis.** Each m363/m726 pair is
   byte-identical apart from `wall_s`, `arm`, and the margin value inside
   `config_dump.circuit_config` — not one strengthen decision differed. No race
   gap ever reached +363 pA during growth, so the "stop when won" regime the
   margin derivation targeted never engaged at these learning rates.
2. **The lr arithmetic did what §3 promised locally — and that was not enough.**
   Eval-time margin scales collapsed from the pilot's runaway regime (shipped
   64/64: median ≈ −8,859 pA) to single/double-digit deficits
   (−0.79 … −24.8 pA medians here). Weight motion is now bounded and graded in
   lr, exactly as the derivation intended. But replay stays at the floor
   (0–1/42) regardless.
3. **Task-level outcomes are invariant to a ~600× lr change.** The lr=0.1 arms
   reproduce the pilot's shipped-constant `three_factor_N512_s7` record exactly
   (train 15/126, replay 1/42, calib 10/54) while its internal margin scale
   differs by ~500×. Which tasks pass is set by the degenerate constant
   pass-through composition documented in
   `experiments/readout_compare/FINDINGS.md`, not by readout weight scale.
   Under this curriculum/protocol, dense three_factor output behavior is
   dominated by that degeneracy across four orders of magnitude of lr.

## Consequences (per doc decision rules)

- Rule 1 vacuous: no widening runs spent (seeds {8,9} × N {128,1024} cells
  never executed; the s7/N512 cell of any would-be grid is the screen arm).
- Rule 2: report and stop. No post-hoc tuning, no sensitivity sweep under this
  registration — any such sweep must be a new preregistered experiment citing
  this file (doc rule 3).
- Losing-arm JSONs in this directory are preserved unmodified and uncommitted.
