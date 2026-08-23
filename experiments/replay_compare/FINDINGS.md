# replay_compare_v1 — findings

Dev-gate T6-lite (`docs/research/a013-teaching-processes.md`, replay-consolidation row).
Question: does an ERROR-PRIORITIZED consolidation pass before freeze beat a random-subset
replay and a full-data control on ridge-readout calibration transfer? Protocol frozen in
`PLAN.md` before any run; one shot per arm on 2026-08-23; outputs (this directory) left
uncommitted by design.

## Verdict

**Both preregistered gates FAIL.**

1. **Primary (calibration transfer) — FAIL, and sign-flipped.** Pooled over all 9 cells,
   errtop21 − rand21 = **−0.0309** (47/324 vs 57/324), Newcombe95% [−0.0875, +0.0259]:
   the CI contains zero AND the point estimate is *negative*. Per-cell paired deltas:
   **0/9 positive** (three exactly 0, six negative). Error-prioritized replay did not beat
   random replay at any cell; it was consistently equal-or-worse.
2. **Secondary (train preserved / no catastrophic forgetting) — FAIL by an order of
   magnitude.** Full-train drop vs the same-cell all42 control: pooled **+23.4 pp**
   (guard: < 5 pp); at N=1024 **+44.2 pp pooled, +45.2 pp worst cell**. Growing on only
   K=21 tasks consolidates those tasks nearly perfectly (label replay ≈ 20–21/21) while
   losing most of the unconsolidated half (~54% full-train retention regardless of arm,
   N, or seed).

## Fixed configuration (all arms)

readout=`ridge` (local rules falsified at this scale, `experiments/local_rule_screen/
FINDINGS.md`), schedule=`massed` (order-invariant for ridge, `experiments/schedule_compare/
FINDINGS.md`; pinned for reproducibility), supervision=`all_transition`, pool scale 42 TRAIN
tasks + fixed 18-task calibration cohort, codebook=`explicit`, corpus_passes=3, N ∈ {512,
1024} × seed ∈ {7, 8, 9}. Harness `benchmarks/ladder_experiments.py` imported UNMODIFIED
at worktree head `c7c08ed4b8d52147601fc14ad70484d9e1001a3c`; Python 3.11.15 venv;
`HUMANBRAIN_DEVICE=cpu`; runs serialized under `benchmarks/locks/cpu-bench.lock`
(created and released by the driver; no contention encountered).

Two passes per cell through the harness primitives: pass 1 grows all 42 tasks and records
the per-task error table (its deterministic record doubles as the full-data control);
pass 2 grows a FRESH circuit on only the K=21 highest-error tasks or a seeded random
21-subset, then freezes and evaluates on the FULL 42-task train cohort (forgetting
visible) + the same calibration cohort. 27 circuit runs (= 9 shared pass-1/control
grows + 18 pass-2 grows) produce 18 metrics records, since each cell's single
deterministic all-42 grow serves as both ranking source and full-data control;
plus one faithfulness probe.

## Integrity checks

**Concurrent-worktree note:** a separate agent modified
`benchmarks/ladder_experiments.py` (+222/−18: a `--device cuda` path and an
`--inhibitory-fraction` knob) and `experiments/SCHEMA.md` mid-batch (write time
12:53:34 local; batch window 12:52:25–12:55:39). Cells through `N1024_s7`
imported pre-edit harness bytes; `N1024_s8`/`N1024_s9` started after the write
(process-start reconstruction from per-cell wall times). This is **proven
behaviorally inert** for this experiment's cpu/explicit/default path:
(a) `src/humanbrain/recurrent.py` is byte-stable throughout (git-clean);
(b) the new plumbing passes `inhibitory_fraction=0.2`, which already was the
committed `RecurrentCircuitConfig` default (`recurrent.py:114`) that the old
harness relied on implicitly;
(c) an A/B probe executing HEAD-bytes vs current-bytes `run_experiment` with
identical arguments (ridge/massed/all_transition/scale42/N128/s7) returned
byte-equal records minus `wall_s` — the only delta is two added `config_dump`
provenance keys (`inhibitory_fraction`, `simulation_backend`), which my driver's
records do not carry (cosmetic asymmetry in `N1024_s8`/`_s9` metadata only).
No verdict below is affected. The working tree also gained unrelated untracked
files from that concurrent work; none were touched by this experiment.

- **Faithfulness**: the driver's mirrored pass-1 record equals a direct
  `ladder_experiments.run_experiment()` return byte-for-byte minus `wall_s`
  (pinned pre-execution gate, cell N512/s7) — the mirror introduces no behavior change.
- **Replication**: `all42_N512_s{7,8,9}` reproduce the pilot's
  `schedule_compare/massed_N512_s*` integer metrics exactly (106/34/5, 55/16/7,
  52/16/8 train/replay/calib) — same effective configuration, independent process.
- One `curriculum_sha256` per cell across all three arms (paired design held); the two
  selections overlap 8–12 of 21 tasks (recorded per cell below).

## Results — calib cases /54 (primary family) and full-train /126

| cell | all42 train | all42 calib | errtop21 train | errtop21 calib | rand21 train | rand21 calib | errtop−rand calib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N512 s7 | 106 | 5 | 69 | 7 | 71 | 10 | −3 |
| N512 s8 | 55 | 7 | 69 | 10 | 71 | 10 | 0 |
| N512 s9 | 52 | 8 | 65 | 10 | 68 | 10 | 0 |
| N1024 s7 | 123 | 8 | 69 | 8 | 70 | 11 | −3 |
| N1024 s8 | 126 | 10 | 69 | 5 | 72 | 6 | −1 |
| N1024 s9 | 122 | 7 | 66 | 7 | 68 | 10 | −3 |

Gate statistics (mechanically recomputed from the JSONs by `_analyze_gates.py`,
output preserved as `_gates_output.txt`):

- Primary pooled: errtop21 47/324 (0.1451) vs rand21 57/324 (0.1759),
  Δ = −0.0309, Newcombe95% [−0.0875, +0.0259]. Per-N: N512 Δ = −0.0185
  [−0.1017, +0.0649]; N1024 Δ = −0.0432 [−0.1206, +0.0342].
- Secondary: errtop21 drops vs control: N512 +2.65 pp pooled but worst-cell
  +29.4 pp; N1024 +44.2 pp pooled, worst-cell +45.2 pp; pooled +23.4 pp.
  (rand21 diagnostic: pooled +21.7 pp — forgetting is a property of SUBSET
  consolidation, not of the prioritization.)

## Why the primary could not move (selection composition)

The pass-1 error table shows the manipulation's leverage collapses exactly where
capacity is highest:

| cell | pass-1 error hist (err: count) | top-21 min error | zero-error tasks pulled in by tie-break |
| --- | --- | --- | --- |
| N512 s7 | 3:4, 2:4, 1:0, 0:34 | 0 | 13 |
| N512 s8 | 3:19, 2:7, 1:0, 0:16 | 2 | 0 |
| N512 s9 | 3:22, 2:4, 1:0, 0:16 | 3 | 0 |
| N1024 s7 | 3:1, 2:0, 1:0, 0:41 | 0 | 20 |
| N1024 s8 | 3:0, 2:0, 1:0, 0:42 | 0 | 21 |
| N1024 s9 | 3:0, 2:0, 1:0, 0:40 | 0 | 19 |

At N=1024 the grow phase makes almost every task fit (41–42 of 42 tasks at zero
errors), so "top-K-error" degenerates to the frozen index tie-break — the arm becomes
a plain first-21 prefix, and the error axis is vacuous BY CONSTRUCTION of the cohort.
The real test is N512 s8/s9, where the top-21 consists almost entirely of genuinely
high-error tasks (min error 2–3) — and there the arms tie exactly on calibration
(10 vs 10 in both cells). Where prioritization has signal, it buys nothing; everywhere
else it cannot even be expressed.

## Interpretation (against the preregistered predictions)

This is the T6 row's preregistered null landing cleanly: *"replay re-imbues
task-specific traces unless paired with T5/codes; this null IS the sealed
interpretation's prediction."* Consolidation choice reshuffles WHERE the sealed
readout is sharp (subset arms show inflated margin medians, +150…+436 pA vs controls'
+96…+226 pA, and near-perfect label replay on consolidated tasks) without moving
calibration transfer, which stays bounded by the degenerate constant-pass-through
composition documented in `experiments/readout_compare/FINDINGS.md`. Two further
diagnostics point the same way:

1. **Subset beats full-data on transfer in several cells** (e.g. N1024 s7: rand21
   calib 11 vs control 8): spending capacity on FEWER observations can improve
   generalization here — the opposite direction from "consolidate more/higher-error".
2. **Error concentration is mildly harmful**: the high-error tasks are precisely the
   ones whose programs never compile/pass (degenerate compositions); weighting toward
   them produces the only negative deltas observed.

## Scope and honest limits

- This dev-gate instantiates replay as PRE-FREEZE SUBSET SELECTION (fresh grow on a
  K-subset then freeze). It answers the delegated question — does error-prioritized
  selection of strengthened observations beat random/full controls — but NOT a
  biological store-and-reactivate pass over an already-grown circuit; no store/replay
  mechanism exists in the harness (T6 marks that as M-scale new machinery).
- At N=1024 the error signal itself is nearly absent (see composition table), so those
  six cells test "prefix-21 vs random-21", not "hard-task vs random" prioritization.
- Schema note: metrics files are schema-v1-compatible plus ONE namespaced key
  `replay_extension_v1` (per-task error tables on pass-1 records, selection rules and
  indices on all records) — declared in PLAN.md before execution. `data_scale` reports
  the materialized pool scale (42); actual growth-set size is `config_dump.train_task_count`.
- Per-task error counts were not available in schema-v1 aggregates; they were recorded
  through the harness's own completion/case-evaluation primitives inside pass 1, with
  aggregate equality to `run_experiment` verified (faithfulness check above).

## Consequences

- No widening: error-prioritized pre-freeze consolidation is not a lever for ridge
  transfer in this harness at either capacity; do not spend ladder-v2 budget on it.
- The forgetting result stands on its own: ANY 21-of-42 subset consolidation costs
  ~23–44 pp of full-train retention — consolidation coverage, not priority, is the
  binding constraint if subset-style consolidation is ever revisited.
- Losing-arm JSONs in this directory are preserved unmodified and uncommitted.
