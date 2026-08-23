# replay_compare_v1 — preregistration (frozen before execution)

Dev-gate T6-lite (`docs/research/a013-teaching-processes.md`, replay-consolidation row):
does an ERROR-PRIORITIZED consolidation pass before freeze beat a random-subset
replay and a full-data control on ridge-readout calibration transfer?

Frozen 2026-08-23, before any run of this cohort. Harness:
`benchmarks/ladder_experiments.py` @ worktree head `c7c08ed4b8d52147601fc14ad70484d9e1001a3c`
(unmodified; driver imports it, never edits it). Python 3.11.15 venv
(`C:/Users/laure/Desktop/DeepSim/Human Brain/.venv`), `HUMANBRAIN_DEVICE=cpu`.

## Rationale (why this is not the falsified schedule axis)

The schedule experiment (`experiments/schedule_compare/FINDINGS.md`) proved ridge is
order-INVARIANT for a fixed observation multiset. Subset SELECTION changes the multiset
itself: consolidating only K tasks' observations before freeze yields a different
`(A, Y)` and hence a different sealed readout. The manipulated variable is WHICH tasks
get strengthened, not in what order.

## Fixed configuration (all arms)

readout=`ridge`, schedule=`massed` (inert for ridge; pinned for reproducibility),
supervision=`all_transition`, data pool scale=42 TRAIN tasks + fixed 18-task calibration
cohort, codebook=`explicit` (assembly-k 16), corpus_passes=3, harness defaults otherwise.
Cells: N ∈ {512, 1024} × seed ∈ {7, 8, 9} (9 cells).

## Arms (per cell) — 27 circuit runs total

1. **pass 1 / `all42_N{n}_s{s}`** — grow on all 42 tasks, freeze_learning, evaluate.
   One grow per cell serves BOTH as ranking source and as the full-data control arm
   (the harness is deterministic per seed; re-growing would reproduce it byte-for-byte).
   Its metrics record embeds the per-task train outcome table (schema extension below).
2. **`errtop21_N{n}_s{s}`** — fresh circuit; pass-2 growth over ONLY the K=21 tasks with
   the highest pass-1 error counts, then freeze, then evaluation on the FULL 42-task
   train cohort (126 cases) + the same calibration cohort.
3. **`rand21_N{n}_s{s}`** — identical mechanics with a uniformly random 21-subset.

Per-task error count := `cases_total − cases_passed` for that task's pass-1 train
completion (each task has exactly 3 cases). Ranking key (pinned): `(-error_count,
train_index)` ascending — ties broken by lower train index. Selected indices recorded.

Random-subset RNG (pinned): `np.random.default_rng([seed, 0x5EEDC0DE]).choice(42,
size=21, replace=False)`, indices sorted ascending before use; salt domain-separated
from every harness stream documented in SCHEMA.md.

## Evaluation semantics

Pass-2 arms are evaluated on ALL 42 train tasks so forgetting of unconsolidated tasks
is visible (denominator 126 everywhere); calibration is always the same fixed 18-task /
54-case cohort. Within a cell all three arms share one `curriculum_sha256` (asserted).

## Schema note (declared deviation)

Metrics files keep every schema-v1 field byte-compatible (`data_scale` reports the
materialized pool scale 42; actual growth-set size lives in
`config_dump.train_task_count`). ONE namespaced top-level key `replay_extension_v1` is
added per record carrying: pass role, selection rule string, `selected_train_indices`,
and (pass-1 records only) the full per-task error table. This is a declared schema-v1
extension, listed here BEFORE execution.

## Gates (mechanical, recomputed from the JSONs by `_analyze_gates.py`)

CI method pinned to the schedule_compare precedent: Wilson score intervals per
proportion (z = 1.959963984540054) combined into Newcombe method-10 hybrid-score
intervals for the difference of pooled independent binomials.

- **PRIMARY**: calibration fraction delta (errtop21 − rand21) > 0, pooled over all
  9 cells (k/n = summed passed/total per arm), Newcombe 95% CI excluding zero on the
  positive side ⇒ PASS. Per-N breakdowns and per-cell paired deltas are REPORTED, not
  gating.
- **SECONDARY (viability guard on the candidate)**: no catastrophic forgetting —
  errtop21 full-train fraction drop vs the same-cell all42 control < 5 pp at EVERY
  cell AND pooled. rand21 drops reported alongside (diagnostic).
- Exploratory, non-gating: train_replay_tasks, margin min/median, error-count
  distribution, overlap between the two selections, all42-vs-schedule_compare
  replication cross-check (N512 cells share massed/ridge/all_transition/scale42 config
  with `schedule_compare/massed_N512_s*`; integer fields must replicate exactly).

## Discipline

One shot: all 27 runs execute once, serialized under `benchmarks/locks/cpu-bench.lock`;
no rerun, repair, or retune regardless of outcome. Metrics files refuse overwrite
(fail-closed). Outputs left uncommitted. Faithfulness check pinned NOW: for cell
(N=512, s=7) the driver's mirrored pass-1 record must equal a direct
`ladder_experiments.run_experiment(...)` return value modulo `wall_s`; the batch aborts
if not.
