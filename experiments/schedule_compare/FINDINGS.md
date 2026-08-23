# A013 spacing-vs-massing schedule experiment: FINDINGS

Experiment ID: `schedule_compare` · run date 2026-08-23 · device `HUMANBRAIN_DEVICE=cpu`
Spec: `docs/research/a013-schedule-experiment-spec.md` (preregistered) · harness
`benchmarks/ladder_experiments.py` @ `e075b93` (untouched since; `src/` byte-stable
`e075b93..e478aff`, verified by empty diff). Runs at git head `e478aff`. All 18 runs
executed serially in one batch (11:23:02–11:25:47 local), total wall 162.1 s, every run
exit 0 (`_run_log.txt`). Outputs left uncommitted.

**Concurrent-worktree note:** a separate agent modified `benchmarks/ladder_experiments.py`
(+187 lines, first write 11:34:27 local) *after* this batch completed — all 18 metric
files pre-date the modification, so the entire cohort ran on the unmodified `e075b93`
harness whose behavior is documented here and in `experiments/SCHEMA.md`. The working
tree also gained unrelated untracked files (`_probe_timing.py`,
`tests/test_ladder_codebook.py`) from that concurrent work; none were touched by this
experiment.

## Fixed configuration

18 arms = schedule {massed, spaced, interleaved} × N ∈ {128, 512} × seed ∈ {7, 8, 9};
data_scale=42 TRAIN tasks (+ fixed 18-task calibration cohort); supervision=
`all_transition`; readout=`ridge`; 3 corpus passes; A2 framing.

**Readout deviation from the arms table, per the spec's own contingency clause:** the
spec pinned `sparse_three_factor` on the assumption it would win the pilot. The pilot
(commit `410c720`, `experiments/readout_compare/FINDINGS.md`) rejected both local rules
at shipped constants (degenerate constant-program collapse), so this execution uses
**ridge**, as the spec directs ("if pilot rejects local rules, rerun spec with ridge and
note") — chosen to keep the schedule axis clean of readout confounds.

## Results — train_cases_passed /126 and calib_cases_passed /54 per arm

| schedule | N | seed | train cases passed | train replay /42 | calib cases passed | margin median pA |
| --- | --- | --- | --- | --- | --- | --- |
| massed | 128 | 7 | 15 (0.1190) | 1 | 7 (0.1296) | −12.6 |
| massed | 128 | 8 | 20 (0.1587) | 3 | 10 (0.1852) | +4.7 |
| massed | 128 | 9 | 20 (0.1587) | 4 | 7 (0.1296) | +17.4 |
| massed | 512 | 7 | 106 (0.8413) | 34 | 5 (0.0926) | +95.7 |
| massed | 512 | 8 | 55 (0.4365) | 16 | 7 (0.1296) | +105.7 |
| massed | 512 | 9 | 52 (0.4127) | 16 | 8 (0.1481) | +100.7 |
| spaced | 128 | 7–9 | **identical to massed** | = | **=** | = |
| spaced | 512 | 7–9 | **identical to massed** | = | **=** | = |
| interleaved | 128 | 7–9 | **identical to massed** | = | **=** | = |
| interleaved | 512 | 7–9 | **identical to massed** | = | **=** | = |

Headline: **presentation order has exactly zero effect on every discrete outcome.**
All 18 runs share per-(N, seed) metrics bit-for-bit on every count field (train cases,
replay, calibration cases, margin counts); the only cross-schedule differences in any
file are the `schedule` key itself, its echo in `config_dump`, `wall_s`, and a **1-ulp
float perturbation of margins (~1e-9 pA)** in one comparison (N=512 s=7:
`margin_min_pa` −30.627828737 vs −30.627828738; one median digit differs vs
interleaved). Every other cross-schedule pair is byte-identical on all metric fields.
Pooled fractions are identical across arms: train 268/756 = 0.3545 each; calib
44/324 = 0.1358 each.

## Preregistered-gate verdicts

Wilson score intervals per proportion (z = 1.96) combined into Newcombe hybrid-score
intervals for the difference of pooled independent binomials (implementation:
`_analyze_gates.py` alongside this file).

**1. Primary — spaced > massed on train_cases_passed fraction: FAIL.**
Delta is exactly +0.0000 at N=128 ([−0.0505, +0.0505]), at N=512
([−0.0704, +0.0704]), and pooled ([−0.0481, +0.0481]); all CIs contain zero. Not even
sign-consistent — all six per-(N, seed) paired deltas are identically zero. There is no
effect to detect: the preregistered "directional but modest" prediction is falsified in
the strongest possible sense.

**2. Secondary — interleaved > blocked (= {massed, spaced} pooled; the arms table
defines no separate blocked arm) on calibration cases: FAIL.**
Delta exactly +0.0000 at N=128 ([−0.0632, +0.0717]), N=512 ([−0.0583, +0.0673]), pooled
([−0.0437, +0.0481]); all twelve per-(N, seed) paired deltas identically zero.

**3. Exploratory (non-gating) — margin distribution shift: none.**
Margin medians are identical across schedules at every (N, seed) (table above).
p10 is **not computable**: schema v1 persists only min/median/count of margins, not the
distribution; recorded as unavailable, not imputed.

## Why nothing could move (mechanism, read off `src/humanbrain/recurrent.py`)

Under the ridge path, presentation order is structurally inert by construction:

- `reset_sequence()` zeroes the eligibility trace AND resets reservoir state between
  presentations (`recurrent.py:645`) — no state crosses presentation boundaries;
- `strengthen` accumulates least-squares sufficient statistics — per-target feature
  sums (`_target_moments += feature`, line 741) and batched Gram matrices
  (`_normal_matrix += featuresᵀ @ features`, line 771, batch 128);
- `freeze_learning` solves `(Σxxᵀ + λI) W = Σxyᵀ` in closed form (line 789).

The fit depends only on the multiset of per-presentation feature/target pairs, which is
identical across schedules (same sequences, different order). The sole channel from
order to outcome is float summation grouping at 128-example batch boundaries — the
observed ~1e-9 pA margin noise, nine orders of magnitude below the tens-of-pA margins
that decide races, hence zero discrete flips anywhere. The spec's trace-math anchor
(ρ²⁰⁷ ≈ 2.3e-06 under massing) presupposed traces persisting across presentations; in
this harness they never do, so the inter-presentation gap distribution the schedules
manipulate is invisible to the readout by arithmetic necessity.

Honest scope of the negative: it certifies T3 dead **for this architecture under
closed-form readouts** — scheduling cannot matter here at any N or seed. The local
rules remain order-sensitive in principle (online competitive updates) but are
unavailable as instruments until their shipped-constant collapse is fixed (pilot,
`410c720`). A schedule effect could only re-enter via mechanisms that let state or
statistics persist across presentations (no-reset variants, online-only readouts).

## REPORT-GAP: frozen permutations were not loadable (recorded, not patched)

The task instructed using `perm_appendix.json` orders for interleaved if the harness
supports loading them, otherwise documenting the gap precisely:

1. **No loading path exists.** `_presentation_order()` (`ladder_experiments.py:227`)
   generates interleaved orders at runtime: `np.random.default_rng([seed, 0x1A77E5])`,
   three concatenated `rng.permutation(task_count)` draws. Grep over all `*.py` in the
   repo finds zero references to `perm_appendix.json`; nothing refuses to shuffle when
   the appendix provides an order — red test #3 from the spec ("frozen-permutation
   load") is unimplemented. Per instruction, I did not patch around it; the 6
   interleaved arms used harness-native runtime orders.
2. **Appendix orders diverge from harness orders.** The appendix lists are valid
   multisets (each of tasks 0–41 exactly 3×, length 126), but differ from what the
   harness generates for the same seed label at 123–124 of 126 positions
   (positional agreement 2/126, 3/126, 3/126 for seeds 7/8/9) — a different RNG
   construction entirely; the appendix's generator is not recorded in-repo.
3. **Mootness for this experiment's verdicts.** Given the demonstrated structural
   order-invariance above, appendix-vs-harness order choice cannot move any reported
   metric beyond 1e-9 pA under ridge. The gap is recorded for protocol hygiene and
   because it becomes load-bearing again for any future order-sensitive (local-rule)
   rerun, where byte-equal frozen orders would be required.
4. **Secondary definitional note.** The harness's `spaced` is forward round-robin
   passes `[0..41]×3` (same-task gap = 42 slots); the spec's worked example reverses
   later passes (`[0..5, 5..0]`). Same-task gap is T slots in both, so the manipulated
   quantity matches; `SCHEMA.md` documents the harness behavior as canonical.
   Recorded, not patched.

## Integrity checks

- `curriculum_sha256`: exactly 3 distinct values, one per seed, shared across all six
  arms per seed — the design is paired as intended (arms differ only in growth-time
  presentation order).
- Cross-process replication: today's six massed arms are the same effective
  configuration as the pilot's `readout_compare/ridge_N{128,512}_s{7,8,9}` (massed +
  all_transition + scale 42 + ridge). All integer metrics replicate exactly; float
  fields agree to ≤1e-9 (last-digit differences in margin min/median at N=512 only),
  consistent with the harness determinism contract modulo last-ulp float noise.
- `config_dump` differs across schedules only in the `schedule` key.

## Falsification note (per spec)

At both N, all three schedules are not merely statistically indistinguishable but
arithmetically identical in every discrete outcome. Per the spec: **the teaching
matrix's T3 (spacing/massing as a lever) is marked dead in this harness** — scheduling
is second-order to representation under closed-form readouts, exactly the branch the
spec pre-committed ("if no schedule moves anything… recorded as a clean negative either
way"). Representation work absorbs the priority.
