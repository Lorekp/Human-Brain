# EI-balance sweep (first L2 architecture experiment): inhibitory sign structure moves state separation enormously but only by breaking the operating regime — margins fall monotonically and transfer never leaves the floor

Experiment ID: `ei_sweep` · run date 2026-08-23 · device `HUMANBRAIN_DEVICE=cpu`
· 12 runs at worktree HEAD `c7c08ed` plus this experiment's `--inhibitory-fraction`
knob (committed immediately after collection) · outputs in `experiments/ei_sweep/`
· preregistration frozen before execution: `PREREGISTRATION.md` · analysis:
`_analyze.py` → `_analysis_output.txt` · mechanism probe: `_mechanism_probe.py`
→ `_probe_results.jsonl`, `_probe_output.txt`

## Question

V3a (`docs/research/a013-cell-construction-variants.md`, as corrected by review
522551d): `RecurrentCircuitConfig.inhibitory_fraction` (default 0.2,
recurrent.py:114) flips recurrent-weight signs on a random subset of active
cells — wired since before the seal but NEVER swept. Does E/I balance move
state separation, decision margins, or calibration transfer? The sealed a012
failure included null state separation (~0.001); sign-structure heterogeneity
was the cheapest candidate to move it.

## Design and execution integrity

- **Arms (preregistered)**: fraction {0.0, 0.1, 0.2, 0.35} × N=1024 × seeds
  {7,8,9}, `--codebook explicit` (isolate the E/I axis), protocol pinned to
  the codebook_compare confirmatory settings (ridge / massed / data-scale 42 /
  all_transition) so rows are comparable across experiments. 12 runs.
- **Determinism**: every arm ran twice in-process (`--check-determinism`);
  byte-identical records minus `wall_s`, exit 0 ×12 (`_run_log.txt`).
- **Red tests** (`tests/test_ladder_inhibitory_fraction.py`, 14 tests, green):
  knob determinism incl. default-is-status-quo anchoring; fraction bounds
  rejected at CLI and `run_experiment`; sign-hash stability (same seed+fraction
  → same canonical int8 sign-vector sha256; distinct across fractions/seeds);
  counts equal `round(f·N_active)`; group magnitudes equal
  `recurrent_current_pa`; topology invariance (sign draws consume the circuit
  RNG after targets, so afferent/recurrent wiring is bit-identical across
  fractions sharing a seed).
- **Fairness**: per seed, `curriculum_sha256` is identical across all four
  fractions — byte-identical task cohorts everywhere.
- **External anchor**: the f=0.200 arm reproduces the codebook_compare
  explicit_N1024 rows exactly — calib 8/10/7 per seed, train
  123/126/122, margin_median 223.22 ± 3.26 vs their 223.22 ± 4.00 — confirming
  both the knob-at-default no-op and that the concurrent `--device` harness
  work did not perturb the CPU path.
- Shared-box caveat: wall_s informational only (a sibling agent's GPU work
  overlapped); all verdict metrics are deterministic.

## Results

| fraction | calib_cases(/54) | train_cases(/126) | train_replay(/42) | margin_median_pa | margin_min_pa |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 7.00 ± 1.41 | 121.0 ± 2.8 | 40.3 ± 0.9 | **319.06 ± 10.14** | −35 ± 28 |
| 0.100 | 6.00 ± 1.41 | 116.3 ± 12.3 | 38.3 ± 4.5 | 279.78 ± 9.42 | −124 ± 173 |
| 0.200 | 8.33 ± 1.25 | 123.7 ± 1.7 | 41.0 ± 0.8 | 223.22 ± 3.26 | −38 ± 42 |
| 0.350 | 6.00 ± 0.82 | 65.0 ± 5.7 | 19.0 ± 2.4 | 193.36 ± 25.42 | −520 ± 231 |

Per-seed raw (calib/54 · margin_median_pa · train/126):

- f=0.000: 8·304.9·123 / 5·324.2·117 / 8·328.1·123
- f=0.100: 8·268.2·124 / 5·279.9·99 / 5·291.3·126
- f=0.200: 8·225.2·123 / 10·225.8·126 / 7·218.6·122
- f=0.350: 5·162.7·58 / 6·192.4·72 / 7·225.0·65

Paired per-seed wins vs f=0.0 (G1 needs ≥2/3 on BOTH metrics):
f=0.1 calib 0/3, margin 0/3 · f=0.2 calib 1/3, margin 0/3 · f=0.35 calib
1/3, margin 0/3.

## Mechanism probe (beyond-schema diagnostics; non-gating)

Sign-pattern hashes are stable under duplicate regrow and distinct across
every (fraction, seed); inhibitory counts hit round(f·1024) exactly
(102/205/358); mean|w| = 650 pA in BOTH groups (mechanism engaged; the knob
changes signs only, never magnitudes).

State separation (pooled median 1−cosine between different-first-teacher-task
states after framing observation; sealed-diagnostic analog on two channels):

| fraction | eligibility channel | synaptic-current channel |
| --- | --- | --- |
| 0.000 | 0.00056–0.00057 | 0.0057–0.0060 |
| 0.100 | 0.00068 (all seeds) | 0.0101–0.0103 |
| 0.200 | 0.00104–0.00118 | 0.0212–0.0261 |
| 0.350 | **0.029–0.175** | **0.46–0.71** |

Margin skew (Fisher–Pearson g1) is small and unsystematic below 0.35
(−0.40–0.77, one 3.03 outlier at f=0.2 s8), larger-spread at 0.35
(0.37–1.21) consistent with heavier negative margin tails
(margin_min −520 ± 231 pA).

## VERDICTS

- **P-EI: FALSIFIED (G1 failed; G2 moot).** No intermediate fraction beats
  f=0.0 on calibration cases AND margin_median — margin_median is MONOTONE
  DECREASING in inhibitory fraction (319 → 280 → 223 → 193 pA), so the
  status-quo 20% already sits past any would-be optimum and there is no
  inverted U below it. Calibration transfer never leaves the constant-program
  floor band (5–10/54) in ANY arm; f=0.2's nominal 8.33 ± 1.25 vs 7.00 ± 1.41
  at f=0.0 is seed noise (paired wins 1/3). The degradation half of the
  prediction is right at 0.35 (fit collapses 121 → 65/126; margins erode),
  but the claimed intermediate sweet spot does not exist.
- **State-separation endpoint: the mechanism is REAL but mispriced.**
  Inhibitory sign structure moves the sealed ~0.001 null by up to two orders
  of magnitude — but only at f=0.35 (eligibility 0.03–0.18; synaptic-current
  0.46–0.71), exactly where fit capacity collapses and margins shrink.
  At operationally viable fractions the move is modest (≈1.2–2.1× at 0.2).
  Separation, capacity, and margin trade off along ONE axis here: you can
  buy state diversity only by spending the homogeneous regime that ridge
  readout exploits.
- **Transfer implication (reportable): raising state separation is not
  sufficient — and at these knobs not even compatible — with transfer.**
  f=0.35 separates states ~100× better than the sealed null yet fits worse
  and transfers no better. Together with the schedule_compare and
  codebook_compare nulls, this is a third independent input/dynamics-side
  manipulation that fails to move calibration: transfer remains blocked at
  the constant-collapse/binding level (feature QUALITY), not at the
  state-diversity or encoding-distribution level.

## Caveats

- Sign-flipped magnitude into homogeneous excitatory-type GLIF5 cells is not
  true inhibitory cell classes (no inhibitory human fit exists in data/allen);
  conclusions speak to recurrent sign structure only.
- Changing the fraction shifts the downstream token-cell heterogeneity draw
  (RNG stream consumption depends on the choice size; σ=0.01 pA on 277 token
  cells). Wiring itself is provably invariant (red test).
- Single size N=1024, three seeds, ridge readout; sparse/local-rule readouts
  not swept here.

## Files

- `PREREGISTRATION.md` — frozen before execution (arms, P-EI verbatim, gates)
- `f*_N1024_s*.json` — 12 fixed-schema v1 metrics records
- `_run_matrix.sh`, `_run_log.txt` — serialized launcher + full run log
  (`--check-determinism` exit 0 ×12)
- `_analyze.py`, `_analysis_output.txt` — aggregates + preregistered gate math
- `_probe_results.jsonl` — sign metrics, margin skew, two-channel state
  separation per run
- Harness/tests (committed): `benchmarks/ladder_experiments.py` knob +
  `recurrent_sign_metrics`, `experiments/SCHEMA.md` knob section,
  `tests/test_ladder_inhibitory_fraction.py`
