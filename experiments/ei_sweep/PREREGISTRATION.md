# PREREGISTRATION — ei_sweep (V3a inhibitory-fraction sweep)

Written and frozen BEFORE any matrix execution. Source tasking:
`docs/research/a013-cell-construction-variants.md` V3a as corrected by review
522551d — `RecurrentCircuitConfig.inhibitory_fraction` already exists
(default 0.2, recurrent.py:114; signs built :261-266, applied :268) but was
never swept. This is the first L2-level architecture experiment.

## Question

Does E/I balance (recurrent sign structure) move state separation,
decision margins, or calibration transfer? The sealed a012 failure included
null state separation (~0.001 1−cosine between different-task states);
heterogeneity of sign structure is the cheapest candidate to move it.

## Arms (frozen)

- fraction ∈ {0.0, 0.1, 0.2, 0.35} × N = 1024 × seeds {7, 8, 9}
- `--codebook explicit` everywhere (isolate the E/I axis from representation)
- Protocol fixed to the codebook_compare confirmatory settings for
  cross-experiment comparability: `--readout-rule ridge --schedule massed
  --data-scale 42 --supervision all_transition`
- 12 runs, every run with `--check-determinism` (byte-identical records minus
  wall_s enforced in-process), serialized, CPU only.
- fraction 0.2 is the shipped-default anchor; explicit_N1024 rows of
  `experiments/codebook_compare/FINDINGS.md` (git_head 081a492) are the
  external reference point for it.

## Preregistered prediction P-EI (verbatim from tasking)

Some intermediate fraction (0.1–0.2) yields higher calib_cases_passed AND
higher margin_median than 0.0; monotone degradation toward 0.35 (too much
inhibition suppresses races).

## Operational gates (fixed before execution)

- **G1 (primary)**: there exists f* ∈ {0.1, 0.2} such that, per-seed paired
  against f = 0.0, BOTH calib_cases_passed(f*) > calib_cases_passed(0.0) in
  ≥ 2/3 seeds AND margin_median_pa(f*) > margin_median_pa(0.0) in ≥ 2/3 seeds.
- **G2 (degradation tail)**: at f = 0.35, margin_median_pa is strictly below
  the margin_median_pa at f* AND calib_cases_passed(0.35) ≤
  calib_cases_passed(f*).
- **Verdict**: P-EI CONFIRMED iff G1 ∧ G2; PARTIAL iff G1 only;
  FALSIFIED iff ¬G1. No re-running, no arm additions, no post-hoc gate edits.

## Diagnostic endpoints (non-gating, recorded per run)

- Recurrent sign-pattern hash (sha256 over the canonical int8 sign vector,
  column 0 of `_recurrent_weights_pa`): must be identical across duplicate
  determinism reruns, distinct across fractions and across seeds.
- Mean |recurrent weight| by sign group (mechanism-engagement check: both
  groups equal `recurrent_current_pa`; counts equal round(f · N_active)).
- Margin distribution skew (Fisher–Pearson g1 over TRAIN decision margins).
- State separation, sealed-diagnostic analog: after growth + freeze, replay
  each TRAIN task's framing observations and snapshot (a) eligibility trace
  and (b) synaptic current vector at the first-decision state; pairwise
  1−cosine over task pairs with different first teacher token; medians per
  stratum (Int / List[Int]) and pooled. The sealed null is ~0.001.

## Honesty notes

- Sign draws consume the circuit RNG AFTER afferent/recurrent targets are
  drawn, so changing ONLY the fraction leaves `_afferent_targets` and
  `_recurrent_targets` bit-identical across arms sharing a seed (pinned by
  red test). The downstream output-heterogeneity draw DOES shift with
  fraction because `rng.choice(k)` stream consumption depends on k; its
  magnitude is σ = 0.01 pA on the 277 token cells — documented, accepted.
- "Inhibition" here is sign-flipped recurrent current magnitude into
  homogeneous excitatory-type GLIF5 cells (no inhibitory cell model exists in
  data/allen); results speak to sign structure only, not true E/I cell classes.
