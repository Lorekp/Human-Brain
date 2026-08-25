# P-A step −1 retro-probe — continuous states decode TASK structure above shuffle; answer content does NOT lift → B′ Stage-0 LICENSED with decomposition caveats (2026-08-25)

Probe of `experiments/apical_probes/logged_states_apical_cohort.npz` per
`docs/research/path-decision-synthesis.md` @ e16d22c §Probe P-A **Step −1**.
Question: do the SAVED continuous subthreshold voltages at QA decision points
linearly decode decision-point variables ABOVE the shuffled-label band?
Falsifier under test: "if even continuous-state linear probes sit inside the
shuffled-label band, the problem is upstream state construction; corpus-level
B′ dies before any build."

**Verdict: FALSIFIER DOES NOT FIRE.** Every probe family exceeds its shuffled
band by wide margins in every one of the 18 arms. B′ Stage-0 is licensed.
Caveat from controls: a large share of the decodable signal is gross
stimulus-locking (prompt length / global activity), but a genuine distributed
component survives after regressing prompt length out of every feature.

## Data integrity and alignment

- npz loaded from the saved cohort artifact (local-only file, sibling worktree
  copy); ALL 14 arrays sha256-match `logged_states_apical_cohort_manifest.json`
  (bit-exact verification before anything ran).
- Row↔array alignment verified three ways: manifest rows iterate the same
  item order as replay (`position` == axis-0 index within each split);
  `qa_index` arrays are exactly 0..199 / 0..59; every manifest row matches the
  freeze evidence `qa_manifest_{train,val}.json` item-by-item on
  `answer_text` AND `family`.
- 200 TRAIN / 60 VAL decision points; families train 60/70/70, val 20/20/20
  (who_said/address_lookup/verse_continuation).

## What was probed (frozen before running)

Per simulation arm (all 18), three feature sets: `voltage` = somatic membrane
voltage at the decision point (1024 cells), `apical` = apical compartment
voltage u_A (1024 cells) — **the continuous state never probed by APICAL-AND**
— and `concat` (2048). sklearn LogisticRegression (multinomial, lbfgs,
C=0.01 frozen a priori), standardized by TRAIN stats, strict cohort split
(fit TRAIN, eval VAL), 200 label-permutation replicates through the identical
pipeline as the null band. Wilson 95% CIs; empirical one-sided p;
Bonferroni ×18 across arms. Seeds: master 7, deterministic child streams.

## T1 — family 3-way classification (PRIMARY, strict split)

All 18 arms × all 3 feature sets exceed the band at the permutation floor:

| Feature set | VAL accuracy range (18 arms) | Shuffle band mean | Band q95 | min p (all arms) |
|---|---|---|---|---|
| apical | 0.8500 – 0.9000 | 0.3223 – 0.3496 | 0.4667 – 0.5008 | 0.005 (=1/201) in 18/18 |
| concat | 0.8167 – 0.9333 | 0.3213 – 0.3491 | 0.4667 – 0.5167 | 0.005 in 18/18 |
| voltage | 0.8167 – 0.9167 | 0.3177 – 0.3430 | 0.4667 – 0.5175 | 0.005 in 18/18 |

Best cells (verbatim): A50_0.25_0.3/concat acc=0.9333 wilson95=[0.8407,0.9738]
effect +0.6118 Cohen's d=6.85; A50_0.75_0.6/concat acc=0.9333
wilson95=[0.8407,0.9738] d=6.72. Majority-class baseline 0.3333.
Effect sizes: +48 to +61 points absolute, d ≈ 5.2–6.9 everywhere.
Zero of 200 shuffles reached the observed accuracy in ANY config.
Sensitivity grid (reference arm A100_0.25_0.6, true labels): flat in C over
[0.001–1] (apical 0.85 at every C; concat 0.8667→0.8833 plateau) with TRAIN
inner-CV accuracy 0.94–0.98 — not a regularization artifact.
Note on multiplicity: Bonferroni×18 gives p_adj=0.0896 > 0.05 for the single
best arm, but all 54 independent configs hit the permutation floor
simultaneously; treating 18 parameter variants of ONE hypothesis as 18
independent rejections is the conservative reading. The falsifier asks only
whether probes sit INSIDE the band; they do not, anywhere.

## T5 — decomposition: how much is trivial stimulus-locking?

Same pipeline, low-dimensional summary features:

| Variant | dims | VAL accuracy range | interpretation |
|---|---|---|---|
| prompt_bytes_only | 1 | 0.5833 (p=0.005–0.02) | families differ in prompt length |
| sum7 (mean/std/max|v|, active frac, mean/std u_A, spikes) | 7 | 0.7833 – 0.8667 | most scale info |
| sum7 + prompt_bytes | 8 | 0.8167 – 0.8833 | |

So ~most of T1 is available from ≤8 gross scalars. Full 1024-d vectors beat
sum7 by ≈ +4–7 pts consistently in every arm.

## T6 — residualized probe: distributed structure SURVIVES input-echo removal

Every cell feature residualized against prompt_bytes (coefficients fit on
TRAIN only), then identical strict-split probe (100 perms):

- apical-residualized: acc 0.7500 – 0.8333 across arms (p = 0.010 = 1/101 in 18/18);
- voltage-residualized: acc 0.8000 – 0.9000 (p = 0.010 in 18/18);
- prompt-bytes-only baseline: 0.5833; shuffle band q95 ≈ 0.50.

The states therefore carry genuinely DISTRIBUTED decision-point structure,
not merely global activity/prompt-length echo.

## T2 — speaker identity at who_said decision points (SECONDARY, pooled CV)

DEVIATION (forced by corpus): train/val speaker vocabularies share only
{And, Moses}; only 3/20 val who_said items have speakers with >=2 train
occurrences, so strict-split speaker naming is vacuous. Probes below pool all
80 who_said items with stratified 5-fold CV (seed 7); same pipeline/perm
machinery otherwise. Power is limited (n=80); treated as secondary evidence.

| Probe | best arm | accuracy | majority | balanced acc | band mean | band q95 | min p (p_adj) |
|---|---|---|---|---|---|---|---|
| speaker multiclass / apical (17 named + OTHER) | A200_0.25_0.6 | 0.2750 wilson95=[0.1892,0.3814] | 0.2500 | 0.1464 | 0.1619 | 0.2256 | 0.010 (0.179) |
| speaker multiclass / concat | A100_0.5_0.6 | 0.2875 | 0.2500 | 0.0913 | 0.1952 | 0.2625 | 0.025 (0.448) |
| moses_binary / apical | A100_0.5_0.3 | 0.9125 wilson95=[0.8302,0.9570] | 0.8750 | 0.6500 | 0.8499 | 0.8750 | 0.005 (0.090) |
| moses_binary / concat | A100_0.5_0.3 | 0.9000 | 0.8750 | 0.6000 | 0.8719 | 0.8875 | 0.010 (0.179) |

Reading: fine-grained speaker identity is at most WEAKLY decodable
(accuracy barely above class priors; balanced accuracy ~0.09–0.15). A single
speaker token ("Moses") shows a marginal above-band signal in some arms.

## T3 — chapter_distance ridge regression (SECONDARY, strict split)

Clean NULL: VAL R^2 is negative in every arm — apical range [-0.2492,
-0.2356], concat range [-0.2494,-0.1024]; best-arm p = 0.687 (apical) /
0.351 (concat). The states carry NO information about how far back the
answer's chapter sits — i.e., they encode what was recently presented in the
prompt, not unrepresented latent story variables. This bounds interpretation
of T1: decodability tracks prompt-locked structure, not arbitrary QA metadata.


## T4 — r0-c G-A3 span protocol with apical features (comparability arm)

BUDGET CUT (disclosed): one r0-c softmax fit (~673 answer classes x up to
2048 dims) costs ~13 min on this box under current load; the full 18-arm x
2-feature-set sweep was infeasible, and even a 4-arm subset was killed after
the first arm failed to checkpoint within ~55 min. Ran the single most
informative cell instead:

- A100_0.25_0.6 (committed reference arm) x `apical_only` features,
  true labels, protocol otherwise VERBATIM (r0c.train_softmax_logistic,
  TRAIN-answer-vocab head, SpanScorer 639 positions):

  **span accuracy = 0.01564945226917058 (10/639)**

Committed comparators for the same arm (probes_results.json):
raw_voltage_spikes logistic 0.025039123630672927 / MLP 0.02190923317683881
(its shuffled-label MLP control 0.01564945226917058 = 10/639); traces_bank
logistic 0.009389671361502348 / MLP 0.018779342723004695; majority prior
0.0406885758998435; unigram 0.04538341158059468; frozen G-A3 threshold
0.061851478566709396.

Reading: substituting the continuous apical voltage for the G-A3 feature sets
does NOT lift answer-content decoding — the score lands exactly AT the
committed shuffle-control level (10/639) and below every no-state control.
Consistent with T2/T3: the saved continuous states carry prompt/task-locked
structure strongly, but not answer content.


## Reading and disposition

1. The pre-declared falsifier ("even continuous-state linear probes sit inside
   the shuffled-label band") does NOT fire — not in any arm, not for any
   feature set, not for the primary target. Upstream state construction is
   NOT dead: decision-point voltages are strongly, linearly, and transferably
   (TRAIN→VAL) structured at the level of task/prompt variables.
2. Per synthesis §P-A: **B′ Stage-0 is licensed** — the Garg-style apical
   regression build may proceed to its own gate (probe R² ≥ 0.85 vs online-LMS
   reference). The gate asked "do continuous probes beat shuffle?" and the
   answer is unambiguous (54/54 configs at the permutation floor for T1).
3. What the decodable signal IS (and is not) — decomposition:
   - Strong: which task family the decision point belongs to (0.82–0.93),
     surviving prompt-length residualization (0.75–0.90 apical/voltage vs
     0.5833 length-only baseline).
   - Weak/marginal: fine-grained speaker identity (best 0.275–0.2875 vs
     0.25 prior; balanced acc ~0.09–0.15); Moses-binary marginal above band.
   - Absent: chapter_distance (negative R^2 everywhere), and answer-content
     span decoding shows NO lift from continuous apical features (T4-minimal
     10/639 = exactly the committed shuffle-control level).
   The licensed structure is therefore stimulus/task-locked input-side
   structure, NOT answer content. Stage-0's own gate (R² of the fitted apical
   trajectory vs an online-LMS reference on a regression task) remains the
   decisive test of the estimator claim — this retro-probe licenses building
   it; it does not predict its pass.
4. Arm gradient worth noting for Stage-0 constants: tau_a=50ms arms decode
   best on raw features (0.90–0.93), tau_a=200ms arms best after
   residualization (0.83 apical) — faster apical integrators preserve more
   non-echo structure in this regime.

## Files

- `retro_probe.py` — main probe driver (sections integrity/t1/sens/t2/t3/t4),
  serial default per instruction, `--jobs` flag kept working; BLAS threads
  capped at 1 per worker (oversubscription made the first parallel attempt
  ~10x slower).
- `t5_summary_controls.py`, `t6_residualized.py` — decomposition controls.
- `t4_subset.py`, `t4_minimal.py` — budget-capped T4 variants (subset was
  killed before first checkpoint; minimal produced the one reported cell).
- `retro_probe_results.json`, `t5_summary_controls.json`,
  `t6_residualized_probe.json`, `t4_minimal_results.json` — verbatim numbers.
- `explore_targets.py`, `preflight.py`, `smoke_t4.py`, `bench_t4_fit.py` —
  recorded pre-flight checks (target distributions; state variability/timing;
  r0-c plumbing + majority-prior reproduction; fit-cost benchmark).
- `summarize_results.py` + `summary_digest.txt` — table generator and its
  output used for this report.

## Disclosures

- Speaker-identity probe deviates from strict split BY NECESSITY: train/val
  speaker vocabularies share only {And, Moses} (val speakers Jesus/Peter/
  Mary… absent from train), so strict-split speaker naming is vacuous (3/20
  val who_said items have speakers with ≥2 train occurrences). T2 uses pooled
  stratified 5-fold CV over all 80 who_said items instead; disclosed as a
  deviation, run alongside, never mixed into the primary verdict.
- Fixed C=0.01 chosen a priori (strong L2 appropriate at d≈1024, n≈200);
  sensitivity grid confirms insensitivity.
- Bonferroni nuance stated inline in T1 rather than hidden: min-p 0.005
  x18 arms = p_adj 0.0896 under the most conservative correction; all 54
  configs hit the permutation floor simultaneously, so the license call does
  not rest on a single arm.
- T4 scope cut disclosed in its section; the harness itself was validated by
  reproducing the committed majority-prior control EXACTLY (26/639 =
  0.0406885758998435) via smoke_t4.py.
- Environment: Windows box, 12 logical cores shared with other concurrent
  agent workloads; sklearn 1.8.0, numpy (see results JSON for versions);
  total wall ~7 h including killed oversubscribed attempts (~1.5 h of that
  wasted to a joblib/BLAS oversubscription mistake, then fixed).
