# GLIF3-as-GLIF5 exact expressibility, and what the V1 mixture costs

Status: preregistered ANALYSIS — verdict reached, empirically verified against
the pinned specimens and the shipped native kernel; red tests committed at
`tests/test_glif3_as_glif5_reduction.py`. No simulation-arm conclusions here.
Provenance: researched + computed by subagent (verification harness retained at
`artifacts/scratch/glif3_reduction_check.py` and
`artifacts/scratch/reduction_aux_checks.py`; raw outputs in
`artifacts/scratch/reduction_results.txt` and
`artifacts/scratch/reduction_aux_results.txt`). Verify pass required before
merge.

**Verdict: REDUCIBLE-WITH-HARMLESS-OVERHEAD.** GLIF5's kernel reduces exactly
(bit-level) to GLIF3's `'inf'` mode for threshold dynamics, voltage reset, and
refractory handling. The after-spike-current path reduces exactly in real
arithmetic via an amplitude time-shift (`Ã = δ⊙A`) but not bit-exactly in
IEEE-754: the post-spike ASC commit rounds differently, producing ≤ 6 ulp
(≤ 1.14e-13 pA) ASC deviations and ≤ 4 ulp (≤ 7.1e-15 mV) voltage deviations,
with spike trains verified identical across every stimulus tried. V1 therefore
needs **no dual-model dispatch**: one kernel, two configs. A hypothetical
native GLIF3 path would buy nothing measurable at experiment scale.

## 1. The two update systems, side by side

Both implementations operate on leak-relative voltage, two ASC channels
(`after_channel_count == 2` on both fits; valid_common pins it, and the AVX2
recurrent lane hard-requires it — `_native_src/src/lib.rs:226,388`).

Constants: `k = dt/C` (`voltage_step_scale`, cells.py:265),
`g = 1000/R_input` nS (`leak_ns` property), per-channel decays
`δ_i = exp(-dt_ms/τ_i)` and cut decays `δcut_i = exp(-spike_cut_ms/τ_i)`
(cells.py:249-253, 438-439), amplitudes `A_i`.

### 1a. GLIF3 `'inf'` mode — `simulate_glif3_current` (cells.py:443-461)

Per step, non-refractory cell (`refractory > 0`: freeze v and a, decrement,
skip — lines 444-447):

```
a'      = δ ⊙ a                                   # decay FIRST   (line 448)
v_{t+1} = v + k·(I_t + Σ a'_i − g·v)              # integrate     (449-453)
if v_{t+1} > Θ_rel:                               # strict >      (454)
    spike at (t+1)·dt                             #               (455)
    v ← 0.0                                       # zero reset    (457)
    a ← A + δcut ⊙ a'                             # cut reset     (458)
    refractory ← spike_cut_steps (=77)            #               (459)
else:
    a ← a'
```

Threshold is a fitted constant (`threshold_relative_mv`, cells.py:68-69);
there is no threshold state. Reset rule `voltage_reset_method: "zero"`.

### 1b. GLIF5 `'three_components_exact'` — native kernel `glif_step_collect`
(lib.rs:66-123; the NumPy equivalence oracle
`tests/test_simulation_equivalence.py:18-81` implements the same order)

Per step, non-refractory cell (`refractory != 0`: freeze everything except the
counter, decrement — lib.rs:67-73):

```
Σã      = ((0.0 + ã_0) + ã_1)                     # PRE-decay sum (76-79)
total   = I_t + Σã                                #               (80)
v_{t+1} = ṽ + k·(total − g·ṽ)                    # integrate     (82-84)
u_{t+1} = E_dec·u + vfac·ṽ + bfac·(total·βscale)  # th-voltage    (86-90)
s_{t+1} = s·spike_threshold_decay                 # th-spike      (91)
ã       ← δ ⊙ ã                                   # decay AFTER   (96-98)
if v_{t+1} > Θ_inf + u_{t+1} + s_{t+1}:           # strict >      (100-102)
    ṽ ← scale·v_{t+1} + offset                    # v_before      (114)
    ã ← Ã + δcut ⊙ ã                              #               (115-119)
    s ← s_{t+1}·spike_cut_threshold_decay + Δ     #               (120-121)
    refractory ← spike_cut_steps                  #               (122)
```

`GlifConfig` fields flow verbatim from `AllenGlif5Parameters` /
`FittedGlif5Constants` (`native.py:_glif_config`, 1095-1114), so degenerate
values chosen below reach the kernel untransformed.

The only structural difference between 1a and 1b is the **ASC decay order**
relative to the voltage update: GLIF3 integrates the post-decay ASC working
value, the kernel integrates the pre-decay stored value.

## 2. The reduction

### 2.1 Parameter recipe

| GlifConfig field | Reduced value | Why |
| --- | --- | --- |
| `threshold_voltage_coupling_per_s` (a) | **0.0** | kills adaptation at the source |
| `threshold_voltage_decay_per_s` (b) | any validated value ≠ leak_per_s | see NaN hazard, §2.3 |
| `threshold_spike_decay_per_s` | any validated positive value | inert |
| `threshold_spike_increment_mv` (Δ) | **0.0** | s stays identically 0 |
| `voltage_reset_scale` / `voltage_reset_offset_mv` | **0.0 / 0.0** | reproduces `"zero"` reset |
| `threshold_inf_relative_mv` | GLIF3's `threshold_relative_mv` = 25.631077893053025 | same comparison constant |
| `spike_cut_steps` | 77 | GLIF3's cut length |
| passive (C, R, rest) | GLIF3's | same voltage equation |
| `after_spike_taus_ms` | GLIF3's (33.33…, 3.33… ms) | same δ |
| `after_cut_decay` | GLIF3's δcut | same cut dynamics |
| `after_amplitudes_pa` | **Ã = fl(δ ⊙ A)** — the time-shifted amplitudes | §2.4 |

Validation accepts every degenerate choice: `__post_init__`
(cells.py:135-154) constrains only dt, C, R, Θ_inf, b_spike, b_voltage as
positive; reset scale/offset, spike increment, and voltage coupling are
unchecked. Verified by construction in the harness and by red test.

### 2.2 Threshold machinery reduces BIT-EXACTLY

With a = 0: `threshold_phi = a/(b − leak_per_s)` (cells.py:237) is a signed
zero (measured φ = −0.0 for b < λ; b = 19.77 s⁻¹ vs λ = 51.48073062244647
s⁻¹), hence `threshold_voltage_factor = φ·(E_leak − E_dec) = ±0.0`
(cells.py:241-243) and `threshold_beta_factor = −vfac + (a/b)(1 − E_dec) =
±0.0` (244-247). Kernel propagation of the inert state, from initial zeros
(cells.py:309-317):

- `u_{t+1} = E_dec·0 ± 0·ṽ ± 0·β = +0.0`; sums of signed zeros under
  round-to-nearest yield +0.0 (lib.rs:87-90,94);
- `s_{t+1} = 0·decay`, and on spike `fl(±0·δcut) + 0 = +0.0` (91, 120-121);
- threshold compare: `fl(fl(Θ + +0.0) + +0.0) = Θ` bitwise for the positive
  fitted Θ (100-102); Python side compares the same constant with the same
  IEEE strict `>` (cells.py:454);
- reset: `fl(fl(0.0·finite) + 0.0) = +0.0` — bit-identical to GLIF3's literal
  `relative_v = 0.0` (114 vs cells.py:457);
- "ASC added even when zero": `I + fl((0.0+a₀)+a₁)` preserves I bitwise for
  finite I (only the −0.0 input maps to +0.0; value-equal either way, and a
  zero-sign difference cannot flip a strict `>` against a positive threshold).
  The 2-term summation orders match exactly: NumPy's pairwise sum over n=2 is
  sequential `(a₀+a₁)`, the kernel computes `((0.0+a₀)+a₁)`, and `0.0+x = x`
  except the same zero-sign caveat — **verified: 2000 randomized one-step
  trials from hand-set nonzero ASC states, 0 mismatches, bitwise**.

Refractory handling is control-flow identical: freeze all integrated state,
decrement, re-arm to the configured length on spike (lib.rs:67-73,122;
cells.py:444-447,459). Cut lengths 77 vs 113 are data, not structure.

Empirical closure: the degenerate zero-ASC LIF (both sides) is bit-exact for
6000 steps including 17 spikes and all refractory intervals — max |dv| = 0.0
(control D, §4). Everything above is thereby tested, not just argued.

### 2.3 One genuine hazard: φ = 0/0

If a = 0 and `b == leak_per_s` exactly, φ = NaN, the inert-state algebra dies
(u becomes NaN, the cell never spikes silently). Validation does not prevent
it (b only needs to be positive, cells.py:146-147). The recipe must pick
`b ≠ leak_per_s` (any distinct positive value works); a mixture constructor
should assert this explicitly. Red test pins it.

### 2.4 ASC ordering: exactly absorbable, not bit-absorbable

Under the substitution "kernel stored state := GLIF3's post-decay working
value", i.e., initialize `ã₀ = a₀ = 0` and set `Ã = fl(δ ⊙ A)`, the two
systems' voltage recursions become term-by-term identical:

- free evolution: GLIF3 stores `a_{t+1} = fl(δ⊙a_t)` and integrates
  `d_t = fl(δ⊙a_t)`; kernel stores `ã_{t+1} = fl(δ⊙ã_t)` and integrates `ã_t`.
  By induction `ã_t ≡ d_t` bitwise, so Σ, leak, delta, and commit arithmetic
  are the same expressions on the same bits;
- spike commit: GLIF3 computes `fl(A + fl(δcut⊙d_t))` (then applies δ once at
  the first post-refractory step); kernel computes
  `fl(fl(δ⊙A) + fl(δcut⊙fl(δ⊙ã_t)))`. These are equal in ℝ
  (both = δ(A + δcut·d)) but round at different points in IEEE-754.

No parameter choice removes the residual: bit-equality would require
`fl(Ã + X) = fl(δ·fl(A + X))` for every reachable X, which forces δ to be a
power of two — impossible for `δ = exp(-dt/τ)` with the pinned τ. Setting
Ã = A without the shift (the naive clone) leaves the kernel integrating
pre-decay currents against GLIF3's post-decay reference and diverges
macroscopically (control F: 5.6 mV, refractory divergence by step 373).

Measured residual (controls A/B/C, three stimulus regimes, pinned GLIF3 with
its fitted nonzero ASCs): ASC working value ≤ 6 ulp, max |Δa| = 1.137e-13 pA
— exactly one ulp of the 755.84 pA channel; voltage ≤ 4 ulp, max |Δv| =
7.105e-15 mV ≈ 2.8e-16 of threshold; **spike steps identical everywhere**;
refractory counters identical everywhere. Deviations appear only after the
first spike, stay bounded (decays are contractive; each spike re-injects a
deterministic parameter-level rounding), and do not accumulate.

## 3. Verdict

**REDUCIBLE-WITH-HARMLESS-OVERHEAD**, scoped precisely:

1. Threshold adaptation, spike threshold, v_before reset, refractory/cut:
   **exact at bit level** (§2.2, tested by controls D/E).
2. ASC dynamics: exact in real arithmetic under `Ã = δ⊙A`; IEEE deviation
   bounded ≤ ~2 ulp per spike commit (measured ≤ 6 ulp steady state,
   ≤ 1.14e-13 pA), propagating ≤ 7.1e-15 mV into voltage. Spike trains
   invariant across all probes. **Not bit-exact, and provably cannot be made
   so without changing the pinned GLIF3 reference or adding a native
   `'inf'`-mode kernel.**
3. "EXACT-REDUCIBLE" holds only for the degenerate zero-ASC caricature — which
   is NOT specimen 503612028 and would silently misattribute provenance.
4. NOT-REDUCIBLE would require a structural impossibility (different channel
   count, reset family, or refractory semantics); none exists.

## 4. Verification protocol and results

Harness: `artifacts/scratch/glif3_reduction_check.py` replays
`simulate_glif3_current`'s exact loop against a 1-cell
`AdaptiveGlifPopulation` (native kernel, ABI v3) recording committed
state entering each step, compared bitwise (ulp distances capped at 65).
Stimuli: A 600 pA const 4000 steps; B uniform[−40, 900] pA 6000 steps seed
20260823; C piecewise 450/0/−80/750 pA; D zero-ASC both sides, noisy; E
subthreshold noisy; F naive Ã = A red control.

| Control | spikes glif3/reduced | identical | thresholds bits | worst |dv| | worst |da| | refractory |
| --- | --- | --- | --- | --- | --- |
| A const600 | 14/14 | yes | {0} | 3.55e-15 mV (2 ulp) | 1.137e-13 pA (≤6 ulp) | none |
| B noisy | 13/13 | yes | {0} | 7.11e-15 mV (4 ulp) | 1.137e-13 pA (≤2 ulp) | none |
| C piecewise | 10/10 | yes | {0} | 3.55e-15 mV (2 ulp) | 1.137e-13 pA (≤2 ulp) | none |
| D zero-asc | 17/17 | yes | {0} | 0.0 (bit-exact) | 0.0 | none |
| E subthreshold | 0/0 | yes | {0} | 0.0 (bit-exact) | 0.0 | none |
| F naive Ã=A | 14/14 | **no** | {0} | **5.60e+00 mV** | 1.95e+02 pA | diverges @373 |

Auxiliary measurements
(`artifacts/scratch/reduction_aux_checks.py`, outputs in
`reduction_aux_results.txt`): one-step sum-order identity 2000/2000 bitwise;
throughput at N = 65 536 cells over two runs: full GLIF5 0.306–0.318 ms/step
vs reduced GLIF3 config 0.292–0.304 ms/step (ratios 0.92–0.99 — the reduced
config is never slower; any difference is timing noise, ≈ 206–224
Mcell-steps/s); wrapper timestamp offset: `simulate_glif3_current` labels the
same crossing (t+1)·dt vs `simulate_glif5_current` t·dt (cells.py:455 vs 491)
— a trace-layer convention difference, not a dynamics difference.

## 5. Implications for the V1 multi-specimen mixture

- **No dual-model dispatch.** V1 is two `GlifConfig`s (one per subpopulation)
  driven through the existing entry points (`humanbrain_glif_step`,
  lib.rs:675-708; serial/parallel/frontier recurrent paths reused unchanged).
  The kernel is untouched. Per-cell model-type branching inside
  `glif_step_collect` — the alternative the variants doc flagged — is
  unnecessary complexity plus a determinism risk; reject it.
- **Cost of the reduction:** measured zero dispatch overhead (two runs at 65k
  cells: reduced-config/full ratios 0.92–0.99, never slower). Analytic waste
  vs a hypothetical fused GLIF3 kernel: ~5 extra
  f64 multiplies/adds and +32 B memory traffic per integrating cell-step (the
  inert u/s loads-stores), +16 B/cell resident. At V1 experiment scale this is
  inside timing noise; at 100B-cell scale it projects to ~1.6 GB extra
  resident state — the only scenario where a native `'inf'` mode would pay,
  and irrelevant to the preregistered V1 question.
- **Determinism ledger.** The GLIF5 half keeps its existing bit-determinism
  guarantees. The reduced GLIF3 half is deterministic relative to its own
  config bytes, but differs from `simulate_glif3_current` by the §4 residuals;
  any cross-check gate must compare spike steps exactly and voltages under an
  explicit ulp budget, honoring the one-step wrapper timestamp offset. Never
  compare the two wrappers bitwise.
- **Provenance guard must be extended before V1 lands.** `AllenGlif5Parameters.__post_init__`
  (cells.py:136-140) forces `model_id/specimen_id == HUMAN_GLIF5_IDENTITY`, so
  a reduced set silently masquerades as fit 523000114/specimen 508378649 —
  attributing specimen 488392806's dynamics to another specimen's config hash.
  V1 needs an explicit third identity (proposal:
  `HUMAN_GLIF3_AS_GLIF5_IDENTITY` with `derived_from = HUMAN_GLIF3_IDENTITY`)
  or per-specimen identity records inside the composition manifest, and
  `config_sha256` checks extended to mixtures (both raw Allen JSONs verified
  per specimen). Red tests §6 pin this contract.

## 6. Red tests — mixture-constructor contract

Committed at `tests/test_glif3_as_glif5_reduction.py`:

Active now (green guards on this document's claims):

1. reduced parameter set constructs and passes validation with all four
   degenerate constants zeroed;
2. inert threshold state stays bit-zero through spiking runs;
3. degenerate-LIF reduction is bit-exact end-to-end (control D, shrunk);
4. shifted-amplitude reduction matches pinned GLIF3: identical spike steps,
   |dv| ≤ 16 ulp, |da| ≤ 8 ulp budgets (controls A/B/C, shrunk);
5. unshifted naive reduction diverges macroscopically (necessity guard);
6. wrapper timestamp offset equals one step.

Red until V1's constructor exists (skipped with `reason="RED until ..."`,
auto-activating on landing; names are proposals — rename with a one-line
patch and keep the assertions):

7. `MixedSpecimenPopulation` exists and accepts a deterministic per-cell
   specimen assignment manifest;
8. the manifest records BOTH identities and verifies both raw
   `data/allen/*_neuron_config.json` bytes against `config_sha256`;
9. the GLIF3 subpopulation is NOT attributed `HUMAN_GLIF5_IDENTITY` — a
   derived identity or explicit per-specimen records are required, and the
   `b ≠ leak_per_s` φ-NaN guard is enforced at construction;
10. checkpoint round-trip preserves per-cell specimen labels (format v5).

## 7. Claims ledger

| Claim | Verdict | Evidence |
| --- | --- | --- |
| GLIF5 reduces bit-exactly to GLIF3 `'inf'` for threshold/reset/refractory | CORRECT | §2.2 algebra; controls D/E bit-exact incl. 17 spikes; inert-state bits {0} in every run |
| ASC ordering difference is absorbable by `Ã = δ⊙A` | CORRECT in ℝ; ≤6 ulp in IEEE | §2.4 derivation; controls A/B/C (identical spikes, ≤4 ulp v) |
| GLIF3-as-GLIF5 is bit-exact for the pinned specimen | REFUTED | reset-path rounding has no parameter fix (§2.4 functional argument); measured residual |
| Naive clone (`Ã = A`) expresses GLIF3 | REFUTED | control F: 5.6 mV divergence, refractory divergence |
| V1 requires per-cell model-type dispatch in the kernel | REFUTED | single-kernel reduction; measured dispatch ratios 0.92–0.99 across two runs (§5) |
| V1's only real reduction cost is inert-state overhead + provenance work | CORRECT | §5 quantification; cells.py:136-140 guard gap |
