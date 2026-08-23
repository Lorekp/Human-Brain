# Local-rule constant derivation from circuit arithmetic (pre-tuning)

Design research, no tuning runs. Derives principled starting constants for the
dormant local readout rules (**three_factor** dense, **sparse_three_factor**)
from the pinned circuit equations and harness reality, and preregisters a
minimal constant screen. Companion to `experiments/readout_compare/FINDINGS.md`
(pilot: local rules collapse at shipped constants) and
`docs/research/a013-biology-learning-fusion.md` (rules are right, constants were
never fitted). Written against worktree HEAD `081a492`; equations extracted at
lib.rs as shipped in this tree.

Shipped constants under test: `local_learning_rate_pa = 64.0` and
`local_competitive_margin_pa = 64.0` (`src/humanbrain/recurrent.py:112-113`,
ladder arms inherit them because `_grow_circuit` builds
`RecurrentCircuitConfig` without overrides).

Headline result: **the shipped learning rate overshoots the stable scale by a
factor of ~620.** One applied update at lr=64 pA moves the target cell's output
current by ≈ 13,951 pA — 31× the entire `output_base_current_pa` (450 pA). The
derived constant is **lr\* = 0.1 pA**, with margin **m ∈ {363, 726} pA**
(half/whole of the sealed ridge-converged median 726 pA).

---

## 1. Exact update equations (extracted, with every constant)

### 1.1 Dense competitive three-factor — `humanbrain_three_factor_update` (lib.rs:817-876)

Per supervised `strengthen(target)` call (`recurrent.py:725-733`), with
eligibility vector `e` (reservoir traces after the observed token):

```
I_o        = P_o + H_o + Σ_c W[o,c]·e_c          for every output o   (:836-843)
winner     = argmax_{o ≠ target} I_o                                  (:845-856)
if I_target ≥ I_winner + m:  return            # race already won: no update (:857-859)
step       = ℓ / sqrt(1 + ‖e‖₂²)                                      (:861-865)
W[target,:] += step·e ;  W[winner,:] −= step·e                        (:866-872)
P_target    += step ;    P_winner   −= step                           (:873-874)
```

Constants reaching this path: ℓ = `local_learning_rate_pa`, m =
`local_competitive_margin_pa`, H = fixed per-output heterogeneity
`N(0, 0.01²)` pA (`recurrent.py:274`). **W and P are unbounded** — no clamp
exists anywhere in this path.

Consequences used below (exact algebra, no approximation until noted):

- Per-event weight-vector change: ‖ΔW_row‖₂ = ℓ·‖e‖/sqrt(1+‖e‖²).
- Per-event **current** change on target (and symmetrically on winner):
  ΔI = Σ_c ΔW·e_c = ℓ·‖e‖²/sqrt(1+‖e‖²) ≈ ℓ·‖e‖ for ‖e‖ ≫ 1.
- The race gap `I_target − I_winner` moves by 2·ΔI per applied event.

### 1.2 Sparse apposition three-factor — `humanbrain_sparse_three_factor_update` (lib.rs:931-1040)

Called from `recurrent.py:677-724`. Race is computed outside the update; if the
target loses, teacher current `I_target + output_synapse_gain_pa` evokes the
postsynaptic spike (`recurrent.py:686-699`). Gates are 1.0/1.0
(`recurrent.py:708-709`).

```
ages += 1 ;  EMA_o ×= activity_decay=0.99, then += (1−decay)·gate      (:981-990)
a[o][cand] ×= trace_decay = exp(−5 ms/80 ms) = 0.93941   ALL rows, EVERY call (:992-994)
a[target] += e_source·1 ;  a[winner] += e_source·1  (if winner≠target) (:995-1005)
if winner == target: return                       # won race: NO depression (:1006-1008)
contacts (nonzero slots only): w ← clamp(w + ℓ·(±1)·a_slot, 0, W_max)   (:1010-1026)
pacemaker:                     p ← clamp(p + ℓ·(±1)·gate, 0, W_max)     (:1027-1035)
```

Constants: ℓ shared with dense; W_max = `maximum_weight_pa` =
`output_synapse_gain_pa` = 1600 pA (`recurrent.py:704,721`);
fan_in = `readout_fan_in` = 64 contacts/output; structural turnover every 16
presentations, grace 32, hazard 0.1, homeostatic rate 0.05
(`recurrent.py:107-110`). Weights are excitatory-nonnegative by invariant
(`readout.py:263-266`).

Key asymmetry vs dense: the sparse rule has **no competitive-margin gate** —
its skip condition is `winner == target`. The margin knob is dense-only
(`recurrent.py:730`); lr is shared by both rules.

## 2. Eligibility-feature scale from harness reality

### 2.1 First-principles arithmetic (pinned GLIF5 human cell, `cells.py:105-133`)

| Quantity | Value | Source arithmetic |
| --- | --- | --- |
| Rheobase | 79.7 pA | th_inf/R = 23.807 mV / 298.63 MΩ |
| Input pulse drive | 5000 pA = 63× rheobase into fanout 16 | fires each afferent in ≈1.3 ms < 1.5 ms pulse (30 substeps @ dt 0.05 ms) |
| Recurrent drive | ±650 pA = 8× rheobase into fanout 8 | one event alone reaches threshold in ≈10.3 ms > 5 ms token, but synaptic current persists with τ_syn = 8 ms (×0.535/token) and accumulates across tokens → network-wide cascade |
| Spike cut | 5.65 ms > 5 ms token | ≤ 1 spike/cell/token ⇒ eligibility cap e_ss = 1/(1−exp(−5/80)) = 16.49 |
| Eligibility decay | exp(−5/80) = 0.93941 per token | `recurrent.py:705-707`; sub-step 0.99938 |

Both drives sit far above rheobase, so within a task presentation the reservoir
saturates: most cells fire repeatedly, once per token at most.

### 2.2 Read-only probe over the real curriculum (measurement, not a run)

Observe-only probe (no `strengthen`, no grow, no evaluation — same class of
read-only harness use as the FINDINGS post-hoc completion probe): seed-7 /
data_scale-42 cohort, N=512, explicit codebook (status-quo wiring identical to
the pilot's), first 10 TRAIN tasks → **2,111 strengthen positions**, recording
the exact eligibility vector the learner would see:

| Statistic | Value |
| --- | --- |
| ‖e‖₂ at strengthen time — **median [p10, p90]** | **217.98 [146.04, 224.06]** |
| Nonzero cells | median 500 / 512 |
| Reservoir spikes per token | median 293 [p10 269, p90 312] |
| max_c e_c | median 13.07 (< cap 16.49 ✓) |
| RMS e over active cells | 218/√500 ≈ 9.75 |
| E[e_c²] per active cell | ‖e‖²/nnz ≈ 95.0 |

The first-principles picture and the probe agree (spike counts consistent with
cascade saturation; max e just under the once-per-token cap). All derivations
below use ‖e‖₂ = 218 (median); the p10–p90 band 146–224 brackets the derived
operating fraction between 3.2% and 4.9% of base current (§3.1).

## 3. Derived constants

### 3.1 Learning rate — dense-rule current-arithmetic, target f = 5% of base current

Requirement (task-fixed): a single mispredicted transition must move the
affected output current by f·`output_base_current_pa` = 0.05 × 450 pA = 22.5 pA
— large enough to learn from, small enough that no single event dominates the
race the way shipped lr does.

From §1.1: ΔI = ℓ·‖e‖²/sqrt(1+‖e‖²) = 217.98·ℓ at the measured median.
Setting ΔI = 22.5 pA:

**ℓ\* = 22.5 / 217.98 = 0.1032 pA → preregistered round value 0.1 pA** (f =
4.84%; grid arm fractions: 0.025 → 1.21%, 0.4 → 19.4%).

Shipped-diagnosis check: at ℓ = 64 pA, ΔI = 13,951 pA = **31.0× the entire base
current per single event**, i.e. ~620× overshoot. One event saturates any race;
the 64 pA competitive margin is swamped immediately; weights run away until the
eval margins seen in the pilot artifacts (dense three_factor margins
−4,402…−25,640 pA across all nine arms; sparse arms pinned at exactly −64.0 pA
or ≈0 when fully collapsed). This is precisely the constant-output degeneracy
FINDINGS documents, now explained arithmetically.

Cumulative-coherence check at ℓ\*: the race gap moves 2×21.8 = 43.6 pA per
applied event, so a lost race sitting at the sealed deficit scale (−726 pA)
needs ≈ 17 net consecutive mispredictions to flip — well within the recurrence
budget of the massed protocol (~26.6k strengthen events per arm over 42 tasks ×
211 transitions × 3 passes).

Sparse cross-check at the shared knob: per row-event, contact Δw = ℓ·a_slot
with apposition trace a_slot ≈ O(e_RMS) ≈ 9.75 (traces decay 0.93941 per call
network-wide, so a row's trace at its own event ≈ fresh increment, not an
accumulated sum). Correlated re-race row swing ≈ ℓ·Σ_s e_s² ≈ 0.1032 × (64×95)
≈ 628 pA — above m*/2 but self-limited: once the row wins its race, the sparse
rule early-returns (no further target-side updates) and dense's margin gate
stops updates at +363 pA. Hot rows (framing-delimiter targets recur thousands
of times per arm) can still ride the 1600 pA clamp — flagged as a screen
observable, §5.

### 3.2 Competitive margin — half of the sealed converged median

Sealed a012 N=128 ridge-converged competitive-margin distribution: 246
positions, min +56 pA, **median 726 pA** (`docs/research/a013-teaching-processes.md:73`);
independent sanity arm under spaced/program-only: median 1,079 pA
(`experiments/a013-ladder-sanity-v1/sanity-ridge.json`). Race noise floor:
heterogeneity σ = 0.01 pA — negligible at these scales.

Gate semantics (§1.1): an update is skipped iff `I_target ≥ I_winner + m`, so m
is the "won" threshold. Setting it **below** the converged operating point makes
training concentrate on genuinely weak races and stop once a healthy margin is
reached (shipped 64 pA sits far below even pathological margins, so training
never stops — churn without convergence):

**m* = 726/2 = 363 pA** (primary), second arm **m = 726** ("stop only at
sealed-converged scale"). Step coherence: ≈ 8 net events to cross 363 pA at
ℓ\* — responsive, not one-shot.

### 3.3 Clamping bounds — consistent with weight semantics

Weight semantics constraint (both rules): contact/synapse currents sum against
`output_base_current_pa` = 450 pA, and the ridge rule scales its fitted
coefficients by exactly `output_synapse_gain_pa` = 1600 pA
(`recurrent.py:790-792`) — that gain is the established per-synapse dynamic
range of this readout.

| Rule | Clamp | Status at derived constants |
| --- | --- | --- |
| sparse | w, pacemaker ∈ [0, 1600 pA] (existing) | **Keep.** Events-to-saturate one contact ≥ 1600/(ℓ\*·9.75) ≈ 1,589 row-events (≈397 at the ×4 arm) — non-binding for typical rows; hot delimiter rows may bind at ×4, which the screen should observe, not prevent. |
| dense | none today — **derive**: \|W[o,c]\| ≤ 1600 pA (gain; same semantics as ridge scaling) and \|P_o\| ≤ 450 pA (learned bias must not exceed the tonic drive itself ⇒ resting output current stays within [0, 900] pA) | **Guardrails only**: binding needs ≥ 1600/(step·e_RMS) ≈ 3.5×10⁵ aligned events and 450/(ℓ\*/‖e‖) ≈ 9.5×10⁵ events vs the ≈26.6k-event arm budget — non-binding even at 4ℓ\* (87k / 237k). They bound pathology under mis-set constants, not the derived operating point. Requires a native signature change → explicitly out of screen scope; preregistered follow-up (T-clamp), not bundled into a constants screen. |

## 4. Preregistered constant screen — `local_rule_constant_screen_v1`

Minimal arm matrix, derived constants as arm centers, pilot-matched protocol.
No other mechanism changes; no codebook arms; frozen before first run.

Fixed configuration (all arms): rule = `three_factor` (margin knob is
dense-only, §1.2), schedule = `massed`, data_scale = 42,
supervision = `all_transition`, N = 512, seed = 7, harness defaults otherwise;
runner `benchmarks/ladder_experiments.py`; metrics JSON per existing schema.
Cost ≈ 11 s/arm CPU (pilot three_factor N=512 wall_s ≈ 10.7).

Arm matrix (6 combos = lr {derived/4, derived, derived×4} × margin {median/2,
median}):

| Arm id | local_learning_rate_pa | local_competitive_margin_pa |
| --- | --- | --- |
| `tf_lr0.025_m363_N512_s7` | 0.025 | 363 |
| `tf_lr0.1_m363_N512_s7` | 0.1 | 363 |
| `tf_lr0.4_m363_N512_s7` | 0.4 | 363 |
| `tf_lr0.025_m726_N512_s7` | 0.025 | 726 |
| `tf_lr0.1_m726_N512_s7` | 0.1 | 726 |
| `tf_lr0.4_m726_N512_s7` | 0.4 | 726 |

Primary widen-gate (per arm, evaluated from its metrics JSON):
**non-degenerate margins AND real replay** —

```
margin_count > 0  AND  isfinite(margin_min_pa)  AND  margin_min_pa > 0
AND  train_replay_tasks >= 21   (of 42)
```

Secondary (reported, not gated): train_cases_passed (/126),
calib_cases_passed (/54), margin_median_pa, and the FINDINGS degeneracy probe
(share of completions equal to the constant pass-through program) applied to
any passing arm before it is credited.

Decision rules (preregistered):

1. Only combos passing the primary gate widen, to seeds {7,8,9} × N
   {128,512,1024} (9 runs/combo, identical protocol). Failing combos close.
2. If **no** combo passes: the derived-constant family is falsified at this
   feature scale; report and stop. No post-hoc tuning — same discipline FINDINGS
   recorded ("a sensitivity sweep would be a new preregistered experiment";
   this document is that preregistration).
3. Any wider run motivated by these results must cite this file and keep the
   losing arms' JSONs unmodified.

Preregistered predictions (falsifiable):

- P1: `lr=0.1, m=363` passes all primary gates.
- P2: `lr=0.4` arms fail via margin degeneration (min ≤ 0) or replay < 21 —
  the overshoot boundary.
- P3: `lr=0.025` arms fail via replay < 21 — under-training boundary
  (5.4 pA/event gap motion).
- Family falsifier: any combo passes while P1's combo fails.

## 5. Harness support needed (constant-injection knobs)

Current gap (verified at HEAD `081a492`): `_grow_circuit`
(`benchmarks/ladder_experiments.py:439-453`) constructs `RecurrentCircuitConfig`
without lr/margin kwargs; `run_experiment` (:611) has no such parameters; the
CLI parser has no flags. `RecurrentCircuitConfig` already owns the fields
(`recurrent.py:112-113`) with positivity validation (`:139,142`), and the
metrics record auto-documents whatever config was used via
`asdict(...)` → `config_dump.circuit_config` — so **plumbing alone** suffices;
no schema change, no native change.

Required plumbing (mirrors the landed `--codebook` pattern exactly):

1. `run_experiment(..., local_learning_rate_pa: float = 64.0,
   local_competitive_margin_pa: float = 64.0)` — defaults preserve shipped
   behavior byte-for-byte; validate both > 0 and finite up front (before
   curriculum build) raising `ValueError`.
2. Thread through `_grow_circuit` into `RecurrentCircuitConfig(...)`.
3. CLI: `--local-lr-pa` / `--local-margin-pa` (float, dest
   `local_learning_rate_pa` / `local_competitive_margin_pa`, defaults 64.0).
4. Determinism guarantee carries over unchanged (identical arguments →
   byte-identical record minus `wall_s`; injected values appear in
   `config_dump.circuit_config`).

Red tests (verbatim, ready to add as `tests/test_ladder_local_constants.py`;
following the `test_ladder_codebook.py` importlib convention). **All four are
RED at HEAD** (knobs absent → TypeError / missing flags) and green only when
the plumbing above lands. They are specified here rather than committed
because this task's write scope is restricted to this document:

```python
"""Red tests: local-rule constant injection knobs for ladder_experiments.

Contract for the preregistered local_rule_constant_screen_v1
(docs/research/local-rule-constant-derivation.md §5). Red against harness
HEAD 081a492 (constants hardcoded at shipped 64.0/64.0); green only when
run_experiment/_grow_circuit/CLI plumb RecurrentCircuitConfig's
local_learning_rate_pa and local_competitive_margin_pa.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("ladder_experiments", HARNESS)
SPEC.loader.exec_module(HARNESS)


def test_run_experiment_accepts_local_rule_constants() -> None:
    params = inspect.signature(HARNESS.run_experiment).parameters
    assert params["local_learning_rate_pa"].default == pytest.approx(64.0)
    assert params["local_competitive_margin_pa"].default == pytest.approx(64.0)


def test_grow_circuit_forwards_constants_into_config() -> None:
    src = inspect.getsource(HARNESS._grow_circuit)
    assert "local_learning_rate_pa=local_learning_rate_pa" in src
    assert "local_competitive_margin_pa=local_competitive_margin_pa" in src


def test_cli_exposes_local_constant_flags(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        HARNESS.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--local-lr-pa" in out
    assert "--local-margin-pa" in out


def test_nonpositive_local_constants_rejected_up_front() -> None:
    with pytest.raises(ValueError):
        HARNESS.run_experiment(
            experiment_id="knob-validation-probe",
            arm="invalid",
            n_cells=128,
            seed=7,
            readout_rule="three_factor",
            schedule="massed",
            data_scale=42,
            supervision="all_transition",
            local_learning_rate_pa=0.0,
            local_competitive_margin_pa=363.0,
        )
```

Note: the dense-rule clamps of §3.3 would additionally need a native
signature change plus Python plumbing — deliberately NOT part of the screen's
red-test contract (constants-only screen; clamps are guardrails that cannot
bind at the derived constants, §3.3).

## 6. Threats to validity

- Feature-scale measurement is single-cohort (seed 7, N=512, 10 tasks,
  2,111 positions). Cross-seed/cross-N variance was not measured; the dense
  normalizer partially self-corrects (ΔI = ℓ·‖e‖²/sqrt(1+‖e‖²) adapts to ‖e‖),
  and the ×4/÷4 grid brackets residual error.
- ‖e‖ is schedule-dependent (massed carryover inflates traces vs spaced); the
  screen pins massed to match the pilot, so the derived scale matches the
  protocol it will run under. Re-derive if the schedule axis changes.
- Eval-time margins are computed among allowed-token subsets
  (`_complete_program`), teach-time gates among all outputs — the sealed 726 pA
  median is the eval-side distribution; the gate comparison is approximate by
  construction, which is why the margin axis has two levels instead of one.
- Sparse trace-scale estimates assume correlated re-races (upper bound on the
  per-row swing); the sparse rule shares only the lr knob, so its response to
  the sweep is observational, not gated.
