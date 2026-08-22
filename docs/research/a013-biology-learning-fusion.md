# Biology-of-learning survey: mapping real plasticity onto the Human Brain simulator

**Mission:** read-only survey. All claims verified against source; no files modified. Deliverable below.

## 0. The substrate (verified)

- **Cells** (`src/humanbrain/cells.py`): one pinned Allen human MTG-L3 pyramidal **GLIF5** fit (`AllenGlif5Parameters`, cells.py:106) — adaptive threshold (spike increment + voltage coupling), two exponential after-spike currents, voltage reset, 5.65 ms spike cut, dt = 0.05 ms. Integrated natively (`AdaptiveGlifPopulation.step`, cells.py:391 → `NativeGlifKernel`). Point cells: `AdaptiveGlifState` holds exactly the **8 mutable fields** listed in `procedural_runtime.py:1-11` (voltage, 2 ASCs, spike/voltage threshold components, refractory counter, recurrent synaptic current, eligibility).
- **Circuit** (`src/humanbrain/recurrent.py`, `RecurrentCellCircuit`): token → fixed random afferents (5,000 pA, 1.5 ms pulse per 5 ms token); sparse signed recurrent fanout (τ=8 ms current decay, recurrent.py:99); **eligibility trace τ=80 ms** (recurrent.py:100, decay precomputed :293); output by **first-spike race** among token-cells using a compiled resting GLIF5 unit response (`_compile_output_race` :332; `race` :833).
- **Sealed failure** (`docs/typed-composition-a012-v1-results.md`, run `hb-a012-once-20260822`, N=128): TRAIN replay 40/42 and cases 120/126 (**capacity shortfall**); calibration replay 0/18, cases 7/54 (**zero transfer**) despite strong margins (246 competitive positions, min +56 pA, median 726 pA). Interpretation on record: "grown synapses carry task-specific traces rather than transferable program structure." Note: the sealed run used the **ridge** readout (`typed_program.py:_CIRCUIT_CONFIG`:327 leaves `readout_learning_rule="ridge"`, 3 epochs, ≤256 examples, A2 program-token-only supervision).

---

## 1. STDP & eligibility traces — what the simulator already implements

**Biology.** Hebbian plasticity is bidirectional and timing-sensitive: pre-before-post spiking within ~10–20 ms potentiates a synapse, post-before-pre depresses it, with the coincidence detected by NMDAR calcium entering via the dendritic spine. Because the coactivity signal decays over hundreds of ms, modern theory splits learning into a fast local **eligibility trace** (synapse-specific "was this synapse involved?") and slower modulatory gates. Gerstner et al. (2018) formalize this as the **three-factor rule**: Δw ∝ (local eligibility) × (postsynaptic activity) × (neuromodulatory/teacher signal).

**What the code implements (exact equations).**
- *Eligibility:* pure presynaptic spike-count trace. In `_native_src/src/lib.rs` → `humanbrain_observe_token_reference_v1` (~line 795): every 0.05-ms step `e *= exp(-dt/80ms)`; on a presynaptic spike `e[pre] += 1.0`. No postsynaptic term at the reservoir level.
- *Dense three-factor readout* (`lib.rs:817` `humanbrain_three_factor_update`, called via `recurrent.py:725-732`): s_j = pacemaker_j + heterogeneity_j + Σᵢ w_ji·eᵢ; let k = strongest non-target output. If s_target < s_k + margin: step η = lr/√(1+Σeᵢ²), then w_target,i += η·eᵢ, w_k,i −= η·eᵢ, pacemaker_target ±= η. This is a perceptron-style competitive Hebbian rule with a trace-norm homeostatic scalar — precisely what `docs/local-readout-learning.md` documents, citing Gerstner 2018 / e-prop.
- *Sparse three-factor* (`lib.rs:931` `humanbrain_sparse_three_factor_update`, driven from `recurrent.py:677-723`): the only genuine **pre×post coincidence trace** — apposition[o,c] ← apposition·exp(−token_interval/80ms) + e_pre(source)·(actual output spike gate), where a teacher-evoked spike is synthesized when the race winner ≠ target (`recurrent.py:685-699`: teaching current = current[target] + `output_synapse_gain_pa`). Weights: w += lr·modulation(±1)·trace, clamped [0, 1600 pA].
- *Ridge path used in the sealed run* (`recurrent.py:667-792`): `strengthen` accumulates moments y += f and batched normal matrix A += ffᵀ with f=[e;1] (`_flush_growth_statistics` :764), solved once at `freeze_learning` as (A+λI)w=Y with λ reduced 99% for the pacemaker bias (:774-792). This is FORCE-style least-squares readout (Sussillo & Abbott 2009) on eligibility features — not online STDP.
- *Missing vs biology:* no LTD window or pre/post asymmetry, no timing-dependent weight change at recurrent synapses (those weights are frozen except developmental scaling/pruning), no inhibitory-side plasticity.

**Mapping change:** none needed to claim STDP-analogue status — cite `lib.rs:931`. To get closer biology: make recurrent-synapse eligibility edge-resolved (currently cell-resolved) and add an LTD term −η·pre·post for mistimed pairs.

**Impact on sealed failure:** medium. Eligibility features are shared across examples (same substrings → same traces), so transfer capacity exists in principle; the failure is that ridge fits each example's full transition set equally, saturating margins on task-specific transitions. **Cost:** low (rules already exist behind `READOUT_LEARNING_RULES`, recurrent.py:22). **Risk:** low.

---

## 2. Neuromodulation: dopamine RPE as third factor

**Biology.** Dopaminergic VTA/SNc neurons broadcast a reward-prediction-error δ = r − V(s) as a global brain-wide signal that gates cortico-striatal and cortical plasticity: no dopamine, no LTP consolidation (zero δ blocks potentiation even with coincident pre/post). This is exactly temporal-difference credit assignment instantiated in tissue, and it explains how animals learn from sparse, delayed reward rather than every moment being equally imprinted.

**Mapping.** The third factor already exists but is a flat instruction: `strengthen(target)` always applies modulation ±1 toward the supervised token (lib.rs:931 `update_contacts(target, +1); update_contacts(winner, -1)`; dense analogue lib.rs:817). What would change: compute a δ-like scalar from machinery already present in `recurrent.py:677-711` — race outcome, winner identity, teaching current, margin deficit are all computed there — and pass `learning_rate_pa = f(δ)` into `self._sparse_readout.teach(...)` (or skip the update when δ≈0). Zero new state; the native kernel already takes a scalar rate per call. CS fusion: this is policy-gradient-with-baseline credit assignment (update only where prediction was wrong), replacing "imitate every transition."

**Impact on sealed failure:** high for **zero calibration transfer**. RPE gating suppresses updates on already-predicted (shared-structure) transitions, freeing representational budget for surprising ones and reducing interference between tasks sharing substrings — the documented mechanism of failure ("task-specific traces," results doc §Interpretation). Low direct effect on raw capacity. **Cost:** very low (one call-site scalar). **Risk:** low-medium (reward shaping choices matter; must keep the preregistered supervision contract intact — `supervision_scope` gating in `typed_program.py` grow loop :1216-1228 must not be altered).

---

## 3. Consolidation & hippocampal replay

**Biology.** During quiet rest and sleep, hippocampal sharp-wave/ripples re-activate waking sequences at compressed speed, and systems consolidation gradually transfers structure into cortex through repeated offline reactivation. Replay is sparse, error-and-novelty weighted (more replay of rewarded/surprising trajectories), and happens *after* experience without changing the environment — the brain's offline second pass.

**Mapping.** Growth today is strictly forward: `grow()` in `typed_program.py` (~:1183-1232) runs 3 ordered epochs of observe→strengthen with `reset_sequence()` per example, then freezes forever (`freeze_learning`, recurrent.py:794; completion asserts byte-level state immutability). There is no post-pass. A consolidation stage would store the ≤256 tiny teacher-forced sequences and, before freeze, re-run passes applying `strengthen` **only on transitions whose current race violates the target** (error-driven = prioritized experience replay à la DQN's PER). For ridge this is importance-weighted moment accumulation (reweight rows of `_target_moments`/`_normal_matrix`); for three-factor it is genuine replay. `development.py`'s staged controller (`grow_developmentally`, :455+; `close_stage`) is the natural place to hang a "rest phase" between stages.

**Impact:** high for **capacity** (the 2 failed TRAIN tasks / 6 failed cases are precisely low-priority tail examples that uniform epochs under-weight); medium-low alone for calibration transfer (replay re-imbues task-specific traces unless combined with #2's gating — the fusion is the point). **Cost:** medium (example store + scheduler; protocol-wise it must enter the preregistered a013 ladder — the A012 cohort is permanently spent, docs results doc §Dispositions). **Risk:** medium (naive replay deepens memorization without transfer; measure both axes separately as a013 already plans).

---

## 4. Synaptic scaling / homeostasis — why unbounded potentiation degrades

**Biology.** Turrigiano-style synaptic scaling multiplicatively rescales a neuron's excitatory inputs to stabilize firing rates, preserving relative weight ratios while preventing runaway LTP; without it, saturated synapses destroy the coding headroom needed to encode anything new. Homeostatic failure looks exactly like this system's symptom: everything learned, nothing discriminated.

**Mapping.** Three homeostat fragments already exist: (a) sparse-readout turnover scales all weights of an output cell by exp(rate·(1/N_out − activity_EMA)) toward uniform activity (`lib.rs` `humanbrain_sparse_structural_turnover` :1060+, invoked recurrent.py:712-723 every 16 target presentations); (b) developmental scaling multiplies incoming recurrent weights toward median activity (`development.py apply_activity_homeostasis` :337-363); (c) the trace-norm denominator η = lr/√(1+Σe²). **But the sealed ridge path has none**: `_consolidate_readout` (recurrent.py:774) applies only fixed λ=1.0 regularization and the ×1600 pA gain — nothing constrains the resulting 726 pA median margins, which is unbounded-potentiation-by-proxy: absolute magnitudes carry task identity instead of relative pattern. Change: post-solve multiplicative row normalization (preserves race ordering since first-spike latency is monotone in current — see `_compiled_first_spike_steps`, recurrent.py:974) plus λ treated as a scale knob.

**Impact:** addresses the "strong-margins-but-no-transfer" signature directly: compressing magnitudes converts absolute-level coding into pattern coding and frees dynamic range, though by itself it cannot create transferable features (diagnostics showed state separation ≈0.001 cosine regardless of framing). Best as stabilizer for #1/#3. **Cost:** very low (transform after existing solve). **Risk:** low — overshoot risks dropping min margin (+56 pA) below competitive threshold; guard with the existing race RuntimeError ("cells did not spike").

---

## 5. Dendritic nonlinearity & astrocytic modulation — feasibility note only

**Biology.** Pyramidal neurons compute with active dendrites: clustered spines trigger local NMDA nonlinearities so a cell performs multi-branch feature binding — single-neuron XOR-like computations that expand the state space without extra somas. Astrocytes ensheath thousands of synapses and regionally modulate plasticity thresholds and extracellular glutamate/K⁺, acting as slow, spatially organized gain controllers.

**Feasibility in this simulator: NOT possible without model upgrades.** GLIF5 is a single-compartment point model (`AdaptiveGlifState`, cells.py:287 — 8 scalars/cell); the output race collapses each output cell to one compiled linear resting response (`compile_resting_response`, cells.py:349; monotonicity enforced recurrent.py:343-346), i.e., current→latency is deliberately linear-in-log only. Compartments would mean a two-compartment/multi-timescale upgrade invalidating the Allen fit identity checks (`HUMAN_GLIF5_IDENTITY.config_sha256` guards, recurrent.py:559-562) and every checkpoint. Cheap proxy available now: astrocyte-like regional gain — one slowly varying scalar multiplying `recurrent_current_pa` or local plasticity rate per cell neighborhood (no morphology needed, checkpointable as one small array). Potential payoff for the failure is real (dendritic nonlinearities raise effective state rank; the diagnostics explicitly found state separation unsupported at 128 cells) but this is a next-generation item: **high cost, high risk, defer**.

---

## 6. Critical periods & neuromodulatory gating of plasticity rate

**Biology.** Cortical plasticity is developmentally scheduled: juvenile circuits learn nearly anything, then molecular brakes (perineuronal nets, Nogo-family signaling) and maturing inhibition lock rates down; adult re-opening requires neuromodulators (acetylcholine/norepinephrine restore juvenile-like plasticity, e.g., during attention/surprise). Learning thus alternates between open windows and protected consolidation.

**Mapping.** Fragments exist: staged cell recruitment (`development.py recruit_to`/`stage_active_cells`), lane-based supervision gating (A0/A1 strengthen all transitions vs A2 program-token-only — grow loop comment, typed_program.py:1216-1228), and the terminal lock (`freeze_learning`). Missing: any within-growth rate schedule — `local_learning_rate_pa`/margin (recurrent.py:112-113) are constant. Change: anneal (lr, margin) across the 3 epochs — wide-open early epoch, gated late epochs — and/or open transient "cholinergic" windows (briefly raised lr) when curriculum phase changes. Trivially composable around existing calls.

**Impact:** protects **capacity** (early broad acquisition, late fine-tuning reduces late-example overwriting of early structure — consistent with 40/42 after 3 identical epochs); indirect for transfer; strongest as a wrapper around #1+#3. **Cost:** very low. **Risk:** low (schedule mis-tuning just wastes an epoch).

---

## Ranked top-3 biology × CS fusion proposals

**#1 — Error-gated three-factor plasticity ("dopamine RPE × policy-gradient credit assignment").**
Biology: dopaminergic third factor gating Hebbian consolidation (§2). CS: baseline-subtracted policy gradient — update only on prediction error. Concretely: switch the typed-program path to the already-implemented `sparse_three_factor` rule (it is dormant code — recurrent.py:22, lib.rs:931) and derive its modulation from the race outcome already computed in `recurrent.strengthen` (winner vs target, margin deficit), skipping updates when the target already wins by margin. Targets **zero-calibration transfer** head-on: stops imprinting every transition, concentrates synaptic budget on surprising/shared-structure decisions.
**Decisive first experiment (public data only, inside the preregistered a013 dev-gate):** at matched N=128 and N=512, run ridge vs δ-gated sparse-three-factor under the identical framing/supervision contract; predeclare success as TRAIN-case parity AND >0 opposite-template dev-gate improvement (the exact axis that scored 7/54). One paired-seed family per docs/research/human-ei-cell-types-and-balanced-dynamics.md §preregistration discipline.

**#2 — Rest-phase replay consolidation ("hippocampal replay × prioritized experience replay").**
Biology: sharp-wave-ripple replay during rest (§3). CS: DQN-style prioritized replay buffer. Store the ≤256 tiny sequences; insert consolidation passes before `freeze_learning` that re-strengthen only margin-violating transitions (error priority), optionally interleaved across task families to force shared-trace reuse. Targets **capacity** (the 40/42 shortfall) directly and cheaply.
**Decisive first experiment:** at N=128 ridge, add one error-driven replay pass and test whether TRAIN reaches 42/42 while dev stays flat. If yes → capacity was statistics-limited (replay suffices); if TRAIN recovers only with #1 also on → representation-limited, confirming the fusion requirement. Either outcome cleanly partitions the two deficits a013 needs separated.

**#3 — Homeostatic margin normalization + critical-period schedule ("synaptic scaling × normalization/LR schedules").**
Biology: Turrigiano scaling + developmental plasticity windows (§4, §6). CS: layer-normalization + LR warmup/decay. Post-`_consolidate_readout` multiplicative row normalization to a target margin quantile (ordering-safe by latency monotonicity, recurrent.py:974) plus epoch-annealed (lr, margin) during growth. Doesn't create transfer by itself; prevents #1/#2 from re-creating the strong-margin saturation pathology and keeps min-margin above the race-failure floor.
**Decisive first experiment:** sweep one post-solve scale α ∈ {0.25, 0.5, 1, 2} on the public TRAIN corpus; verify race-winner invariance and measure margin distribution + TRAIN replay. Establishes the safe operating band before #1 lands.

*(Dendritic/astrocytic mechanisms: excluded from top-3 — infeasible on point-cell GLIF5 without invalidating fitted-model identity and checkpoints; revisit only as a regional-gain proxy if #1–#3 stall.)*

**Key files referenced:** `src/humanbrain/cells.py` (GLIF5 params/state/race compilation), `src/humanbrain/recurrent.py` (strengthen dispatch :667, ridge stats :764-792, teach :677-723, race :833-949), `src/humanbrain/_native_src/src/lib.rs` (observe kernel ~:740, three-factor :817, sparse three-factor :931, structural turnover :1060), `src/humanbrain/readout.py` (sparse dendrite state/plasticity API), `src/humanbrain/development.py` (staged recruitment/homeostasis :280-500), `src/humanbrain/typed_program.py` (grow :1183-1232, complete :1253+, config :325-344), `docs/local-readout-learning.md`, `docs/typed-composition-a012-v1-results.md`, `docs/research/human-ei-cell-types-and-balanced-dynamics.md`.

---

## Summary

- **Did:** Read-only survey of the Human Brain simulator's learning substrate (cells/recurrent/native kernel/readout/development/typed-program paths, both named worktrees confirmed merged) plus prior neuro research docs; mapped six biological learning mechanisms onto the exact implemented equations and ranked fusion proposals against the sealed A012 failure (TRAIN 40/42, calibration 0/18, margins median 726 pA, N=128, ridge readout).
- **Found:** The simulator already implements a genuine Gerstner/e-prop-style three-factor rule — presynaptic eligibility τ=80 ms (`lib.rs` observe kernel) with competitive margin updates (`lib.rs:817`) and a true pre×post coincidence trace in the sparse path (`lib.rs:931`) — but the sealed run used the ridge least-squares path, which has **no** error gating, **no** homeostatic normalization, and **no** replay/consolidation phase; those three gaps map one-to-one onto the zero-transfer and capacity deficits. Top proposals: (1) δ/RPE-gated three-factor plasticity (= dopamine × policy-gradient), (2) error-driven rest-phase replay (= hippocampal replay × PER), (3) homeostatic margin scaling + critical-period LR schedule (= synaptic scaling × norm/LR schedules), each with a decisive first experiment runnable on public data inside the preregistered a013 ladder. Dendritic/astrocytic computation is infeasible on point-cell GLIF5 without invalidating model identity — noted as feasibility-only.
- **Created/modified:** Nothing (read-only mission honored); deliverable is the survey text above only.
- **Issues:** None blocking. `.pytest_cache` permission denial and binary `.pdb` grep noise were irrelevant; worktrees live under `DeepSim/.worktrees/` (not inside the repo) — both inspected read-only.