# Regioned architecture A013: Region E as the distributed-representation mechanism, Region P as processor

**Status:** design proposal, read-only research worktree; no code changed.
**Branch context:** `codex/a013-cognition-design` @ `2fdd3d0`.
**Sealed inputs:** `docs/typed-composition-a012-v1-results.md` (run `hb-a012-once-20260822`),
`docs/research/a013-biology-learning-fusion.md`, `docs/research/sparse-cell-output-readout.md`,
`docs/research/human-ei-cell-types-and-balanced-dynamics.md`,
`benchmarks/procedural_runtime_gate3_slice_protocol_v1.json`, `docs/native-scaling-v2.md`,
`docs/research/100-billion-cell-architecture.md`.

## 0. Position statement (binding recast)

The core unsolved problem is **organizing complexity via distributed representation**:
concepts must be *patterns across populations*, not `this-neuron-equals-the-word-cat`.
Region E's primary job is therefore **the distribution mechanism itself**: it maps
tokens into **overlapping sparse patterns** so that semantic and structural similarity
become physical overlap, and downstream learning automatically shares features across
related tasks.

Said plainly: **a region that merely relocates grandmother cells buys nothing.** If E
assigns token *t* a fixed private assembly that no other token ever shares, we have moved
the explicit-token-identity failure into a new address space. The sealed A012 result is
precisely what explicit identities produce — N=128 monolithic reservoir with each token
owning fixed random afferents reached TRAIN replay 40/42, calibration replay **0/18**,
calibration cases 7/54, and state separation ≈ 0.001 in every lane. A two-region layout
whose E-codebook is grandmother-style reproduces that null by construction. Every design
choice below is judged by one question: *does similarity in the task domain become
overlap in the physical substrate?*

## 1. Why today's single reservoir fails this test

In the current circuit (`src/humanbrain/recurrent.py`), each token injects a 5,000 pA,
1.5 ms pulse into its own fixed random afferent subset of the reservoir every 5 ms token
interval (`input_pulse_ms = 1.5`). Two tokens' subsets overlap only at chance. Consequences:

1. Learned traces anchor to token-specific cells; there is no physical channel by which
   training on "5+7" partially prepares "7+5" or any other shared-substring task.
2. Readout margins grow around absolute current levels tied to those private cells
   (median +726 pA in the sealed run) — magnitude coding, not pattern coding.
3. State separation between tasks stays at chance (~0.001 1−cosine) because the state
   distributions barely differ structurally: different private cells, same statistics.

The a013 biology-fusion survey locates the missing machinery in learning-rule upgrades
(RPE gating, replay, homeostasis). This document attacks the layer underneath: **what the
input representation even allows the learning rules to share.** The dormant sparse
three-factor rule (`lib.rs:931`) already generalizes across examples *whose eligibility
features overlap*; the failure is that explicit token identities make that overlap
chance-level. Regions fix the geometry first; the fusion proposals then act on a
substrate worth acting on.

## 2. (A) Concrete region model

### 2.1 Region E — the distribution mechanism

- **Population:** `N_E` configured GLIF5 cells, contiguous gid range `[0, N_E)`.
  Same fitted cell model (`config_sha256 69fe7260…`); no new cell class.
- **Computation:** token → **k-of-N_E sparse assembly**. For incoming token `t`, the
  frozen codebook assigns `C(t) = {g₁ < … < g_k} ⊂ [0, N_E)` and drives exactly those k
  cells with today's injection constants (5,000 pA, 1.5 ms pulse per 5 ms token).
- **No internal recurrence in v1.** E has no recurrent synapses and no plasticity; it is
  deliberately an expander, not a second reservoir. All recurrence (τ=8 ms signed fanout)
  and all learning stay in P. Rationale: the distribution mechanism lives in the
  *geometry of the codebook*; giving E its own dynamics would confound the decisive
  experiment (§5). Experience-formed codes (§2.2, family 3) later add E-side plasticity
  as a separate arm.
- **Overlap property:** for tokens `s, t`, `m(s,t) = |C(s) ∩ C(t)|` is the *only*
  channel through which similarity enters the system. P receives from shared E cells via
  literally the same synapses, so any trace grown while processing `s` is partially
  pre-grown for `t`. That is the mechanism by which "downstream learning automatically
  shares features across related tasks."

### 2.2 Codebook design (first-class section)

The codebook is the load-bearing component of the whole proposal. Three families,
in increasing ambition; all are frozen before training and hash-committed like
`source_binding` artifacts.

**Family 1 — random sparse codes (control).**
Each token draws k distinct gids from the seeded philox stream
(`random123-philox2x64-10-v2`). Similarity structure: none beyond chance. This arm
isolates *distributedness alone* from *similarity structure*: if Family 1 already beats
grandmother-cell wiring on calibration transfer, overlap-per-se is doing real work;
if not, structured families carry the effect.

**Family 2 — embedding-projected top-k (similarity-structured by construction).**
Token embedding `x_t ∈ ℝ^d` (byte-ngram hashed, or any frozen pretrained vector);
project `y_t = W x_t` with a **frozen seeded Gaussian** `W ∈ ℝ^{d×N_E}`; take top-k
components of `y_t`. For random `W`, expected set-overlap increases monotonically in
`cos(x_s, x_t)` (a random-projection/JL argument: nearby vectors agree on their large
coordinates more often than distant ones). Semantic/structural similarity thus becomes
physical overlap *before any learning happens*. Determinism rule: the projection runs
host-side once, offline; the committed artifact is the **sorted uint64 gid list**, and
the native hot path stays integer-only. Top-k ties break by ascending gid.

**Family 3 — experience-formed Hebbian assemblies.**
Start from Family 1 or 2 seeds; during growth, tokens co-active within the τ=80 ms
eligibility window strengthen shared membership through local apposition traces on a
fixed candidate pool (per `sparse-cell-output-readout.md`: pool C ≤ 4k per assembly,
K active slots, slow seeded stochastic turnover, no argmax rewiring). Assemblies drift
toward corpus statistics. Named risk: overly strong competition collapses assemblies
back toward grandmother cells — participation entropy per assembly is a preregistered
diagnostic (human-ei gates §activity), and collapse is a *reportable failure mode*, not
a tuning target.

**Interference / superposition arithmetic (shared treatment).**

| Quantity | Formula | Meaning |
| --- | --- | --- |
| unit load | `L = T·k / N_E` | expected #token-codes containing a given E cell |
| chance overlap | `E[m] = k² / N_E` (hypergeometric mean) | Family-1 pairwise sharing |
| transfer coefficient | `ρ(s,t) = m(s,t) / k ∈ [0,1]` | fraction of a trace grown for s pre-grown for t |
| P-afferent activity | `k / N_E` | family-invariant drive statistics |

Worked point (illustrative constants): `N_E = 4096`, `k = 64` → `L = T/64`; two
*unrelated* tokens share `E[m] = 64²/4096 = 1.0` units; under a similarity-structured
codebook related pairs move `ρ` toward 1 while unrelated pairs stay near `1/N_E·k⁻¹`
noise floor. Distinguishable-code capacity is combinatorially enormous
(`C(N_E, k)` draws vs. vocabulary `T ≤ 10⁵`), so capacity is not the binding
constraint — **correlation structure is**. For a P cell with cross-region indegree `f`,
crosstalk per synapse over training scales with `L·(k/N_E)`, i.e. interference is
bounded by the same sparsity knob `k/N_E` that controls drive strength; both families
1–2 let us measure the full `ρ(s,t)` matrix **before training**, which becomes the
preregistered predictor variable in §5.

### 2.3 Region P — processor

- **Population:** `N_P` configured cells, contiguous gids `[N_E, N_E + N_P)`.
  Identical GLIF5 fit, identical sparse signed recurrent fanout (τ=8 ms),
  eligibility τ=80 ms, sparse three-factor local learning (dormant path,
  `lib.rs:931`), first-spike race among output cells via compiled resting response.
- **Input change:** P never sees token identities. Its excitation arrives as spike
  events from E over cross-region edges. Everything downstream — teaching currents
  (`current[target] + output_synapse_gain_pa`), margin competition, race readout,
  structural turnover — is today's machinery unchanged.

### 2.4 Interface contract between regions

Cross-region delivery is **ordinary spike-generated events inside one runtime
instance**: canonical key `(due_step, target_gid, source_gid, edge_slot)`, minimum
delay ≥ 1 step (0.05 ms). Token drive enters through a third tiny gid range of
synthetic input ports above P, keeping event provenance clean with no sentinel values
(Gate-3 `source_gid_semantics`).

Gate-3 guarantees that carry over **unchanged**:

- event total order `(due_step, target_gid, source_gid, edge_slot)`; resident update
  order ascending gid; spike expansion ascending source gid then edge slot;
- duplicate canonical-key rejection before mutation, non-poisoning (status 4);
  returned-spike capacity exhaustion fail-stop poisoning (status 9); poisoned
  follow-up status 5; counters freeze at last completed interval boundary;
- residency: due nonzero events materialize targets immediately before update;
  eviction only when all eight mutable fields are canonical zero; future events do
  not force residency; nonzero tails stay resident until exact underflow;
- floating-point rule: no reassociation, tolerance, or analytic catch-up; signed zero
  canonicalized at every event/state write boundary — cross-region events get this
  automatically because they are normal events;
- exactness: bit-exact byte comparison against the dense oracle below 65,536 cells
  (the oracle simply spans both regions — it is just more gids);
- resource bounds: memory `O(R_capacity + Q_capacity + S_capacity + F +
  D_capacity)`, work never proportional to configured count,
  `configured_ids_scanned == 0`. Regions add reporting tags derived from gid-range
  comparison at accounting time — **no per-cell storage**.

New host-side surface, kept outside the kernel: codebook expansion (token → gid set)
happens Python-side during batch marshalling — consistent with the public seam role
"validate and marshal one batch per advance; no per-cell or per-step simulation."
Projected codes are committed as integer artifacts, so determinism of the hot path is
unaffected.

## 3. (B) Wiring / topology changes and bit-exactness implications

**Removed:** the token→fixed-afferent table — the explicit-identity substrate that the
sealed diagnostics indict.

**Added:**

1. *E drive events* — host-marshalled per advance, canonical keys, source gid drawn
   from the input-port range; targets `C(t)`.
2. *Cross-region edges E→P* — target-owned fixed indegree from E: each P cell draws a
   deterministic seeded sample of its incoming block from `[0, N_E)` (uniform-no-self),
   matching the existing reverse-CSR ownership pattern (`procedural_topology.py`) so P's
   physical edge layout stays recognizable. Degree-conserving rule (human-ei discipline:
   preserve total edges and incoming degree): the cross block replaces an equal slice of
   what would have been P→P fanout. Total edge count is therefore unchanged vs. the
   monolithic layout of equal total cells.
3. *Per-region staged recruitment* — `DevelopmentalGrowthConfig.stage_active_cells`
   becomes two tuples (E stages, P stages); `recruit_to` / `close_stage` /
   `apply_activity_homeostasis` run per region, with homeostasis computed against each
   region's own median activity (E's duty cycle must not be scaled by P's statistics,
   nor vice versa). Orchestration only; every surviving value still lives on a fitted
   cell or sparse synapse.

**Bit-exactness implications:**

- Same-step E spikes targeting shared P cells already receive a total order from the
  event-order keys — no new ordering rule. New exactness cases required:
  shuffled same-step **cross-region** events; boundary gids (last E gid, first P gid);
  recurrent delays 1–3 exercised with mixed intra-P and cross-region sources;
  quiescent-E / quiescent-P continuation; cross-region returned-spike capacity
  exhaustion before step commit.
- The dense oracle needs no new adapter (both regions are ordinary gids below its
  65,536-cell limit at experiment scale).
- Host-side top-k projection floats never enter native state; recompute-and-hash pins
  the codebook artifact. Risk flagged: this is the one genuinely new floating-point
  surface in the pipeline, and it sits outside every Gate-3 contract by design.
- If E later gains internal recurrence or plasticity, eligibility becomes a second
  mutable-field class and checkpoint/identity guards
  (`HUMAN_GLIF5_IDENTITY.config_sha256`) must be versioned. v1 avoids this entirely;
  it is listed as a known future migration, not hidden.

## 4. (C) Cost model toward 1e9 configured cells

All anchors measured, not projected: 60 B/cell mutable floor (44 B GLIF5 state +
8 B recurrent synaptic current + 8 B eligibility); 132 B/cell for floor plus fanout-6
int32-target/f64-weight records; 181.04 B/cell measured on the 1M CPU artifact after
subtracting the dense readout; 185.50 host / 138.16 device B/cell on the 200k
sparse-readout CUDA artifact. Decimal TB (10¹² B).

| Layout | B/cell | 1e7 | 1e8 | 1e9 |
| --- | ---: | ---: | ---: | ---: |
| mutable floor | 60 | 0.60 TB | 6.0 TB | 60 TB |
| floor + fanout-6 local records | 132 | 1.32 TB | 13.2 TB | 132 TB |
| measured CPU layout | 181.04 | 1.81 TB | 18.10 TB | 181.04 TB |
| measured device (sparse readout) | 138.16 | 1.38 TB | 13.82 TB | 138.16 TB |

The dense 257-output readout costs 2,056 B/cell — at 1e9 cells that is
2,056 GB — and is **retired** along with the ridge path for large arms (O(N·V)
storage, O(N²)/O(N³) solve); the sparse readout's `O(VK)` cost is independent of N
and is assumed throughout the table.

**Region deltas vs. monolithic at equal total N:** regions partition one configured
pool (`N = N_E + N_P + ports`), so none of the table multiplies. Additions:

- Codebook: `T·(k+1)` uint64 ≈ 50k tokens × 65 × 8 B ≈ **26 MB flat** at any scale.
- Cross-region edges: zero net when degree-conserving; worst case (added on top,
  f_x = 6, int32 local targets) ≤ 24 B × 6 × N_E ≤ **144 GB at 1e9** — under 0.25% of
  the floor row.
- Second eligibility field: none in v1 (E carries eligibility as today — all GLIF5
  states do — but no E-side synapses to learn on).

**Throughput — assumptions, pending measurement.** Measured frontier curve ends at
1,048,576 configured cells: 843.58 tok/s at 98,304 cells @ 2.51% updated; density
stress at 131,072 cells gives 568.02 / 176.15 / 51.50 / 11.71 tok/s at 1 / 5 / 20 /
100% updated cells; at 1M cells 193.36 tok/s natural but only 1.41 tok/s at 100%.
Everything beyond 1M configured is an **extrapolation assumption**: the active-working-
set model says cost tracks resident set F and updated fraction, both activity-driven
and independent of N_configured, and the resource gate demonstrated constant-memory
behavior at 1e6 and 1e11 configured counts — but no throughput measurement exists above
1M. Assumptions carried into the table: (i) frontier rate at fixed *updated-cell count*
stays roughly flat as N_configured grows (the sparse-work bound supports this; nothing
measured confirms it); (ii) regional split leaves per-token updated counts similar to
monolithic at equal activity; (iii) distributed codes raise E-side participation
(k-of-N drive touches k cells/token regardless of family), so E adds ~k updated cells
per token against P's existing budget — bounded and small at large N_E, but at small
scale it visibly raises the updated fraction and therefore lowers tok/s per the
density curve. The CPU-optimizer benchmark already sequenced behind the sealed run
must measure points at ≥1e7 configured before any row above 1M is treated as more
than assumption.

## 5. (D) Why regions + distributed codes might fix the sealed failure — and why they might not

### 5.1 The mechanism claim

The sealed interpretation reads: *"grown synapses carry task-specific traces rather
than transferable program structure."* The region model gives that sentence a concrete
failure surface and a concrete repair:

- **Compositional codes vs. task-specific traces.** With explicit token identities,
  the eligibility features feeding any readout update are (mostly) disjoint across
  tasks, so every grown trace is task-private *by construction of the input map* — no
  readout rule can share what the features do not overlap. With distributed codes, the
  feature sets of related tasks overlap by `ρ(s,t) = m(s,t)/k` before training starts.
  Three-factor or ridge updates on task A then pre-move the same weights task B needs;
  the transfer the sealed run scored 0/18 becomes a structural consequence rather than
  an emergent hope.
- **Superposition spreads load.** A concept is a pattern over k cells; the same cell
  participates in ~L = T·k/N_E codes. Readout margins stop being per-task absolute
  levels (the +726 pA median pathology) because no single task owns enough substrate to
  saturate; magnitude coding gives way to pattern coding, which the homeostatic
  normalization proposal (fusion survey #3) then protects.
- **Regions separate timescales of failure.** E-codebook quality (representation) and
  P-learning (processing) become independently measurable: capacity gates exercise P
  given fixed E; calibration gates exercise the E-induced overlap structure. The a013
  ladder's "too small to fit" vs. "fits but does not generalize" distinction becomes
  attributable to a component.

### 5.2 Honest counterarguments

1. **Random overlap may just add noise.** Family-1 overlap is chance-level; a shared
   random unit carries no semantic content, so P's shared traces could be pure crosstalk
   — degrading capacity (interference) without buying transfer. The sealed run's
   capacity shortfall (40/42) could plausibly *worsen* under distributed drive, since
   every token now activates k cells whose combined current is less stereotyped than
   one private subset. If Family 1 fails calibration while Family 2 passes, the
   mechanism is similarity structure, not distributedness; if both fail, regions are
   refuted as the fix at this scale.
2. **Two small regions lose the wide single-reservoir mixing.** A monolithic reservoir
   lets every cell talk to every cell; splitting into E and P with a thin cross block
   removes E↔P feedback and shrinks the recurrent mixing volume to N_P. Recurrent
   depth (multi-step mixing through τ=8 ms fanout) may be exactly what composed
   problems need; the region split trades that away for input-structure hygiene.
   The merged-reservoir arm in §6 exists to measure precisely this.
3. **E-codebook quality becomes a new single point of failure.** The codebook is
   frozen, host-side, and outside every biological-learning mechanism: if the embedding
   is bad, similarity becomes *wrong* physical overlap, and nothing downstream can
   repair it — worse, the failure is invisible to all Gate-3-style contracts because
   the codebook is not part of the runtime. Family 3 (experience-formed) is the escape
   hatch but reintroduces the collapse-to-grandmother risk and a new checkpoint/
   identity-versioning burden.
4. **The readout may still learn task-private margins.** Distributed inputs lower the
   barrier to sharing but do not force it; a margin-driven rule can still satisfy
   training by exploiting the non-overlapping (1−ρ) fraction of features. This is why
   the region change is proposed *together with* the fusion survey's RPE-gated
   three-factor rule, and why the experiment below keeps the readout rule as a
   controlled factor rather than assuming transfer follows from overlap alone.
5. **No consciousness/cognition claim is implied.** This is a representation-
   engineering proposal evaluated by replay and transfer counts on a tiny curriculum;
   nothing here speaks to phenomenal claims of any kind.

## 6. (E) Smallest decisive experiment — preregisterable protocol skeleton

**Question:** does replacing explicit token identities with distributed E-codes fix
calibration transfer at matched capacity, or does it merely add noise?

**Design: two-region vs. one-region, same total cells, same teaching schedule.**

- **Total substrate fixed:** `N_total = 512` (one rung of the a013 N-ladder;
  128-cell replication as pilot). Two-region arm: `N_E = 256`, `N_P = 256` (plus
  output cells as today). Merged arm: single 512-cell reservoir driven by the same
  codebook families directly (token → k-of-512), i.e. distributed input without
  regionalization. This isolates "distributed codes" from "two regions."
- **Arms (codebook × topology), all with identical total cells, total edges,
  incoming degree, injection constants, token schedule, seeds:**
  1. explicit identities (sealed A012 wiring) — replication control;
  2. two regions, Family-1 random distributed codes;
  3. two regions, Family-2 similarity-projected top-k codes;
  4. merged reservoir, Family-2 codes (region ablation);
  5. optional Family-3 experience-formed codes, only after 1–4 are frozen and run.
- **Frozen before any run:** tokenizer, curriculum, 3-epoch grow schedule and
  supervision scope (A0/A1/A2 lanes unchanged from the a012 contract), readout rule
  (sparse three-factor, dormant path, fixed lr/margin), codebook artifacts + hashes,
  seed list, source commit, stopping rule. The `ρ(s,t)` matrix for arms 3–4 is
  computed and committed *before training* as the preregistered predictor.
- **Gates — split capacity vs. calibration exactly as the a013 ladder does:**
  - *Capacity gate:* TRAIN replay 42/42 tasks and 126/126 cases. A region arm that
    fails here is not interpretable on transfer; report and stop (matches the sealed
    run's own gate semantics).
  - *Calibration gate:* replay 18/18, cases ≥ preregistered threshold above the
    explicit-identity arm's 7/54, decided by paired-seed quantiles (two-step
    threshold-freezing discipline from the human-ei doc: lock baseline quantiles
    first, then freeze thresholds before candidates run).
  - *Mechanism gate (diagnostic, non-veto):* correlation between per-pair calibration
    success and preregistered ρ(s,t); state-separation (1−cosine) between task
    strata, testing whether the ~0.001 null moves; participation entropy per E
    assembly (grandmother-collapse detector for arm 5).
- **Outcomes and what each would mean:**
  - arms 2–3 pass capacity + calibration where arm 1 fails → distributed
    representation is sufficient at this scale; proceed up the N-ladder;
  - arm 3 passes, arm 2 fails → similarity structure, not mere overlap, is the
    active ingredient;
  - arm 4 ≈ arm 3 → regions add nothing beyond distributed codes at this scale
    (counterargument 2 supported); regionalization must earn its complexity at
    larger N or be dropped;
  - arms 2–3 fail capacity where arm 1 passed → distributed drive costs capacity;
    revisit k, drive gain, or homeostasis before any transfer claim.
- **Preregistration discipline:** one paired-seed family per arm; every seed, failed
  condition, and diagnostic published; no gate threshold chosen after seeing
  candidate results; cohort identity consumed once per the a012 disposition rules;
  no numeric consciousness claims; results reported matter-of-factly against the
  frozen gates regardless of direction.

## 7. Summary

- **Did:** specified the two-region model — E as the token→k-of-N overlapping-sparse-
  assembly distribution mechanism with codebook design treated first-class (random /
  projected / Hebbian families, interference arithmetic, transfer coefficient ρ), P as
  today's eligibility+race processor unchanged; interface contract showing which
  Gate-3 determinism/residency/exactness/resource guarantees carry over verbatim and
  which new cases are needed.
- **Found:** the sealed failure is reproducible as a property of the input map, not
  only the learning rule; distributed codes convert calibration transfer from an
  emergent hope into a structural consequence of pre-training feature overlap, at
  ~zero memory cost (26 MB codebook, degree-conserving edges) against measured
  per-cell anchors of 60/132/181.04 B/cell toward 60–181 TB at 1e9 configured cells;
  all throughput beyond 1M configured cells is flagged as assumption pending the
  sequenced CPU-optimizer benchmark.
- **Risks recorded honestly:** random overlap may be pure noise; regions forfeit
  single-reservoir mixing; the codebook is a new frozen single point of failure; the
  readout can still learn task-private margins. The decisive experiment (§6) is
  sized to refute or support the mechanism in one preregistered rung.
- **Created/modified:** this file only
  (`docs/research/a013-regioned-architecture.md`); committed on
  `codex/a013-cognition-design`.
