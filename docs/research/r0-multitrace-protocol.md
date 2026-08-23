# R0 preregistered protocol — multi-timescale eligibility traces (the keystone)

Status: PREREGISTERED DESIGN — gates frozen BEFORE implementation. No arm of
this protocol has been executed. Written against worktree
`human-brain-a013-cognition-design` @ `c5b9369`; every code citation was
verified against this tree. Tasking: `roadmap-reading-to-coding.md` Phase R0
(:111-120) and immediate-queue item 2 (:166). Exclusive file: this doc.

## Question

Does replacing the single eligibility trace (τ = 80 ms ≈ 16-token window,
recurrent.py:100) with a bank of N_TRACE multi-timescale traces concatenated
into the readout feature vector de-collapse the readout's view of state?
Measured root cause being attacked: post-framing eligibility cosine 0.9989
across tasks (roadmap :16) — every prompt collapses to ONE point in feature
space, so ridge emits the unigram prior. Corroborating measurement at the
gate's exact arm family: ei_sweep f=0.0 N=1024 s7 eligibility channel median
1−cosine ≈ 0.00056 (experiments/ei_sweep/FINDINGS.md), i.e. cosine ≈ 0.99944.

## 1. Mechanism (exact design)

### 1.1 Trace bank and taus

N_TRACE = 8 traces with time constants

```
ELIGIBILITY_TAUS_MS = tuple(np.geomspace(80.0, 4000.0, 8))
                   = (80.0, 139.894, 244.630, 427.779,
                      748.049, 1308.097, 2287.441, 4000.0)
```

Justification (frozen arithmetic, dt-free): at token_interval_ms = 5
(recurrent.py:95) an exponentially decaying trace's characteristic memory
window is τ/T tokens. The bank spans 80 ms → 4 s, i.e. windows of
16 → 800 tokens:

| k | τ_k (ms) | window (tokens) | per-token decay exp(−T/τ_k) |
| --- | --- | --- | --- |
| 0 | 80.000 | 16.0 | 0.939413 |
| 1 | 139.894 | 27.98 | 0.964890 |
| 2 | 244.630 | 48.93 | 0.979768 |
| 3 | 427.779 | 85.56 | 0.988380 |
| 4 | 748.049 | 149.61 | 0.993338 |
| 5 | 1308.097 | 261.62 | 0.996185 |
| 6 | 2287.441 | 457.49 | 0.997817 |
| 7 | 4000.000 | 800.00 | 0.998751 |

- Geometric spacing (ratio 50^(1/7) ≈ 1.749) gives log-uniform coverage so no
  decade of context length is unrepresented while adjacent traces stay
  correlated enough for ridge stability.
- τ_0 = 80 ms is deliberately the current default: trace 0 IS today's trace;
  a single-tau config reproduces current behavior exactly (see §1.5).
- 800 tokens covers a full QA framing (context verses + question) plus margin;
  16 tokens matches the shortest program-transition dependencies already
  served by the incumbent trace.
- Sub-step maintenance stays at the registered 0.05 ms step (cells.py:32,
  100 steps/token): per-step decay λ_k = exp(−dt/τ_k) (trace 0: 0.999375,
  matching local-rule-constant-derivation.md:90).

### 1.2 Where the traces live (recurrent.py)

- `RecurrentCircuitConfig.eligibility_tau_ms: float = 80.0` (:100) is replaced
  by `eligibility_taus_ms: tuple[float, ...] = ELIGIBILITY_TAUS_MS`.
  Validation (`__post_init__`): finite, positive, strictly increasing, ≥ 1
  trace. A deprecated `eligibility_tau_ms=` keyword shim constructs the
  single-element tuple `(value,)` so existing callers and v3 checkpoint
  metadata keep constructing valid configs.
- Circuit state: `self._eligibility = np.zeros(reservoir_cells)` (:221)
  becomes one contiguous f64 buffer
  `self._eligibility_traces = np.zeros(N_TRACE * reservoir_cells)` viewed as
  `(N_TRACE, reservoir_cells)` C-order. Row k is trace k. A flat view
  `self._feature_traces = self._eligibility_traces.reshape(-1)` is THE
  feature-space addressable by readouts (source id s ∈ [0, N_TRACE·R) decodes
  as `k, cell = divmod(s, R)`).
- Per-trace step decays: `self._eligibility_decays = np.exp(-dt_ms / taus)`
  (replaces the scalar at :293-295); the scalar passed into native configs
  becomes a K-length vector (§2).
- Back-compat views: `.eligibility` property (:634) returns trace 0 copy;
  `reservoir_activity_density` (:619) keeps computing on trace 0 so the
  published activity-density accounting does not silently change meaning.
- `reset_sequence()` (:645-650) zeroes ALL K rows (one `fill(0.0)` on the
  flat buffer) before the native resync — unchanged call count.
- Feature dimension: `feature_count = reservoir_cells + 1` (:231) becomes
  `N_TRACE * reservoir_cells + 1`; `_growth_feature_count` inherits this, so
  ridge storage shapes (`_normal_matrix` (F,F), `_target_moments`
  (token_cells,F), `_feature_batch` (128,F)) resize through the existing
  allocation path (:746-762) with zero new plumbing.

### 1.3 How observe/strengthen consume them

- observe(): unchanged call shape. The native kernel advances every trace:
  per sub-step, row-wise `e_k ← e_k · λ_k` for all cells (decay fused exactly
  where the single trace decays today, lib.rs:200-201 comment contract), then
  `e_k[cell] += 1.0` for every ACTIVE spiking cell — once per spike event,
  into every row (matches the incumbent once-per-active-cell increment,
  lib.rs:644 serial / :793 AVX / cuda.cu:157-158 device form). Ordering
  contract preserved per row: compute spikes first, then decay-all, then add
  this step's spikes. No cross-trace interaction exists anywhere.
- strengthen(), ridge path (:735-742):
  ```
  feature[:-1] = self._feature_traces      # flat K·R concatenation
  feature[-1]  = 1.0                       # pacemaker bias slot, still last
  ```
  Everything downstream (batched outer products, ridge solve with the
  bias-friendly 0.99 diagonal relief at :788, consolidation into
  `readout_synapses_pa`/`pacemaker_synapses_pa`) is dimension-generic already.
- Inference current (:820-831): `self._readout_synapses_pa @ self._eligibility`
  becomes `@ self._feature_traces` (weights are (token_cells, K·R)).
- three_factor dense path: `NativeThreeFactorLearner` binds the flat buffer;
  `readout_synapses_pa` becomes (token_cells, K·R). No rule change — the
  normalized Hebbian step consumes whatever feature vector it is handed.
- sparse_three_factor path: `SparseCellReadout` potential/contact sources now
  address the K·R feature space; canonical candidate count becomes
  `min(K·R, CANDIDATE_FACTOR * fan_in)` (validation at readout.py:193 updated
  to match). The native current/update calls (lib.rs:912, :931+) already take
  `(cell_count, eligibility*)` — passing the flat buffer with
  cell_count = K·R requires NO consumer-signature change. The apposition
  trace's per-token decay (currently scalar exp(−T/τ), recurrent.py:705-707)
  becomes per-contact: `exp(−T/τ_k)` with k = source // R, applied as a
  per-slot vector inside `SparseCellReadout.teach`.

### 1.4 What deliberately does NOT change

Tokenization, framing, curriculum generators, supervision scope, race
dynamics, GLIF5 cell integration, recurrent topology/signs, developmental
controller, QA prompt framing, presentation schedules. The experiment varies
ONE axis: the temporal basis of the readout features. Any drift in the other
axes invalidates the run (detectable via the curriculum-digest anchor, §3).

### 1.5 Checkpoint layout v5 migration

Current formats: model artifact `recurrent-human-cell-checkpoint-v3`
(recurrent_artifacts.py:16) wrapping the neutral envelope
`recurrent-cell-synapse-envelope-v1` (recurrent_state.py:13, container
unchanged). Eligibility/traces are transient state and remain UNPERSISTED
(`_BASE_STATE_ARRAYS`, recurrent.py:24-30, never included them); what forces
the layout bump is feature-space-dependent PERSISTED state.

- `FORMAT` bumps to `"recurrent-human-cell-checkpoint-v5"`. Writers always
  write v5. Metadata additionally carries `"eligibility_taus_ms"` (list) and
  `"feature_cell_count": K·R` as explicit guards alongside the asdict'd
  circuit_config.
- v5 loading (strict): taus in metadata must equal the config's; every
  feature-space array shape checked against F = K·R + 1 (ridge statistics)
  or K·R widths (dense weights); mismatch raises (existing shape-guard
  pattern, recurrent.py:462-477).
- v3 loading (legacy, exact-by-construction): metadata's
  `circuit_config.eligibility_tau_ms` maps to a single-tau config (K=1).
  With K=1 the feature space IS R, so every v3 payload — including sparse
  layouts, whose candidate formula degenerates to the v3 canonical form —
  restores bit-identically. Zero semantic migration needed; the version bump
  exists to make mixed-era payloads loudly distinguishable, not to convert.
- Ridge statistics under a K>1 RESTART from a v3 checkpoint are NOT migrated:
  the normal matrix embeds single-trace geometry and cannot synthesize
  multi-trace statistics post hoc without corpus replay. Declared honestly:
  continue-training after migration starts blind on traces 1..K−1 (algebraic
  zero-padding embedding would be exact but we do not ship it); retraining is
  the sanctioned path. This is a disposition, not a silent fallback.

## 2. Native mirror requirements (bit-exactness contract)

How eligibility is mirrored TODAY (the contract any change must preserve):

- CPU/Rust: host numpy arrays ARE the live kernel state; synchronize calls
  are documented no-ops (native.py:1240-1247). The serial kernel decays
  eligibility inside the fused cell pass (lib.rs:200-201) and increments
  active spikers afterward (lib.rs:644); the AVX path does the same with the
  decay constant broadcast per lane (lib.rs:239-240, 252-253, 260, 789-793).
- CUDA: device-owned cooperative kernel recomputes each step via
  `eligibility = eligibility * decay + spike_mask` in f64 (cuda.cu:157-158);
  after every token launch the host-mirror arrays are downloaded
  (cuda.py:645-647). Mirror mode is chosen by readout rule at binding
  (recurrent.py:315-319): `"eligibility"` (sparse_three_factor) downloads
  only the trace array; otherwise `"full"` (60 B/cell/token public-state
  contract, docs/simulation-speed.md). Mirror membership and byte accounting
  live in cuda.py:473-475 / :554-569. Bit-exactness is real, not aspirational:
  NVRTC compiles with FMA disabled and precise division (simulation-speed.md),
  and a 64-token test asserts every mutable array bit-for-bit vs Rust
  (test_cuda.py:168, `assert_array_equal` on `_eligibility` at :47/:228/:248).

Multi-trace requirements (all mandatory before the gate run):

1. `RecurrentConfig` grows an f64 decay-vector parameter of length K
   (ctypes struct field + Rust struct + CUDA `__constant`-style struct
   member). Scalar `eligibility_decay` fields survive ONLY in single-tau
   builds.
2. Every kernel (Rust serial, Rust AVX, CUDA) maintains all K rows in FIXED
   index order 0..K−1 with identical per-row operation order
   (spike-determine → decay → increment). No trace may shortcut (e.g. no
   recurrence e_{k} derived from e_{k−1}).
3. The increment applies to every row for each active spiking cell, once per
   spike event (never per fanout edge) — same event set as lib.rs:644 today.
4. CUDA host mirrors deliver all K rows contiguously; the eligibility mirror
   payload becomes K·8 B/cell/token; `device_to_host_bytes_per_token`
   (:473-475) and the full-contract figure in simulation-speed.md are updated
   to state K explicitly rather than silently doubling traffic.
5. Parity tests extend to every row: 64-token CPU-vs-CUDA equality asserted
   per trace row, plus reset synchronization (zeroes visible on device after
   `reset_sequence`) — extension of tests/test_cuda.py:168 and
   tests/test_simulation_equivalence.py:227/:273, not new harnesses.
6. Quiescent/frontier paths (lib.rs:568 `eligibility[cell] != 0.0` style
   activity checks) consider a cell active if ANY row is nonzero; a cell
   quiet in row 0 but not row k must not be dropped.

## 3. THE GATE (preregistered, mechanical)

Single confirmatory run; nothing may be tuned, re-seeded, or added after the
first token is observed. Diagnostics never gate (house rule).

### 3.1 Frozen run configuration

Ladder arm, byte-identical settings to the recorded single-trace anchors
(experiments/readout_compare/ridge_N1024_s7.json,
experiments/ei_sweep/f0.000_N1024_s7.json):

- `n_cells=1024`, `seed=7`, `schedule=massed`, `data_scale=42`,
  `supervision=all_transition`, `readout_rule=ridge`, `codebook=explicit`,
  A2 lane framing, 3 corpus passes, input_fanout=16, recurrent_fanout=8,
  harness defaults otherwise — plus the R0 change: the 8-tau bank (§1.1).
- Backend pinned `HUMANBRAIN_DEVICE=cpu` (exact reference backend);
  `--check-determinism` enforced (byte-identical duplicate record modulo
  wall_s). Optional CUDA replication is non-gating evidence only.
- Freeze anchor: `curriculum_sha256` must equal
  `48219eda110f09f735525a7de00e56596ffac729746654e809c3c5eb525ccae6`
  (recorded by BOTH prior arms at this seed/config). If it differs, STOP —
  generators drifted and the cohort is not spent yet; fix provenance first.
- git_head of the implementing branch recorded in the run record; all §4 red
  tests green on that exact head (preflight, verified before execution).

### 3.2 Gates

State vectors for G1: after growth + freeze, replay each of the 18
CALIBRATION tasks' framing observations (`reset_sequence()` then `observe`
over `typed_program._model_input(...)`), snapshot the FULL feature state the
first decision race sees — the flat `[e_0; …; e_7]` buffer (bias excluded;
identical exclusion to the 0.9989 baseline measurement) — and compute
pairwise cosine over task pairs with DIFFERENT first teacher token within the
same stratum (Int / List[Int]), exactly the `collect_state_separation`
definition (benchmarks/ladder_experiments.py:666-731) pointed at calibration
tasks per roadmap :116-117.

- **G1 (separation, primary)**: pooled MEDIAN cosine across those pairs
  < 0.90. Baseline being beaten: 0.9989 (roadmap :16); corroborated ≈ 0.99944
  at this arm family (ei_sweep FINDINGS, f=0.0 eligibility channel). Gate
  boundary expressed identically as median 1−cosine > 0.10 — a ≥ ~90× move
  from the measured null.
- **G2 (answers, primary)**: on the bundled QA fixture corpus
  (qa_corpus.py FIXTURE_CORPUS, 30 KJV verse-context triples), run
  `benchmarks/qa_experiments.run(ExperimentConfig(seed=7))` at committed
  defaults (reservoir 96, growth_epochs=2, CALIBRATION cohort held out);
  free-run completions on held-out tasks yield
  `distinct_answers >= 3` (GUARD_DISTINCT_MIN, qa_experiments.py:45).
  modal_share is recorded diagnostically alongside (guard pair exists at
  :146-153; only the distinct-answer half gates here).
- **G3 (capacity non-regression, secondary)**: `train_replay_tasks >= 39`
  of 42. Baseline: seed-7 single-trace ridge = 41/42 = 0.9762
  (readout_compare record); 5 percentage points = 0.05 → floor 0.9262;
  39/42 = 0.9286 passes, 38/42 = 0.9048 fails. train_cases_passed (/126) and
  calib_cases_passed (/54) recorded diagnostically.

### 3.3 Verdict semantics (fixed now)

- **PASS** ⇔ G1 ∧ G2 ∧ G3 in the single run. Proceed to R2 sizing with
  multi-trace features as the shipped readout basis.
- **PRIMARY FAILURE** (¬G1 ∨ ¬G2) on an honestly implemented stack (red tests
  green, parity proven, curriculum anchor matched) ⇒ **the
  eligibility-feature family is FALSIFIED** as the readout basis at this
  operating point. See §6. This is a publishable result; it spends the
  protocol permanently.
- **SECONDARY-ONLY FAILURE** (G1 ∧ G2 ∧ ¬G3): features separate and answers
  de-degenerate but capacity regressed — the family is NOT falsified; the
  disposition is "capacity-regressive", the result is banked, and ANY follow-up
  (trace-count reduction, tau re-span, regularization change) requires a NEW
  protocol v2 with a fresh cohort. No repair pass under this document.
- One shot means one shot: no alternate seeds, no arm additions, no post-hoc
  threshold edits, no rerunning "to check" (preregistered-experiments
  discipline).

## 4. Red tests (must FAIL before implementation, PASS after)

New file `tests/test_multitrace.py` unless noted; each names its failure
condition mechanically.

- **R0-T1 tau-bank determinism**: default config yields exactly 8 traces with
  values bitwise-equal to `tuple(np.geomspace(80.0, 4000.0, 8))`;
  independent of `random_seed`; `eligibility_tau_ms=80.0` shim yields the
  1-tuple; duplicate, non-monotone, non-positive, or non-finite tau lists are
  rejected with ValueError.
- **R0-T2 per-trace decay vs hand-computed**: drive a scripted spike train
  (including refractory-suppressed steps and a quiescent tail ≥ 5·τ_7) and
  compare every row against a NumPy reference applying EXACTLY
  `e ← e·λ_k` then `+= spike` per 0.05 ms step — bit-exact (same op order),
  not allclose. Includes: a spike appearing in row k must appear in ALL rows;
  refractory cells contribute 0 to every row.
- **R0-T3 token-window sanity**: after one pulse followed by silence,
  per-token amplitude ratios between consecutive observes match
  exp(−T/τ_k) to 1e-12 relative (T = 5 ms; row 0 = 0.939413…).
- **R0-T4 checkpoint round-trip v5**: grow → freeze → save → load →
  `cell_synapse_state_sha256` identical; predictions on a fixed context
  byte-identical pre/post round-trip. v5 metadata rejects tau-list mismatch
  and wrong `feature_cell_count`; ridge statistics restored at F = K·R + 1.
- **R0-T5 v3 legacy load**: a v3-format fixture (single-tau) loads under the
  K=1 interpretation with bit-identical `output_current_pa` and race winners
  vs pre-upgrade behavior on the same payload; v3 payload loaded while
  requesting K>1 taus raises loudly (never silently reshapes).
- **R0-T6 bit-exact CUDA parity** (extends tests/test_cuda.py:168): 64-token
  loop, every trace row `assert_array_equal` Rust vs CUDA; reset zeroes all
  rows on device; repeated-token persistence per row; mirror download
  contains K·R values with updated `device_to_host_bytes_per_token` ==
  8·K·cells + 8 (eligibility mode).
- **R0-T7 kernel equivalence across backends** (extends
  tests/test_simulation_equivalence.py:227/:273): Python-reference, Rust
  serial, Rust AVX, CUDA produce identical all-row trace matrices after the
  64-token sequence at two activity densities; quiescent-frontier handling
  treats any-nonzero-row cells as active (§2 requirement 6).
- **R0-T8 feature-vector shape guards**: ridge `_feature_batch`,
  `_normal_matrix`, `_target_moments` shaped (128, F), (F, F),
  (token_cells, F) with F = K·R + 1; `feature[-1] == 1.0` and
  `feature[:-1]` bitwise-equals the flat trace buffer at strengthen time;
  restore-time shape mismatch raises with the offending key named
  (existing message contract, recurrent.py:475-477).
- **R0-T9 sparse source-space validity**: potential/contact sources sampled,
  unique, ascending over [0, K·R); canonical candidate count ==
  min(K·R, CANDIDATE_FACTOR·fan_in); per-contact apposition decay equals
  exp(−T/τ_(source//R)); native sparse current via flat buffer matches a
  NumPy reference over the same sources.
- **R0-T10 accounting invariance**: for fixed seed and curriculum, token
  counts, presentation order, and observed-token counters are identical
  between single-tau and multi-tau runs; ONLY declared fields differ
  (ladder red-test accounting-block pattern, a013-teaching-processes.md:177);
  `reservoir_activity_density` remains a trace-0 statistic.

## 5. Cost estimate (arithmetic on measured baselines; estimates flagged)

Measured anchors: 1M-cell CUDA text training 22.7 tok/s grow / 22.1 read,
VRAM 2.7/12 GB (roadmap :11, lm_million_smoke sparse_three_factor →
eligibility-only mirror); DtoH contract 60 B/cell/token full /
8 B/cell eligibility-only (simulation-speed.md); ladder ridge N=1024 s7 arm
wall_s = 16.99 (readout_compare record); host ridge budget 20 GiB binding the
dense-ridge ceiling at ≈ 50k features (`_HOST_RIDGE_BUDGET_BYTES`,
benchmarks/ladder_experiments.py:130).

Per ADDED trace (K−1 of these), at steady state:

| Quantity | Scaling | 128k cells | 1M cells |
| --- | --- | --- | --- |
| Device/state memory (f64/cell) | +8 B | +1.0 MB | +8.0 MB |
| DtoH mirror traffic per token (eligibility mode) | +8 B/cell | +1.0 MB | +8.0 MB |
| Kernel state-pass DRAM traffic per step | +16 B/cell (r+w) vs ≈61 B base | +26 %/trace of state pass | +26 %/trace |
| Kernel compute per step | +1 FMA/cell vs ≈50 flops | ≲ +4 %/trace | ≲ +4 %/trace |

K = 8 totals (vs single-trace baseline):

- **CUDA 1M**: VRAM +56 MB (2.7 → ~2.76 GB of 12 — negligible). DtoH
  8 → 64 MB/token; at the baseline 44 ms/token budget this is the dominant
  term: serialized-copy arithmetic bounds throughput near
  1000/(44 + ~5.6) ≈ 20 tok/s, with overlap potentially hiding more —
  ESTIMATE band 15–22 tok/s (×1.05–1.5 wall clock). Must be re-measured with
  the registered cuda_scaling protocol extended to record K; kernel-side
  traffic/compute deltas predict ≤ ~×1.3 kernel time.
- **CPU ladder/gate path (N=1024)**: the fused decay pass adds 7×16 B/cell/
  step of traffic (state-pass bound ×≈2.8 worst case) but the arm is dominated
  by ridge algebra: normal matrix 8.4 → 537 MB host RAM, per-flush flops
  ×64 (2.7e8 → 1.7e10 per 128-batch), solve flops ×512 (7.2e8 → 3.7e11,
  seconds-to-minutes under OpenBLAS threads). ESTIMATE: gate run minutes,
  not 17 s — acceptable one-time; flagged as estimate, measured value goes in
  the record.
- **Dense-ridge feasibility ceiling**: F_max ≈ 50k features at 20 GiB ⇒
  max cells ≈ 50k/K: 50k (K=1) → 12.5k (K=4) → 6.25k (K=8). Dense ridge was
  never the ≥100k path; the sparse/CUDA path is structurally unaffected
  (fixed fan_in; source space grows, contact count does not).
- **Python strengthen path**: per-example feature build O(K·R) memcpy —
  ×8 at fixed R; irrelevant at ladder N, subsumed at scale by batched BLAS.
- Cost is NOT part of the pass gate. A measured > 2× end-to-end slowdown at
  1M triggers a scaling review before R2 sizing (recorded as engineering
  disposition), never a silent acceptance or a gate edit.

## 6. Falsifier

If the gate ends in PRIMARY FAILURE after honest implementation (all §4 red
tests green, backend parity proven, curriculum anchor matched, single
unmanipulated run), the eligibility-feature family — multi-timescale traces
as the readout's temporal basis — is FALSIFIED at this operating point, and
the readout must move to DIFFERENT STATE FEATURES entirely. Per the tasking,
this protocol decides nothing about successors; it only names the candidates
for the next protocol:

1. **Spike counts per window** — per-cell binned spike counts at several bin
   sizes (count-based temporal features, no exponential kernel).
2. **Membrane-potential quantiles** — per-cell running quantiles/summary
   statistics of voltage over the token interval or sliding windows.
3. **Assembly-overlap counts** — overlaps between currently co-active cell
   sets and previously recruited assemblies.

Distinction fixed NOW so it cannot be negotiated later: if G1 passes
(features separate) but G2 fails (answers stay degenerate), that is NOT an
eligibility-family falsification — it localizes the blocker downstream to the
readout/race layer (margins exist but never re-enter computation; HOT-2
seed, roadmap :119-120) and routes there. Only separation-or-answers failure
of the feature state itself kills the family. Either way the cohort is spent
once and the outcome stands.

## Honesty notes

- The 0.9989 headline number is orchestrator-pinned (roadmap :16); this
  tree's nearest direct measurement at the same arm family is 1−cos =
  0.00056 (cosine 0.99944, ei_sweep f=0.0). Both are far above the 0.90
  gate boundary, so the gap does not move the verdict.
- All §5 figures marked ESTIMATE are arithmetic on measured anchors, not
  measurements; the record will carry measured wall-clock, VRAM, and DtoH
  bytes.
- The QA leg runs at fixture scale (96-cell circuit, 30-triple corpus) — it
  evidences answer de-degeneration, not reading skill; R2 owns the real
  corpus question.
