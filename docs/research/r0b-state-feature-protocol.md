# R0-b preregistered protocol — successor state-feature families (spike-count windows, voltage quantiles, assembly overlaps) and the combination arm

Status: PREREGISTERED DESIGN — gates frozen BEFORE implementation. No arm of
this protocol has been executed. Written against worktree
`human-brain-a013-cognition-design` @ `dcc7d17`; every code citation was
verified against this tree. Tasking: the falsifier clause of
`docs/research/r0-multitrace-protocol.md` §6 (:388-394), which named exactly
these three successor families when the eligibility-feature family was
falsified. Exclusive file: this doc (plus its future red-test file and gate
runner, named below).

## Question

Does replacing the readout's temporal basis — the falsified multi-timescale
eligibility-trace bank — with NON-exponential state features (sliding
spike-count windows, membrane-potential distribution quantiles, or
assembly-overlap counts) de-collapse the readout's view of state?

Measured root cause being attacked: post-framing feature cosine across tasks
≈ 0.99993 under the 8-trace bank (R0 gate: pooled median 1−cosine
6.7713e-05 vs required > 0.10; `experiments/r0_multitrace/gate_results.json`,
committed `dcc7d17`). The incumbent basis accumulates every token with slow
exponential decay, so the long SHARED framing prefix dominates the feature
norm and every prompt collapses to one point. The three families below share
one mechanistic bet the traces did not have: **hard-edged sliding windows
FORGET** — a 16-token window contains only the last 16 tokens, so the shared
prefix physically exits the feature, and whatever differs between tasks is
what remains.

## 0. What R0 established and what carries over unchanged

The R0 one-shot (`docs/research/r0-multitrace-protocol.md @ 771bd93`, run at
git head `614856a`) spent its cohort and returned PRIMARY FAILURE:

| Gate | Result | Required | Verdict |
| --- | --- | --- | --- |
| G1 separation | median 1−cos pooled **6.7713e-05** (Int 6.9421e-05, 44 pairs; List[Int] 5.5643e-05, 4 pairs) | > 0.10 | FAIL |
| G2 answers | distinct_answers **6**, modal_share 0.167 | ≥ 3 | PASS |
| G3 capacity | train_replay **42/42** (126/126 cases) | ≥ 39/42 | PASS |

Diagnostics: wall_s 48.968 at N=1024, ridge backend scipy-openblas-dsyrk,
curriculum anchor matched. Note the inversion against R0 §6's contingency:
features did NOT separate yet answers were NOT degenerate — G2's six distinct
answers at a 0.196 span accuracy come from a diverse-but-wrong prior, not from
state discrimination. This does not soften G1: the mechanical definition of
de-collapse is separation, and it failed by ~1500×.

Everything R0 built is REUSED by this protocol; nothing downstream of the
feature vector changes meaning:

- **Trace-bank machinery** (recurrent.py): `(trace_count, reservoir_cells)`
  contiguous f64 buffer `self._eligibility_traces` (:271-274), flat view
  `_feature_traces`, per-row sub-step decays `_eligibility_decays`
  (:349-352), config validation (:151-171).
- **v5 checkpoints** (`recurrent-human-cell-checkpoint-v5`, recurrent.py
  :96-99) with `feature_cell_count` metadata guards (:463, :676-694).
- **Native mirrors**: CPU/Rust host-live arrays with `spike_mask` exposed
  per call (native.py :1126, :1310); CUDA mirror modes `"full"` /
  `"eligibility"` with per-token DtoH accounting (cuda.py :371-372,
  :599-602, :505-508).
- **scipy ridge backend** (`install_scipy_ridge`, ladder_experiments.py
  :600-667; dsyrk flush + Cholesky consolidate).
- **Gate tooling**: `tools/r0_gate.py` (anchor preflight :42-58, arm
  invocation :63-94, G1/G2/G3 derivation :102-135) — this protocol ships a
  sibling runner `tools/r0b_gate.py` with the SAME structure.
- **Harness knobs**: `--codebook {explicit,random,similarity}`
  (ladder_experiments.py :1157, CODEBOOKS :118) with frozen similarity
  projection (`_similarity_assembly_cells` :223-269) and deterministic
  random arm (:289-296); `collect_state_separation` (:670-737).

The successor families REPLACE ONLY what feeds the feature vector. Tokenization,
framing, curriculum, schedules, race dynamics, cell integration, topology,
readout algebra, QA leg, and the shared gate (§4) are byte-identical to R0's.

## 1. Shared skeleton: the window bank replaces the trace bank

All three families share one container so that everything downstream of the
feature vector stays dimension-generic exactly as in R0 §1.3.

### 1.1 Window bank

`N_WIN = 8` sliding windows with hard edges, in TOKENS (not ms):

```
WINDOW_TOKENS = tuple(int(round(w)) for w in np.geomspace(16.0, 800.0, 8))
              = (16, 28, 49, 86, 150, 262, 457, 800)
```

- Same geomspace coverage as the falsified tau bank (16..800 tokens at
  `token_interval_ms = 5`, recurrent.py:95) — log-uniform, no uncovered
  decade of context length, adjacent windows correlated enough for ridge
  stability.
- Deliberate contrast with traces: a trace weights token u at time t by
  `exp(−(t−u)·T/τ_k)` — never zero, so the shared framing prefix always
  dominates the norm. A count/quantile/overlap window is ZERO outside its
  edge: token u contributes iff `t − u < W_k`. W_0 = 16 matches the shortest
  program-transition dependencies; W_7 = 800 covers a full QA framing plus
  margin.
- Warm-up convention (all families): before any observation all rows are
  zero; after t < W_k tokens, window k summarizes the available t samples
  (truncated window). No synthetic padding.
- Validation (`__post_init__`, mirroring the tau checks recurrent.py
  :151-171): positive, strictly increasing, ≥ 1 window, each ≤ 100_000.

### 1.2 Config surface and state locations (recurrent.py)

- `RecurrentCircuitConfig` gains `feature_family: str = "traces"`.
  `FEATURE_FAMILIES = ("traces", "spike_counts", "voltage_quantiles",
  "assembly_overlap")`; combination values use the grammar
  `"combo:<f1>+<f2>"` with f1 ≠ f2, both single names in canonical order
  (A < B < C below), validated against a strict regex. Default `"traces"`
  reproduces incumbent behavior bit-for-bit.
- The trace bank REMAINS MAINTAINED in every arm (`self._eligibility_traces`,
  decays, native updates — unchanged dynamics): minimal-diff principle and
  anchor comparability (every R0-b arm differs from the R0 anchor arm only
  in what feeds the readout). Traces acquire no consumer under non-trace
  families; deactivation is a post-verdict engineering change, never
  verdict-relevant (recorded disposition).
- New per-family transient state (all UNPERSISTED, same status as traces in
  `_BASE_STATE_ARRAYS`; exact formulas per family in §2):
  - `self._window_tokens: tuple[int, ...]` — the frozen bank.
  - A: `self._token_spike_counts` u32 `(reservoir_cells,)` kernel-maintained;
    `self._count_ring` u32 `(800, reservoir_cells)` ring of per-token count
    vectors; rows `self._window_counts` f64 `(N_WIN, reservoir_cells)`.
  - B: `self._voltage_ring` f64 `(800, reservoir_cells)` boundary samples +
    fill pointer.
  - C: `self._token_ring` int32 `(800,)`; `self._occurrences` int32
    `(N_WIN, token_count)`; rows `self._assembly_counts` f64
    `(N_WIN, reservoir_cells)`.
  - Combo: concatenation of member buffers in canonical family order.
- ONE accessor defines THE readout feature space:
  `circuit.feature_state_vector() -> np.ndarray` returns the flat f64 view
  of the ACTIVE family's buffer (source id s ∈ [0, F−1); decode per family
  in §2). Feature dimension `feature_count = F + 1` replaces
  `trace_count * reservoir_cells + 1` (:285); `_growth_feature_count`
  (:289), ridge allocations, and sparse source space resize through the
  existing dimension-generic paths exactly as in R0 §1.2.
- `collect_state_separation` (ladder_experiments.py:670-737) switches its
  snapshot channel from `circuit._feature_traces` (:700) to
  `circuit.feature_state_vector()`; pairing, strata, first-teacher-token
  exclusion, and medians untouched.

### 1.3 How observe/strengthen/race consume them

- `observe(token_id)` (:789-802): unchanged call shape; after the native
  token advance returns, the family's boundary update runs (§2), then the
  flat feature buffer is current for whatever comes next.
- `strengthen(target)` ridge path (:872+):
  `feature[:-1] = self.feature_state_vector(); feature[-1] = 1.0` — the
  R0 §1.3 pattern verbatim. Downstream (batched outer products, dsyrk
  flush, 0.99-diagonal relief consolidation) is already dimension-generic.
- Inference/race: readout currents consume the active buffer via the same
  matrix-product the trace path uses today (:820-831 region).
- sparse_three_factor: `SparseCellReadout` binds the flat family buffer as
  its source space at construction (:329-343 pattern); canonical candidate
  count `min(F, CANDIDATE_FACTOR * fan_in)` (readout.py validation).
  Per-contact apposition decay generalizes dt-free: block k's contacts
  decay per token by `exp(−1 / W_k)` (window length substituted for the
  τ/T ratio; identical formula shape to `_source_trace_decays` :355-361,
  and coincidentally equal to today's row-0 value since τ_0/T = 16 = W_0).

### 1.4 What deliberately does NOT change

Tokenization, framing, curriculum generators, supervision scope, race
dynamics, GLIF5 integration, recurrent topology/signs, developmental
controller, QA prompt framing, presentation schedules, codebook install
semantics, curriculum digest. Any drift invalidates the run (curriculum
anchor, §4).

### 1.5 Checkpoint layout v6

Same discipline as R0 §1.5: feature banks are TRANSIENT and stay
unpersisted; the bump exists because feature-space-dependent PERSISTED
state (ridge statistics, dense/sparse weights) changes width.

- `FORMAT` bumps to `"recurrent-human-cell-checkpoint-v6"`. Writers always
  write v6. Metadata additionally carries `"feature_family"` (str),
  `"feature_window_tokens"` (list), and `"feature_cell_count": F` beside
  the existing guards (:463, :676-694).
- v6 loading (strict): family and window list must equal the config's; every
  persisted array shape checked against the family's F (existing shape-guard
  message contract, recurrent.py:475-477 style).
- v5 loading (legacy, exact-by-construction): permitted ONLY with
  `feature_family="traces"`; a v5 payload requested under any other family
  raises loudly (never silently reshaped). With `"traces"` the layout IS
  v5's, so restore is bit-identical.
- Ridge statistics are NOT migrated across feature geometries (same
  disposition as R0 §1.5): switching family restarts readout learning blind;
  retraining from scratch is the sanctioned path. Declared, not hidden.

## 2. Family definitions (exact feature formulas)

Canonical family order for all ordering decisions: **A** spike_counts <
**B** voltage_quantiles < **C** assembly_overlap. `t` = number of tokens
observed since the last `reset_sequence()`; `R` = reservoir_cells;
`T_tok = token_interval_ms`.

### 2.A Spike-count windows (`spike_counts`)

Per-token count vector, kernel-maintained:

```
s_t[c] = Σ over the token's token_steps sub-steps of spike_mask_step[c] ∈ {0,1}
```

i.e. the number of times cell c crossed threshold during token t's
integration. Counts are small non-negative integers (the fitted GLIF5
spike-cut refractory bounds firing rate; cells.py `AllenGlif5Parameters`)
so f64 storage is exact.

Feature rows — hard-edged sliding sums:

```
A_k[c](t) = Σ_{u = max(1, t−W_k+1)}^{t} s_u[c]        k = 0..7
F_A       = N_WIN · R          (8192 at R=1024)
source id s decodes as (k, cell) = divmod(s, R) — same decode as traces.
```

Update at each token boundary (exact integer arithmetic in f64):

1. push `d_t := s_t` into `_count_ring`;
2. for every k: if `t > W_k`, subtract the ring entry exiting window k
   (`_count_ring[t − W_k]`), then add `d_t`.

Cost: O(N_WIN·R) adds/token host-side + the kernel accumulation below.
No exponential kernel exists anywhere in this family — that is the point.

### 2.B Membrane-potential quantiles (`voltage_quantiles`)

Boundary capture: after `observe()` completes its full sub-step integration
and BEFORE strengthen/race, sample

```
b_t[c] = state.voltage_relative_mv[c]     (f64 relative mV, cells.py :290/:310)
```

into `_voltage_ring`. One sample per cell per token boundary ("boundary
captures"); intra-token trajectory is deliberately NOT stored.

Feature rows — exact empirical quantiles of the last W_k boundary samples:

```
B_{k,q}[c](t) = np.quantile({ b_u[c] : max(1, t−W_k+1) ≤ u ≤ t }, q,
                             method="linear")     (numpy default interpolation)
block index m = 3·k + j, j ∈ {0,1,2} for q ∈ (0.10, 0.50, 0.90); row order
[q10 win0, q50 win0, q90 win0, q10 win1, ...]
F_B       = 3 · N_WIN · R      (24576 at R=1024)
source id s decodes as (m, cell) = divmod(s, R).
```

- Quantile method PINNED: numpy `method="linear"` on the raw sample set —
  exact, deterministic, and platform-stable; running/approximate estimators
  (P², histograms) are REJECTED ex ante because they change semantics under
  reordering and cannot be parity-tested bit-exactly.
- Rows are recomputed from the ring lazily at consume points (strengthen /
  pre-race), not per sub-step; §6 accounts the cost honestly including the
  1M-scale open flag.
- On CPU the boundary read is free (host-live state); on CUDA it requires
  the mirror extension of §3 — hence "needs boundary captures".

### 2.C Assembly-overlap counts (`assembly_overlap`) — codebook random/similarity arms only

Assemblies are the installed afferent table itself:
`A(t) = circuit._afferent_targets[t]` (post-`install_codebook`,
ladder_experiments.py:329-352) — the set of reservoir cells token t's pulse
injects. Under `explicit`, this table IS the incumbent wiring and family C
would re-derive input bookkeeping with no new structure (expected cross-token
overlap 2/16, a013-representation-choice.md :10-14), so C is RESTRICTED to
the random/similarity codebook arms where assemblies exist as first-class
distributed objects with controlled overlap statistics.

Feature rows — how many DISTINCT recently-active assemblies cover each cell:

```
Active(t, W_k) = 1 iff token t was observed within the last W_k tokens
C_k[c](t)      = |{ t' distinct : Active(t', W_k) ∧ c ∈ A(t') }|
F_C            = N_WIN · R          (8192 at R=1024)
source id s decodes as (k, cell) = divmod(s, R).
```

Distinctness is handled by occurrence counters so repeated tokens count once:

- `_occurrences[k, x]` += 1 when token x is observed; on a 0→1 transition,
  `C_k[c] += 1` for every `c ∈ A(x)`.
- When token y exits window k at the boundary, `_occurrences[k, y]` -= 1;
  on a 1→0 transition, `C_k[c] -= 1` for every `c ∈ A(y)`.

Mechanistic role: under the similarity projection, structurally related
tokens share cells (blake2b trigram votes, ladder_experiments.py:223-269),
so C_k reports WHICH SEMANTIC CONTENT recently surrounded each cell —
context composition relative to wiring, something neither own-spikes (A)
nor own-voltage (B) encode. The random arm serves as the distribution-only
control (§4). Host-only bookkeeping: no kernel or mirror changes (§3).

### 2.D Combination arm

`feature_state_vector()` under `"combo:<f1>+<f2>"` returns the flat
CONCATENATION of the two members' buffers in canonical family order,
followed by nothing else; bias stays the separate final slot as everywhere.
`F_combo = F_f1 + F_f2` (e.g. A+B → 32768 at R=1024). Members keep their own
boundary updates; composition is an addressing change only. Which two run is
frozen by the selection rule in §4.4 — chosen BEFORE any combo token is
observed, from diagnostics the protocol pins now.

## 3. Native mirror requirements

How state is mirrored TODAY (the contract any change must preserve):
CPU/Rust host numpy arrays ARE the live kernel state; the serial/AVX kernels
write `spike_mask` per sub-step at fixed sites (lib.rs :107-109 scalar step,
:158-191 fused pass) and decay/increment every trace row with fixed op order
(spike-determine → decay → increment; multi-row debug contract lib.rs
:153). CUDA owns state device-side; mirror mode is chosen at binding
(recurrent.py :386-390), `"eligibility"` downloads only the K·R bank,
`"full"` everything mutable (cuda.py :587-602); DtoH bytes/token are
computed from the mirrored set (:505-508).

Per family (all mandatory before that family's gate arm):

1. **A (`spike_counts`) — kernel accumulation, all three backends.** Each
   kernel gains a u32 `(cells,)` accumulator `token_spike_counts`, cleared
   at token start and folded at EXACTLY the existing mask-write sites
   (`counts[cell] += crossed`). No new pass over cells; no change to spike
   determination or to any trace row. Rust serial + AVX and CUDA implement
   identical accumulation order. The buffer is host-live on CPU (like all
   state) and added to the CUDA eligibility-mode mirror payload.
2. **A — CUDA mirror byte contract.** New mirror payload for family-A arms:
   traces bank (8·K·R B, kept warm per §1.2) + counts (4·R B).
   `device_to_host_bytes_per_token` (:505-508) and simulation-speed.md
   updated to state both terms explicitly rather than silently growing
   traffic.
3. **B (`voltage_quantiles`) — boundary capture only; NO kernel changes.**
   CPU: `_voltage_ring` append reads `state.voltage_relative_mv` directly
   post-observe (host-live). CUDA: new mirror mode
   `"eligibility_plus_voltage"` = traces bank + voltage f64 (8·R B/token);
   quantile math happens HOST-side from downloaded boundaries so CPU/CUDA
   consume identical sample sets (parity testable bit-exactly, R0B-T6).
   Forcing the existing `"full"` mirror was REJECTED: it would drag 60+
   B/cell/token of unrelated arrays into a sparse-readout arm's budget.
4. **C (`assembly_overlap`) — zero kernel/mirror delta.** Token ids are
   known host-side; assemblies are the host-owned afferent table. Nothing
   native moves. This family is the control for "native cost" claims in §6.
5. **Quiescent/frontier safety.** Activity/frontier checks
   (`eligibility[cell] != 0.0`-style, lib.rs :568 region; active-count
   accounting native.py :1277-1278) consider ONLY trace rows and cell
   dynamics — never the new accumulators/rings. A zero-trace cell with
   nonzero counts stays quiescent exactly as today; frontier membership is
   bit-identical across families by construction (R0B-T7 asserts it).
6. **Parity tests extend per family** (§5): 64-token CPU-vs-CUDA equality
   asserted on the counts accumulator (A) and on boundary-voltage samples
   (B); reset synchronization zeroes accumulators on device; C asserts
   backend-invariance of its host bookkeeping after switching mid-sequence.

## 4. THE GATE (preregistered, mechanical, per-family verdicts + combination arm)

Single gate execution; five confirmatory runs; nothing may be tuned,
re-seeded, or added after the first token is observed. Diagnostics never
gate (house rule). Thresholds are IDENTICAL to R0 §3 — only the feature
state under test differs.

### 4.1 Frozen run configuration (every arm)

Byte-identical to the recorded R0 anchor arm
(`experiments/r0_multitrace/ridge_N1024_s7.json`):

- `n_cells=1024`, `seed=7`, `schedule=massed`, `data_scale=42`,
  `supervision=all_transition`, `readout_rule=ridge`,
  `--ridge-backend scipy`, `codebook=explicit` (except C arms below),
  A2 lane framing, 3 corpus passes, input_fanout=16, recurrent_fanout=8,
  `HUMANBRAIN_DEVICE=cpu` pinned, `--check-determinism` enforced
  (byte-identical duplicate record modulo wall_s).
- Freeze anchor: `curriculum_sha256 == 48219eda110f09f735525a7de00e56596f
  fac729746654e809c3c5eb525ccae6` — checked ONCE before the first arm; a
  mismatch ABORTS with the cohort NOT spent (r0_gate.py :42-58 pattern).
- git_head recorded per arm record; all §5 red tests green on that exact
  head (preflight).

### 4.2 Arms, in pinned execution order

| # | experiment_id | family | codebook | role |
| --- | --- | --- | --- | --- |
| 1 | `r0b_A_spikewin` | spike_counts | explicit | confirmatory |
| 2 | `r0b_B_voltq` | voltage_quantiles | explicit | confirmatory |
| 3 | `r0b_C_asmbl_sim` | assembly_overlap | similarity (`--assembly-k 16`) | confirmatory |
| 4 | `r0b_C_asmbl_rand` | assembly_overlap | random (`--assembly-k 16`) | control (gated mechanically; see caveat) |
| 5 | `r0b_combo_<X>_<Y>` | combo:X+Y | as members require | confirmatory (§4.4) |

Outputs to `experiments/r0b_state_features/<arm>.json` +
`<arm>.gate.json`; aggregate `gate_results.json`. Runner:
`tools/r0b_gate.py` (structure of `tools/r0_gate.py`: anchor preflight →
ladder invocation → G1 via `collect_state_separation` on
`feature_state_vector()` → G2 via `qa_experiments.run(ExperimentConfig(
seed=7, circuit=RecurrentCircuitConfig(reservoir_cells=96,
feature_family=<family>)))` → G3 from the arm record). Combo members'
codebooks: if both members are A/B → explicit; any member C → that C arm's
codebook; the two C arms never co-member (§4.4 candidates exclude the
control). QA guard unchanged: `GUARD_DISTINCT_MIN = 3`
(qa_experiments.py:45), modal_share recorded diagnostically.

Control-arm caveat (fixed now): if `r0b_C_asmbl_rand` passes where
`r0b_C_asmbl_sim` also passes, semantic structure added nothing beyond
distributed overlap — recorded as the H-code-analog caveat and carried into
R2 sizing; it does NOT alter any mechanical verdict.

### 4.3 Gates (per arm, identical thresholds to R0 §3.2)

State vectors for G1: after growth + freeze, replay each of the 18
CALIBRATION tasks' framing observations (`reset_sequence()` then `observe`
over `typed_program._model_input(...)`), snapshot `feature_state_vector()`
(bias excluded; identical exclusion to every prior separation measurement),
pairwise cosine over task pairs with DIFFERENT first teacher token within
the same stratum (Int / List[Int]) — exactly `collect_state_separation`
(ladder_experiments.py:666-737) with the channel swapped.

- **G1 (separation, primary)**: pooled MEDIAN cosine < 0.90 across those
  pairs ⇔ median 1−cosine > 0.10. Null being beaten: 6.7713e-05 (R0 gate)
  and 0.9989-cosine headline (roadmap :16); the boundary demands a ≥ ~1500×
  move from the measured R0 null.
- **G2 (answers, primary)**: distinct_answers ≥ 3 on the bundled QA fixture
  corpus at committed defaults, seed 7, held-out CALIBRATION free-run.
- **G3 (capacity non-regression, secondary)**: train_replay_tasks ≥ 39/42
  (baseline 41/42 single-trace seed-7 ridge; floor 39). train_cases_passed
  (/126) and calib_cases_passed (/54) recorded diagnostically.

Per-arm verdict semantics (fixed now):

- **PASS** ⇔ G1 ∧ G2 ∧ G3 for that arm.
- **FAMILY-FALSIFIED** (¬G1 ∨ ¬G2) on an honestly implemented stack ⇒ THAT
  family is falsified as the readout basis at this operating point.
  Publishable; spends that arm permanently.
- **CAPACITY-REGRESSIVE** (G1 ∧ G2 ∧ ¬G3): banked disposition; ANY follow-up
  needs a new protocol v2 with a fresh cohort. No repair pass.

### 4.4 Combination-arm selection rule (frozen BEFORE execution)

After arms 1–4 complete within the SAME gate execution:

1. Candidates: {A, B, C_sim}. The control C_rand is never a candidate.
2. Rank by `median_1minuscos_pooled` DESCENDING on the G1 diagnostic;
   tie-break (a) higher train_replay_tasks, (b) lower wall_s, (c) name
   ascending. Take the top TWO as `<X>_<Y>` in canonical family order.
3. Legality: this consumes diagnostics that do not gate the singles'
   verdicts; the RULE is frozen here, before any token is observed, which
   is what distinguishes it from post-hoc selection. The combo's own gates
   (§4.3) are evaluated identically to every other arm.

### 4.5 Program-level outcomes (fixed now)

| Outcome across the five arms | Disposition |
| --- | --- |
| Any arm PASS | Feature de-collapse achieved. R2 sizing proceeds with the passing basis; if several pass, ship the one with lowest median_1minuscos_pooled (tie-break lower wall_s). |
| ALL arms ¬G1 | **STATE-FEATURE CLASS FALSIFIED** at this operating point — features are NOT the bottleneck. Program PIVOTS to the race/output layer: winner-feedback broadcast loop + HOT-2 seed (roadmap :119-120; mechanism pre-scoped at :62-65). See §7. |
| ∃ G1 passes, ALL ¬G2 | Downstream localization WITHOUT class falsification: features separate but answers stay degenerate ⇒ race/output layer, same destination, different label ("readout/race blocker", R0 §6 distinction preserved). |
| Only capacity failures | Banked dispositions; new protocols required per family. |

Exit-code semantics for `tools/r0b_gate.py`, fixed in advance: **0** = gate
executed to completion with verdicts recorded (pass or fail — failure is a
result); **2** = ABORTED before the cohort was spent (anchor mismatch or an
arm crashed before producing its record). A crash mid-execution preserves
all completed records, marks the cohort spent-with-partials in
`gate_results.json`, and is NEVER retried silently.

One shot means one shot: no alternate seeds, no arm additions, no post-hoc
threshold edits, no rerunning "to check".

## 5. Red tests (must FAIL before implementation, PASS after)

New file `tests/test_state_features.py`, parametrized over the family
registry; each names its failure condition mechanically. The R0 suite
(`R0-T1..T10`) is mirrored 1:1 with family-specific instances; two
combo-level tests extend it. "Per family" below means: instantiated for
each of {spike_counts, voltage_quantiles, assembly_overlap} and for both
legal combos.

- **R0B-T1 window-bank determinism & config validation** (mirrors R0-T1):
  default `WINDOW_TOKENS` bitwise-equals
  `(16, 28, 49, 86, 150, 262, 457, 800)`, seed-independent;
  `feature_family="traces"` reproduces incumbent behavior bit-for-bit
  (same F, same strengthen output); unknown families, malformed
  `"combo:"` grammar (dupes, wrong order, >2 members), and
  non-monotone/non-positive window lists raise ValueError.
- **R0B-T2 scripted-sequence reference equality** (mirrors R0-T2): a
  scripted token sequence (repeats, refractory-suppressed cells, quiescent
  tail ≥ 5·W_7) compared against a NumPy reference implementing EXACTLY
  §2's update — bit-exact, not allclose:
  A: rows equal sliding bincounts; a spike in window k appears in every
  window whose span covers it; expiring tokens subtract exactly once.
  B: quantile rows equal `np.quantile(..., method="linear")` over the ring
  slice; warm-up truncation matches the pinned convention.
  C: occurrence transitions fire only on 0→1 / 1→0; a token repeated
  inside a window counts ONCE; expiry after repeats decrements once.
- **R0B-T3 window-edge sanity** (mirrors R0-T3): after one distinct pulse
  followed by W_7 identical-pair-free silence — A: each row drops to zero
  at exactly t = pulse + W_k (integer expiry, no decay tail);
  B: constant-signal windows quantile to that constant exactly;
  C: coverage counts hit zero exactly at each window edge. Contrast case:
  the trace bank under the same stimulus retains exp(−T/τ_k) tails —
  asserted DIFFERENT from A's hard expiry (the falsification-attack made
  explicit).
- **R0B-T4 checkpoint round-trip v6** (mirrors R0-T4): grow → freeze →
  save → load → `cell_synapse_state_sha256` identical; predictions on a
  fixed context byte-identical pre/post; v6 metadata rejects family
  mismatch, window-list mismatch, and wrong `feature_cell_count`; ridge
  statistics restored at the family's F + 1.
- **R0B-T5 v5 legacy load discipline** (mirrors R0-T5): a v5 fixture loads
  ONLY under `"traces"` with bit-identical outputs and race winners;
  loading it under any successor family raises loudly naming the conflict
  (never silently reshapes or re-trains in place).
- **R0B-T6 CUDA parity per family** (mirrors R0-T6, extends
  tests/test_cuda.py pattern): 64-token loop — A: `assert_array_equal` on
  the u32 accumulator Rust-vs-CUDA after every token, reset zeroes it on
  device, mirror payload carries 8·K·R + 4·R bytes;
  B: boundary-voltage samples `assert_array_equal`, hence host-side
  quantile rows identical by construction, mirror payload
  8·K·R + 8·R bytes (`"eligibility_plus_voltage"`); C: bookkeeping
  invariant across a mid-sequence backend switch.
- **R0B-T7 kernel equivalence across backends** (mirrors R0-T7, extends
  tests/test_simulation_equivalence.py): Python-reference, Rust serial,
  Rust AVX, CUDA produce identical family buffers after the 64-token
  sequence at two activity densities; frontier membership and
  `reservoir_activity_density` are bit-identical across ALL families
  (§3 requirement 5).
- **R0B-T8 feature-vector shape guards** (mirrors R0-T8): `_feature_batch`,
  `_normal_matrix`, `_target_moments` shaped (128, F+1), (F+1, F+1),
  (token_cells, F+1) with F = F_family (8192 / 24576 / 8192 / sums);
  `feature[-1] == 1.0` and `feature[:-1]` bitwise-equals
  `feature_state_vector()` at strengthen time; restore-time mismatch
  raises naming the offending key.
- **R0B-T9 sparse source-space validity** (mirrors R0-T9): potential/
  contact sources sampled unique ascending over [0, F_family); canonical
  candidate count `min(F_family, CANDIDATE_FACTOR · fan_in)`; per-contact
  apposition decay equals `exp(−1 / W_(source // R))` (§1.3 generalization)
  and equals today's row-0 value when W_0 substitutes; native sparse
  current via the flat buffer matches a NumPy reference over the same
  sources.
- **R0B-T10 accounting invariance** (mirrors R0-T10): for fixed seed and
  curriculum, token counts, presentation order, observed-token counters,
  and curriculum digest are identical between `"traces"` and every family
  arm; ONLY declared fields differ (ladder red-test accounting-block
  pattern, a013-teaching-processes.md:177); `reservoir_activity_density`
  remains a trace-row statistic under every family.
- **R0B-T11 combo composition**: `"combo:X+Y"` feature vector equals the
  concatenation of members' vectors in canonical order (bitwise, per
  block); F_combo arithmetic; illegal grammars rejected; member boundary
  updates independent of composition.
- **R0B-T12 frozen selection rule determinism**: the §4.4 ranking function,
  run on synthetic gate records (constructed ties, missing diagnostics,
  control-arm entries), returns the documented choice deterministically
  and ignores the control; unit-pinned so the live gate cannot improvise.

## 6. Cost estimate (arithmetic on measured baselines; estimates flagged)

Measured anchors: **22.7 tok/s grow / 22.1 read** at 1M-cell CUDA text
training, VRAM 2.7/12 GB (roadmap :10-11) ⇒ ~44 ms/token budget; DtoH
contract 8 B/cell/token eligibility-only mirror in that record's era;
R0 arms carry the K=8 bank (64 B/cell/token DtoH) and stay warm per §1.2;
ladder gate wall_s = 48.968 at N=1024 (R0 arm, scipy dsyrk); host ridge
budget 20 GiB binding dense-ridge ceiling ≈ 50k features
(ladder_experiments.py:134). Serialized-copy arithmetic assumes ~10–20 GB/s
effective DtoH (same basis as R0 §5).

Per-family deltas at steady state:

| Quantity | A spike_counts | B voltage_quantiles | C assembly_overlap | worst combo (A+B) |
| --- | --- | --- | --- | --- |
| Kernel compute/step @1M | +1 u32 add on mask writes ≲ +2% | none | none | A's delta |
| DtoH B/cell/token | +4 (u32 counts) | +8 (boundary f64) | +0 | +12 |
| DtoH MB/token @1M (on top of bank's 64) | +4 | +8 | 0 | +12 |
| Est. added copy time/token | +0.2–0.4 ms | +0.4–0.8 ms | 0 | +0.6–1.2 ms |
| Host RAM transient | ring 800·R·4 B = 3.2 GB | ring 800·R·8 B = 6.4 GB | occ K·T ints + token ring (KB-scale) | sum of members |
| Device memory @1M | +4 MB | +8 MB staging | 0 | +12 MB |

Throughput bands at 1M (ESTIMATES, ×wall vs the R0-bank baseline, not vs
22.7 single-trace): **A** ≈ ×1.01–1.05; **B** ≈ ×1.02–1.08 copy-side plus
the quantile flag below; **C** ≈ ×1.00 (host O(k)/token bookkeeping,
noise); **A+B** ≈ ×1.03–1.11. Cost is NOT part of any pass gate; a measured
> 2× end-to-end slowdown at 1M triggers a scaling review as an engineering
disposition, never a silent acceptance or gate edit.

- **Family B open feasibility flag (declared honestly)**: exact quantile
  refresh is `O(Σ W_k · R) ≈ 1848·R` comparisons per consume. At the gate's
  N=1024 this is ~2M compares (~ms) — irrelevant. At R=1M it is ~1.9e9
  compares per consume against a once-per-token strengthen cadence — NOT
  viable as written. Pinned dispositions for a future v2 (none apply to
  this gate): (i) accept the cost behind sparse-readout arms only,
  (ii) subsampled refresh cadence (changes semantics → new protocol),
  (iii) approximate running estimators (REJECTED ex ante in §2.B; would
  need their own protocol). The gate runs at N=1024 where B is cheap.
- **Dense-ridge ceilings** (20 GiB host budget): F+1 widths at R=1024 —
  traces/A/C: 8193² · 8 B ≈ 0.54 GB; B: 24577² ≈ 4.8 GB; A+B / B+C:
  32769² ≈ 8.6 GB — all inside budget; solve flops scale ~(F/8193)³
  under OpenBLAS (~248 GFLOPS measured), so B/combo consolidation moves
  from seconds toward tens of seconds per solve: ESTIMATE gate wall
  minutes-to-tens-of-minutes, acceptable one-time, measured value recorded.
  Sparse/CUDA path structurally unaffected (fixed fan_in; source space
  grows, contacts do not).
- **Python boundary updates**: O(N_WIN·R)/token (A), one memcpy (B),
  O(k + transitions·k)/token (C) — negligible at ladder N, subsumed at
  scale by batched BLAS on the consume side.

## 7. Falsifier (program level)

If ALL FIVE arms end ¬G1 after honest implementation (§5 red tests green,
backend parity proven per family, curriculum anchor matched, single
unmanipulated execution), then **the state-feature class — per-cell windowed
summaries of spikes, voltage, and assembly coverage — is FALSIFIED as the
readout basis at this operating point**, and the conclusion is stronger than
R0's: three mechanistically distinct feature families, including hard-edged
windows that cannot retain the shared framing prefix, still collapse tasks
into one point. Under the preregistered interpretation this localizes the
bottleneck OUTSIDE the feature state: features are NOT the bottleneck.

**Pivot (frozen now, not negotiated later): the race/output layer.**
Concretely, the winner-feedback broadcast loop — the first-spike winner
re-entering the network as input, turning the existing argmax-current race
(roadmap :18) into an iterative computation rather than a one-shot read —
seeded by the HOT-2 hypothesis already queued at roadmap :119-120 ("margins
exist but never re-enter computation"), mechanism pre-scoped at roadmap
:62-65 as the R3 global-workspace half pulled forward. The pivot protocol
owns its own gates; this document only fixes THAT the pivot happens and its
first mechanism.

Boundary distinctions fixed NOW so they cannot be renegotiated:

- ∃ G1 passes but ALL ¬G2 ⇒ NOT a feature-class falsification — features
  separate; answers degenerate downstream. Same destination (race/output
  layer), different disposition label ("readout/race blocker").
- Only capacity failures ⇒ features exonerated twice over; dispositions
  banked; follow-ups require fresh v2 protocols per family.
- The control arm C_rand passing alongside C_sim does not flip verdicts;
  it downgrades the SEMANTIC claim only (§4.2 caveat).
- Either way each arm's cohort is spent once and every outcome stands.

## Honesty notes

- Every §6 figure marked ESTIMATE is arithmetic on measured anchors, not a
  measurement; the records will carry measured wall-clock, VRAM, and DtoH
  bytes per arm.
- The G1 null cited (6.7713e-05) was measured on the TRACE buffer; the
  successor families' true nulls are unknown until run — that is why G1 is
  a gate and not a prediction. The ≥1500× framing describes the threshold's
  distance from the R0 measurement, nothing more.
- Mechanistic bet, stated so it can be wrong in public: R0 failed because
  exponentially-decaying features never forget the shared framing prefix;
  hard-edged windows forget by construction. If families STILL fail G1, the
  shared-prefix theory of the collapse dies with them, which is precisely
  what licenses the §7 pivot.
- Family C's assemblies are the installed afferent table, not emergent
  recruited assemblies; "assembly" here means the codebook object of
  a013-representation-choice.md. Claims about emergent assemblies are out
  of scope.
- The QA leg runs at fixture scale (96-cell circuit, 30-triple corpus) and
  evidences answer de-degeneration, not reading skill; R2 owns the real
  corpus question.
- Traces remain maintained-but-unconsumed in non-trace arms (§1.2): a
  deliberate waste bought for anchor comparability, recorded here so nobody
  "optimizes" it mid-cohort.
