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

<!-- R0B-CONTINUES -->



