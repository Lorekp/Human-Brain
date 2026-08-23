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

<!-- R0B-CONTINUES -->

