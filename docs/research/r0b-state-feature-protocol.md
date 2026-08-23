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

<!-- R0B-CONTINUES -->
