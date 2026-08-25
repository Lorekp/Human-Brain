# Post-falsification strategy report (R0/R0-b → next mainline)

Status: ADOPTED as program mainline decision, 2026-08-24. Author: strategy
agent (decision-grade report, verified against committed records); adopted by
orchestrator. Full text preserved in delegation transcript
deleg_f4733c7b/task-0.log; this doc carries the operative content.

## Verdict table (what five failures established)

| Family | G1 sep | G2 | G3 | Verdict |
|---|---|---|---|---|
| Multi-taus eligibility (R0) | 6.77e-05 | 6 | 42/42 | FALSIFIED |
| Spike-count windows | 1.10e-04 | 6 | 42/42 | FALSIFIED |
| Voltage quantiles | 7.59e-04 | 6 | 42/42 | FALSIFIED |
| Assembly-overlap similarity | 0.0978 | 6 | **1/42** | FALSIFIED |
| Assembly-overlap random | **0.1019 PASS** | 6 | **12/42** | BANKED capacity-regressive |

Derived findings: (1) §7 class-falsification did NOT fire — pivot is a
choice; (2) hard-window failure is POSITIONAL (carry early-context identity
to late decision point) — windowing retries demoted; (3) separation–capacity
tradeoff observed three independent times — a lookup store escapes it (store
grows; replay untouched); (4) G2 guard satisfiable by diverse-but-wrong
priors — the only bar now is Bible-VAL span accuracy vs unigram 4.54% /
trigram 16.28%.

## Ranking

1. **WINNER: (c) hybrid associative memory through the race layer** — new
   `associative_store.py` (Hebbian write, thresholded-overlap read), keys =
   binarized `feature_state_vector()` (C_rand proved separability at 0.102),
   values = answer-token multi-hot, injected at `output_current_pa()`
   (`recurrent.py:858-869`) before `race()` (unchanged). Escapes linear
   separability AND replay-capacity traps; roadmap R3 contingency pulled
   forward with sanction.
2. Runner-up: (a) winner-feedback broadcast + HOT-2 margins — parallel red
   tests only (shared injection seam).
3. (d) modern learned readout — DIAGNOSTIC ONLY (probe P1).
4. E2 TD/delta features — queued behind probes.
5. E3 scale-vs-N separation calibration — background.

## Two-week plan (three threads)

Thread 1 mainline: D1-2 logging harness (Bible TRAIN 200 / VAL 60 at
N=1024 + N=96, per-token features + raw states via observation_sink);
D2-4 offline probes — P1 ceiling (logistic+MLP on raw states: < unigram ⇒
reservoir-state route dead; ≥ trigram ⇒ escalate (d) to user), P2 memory
route (SDM store/query sweep radius/top-k: pass = >4.54% span-acc at ≥2σ,
distinct ≥3, shuffled-key control ≈ unigram), P3 attribution by chapter
distance; D5 mechanical decision checkpoint; D6-9 integration behind
config flag (red tests T-A1..A5 incl. flag-off bit-identity); D10-14
Bible VAL run at N=1024 + capacity non-regression asserted.

Thread 2 runner-up prep: broadcast-seam spec + margin-consumer red tests.
Thread 3 background: 128k separation-vs-N calibration (E3); land sandbox
trampoline fix (needed regardless).

## Honest probabilities

P(clean Week-1 probe decision): ~85%. P(beats unigram on VAL in 2wk): ~35%
(45% conditional on P2 pass). P(beats trigram in 1mo with 1M CUDA): ~25%.
P(above-trigram reading in 6wk): ~25–30%. Most likely good outcome of the
fortnight: first cleanly localized statement of where context identity dies
(dynamics vs decode chain vs representation) — decision-grade either way.
