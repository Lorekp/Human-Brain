# Readout-rule comparison pilot: ridge vs local three-factor at matched N

Experiment ID: `readout_compare` · run date 2026-08-23 · device `HUMANBRAIN_DEVICE=cpu`

## Question

Single-axis pilot from the sealed a012 negative result
(`docs/research/a013-teaching-processes.md`, baseline block): does switching the
readout rule from closed-form **ridge** (FORCE-style least squares on
`[eligibility; bias]`) to the dormant local biology-fusion rules (**three_factor**
dense perceptron-style competitive Hebbian; **sparse_three_factor** apposition-trace
contact updates, lib.rs:931) change capacity or calibration transfer at matched
reservoir size?

## Fixed configuration

27 arms = N ∈ {128, 512, 1024} × rule × seed ∈ {7, 8, 9}, driven by
`benchmarks/ladder_experiments.py` @ commit `e075b93`: schedule=`massed`,
data_scale=42 TRAIN tasks, supervision=`all_transition`, A2 framing, 3 corpus
passes, fresh per-seed a013 cohorts (literals disjoint from the spent a012 set),
harness defaults otherwise (input_fanout=16, recurrent_fanout=8,
local lr/margin = 64 pA). Per-seed curricula are byte-identical across rules and
N (verified via `curriculum_sha256`). No frozen protocol was executed; nothing
under `artifacts/` was touched. Determinism re-run of one arm
(`--check-determinism`) passed byte-exact modulo `wall_s`. All 30 grow+eval
passes totalled 294 s; every arm exited 0. `src/humanbrain` was byte-stable
across the whole matrix (only doc commits landed mid-window); outputs left
uncommitted for orchestrator collection.

## Results — mean ± sd over seeds {7,8,9}

| N | rule | train_cases_passed frac (/126) | train_replay_tasks frac (/42) | calib_cases_passed frac (/54) |
| --- | --- | --- | --- | --- |
| 128 | ridge | 0.1455 ± 0.0229 | 0.0635 ± 0.0364 | 0.1481 ± 0.0321 |
| 128 | three_factor | 0.0899 ± 0.0279 | 0.0079 ± 0.0137 | 0.1852 ± 0.0000 |
| 128 | sparse_three_factor | 0.0503 ± 0.0451 | 0.0000 ± 0.0000 | 0.1235 ± 0.1069 |
| 512 | ridge | 0.5635 ± 0.2409 | 0.5238 ± 0.2474 | 0.1235 ± 0.0283 |
| 512 | three_factor | 0.0899 ± 0.0279 | 0.0079 ± 0.0137 | 0.1852 ± 0.0000 |
| 512 | sparse_three_factor | 0.0503 ± 0.0451 | 0.0000 ± 0.0000 | 0.1235 ± 0.1069 |
| 1024 | ridge | 0.9815 ± 0.0165 | 0.9762 ± 0.0238 | 0.1543 ± 0.0283 |
| 1024 | three_factor | 0.0899 ± 0.0279 | 0.0079 ± 0.0137 | 0.1852 ± 0.0000 |
| 1024 | sparse_three_factor | 0.0503 ± 0.0451 | 0.0000 ± 0.0000 | 0.1235 ± 0.1069 |

Per-seed raw passes (train_cases/126 · train_replay/42 · calib/54):

- ridge 128: 15·1·7 / 20·3·10 / 20·4·7 — 512: 106·34·5 / 55·16·7 / 52·16·8 — 1024: 123·41·8 / 126·42·10 / 122·40·7
- three_factor (all N): 15·1·10 / 11·0·10 / 8·0·10
- sparse_three_factor (all N): 0·0·0 / 11·0·10 / 8·0·10

Local-rule cells are literally N-invariant per seed: same pass counts, same
margin samples (42 collected per train eval), margin medians ≈ −42.7 pA at every
N. Sparse seed 7 is a fully collapsed variant (0 passes everywhere, margin_count
148). Ridge at 512 shows unstable emergence (seed 7: 106 vs 52–55).

## Interpretation

**1. Does any local rule beat ridge on calibration? Nominal yes, mechanistic no.**
On mean calibration fraction, `three_factor` nominally beats ridge at every N
(0.185 vs 0.148/0.124/0.154) and never loses the paired per-seed comparison
(wins 6, ties 3 of 9 cells); `sparse_three_factor` does not beat ridge on means.
But a post-hoc per-task probe (read-only use of the harness functions) shows
**every** local-rule completion — 18/18 calibration and 42/42 TRAIN tasks under
sparse_three_factor, and nearly all under three_factor — emits the single
degenerate token `variable-0` then stops. Two of the 18 calibration tasks are
identity prompts for which the constant pass-through program is *correct*
(6/54 = 11.1% floor, replay 1); the remaining local-rule "passes" are ≤4 lucky
single-case coincidences of that constant. Ridge's low-N calibration has the
same composition (14/18 constant programs at N=128). Excluding identity tasks,
genuine transfer is ≤4/54 (≤7%) for **every** rule at **every** N. The honest
verdict: **no local rule demonstrates transfer advantage over ridge; the nominal
three_factor edge is an artifact of failing with a less-wrong constant program.**

**2. Capacity ordering vs the feature-collision thesis: supported.**
The thesis (a013-teaching-processes.md baseline section) says capacity is set by
how linearly separable tasks are inside the shared eligibility geometry, and
that changing the readout *rule* without changing feature geometry should not
create transfer. Observed exactly: ridge capacity rises monotonically with
reservoir dimensionality (0.146 → 0.563 → 0.981 train cases; full capacity
family 42/42 replay + 126/126 cases at N=1024 seed 8) while both local rules are
flat at ~0.05–0.09 at every N — their binding constraint is learning dynamics,
not feature dimensionality. Meanwhile calibration stays pinned at the identity
floor plus noise for all nine (N, rule) cells, including ridge's perfect-capacity
point: fits-but-does-not-generalize, the ladder's predicted branch for rule-only
interventions. No result contradicts the collision thesis; the dormant
biology-fusion rules, at shipped hyperparameters (lr=64 pA, margin=64 pA), do
not rescue either deficit.

## Caveats

- Pinned knobs (massed, all_transition) are deliberately adverse relative to the
  sealed a012 operating point (ordered corpus passes ≡ `spaced` here, A2
  program-only); absolute levels are not comparable to sealed numbers. Harness
  sanity runs (`experiments/a013-ladder-sanity-v1/`, spaced+program_only+21)
  confirm the collapse is not schedule-induced: ridge memorizes 21/21 & 63/63
  there while three_factor gets 6/63 and sparse_three_factor 0/63.
- Local rules ran only at harness defaults; this pilot tests the rules as they
  ship, with no hyperparameter search (preregistration discipline: no tuning
  after failure). A sensitivity sweep would be a new preregistered experiment.
- Margin statistics under local rules reflect the degenerate completions
  (medians ≈ −42.7 pA ≈ systematic first-token loss near the 64 pA competitive
  margin), so margins cannot arbitrate "close but wrong" learning for these arms.

## Preregistered-gate verdicts

1. *"Any local rule beats ridge on calibration at any N?"* — **REJECTED as a
   transfer claim**: nominal mean-level win for three_factor at all three N, but
   decomposition shows it is identity-floor + luck from degenerate constant
   programs, not learned generalization. Zero-transfer reading of sealed a012
   stands for all rules.
2. *"Capacity ordering matches the feature-collision thesis?"* — **CONFIRMED**:
   ridge-only monotone capacity scaling in N; local rules N-flat; calibration
   immovable at floor across all 27 arms. Rule substitution alone does not break
   the collision barrier; representation/code-geometry work remains the lever.
