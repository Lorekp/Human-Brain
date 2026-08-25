# Race-pivot gate verdict — TERMINAL INELIGIBLE at N=96 (2026-08-24)

Status: executed one-shot, recorded permanently. Implementation complete
(21/21 tests green, 4 commits cef8508→8854542 on codex/a013-cognition-design).
N=1024 leg deliberately NOT spent per protocol §5 (stop-on-failure).

## Verbatim terminal numbers (N=96 fixture leg)

| Arm | distinct | exact | replay | modal_share | margin_med | notes |
|---|---|---|---|---|---|---|
| P-S1 | 6 | **0** | 42/42 | 0.167 | 224.6 pA | stable_frac 0.46 |
| P-S2 | 6 | **0** | 42/42 | 0.167 | 184.3 pA | stable_frac 0.52 |
| P-S4 | 6 | **0** | 42/42 | 0.167 | 122.7 pA | stable_frac 0.41 |
| P-R4 | 6 | **0** | 42/42 | 0.167 | 205.1 pA | flip_rate 0.508 under chase |
| P-M1 | 6 | **0** | 42/42 | 0.167 | selective=0.081 | coverage 1.0 |

G-P1 (distinct≥3) PASS everywhere; G-P3 (replay≥39/42) PASS everywhere;
G-P2 (exact≥1) FAILED everywhere → LEG INELIGIBLE.

## What the pivot achieved and what it did not

ACHIEVED: answer de-collapse — broadcast iteration produced 6 distinct
answers with modal share 0.167 (vs the [variable-0] monoculture of R0/R0-b),
while preserving training replay perfectly. The degeneracy mechanism is
understood and controllable.

NOT ACHIEVED: zero exactly-correct completions. Diverse-but-wrong: the
broadcast amplifies state differences into varied outputs, but the states
still do not carry enough context identity to select correct answers.

## Falsifier clause status

Per race-pivot-protocol.md §falsifier: features (R0), successor state
features (R0-b), associative store on reservoir keys (R0-c P2), scale alone
(R2 pilot at 131k), and now the decision layer (race-pivot) have all been
executed honestly under preregistered gates. The reservoir-computing
architecture family is falsified for QA reading ON THIS EVIDENCE BASE at
small N. The protocol's named consequence activates: custom-architecture
pivot, with the shortlist (STDP-trained spiking arrays + workspace
broadcast; HTM-style sparse distributed representations) requiring user
direction before any compute is spent.

## What survives regardless

All speed work (L3 event-proportional, kernel bitmap 3.06x, overlap +17.5%,
Seam A read_cells 36.9us), v6 checkpoints, K-bank mirrors, QA/corpus/
baseline/analysis stack, SWE ceremony, concept tokenizer (+19% compression),
and a fully instrumented negative-result ledger that redirects future effort
with evidence instead of taste.
