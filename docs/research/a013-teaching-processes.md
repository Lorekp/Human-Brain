# A013 teaching-process experiment matrix

Status: design input — no protocol executed. Companion to
`a013-representation-choice.md` (owns the encoding axis) and
`typed-composition-a013-ladder-v1.md` (owns scale arms). Preregistration
discipline: every experiment declares gates before running; no tuning after
failure.

## Baseline being improved

Sealed a012: 3 identical epochs of observe->strengthen over 42 TRAIN tasks,
supervision per lane (all-transition A0/A1, program-only A2). Result at N=128:
TRAIN replay 40/42 (capacity shortfall), calibration 0/18 (no transfer),
margins strong where committed (median 726 pA) — task-specific saturation.

Feature-space fact (verified, recurrent.py:231): the ridge feature vector is
exactly `[eligibility vector; bias]` — `feature_count = reservoir_cells + 1`.
There are NO token-identity features in the readout itself. Token specificity
enters only through which cells the fixed random fanout excites. Consequence:
the readout's capacity ceiling is set by how linearly separable different
tasks are within one shared eligibility geometry — under explicit token
identities with near-null state separation (~0.001), tasks collide in feature
space. This sharpens T1/T2 below: adding data cannot help once features
collide; changing the code geometry can.

## Axis matrix

| # | Axis | Mechanism hypothesis (code-cited) | Capacity effect | Generalization effect | Cost | Decisive experiment |
|---|---|---|---|---|---|---|
| T1 | Data-scale dose-response | ridge moment accumulation saturates when examples share eligibility features; distinct-task count is the binding constraint (recurrent.py:764-792 `_flush_growth_statistics`) | monotone up to a knee then flat | none directly | S | 21/42/84/168-task cohorts at fixed N=512 local-readout; gate = knee location |
| T2 | Kind diversity — **CONDITIONAL ON REPRESENTATION** | under explicit token identities, new kinds add disjoint features -> interference in shared readout; under distributed overlapping codes, diversity fills shared structure | +/- depends on codes | the main transfer lever ONLY with distributed codes | S | diversity-x-representation interaction row below |
| T3 | Repetition schedule: spaced vs massed vs multi-pass interleave | eligibility decay exp(-d*5ms/80ms): massed repetition within 80ms adds ~nothing (trace already saturated); spacing re-fires on decayed traces => stronger consolidation per presentation (observe/strengthen mechanics recurrent.py:667-723) | spaced >= massed at equal presentations | spaced slightly better (less catastrophic overwrite) | S | equal-presentation-count spaced vs massed vs interleaved; gate = train replay + margin distribution |
| T4 | Supervision-density mixing | all-transition supervision spends budget on framing tokens; program-only starves context learning; scheduled mix could get both (grow loop :1218-1228 switches scope per epoch) | ++ vs program-only alone | + vs all-transition alone | S | epoch schedule {all,all,program} vs fixed single scope; gates split capacity/transfer |
| T5 | Error-comparison teaching | sparse three-factor already updates target AND current winner (+/-); errorless teaching (target only, no competitor depression) removes contrast | neutral | likely worse (no discrimination pressure) | S (rule exists, lib.rs:931) | three-factor with modulation gated by race outcome vs always-on; gate = calibration cases |
| T6 | Replay consolidation pass | store teacher-forced sequences; pre-freeze error-prioritized re-strengthen (biology survey #2; development.py stage hooks) | direct: recovers tail tasks (the 40/42 gap) | flat alone (re-imbues task traces) — pairs with T5/codes | M | one replay pass before freeze; gate = TRAIN 42/42 recovery with calibration flat |

## The interaction row (first-class)

| Teaching \ Codes | Explicit identities (current) | Random distributed k-of-N | Similarity-projected top-k |
|---|---|---|---|
| Single-kind curriculum | baseline (sealed): memorize | memorize, no extra | memorize, mild structure |
| Diverse kinds, interleaved | interference risk: disjoint feature sets compete in readout | overlap gives partial sharing | shared assemblies across families => transfer predicted here |
| Spaced + diverse + replay | capacity recovers, transfer still ~0 | partial transfer | full prediction: transfer appears |

Prediction to preregister: the diversity benefit materializes only in the
right column. If diverse+spaced+replay moves calibration under explicit codes,
this thesis is falsified and representation work deprioritized.

## Ranking by information gain per compute-minute

1. T6 replay (targets the exact sealed gap; cheap; clean interpretation)
2. T3 spacing (pure schedule change; isolates trace mechanics)
3. T5 error-gating (dormant rule; one flag flip)
4. T1 dose-response (cheap but expected: confirms saturation)
5. T2/T6-x-representation interaction (highest ultimate value; needs the
   codebook implementation first — sequence AFTER representation choice lands)

All experiments: fixed seeds, preregistered gates, capacity and calibration
families reported separately, one-shot execution per configuration.
