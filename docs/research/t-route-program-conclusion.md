# T-ROUTE program-grade conclusion — local-learning escalation ladder exhausted

Status: recorded 2026-08-25. E1/E2/E3 all one-shot, preregistered, pushed
(E3: a9ace9b). This document is the evidence-backed termination of the
Bible-QA gate for the fixed-local-connectivity architecture class.

## The complete chain

| Probe | Question | Answer |
|---|---|---|
| E1 | Does episode-scoped three-factor learning create routing? | NO (base_med 0.0; homeostat identified as annihilator) |
| E2 | Does removing normalization + adding sparsity recover it? | NO (overlap recovers but winners are query-independent, shuffled e=1.0) |
| E3 | Does quote-span-specific gating fix it? | NO (random-span control reproduces gated arm exactly, spec 0.98–1.007) |

## The mechanistic statement

Token-*identity* bits cannot express token *position*. Quote-specific
eligibility redistributes mass within the static prior but cannot make the
substrate answer "where" — position is not addressable in this representation.
The random-span control proved placement carries zero signal: identical
winner sets from arbitrary windows.

## Program-grade conclusion

Within the constraint set (local learning rules only, fixed connectivity,
token-identity SDR substrate), Bible-QA routing requires mechanisms this
architecture class cannot express. The remaining paths require relaxing a
constraint:
1. **Non-local optimization** (relax local-learning-only), or
2. **Trained-by-construction connectivity** (R3 episodic-store hybrid:
   connectivity shaped during growth by passage statistics), or
3. **Task pivot** (different QA formulation where identity bits suffice).

This is an evidence-backed termination, not a fudge: 14 preregistered
executions, every failure localized to mechanism, zero unexplained.

## What survives (complete inventory)

Speed stack (L3+SeamA+SeamB+marshal+bitmap+overlap: EP default everywhere,
2× sparse band, 38× mid-density), v6 checkpoints, K-bank mirrors on three
backends, concept tokenizer (+19% compression), QA/corpus/baseline/analysis
stack, SWE ceremony + M0/M1 harness, synthetic-episode synthesizer (2000+
episodes ready), and a fully instrumented negative-result ledger.

## Escalation decision — user's call

The strategy agent's next cycle should evaluate paths 1–3 against the
program's actual goal (coding competence). Note: path 2 (R3 episodic store)
was always roadmap-scheduled and does NOT violate neuron-derived constraints
— connectivity trained during development by passage statistics is what
brains do. Path 1 relaxes "local learning rules only" but keeps everything
else. Path 3 abandons verbatim-quote QA for tasks identity-SDRs can express.

**Update 2026-08-25:** the strategy cycle completed six primary-source deep
dives (Assembly Calculus/NEMO, DendriCL + CTP-Hybrid, HTM Temporal Memory,
NEST e-prop + HuMAM/DTB, SpikeGPT + NeuronSpark, spiking-reservoir-lm).
Their consolidated effect on this fork — re-priced prospects per path, a
master adopt/adapt/avoid ledger grouped by path, two new cheap probes
(B′ = APICAL-LMS Stage-0 via TwoCompartmentGLIF5; HTM predictive-state
depolarizing bias into the first-spike race), and a recommended execution
order under the single-GPU + $70 vast.ai envelope — is in
[path-decision-synthesis.md](path-decision-synthesis.md). No finding reopens
the termination recorded above; the three independent external corroboration
angles are summarized there.
