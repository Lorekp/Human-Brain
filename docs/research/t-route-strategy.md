# Second strategy cycle — T-ROUTE: trained question-space↔passage-space mapping

Status: adopted as program mainline (2026-08-25). Full report in delegation
transcript deleg_2877a5e2; operative content here.

## The decisive new fact

Who-said questions are built as `Who said, "{speech}"?` where speech is a
VERBATIM contiguous substring of the context verse, and framing is
context-first/question-last — at decision time the passage representation
is fully formed and the question arrives as a query into it. The learnable
target therefore decomposes into quote-subassembly→same-subassembly-at-
passage-position correspondences plus local adjacency — statistics that
one-shot Hebbian/three-factor rules estimate unbiasedly, with compositionality
from the SDR code providing weight sharing across items keyed by word identity.

## Ranking

1. **T-ROUTE (RECOMMENDED)**: trained Q→P projection, local three-factor at
   projection synapses + two-hop routed retrieval. Q-pop = undecayed union of
   question stream SDRs (reuses sdr_union.py + concept_tokens.py); P-pop =
   existing passage union state; W = sparse projection trained only at
   projection synapses (ΔW ∝ pre·post·M); homeostat supplies IDF-like scaling.
   Hop 2 is ordinary temporal contiguity (trace machinery's home turf).
   ~400 LoC numpy, CPU-only, $0. P(exact≥1/4wk) ~35%.
2. Correlational soft-binding — folds in as E1 control arm B (dense
   unnormalized correlation matrix); measured failure signature twice already.
3. Neuromodulator-gated broadcast — folds in as #1's third factor (question-
   onset eligibility window); not a mapping mechanism solo (~5%).
4. Predictive-coding bidirectional loop — deferred to cycle 3.

## Gates (E1, one-shot preregistered, $0)

Arms: A sparse trained W + homeostasis; B dense unnormalized correlation;
controls shuffled-question / randomized-W / wrong-passage.

- G-T1 routing enrichment: median overlap(retrieved winners, gold answer-region
  SDRs) >= 2x shuffled-question AND >= 1.5x randomized-W, per family stratum,
  TRAIN-self vs VAL split reported (early-warning contrast)
- G-T2: full pipeline VAL exact >= 3/639
- G-T3 anti-leak: wrong-passage routing <= chance; shuffled-W inside random band

Decision semantics: T1+T2 pass -> install behind config flag, one-shot official
VAL leg wk2. T1 pass + T2 fail -> NEGATIVE_WITH_LOCALIZATION_V3 (routing works,
extraction broken) — escalate extraction only. T1 fail -> trained-inter-space
hypothesis takes its core wound; directions die together at the shared root.

Falsifier: G-T1 failure = locally-trained projections carry no more cross-space
structure than raw fixed-random overlap => kills the whole trained-mapping family.

## Reserve ($70 case if E1 passes)

Synthetic-episode scaling: generate thousands of extra who-said episodes from
the corpus builder's own narrative-pattern miner; attacks the N=200 sample limit.

## Honest limits

Local rules handle verbatim anchors + local adjacency; they cannot do deep
credit assignment (paraphrase-invariant matching beyond mined concepts,
syntactic transformation). Escalation ladder if G-T1 fails: synthetic episodes
first, then a global-scalar error/RPE third factor (most biologically defensible
supervised option, still synapse-local). If both fail: program-grade conclusion
that Bible-QA is out of reach for this architecture class under the
neuron-derived constraint set — clean termination, evidence-backed.

## Week-1 plan

D1 freeze trained-inter-space-preregistration.md -> D2-3 hebbian_projection.py +
episode builder + red tests -> D4 run E1 once -> D5 mechanical verdict.
