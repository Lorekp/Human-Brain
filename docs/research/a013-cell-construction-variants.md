# A013 cell-construction variants

Status: DESIGN — nothing implemented or run. Provenance: researched and
primary-source-verified by subagent (transcript-mined after write-time
truncation), authored by orchestrator; verify pass REQUIRED before merge.

## Current construction (verified)

- ONE fitted GLIF5 parameterization (`AllenGlif5Parameters`, cells.py) is
  cloned across all reservoir cells:
  `AdaptiveGlifPopulation(N, cell_parameters)` (recurrent.py:218-219).
- Per-cell diversity today comes from exactly three sources:
  1. threshold offset noise `rng.normal(0.0, 0.025, N)` in the cortex layer
     (cortex.py:78-79);
  2. output heterogeneity current per token cell
     (`_output_heterogeneity_pa`, recurrent.py:28);
  3. random signed sparse recurrent fanout with fixed seed
     (recurrent.py:242+, signs * recurrent_current_pa :268).
- The native kernel steps every cell with identical GlifConfig
  (procedural.rs GlifConfig; lib.rs kernel loops over cells with shared
  config, per-cell arrays only for state).

So the "brain" today is a clone army with three small noise sources. All
computational diversity is wiring, not cells.

## Variants, each tied to a sealed failure mode

### V1 Multi-specimen mixture

Mechanism: pool both pinned Allen MTG fits as subpopulations
(proportion arms 50/50, 75/25, 90/10). Requires per-cell parameter tables
instead of one GlifConfig — native signature change (GlifConfig -> per-cell
arrays or K configs with cell-range partitioning).
Determinism: identity hashes must extend to per-specimen maps
(HUMAN_GLIF5_IDENTITY covers one fit today; add a composition manifest with
per-cell assignment seeded deterministically).
Prediction: two distinct response kernels raise the effective state rank =>
state-separation diagnostic moves off ~0.001; transfer may improve if task
families recruit different subpopulations.
Cost: M (native + checkpoint format v5).

### V2 Parameter jitter within specimen

Mechanism: jitter a SUBSET of parameters per cell with seeded RNG —
candidates ranked by dynamics leverage: th_inf (threshold), asc_amp_array
(adaptation), C (membrane), spike_cut_length. Ranges: no second human
specimen variability data in data/allen/ (two files = the two identities),
so declare +-10% lognormal as declared-assumption arms.
Determinism: same as V1 (seeded assignment manifest).
Prediction: heterogeneity decorrelates race timing => fewer decision
collisions AND higher separation; too-wide jitter degrades margins
(inverted-U prediction — preregister width sweep).
Cost: S/M (python-side first: AdaptiveGlifPopulation accepts per-cell param
arrays even before native support).

### V3 EI balance

Verified finding: the pinned specimens' neuron_config records show
`init_AScurrents: [0.0, 0.0]` and the fits are excitatory-type; there is NO
inhibitory fit in data/allen/. Honest options: (a) derive inhibitory
equivalents by sign-flipping synaptic signs only (recurrent signs already
exist! recurrent.py:268) — i.e., an inhibitory POPULATION via wiring, not
cell type — cheap and honest; (b) declare true inhibitory cell models
out-of-scope until an inhibitory human fit is licensed.
Prediction: balanced E/I via sign structure reduces runaway synchrony,
raises state-rank, improves separation.
Cost: S for (a).

### V4 Axonal-delay heterogeneity

Verified: delays exist in the event runtime as integer step delays
(`recurrent_delay_steps`, procedural.rs:66, minimum 1 enforced). Today it is
ONE global value. Variant: per-edge delay drawn from {1..D_max} seeded;
determinism preserved (delay participates in canonical event ordering keys
already).
Prediction: delay spread desynchronizes races, raises state-rank; risk:
longer delays blur the 80ms eligibility window alignment.
Cost: M (event key schema unchanged; marshalling change only).

### V5 Class subpopulations mapped to regions

Bridges regioned doc: E-population and P-population get different
construction variants (e.g., E = high-heterogeneity broad codebook cells;
P = low-jitter precise race cells). Prediction: division of labor beats
homogeneous split at equal total N.
Cost: M after V1/V2 exist.

## Ranked top-3 by information/cost

1. V3a sign-structure EI (S, tests inhibition hypothesis immediately)
2. V2 threshold/adaptation jitter sweep (S/M, direct separation prediction)
3. V1 dual-specimen mixture (M, uses real human data diversity we already own)

V4/V5 follow once the above establish the separation diagnostic responds to
construction at all.

## Red tests (test-first)

- Per-cell parameter manifest hash: construction determinism under fixed seed
- Identity guard: HUMAN_GLIF5_IDENTITY.config_sha256 still verifies per
  specimen in mixtures
- Checkpoint round-trip with per-cell params (format v5)
- Separation diagnostic sensitivity smoke: synthetic jitter must move the
  metric (guards against a dead metric)
- Race-marginal integrity: margins remain finite-positive under widest jitter
