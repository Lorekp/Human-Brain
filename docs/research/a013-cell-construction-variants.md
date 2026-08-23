# A013 cell-construction variants

Status: DESIGN — nothing implemented or run. Provenance: researched and
primary-source-verified by subagent (transcript-mined after write-time
truncation), authored by orchestrator; cross-reviewed against primary
sources (verify pass #2, corrections in place) — see Verification log.

## Current construction (verified)

- ONE fitted GLIF5 parameterization (`AllenGlif5Parameters`, cells.py) is
  cloned across all reservoir cells:
  `AdaptiveGlifPopulation(N, cell_parameters)` (recurrent.py:218-219).
- Per-cell diversity today comes from exactly three sources:
  1. threshold offset noise `rng.normal(0.0, 0.025, N)` in the cortex layer
     (cortex.py:78-79);
  2. output heterogeneity current per token cell, drawn
     `rng.normal(0.0, 0.01, N_tokens)` (named in the state registry at
     recurrent.py:28, initialized at recurrent.py:274);
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
instead of one GlifConfig — CORRECTION (cross-review): the two pinned fits
are different model classes, not two parameterizations of one model —
GLIF3 (503612028, `threshold_dynamics_method: "inf"`, no threshold
adaptation, spike_cut_length 77) vs GLIF5 (523000114,
`three_components_exact`, spike_cut_length 113) per data/allen configs and
metadata. So this needs either a per-cell model-type dispatch in the kernel
or proof that GLIF3 is exactly expressible as GLIF5 with zeroed
threshold-adaptation coefficients (ASC channel count is compatible:
valid_common pins `after_channel_count == 2`, both fits carry two ASC
channels). Native signature change either way (GlifConfig -> per-cell
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

Verified finding: BOTH pinned specimens' neuron_config records show
`init_AScurrents: [0.0, 0.0]` and the fits are excitatory-type
(data/allen/README.md: "Both specimens are spiny human cortical cells";
spiny = excitatory); there is NO inhibitory fit in data/allen/.
CORRECTION (cross-review): a sign-flipped inhibitory subpopulation ALREADY
EXISTS as a wired knob — `RecurrentCircuitConfig.inhibitory_fraction`
defaults to 0.2 (recurrent.py:114) and assigns negative recurrent weight
signs to a random 20% of active cells (signs built recurrent.py:261-266,
applied at :268). Honest options: (a) treat V3a as a preregistered
sweep/tuning of the EXISTING inhibitory_fraction knob (optionally its
assignment rule) rather than new machinery — cheaper than originally
framed; (b) declare true inhibitory cell models out-of-scope until an
inhibitory human fit is licensed.
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

## Verification log

Cross-reviewed against primary sources (verify pass #2, 2026-08-23), worktree
human-brain-a013-cognition-design. Every cited line re-checked in place.

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Clone army: single GLIF5 parameterization cloned across reservoir | CORRECT | recurrent.py:218-219 (`cell_parameters or AllenGlif5Parameters()`; `AdaptiveGlifPopulation(self.config.reservoir_cells, self.cell_parameters)`); class docstring "homogeneous Allen GLIF5 cells" cells.py:301-304 |
| Diversity 1: threshold offset `rng.normal(0.0, 0.025, N)` in cortex layer | CORRECT | cortex.py:78-79 exact |
| Diversity 2: `_output_heterogeneity_pa` per token cell | CORRECT; citation sharpened | name at recurrent.py:28 (state registry); actual draw `rng.normal(0.0, 0.01, self.token_cells)` at recurrent.py:274 — added above |
| Diversity 3: signed sparse fanout, fixed seed; `signs * recurrent_current_pa` | CORRECT | seed recurrent.py:242; targets :243-260; signs :261-266; product exactly at :268 |
| Native kernel steps every cell with identical GlifConfig; per-cell arrays only for state | CORRECT | GlifConfig lib.rs:18; `glif_step_collect(config: &GlifConfig, ..)` loops `for cell in 0..config.cell_count` (lib.rs:49-66) over per-cell state arrays only; procedural runtime carries ONE GlifConfig (procedural.rs:37 CommonConfiguration tuple) |
| V3: `init_AScurrents: [0.0, 0.0]` in both pinned configs; no inhibitory fit | CORRECT | verified directly in data/allen/503612028_neuron_config.json and 523000114_neuron_config.json; excitatory-type supported by data/allen/README.md ("Both specimens are spiny human cortical cells") |
| V3a framing ("signs already exist", cheap new option) | OVERSTATED — fixed | `RecurrentCircuitConfig.inhibitory_fraction` defaults to 0.2 (recurrent.py:114): wired inhibition already exists as a knob; V3a reframed as sweep of existing mechanism |
| V4: `recurrent_delay_steps` procedural.rs:66, min 1 enforced, ONE global value | CORRECT | field in ProceduralConfig procedural.rs:66; `config.recurrent_delay_steps >= 1` in valid_common :1212; single scalar used at :853 and :1155 (`checked_add` into event due_step ordering key) |
| V1 "both pinned fits" = parameter-table-only change within one model | IMPRECISE — fixed | fits are different model classes: GLIF3 503612028 (`threshold_dynamics_method: "inf"`, spike_cut_length 77) vs GLIF5 523000114 (`three_components_exact`, spike_cut_length 113); mechanism text corrected |
| GLIF5 taus: no stale figures | CLEAN | doc's only ms figure is the 80ms eligibility window, matching `eligibility_tau_ms = 80.0` (recurrent.py:100). Fitted values confirmed: membrane tau = R*C = 298.63483 MΩ x 263.16487 pF = 78.59 ms (~78.6 ms); ASC taus (100.0, 33.33) ms (cells.py:119-127, :161-162). No 20ms/200ms figures present anywhere |
| Identity guard: HUMAN_GLIF5_IDENTITY covers one fit | CORRECT | human_cell_identity.py:39-52 — one identity object, model 523000114; GLIF3 has its separate HUMAN_GLIF3_IDENTITY |
| V2: "two files = the two identities" | CORRECT | 503612028 ↔ HUMAN_GLIF3_IDENTITY, 523000114 ↔ HUMAN_GLIF5_IDENTITY |

Corrections made in place: status line (pass #2 recorded); diversity source 2
citation (draw location + magnitude added); V1 mechanism rewritten
(cross-model-class mixture, dispatch-or-expressibility requirement);
V3a reframed around the existing `inhibitory_fraction` knob.

Residual risks:
- "Excitatory-type" rests on the pinned README's "spiny" description plus the
  absence of any inhibitory fit; the pinned Allen metadata files carry no
  explicit electrophysiological E/I label for these specimens.
- V1 cost estimate (M) assumes GLIF3 is exactly expressible inside the
  GLIF5-shaped GlifConfig (zeroed threshold-adaptation coefficients); if not,
  the kernel needs a per-cell model-type dispatch and cost rises above M.
- Ranked top-3 order is unaffected by these corrections (V3a gets cheaper,
  V1 possibly dearer; ranking still holds by information/cost).
