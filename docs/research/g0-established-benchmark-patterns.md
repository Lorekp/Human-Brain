# Established benchmark patterns for the G0 cross-domain gate

Status: primary-source research note and concrete preregistration proposal; no
gate has been generated or spent.

Date: 2026-08-30.

## Recommendation in one sentence

Use two small but nontrivial procedural benchmarks: 512 learning systems and
128 whole-system holdouts in each domain, five fixed learner seeds, an
isomorphic renamed-symbol rerun, stratum-level thresholds, focused
learning/memory/derangement ablations, and a data-only adapter contract.  This
is the smallest design I would regard as credible evidence that a frozen
learning architecture acquired two new ontologies rather than that two adapters
solved them.

The numerical values below are **proposed preregistration choices**, not numbers
reported by the cited benchmarks.  They must be frozen before candidate G0
outcomes exist.

## What established benchmarks already get right

### Whole generated systems and withheld interventions

GeneNetWeaver was built to generate in-silico regulatory-network benchmarks and
was used for DREAM3, DREAM4, and DREAM5.  Its benchmark role is useful here even
though G0 should use a much smaller domain-neutral simulator rather than its
biology-specific kinetic model.  [The GeneNetWeaver paper](https://doi.org/10.1093/bioinformatics/btr373)
describes generated dynamical networks, standard precision/recall and ROC
profiling, and use in the DREAM challenges.

DREAM4 is an especially good procedural precedent: participants received
wild-type, perturbational, and time-series behavior for multiple synthetic
networks, while the bonus round withheld dual-knockout outcomes and asked for
the response to a novel joint intervention.  The official challenge specified
five separate networks per subchallenge, 21-point perturb-and-release time
series, and hidden double-knockout responses ([official DREAM4 challenge
description](https://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.pdf)).
The pattern to copy is not the biological detail; it is **learn from behavior on
several complete systems, then score new interventions and entirely new
systems**.

CausalBench makes the observation/intervention distinction explicit by defining
observational, partially interventional, and fully interventional training
regimes over perturbation data ([official CausalBench repository](https://github.com/causalbench/causalbench)).
G0 should similarly report prediction and intervention slices separately rather
than allowing high observational accuracy to hide weak causal behavior.

### Separate association, intervention, and counterfactual strata

CLadder balances questions across causal graph structures and distinct
association, intervention, and counterfactual query families.  Its graph set
includes chains, forks, collisions, diamonds, mediation, and confounding, and
its deterministic-counterfactual slice is reported separately
([CLadder paper, Appendix A](https://papers.nips.cc/paper_files/paper/2023/file/631bb9434d718ea309af82566347d607-Paper-Conference.pdf)).
The G0 regulatory arm should borrow the **stratification**, not CLadder's text
templates or causal oracle.

CLadder also publishes a “nonsense” variant in which semantically meaningful
variables are replaced with random four-letter words while graph structure is
preserved ([official CLadder repository](https://github.com/causalNLP/cladder)).
That is a direct precedent for an isomorphic symbol-renaming control.

### Controlled interventions and non-overlapping train/test factors

CausalWorld defines evaluation protocols as sets of episodes that vary a chosen
environment property, treats paired episodes differing in one property as
counterfactual variants, and gives environment variables non-overlapping A/B
spaces for training and evaluation ([official evaluation-protocol
documentation](https://causal-world.readthedocs.io/en/latest/guide/evaluating_policy.html)).
The useful pattern is **one declared axis per control plus disjoint factor
support**, not the robotic domain itself.

### Partial observation, persistent objects, memory, and action chains

MiniGrid provides a compact partially observable view rather than the full
world: the standard observation contains a `7 x 7 x 3` image plus direction and
a mission string, while the image tiles encode object, color, and state
([official MiniGrid source](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/minigrid_env.py)).
G0 should retain the partial image and direction but remove the mission string.

The official environments isolate the mechanisms G0 needs:

- `Memory` shows one object before a corridor and later requires choosing the
  matching object after the cue has left view ([official Memory
  documentation](https://minigrid.farama.org/main/environments/minigrid/MemoryEnv/)).
- `DoorKey` requires picking up a key, unlocking a door, and reaching a goal
  through discrete navigation, pickup, and toggle actions ([official DoorKey
  documentation](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/)).
- `KeyCorridor` hides the key in another room and places the target behind a
  locked door, forcing exploration plus persistent object use ([official
  KeyCorridor documentation](https://minigrid.farama.org/main/environments/minigrid/KeyCorridorEnv/)).

BabyAI demonstrates the established practice of arranging the same grid/object
substrate into levels of increasing compositional difficulty; its original
benchmark contains 19 levels and reports both reinforcement-learning and
behavioral-cloning baselines ([BabyAI ICLR paper](https://openreview.net/pdf?id=rJeXCo0cYX)).
G0 should borrow the curriculum/composition pattern but expose **no language
instruction**.

### Seed-disjoint procedural layouts and uncertainty-aware reporting

Procgen evaluates policies on unseen procedurally generated levels and found
that diversity of training levels is necessary to avoid overfitting
([Procgen paper](https://proceedings.mlr.press/v119/cobbe20a.html)).  The
official release reports that its environments commonly need roughly 500–1000
training levels to generalize and exposes `start_level`/`num_levels` as the
level-set identity ([OpenAI Procgen benchmark page](https://openai.com/index/procgen-benchmark/),
[official Procgen repository](https://github.com/openai/procgen)).  This is why
the proposal below uses 512 learning layouts rather than a visually impressive
but memorisable handful.

Gymnasium's environment contract makes reset and action-space randomness
explicitly seedable ([official `Env` documentation](https://gymnasium.farama.org/v0.28.1/api/env/)).
The generator, learner, exploration, and rename seeds should therefore be
separate manifest fields, not one implicit global seed.

Finally, a point estimate over a few stochastic runs is not enough.  The
NeurIPS rliable work recommends interval estimates, performance profiles, and
robust aggregates such as the interquartile mean
([paper](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html));
its official implementation demonstrates 95% stratified-bootstrap intervals
with 50,000 resamples ([repository](https://github.com/google-research/rliable)).

## Common frozen contract

The two domain arms and their renamed replicas start from the **same exact
architecture/source/checkpoint identity**.  Each arm may learn new weights and
persistent state through the already-frozen learning rule; it may not receive a
new encoder, head, planner, primitive, loss, memory rule, or optimizer.  Generic
input/output populations already present at freeze time are assigned raw tokens
by a manifest, not extended for a domain.

The first G0 result tests coexistence of learning mechanisms, not positive
cross-domain transfer.  Therefore each domain/rename arm begins from an
independent clone of the common checkpoint.  Cross-domain sequential transfer
is a later gate.

Freeze these artifacts before generating the adjudicative manifests:

1. architecture source commit and checkpoint SHA-256;
2. cell, plasticity, memory, competition, and update-budget configuration;
3. both generator sources and complete generator configurations;
4. the raw data-only serializer and action/output token assignment;
5. baseline and ablation definitions;
6. learner, generator, rollout, and rename seeds;
7. split sizes, query counts, metrics, thresholds, and gate logic;
8. evaluator source and 50,000-resample bootstrap procedure;
9. dependency lock and environment/container identity.

After freeze, generate manifests once, record their hashes, and make no
candidate-dependent rejection, repair, resampling, or seed replacement.

### Domain-neutral transport

Use one fixed event envelope in both domains:

```text
OBSERVATION(raw byte tensor, shape, time)
ACTION(discrete token and optional raw entity/value operands)
OUTCOME(raw scalar or raw next observation)
BOUNDARY(terminated, truncated)
QUERY(raw target token, answer-slot token)
```

Shape and padding are serialization facts, not semantic features.  The same
generic sensory, action, answer, and memory populations consume the envelope.
The domain may define available raw actions, as every environment must, but the
adapter may not define semantic operations such as `ACTIVATE`, `CAUSES`,
`MATCHING_KEY`, `OPEN_PATH`, or `PLAN`.

### Adapter anti-solver boundary

The candidate process receives only the event envelope over a narrow IPC/file
boundary.  It cannot import or inspect generator/evaluator objects.

The regulatory adapter may expose current binary entity states, the raw
intervention `(entity, value)`, time, and observed consequences.  It may not
expose adjacency, signs, update equations, ancestors, reachability,
topological order, causal effects, counterfactual twins, graph/motif IDs, or
gold answers.

The spatial adapter may expose only MiniGrid's partial `image`, `direction`, raw
action ID, reward, and termination/truncation.  It must drop `mission`, full
grid, absolute agent position, visibility mask, shortest paths, object-tracking
IDs, solver state, codebook inverse, layout/family ID, and oracle actions.

Both adapters must satisfy all of the following before G0:

- deterministic, parameter-free serialization with no fitted state;
- a byte-for-byte round-trip test for permitted fields;
- a negative-field audit showing that every forbidden field is absent;
- a transcript SHA-256 for exactly what the candidate can read;
- a separate evaluator process that sees gold state only after candidate output;
- no domain-specific source dependency reachable from candidate code.

An oracle may prove that generated systems are valid/solvable and provide a
ceiling in the report.  It is generator/evaluator infrastructure, never a
candidate baseline or input.

## Proposed G0-B: signed causal/regulatory systems

### Generator and splits

Use a deliberately abstract lagged Boolean regulatory system, not biological
names or GeneNetWeaver kinetics.

- Every system has 8 binary entities and 10–16 directed edges, no self-edge,
  in-degree 1–3, and one connected underlying component.
- Each edge weight is `+1` or `-1`.  A node's next state is a deterministic
  threshold of signed parent states and a bias in `{-1, 0, +1}`.
- Rejection sampling, performed only by the generator, requires every declared
  edge to change its child under at least one of the 256 possible current
  states, every node to take both values, and every requested query to have a
  unique binary answer.
- Cycles are permitted in the static graph because all influence is from time
  `t` to `t+1`; the unrolled transition remains well-defined.
- Entity names are eight distinct opaque 32-bit tokens.  They contain no
  biology word, ordinal, degree, sign, motif, or split information.

Freeze the following disjoint cohorts:

| Surface | Seed | Systems | Candidate use |
| --- | ---: | ---: | --- |
| generator smoke | 470000 | 64 | validate balance/uniqueness only; quarantined |
| learning | 470001 | 512 | fixed learning schedule |
| adjudicative | 470002 | 128 | whole-system holdout |
| rename bijection | 470003 | same systems | isomorphic full rerun |

No canonical adjacency hash may occur in more than one surface.  Of the 128
adjudicative systems, 64 are structure holdouts from the same generator and 64
are composition holdouts containing either a signed path of length 3–4 joined
to a fork or a nested/converging diamond.  Those complete motifs are excluded
from learning systems, while every individual sign, edge, two-edge chain,
fork, and convergence primitive occurs in learning.

For each learning or adjudicative system, present exactly 40 support episodes:

- 8 unmanipulated trajectories of 8 transitions;
- 32 single-entity clamp trajectories: 8 entities × 2 values × 2 initial
  states, each 8 transitions.

The initial states and intervention tapes are disjoint between support and
queries.  Adjudicative support may update the frozen learner, but each held-out
system starts from an independent clone of the post-learning checkpoint so no
test system teaches another.

Score exactly 48 balanced target-bit queries per adjudicative system:

| Stratum | Queries/system | What is held out |
| --- | ---: | --- |
| next-state prediction | 12 | state/transition rows |
| single intervention | 12 | initial state and intervention tape |
| paired counterfactual | 12 | alternate clamp from the same factual prefix |
| relation composition | 6 | length-3/4 or converging signed effect |
| delayed memory | 6 | causal support followed by 16 irrelevant events |

This yields 6,144 scored queries per learner seed.  Query generation balances
the answer bit within each stratum and records that balance before candidate
execution.

### Regulatory pass thresholds

Across learner seeds `[17, 29, 43, 71, 101]`, require all of:

| Metric | Required point estimate | Required 95% lower bound |
| --- | ---: | ---: |
| all held-out queries | ≥ 0.82 | ≥ 0.78 |
| next-state prediction | ≥ 0.85 | ≥ 0.80 |
| single intervention | ≥ 0.80 | ≥ 0.75 |
| paired counterfactual | ≥ 0.75 | ≥ 0.70 |
| relation composition | ≥ 0.75 | ≥ 0.70 |
| delayed memory | ≥ 0.75 | ≥ 0.70 |

In addition:

- the strongest fixed simple baseline must be beaten by at least 0.10 overall,
  with a paired 95% lower confidence bound above 0.05;
- the renamed arm must have mapped query-by-query agreement ≥ 0.90 and an
  absolute accuracy change ≤ 0.03;
- no adjudicative structure/composition half may fail its relevant threshold.

## Proposed G0-C: non-linguistic persistent-object microworld

### Generator and splits

Pin the MiniGrid source/dependencies, use the standard partial `7 x 7 x 3`
encoded view plus direction, and remove the mission string.  Preserve the
standard seven discrete action IDs and sparse terminal reward; do not add
distance or subgoal shaping.  Use the environment's natural timeout rule,
fixed here as `4 * side_length^2` steps, a convention shown in MiniGrid's
[official environment tutorial](https://minigrid.farama.org/v2.5.0/content/create_env_tutorial/).

| Surface | Seed | Layouts | Candidate use |
| --- | ---: | ---: | --- |
| generator smoke | 470100 | 64 | validate solvability/balance only; quarantined |
| learning | 470101 | 512 | 32 episodes/layout, fixed order |
| adjudicative | 470102 | 128 | four independent rollouts/layout |
| rename bijection | 470103 | same layouts | isomorphic full rerun |

The 512 learning layouts contain 192 `MemoryS11Random`-style layouts, 160
`DoorKey-8x8`-style layouts, and 160 `KeyCorridorS3R2`-style layouts.  They
teach the primitive consequences of movement, pickup, carrying, door state,
toggle, cue retention, and terminal reward, but never the held-out whole
composition below.

The 128 adjudicative layouts are:

- 32 fresh `MemoryS13Random`-style layouts;
- 16 fresh `DoorKey` and 16 fresh `KeyCorridor` layouts;
- 64 generated compositions in which an initially observed cue selects one of
  two keys, the matching key enables a locked door, and the door gates the
  terminal object/goal.  The cue leaves the partial view at least 16 actions
  before the first irreversible choice.

The composition generator may reuse MiniGrid's existing object and action
dynamics but may not communicate its grammar or solver trace.  All 128 layouts
must be pairwise distinct and absent from learning by canonical full-state
layout hash.  An independent oracle verifies solvability before the manifest is
sealed; candidate outcomes may not affect acceptance.

Learning is exactly 16,384 episodes per learner seed (512 × 32), with layouts
visited in the fixed seed-derived shuffle and no validation selection or early
stopping.  Each adjudicative rollout starts from the final learning checkpoint
with no inter-layout or inter-rollout weight leakage; recurrent/working memory
within the episode remains active.

The rename arm applies an independently generated bijection to every object and
color token in **both learning and adjudicative transcripts**, starts from the
same clean architecture checkpoint and learner seed, and is scored after
mapping outputs back.  It does not expose the inverse bijection to the learner.

### Microworld pass thresholds

Four rollouts per layout × 128 layouts × five learner seeds produce 2,560
adjudicative episodes.  Require all of:

| Metric | Required point estimate | Required 95% lower bound |
| --- | ---: | ---: |
| all held-out layout success | ≥ 0.70 | ≥ 0.65 |
| fresh Memory layouts | ≥ 0.75 | ≥ 0.68 |
| fresh key/door layouts | ≥ 0.70 | ≥ 0.63 |
| held-out compositions | ≥ 0.60 | ≥ 0.55 |
| delayed-cue subset | ≥ 0.70 | ≥ 0.63 |
| multi-step intervention subset | ≥ 0.60 | ≥ 0.55 |

Also require the strongest fixed simple baseline to be beaten by at least 0.10
overall with a paired 95% lower bound above 0.05.  The renamed arm must change
overall success by no more than 0.05 and must agree on success/failure for at
least 0.85 of paired rollouts.

## Baselines, ablations, and statistics

### Fixed baselines

Baselines are evaluation comparators; they are not permitted candidate
components.

Regulatory baselines:

1. balanced coin;
2. persistence (`target(t+1) = target(t)`);
3. per-target marginal majority from support;
4. nearest-neighbor lookup over serialized support prefixes;
5. observational correlation predictor that is not told intervention semantics.

Microworld baselines:

1. uniform random action;
2. fixed wall follower;
3. nearest-observation action memorizer;
4. a memoryless feed-forward policy trained for the identical transition
   budget.

A generic GRU policy with a matched update/parameter budget should be reported
as a non-gating competence reference.  The hidden generator oracle is reported
only as a ceiling.

### Focused causal ablations

Run the following from the same checkpoints and on the same manifests:

| Ablation | Required falsification |
| --- | --- |
| learning off | full score drops ≥ 0.15 in each domain and falls below its absolute gate |
| cell/output population off | no better than the strongest non-learning simple baseline + 0.02 |
| working/persistent memory reset each observation | delayed-memory slice drops ≥ 0.20 in each domain |
| fixed representation→consequence/action derangement | intervention/composition slice drops ≥ 0.15 in each domain |

These four tests target the claimed causes directly.  Do not expand G0 into an
unbounded ablation catalogue after seeing results.

### Estimation

Use learner seeds `[17, 29, 43, 71, 101]` everywhere.  Report every seed and
every system/layout result, mean, interquartile mean, and 95% interval.  Compute
50,000 deterministic stratified bootstrap resamples, resampling whole
systems/layouts and learner seeds rather than individual correlated timesteps.
For baseline gaps and rename invariance, preserve paired system/layout identity
inside each resample.

The gate thresholds above apply to the stated means and confidence bounds.
IQMs and performance profiles are mandatory diagnostics but may not rescue a
failed stratum.  Report update events, simulated cell events, environment
transitions, and wall time; do not tune the pass threshold to observed compute.

## Terminal gate logic

`PASS_FROZEN_ARCH_CROSS_DOMAIN_G0` requires:

1. both non-coding domains meet every absolute, confidence, baseline, rename,
   and ablation requirement;
2. all domain/rename arms authenticate the same frozen architecture and
   learning-mechanism identities;
3. the domain-specific computational-primitive delta is exactly zero;
4. adapter audits and transcript hashes are clean;
5. no protected coding DEV/M0 or hidden G0 outcome influenced design.

Do not average one domain's success over the other's failure.  Any failed item
is a localized G0 STOP.  Preserve the spent cohort, repair only a genuinely
domain-general mechanism under a new name, and generate a fresh successor
cohort.  A PASS supports the narrow claim “candidate domain-general learner”; it
does not establish AGI, positive cross-domain transfer, real-biology competence,
or protected coding performance.

## Exact primary/official sources

- Schaffter, Marbach, and Floreano, [“GeneNetWeaver: in silico benchmark
  generation and performance profiling of network inference
  methods”](https://doi.org/10.1093/bioinformatics/btr373),
  *Bioinformatics* 27(16), 2011.
- GeneNetWeaver/DREAM, [official DREAM4 in-silico challenge
  description](https://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.pdf).
- Jin et al., [“CLadder: Assessing Causal Reasoning in Language
  Models”](https://papers.nips.cc/paper_files/paper/2023/file/631bb9434d718ea309af82566347d607-Paper-Conference.pdf),
  NeurIPS 2023, and the [official generator/data repository](https://github.com/causalNLP/cladder).
- Chevalley et al., [official CausalBench repository and training-regime
  contract](https://github.com/causalbench/causalbench).
- Ahmed et al., [official CausalWorld evaluation-protocol
  documentation](https://causal-world.readthedocs.io/en/latest/guide/evaluating_policy.html).
- Farama Foundation, [official MiniGrid source](https://github.com/Farama-Foundation/Minigrid),
  [Memory](https://minigrid.farama.org/main/environments/minigrid/MemoryEnv/),
  [DoorKey](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/),
  and [KeyCorridor](https://minigrid.farama.org/main/environments/minigrid/KeyCorridorEnv/)
  documentation.
- Chevalier-Boisvert et al., [“BabyAI: A Platform to Study the Sample
  Efficiency of Grounded Language Learning”](https://openreview.net/pdf?id=rJeXCo0cYX),
  ICLR 2019.
- Cobbe et al., [“Leveraging Procedural Generation to Benchmark Reinforcement
  Learning”](https://proceedings.mlr.press/v119/cobbe20a.html), ICML 2020, and
  the [official Procgen release](https://github.com/openai/procgen).
- Agarwal et al., [“Deep Reinforcement Learning at the Edge of the Statistical
  Precipice”](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html),
  NeurIPS 2021, and [official rliable implementation](https://github.com/google-research/rliable).
- Farama Foundation, [official Gymnasium environment and seeding
  contract](https://gymnasium.farama.org/v0.28.1/api/env/).
