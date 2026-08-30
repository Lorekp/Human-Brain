# Frozen-architecture cross-domain generalization gate

Status: roadmap protocol; gate unspent.

Date: 2026-08-30.

## Purpose

This gate exists to distinguish a domain-general learning architecture from a
highly engineered programming system implemented with neurons.

Coding remains the primary real-world target, but success on coding alone is not
evidence that the same network can learn arbitrary domains. A system can become
excellent at program synthesis by accumulating coding-specific ontology and
compiler structure without acquiring a general learning algorithm.

The gate therefore asks a stronger question:

> Can one frozen architecture acquire new latent abstractions and compositional
> behavior in structurally different domains using the same learning,
> representation, memory, competition, and credit-assignment mechanisms?

The architecture is frozen before any adjudicative data from the new domains is
seen. Domain-specific computational primitives are forbidden after that freeze.

## Claim boundary

Passing this gate would be evidence for cross-domain learning by one architecture.
It would not establish AGI, consciousness, human-level biology knowledge, broad
real-world competence, or transfer to every possible domain.

Failing one domain is informative: it localizes which supposedly general
mechanism still depends on the programming ontology. A failed domain is not to
be repaired by adding an equivalent domain-specific operator library and then
re-running the same gate.

## The anti-overfitting rule

After the G0 freeze, a new domain may provide only:

- raw observations or text;
- actions/interventions available in that environment;
- rewards, outcomes, or teacher signals permitted by the frozen protocol;
- domain-neutral serialization needed to present those observations to the
  existing substrate.

It may not add:

- biology-specific operators such as `ACTIVATE_GENE`, `INHIBIT_PROTEIN`, or
  `CAUSES_DISEASE`;
- domain-specific type systems or grammars that encode the answer structure;
- hand-written planners, relation solvers, or causal engines;
- task/family IDs used as semantic features;
- hidden evaluation outcomes used to tune the architecture;
- a new memory, attention, credit-assignment, or world-model mechanism solely
  because the new domain requires it.

A change that would be useful in every domain may be proposed only in a newly
named successor gate. The spent G0 cohort remains immutable.

## Generality budget

Every material component is classified before the freeze.

| Component | Expected classification |
| --- | --- |
| fitted-cell dynamics and recurrent synapses | general substrate |
| local three-factor / teacher-gated plasticity | general learning |
| winner-take-all / workspace competition | general control |
| learned successor/world-model mechanism | general learning |
| episodic/persistent memory | general memory |
| generic variable/entity binding | candidate general mechanism |
| generic relation composition | candidate general mechanism |
| Python AST grammar and compiler | coding adapter |
| `List[Int]`, `MAP_BIND`, Python-specific syntax | coding adapter |
| sandbox/test harness | coding environment |
| biology-specific relation solver | forbidden in G0 |

The experiment report must include an architecture-delta ledger. The desired
G0 delta for domain-specific computational machinery is zero.

## Domains

The first G0 execution uses at least two non-coding domains plus coding as a
control. Exact generators, seeds, split sizes, and numerical pass thresholds
must be preregistered before the adjudicative examples are generated.

### G0-A — Programming control

Use the frozen coding stack only as a control that the architecture still
performs its already-licensed learning behavior after the G0 freeze. This arm
is not the generalization claim.

### G0-B — Synthetic biology / regulatory systems

Generate small causal systems from generic entities and directed signed
relations, for example:

- entity A activates B;
- B inhibits C;
- D activates C;
- interventions clamp an entity on/off or alter its state.

Names are randomly generated symbols, not biological vocabulary. Training and
evaluation use disjoint graphs and disjoint symbol assignments. Evaluation must
include unseen chains, forks, competing activation/inhibition, and
interventions/counterfactuals.

The system must learn the needed abstractions from experience. No
activation/inhibition engine is supplied as a domain primitive.

A later, separately frozen follow-on may use real biology text after the
synthetic causal gate is understood; factual memorization is deliberately not
allowed to hide the mechanism question in the first gate.

### G0-C — Non-linguistic causal/spatial microworld

Use a small environment with persistent objects and action consequences, for
example keys/doors, switches, movable objects, containers, locations, and
simple delayed causal dependencies.

The observation channel must not expose task labels or symbolic solution
programs. Evaluation uses unseen layouts, object identities, and causal
combinations. At least one task class must require an intervention sequence or
stateful plan rather than static classification.

## Required controls

Each non-coding domain must include:

1. **Symbol renaming.** Rename all entities while preserving structure. A pass
   that disappears under renaming is lexical memorization, not relational
   learning.
2. **Structure holdout.** Hold out complete systems/graphs/layouts, not merely
   examples from systems seen during training.
3. **Composition holdout.** Test combinations of learned primitive relations
   that did not occur as whole training examples.
4. **Memory control.** Include cases whose answer depends on information no
   longer present in the immediate observation.
5. **Intervention control.** Where the domain permits actions, distinguish
   prediction from intervention/counterfactual reasoning.
6. **Cell/learning ablations.** Frozen cell-off, learning-off, and appropriate
   derangement controls must fail or materially degrade.
7. **No hidden-label tuning.** Evaluation outcomes may not alter the spent
   architecture or domain adapter.

## Metrics

The preregistration must define numerical thresholds, but the report always
includes these dimensions:

- held-out system/task success;
- composition success on unseen relation combinations;
- symbol-renaming invariance;
- intervention/counterfactual success;
- long-delay recall where applicable;
- learning efficiency / number of update events;
- frozen-checkpoint and source identities;
- domain-specific architecture delta;
- comparison with memorization and simple heuristic baselines.

A single aggregate score is insufficient. A model that memorizes biology facts
but fails renamed synthetic networks does not pass.

## Gate logic

`PASS_FROZEN_ARCH_CROSS_DOMAIN_G0` requires, at minimum:

- the identical frozen general-learning architecture is used in all adjudicative
  domains;
- at least two structurally different non-coding domains meet their
  preregistered held-out and compositional thresholds;
- symbol-renaming controls pass;
- the domain-specific computational-primitive delta is zero;
- relevant learning/cell ablations fail as expected;
- no protected coding DEV/M0 result is used to tune the cross-domain gate.

If one non-coding domain fails, the verdict is a STOP with localization. A
successor may change a genuinely domain-general mechanism, but it must freeze
that change before a fresh cross-domain cohort is generated.

## Cross-domain transfer extension

A stronger follow-on asks whether learning in one domain improves another rather
than merely coexisting in separate heads. Examples:

- synthetic causal training improves intervention planning in coding/debugging;
- reading/relational training improves sample efficiency on a fresh causal
  microworld;
- coding training does not catastrophically erase previously learned biology or
  spatial relations.

This is not required for the first G0 pass, but positive transfer would be much
stronger evidence that shared cognitive machinery is being learned.

## Sequencing

G0 is intentionally scheduled before protected coding evaluation expands the
architecture further. The intended order is:

1. finish and freeze the current generic TRAIN semantic engine;
2. finish and freeze canonical TRAIN prompt-to-factor learning and joint
   integration;
3. freeze the common architecture and all domain-neutral learning mechanisms;
4. preregister and spend G0 on fresh non-coding domains;
5. only then proceed to the family-disjoint public DEV gate and canonical
   Simple-SWE M0 freeze/reveal sequence.

The coding DEV surface remains sealed while G0 is designed and executed. This
prevents later coding-specific repairs from quietly becoming part of the
architecture that is claimed to be general.

## Relation to the roadmap

The main roadmap retains the general-learning philosophy and phase ordering.
This document owns the detailed G0 protocol so the roadmap does not accumulate
experiment-level procedure.
