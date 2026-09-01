# Roadmap: general intelligence through language, software, memory, and learning

Status: living decision document. Updated 2026-09-01.
Owner: orchestrator.

## North star

Build a **general learning system capable of becoming superhumanly intelligent**:
one that can acquire knowledge, form abstractions, reason, plan, learn continually,
remember over long timescales, revise beliefs, and eventually operate through
arbitrary interfaces.

The immediate research focus is **software, language, coding, reasoning, and
knowledge acquisition**. Vision, robotics, and motor control are explicitly out
of scope for the current roadmap. They may become useful future interfaces, but
the present objective is to solve the cognitive core first.

Coding is the primary application and benchmark because it gives exact feedback,
rich compositional structure, cheap experimentation, and increasingly difficult
closed-loop tasks. It is **not the definition of intelligence**.

The end target is not a neural program synthesizer. It is a learner whose useful
mechanisms survive outside programming and whose architecture can continue to
change substantially as experiments teach us what intelligence requires.

Reading and persistent memory remain important because:

> "a network that can't remember the Bible won't remember code."

The project therefore has five standing obligations:

1. **learn continually** from experience rather than depend on one offline fit;
2. **acquire and retain knowledge** over long timescales;
3. **compress experience into reusable abstractions** rather than memorize every
   case independently;
4. **reason, plan, act, and revise** in closed loops;
5. **generalize beyond the ontology used to develop it**.

---

## What this roadmap should and should not specify

This document specifies **goals, scientific constraints, and evidence gates**.
It should not freeze a detailed architecture.

Current cell populations, memory systems, factor schemas, workspaces, planners,
plasticity rules, representations, and compiler interfaces are **research
hypotheses**. Any of them may be replaced if a better mechanism is discovered.
A roadmap update should not be needed merely because one implementation is
replaced by another.

The roadmap should answer:

- What capability are we trying to obtain?
- What failure would falsify the current approach?
- What evidence would justify a stronger claim?
- What must remain sealed to preserve that evidence?
- Does a proposed mechanism appear broadly useful, or only useful for one task?

Implementation detail, exact experiment chronology, hashes, and retired variants
belong elsewhere:

- Current evidence / claim boundaries: [`docs/benchmark-status.md`](../benchmark-status.md)
- Current implementation map: [`docs/program-map.md`](../program-map.md)
- Experiment history: [`roadmap-experiment-history.md`](roadmap-experiment-history.md)
- Cross-domain gate details: [`frozen-architecture-cross-domain-generalization.md`](frozen-architecture-cross-domain-generalization.md)

---

# Design philosophy

## 1. Intelligence is learning reusable structure

Repeated failures in both reading and coding have pointed to the same broad
problem: storing or replaying examples is not enough. Useful cognition requires
internal states that can be distinguished, related, reused, and recombined in
new situations.

The working thesis is:

> **Intelligence grows by discovering reusable structure in experience and using
> that structure to predict, remember, reason, and act.**

This is deliberately more general than any current implementation.

## 2. Coding is the laboratory, not the ontology

Python syntax, ASTs, types, compilers, sandboxes, and tests are useful parts of a
coding environment. They must not quietly become the definition of the brain's
internal ontology.

When coding performance improves, ask:

> Did the learner become better at acquiring/composing arbitrary structure, or
> did we simply make Python easier for it?

The former is progress toward the north star. The latter may still be useful
engineering, but it is not evidence of general intelligence.

## 3. Architecture is provisional

Do not protect an architecture merely because substantial work has gone into it.
The project is expected to change architecture repeatedly.

A failed mechanism should normally lead to:

```text
localize failure
  -> propose a more general hypothesis
  -> test it on controlled evidence
  -> keep it only if it improves the relevant capability
```

Avoid:

```text
failure on X
  -> encode X's answer more directly
  -> call the benchmark solved
```

The purpose of experiments is to discover the architecture, not to defend it.

## 4. Prefer learned abstractions over hand-supplied abstractions

Some scaffolding is unavoidable while mechanisms are being discovered. But the
long-run direction must move from:

```text
human defines concepts and relations
network learns values inside them
```

toward:

```text
experience
  -> recurring structure discovered
  -> reusable abstraction formed
  -> abstraction becomes a building block for later learning
```

Recursive abstraction formation is a core candidate capability for intelligence.

## 5. Closed-loop learning beats static decoding

The preferred pattern is:

```text
observe
  -> internal state / memory
  -> choose or infer
  -> act / predict / answer
  -> observe consequence
  -> update beliefs or policy
```

Coding sandboxes already provide this loop. Future domains may use different
interfaces without requiring a different cognitive core.

## 6. Ambiguity is information

If available evidence supports multiple hypotheses, the system should be able to
represent uncertainty, abstain, seek more evidence, retrieve relevant knowledge,
or run an experiment.

Do not resolve genuine uncertainty using hidden tests, fixed candidate order, or
host-side semantic knowledge.

## 7. Scale follows useful learning

Large cell counts and fast kernels are enabling technology. They are not a
substitute for a useful learning mechanism.

Demonstrate the mechanism at a scale where it can be falsified, then scale it
without changing what the experiment claims.

---

# Knowledge, memory, and compression

Knowledge is a first-class part of the intelligence target, not merely training
data for a reasoning system.

A sufficiently capable learner with access to a large fraction of public human
knowledge could gain an enormous advantage, but only if it can **retain,
organize, retrieve, generalize, and revise** that knowledge rather than merely
memorize text.

The roadmap intentionally does **not** prescribe whether this ultimately lives
in synaptic state, episodic memory, learned latent structures, external stores,
or a hybrid. Research should determine that.

The capabilities that matter are:

### Acquisition

Learn useful information from text, code, interaction, tools, and future data
sources without requiring full retraining.

### Episodic retention

Preserve particular experiences and source-specific details when abstraction
would lose important information.

### Semantic consolidation / compression

Turn many related observations into shorter reusable models, concepts, rules,
relations, or predictive structures.

A strong learner should not need one independent memory for every superficially
different example when a common explanation exists.

### Retrieval

Current cognitive state should make relevant prior knowledge available. Retrieval
must ultimately be learned and context-sensitive rather than a fixed keyword
lookup.

### Provenance and confidence

Knowledge should retain enough information to distinguish observation, inference,
source, uncertainty, and conflict. Internet-scale knowledge contains errors and
contradictions; superhuman knowledge requires revision, not blind accumulation.

### Revision

New evidence must be able to update or supersede old knowledge without requiring
an architectural reset.

### Transfer

Knowledge and abstractions learned in one context should improve learning and
reasoning elsewhere when the underlying structure is shared.

A useful conceptual target is:

```text
raw experience
     |
     +--> particular memories / provenance
     |
     +--> reusable compressed structure
                  |
                  v
        prediction / reasoning / planning
                  |
                  v
              new evidence
                  |
                  v
         confirmation or revision
```

The current reading/persistent-memory work and the broader learning experiments
should be judged partly by whether they move toward this capability.

---

# Cognitive capability spine

The original 11-property framework remains useful as a checklist, but it does
not prescribe implementations.

| Property | Capability requirement |
| --- | --- |
| **P1 Strong recurrence** | Past internal state can affect later computation in a useful, learned way. |
| **P2 Integrated representations** | Information from different sources/roles can become a coherent reusable state. |
| **P3 Global workspace** | Relevant information can win competition and influence subsequent computation. |
| **P4 Persistent internal state** | Useful state survives long enough to support memory, reasoning, and learning. |
| **P5 Metacognition** | Confidence, uncertainty, error, and missing information can change later behavior. |
| **P6 Attention** | The system learns what information is relevant rather than receiving the answer through routing metadata. |
| **P7 Self-model** | With sufficient agency/history, the system can represent its own capabilities, actions, and failures. |
| **P8 Autobiographical / episodic memory** | Specific experiences can be retained and recalled by learned cues. |
| **P9 Goal-directed agency** | The system can select actions over multiple steps to achieve a goal. |
| **P10 Closed-loop interaction** | Actions alter future observations and learning uses those consequences. |
| **P11 Homeostasis / valence** | Internal signals stabilize learning and prioritize behavior in measurable ways. |

These properties are not a mandatory build order. Their interactions should be
discovered experimentally.

---

# Cross-cutting capability gates

Do not create one gate or module per cognitive property. Use a small number of
architecture-neutral gates to force the weaker capabilities to become real as
the rest of the system matures.

| Gate | Main properties | Required evidence |
| --- | --- | --- |
| **Continuity** | P1, P4, P8 | Across a long fresh workload, retain specific experiences, retrieve them from learned cues, consolidate reusable knowledge, revise contradictions, and continue learning without destructive forgetting. |
| **Autonomous correction** | P3, P5, P6, P9, P10 | Given a novel goal and tools, choose relevant information, act over multiple steps, detect uncertainty or failure, seek/retrieve evidence, revise the plan, and recover without a hidden oracle. |
| **Self-regulation** | P7, P11 | After substantial prior experience, represent useful facts about the system's own capabilities/state/resources and use internal regulation to measurably improve strategy, stability, or learning. |

The gates are staged rather than simultaneous. **Continuity** should mature in
parallel with R2 and knowledge ingestion. **Autonomous correction** becomes a
prerequisite for strong autonomous-agent claims. **Self-regulation** is
intentionally later: forcing a self-model or drives before meaningful memory and
agency exist would mostly create hand-built theater.

For each gate: freeze the relevant architecture, use fresh tasks, require causal
ablations/controls, keep the spent cohort immutable, and treat failure as
localization of a general mechanism. These gates should guide research without
prematurely blocking current R3/R4 experimentation or the G0 falsification work.

---

# Generalization ladder

Claims should distinguish several levels:

1. **Replay** — reproduce a learned case.
2. **Compositional generalization** — recombine learned pieces in a new way.
3. **Within-domain transfer** — solve genuinely new families inside one domain.
4. **Cross-domain learning** — acquire useful structure in a substantially
   different domain without domain-specific cognitive machinery.
5. **Cross-domain transfer** — knowledge or mechanisms learned in one domain
   measurably improve another.

Coding work is currently the main laboratory for levels 1-3. Cross-domain tests
exist to prevent us from mistaking specialization for general intelligence.

---

# Current research focus

## Priority A — Language, coding, and reasoning

This is the main active frontier.

The system should increasingly be able to:

- understand natural-language task descriptions;
- acquire reusable concepts and relations from examples and experience;
- write, inspect, execute, debug, and revise code;
- plan across multiple dependent steps;
- use feedback without brute-force hidden search;
- represent uncertainty and request/retrieve missing information;
- transfer learned abstractions to new task families;
- improve continually rather than reset between tasks.

Current experiments on semantic synthesis, learned prompt structure, factor
binding, planning, and closed-loop coding are **means to this end**, not frozen
architectural commitments.

The current local integration gap and exact implementation status belong in
[`docs/benchmark-status.md`](../benchmark-status.md), not in the roadmap.

## Priority B — Persistent knowledge and memory

Develop and test the ability to ingest long streams, retain particular facts and
relationships, retrieve relevant information later, and consolidate repeated
experience into reusable structure.

Near-term evidence should include:

- long-delay retention;
- source/relationship recall;
- useful retrieval from learned cues;
- resistance to catastrophic forgetting;
- updating knowledge after contradictory evidence;
- measurable compression/generalization rather than independent memorization of
  every example.

The Bible/reading lane is one convenient stress test, not the eventual knowledge
corpus.

## Priority C — Generality guardrails

Maintain cross-domain tests as a falsification tool for coding overfit.

They should constrain claims and reveal architectural specialization, but they
do not need to become the main engineering priority while the software/language
core is still being developed.

Vision and motor competence are not required in the present roadmap. A future
robot should ideally connect sensors and actions to a mature general learner;
robotics should not be used now to distract from solving cognition.

---

# Roadmap stages

These are capability stages, not architectural prescriptions.

## R1 — Auditable learning substrate — CONTINUOUS

Maintain an experimental substrate where learning, memory, competition, and
inference can be measured and falsified.

Important constraints:

- read-only inference should not silently modify learned state;
- causal controls should distinguish real learning from host scaffolding;
- performance optimizations should preserve the scientific result;
- host code should not secretly make the decision attributed to the learner.

The specific neuron models, data structures, and execution backends may change.

## R2 — Persistent memory and knowledge — ACTIVE

Demonstrate that the system can acquire and retain large bodies of information
and later use them productively.

Required progression:

- reliable long-term retention;
- retrieval by learned context;
- relationships and source provenance;
- continual updates without catastrophic destruction;
- consolidation/compression into reusable abstractions;
- reasoning using both retained particulars and generalized knowledge.

The existing reading replay is one experiment inside this stage, not the stage's
architectural definition.

## R3 — General reasoning, abstraction, and world models — ACTIVE

Develop mechanisms that can learn consequences, relations, latent structure,
and multi-step plans from experience.

Evidence should increasingly demonstrate:

- novel composition;
- causal/predictive learning;
- planning without host enumeration of answers;
- abstraction that reduces repeated work;
- uncertainty when evidence is insufficient;
- reuse of learned structure across tasks.

Coding semantics are currently the richest development environment for this.

## R4 — Language, software, and continual coding intelligence — ACTIVE

Combine language understanding, knowledge, reasoning, memory, planning, and
feedback into increasingly autonomous software capability.

Important milestones include:

- robust TRAIN generalization without family-specific patches;
- family-disjoint coding transfer;
- prompt-only coding performance;
- debugging and revision loops;
- persistent learning across tasks;
- knowledge-assisted coding using previously learned documentation, codebases,
  algorithms, and general world knowledge.

Prior adjudicative narrow evidence (`f586e990f9441dacb5d51437dd99d6d5e57adfa5`, result
`494e3a4696f2464cc4ed7d4ba9defd4c614d2fc13a5afc921a7a77741d9ef37c`): in one
frozen synthetic two-action iterator cohort, a single immutable continual lexical
memory sequentially incorporated four host-supplied successful lessons without
replaying or mutating prior checkpoints, promoted both candidate-provided actions
under the two-task/two-source quorum, and on two fresh exact-factor-overlap
paraphrases issued physical recalls whose prebound Rust sources each compiled once,
ran once on a trusted Windows host, and matched an oracle first opened after both
executions. This is lexical association/reuse plus authenticated source custody and
execution—not live consequence learning, source synthesis, semantic understanding,
debugging, transfer, SWE performance, or intelligence.

Current public-development successor (`a364d0a`, result
`692c8545408de2d71b803aaec39f9e392b909f35ccf3f0b524b23a97da9814a6`):
on one exact rooted task and input batch, a promoted prebound Rust action was
selected and executed, received an authenticated public failure, and was inhibited
in an immutable successor; the revised physical race selected the other promoted
action, whose execution passed and produced another immutable successor. A
plasticity-off control preserved the original winner, an unrelated exact-factor
probe retained its action, and two evaluation executions completed before one
oracle callback. The retained external transcript authenticates input, authority,
binding, release, execution, feedback, lesson, and checkpoint custody. This is
same-task binary consequence correction and persistence under exact lexical-factor
reuse—not source synthesis, semantics, diagnosis, generic debugging, transfer,
SWE performance, or intelligence. The next mechanism question was whether two
independent failed-to-successful episodes can consolidate a generic parameterized
edit schema that transfers to a fresh family.

Banked C1 public-development successor (`f2de425`, exact spec
`3dedb36b3ae4c7a916e1626a32b598f9ed1ee7c23cde2b9a1bc10dabea3d6609`): two
independently rooted failed→successful Rust episodes selected and consolidated
one member of a fixed lexical token-edit bank. One source alone physically
abstained; both sources produced a singleton promoted schema. On a structurally
fresh `for`-loop source, that frozen schema bound a new query-local donor and
proposed one identifier replacement without replaying either training source.
Four training sources compiled and ran once each; the fresh proposal was not
executed. The trusted test authenticated its full result in memory but did not
retain the result root/transcript, so the external attestation is explicitly
weaker and records that limitation. This is selection and reuse inside a supplied
lexer/ontology/schema bank—not learned semantics, diagnosis, proposal
correctness, C2 transfer, SWE performance, or intelligence. The next mechanism
question was whether the frozen schema's proposal survived execution and public
consequence on a fresh family without C1 receipt reuse.

Current C2 public-development successor: one spent preregistered cohort rebuilt
the frozen lexical mechanism from two new content-root-distinct
synthetic episodes rather than importing C1 checkpoint or receipt lineage. On a
held-out source, the unedited baseline compiled and failed; the exact schema-3
one-token proposal compiled and passed, with one oracle opening only after both
executions and no retry or alternative execution. This establishes executable
proposal behavior for that frozen mechanism and cohort. It is not consequence
learning because the transfer consequence did not update the checkpoint, nor
does it establish C1→C2 checkpoint-lineage continuation, learned semantics,
diagnosis, generic repair, protected transfer, SWE performance, or intelligence.

The next mechanism question is whether consequences from executed proposals can
change reusable state and improve later fresh behavior on a new unspent cohort.
For the spent C2 cohort, the authenticated terminal is authoritative; a
reservation-only or pending-only interruption state would be indeterminate,
never PASS and never permission to rerun.

Current internal prompt/factor/occurrence representations are provisional.
Replace them if a better learning formulation is found.

## G0 — Cross-domain falsification gate — REQUIRED FOR STRONG GENERALITY CLAIMS

Use at least two non-coding domains to test whether the learned cognitive
mechanisms survive outside the programming ontology.

The detailed protocol is frozen separately. The roadmap only requires that:

- domain-specific cognitive solvers are not added after the relevant freeze;
- whole systems/problems are held out;
- names/symbols can be changed without destroying learned relations;
- new relation compositions are tested;
- learning and memory ablations behave causally;
- failures are treated as evidence about general mechanisms.

G0 is a claim/evidence gate, not a requirement to prioritize vision or robotics.

## R5 — Protected family-disjoint coding transfer — SEALED

Use the protected coding split to test whether the developed software/language
system transfers beyond the TRAIN families.

No tuning on the protected result.

## R6 — Prompt-only coding ladder — SEALED

Run the existing Simple-SWE M0-M3 ladder after the prerequisites for a meaningful
transfer claim are frozen.

The exact milestone baselines remain in the benchmark documentation.

## R7+ — Real-world software intelligence

Progress toward increasingly difficult real repositories, debugging, design,
tool use, long-horizon tasks, and continual improvement.

External-model comparisons matter here, but the ultimate objective is capability,
not benchmark theater.

---

# Research discipline

## Preserve falsifiability

A spent protected/adjudicative gate is immutable. Fixes receive a new experiment
or fresh cohort.

## Keep evaluation surfaces sealed

Do not use DEV/evaluation/held-out outcomes to choose the architecture that will
be scored on them.

## Separate environment adapters from cognition

Parsers, compilers, sandboxes, serializers, and tools can exist around the
learner. They should not silently encode the reasoning result.

## Prefer narrow true claims

Distinguish:

- mechanism success;
- learned replay;
- compositional generalization;
- within-domain transfer;
- cross-domain learning;
- broad autonomous capability.

## Track architecture debt

Whenever a new mechanism is added, ask whether it is:

- plausibly reusable across many domains;
- temporary scaffolding for an experiment;
- or a domain-specific shortcut.

Temporary scaffolding is allowed, but should not become invisible evidence for a
generality claim.

---

# Immediate direction

The agent should **not** interpret this roadmap as an instruction to preserve the
current architecture.

Near-term work should optimize for progress on these questions:

1. **Can the current language/coding system learn the remaining missing
   relationships from causal data rather than host annotations?**
2. **Can learned representations become reusable abstractions instead of growing
   one new mechanism per coding pattern?**
3. **Can persistent memory evolve from exact retention toward useful knowledge
   consolidation, retrieval, provenance, and revision?**
4. **Can coding competence become continual—learning from repositories,
   documentation, executions, and prior tasks without full retraining?**
5. **Do proposed improvements look useful for arbitrary structured experience,
   even while coding remains the primary testbed?**

When deciding between two implementation directions, prefer the one that teaches
us more about these questions, even if it requires replacing substantial existing
code.

Do not broaden into computer vision or robotics merely to appear more general.
Solve language, software, knowledge, memory, reasoning, and continual learning
first; keep cross-domain tests as a guardrail against accidental specialization.

The long-run standard remains demanding:

> A system that can ingest enormous amounts of human knowledge, compress it into
> useful abstractions while retaining important particulars and provenance,
> reason and plan beyond what it has seen, learn continually from consequences,
> revise itself when wrong, and transfer those capabilities to new problems.

That—not any specific current architecture—is the roadmap's target.
