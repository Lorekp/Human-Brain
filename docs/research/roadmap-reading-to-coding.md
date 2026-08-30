# Roadmap: from reading to coding to general learning

Status: living decision document. Updated 2026-08-30.
Owner: orchestrator.

## North star

Build a biologically grounded simulated-cell learning system that is useful in
the real world, with **coding as the primary benchmark and application — not as
the definition of intelligence**.

The architecture is not successful merely because it becomes a strong neural
program synthesizer. The stronger target is a system whose **same frozen
learning mechanisms can acquire useful abstractions in domains they were not
designed around**.

Reading and persistent memory remain an intermediate requirement:

> "a network that can't remember the Bible won't remember code."

The project therefore has three simultaneous obligations:

1. **learn and remember** rather than replay or retrieve;
2. **compose and act** rather than merely classify;
3. **generalize across domains** rather than accumulate a coding-specific
   ontology until benchmarks pass.

## How to read this document

This file is intentionally the roadmap, not the lab notebook.

- Current evidence and exact claim boundaries:
  [`docs/benchmark-status.md`](../benchmark-status.md)
- Current implementation/program map:
  [`docs/program-map.md`](../program-map.md)
- Chronological experiment history, retired variants, and failure localization:
  [`roadmap-experiment-history.md`](roadmap-experiment-history.md)
- Detailed frozen-architecture cross-domain gate:
  [`frozen-architecture-cross-domain-generalization.md`](frozen-architecture-cross-domain-generalization.md)
- Compact typed-output engineering protocol:
  [`compact-typed-output-encoding.md`](compact-typed-output-encoding.md)
- Six-family semantic TRAIN gate verdict:
  [`public-typed-semantic-six-family-train-gate-v1-verdict.md`](public-typed-semantic-six-family-train-gate-v1-verdict.md)

Exact hashes, per-run counters, chronological STOP narratives, and retired
mechanism internals belong in those linked records. This roadmap should grow by
**new decisions and gates**, not by every experiment.

---

# Design philosophy

## 1. Reading, coding, and reasoning share the same core problem

The project began with a measured feature-starvation failure: different prompts
collapsed into nearly the same usable state, so the readout had nothing
meaningful to distinguish. Later coding failures repeatedly reproduced the same
general pattern in different clothing: representations could store or replay
examples, yet failed when the system had to bind new contexts, relations, or
compositions.

The core thesis remains:

> **Useful cognition requires differentiated internal state plus learned binding
> between states, roles, relations, memories, and consequences.**

A network that cannot distinguish two verses cannot reliably distinguish two
function signatures. A network that can distinguish them but cannot bind a
question to the right passage, or an instruction to the right semantic factors,
still cannot generalize.

This is why reading is not a detour from coding. It probes the same learning and
binding machinery without allowing a compiler to hide weaknesses.

## 2. Coding is a benchmark, not an ontology

Coding is attractive because success is objective: programs can be executed,
tests can be hidden, actions have consequences, and compositional structure is
rich.

But the project must not quietly redefine intelligence as the ability to use a
hand-built programming ontology. Components such as Python syntax, AST grammar,
`List[Int]`, compiler binders, and a sandbox are **coding adapters**. They may be
useful environments, but their existence is not evidence that the network can
learn arbitrary concepts.

The general-learning claim is therefore deliberately held behind the new
cross-domain G0 gate. See
[`frozen-architecture-cross-domain-generalization.md`](frozen-architecture-cross-domain-generalization.md).

## 3. Prefer learned structure over task-specific structure

When an experiment fails, the default response is not to add the missing answer
as a new primitive.

Allowed direction:

```text
failure
  -> localize missing computation
  -> propose a domain-general mechanism
  -> freeze it on target-independent / self-posed evidence
  -> spend a fresh gate
```

Dangerous direction:

```text
failure on X
  -> add X-specific operator / type / planner
  -> pass X
```

The second path can produce a strong engineered system while moving away from a
general learner.

Every major new component should therefore be classified in a **generality
budget**:

| Component | Roadmap treatment |
| --- | --- |
| fitted-cell dynamics / recurrent synapses | general substrate |
| local / teacher-gated plasticity | general learning candidate |
| winner-take-all / workspace competition | general control candidate |
| persistent / episodic memory | general memory candidate |
| learned successor/world model | general learning candidate |
| generic entity/role/relation binding | candidate general mechanism |
| Python AST grammar / compiler | coding adapter |
| Python-specific types or operators | coding adapter |
| sandbox and tests | coding environment |
| biology-specific causal solver | forbidden for G0 |

The desired domain-specific architecture delta in G0 is **zero**.

## 4. Distributed representation remains a standing constraint

Explicit token identities were insufficient for transfer, but arbitrary random
distribution was not enough either. Distributed representation must be
**structured and causally useful**, not decorative.

No phase should quietly reintroduce one-cell-per-concept identity as the main
learning mechanism. Sparse assemblies, factorized physical populations, role
binding, learned relations, and distributed persistent state are preferred.
Compact engineering representations may record winners, but the scientific
identity must still correspond to the same physical population and race.

## 5. Closed-loop evidence beats static decoding

Static prompt decoders and whole-program continuation repeatedly failed to
compose. The project consequently moved toward:

```text
observe
  -> internal state
  -> physical competition / action
  -> environment changes
  -> observe consequence
  -> learn / replan
```

This principle applies beyond coding. A future biology or spatial domain should
exercise the same causal learning machinery rather than receive a custom static
classifier.

## 6. Ambiguity is a first-class state, not an excuse to leak labels

Several experiments have shown that the visible evidence can legitimately
support multiple hypotheses. For example, four visible I/O rows may not
identify a deeper composition when a cheaper program matches all four.

The correct behavior is explicit ambiguity, abstention, further information
seeking, or a later prompt/memory signal — **not hidden-test leakage or a host
index tie-break**.

This principle is central to metacognition and to domain-general learning.

## 7. Scale follows learning

Million-cell operation and a path toward much larger configured populations are
engineering achievements, not evidence of intelligence. Do not use scale to
compensate for a missing learning mechanism.

Prove useful learning at the smallest size that exposes the mechanism, then
scale it without changing its scientific semantics.

---

# The 11 cognitive properties

The original property table remains the roadmap's cognitive spine. These are not
marketing labels; each needs an operational mechanism and a falsifiable test.

| Property | Current interpretation and dependency |
| --- | --- |
| **P1 Strong recurrence** | Recurrent substrate exists; useful recurrence requires differentiated state and learned re-entry. R2/R3 exercise it directly. |
| **P2 Integrated representations** | Requires binding across learned populations rather than isolated feature banks. R2-R4 and G0 test whether integration survives new contexts/domains. |
| **P3 Global workspace** | Physical competition exists; the critical capability is selective broadcast/re-entry of winners into subsequent computation. |
| **P4 Persistent internal state** | Core R2 obligation. State must remain informative over long gaps without accidental mutation during read-only use. |
| **P5 Metacognition** | Confidence/ambiguity must affect later computation. Explicit STOP/UNKNOWN/AMBIGUOUS states are seeds; self-checking and information seeking come later. |
| **P6 Attention model** | Which context/span/relation is selected must be learned and perturbation-sensitive, not supplied by task identity. |
| **P7 Self-model** | Not credibly exercised by token prediction. Earliest meaningful tests are agent loops that represent their own success/failure history. |
| **P8 Autobiographical memory** | R2 persistent/episodic lane. Must retain events and relationships over long context and later retrieve them by learned cues. |
| **P9 Goal-directed agency** | Coding/debugging and causal microworlds provide act-observe-revise loops. Becomes a formal requirement in later agent phases. |
| **P10 Embodiment / closed loop** | The environment need not be a robot: a sandbox or microworld counts if actions causally alter future observations. |
| **P11 Homeostatic drives / valence** | Activity/turnover homeostasis and useful confidence/valence signals are continuous stabilizers; avoid motivation-theater without a measurable control role. |

The dependency logic remains:

```text
P4/P1 persistent differentiated state
        -> P2 integration
        -> P3/P6 workspace + attention
        -> P5 metacognition
        -> P8 long-term autobiographical memory
        -> P9/P10 agency + closed-loop behavior

P11 stabilizes the stack throughout.
```

P7 becomes credible only once the system has enough persistent history and
agency to represent itself as an actor.

G0 adds a new constraint across this entire table: a property implemented only
inside the coding ontology is not yet evidence that the property is general.

---

# Generalization ladder

The word "generalization" is too easy to overclaim. The project distinguishes:

1. **Replay generalization** — same learned example under harmless perturbation.
2. **Compositional generalization** — new combination of learned factors.
3. **Family generalization** — unseen task family inside one ontology.
4. **Domain generalization** — learn a new ontology/domain with the same frozen
   learning architecture.
5. **Cross-domain transfer** — learning in one domain measurably helps another.

Recent coding work provides evidence at levels 1-2 and some controlled
within-ontology transfer. Level 3 remains protected behind the public DEV gate.
**Level 4 is unproven and is now a required G0 gate.** Level 5 is a stronger
follow-on, not yet required for the first G0 pass.

The project must not describe itself as a domain-general learner before G0.

---

# Current frontier — 2026-08-30

| Lane | Current state | Next hard question |
| --- | --- | --- |
| **Substrate / runtime** | Large native/CUDA execution demonstrated; compact typed outputs pass parity gates; lazy packed proof plans preserve cost-5 physics and make cost 7 fit under the unchanged cap | Can capability mechanisms scale without changing scientific semantics? |
| **R2 reading / persistent memory** | Dense 49-book TRAIN build exists; factorized successor criterion is pinned; fresh source-bound replay is preflight-ready but unspent | Does persistent identity survive the full fresh TRAIN replay under the corrected factorized criterion? |
| **R3 semantic learning / planning** | Generic scalar/List/String breadth passes; the fixed six-row canonical panel stops 3 selected / 3 stopped, localizing explicit scalar ambiguity and two unsupported `List[Int] -> Int` compositions | Can a fresh generic cross-type layer add clamp-sum and add-count without changing the spent panel, caps, or existing routes? |
| **R4 language → semantic factors** | Role-bound factor-workspace V4 protocol is frozen after V3's 43/48 FIT STOP; prompt-selection V2 stops at 16/24 FIT and 0/18 heldout, localizing whole-target aliasing and unsupported output atoms | Does one V4 learner spend pass while a fresh compositional prompt V3 protocol supplies atom/transition support independent of canonical targets? |
| **Joint language + semantics** | Self-posed joint matrix passes 3/3 treatments + 4/4 controls | Does the joint seam survive canonical TRAIN breadth without task/family identity or hidden outputs? |
| **G0 domain generality** | Numerical/seeding/adapter/ablation preregistration is machine-frozen; no common-architecture binding, generator execution, cohort, or outcome exists | Can the frozen architecture learn at least two structurally different non-coding domains with zero domain-specific computational primitives? |
| **R5 family-disjoint code DEV** | Sealed | Does the frozen TRAIN architecture transfer to the 27 untouched public DEV tasks? |
| **R6 Simple-SWE M0-M3** | Sealed | Can prompt-only coding beat the preregistered baselines? |

Exact historical sequences and artifacts are in
[`roadmap-experiment-history.md`](roadmap-experiment-history.md).

---

# Roadmap phases and gates

## R0 — Falsify cheap representations and readouts — COMPLETE

The original reservoir/readout path was tested across feature families,
decision layers, memory stores, and scale. It established several durable
negative results and the central separation-without-generalization diagnosis.

**Decision:** do not retune the old reservoir/ridge family. Preserve its
engineering infrastructure where useful; build learned relational structure
instead.

Historical detail:
[`roadmap-experiment-history.md`](roadmap-experiment-history.md#1-reservoir--readout-era--closed).

## R1 — Fast, auditable simulated-cell substrate — CONTINUOUS

Goal: make the physical mechanisms inexpensive enough to test without allowing
runtime shortcuts to change their semantics.

Requirements:

- exact checkpoint identity and immutable read-only inference;
- explicit physical winner/STOP/UNKNOWN semantics;
- event-proportional work where possible;
- compact storage may replace dense storage only with parity evidence;
- no host candidate execution/ranking hidden inside a neuronal selection claim.

Current status includes successful compact typed-output cutover and substantial
batching/runtime work. Scale beyond the current host remains secondary to
learning.

## R2 — Reading and persistent memory — ACTIVE

Purpose: prove that the system can retain and use relationships over long
natural-language streams without relying on the programming compiler.

Main line:

1. run the fresh source-bound 49-book TRAIN factorized-successor replay from
   empty state;
2. require stable complete successor tuples and stable owner identity per exact
   source/output column while summing actual allocations;
3. keep matching/active diagnostics non-gating;
4. only after a clean TRAIN result design the held-out/C1 opening;
5. proceed to long-context recall, attribution, entity tracking, and arbitrary
   continuation/QA gates.

Protected held-out books and C1 remain sealed.

This lane is scientifically important even if coding advances faster: a coding
system that cannot acquire and retain ordinary text relationships is not the
learner this roadmap is trying to build.

## R3 — Generic semantic learning and world models — ACTIVE

Purpose: learn/combine useful semantic structure without enumerating one concrete
program per hypothesis or baking task families into the neuronal ontology.

Banked positive mechanisms include:

- learned primitive consequences and bounded planning;
- typed AST site/rewrite planning;
- bottom-up programming-by-examples on the first 30 TRAIN tasks;
- compact exact typed physical outputs;
- packed semantic families with selected-only proof materialization;
- scalar same-cap V3 selection;
- lazy proof-plan V4, which preserves the cost-5 physical ledger and selected
  proof identity while reducing the 19-literal cost-7 build to 22,456 relation
  units under the unchanged 30,000 cap;
- first full-typed `List[Int] -> Int` sum/count packed slice;
- construction-certified desired-free `Int -> Int` TRAIN breadth for generic
  signed offset, nonnegative maximum, and nested clamp;
- construction-certified desired-free `List[Int] -> List[Int]` TRAIN breadth
  for generic map-absolute and map-add, with zero scalar-route work.
- construction-certified desired-free `String -> String` TRAIN breadth for
  direct reverse, generic ASCII-class filters, and reverse-after-filter through
  physical class/contains/String-copy semantics and V4 lazy proof plans, with
  causally zero scalar/List-route work.

The six-family TRAIN STOP established the current constraints:

- factorize concrete candidate tiers rather than raise work caps;
- factorize/normalize bound-context semantics rather than silently top-k prune;
- preserve explicit version-space ambiguity when visible evidence is
  insufficient.

### Spent R3 panel and successor

`PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V1` froze all six ordinal-zero row
identities before the first outcome and stopped honestly at three selected / three
stopped. Scalar clamp and both String rows selected. Scalar nonnegative retained
two exact proofs and stayed `STOP_AMBIGUOUS`; the two `List[Int] -> Int` rows
stopped unsupported with zero build work. No row substitution, task/family
evidence, cap increase, DEV read, or hidden host tie-break occurred.

The smallest successor is a separately named generic cross-type breadth layer
for clamp-sum and add-count, followed by a fresh canonical panel version using
the already frozen roster. Preserve V2 bytes, existing routes, work accounting,
selected-only proof materialization, and the honest scalar ambiguity.

## R4 — Learn language → semantic factors — ACTIVE

Purpose: make natural-language instructions constrain generic semantic families
through learned physical factors rather than static prompt patterns.

Current lessons:

- the original canonical 90-prompt manifest is insufficiently diverse after
  semantic normalization;
- simple local factor learning works;
- independent whole-union heads and early local-span learners failed
  compositional/held-out credit assignment;
- causal teacher-gated V5 branch-local learning passes the complete self-posed
  atomic/composition/control chain;
- the first joint self-posed V5 → packed semantic matrix passes.

### Spent R4 factor workspace V3 and successor

V2 is terminal at `STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V2_INPUT_ALIAS`. Two
post-selector 12-cell inputs each demanded both minimum and maximum as different
singletons; seven FIT and three heldout lessons were contradictory before any
electrical control, array allocation, or learning attempt. The fresh V3 protocol
now defines causal input keys without labels/hashes and maps each key to one exact
factor set. Its 12 factor-set classes contain 72 fresh lessons (48 FIT / 24
heldout), exactly four/two distinct causal inputs per class, with zero stripped-key
conflicts. In particular, the shared minimum/maximum assembly teaches
`{minimum, maximum}` through repeated masked WTA, matching the canonical
same-span semantics. Different factors may overlap or nest; a repeated factor
poisons the case and a fifth distinct factor triggers capacity STOP. Candidate
admission must compare exact physical factor sets. The data-only protocol froze
cleanly, but its single learner execution stopped at
`STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_FIT_READINESS`: 43/48 FIT lessons were
exact, while all four minimum/maximum inputs and one once-seen add input
collided. The 68/68 electrical/control preflight passed, immutable read-only
controls stayed clean, and heldout treatment attempts were zero.

V3 is terminal. The fresh V4 protocol now freezes a 5,120-cell physical
content×selector-role conjunction population and 24 independent simultaneous
factor-membership relays over a new 48/24 cohort. Content-only projections retain
six target conflicts while role-bound projections have none, making the repair
causal. Execute it once; do not retry V3 or inspect its heldout treatment rows.

### Spent prompt selection V2 and compositional successor

Factor sets alone cannot identify an AST. The spent
`PUBLIC_TRAIN_PROMPT_SELECTION_FREEZE_V1` binds 90 role-normalized prompt/signature
appearances to 42 physical causal identities and freezes the required root/edge,
operand-orientation, nesting, parameter-role, literal-slot/copy, and terminal
root/e-class relay vocabulary. It stops before learning at
`STOP_PUBLIC_TRAIN_PROMPT_SELECTION_FREEZE_V1_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE`:
the banked annotations expose only span/factor fields and contain zero exact
structural target sets.

The separately frozen compiler-owned public-TRAIN structural annotation capsule
now passes identifiability. It maps all 42 causal identities to 13 generic
structural target sets and 722 abstract relay labels without exposing executable
programs/ASTs/sources, desired/property rows, task/family identity, entrypoint
spelling, audit hashes as neural evidence, or protected surfaces.

Prompt-selection V2 then spent the frozen 24/18 split once. It stopped at
`STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V2_FIT`: 16/24 FIT identities were exact and
all 18 heldout identities stopped. The causal controls all falsified, but its
whole-target 12-lane trace still aliased FIT identities and its FIT-only output
registry made unseen structural atoms unrepresentable. V2 is terminal.

Freeze a fresh V3 protocol with compiler-grammar-owned compositional
role/filler/relation atoms declared independently of canonical targets and a
matched self-posed TRAIN support cohort. Only after that protocol passes may a
new learner attempt the abstract structural relays. Require one
physical prompt-only retained e-class winner with zero desired reads; proof
exposure requires prompt winner, authoritative retained root/e-class, and
selected proof e-class to agree.

After both checkpoints freeze, join them once to the generic packed TRAIN engine
on fixed TRAIN rows and rerun the terminal joint controls. Learning must exclude
desired outputs, programs, family/task IDs, DEV material, and hidden property
rows. These passes establish a canonical TRAIN language-to-semantics freeze,
not family-disjoint transfer or domain generality.

## G0 — Frozen-architecture cross-domain generalization — NEW HARD GATE

Purpose: directly test the concern that coding-specific engineering is replacing
general learning.

**This gate is required before protected coding evaluation is expanded.**
TRAIN-only engineering can continue, but R5 DEV and R6 M0 remain sealed until G0
has an adjudicative result.

Freeze the common architecture after R3/R4 TRAIN mechanisms are ready, then test
the same learning machinery on at least two structurally different non-coding
domains without adding domain-specific computational primitives.

Initial domains:

- synthetic regulatory/biology-like causal networks with renamed entities,
  unseen graphs, intervention/counterfactual tests, and no biology-specific
  relation solver;
- a non-linguistic causal/spatial microworld with persistent objects, unseen
  layouts, action consequences, delayed state, and multi-step interventions.

Coding remains a control arm, not the G0 claim.

Core requirements:

- same frozen learning architecture across domains;
- zero domain-specific computational-primitive delta;
- held-out systems/layouts rather than merely held-out rows;
- symbol-renaming invariance;
- unseen relation composition;
- memory and intervention controls;
- cell/learning/derangement ablations;
- no post-result repair of the spent cohort.

Detailed protocol and claim boundary:
[`frozen-architecture-cross-domain-generalization.md`](frozen-architecture-cross-domain-generalization.md).
Established primary-source benchmark patterns and proposed numerical
preregistration values:
[`g0-established-benchmark-patterns.md`](g0-established-benchmark-patterns.md).
The exact values are now machine-frozen in
`experiments/g0/frozen_cross_domain_g0_preregistration_v1.py` with generation
fail-closed until a separate common-architecture binding authenticates all nine
required artifact classes. This is still an unspent design result.

`PASS_FROZEN_ARCH_CROSS_DOMAIN_G0` would justify calling the architecture a
**candidate domain-general learner**. It would not imply AGI or human-level
competence.

A G0 STOP outranks the desire to rush into coding benchmarks: the next successor
must repair a genuinely domain-general mechanism on a fresh cohort, not add a
biology/spatial adapter and retry.

## R5 — Family-disjoint public DEV transfer — SEALED

The 27 fixed public DEV tasks are the first protected test of whether the coding
architecture transfers beyond TRAIN families.

Opening conditions:

1. a fresh successor to the terminal canonical packed panel passes and freezes;
2. factor-workspace V4 passes and freezes;
3. a fresh successor to terminal prompt-selection V2 passes and freezes;
4. the canonical TRAIN joint matrix and terminal controls pass unchanged;
5. G0 receives its frozen-architecture cross-domain verdict under the same
   common-learning architecture;
6. no DEV desired vector, result, or family-specific patch has entered design.

The DEV gate should pair:

- visible-I/O semantic selection;
- canonical prompt-only factor/semantic selection.

It is spent once. No tuning on DEV.

## R6 — Canonical prompt-only Simple-SWE M0-M3 — SEALED

After the family-disjoint DEV gate, freeze the complete prompt-semantic
checkpoint and run the existing freeze/reveal protocol.

Milestones remain:

- **M0:** beat random completion;
- **M1:** beat n-gram;
- **M2:** beat template heuristic;
- **M3:** beat a tiny neural LM.

The earlier programming-by-examples TRAIN passes do not partially score M0.

## R7 — Real held-out SWE benchmark

Choose and contamination-audit a real benchmark only after the Simple-SWE
ladder demonstrates meaningful transfer. Preserve the same no-tuning-on-test
rules.

## R8 — Fair external-model comparison

Compare against the frozen external baseline specified by the comparison design
(e.g. Gemma-class local model) under matched task access and contamination rules.
The comparison is meaningful only if the preceding generalization gates pass.

## R9 — Broad autonomous coding / agent capability

Long-horizon coding, debugging, self-checking, tool use, memory across tasks, and
persistent improvement become the application frontier. This phase exercises
P5/P7/P8/P9/P10 directly.

It is not the place to discover that the architecture cannot learn outside a
compiler; G0 exists specifically to force that question earlier.

---

# Cross-domain transfer extension

A stronger result than G0 coexistence is **positive transfer** between domains.
After the first G0 result, measure whether:

- causal/regulatory training improves intervention planning on fresh coding or
  debugging tasks;
- reading/persistent-memory training reduces samples needed in a fresh causal
  microworld;
- coding training preserves rather than catastrophically erases previously
  learned non-coding relations.

This would be evidence that the domains share learned cognitive machinery rather
than merely using separate heads behind one runtime.

Do not require positive transfer for the first G0 pass; record it as the next
scientific strengthening step.

---

# Scientific discipline

## Freeze / reveal

- TRAIN mechanisms may be developed only on declared public/self-posed surfaces.
- A spent gate is immutable even when the failure is obviously fixable.
- New mechanisms receive new names and fresh cohorts.
- Protected DEV/eval/held-out/C1 data must not influence architecture selection.

## Host-code boundary

Host code may:

- enforce serialization, syntax, type safety, and sandbox policy;
- materialize a proof only after the authoritative physical winner;
- report/check outputs after selection.

Host code may not, when the claim is neuronal selection:

- execute/rank candidate programs to choose the winner;
- select a task-specific skeleton;
- inject family/task identity as semantic evidence;
- resolve a neural tie by fixed candidate index;
- use hidden evaluation feedback to update the same spent model.

## Domain-adapter boundary

Adapters may expose observations/actions in a stable domain-neutral format. They
must not solve the domain.

G0 explicitly forbids adding the biology/spatial equivalents of the coding
compiler's semantic operators after architecture freeze.

## Claim language

Prefer the narrowest true claim:

- mechanism PASS;
- self-posed composition PASS;
- public TRAIN programming-by-examples PASS;
- family-disjoint transfer PASS;
- frozen-architecture cross-domain PASS;
- canonical prompt-only coding PASS.

Do not collapse these into "general intelligence."

---

# Immediate queue

## Parallel lane A — generic semantic engine

1. Freeze the separately named generic `List[Int] -> Int` cross-type breadth
   layer, then rerun the already frozen six-row roster in a fresh panel version.
2. Preserve explicit ambiguity/version-space states and selected-only proof
   materialization.
3. No family-specific operators or cap increases based on the spent six-family
   results.

## Parallel lane B — canonical language learning

1. Execute the frozen factor-workspace V4 role-binding protocol once; V3 remains
   terminal and its heldout treatment remains unopened.
2. Freeze prompt-selection V3 around compositional role/filler/relation atoms
   and a matched self-posed support cohort; V2 remains terminal.
3. Use the V5 causal teacher-gated/branch-local mechanism; do not restore task,
   family, ordinal, entrypoint spelling, or audit hashes as semantic features.
4. Freeze both checkpoints and the joint TRAIN controls before any DEV desired
   vector is visible.

## Parallel lane C — reading/persistent memory

1. Spend/resume the prepared fresh source-bound 49-book factorized-successor
   TRAIN replay in bounded chunks.
2. Do not import the completed dense build into the adjudicative scorer.
3. Keep held-out/C1 sealed until the corrected TRAIN criterion has a verdict.

## New lane D — domain generality

1. Implement/authenticate the frozen regulatory and spatial generator/adapter
   sources without executing the adjudicative seeds.
2. Freeze the common post-R3/R4 architecture, nine-part artifact binding, and
   generality-budget ledger against the machine-frozen G0 protocol.
3. Generate fresh synthetic regulatory and causal/spatial cohorts only after the
   freeze.
4. Spend G0 before R5 DEV.
5. If G0 STOPs, prioritize the localized domain-general mechanism before adding
   more coding-specific machinery.

## Protected opening order

```text
R3 packed TRAIN breadth freeze
        +
R4 canonical factor workspace
        +
R4 prompt selection + joint TRAIN freeze
        |
        v
G0 frozen-architecture cross-domain gate
        |
        v
R5 paired 27-task family-disjoint public DEV
        |
        v
R6 Simple-SWE M0 -> M3
        |
        v
real held-out SWE / model comparisons
```

This ordering is deliberate. It makes it harder to accidentally optimize the
architecture around coding benchmarks and then mistake that specialization for
a generally learning network.
