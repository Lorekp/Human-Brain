# Typed composition: primary-source design for the next learning increment

**Date:** 2026-08-22

**Status:** Design study at exact source base `9ee5fb7fa1277001c1a8e386d6ddfcdb3250923d`; no training or evaluation was run
**Scope:** Public typed-program architecture and public primary literature only. No development-task contents, canonical/private evaluation material, Fly Brain source, or model artifacts were inspected.

## Decision

The smallest informative capability increment is to add one general primitive,
`subtract : Int x Int -> Int`, while keeping the existing recurrent reservoir,
typed prefix decoder, and one-race-per-token inference boundary. Train subtraction
as a scalar operation, retain `map` composition with the already expressible
`add`, and hold out the operator-combinator pairing
`map(bind k - variable-0, variable-0)` for a teacher-free public composition
probe. This is the typed-program analogue of SCAN's "new primitive in a learned
composition" test: every constituent is familiar, but their combination is not
([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html)).

This increment adds one output-token cell and one operation compiler lowering, but
no new type, binder, search procedure, family macro, or semantic decoder. The
current semantic vocabulary contains only `add` and `map`, and the compiler
lowers those literally to Python addition and a list comprehension
([current primitive table](../../src/humanbrain/typed_program.py#L279),
[current literal lowering](../../src/humanbrain/typed_program.py#L1210)).
**Inference:** merely nesting the existing operations would be a weak
composition test because nested additions, including additions nested under
multiple maps, can be algebraically collapsed to an already expressible
single-add program. Reverse subtraction `k - x`, by contrast, is not expressible
by the current `add`/`map` language, while still being a general operation rather
than a benchmark-family macro
([current primitive table](../../src/humanbrain/typed_program.py#L279),
[current compiler](../../src/humanbrain/typed_program.py#L1210)).

Do not combine this capability increment with recurrent-weight learning, a
larger reservoir, a new memory architecture, or a different decoder. Jaeger's
echo-state construction deliberately holds the recurrent transformation fixed
and fits the output map, while the liquid-state framework explicitly separates
input-history state formation from task-specific readout approximation
([Jaeger corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf),
[Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955)).
**Inference:** changing one representational degree of freedom first gives a
causal result: either the present cell state/readout can transfer a primitive
through a familiar combinator, or it cannot.

## 1. Reservoir state and readout stay separate

Jaeger's echo-state property makes the current recurrent state a function of
input history rather than initialization, and his supervised construction trains
only network-to-output weights by linear regression after discarding an initial
transient. The corrected report also limits its simple algebraic guarantees to
the stated sigmoid-network assumptions; it does not make spectral radius below
one a general proof for arbitrary neuron dynamics
([Jaeger corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf),
[official Fraunhofer record](https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02)).

The liquid-state machine makes the same separation in spiking form. A generic
recurrent liquid supplies a high-dimensional, fading-memory state; a memoryless,
task-specific readout maps the current state to an output. Its two distinct
requirements are separation of relevant input histories in liquid state and
approximation of the target by the readout; multiple readouts can use the same
state trajectories for different tasks
([Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955),
[primary ECCC record](https://eccc.weizmann.ac.il/eccc-reports/2002/TR02-022/index.html)).

The current public API already follows that seam: `grow` constructs one recurrent
cell circuit, presents whole token sequences, and freezes it; `complete` restores
the frozen circuit, observes the framed prompt, and feeds each emitted token back
into the same continuing state
([`grow`](../../src/humanbrain/typed_program.py#L1073),
[`complete`](../../src/humanbrain/typed_program.py#L1115),
[`grow_sequence`](../../src/humanbrain/recurrent.py#L803)).

**Inference for this repo:** before increasing cell count, preregister four
train-only state diagnostics on the exact prompt/signature/program framing:

1. **Forgetting:** start the same token stream from two bounded, deliberately
   different admissible cell states and measure state distance after every token;
   freeze a wash-in point at which initialization influence is below a stated
   tolerance. This operationalizes input-history dependence without pretending
   that the Allen GLIF5 circuit satisfies Jaeger's tanh-network theorem
   ([Jaeger corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)).
2. **Semantic separation:** use paired public-training prompts that differ in
   exactly one factor (operation, operand order, container type, or literal) and
   measure cell-state distance at the delimiter and each output-prefix position.
   This is a diagnostic of the liquid separation property, not an inference-time
   classifier
   ([Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955)).
3. **Readout adequacy:** on those fixed states, report target-versus-runner-up
   output current and first-spike margin separately from state separation. A good
   liquid can still have an inadequate readout, because separation and
   approximation are different requirements
   ([Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955)).
4. **Fading-memory horizon:** perturb one early public-training token while
   holding the suffix fixed, then report how its effect decays across later token
   positions. The reported horizon must cover the longest preregistered training
   composition; otherwise add heterogeneous synaptic/adaptation timescales before
   adding neurons. Liquid-state theory relies on fading memory and highlights
   heterogeneous neuronal and synaptic time constants as useful for separating
   time-varying inputs
   ([Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955)).

These are measurements, not new chooser inputs. They may explain a failure, but
they must not be consulted by the compiler or decoder during completion
([current completion boundary](../../src/humanbrain/typed_program.py#L1115)).

## 2. Curriculum and probe construction

SCAN demonstrates why random held-out examples are insufficient. Standard
sequence-to-sequence models reached about 99.8% on its random split, yet the best
model reached only 1.2% when a primitive seen in isolation had to be used in
held-out compositions. The latter split keeps the primitive and composition
rules individually present in training while withholding their cross-product
([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf)).

Lake's later meta-sequence-to-sequence study distributed training across many
small support/query episodes and permuted primitive-to-action assignments. That
procedure solved several held-out SCAN compositions, while still performing
poorly on length extrapolation. The primary lesson for this increment is about
data construction, not importing its external memory or attention architecture:
vary surface assignments/environments during training and test operator
recombination separately from sequence-length extrapolation
([Lake, 2019](https://papers.neurips.cc/paper/9172-compositional-generalization-through-meta-sequence-to-sequence-learning.pdf)).

Dekker, Otto, and Summerfield found that human compositional generalization in
their function-recomposition task benefited when examples were axis-aligned and
temporally correlated, and they modeled the curriculum effect with Hebbian
gating. This is direct evidence about that human task, not a theorem about this
spiking program learner
([Dekker, Otto, and Summerfield, 2022](https://doi.org/10.1073/pnas.2205582119)).
**Inference:** use a fixed factor-aligned acquisition phase followed by a fixed
mixed rehearsal phase; do not choose the switch point from held-out performance.

### Preregistered public ladder

Use a generated Cartesian curriculum rather than hand-writing a few task
sentences. Freeze literal sets, prompt-template sets, task order, seeds, token
budget, and every later rung before the first material run. SCAN's systematic
splits and meta-sequence-to-sequence episodes both make the withheld relation
explicit rather than relying on an ordinary random example split
([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html),
[Lake, 2019](https://papers.neurips.cc/paper/9172-compositional-generalization-through-meta-sequence-to-sequence-learning.pdf)).

1. **Axis-aligned acquisition:** present scalar `add`, scalar
   `subtract(x, k)` and `subtract(k, x)`, and list `map(add(_, k), xs)` in
   separate, fixed blocks. Balance positive/negative/nonzero literals and both
   subtraction operand orientations, and use
   several meaning-preserving public prompt templates per semantic program. This
   is an inference from the axis-aligned curriculum result; the schedule is a
   hypothesis to test, not a claimed biological law
   ([Dekker, Otto, and Summerfield, 2022](https://doi.org/10.1073/pnas.2205582119)).
2. **Fixed mixed rehearsal:** interleave exactly the same three acquired factors
   for a preregistered number of presentations. Never include list subtraction.
   Surface variation should not alter the teacher program, and program/literal
   combinations should be generated mechanically so no benchmark-family label is
   available to the learner. Training across changing but rule-related problems
   is the operative pattern in meta-sequence-to-sequence learning
   ([Lake, 2019](https://papers.neurips.cc/paper/9172-compositional-generalization-through-meta-sequence-to-sequence-learning.pdf)).
3. **Teacher-free composition probe:** ask for list reverse subtraction,
   `map(bind subtract(integer k, variable-0), variable-0)`, using literal and
   prompt-template cross-products absent from training. Every atomic semantic
   choice (`map`, `bind`, `subtract`, literal, variable, and operand order) has
   training evidence, but the `map x subtract` pairing is withheld. This is the
   direct typed analogue of SCAN's held-out primitive-composition design
   ([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf)).
4. **Later rungs, frozen now but not run now:** if the zero-bridge probe fails,
   expose a preregistered small number of list-subtraction bridge examples and
   test on a new disjoint literal/template stratum. Each rung is a new public
   development experiment, never a retry of the same items. SCAN explicitly
   measured generalization as the number of composed examples increased; it did
   not treat training replay as systematicity
   ([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf)).

The promotion result is semantic pass-at-one across all probe cases, with one
fresh checkpoint restore and one completion per prompt. Record syntax/type
validity, compilation, semantic cases, every output-cell race, and the train-label
replay control separately. Do not require the teacher's exact AST when a different
valid program has the same semantics. SCAN evaluates novel input-output behavior,
and its paper separately checked that failures were not merely greedy-decoder
search failures
([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf)).

Keep length extrapolation out of this gate. Meta-sequence-to-sequence learning
could solve held-out primitive compositions while remaining weak on longer
sequences, so composition transfer and length transfer are distinct claims and
should get distinct frozen protocols
([Lake, 2019](https://papers.neurips.cc/paper/9172-compositional-generalization-through-meta-sequence-to-sequence-learning.pdf)).

## 3. Local eligibility learning: a separate, matched lane

At this exact base, the typed circuit overrides size and fanout but inherits
`readout_learning_rule="ridge"`; training accumulates state/target statistics and
consolidates the output readout after sequence presentations
([typed circuit configuration](../../src/humanbrain/typed_program.py#L307),
[configuration default](../../src/humanbrain/recurrent.py#L90),
[`grow_sequence`](../../src/humanbrain/recurrent.py#L803)).
That is a defensible reservoir-computing baseline because Jaeger's construction
fits only the output map, but it is not a biologically local plasticity result
([Jaeger corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)).

E-prop factorizes a recurrent-network update into a synapse-indexed eligibility
trace and a neuron-specific learning signal. For LIF cells the eligibility
contains a filtered presynaptic train multiplied by a postsynaptic
pseudo-derivative; for adaptive cells the eligibility recurrence also tracks the
hidden adaptation state. Its online approximation still constructs current
output errors and routes them through feedback weights, so e-prop removes
backward-in-time gradient transport but not the learning signal
([Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y)).

Izhikevich's dopamine-modulated STDP model supplies a complementary primary
precedent: pre/post spike timing leaves a synapse-specific trace, and a later
global diffusive dopamine signal converts eligible traces into weight changes.
It supports delayed three-factor credit, but it does not provide a vocabulary
token label locally at every synapse
([Izhikevich, 2007](https://www.izhikevich.org/publications/dastdp.htm),
[paper DOI](https://doi.org/10.1093/cercor/bhl152)).

**Inference for this repo:** keep ridge readout learning for the single-variable
composition experiment, and run a matched non-promoting local-learning lane after
the composition protocol is frozen. In that lane, change only the output-readout
rule: a contact may use its own presynaptic trace, its output cell's membrane/spike
gate, an explicit output-cell teaching signal, and a broadcast surprise/reward
modulator. Do not change recurrent weights in the same experiment
([Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y),
[Izhikevich, 2007](https://doi.org/10.1093/cercor/bhl152)).

Do not call a rule "e-prop for GLIF5" merely because it multiplies an eligibility
array by a teaching signal. E-prop's eligibility depends on the neuron's full
hidden-state recurrence. A GLIF5 claim therefore needs a GLIF5-specific
derivation and finite-difference validation on a tiny public circuit; until then,
"sparse three-factor readout" is the accurate label
([Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y)).

## 4. Cell-only constrained decoding boundary

PICARD is a useful but deliberately limited precedent. It incrementally parses
partial SQL and assigns inadmissible token predictions negative-infinite scores,
preserving accepted model scores. However, the published system first selects
top-k model tokens, participates in greedy/beam search, and its stronger guards
use database-schema facts. Its whole algorithm is therefore broader than this
project's permissible syntax/type mask
([Scholak, Schucher, and Bahdanau, 2021](https://aclanthology.org/2021.emnlp-main.779/),
[paper PDF](https://aclanthology.org/2021.emnlp-main.779.pdf)).

Abstract Syntax Networks likewise show that a grammar can make AST construction
well formed while a learned decoder selects constructors and primitive values.
That supports separating structural admissibility from learned content choice,
but its modular neural decoder is not a component to import here
([Rabinovich, Stern, and Klein, 2017](https://aclanthology.org/P17-1105/)).

The current boundary is already narrower: `_TypedHoleState.allowed` admits
operations, variables, binders, and literals only by the expected type/hole state,
and `complete` passes those allowed token-cell IDs into the cell race
([typed admissibility](../../src/humanbrain/typed_program.py#L618),
[cell-race completion](../../src/humanbrain/typed_program.py#L1115)).
Preserve these invariants for `subtract`:

- the mask may read only the frozen grammar, type/signature state, and emitted
  prefix; it may not read prompt meanings, examples, tests, output values, or
  executable semantics;
- it returns an admissible set, never a ranking, bonus, fallback, repair, or
  constant-folded preference;
- all admissible token cells retain their cell-derived currents and compete in
  one race; and
- at an `Int` hole, the grammar may admit `add`, `subtract`, a compatible
  variable, or a literal, but cells alone choose the operation and the order of
  `subtract`'s operands.

Those restrictions are an inference from the score-preserving rejection part of
PICARD, with PICARD's top-k, beam, and schema-guard mechanisms intentionally
excluded
([Scholak, Schucher, and Bahdanau, 2021](https://aclanthology.org/2021.emnlp-main.779.pdf)).

## Implementation order and stop conditions

1. Add and test the one typed `subtract` primitive and subtraction AST lowering.
2. Freeze the Cartesian curriculum, later bridge rungs, public probe, hashes,
   seeds, resource bounds, and evidence schema before a material training run.
3. Run only the primitive/map training controls, then one teacher-free
   composition probe with no retries or repair.
4. Diagnose a failure with the already frozen state-separation, readout-margin,
   and fading-memory measurements; do not immediately add neurons.
5. Only if those measurements show an inadequate state horizon or separation
   should the next design consider more reservoir cells or heterogeneous
   timescales. Only after the ridge capability result should the matched local
   three-factor readout lane become promotion-relevant.

This order follows the reservoir practice of separating state quality from
readout fit and the compositional-benchmark practice of withholding a complete
factor cross-product rather than tuning against a random example split
([Jaeger corrected report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf),
[Maass, Natschlaeger, and Markram, 2002](https://doi.org/10.1162/089976602760407955),
[Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html)).

## Primary sources

- Herbert Jaeger, *The "echo state" approach to analysing and training recurrent
  neural networks*, corrected report with erratum
  ([PDF](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf),
  [Fraunhofer record](https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02)).
- Wolfgang Maass, Thomas Natschlaeger, and Henry Markram, *Real-Time Computing
  Without Stable States*
  ([DOI](https://doi.org/10.1162/089976602760407955),
  [ECCC report record](https://eccc.weizmann.ac.il/eccc-reports/2002/TR02-022/index.html)).
- Brenden Lake and Marco Baroni, *Generalization without Systematicity*
  ([PMLR](https://proceedings.mlr.press/v80/lake18a.html)).
- Brenden Lake, *Compositional generalization through meta sequence-to-sequence
  learning*
  ([NeurIPS paper](https://papers.neurips.cc/paper/9172-compositional-generalization-through-meta-sequence-to-sequence-learning.pdf)).
- Ronald Dekker, Fabian Otto, and Christopher Summerfield, *Curriculum learning
  for human compositional generalization*
  ([DOI](https://doi.org/10.1073/pnas.2205582119)).
- Guillaume Bellec et al., *A solution to the learning dilemma for recurrent
  networks of spiking neurons*
  ([DOI](https://doi.org/10.1038/s41467-020-17236-y)).
- Eugene Izhikevich, *Solving the Distal Reward Problem through Linkage of STDP
  and Dopamine Signaling*
  ([author-hosted paper and code](https://www.izhikevich.org/publications/dastdp.htm)).
- Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau, *PICARD*
  ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.779/)).
- Maxim Rabinovich, Mitchell Stern, and Dan Klein, *Abstract Syntax Networks for
  Code Generation and Semantic Parsing*
  ([ACL Anthology](https://aclanthology.org/P17-1105/)).
