# Human Brain architecture

`Human Brain` is a sibling of `Fly Brain`. The projects share no runtime
package and neither imports the other.

## Scope and current direction

Updated policy pointer: 2026-09-07. The
[roadmap](research/roadmap-reading-to-coding.md#learning-and-action-architecture) owns the
simulated-neuron learning commitment and architecture policy.
Cell equations and GLIF may change. Tokenization and ordinary tools may encode
observations and execute actions; externally supplied reasoning or answers are
assisted-system evidence and cannot establish the requested neuron-only capability.
The main learner now requires local, online cell/synapse updates. Historical
global-gradient and matrix-solve implementations remain documented comparisons;
their presence does not satisfy that current learning constraint.

The descriptions below document existing implementations. Their cell-only and
fidelity constraints apply to those lanes and frozen experiments, not to every
future architecture. Current integration work follows the
[persistent language/action plan](research/persistent-language-action-development-v1.md).

The measured `RecognitionSequenceMemoryLearner` / `RecognitionSequenceSpliceLearner`
pair retains every old numerical prior and adds one private recognition gain
per input-defined event. A fixed canonical self-activity supplies its one-input
local current rule; activation/preparation precedes the current teacher byte.
That same prepared owner's outgoing byte currents and the acting motor receive
their separate declared local updates. The previous shared router theta stays
fixed. Lossless capacity growth provides1024 event owners and2048 factual slots.
Every occupied owner and factual address remains eligible. This remains episodic
prototype recognition, not demonstrated semantic abstraction. Ordinary integration
QA passes; the [sixth broader result](../experiments/r4_integrated/persistent_language_action_v6_result.md)
is partial after a time limit and establishes no fresh transfer.

`TournamentRecognitionMemoryLearner` and its combined action wrapper preserve
those numerical populations while replacing only factual-address competition.
Fixed adjacent comparators drive exact potential/identity copy relays; all
occupied addresses compete with lowest-index ties. This usesN-1 comparisons
and explicit relay stages, rather than the old two-layer all-pairs circuit.
[Ordinary circuit and executed-learning parity QA](research/factual-address-tournament-qa-v1.md)
passes. The changed metadata/work schema adds no learned tensor or external
answer selector; full-workload speedup remains unmeasured.

The implemented successor `DendriticSequenceMemoryLearner` / `DendriticSequenceSpliceLearner`
adds paired basal/apical contacts to byte/EOS cells. Basal contacts recognize
ordered raw prefixes; apical contacts recognize fixed raw-request activity on
the same branch. Source bytes teach basal contacts, and declared correction
bytes teach their own sensory columns with local signed/clipped permanence
updates. All byte columns enter a class/prior/identity tournament. Old neuronal
priors stay fixed while these contacts and the acting heads receive their own
local signals. [Ordinary QA](research/dendritic-request-binding-qa-v1.md) passes.
The [V7 partial result](../experiments/r4_integrated/persistent_language_action_v7_result.md)
finds one exposed exact reading recall,0/4 changed forms,0/16 fresh and0/8
recovery; online loses one prior success that disabled retains. Current apical
contacts see request bytes, while files and feedback enter the inherited prior.
Categorical branch priority makes observation relevance a concrete design concern,
but the saved aggregates do not identify it as the cause of these failures.

The [parent checkpoint store](research/persistent-session-checkpoint-v1.md) binds
both actors, tapes, actual workspace/feedback, reference history and evidence
frontiers to a single safe boundary. It is development/evidence infrastructure,
not neuronal memory or an answer source. An outstanding interrupted intent
prevents automatic replay; V6's missing final state remains unavailable. V7
preserves a clean boundary after185 attempts with authenticated numeric payloads;
the saved review does not decode or restore them. Its measured repeated checkpoint
traffic motivates the [implemented owned immutable-payload path](research/persistent-checkpoint-owned-payload-v1.md).
Windows NTFS read leases block ordinary writes/deletion while new payloads are
owned; later boundaries reuse their authenticated identities with live path
checks. Other supported execution platforms keep full byte checks. Mutable
actor, tape, source, journal and fresh-restore integrity remain checked.
[Ordinary QA](research/persistent-checkpoint-owned-payload-qa-v1.md) passes;
the frozen V7 session is not replayed and no workload speedup is yet measured.

The preceding reader/action pair uses
[factorized local sequence routing](research/factorized-sequence-routing-v1.md).
Fixed byte, adjacent-pair and relative-position populations produce sparse
coincidence activity for every event owner. One shared float64 routing vector
receives direct outcome teaching from its own selected pre-clamp emitted byte;
no downstream derivative or correct-owner label is transported backward.
Teacher-time exact-context reuse/recruitment remains a separate input-only
circuit. Routing, assembly and action updates publish jointly after validation.
The same reader serves language and action payloads with all1024 factual slots
eligible. Existing prototypes are authoritative; sparse views are derived caches
with source and buffer custody checks. Ordinary integration QA passes, while the
broader continuation's measured usefulness remains unestablished.

The released `SequenceAssemblyMemoryLearner` continues the saved reader and
505-slot tape, with a 512-slot context assembly bank shared by reading and
action payloads. Its observation includes complete bounded requests, files and
actual diagnostic bytes. Each offered teacher-prefix context selects or recruits
an assembly before the current output label is supplied. Only the selected
assembly's outgoing currents receive their own local error; inherited B/C/U
and codec tensors remain fixed. Free responses use simulated agreement and
competition over occupied assemblies. This episodic mechanism has no established
generalization claim. The [selected contract](research/context-specific-sequence-assemblies-v1.md)
specifies input geometry, recruitment, causal sequencing and resource limits.

`SequenceAssemblySpliceLearner` owns that same reader, expanded block-simulated
action heads and atomic action-before-teacher custody. It consumes actual
workspace and public execution feedback across continued projects and retries.
No host repair, target-address routing or separate executor fitting occurs.
Ordinary affected QA and the complete v4 saved audit pass. The
[measured continuation](../experiments/r4_integrated/persistent_language_action_v4_result.md)
changes generation but establishes no correct broader neuronal task response.

The preceding `ActivePathMemoryLearner` materializes fixed raw-request/key
and earlier-output/key coincidence activities before learned address cells.
Two fixed threshold layers select among every occupied address at each output
step. Fixed mixed context cells then supply a locally learned byte/EOS output
population, with the old codec retained as a fixed current prior. During
correction, the first mismatch in the actual saved response supplies one direct
output-current clamp using that event's own selected feature. Full offered
teacher prefixes still supply B/C address credit with U held; there are no
all-address U updates. Every proposal is validated before atomic publication.
No error travels backward through fixed features or history. The incremental
prefix helper reuses Boolean fixed-cell activity and applies the same full
float32 normalization at every step. See the
[exact selected equations](research/active-path-output-learning-v1.md).

`ActivePathSpliceLearner` retains this reader and the existing local
action heads through proposal, real execution and subsequent correction. STOP
corrections and teacher-free actions explicitly resolve the nested reader
without inventing a payload teacher. The
[completed third broader session](../experiments/r4_integrated/persistent_language_action_v3_result.md)
preserves the prior model and whole accumulated tape, but establishes no useful
task accuracy. Its state and evidence remain available for continued development.

The following development descriptions retain historical implementation
contracts. In particular, earlier reverse error-transport paths are not the
current learner and are not authorized for new training under the user's
no-backpropagation requirement.

Current local development combines a streaming rate-cell document writer,
explicit fast synapses and retained local credit with byte output. The closed
recurrent output candidate fails useful context composition. Its replacement
candidate, `PhaseContextLMS`, uses a three-neuron clock, generic continuous
phase-by-context multiplication neurons and 257 learned byte/EOS readouts.
Every readout synapse updates from its own input activity and its output
neuron's potential error; no optimizer or historical reverse gradient is used.
The [qualified output result](../experiments/r4_integrated/local_context_lms_v1_result.md)
demonstrates two-byte composition on supplied factor inputs, with a separate
certificate on the learned weights. Clock and representation are explicit priors.

The selected document return owns byte input codes independently of the output
module and connects its pre-update context-neuron teaching signal to the
writer/query cells' existing local traces. The
[integrated fit](../experiments/r4_integrated/local_memory_two_byte_v1_result.md)
now completes with a passing saved audit, but fails useful acquisition and
learned-writing criteria. Nuisance writes influence output while key/value
assignment swaps do not; the tagged and frozen-writer arms produce identical
final answers. Its fresh-weight, two-byte workload and cost admission remain
separate from earlier variable-length experiments. The legacy Python
motor supplies a narrow edit-selection schema; learned neuronal action selection
and learning from execution feedback remain required work.

The current query-credit implementation replaces the fixed random query broadcast with
instantaneous transport through the retained read matrix and query map. It
applies the full normalization error pool once, then uses raw presynaptic
query-map eligibility and adds the indirect read signal to the query cell's
direct context signal. All connections used for transport precede the upstream
updates. Writer broadcasts and recurrent self-path traces retain their local
approximations; this is not differentiation through document or query history.
The [integrated comparison](../experiments/r4_integrated/local_memory_query_credit_v1.md)
keeps an explicit experiment-only copy of the old query rule as a matched arm.
Implementation and arithmetic QA do not establish useful learning.

The current value-credit implementation also supplies each value neuron with its
corresponding read error. Its existing tagged eligibility transports the effect
of that neuron's value synapses through later overwrites, giving a conditional
value-map partial derivative with fixed initial memory and slow weights during
ingestion. Aggregate traces and the other writer pathways remain approximate.
The [writer review](research/local-memory-writer-credit-review.md) defines this
limited claim and the next integrated comparison; binding and interference
remain unresolved. The removed random value buffer's initialization draw is
still consumed to preserve the other feedback matrices exactly.

The writer cell now accepts a fixed scalar or a copied per-neuron update-fraction
buffer. Its default joined 64-neuron bank uses 32 fractions of 0.5, 16 of 0.125
and 16 of 0.03125; other writer geometries require explicit fractions. Query
cells retain scalar 0.5. For neuron j, the forward law is
`h' = h + alpha_j * (tanh(Wx + Uh + b) - h)`. Its own-synapse eligibility uses
`d = alpha_j * (1 - activation^2)` and carry `1 - alpha_j + d * U_jj`;
off-diagonal forward connections remain active while their indirect derivative
paths remain omitted. All writer maps can read all neurons; no identifier parser
or privileged address is supplied. The 256-byte fixed vector adds no retained
activity or credit dimensions. Scalar 0.5 preserves exact predecessor arithmetic
and state keys, verified against frozen source in both upstream dtypes.

The [multiscale comparison](../experiments/r4_integrated/local_memory_multiscale_v1_result.md)
finds a narrow fresh-value gain shared by a frozen writer, with no learned-writing
advantage or interference success. This documents the current implementation,
not scientific qualification. The generic temporal prior changes both average
rate and heterogeneity; it does not demonstrate uniquely heterogeneous benefit
or directly measure improved identifier retention in the completed fit.

## Existing cell implementations

The computational object in the dendritic lane is a sparse simulation of
human-data-fitted cortical cells. A token activates a token cell. Replayed sequences grow local synapses
onto dendritic branches. At recall, matching branches converge from short basal
contexts to specific apical contexts. Distinct continuations determine how much
lower-order current escapes into the next level, preventing a sparse long
branch from discarding reliable shorter evidence. The resulting NMDA-like
plateau currents drive postsynaptic human GLIF3 cells, and a fast-spiking
basket-cell circuit suppresses later competitors. The first pyramidal spike is
the next token.

Before that first spike, the fitted GLIF3 membrane and exponentially decaying
NMDA current form a linear system. The runtime compiles its exact forward-Euler
unit response when the cortex is created, then scales that response for the
active synaptic currents during each race. This changes neither simulated spike
time nor winner; it removes repeated integration of identical subthreshold
dynamics.

Stateful GLIF5 reservoir tokens use separate exact native backends. Python owns
tokenization, direct sparse-topology construction, and checkpoint arrays, but
one Rust C-ABI call or one persistent CUDA cooperative launch advances the
entire biological token interval. Both apply the same human-fitted membrane,
adaptive-threshold, after-spike, reset, and refractory equations over fixed
sparse edges. They have no ANN runtime or dense cell-by-cell recurrent matrix.

Readout learning has an explicit seam with two rules. Exact ridge remains the
default comparison. The scalable `three_factor` rule combines the presynaptic
reservoir eligibility trace, the active postsynaptic output-cell competitor,
and a signed target neuromodulatory signal. Its native update changes only the
target and competitor synapses, stores no optimizer state, and requires no
reservoir-sized normal matrix or solve. Language-model growth freezes either
rule before validation and checkpointing. See `docs/local-readout-learning.md`
for the equation, locality argument, complexity, and recorded limitations.

Developmental growth is an explicit layer over the three-factor rule. A frozen
schedule recruits a prefix of stable reservoir cell IDs, leaving all unborn
GLIF5 membrane, after-spike, threshold, synaptic-current, and eligibility state
at rest. The prefix can be a seeded random permutation or a deterministic wave
through a six-layer sheet. Recruitment mutates fixed-capacity sparse recurrent
arrays in place: existing edges persist, new cells grow outgoing edges, old
cells overgrow edges toward the new cohort, and limited training-time turnover
replaces the weakest local coactivity traces. Final pruning selects normalized
pre/post eligibility evidence or a matched random ablation. A homeostatic
condition scales incoming synapses from postsynaptic activity during training
and conserves incoming absolute strength after pruning.

The schedule itself is ordinary orchestration. Persistent state remains on
cells and synapses: birth order, active mask, cell homeostatic gain, recurrent
targets/weights, structural score/exposure, and three-factor output weights.
The checkpoint format stores those arrays. The transition kernels need no
developmental branch because recruitment changes the sparse arrays they already
consume. CPU reads the live host arrays; CUDA rebuilds its stable reverse CSR
and uploads topology/weights at the explicit mutation boundary. Python, CPU,
and CUDA lifecycle tests remain exact through recruitment, homeostasis,
pruning, and reset.

The external interface is deliberately small:

```python
model = CellLanguageModel.grow(corpus)
prediction = model.predict_next(prompt)
text = model.complete(prompt, max_new_tokens=32)
```

Behind that seam, the implementation owns token/cell identity, structural
plasticity, sparse branch lookup, membrane integration, inhibition, and spike
readout. Tokenization is permitted non-cell plumbing. Repository retrieval,
patch application, and benchmark execution are separate mini-swe-agent
adapters. They do not get access to branch strengths or choose output tokens.
Decoding may stop after the first complete bash action, but the action text
itself is emitted token by token by cell spike races.

## Recurrent scaling layer

The scaling path uses Allen human GLIF5 reservoir cells. Each input token sends
sparse afferent current into the active population; recurrent spikes propagate
over a fixed-fanout topology; exponentially decaying eligibility traces carry
recent cell state into learned output synapses; and the next token is selected
by a race among GLIF output cells. Dormant allocated cells receive no afferent
or recurrent drive and cannot contribute eligibility.

The first reservoir regime was pathologically synchronized: 96 cells produced
only about 1.1 effective state dimensions and fired near 100 Hz. A train-only
dynamics calibration selected sparse afferents, 5 ms token intervals, lower
recurrent current, and a 40 ms eligibility trace. At 96 cells this raises
effective rank to roughly 18-21, reduces mean absolute cell correlation to
about 0.15, and keeps firing near 12-14 Hz. Cell-count conclusions are made
only inside such dynamically matched regimes.

Supported degree-matched topology probes include uniform random connectivity,
single-scale and nested groups, a 1D ring, a 2D sheet, a six-layer sheet, a 3D
volume, an abstract 4D functional lattice, and overlapping mixed membership.
The 4D lattice is a computational dimensionality control, not an anatomical
claim. Calibrated validation v2 found only the layered sheet met its primary
criterion at 96 cells; no structured variant beat uniform connectivity at 192
cells. These are diagnostic populations, not claims about brain-scale
geometry.

## Native simulation boundary

The CPU recurrent hot loop is a Rust `cdylib` loaded through a small Python FFI.
One native call advances the complete token interval, including 100 GLIF5 steps
and sparse recurrent scatter. Deterministic dense Rayon chunks serve high
activity; an exact frontier scans all cells once and skips only provable
zero-to-zero steps. The CUDA backend instead touches every enabled cell and edge
inside one cooperative launch. The frozen serial native oracle, optimized CPU,
and CUDA paths are bit-exact over registered token sequences for every
persistent cell and synapse state array.

Speed claims are activity-qualified. The exact CPU frontier passes 100 tokens/s
at 131,072 cells through roughly 5% updated cells, but not at 20%. CUDA passes
the full-touch gate at 200,000 cells and 144.03 end-to-end tokens/s. Neither
result is accepted as language capability without responsive-cell, firing,
state-rank, and held-out prediction measurements.

The proposed 100-billion-cell scale is an architectural target, not a current
benchmark claim. Every future scale artifact must separately report
addressable, materialized, updated, and spiking cells. A procedural address
space or dormant population receives no capability credit. Reaching that scale
requires 64-bit cell IDs, procedural immutable topology, sparse learned
overlays, tile/event frontiers with no population-wide scan, and distributed
target-owned state. The design, resource extrapolation, and staged acceptance
gates are recorded in
`docs/research/100-billion-cell-architecture.md`.

Inference allocates only linear-sized cell, afferent, and sparse recurrent
state. It does not allocate the training-only dense ridge statistics. The
exact ridge learner is therefore deliberately identified as a separate
`O(N^2)` memory / `O(N^3)` consolidation comparison. The three-factor and
developmental paths remove that training-state bottleneck, although their
predictive quality is still below the small ridge model and remains an active
research target.

## Current layer

The first working layer uses Allen Cell Types human specimen `488392806` and
its fitted GLIF3 model `503612028`. The port retains the artifact's membrane
capacitance, input resistance, threshold, reset, refractory cut, and two
after-spike currents. It is a human-data-fitted point cell, not yet a whole
human brain and not a claim of full Hodgkin–Huxley morphology. The separate
fidelity research track targets the Eyal multicompartment model followed by a
validated Neuron_Reduce/Jaxley port; that track is not a prerequisite for the
roadmap's learning milestones or custom neuron models.

## Cell-only contract

This contract defines the existing strict cell-only lane and claims made under
it. It is not a blanket prohibition on the roadmap's hybrid research. New
components require a separately declared experiment and contribution controls;
they do not retroactively qualify under an earlier cell-only freeze.

- No transformer, dense ANN, pretrained embedding, or ANN runtime participates
  in inference.
- Learned state consists of cell identity, dendritic branches, and local
  synaptic strength.
- A next token must be the winner of a simulated membrane/spike race.
- Tokenization and byte decoding may be ordinary deterministic code.
- SWE adapters may retrieve files and apply emitted edit programs, but
  may not select repairs or rewrite cell output into a different answer.

## SWE episodic layer

For the first end-to-end SWE gate, each teacher interaction is replayed into a
64-token variable-order dendritic memory. At inference, matched context
branches produce recursively discounted synaptic current and start a race among
fitted human GLIF3 token cells; the first spike emits the next token. This layer
contains no recurrent ANN, transformer, pretrained embedding, or external
command selector.

The current SQLFluff checkpoint is deliberately instance-specific. It is a
plumbing and capacity demonstration, not the held-out checkpoint required for
the final Gemma comparison.

The historical typed-decoding gate is deliberately smaller than SWE-bench: one
frozen generic checkpoint must emit one complete function for family-disjoint hidden coding
tasks. The implemented Layer-1 typed prefix language lets recurrent fitted-cell
output races choose semantic program tokens while a frozen deterministic
grammar and compiler provide syntax-only plumbing—without search, retries, or
hidden-test feedback. The historical v1 TRAIN/DEV executables were retired
after the immutable 0/2 DEV result; their protocols and evidence remain frozen
and their bound commits are the reproducibility anchors. The incompatible v4
A0/A1/A2 increment is preregistered but has not been materially run; see
`docs/typed-composition-a012-v1.md`.
The raw-Python/compiled-DSL-agnostic completion seam,
freeze/reserve/anchor/reveal state machine, sealed production runner, and
pass@1 artifact chain are now implemented, but no canonical freeze or
evaluation seed exists. The exact protocol is
in `docs/simple-swe-curriculum.md`; the broader typed decoder design is in
`docs/research/cell-program-decoding.md`. Before that freeze, the public
task/request projection now binds explicit parameter and result types in one
immutable typed signature without exposing family or private-test metadata.
The implemented scalar-only Layer-0
plumbing and its committed-and-read but unexecuted and unscored public
composition definitions are in `docs/cell-program-layer0.md`.
Layer 0 is not a held-out capability result or the future typed DSL.
