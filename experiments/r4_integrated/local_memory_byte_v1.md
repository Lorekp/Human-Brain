# Local online memory and byte-action development

Status: preparation, 2026-09-06. No cost admission or capability run yet.
This is a new development candidate. The withdrawn BPTT/Adam recipe and its
admission remain unchanged and do not authorize this run.

## Failure, hypothesis and scope

The persistent forward memory and byte decoder have working interfaces, but
their original slow-weight fitting violated the user's local-learning policy.
The new local cell and output updates pass arithmetic and state checks. Useful
local learning of document writing and question addressing remains unproved.

Test whether local eligibility, neural memory-address tags and fixed neuronal
feedback can acquire writing representations that retain previously unseen
values and emit their exact bytes after source removal. Use the equations and
event clocks in [local writer credit](../../docs/research/local-writer-credit.md).
These are approximate learning rules, not the full recurrent gradient.

Acquisition uses explicit byte supervision. Each lesson has a genuine read-only
first byte attempt before a separate teacher-forced correction. Final generated
values enter the existing Python editing motor and execute in real children.
This tests supervised acquisition followed by action transfer. It does **not**
establish learning from execution feedback or autonomous recovery.

Executing every acquisition attempt would require 5,222 children including the
final checks and motor demonstrations. Twice the prior measured maximum action
package time alone projects about 1,262 seconds, exceeding the 900-second ceiling
before local learning. Choose byte-only acquisition before admission; do not
quietly omit executions from a claimed action-feedback experiment.

## Fixed learning configuration

Use one seed, 733; byte embedding 32, writer 64, key/value populations 64,
query cells 32, decoder cells 64, context 96. All recurrent populations use the
shared leaky tanh rate-cell dynamics. The decoder's fixed byte embedding supplies
document and query inputs too; byte 256 terminates a document/query, while
decoder BOS 257 is input-only. No pretrained encoder or LLM enters the learner.

Use learning rate 0.01 for each local update, with no clipping, optimizer,
backward call, rate sweep or implicit replay. This rate is a prospective design
choice, not selected by capability scores. The fixed feedback projections and
their seeds are serialized with the model. Local synapse updates receive only
their own traces and declared neuron-level signals.

| Arm | Slow learning and fast memory |
| --- | --- |
| tagged | Local cell/map learning with address-tagged overwrite eligibility |
| aggregate | Same learner, scalar overwrite eligibility without key/query tags |
| writer_maps_frozen | Controller and write key/value/strength maps fixed; query map/cells and decoder learn |
| frozen | All slow parameters fixed; document-triggered fast writes remain |
| writes_disabled | Same local feedback schedule, but fast synapses and write credit stay zero |

All arms start with identical forward weights and fixed feedback. The aggregate
rule is `E <- (1-beta)*E + beta*e`; tagged credit uses the neural key/query rule
from the linked contract. The aggregate arm is a credit comparison, not a
resource-matched memory representation. Count its lower storage explicitly.
Frozen controls receive the same inputs, first-attempt opportunities and teacher
byte/EOS offers. A fully frozen learner does not change slow weights.

## Experience and clocks

Reuse the withdrawn recipe's deterministic **offline authoring** schedule:
256 groups of four episodes flattened into 1,024 sequential lessons per arm.
There are no group-averaged updates. Each episode independently samples two
distinct keys and two distinct values from the eight-item training vocabulary,
then chooses one question. Key-to-value assignments vary across lessons.
The full serialized schedule and its digest are frozen before the run.

One lesson starts with retained slow weights and zero fast memory and traces.
Read both complete raw documents, including terminators, before supplying the
question or any target. Remove raw/token inputs and transient writer activity
from retained state. Query, then greedily generate the first answer, capped at
32 bytes plus the final EOS decision. Record exact bytes and stopping reason.

Only then supply the independently authored correct bytes and EOS as explicit
teacher feedback. The decoder may update after each teacher byte. Accumulate its
before-update, unaveraged context broadcasts, average once over teacher steps,
then apply upstream local updates once. Retire write/query credit after this
lesson. Never rebuild the old memory or recompute old traces after updating
slow weights. The next lesson's document ingestion is fresh counted exposure.

Per arm: 1,024 first attempts, 58,431 document byte/terminator steps, 26,624
query byte/terminator steps per query traversal, and 6,733 teacher byte/EOS
steps. Query recomputation for correction is additional processing and must be
counted. Across five arms there are 33,665 teacher offers; four updating decoder
arms have 26,932 update opportunities. Actual parameter changes are reported
separately. Inference cap is 33 decisions per first attempt, not an observed
decision count. No earlier QA weights are reused.

## Source removal and executable checks

Use the existing twenty-check geometry per arm: eight fit, four new-value,
four delayed and four interference checks, with eight shared memory pools.
The offline roster remains outside learner requests. New values are byte
recombinations of the exposed vocabulary, not independent prose generalization.

Each two-document pool answers both keys from the same exact saved memory.
Fit versus novel pools also ask the same keys with different values, preventing
query-only correctness from satisfying the paired checks. Delayed queries use
the same memories after 32 unrelated read-only queries. Interference extends
the existing novel memories with three unrelated documents; no base reingestion.

A fresh process receives only detached model/plasticity tensors and query-only
requests. Count retained eligibility tags as source-derived state, even though
they are denied as decoder inputs. The worker does not invoke authoring or
teaching. This is a verified data-routing boundary, not an OS secrecy sandbox.
Require exact state and weight equality before/after read-only inference.

Generated bytes alone enter the existing occurrence motor and strict UTF-8
executor. Nonempty valid UTF-8 with EOS may be actuated; other output is recorded
as failure and still receives its declared final action attempt. Lossless
Python quoting is transport, with no answer selection or repair. Expected
answers enter only parent-side scoring after generation. Count all 100 final
actions and the two common motor demonstrations.

## Budget, accounting and decision

Allow one disposable cost admission of at most 900 seconds, then at most one
900-second run if the measured conservative projection and process memory fit.
Total live process peak sum limit is 2,048 MiB. Reuse the reviewed outer launcher
with this new runner/configuration explicitly bound; its 500-ms sampling can
miss brief children. Freeze source identities, exact command, packages, native
library identity, schedule digest and cost receipt before admitting the fit.
Old gradient timing is not evidence for local O(PK) eligibility work.

Measure complete lessons, full-length generation, ingestion, source-free reload,
all final actions and finalization. Count slow weights, fixed feedback, fast
synapses, writer and query/decoder traces and serialized retained bytes. At the
selected dimensions writer/map tags alone use 3,735,808 bytes per stream.
No hidden input cache, optimizer state or unreported per-write history is allowed.

Require tagged 8/8 fit and 4/4 new-value executed exact answers, with a strict
new-value advantage over writer_maps_frozen, frozen and writes_disabled. Report
aggregate comparison separately: superiority over it is required to attribute
an advantage to address tags specifically. Delay and interference preservation
require 4/4 in each lane. Publish every arm and first-attempt/teacher distinction.

A cost rejection, nonfinite learning, failure to fit or lack of useful advantage
closes this fixed candidate. Record the failure and revisit its explanation;
no automatic extra seed, rate, lesson budget, source filter or host answer solver.
This synthetic experiment cannot advance G0/R5/R6 or establish public-exam
comprehension, general intelligence, old-skill retention or autonomous recovery.
