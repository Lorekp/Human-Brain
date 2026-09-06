# Actual read-path query-credit return

Status: prospective; no cost or acquisition run. 2026-09-06.

The [selected follow-up](../../docs/research/local-memory-localization-followup.md)
owns this single return after the completed frozen-checkpoint localization.
Existing updates had a beneficial finite sign on all tested cases, but no answer
changed. This comparison replaces only query-read credit in the main learner;
it does not assume complete erasure, identical addresses, or an output-only defect.

## Fixed rule and comparator

The repaired learner transports the actual mean LMS read error through retained
memory synapses, the complete instantaneous query normalization and pre-update
query map. Its query-cell signal combines the direct context slice with that
read-path contribution. Query-map traces use raw presynaptic activity, avoiding
a second normalization gain. Writer learning, the query cell's recurrent local
self-path eligibility, LMS learning and input codes retain their existing rules.
All propagation uses pre-update connections; the float64 teacher sum/mean and
single checked cast to upstream float32 remain explicit. This is instantaneous
weight transport and neuronal error broadcasting, not BPTT or a global optimizer.

Use six ordered arms: `tagged`, `aggregate`, `writer_maps_frozen`, `frozen`,
`writes_disabled`, `current_tagged`. The first five use the repaired production
learner. The last is an experiment-only reference to the current rule at
`d724e06183ea7232f59a506febe2f8f0b02b5b1b`: fixed query-error projection,
diagonal query-normalization eligibility and direct query-cell context slice.
It preserves the old learner and changed read-event/read-validation methods in
an explicit reference module; unchanged memory/write helpers are shared only
with source/trajectory parity checks. There is no production compatibility mode,
runtime Git source loading, mutable module-global patch or checkpoint migration.
The reference source and dependencies enter the admitted source manifest.

Start every arm fresh at seed 733: writer seed 733, query seed 734, fixed input
codes from seed 735 and zero float64 LMS output. Shared initial tensor values
must match exactly. The removed 24,576-byte inactive query-feedback buffer is reference-only
state and remains counted there; configuration metadata intentionally identifies
the different rule. No fitted or disposable localization weights initialize this
experiment. Production geometry remains input/query 32, writer/key/read 64,
context 96, fixed three-phase output and all 257 output neurons.

## Workload and evidence boundary

Reuse exactly the [two-byte roster and schedule](local_memory_two_byte_v1.md):
TRAIN `aa, ab, bb, bc, cc, cd, dd, da`, private fixed seed, 1,024 episodes per
arm and rate 0.01. Novel value swaps `ac/ca` and `bd/db` use separately ingested
memories. They are withheld from this TRAIN stream but previously exposed
development combinations. Final checks remain 8 fit, 4 new-value, 4 delayed and
4 interference actions per arm, with twelve retained memories per arm.
Delay consists of 32 read-only distractor questions; it is not neuronal decay
or retention through further teaching. Interference adds unrelated document writes.

Each lesson ingests complete raw documents, removes them from model input,
records one genuine free output, then offers exactly two target bytes and EOS.
Those labels train the byte learner; later execution receipts never act as
teaching signals. Final query-only workers receive only bound weights/memories
and case ID, memory ID, query bytes and distractor-read count. No target, source
document, offline roster, value span or solver enters the model's inference path.
All final free bytes are passed once through the existing host-authored occurrence
motor and real Python executor. Two fixed motor demonstrations are separate.
Neuronal answer bytes plus that supplied action schema do not establish entirely
neuronal action selection, autonomous correction or general software capability.

Qualification requires repaired tagged exactly 8/8 fit and 4/4 in each remaining
stratum, plus strict new-value action advantage over repaired writer-frozen,
fully frozen, writes-disabled and `current_tagged`. Report the aggregate/tagged
comparison separately. An audit pass, lower loss or checkpoint change is not a
qualification. Failure closes this fixed comparison without automatic rate,
seed, exposure or rule sweeps. A useful result returns to learned action selection
and broader persistent tasks under the roadmap; no protected gate opens here.

## Fresh cost admission and artifacts

Before acquisition, one disposable cost workload uses the original four longest
episodes per arm (document bytes, teacher bytes, query bytes, original index tie),
six capped predictions, full final memory export/query/action work and two motor
demonstrations. It counts 24 first attempts, 72 teacher offers, 60 enabled decoder
step opportunities, 18,504 all-neuron residual offers (15,420 enabled), 120 actions
and two demonstrations. All cost weights are discarded.

Recompute the original bound: twice 1,024 times the sum across all six arms of
their maximum complete sampled lesson plus separately measured capped prediction,
plus twice the outer cost elapsed time less sampled lesson time, plus 30 seconds.
Every timing sample must be finite and strictly positive. Require projection
at most 900 seconds and conservative parent-plus-worker/observed tree peak at
most 2,048 MiB. Only a newly source-bound, strictly validated cost result admits
one acquisition under the same 900-second/2,048-MiB watchdog. Old five-arm or
diagnostic admissions cannot transfer. Root owns both launches and saved audits.

Full acquisition counts 6,144 first attempts, 18,432 teacher offers and 15,360
enabled decoder step opportunities: 4,737,024 all-neuron residual offers and
3,947,520 enabled. Report actual nonzero residuals, changed steps, generation
decisions and all ingestion/query traversals. Export seventy-two retained memories,
six final models, query-only requests and 120 real actions; fixed codes, inactive
reference buffers, credits, snapshots and transient events remain counted.

Use absolute output/admission/checkpoint/request paths and the exact parent
`-X utf8 -m experiments.r4_integrated.local_memory_query_credit_v1 cost` prefix.
Set CPU, checkout native cache and single-thread variables before Python imports.
Elapsed durations use the performance counter with bound clock metadata; Unix
deadlines stay separate. Include imports, serialization, source hashing, progress
output every 128 episode indices, child workers and actions in measured budgets.
Admission binds source revision/closure, reference, recipe, packages/native
identity, roster/schedule and launcher ownership. Preserve source start/end hashes,
initial/final weights, first-before-teacher hashes, all teacher counters and
completion artifact hashes for independent saved-only audit.

The new runner reuses suitable explicit-argument roster, schedule, lesson,
capped-generation, memory-export and digest helpers. It owns six-arm factory,
worker, action/count/qualification validation and cost/run orchestration because
the closed predecessor binds five-arm globals internally. Closed code and recipes
remain unchanged; no injected global spec substitutes for explicit identity.

## Engineering QA before freeze

Run doctor first, then reviewed affected fixtures in existing test files. Require
independent scalar read/normalization/update arithmetic, pre-update transport,
finite/frozen behavior, untouched forward predictions and source-bound reference
parity against the old commit. New runner fixtures must check the six-arm roster,
original strict cost arithmetic and source-free worker routing in a fresh process.
Use only tiny disposable examples, not the full schedule or a model fit. Preserve
failed receipts and record actual bounded QA exposure. Root freezes reviewed source
before any cost or acquisition launch.

### Verified engineering receipts

The [independent preflight](../../docs/research/local-memory-query-credit-review.md)
is included in this experiment's source closure. Doctor passes are recorded at
`.pytest-tmp/dev/doctor-7e4fia4c/receipt.json` for this runner work and
`.pytest-tmp/dev/doctor-9o8kovpx/receipt.json` for the core change. Root separately
prepares the native environment at `.pytest-tmp/dev/native-do_1pfzq/receipt.json`.

The owned initial selection in `.pytest-tmp/query-credit-owned-qa-v1.xml` has
nine passes and two fixture failures before model construction: a source-parity
check used a nonexistent class name. Correcting that fixture to enumerate the
actual unchanged classes yields two passes in
`.pytest-tmp/query-credit-reference-qa-v2.xml` (2.03 seconds). The checks compare
the archived learner/read methods and every inherited old memory method against
the declared Git source, then compare complete three-lesson float32/float64
trajectories with independently loaded old Git modules. Source loading is QA-only;
the experiment imports only its declared reference module.

The requested affected-file selection is retained in
`.pytest-tmp/query-credit-full-qa-v1.xml`: 112 passes and four failures in the
generalized historical cost fixture, in 8.09 seconds. The fixture incorrectly
equated the old teacher length with the generation cap. Restoring the historical
ten teacher offers per lesson, while preserving the new three, yields all six
cost cases passing in `.pytest-tmp/query-credit-cost-qa-v2.xml` (1.73 seconds).
Only those cost fixtures were rerun; no model/source correction was necessary.
This verifies all 116 selected cases across the preserved receipts.

One new source-free worker fixture passes in
`.pytest-tmp/query-credit-worker-qa-v1.xml` (6.20 seconds) under the prepared
native environment. Six untrained models each transport twelve empty memories
and answer two query-only requests. All twelve predictions emit `00 00` at the
cap with three decisions; distinct child PID, unchanged weights/states, input
hashes and source closure pass. There is no ingestion, teaching or real motor
action. A pytest cache-write warning does not affect the XML receipt. The source
files were held unchanged during the worker's request/hash window.

Two additional direct repaired-versus-current forward comparisons pass in
`.pytest-tmp/query-credit-forward-qa.xml` (1.68 seconds): equal complete ingested
state, query context and free bytes before teaching in both upstream dtypes.
The runner/decoder selection thus verifies 119 distinct cases, counting the
worker and these two fixtures. Closed predecessor and duplicate worker fixtures
were excluded; the ordinary full file was not rerun after these additions.

The new reference parity fixtures, including their intentional repeat in the
affected-file selection, execute 24 tiny lessons and 72 byte/EOS teacher offers.
Together with the forward and worker fixtures, new parity/transport QA totals
28 tiny ingestions/216 document-byte steps, 100 query traversals/432 query-byte
steps and 44 free generations. The older joined/numerical fixtures in the selected
file have their own repeated disposable exposure; these totals do not substitute
for the future protocol ledger. No full episode schedule or acquired checkpoint
was evaluated. Ruff and diff checks pass. Source and recipe will be frozen before
the first cost launch.
