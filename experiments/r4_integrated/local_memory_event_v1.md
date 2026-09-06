# Fixed neuronal byte-event integrated return

Status: affected QA passed; prospective before source freeze or cost/acquisition.
The [byte-event interface](../../docs/research/local-token-event-interface.md)
and completed host preflight earn this integration check, not a capability claim.

All five candidate arms use one fixed ordinary 32-neuron encoder, private seed+4,
scalar fraction 0.5, reset per event. Ordered event bytes use the existing input
codes; only the final encoder activity reaches the writer/query controller.
Cap-16 ASCII word/whitespace runs and singleton other bytes preserve exact raw
bytes. Existing EOS256 is encoded once separately per document/question. There
are no new EOT, length, answer-position or parsed-role features, encoder updates,
encoder eligibility or memoized token vectors. Purge all transient event/raw
buffers before public ingestion returns. Fixed recurrence is not guaranteed
injective or compositionally readable by the downstream learner.

All six arms retain the same 32/16/16 multiscale writer, current local query/value
credit and byte/EOS LMS output. The immediate current_tagged comparator is an
explicit experiment-only a1ab5ee v4 byte-input learner, fresh seed733; it is not
the old scalar reference or a learned checkpoint warm start. Its full trajectory
must reproduce the authenticated multiscale tagged run. No main migration branch
or additional seventh arm is added. The only extra candidate tensors are fixed
event_encoder.weight/recurrent/bias, 8,320 float32 bytes per model. The writer-rate
vector is now common to all six. Common initial tensors must match exactly.

Writer/maps frozen uses the same candidate encoder and writer prior. It measures
useful learned writing beyond that prior. Frozen and writes-disabled also share
the encoder; their prior byte-query trajectories need not match because questions
are eventized too. Encoder parameters remain fixed in every arm. Separate archived
byte fixtures preserve old tracer/reference QA; the old raw-byte-index tracer
must never consume a shorter event sequence. No event endpoint tracer is added.

## Fixed workload, supervision and budget

Preserve the exact six-arm order, 1,024 lessons per arm, local rate .01, seed733,
raw source-assignment roster, twelve retained memories and 8/4/4/4 final strata.
First attempts precede three teacher byte/EOS offers. All 6,144 lessons retain
18,432 teacher offers, 15,360 enabled decoder opportunities, 120 final actual
Python actions and two motor demonstrations. New-value pairs remain previously
exposed development material; teachers do not count as autonomous action recovery.

One fresh disposable cost phase uses four selected lessons per arm, the same six
capped predictions and complete export/source-free/action path. Discard all cost
weights. Preserve twice complete lesson/cap projection, twice other cost and
30-second allowance, within 900 seconds/2,048 MiB. Root owns native preparation,
absolute-path UTF-8 launch, independent audit, admission and the one conditional
fit. No count reduction or host preflight result substitutes for this fresh cost.

## Separate input clocks and host overhead

Record raw input bytes and string EOS symbols separately from actual encoder
steps and writer/query controller events. Candidate encoding advances once per
raw byte plus EOS; its controller advances once per segment plus EOS. The byte
reference has zero encoder steps and one controller step per raw byte plus EOS.
Record all document lists, both acquisition query traversals, final memory export
including unrelated writes, final query/distractor repetitions and capped cost
queries. Generated output bytes/teacher residuals retain their own unchanged counts.

The legacy report names document_byte_steps/query_byte_steps, where retained,
mean raw bytes plus explicit string ends, never event-controller ticks. Nested
input_work receipts give actual clocks by arm/stage and identify writer versus
query work. The input clock changes; leaks are not rescaled by event length.

Count derivation may perform an additional host-only segmentation of each input,
explicitly recording accounting_host_segmentations/accounting_host_bytes. This
does not call the encoder or model again. Repeated identical query counts may be
multiplied only when the actual repetitions execute; count this accounting pass
once and actual encoder/controller steps at their executed multiplicity. Production
segmentation work is recorded separately. All counting/copy/hash costs are included
in enclosing lesson/cap or export/worker timing and the outer budget. No wall-clock
speedup or neural information-retention gain is inferred from fewer events.

Worker per-output `seconds` begins after its delay distractor reads, although that
output's `input_work` includes them. The enclosing `worker.package_seconds` and
outer watchdog include every distractor traversal, output and accounting pass.
Do not compare per-output time against its whole input-work count as a step rate.

The report also distinguishes document-controller events from three operation
opportunities, including acquisition and final ingestion. `memory_write_event_opportunities`
excludes writes-disabled. `writer_controller_eligibility_events` excludes frozen
and writer/maps-frozen; writes-disabled still advances these local cell traces.
`writer_credit_transport_event_opportunities` includes only tagged, aggregate and
current_tagged. Both frozen controls still write persistent memory through their
forward-only path, but omit unused writer eligibility and credit transport.
These counts do not assert a numerically changed matrix or a slow-weight update;
actual lesson update flags retain that separate meaning.

## Decision and QA

Qualification remains tagged8/8 fit,4/4 novel,4/4 delayed,4/4 interference and strict
novel superiority over candidate writer-frozen, frozen, writes-disabled and the
immediate byte reference. Report aggregate/tag advantage separately. A fixed-prior
gain without writer-frozen superiority does not establish learned writing. No
automatic cap/seed/rate/epoch sweep or output-only diagnostic follows failure.

Before freeze, reviewed existing tests must verify fixed finite read-only encoding,
same document/query events, exact shared initialization and archived byte-reference
trajectories, correct five controls, no backward/hidden teacher calls, raw/event
accounting and fresh-process source-free transport. Use public disposable strings,
not full acquisition or novel endpoint probes. Record all failure/recovery receipts
and disposable exposure below. The host-authored occurrence motor remains declared
scaffolding; no general neuronal action policy or protected SWE capability follows.

### Implementation QA receipts

Doctor `.pytest-tmp/dev/doctor-8a1zkuyf/receipt.json` verifies this checkout and
venv. The reviewed existing decoder file passes 144 selected cases in 10.70s
(`.pytest-tmp/event-joined-v1.xml`); three worker cases were excluded. The current
event worker then passes separately in the prepared environment in 6.28s
(`.pytest-tmp/event-worker-v1.xml`). Its pytest cache-write warning does not affect
the successful worker or saved XML. The two unchanged historical workers remain
excluded. No model assertion or process failure occurred. Import-order-only Ruff
fixes follow the first receipt.

The new reference's two dtype fixtures were then tightened from explicit tiny
scalar geometry to the actual 64-cell multiscale/32-input comparator. Only those
two cases were rerun: both pass in 2.18s
(`.pytest-tmp/event-reference-production-v2.xml`). Each compares three sequential
lessons, initial/final tensors, retained states, query contexts, first outputs and
all teacher feedback against a1ab Git source with its original memory dependency.
The experiment runtime uses a static declared archive; dynamic old-source loading
occurs only in this disposable QA. The original tiny-case receipt is preserved.

Encoder-only QA contributes 13 distinct cases in 2.21s
(`.pytest-tmp/event-encoder-unit-v1.xml`, doctor `doctor-ra3lk5gn`). Its separate
exposure record `.pytest-tmp/event-encoder-unit-exposure.json` has SHA256
`9663edf021cb137e3035c8f1a95e6e212162ca7270dd5b8f772a0c3f5bac7dbc`:
71 actual encoder steps (one deliberately rejected nonfinite step), 70 independent
NumPy reference steps, zero writer/query/teacher/output/action calls. Together the
lanes verify 158 distinct affected cases; the two targeted reruns add exposure,
not new distinct cases.

The new/amended integration subset executes 24 archive/reference lessons across
the two parity invocations (72 teacher offers), ten candidate-control lessons
(30 offers), and one complete-accounting lesson (three offers). The control tests
also make ten duplicate ordinary ingestions to check exact read-only recurrence.
The accounting lesson records two document ingestions and exactly two query
traversals: 31 encoder steps, eight writer events and six query events, with no
encoder replay for counting. Other pre-existing disposable fixtures also execute
in the full affected-file receipt; these subset figures are not a claim about
the entire file's exposure.

The fresh worker constructs untrained models, transports twelve empty states per
arm and answers two public questions per arm. Its twelve free calls perform 140
candidate encoder steps and 68 total query-controller events (40 candidate,
28 reference), 36 output decisions, zero document ingestion, teachers or actions.
No admitted TRAIN acquisition or final development roster is executed in QA.

The final reporting-only write/eligibility/credit opportunity sums pass an
independent authored scalar fixture (`.pytest-tmp/event-write-accounting-qa.json`).
It executes the extracted reporting loop on synthetic counts, with zero model
imports/calls or input segmentation. No model QA is repeated for these sums.
