# Local memory/byte preflight review

2026-09-06. Bounded, read-only implementation/scientific review requested by the
orchestrator. This report is the only edited file. No training, timing, tests or
new evaluation exposure was performed. The roadmap, active persistent-memory
plan, writer-credit contract and v1 recipe/result were reread. The canonical
graph root matched this checkout; new learner files were `not_tracked` and motor
files `metadata_changed`, so conclusions use direct source. Mem0 search supplied
development context only, with no result used as scientific evidence.

**Recommendation:** continue the optimized, separately admitted local byte-memory
candidate, preserving its learning rule and controls. There is no acquired
capability result yet. The existing full action path must not be described as
exclusively learned neuronal intelligence: its legacy motor contains a host
derived and host evaluated edit schema. This does not invalidate a properly
scoped test of neuron-generated answer bytes reaching executable edits.

## Prioritized findings

### 1. Separate neuron-generated answers from the legacy action schema

[LocalMemoryLearner](../../src/humanbrain/local_memory_learner.py) feeds raw byte
codes into recurrent cells, synaptic memory and the byte decoder. Its embedding
is randomly initialized and fixed, not pretrained. No external encoder or LLM
enters that path. The final worker accepts detached tensors and queries; target
answers remain in parent-side scoring. In the examined path, no host algorithm
selects or reconstructs the answer value.

The action-selection boundary is different.
[ContextualOccurrenceMotor.learn](../../experiments/r4_integrated/contextual_occurrence_motor.py)
extracts an edited occurrence from two verified before/after demonstrations.
[VariabilityAwareSegmentBinder.grow](../../src/humanbrain/variability_aware_segment_binder.py)
then computes shared role offsets, stable/wildcard intermediate events and
boundary envelopes with Python indexing and set comparisons. Its `respond`
method computes equality matches, the next candidate index, stable-event
permission and boundary matches in Python before supplying zero or
`_DRIVE_SCORE` to physical races. A final physical race does not establish that
those computations were acquired through local neuronal plasticity.

**Required claim correction:** retain this as an explicitly supplied narrow
literal-edit motor prior and report its two demonstrations. Attribute a future
successful run to locally learned source-absent byte answers used by that motor,
not general coding or an entirely neuron-learned action policy. Under the user's
current exclusive-neuron objective, eventual motor selection and schema
acquisition must move into the neuronal learner; replacing transport-only
quoting or file materialization is not the issue. This is an integration gap,
not a reason to discard the reusable memory/decoder experiment.

### 2. Preserve the runtime rejection and repair the successor admission

The [v1 result](../../experiments/r4_integrated/local_memory_byte_v1_result.md)
closes the old configuration at a 2,013.103-second projection against 900 seconds.
Its twenty disposable lessons are not acquisition evidence. Optimized source
needs a distinct source/configuration identity and measured complete-lesson
admission with the original margin.

The old runner also asserts `launch["child_pid"] == cost_report["pid"]`, although
the saved receipt distinguishes the Windows venv redirector from actual Python.
Use the verified launcher-owned process ancestry, preserving PID role checks and
the failed receipt. Fix this before a successor full launch; do not weaken native
identity checks or repurpose the closed admission.

### 3. Most plausible acquisition risk: sequence order is weakly represented in upstream credit

This is a source-derived hypothesis, not a measured learning failure.
[ByteSequenceDecoder.teach_local](../../src/humanbrain/byte_sequence_decoder.py)
broadcasts `g_t = B_context (p_t - onehot(y_t))` through fixed feedback. The joined
learner averages those signals over all teacher bytes/EOS and updates each
upstream group once using `C_group mean(g_t)`. Therefore, exactly in real
arithmetic,

```text
mean(g_t) = B_context (mean(p_t) - normalized_target_byte_histogram).
```

The target-dependent component has no ordering information. The declared novel
value `sora brin` has the same byte/EOS histogram as taught `brin sora`; likewise
`lune mavi` and `mavi lune`. Address tags select past writes by the question key,
but receive no byte-position-dependent learning signal at this upstream update.
Distinct per-position errors can cancel in their average.

This is **not** a proof that permutation learning is impossible. Teacher-forced
decoder states, predicted probabilities, online decoder updates and memory
states can differ by order. Decoder hidden/readout updates themselves occur per
byte. The risk is that useful contextual order credit must emerge through these
indirect routes while teacher forcing makes the decoder's local next-byte task
easier than free generation.

**Action:** keep the fixed successor rule for its first honest fit; do not silently
replace this average or tune a rate against the four novel answers. If acquisition
fails, first inspect already generated exact bytes, EOS/limit reasons, per-lesson
loss and first-attempt accuracy. Distinguish low teacher loss with failed free
generation from failure to fit even taught values. For a separately declared
follow-up, one order-sensitive neuronal feedback/eligibility proposal is more
informative than increasing neuron count or replaying the same benchmark.

### 4. Second acquisition risk: useful address separation and writer credit must emerge together

[PlasticMemory.write_local](../../src/humanbrain/plastic_memory.py) writes at every
byte, including repeated sentence boilerplate and terminators, with initially
untrained recurrent/key maps and sigmoid write strength. Correlated keys cause
later writes to overwrite earlier contributions. The same fixed-key overwrite
operator updates eligibility tags, so it tracks this loss; it does not itself
preserve a fact that the forward synapses erased.

The writer contract correctly discloses omitted key/strength derivatives and
off-diagonal recurrent/normalization paths. Map-local derivative factors and
independent fixed neuron broadcasts are legitimate declared local rules, but
they are not a descent guarantee for writing an exact ordered value. In
particular, key and strength learning omit the full effect of their change on
memory readout. Do not treat moving all parameter groups as learned addressing.

**Action:** use the existing two-key/shared-memory, same-key/different-memory and
interference checks to diagnose behavior. If needed after a negative result,
record readout separation and key-overlap/write-strength summaries on a bounded
public diagnostic stream, keeping those summaries out of learner inputs. Require
useful improvement over `writer_maps_frozen` before more scale. A failed value
decoder and a failed address representation need different fixes.

## Controls, exactness and evidence limits

- The five modes share initial forward tensor values and fixed feedback. Frozen
  slow parameters still permit document-triggered fast writes; writes-disabled
  removes those writes. Writer/maps-frozen preserves query/decoder learning.
  These controls can distinguish useful writing from a trained decoder over a
  random reservoir. Aggregate superiority is separately required for a
  tag-specific claim.
- The schedule independently varies two keys and values. This prevents a fixed
  query-to-answer table from satisfying paired final checks. The four new values
  are exposed-byte recombinations under the same document/query grammar, not
  new-language or family-disjoint software transfer. Fit checks are familiar
  vocabulary; delayed checks repeat the same memories and read-only queries.
  Read-only delay does not test forgetting under further slow learning.
- `predict` precedes separately supplied teacher bytes. Ingestion receives no
  target/question and purges writer activity; persistent tags are counted
  source-derived plasticity state. Snapshots contain neither raw source arrays
  nor autograd history, and worker inference checks unchanged tensors. This is a
  reviewed routing boundary, not an OS secrecy boundary.
- Local updates use detached pre/post traces and declared neuron-level signals;
  the examined main path does not call backward or an optimizer. Self-path and
  map-diagonal approximations are explicit. Existing tests verify arithmetic,
  masks, control behavior and transport, not useful acquisition. They were read,
  not rerun for this report.
- `generate` distinguishes real EOS from a byte limit and leaves invalid UTF-8
  untouched. Final action scoring requires emitted bytes to equal the offline
  answer, plus exact generated bytes followed by a newline in successful Python
  stdout, no stderr and unchanged decoys. The broader executor's whitespace
  stripping does not weaken this runner's stricter `lossless_generated_execution`
  check. Rejected output still gets an action receipt and cannot score success.
- All acquisition episodes reset fast memory and retain slow weights. Final
  actions receive no execution-feedback updates. Even a complete narrow pass
  therefore leaves persistent multi-task retention, autonomous correction,
  authentic comprehension and general coding unestablished. A retrieval-only
  reference and recurrent-controller bypass are absent from these five arms;
  do not claim advantage over retrieval or causal necessity of recurrence.

The immediate useful outcome is a measured, honestly scoped acquisition result
for the optimized local learner. Keep the negative result if it fails; use its
failure distribution to choose one bounded return to the integrated workload.

## One contingency, only after the current capability result

If the current fit fails in an order-sensitive lane, compare its mean feedback
with one fixed neuronal-activity-modulated feedback rule on the same integrated
workload. For example, let a fixed, serialized projection of the decoder's
**before-update** recurrent activity produce `a_t = tanh(R h_t)` in context
neurons, and accumulate `mean(a_t * B_context delta_t)` instead of
`mean(B_context delta_t)`. All factors are current neuron activity, explicit
byte error and fixed connections; existing synapse-local writer/query
eligibilities remain the update factors. No retained-network backward pass or
externally computed per-synapse correction is introduced. This is a specific
heuristic candidate, not a predicted improvement or an implementation request.

Freeze one projection seed/scaling, learning rate, existing lesson schedule and
cost ceiling before exposure. Run the current-mean and modulated rules from
matched initial weights with the same byte offers and genuine first attempts;
retain the writer-frozen comparison and full action scoring. Reserve fresh
swapped-value questions before the run, without recycling the four exposed novel
values as new evidence. The smallest falsifier is failure to distinguish both
orders in a paired same-query/different-memory check at the fixed endpoint, or
no exact fresh-value action gain over current mean feedback. Either outcome
closes this one proposal; no projection/seed/rate sweep. Passing that pair alone
does not replace the existing fit, delay, interference and control requirements.

## Successor preparation review

The orchestrator subsequently requested review of
[v2](../../experiments/r4_integrated/local_memory_byte_v2.py), its recipe,
admission checks and the frozen-control credit omission. No blocking source
defect was found in this review. This is readiness for separate measured cost
admission, not permission inferred from the old rejected receipt.

- Imported v1 factories/lesson/export helpers retain their original numerical
  configuration; v2 changes only the schema identifier. Its source manifest
  extends the predecessor/native source manifest with both successor files.
  The reviewed closure includes local memory/decoder/controller code, motor
  modules, native bindings/source and the human-cell configuration. Runtime
  source/package/native/schedule checks remain in force.
- Copied worker and action logic preserve the query-only fresh process,
  generated-byte actuation and exact parent-side scoring. Their own module
  path/source manifest now bind the v2 worker. Metadata explicitly states the
  legacy host-computed motor boundary identified above.
- The successor recomputes complete-lesson cost using the original factor-two
  margin, additional capped-generation charge and thirty-second finalization
  allowance. It validates the exact v2 cost command and cost admission hash;
  actual Python, redirector and worker must each occur once in the launcher's
  creation-keyed owned process inventory. This repairs the invalid direct PID
  equality while retaining the documented sampling limitation.
- The two added runner tests cover the projection arithmetic, over-budget and
  nonfinite samples, missing actual-Python identity and PID reuse. The author
  reports 24 decoder tests passing in
  `.pytest-tmp/local-memory-byte-v2-decoder.xml`; this reviewer read the tests
  and did not rerun them.

The control optimization uses the existing forward memory writer only for
`frozen` and `writer_maps_frozen`, retaining independent allocated zero credit
arrays of the original shapes. Credit is not an inference input. Frozen slow
writer parameters cannot consume it; `writer_maps_frozen` now rejects attempted
writer/controller/map unfreezing before ingestion or teaching. Fully frozen
teaching already suppresses all updates. Query/decoder learning stays intact.
This is a causally justified omission within these fixed experiment modes,
not full pre-feedback state equivalence or support for later arbitrary mode
mutation. Credit contents and their hashes intentionally differ from v1.

The four new parametrized parity checks compare float32/float64 cumulative
fast synapses, predictions, feedback, all updated slow tensors, post-retirement
full state, input independence and detached restore against the original
credit-tracking path. A separate test rejects unfreezing. These checks were
source-reviewed; performance remains subject to the root's separate timing and
complete-workload admission. Source-derived live tags remain required and
counted for the learning arms; zero arrays still occupy storage in frozen arms.

A disposable saved-only auditor is prepared at
`.pytest-tmp/local-memory-byte-v2-audit.py`. It imports no project/model/native
modules and never regenerates answers or reruns children. It independently
checks source/receipt hashes, the deterministic teaching/check roster, cost
projection, owned process identities, checkpoint chains, fixed control tensors,
zero versus live writer credit, exact source-free state keys, all 100 action
workspaces and 102 execution receipts. Its syntax/argument parser was checked;
no v2 cost artifacts had yet been audited when this section was written.
