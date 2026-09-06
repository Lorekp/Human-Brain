# Competitive reader receipts: successor integration contract

2026-09-06. Prospective source contract for root review. The
[component QA bank](../../benchmarks/neuronal_address_competition_v1.qa.json)
establishes the six selected mechanics checks, including a preserved wrong
action followed by inherited same-instance correction. It does not admit a
successor benchmark. Root has selected implementation of receipt serialization
and independent saved-trace validation only. The first authorized source slice
is the v2 serializer; the independently implemented saved checker follows under
separate ownership.
No model, checkpoint, action, test, roster generation or exposed endpoint was
executed/read to prepare it.

## Smallest first implementation

Add two files without changing frozen v1 sources:

| File under `experiments/r4_integrated/` | Responsibility |
|---|---|
| `neuronal_splice_receipts_v2.py` | Teacher-free, model-free compact serialization and writer-side consistency checks. Reuse v1 primitive guards, action-head/proposal helpers and exact full-branch teacher compaction. |
| `check_neuronal_address_competition_v1.py` | Independently implemented scalar validation of saved competition/free-reader traces. Import no neuronal model, executor, roster or tensor library. Do not call the production competitor or merely delegate its numerical checks back to the writer. |

Root assigns these two files separately for independent review; implementation
owners must agree on the schema below first. Existing `neuronal_splice_io_v1`
already takes explicit limits from `spec` and can remain unchanged. Do not
mutate imported globals, patch frozen validators, or add a policy to the old
serializer through permissive fallback.

The existing v1 `compact_reader` starts every occupied branch and checks omitted
cursors against `range(N)`. Its `validate_reader` requires N-wide beliefs.
It therefore cannot serialize a competitive free receipt truthfully. Retain
that implementation for the unchanged teacher and matched marginal reference.

## Exact policy separation

Root selects the flat format with an explicit expected phase. The integration
caller must supply `expected_policy="competitive_free"`, `"marginal_free"` or
`"all_address_teacher"`; never infer policy or phase from an untrusted row.
Saved `kind` and actual receipt generation presence must agree with that
expectation before dispatch. Teacher-as-free and free-as-teacher are rejected. Marginal
free and all-address teacher rows retain the exact six-key v1 schema unchanged.
Competitive free rows retain those six fields and add `schema` with value
`neuronal-splice-competitive-reader-v2`, `competition`, and
`diagnostic_log_prior`. The six fields are `kind`, `initial_log_belief`, `steps`,
`generated_hex`, `stop_reason`, `work`. Each step retains `token`, `nll`,
`mixture`, `posterior`. Competitive free activity has one branch throughout;
marginal free and corrective teaching have N branches. The latter two activity
records must be exactly the existing v1 compactor's output for the supplied
actual receipt, without a new forward call. STOP feedback still has no reader
receipt or reader-update flags. Full teaching has no competition trace.

Proposed callable seam:

- `compact_reader(receipt, state, *, expected_policy)`.
- `compact_prediction(prediction, state, *, expected_policy)`, accepting only
  the two free policies.
- `compact_feedback(feedback, state)`, internally selecting full-address teaching
  only when an actual reader feedback exists.
- Writer-side `validate_reader(row, successor, *, expected_policy, max_bytes=16,
  targets=None)`, `validate_prediction(row, successor, *, expected_policy,
  boundaries=None)` and `validate_competition(trace, *, used)`. The future
  independently implemented checker uses the same explicit policy contract.

The expected free policy comes from the frozen arm/model metadata, not a
caller-controlled field inside a receipt. A matched marginal reference gets
`marginal_free` and its own unchanged full-population policy. It must never be
retroactively assigned the competitive winner or charged competition work.
All competitive matched controls use the same gate, including zero-memory ties.
This supports the reference without selecting its future exposure or prior.

Keep the existing outer prediction/action identity and head fields. Its payload
uses the explicitly selected reader format. Recompose the small prediction validator
around reused head/proposal helpers; only reader validation changes. Preserve
actual pending attempt, proposal hash, model/tape hashes, action-line binding,
external grade and action-before-teacher clocks.

## Complete competition trace

The `competition` object has exactly `schema`, `scores`, `pair_currents`,
`comparator_activity`, `winner_currents`, `winner_activity`, `winner_index`,
`work`; its schema is `neuronal-address-competition-v1`. Retain every actual
value, in production upper-triangle i<j order. Shapes are N, P, P, N, N with
P=N(N−1)/2. Activities are JSON booleans, not arbitrary numerical substitutes.
The trace work dictionary preserves every declared competition counter.

Writer and independent checker must establish:

1. N equals authenticated occupied tape slots, 1..128. Scores are finite and
   round-trip exactly through float32 (stdlib `struct` is sufficient in the
   saved checker). They describe every slot, without EOS/content filtering.
2. Every pair current equals widened `score_i-score_j`; every comparator is its
   nonnegative threshold. Every winner current equals
   `i + sum(j>i)c_ij - sum(j<i)c_ji - (N-1)`. Check the actual threshold bits,
   exactly one winner, and the fixed `sum(i*w_i)` index readout, including N=1.
   A saved-only scalar loop may independently check this circuit; it does not
   reconstruct the neuronal query/key feature bank or learned score computation.
3. `diagnostic_log_prior` is separately N-wide and agrees with stable log-softmax
   of the actual scores. Center by their maximum **before** computing the
   normalization: adding a small log-normalizer to a huge score first loses it.
   Exact tied large float32 scores are an authored validation case. This
   diagnostic never enters competitive free mixtures.
4. Competitive activity starts at the actual winner cursor and has prior and
   posterior `[0.0]`. Check every actual supplied step's prior/cursor and final
   belief before omitting those redundant fields. The saved checker follows
   authenticated forward/EOS-absorbing adjacency from that winner. Retain the
   complete 257-way mixtures, NLLs and observed output tokens; validate actual
   codec-output maximum, byte assembly and EOS/LIMIT semantics. No target byte
   or teacher-selected address may fill in a missing free receipt.
5. Work equals the actual source contract: one query/competition, N initial
   codec evaluations and257N potentials, one bank index per occupied slot,
   one gather of1,028 values/7,196 bytes per free decision, one branch update
   per decision and one advance between continuing decisions. Teacher and
   marginal-reference counters remain their original full-population formulas.
   Verify all gate counters, including `45N+49P+16` named allocated bytes and
   `13N+9P` returned tensor bytes; neither is a measured RSS peak.

No saved validator may claim to recompute unsaved codec outputs, score synapse
currents from X, or intermediate parameter updates. Those remain supported by
source review and ordinary arithmetic/parity QA. Neither the N-wide teaching
surrogate nor trace checking differentiates through the hard gate.

## Guard and cost proposal before any integrated admission

Keep complete traces inline in the owning free row. This avoids a second trace
journal, join protocol and overlapping row lifetime. At N=128 and17 free plus
17 teacher decisions, the old14,640 named numeric-vector entries lose2,286
free-branch entries and gain8,512 trace/diagnostic floating entries. The new
maximum is **20,866 numbers plus8,256 booleans**, excluding control scalars/work.
This is a dimension bound, not a serializer-size measurement.

The old maximal fixture's735,558-byte host tree cannot establish the new bound.
Adding the nominal no-alias vector delta gives roughly1,232,006 bytes before
container changes, already beyond1MiB. Propose **2MiB line and tree guards**,
qualified with an authored complete-row fixture before freezing them. Retain
two live rows/two line buffers per process and factor-two tree reserve. That
gives **E=24MiB** across parent and worker, replacing12MiB only in a successor
protocol and its independent formula. It is not a post-cost enlargement.

The fixture must retain maximum head vectors, full teacher mixtures/posteriors,
full gate trace, maximum declared text/output fields, and work metadata. Check
both encoded size and writer/parsed tree guards, including round-trip JSON.
If it fails, stop for a prospectively reviewed representation/guard change;
never truncate pair activity or silently drop trace fields.

Every pair is also checked by the writer and independent auditor. Those O(N²)
scalar checks, conversion, guarded serialization, flush and hashing belong in
complete journal/audit timing. No old cost admission proves the new footprint
or duration. Recompute full retained tensor/payload scope once the reference,
prior and roster are selected; do not carry forward a four-arm total into a
five-arm comparison. Neuronal weights remain the same shapes, but new metadata
changes model-state hashes and must be authenticated explicitly.

## Gates after the two receipt modules

Root selects the [three authored receipt QA nodes](neuronal-address-competition-receipt-qa-v1.md)
in the existing decoder test file. They exercise six synthetic receipts with
42 step records, with zero model, checkpoint or real-action calls. After reviewed
source freeze and the selected-checkout doctor, run exactly these nodes under
the existing 60-second/2,048-MiB watchdog using fresh
`.pytest-tmp/neuronal-address-competition-receipt-qa-v1` artifacts and basetemp.
Preserve failures; do not repeat the already passed six neuronal component tests.
The small first
tier needs: N=1/tie/adjacent/extreme-float/128-candidate trace round trips;
mutations of pair current/bit, winner/index, diagnostic prior and work;
wrong omitted cursor/prior; competitive/marginal/teacher policy confusion;
EOS and limit assembly; and the maximal complete-row bound. Use plain authored
values and tensor-like stubs, with no model, old checkpoint or benchmark case.
Do not rerun the already banked component mechanics as incidental helper QA.

Later integrated work requires new `neuronal_splice_worker_v2.py`,
`neuronal_splice_endpoint_v2.py`, `neuronal_splice_online_v2.py`,
`neuronal_splice_prior_v2.py` and `check_neuronal_splice_online_v2.py`, plus a
newly selected roster/recipe. These are integration tasks, not authorized by
this receipt implementation slice. Keep the fixed teacher-free closure explicit:
it will need both reader classes, the competitor, v2 receipts and their frozen
v1 helper dependency. No parent roster/executor enters that package.

Before acquisition: qualify one bounded actual isolated exchange with declared
new-schema receipts and unchanged final weights/tape; verify selected prior
numerical identity and permitted metadata adaptation; test independent cost
decision/source/PID bindings with synthetic artifacts; freeze new withheld
cases, matched policies and memory/timing formulas; then obtain fresh completed
cost and independent-audit admission. Preserve v1's closed outcomes throughout.
