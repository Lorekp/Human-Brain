# Affine neuronal readout: recovered execution at saved weights

Completed 2026-09-05 from corrected source `8f1f176`, with the unchanged readout
implementation from `3c7c474`. The [completed artifact](../../benchmarks/contextual_affine_readout_v2.json)
follows the [protocol](contextual_affine_readout_protocol.md) and
[exam-harness correction](contextual_affine_readout_correction.md). This reuses
the exploration weights and cases. It establishes readout recovery, not new
learning or independent generalization.

| Neuronal policy | Fresh edits | Retained edits | Unsupported STOPs |
| --- | ---: | ---: | ---: |
| BINDING affine + interpolated GLIF | 4/8 | 2/16 | 0/2 |
| BINDING original interpolated GLIF | 0/8 | 0/16 | 2/2 |
| GAIN16 affine or original interpolated GLIF | 0/8 | 9/16 | 0/2 |
| Frozen affine or original interpolated GLIF, either arm | 0/8 | 0/16 | 2/2 |

Affine neuronal selection matches all 52 updating raw occurrences and their
executed outcomes, plus all 52 frozen raw STOPs. BINDING now executes the four
existing fresh successes and two retained successes that saturation previously
masked. No fresh entity has both properties correct; GAIN16 retains both for
three old entities. Both updating affine policies wrongly act on unsupported
requests. Correctly exposing the learned score ordering also exposes its errors.
No capability or protected-evaluation gate passes.

## Verification and the interrupted attempt

All 416 archived ordinary decisions reproduce exactly before new actions.
The completed run records 208 actual executions, with unchanged motor/checkpoint
state and no decoy failures. Frozen weights remain zero. Seven original exam
policies per arm match their archived answers/scoring and original end
checkpoints. Affine exam answers match the old raw selections: BINDING gives
`tenrec that live` / `tenrec`; GAIN16 gives `worms and beetle` / `beetle`.
Each arm's two marks remain pending official-rubric human scoring, with zero
confirmed marks. Frozen answers STOP. No new questions or marking policy were used.

The [first artifact](../../benchmarks/contextual_affine_readout_v1.json) remains
immutable and terminal `error`: a second exam-helper invocation ingested the
passage again, correctly triggering the checkpoint assertion. It had completed
all 416 old-decision checks and 104 BINDING executions. The corrected harness
ingests once for all old/new policies. Every repeated BINDING decision and
deterministic execution outcome matches the interrupted attempt; this repetition
is not an independent replicate. The readout formula was never tuned.

All 39 focused core tests pass, including overflow-safe large finite inputs,
true and STOP ties, finite-precision aliases, candidate permutation, frozen
states, custody and unchanged old identities. Inert harness checks cover single
exam ingestion and rejection of old-answer/checkpoint mismatch. Ruff passes.
All 37 source files match the corrected source commit; artifact, NPZ, encoder,
loaded checkout-native DLL and reconstructed synapse identities verify. No native
source or ABI changed, and no evidence learning or new teacher execution ran.
The original motor was restored from its two existing certified demonstrations.

## Cost and remaining capability failure

The completed run takes 77.65 seconds; the interrupted attempt takes 66.82,
for 144.47 seconds total. Completed-run parent peak working set is 301.10 MiB,
final private memory 591.93 MiB and traced Python peak 106.16 MiB. The interrupted
run peaks at 379.57 MiB. These timings include child waits; child memory is
excluded. BINDING/GAIN16 reconstruction takes 2.01/1.72 seconds, old-decision
verification 13.33/4.45, and exam comparison 12.46/2.05. Motor restoration takes
0.015 seconds. This is not a training-runtime or speedup comparison.

Each learner retains the original eleven documents, 4,708 raw bytes, 2,298
candidates and 23 windows after the exam. Counted array/byte payload remains
7,256,365 bytes for BINDING and 7,207,267 for GAIN16. Encoder, external correction
ledger and Python object overhead remain separate, as disclosed in the
[exploration result](contextual_exploration_result.md). All controls share the
learned motor and isolate evidence selection/readout, not the whole neuronal core.

The restored TRAIN receipts expose a separate stability problem. Six of 32
BINDING and two of 32 GAIN16 certified pair margins are negative at the final
weights, although each was positive after its original correction. BINDING
pair 26 was at +17.218 in the final rehearsal, then subsequent replay updates
left it at -8.665. However, nine of fourteen failed BINDING old queries satisfy
every recorded contrast by at least one: erased margins cannot explain all
global selection errors. The remaining issues include interference and incomplete
coverage of competing candidates; representation incapacity is not established.

Next compare the existing update with a step capped at the amount required for
unit margin, using the integrated affine action path and the same exposure
budgets. Smaller steps may reduce interference but lose protective margins or
learn too slowly. The earlier
[conditioning diagnostic](../../benchmarks/contextual_conditioning_diagnostic_v1.json)
already found zero wrong pairs but 23 subunit margins with an uncapped
margin-deficit rule on another ledger; the fixed-rate comparator had none.
Judge the new bounded comparison by retained and fresh neuronal execution,
not closeness to a numerical margin. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns that next workload and its limits.
