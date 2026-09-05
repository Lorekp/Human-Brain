# Interaction conditioning: executed result

Completed 2026-09-05 under the [declared protocol](contextual_conditioned_protocol.md).
The [artifact](../../benchmarks/contextual_conditioned_python_development_v1.json)
and [synapses](../../benchmarks/contextual_conditioned_python_development_v1.npz)
record repeatable public development. Teaching requests were openly reused;
transfer cases were declared before outputs. No protected gate passes.

## Behavior

| Policy | Fresh executed | Retained executed | Both-property retained entities | Unsupported STOPs |
| --- | ---: | ---: | ---: | ---: |
| GAIN1 updating neurons | 0/8 | 0/16 | 0/8 | 2/2 |
| GAIN16 updating neurons | 0/8 | 1/16 | 0/8 | 2/2 |
| GAIN1 raw or mapped learned-score reference | 0/8 | 1/16 | 0/8 | 0/2 |
| GAIN16 raw or mapped learned-score reference | 0/8 | 11/16 | 3/8 | 0/2 |
| Frozen neurons, either gain | 0/8 | 0/16 | 0/8 | 2/2 |
| External encoder cosine, either gain | 0/8 | 0/16 | 0/8 | 0/2 |

Conditioning produces useful query-dependent retention in the host readouts:
GAIN16 emits fourteen distinct old-task answers; GAIN1 always selects `ebony`.
The eleven successes are actual executed edits on previously taught requests.
They establish neither fresh generalization nor successful neuronal execution.
All decoys remain unchanged.

Two distinct failures are now visible. GAIN16 has fifteen collision STOPs on
sixteen retention requests, blocking ten otherwise-correct learned selections.
Raw and mapped winners agree throughout teaching and evaluation, so there is no
observed winner loss from the saturating current map in this run. The recorded
loss occurs at the discrete first-spike readout; compressed currents may still
reduce temporal separation. Exact ties must remain ambiguous.

Fresh ranking fails even without that readout. GAIN16 raw/mapped references
select the old answer `indigo` on every fresh audit query; transport queries
select old values or a neighboring service's value. The learner has not
demonstrated reusable entity/property binding on this cohort.
Neuronal STOPs also affect most supported requests, so unsupported STOPs alone
do not demonstrate selective abstention.

## Teaching and controls

Each gain makes 48 neuronal first attempts, with zero successes. GAIN1 executes
34 fallback proposals and solves one; GAIN16 executes 38 and solves fifteen,
with 0, 5 and 10 fallback successes across the three passes. Fallback and teacher
successes never count as neuronal first-attempt success or autonomous recovery.

Both arms admit 32 actual FAIL/teacher-PASS pairs and offer 118 older receipt
replays. GAIN1 applies 116 native updates (32 new, 84 replay); GAIN16 applies 57
(32 new, 25 replay). All newly admitted pairs trigger updates. Their negatives
differ: GAIN1 admits 14 neuronal, 17 raw-score and one external proposal;
GAIN16 admits 10, 21 and one respectively. This is the effect of conditioning
inside an on-policy learning loop, not a comparison on identical negatives.
Matched frozen mates receive the same 300 feedback events as their updater and
apply no updates; their synapses remain unchanged.

The pinned pretrained MiniLM encoder, retained raw documents/vectors, generic
span candidates and external receipt ledger remain disclosed components.
Native plasticity changes the evidence-ranking synapses. All policies share
the same fixed learned occurrence motor prior: the host and external references
isolate evidence selection, not removal of every neuron from the entire system.
Live sources disappear from requests, but remain in the external retained store.

## Resources and validation

The complete comparison takes 156.64 seconds, including execution subprocess
waits. Parent peak working set is 288.67 MiB; final private memory is 574.35 MiB.
These process measurements include the evaluator and ledger, exclude child
executor memory, and are not a repeated performance benchmark.

After fresh ingestion and the original exam, each arm retains 4,823 raw bytes,
2,319 candidates and 27 windows from fifteen documents. Counted array/byte
payloads total 7,284,967 bytes per arm, including 1,152 storage-ancestry bytes.
Compact external ledgers add 2,185,816 bytes for GAIN1 and 2,133,457 for GAIN16.
Each contains 786,944 bytes of numeric eligibility payload; that figure overlaps
the serialized ledger rather than adding to it. Python object overhead and the
shared 90,405,214-byte encoder are separate.

The original KS2 Q6a/b remains untuned. Updating and frozen neuronal answers
STOP and score 0/2. GAIN16 raw/mapped answers are `vibrations` and
`create vibrations`; GAIN1 emits `ebony`, and external references emit `to`.
Noncanonical reference answers retain official-rubric human scoring pending.
No exam improvement is claimed from unresolved marks.

Twenty-two focused tests pass with the real encoder/native learner, covering
gain identity, exact scaled updates, cross-gain receipt rejection, retained
occurrences, and the raw reference's tie/STOP behavior. Source and NPZ hashes
match the completed artifact; the loaded native DLL is checkout-local. No native
kernel code changed. Numeric receipts use compact LF JSON.

## Next decision

Keep gain 16 as a tested development candidate; do not raise gain or replay
budgets on this cohort. Prioritize candidate-relative token interactions that
can bind a fresh entity and property to its answer occurrence, retaining all
candidates and learning the selection from executed feedback. The existing
[research review](../../docs/research/contextual-learning-review.md) motivates
that comparison.

A separate bounded first-crossing/readout refinement is now justified to recover
demonstrated retained choices. It must preserve exact-tie STOPs and be checked
against declared cell dynamics; disabling uniqueness would introduce arbitrary
index selection. It cannot resolve the fresh ranking failure. Return both changes
to the integrated loop with separate attribution. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md) owns
the next experiment and unchanged 6/8 fresh, 12/16 retained neuronal targets.
