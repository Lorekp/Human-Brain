# Initial address competition: bounded resource and integration review

2026-09-06. Source/math review of the root's proposed pairwise threshold family,
not implementation selection or a measured result. No model, checkpoint, test,
action, alternate prediction or timing probe was executed. The active
[learning review](neuronal-splice-online-v1-learning-review.md) supplies the
scientific motivation; this note does not infer new outcomes from saved ranks.

## Concrete forward circuit

Keep the existing raw-query lag population, actual materialized joint features
`X_s = Z K_s^T`, and learned score `s_s = <B,X_s>`. Compare **all N occupied
slots**, including EOS and zero-sensory slots in the memory-disabled control.
No content, path, answer length or teacher filters the candidates.

For each pair `i < j`, form the fixed two-input threshold neuron
`c_ij = 1[float64(s_i) - float64(s_j) >= 0]`. Its gains are exactly +1 and −1,
with zero threshold. Then form N winner neurons:

`w_i = 1[i + sum(j>i)c_ij - sum(j<i)c_ji >= N-1]`.

The integer current counts opponents beaten by slot i. Equal scores favor the
lower tape index, explicitly introducing an order prior. This total ordering
has exactly one winner; N=1 gives w_0=1 without any pair comparators. Finite
float32 scores are cast before subtraction, avoiding float32 difference
overflow. Winner counts up to 127 are exactly representable. There is no
gain search, temperature, recurrent settling tolerance or hidden final argmax.
Two threshold stages are followed by the fixed readout `sum(i*w_i)` into the
compact cursor. Count its N weighted inputs/N−1 additions and verify one active
winner. This index is a representation of actual circuit activity.

The quadratic two-layer threshold construction is supported by section 4 of
[Maass, Neural Computation with Winner-Take-All as the Only Nonlinear Operation](https://proceedings.neurips.cc/paper/1999/file/1c54985e4f95b7819ca0357c0cb9a09f-Paper.pdf).
The exact lower-index tie convention and cursor readout above are prospective
implementation choices, not claims of biological timing or universal semantics.

| Work at N=128 | Exact count |
|---|---:|
| Pair comparator neurons, P=N(N−1)/2 | 8,128 |
| Winner threshold neurons | 128 |
| Total new threshold neurons | 8,256 |
| Comparator fixed synaptic applications, 2P | 16,256 |
| Winner fixed synaptic applications, 2P | 16,256 |
| Total fixed applications before cursor readout | 32,512 |
| Cursor readout weighted inputs/additions | 128 / 127 |
| Existing joint-feature products, unchanged | 4,194,304 |
| Existing joint-feature channel reductions | 4,063,232 |
| Existing learned score products/reductions | 131,072 / 130,944 |

A straightforward vectorization uses all upper-triangle pair indices, one
float64 difference bank, binary thresholds, then signed indexed accumulation
into N winner currents. It implements the stated fixed connectivity directly;
it does not need a dense N-by-P weight matrix. Fixed ±1 connections permit
P subtractions and 2P signed accumulations, while the logical synaptic work
remains 4P. Report both conventions rather than calling the operation free.

## Activity storage and sparse retrieval

A conservative named-array envelope for that vectorization at N=128 is
**403,528 bytes**: pair indices 130,048; cast scores 1,024; two gathered score
arrays 130,048; differences 65,024; comparator bits 8,128; float64 comparator
outputs 65,024; winner currents 1,024; winner bits 128; integer winner/readout
positions/products 3,072; scalar cursor 8. This deliberately counts named
intermediates together, although their lifetimes can be shortened. Validation,
framework temporaries and allocation overhead remain additional. This is not
an RSS bound. Generating the fixed pair-index topology per call adds no learned
parameters or persistent tensor state; topology-generation work must be counted.
If indices are retained instead, disclose their 130,048 bytes per stored bank
and do not silently omit duplicated per-model storage.

Implementation amendment: the held competitor source
`d222dcc41323acb7ef3c82ad54c39c2c72d6f6428587994bea62707d6c65eea3`
also creates an owned float32 score clone (4N bytes) and an int64 winner-count
validation scalar (8 bytes). Its exact explicitly named allocation sum is
`45N + 49P + 16`, or **404,048 bytes at N=128**, which is 520 bytes above the
prospective envelope. Returned tensor payload is `13N + 9P` = **74,816 bytes**;
pair-index, gather and readout temporaries do not survive the call. These are
payload/allocation counts, not measured peak memory. This reconciliation is
source-only and does not alter the earlier prospective arithmetic.

The subsequently selected [implementation contract](neuronal-address-competition-v1.md)
retains the full occupied codec bank and gathers one cursor at every free
decision. Therefore the visited-only evaluation calculation below is an
unselected design bound, **not the implemented workload**. Implemented initial
work remains N codec evaluations and 257N potentials. Free gathers are 7,196
bytes per decision (122,332 bytes for 17), with a full held bank plus two
successive single-row responses: at most 935,480 bytes of these named response
arrays at N=128. Teaching keeps the original full-branch work.

After the actual winner, one committed cursor can follow the existing immutable
EOS-absorbing adjacency. Evaluate the unchanged 257-way codec only at that
cursor at each decision. With payload cap 16 this is at most 17 actual codec
evaluations and 4,369 output potentials, versus current 128 evaluations/32,896
potentials for the all-slot held bank. Early EOS means fewer evaluations.
No source-token cache or target lookup is needed. The winner may nevertheless
be wrong, or its codec may emit an incorrect byte/EOS; no success follows from
this accounting.

One conditional response occupies `257*(4+3*8)=7,196` bytes. Two successive
responses may overlap during replacement. Current full-branch responses occupy
921,088 bytes per bank; held and old/new gathered banks can overlap at
2,763,264 bytes. At 17 decisions the current 16 later full gathers copy
14,737,408 response bytes. Sparse continuation removes that N-wide gather and
mixture path only if it actually evaluates a single active cursor. Applying a
one-hot mask while still computing all 128 codec responses earns no such saving.
Softmax and greedy selection among the codec's 257 output neurons remain work;
the address gate does not implement the byte-output winner circuit itself.

Current public `codec.evaluate` checks all 530,448 bytes of codec parameters on
every call. Calling it 17 times would repeat those scans over 9,017,616 bytes;
single-row arithmetic does not make validation free. An eventual implementation
must either count this work or reuse the established once-per-call parameter
validation/private-checked-kernel pattern, with per-activity finite checks and
no parameter publication during the owning prediction. This is an implementation
choice requiring ordinary QA, not permission to weaken finite validation.

These savings do not remove the existing 524,288-byte X bank or its score
product transient. Preserve X-before-B in both acting and teaching. Existing
teacher X casts and weighted local products each occupy 1,048,576 bytes, plus
8,192-byte accumulation and 4,096-byte cast/proposal buffers. Full teacher codec
weight/bias error buffers remain 528,392/2,056 bytes, with per-prefix branch
error and scaled-column buffers each 263,168 bytes. No sparse free-read claim
reduces these retained full-branch teaching costs.

## Same-instance teaching and integration boundary

The narrow forward seam is in `LocalMemoryLearner.predict_with_receipt`, after
`_query` has produced actual scores and before `_codec_bank` currently expands
all branches. Reuse `ConditionalByteCodec.evaluate` and the tape's checked
forward cursor gather/advance kernels. Leave the learned action heads, raw
request, payload cap, STOP/SPLICE ABI and same-instance pending action/feedback
ordering intact. Do not bypass codec finite checks or atomic publication.

Retaining full posterior supervision in `teach` still evaluates all N candidate
branches and every teacher byte/EOS. It preserves the expensive full teacher
phase and uses the same B/codec parameters, but its distributional loss is **not
the loss of the committed deterministic read**. Its `p - posterior` score error
must be declared as an auxiliary direct score-teaching rule, not advertised as
the derivative through either threshold layer. No comparator Jacobian, reverse
key-error transport, straight-through estimator or cross-call cache is implied.
The learning review must settle the actual teaching rule before implementation;
this cost note does not select a replacement surrogate.

A successor receipt needs the initial score/competition identity, N and P,
two executed stages, tie policy, one-hot winner or equivalent validated cursor,
actual codec evaluations and continuation count. Keep full teacher mixtures and
posteriors if that phase survives. Do not retain all pair activity in every
JSON row: fixed topology plus scores/final activity permits an independent
saved scalar gate check. New sparse free receipts require a versioned validator;
the old all-branch posterior schema cannot silently change meaning.

The circuit's O(N²) overhead is bounded and small relative to existing fixed
feature arithmetic at N=128, but arithmetic ratios are not runtime predictions.
The completed run needed only 96.14 seconds including audit; speed is not the
reason to adopt this behavioral change. Longer actual payloads could increase
teacher-free execution time despite sparse reads. Fresh caps, receipt bounds,
withheld cases and measured admission remain necessary if root selects it.

Source inspection used the canonical graph project with verified checkout root.
Graph discovery returned no current learner symbols and reported untracked
freshness; direct held source was therefore read. Sources: reader
`2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8`,
its `ConditionalByteCodec`, `_query`, `_codec_bank`, `_codec_responses`, `teach`
and `accounting`, plus `NeuronalSequenceMemory` forward adjacency. Mem0 returned
unrelated historical development notes; none supplies this review's mathematics
or scientific evidence.
