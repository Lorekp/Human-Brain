# Initial-address competition: bounded ordinary QA

Prospective source contract, 2026-09-06; no tests, models, checkpoints or actions
were run to prepare it. Root owns implementation of these six appended nodes in
`tests/test_byte_sequence_decoder.py` and their eventual execution. Read the
[selected design](neuronal-address-competition-v1.md) and run the selected
checkout's doctor after reviewed source freeze. Preserve existing test AST and
all receipts. Do not collect/run the broad historical file or old backpropagating
references. Use only fresh seed-733 readers and the authored bytes below.

## Six cases, in this order

1. **`test_address_competition_scalar_circuit_contract`** — Eight direct circuit
   calls, no reader. Float32 vectors: `[-3]`, `[1,-2]`, `[-3,0,5,2]`, `[3,3,1]`,
   128 zeros, `[f32_max,-f32_max,0]`, `[1,1+2**-23]`, and ascending integers
   0..127. Expected winners: 0,0,2,0,0,0,1,127. Independently compute every
   upper-triangle pair current using Python floats, each binary comparator and
   each signed winner current from the literal equations. Compare all receipt
   activities, the identity readout and exact work/allocated-byte declarations.
   Verify 8,128+128 cells and 32,512 signed applications at N=128. A host maximum
   is permissible only as an independent test oracle, never production selection.
   Snapshot inputs and require no mutation. This covers exact ties, adjacent
   float32 values, negative inputs and widened subtraction without gain tuning.

2. **`test_address_competition_input_guards`** — Seven rejected circuit calls:
   empty float32 vector; length129; rank-two `[1,1]`; float64 vector; `[NaN]`;
   `[+inf]`; and a float32 vector with `requires_grad=True`. Require explicit
   validation failure and unchanged supplied tensors. No reader or gradient
   computation. Do not create CUDA state solely for another invalid-device case.

3. **`test_competitive_reader_single_cursor_and_full_bank`** — One fresh learned
   competitive reader; zero B under `no_grad` as an authored initialization.
   Ingest `[b'ab', b'c']` once, N=5. Raw query `b'a'`, max16 must emit `b'ab'`
   then EOS: three decisions with cursors `[0],[1],[2]`. A second call at max1
   must return `b'a'`, LIMIT, with two actual decisions: a then b. Separately
   ingest `[b'']` into a fresh tape; query `b'a'`, max16 emits immediate EOS in
   one decision. Identity-biased codec weights stay untouched. Every free
   belief/prior/posterior is length one and normalized; diagnostic soft prior
   is separately N-wide. The full bank evaluates all occupied slots exactly once
   per call; each decision gathers one response, including the first. Check
   actual query, competition, full-bank, gather and single-cursor work; no
   visited-only evaluation claim. Free calls change neither parameters nor tape.

4. **`test_competitive_reader_inherited_publication_modes`** — One fresh reader
   for each `learned`, `address_frozen`, `slow_frozen`, `writes_disabled` mode,
   B zero, each ingesting `[b'ab']` once. Each receives exactly one inherited
   teacher call for query `b'a'`, teacher `b'b'`: two offered symbols including
   EOS, and three address branches. Check permitted codec/address publication
   against actual parameter equality, disabled tape zeros, held-state custody,
   and absence of autograd state. On the same learned instance, offer that
   correction once more with `learning_enabled=False`; both groups must remain
   unchanged despite counted offers. Total five teacher calls/ten symbols, no
   free calls. Do not require enabled and changed to coincide in zero-information
   memory-disabled address activity. There must be zero competition work in
   these inherited teacher calls.

5. **`test_competitive_reader_metadata_and_snapshot`** — One fresh competitive
   reader, one ingestion `[b'ab']`; no query or teacher. Snapshot, change B
   and a cloned snapshot tensor, and restore the original snapshot. Verify
   exact parameter/tape restoration and no aliasing. Metadata must identify
   committed free recall, full-bank execution and reopened all-address teaching.
   The overridden prediction method must be distinct, while `teach` and `_query`
   are inherited by method identity from the current no-backpropagation base.
   Check all parameters remain outside autograd. Do not relabel a base checkpoint
   as metadata-equivalent or load any historical artifact.

6. **`test_competitive_splice_wrong_action_then_teacher_parity`** — Last, one
   actual qualified child. Construct one competitive reader and inject that
   exact object into one fresh `NeuronalSpliceLearner`; construct one additional
   current no-backpropagation base reader solely as teacher-parity reference.
   Initialize both B matrices to zero and ingest `[b'ab']` once in each. Before
   any teacher, require all numerical reader parameters and tape tensors equal;
   version metadata is intentionally different. No base free prediction runs.

   Authored workspace: `a.py = b"x = 'old'\n"` and
   `main.py = b"import a\nprint(a.x)\n"`; raw request `b'a'`. Zero all action
   weights, set operation bias `[0,1]`. These disclosed fixture weights cause
   the actual neurons to choose SPLICE at the first path/boundary, end0, payload
   `b'ab'` plus EOS. The uncorrected result is `abx = 'old'`: verify that exact
   raw proposal through the existing qualified temporary-module executor once.
   Save raw proposal, files, child receipt and hashes exclusively before outcome
   assertions. Require genuine nonzero execution with AttributeError for missing
   `a.x`; do not repair or replace the action. This is mechanics, not a skill
   example or a successful task.

   Only after that real envelope, correct the same pending splice object on the
   immutable pre-workspace with teacher `SPLICE a.py 5:8 -> b'b', EOS`. Require
   three action labels and the same reader's two-symbol inherited correction
   over all three branches. Compare every reader teacher step, loss, work,
   address error and final numerical parameter tensor against one identical
   `base.teach(b'a', b'b', ...)` from the equal pre-state. Require a nonzero
   address update even though free recall committed to slot0. Do not predict
   again to manufacture post-correction success. The corrective phase performs
   no winner competition. Check pending acknowledgement, same instance token
   and unchanged literal tapes. The base is the held current compliant reader,
   never an older backpropagating implementation.

## Fixed exposure ledger

These are six pytest function nodes; the four modes in node4 are one explicit
loop, not an invitation to expand a parameter sweep. Planned totals:

| Exposure | Count |
| --- | ---: |
| Fresh competitive readers / current base readers | 7 / 1 |
| Additional splice wrapper and action circuit (uses injected reader) | 1 / 1 |
| Ingestion calls / raw documents / raw bytes / document EOS | 9 / 10 / 17 / 10 |
| Free reader calls / actual free decisions | 4 / 9 |
| Teacher calls / byte-or-EOS offers | 7 / 14 |
| Query traversals (free plus teacher) | 11 |
| Direct valid / rejected competition fixture calls | 8 / 7 |
| Additional free-call competitions / teacher competitions | 4 / 0 |
| Splice proposals / corrective action labels | 1 / 3 |
| Real qualified verification children | 1 |
| Historical checkpoint loads / exposed roster cases | 0 / 0 |

Queries are the one-byte `b'a'`, so eleven query traversals mean eleven raw query
bytes plus eleven EOS. The repeated disabled-publication correction still counts
its two teacher offers. Node6's base correction also counts; numerical parity is
not free. Snapshot operations add tensor copies, not ingestions or forwards.
If implementation needs a different fixture or call count, amend this document
before execution rather than silently adding trials. Root records actual return
counts and preserved failures, and approves the last real-child node separately
after the first five pass. Do not rerun passed unaffected cases.

## Source review before the gate

Verify the competitor has actual fixed comparison/winner activity with the
selected tie orientation, no sort/argmax substitute, and no trainable comparator
weights. The subclass must call `_query` once, competition once, full codec bank
once, and gather the single actual cursor each decision; its N-wide diagnostic
prior must never weight the free mixture. Check bounded EOS/LIMIT handling,
finite publication inheritance and versioned metadata. Inherited teacher parity
does not make its local marginal surrogate a derivative of hard selection.

These checks qualify component mechanics only. Successor serializer, isolated
worker, audit, new withheld roster and fresh cost admission remain separate.
No benchmark result or new acquisition is authorized by this QA document.
