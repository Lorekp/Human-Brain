# Prospective operation-intent component QA

Status: paused, authored source only; not executed. The broader persistent
language/action workload supersedes this proposed micro-gate. Do not append or
run the unreconciled fragment automatically. Any future use must review the core
and this exact fragment, append it to the existing decoder test file, freeze
the source, obtain a clean doctor receipt, and select the three nodes explicitly.
The selected mechanism is defined in
[the operation review](neuronal-operation-intent-review-v1.md). This qualification
does not authorize a fit, checkpoint prior load, endpoint replay or cost run.

The prospective fragment is
`.pytest-tmp/operation-intent-qa-v1-fragment.py`; its destination is the existing
`tests/test_byte_sequence_decoder.py`. It reuses that file's nested equality,
plain receipt writer, authored workspace and simulated-execution helpers. The
third node alone invokes the already qualified real executor. No new test file,
reader architecture, historical learner or external memory enters execution.

## Three selected nodes

1. `test_operation_intent_raw_request_population`: six authored raw requests
   (empty, byte zero, byte 255, `a`, `ab`, and bytes 0..30), plus rejected wrong
   type and 32-byte inputs. An independent bit-loop oracle checks all positions,
   EOS, empty padding, 576-column CPU float32 geometry, energy 32 and encoder
   work. Prefix support is equal for `a` and `ab`; amplitudes deliberately differ
   because each whole request is normalized independently. There are exactly
   42 encoded byte/EOS symbols across the six successful calls, no models.
2. `test_operation_intent_local_update_controls_and_atomicity`: one old and one
   successor standalone action circuit establish zero-U exact logits and
   probabilities on identical old features. One nonzero authored U and one
   independent scalar softmax/error/product oracle test the new direct local
   update and absence of publication during proposal preparation. Four fresh
   combined modes each ingest `ab`, issue one actual neuronal free proposal and
   receive one explicitly simulated STOP correction. Only learned and
   memory-disabled publish the new branch. The learned instance also checks
   missing/wrong-shape U checkpoint rejection, an in-memory strict snapshot
   round trip without another forward, and snapshot nonaliasing. One further
   free proposal receives a SPLICE correction with an infinity injected into
   the final *temporary* U proposal. The real finite guard must reject it with
   all model/tape tensors unchanged and before any reader teacher is called.
3. `test_operation_intent_same_request_action_before_teacher`: one fresh
   coincidence reader and successor combined model ingest `ab`. Authored zero
   address weights and STOP-biased action weights make this an explicit
   execution fixture, not learned success. One neural proposal is passed through
   the unchanged qualified executor, which runs the unchanged two-file workspace
   once and emits `old\n`. Its raw receipt is saved before the same model teaches
   the authored `ab` SPLICE. A delegating observation hook records, without
   replacing computation, the identical request activity in free and corrective
   phases. Three actual reader teaching offers and three action labels follow
   the real execution, and the tape remains unchanged.

## Prospective exposure ledger

| Quantity | Node 1 | Node 2 | Node 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| Combined model constructions | 0 | 4 | 1 | 5 |
| Reader constructions | 0 | 4 | 1 | 5 |
| Action circuit constructions, including combined | 0 | 6 | 1 | 7 |
| Fresh document ingests | 0 | 4 | 1 | 5 |
| Free reader queries/proposals | 0 | 5 | 1 | 6 |
| Combined teacher attempts | 0 | 5 | 1 | 6 |
| Reader teacher calls | 0 | 0 | 1 | 1 |
| Reader teaching offers | 0 | 0 | 3 | 3 |
| Standalone operation update offers | 0 | 1 | 0 | 1 |
| Actual execution children | 0 | 0 | 1 | 1 |
| Historical/file checkpoint loads | 0 | 0 | 0 | 0 |

The five ingests contain ten raw bytes plus five EOS symbols. Four normal
node-2 STOP corrections each offer one operation label. The rejected SPLICE
attempt completes start/end preparation and reaches the operation proposal
check; it publishes nothing and never enters reader teaching. Node 3 offers
three action labels. Thus there are ten completed head-update preparations
including the standalone oracle, plus one rejected operation preparation.
Node 2 makes two rejected state-dictionary loads and one successful same-instance
snapshot restore (one memory reconstruction and one valid in-memory load).
No second restored model or extra parity forward is constructed.

Each proposal is capped at sixteen bytes plus its final EOS/limit decision.
The authored source fixture anticipates five enabled-sensory proposals of
three decisions and one memory-disabled proposal of seventeen: 32 decisions
total. Saved receipts must report actual counts; the unconditional execution
ceiling is six times seventeen, or 102 free decisions. This is not permission
to retry after an unexpected result. The constructor must allocate exactly one
reader and one successor action circuit per combined model, with no discarded
old circuit hidden from this ledger.

## New contract checks and evidence

The new parameter is exactly 2x576 float32 values (4,608 bytes); standalone
action state is 71,472 bytes and the combined coincidence model is 639,040
tensor bytes. Snapshot schema and work expectations must be reconciled to the
held source before root selects execution. Preserve the old 4,244-term score
reduction and bias order, then add the separately reduced request current.
The scalar oracle checks all three parameter proposals, including the bias,
with one float32 gradient cast and unchanged eta .01. No autograd or backwards
transport participates in any fixture.

Actual raw request, head receipts, all new work, numerical update proposals,
control outcomes, raw execution and temporal ordering are retained beneath
the pytest basetemp. Raw execution and teacher receipts are written before
their corresponding assertions. Root also retains the exact source hashes,
JUnit, process monitor and doctor. Failures remain evidence; there is no
automatic broadened test selection or repeat of earlier component QA.

The proposed launch is one explicit three-node pytest selection under the
existing monitored 900-second/2,048-MiB ceiling, after source review and doctor.
Those ceilings are guards, not a measured performance prediction. No tests,
models, feature calls, children or checkpoint loads have been executed while
authoring this document. Mem0 was consulted solely as development context and
provided no premise used by these numerical tests.
