# Fixed byte-event preprocessing preflight

Status: prospective; implementation preparation only, before source freeze/run.
This host-only qualification implements the committed
[interface contract](../../docs/research/local-token-event-interface.md).
It performs no neuronal encoding, learning, query inference or action execution.

## Fixed inputs and processing

Use only the 1,024 TRAIN schedule entries in the authenticated multiscale run's
`offline_teacher_roster.json`: its 256 groups of four entries, in saved order.
For each entry process the two document byte strings then its query byte string.
The loader accesses `schedule`, `document_hex` and `query_hex`; it never passes
`checks`, `train` records or `target_hex` to segmentation. Full input-file bytes
are hashed for custody, not used as segmentation guidance. Bind prior completed
run/audit hashes and the extracted document/query-only canonical hash.

Use the fixed cap-16 generic ASCII word/whitespace runs and singleton remaining
bytes. No options, vocabulary, Unicode decoding/normalization or target-guided
merging. For each schedule string perform one single-chunk invocation and one
byte-at-a-time invocation; empty input uses one empty feed. Require identical
ordered event payloads/half-open offsets and exact reconstruction. Each invocation
finishes exactly once and leaves no buffered bytes. No chunk arrival flushes a run.

Authored fixtures are fixed before data: empty bytes, all 256 byte values together,
each of the 256 bytes separately, `a` runs and alternating ASCII space/tab runs
of lengths 15/16/17/31/32/33, mixed ASCII punctuation/control bytes, composed and
combining/multibyte Unicode represented as bytes, malformed UTF-8, and a mixed
alphanumeric/underscore string. Apply the same two invocation styles; for fixtures
of at most 64 bytes also apply every two-part split, including empty halves.
Every repeated byte and invocation is counted. No model or TRAIN output labels
enter authored fixture selection.

## Receipts and fixed decision

Record one input entry with raw hex and reference events plus each invocation's
chunk lengths, event digest, offered-byte count, emitted-event count, feed/finish
counts and maximum internal buffering. Record contiguous offsets, payload equality,
cap bounds and zero buffer after finish. These strings live only in this external
host QA evidence; no source-free neuronal checkpoint is constructed.

All actual host-processing totals include repeats from chunk-invariance checks.
Separate one-pass schedule document/query totals and event-count distributions
preserve their multiplicities. A conceptual end event is counted once per
segmentation invocation for future encoder/controller accounting; the utility
does not emit EOS, and no EOS or neuronal step is actually executed here.
Conceptual byte baseline is raw bytes plus string ends; conceptual writer/query
events are emitted raw events plus those ends. Actual teacher/output/action and
neuronal encoder/controller counts are all zero.

Pass only if every preservation/count check succeeds and the one-pass schedule
documents have strictly fewer conceptual writer events than byte-plus-EOS steps.
No event reduction or a preservation failure closes the candidate for review;
no extra input, cap or repeat is tried. Passing earns implementation of the fixed
encoder/integration QA only. It establishes neither learned binding nor a neural
speedup and does not admit acquisition or advance a protected benchmark.

## Resources and custody

One runner plus one independent saved audit must fit an aggregate 60 seconds
and 2,048 MiB. The orchestrator uses the existing monitored launcher separately
for each process, giving the auditor only the remaining aggregate allowance.
Import/startup, hashing, segmentation, all repetitions, JSON logging and audit
reconstruction count. Orchestration gaps are not computation. The watchdog is
authoritative for process-tree memory; the runner checks its injected Unix
start/deadline and perf_counter elapsed time without importing resource packages.

The runner uses the selected `.venv`, normal `humanbrain.byte_events` import,
and only standard-library modules otherwise. Its source manifest includes the
utility, executed package initializers, runner, recipe, interface contract and
launcher. It must not import Torch, NumPy, model/native/action modules or launch
child processes. The independent audit uses saved bytes and an independently
written byte regex; its processing/reconstruction counts are reported separately.

Use absolute output/admission paths and exactly:

```text
.venv/Scripts/python.exe -X utf8 -m experiments.r4_integrated.local_byte_event_preflight_v1 run --output ABS --admission ABS --admission-sha SHA --revision REV
```

The output directory must not exist. Write `records.jsonl`, `report.json` and
`completion.json` once, with source/input/admission/artifact hashes and completion
status. Source bytes are checked at both boundaries. Admission binds the fixed
policy, original TRAIN inputs, revision, sources and independent auditor identity.
The root owns source freeze, admission and both launches. No qualification run
is authorized by writing this recipe.

## Pre-freeze implementation QA

Doctor passed at `.pytest-tmp/dev/doctor-ssfi6kba/receipt.json`. The new runner
accounting case in the existing `tests/test_tokenizer.py` passed in 0.16 seconds:
`.pytest-tmp/byte-event-preflight-unit-v1.xml`. It checks exact spans and all
chunking/counter arithmetic for only `ab ` followed by byte ff and empty bytes:
ten disposable segmentation streams and 28 raw bytes, no TRAIN schedule input.
There are ten conceptual end events, but zero executed EOS/neural steps.

The utility's separate existing-file QA passed 28 cases in 0.163 seconds, including
26 new segmenter cases: `.pytest-tmp/byte-events-unit-v1.xml`, with doctor
`.pytest-tmp/dev/doctor-exqoufxn/receipt.json`. Its explicit exposure receipt
`.pytest-tmp/byte-events-unit-exposure.json` records 404 valid streams, 10,822
authored bytes, ten invalid calls and one closed-stream feed rejected. These
ordinary tests are separate from this prospective preprocessing qualification;
they do not process the admitted TRAIN schedule. The two new layers therefore
have 29 distinct verified existing-file cases. Ruff and diff checks passed.
