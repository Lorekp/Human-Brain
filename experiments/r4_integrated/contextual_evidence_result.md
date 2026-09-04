# Contextual evidence selection: development result

Recorded 2026-09-05. **The candidate failed fresh executable transfer.** This
closes the one-encoder/two-rate comparison in the [protocol](contextual_evidence_protocol.md).
The encoder remains an experimental feature source, not an adopted architecture.
No protected gate or independent benchmark was run.

## Measured behavior

Eight teaching cases, six transfer cases (five supported and one unsupported),
eight old-task replays, and the original KS2 Q6a/b were evaluated. Both learning
rates received 16 executed feedback events; their frozen controls replayed the
identical action/outcome exposure while retaining identical source stores.
The common occurrence motor was learned from two executed demonstrations.

| Evidence selector | Fresh executed tasks | Old-task replay | Unsupported no-action |
| --- | --- | --- | --- |
| Updating neuronal, rate 1 | 0/5 | 1/8 | 0/1 |
| Updating neuronal, rate 3 | 0/5 | 0/8 | 1/1 |
| Frozen neuronal, either rate | 0/5 | 0/8 | 1/1 |
| Conventional learned scores, either rate | 0/5 | 1/8 | 0/1 |
| External cosine reference, either rate | 0/5 | 0/8 | 0/1 |

Rate 1 emits `velvet` on every transfer and old-task request. Rate 3's
conventional selector emits `cedar` throughout; its neuronal path stops on
12–19 simultaneous spikes in those phases. Thus the single old-task success
is a constant-answer match, not evidence of retained query-dependent skills.
Rate 3's correct unsupported no-action comes from the same collision failure,
not demonstrated recognition of missing evidence. All frozen responses are STOPs.

Every selected evidence span reaches a real Python edit and execution. No
selected action stops at the motor, and all decoys remain unchanged. The
external reference sometimes selects multiword spans; these are copied and
executed without filtering, including a quote-containing span that causes a
Python syntax failure. No teacher repair counts as learner success.

All exam arms have zero **confirmed** marks. Rate 3 neuronal and frozen STOPs
score 0/2. Other responses (`velvet`, `cedar`, or `to`) remain formally pending
the existing official-rubric human scoring rule; no LLM mark was substituted.
The exam was an untuned diagnostic with the earlier coding documents still in
the external store. No complete reading-comprehension score is claimed.

## Why learning did not generalize

The shared score contains pure query, span, query–span interaction,
query–window interaction, lexical-overlap, and bias terms. Pure query and bias
terms are identical across candidates and cannot change their ranking; the
window term cannot distinguish two values inside the same window.

The local GO/NO_GO update rewards an individual candidate without requiring it
to outrank the failed candidate for that query. At rate 3 the last three teacher
positives produce no update even though global selection remains wrong: their
individual margin is already satisfied. The query-independent span block holds
68.5%/93.0% of squared weight norm at rates 1/3. These are weight measurements,
not causal attribution percentages. Constant outputs and poor conventional
results identify ranking as the primary failure; finer spike resolution alone
would still leave fresh transfer at 0/5.

The [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
therefore compares absolute feedback with a same-query pairwise learning rule
under counterbalanced teaching, retaining this encoder, candidate family and motor.

## Resources, reproduction and verification

The [initial result](../../benchmarks/contextual_evidence_python_development_v1.json)
used the attested temporary native cache. The
[checkout-cache replay](../../benchmarks/contextual_evidence_python_development_v1_local_native.json)
used the preflight's exact DLL and took 22.48 seconds with allocation tracing.
Both runs have identical source hashes, selection decisions, learned checkpoints,
synaptic arrays, exam responses and outcomes. The replay is an environment check,
not a second independent cohort. Both JSON files and adjacent `.npz` synaptic
checkpoints are retained; source files stayed unchanged throughout each run.

The replay's process peak working set was 244.95 MiB and final private memory
553.30 MiB, including the shared encoder, paired memories, evaluator and result
records but excluding child Python executors. Traced Python peak was 91,436,416
bytes. Per-arm retained byte/array payload grew to 1,986,829 bytes for 14 coding
documents and 615 candidates, then 7,357,628 bytes for 15 documents and 2,343
candidates after the exam. These payload counts exclude Python object overhead
and shared encoder memory; the ONNX model file is 90,405,214 bytes. Raw source
text remains stored: this is hybrid episodic retention, not semantic compression.

Before a direct experiment invocation, set the native environment explicitly:

```powershell
$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'
$env:HUMANBRAIN_DEVICE = 'cpu'
./.venv/Scripts/python.exe -m experiments.r4_integrated.contextual_evidence_python_session --output benchmarks/contextual_evidence_python_repeat.json
```

Six real-encoder/native tests and one motor regression passed. They cover
feedback and source custody, frozen weights, immutable inference, UTF-8 pooling,
overlength rejection, incidental query words, multiword actuation, and ambiguous
motor selection. These verify the implementation contracts; the capability
comparison above failed. Existing frozen modules and evidence were not changed.
