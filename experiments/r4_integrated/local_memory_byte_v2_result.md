# Local memory/byte v2: measured cost return

2026-09-06. The optimized implementation completes its disposable workload and
independent saved-only audit, but the full acquisition is **not admitted**:
**1,468.472 seconds** projected against 900 seconds. The same prospective formula
previously projected 2,013.103 seconds. This is a 27.1% lower projection, not a
capability result or a pass of the resource gate.

Frozen source: `ea2a2954e700c0bff9095a1ee6269bdfb3031be5`.
The [recipe](local_memory_byte_v2.md) and
[cost decision](../../benchmarks/local_memory_byte_v2.cost_decision.json) retain
the five arms, local rule, schedule, original two-times margin and additional
capped-generation charge. No full fit, additional seed or rate sweep ran.

| Arm | Maximum complete sampled lesson | Additional capped generation |
| --- | ---: | ---: |
| Tagged | 0.219 s | 0.015 s |
| Aggregate | 0.093 s | 0.016 s |
| Writer/maps frozen | 0.094 s | 0.016 s |
| Slow weights frozen | 0.078 s | 0.015 s |
| Writes disabled | 0.110 s | 0.016 s |

Projected learning is 1,376.256 seconds, other work 62.216 seconds, and reserved
finalization 30 seconds. Actual outer cost execution took 33.311 seconds;
conservative parent-plus-worker peaks were 792.051 MiB, below 2,048 MiB.
The one launch passed native/source identity and process-tree checks.

The check uses four lessons per arm: twenty genuine first attempts, 200 teacher
byte/EOS offers and 160 enabled decoder update opportunities. It then performs
source-free reload, 100 action attempts and two motor demonstrations. The five
models and their eight shared memory pools retain 124,109,512 tensor bytes.
Unused frozen-writer tags are allocated zeros; no persistent-byte saving is
claimed. All disposable action scores are zero; four lessons per arm do not
establish an acquisition failure or useful acquisition. These weights are discarded.

The independent audit verifies 1,127 source hashes, five artifact hashes,
initial/frozen weights, checkpoint chains, shared memories, zero frozen-credit
tensors, query-only state transport, 102 execution receipts and 100 workspaces.
It does not reexecute models or children. The legacy motor remains a supplied
host-computed edit schema, so even a future exact byte success would require a
separate neuronal action-policy result for the user's exclusive-neuron goal.

Evidence:

- Cost report: `.pytest-tmp/local-memory-byte-v2-cost/report.json`, SHA-256
  `ce9720a42a6d598169099d3a46b05426c8b8dc8be57954cde9de7df36a00f179`.
- Independent audit: `.pytest-tmp/local-memory-byte-v2-cost-independent-audit.json`,
  SHA-256 `cb91f11d91ae5f1ce67e2e1c11d2aafd5111d8090b5261654c8894d31cc8057e`.
- Source-bound admission, completion and launcher hashes are in the cost decision.
- The [speed-development record](../../docs/research/local-writer-speed-development.md)
  retains the 53 passing focused tests, exact forward/update comparisons and
  separate paired timing results.

A subsequent five-lesson profile attributes about 36% of summed complete-lesson
time to finite checks. Hashing and snapshot copies are smaller costs. Its receipt
is `.pytest-tmp/local-complete-lesson-profile.result.json`, SHA-256
`796543ea50a5316c83a6d8b3b3f10b511be3c6d77eb6adfb0771b7dff7bdb038`.
This instrumented diagnostic offered fifty teacher byte/EOS events and reused
no weights. One initial launcher import failed before any lesson; both attempts
remain available. It is not a replacement cost admission.

Keep v2 closed. The next bounded engineering hypothesis is a cheaper, exactly
equivalent finite predicate using existing NumPy on supported CPU tensors,
preserving every validation and leaving cell/synapse arithmetic unchanged.
Require complete-lesson parity and measured gain before another independently
bound cost return. Native predicate/pointwise fusion is a higher-effort fallback,
not an automatic parallel implementation or permission to change the local rule.
