# FINDINGS: host-device overlap for the CUDA LM grow epoch (double buffering)

Worktree branch `codex/scale-opt-gpu`, RTX 4070 SUPER (sm_89), CUDA 12.6 NVRTC,
`benchmarks/lm_million_smoke.py` grow pattern (tokenizer fit -> observe+strengthen
over verse texts -> read/race eval), sparse_three_factor readout,
host mirror = eligibility only.

## Task 1 - measured idle gaps (instrument mode, one grow epoch, N=131072)

Per-token timeline of the synchronous baseline (`experiments/lm_scale_bridge/
overlap_instrument_pre_131072.json`, 400 tokens):

| segment                          | mean      | share of wall |
| ---                              | ---       | --- |
| CUDA cooperative kernel (events) | 3.887 ms  | ~85% |
| D2H eligibility mirror (1 MB)    | 0.193 ms  | ~4%  |
| strengthen() host learning       | 0.376 ms  | ~8%  |
| python launch/marshal overhead   | 0.496 ms  | ~11% |

GPU-idle fraction of wall: **15.1%** (10.9% even when DMA-busy time counts as
busy). Decision threshold was 5% -> overlap work justified. H2D traffic is
**zero by design**: tokens select rows of a device-resident afferent table via
pointer offset, so "prefetch next batch H2D" does not apply; the real
serializers are the blocking `cuEventSynchronize`, the synchronous pageable
D2H mirror copy, and host learning between launches.

Token preparation (task 3 premise) measured negligible: tokenizer encode is a
once-per-text trie walk (`encode_prep_seconds` = 0.000-0.004 s per epoch);
there are no per-token python token objects to batch. The per-token python cost
lives in ctypes argument marshaling, which we made O(1) instead (cached launch
argument block, two patched pointer cells per call).

## Task 2 - double-buffered overlap pipeline (implemented)

`src/humanbrain/cuda.py` (`CudaRecurrentKernel`):

- compute stream + non-blocking copy stream;
- per-slot device eligibility snapshots: compute stream runs
  `[kernel -> D2D primary->snapshot]` back-to-back so the snapshot stays stable
  while later kernels overwrite the shared primary buffer (no .cu ABI change);
- pinned host staging ring (`cuMemHostAlloc`, depth 4) drained by
  `cuMemcpyDtoHAsync` on the copy stream, gated on per-slot kernel-done events;
- `observe_async(token)` enqueue-only submit + `collect(handle)` publish into
  the host mirror; `drain()` guards every synchronous entry point;
- cached cooperative-launch argument block (per token only two pointer cells
  change), removing marshal cost from the critical path;
- `observe()` keeps exact synchronous semantics; pipeline is on by default and
  disabled with `HUMANBRAIN_CUDA_ASYNC=0`.

`src/humanbrain/recurrent.py`: `grow_sequence` drives submissions up to depth-1
tokens ahead of host learning (`_grow_sequence_pipelined`), vectorized token-id
normalization + up-front range validation.

Entanglement note: the decayed-state activity bitmap (commit af14c90, sibling
workstream) landed in the same `cuda.py` working tree and its commit swept the
pipeline code in; this commit carries the recurrent/benchmark/test halves.
All post-sync vs post-async comparisons below run under identical committed
kernel code, isolating the overlap contribution.

## Gates

Byte-identical trajectory, seed 17, N=131072, 600 grown tokens
(`gate_pre_sync.json`, `gate_post_sync.json`, `gate_post_async.json`):

| label        | HUMANBRAIN_CUDA_ASYNC | checkpoint sha256 | winner replay sha256 |
| ---          | --- | --- | --- |
| pre-sync     | unset | d3a08356634ae007...35974 | f25b6886f2afbbad...2de7 |
| post-sync    | 0     | d3a08356634ae007...35974 | f25b6886f2afbbad...2de7 |
| post-async   | 1     | d3a08356634ae007...35974 | f25b6886f2afbbad...2de7 |

Identical learned synapses (7), local updates (597), structural recruitments
(13); 300-token A/B replay through the pipelined `grow_sequence` also hashes
equal (`3ea8f3e508f1ce77...`). PASS.

## Throughput

N=131072, 400 tokens, warmup pass then measured pass, two interleaved runs each:

| pipeline | tok/s run 1 | tok/s run 2 | ms/token |
| ---      | ---         | ---         | --- |
| sync     | 232.4       | 232.3       | 4.304 |
| async    | 273.0       | 272.8       | 3.664 |

**+17.5% at 131k** (reproduced independently: interleaved A/B rounds and a
lookahead probe peaked at 272.3 tok/s).

N=1000000, 300 tokens, same protocol:

| pipeline | tok/s run 1 | tok/s run 2 |
| ---      | ---         | ---         |
| sync     | 39.3        | 34.1        |
| async    | 40.8        | 39.2        |

**~+9% at 1M** (mean 36.7 -> 40.0 tok/s; wider variance because the kernel now
dominates the token budget). Absolute rates here also reflect the af14c90
dormancy bitmap, which shrank the 1M kernel far below the historical 43.6 ms;
the async delta is the overlap contribution alone.

Unit coverage: `tests/test_cuda_pipeline.py` (bit-exact sync-vs-async growth +
checkpoint hash, sync observe unchanged while armed, stall/order guards,
drain-on-synchronize equivalence, disabled fallback, up-front validation).

## Caveats

- Async throughput is occasionally noisy on this desktop (single outlier epoch
  at 213 tok/s during concurrent system load); the isolated bench battery is
  the authoritative measurement.
- At 1M the remaining non-kernel budget is small; further gains there need
  kernel-time reduction, not more overlap.
