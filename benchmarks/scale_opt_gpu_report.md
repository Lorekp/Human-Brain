# Scale-optimization GPU report: CUDA cell-circuit path

Branch: `codex/scale-opt-gpu` (worktree `.worktrees/human-brain-scale-opt-gpu`, base `49eb043`)
Host: RTX 4070 SUPER 12 GB, single GPU, every process < 8 GB VRAM (peak process
device allocation observed ≈ 132 MiB arrays + driver context; system GPU pool
stayed ≈ 2.5 GiB including the Windows desktop).
Date: 2026-08-22.

## Scope and constraints kept

- Exactness contracts unchanged: `tests/test_cuda.py` bit-exact CPU/CUDA state
  parity after every token, reset/replay parity, and
  `device_to_host_bytes_per_token` mirror-size assertions all pass unmodified.
- Full-touch semantics kept: every active cell is integrated for all 100
  fitted 0.05 ms steps of every token on every path. The only work elided is
  the reverse-CSR edge scan on steps where **zero** cells spiked, which is a
  bit-exact no-op (every `spike_mask[source]` would read 0, leaving
  `next_current == synaptic_current * recurrent_decay` exactly).
- No protocol artifacts were touched (`artifacts/` untouched, no
  sparse_cuda_engineering `--execute` runs). All measurements come from the
  new engineering harness `benchmarks/scale_opt_gpu_bench.py`, which mirrors
  the registered workload (observe+race inference, observe+strengthen
  training, bootstrap/warmup phases, seed-2317 circuit family) and writes JSON
  next to itself.

## Baseline vs optimized

End-to-end tokens/s over 64-token timed phases (single runs; medians below).
Baseline = branch point `49eb043`; optimized = this branch.

| cells | construct base→opt (s) | infer tps base→opt | train tps base→opt | kernel tps base→opt | spikes/token | device MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5.70*→0.64* | 1041→1115 | 997→1047 | 1329→1363 | 28.9 | ~0 |
| 1 000 | 0.11→0.09 | 869→827 | 793→820 | 1038→972 | 20.5 | 0.2 |
| 10 000 | 0.18→0.15 | 736→808 | 687→768 | 886→969 | 3.98 | 1.3 |
| 100 000 | 0.81→0.15 | 346→**513** | 328→**471** | 395→611 | 3.11 | 13.2 |
| 200 000 | 1.94→0.81† | 131→**308** | 129→**290** | 198→378 | 2.61 | 26.4 |
| 500 000 | 3.46→0.42 | 72→**148** | 71→**144** | 76→164 | 2.55 | 65.8 |
| 1 000 000 | 6.93→0.85 | 26→**35** | 25→**34** | 27→36 | 1.53 | 131.6 |

\* first size in each fresh process pays CUDA context init + NVRTC module load.
† first size in its process pays context/module init (~0.6 s); pure
construction is ≈ 0.22 s (median of repeats).

Three-repeat medians (fresh circuits per repeat, identical harness; raw
per-repeat results committed as `scale_opt_gpu_medians-small.json` /
`scale_opt_gpu_medians-large.json`):

| cells | infer tps median | train tps median | kernel tps median | construction median (s) |
|---:|---:|---:|---:|---:|
| 100 000 | **479.0** | **466.0** | 571.8 | 0.16 |
| 200 000 | **323.8** | **312.7** | 371.0 | 0.22 |

Run-to-run spread across those repeats was ±8.5 % (infer) / ±7.1 % (train) at
100 k — the first repeat pays context/module warm-up — and ±2.7 % / ±1.9 % at
200 k.

## Techniques applied (3)

1. **Vectorized `uniform_no_self` topology construction** (`topology.py`).
   The legacy builder drew `rng.choice(cells-1, fanout)` once per source in
   Python — O(cells) interpreter round-trips. The new path draws an
   `(cells, fanout)` candidate block, detects duplicate values per row via a
   row sort, redraws whole conflicted rows (rejection conditioning stays
   uniform), then applies the identical `c += c >= source` self-exclusion map.
   Size-gated at `_UNIFORM_NO_SELF_VECTOR_MIN_CELLS = 100_000`: below it the
   exact legacy seeded stream is preserved (pinned by
   `test_uniform_sampling_preserves_legacy_seeded_targets`). A first version
   of this fix redrew *columns* flagged in sorted-position space while writing
   original-position columns — correct-looking but non-convergent on some RNG
   stream states (41 s churn at 200 k caught by profiling before any claim);
   the committed row-level redraw converges geometrically.
   Result: topology build at 200 k went ~1.7 s → 0.03 s; at 1 M, 0.13 s.
   Whole-circuit construction at 1 M is now 0.85 s including uploads and the
   reverse-CSR build, removing construction as the million-cell blocker.
2. **Bit-exact zero-spike edge-pass elision** (`cuda.cu`). After each
   per-step `grid.sync()`, all threads read the atomic spike total and
   uniformly skip the reverse-CSR weight scan while it is still zero. That
   counter is zeroed only once per kernel launch — one launch integrates one
   whole token, all 100 of its steps — so elision operates at whole-token
   granularity: a single spike anywhere in a token ends its edge-scan
   elision for every remaining step. On this workload 78 % of tokens are
   entirely spike-free, so ≥78 % of all integration steps skip the scan
   (steps before a spiking token's first spike also skip), and the kernel
   gains (395→611 tps at 100 k, 198→378 tps at 200 k) are consistent with
   removing roughly that share of edge-scan memory traffic while touching
   every cell's neuron update, synaptic decay, and eligibility trace exactly
   as before. The win is fragile with respect to the spike distribution:
   denser firing, or spikes spread across more tokens, drives the
   spike-free fraction — and with it the gain — toward zero.
3. **Eligibility trace fused into the neuron pass** (`cuda.cu`). The trace
   update used only the thread's own `crossed` value, so it moved from the
   post-sync synapse pass into the neuron pass, dropping one full-array
   read+write plus a mask re-read per step. Measured effect is within
   run-to-run noise (bounded ≤ ~15 % by the traffic model); kept because it
   strictly reduces device memory movement with bit-identical results.

## Honest framing (activity-qualified)

Per-token reservoir spike events and responsive (non-zero eligibility) cell
counts are recorded in every result JSON (e.g. 200 k cells: 2.61 spikes/token,
167 responsive cells; 128 cells: 28.9 spikes/token). Throughput numbers above
are array-throughput at those activity levels with full-touch integration;
they are not language or capability claims.

## Where the time goes now / next bottlenecks

- Kernel time scales linearly with cells and is device-bandwidth-bound:
  measured effective bandwidth ≈ 370 GB/s (≈ 73 % of the 504 GB/s spec) at
  1 M cells. The remaining traffic is the irreducible full-touch float64 SoA
  neuron update (~90 B/cell/step). Cutting it further requires breaking a
  contract: fp32/mixed precision breaks bit-exact replay; skipping low-voltage
  cells breaks the full-touch claim.
- End-to-end overhead above kernel is now ~15–20 % at 100–200 k (event sync +
  8 B/cell eligibility download + host race()/strengthen()), pinned by the
  `device_to_host_bytes_per_token == eligibility.nbytes + 8` contract.
- Host-side `race()`/`strengthen()` are constant-cost (token_cells-sized) and
  negligible at scale.
- Construction is no longer a bottleneck at any demonstrated size.

## Feasibility extrapolation on this card

- **100 k cells**: gate comfortably passed — ~479/466 inf/train tps median
  (>100 tps required), 144–209 responsive cells across committed runs.
- **200 k cells**: ~324/313 tps median — 2.05x/2.08x the previously
  registered engineering result lineage (158.18/150.00).
- **1 M cells**: demonstrated on-card end to end (construct 0.85 s, 35/34
  inf/train tps single-run, 131.6 MiB device arrays). >100 tps at 1 M is not
  reachable under current bit-exact float64 full-touch contracts on a 4070
  SUPER: it needs ≤10 ms/token vs the ~21 ms/token bandwidth roofline
  (~48 tps absolute ceiling at 73 % efficiency ≈ 21 ms → measured 29 ms).
- **~80 M cells** is the VRAM ceiling for persistent-resident state
  (~138 B/cell device + context overhead under 12 GB), at which point
  throughput would be ~0.4 tps — memory stops binding long before time does.
- **1 B cells**: not feasible on this GPU (≈ 132 GB resident state, ~21 s per
  token at today's roofline ⇒ ~0.05 tps). Requires distributed multi-GPU
  residency plus an algorithmic relaxation (lower-precision or activity-gated
  integration); both out of scope for this exactness-preserving workstream.

## Files

- `src/humanbrain/topology.py` — vectorized gated `uniform_no_self` builder.
- `src/humanbrain/_native_src/src/cuda.cu` — zero-spike edge elision + fused
  eligibility (NVRTC cache digests invalidate automatically).
- `benchmarks/scale_opt_gpu_bench.py` + `scale_opt_gpu_*.json` — harness and
  raw results (baseline-small/large, optimized intermediate, optimized-final,
  three-repeat medians backing the table above).
- Tests: `pytest tests/test_cuda.py tests/test_sparse_cuda_engineering.py -q`
  green after each change (17 tests incl. `tests/test_topology.py`).
