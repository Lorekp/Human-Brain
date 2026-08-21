# Human-cell simulation performance

Performance is part of the model specification, not an optional deployment
optimization. The current engineering gate requires more than 100,000 active
fitted human cells at more than 100 end-to-end recurrent tokens/s while keeping
the registered 0.05-ms integration step, 5-ms token interval, cellular state,
sparse recurrent propagation, and cell-derived output. Results from dormant
padding, shortened simulation, or learned computation outside cells are
ineligible. Every faster backend must first match the deterministic reference
state transition.

## Large-network CUDA backend

Circuits with at least 100,000 reservoir cells now select a persistent-device
CUDA backend when a cooperative-launch NVIDIA device and NVRTC are available.
`HUMANBRAIN_DEVICE=cpu` forces Rust, `HUMANBRAIN_DEVICE=cuda` requires CUDA and
raises instead of falling back, and `auto` is the default. The selected backend
and physical device are observable through `simulation_backend` and
`simulation_device` on `RecurrentCellCircuit`.

One cooperative kernel launch performs the complete 100-step token interval.
Threads traverse contiguous cell-state arrays with grid-stride loops; two
grid-wide barriers per biological step separate fitted GLIF5 integration from
synaptic propagation. Construction converts outgoing sparse targets to a
stable reverse-CSR view. Each postsynaptic thread therefore accumulates active
incoming weights in the same `(presynaptic cell, edge)` order as the Rust
scatter, avoiding nondeterministic floating-point atomics. NVRTC compiles with
fused multiply-add disabled and precise division/square-root settings. A
64-token test compares every mutable cell/synapse array and spike count
bit-for-bit against Rust, then verifies reset synchronization and repeated-token
persistence.

Device state persists between tokens. The current exact public-state contract
downloads voltage, both after-spike channels, both thresholds, refractory
state, synaptic current, eligibility, and the spike total after every token.
That is 60 bytes per cell plus eight bytes device-to-host per token; steady
observation performs no host-to-device transfer. Reporting this boundary
separately prevents kernel-only throughput from being mistaken for end-to-end
throughput.

This follows NVIDIA's documented
[cooperative-grid synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
pattern and its [best-practices guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
to keep iterative state on device and use coalesced cell-array accesses. The
kernel is compiled directly to device code through NVRTC, so it does not depend
on a host C++ compiler. CPU execution remains the exact backend for smaller
circuits and systems without CUDA.

The registered `benchmarks/cuda_scaling.py` protocol measures 100,000, 200,000,
and 1,000,000 active cells. It records wall-clock and device-event throughput,
cell steps, six sparse edges per cell, host/device persistent memory, exact
transfer bytes, construction/first-token/cold-build time, device identity,
GPU-utilization snapshots, and equivalence evidence. Final performance numbers
must come from an isolated GPU window. Its end-to-end recurrent-observation
timing includes device-to-host synchronization but excludes the separate dense
readout calculation and output-cell race.

The isolated RTX 4070 SUPER run passes the strict size/speed gate at 200,000
enabled cells. Both completed sizes execute all 100 fitted integration steps,
touch every enabled cell and every sparse recurrent edge on every
step, and synchronize all mutable state back to the host:

| Enabled/touched cells | Edges examined/step | End-to-end tokens/s | Kernel tokens/s | Cell-steps/s | Host arrays | Device arrays | D2H/token | Spikes/token | Construct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 600,000 | 263.26 | 363.07 | 2.633B | 214.05 MiB | 13.19 MiB | 6,000,008 B | 3.3125 | 5.75 s |
| 200,000 | 1,200,000 | 144.03 | 183.79 | 2.881B | 428.05 MiB | 26.35 MiB | 12,000,008 B | 1.1250 | 52.61 s |

The 200,000-cell result is above both required strict inequalities: more than
100,000 cells and more than 100 synchronized recurrent observations/s. The
100,000-cell boundary result is 13.1x faster than the 98,304-cell Rust baseline,
although it does not satisfy the strict cell-count inequality. Cold NVRTC
compilation took 0.883 s and a cached lookup 0.255 ms. No host-to-device bytes
are transferred during a normal token; reset is a separate explicit state
upload.

The spike counts are deliberately not hidden. Only 424 spikes occurred across
12.8 million cell-steps at 100,000 cells, and 72 across 6.4 million cell-steps
at 200,000. Thus this result proves exact large-array simulation throughput,
not that the present fixed 32-cell afferent fanout recruits a useful fraction of
the larger brain. Scaling input recruitment, propagation, and state diversity
without losing throughput remains an architecture/capability requirement.

The registered 1,000,000-cell full-circuit attempt was stopped after a 270 s
operational bound during construction, before timing. It reached 2,809.24 MiB
of process-private host memory and 2,709 MiB total observed GPU use; no 1M CUDA
throughput is claimed. The dense 257-by-reservoir readout and Python topology
construction, rather than recurrent CUDA state, are now the full-circuit scale
barriers. The frozen report is `artifacts/cuda_scaling_v1.json`.

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
$env:HUMANBRAIN_NATIVE_CACHE=(Resolve-Path '.runtime/native').Path
$env:HUMANBRAIN_DEVICE='cuda'
python benchmarks\cuda_scaling.py
```

The CUDA artifact was measured on 2026-08-21 with Python 3.11.15 and NumPy
2.4.6 on Windows (AMD64 Family 25 Model 97). Throughput values are medians of
three runs from `benchmarks/cuda_scaling.py`; reported persistent memory is the
sum of unique host arrays or allocated device buffers, not a process peak.

## Exact CPU frontier and activity density

Native ABI 3 retains the frozen serial implementation as a correctness oracle,
adds deterministic Rayon-parallel dense chunks, and adds an exact quiescent
frontier. The frontier scans every configured cell once per token, then skips a
cell's zero-to-zero integration only until an afferent or recurrent event can
change it. It preserves topology, spikes, and all persistent state bytes over
the registered 64-token comparison.

| Configured cells | Natural-trajectory token/s | Maximum updated cells/token | Dense token/s | Persistent arrays |
| ---: | ---: | ---: | ---: | ---: |
| 98,304 | 843.58 | 2,466 (2.51%) | 79.15 | 209.76 MiB |
| 131,072 | 754.17 | 2,004 (1.53%) | 61.53 | 279.67 MiB |
| 393,216 | 449.83 | 937 (0.24%) | 25.92 | 838.92 MiB |
| 1,048,576 | 193.36 | 544 (0.052%) | 3.59 | 2,237.04 MiB |

Configured size is not activity density. Matched seeded states were therefore
run through both frontier and dense execution at controlled nonquiescent
fractions; every pair was byte-identical. At 131,072 cells the frontier records
568.02/176.15/51.50/11.71 tokens/s at 1/5/20/100% updated cells. At 1,048,576
cells it records 53.25/5.27/3.00/1.41. The CPU >100-token/s claim is therefore
valid through roughly 5% activity at 131,072 cells, while the million-cell
193.36-token/s result is explicitly a 0.052%-updated trajectory.

Direct structured builders now produce six million edges at one million cells
without population-sized candidate rows. Registered construction ranges from
15.74 seconds for single-scale groups to 48.43 seconds for a 3D volume, with
linear target-array memory. See `docs/native-scaling-v2.md` for phase timings,
topology invariants, locked Rayon dependencies, protocols, and artifact hashes.

## Frozen native v1 baseline

The fitted GLIF5 integration loop runs in a Rust `cdylib`.
Python creates topology and checkpointable NumPy state, then makes one C-ABI
call per token. All 100 fitted 0.05-ms integration steps, refractory updates,
eligibility decay, and spike-triggered sparse scatter happen inside that call.
The production recurrent path therefore contains no Python simulation loop and
allocates no dense cell-by-cell connectivity.

`benchmarks/native_scaling.py` uses 257 token cells, fixed 32-cell afferent
fanout, six recurrent targets per cell, and three timing repeats. Its Python
reference is the removed production NumPy loop, not a simplified model.

| Reservoir cells | Native tokens/s | Cell-steps/s | Persistent arrays | Recurrent edges |
| ---: | ---: | ---: | ---: | ---: |
| 1,536 | 1,045.45 | 160.58M | 3.34 MiB | 9,216 |
| 6,144 | 301.70 | 185.37M | 13.22 MiB | 36,864 |
| 24,576 | 82.49 | 202.73M | 52.73 MiB | 147,456 |
| 98,304 | 20.05 | 197.10M | 210.79 MiB | 589,824 |

At 1,536 cells, the exact Python reference ran at 154.23 tokens/s and 23.69M
cell-steps/s, so the native token kernel is 6.78x faster. Fitted across all four
native sizes, seconds per token scale as `N^0.949`, persistent memory as
`N^0.997`, and cell-step throughput as `N^0.051`: useful near-linear scaling
through 98,304 fitted human cells rather than a hundreds-cell ceiling.

For the frozen v1 source, a clean release compile took 1.212 seconds, cached
source lookup 1.15 ms, and DLL loading 0.48 ms. The current ABI 3 crate uses the
locked Rayon dependency set described in `docs/native-scaling-v2.md`. The
content-addressed build is cached outside the package; set
`HUMANBRAIN_NATIVE_CACHE` to choose its location.

The v1 bottlenecks motivated the current work. At 98,304 cells it spent 7.66
seconds building topology while the first token took 48.6 ms, and the dense
token-cell readout accounted for 192.75 MiB of 210.79 MiB persistent arrays.
ABI 3 and topology protocol v3 now remove the serial integration and candidate
construction limits. The dense `O(vocabulary*cells)` readout remains the main
large-model storage/output-computation target. Exact ridge learning remains the
separate `O(N^2)` memory / `O(N^3)` solve limit described below.

The frozen v1 result did not pass the project gate: 98,304 is below the required
cell count and 20.05 tokens/s is below 100. It remains the bit-exact historical
baseline against which the current CPU frontier and CUDA paths are measured.

Run the registered native protocol with:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
$env:HUMANBRAIN_NATIVE_CACHE=(Resolve-Path '.runtime/native').Path
python benchmarks\native_scaling.py
```

## Before/after results

The baseline is the unmodified Human Brain source used to create the
`speed-optimization` worktree. Both versions ran the same benchmark script and
deterministic seeds.

| Workload | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| 1,536-cell GLIF5 integration | 27.43M cell-steps/s | 40.82M cell-steps/s | 1.49x |
| 1,536-cell recurrent observation | 72.00 tokens/s | 99.23 tokens/s | 1.38x |
| 1,536-cell inference arrays | 24.35 MiB | 3.41 MiB | 7.13x smaller |
| 1,536-cell recurrent growth | 32.52 examples/s | 64.00 examples/s | 1.97x |
| 256-cell recurrent output race | 119.5 races/s | 14,169 races/s | 118.6x |
| 256-cell dendritic output race | 7,858 races/s | 20,789 races/s | 2.65x |
| Held-out recurrent code evaluation | 103.57 tokens/s | 618.06 tokens/s | 5.97x |

The held-out evaluation retained exactly 30/128 correct tokens, 23.4375%
accuracy, 6.931640625 ms mean biological spike time, and 7,776 learned
synapses in both versions.

## Scaling through 1,536 cells

The optimized short-run measurements were:

| Cells | Raw cell-steps/s | Recurrent tokens/s | Inference arrays | Growth examples/s | Growth peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 6.65M | 348.57 | 0.23 MiB | 303.78 | 1.04 MiB |
| 384 | 17.44M | 208.57 | 0.86 MiB | 122.75 | 5.77 MiB |
| 768 | 26.08M | 121.73 | 1.71 MiB | 92.72 | 16.01 MiB |
| 1,536 | 40.82M | 99.23 | 3.41 MiB | 64.00 | 49.99 MiB |

Across 96/384/768/1,536 cells, fitted elapsed-time exponents were 0.35 for
raw GLIF5 steps, 0.47 for recurrent observation, and 0.56 for the 128-example
growth run. Inference persistent memory changed from a 1.41 empirical exponent
in the baseline to 0.98 after making dense growth state lazy. Growth peak
memory remains superlinear (1.38 empirical exponent) because exact ridge
learning still requires a dense normal matrix.

Run the registered synthetic protocol with:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python benchmarks\simulation_speed.py --sizes 96,384,768,1536 `
  --cell-steps 2000 --tokens 64 --training-tokens 128 --races 256 --repeats 3
```

## Equivalence and implementation boundaries

- 2,000 randomized GLIF5 steps, including spikes and refractory intervals,
  preserve spike masks exactly. Voltage, after-spike current, and spike
  threshold match to `1e-13`; the algebraically reduced voltage-threshold state
  matches to `1e-10` mV.
- Compiled recurrent output races preserve winner, spike time, simulated steps,
  active-cell count, and peak voltage against stepwise GLIF5 populations.
- Dendritic search races match the previous time-by-cell response matrix over
  31 deterministic trained contexts.
- Batched readout sufficient statistics match sequential outer products within
  `2e-13` relative error; learned coefficients match within `2e-10`.
- The full suite passes 50 tests. The cell-only architecture, fitted cell
  parameters, calibrated biological timing, recurrent topology, and checkpoint
  format are unchanged.

## Remaining scaling limits

Inference is now sparse/linear in reservoir size: the dense `N x N` ridge
matrix is not allocated unless the circuit learns. Learning batches 128 cell
traces into a BLAS rank-k update, removing one Python/NumPy `N x N` outer
product per token. Exact ridge consolidation still stores an `O(N^2)` normal
matrix and uses an `O(N^3)` dense solve. That is now the dominant obstacle to
networks much larger than a few thousand learning cells. Replacing it requires
a separately validated local or iterative learning rule; changing that rule in
this speed-only patch would change registered model behavior.

## Experimental local-learning follow-up

The integrated experimental path adds a native competitive three-factor rule
without changing ridge as the default. The recorded train-only-calibrated
comparison is `artifacts/local_learning_scaling_v1.json`:

| Cells | Local train | Ridge train | Local peak | Ridge peak | Local persistent | Ridge persistent | Local / ridge teacher-forced | Local / ridge free-running |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.648 s | 0.978 s | 1.00 MiB | 1.50 MiB | 0.25 MiB | 0.60 MiB | 3.91% / 35.16% | 14.06% / 6.25% |
| 384 | 0.708 s | 1.638 s | 1.62 MiB | 6.21 MiB | 0.86 MiB | 3.12 MiB | 6.25% / 34.38% | 7.81% / 3.13% |
| 1,536 | 1.957 s | 3.914 s | 4.08 MiB | 50.34 MiB | 3.32 MiB | 25.84 MiB | 7.81% / 8.59% | 6.25% / 7.81% |
| 6,144 | 6.657 s | skipped | 13.92 MiB | skipped | 13.14 MiB | skipped | 10.16% / n/a | 4.69% / n/a |

Ridge at 6,144 was excluded by the protocol's safety limit: the normal matrix
alone would occupy 288.09 MiB before its copy and cubic solve. The local run
allocated no normal matrix at any size. Scaling is demonstrated; accuracy
parity and useful free-running generation are not. See
`docs/local-readout-learning.md` for rule details and integration.

A separate fixed-sequence smoke makes no accuracy claim and touches no dataset:

| Cells | Construct | 32-example growth | Examples/updates per s | Synapse-weight events/s | Peak | Persistent | Ridge normal matrix avoided |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24,576 | 3.425 s | 0.377 s | 84.96 | 4.18M | 53.87 MiB | 52.45 MiB | 4.50 GiB |
| 98,304 | 31.257 s | 3.425 s | 9.34 | 1.84M | 211.70 MiB | 209.67 MiB | 72.00 GiB |

Both runs applied 32 competitive updates and asserted that the normal matrix,
target moments, and batch workspace remained absent. The frozen report is
`artifacts/local_learning_large_scale_v1.json`. At 98,304 cells, Python
topology construction (31.26 s) is now much slower than local growth (3.43 s),
so moving construction into the native module is the next clear speed target.
