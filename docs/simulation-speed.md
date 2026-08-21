# Human-cell simulation performance

Performance is part of the model specification, not an optional deployment
optimization. The current engineering gate requires more than 100,000 active
fitted human cells at more than 100 end-to-end recurrent tokens/s while keeping
the registered 0.05-ms integration step, 5-ms token interval, cellular state,
sparse recurrent propagation, and cell-derived output. Results from dormant
padding, shortened simulation, or learned computation outside cells are
ineligible. Every faster backend must first match the deterministic reference
state transition.

Measured on 2026-08-21 with Python 3.11.15 and NumPy 2.4.6 on Windows
(AMD64 Family 25 Model 97). Results are medians of three runs from
`benchmarks/simulation_speed.py`; peak memory includes Python and NumPy
allocations observed by `tracemalloc`.

## Native sparse recurrent kernel

The fitted GLIF5 integration loop now runs in a dependency-free Rust `cdylib`.
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

The Rust crate has no third-party dependencies. On the recorded Windows host,
a clean release compile took 1.212 seconds, cached source lookup 1.15 ms, and
DLL loading 0.48 ms. The content-addressed build is cached outside the package;
set `HUMANBRAIN_NATIVE_CACHE` to choose its location.

The remaining inference bottlenecks are now visible. At 98,304 cells, topology
construction takes 7.66 seconds in Python while the first simulated token takes
48.6 ms. The dense token-cell readout is `O(vocabulary*cells)` and accounts for
192.75 MiB of the 210.79 MiB persistent arrays; sparse recurrent targets and
weights use 6.75 MiB. Steady execution is a single native fused pass and is
primarily cell-state memory traffic. Larger runs should next move topology
construction out of Python and replace or factor the dense readout, then add
carefully validated native parallelism. Exact ridge learning remains the more
fundamental `O(N^2)` memory / `O(N^3)` solve limit described below.

The frozen v1 result therefore does not yet pass the project gate: 98,304 is
below the required active-cell count and 20.05 tokens/s is below 100. It is the
bit-exact baseline against which CPU and GPU work is measured.

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

The separate `local-learning` branch adds a native competitive three-factor
rule without changing ridge as the default. The recorded train-only-calibrated
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
