# Human-cell simulation performance

Measured on 2026-08-21 with Python 3.11.15 and NumPy 2.4.6 on Windows
(AMD64 Family 25 Model 97). Results are medians of three runs from
`benchmarks/simulation_speed.py`; peak memory includes Python and NumPy
allocations observed by `tracemalloc`.

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
- The full suite passes 45 tests. The cell-only architecture, fitted cell
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
