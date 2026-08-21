# Scalable local readout learning

The recurrent GLIF5 model now has two explicit readout-learning rules. `ridge`
remains the default, frozen comparison. `three_factor` is the scalable online
path: it never creates the reservoir-sized normal matrix and never performs a
dense solve.

## Rule and locality

The rule follows the established neo-Hebbian three-factor pattern described by
[Gerstner et al. (2018)](https://doi.org/10.3389/fncir.2018.00053) and the local
eligibility-plus-learning-signal decomposition used by
[e-prop](https://doi.org/10.1038/s41467-020-17236-y). For reservoir eligibility
`e_i`, output-synapse strength `w_ji`, pacemaker strength `b_j`, and fixed
output-cell heterogeneity `h_j`, the native learner computes:

```text
s_j = b_j + h_j + sum_i(w_ji e_i)
k   = strongest non-target output cell
q   = learning_rate / sqrt(1 + sum_i(e_i^2))

if s_target < s_k + competitive_margin:
    w_target,i += q e_i
    w_k,i      -= q e_i
    b_target   += q
    b_k        -= q
```

Each changed synapse therefore receives only its presynaptic eligibility trace,
the state/competitive selection of its postsynaptic output cell, and a signed
target neuromodulatory signal. The trace norm is a transient homeostatic scalar
available to the output cell; it is not learned or checkpointed. Only the two
participating output-cell rows change. The fitted reservoir dynamics,
eligibility decay, and first-spike output race are unchanged.

All persistent learned values are `_readout_synapses_pa` and
`_pacemaker_synapses_pa`. Local checkpoints omit ridge's `normal_matrix` and
`target_moments`. A language model freezes plasticity after growth and before
evaluation or checkpointing.

## Complexity

The competitive current scan is `O(vocabulary * reservoir_cells)` per example,
matching the physical dense readout fan-in. The actual update is
`O(reservoir_cells)`. It uses no per-example allocation and no optimizer state.
Persistent inference state remains `O(vocabulary * reservoir_cells)` because
those are real output synapses; the local rule removes the additional
`O(reservoir_cells^2)` normal matrix and `O(reservoir_cells^3)` solve. If a
model deliberately chooses as many output cells as reservoir cells, the real
readout itself is square; that is distinct from hidden ridge state.

The numerical loops live in the dependency-free Rust kernel. Python selects
the rule and owns checkpointable NumPy synapse arrays, but does not simulate
synaptic updates element by element.

## Integration

```python
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig

model = RecurrentCellLanguageModel.grow(
    training_texts,
    RecurrentModelConfig(
        circuit=RecurrentCircuitConfig(
            reservoir_cells=6_144,
            input_fanout=32,
            recurrent_fanout=6,
            readout_learning_rule="three_factor",
            local_learning_rate_pa=16.0,
        )
    ),
)
assert model._circuit.learning_frozen
```

The registered comparison calibrates candidate learning rates on a named
training-only calibration file, chooses `16 pA`, then opens validation after
the choice and model plasticity are frozen. It defines no test partition and
does not access test data:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python benchmarks\local_learning_scaling.py `
  --sizes 96,384,1536,6144 `
  --ridge-max-size 1536 `
  --output artifacts\local_learning_scaling_v1.json
```

## Recorded validation result

The 2026-08-21 run is in `artifacts/local_learning_scaling_v1.json`. Local
learning reached 6,144 reservoir cells in 6.657 s with 13.92 MiB traced peak
and 13.14 MiB persistent arrays. Its teacher-forced validation accuracy was
10.16% and free-running byte accuracy was 4.69%; the latter remains a failure.

At 1,536 cells, local growth took 1.957 s and 4.08 MiB peak versus ridge's
3.914 s and 50.34 MiB. Accuracy was 7.81% local versus 8.59% ridge. At 96 and
384 cells ridge remained substantially more accurate (35.16%/34.38% versus
3.91%/6.25%), so the local path has demonstrated scaling and basic sequence
capability, not parity with exact ridge. Ridge at 6,144 was intentionally not
run: its normal matrix alone is 288.09 MiB and the cubic solve is outside the
registered safety limit.

## Large-scale smoke (not a capability result)

`artifacts/local_learning_large_scale_v1.json` uses a fixed 32-example
synthetic byte-token sequence and no train, calibration, validation, or test
data. It exists only to exercise allocation and throughput at scales still tiny
relative to a biological brain. At 24,576 cells it sustained 84.96 examples
and competitive updates/s with 53.87 MiB peak and 52.45 MiB persistent arrays.
At 98,304 cells it sustained 9.34/s with 211.70 MiB peak and 209.67 MiB
persistent arrays. The latter avoided a 72.00-GiB ridge normal matrix. Both
runs asserted that all three ridge work arrays remained absent; no accuracy or
generalization claim follows from this smoke.
