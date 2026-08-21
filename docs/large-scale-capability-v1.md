# Large-scale capability protocol v1

This is a preregistered validation experiment, not a completed result. The
runner is plan-only unless the caller supplies both `--execute` and the exact
SHA-256 of `benchmarks/large_scale_capability_protocol_v1.json`. The protocol
and runner were committed before validation. One subsequent 6,144-cell canary
is frozen in `artifacts/large_scale_capability_validation_v1.json`; its 1/148
partial state supports no scale or topology claim.

## Questions and controls

The experiment separates two questions that a single cell-count curve would
confound:

- `fixed_1024` gives every size the same 1,024 training characters and two
  epochs. It measures capability per fixed experience budget, but can
  undertrain a larger circuit.
- `sqrt_scaled` increases per-file training characters with
  `sqrt(cells / 6,144)`: 256, 512, 1,024, and 1,184 characters at 6,144,
  24,576, 98,304, and 131,072 cells. This is a frozen sublinear compromise
  between starving capacity and making total work scale quadratically.

Both uniform tracks use five paired seeds. The fixed-data architecture track
uses three paired seeds and includes uniform no-self, self-edge-permitting
random, single-scale groups, nested groups, mixed membership, a 1D ring, a 2D
sheet, a six-layer 2D sheet, a 3D volume, and `hypercube_4d`. The last topology
is an abstract four-dimensional functional lattice with five local edges and
one sparse long edge per cell. It tests dimensional organization; it is not a
claim about human anatomy.

All conditions use the same fitted Allen human GLIF5 equations, 0.05-ms step,
5-ms token interval, recurrent outdegree six, recurrent current, validation
folds, and evaluation budgets. Afferent fanout scales as `round(cells / 12)` so
direct recruitment density is matched instead of vanishing with cell count.
Only the native three-factor rule is allowed. Its 16-pA rate was selected in a
prior train-only experiment and is not recalibrated here. Any ridge allocation
fails the run.

## Anti-overfitting boundary

The validation files are `src/humanbrain/artifacts.py`,
`src/humanbrain/eval_bridge.py`, and `src/humanbrain/swe_model.py`. Their
canonical hashes are frozen in the protocol, and their paths had zero mentions
in pre-registration artifacts and protocols at base commit `339c3be`. The
runner verifies both properties before execution. There is no test partition,
no validation-time adaptation, and no result-dependent stage selection.

The four stages contain 148 frozen runs: 20 uniform fixed-data, 20 uniform
square-root-data, 36 group, and 72 geometry runs. `--limit-runs` only stops
after the next runs in this order; it cannot select a stage. Every completed
run is written atomically and later invocations resume at the first missing
run.

## Registered analysis

The primary comparison is the five-seed paired change from 6,144 to 131,072
uniform cells on the square-root-data track. It requires at least a one-point
median teacher-forced gain with four positive seed directions, a nonnegative
median free-running change with three nonnegative directions, at least 2%
responsive state cells and readout-source cells, and median firing between 0.5
and 80 Hz. The fixed-data track determines whether a passing effect is robust
to a fixed experience budget or appears only after relieving data starvation.

Every paired comparison reports its seed deltas, mean, median, range, paired
95% t interval, exact one-sided sign statistic, and exact one-sided sign-flip
permutation statistic. Five seeds do not justify presenting those statistics
as strong general evidence.

Nested groups and the layered sheet have separately named confirmatory
criteria at 131,072 cells. Single-scale groups and the ordinary sheet prevent
crediting hierarchy or layering for a generic grouping/locality effect. All
other topology comparisons, including 4D versus 3D, are descriptive; the
protocol forbids choosing a post-hoc winner.

Capability, dynamics, and resource use are separate report sections. The
dynamics section records firing, spike density, eligibility density,
responsive-cell density, state rank, and sampled correlation. The learning
audit records attempted and applied local updates, nonzero readout synapses,
source/output-cell coverage, afferent and recurrent counts, and the absence of
ridge state. Thus sparse or dormant execution cannot be reported as learned
large-network capability.

## Cost and execution

The registered estimate is 12-20 hours for all 148 runs. Fixed-data runs at
98,304 or 131,072 cells are estimated at 5-9 minutes each; square-root-data
runs at those sizes at 18-35 minutes. The largest model is expected to use
roughly 0.28-0.45 GiB of host model arrays plus temporary diagnostic space.
These are planning estimates, not benchmark results.

Plan without accessing validation metrics:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python benchmarks\large_scale_capability.py
```

The plan prints the required hash and explicit execution command. Validation
must not be launched until the topology, runner, tests, and protocol are
committed.
