# Developmental recurrent-cell growth

This layer tests a narrow, falsifiable version of natural development: does
recruiting fitted human cells over training, followed by local synaptic
overgrowth, selection, and homeostasis, improve recurrent sequence dynamics?
It does not claim to reproduce cortical ontogeny.

## Mechanism

`DevelopmentalGrowthController` begins with a dormant-padded GLIF5 reservoir.
Only the first scheduled cohort receives afferent or recurrent drive. Later
cohorts retain stable cell IDs and are recruited without reallocating the cell,
readout, or recurrent arrays.

Two deterministic birth orders are supported:

- `random` is a seeded permutation and is the shuffled-order ablation.
- `layered_prefix` grows outward from the centre of a two-dimensional cortical
  sheet while cycling through all six depth layers at each surface location.

The recurrent array's fixed fanout is structural capacity, not a dense matrix.
Early cells begin with six live outgoing synapses. Recruitment grows the array
to twelve live edges per active source, includes edges into each new cohort,
and permits two weak edges per old source to turn over. Each live synapse stores
a local score

```text
score_ij += eligibility_i * eligibility_j
exposure_ij += 1
selection_ij = score_ij / exposure_ij
```

after every three-factor learning example. Final maturation keeps six edges by
that score. A matched condition prunes six random edges instead. No validation
value enters a structural update.

The homeostatic condition uses each postsynaptic cell's mean training
eligibility as a rate signal. Incoming recurrent synapses are scaled toward the
stage median, with a preregistered `[0.5, 2.0]` bound. After final pruning, the
remaining incoming synapses conserve the target cell's pre-pruning total
absolute recurrent strength when at least one input survives.

This is genuine training-time structural change. It does not copy a trained
readout and zero weights afterward. Newly recruited cells participate in later
GLIF5 transitions and later three-factor output updates; the recurrent targets,
weights, structural scores/exposures, cell order/mask, homeostatic gains, and
readout synapses are the persistent state.

## Frozen validation design

`benchmarks/development_protocol_v2.json` was registered before its named
validation run. It has no test partition. Three fixed training files supply 384
characters; the previously unused `recurrent_artifacts.py` is the only
validation file and is opened after each model's learning has frozen.

At 384 and 1,536 final active cells, three fixed seeds compare five
conditions with the same bytes, attempted next-token updates, epoch count,
final active count, and final fanout:

1. full size from the first update;
2. staged layered order, activity selection, and homeostasis;
3. staged random birth order;
4. staged random structural pruning;
5. staged growth without homeostasis.

The staged cohorts are 25%, 50%, and 100% of final cells over equal thirds of
the training updates. Metrics include teacher-forced and free-running byte
accuracy, effective eligibility-state rank, sampled inter-cell correlation,
firing rate, stage cell/synapse counts, learned/pruned synapses, wall time, and
memory. Before validation, the primary criterion was fixed independently at
each size: staged layered/activity/homeostatic growth must lose no more than one
percentage point of median teacher-forced accuracy, must not lose median
free-running accuracy, and must strictly beat full-start on at least one of
those endpoints in at least two of three paired seeds. Rank and firing remain
secondary diagnostics. The single-file result is still exploratory and cannot
establish general developmental superiority. A 6,144-cell extension is
deferred until concurrent speed measurements release shared compute.

Run the frozen gate with:

```powershell
$env:PYTHONPATH = "src"
python benchmarks\development_gate.py
```
