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

Structural mutations are published at the end of each recruitment,
homeostasis, and pruning boundary through
`RecurrentCellCircuit.synchronize_structure_from_host()`. CPU execution already
reads the live host arrays. CUDA rebuilds its target-indexed reverse CSR and
uploads the fixed-size afferent, recurrent-target, and recurrent-weight arrays;
sequence reset remains a separate mutable cell-state synchronization.

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
establish general developmental superiority. A 6,144-cell post-primary
extension was declared in the same protocol and ran unchanged only after the
1,536-cell runtime and memory measurements established that it was safe.

## Frozen results

The primary criterion failed at both registered sizes. At 384 cells, staged
growth tied full-start teacher forcing but lost 1.5625 percentage points of
median free-running accuracy and recorded no strict paired-seed win. At 1,536
cells, it stayed inside the one-point teacher tolerance and tied free running,
but had only one teacher win and no free-running win. The unchanged 6,144-cell
post-primary diagnostic also failed: staged teacher forcing was 2.34375 points
below full-start and had no paired win.

| Final active cells | Condition | Teacher forced | Free running | Effective rank | Mean firing |
| ---: | --- | ---: | ---: | ---: | ---: |
| 384 | Full from start | 4.69% | 6.25% | 2.031 | 41.57 Hz |
| 384 | Staged layered/activity/homeostatic | 4.69% | 4.69% | 2.425 | 41.84 Hz |
| 384 | Staged random order | 4.69% | 4.69% | 2.208 | 44.84 Hz |
| 384 | Staged random pruning | 4.69% | 4.69% | 1.256 | 80.85 Hz |
| 384 | Staged without homeostasis | 5.47% | 4.69% | 2.023 | 44.20 Hz |
| 1,536 | Full from start | 5.47% | 4.69% | 3.576 | 15.13 Hz |
| 1,536 | Staged layered/activity/homeostatic | 4.69% | 4.69% | 4.055 | 14.98 Hz |
| 1,536 | Staged random order | 6.25% | 4.69% | 4.244 | 14.72 Hz |
| 1,536 | Staged random pruning | 4.69% | 4.69% | 1.190 | 65.72 Hz |
| 1,536 | Staged without homeostasis | 4.69% | 4.69% | 2.626 | 16.63 Hz |
| 6,144 | Full from start | 8.59% | 4.69% | 5.327 | 3.45 Hz |
| 6,144 | Staged layered/activity/homeostatic | 6.25% | 4.69% | 5.414 | 3.60 Hz |
| 6,144 | Staged random order | 7.03% | 4.69% | 5.291 | 3.81 Hz |
| 6,144 | Staged random pruning | 5.47% | 4.69% | 1.133 | 56.75 Hz |
| 6,144 | Staged without homeostasis | 6.25% | 4.69% | 4.470 | 3.80 Hz |

Values are three-seed medians. The 6,144-cell full-start teacher score rising
to 8.59% shows that more cells were not intrinsically harmful. For the staged
condition, effective rank rose 2.425 → 4.055 → 5.414 and sampled absolute
correlation fell about 0.458 → 0.380 → 0.327 as cells increased. Predictive
accuracy did not follow those improved dynamics, so the current limitation is
the learning/development schedule rather than evidence against scale.

Activity-based structural selection was consistently meaningful: random
pruning collapsed rank to about 1.1-1.3 and drove firing to 57-81 Hz, while the
activity-selected condition retained rank 2.4-5.4 and much lower firing.
Homeostasis also raised staged rank over its no-homeostasis ablation at every
size. Neither effect delivered the preregistered prediction win, and layered
birth order did not consistently beat random order.

All conditions received 381 attempted next-token updates and matured to six
live recurrent edges per active cell. At 6,144 cells this is 36,864 surviving
and 36,864 final-pruned recurrent synapses. Staged growth retained about
199,000 nonzero readout synapses versus about 278,000 for full-start because
late-born cells saw fewer updates; this is a plausible mechanism for the
predictive loss, not a post-result parameter change. The staged 6,144-cell run
took a 3.57 s median to train and used 14.77 MiB of persistent arrays.

Run the frozen gate with:

```powershell
$env:PYTHONPATH = "src"
python benchmarks\development_gate.py
python benchmarks\development_gate.py --deferred-scale
```
