# An honest path to a 100-billion-cell address space

**Date:** 2026-08-21
**Status:** Architecture research and engineering targets, not a simulation
result

## Decision

Treat 100 billion cells as an **addressable distributed population**, and
always report four smaller-or-equal counts:

1. **addressable** cells described by the population manifest;
2. **materialized** cells with mutable state allocated on some shard;
3. **updated** cells whose fitted equations were integrated in the measured
interval; and
4. **spiking** cells.

No capability or speed result may use the addressable count in place of the
materialized or updated count. A dormant or procedural cell is useful only as
future capacity. It is not evidence that a larger active network learned more.

The long-term runtime should combine:

- 64-bit global cell identifiers and shard-local ownership;
- counter-based procedural generation of immutable cell traits and base
  connectivity;
- sparse overlays for learned synapses and structural changes;
- exact implicit rest state for cells that have never received an event;
- tile-local active frontiers with no population-wide scan;
- deterministic event buckets and target-routed shard communication; and
- compact homogeneous parameter blocks plus explicit exceptions.

This direction follows established large-network simulator patterns, but the
specific combination with the project's fitted human GLIF5 equations and
cell-only language learning remains an engineering hypothesis.

## Why literal materialization is not the first milestone

The current 1,048,576-cell CPU artifact reports about 2,237.04 MiB of
persistent arrays, or roughly 2,237 bytes per cell. Most of that is the current
dense vocabulary-by-reservoir readout. Removing that matrix with the sparse
readout under development leaves approximately 181 bytes per materialized cell
in the present layout. A literal 100-billion-cell allocation would therefore
still require about 18.1 TB (16.5 TiB), before allocator, routing, checkpoint,
and replication overhead.

The measured full-touch CUDA kernel advances about 2.881 billion fitted cell
steps per second. At 100 biological steps per token, touching all 100 billion
cells would take about 3,471 seconds per token on one current GPU. Reaching
100 tokens/s by replication alone would require about 347,000 devices of the
same effective throughput, ignoring communication. These are extrapolations
from the registered local artifact, not hardware forecasts.

At the same measured throughput, a single GPU can update at most about 288,000
cells per biological step while sustaining 100 tokens/s. That is only
2.88 parts per million of a 100-billion-cell address space. Therefore a
single-workstation 100-billion-cell milestone can only be an explicitly
sparse-activity/addressability milestone. Useful high-activity scale requires
distributed hardware or a materially faster cell solver.

## Patterns worth adopting

### Local ownership, not replicated global metadata

NEST's exascale design work states that structures whose cost is proportional
to total neurons or MPI ranks must be removed, and that each process should
store and receive only information relevant to local nodes
([Jordan et al. 2018](https://doi.org/10.3389/fninf.2018.00002)).
NEST also provides reduced-memory HPC and homogeneous synapse variants, and
documents fixed indegree as preferable for distributed construction
([connection management](https://nest-simulator.readthedocs.io/en/v3.5/synapses/connection_management.html),
[population tutorial](https://nest-simulator.readthedocs.io/en/stable/tutorials/pynest_tutorial/part_2_populations_of_neurons.html)).

Adopt the same invariant: a shard owns mutable state for its local target cells
and the learned synapses that terminate there. Global population descriptions
are small manifests, not arrays with one record per addressable cell.

### Recipes and procedural identity

Arbor separates cheap cell kind queries from potentially expensive cell
descriptions, and asks a recipe for the incoming connections of a particular
global cell identifier
([Arbor recipe interface](https://docs.arbor-sim.org/en/latest/cpp/recipe.html)).
This is a useful interface model for a population too large to instantiate
eagerly.

Random123 describes counter-based random generators as stateless functions of
a counter and key
([official source](https://github.com/DEShawResearch/random123);
[paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)).
CUDA also exposes Philox as a counter-based generator
([NVIDIA documentation](https://docs.nvidia.com/cuda/curanddx/api/description_ops.html)).
Use a specified integer Philox/Threefry mapping keyed by model seed, cell ID,
edge slot, developmental epoch, and purpose tag. A cell or base edge can then
be reconstructed on any shard without storing RNG streams or depending on
execution order.

The procedural function must be frozen and tested against materialized
topologies at small sizes. It must generate fixed degree, valid targets, no
self edge when forbidden, no duplicate target per row, and identical CPU/GPU
results. Plasticity does not rewrite this function: learned additions,
deletions, and weights live in a sparse overlay.

### Event exchange and deterministic epochs

Arbor communicates spikes as discrete timestamped events, sorts local spikes,
exchanges them at epoch boundaries, and builds target event queues
([communication design](https://docs.arbor-sim.org/en/latest/dev/communication.html)).
SpiNNaker likewise executes spiking networks through event-oriented,
distributed many-core software
([official software documentation](https://spinnakermanchester.github.io/)).
The approach has simulated a roughly 77,000-neuron, 0.3-billion-synapse
cortical microcircuit in real time
([Rhodes et al. 2020](https://arxiv.org/abs/1909.08665)), showing why event
routing matters even far below 100 billion cells.

Use fixed biological-step event buckets. Each outgoing spike expands its small
procedural fanout, partitions events by target shard, and sends only non-empty
batches. Within a target bucket, sort by a canonical integer key such as
(step, target, source, edge slot) before floating-point accumulation. This
preserves a stable arithmetic order across thread counts and shard layouts.
Do not use a global all-gather at the target scale.

### Reordering and homogeneous storage

CoreNEURON exposes GPU execution and recommends cell permutation for GPU
performance
([CoreNEURON documentation](https://bluebrain.github.io/CoreNeuron/doxygen/index.html)).
Arbor groups cells for CPU/GPU load balancing
([GPU tutorial](https://docs.arbor-sim.org/en/latest/tutorial/network_ring_gpu.html)).
The current Human Brain kernel already benefits from structure-of-arrays and a
persistent device launch. Extend this to reorder materialized cells by
(shard, cell type, active tile, incoming-degree class) while retaining stable
global IDs.

Human-fitted parameter sets are shared immutable blocks referenced by a compact
type ID. Per-cell fitted deviations should be generated procedurally or stored
only when learned. This mirrors NEST's homogeneous synapse idea without
claiming that human cells are biologically identical.

## Proposed runtime model

### Population manifest

A manifest describes named regions, six-layer cortical sheets, optional 1D,
2D, 3D, or abstract higher-dimensional functional coordinates, cell-type
fractions, group hierarchy, base fanout, parameter-family IDs, and a 64-bit ID
range. One hundred billion IDs require 37 bits, leaving ample room for typed or
versioned namespaces.

Geometry is a connectivity prior, not a capability claim. Degree and training
budgets must be matched when comparing ring, sheet, layered, volume,
hypercube, nested-group, and mixed-membership recipes.

### Exact implicit state

A cell may be absent from mutable storage only when all of the following are
provably true:

- its membrane, adaptive threshold, after-spike currents, refractory state,
  synaptic current, and eligibility equal their exact rest representation;
- it has no scheduled afferent or recurrent event;
- it owns no learned mutable state that must decay or be checkpointed; and
- no output or diagnostic reads it during the interval.

First delivery of an event materializes its tile and initializes the fitted
state exactly. A tile remains active until every contained value has returned
to the exact removable state. The current exponential recurrences often
approach rather than reach zero, so eviction cannot use an unregistered
tolerance while claiming bit exactness. A future analytic/lazy-decay protocol
may store a last-update step and reconstruct decays on demand, but must first
match the stepwise oracle within a preregistered tolerance and be labeled
approximate if it is not bit exact.

### Hierarchical active frontier

Maintain a hierarchy of:

- shards;
- fixed-size tiles;
- active cells within a tile; and
- due event buckets.

No token may scan all addressable IDs. An empty shard or tile is represented by
one manifest/routing record. Activation and deactivation update explicit dirty
sets. The runtime reports scanned shards, tiles, cells, edges, and events so a
speed artifact reveals hidden population-wide work.

### Plastic overlays

Base topology and shared initial weights are procedural. Learning state is
target-owned and sparse:

- recurrent structural additions/deletions and non-default weights;
- sparse output-cell contact pools, eligibility/apposition traces, and
  turnover clocks;
- cell homeostatic and developmental state; and
- checkpoint event-log segments since the last compact snapshot.

The overlay must be bounded per materialized cell or per output cell. There is
no dense N by N or vocabulary by N learned array.

### Sharded execution

Each shard uses the existing exact Rust or CUDA transition kernel on compacted
local tiles. Shards advance a common fixed-step epoch, exchange spike events,
and only then enter the next epoch. Checkpoints store the manifest and
procedural seed once, plus shard-owned mutable overlays and cell state. Recovery
must not depend on the previous shard count.

MPI-style target ownership is the portable baseline. GPU peer/NCCL transport
can be added behind the same event-batch seam only after profiling. The
communication contract is more important than a particular transport library.

## Scale and capability gates

The project should use a ladder rather than a single 100-billion headline:

| Gate | What must be demonstrated |
|---|---|
| A: procedural exactness | Small procedural topology and fitted transitions match the fully materialized oracle exactly across CPU and GPU. |
| B: 100B addressability | Random cell/edge queries over a 100B manifest use bounded memory and time; no O(N) construction or token scan occurs. Report zero capability credit. |
| C: sparse 100B runtime | A 100B address space with a declared materialized/updated cohort runs end to end. Report all four counts and activity density. |
| D: active scale law | Increasing **updated and responsive** cells improves family-disjoint held-out prediction under matched data/compute controls and paired seeds. |
| E: distributed materialization | Multiple shards materialize and update a population larger than one device, preserve the deterministic oracle, and show useful scaling efficiency. |
| F: 100B materialized | Only a run allocating and maintaining fitted state for 100B cells may use the unqualified phrase “100-billion-cell simulation.” |

Gate C should initially target at least the already-proven 200,000 full-touch
cells inside a 100B address space at more than 100 tokens/s. That validates the
runtime shape, not the user's hypothesis that billions of active cells help.
Gate D is the scientific test of that hypothesis.

## Testing whether more cells and organization help

After the sparse readout is frozen, preregister a second active-scale protocol:

- paired active sizes, with dormant allocation forbidden as the independent
  variable;
- fixed-data, compute-matched, and sublinearly data-scaled tracks;
- fixed responsive-cell, firing-rate, and readout-coverage floors;
- uniform, nested-group, mixed-membership, layered 2D, 3D, and abstract 4D
  conditions with matched fanout;
- developmental birth-order, local growth/pruning, and no-development controls;
- teacher-forced language, free-running language, and family-disjoint coding
  development metrics; and
- a frozen fresh evaluation cohort opened once after architecture selection.

Fit paired capability changes against log active-cell count and report all
seeds, confidence intervals, failures, resource use, and activity. A positive
scale claim requires a consistent held-out trend, not the best topology/seed
selected after inspection. The existing 148-run protocol remains an honest
pre-sparse-readout experiment and must not be retuned.

## Ordered implementation

1. Finish the bounded sparse output-cell readout and exact checkpoints.
2. Add a 64-bit procedural topology recipe with a small materialized
   equivalence oracle.
3. Replace the population-wide frontier scan with tile/event activation while
   retaining exact rest-state semantics.
4. Add addressable/materialized/updated/spiking counters to every benchmark
   artifact.
5. Add a two-shard deterministic CPU prototype, then GPU event batching.
6. Run active scale-law development experiments; improve activity recruitment
   and architecture when larger active networks fail.
7. Freeze one generic cell checkpoint for the hidden simple-coding gate.
8. Attempt a larger SWE benchmark and Gemma comparison only after the generic
   checkpoint solves the simple family-disjoint tasks.

The immediate project priority remains simple coding competence. Procedural
100-billion addressability is useful infrastructure, but it must not distract
from the current result: the cell model does not yet generalize to held-out
software-engineering tasks.
