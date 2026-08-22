# A falsifiable path to a 100-billion-neuron address space

**Date:** 2026-08-22

**Repository baseline:** Human Brain `4df4966`

**Status:** primary-source architecture research; not a simulation result

## Decision

The attainable first milestone is a **100-billion-neuron address space**, not a
100-billion-neuron active simulation. The runtime may configure and resolve
`100,000,000,000` deterministic identities while keeping only a declared subset
resident and advancing only the subset that has state or due events. Every result
must publish the five scale levels below. A procedural identity, an implicit
resting cell, a hardware capacity number, and a resident-but-unupdated cell are
not interchangeable with an updated neuron.

The shortest credible route combines patterns already demonstrated elsewhere:

- NEST-style rank-local ownership and communication that does not replicate
  global per-neuron metadata;
- Arbor-style lazy, global-ID-indexed recipes, while recognizing that Arbor
  materializes the cells assigned to a rank before simulation;
- GeNN-style runtime regeneration of **static** connectivity, with explicit
  sparse storage for every learned exception;
- CoreNEURON-style structure-of-arrays layouts and local reordering;
- SpiNNaker/Loihi-style source-addressed sparse event traffic; and
- an exact active frontier only for cells provably at canonical rest, with no
  scan over the 100-billion address range.

This architecture is falsifiable through the gates at the end of this report.
Nothing here establishes that more actively simulated cells improve language or
software capability.

## The scale contract

### Required neuron counts

All counts are over a named measurement interval `I`; the interval and biological
step size must be recorded.

| Term | Exact meaning | Required report fields |
| --- | --- | --- |
| **Configured** `N_cfg` | Identities declared by the run manifest. Configuration alone may be a number and a recipe. | total and counts by region/type |
| **Addressable** `N_addr` | Configured IDs for which the implementation can validate the ID and deterministically return its type, coordinates, parameters, and base-edge queries. No mutable neuron state is implied. | valid ID ranges, ID width, query count/latency, invalid-ID behavior |
| **Resident** `N_res` | Identities with mutable neuron state physically allocated in host, device, or neuromorphic memory. Report both peak simultaneous residents and distinct residents during `I`. | `resident_peak`, `resident_distinct`, state bytes, allocation tier |
| **Updated** `N_upd` | Distinct neurons whose dynamical state was actually evaluated or analytically caught up during `I`. Also report the real number of cell-step evaluations `U`, because one cell updated 100 times is more work than one cell updated once. | `updated_distinct`, `cell_step_evaluations`; analytic jumps separately |
| **Spiking** `N_spk` | Distinct neurons that emitted at least one spike during `I`. Also report total spike events `S`. | `spiking_distinct`, `spike_events`, firing-rate distribution |

For a non-churning interval, `N_spk <= N_upd <= N_res <= N_addr <= N_cfg`.
With eviction and rematerialization, use `resident_distinct` for that relation and
publish `resident_peak` separately. “Active” is too ambiguous for a scale claim;
if retained in code, it must be qualified as addressable, resident, updated, or
spiking.

An analytic jump over `k` silent fixed steps is one catch-up operation, not `k`
measured cell-step evaluations. It may additionally report `k` **step
equivalents**, clearly labeled. This prevents a lazy recurrence from being sold
as work the machine did not perform.

### Required connection and event counts

Neuron counts alone can hide the dominant cost. Every run must also publish:

- configured base edges `E_cfg` and their rule/version;
- physically resident edge records `E_res`;
- mutable or learned edge records `E_mut`;
- base edges generated or examined during `I`, `E_gen`;
- delivered synaptic events `D`; and
- bytes sent per shard pair and the maximum event-queue occupancy.

A model may truthfully have 600 billion configured degree-six edges and zero
resident base-edge records. It may not describe those 600 billion edges as
“processed” unless `E_gen` or `D` shows that work.

## What established systems actually demonstrate

### NEST: distribute resident state and eliminate global-per-rank costs

NEST is the strongest precedent here for truly large **resident** networks. Its
exascale redesign stores synapse objects with process-local target neurons,
adds sender-side routing information, and uses directed `MPI_Alltoall` spike
communication. The authors identify any per-process structure proportional to
the total neurons or MPI ranks as a scalability failure and require each process
to store and receive only locally relevant information. Their maximum-filling
run reports 1.51 billion neurons and 16.8 trillion synapses on the K computer;
these were distributed neuron and synapse objects, not dormant IDs
([Jordan et al., 2018](https://doi.org/10.3389/fninf.2018.00002)).

Current NEST documentation exposes the distinction directly: its node-ID field
can address up to `2^61 - 2` nodes, vastly more than any reported resident run
([network and simulation limits](https://nest-simulator.readthedocs.io/en/main/nest_behavior/simulation_limits.html)).
NEST also provides fixed-indegree/outdegree construction rules and recommends
fixed indegree for distributed efficiency
([population tutorial](https://nest-simulator.readthedocs.io/en/stable/tutorials/pynest_tutorial/part_2_populations_of_neurons.html)).
Its `_hpc` synapses use thread-local target indices and `_hom` variants share
parameters to reduce per-synapse storage, but neither makes arbitrary mutable
synapses free
([synapse specification](https://nest-simulator.readthedocs.io/en/stable/synapses/synapse_specification.html)).

**Adopt:** 64-bit global IDs, compact shard-local indices, target ownership,
shared parameter blocks, and directed relevant-spike exchange. **Do not infer:**
that a wide ID field is a simulated population.

### Arbor and CoreNEURON: lazy descriptions, then compact local execution

Arbor's recipe is queried by global cell ID and need not hold a complete model
description in memory. The documentation explicitly recommends lazy,
reproducible per-GID descriptions. During model building, however, Arbor
partitions the cells, groups each rank's local cells for CPU/GPU execution, and
builds the local communication network. A lazy recipe therefore reduces model
description and construction memory; it is not evidence that nonresident recipe
IDs are dynamically simulated
([Arbor recipes](https://docs.arbor-sim.org/en/latest/concepts/recipe.html)).

Arbor exchanges discrete spikes at epoch boundaries, but its current documented
implementation gathers the global spike list onto every rank with
`MPI_Allgatherv`, after which each rank selects locally subscribed events
([Arbor communication](https://docs.arbor-sim.org/en/latest/dev/communication.html)).
The recipe is a useful interface pattern for this project; the global all-gather
is not the desired 100-billion transport pattern.

CoreNEURON demonstrates the next layer after partitioning: remove interactive
object overhead, group mechanism properties in structure-of-arrays form,
reorder nodes for locality, and run time-step integration and event delivery on
the GPU. Its paper reports 4-7x less memory and 2-7x less execution time than
NEURON for the evaluated models while preserving binary compatibility. It is a
compact resident compute engine, not an implicit-population mechanism
([Kumbhar et al., 2019](https://doi.org/10.3389/fninf.2019.00063)).

**Adopt:** per-GID recipe queries, construction local to the owner, SoA state,
and local cell reordering that preserves stable global IDs. **Change:** use
target-routed non-empty event batches rather than a global spike all-gather at
the target scale.

### GeNN and Brian2GeNN: procedural edges are real, but constrained

GeNN offers dense, bitmask, sparse CSR, Toeplitz, and procedural synapse matrix
types. Its current documentation says procedural sparse connectivity is generated
on the fly and requires synaptic variables to be constant or generated on the
fly; individual per-synapse state is not resident in that representation
([PyGeNN network construction](https://genn-team.github.io/genn/documentation/5/building_networks.html)).
That is the closest established pattern to the proposed base graph.

Knight and Nowotny used procedural connectivity and kernel merging to simulate a
4.13-million-neuron macaque visual-cortex model with 24.2 billion logical
synapses on one Titan RTX. The paper is explicit that connectivity and weights
are regenerated instead of read from stored synapse records
([Knight and Nowotny, 2021](https://doi.org/10.1038/s43588-020-00022-7)).
The honest interpretation is: millions of neuron states were simulated, while
billions of static synapse identities were procedural. It is not 24.2 billion
resident or plastic synapses.

Brian2GeNN shows the ordinary alternative. Its evaluated models used GeNN's
sparse matrix representation, and the paper reports that synapse matrices were
the GPU memory limit, approximately 8 bytes per synapse in single precision and
12 in double precision for those implementations
([Stimberg et al., 2020](https://doi.org/10.1038/s41598-019-54957-7)).
Brian2GeNN also uses one fixed simulation clock and supports a narrower model
surface than Brian
([documented exclusions](https://brian2genn.readthedocs.io/en/latest/introduction/exclusions.html)).

**Adopt:** generate immutable base edges and shared/default weights on spike,
using counter-based random numbers. **Constraint:** any learned weight, deletion,
addition, delay, or eligibility that differs by synapse must occupy a sparse
overlay; procedural generation cannot erase its information cost.

### SpiNNaker and Loihi: event traffic is sparse; state capacity is physical

SpiNNaker places neuron state on ARM cores, keeps synaptic rows in chip SDRAM,
and sends source-addressed multicast packets when neurons spike. The programming
model is event-driven, but it also uses periodic timer events for neuron-state
updates; event-driven communication does not imply that all neuron integration
is absent. Real-time capacity is limited by neurons × input synapses × input
firing rate, not neuron count alone
([Furber et al., 2014](https://doi.org/10.1109/JPROC.2014.2304638)).
A primary benchmark ran a 77,000-neuron, 0.3-billion-synapse cortical
microcircuit in biological real time on SpiNNaker, with every 0.1-ms model step
executed in 0.1 ms
([Rhodes et al., 2020](https://doi.org/10.1098/rsta.2019.0160)).

Intel's Loihi 2 likewise combines local neuron/synapse memory with sparse
asynchronous messages. Its technology brief lists at most one million neurons
and 120 million synapses per chip, 192 KB flexible memory per neuron core, and
connectivity features including synapses generated from a seed and fanout-list
compression
([Intel Loihi 2 brief](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)).
Intel says Hala Point **supports up to** 1.15 billion neurons and 128 billion
synapses across 1,152 Loihi 2 chips and 140,544 neuromorphic cores. That is a
physical system capacity statement, not a published workload showing 1.15
billion updated or spiking neurons
([Intel Hala Point announcement](https://www.intc.com/news-events/press-releases/detail/1691/intel-builds-worlds-largest-neuromorphic-system-to)).

**Adopt:** source-addressed multicast semantics, local fanout expansion, and
explicit traffic limits. **Do not infer:** installed neuron capacity equals
resident model occupancy, or resident occupancy equals spiking activity.

## What the current Human Brain measurements mean

The repository already contains three different kinds of “large” result. They
must not be merged into one headline.

| Path | Configured / resident | Actually updated | Measured result | Honest interpretation |
| --- | ---: | ---: | --- | --- |
| CPU dense, 1,048,576 | 1,048,576 / 1,048,576 | every cell on each of 100 steps | 3.59 token/s; 376.7 M cell-steps/s | genuine full-touch million-cell CPU execution |
| CPU exact frontier, natural trajectory | 1,048,576 / 1,048,576 | maximum 544 cells in a token (0.052%) | 193.36 token/s | exact sparse update, but it scans all resident state once per token |
| CPU exact frontier, controlled 1% | 1,048,576 / 1,048,576 | 10,518 cells | 53.25 token/s | one-percent density does **not** pass 100 token/s at 1M |
| CPU exact frontier, controlled 100% | 1,048,576 / 1,048,576 | 1,048,576 cells | 1.41 token/s (dense: 4.13) | frontier bookkeeping loses when activity is dense |
| CUDA full-touch | 200,000 / 200,000 | every cell and six reverse-CSR edges on each of 100 steps | 144.03 end-to-end token/s; 2.8806 B cell-steps/s | genuine 200k updated-cell result on the RTX 4070 SUPER |
| CUDA + sparse output readout | 200,000 / 200,000 | every reservoir cell each step | 158.18 inference and 150.00 training token/s | removes most host readout/transfer cost; reservoir execution remains dense |
| Topology construction v3 | 1,000,000 IDs | no neuron dynamics | six million targets generated into a 22.89-MiB `int32` array in 15.74-48.43 s | near-linear **materialized construction**, not runtime procedural connectivity |

The CPU numbers and density equivalence checks come from
[`native_scaling_v2.json`](../../artifacts/native_scaling_v2.json) and
[`native_activity_density_v1.json`](../../artifacts/native_activity_density_v1.json).
The current frontier implementation clears and fills a population-sized mask and
loops over every resident cell before integrating its compact active list
([`lib.rs`](../../src/humanbrain/_native_src/src/lib.rs)). It is exact relative
to the dense oracle, but its scan is `O(N_resident)` per token. It is not yet a
100-billion-address-space frontier.

The CUDA numbers come from [`cuda_scaling_v1.json`](../../artifacts/cuda_scaling_v1.json)
and the optimized [`sparse_cuda_engineering_v1_optimized.json`](../../artifacts/sparse_cuda_engineering_v1_optimized.json).
The cooperative CUDA kernel loops over all cells for neuron updates and again
over all cells plus their incoming CSR rows for recurrent current on every step
([`cuda.cu`](../../src/humanbrain/_native_src/src/cuda.cu)). Its sparse spike
count—1.125 mean spike events per token in the registered 200k run—does not make
the compute sparse. The registered one-million CUDA case hit its 270-second
construction bound and records no throughput result; it must not be claimed as
a one-million CUDA simulation.

The topology artifact covers single-scale groups, nested groups, mixed
membership, a 1D ring, a 2D sheet, a layered sheet, and a 3D volume
([`structured_topology_construction_v3.json`](../../artifacts/structured_topology_construction_v3.json)).
[`topology.py`](../../src/humanbrain/topology.py) also implements an abstract 4D
hypercube candidate rule, but the one-million-cell v3 artifact does not include
that condition. All of these functions currently allocate the full
`(cell_count, fanout)` target array. “Procedural topology” presently describes
how the array is constructed, not how execution stores it.

Finally, the 1M CPU artifact reports 2,237.04 MiB of persistent arrays. With 257
output cells, the dense `float64` output-by-reservoir matrix alone accounts for
2,056 bytes per reservoir cell. Subtracting it leaves 181.04 measured bytes per
resident cell in that layout. The later 200k sparse-readout CUDA artifact
measures 185.50 host bytes and 138.16 device bytes per cell. These measurements
include different execution structures and should be reported separately, not
combined into a synthetic total.

## Arithmetic at 100 billion

All decimal units below use `1 TB = 10^12 bytes`; binary equivalents are shown
where useful. These are arithmetic consequences of the specified representation,
not hardware forecasts.

### Identity and flags

`100,000,000,000` identities require 37 bits. A 64-bit global ID is therefore
sufficient, but a 32-bit global target index is not.

| One item per configured ID | Total |
| --- | ---: |
| one bit | 12.5 GB = 11.64 GiB |
| one byte | 100 GB = 93.13 GiB |
| one `uint32` | 400 GB = 372.53 GiB |
| one `uint64` | 800 GB = 745.06 GiB |

There is no per-ID memory lower bound for a purely mathematical address range:
a manifest can describe a contiguous range in constant space. The table is the
price paid the moment the design adds a global membership bitset, status byte,
owner array, or ID array. Shard ownership and topology must therefore be
computed from ID/range metadata, not stored once per addressable neuron.

### Resident state and edges

The current GLIF5 state uses one `float64` voltage, two `float64` after-spike
currents, two `float64` threshold components, and one `int32` refractory counter:
44 bytes per resident cell. Recurrent synaptic current and eligibility add 16,
for **60 bytes of mutable dynamical state before topology, workspaces, readout,
allocator overhead, or replication**
([`cells.py`](../../src/humanbrain/cells.py),
[`recurrent.py`](../../src/humanbrain/recurrent.py)). Literal state for 100
billion residents is therefore at least 6.0 TB (5.46 TiB) in the current exact
dtypes.

At fanout six:

- current `int32` targets alone cost 24 bytes/cell, or 2.4 TB, but cannot name a
  100-billion global target without sharding;
- six global `uint64` targets cost 48 bytes/cell, or 4.8 TB;
- six `float64` weights cost another 48 bytes/cell, or 4.8 TB; and
- current local-`int32` target plus `float64` weight records combined with the
  60-byte state floor total 132 bytes/cell, or 13.2 TB, before reverse CSR,
  queues, masks, readout, and checkpoints.

The measured extrapolations are worse: 181.04 bytes/cell after subtracting the
dense 257-output readout implies 18.10 TB; the sparse-readout host layout implies
18.55 TB; and its device layout implies 13.82 TB. Preserving the current dense
readout would imply 223.70 TB. These extrapolations are useful lower-order checks,
not promises of linear allocation at that scale.

There is also an information lower bound on **arbitrary** learned connectivity.
For six unordered distinct targets among 100 billion possible targets,
`log2(C(100,000,000,000 - 1, 6))` is approximately 209.8 bits per source, or
2.62 TB across 100 billion sources, before weights, delays, and indexing. A seed
compresses a restricted procedural graph family; it cannot compress arbitrary
learned rewiring without storing the deviations.

### Compute and traffic

The measured 200k CUDA run performs 2.8806 billion fitted cell steps/s. At the
current 100 steps per 5-ms token, a full touch of 100 billion cells would take
about **3,472 seconds per token** (0.000288 token/s) on one equally effective
GPU. One biological second at the 0.05-ms step would take about 694,302 wall
seconds—roughly 8.0 days. Sustaining 100 token/s by ideal replication would take
about 347,151 such GPUs before any communication cost. Conversely, that measured
rate can fully update at most about 288,000 cells at 100 steps/token and 100
token/s.

Sparse firing changes work only after the kernel changes. If 100 billion
updated neurons fired at 1 Hz with degree six, the system would still face 100
billion spike events and 600 billion synaptic deliveries per biological second;
at 10 Hz those become one trillion and six trillion. Event-driven execution
replaces work proportional to all configured edges with work proportional to
actual state updates and deliveries. It does not make a high-activity network
free.

## What one RTX 4070 SUPER cannot do

The registered device reports 12,282 MiB total VRAM. A single one-bit flag for
each of 100 billion configured IDs is 11,920.93 MiB—97.1% of total VRAM before
neuron state, topology, queues, CUDA context, display use, or outputs. At the
captured sparse-CUDA baseline only about 9,967 MiB was free. Therefore this GPU
cannot:

- hold one global activity bit per addressable ID alongside a simulator;
- hold 100 billion current GLIF5 states (state alone is 6 TB);
- hold six explicit targets and weights per 100 billion residents;
- fully update 100 billion cells at interactive speed—the measured-throughput
  extrapolation is about 58 minutes per token;
- provide a one-million CUDA throughput result from the current artifact, which
  explicitly stopped during construction; or
- turn unmaterialized IDs or regenerated edges into evidence of 100 billion
  learned, responsive, updated, or spiking cells.

It **can** host a bounded-memory manifest for 100 billion identities, verify
random high-ID cell/edge queries, and simulate a much smaller declared resident
and updated cohort within that address space. That is a useful infrastructure
result if labeled with all five counts.

## Proposed representation and execution

### 64-bit manifest and shard-local ownership

Use contiguous 64-bit ID ranges for named populations. Each manifest entry
contains a topology recipe version, parameter-family ID, dimensions, group
radices, fanout quotas, and seed. Region and shard lookup is a range operation;
no owner array is permitted. A shard owns mutable target-cell state and mutable
synapses terminating on those cells, following NEST's local-target principle.
Within a shard, use 32-bit offsets only when the shard-local count and edge index
are proven below their limits.

The manifest is the configured model. A successful random query makes an ID
addressable. Allocation makes it resident. Neither operation increments the
updated or spiking counters.

### One counter-based base-edge function

Freeze one versioned integer function:

```text
base_edge(model_seed, topology_version, source_gid, edge_slot, epoch, purpose)
    -> (target_gid, shared_weight_id, delay_bucket)
```

Use a counter-based generator such as Philox/Threefry so the result is a pure
function of the tuple, independent of thread order, shard count, and checkpoint
history. Counter-based generators were designed precisely as keyed functions of
counters rather than mutable RNG streams
([Salmon et al., 2011](https://doi.org/10.1145/2063384.2063405)).
Sampling without replacement must have a specified bounded algorithm; rejection
behavior, self-edge policy, and duplicate policy are part of the topology
version.

The same function supports each present topology without coordinate arrays:

- **single-scale groups:** derive a microgroup range from mixed-radix ID digits;
  allocate fixed local and global edge-slot quotas;
- **nested groups:** derive micro/meso/region ancestors from the ID, with explicit
  quotas at each level;
- **mixed membership:** derive two or more memberships from independent keyed
  permutations and reserve a declared shared-membership quota;
- **1D/2D/3D:** obtain coordinates by integer division/modulo, generate bounded
  local offsets for local slots, and use keyed global candidates for long-range
  slots;
- **layered 2D:** use `(x, y, layer)` with separate in-layer, adjacent-layer,
  and long-range quotas; and
- **4D:** use the same integer lattice rule as an explicitly **abstract
  functional topology**, never as an anatomical claim.

Small and partially filled boundary grids need deterministic fallback rules.
Every topology comparison must match fanout, weight/delay distributions, input,
training data, and update budget. Geometry is a prior, not a scale result.

At runtime, a source spike expands its base fanout and routes the resulting
events. Do not build the current full forward target array or reverse CSR for
the 100-billion path. Optional bounded caches may retain hot rows, but cache
contents are resident acceleration data and must be reported.

### Sparse mutable overlays

Procedural state is valid only for immutable defaults. Store target-owned sparse
records for:

- non-default learned weights or delays;
- structural additions and deletion tombstones;
- per-synapse eligibility/traces;
- per-cell learned/homeostatic state; and
- checkpoint deltas.

Overlay bytes and records per resident/updated cell are hard budget metrics. A
model that learns a distinct value for every configured edge has abandoned the
procedural-memory advantage and must say so.

### Event-driven active frontier with exact limits

A neuron may remain nonresident only while its exact state is the canonical
rest state, it has no due event, no mutable overlay that requires decay, and no
probe or output reads it. The first incoming event materializes its tile and
state. The runtime maintains sets of non-empty shards, resident tiles, active
cells, and due event buckets; no token may iterate across `N_cfg` or `N_addr`.

The current frontier's exact-zero rule is safe for never-activated cells.
Eviction is harder: the GLIF5 voltage, two after-spike currents, threshold
components, synaptic current, and eligibility use exponential recurrences that
can remain nonzero for many steps. A tolerance-based eviction changes the model.
An analytic lazy-decay catch-up changes floating-point operation order and must
be labeled tolerance-equivalent unless it passes a bitwise oracle. Until one of
those gates passes, keep decaying cells resident and updated; do not silently
drop them.

Events are bucketed by biological step. Generate base fanout only for spiking
sources, partition by target shard, and send only non-empty batches. Within a
target bucket, sort by a frozen integer key such as
`(step, target_gid, source_gid, edge_slot)` before accumulation. This replaces
the current reverse-CSR scan while preserving a deterministic summation order.

### Distributed epochs and checkpoints

Start with two CPU shards behind the same event-batch contract, then add GPU
shards. Each shard advances only to the next safe epoch boundary, exchanges
target-routed spike batches, and resumes after all earlier events are known.
Checkpoint the manifest and base seed once, plus shard-owned neuron state,
mutable overlays, pending events, and counter totals. Restore onto a different
shard count and compare against the same canonical event/state hash.

No all-gather, all-ID bitset, or per-address owner table is allowed on this path.
Transport choice is secondary to that contract.

## Staged engineering and capability gates

Each gate authorizes only the claim in its final column. Thresholds may be
preregistered more tightly, but not weakened after results are seen.

| Gate | Falsifiable acceptance condition | Claim authorized |
| --- | --- | --- |
| 0. Scale instrumentation | Every benchmark emits the five neuron counts, `U`, `S`, all connection/event counts, host/device bytes, and the measured interval; tests reject impossible relations. | Existing results can be relabeled honestly. |
| 1. 64-bit topology oracle | For every topology, all rows of small graphs match a materialized reference; sampled IDs on both sides of `2^32` match on CPU/GPU; degree, quotas, no-self/no-duplicate policy, and seed reproducibility pass. | Procedural topology correctness, not 100B scale. |
| 2. 100B addressability | A `10^11` manifest starts without a loop/allocation proportional to `N_addr`; random valid/invalid cell and edge queries pass. Across `10^6`, `10^9`, and `10^11` configured IDs, peak memory is bounded by manifest/cache settings and fixed-query runtime has no material slope with `N_cfg`. | “100-billion addressable neurons.” |
| 3. Runtime-procedural 200k parity | With 200k resident and fully updated cells, generated-on-spike base edges reproduce the explicit graph's registered state/spikes within the frozen exactness rule and retain at least 100 end-to-end token/s on the same RTX 4070 SUPER. | No-regression runtime seam at the already measured active scale. |
| 4. No global scan | Hold resident, updated, spikes, and inputs fixed while increasing addressable IDs from `10^6` to `10^11`; token work counters scan zero dormant IDs and time/memory remain within a preregistered noise band. | Sparse execution cost is independent of dormant address space. |
| 5. Density curve | Measure 1%, 5%, 20%, and 100% updated fractions at each resident size, including construction, queue pressure, generated edges, deliveries, and exactness. A size passes the speed gate only at densities that actually exceed 100 token/s. | Density-qualified speed, never an unqualified size claim. |
| 6. Two-shard determinism | One- and two-shard small oracles produce the same canonical states/spikes; restore with a changed shard count also matches. Network bytes scale with routed events, not total configured IDs or ranks. | Correct distributed execution seam. |
| 7. Distributed resident scaling | Increasing hardware increases `resident_peak` and `updated_distinct`; memory is local, no rank holds global per-ID state, and measured weak/strong scaling efficiency plus failures are published. | Hardware- and density-qualified distributed scale. |
| 8. Capability scaling | Paired runs vary **updated/responsive** cells, not dormant IDs; data, fanout, topology budgets, firing-rate floors, and compute controls are fixed or explicitly stratified. Held-out capability improves with confidence intervals across preregistered seeds. | Evidence that active scale helps the task. |
| 9. 100B resident/updated | Mutable fitted state is allocated for 100 billion neurons and all 100 billion are advanced in the declared interval, with memory, wall time, spikes, events, and accuracy reported. | Only this gate permits the unqualified phrase “100-billion-neuron simulation.” |

Gate 2 is the workstation-sized near-term target. Gate 3 is the critical bridge
to the current measured product: a 100-billion address space with 200,000 fully
updated cells must not be slower or less exact merely because its IDs are wider.
Gate 4 proves that the headline address range is not hiding an `O(N_cfg)` scan.
Gate 8, not Gate 2, tests whether scale improves capability.

### Procedural v2 Gate 1/2 engineering protocol

The staged CPU scalar implementation lives behind one small seam:
`TopologyManifest` declares a contiguous uint64 GID range and
`ProceduralTopology` exposes `describe_cell(gid)` and `base_edges(gid)`.
Constructing a manifest or oracle does not iterate over the configured range and
does not allocate a GID array, owner array, topology array, or membership bitset.
The private topology-spec table selects all ten established random, grouped,
mixed-membership, 1D, 2D, layered, 3D, and abstract-4D policies without a
topology-name switch in either public query.

This is an explicit protocol break from the prior NumPy global-RNG arrays. The
only executable version is `random123-philox2x64-10-v2`; there is no legacy seed
mode, compatibility fallback, or v1 execution path. The v1 protocol and
[`procedural_address_space_v1.json`](../../artifacts/procedural_address_space_v1.json)
are immutable historical evidence. Their recorded outcome remains **FAIL** and
must not be rerun, rewritten, or reinterpreted. All future procedural
address-space execution is governed only by
[`procedural_address_space_protocol_v2.json`](../../benchmarks/procedural_address_space_protocol_v2.json).

The v2 Philox-2x64-10 constants, numeric word order, counter packing, topology
streams, purpose streams, bounded rejection, and the official Random123
[implementation](https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h)
and [known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)
are pinned in
[`procedural_address_space_protocol_v2.json`](../../benchmarks/procedural_address_space_protocol_v2.json).
The scalar implementation is checked against the same official vectors and a
frozen materialized all-row reference for every topology on a small graph.
The protocol also pins an exact six-file raw-byte implementation bundle,
including the runner, oracle, identity, accounting, and atomic-publication
sources. A changed path, size, file SHA-256, or canonical bundle SHA-256 rejects
execution before any worker launches.

Each descriptor reports exact integer lattice coordinates or group/layer
memberships only where the topology defines them. `outgoing_connectivity_role`
is deterministic Dale-like source-polarity metadata. It is not distinct
inhibitory dynamics or molecular E/I biology: every descriptor identifies the
same pinned Allen human GLIF5 model `523000114`, specimen `508378649`, and
configuration SHA-256
`69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149`.
The 4D coordinate system is labeled abstract rather than anatomical.
For every sampled mixed-membership descriptor, v2 explicitly checks that the
`primary` and `secondary` group IDs differ. Each mixed-topology result reports
the sampled count and pass/fail value; the parent aggregates that claim and
includes it independently in the Gate 2 decision.

For a row of fanout `F`, each slot performs at most 128 counter candidates. If
those candidates are rejected, one row-level fallback cursor advances without
resetting or revisiting a GID. Across the entire row it examines at most `F+1`
fallback GIDs: each rejected fallback GID is either the source or one of at most
`F` already selected targets. The worst-case row bound is therefore at most
`129F+1` candidate examinations and `O(F)` memory, independent of configured
`N`. Tests force every normal candidate to collide while increasing fanout and
enforce that bound.

The frozen engineering run uses `N = 10^6, 10^9, 10^11`, fanout six, twelve
fixed valid queries and four invalid queries per topology, including GIDs
immediately below, at, and above `2^32` in the 100-billion case. Its artifact is
strictly address/query evidence: `configured = addressable = N`, resident,
updated, and spiking counts are zero, configured base edges are `N*6`, resident
and mutable edge records are zero, and generated edges count only the 72
returned/examined base edges. It reports no token throughput, GPU result,
resident simulation, or capability result.

Invoking the runner without flags only prints the plan, hypotheses, and raw
protocol SHA-256. Execution requires both `--execute` and
`--confirm-protocol-sha <exact raw v2 protocol SHA-256>`. The current exact v2
raw protocol SHA-256 is
`cb3c66402c9d1b88a4a6bf804bf232d7cba2741d4bc469af32a97bad53f40139`.
Before any worker starts, the runner validates that protocol, its complete
implementation contract, and its source bundle, then records the official
Philox known-answer tests and frozen small reference. Every internal worker must
receive the validated source-bundle hash and return the same attestation. The
parent verifies each result. Immediately before same-directory, non-overwriting
atomic publication, the parent rechecks a clean worktree, the exact starting
HEAD, and the source-bundle hash; any drift leaves no final artifact.

## Ordered path

1. Add the scale/event counters and ban unqualified “active” in benchmark claims.
2. Replace materialized `int32` base targets with one versioned 64-bit
   counter-based edge recipe and a materialized small-graph oracle.
3. Execute that recipe source-first on spikes while keeping the existing 200k
   dense CUDA path as the performance/exactness reference.
4. Replace the population-wide frontier scan with resident-tile and due-event
   registries; keep nonzero decays resident until an analytic/tolerance protocol
   is separately accepted.
5. Add bounded target-owned mutable overlays and checkpoint/reshard tests.
6. Pass 100-billion addressability and dormant-range invariance on one machine.
7. Add two-shard CPU determinism, then GPU event batching and distributed scale.
8. Run density-qualified speed curves and only then preregister active-cell
   capability scaling.

The defensible near-term sentence is: **“The runtime addresses 100 billion
procedural neuron identities and exactly simulates a reported resident/updated
subset.”** Any shorter sentence must wait for the corresponding gate.
