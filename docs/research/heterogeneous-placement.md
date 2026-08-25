# Heterogeneous placement: splitting large circuits across GPU, pinned RAM, and virtual identity

**Date:** 2026-08-23

**Status:** Preregistered design note for the next optimization campaign. No model
artifact, sealed protocol run, or product code path was changed by this document.
Every number cited below is either computed from committed result JSONs or marked
as an assumption.

**Code baseline:** `d4b2f23` (branch `codex/scale-opt-gpu`).

**Scope boundary:** Human Brain GLIF5 recurrent/procedural paths only. Not Fly
Brain; no evaluation data inspected. This note proposes engineering placement
policy; it authorizes no capability claim.

## Directive

Optimize LARGE networks split across CPU + GPU + RAM simultaneously: hot sets
resident in VRAM, warm partitions in pinned host RAM, cold identities virtualized
in the counter64-v2 address space with residency faulted on first spike. The
procedural runtime's event-driven activation is the natural faulting mechanism.
This note defines the partitioning policy, the inter-tier transfer protocol, a
cost model at 1e7/1e8/1e9 configured cells, the honest counterarguments, and the
smallest decisive experiment.

## Measured grounding

### GPU lane (this branch, `benchmarks/scale_opt_gpu_*.json`)

- Persistent device state is **≈138 B/cell** (`device_persistent_array_mib`
  divided by cells is 138.0-138.3 at every size from 100 k to 1 M).
- Full-touch float64 integration is device-bandwidth-bound: ~90 B/cell/step,
  measured ~370 GB/s effective (73 % of the 504 GB/s spec) at 1 M cells;
  the roofline is ~21 ms/token at 1 M (~48 tps absolute ceiling) and the card
  measured 29 ms/token (34.8 tps end-to-end). Full-touch on one 4070 SUPER dies
  around 1 M cells; VRAM alone would allow ~80 M residents at 138 B/cell, so
  **time binds long before memory**.
- Zero-spike edge-pass elision operates at whole-token granularity (`total_spikes`
  is zeroed once per kernel launch; one launch = one token): **78 % of tokens are
  entirely spike-free**, so >=78 % of steps skip the reverse-CSR scan. The win is
  fragile w.r.t. the spike distribution.
- Demonstrated end-to-end: 479/466 inf/train tps median at 100 k cells,
  323.8/312.7 at 200 k (three-repeat medians, raw repeats committed).
- License for tier hops: `tests/test_cuda.py` pins **bit-exact CPU/CUDA state
  parity after every token**. A cell whose state words move between engines
  bit-for-bit, under the same equations, has a continuous trajectory.

### CPU lane (`scale_opt_cpu_baseline_v1.json`, branch `codex/scale-opt-cpu`)

- Exact frontier throughput is **~40 M cell-steps/s flat through 100 k residents**
  (36.2-49.4 M across capacities 1 k-100 k and densities 0.01-1.0) and falls to
  **9.18 M cell-steps/s at 1 M residents at density 1.0**.
- Accounted native payload is **110-177 B/resident** at density 1.0 for resident
  counts 10 k-1 M (fixed overhead amortizes; 703 B/cell at 1 k residents). The
  ~181 B/cell planning figure for pinned warm partitions quoted in the campaign
  brief sits at the upper end of this measured band and is used as a
  conservative constant below.

### Residency and ordering contracts already in force

| Contract | Source | What placement may rely on |
| --- | --- | --- |
| Nonresidency rule | `docs/research/100-billion-cell-architecture.md`, Gate-3 slice | A cell may be nonresident only while its exact state is canonical rest, it has no due event, no mutable overlay requires decay, and nothing reads it. First incoming event materializes state. |
| Eviction rule | Gate-3 slice | Eviction only after a real fixed-step transition leaves all eight mutable fields at canonical zero. Signed zero is canonicalized at all write seams. |
| Fail-stop | Gate-3 slice | Exhausting resident/event/spike/interval/allocation bounds poisons the runtime (statuses 4/5/9); it never spills into a dense or approximate path. |
| Event ordering | Gate-3 slice | Events accumulate in `(due_step, target_gid, source_gid, edge_slot)` order; residents update in ascending GID; spike expansion in ascending `(source, slot)`. Delays are positive integers, so a spike never affects its source's own step. |
| Storage bound | Gate-3 slice | `O(R_capacity + Q_capacity + S_capacity + F + D_capacity)`, independent of configured/addressable `N`. |
| Topology on demand | 100 B architecture | One versioned counter-based `base_edge(...)` function regenerates a spiking source's row; no stored forward array or reverse CSR is required on the 100 B path. |

## Tier definitions

| Tier | Residence | Bytes/cell | Integration duty |
| --- | --- | --- | --- |
| Hot | VRAM, persistent cooperative kernel | 138.0 (measured) | Every enabled cell integrated for all 100 steps of every token (full touch), plus edge scan on non-elided steps. |
| Warm | Pinned host RAM, CPU exact frontier | ~181 planned (measured band 110-177) | Same equations, same op order, every step, for cells with nonzero state or due events; enforced by the Gate-3 nonresidency rule, not by sampling. |
| Cold | Manifest identity only | ~0 marginal (population manifest entries + recipe) | Nothing. Faulted to warm on first spike/event via `base_edge` row regeneration; unborn developmental cohorts are canonical-rest by construction. |

Candidate placement units are the regioned architecture's manifest populations:
contiguous 64-bit ID ranges carrying topology recipe version, parameter family,
dimensions, radices, quotas, and seed - subdivided into fixed-size tiles
(e.g. 4096 cells) so placement metadata stays `O(tiles)`, never `O(cells)`.
Family-scoped cohorts are concrete near-term instances: the current circuit's
signed-sign E/I labeling, the typed E/I families proposed in
`docs/research/human-ei-cell-types-and-balanced-dynamics.md` (the campaign
brief's "E/P" excitatory/pyramidal vs inhibitory split), and developmental
recruitment prefixes (unborn cells beyond the recruited prefix are canonical-rest
and therefore legally cold today).

## Partitioning policy

All decisions are made once per epoch boundary, in ascending tile-GID order,
from quantities that are exact functions of simulation state. No wall-clock,
occupancy, or thread-order input is admissible; this is what makes placement
itself deterministic and checkpointable.

Signals kept per tile (integer/fixed-point only):

1. Spike-count EMA over a sliding window of epochs (fixed-point, shift-based).
2. Last-spiked step index (recency).
3. Nonzero-tail cell count (cells failing the canonical-zero eviction rule).
4. Due-event pressure: queued events routed into the tile within the next epoch.
5. Manifest role: declared, never inferred - e.g. typed family (E/I),
   recruitment cohort, readout-adjacent region.

Policy (evaluated in order; every comparison is a total order with GID
tie-break):

- **Hot admission:** sort tiles by (EMA desc, last-spiked recency asc, GID asc);
  admit until the VRAM budget (cells x 138 B <= budget) or the hot-cell
  bandwidth budget is reached. Tiles with due events next epoch preempt spare
  room ahead of EMA rank.
- **Warm retention:** any tile containing at least one nonzero-tail or due-event
  cell must live in hot or warm; the policy may never delete state to dodge the
  duty. Warm capacity is pinned-RAM-bound.
- **Cold:** everything else, including all never-touched identities.
- **Role override:** manifest roles may pin a tile hot (readout-critical cohort)
  or force cold (declared silent cohort). Overrides are configuration, reported
  in every checkpoint.
- **Hysteresis:** promote threshold strictly above demote threshold, and at most
  one direction change per tile per epoch, so the policy cannot thrash. Every
  decision appends `(epoch, tile_id, from_tier, to_tier)` to the checkpoint
  delta log.

Determinism argument: tier assignment feeds back into execution only through
*where* the identical bit patterns live and *when* batches are cut (epochs).
Because both engines are pinned bit-identical by `tests/test_cuda.py`, event
keys are position-independent GID tuples, and accumulation order is frozen, the
canonical state hash per epoch is a function of the token stream and seeds
alone - not of the placement map. This is asserted, not assumed, by the
experiment below.

## Transfer protocol

- **Epoch boundaries are the only transfer points.** A shard/tier advances to
  the next safe epoch boundary, exchanges target-routed spike batches, and
  resumes only after all earlier events are known (the two-CPU-shard start
  prescribed by the 100 B architecture, extended with GPU shards).
- **Promotion (warm -> hot):** serialize the tile's eight-field records plus
  sparse overlays verbatim (little-endian f64s, canonical zeros), upload over
  pinned staging, and verify a per-tile SHA-256 against the host copy before the
  epoch commits. The GPU continues from identical bit patterns; no re-scaling,
  no re-initialization, no tolerance.
- **Demotion (hot -> warm):** download the same byte-exact records; the CPU lane
  adopts them untransformed. Dropping a tile to cold outright remains illegal
  unless every cell passes the canonical-zero rule - decaying eligibility tails
  (tau = 40 ms) keep tiles warm for hundreds of steps after their last spike.
- **Events survive hops structurally.** Keys are `(due_step, target_gid,
  source_gid, edge_slot)` - GIDs, not device addresses - and delays are positive
  integers, so an event generated before a hop is delivered identically
  regardless of where the target lived at either endpoint. After any batch moves
  tiers it is re-sorted by the frozen key before accumulation, preserving one
  global summation order.
- **Failure semantics:** a promotion that would exceed a tier budget fails
  before any tier mutates (transactional rejection at the orchestrator seam),
  and repeated overflow poisons the run - mirroring Gate-3's
  fail-stop-never-approximate principle. There is no silent spill path.
- **Checkpoints:** manifest + base seeds once; per-tier tile records, pending
  event buckets, counter totals, and the placement delta log. Restore onto a
  different tier layout must reproduce the same canonical event/state hash.

## Cost model (this card: RTX 4070 SUPER 12 GB, ~25 GB/s pinned H2D assumed)

Per-token tier budgets at the 100 tps gate (10 ms/token):

- Hot: bandwidth-bound at ~24 ns/cell/token beyond the latency floor (~1.6
  ms/token per 100 k residents measured kernel-side) -> **~0.4 M hot cells**
  inside a 10 ms budget; 8 GB VRAM allows 58 M, so **bandwidth binds first**.
- Warm: 40 M cell-steps/s flat => **~4 k fully-integrated warm cells per token**
  at 100 tps (100 steps each); measured tail fractions at benchmark activity are
  135-209 responsive cells per 100-200 k configured (0.07-0.21 %), comfortably
  inside this at 200 k scale - but *linear scaling of tail fraction with N is
  unverified and is the model's main load-bearing assumption*.
- Transfer: promoting a 100 k-cell tile moves ~18 MB ~ 0.7-1 ms. Amortization
  rule: a promoted tile must stay hot >= K epochs, K ~ t_transfer / t_marginal,
  else the bus eats the win (see counterarguments).

| Configured | Assumed hot (policy cap) | Warm (0.1 % tail frac.) | Cold | Hot VRAM | Warm pinned | Expected binding term |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e7 | 100 k | ~10 k | ~9.89 M | 13.8 MB | ~1.8 MB | GPU latency floor; ~300-500 tps plausible, >100 tps gate met |
| 1e8 | 200 k | ~100 k | ~99.7 M | 27.6 MB | ~18 MB | Warm duty approaches the 4 k/token budget -> overflow must promote to hot (room exists) |
| 1e9 | 300 k | ~1 M | ~998.7 M | 41 MB | ~181 MB | Warm budget exceeded ~250x if tail fraction holds linearly -> hot bandwidth saturates; 100 tps **fails** unless simultaneous-tail fraction falls to ~<=0.04 % of configured or warm duty is batched across epochs |

The 1e9 row is the honest headline: on this card, heterogeneous placement buys
configured-scale *capacity*, not proportional *throughput*. Feasibility at 1e9
hinges on tail-fraction physics (how quickly eligibility tails cancel/underflow
as recruited cohorts grow), which is unmeasured beyond 200 k configured.

## Honest counterarguments

- **PCIe may dominate.** With churny activity, tiles ping-pong: at one
  promotion per token per 100 k-cell tile, transfers add ~1 ms/token against a
  2-10 ms token budget - a 10-50 % tax that can exceed the GPU's marginal
  advantage for exactly the mid-size tiles placement loves. The hysteresis and
  K-epoch amortization rules exist for this; if real churn defeats them, the
  design reduces to "keep the hot set static," and the dynamic policy earns
  nothing.
- **Sync at region boundaries may dominate.** Every epoch ends in a
  sort-and-exchange of crossing events; cost scales with crossing-edge count,
  not N. If activity concentrates *across* placement units (E<->I blocks are
  deliberately cross-family), crossing volume can rival intra-tile work, and
  finer tiles (better load balance) make it worse. Tile size is therefore a
  real tuning knob with opposing pressures, not a free constant.
- **Dual-engine correctness surface.** Bit-exact parity is pinned today for the
  recurrent reservoir path at tested sizes; extending it to procedural-runtime
  tiles, overlays, and mid-stream hops is unproven engineering, and any equation
  change must re-pass parity on both engines forever.
- **Elision cuts both ways.** 78 % spike-free tokens mean the GPU edge scan is
  often idle anyway; at low activity the hot tier's marginal value over a good
  CPU frontier shrinks toward the neuron-update floor, while dense firing
  destroys both elision *and* the warm tier's sparsity assumption.
- **Warm-tier collapse under bursts.** 40 M cell-steps/s holds only below
  ~100 k residents; a burst that parks 1 M tails on the CPU drops it to
  9.2 M cell-steps/s - 25x slower - right when queue pressure peaks.
- **Placement units are partly aspirational.** Typed E/I families and region
  roles are documented plans, not executed circuit features; today's only real
  signals are activity statistics and developmental cohort boundaries.

## Smallest decisive experiment (preregistered)

One machine, one card, one token stream (seeded as in
`scale_opt_gpu_bench.py`), 1 M configured cells, 100-step tokens, fanout 6:

- **Arm A - heterogeneous:** 100 k-cell hot cap on GPU; warm remainder on the
  CPU exact frontier; unborn identities cold; EMA-driven promotion/demotion
  with dual thresholds at every epoch boundary.
- **Arm B - pure-CPU control:** all 1 M cells on the CPU lane frontier, same
  stream, same epoch cadence, matched updated-cell work per epoch.
- **Arm C - shuffled-placement negative control:** Arm A's machinery with tile
  assignment randomized per epoch (ignoring EMA), to show the policy, not the
  hardware, produces any advantage.

Primary metrics: end-to-end and kernel tokens/s; cell-steps/s per tier;
`resident_peak`/`resident_distinct` per tier; promotions/demotions per epoch;
transfer bytes; per-epoch canonical state hash.

Falsifiable acceptance (preregistered; may be tightened, not loosened):

1. **Determinism:** hash(A, e) == hash(B, e) == hash(C-state-trajectory legality)
   for every epoch e - any mismatch is a determinism bug, not noise. (C differs
   in placement only; its hashes must also equal A's.)
2. **Value:** tokens/s(A) >= 1.5 x tokens/s(B) at matched updated-cell counts,
   and tokens/s(A) > tokens/s(C) - otherwise heterogeneous placement is rejected
   for this workload.
3. **Stability:** median promotion churn <= 2 tiles/epoch after warm-up; zero
   budget-induced poisonings in A.
4. **No-contract-violations:** zero events delivered out of frozen-key order;
   zero reads of nonresident state (Gate-3 audit counters).

Abort conditions: any Arm-A poisoning, any hash inequality, or GPU parity test
failure at transferred-tile boundaries ends the experiment with a written
negative result (house rule: sealed negatives are recorded, cf.
`docs/typed-composition-a012-v1-results.md`).

## What would falsify the design

- Tail fractions growing with configured N (kills the 1e9 row and probably 1e8).
- Cross-tier event volume scaling superlinearly with tile count.
- CPU/CUDA parity breaking on any transferred-tile state pattern.
- Promotion churn persistently above the amortization threshold under natural
  token streams.

## Provenance

- GPU numbers: `benchmarks/scale_opt_gpu_report.md`,
  `benchmarks/scale_opt_gpu_{medians-small,medians-large,baseline,baseline-large,optimized*}.json`
  (this branch).
- CPU numbers: `logs/scale_opt_cpu_baseline_v1.json` (branch
  `codex/scale-opt-cpu`, commit `6bf731e`).
- Contracts: `docs/procedural-runtime-gate3-slice.md`;
  `docs/research/100-billion-cell-architecture.md`;
  `docs/research/human-ei-cell-types-and-balanced-dynamics.md`.
