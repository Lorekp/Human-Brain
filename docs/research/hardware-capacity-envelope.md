# Hardware capacity envelope: does 10^10 configured fit on this box?

**Date:** 2026-08-23
**Branch:** `codex/scale-opt-cpu` (worktree `human-brain-scale-opt-cpu`)
**Status:** design research. Measured-bytes arithmetic only; no new long runs were executed for this document. Every number is quoted from a committed artifact/source line or computed inline with the formula shown. Decimal GB/TB (`1 GB = 10^9 B`) unless marked GiB/MiB.

**Question answered:** *Is 10^10 configured neurons achievable on this machine (31 GB RAM total / ~13 GB available, RTX 4070 SUPER 12 GB VRAM, NVMe scratch), assuming CPU+GPU+RAM usage with further optimizations? How much of the network could be ACTIVE at once?*

Note on magnitudes: the repository's headline target in
[`100-billion-cell-architecture.md`](100-billion-cell-architecture.md) is
`100,000,000,000 = 10^11` identities (which is why that document correctly says
37 bits: `ceil(log2(10^11)) = 37`). The question assigned here is about
`N_cfg = 10^10`; where the factor-10 difference changes a conclusion, both
columns are shown.

## 0. Measured machine envelope (this session, `wmic`/`nvidia-smi`/`df`)

| Resource | Measured | Value |
| --- | --- | --- |
| RAM total | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory` | 33,378,181,120 B = 31.09 GiB |
| RAM available | `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory` | 13,506,756 KiB = 13.83 GB (12.88 GiB) |
| VRAM total | `nvidia-smi`, RTX 4070 SUPER | 12,282 MiB = 12.88 GB |
| VRAM free | `nvidia-smi` (now); 9,967 MiB at the registered capture | 9,470 MiB = 9.93 GB; 10.45 GB |
| NVMe scratch free | `df /c` | 1,902,957,836 KiB = 1.949 TB |

Budgeting convention below: **13.0 GB usable RAM** (conservative against the
measured 13.83 GB free) and **9.93 GB free VRAM** (or 12.88 GB minus a declared
CUDA-context/kernel-workspace reserve, taken as 1.5 GB).

## 1. Per-tier byte budgets from the runtime's own accounting

The five-level residency model is the one defined in
[§The scale contract](100-billion-cell-architecture.md):
`N_spk <= N_upd <= N_res <= N_addr <= N_cfg`. A configured identity, a resident
cell, and an updated cell are different objects with different byte costs. The
tiers below are exactly those levels priced out.

### 1.1 Cold tier — configured identity records

The procedural runtime's configured address space is **metadata, not storage**:
`TopologyManifest` declares a contiguous uint64 GID range and
`describe_cell(gid)`/`base_edges(gid)` are computed on demand
([`procedural.rs`](../../src/humanbrain/_native_src/src/procedural.rs),
[`procedural_runtime.py`](../../src/humanbrain/procedural_runtime.py)). The
recorded address-space artifact measured **20,087 host bytes total at
`configured = 10^6`, `resident_peak = 0`**
([`procedural_address_space_v1.json`](../../artifacts/procedural_address_space_v1.json),
`workers.0.results.0.scale_accounting.memory.host_bytes`), with tracemalloc
log-log slopes of 0.0012–0.0040 across topologies (flat in `N`). The v2
protocol pins the frozen `N = 10^6, 10^9, 10^11` engineering run; **no v2
artifact has been published yet**, so the O(1)-manifest property at `10^10` is
extrapolated from v1 evidence plus protocol, not yet a recorded result at that
scale.

If per-ID cold records *were* materialized (formula: `bytes = rate × N`):

| Cold record | Rate | `N = 10^10` | `N = 10^11` | Fits where |
| --- | ---: | ---: | ---: | --- |
| membership bit | 1 bit | 1.25 GB | 12.5 GB | RAM (10^10); NVMe (both) |
| status byte | 1 B | 10 GB | 100 GB | NVMe comfortably; RAM marginal |
| `uint64` gid index | 8 B | 80 GB | 800 GB | NVMe (1.949 TB free) only |
| ID width needed | — | 34 bits (`2^33 = 8.59e9 < 10^10 <= 2^34`) | 37 bits | `uint32` cannot address either globally |

### 1.2 Warm tier — resident mutable-state floor: 60 B/cell

Current GLIF5 state = `float64` voltage + two after-spike currents + two
threshold components + `int32` refractory = 44 B; recurrent synaptic current +
eligibility add 16 B ⇒ **60 B/cell before topology, workspaces, readout,
allocator overhead**
([arithmetic in 100-billion-cell-architecture.md §Resident state](100-billion-cell-architecture.md),
from [`cells.py`](../../src/humanbrain/cells.py) +
[`recurrent.py`](../../src/humanbrain/recurrent.py)).
The native event-driven runtime stores the same fields **plus the u64 gid in an
AoS record**: `CellState`/`ProceduralResidentRecord` =
gid 8 + voltage 8 + after[2] 16 + threshold_spike 8 + threshold_voltage 8 +
refractory 4 (+4 padding) + synaptic_current 8 + eligibility 8 =
**72 B/cell resident record**
([`procedural.rs`](../../src/humanbrain/_native_src/src/procedural.rs), struct defs).

### 1.3 Hot tier — working set including event buffers (measured)

`native_payload_bytes` in
[`logs/scale_opt_cpu_baseline_v1.json`](../../logs/scale_opt_cpu_baseline_v1.json)
counts reserved capacities, not occupancy
([`payload_bytes()`](../../src/humanbrain/_native_src/src/procedural.rs)):
residents `R×72` + staged `min(R,Q)×72` + pending/submitted/work event vecs
`3×Q×40` + spikes `S×16` + step-spike and gid-run scratch `2×R×8` + two
interval ledgers `2×D×(8+1)` + row scratch, where `R` = resident capacity,
`Q` = event capacity, `S` = spike capacity, `D` = interval-distinct capacity
(`D >= R` is a constructor contract;
[`procedural_runtime.py`](../../src/humanbrain/procedural_runtime.py) status 10).
Record sizes: **event record 40 B** (due_step/target/source/edge_slot u64 +
current f64), **spike record 16 B**.

Measured payload columns (baseline JSON, `payload_bytes_per_resident_cell`):

| R | d=1% | d=5% | d=20% | d=100% |
| ---: | ---: | ---: | ---: | ---: |
| 10^3 | 70,341.6 | 14,068.3 | 3,517.1 | 703.4 |
| 10^4 | 15,381.0 | 3,076.2 | 769.1 | 176.7 |
| 10^5 | 10,343.7 | 2,068.7 | 517.2 | 121.8 |
| 10^6 | 10,986.9 | 2,197.4 | 549.4 | **110.42** |

Reading the table honestly: the per-cell column **falls** with density because
the denominator (`final_residents = d×R`) grows while most components scale
with capacities, not occupancy. The asymptotic **marginal cost of one more
resident-capacity slot is ~88 B** (72 record + 8 step_spikes + 8 gid_run) plus
**18 B per interval-distinct slot** (two ledgers × 9 B) when `D = R`, i.e.
**~106 B/resident-of-capacity**, plus a fixed `3×Q×40 + S×16` term (only
491,520 + 16,384 B at the benchmark's `Q = 4096, S = 1024` floors — noise above
`R >= 10^5`). Measured totals sit ~2.9 % above this component sum at `R = 10^6`
(allocator granularity + the runtime struct itself), so the **110.42 B
measured** column, not the idealized sum, is what budgets below use. The
delta check is exact where it matters: raising `Q` from 4096 to 8192 at
`R = 10^6` moved the measured payload by exactly
`3 × 4096 × 40 = 491,520 B` (109,868,856 → 110,360,376), confirming three
event vectors at 40 B/record.

### 1.4 GPU tier — device-resident arrays + kernel working space

From [`sparse_cuda_engineering_v1_optimized.json`](../../artifacts/sparse_cuda_engineering_v1_optimized.json),
200,000-cell registered run: device persistent arrays 27,632,828 B =
**138.16 B/cell**; host persistent arrays 37,099,184 B = 185.50 B/cell; the
dense control readout (409,600,000 B) is excluded by the sparse-readout design;
device→host transfer is 1,600,008 B/token of kernel working traffic. These
match the figures quoted in
[100-billion-cell-architecture.md §What the current measurements mean](100-billion-cell-architecture.md).

## 2. Capacity table and the combined configured-N ceiling

Formulas: `residents_tier = usable_bytes / bytes_per_cell`.

| Tier | Budget | Rate | Max cells | Source of rate |
| --- | --- | ---: | ---: | --- |
| CPU warm floor only | 13.0 GB | 60 B/cell | **2.17e8** | §1.2 |
| CPU native AoS record only | 13.0 GB | 72 B/cell | 1.81e8 | §1.2 |
| CPU full measured hot set | 13.0 GB | 110.42 B/cell | **1.18e8** | §1.3 |
| GPU device arrays | 9.93 GB free (now) | 138.16 B/cell | **7.19e7** | §1.4 |
| GPU device arrays | 12.88 GB − 1.5 GB ctx/ws | 138.16 B/cell | 8.24e7 | §1.4 |
| Combined CPU + GPU resident shards | 13.0 + 9.93 GB | mixed | **≈ 1.90e8** | sum of rows 3+4 |
| Cold bitset (optional) | RAM | 1 bit/ID | 10^10 → 1.25 GB; 10^11 → 12.5 GB | §1.1 |
| Cold u64 index | NVMe 1.949 TB | 8 B/ID | 10^10 → 80 GB; 10^11 → 800 GB | §1.1 |
| Max cold B/configured-ID | NVMe 1.949 TB | — | 194.9 B at 10^10 | division |

**Combined configured-N ceiling with tiering.** Because configuration is
O(1) manifest space (§1.1), the configured ceiling is not byte-bound at all on
this box:

- `N_cfg = 10^10`: reachable **today as address space** — zero per-ID bytes,
  or a 1.25 GB cold bitset in RAM, or even a full 80 GB u64 gid index paged
  from NVMe. This is Gate 2 territory ("100-billion addressability", here at
  10^10) and is consistent with the residency model: configured ≠ resident.
- `N_cfg = 10^11` (the repo headline): same conclusion — manifest O(1);
  bitset 12.5 GB would have to live partly in NVMe-backed pages; u64 index
  800 GB fits the 1.949 TB free NVMe.
- **Resident** `N_res`: 10^10 residents need ≥ 600 GB at the 60 B floor —
  44× the free RAM and far beyond any honest "resident" definition even with
  page-file backing (see §4). Resident ceiling on this box is **~1.9e8**
  (CPU+GPU split), i.e. **~1.9 % of 10^10 configured**.
- Page-file caveat: Windows commit can extend past RAM onto the NVMe, but the
  runtime's contracts are **fail-stop, not spill**: exhausting resident/event/
  spike/distinct/allocation capacity raises a deterministic stop (statuses 2,
  3, 8, 9, 10 in
  [`procedural_runtime.py`](../../src/humanbrain/procedural_runtime.py)); there is
  no spill path. Letting the OS page warm state converts fail-stop into
  unbounded-latency thrashing inside the advance loop. NVMe belongs in the
  **cold/checkpoint tier**, not warm overflow.

## 3. Active-cell budget: concurrent active = resident × density

**What the runtime sustains by its own contracts.** The fail-stop bounds cap a
single `advance()` working set: resident, event-queue, spike, and
interval-distinct capacities. Because the constructor requires
`interval_distinct_capacity >= resident_capacity`
([`procedural_runtime.py`](../../src/humanbrain/procedural_runtime.py)), the
runtime is contracted to track **100 % of residents as distinct-and-updated in
an interval**. The baseline exercised exactly that: `density_target = 1.0`
completed fail-stop-free at every `R` up to `10^6`
(`final_residents = 999,423`, achieved updated fraction 0.50 of capacity — the
rest is the ramp schedule, not a cap;
[`logs/scale_opt_cpu_baseline_v1.json`](../../logs/scale_opt_cpu_baseline_v1.json)).
So the *contractual* active fraction is **up to 1.0 × resident population**;
what limits useful activity is measured throughput, not the contracts:

- CPU event-driven engine: peak **4.94e7 cell-steps/s** (R=10^4, d=20 %);
  9.18e6 at R=10^6/d=100 %; peak combined event throughput **2,638.6
  event-items/s** (delivered + generated ÷ wall, same JSON).
- CUDA dense engine: **2.8806e9 cell-steps/s** at the registered 200k run
  ([100-billion-cell-architecture.md §Compute](100-billion-cell-architecture.md)).

**What real cortical-style workloads actually need (recorded stats).**

- Task-curriculum-style validation, 6,144 cells
  ([`large_scale_capability_validation_v1.json`](../../artifacts/large_scale_capability_validation_v1.json)):
  mean firing **14.97 Hz/cell**, **0.0749 spikes/cell/token**
  (29,441 reservoir spikes / 6,144 cells / 64 tokens), spiking in 98.4 % of
  tokens, responsive-state fraction 0.941, mean active-eligibility fraction
  0.773.
- Registered 200k CUDA task run: **167–168 unique spiking reservoir cells per
  repeat = 0.084 % of resident**, while `updated_reservoir_cells_per_integration_step
  = 200,000` — i.e. the production kernel is compute-dense even when spiking is
  sparse (`sparse_cuda_engineering_v1_optimized.json`); the registered run
  averaged 1.125 spike events/token
  (quoted in 100-billion-cell-architecture.md from `cuda_scaling_v1.json`).
- Exact frontier, natural trajectory: **0.052 %** of 1,048,576 residents
  updated in a token; controlled seeding at 1 % density updates 1.00 %
  (`native_activity_density_v1.json`, `native_scaling_v2.json`).
- **a013:** the typed-composition-a013 scale ladder is *preregistered* as the
  successor to the sealed A012 negative result
  ([`docs/typed-composition-a012-v1-results.md`](../typed-composition-a012-v1-results.md),
  "Both deficits motivate the preregistered scale ladder (typed-composition-a013
  family)"). **No a013 experiment JSON exists in this worktree** — there are no
  recorded a013 spike stats to cite. The 6,144-cell validation run above is the
  closest recorded task-workload activity measurement.

**Active-fraction arithmetic** (formulas shown):

- Memory-feasible simultaneous active (all residents updated, density 1.0,
  which the contracts and the d=1.0 runs support at small R):
  `A_max = 1.90e8` cells = **1.9 % of 10^10 configured**. For reference, 2 % of
  10^10 (the human-brain simultaneous-firing figure supplied with the question)
  = `2.0e8` cells — the same order as our resident ceiling. The box can
  therefore *hold* a 2 %-active cohort as residents, with the other ~98 %
  living as zero-byte procedural identities.
- Interactive-rate active: at 100 token/s the measured CUDA engine updates
  `2.8806e9 / (100 steps/token × 100 token/s) = 288,060` cells/token =
  **0.0029 % of 10^10** (and matches the ~288,000 extrapolation already
  published in 100-billion-cell-architecture.md). At 1 token/s: 2.88e7 cells =
  0.29 %.
- Whole-cohort full touch: `1.9e8 × 100 = 1.9e10` cell-steps ⇒ **6.9 s/token**
  on the measured GPU engine (0.145 token/s), or **~4.8 days/token** on the
  scalar CPU engine at 4.6e7 cell-steps/s.
- Event-pipeline reality check at scale: a 10^8-resident net firing at the
  recorded task rate emits `1e8 × 0.0749 × (1 + 8) ≈ 6.7e7` event-items per
  token; the measured CPU procedural engine delivers ~2.6e3 event-items/s ⇒
  **~25,500 s (~7 h) of event traffic per such token**. The event path — not
  memory — is the binding constraint on high-active-fraction operation today;
  the CUDA engine sidesteps routing by updating everything densely, which is
  precisely the trade the frontier work is trying to remove.

## 4. The honest verdict

1. **`N_cfg = 10^10` configured: YES — already in reach.** Identity is a
   contiguous uint64 range (34 bits needed; `uint32` insufficient) with O(1)
   manifest bytes at recorded `10^6`, protocol-pinned (not yet recorded) at
   `10^9–10^11`. Even materialized cold tiers fit: 1.25 GB bitset in RAM, 80 GB
   gid index on 1.949 TB NVMe. Nothing in the measured byte budget blocks
   configured 10^10 today.
2. **`R = 1e7–1e8` resident: YES.** `R = 1e7` costs 1.10 GB hot at the measured
   110.42 B/cell (8 % of free RAM). `R = 1e8` costs 11.04 GB vs 13.83 GB free —
   feasible on CPU alone with ~2.8 GB headroom, comfortable (~6–7 GB) after the
   §5 packing wins, and impossible on the GPU alone (13.8 GB > 12.88 GB total),
   hence the CPU+GPU shard split. `1e8` sits inside the claimed range; `1e9`
   resident does not (≥ 110 GB).
3. **Active fraction:** contractual maximum is 100 % of residents (fail-stop
   caps, d=1.0 runs pass), giving a memory-feasible simultaneous-active ceiling
   of **≈ 1.9e8 cells ≈ 1.9 % of configured — essentially the 2 % human
   simultaneous-firing cohort**. Per-token interactive activity at measured
   throughput is far smaller: 2.9e5 cells/token at 100 token/s (0.0029 % of
   configured), and recorded task workloads only *need* 0.084 %–1 % of resident
   cells updating/spiking per token anyway — well inside what the memory tiers
   hold. The gap between "can hold 2 %-active" and "can drive 2 %-active at
   interactive rates" is compute (event-delivery throughput), not bytes.
4. **What would falsify this verdict:** a v2 address-space artifact showing
   super-O(1) manifest growth at 10^10; a measured hot-set slope steeper than
   the ~106 B/resident component model at larger R; or an event-throughput
   curve that stays near 2.6e3 events/s after the frontier rewrite (then the
   §3 active fractions drop proportionally).

## 5. What moves each ceiling (estimated savings, determinism risk)

Baseline for comparison: 110.42 B/resident measured hot; 40 B/event record;
18 B/ledger slot. Determinism bar: all 16 canonical digests in
[`benchmarks/scale_opt_cpu_golden_v1.json`](../../benchmarks/scale_opt_cpu_golden_v1.json)
must reproduce bit-exactly (struct-packed resident/event/spike bits), per the
golden-file contract.

| Optimization | Ceiling it moves | Estimated saving | Determinism risk |
| --- | --- | --- | --- |
| **SoA packing + drop embedded gid** (implicit sorted shard-local index; keep sparse gid map only for evicted cells) | Hot resident cost 110.42 → ~102 B/cell; warm floor 72 → 68 B (pad removal) | −8 B/resident (−7 %); unbounded rematerialization headroom | **Low**: identical f64 op order; risk is eviction bookkeeping correctness, covered by existing fail-stop tests |
| **Interval-ledger compression**: single-buffered tile-local bitmap instead of 2×(8+1) B/slot ledgers | Hot overhead −16 B/resident when `D = R` | −89 % of ledger component | **Low**: same flags, same fail-stop at capacity; requires tile-dense gid ranges |
| **u32 shard-local gids in event/spike records** (global IDs stay u64; 10^10 needs 34 bits so u32 is shard-only, valid for shards < 2^32) | Event record 40 → 24 B; spike 16 → 12 B; event-slot cost 120 → 72 B | −40 % event component | **Low** if the shard mapping is frozen and the `(step, target, source, slot)` sort order is preserved — u32 comparisons induce the same total order |
| **Delta-encoded event records** (fixed-width deltas inside sorted batches: u16 step delta, u32 target delta, u8 slot) | Event record → ~12–14 B | −65–70 % on the event component | **Medium**: decode must be exact integer math; variable-length encodings are platform-divergence hazards; tie-breaks must follow the frozen key |
| **mmap-backed cold tier on NVMe** (cold identity/status/param records; 10 GB/byte-of-status or 80 GB/u64-index at 10^10) | Configured tier gains arbitrary per-ID cold metadata; resident ceiling unchanged | Enables up to 194.9 cold B/ID within free NVMe (§2) | **Low for read-only paging** (values identical); **Medium operationally**: page-fault latency is unbounded, so cold reads must stay outside the fail-stop advance path or carry a preregistered fault bound; dirty write-back is checkpointing, a semantics change to declare |
| **f32 state halves** (voltage/currents/thresholds f32) | Warm floor 60 → 32 B; hot record 72 → ~40 B | −45–55 % across warm+hot | **High**: breaks every golden digest; only legal behind a new tolerance-oracle protocol per the exactness rules in 100-billion-cell-architecture.md — not a drop-in |
| Event-capacity right-sizing | none worth having | Fixed 3·Q·40+S·16 term is 0.51 MB at benchmark floors — negligible at R ≥ 10^5 | n/a — listed to preempt wasted effort |

Net effect at `R = 1e8`: SoA (−8) + ledger bitmap (−16) ⇒ ~86 B/resident ⇒
**8.6 GB hot (vs 11.04 GB)**, lifting RAM headroom from 2.8 GB to ~5.2 GB and
the CPU-resident ceiling from 1.18e8 to ~1.5e8; with the GPU shard the combined
ceiling approaches **~2.3e8 residents ≈ 2.3 % of 10^10 configured**. None of
these moves the resident ceiling by the 53× needed for 10^10 residents — that
requires different hardware, not different packing.

## 6. Preregistered validation run (proposed, not yet executed)

- **Hypothesis H1:** an SoA-packed resident layout with single-buffered
  tile-local ledger flags (rows 1–2 of §5) reproduces all 16 canonical SHA-256
  state digests in `scale_opt_cpu_golden_v1.json` exactly, while reducing
  `native_payload_bytes` in the `R = 10^6, d = 1.0` scenario from the measured
  110,360,376 B to **≤ 92e6 B** (predicted ≈ 102 B × 1e6 + staged/events ≈
  88.5–91 MB).
- **Protocol:** implement behind the existing public `advance()` seam; rerun
  `benchmarks/scale_opt_cpu_throughput.py --check-golden` against the unchanged
  golden file; add one scenario at `R = 10^8` (window per the existing
  `window_steps` rule) with a preregistered peak-private-memory bound of
  **13.0 GB** and fail-stop-free completion.
- **Gates:** digest equality is binary — any mismatch fails H1 and rejects the
  layout (no tolerance relabeling). Memory overshoot fails the R=1e8 claim but
  not H1. Publish the artifact as `logs/scale_opt_cpu_soa_v1.json` with the same
  counter schema.

## Sources

- Measured baseline curves/payload columns: [`logs/scale_opt_cpu_baseline_v1.json`](../../logs/scale_opt_cpu_baseline_v1.json); oracle digests: [`benchmarks/scale_opt_cpu_golden_v1.json`](../../benchmarks/scale_opt_cpu_golden_v1.json); capacity-sizing code: [`benchmarks/scale_opt_cpu_throughput.py`](../../benchmarks/scale_opt_cpu_throughput.py)
- Struct sizes and `payload_bytes()` components: [`src/humanbrain/_native_src/src/procedural.rs`](../../src/humanbrain/_native_src/src/procedural.rs); config contracts and fail-stop statuses: [`src/humanbrain/procedural_runtime.py`](../../src/humanbrain/procedural_runtime.py)
- 60 B/cell floor, 185.50/138.16 B/cell measurements, residency model, gates: [`docs/research/100-billion-cell-architecture.md`](100-billion-cell-architecture.md)
- Address-space evidence (v1, `10^6`, 20,087 host bytes): [`artifacts/procedural_address_space_v1.json`](../../artifacts/procedural_address_space_v1.json) (v2 run protocol-pinned, artifact pending)
- Activity stats: [`artifacts/large_scale_capability_validation_v1.json`](../../artifacts/large_scale_capability_validation_v1.json), [`artifacts/sparse_cuda_engineering_v1_optimized.json`](../../artifacts/sparse_cuda_engineering_v1_optimized.json), [`artifacts/native_activity_density_v1.json`](../../artifacts/native_activity_density_v1.json), [`artifacts/native_scaling_v2.json`](../../artifacts/native_scaling_v2.json); a013 status: [`docs/typed-composition-a012-v1-results.md`](../typed-composition-a012-v1-results.md)
- Machine envelope: `Get-CimInstance`/`nvidia-smi`/`df` outputs reproduced in §0 (measured 2026-08-23)
