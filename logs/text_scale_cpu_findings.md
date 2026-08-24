# Million-scale TEXT training on the CPU/event-driven path: connectivity, throughput, crossover

**Date:** 2026-08-23
**Branch:** `codex/scale-opt-cpu` (worktree `human-brain-scale-opt-cpu`)
**Status:** feasibility arithmetic + gap analysis from committed measurements. No new long runs; every number is quoted from a committed artifact/source line or computed inline with the formula shown.

**Mission:** establish what the CPU/event-driven procedural path can contribute to million+-scale *text* training: (1) does `RecurrentCellLanguageModel` grow/complete run over the procedural runtime? (2) if not, what adapter closes the gap (eligibility/strengthen mapping)? (3) tokens/sec estimates from measured anchors; (4) CUDA-vs-sparse division of labor and crossover.

---

## 1. Connectivity audit: the language model does NOT touch the procedural runtime

**Verdict: fully disconnected.** `RecurrentCellLanguageModel` grows text over the **native materialized-array kernels** (CPU AVX2/Rayon or CUDA), never over `NativeProceduralRuntime`. Exact chain:

1. `src/humanbrain/recurrent_language.py:12` — `from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig, RecurrentRace`. `grow()` constructs the circuit directly (`recurrent_language.py:56-59`): `RecurrentCellCircuit(token_cells=tokenizer.vocab_size, config=cfg.circuit)` and drives `circuit.grow_sequence(...)` (`recurrent_language.py:61`). There is no circuit abstraction to swap.
2. `RecurrentCellCircuit.__init__` (`recurrent.py:299-320`) binds `self._native_kernel = recurrent_kernel(...)`, which dispatches to `CudaRecurrentKernel` or `NativeRecurrentKernel` (`native.py:1634-1643`, `should_use_cuda(population.cell_count)`; tests confirm the `HUMANBRAIN_DEVICE=cpu/cuda` switch, `tests/test_cuda.py:280-285`). Both kernels operate on **materialized numpy/device arrays sized by `config.reservoir_cells`** — afferent rows `(token_cells, input_fanout)` are allocated in `__init__` (`recurrent.py:243-251`), recurrent weights `(reservoir_cells, recurrent_fanout)` at `recurrent.py:252-271`.
3. `recurrent.py` contains **zero references** to `procedural`; conversely `grep -rln NativeProceduralRuntime` hits only `benchmarks/scale_opt_cpu_throughput.py` and `tests/test_procedural_runtime.py`. Nothing connects the two engines except shared GLIF5 constants (`AllenGlif5Parameters`/`FittedGlif5Constants` feed both `_GlifConfig`s) and the same eight-field canonical cell-state model.

**Why they cannot be swapped today** (interface mismatches, all cited):

| Language-model requirement | Procedural runtime reality |
| --- | --- |
| `token_cells == tokenizer.vocab_size`; per-token afferent fan-out rows (`recurrent_language.py:33-34`, `recurrent.py:243-251`) | Address space is one uniform `uniform_no_self` population; other topologies fail before FFI (`procedural_runtime.py:116-119`). No reserved token/output GID block, no afferent structure |
| `observe(token_id)` injects afferent current and advances exactly `token_steps`=100 steps (`recurrent.py:652-665`) | `advance(step_count, events)` accepts caller-injected `ProceduralInputEvent`s (`procedural_runtime.py:561-596`) — token injection **is** expressible today as external events |
| `strengthen(target)` mutates learned synapses: ridge statistics / native three-factor / `SparseCellReadout.teach` + structural turnover (`recurrent.py:667-744`) | **No learning write-path.** Edges are procedurally expanded at fixed weight `±recurrent_weight_pa` by presynaptic sign; `EdgeCounts.mutable=0` (`procedural_runtime.py:623-629`) |
| `race()` picks the first output cell to spike from compiled GLIF5 resting-response curves over assembled output currents (`recurrent.py:332-346, 833-949`) | Returns raw `(step, gid)` spike records only; no output-cell concept, no per-output current assembly, no tie-break/exclusion semantics |
| `freeze_learning()` + `capture_cell_synapse_state()` checkpoints (`recurrent.py:348-396, 794-801`) | Snapshot is read-only introspection; fail-stop poisoning replaces checkpointing (`docs/procedural-runtime-gate3-slice.md`) |

One important nuance the connectivity check surfaced: **the dense path is no longer dense at execution time.** Native ABI 3 adds a quiescent frontier that skips zero-to-zero integration algebraically and bitwise-exactly (`docs/native-scaling-v2.md`), and it is what `RecurrentCellLanguageModel` already benefits from on CPU. So "dense python circuit" is a misnomer for the incumbent: the incumbent is a *materialized-topology* engine with sparse execution; the procedural runtime is an *O(1)-manifest* engine with sparse execution and no learning.

---

## 2. Gap analysis: what an adapter must provide at 1e6–1e8 configured cells

### 2.1 Architecture: three layers, ordered by how much kernel change they need

**L1 — Token drive (works with today's FFI).** Reserve no topology at all: encode token *t* as caller-injected `ProceduralInputEvent`s delivering `input_current_pa` to *t*'s chosen target GIDs at the token boundary (every `token_interval_ms/dt_ms = 100` steps). The external-event seam is validated, transactional, and proven in the throughput harness (`benchmarks/scale_opt_cpu_throughput.py`). Input side needs no kernel change.

**L2 — Host-side readout (small native additions).** Mirror `SparseCellReadout` in Python outside the runtime: after each `advance`, consume returned spikes, then `teach(target, winner, trace_decay=exp(-5.0/80.0), ...)` exactly as `recurrent.py:700-723` does, plus the periodic `turnover` cadence. Required new native seams:
- *targeted eligibility read*: `snapshot()` copies **all** residents through Python objects (`procedural_runtime.py:658-724`) — O(R) churn per token defeats sparsity. Need `read_state(gids[]) -> eligibility/voltage` FFI.
- *output-current assembly*: race inputs must be computable for the token GID block only.

**L3 — In-kernel mutable edges (the real fix).** Store one `f64` weight per **resident generated edge** (+8 B/active edge; at measured text-like 500–2,500 updated cells/token this is kilobytes) and apply local plasticity at delivery. `ridge` cannot come along: its sufficient statistic is a `(N+1)²` normal matrix (`recurrent.py:746-772`) — precisely the object that cannot scale; million-scale text growth must ship on local rules (`sparse_three_factor`/`three_factor` semantics), for which repo precedent exists (`docs/local-readout-learning.md`, ladder screens).

### 2.2 Eligibility/strengthen semantics — the hard part, mapped

Dense semantics (`recurrent.py:100, 292-295`): `eligibility_i` decays ×`exp(-dt/80 ms)` **every step**, is driven by presynaptic spikes, and is consumed instantaneously and globally by the readout rules. Three mapping problems:

1. **Lazy decay vs bit-exactness.** An event-driven kernel wants decay-on-read: store `(value, last_update_step)`, analytically decay when touched. But iterated multiplication ≠ `pow()` in float64 bits, and the runtime's eviction contract is *canonical exact zero* after real fixed-step updates ("Nonzero exponential tails therefore remain resident and updated until they underflow or cancel exactly", `procedural_runtime.py:7-14`; signed-zero canon at all seams per Gate-3 doc). Two honest options: (a) accept a documented, tested bit-deviation for eligibility only (it feeds learning, not dynamics); or (b) keep per-step iterated decay but skip *integration* of quiescent residents while decaying their traces in a cheap separate pass (trace pass is one multiply — the expensive part is GLIF evaluation, not decay). Option (b) preserves exactness and is the recommendation. Note `ScaleAccounting.WorkCounts` already reserves `analytic_catchups`/`analytic_step_equivalents` (`procedural_runtime.py:618-622`) — the schema anticipated lazy advancement.
2. **Delay-carrying credit.** `recurrent_delay_steps ≥ 1`: a spike cannot affect its target in the same step (Gate-3 contract). A strengthening decision at delivery needs the *presynaptic eligibility at emission time*. Today's `ProceduralInputEvent` carries only `current_pa` (`procedural_runtime.py:164-188`). Kernel change: capture `pre_eligibility` into natively generated events (or stamp emission step and look the source up at delivery — source may have been evicted, so capture-at-emission is safer).
3. **Competitive winner selection.** `three_factor` uses a global competitive margin over output currents (`recurrent.py:725-733`); `sparse_three_factor` restricts competition to a per-output fan-in (`SparseCellReadout`). Only the latter is event-mappable at scale — adopt it as the sole growth rule for the procedural path and pin its gates (`post_gate`, `activity_decay`, turnover interval) to the shipped values.

Also required for parity, not optional: host-side replication of the output race (compiled unit-response curves are pure functions of current — portable), ascending-GID tie-breaking to match candidate-order semantics, and a checkpoint codec (poison/fail-stop is not a persistence story).

### 2.3 What the procedural runtime uniquely contributes

The dense/frontier path scans **every configured cell once per token** (`median_full_state_scan_microseconds`: 374.55 µs @98 k → 1,807.55 µs @393 k, `native_scaling_v2.json`; scan cost ∝ N_cfg). The procedural runtime has **no N_cfg term**: the address space is metadata (measured 20,087 host bytes at `configured=1e6`, `artifacts/procedural_address_space_v1.json`), and throughput depends only on the resident/updated set. That asymptotic difference is the entire reason to build the adapter.

---

## 3. Throughput arithmetic from measured anchors

Anchors (all measured, committed):

- **A1 — CPU procedural:** ~36–47 M cell-steps/s flat across densities up to 100 k residents (e.g. 36.24 M @100 k full-touch, 47.36 M @1 k full-touch); **9.18 M cell-steps/s at 1 M full-touch**; payload **110–177 B/resident** at large resident counts (`logs/scale_opt_cpu_baseline_v1.json`: `cell_steps_per_second`, `payload_bytes_per_resident_cell`).
- **A2 — CPU dense frontier:** 843.58 tok/s @98,304 cfg (2.51% updated), 754.17 @131 k, 449.83 @393 k, **193.36 @1,048,576 cfg (max 544 updated cells/token = 0.052%)**; density ladder @131 k: 568/176/52/11.7 tok/s @1/5/20/100% updated; @1 M: 53.25/5.27/3.00/1.41 (`artifacts/native_scaling_v2.json`, `docs/native-scaling-v2.md`, `docs/benchmark-status.md:73-85`).
- **A3 — CUDA full-touch (RTX 4070 SUPER):** 263.26 tok/s @100 k, 144.03 @200 k, touching every enabled cell on all 100 steps (`docs/benchmark-status.md:88-91`); engineering artifact: training 153.9 tok/s @200 k full-touch (`artifacts/sparse_cuda_engineering_v1_optimized.json`); post-optimization merge claims +497 tps @100k, 1 M cells demonstrated (commit `112a83c`).
- **A4 — Spike load:** ~293 spikes/token measured at 128 cells (campaign measurement, GPU-lane context). Cross-checks: `cuda_scaling_v1.json` `mean_spikes_per_token` = 3.31 (tiny N); `large_scale_capability_validation_v1.json` @6,144 cells: 0.0749 spikes/cell/token → ≈460 spikes/token. Per-cell spike rate falls with N because fixed `input_fanout=24` touches a shrinking fraction of the pool; at ≥100 k cfg the better activity proxy is the frontier's *updated cells/token*: 544–2,466.

Convention: `dt=0.05 ms`, `token_interval=5 ms` → **100 integration steps/token** (matches the benchmark's v2-equivalent-tokens column and `"token_steps": 100` in artifacts).

### 3.1 Current procedural semantics (every resident integrates all 100 steps/token)

`tokens/s = CPS / (100 · R_resident)`:

| Residents | CPS (measured regime) | tokens/s |
| ---: | ---: | ---: |
| 128 | 40 M | 3,125 |
| 10,000 | 40 M | 40 |
| 100,000 | 40 M | 4 |
| 1,000,000 (full-touch) | 9.18 M | **0.092** (≈11 s/token) |

Even if the flat 40 M rate held at 1 M residents, that is 0.4 tok/s. **A 1 M-resident text run under current semantics cannot train** (a 10⁹-token curriculum ≈ 3 centuries). Residency is also sticky: the 80 ms eligibility tau reaches float64 underflow only after ~10⁶ steps, so touched cells stay resident near-indefinitely absent exact cancellation — steady-state residency under continuous text ratchets toward "all cells ever touched."

### 3.2 Event-proportional bound (what the adapter + quiescent-skip unlocks)

If work ∝ event-touched cells rather than residents: with A4, events/token ≈ 293 × (1 + fanout 8) ≈ 2,600 touches, plus ≤293 × 100 spiking-cell integration steps → order 3–5 ×10⁴ cell-step-equivalents/token → **~800–1,300 tok/s optimistic** at 40 M CPS. Mapping the *measured* large-N activity (frontier updated-set) onto procedural cost gives the defensible band:

| Updated cells/token (measured at…) | cell-steps/token | tokens/s @40 M CPS |
| ---: | ---: | ---: |
| 544 (@1 M cfg) | 54,400 | **735** |
| 937 (@393 k cfg) | 93,700 | 427 |
| 2,004 (@131 k cfg) | 200,400 | 200 |
| 2,466 (@98 k cfg) | 246,600 | 162 |

**Realistic procedural band at text-like activity: ~160–750 tok/s**, independent of configured N. Caveat: the updated-set measurements come from ≤64-token runs; long training-length tail accumulation could push residency (and hence cost) upward — the 100%-updated rows of the density ladder (1.41 tok/s @1 M) remain the honest worst case.

Event-side caveat: the baseline's `events_per_second` (825–2,638) reflects the benchmark's synthetic injection rate, not a delivery ceiling; queue capacity and per-event cost at text injection rates (~2,600 events/token × hundreds of tok/s) need a dedicated microbenchmark before the optimistic band is promised. Flagged as follow-up, not blocking the analysis.

### 3.3 The two headline answers

- **1 M-resident text run:** current semantics → 0.09 tok/s (unusable). With the L2/L3 adapter + quiescent-skip and text-like activity → **~700 tok/s class** (band 160–750), i.e. a ~4 orders-of-magnitude swing hinging entirely on whether non-event residents are stepped.
- **1e7 configured, sparse residency:** configured N is irrelevant to procedural cost (O(1) manifest, no scan) → same **160–750 tok/s** band at equal resident counts; memory trivial (manifest KBs + 100 k residents ≈ 11–18 MB at 110–177 B/resident; even 1 M residents ≈ 110–177 MB against 13 GB usable RAM). By contrast at 1e7 configured the frontier's scan alone costs ~46 ms/token (linear extrapolation of 1,807.55 µs @393 k) → **≤22 tok/s absolute ceiling**, and CUDA full-touch computes at ~2.9–5.0 tok/s (2.88–4.97 ×10⁹ cell-steps/s ÷ 10⁹). **At 1e7, the procedural path is not just the best option; it is the only option with headroom.**

---

## 4. Division of labor and the crossover point

Cost models: CUDA full-touch ∝ N_cfg (2.88–4.97 ×10⁹ cell-steps/s measured-class); dense frontier ∝ (scan(N_cfg) + 100·U); procedural ∝ 100·R_res only (40 M cell-steps/s).

| Regime | Owner | Measured/projected rate | Why |
| --- | --- | --- | --- |
| N_cfg ≤ ~200 k, any activity | **CUDA full-touch** | 144–497 tok/s | Complete training tooling (grow/strengthen/readout/checkpoints/LM); fastest per cell-step |
| 200 k < N_cfg ≤ ~4–6 M, activity ≲5% | **CPU dense frontier** (today) → procedural (after adapter) | 193–843 tok/s measured | Learning intact, bit-exact, already ships; scan tax grows with N_cfg |
| N_cfg ≥ ~1e7 or >VRAM | **CPU procedural** (after L2/L3 adapter) | 160–750 tok/s projected | Only architecture with O(1) configured-cost; RAM sized by residents |

**Crossover points:**

1. **Speed, dense↔procedural at equal text-like activity:** frontier 449.83 tok/s @393 k vs procedural-equivalent 427 tok/s @937 updated → parity at **N_cfg ≈ 4 ×10⁵**; at 1 M cfg procedural is ~3.8× ahead (193 vs 735). Crossover sits at **~4 ×10⁵–10⁶ configured cells**.
2. **Speed, CUDA-vs-procedural:** parity when `N_cfg/R_res ≈ 66–124` (from `40e6·N > CPS_CUDA·100·R` with CPS_CUDA = 2.63–4.97 ×10⁹). Because CUDA pays the configured tax on every token, **any workload sparser than ~1% of configured (at equal rates) belongs to the event-driven path on speed alone** — e.g. at N_cfg = 1e7, R_res = 1e5: procedural ~400 tok/s vs CUDA full-touch ~3–5 tok/s.
3. **Capability wall (the binding constraint):** CUDA requires the working set in VRAM (~9.9 GB free per `hardware-capacity-envelope.md`; measured persistent arrays 2,237 MiB @1 M cells on the CPU frontier configuration). Practical CUDA ceiling on this card ≈ **4–6 M configured cells**; beyond it CUDA is impossible without sharding that does not exist, while procedural configured-N is bounded only by uint64.

**Recommendation:** keep CUDA full-touch as the training-grade lane through the current 1 M target; build the L1+L2 adapter now (token injection works today; add targeted-state FFI + host-side `SparseCellReadout` mirror) to make the procedural runtime *measurable* on a real text gate at 1e6–1e7 configured; land L3 (resident-edge weights + emission-captured eligibility) to make it *learnable*. Do not port `ridge` — pin `sparse_three_factor` semantics as the million-scale growth rule. Acceptance gate for the adapter: reproduce the dense circuit's teacher-forced accuracy delta within noise on a fixed corpus at ≥1e6 configured, ≥100 tok/s.

## 5. Source index

- `src/humanbrain/recurrent_language.py:12,33-61,117-131` — LM↔circuit coupling, grow/complete flow
- `src/humanbrain/recurrent.py:190-320,652-744,833-949` — circuit init/kernels/observe/strengthen/race
- `src/humanbrain/native.py:1634-1643` — backend dispatch; ABI 3 frontier in `docs/native-scaling-v2.md`
- `src/humanbrain/procedural_runtime.py:7-14,98-159,164-188,561-656,623-629` — contract, config, events, advance, fixed-edge accounting
- `docs/procedural-runtime-gate3-slice.md` — fail-stop/exactness contract
- `logs/scale_opt_cpu_baseline_v1.json` — A1; `artifacts/native_scaling_v2.json` + `docs/benchmark-status.md:73-92` — A2, A3
- `artifacts/procedural_address_space_v1.json`, `docs/research/hardware-capacity-envelope.md` — O(1) manifest, VRAM budget
- `artifacts/cuda_scaling_v1.json`, `artifacts/large_scale_capability_validation_v1.json` — A4 cross-checks
