# Experiment level ladder: single cell → brain, dependency graph, and box-aware parallelization plan

Status: survey + planning doc. Read-only research pass over this worktree
(`codex/a013-cognition-design` @ `081a492`) and its sibling worktrees; answers
the program-level question *"have you run low and high level experiments —
single neuron → brain architecture levels?"* honestly, then derives what
blocks what and how the remaining work partitions across agents sharing one
box. No experiment was executed for this document.

**Naming collision warning:** `docs/typed-composition-a013-ladder-v1.md`
(sibling branch `codex/typed-ladder-a013`) uses **L1–L6 as arm labels** for
reservoir sizes inside one experiment family. The **L0–L5 below are organism
levels**, not those arms. Cross-references always say "level Ln" vs "arm Ln".

## 0. Answer to the level question, up front

- **L0–L1 (cell, microcircuit): YES — heavily, including this week.** Sealed
  a012 at N=128; three preregistered pilot matrices collected 2026-08-23
  (`readout_compare` 27 runs, `schedule_compare` 18 runs,
  `a013-ladder-sanity-v1`); `codebook_compare` running now. Cell-side:
  two pinned Allen human MTG fits with SHA-256 identity guards, native ABI
  bit-exactness vs the frozen serial kernel, and a train-only
  dynamics-regime calibration (rank 1.19 → 20.84 at 96 cells).
- **L2 (mesocircuit 1e3–1e5): PARTIAL.** Throughput and memory are measured
  to 131k–1M configured; *learning* outcomes exist only to 6,144 cells
  (local-rule scaling trend, developmental-gate diagnostics) plus exactly
  **one** canary run of the frozen 148-run large-scale capability grid
  (execution paused). No E/I-structure result exists at any scale.
- **L3 (region 1e6–1e7): CONSTRUCTION/THROUGHPUT ONLY.** 1M-cell builders,
  frontier throughput at 0.052%-updated activity, density-controlled curves.
  Zero learning above ~6k cells; the ≥1e7 throughput point required before
  any larger claim does not exist yet.
- **L4 (multi-region 1e8+): DESIGN + ADDRESSABILITY ONLY.** No resident
  simulation above 1M configured anywhere; the two-region design exists on
  paper only; shard determinism unbuilt.
- **L5 (full architecture 1e9+): ADDRESS SPACE ONLY, BY EXPLICIT CONTRACT.**
  The 100-billion milestone is defined as *addressable identities*, not
  simulated neurons; the 1e9/1e11 artifact proves configured=addressable with
  zero resident/updated/spiking cells.

So: the ladder's bottom two rungs carry real experimental weight; the middle
rung is half-built; the top three rungs are engineering evidence and
falsifiable plans, not biology-scale results — which matches the scale
contract's own refusal to sell configuration as simulation.

## 1. Level ladder

Columns: what EXISTS (named artifacts), what's MISSING, and the ONE decisive
missing experiment per level.

### L0 — single-cell dynamics validation

| | |
|---|---|
| Exists | Pinned human MTG GLIF5 fit `523000114` + GLIF3 fit `503612028` with identity guards (`HUMAN_GLIF5_IDENTITY.config_sha256`, human_cell_identity.py:39-52; verification table in `a013-cell-construction-variants.md`). Native ABI bit-exact vs frozen serial kernel over 64-token trajectories (`docs/benchmark-status.md` §Native simulation scaling; native_scaling_protocol_v1/v2). Train-only dynamics calibration over 11 regimes → `artifacts/dynamics_calibration_train_v1.json`: selected sparse/slow/short-trace regime moved effective rank 1.19→20.84 at 96 cells, firing 107.1→12.4 Hz. Corrected trace-horizon arithmetic (`a013-cognitive-properties.md`: membrane R·C ≈78.6 ms → ≈59 s horizon; eligibility 80 ms → ≈60 s; ASC 100/33.3 ms). |
| Missing | Kernel-vs-fit validation against the Allen specimens' own electrophysiology (f-I curve, voltage response, threshold adaptation) — the fits are trusted via hash, never replayed through our dt=0.05 ms forward-Euler kernel. No proof that GLIF3 is exactly expressible inside the GLIF5-shaped GlifConfig (load-bearing for variant V1's cost). No inhibitory human fit exists in `data/allen/` at all (both specimens spiny/excitatory-type). |
| Decisive missing experiment | **Specimen-replay validation**: drive the native kernel with the pinned GLIF5 fit's own stimulus protocol and compare emitted f-I/voltage/adaptation response against the Allen target responses (and the GLIF3-as-GLIF5 zeroed-threshold-adaptation equivalence check). Without it, every higher level inherits an unvalidated cell; with it, V1 dual-specimen costing collapses from open question to arithmetic. |

### L1 — microcircuit (10–100s of cells, 128-cell bridge) with plasticity

| | |
|---|---|
| Exists | Sealed a012 @N=128 ridge (`hb-a012-once-20260822`, `typed-composition-a012-v1-results.md`): TRAIN replay 40/42, cases 120/126; calibration replay **0/18**, cases 7/54; margins min +56 pA / median 726 pA; state separation ≈1e-3. Pilot `experiments/readout_compare/FINDINGS.md` (@`410c720`; 27 runs + determinism recheck): ridge capacity monotone in N (train 0.146→0.563→0.981), both local rules **N-flat and collapsed** to the degenerate constant program `variable-0`; genuine transfer ≤4/54 for every rule at every N; feature-collision thesis confirmed. `experiments/schedule_compare/FINDINGS.md` (@`0cb7359`; 18 runs): presentation order **structurally inert** under ridge — teaching-matrix T3 marked dead for closed-form readouts; REPORT-GAP records that frozen `perm_appendix.json` orders have no loading path. `experiments/a013-ladder-sanity-v1/` (3 arms). In flight on this box right now: `experiments/codebook_compare/` (explicit/random/similarity × N∈{128,512} × seeds 7-9) + `codebook_probe`. Earlier generation: typed-program public TRAIN v1 (memorization 3/3, 9/9) and DEV v1 0/2 (retired); Layer-0 plumbing smoke (`docs/cell-program-layer0.md`). |
| Missing | Local-rule revival (blocked: shipped constants collapse — see dependency D2). Unexecuted teaching-matrix rows: T6 replay consolidation, T5a error-gating, T5b errorless, T4 supervision mixing, T1 dose-response knee, T2 diversity×representation interaction. V3a `inhibitory_fraction` sweep (knob exists, default 0.2, never swept). V2 parameter jitter. Two-region E/P pilot (design-only, `a013-regioned-architecture.md` §6). |
| Decisive missing experiment | **Constant-collapse repair, then revive sparse three-factor as an instrument** (rerun the schedule axis under the online local rule with byte-frozen permutation orders). Every order-sensitive and interaction experiment (T2, T3-local, T5b, the regioned decisive experiment, cell-LM v1) pins the local rule; until its degenerate-constant failure is fixed, L1 has no functioning biological learner, only ridge. |

### L2 — mesocircuit (1e3–1e5 cells) with E/I structure

| | |
|---|---|
| Exists | Predictive local learning to 6,144 active cells (`local_learning_scaling_v1`: teacher-forced 3.91%@96 → 10.16%@6144, trails ridge; linear memory). Dataset-free 32-example smoke at **98,304** active cells (`local_learning_large_scale_v1`: 9.34 updates/s, 211.70 MiB, avoids a 72 GiB normal matrix). Developmental gate v2 at 384/1536/6144 (`development_validation_v2.json`, `development_scale_6144_v2.json`): numeric primary FAILED, positive diagnostics (staged rank 2.43→5.41; activity-selection+homeostasis help circuit health). Frozen 148-run large-scale capability protocol v1 with exactly **one** 6,144-cell canary recorded (`17.06%` teacher-forced / 1.04% free-running / 94.14% responsive / 14.97 Hz / 1,740 updates), execution paused pending the simple-SWE curriculum. Throughput: CPU frontier 843.58 tok/s @98k and 754.17 @131k natural activity; controlled-density exactness curves; CUDA full-touch 263.26 tok/s @100k, 144.03 @200k on RTX 4070 SUPER (`cuda_scaling_v1`). Topology diagnostics @96–192 cells (`topology_validation_v2.json`: layered sheet meets criterion at 96, nothing beats uniform at 192). |
| Missing | Any plasticity/predictive outcome above 6,144 cells with data-bearing runs (the 148-run grid is 1/148 done). **Any E/I-structure measurement anywhere** — `inhibitory_fraction=0.2` is a wired default whose effect has never been preregistered or swept. Heterogeneity variants (V1/V2/V4/V5 of `a013-cell-construction-variants.md`) unimplemented. |
| Decisive missing experiment | **Execute (or trim-and-refreeze) the registered large-scale capability grid** at 6,144–131,072 cells with five paired seeds and activity/readout-density controls — the first scale where "does more substrate improve prediction" is testable rather than assumed — paired with the preregistered **V3a inhibitory_fraction sweep** as the E/I-structure axis the level definition demands. |

### L3 — region (1e6–1e7 cells), sparse event-driven

| | |
|---|---|
| Exists | 1M-configured CPU frontier 193.36 tok/s natural (0.052% updated), density-controlled 53.25/5.27/3.00/1.41 tok/s at 1/5/20/100% updated (`native_scaling_v2`, `native_activity_density_v1`; 2.24 GiB persistent arrays at 1M). Structured topology builders at 1M: six million edges in 15.74–48.43 s, near-linear exponents, no self/duplicate edges (`structured_topology_construction_v3`). Measured memory anchors 60/132/181.04 B/cell with explicit extrapolation flags above 1M (`a013-regioned-architecture.md` §4). Constant-memory resource-gate behavior demonstrated at 1e6 **and** 1e11 configured (procedural runtime slice). Sequenced-but-unmeasured CPU/GPU optimizer threads exist as worktrees (`human-brain-scale-opt-cpu/-gpu`). |
| Missing | **No throughput measurement at ≥1e7 configured** — stated outright as required before any row above 1M "is treated as more than assumption." No learning of any kind above 1M. The E/P two-region split is design-only; no region-level runtime object exists. |
| Decisive missing experiment | **The ≥1e7-configured CPU-optimizer throughput point** (first point of the sequenced scale-opt work), plus extending the dataset-free 1M local-learning smoke to one data-bearing predictive run — together they convert the L3 row from extrapolation to measurement. |

### L4 — multi-region brain (1e8+)

| | |
|---|---|
| Exists | Nothing measured. Design capital: `100-billion-cell-architecture.md` staged ordered path (steps 1–8: counters → 64-bit counter edge recipe → source-first execution → resident-tile/due-event registries → mutable overlays → 100-billion addressability → two-shard determinism → density-qualified speed curves); memory tables (1e8 ⇒ 13.82 TB device layout — three orders beyond this box); Gate-3 event-runtime correctness slice with bounded O(working-set) storage. |
| Missing | Resident state at 1e7–1e8; tile/registry residency control; two-shard CPU determinism; any multi-region wiring (only the 2-region E/P design exists); any capability claim (contractually out of scope until step 8). |
| Decisive missing experiment | **Resident-tile + due-event registry slice, then two-shard CPU determinism** (ordered-path steps 4 and 7) at the largest resident floor this box affords (≈1e7–1e8 configured ≈ 0.6–6 GB at the 60 B/cell mutable floor). This establishes the shard/residency scaling law every 1e9+ sentence rests on, and is the only L4-grade experiment runnable here at all. |

### L5 — full architecture (1e9+ configured)

| | |
|---|---|
| Exists | Address-space artifact: `configured = addressable = 1e9/1e11` identity/query evidence including GIDs straddling 2^32 (procedural_address_space v2, protocol SHA-256 `cb3c6640…`; strictly address/query evidence, resident=updated=spiking=0). Counter-based base-edge recipe with tested worst-case row bounds (≤129F+1 candidate examinations, O(F) memory). `procedural_runtime_gate3_slice` correctness suite (ordering, delays, quiescence, underflow, duplicate-key, capacity fail-stop). |
| Missing | Everything resident/updated/spiking at 1e9 (60 TB mutable floor vs a 24 GB-class box); density-qualified speed curves; all capability gates — deliberately deferred by the scale contract. |
| Decisive missing experiment | The doc's own gate: **pass 100-billion addressability + dormant-range invariance on one machine** (step 6), then two-shard determinism (step 7). Capability experiments at this level are pre-committed as out of scope until step 8's speed curves exist. |

## 2. Dependency graph

Edges are stated as `blocked → blocker`. "Verdict" means a committed
FINDINGS/preregistered-gate outcome, not an opinion.

```
                            ┌── D6 GLIF3-in-GLIF5 proof ──┐
D0 dynamics specimen-replay │                             ▼
   (L0 decisive) ───────────┴──▶ D4 hetero placement doc ──▶ D5 tiering impl
                                     │                        (per-cell dispatch,
                                     │                         ckpt format v5)
                                     ▼
                          V1/V2/V5 construction variants

D1 codebook verdict ◀── RUNNING (codebook_compare, this box)
   └─▶ representation arm freeze ─▶ cell-LM v1 freeze
                                 ─▶ regioned §6 arm selection
                                 ─▶ T2 interaction cells

D2 constant derivation (collapse fix)
   └─▶ local-rule revival ─▶ T3 re-test under local rule (needs D3)
                           ─▶ T5b errorless
                           ─▶ T2 interaction (sparse column)
                           ─▶ regioned decisive exp (pins sparse rule)
                           ─▶ cell-LM v1 (ridge retired at scale ⇒ local-only)

D3 perm_appendix loader (REPORT-GAP) ─▶ any order-sensitive rerun
                                          with byte-equal frozen orders

D6 T6 replay outcome ─▶ capacity-vs-representation partition reading
                        (sharpens, does not gate, D1/D2 priorities)

D7 resident-tile registries (path step 4) ─▶ two-shard determinism (step 7)
    ─▶ ≥1e7-resident claims (L4) ─▶ density speed curves (step 8)
    ─▶ any future capability-scaling preregistration

D8 ≥1e7 CPU-optimizer point ─▶ validates regioned §4 rows >1M
    ─▶ feeds typed-ladder {PARAM} arm-size finalization (needs
       "measured resident-cost curve" before freeze)

D9 simple-SWE curriculum build ─▶ unpauses 148-run large-scale grid
                                ─▶ SWE rung opening preregistration execution

D10 inhibitory human fit licensing (EXTERNAL) ─▶ true inhibitory cell models;
    until then V3a sign-structure sweep is the declared stand-in

D11 Gutenberg mirror fetch + SHA-256 ─▶ cell-LM v1 freeze (failure BLOCKS
    freeze by design; no silent corpus switch)
```

Named blocking pairs (from the task context, resolved to artifacts):

1. **Codebook verdict blocks arm freeze** — `experiments/codebook_compare/`
   FINDINGS (in flight; 4/18 arms landed as of 11:51 today after a first
   aborted attempt with a wrong relative path, exit=2 logged in
   `experiments/_run_log.txt`) → freezes the representation arm in cell-LM v1
   (open item: "Representation doc must land before freeze"), regioned §6
   arms 2–4, and the T2 interaction columns.
2. **Constant derivation blocks local-rule revival** — `readout_compare`
   FINDINGS verdict #1 (degenerate constant programs; nominal transfer wins
   exposed as identity-floor + luck) → the sparse three-factor rule cannot
   serve as an instrument anywhere until completions can derive the correct
   constant/identity program instead of collapsing to `variable-0`.
3. **Hetero placement doc blocks tiering implementation** — the per-cell
   assignment manifest (which jitter/specimen/class goes to which population
   or region; seeded, hashed, checkpoint-stable) must be authored before
   per-cell model-type dispatch (V1's kernel requirement) or per-cell
   parameter arrays (V2) are implemented, because the manifest decides the
   tier schema and checkpoint format v5 layout.

Soft sequencing (not hard blocks): T6/T5a/T4/T1 are runnable today; their
interpretation sharpens once D2 lands. V3a is runnable today (existing knob).
A7/A8-class engineering (below) is independent of all science threads except
the src-freeze rule (§4).

## 3. Parallelization plan (shared-box aware)

Box reality: one Windows host, RTX 4070 SUPER, other agents actively using it
(today: a failed-then-relaunched `codebook_compare` matrix; during
`schedule_compare` another agent modified the harness mid-batch — recorded in
its FINDINGS as a concurrent-worktree note). **Contention baseline for every
thread: the box may be used by other agents at any moment; measure around
noise — use medians over ≥3 repeats, report spread, never trust single-run
numbers; echo `git_head` per run (schema already carries it) and re-verify
byte-stability of `src/humanbrain` across a batch before interpreting.**

Measured cost anchors: ladder-harness run ≈ 5–11 s/arm (readout_compare: 29
passes / 294 s; schedule_compare: 18 runs / 162 s); harness caps 2 GiB RAM
per process, N ≤ 4096, CPU-only; 1M-cell native point ≈ 2.24 GiB persistent;
Gemma 12B Q4_0 GGUF ≈ 8 GB VRAM; CUDA measured layouts < 1 GB VRAM.

### Wave 1 — start now, mutually compatible (no cross-blockers)

| Thread | Work | Depends on | Resource budget | Contention note |
|---|---|---|---|---|
| A1 | Constant-collapse repair ("constant derivation"): red tests + fix so local-rule completions derive the correct constant/identity program | nothing | S code change (readout/completion path); validation batch ≤5 CPU-min, ≤2 GiB, 0 VRAM | Touches `recurrent.py`/lib.rs — see src-freeze rule; lands BEFORE A7 starts editing natives |
| A2 | T6 replay consolidation dev-gate (error-prioritized pre-freeze pass) | nothing (harness seam exists) | ~5–10 CPU-min for the arm set; ≤2 GiB; 0 VRAM | Serial under cpu-bench.lock with A3/A4; deterministic outputs immune to box noise, wall_s is not |
| A3 | T4 supervision-mixing scheduler + T1 dose-response (21⊂42⊂84⊂168 cohorts) | nothing | ≈10-line scheduler + ~15–20 arms ≈ 5–10 CPU-min; ≤2 GiB | Same lock queue as A2; fresh cohorts are cheap, generate inside the locked window |
| A4 | V3a `inhibitory_fraction` preregistered sweep (existing knob) | nothing | Sweep spec authoring + ≤10 CPU-min runs; ≤2 GiB | First actual E/I-structure data; keep arms ≤4096 (python circuit limit) |
| A5 | Hetero placement doc (D4 blocker): per-cell assignment manifest design for V2 jitter / V5 class-to-region maps, seed+hash discipline | nothing (reads A6 output for V1 feasibility line) | 0 compute; docs only | Fully parallel with everything; no locks |
| A6 | GLIF3-in-GLIF5 expressibility proof (zeroed threshold-adaptation coefficients, ASC channel compatibility) | nothing | Minutes of numeric checks | Non-timing; no lock |
| A7 | Resident-tile + due-event registry slice (ordered path step 4) | nothing code-wise; **waits for A1's native edits to land** | Dev + unit tests; non-timing | Real merge-conflict hazard with A1 in `_native_src/src/` — sequence, don't interleave |
| A8 | cell-LM v1 freeze prep *except* the arm choice: corpus mirror fetch + SHA-256, train-only tokenizer/BPE vocab sweep, pure-numpy B0/B1 baseline harness | D11 (fetch success) | One network fetch + ~30–60 CPU-min sweeps; no GPU (torch would need an extra — stay numpy per pyproject) | Tokenizer sweep is not timing-sensitive: lock-free, but drop process priority while any bench lock is held |

Wave-1 conflicts: A1↔A7 (same native files — serialize); A2/A3/A4 (same
harness + `experiments/<id>/` namespaces — serialize under cpu-bench.lock);
everything else overlaps freely.

### Wave 2 — after the codebook verdict (D1) and A1 land

| Thread | Work | Depends on | Resource budget | Contention note |
|---|---|---|---|---|
| B1 | Arm freeze everywhere: cell-LM v1 protocol hashes; regioned §6 arm selection; revived-schedule spec | D1, A1, D11, D3 | 0 compute; JSON/docs | None |
| B2 | Implement perm_appendix loader (red test #3 from the schedule spec) + rerun schedule axis under revived sparse rule, frozen orders | A1, B1 | Harness patch + 18 arms ≈ 5 CPU-min | Lock queue |
| B3 | Two-region E/P implementation pilot @ N_total=512 (Gate-3 exactness cases from regioned §3) | D1, A1 | M-sized code; red tests first; smoke runs minutes | Another `_native_src` editor — after A7 rebases |
| B4 | T2 diversity×representation interaction cells | D1, A1 | ≤9 cells of the matrix ≈ 10 CPU-min | Lock queue |
| B5 | ≥1e7-configured CPU-optimizer FIRST point (scale-opt-cpu worktree) | none scientifically; long box hold | ~1.3–4 GiB RAM (132 B/cell layout), est. 1–4 h CPU — measure the first point before scheduling the rest | Long cpu-bench.lock hold; schedule idle/overnight; medians impossible for one point — repeat key sub-measurements ≥3× |
| B6 | Resume-or-refreeze decision for the 148-run large-scale grid (trim vs continue) | D9 progress | Decision only | Grid itself is Wave 3 |

### Wave 3 — capability executions (big, exclusive slots)

| Thread | Work | Depends on | Resource budget | Contention note |
|---|---|---|---|---|
| C1 | Regioned decisive experiment: 5-arm paired-seed batch (§6) | B1, B2, B3 | 15–25 python-scale runs ≈ 10–15 CPU-min; ≤2 GiB | Lock queue |
| C2 | Large-scale capability grid execution (registered estimate 12–20 h) | B6, D9 | Hours of exclusive-ish CPU; ≤~3 GiB/run at 131k | Longest cpu-bench.lock hold in the plan; announce max_minutes honestly |
| C3 | cell-LM v1 execution under run-hold authorization | B1 | Declared cell-side budget at freeze; B2 arms ≤30 min/GPU-arm if GPU chosen — prefer CPU-equivalent | If any arm goes GPU: gpu.lock |
| C4 | CUDA sparse-engineering + Gemma-tool-canary reruns | independent | <1 GB VRAM measured / ~8 GB VRAM for Gemma; minutes–hours | gpu.lock exclusive |
| C5 | Two-shard CPU determinism slice (ordered path step 7) | A7 | Dev + medium runs; RAM scales with resident floor | Coordinates with B5/C2 for quiet windows |

### Never-concurrent pairs

1. **GPU-exclusive vs CPU timing-sensitive — mutual exclusion, always.**
   Anything under gpu.lock (cuda_scaling, sparse_cuda_engineering,
   gemma_tool_canary, any GPU B2 arm) vs anything under cpu-bench.lock
   (native_scaling, simulation_speed, native_activity_density, ladder
   batches whose cost estimates matter): GPU driver/host activity perturbs
   CPU wall-clock, and hour-long CPU benches starve GPU responsiveness.
   Neither lock may be held while the other is requested — wait, don't stack.
2. **Two writers to the same `experiments/<id>/` namespace** — pointless and
   fail-closed anyway (never-overwrite rule); pick disjoint experiment ids.
3. **Source edits vs running batches** (the 2026-08-23 11:34 incident):
   never modify `src/humanbrain/**` or `benchmarks/*.py` while any batch
   holds a lock; land edits as commits and let in-flight runs finish on their
   recorded `git_head`.
4. **Gemma llama.cpp server vs any other GPU process** — 8 GB VRAM leaves no
   room; gpu.lock covers it.
5. **Heavy lock-free work (full pytest, corpus tokenization sweeps) vs any
   held bench lock** — allowed but only at below-normal priority; if you
   can't set priority, defer.

### Lock discipline (proposal)

Directory: `benchmarks/locks/`. Two lock classes, mutually exclusive:

- `benchmarks/locks/gpu.lock` — any process touching the GPU (CUDA benches,
  llama.cpp/Gemma, GPU neural-LM arms).
- `benchmarks/locks/cpu-bench.lock` — any wall-clock-sensitive CPU
  benchmark (scaling curves, speed gates, cost-estimate ladder batches).

File format (JSON, created atomically via exclusive-create so double-acquire
fails loudly):

```json
{
  "lock": "gpu",
  "owner_agent": "<agent-name>",
  "pid": 12345,
  "purpose": "cuda_scaling_protocol_v1 full curve",
  "started_utc": "2026-08-23T12:00:00Z",
  "max_minutes": 30
}
```

Rules:

1. Acquire = create-new exclusively. On collision: read the holder; if the
   PID is dead AND age > 2×max_minutes, rename to `<name>.stale-<ts>` and
   take the lock (log the theft in the run notes). Otherwise wait and poll.
2. gpu.lock and cpu-bench.lock refuse concurrent holds (rule pair 1 above).
3. Lock-free work needs no file, but must honor pairs 2/3/5.
4. Measurement hygiene while sharing: medians over ≥3 repeats, report
   min/median/spread, never gate on single-run numbers; verify
   `curriculum_sha256`/`git_head` echoes; treat `wall_s` as noise-prone
   always (the schema already excludes it from determinism).
5. Release = delete your own lock file; a crashed holder's staleness clause
   (rule 1) is the only legitimate takeover path.

This costs one directory and a ten-line helper; it converts today's
implicit "hope nobody is benchmarking" into an auditable convention.

## 4. Summary judgment

Levels L0–L1 are experimentally real and current; L2 is half-built (throughput
rich, learning poor, E/I empty); L3–L5 are honest engineering evidence plus
pre-registered gates, not biological results — exactly as the scale contract
demands. The three keystone blockers (codebook verdict, constant derivation,
hetero placement doc) sit upstream of nearly every next decisive experiment,
all three are cheap, and two of them are unblocked *right now*. Waves 1–2
above can run three-to-four agents wide on this box without ever stacking a
timing-sensitive CPU bench against a GPU job, provided the two-lock
discipline is adopted before the next long hold (C2's 12–20 h grid being the
one that most needs it).
