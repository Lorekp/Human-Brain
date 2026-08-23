# Procedural adapter spike — L1/L2 (roadmap R5 prerequisite)

**Date:** 2026-08-23
**Branch:** `codex/scale-opt-cpu` (worktree `human-brain-scale-opt-cpu`)
**Status:** executed spike. L1 verified live on this branch; L2 measured live at 2^20 configured cells; no native code changed. Scripts: `l1_token_injection.py`, `l2_eligibility_read_cost.py`; raw numbers: `l2_eligibility_read_cost.json`. All runs CPU-only, each script <60 s.
**Context:** `logs/text_scale_cpu_findings.md` §2 defines the three-layer adapter. This spike answers its two gating questions: does token injection work today, and is the existing snapshot FFI an eligible-read seam at scale.

---

## 1. Layer 1 — token injection via caller-created `ProceduralInputEvent`s

**Verdict: VALIDATED. Works today, zero kernel changes, demonstrated at 1e6 configured cells.**

Encoding: token *t* → one event `(due_step=boundary, target_gid=t, source_gid=t, edge_slot=0, current_pa=150_000)` passed to `advance(step_count=100, events)` at every token boundary (`token_interval_ms/dt_ms = 100` steps). Working example (from `l1_token_injection.py`, abridged):

```python
config = ProceduralRuntimeConfig(
    manifest=TopologyManifest(cell_count=1_048_576, fanout=8,
                              topology="uniform_no_self", seed=71),
    resident_capacity=4_096, event_capacity=4_096, spike_capacity=8_192,
    interval_distinct_capacity=4_096,
    recurrent_weight_pa=325.0, recurrent_delay_steps=1,
)
with NativeProceduralRuntime(config) as runtime:
    runtime.advance(100)                      # settle to step 100
    drive = ProceduralInputEvent(due_step=runtime._current_step,
                                 target_gid=token_gid, source_gid=token_gid,
                                 edge_slot=0, current_pa=150_000.0)
    result = runtime.advance(100, [drive])    # one token window
```

Observed evidence (asserted by the script, all passing):

- Empty advance over 2^20 configured cells: 0 spikes, 0 residents, **`configured_ids_scanned = 0`** — configured N stays metadata while the token path works.
- Injected token: target cell spikes inside its window; accounting shows `delivered_events = 9` (1 injected + 8 natively generated), `generated_edges = 8 == fanout`; 8 downstream cells become resident with nonzero synaptic current; target's own eligibility trace ≈ 0.94 (= +1.0 on spike, then ×exp(-dt/80 ms) per step).
- A distinct token fires on its own boundary; re-injecting the same token later re-fires it (distinct `due_step` keeps canonical keys unique).
- Identically-seeded runtimes produce bit-identical spike traces.
- Duplicate canonical key / stale `due_step` / out-of-range gids are rejected in Python before FFI (`procedural_runtime.py:_event_batch`).

**L1 caveat found by measurement (new):** Python-side event construction costs ~0.96 µs/event. The text-like drive (~2,600 events/token, per findings §3.2) marshals in **~2.5 ms**, and `advance(1, batch)` end-to-end is ~4.4–4.9 ms — a ~200–400 tok/s ceiling on the *host* side alone, independent of kernel speed. Not blocking L1 correctness; must be engineered away (numpy-view batching of `_NativeEvent` arrays, or a native token-drive seam) before promising the 735 tok/s band.

## 2. Layer 2 — eligibility-read cost via the existing snapshot FFI

**Verdict: INVALIDATED as a per-token read seam. Cost is O(all residents), independent of K.**

The only state-read seam today is `snapshot()` (`procedural_runtime.py:_read_snapshot` → `humanbrain_procedural_runtime_snapshot`): there is no GID filter parameter; native merges `residents`+`staged`, copies **every** resident record, and Python converts every record into a frozen dataclass.

Measured at N_cfg = 2^20, K = 16 requested eligibilities (median wall time; full table in `l2_eligibility_read_cost.json`; three runs within ±8%):

| residents | counts() µs | raw FFI copy µs | full `snapshot()` µs | ns/resident | extract K=16 µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,576 | 0.9 | 5.1 | 2,006 | 1,273 | 0.5 |
| 10,568 | 0.8 | 24.0 | 14,522 | 1,374 | 0.4 |
| 50,539 | 1.6 | 906 | 80,256 | 1,588 | 0.5 |
| 100,528 | 0.9 | 1,278 | 161,348 | 1,605 | 0.5 |

Reading eligibility for 16 hot cells at 100 k residents costs **161 ms/token** — 26× over the entire 6.25 ms/token budget at the 160 tok/s floor and 119× the 1.36 ms budget at 735 tok/s. Linear fit projects **~1.6 s/token at 1e6 residents**. Decomposition: the actual read is free (0.5 µs flat); ~99% of cost is Python dataclass conversion (~1.6 µs/resident) with the raw native copy at only ~13 ns/resident. Conclusion matches the code reading: a snapshot-per-token mirror is impossible at text scale; a targeted read is required.

### 2.1 What a host-side mirror needs (and what is already cheap enough)

`SparseCellReadout` binds a dense `reservoir_eligibility` array sized by configured cells (8 MB @1e6, 80 MB @1e7 — affordable). Mirror maintenance measured at 1e6:

- persistent buffer scatter+reset of R=100 k touched slots: **147 µs** ✔
- fresh `np.zeros(1e6)` per token (naive reset): **14 µs** ✔

So the mirror itself is viable if per-token population uses targeted reads/writes of only the touched set (winner/target fan-ins ≤ 2·candidate_count sources). That is exactly the missing seam:

### 2.2 Minimal C-ABI extension proposal (spec only — NOT implemented here)

One additive export in `src/humanbrain/_native_src/src/procedural.rs`, immediately after `humanbrain_procedural_runtime_snapshot` (ends ~line 1787):

```rust
/// Targeted resident-state read. For each gid: binary-search `residents`
/// (ascending-GID Vec<CellState>) then `staged`; staged wins if present.
/// Writes exactly gid_count ProceduralResidentRecord values to `out` in input
/// order -- canonical zeros when the cell is not resident -- and 1/0 presence
/// flags to `found`. O(K log R); no allocation; touches nothing else.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_read_cells(
    runtime: *const ProceduralRuntime,
    gids: *const u64,
    gid_count: usize,
    out: *mut ProceduralResidentRecord,   // same repr(C) layout as snapshot
    found: *mut u8,
) -> u32                                // PROCEDURAL_OK / _INVALID / _POISONED
```

Placement and binding:

- Rust: new export beside the existing snapshot export; implementation is two `binary_search_by_key` calls per gid plus `CellState::record()` copies (~30 lines, reuses existing types; no new status codes needed).
- Python: declare argtypes in `_procedural_library()` next to the snapshot binding (`procedural_runtime.py` ~line 340–350); add `NativeProceduralRuntime.read_cells(gids: Sequence[int]) -> tuple[ProceduralResidentState | None, ...]` near `snapshot()` (line 726).

Contract notes for review: output order mirrors request order (no implicit sort); absent-cell records carry canonical positive zeros so the Gate-3 signed-zero canon holds at the seam; poisoned runtimes fail closed with `PROCEDURAL_POISONED`; additive symbol, no change to `humanbrain_kernel_abi_version` semantics required (version bump decision belongs to the reviewed change). Alternatives considered: (a) GID-filter parameter on `snapshot` — rejected, muddies the all-state checkpoint contract and capacity semantics; (b) returning only `(gid, eligibility)` pairs — smaller payload but breaks the one-canonical-record-layout principle for negligible gain (72 B/record × K≪R).

Expected effect: per-token read cost drops from O(R)·1.6 µs to O(K log R)·~50 ns-class, i.e. microseconds at K≈10²–10³.

### 2.3 Growth rule pinned for this scale

**`sparse_three_factor` is the only supported readout rule on the procedural path at ≥1e6 configured.** `ridge` cannot come along: its sufficient statistic is the `(N+1)²` normal matrix (`recurrent.py:746-772`) — the exact O(N²)-memory object the procedural runtime exists to avoid (at 1e6 that is ~8 TB). Plain `three_factor`'s global competitive margin also has no event-mappable form at scale. Pin `SparseCellReadout` gates to shipped values (`post_gate=1.0`, `trace_decay=exp(-5/80)`, activity decay, structural turnover cadence) as `recurrent.py:700-723` does.

## 3. Layer 3 scope statement — emission-captured pre-eligibility (design sketch only)

L3 adds mutable resident-edge weights (+8 B/resident generated edge) with local plasticity applied at delivery. The prerequisite semantics gap: with `recurrent_delay_steps ≥ 1`, a strengthening decision at delivery needs the source's eligibility **at emission time**, not at delivery time. Sketch:

- Extend the natively generated event record with one captured field, `pre_eligibility: f64`, stamped when the source spikes and its base edges are expanded (capture-at-emission). At delivery, plasticity consumes `pre_eligibility` directly.
- Rejected alternative: stamping only the emission step and looking the source up at delivery — the source may have been evicted to canonical zero by then, silently corrupting credit.
- Cost: +8 B per queued event (bounded by `event_capacity`, kilobytes at text rates); no new per-resident storage until L3's weight store lands.
- This requires the reviewed native change (record layout + delivery-path plasticity) and is explicitly out of scope for this spike. The L2 `read_cells` seam above is independent of and prior to it.

Also required for parity eventually, not in this spike: host-side output race over compiled unit-response curves (pure function of assembled current), ascending-GID tie-break, and a persistence story (fail-stop poisoning is not a checkpoint codec).

## 4. Recommended minimal viable experiment: text-grow over the procedural runtime at 1e6 configured

Goal: make the procedural runtime *measurable* on a real text gate at N_cfg = 1,048,576 (accuracy parity is explicitly not the gate; that is the post-L3 acceptance in findings §4). Three phases, each independently shippable:

- **Phase A — L1 activity gate (works today; ~1 day).** Character-level corpus (small vocab V≤128). Map token id → E=8 spread drive GIDs; inject 150 k pA at every 100-step boundary; run ≥1,000 tokens. Measure: tok/s end-to-end, residency trajectory (ratchet check against the 80 ms tau analysis), spikes/token, updated-set size, event-queue peak. Validates the §3.2 event-rate caveat under real injection and pins the empirical activity number the throughput band hangs on. No learning, no L2.
- **Phase B — L2 growth parity at small cfg (needs `read_cells`; the reviewed native change).** Same harness at cfg ≤65,536 using `NativeDenseProceduralReference.materialized` as oracle: host mirror = persistent dense eligibility buffer (§2.1) populated via `read_cells` on winner/target fan-ins; `SparseCellReadout.teach` + turnover exactly as `recurrent.py:700-723`; race mirrored host-side with ascending-GID tie-break. Gate: teacher-forced accuracy delta vs the dense circuit within noise on a fixed corpus.
- **Phase C — scale-up.** Phase B harness at 2^20 configured, residency left sparse by construction; report tok/s against the 160–735 band and the Phase-A activity anchor. Overflow behavior (resident capacity) is fail-stop by contract and is itself a reported metric.

Sequencing rationale: Phase A de-risks the only unmeasured throughput unknown (injection rate ceiling) without any native change; B proves learning parity where the oracle still exists; C inherits both. The `read_cells` FFI (§2.2) is the single native prerequisite of B/C.

## 5. Source index

- `src/humanbrain/procedural_runtime.py:164-188,440-475,554-656,658-730` — events, pre-FFI validation, advance, `_read_snapshot`/`snapshot`
- `src/humanbrain/_native_src/src/procedural.rs:652-664,725-727,586-590,1654-1787` — `ProceduralRuntime` (sorted `residents`+`staged`), `total_residents`, eligibility update (+1.0 own-spike, ×decay per step), C-ABI exports incl. unfiltered snapshot
- `src/humanbrain/readout.py:21-110,292-346` — `SparseCellReadout.create/teach/turnover`, dense `reservoir_eligibility` binding
- `src/humanbrain/recurrent.py:652-744` — `observe`/`strengthen` reference semantics (sparse_three_factor path)
- `logs/text_scale_cpu_findings.md` §2-3 — layer definitions, throughput anchors/budgets quoted here
- This directory — `l1_token_injection.py`, `l2_eligibility_read_cost.py`, `l2_eligibility_read_cost.json`
