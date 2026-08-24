# CPU seam specs: `read_cells` FFI + indexed staged membership

**Date:** 2026-08-24
**Branch:** `codex/scale-opt-cpu` (worktree `human-brain-scale-opt-cpu`) @ `51762c4`.
**Status:** implementation-grade spec. Written so any agent can implement without re-deriving
context; every anchor cites `procedural.rs` / `procedural_runtime.py` line numbers **as of
`51762c4`** plus a stable symbol anchor (line numbers drift, symbols don't).
**Provenance:** FINDINGS.md §2.2 (`read_cells` sketch) and §3.3(c) (staged-scan wall) in
`experiments/procedural_adapter_spike/FINDINGS.md`; budgets from
`logs/text_scale_cpu_findings.md` (6.25 ms/token @160 tok/s floor, 1.36 ms/token @735 tok/s).
**Scope guard:** these are the only two seams specced here. Token-drive batching (~0.96 µs/event
host marshal, L1 caveat) and emission-captured pre-eligibility (FINDINGS §4) are out of scope.

Both seams are **performance-only**: neither may change one bit of simulated state, spike traces,
event accounting, or fail-stop timing relative to `51762c4`. The acceptance sections below make
that testable.

---

## Seam A — `humanbrain_procedural_runtime_read_cells`

### A.1 Problem and budget

The only state-read seam today is `snapshot()`
(`procedural.rs:1939–2007` → `humanbrain_procedural_runtime_snapshot`; Python
`procedural_runtime.py:_read_snapshot`, lines 685–751). It has no GID filter: native merges
`residents`+`staged`, copies every record, and Python converts every record into a frozen
dataclass. Measured (FINDINGS §2, `l2_eligibility_read_cost.json`): reading eligibility for
K=16 hot cells at 100,528 residents costs **161 ms/token** — 26× the entire 6.25 ms/token
budget at the 160 tok/s floor. ~99% of that is Python dataclass conversion over R records;
the raw FFI copy alone is ~13 ns/resident. Linear projection: ~1.6 s/token at 1e6 residents.
A per-token host mirror (Phase B/C of FINDINGS §5) is impossible on this seam; a targeted
O(K log R) read is required.

Budget to beat: **< 50 µs per K=16 call at R = 1e6 residents**, i.e. ≤ ~3 µs/request including
FFI entry/exit — roughly four orders of magnitude under snapshot() at equal R.

### A.2 Exact signature and ABI surface

One additive export placed immediately after `humanbrain_procedural_runtime_snapshot`
(ends at `procedural.rs:2007`). No new types; the out-record reuses the existing
`#[repr(C)] ProceduralResidentRecord` (`procedural.rs:99–111`) so the "one canonical record
layout" principle holds and Python can reuse `_NativeResident`
(`procedural_runtime.py:276–287`) verbatim.

```rust
/// Targeted resident-state read (active residency only; see A.3).
///
/// For each `gids[i]`: binary-search `residents` (ascending-GID `Vec<CellState>`)
/// via `resident_index`, then linear-binary-search `staged`; staged wins if
/// present (unreachable under the disjointness invariant D1, kept defensive).
/// Writes exactly `gid_count` `ProceduralResidentRecord` values to `out` in
/// input request order and one presence flag per request to `found`.
/// Absent cells get the canonical-rest record for the requested gid.
/// O(K log R). Allocates nothing. Mutates nothing. Touches no counters.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_read_cells(
    runtime: *const ProceduralRuntime,
    gids: *const u64,
    gid_count: usize,
    out: *mut ProceduralResidentRecord,
    found: *mut u8,
) -> u32 // PROCEDURAL_OK | PROCEDURAL_INVALID | PROCEDURAL_POISONED
```

Out-record byte layout (72 bytes, 8-byte aligned — identical to the snapshot element type;
Python `_NativeResident` must stay field-for-field identical):

| offset | size | field |
| ---: | ---: | --- |
| 0 | 8 | `gid: u64` |
| 8 | 8 | `voltage_relative_mv: f64` |
| 16 | 8 | `after_spike_current_0_pa: f64` |
| 24 | 8 | `after_spike_current_1_pa: f64` |
| 32 | 8 | `threshold_spike_mv: f64` |
| 40 | 8 | `threshold_voltage_mv: f64` |
| 48 | 4 (+4 pad) | `refractory_steps: i32` |
| 56 | 8 | `synaptic_current_pa: f64` |
| 64 | 8 | `eligibility: f64` |

Error codes — existing constants only (`procedural.rs:39–49`), no new statuses:

| code | when |
| --- | --- |
| `PROCEDURAL_OK` (0) | read completed; results valid |
| `PROCEDURAL_INVALID` (1) | `runtime` null; or `gid_count > 0` and any of `gids`/`out`/`found` null. (Mirrors the null-check style of `counts`, `procedural.rs:1925–1931.`) |
| `PROCEDURAL_POISONED` (5) | `(*runtime).poisoned != PROCEDURAL_OK` |

Validation order is normative (pin it so tests can't pass against a reordered impl):
null checks → poison check → `gid_count == 0` short-circuit returning `OK` with no writes →
per-gid read loop.

### A.3 Pinned semantics contract

Each decision below is deliberate; do not "improve" them without re-opening this spec.

1. **Output order = request order.** No implicit sort, no dedup. Duplicate gids in the request
   each get their own independent search and result. This lets callers zip `(gids, out, found)`
   positionally.
2. **Absent-cell records carry canonical positive zeros.** `found[i] = 0` ⟺
   `out[i] == CellState::canonical_rest(gids[i]).record()` — i.e. `gid` echoes the requested
   gid, every dynamics field is exactly `+0.0`, `refractory_steps == 0`. Constructed by calling
   `canonical_rest(...).record()` (never by arithmetic), which preserves the Gate-3 signed-zero
   canon at the seam (`+0.0`, not `-0.0`; see `canonical_zero`, `procedural.rs:208–215`).
3. **Precedence: residents searched first, then staged; staged wins if present.** Under
   invariant D1 (below) a gid is never in both, so precedence is unreachable; it exists to make
   the contract total rather than rely on an invariant an FFI caller can't observe.
   - **D1 (disjointness invariant):** `residents`, `staged`, and `dormant` are pairwise
     disjoint by GID at every observable point. Residents gain members only via `absorb_staged`
     (which empties `staged`, `procedural.rs:842–875`); dormant→staged reactivation removes the
     key from `dormant` before pushing to `staged` (`procedural.rs:913–947`); a fresh staging
     happens only after misses in all three structures (`procedural.rs:950–975`).
4. **Active residency only: dormant cells report `found = 0` + canonical zeros.** This matches
   the documented `snapshot()` divergence in event-proportional mode ("snapshot reports active
   residents + staged only", FINDINGS §3.2): the two read seams must agree on what "is resident"
   means or the Phase-B mirror will silently disagree with checkpoints. Host code needing
   dormant eligibility uses `humanbrain_procedural_runtime_dormant_snapshot`
   (`procedural.rs:2024–2053`) and replays the catch-up multiplies itself. Call this out in the
   Rust doc-comment; it is the most likely future-usability complaint.
5. **Fail closed on poison.** Unlike `snapshot()` (which intentionally serves diagnostics of a
   poisoned runtime via `counters.overflow_code`), `read_cells` returns `PROCEDURAL_POISONED`
   without touching `out`/`found`. Rationale: this seam feeds learning (eligibility mirrors that
   drive weight updates). Serving pre-poison state silently would corrupt training with no
   signal; checkpoint/diagnostic reads keep using `snapshot()`. This asymmetry is intentional —
   document it beside the export.
6. **No GID range validation.** A requested gid ≥ `configured_cells` is simply absent:
   `found = 0`, canonical zeros, status `OK`. Range rejection stays a Python-side courtesy
   (A.5), matching the precedent that out-of-range event gids are rejected in Python before FFI
   (FINDINGS §1). Keeps the hot loop branch-free beyond the two binary searches.
7. **Read-only.** No counter mutation, no `current_step` change, no allocation, no lock
   discipline change (Python serializes via `_operation_lock`, same as `snapshot()`).
   Safety precondition on the caller: `out`/`found` must not alias runtime-owned memory
   (unreachable from Python, where both are freshly allocated ctypes arrays).
8. **ABI version:** additive export, `humanbrain_kernel_abi_version` unchanged — same
   rationale as the `dormant_snapshot` precedent (comment at `procedural.rs:2019–2023`). The
   reviewed change may bump it; that decision belongs to the reviewed PR, not this seam.

### A.4 Implementation plan (native)

~30 lines besides tests. Edit sites:

1. Helper on `ProceduralRuntime`, directly after `resident_index` (`procedural.rs:877–879`):

```rust
fn read_cell(&self, gid: u64) -> Option<&CellState> {
    if let Ok(index) = self.resident_index(gid) {
        return Some(&self.residents[index]);
    }
    match self.staged.binary_search_by_key(&gid, |state| state.gid) { ... }
}
```

   Note `staged` is sorted whenever this can run: mid-step delivery leaves `staged` unsorted in
   general, but `read_cells` is only callable between top-level FFI calls, and every non-poisoned
   exit path of `advance` ends a step either post-absorb (staged empty) or with staged never
   populated (population 0). A mid-delivery poisoned abort is excluded by rule A.3.5. Add a
   `debug_assert!(self.staged.windows(2).all(|w| w[0].gid <= w[1].gid))` on the staged branch —
   cheap (K executions over ≤ min(resident_cap, event_cap) elements) and turns any lifecycle
   mistake into a test failure instead of silent misorder.
2. Export after line 2007, following the validation order of A.2 and writing records via
   `state.record()` / `CellState::canonical_rest(gid).record()` through `ptr::add` writes
   bounded by `gid_count` (style-match `snapshot`'s merge loop, `procedural.rs:1962–1985`).
3. Unit tests go in the existing `#[cfg(test)] mod tests` (`procedural.rs:2220+`).

### A.5 Implementation plan (Python binding)

File: `src/humanbrain/procedural_runtime.py`.

1. ctypes declaration inside `_procedural_library()`, after the `dormant_snapshot` block
   (lines 370–376): `argtypes = [void_pointer, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
   ctypes.POINTER(_NativeResident), ctypes.POINTER(ctypes.c_uint8)]`, `restype = c_uint32`.
2. Method on `NativeProceduralRuntime` directly after `snapshot()` (line 757):

```python
def read_cells(self, gids: Sequence[int]) -> tuple[ProceduralResidentState | None, ...]:
    """Targeted read of active resident state, in request order.

    None marks a requested cell that is not actively resident (not in
    `residents`/`staged` at this instant). Dormant cells (event-proportional
    mode) read as absent; use `dormant_cells()` for those.
    """
```

   Contract details, pinned: take `self._operation_lock` + `self._require_open()` exactly like
   `snapshot()` (lines 753–757); validate every gid as exact `int` via `_exact_uint64` then
   reject `>= self.config.manifest.cell_count`-equivalent configured bound with `ValueError`
   (courtesy rule A.3.6 — mirrors `_event_batch`'s pre-FFI rejection); empty sequence returns
   `()` without an FFI call; allocate `( _NativeResident * K )()` and `(ctypes.c_uint8 * K)()`,
   call, raise `ProceduralRuntimeError(status)` on nonzero, convert hits via the existing
   `_resident()` converter (line 505) and map `found == 0` to `None`.
3. Export the name in the module `__all__`-style tuple only if new public names are introduced
   (none needed — method lives on the runtime class).

### A.6 Test plan

Rust (`procedural.rs` tests module, config style per `collision_heavy_row_matches_oracle...`,
lines 2234–2270):

- `read_cells_matches_snapshot_records_for_present_absent_and_staged` — small FT config;
  drive some cells resident, others staged (deliver due-next-step events, read before their
  fused pass via a second advance boundary… simplest deterministic recipe: one `advance(n,
  events)` whose events land on the final step leaves them absorbed; instead stage via
  `advance(1, events_a)` then inject `events_b` due at step n and call read through the C-ABI
  helper between advances). Assert per-request: present gids byte-equal the corresponding
  `snapshot().residents` record; absent gids byte-equal `canonical_rest(gid).record()`
  (compare `f64::to_bits`, not `==`, to catch `-0.0`); `found` flags exact.
- `read_cells_reports_poisoned_without_touching_outputs` — force a capacity fail-stop, pre-fill
  `out`/`found` sentinels, assert status == `PROCEDURAL_POISONED` and sentinels untouched.
- `read_cells_handles_empty_null_and_duplicate_requests` — `gid_count == 0` with null buffers
  → OK; duplicates → consistent repeated results; out-of-range gid → absent semantics.
- `read_cells_is_bit_identical_across_identically_seeded_runtimes` — two runtimes, same seed
  and schedule, compare raw out-buffer bytes via slice of `u64` reinterpretation.

Python (`tests/test_procedural_read_cells.py`, mirroring `tests/test_procedural_runtime_ep.py`
conventions — production GLIF constants, small manifests):

- Order/dupes/partial-found matrix: `[present, absent, present, dormant, out_of_range→ValueError
  raised before FFI]`; assert positional alignment and `None` placement.
- Dormant semantics in EP mode: retire a cell to dormancy, assert `read_cells` returns `None`
  while `dormant_cells()` still lists it, and that a reactivating advance makes it readable.
- Determinism: identically-seeded runtimes → identical tuples.
- Poison: after forced overflow, `read_cells` raises `ProceduralRuntimeError(5)`.
- Lock/open hygiene: call on closed runtime → the standard closed-runtime error.

### A.7 Acceptance microbenchmark

Script: `experiments/procedural_adapter_spike/l4_seams_acceptance.py`, artifact
`l4_seams_acceptance.json` (same directory/conventions as L2/L3 scripts; CPU-only).

Fixture (the hard part — prescribe it, don't improvise): cfg = 2^20,
`resident_capacity = interval_distinct_capacity = 1_048_576 + slack`,
build R = 1,000,000 residents with the L2 bulk recipe (`fill_residents`,
`l2_eligibility_read_cost.py:79–86`: one `advance(TOKEN_STEPS, [one subthreshold event per
gid])`). **Known cost warning:** three O(R²)-class paths make the fixture build minutes-long at
1e6 (linear `.position()` staged scans ≈ R²/2 comparisons during delivery;
`IntervalLedger::mark_single` sorted-insert shifts ≈ 2 TB of memmove traffic; one final
O(R) `absorb_staged`). Mitigate by building in k = 8 chunks of 125k across separate `advance`
calls (cuts the scan term k×; ledger term is irreducible through the public API) and expect
**~3–6 min one-time build per process**. Cache nothing across processes; the benchmark's timed
section starts only after the build.

Measurement protocol:

- Request sets: for each repetition draw K=16 gids uniformly **without replacement from a
  shuffled pool spanning the full [0, R) range**, never reusing a pool across repetitions, so
  the probed working set (~reps × K × 72 B) far exceeds LLC and steady-state numbers stay
  honest about cache-miss cost.
- Timing: 3 warmup calls, then median of ≥ 31 timed calls (`time.perf_counter_ns` around the
  `runtime.read_cells(gids)` call, conversion included — the seam is judged end-to-end).
- Report: median µs, p10/p90, ns/request, and the snapshot()-based extraction equivalent for
  context (K lookups out of one snapshot at this R — measured once, not per rep; it is
  ~seconds).

Gates (pass/fail, recorded in the JSON):

| id | condition |
| --- | --- |
| A-G1 | median `read_cells` K=16 @ R=1e6 < **50 µs** (expect single-digit µs warm; cold-miss arithmetic: ≤ 20 cache-missing probes/gid × ~100 ns × 16 ≈ 32 µs worst case, so 50 µs holds even fully evicted) |
| A-G2 | median `read_cells` K=16 @ R≈100k < 20 µs (sanity scaling; log2 delta only) |
| A-G3 | speedup vs snapshot-extract ≥ 1000× @ R=100k (161 ms → sub-161 µs) |
| A-G4 | determinism gate: two identically-seeded runtimes, byte-identical raw buffers |
| A-G5 | bit-parity regression: full existing suite green (`pytest tests/test_procedural_runtime.py tests/test_procedural_runtime_ep.py tests/test_procedural_read_cells.py`) — proves touch-nothing |

### A.8 Effort estimate

| item | hours |
| --- | ---: |
| Rust helper + export + validation | 3 |
| Rust unit tests (4 above) | 3 |
| Python argtypes + `read_cells` + validation | 2 |
| Python seam tests | 2 |
| Benchmark script + JSON artifact + doc note | 2 |
| **Total** | **12 ± 4** |

### A.9 Risk register

| risk | impact | mitigation (spec-pinned) |
| --- | --- | --- |
| Caller assumes all-or-nothing reads; partial-found misread corrupts mirror | learning corruption | `found[]` semantics pinned (A.3.1–2); dedicated partial-found tests in A.6; `None`-per-slot Python return makes absence explicit |
| `-0.0` leaks into absent records, violating Gate-3 signed-zero canon | downstream bit-parity breaks | zeros constructed only via `canonical_rest().record()`; bit-level (`to_bits`) assertions, never `==` |
| Staged vec encountered unsorted (lifecycle regression elsewhere) | wrong record returned for staged gid | debug_assert sortedness on staged branch (A.4.1) + poison-fail-closed excludes mid-delivery states (A.3.5) |
| Someone later "extends" `ProceduralResidentRecord` for this seam | silent ABI drift for existing snapshot consumers | forbidden by A.2; additive fields would require a new versioned struct + review |
| Mirror caches across `advance` without re-read | stale eligibility into plasticity | caller obligation, documented on the Python method; native cannot enforce |
| Poison asymmetry surprises maintainers (snapshot reads, read_cells refuses) | confusion → "fix" removes fail-closed check | rationale recorded in A.3.5 and required in the export's doc comment |
| Fixture build time misread as seam cost | false benchmark failure | A.7 prescribes chunked build + timing starts post-build; warning printed by script |

---

## Seam B — indexed staged membership

### B.1 Problem and hot-loop citation

When an event's target is neither resident nor dormant, `apply_event` scans the staged Vec
linearly to decide accumulate-vs-stage:

```rust
// procedural.rs:950–958 @51762c4 (inside fn apply_event, opened at :906)
if let Some(slot) = self
    .staged
    .iter()
    .position(|state| state.gid == event.target_gid)
{
    let value = self.staged[slot].synaptic_current + current;
    self.staged[slot].synaptic_current = canonical_zero(value);
    return Ok(());
}
```

Stable anchor: the only `.iter().position(` in the file; inside `apply_event`'s
`Err(_) =>` arm, immediately after the `event_proportional()` dormant block.

Cost model (FINDINGS §3.3(c)): with activity A events/token and fanout f, ~f·A fresh
recipients stage per token and subsequent deliveries to those targets each pay O(S)
(S = |staged|) — O((f·A)²) per token. Measured consequence at N_cfg=1e6, 5% uniform activity
(50k events/token, fanout 4): **EP 9.01 s/token vs FT 4.99 s/token**, and cutting fanout to 1
speeds *both* modes ~9–12× superlinearly — the shared injection path dominates, and this scan
is its mode-independent core. This wall masks the stepping-mode economics the L3 mode exists to
expose, and keeps both modes under the 160 tok/s gate at ≥1% uniform activity.

### B.2 Chosen design: GID-keyed slot index (`BTreeMap<u64, u32>`)

Add one field to `ProceduralRuntime` (struct at `procedural.rs:708–735`), next to `staged`:

```rust
/// Delivery-phase membership index for `staged`: gid -> slot offset into the
/// `staged` Vec. Valid only while staged accumulates between fused passes
/// (invariant D2 below); emptied by `absorb_staged`. Never iterated for
/// semantics; pure lookup structure. Replaces the O(S) `.position()` scan in
/// `apply_event` with O(log S).
staged_index: BTreeMap<u64, u32>,
```

`apply_event` replacement for lines 950–958:

```rust
if let Some(&slot) = self.staged_index.get(&event.target_gid) {
    let value = self.staged[slot as usize].synaptic_current + current;
    self.staged[slot as usize].synaptic_current = canonical_zero(value);
    return Ok(());
}
```

and every `self.staged.push(state)` site gains `self.staged_index.insert(state.gid,
self.staged.len() as u32 - 1)` **after** the push (slot = len−1). Push sites: fresh staging
(`procedural.rs:975`) and dormant reactivation (`procedural.rs:946`). Slot fits `u32` because
`staged.capacity() = min(resident_capacity, event_capacity)`
(`procedural.rs:755`) and `resident_capacity ≤ configured_cells ≤ usize`-practical bounds;
add a constructor-time `debug_assert!(staged_capacity <= u32::MAX as usize)`.

Why `BTreeMap<u64, u32>` and not the alternatives (decision record):

| alternative | verdict |
| --- | --- |
| Sorted `Vec<(u64, u32)>` index + binary search | rejected: fresh-staging insertions memmove O(S) — reintroduces a quadratic (at S=50k, ~15 GB memmove traffic per token) |
| `HashMap<u64, u32>` | workable but rejected: hash iteration nondeterminism must never leak into state (it wouldn't — but proving that forever is a tax), and `BTreeMap` matches the existing dormant-store precedent (`procedural.rs:731`), keeping one idiom |
| Sort `staged` + binary search in place | rejected: destroys arrival-order slots mid-delivery and buys nothing — absorb already sorts once per step |
| Per-cell hash marks like the old dense path | rejected: that pattern was already removed for the fused pass (comment at `procedural.rs:1275–1279`); a second scratch structure would need its own capacity story |

### B.3 Lifecycle and invalidation rules (normative)

The index is a **derived scratch structure with a strictly scoped validity window**. Rules:

1. **D2 (validity invariant):** `staged_index` maps exactly `{staged[i].gid → i}` only during
   a step's delivery phase (after the fused pass of step t−1 cleared it, until `absorb_staged`
   of step t runs). Outside that window (post-absorb, mid/post-fused-pass, or any poisoned
   state) its contents are undefined-but-empty-or-stale and must never be read.
2. **Clear-on-absorb:** `absorb_staged` clears the index together with the Vec
   (insert `self.staged_index.clear();` next to `self.staged.clear();`, line 873). This is the
   single removal mechanism — there are no individual tombstones because nothing ever removes a
   single staged entry (verified: the only mutators of `staged` are the two pushes and
   `clear()`).
3. **Sort hazard handled by scoping:** the fused pass sorts `staged` by gid
   (`sort_unstable_by_key`, line 1250) *after* delivery completes, which invalidates slots —
   but no `apply_event` can run between that sort and `absorb_staged` (same step, same
   `&mut self` call chain), and absorb then clears everything (rule 2). Do **not** "fix" this by
   re-keying the map after sorting; the scoping rule is the fix.
4. **Poison paths:** early `return self.poison(code)` between sort and clear leaves a stale
   index. Harmless: poisoning is terminal (`advance` fails closed at `procedural.rs:1670–1672`),
   `apply_event` is unreachable afterward, and `read_cells`/`snapshot` deliberately do not
   consult the index (A.4). State this in the field doc-comment so nobody adds a reader.
5. **Capacity accounting untouched:** `total_residents()` (line 830) and the resident-capacity
   check (line 962) keep counting `staged.len()`; the index changes no fail-stop timing or
   trip point. The capacity checks run *before* the staging push exactly as today.
6. **New allocation-failure surface:** `BTreeMap::insert` allocates nodes lazily and cannot
   pre-reserve. An OOM mid-delivery becomes `self.poison(PROCEDURAL_ALLOCATION)` from
   `apply_event` — consistent with the existing bounded-allocation philosophy
   (`reserved_vec`, `PROCEDURAL_ALLOCATION` at `procedural.rs:568`) but genuinely new behavior,
   so record it: worst case is bounded by `min(resident_capacity, event_capacity)` entries
   (≤ ~48–64 B each with node overhead; ≤ ~1 MB at the shipped 16k staged cap). Accepted;
   do not add a fallback path.

### B.4 Complexity and memory delta

| quantity | before | after |
| --- | --- | --- |
| staged hit lookup (`apply_event`) | O(S) | O(log S) |
| staged insert (fresh recipient) | O(1) push + O(S) paid later by scan traffic | O(log S) insert |
| wholesale clear (absorb) | O(1) (`Vec::clear`) | O(capacity) for BTreeMap `clear` — amortized negligible next to the O(residents+staged) merge itself |
| per-token injection (A events, fanout f) | O((fA)²) worst | O(fA · log(fA)) |
| memory | 0 extra | ≤ min(R_cap, E_cap) × ~(16 B entry + ~32–48 B B-tree overhead); ≤ ~1 MB at 16k staged cap, ≤ ~80 MB at the 1e6 stress cap — include it in `payload_bytes()` |

Accounting honesty: extend `payload_bytes()` (lines 800–816) with a conservative term for the
index, styled after the dormant-store estimate (`len * (key + value + 4*usize)`). Tests pin
`native_payload_bytes` only for *stability across identical workloads*
(`tests/test_procedural_runtime.py:789–798`), not absolute values, and the index is
len-deterministic — so this is safe. Note in the PR: absolute payload bytes shift in **both**
stepping modes (unlike the dormant term, which is FT-zero); no dynamics impact.

### B.5 Determinism argument (why this cannot move a bit)

All floating-point accumulation into `staged[slot].synaptic_current` happens in canonical
pending-queue order (events sorted by `(due_step, target, source, edge_slot)` at
`append_events`, line 1004). The lookup structure only answers *which slot*, never *in what
order* additions occur; slot assignment is push-order-determined and unchanged. The index is
never iterated for semantics. Therefore state bits, spike traces, counters, and fail-stop
timing are bit-identical to `51762c4` — same class of argument as the L3 lazy-replay parity
(FINDINGS §3.1–3.2), and enforced by the same test suites (B.7).

### B.6 Implementation plan (edit sites)

1. Field + constructor init (`staged_index: BTreeMap::new()`) in `ProceduralRuntime::new`
   (lines 760–776) + `debug_assert` on staged capacity (B.2).
2. `apply_event`: replace lines 950–958; add `staged_index.insert` after both push sites
   (lines 946, 975).
3. `absorb_staged`: add `self.staged_index.clear();` beside `self.staged.clear();` (line 873),
   with a comment citing D2.
4. `payload_bytes()`: add the index term (B.4).
5. Optional hardening (recommended, tiny): in the `snapshot` merge (lines 1962–1985), replace
   the implicit ascending-`staged` assumption with an explicit `debug_assert!` that staged is
   sorted or empty whenever the merge runs with staged non-empty (only reachable on poisoned
   mid-aborts today). Documents a pre-existing quirk this seam must not silently worsen.
6. No Python-side changes required (behavior-preserving); no ABI change (internal field only).

### B.7 Test plan

Rust unit tests (existing `mod tests`, style per lines 2229–2270):

- `staged_index_maps_exactly_the_staged_vec_during_delivery` — drive a mixed schedule; between
  advances, reconstruct expected map from a snapshot of staged gids and compare (test-visible
  via `#[cfg(test)]` accessor or direct field access in-module).
- `indexed_apply_event_is_bit_identical_to_linear_scan_reference` — oracle: reimplement the old
  `.position()` lookup in the test, run both against cloned runtimes with identical schedules,
  compare full snapshots bit-for-bit (`to_bits` on every f64 field) including counters.
- `staged_index_cleared_by_absorb_and_unused_after_sort` — after a normal advance,
  `assert!(staged_index.is_empty())`; after a poison mid-fused-pass, assert subsequent
  `advance` returns `PROCEDURAL_POISONED` (rule B.3.4 reachability).
- `duplicate_delivery_accumulation_order_preserved` — many events to one staged target in one
  step; final `synaptic_current` bit-equals the sequential-sum oracle (guards the B.5 argument
  mechanically).

Python regressions (existing suites are the real gate — run all):

- `tests/test_procedural_runtime.py`, `tests/test_procedural_runtime_ep.py` green unchanged.
- Identically-seeded EP-vs-baseline spike-trace equality on the L1/L3 fixtures (scripts
  double as checks; assert no diff in delivered/generated counts and spike tuples).

### B.8 Acceptance microbenchmark

Two layers: a precise native gate plus an integration gate proving the wall actually moved.

**(a) Native op gate** (in-file `#[test]` with `std::time::Instant`, excluded from CI timing
asserts on slow runners via a generous threshold):

- Build S = 10,000 staged entries (one delivery burst of 10k distinct fresh targets, none yet
  absorbed). Then time 10,000 mixed operations: 50% hits (accumulate into existing staged
  targets) + 50% misses (fresh staging inserts), plus one wholesale clear, repeated ≥ 11 times
  after 2 warmups; report mean amortized µs/op.
- Gate **B-G1: amortized < 1 µs/op at S = 10k** (expect ~0.1–0.3 µs; the old scan averages
  S/2 ≈ 5,000 comparisons/op ≈ several µs — the gate is set to be unmissable while allowing
  debug-build slack; run in `--release`).

**(b) Integration gate** (script `l4_seams_acceptance.py`, same artifact JSON as Seam A):
re-run the FINDINGS §3.3(c) diagnostic row — N_cfg = 1e6, 5% uniform activity (50k
events/token), fanout 4, EP and FT, 15 tokens, stress GLIF config — on the post-change build:

| id | condition |
| --- | --- |
| B-G2 | EP s/token ≤ **3.0** (vs 9.01 @`51762c4`, ≥ 3× improvement) and FT s/token ≤ **1.7** (vs 4.99) |
| B-G3 | spike traces + delivered/generated counts bit-identical to `51762c4` on the identical schedule (store baseline outputs in the JSON artifact keyed by commit) |
| B-G4 | sparse-regime non-regression: FINDINGS §3.3(a)-style row (640 events / 23.4k residents) within ±5% of the recorded 363 tok/s EP figure (the index must not tax the regime L3 won) |
| B-G5 | `payload_bytes` stability test (`test_native_scratch_payload_is_stable_across_repeated_bounded_workloads`) green with the new term included |

### B.9 Effort estimate

| item | hours |
| --- | ---: |
| Field, apply_event rewrite, absorb clear, payload term, asserts | 3 |
| Rust unit tests incl. linear-scan oracle parity | 4 |
| Native op-gate test + integration re-measure script updates + JSON artifacts | 3 |
| Full-suite regression runs + baseline-trace capture @`51762c4` | 2 |
| Doc notes (field comments, FINDINGS follow-up paragraph) | 1 |
| **Total** | **13 ± 4** |

### B.10 Risk register

| risk | impact | mitigation (spec-pinned) |
| --- | --- | --- |
| Slot invalidation after fused-pass sort read by a future code path | silent wrong-cell accumulation (bit corruption) | D2 scoping is normative (B.3.2–3); clear-on-absorb is the single removal path; dedicated lifecycle test; field doc-comment forbids readers outside delivery phase |
| Arithmetic reorder sneaks in (e.g. "optimize" by batching inserts) | breaks bit-parity contracts | B.5 argument + mechanical duplicate-accumulation oracle test; review checklist item: no change to addition order |
| BTreeMap OOM poisons a run that previously succeeded | new failure mode mid-training | accepted & documented (B.3.6); bound ≤ staged cap; PROCEDURAL_ALLOCATION is the existing fail-stop channel |
| `payload_bytes` shift mistaken for regression | churn/false alarms | B.4 note: stability-pinned only; absolute shift expected in both modes; called out in PR description |
| Index taxes the sparse regime L3 won | 363 tok/s figure regresses | B-G4 non-regression gate ±5%; index ops are O(log S) with S tiny (≤ hundreds) in that regime |
| Pre-existing unsorted-staged snapshot quirk gets blamed on this change | misdirected debugging | optional hardening debug_assert (B.6.5) documents reachability (poisoned mid-abort only) |
| `u32` slot truncation at absurd capacities | wrong slot | constructor `debug_assert!(staged_capacity <= u32::MAX)`; capacities are `usize`-bounded far below this in practice |

---

## Shared rollout policy

- **Sequencing:** independent seams; either may land first. Seam A unblocks FINDINGS §5
  Phase B/C (learning parity + scale-up); Seam B unblocks the ≥1%-activity throughput band for
  both modes. If landed together, land B first so A's benchmark fixture build (which pays the
  staged scan quadratically) gets faster for free.
- **ABI:** both are additive/no-version-change by the `dormant_snapshot` precedent; the actual
  bump decision belongs to the reviewed PR.
- **Definition of done (checklist):**
  - [ ] All gates A-G1..G5 and B-G1..G5 recorded in `l4_seams_acceptance.json` with commit SHA.
  - [ ] `cargo test` (native crate) + the three Python test files green on the landing commit.
  - [ ] Bit-parity vs `51762c4` demonstrated for Seam B on an identical schedule (B-G3).
  - [ ] FINDINGS.md amended with a short §2.2/§3.3(c) "landed" note referencing this spec.
  - [ ] No new error codes, no record-layout changes, no counter semantics changes.
