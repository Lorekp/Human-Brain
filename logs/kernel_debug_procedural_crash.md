# KERNEL-DEBUG notes: native crash in procedural.rs queue-management rewrite

## Symptom
- `pytest tests/test_procedural_runtime.py` died with **exit code 127, zero output**
  at `test_sparse_runtime_is_bit_exact_to_independent_dense_native_reference`
  (first bit-exact sparse-vs-dense test).
- faulthandler produced nothing → crash was inside the native FFI kernel.
  A Rust panic unwinding out of an `extern "C"` fn aborts the process, which
  matches the silent hard-death signature.

## Root cause (two defects, one family — both in the new staged-cell merge)

The optimizer split materialized cells into a separate `staged: Vec<CellState>`
array (`residents` stays sorted; staged cells are O(1)-pushed during delivery
and merged back during the step pass). Both merge sites iterated the *logical*
union of `residents ∪ staged` and wrote survivors back with
`self.residents[write] = state`, but `residents.len()` does **not** include
`staged.len()`:

1. **Out-of-bounds write (the crash).** In the fused fast-path pass and in
   `absorb_staged()`, the write cursor can reach `write >= residents.len()`
   as soon as any staged cell survives its step (newly materialized cells
   almost always do). Vec indexing then panics → abort across FFI → exit 127,
   no output. Deterministic; first triggered by the first test that delivers
   events to fresh GIDs mid-run.

2. **Forward in-place merge corruption (latent).** Even where indexing stayed
   in bounds, merging ascending `staged` into `residents` from the FRONT can
   overwrite not-yet-read resident slots whenever a staged GID sorts before an
   unread resident (e.g. residents=[5,6,7], staged=[1,2,3] destroys 6 and 7).

## Fix (minimal, optimization preserved)
- `absorb_staged()` rewritten as a **backward** in-place merge:
  - guard `merged_len <= residents.capacity()` → `Err(PROCEDURAL_ALLOCATION)`
    (fail-stop allocation contract kept),
  - pre-grow `residents` to `main + staged` with placeholder rest states
    (every slot is overwritten exactly once by the merge),
  - merge from the back keeping the invariant `write == read + staged_index`,
    so writes always land at/after every unmerged slot — never OOB, never
    clobbers unread residents. Returns `Result<(), u32>` now.
- Fused fast path now calls the fixed `absorb_staged()` first, then runs the
  single-array update+compact sweep over `residents` alone (`write <= read`
  always). The optimization is intact: sorted-run interval ledger merges
  replace the per-cell HashMap marks, the pending queue keeps its consumed-
  prefix offset scheme, only the generated tail gets sorted and linear-merged.
  Cell visiting order is unchanged (ascending GID), so spikes, counters and
  snapshots stay bit-exact.
- Exact fallback branch propagates the new absorb error via `poison()`.

## Verification
- `cargo build --locked --offline --release` green;
  DLL cache `$LOCALAPPDATA/Temp/humanbrain-native` cleared so tests load the
  fresh binary keyed by source digest.
- `cargo test --locked --offline --release`: 3/3 passed.
- Previously crashing test passes; full
  `pytest tests/test_procedural_runtime.py tests/test_procedural_address_space_benchmark.py`
  green; benchmark quick run confirms perf gains survive (see gate logs).
