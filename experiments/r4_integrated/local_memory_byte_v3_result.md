# Local memory/byte v3: invalid cost timing

2026-09-06. The disposable workload completes, but cost admission is **invalid**:
the frozen arm's 33-decision capped prediction records zero elapsed seconds.
The positive-duration requirement correctly rejects it. No full fit runs and
no cost projection or capability result is claimed.

Frozen source: `9199aa1cf660c81a027c8d5f3116ccdf0368b5f5`.
The [decision record](../../benchmarks/local_memory_byte_v3.cost_decision.json)
preserves source/admission/report and both independent audit hashes.

The local Python 3.11 clock inspection identifies `time.monotonic()` as
`GetTickCount64()` with 15.625 ms resolution. `time.perf_counter()` uses
`QueryPerformanceCounter()` with 100 ns resolution. All 33 decisions and their
bytes are present: this is insufficient timing resolution, not absent work.
The zero is not replaced with a guessed floor or treated as free computation.

Outer execution took 27.316 seconds; conservative peak sum was 780.270 MiB.
The twenty disposable lessons offered 200 teacher byte/EOS events and 160
enabled decoder update opportunities, followed by 100 action attempts and two
motor demonstrations. Retained tensor state is 124,109,512 bytes. These weights
are discarded; four lessons per arm are not an acquisition experiment.

The original strict audit failure stays unchanged. A separate saved-only audit
checks the unaffected invariants before source changes: 1,130 source hashes,
five artifact hashes, initial/frozen weight chains, zero frozen credit, shared
source-free memories, exact worker command, 102 execution receipts and 100
action workspaces all pass. Its overall verdict remains invalid timing and
`admissible=false`; no model or child is rerun.

The next return fixes elapsed-time instrumentation with a deterministic
regression on the actual capped-generation seam and a fresh experiment identity.
Use the performance counter for durations and retain Unix time for deadlines.
Preserve the original workload, local learning rule, 900-second ceiling, extra
capped-prediction charge and two-times timing margin. The timer fix must not
silently reopen v3 or reinterpret its zero sample.
