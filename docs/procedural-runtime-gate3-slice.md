# Native procedural runtime: Gate-3 correctness slice

This layer is the smallest end-to-end source-driven runtime inside the existing
counter64-v2 address space. It is a correctness and resource-bound result, not
the material 200,000-cell Gate-3 speed result.

`NativeProceduralRuntime` is the production seam. It owns an opaque Rust
runtime and exposes only `advance`, `snapshot`, and lifecycle operations.
Python validates and marshals an event batch once per call; the biological-step
loop, fitted Allen GLIF5 equations, resident frontier, event queue, spike
expansion, and counter updates all remain native. The first slice intentionally
supports only `uniform_no_self`. Unsupported topology names fail before FFI;
there is no explicit-target fallback.

Each adapter serializes `advance`, `snapshot`, and idempotent `close` with one
per-instance reentrant lock. The lock covers the complete snapshot
count/allocate/copy sequence and native handle lifetime. Raw C callers do not
receive this synchronization and must externally serialize handle use and
destruction. Both raw snapshot functions receive every destination capacity and
reject an insufficient supplied buffer before writing any destination or count.
A `NULL, 0` pair explicitly omits that output.

`NativeDenseProceduralReference` is a separately named small-domain oracle. It
receives uint64 edge rows independently materialized by `ProceduralTopology`,
allocates state for every small-domain GID, and advances them with a separate
dense Rust loop. It is capped at 65,536 cells and is never selected by the
production adapter.

## Exact execution contract

Events are accumulated in `(due_step, target_gid, source_gid, edge_slot)` order.
Residents update in ascending GID order; spiking rows expand in ascending source
GID and edge-slot order. Recurrent delay is an exact positive integer, so a
spike cannot affect a target in the same biological step. Both source and target
GIDs must name configured cells; the first slice defines no out-of-range
provenance sentinel. Canonical keys in a submitted input-event batch must be
unique against both that batch and the already queued frontier. Rejection occurs
before native simulation-state mutation, is transactional, and does not poison
the runtime. A collision that only appears when a later spike generates an event
is detected after step mutation and is deterministic fail-stop.

The eight mutable fields are voltage, two after-spike currents, the spike and
voltage threshold components, refractory count, recurrent synaptic current,
and eligibility. Signed zero is canonicalized to positive zero at all
event/state write seams in both engines. This is an exact representation rule,
not a tolerance. A due positive-zero event does not materialize a cell. A cell
is evicted only after a real fixed-step transition leaves all eight fields at
canonical zero. Nonzero exponential tails may therefore retain touched cells
indefinitely until exact cancellation or floating-point underflow. When the
resident, event, returned-spike, interval-distinct, or reserved-allocation bound
is exhausted, the runtime becomes poisoned and reports the failure; it never
spills into a dense or approximate path. Snapshot-buffer and duplicate-input
validation failures are non-poisoning.

Returned-spike capacity exhaustion reports native status `9` and poisons the
handle. A spike-generated event key that collides with an already queued key
reports status `4` and also poisons the handle; by contrast, status `4` for a
duplicate submitted input is rejected before simulation-state mutation and is
non-poisoning. Any later operation on a poisoned handle reports status `5`.
Neither fail-stop path commits the in-progress biological step or publishes its
work as completed-interval counters. Bounded in-progress cell, event, and spike
state remains deterministic and snapshot-visible; the runtime does not claim
transactional rollback after step execution has begun.

The runtime stores no base-edge records. Each spiking source regenerates its
counter64-v2 row, applies the source's deterministic Dale-like sign, and queues
fixed-delay events. Row membership uses a reusable bounded hash set while target
slot order remains deterministic. The fallback expression uses widened `u128`
addition before reducing modulo the configured uint64 address range, including
across the `2^64` wrap. A row examines at most `129F+1` candidates and uses
`O(F)` scratch. Mutable edge count is zero in this slice.

## Scale accounting and bounds

Each successful advance returns canonical configured/addressable/resident/
updated/spiking counts, real cell-step and spike work, configured/resident/
mutable/generated edge counts, delivered events and queue peak, and exact
accounted native payload bytes. Analytic catch-ups are always zero. The
additional `configured_ids_scanned` counter is zero because no production loop
iterates the address range.

The public configuration explicitly bounds residents `R_capacity`, queued
events `Q_capacity`, returned spikes `S_capacity`, and distinct interval GIDs
`D_capacity`; fanout is `F`. Persistent and reusable scratch storage is
fallibly reserved during construction, and the native hot loop does not grow
it. Submitted keys are copied into reusable scratch, sorted in place, checked by
adjacent comparison, and merged against the sorted queue without quadratic
membership scans. Due events are copied through bounded scratch and removed only
after delivery; spike and topology scratch are reused.

Production storage is therefore
`O(R_capacity + Q_capacity + S_capacity + F + D_capacity)`, independent of
configured/addressable `N`. Work is proportional to resident steps, event
sorting/merging and delivery, bounded sorted-resident insertion, and bounded
spiking-source row generation. Neither construction nor advance allocates or
scans an object proportional to configured IDs. Logical capacity failures and
raw `usize` capacity-overflow construction are covered by tests. Safely forcing
the process-global allocator itself to exhaust memory is not practical in this
gate.

`native_payload_bytes` includes the runtime struct, persistent vector
capacities, reusable vector scratch, and logical hash-set/hash-map entry
capacities. It does not introspect allocator metadata or hash-table control
bytes, and it excludes the small dense oracle. This is canonical, deterministic
accounting for this slice, not allocator-exact resident-set measurement.

The protocol binds every production source/configuration input through an exact
path/size/SHA-256 manifest and a framed bundle digest. Framing includes the
namespace and each ordered UTF-8 path length/path plus uint64 payload length/raw
bytes, so neither file-boundary nor path-boundary concatenation can collide.

The complete preregistration is
`benchmarks/procedural_runtime_gate3_slice_protocol_v1.json`. Material timing,
the Gate-3 timing sweep, GPU execution, capability evaluation, DEV/SWE/
canonical/private evaluation, and Fly Brain remain frozen. A material timing
phase requires explicit approval and a separately fixed execution protocol.
