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

`NativeDenseProceduralReference` is a separately named small-domain oracle. It
receives uint64 edge rows independently materialized by `ProceduralTopology`,
allocates state for every small-domain GID, and advances them with a separate
dense Rust loop. It is capped at 65,536 cells and is never selected by the
production adapter.

## Exact execution contract

Events are accumulated in `(due_step, target_gid, source_gid, edge_slot)` order.
Residents update in ascending GID order; spiking rows expand in ascending source
GID and edge-slot order. Recurrent delay is an exact positive integer, so a
spike cannot affect a target in the same biological step.

The eight mutable fields are voltage, two after-spike currents, the spike and
voltage threshold components, refractory count, recurrent synaptic current,
and eligibility. Signed zero is canonicalized to positive zero at all
event/state write seams in both engines. This is an exact representation rule,
not a tolerance. A due positive-zero event does not materialize a cell. A cell
is evicted only after a real fixed-step transition leaves all eight fields at
canonical zero. Nonzero exponential tails may therefore retain touched cells
indefinitely until exact cancellation or floating-point underflow. When the
resident or event bound is exhausted, the runtime becomes poisoned and reports
the failure; it never spills into a dense or approximate path.

The runtime stores no base-edge records. Each spiking source regenerates its
counter64-v2 row, applies the source's deterministic Dale-like sign, and queues
fixed-delay events. Mutable edge count is zero in this slice.

## Scale accounting and bounds

Each successful advance returns canonical configured/addressable/resident/
updated/spiking counts, real cell-step and spike work, configured/resident/
mutable/generated edge counts, delivered events and queue peak, and exact
accounted native payload bytes. Analytic catch-ups are always zero. The
additional `configured_ids_scanned` counter is zero because no production loop
iterates the address range.

Production memory is bounded by resident capacity, pending events, last-call
spikes, and one generated fanout row. Work is proportional to residents,
delivered events, event ordering, and spiking-source row generation. Neither
construction nor advance contains an operation proportional to configured or
addressable IDs. The independent dense oracle is explicitly excluded from this
claim.

The complete preregistration is
`benchmarks/procedural_runtime_gate3_slice_protocol_v1.json`. Material timing,
the Gate-3 timing sweep, GPU execution, capability evaluation, DEV/SWE/
canonical/private evaluation, and Fly Brain remain frozen. A material timing
phase requires explicit approval and a separately fixed execution protocol.
