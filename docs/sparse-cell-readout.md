# Bounded sparse output-cell readout

The experimental `sparse_three_factor` rule removes the remaining dense
`vocabulary x reservoir` classifier from the recurrent human-cell model. Each
token is still represented by one fitted GLIF5 output cell. Its current is the
sum of a tonic pacemaker synapse, fixed cell heterogeneity, and explicit
excitatory contacts from reservoir cells. There is no Transformer, MLP, SSM,
dense score layer, optimizer state, or reservoir-sized normal matrix behind
the race.

This is an implemented mechanics lane, not a capability result. No validation,
held-out test, or SWE benchmark was run while choosing it. The frozen research
plan is `benchmarks/sparse_readout_protocol_v1.json`; execution remains a
separate step.

## Anatomy and state

For `V` output cells, `N` reservoir cells, and fan-in capacity `K`, each output
cell owns:

- `C = min(N, 4K)` immutable, sorted, unique potential-apposition source IDs;
- one nonnegative pre-times-post coactivity trace per potential apposition;
- exactly `K` sorted, unique capacity-slot source IDs, always a subset of the
  potential pool;
- one nonnegative excitatory weight and age per capacity slot;
- one output-activity exponential moving average and structural clock.

A zero-weight capacity slot is vacant, not an active biological synapse.
Functional plasticity skips it. Only a structural event can recruit it, and a
single event can recruit at most one apposition and delete at most one mature
contact. This prevents one wrong example from instantly creating `K` target
contacts. Candidate pools are sampled before text or labels with a seeded,
bounded-memory sampler. They are a controlled potential-anatomy hypothesis,
not a claim that random IDs reproduce cortical geometry.

The internal `SparseCellReadout` module owns all of these arrays, invariants,
native bindings, counters, and its checkpoint seam. `RecurrentCellCircuit`
only asks it for current, teaching, turnover, and diagnostics.

## Local learning and structural turnover

After a reservoir observation, the fitted output cells race. If the natural
winner is not the target, the target receives an explicit additional teaching
current. The same compiled fitted-cell response used by the race must show a
real threshold crossing before the target post gate is asserted.

For potential apposition `i -> j`, native Rust updates the stored trace as:

```text
e_ji <- trace_decay * e_ji + reservoir_trace_i * output_post_gate_j
```

The target and natural winner are the only output rows with post events. A
signed cell-level teaching factor potentiates occupied target contacts and
depresses occupied winner contacts. Excitatory contact and pacemaker weights
are clamped to `[0, maximum_weight]`, preserving Dale's sign at this readout.
Pre alone, post/modulator alone, and an expired trace produce no contact
change. Perturbing an unrelated output row cannot affect participating
weights; global trace decay and age advance represent elapsed time, not an
error signal.

On a much slower per-target structural clock, the native turnover kernel runs
on scheduled target presentations whether the natural winner was correct or
wrong. This is intentional homeostatic/structural maintenance, not
error-triggered recruitment. It:

1. multiplicatively scales occupied incoming weights toward the output
   activity target `1 / V`;
2. considers deletion only for age-qualified contacts, with a seeded hazard
   increasing as weight weakens;
3. fills one vacancy from unassigned members of that output cell's fixed pool,
   using seeded stochastic selection biased by local apposition coactivity;
4. restores ascending source order without allocation.

The randomness is counter-based over the configured seed, output ID, and saved
structural event number. Checkpoint restoration therefore needs no hidden RNG
object and continues the next turnover bit-for-bit.

## Output race and present limitation

Inference sums the `K` explicit capacity slots in ascending source order in
native code, then applies the fitted GLIF5 resting response. The earliest
threshold crossing selects the token. When several cells cross on the same
integration bin, the current implementation deterministically keeps the first
cell and reports both `simultaneous_spike_candidates` and
`discarded_simultaneous_candidates`.

That tie report is arbitration bookkeeping. It is **not** evidence of the
shared inhibitory interneuron circuit recommended by the biological review.
An explicit inhibitory-cell implementation and remove-inhibition ablation
remain required before making a winner-take-all inhibition claim.

## Complexity and exactness

Sparse-readout persistent state is proportional to the declared contacts:

```text
potential IDs and traces: V * C * (4 + 8) bytes
contact IDs/weights/ages: V * K * (4 + 8 + 4) bytes
activity and clocks:      V * (8 + 8) bytes
```

With `C = 4K`, that is approximately `64 * V * K` bytes plus the small
per-output terms, versus `8 * V * N` bytes for dense float64 weights. The exact
ratio is therefore tied to both `N` and the selected `K`; no fixed 100x claim
is made at 6,144 cells. The engineering test uses `N = 200,000`, `K = 64`, and
checks the actual sparse arrays exceed a reachable 100x reduction. Inference
is `O(VK)`, online trace work is `O(VC + VK) = O(VK)`, and a turnover event for
one output is `O(C + K)`.

At `K = N`, both potential and capacity IDs are exactly `arange(N)`. Tests use
an independent ascending scalar oracle for currents and local updates, then
check the chosen race winner, update count, weights, ages, activity, reset, and
continued prediction exactly. Checkpoint format v3 stores the complete sparse
anatomy, apposition traces, ages, activity, clocks, and counters; malformed
ordering or contact/potential subsets are rejected. Dense `three_factor`
remains an explicit historical control, not a compatibility implementation of
the sparse rule.

Sparse readout has a registered exact hybrid CUDA path. The recurrent fitted
cells advance on the persistent-device CUDA kernel; their eligibility is then
consumed by the existing native Rust sparse-current and synapse-local
plasticity kernels. A real-device test asserts `backend == "cuda"` and exact
CPU parity for observe state and spikes, output current, first-spike race,
teaching, scheduled turnover, reset, checkpoint load, and post-checkpoint
continuation. Forced CUDA still fails if no CUDA backend is available; it never
silently falls back to CPU.

Only eligibility and the total spike count cross from device to host after a
sparse-readout token, because those are the recurrent outputs consumed by the
fitted-output current and plasticity kernels. At 200,000 GLIF5 cells this is
1,600,008 bytes/token rather than 12,000,008 bytes/token for all seven mutable
arrays. The CUDA device remains canonical for voltage, after-spike current,
adaptive thresholds, refractory state, and recurrent synaptic current between
tokens. `synchronize_state_from_device()` is the explicit full-state barrier
for diagnostics; `reset_sequence()` still resets and uploads every mutable
array. Dense and developmental CUDA lanes retain their full per-token host
mirror. The real-device parity test exercises sparse current and teaching
before the diagnostic barrier, then downloads and bit-compares every mutable
array.

This registration does not imply that sparse current or plasticity runs on the
GPU, nor is it a capability result. The separate synthetic 200,000-active-cell
timing protocol in `benchmarks/sparse_cuda_engineering_protocol_v1.json`
profiles kernel, transfer, fitted-output, and teaching phases before deciding
whether any additional CUDA work is justified.

## Diagnostics and claim boundary

Every later run must report attempted/applied teaching events, recruitments,
deletions, occupied output contacts per token cell, reservoir activity density,
co-winner rate, fitted-cell race latency, memory, and throughput. The
preregistered protocol also requires:

- pre-only, post-only, expired-trace, and unrelated-output locality controls;
- nonnegative/Dale-sign checks and an isolated signed-weight diagnostic only
  as a non-production ablation;
- finite-difference comparison of the local trace against the fitted-cell
  response under reset and adaptation, reported as approximation error rather
  than assumed e-prop equivalence;
- no-turnover, random, weak-contact, coactivity-stochastic, and top-eligible
  engineering turnover controls;
- candidate-pool and structural-timescale controls;
- a one-access external confirmation cohort after all selection is frozen.

See `docs/research/sparse-cell-output-readout.md` for the primary-source basis
and the limits of the biological interpretation.
