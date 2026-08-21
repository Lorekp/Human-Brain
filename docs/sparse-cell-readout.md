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

A source-wide post-`observe()` consumer audit found that sparse production
paths proceed only through `race()`, `strengthen()`, the public eligibility
snapshot, or the eligibility-based activity diagnostic. Sparse current and
teaching both consume that mirrored eligibility array. No production caller
reads host voltage, after-spike current, threshold, refractory, or recurrent
synaptic current after a sparse CUDA observe. Developmental growth explicitly
rejects any rule other than dense `three_factor`, so its plasticity recorder
retains the full mirror. Recurrent checkpoints persist topology and long-term
readout anatomy, not transient reservoir state.

This registration does not imply that sparse current or plasticity runs on the
GPU, nor is it a capability result. The separate synthetic 200,000-active-cell
timing protocol in `benchmarks/sparse_cuda_engineering_protocol_v1.json`
profiles kernel, transfer, fitted-output, and teaching phases before deciding
whether any additional CUDA work is justified.

### Registered 200,000-cell engineering result

The unchanged protocol SHA-256
`4e9b61790cd1dfbdbad2309070b22298407a0f73712108cbe3cfc6beb40322ce`
ran three independent synthetic repeats on an NVIDIA GeForce RTX 4070 SUPER.
Both measured implementations were committed and clean before execution.

| Median metric | Full mutable mirror | Eligibility mirror |
| --- | ---: | ---: |
| implementation | `de00ad8` | `14bc5df` |
| inference tokens/s | 127.265 | 158.177 |
| online-teaching tokens/s | 125.877 | 149.996 |
| inference kernel tokens/s | 178.724 | 186.150 |
| device-to-host bytes/token | 12,000,008 | 1,600,008 |
| host-to-device bytes/token | 0 | 0 |
| construction seconds | 2.223 | 2.796 |
| host persistent arrays | 35.381 MiB | 35.381 MiB |
| device persistent arrays | 26.353 MiB | 26.353 MiB |

The transfer volume fell 7.5x (86.67%). End-to-end inference improved 24.29%
and teaching improved 19.16%. Sparse current itself was only 70.6 microseconds
per call in the baseline median, below 1% of inference time, so the registered
decision rule does not justify moving fitted-output current or plasticity onto
CUDA.

The final run configured, activated, and updated all 200,000 reservoir cells
on each of 100 integration steps/token: no dormant padding was counted. It
reported 167 inference and 168 training spike events across 64 measured tokens,
with 168 unique training-sequence spiking cells and reservoir activity density
0.00084. Online teaching applied 64 updates and 64 recruitments with zero
deletions. The sparse readout occupied 1,052,672 bytes versus 409,600,000 bytes
for the dense control, a 389.105x ratio. These low activity counts are reported
as an architecture diagnostic, not hidden behind the updated-cell count.

The immutable reports are
`artifacts/sparse_cuda_engineering_v1_baseline.json` and
`artifacts/sparse_cuda_engineering_v1_optimized.json`. They contain no text,
held-out examples, prediction accuracy, or other capability result.

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
