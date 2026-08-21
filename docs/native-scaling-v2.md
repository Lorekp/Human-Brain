# Exact native scaling v2

The v2 CPU simulator keeps the fitted human-cell equations, all persistent cell
and synapse state, sparse recurrent events, and the 100 integration steps per
token unchanged. It adds three exact execution strategies behind native ABI 3:

- the frozen v1 serial kernel, retained only as a bit-exact oracle;
- a dense AVX2 kernel with deterministic Rayon cell chunks and ascending-order
  recurrent scatter;
- a quiescent frontier that scans every configured cell once per token, then
  integrates only cells whose complete checkpointable state is nonzero or that
  are reached by the current afferent/recurrent events.

The frontier is not an approximation or dormant-cell padding. A cell with zero
voltage, after-spike currents, thresholds, refractory counter, synaptic current,
and eligibility cannot change before it receives an event, so skipping its
zero-to-zero integration steps is algebraically and bitwise exact. Newly reached
cells join the frontier before their next eligible integration step. Spikes are
still scattered in ascending presynaptic cell order, preserving floating-point
accumulation order.

Frontier speed depends on activity density. Scaling artifacts therefore report
configured cells separately from cells actually updated, and report the full
state scan, frontier construction, and integration phases independently. The
million-cell low-activity result must not be read as million simultaneously
nonquiescent cells; the density stress artifact measures that boundary directly.

## Memory and construction

The recurrent token path no longer keeps an eight-byte step-current copy for
every cell. A one-byte spike mask, one-byte frontier membership mask, and
four-byte frontier index replace the former step-current and compact-spike
workspaces. Readout synapses remain the dominant persistent array at large cell
counts; the local learner removes ridge's quadratic statistics but does not yet
factor the output-by-reservoir readout matrix.

`uniform_no_self` now samples the same candidate indices as the registered v1
implementation and maps around the source cell, preserving seeded topology
arrays exactly without allocating an almost-population-sized temporary per
source. Structured group, mixed-membership, 1D, 2D, layered, and 3D builders use
direct group members or bounded coordinate offsets plus sparse long links. That
structured change is an explicit topology-protocol v3 break; old topology v2
artifacts remain tied to their original candidate-weighted builders.

## Build and third-party licenses

The Rust crate is no longer dependency-free. It uses Rayon 1.12 for deterministic
dense cell chunks. The locked transitive set is Rayon, Rayon Core, Either, and
Crossbeam Utils/Epoch/Deque; each is licensed `MIT OR Apache-2.0`.

The normal Python content-addressed build invokes `cargo build --release
--locked` and may need network access the first time those locked crates are not
present in Cargo's cache. A reproducibility machine with the locked crates
already cached can verify an offline build with:

```powershell
cargo build --release --locked --offline --manifest-path src/humanbrain/_native_src/Cargo.toml
```

The benchmark reports circuit construction separately from native compilation;
workers load the cached native library before starting the construction timer.

## Reproduced results

All token measurements below include Python-to-native dispatch and exactly 100
fitted 0.05-ms GLIF5 integration steps. The dense rate is the deterministic
five-thread AVX2 path and therefore reports actual integrated cell-steps rather
than configured-cell equivalents.

| Configured cells | Frontier token/s | Max cells updated/token | Dense token/s | Dense cell-steps/s | Construction | Persistent arrays |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 98,304 | 843.58 | 2,466 (2.51%) | 79.15 | 778.1 M | 0.838 s | 209.76 MiB |
| 131,072 | 754.17 | 2,004 (1.53%) | 61.53 | 806.5 M | 0.976 s | 279.67 MiB |
| 393,216 | 449.83 | 937 (0.24%) | 25.92 | 1,019.0 M | 3.095 s | 838.92 MiB |
| 1,048,576 | 193.36 | 544 (0.052%) | 3.59 | 376.7 M | 8.702 s | 2,237.04 MiB |

The 64-token, 98,304-cell comparison is bit exact against the frozen v1
kernel: topology arrays and spike counts match, and both persistent states hash
to `1cd1c50a5cab36b828a63e8ee25c9ec0f46ce71f854b7124d17add1b88c23366`.
The exact frontier ran at 789.59 token/s versus 17.70 token/s for v1 (44.62x).

Activity-density stress testing prevents the sparse natural trajectory from
being mistaken for a high-density result:

| Configured cells | Updated fraction | Frontier token/s | Dense token/s |
| ---: | ---: | ---: | ---: |
| 131,072 | 1.02% | 568.02 | 69.07 |
| 131,072 | 5.02% | 176.15 | 63.27 |
| 131,072 | 20.02% | 51.50 | 64.43 |
| 131,072 | 100% | 11.71 | 45.69 |
| 1,048,576 | 1.00% | 53.25 | 3.92 |
| 1,048,576 | 5.00% | 5.27 | 3.56 |
| 1,048,576 | 20.00% | 3.00 | 4.05 |
| 1,048,576 | 100% | 1.41 | 4.13 |

Every density result is byte-identical to dense execution from the same seeded
state. The >100 token/s gate holds at 131,072 cells through roughly 5% updated
cells, not at 20%. At one million cells, even 1% updated cells falls below the
gate; the 193.36 token/s natural result is specifically a 0.052%-updated result.

The direct structured builders produced six million edges for every one-million
cell topology without self-edges or duplicate targets. Construction ranged from
15.74 s (`single_scale_groups`) to 48.43 s (`volume_3d`), used 22.89 MiB for
the target array, and had measured 100k-to-1M time exponents from 0.888 to 1.114
and an output-memory exponent of exactly 1.0. Group quotas and bounded-local plus
long-range spatial-link invariants were checked in both tests and the artifact.

Reproducibility inputs and captured outputs are:

- `benchmarks/native_scaling_protocol_v2.json` and
  `artifacts/native_scaling_v2.json` (protocol SHA-256
  `ef54dc44a40eb14f23c50ddd85554fe0aec304340eea1513d0546ee2c9404545`);
- `benchmarks/native_activity_density_protocol_v1.json` and
  `artifacts/native_activity_density_v1.json` (protocol SHA-256
  `a9e8d8fceaad6b895c9b277342f903e3d709e33454d4fc623e64a6f69105b5a4`);
- `benchmarks/structured_topology_protocol_v3.json` and
  `artifacts/structured_topology_construction_v3.json` (protocol SHA-256
  `2e56a4c30a64e2cf23178281c1ae1d052c148f505a49e19a143ad5b82edf702c`).

Run the matching benchmark script from the repository root to reproduce an
artifact. Machine timings will vary; the scripts embed the exact protocol hash,
environment, target-array hashes, equivalence checks, and phase timings needed
to distinguish performance drift from semantic or topology drift.
