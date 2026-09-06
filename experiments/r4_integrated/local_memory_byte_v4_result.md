# Local memory/byte v4: valid timing, cost still above budget

2026-09-06. Precise duration measurement and the strict saved-only audit pass.
The full acquisition remains **not admitted**: the unchanged conservative
formula projects **995.972 seconds**, above 900 seconds. Memory fits at
**774.773 MiB**. V4 is closed; no full fit or capability result follows.

Frozen source: `e25bb2a9901a96e832a4018beac967ddc4882d6a`.
The [cost decision](../../benchmarks/local_memory_byte_v4.cost_decision.json)
binds the [recipe](local_memory_byte_v4.md), source hashes, packages, native
library, schedule, clock and saved completion/launcher evidence.

| Arm | Maximum complete sampled lesson | Additional capped prediction |
| --- | ---: | ---: |
| Tagged | 154.512 ms | 12.068 ms |
| Aggregate | 66.066 ms | 9.735 ms |
| Writer/maps frozen | 58.143 ms | 11.083 ms |
| Fully frozen | 54.377 ms | 9.651 ms |
| Writes disabled | 58.823 ms | 12.291 ms |

Learning projects to 914.939 seconds, other work to 51.033 seconds, plus the
unchanged 30-second finalization reserve. Outer execution took 27.015 seconds.
Every capped sample is strictly positive. Parent, worker and admission agree
on `perf_counter` / `QueryPerformanceCounter()` with 100 ns clock resolution;
Unix deadlines remain separate. No zero floor or adjusted margin is used.

The disposable workload contains twenty first attempts, 200 teacher byte/EOS
offers, 160 enabled decoder update opportunities, 100 action attempts and two
motor demonstrations. Saved tensor state remains 124,109,512 bytes. All weights
from this engineering run are discarded; this is not an acquisition test.

The independent saved-only audit verifies 1,132 source hashes, five artifact
hashes, clock metadata, owned process identities, initial/frozen weight chains,
zero frozen credit, shared source-free memories, 102 execution receipts and
100 workspaces. No model or child is reexecuted. Its receipt is
`.pytest-tmp/local-memory-byte-v4-cost-independent-audit.json`, SHA-256
`a734b0b1646e3873514e18ab1d3ff7fc84e8273dcc6e2b357ef0be8c6cd8991c`.

The correct timer resolves v3's measurement failure; v3 stays invalid. The
current implementation is materially cheaper than the original 2,013-second
projection, but the remaining 96-second gap is not waived. The next bounded
candidate removes query learning traces that read-only prediction discards.
It must preserve supplied-state and actual forward-state validation, predictions,
teacher updates and all exposure. Its explicit inference contract need not
construct or reject overflow in absent, unused eligibility tensors; that is a
declared error-behavior change, not universal exception parity. Require a
measured complete-workload return before any fit.
