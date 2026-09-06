# Local memory/byte v2: optimized implementation return

Status: prepared 2026-09-06; no cost admission or acquisition result yet.
This is one implementation successor to the closed
[v1 cost rejection](local_memory_byte_v1_result.md). Its observed failure was a
2,013.103-second conservative projection against a 900-second ceiling, primarily
from tagged ingestion arithmetic and repeated validation. The reusable hypothesis
is that equivalent local arithmetic with less temporary work permits the same
supervised acquisition workload within its original budget.

## Unchanged experiment contract

The complete [v1 recipe](local_memory_byte_v1.md), its local writer-credit
equations, five arms, seed 733, dimensions, learning rate 0.01, deterministic
1,024 lessons per arm, inference caps, teacher clocks, and all 100 final actions
plus two motor demonstrations apply unchanged. No source filtering, selected
answer ingestion, extra lessons, seed/rate sweep, optimizer, clipping or backward
pass is introduced. The teaching schedule digest remains
`5aa2b390194d28f8f58e8f9cc8ce0c0f1e0b6721ba848177101448b5564a14ac`.

Each lesson ingests both complete documents before the question or target, makes
a read-only first answer, then receives explicit byte/EOS correction. Weights
from profiling, cost admission and earlier experiments are discarded. Final
memory pools support both keys, same-query/different-memory checks, 32-query
read-only delays and incremental interference ingestion. The fresh worker
receives detached tensors and query-only requests; sources and target rosters
remain outside its model input. Retained tags remain counted source-derived
state. Checkpoint and weight immutability checks remain in force.

The byte learner has no pretrained encoder or LLM. Its existing occurrence motor
uses host-authored boundary and candidate drives as a narrow action-schema
scaffold before a physical race. Exact execution therefore does not establish
entirely neuron-only action selection. This is byte-supervised acquisition and
narrow action transfer, not execution-feedback learning, recovery, independent
prose comprehension, broad intelligence or an integrated gate pass.

## Implementation and admission changes

For `frozen` and `writer_maps_frozen`, ingestion now uses the same forward
writer dynamics without computing writer eligibilities that these arms never
consume. Allocated credit tensors retain the same shapes and byte accounting,
but remain zero. These control states intentionally differ from v1's unused
tags; do not claim whole-state parity. Fast synapses, predictions and all
enabled query/decoder updates must match the full-credit reference. Public
boundaries retain finite checks, tensor ownership and purged writer activity.
`writer_maps_frozen` rejects external re-enabling of controller/write-map
learning before ingestion, query, teaching or snapshot; fully frozen already
disables teaching irrespective of parameter flags. No future writer update may
consume the intentionally absent historical credit. Main tagged and aggregate
learning rules remain unchanged.

The successor imports v1's unchanged model factory, lesson, detached-state,
digest, export and scoring/actuation helpers. The run/worker/action orchestration
is copied because the frozen v1 binds its child to its own file and directly
equates the Windows virtualenv redirector PID with actual Python. The successor
uses its own worker path and verifies actual Python, redirector and worker in
the original launcher's observed owned process tree. That launcher checks
ancestry and creation time before recording process identities; unrelated PIDs
and repeated identities for a reused PID are rejected. This remains sampled
process evidence, not an OS sandbox.

The source manifest includes both frozen and successor runners/recipes and the
existing full dependency closure. Freeze reviewed code, exact commands, packages,
native identity, roster hash and QA receipts before costing. Prepare native
identity and launch in the same environment. No v1 admission or cost result
authorizes v2. Review floating-point parity before freezing optimized arithmetic;
no reduction-order change may silently inherit exact-equivalence claims.

## One bounded cost phase, then one conditional acquisition

Allow one disposable phase of at most 900 seconds total, including any failed
identity/preflight attempt; all attempts remain preserved. It runs the same
four longest selected lessons per arm, forced full-length prediction, complete
final ingestion, checkpoint serialization, source-free reload, 100 actions,
two motor demonstrations and finalization. Run on an otherwise idle machine.

Use the exact original prospective estimate:

```text
2 * sum_arm(1024 * (maximum complete sampled lesson + capped prediction))
  + 2 * (outer cost elapsed - sum all sampled lessons)
  + 30 seconds finalization
```

The capped-prediction term is intentionally additional even when a sampled
lesson already emits the cap. `project_cost` recomputes this formula and the
maximum of conservative parent-plus-worker peaks and sampled live tree peaks.
Admit acquisition only at <=900 projected seconds and <=2,048 MiB. A lower
kernel-only time does not qualify. The run rechecks the artifact hash chain,
source/package/native/schedule identity, owned-tree command and recomputed cost;
an admission's unsubstantiated estimated numbers cannot unlock training.

The cost launcher takes a prepared `engineering_qa` admission containing v2
`CONFIG`, source manifest, revision, packages, native identity and schedule hash:

```powershell
$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'
$env:HUMANBRAIN_DEVICE = 'cpu'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$v2Admission = (Resolve-Path '.pytest-tmp/local-memory-byte-v2.cost-admission.json').Path
$v2Hash = (Get-FileHash -LiteralPath $v2Admission -Algorithm SHA256).Hash.ToLowerInvariant()
$v2Revision = git rev-parse HEAD
$v2Output = Join-Path (Get-Location) '.pytest-tmp/local-memory-byte-v2-cost'
./experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1 -Seconds 900 -MaxMiB 2048 -LogBase $v2Output -ChildArguments @('-X','utf8','-m','experiments.r4_integrated.local_memory_byte_v2','cost','--output',$v2Output,'--admission',$v2Admission,'--admission-sha',$v2Hash,'--revision',$v2Revision)
```

After a passing saved-only audit and recomputed projection, write a new
`admitted` record binding `completion_path/sha256`, `launcher_path/sha256`,
`projected_seconds` and `peak_upper_mib` under `cost`, with the same frozen
identities. Launch the same command with `run`, the new admission, and fresh
`local-memory-byte-v2-run` output/log paths. It has its own 900-second ceiling;
there is at most one acquisition, with no warm-start from cost weights.

Require tagged 8/8 fit, 4/4 new-value, 4/4 delayed and 4/4 interference exact
executed outputs, plus strict new-value advantage over writer/maps-frozen,
fully frozen and writes-disabled. Aggregate superiority is separately necessary
to attribute a benefit to address tags. Publish every arm and distinguish first
attempts, teacher corrections, STOP/error and execution counts. A failed cost,
nonfinite learning, failed fit or no useful advantage closes this successor and
triggers explanation review; no automatic extra practice or threshold change.
