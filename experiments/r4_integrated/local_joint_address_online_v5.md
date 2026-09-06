# Joint-address online v5: stdlib actions and held codec responses

Selected integration of the [runtime return](../../docs/research/local-online-runtime-return-v1.md).
Inherit the complete [v4 contract](local_joint_address_online_v4.md) except for
the two explicit changes below. V4's complete cost and audit passed their checks
but rejected the 1,886.974-second projection; no full acquisition ran. Preserve
that rejection and its cost weights. This version starts fresh and receives no
old endpoint or backpropagating reference.

## Execution and learning boundary

Use `stdlib_python_action_bridge_v1.py`, source SHA256
`3bacf37d703af9c6685a4ece8b554cc6779597073fa94adae2e6004cc8dd50c1`.
The neuronal parent stays in the selected development venv. Each actual action
uses the separately authenticated base CPython with `-I -S -X utf8 -B`, a copied
environment with `__PYVENV_LAUNCHER__` removed, and the unchanged regular-source
bootstrap. Keep real files, a fresh OS process, strict UTF-8 byte capture,
ten-second timeout, error/exit handling and complete file verification.

This deliberately supplies the standard library and authored sibling modules;
it does not implicitly provide project or development dependencies. Future
dependency-bearing tasks need a declared provisioned environment. Base prefixes,
absent automatic site/hooks and removed launcher are qualified behavior, not old
venv parity. Process-state isolation is not a filesystem/security sandbox.

`action_execution` has exactly five fields: `qualification_path`,
`qualification_sha256`, `launch_record`, `bootstrap_policy` and
`bootstrap_sha256`. The single eight-key stdlib qualification authenticates the
new record, including its separate parent runtime identity and fixed artifacts.
Load and verify it at preparation and completion. Per-action receipts bind its
record hash, exact command, bootstrap identity and explicit removed environment
key. The source-free request retains its four fields: source hashes, cases,
action launch-record hash and bootstrap hash, assembled before one exclusive
write. The worker validates and echoes these identities; they never enter the
neural computation. Old runtime/bootstrap manifests are historical prerequisites,
not interchangeable new-environment qualifications.

Use main learner SHA256
`2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8`,
schema `local-neuronal-joint-address-held-codec-learner-v1`. At each free or teacher
call, evaluate the held codec on every occupied sensory slot once. The first
decision uses this bank; later decisions gather responses through the actual
EOS-absorbing cursors. Rebuild after every call, including after publication;
do not retain the bank across an action or lesson. Prefix mixtures, current-prefix
local errors, ordered accumulation, actual X-before-B address activity, casts and
all-group finite publication remain unchanged. No target-dependent response
cache, hidden backward error transport, autograd or external learner is added.

For N occupied slots and D actual decisions, each free/teacher call counts N
codec evaluations, 257N output potentials and N normalizations. Its later
decisions gather N(D−1) response rows, 1028N(D−1) scalar values and
7196N(D−1) payload bytes. Query-only delay reads have no codec work. Other
address, mixture, cursor, teacher and update-opportunity counts keep their
original meanings. The 128-slot response bank occupies 921,088 bytes; bank plus
old/new gathered responses can overlap at 2,763,264 bytes. Other validation,
address, publication and Python receipt temporaries remain counted by the
process-tree monitor. These tensor figures are not RSS bounds.

## Fixed workload and acceptance

Keep ordered arms `learned`, `address_frozen`, `slow_frozen`, `writes_disabled`,
seed 733, learning rate .01 and 1,024 lessons per arm. Every one of 4,096 first
attempts executes the actual generated action before external grading and the
same instance/state's supervised three-position correction: 12,288 teacher
byte/EOS offers. Execution grade is evidence, not a claimed numerical reward
input to learning. Keep all eight fit, four novel-value, four delayed and four
interference cases per arm: 80 final actions plus two supplied motor demos,
4,178 total full execution receipts.

The disposable cost uses four lessons per arm: 16 first attempts/actions and
48 teacher offers, all 80 final actions, two motor demos, one supplied accepted
`ab`/EOS action at ordinal 1023 and four ordinary 127-period-plus-EOS cap fixtures.
There are 99 cost executions. The probe is not a neuronal prediction or success;
its complete action time guards every arm's sampled action maximum. Cost and
full retain the same four models and 48 tapes, 33,801,664 persistent tensor bytes.
No full-only checkpoint increment or altered source-memory roster is introduced.

Preserve the scoped gate: compliant local learning and same-instance executed
acquisition, 8/8 fit, 4/4 novel, 4/4 delay and at least 2/4 interference, with novel
superiority over all three matched controls. Eighteen correct outputs preserve
the narrow workload but do not satisfy the original 20/20 qualification gate.
The supplied motor and explicit supervised corrections remain limitations;
there is no general SWE or exclusive-neuron coding claim.

## Prospective complete-cost admission

Stream complete lesson/action/final JSONL rows with the inherited line/tree
guards and bounded pair lifetimes. Model M includes validated ingestion/free
prediction and, after action, teacher work and hashes. Action A includes actual
execution, external grade and action-row conversion/write/flush/hash. Journal J
includes complete lesson conversion/guards/write/flush/hash. These timers are
disjoint. Saved audit pair timing starts before either row's read/parse and ends
after all validation; fixed final-artifact checks stay in its residual.

For each arm let m, a, j be separate sampled maxima and c its complete recorded
cap-prediction time; replace a by max(a, supplied accepted-probe A). Keep exactly:

```text
learning = 2 × 1024 × sum_arms(m + a + j + c)
other = 2 × (cost outer time − sum measured lesson M/A/J)
run projection = learning + other + 30
audit projection = 2 × (1024 × sum_arms(max pair audit time)
                        + finished cost-audit outer time − sum measured pair audit time)
final time projection = run projection + audit projection
```

The actual probe time also remains in `other`; the pessimistic replicated probe
maximum does not remove its cost. Use the unchanged 3-MiB streamed-row allowance,
128-MiB run reserve and 32-MiB audit reserve. Run peak projection uses the larger
of sampled parent-plus-worker peaks and observed owned-tree peak, plus 131 MiB.
Audit projection uses the finished cost-audit owned-tree peak plus 35 MiB.
Their maximum must be at most 2,048 MiB. These remain engineering estimates;
the watchdog enforces actual limits.

Preliminary cost checking leaves admission false until the authenticated finished
audit receipt and monitor supply its time and peak. Only a final projection at
most 900 seconds and 2,048 MiB admits one fixed fresh full run and saved audit;
their actual aggregate must meet the same ceilings. No identical timing retry,
sample exclusion, weakened ceiling, omitted accepted probe or exposure change
follows rejection. Both changes are integrated together, so a future aggregate
speed difference cannot causally isolate either component.

## Completed prerequisites and remaining freeze

The stdlib qualification passed nine authored tests: one identity child, seven
semantic pairs and two independent-process contamination children, 17 children
total and no neuronal/motor calls. Its eight-key manifest is
`.pytest-tmp/stdlib-python-action-v1-qualification.json`, SHA256
`4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c`.
It binds raw/pass receipts, JUnit and archived tested source. The bootstrap is
unchanged, SHA256 `9d4239f54f38a520d3da269be86a3f3f58c901be92b9f05d0a66b6ad4bd11c7e`.

The fixed authored timing comparison passed all semantics and both case medians;
its aggregate new/old ratio was 0.8949510386891091, below the prospective .90
limit. Twelve fresh children ran with fixed alternating order and no warmup,
repeat or omitted sample. Pending medians were 53.0101 versus 57.8297 ms;
corrected medians were 53.5667 versus 58.0082 ms. The report is
`.pytest-tmp/stdlib-python-action-timing-v1/report.json`, SHA256
`d8ac9291bd61841f9fa0de73e05c8879e91a46f6949f204315851b9db1be9bf8`.
The monitor recorded 2.7144727 seconds and 79.44921875 MiB. This earns integration,
not full-workload admission or a claim about v4's slowdown.

Held-codec QA passed 17 initial cases and one added nonzero-sensory free-cursor
case, all first attempt. The initial ledger records 22 model constructions plus
four disposable clones, 22 ingestions, 25 valid free calls/49 decisions, one
query-only traversal and 17 teacher traversals/71 offers, including three offers
in the injected atomic rejection. The added case contributes one construction,
one ingestion, one free call/five decisions and five standalone codec evaluations
of 20 rows; no teacher or action. Invalid inputs rejected before activity remain
separate. Exact parity uses only the preserved immediately preceding no-backprop
main source, never an old adjoint learner. JUnit receipts are
`.pytest-tmp/held-codec-v1-qa.xml` and
`.pytest-tmp/held-codec-v1-nonzero-free-qa.xml`; source extension custody is
`.pytest-tmp/held-codec-tested-source-extension.json`.

V5 reuses the exact single-publication export and same-instance lesson/worker
bodies, with only explicit runtime/configuration/qualification changes at the
wrapper. Source comparisons cover these inherited paths; no redundant model or
export fixture is required for unchanged bodies. Source review, final lint,
independent auditor alignment and a clean freeze precede new cost execution.
