# Source-absent memory and byte action development v1

Engineering preparation, 2026-09-06. Disposable composition and timing passed;
no 256-step joint fit has run. Root must freeze and admit the exact runner,
config, roster and source hashes before `run`. This is separate from the
completed answerability experiment.

`source_absent_memory_byte_v1.py` joins learned byte embeddings (257×32),
PlasticMemory's 64-cell document writer and 64×64 synapses, a 32-cell byte-query
controller, and a 64-cell ByteSequenceDecoder. The query state already has the
memory's 32 input coordinates. Decoder context is query state plus the 64-value
memory read. Original UTF-8 documents are processed byte by byte with delimiter
256; no host key/value parsing, answer extraction or pointer inventory is used.
Slow parameters learn through explicit offline supervised decoder loss and the
differentiable writer/read path. Fast writes then use those learned write rules
without answers. This is conventional BPTT over rate cells plus plastic fast
synapses, not a claim of biologically local slow learning.

## Fixed proposed tiny-fit schedule

Eight neutral records, one narrow grammar: `Record kNN has value <text>.`
Values include spaces, an apostrophe and UTF-8 café. Each of 256 steps samples
four fixed-seed two-document episodes and one associated query per episode.
For EACH episode, sample two distinct keys and independently sample two distinct
original values, zip them, then select one of the two pairs for the question.
Thus key-to-value assignments vary across episodes; a fixed question lookup
cannot solve the training distribution. This pre-admission correction removes
a shortcut in the initial draft; no joint-fit outputs informed it. The exact
256×4 episode schedule and canonical SHA256 are saved in the external teacher
roster and bound by admission; all three updating arms receive the same batch.
The same batch goes to three updating arms: 768 optimizer steps and 3,072
supervised episodes total. Adam .001, clip1, no dropout, seed733; byte CE includes
EOS. There is no early stopping or extra practice after inspecting outcomes.
All stages must fit 900 seconds and 2 GiB across the owned process tree.
Root's outer launcher covers Python startup, fresh-worker execution and final
serialization, recording sampled process-tree peaks. The runner additionally
checks the sum of its peak and the worker peak as a conservative upper bound.

- `full`: all slow parameters train; online fast writes enabled.
- `writes_disabled`: same initial arrays, slow optimizer/schedule and query/output
  interface; writes are disabled both during offline fit and later ingestion.
  Its controller may advance, but memory reads must remain zero from zero M.
- `frozen`: same initial full arrays; zero slow optimizer updates, online fast
  writes enabled. It is a frozen-learning control, not a no-write control.
- `controller_bypass`: slow parameters train on the same schedule; ONLY the
  document writer uses memoryless tanh (no recurrent U). The recurrent query
  controller, recurrent output decoder and plastic write/read rule remain.
  All common initial parameters match. This diagnoses the document writer's
  recurrence, not query/output recurrence or a fully cell-disabled system.

No pretraining or pretrained encoder is added. The shared occurrence motor has
its existing two explicit executed demonstrations; those teach the actuation
schema, never the reading questions or values. All generated answers use the
same ordinal `Configure qaNNN` instruction, with no label-dependent routing.

## Source removal, output and checks

After fitting, ingest raw documents under `no_grad`, purge writer state, and
save detached CPU network state plus eight named synapse snapshots per arm. Only
tensor weights/synapses, zero writer tensors and primitive architecture metadata
are allowed; no optimizer, gradients, autograd graph, byte inputs, documents,
targets, candidate lists or retrieved value store enters this archive. The
separate parent teacher roster retains sources/answers for external scoring.

A fresh isolated Python process loads the archive and a query-only request file
from a separate directory. The worker never calls the fixture author or trainer;
its network receives only query bytes and learned state. State/source hashes
and exact before/after tensors authenticate reload and read-only generation.
This is an auditable routing boundary, not an OS-level file-access sandbox:
the executed script still contains offline authoring constants. The precise
claim is that its worker does not invoke the roster or supply documents/targets
to the network. It is not a claim that source-code constants are inaccessible.

Twenty checks per arm are fixed before fitting. Eight trained-value queries use
four shared two-record pools. Four unseen-value queries use two shared pools:
one containing k00 then k01, the other k03 then k02. BOTH keys of each pool are
queried from its exact same restored state object, giving target positions
first/second/second/first. Those same four queries after 32 unrelated read-only
queries reuse the original two state objects without rewriting or reloading.
Two interference snapshots derive from those purged novel-pool snapshots plus
three unrelated document writes (k04, k05, k06); both keys are again queried
from each shared resulting state. The worker receives only pool IDs and query
bytes: pool IDs select a shared two-answer state, never a target-specific state.
No base document is re-ingested to prepare the interference snapshots.
Unseen values are new byte combinations,
not a new language/task distribution. Delay checks test state persistence only;
the interference condition tests those four novel values under later writes,
not retention of every old skill or recovery from action feedback. No expected value enters
worker requests, and no difficult case is dropped, rewritten or retrained.

Greedy generation caps at32 bytes plus the decoder's final EOS decision. Only
EOS-terminated nonempty valid UTF-8 may be actuated; invalid UTF-8, missing EOS
or empty output stays a separately recorded STOP/failure. There is no spelling,
field, answer-length or semantic repair. Decoder exact ties use its declared
lowest-index byte policy. Lossless Python quoting is fixed transport scaffolding.
The existing occurrence motor edits its selected literal; strict UTF-8 child
execution preserves raw stdout/stderr and decoys. Scoring compares generated
bytes with the parent target only after generation/action, requiring exact
stdout bytes and unchanged decoys. All80 attempts plus2 demonstrations count.

Tiny fit requires full8/8 executed answers. Report every control and every
unseen/delay/interference count separately: source-absent fresh-value evidence
requires full4/4 unseen AND a strict advantage over writes-disabled/frozen;
claim controller contribution only with an actual advantage over bypass.
Delay/interference preservation requires4/4 in each lane. A failed tiny fit or
any failed transfer condition is retained and closes this candidate; no extra
seed, rate, steps, source filter or host answer selector follows automatically.
No public-exam run, broad comprehension, generalization gate or replacement of
the persistent reading-to-action milestone is claimed by these narrow checks.

## Verified preparation and final admission

The original scratch receipts remain unchanged. Composition reproduced all 80
untrained outputs after fresh reload, with shared states and parameters exact.
All eight randomly generated action checks were non-emitting. A separate,
explicitly parameter-authored decoder fixture emitted `a` then EOS after reload
and passed the real motor/strict child; it demonstrates transport only.
Timing performed 1 warmup + 3 timed four-episode updates per updating arm,
using two 31-byte documents, a 25-byte query and a 9-byte target plus delimiters
and EOS. These 12 disposable steps are not the proposed 256-step fit. Full
33-decision generation costs used EOS-suppressed disposable copies only for
measurement, never for learning or scored generation.

The conservative corrected projection is 289.325 seconds; parent peak 342.625 MiB
and worker peak 201.449 MiB sum to 544.074 MiB. It includes 768 updates at twice
per-arm slowest timed step, full 80-query/ingestion/reload charges, capped
byte generation, 82 action children, serialization and an explicitly
overinclusive allowance for native/import/finalization work. Cost evidence:
`.pytest-tmp/source-absent-cost-admission-v2.json`; composition and positive
transport receipts are linked there. This estimate does not replace live guards.

`source_sha256` combines the existing native runtime source manifest (including
native build inputs) with explicit imported motor, bridge, memory, decoder,
resource and runner sources. Admission also pins the exact Git revision, torch
and numpy versions and existing `loaded_library_identity()` record. The same
native identity is checked before learning and after action execution.

Root creates a JSON admission with exactly the reviewed `config`,
`source_sha256`, `source_revision`, `packages`, `native`,
`teaching_schedule_sha256`, and `status="admitted"`;
additional preparation receipt references may be included. No prepare command
implicitly authorizes a fit. The runner requires the admission SHA externally.

Under the outer launcher, invoke the selected checkout Python as:

```text
.venv/Scripts/python.exe -X utf8 -m experiments.r4_integrated.source_absent_memory_byte_v1 run --output <absolute-new-output-directory> --admission <absolute-admission.json> --admission-sha <SHA256> --revision <frozen-commit>
```

Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`,
`HUMANBRAIN_DEVICE=cpu`, and checkout-local `HUMANBRAIN_NATIVE_CACHE`, `TEMP`, `TMP`
before launch. The reviewed launcher is `run_source_absent_memory_byte_v1.ps1`; it supplies
`HUMANBRAIN_RUN_START_UNIX` and `HUMANBRAIN_RUN_DEADLINE_UNIX` from its own clock.
It owns process-tree termination on a deadline
or memory breach. It preserves log/output files and records final exit status, elapsed wall time,
sampled tree peak, root/descendant PIDs, launcher hash and command (including
revision/admission SHA). It does not inspect report/completion hashes: the runner
completion binds those artifacts and the saved-only auditor verifies their
joint identity with the launcher receipt. Sampling every 500 ms may miss brief
children; this is not a hard operating-system memory limit. No retry is allowed.

Output is a new directory: `initial_weights.pt` (all four initial slow states),
`report.json`, `completion.json`, an offline teacher
roster, retained action workspaces and `source-free/` containing only weights/
synapses, query requests and generated output. Worker PID/peak/elapsed are in
its output and primary report. Scientific failure still returns exit 0 if the
whole workload completes; execution/admission/finalization failures return 1.
`report.json` alone is provisional: require matching passed completion and
outer launcher receipts. Primary write costs are included before completion;
only the tiny final completion write is covered by the outer launcher alone.
Original exceptions and completed rows are preserved; finalization failure is
reported separately. No optimizer, source input or teacher label enters the
source-free checkpoint/request. The parent retains the declared teaching and
scoring data; source absence is the fresh worker's model interface boundary.
