# Standalone neuronal splice harness design v1

Source-only implementation contract, 2026-09-06. No runner implementation, model,
checkpoint load, forward, action or test occurred in this review. Root selected
four arms, 32 TRAIN tasks repeated eight times (256 lessons per arm), and 48
endpoint tasks per arm (32 fresh and 16 retention). Exact roster, sample IDs,
category/cap fixtures and criterion still require protocol freeze. This note
does not admit acquisition. Graph discovery failed with Transport closed; known
source paths were read directly. Mem0 returned unrelated development history,
which supplies no scientific evidence here.

## Small standalone module

Use one new runner with explicit `prepare`, `cost`, `run` and `worker` entry
points. Import the actual `NeuronalSpliceLearner`, raw `SpliceWorkspace` /
`SpliceProposal` / `apply_splice`, and qualified
`temporary_module_action_bridge_v1.execute_utf8`. Reuse only the small generic
`JsonlJournal`, `host_tree_bytes` and `read_row` utilities where their source
contracts fit the new spec. Bind their import closure; do not invoke old runner
factories. Write a short new lesson summary and grading function instead of
inheriting motor-specific lesson/configuration functions. No motor, escaping,
`actuate`, `Case.files`, `pending` replacement, old reference arm, or function
global mutation belongs in this module.

Suggested concrete seams are `load_reader_prior`, `make_models`,
`verify_proposal`, `learning_experience`, `compact_prediction`,
`compact_feedback`, `export_endpoint`, `worker`, and
`project_cost(spec, report, launch, *, audit=None, audit_launch=None)`.
Keep these ordinary functions; no generic execution framework is needed.

## One authenticated reader prior, four persistent learners

Bind the completed v6 decision, full audit/checker, completion and source-free
checkpoint as one prior manifest. The saved completion names
`source-free/source_free.pt` with SHA
`03654f61af4cccb93688a75e75a84c089800672530cdaffdc7a30f7cc5181180`;
its own SHA in the decision is
`e4c4836cca69a29dc071097bd4b8afc473f6c521c59731a421be4d462d071b6d`.
The full audit is
`358a7a45c01de58d8d3392717b742a26a91ce63fba123ad639fa8d8e6bfdd0a5`.
These are existing JSON declarations, not a new checkpoint verification. At
admitted preparation/run boundaries, authenticate the entire chain and bytes
before `torch.load(..., weights_only=True, map_location='cpu')`. Reject a partial,
cost, old-schema or non-learned source. Select only the saved `learned.weights`.

Construct four current combined models at seed 733 in exact order `learned`,
`action_frozen`, `slow_frozen`, `memory_disabled`. Each has a fresh action circuit.
The reader schema is already the v6 held-codec/no-backpropagation schema. Validate
the source metadata exactly against that schema in learned mode, and validate
the complete tensor-key set, shapes, dtypes, CPU placement and finite values.
Build each destination reader state from its own metadata and replace only its
tensor fields with clones of the authenticated source tensors. Require that the
sole source/destination metadata difference is the declared control mode; load
strictly through the current reader API. This explicit mode adapter does not
rewrite the artifact or silently relax its loader. Compare all four initial
reader tensors to the source exactly, and all action tensors to the same fresh
private-seed initialization. Record per-group hashes and the adapter policy.

Never construct an old learner or call an old teacher to load this prior. Do not
transfer v6 endpoint tapes by accident: fact storage is a separate, explicitly
selected input. Cost and full may deliberately load the entire authenticated
prior payload for equal custody scope, including unused entries; count those
bytes while live, then release the container after copying selected tensors.

Each arm keeps the same combined object through all 256 lessons. Tapes are
immutable episode states addressed by exact raw document identity, not a growing
cross-task tape that would exceed 128 slots. The frozen roster must state the
observation schedule and retention mapping: first observation constructs a tape
through `model.ingest`, later specified reuse keeps that exact state, and any
specified reobservation is a separately counted ingest. No implicit memoization
may remove scheduled input exposure. The memory-disabled arm uses its normal
ingest path with the same layout and zero keys/sensory. Teacher corrections must
leave every tape unchanged. New raw requests reach reader and action paths
unchanged; the transferred reader is not guaranteed to understand this changed
distribution.

## Actual action first, local correction second

For ordered `(round, task, arm)` experiences, retain only the current immutable
observation, fact state and compact receipts. Hash model/tape, call `propose`,
and pass its actual proposal to `apply_splice`. Valid SPLICE applies exactly
those raw bytes once. STOP or rejected SPLICE retains the original files, with
the rejection recorded. Root selected exactly one qualified verification child
in all three cases; unchanged verification is not a replacement edit or a second
neuronal choice. No new execution-envelope API is required.

`verify_proposal` uses a constant neutral `Case` solely because the frozen
executor signature requires it. Its expected value/module do not enter inputs,
selection or the new grade. Ignore legacy `passed` and `decoys_unchanged` fields.
Independently check actual materialized file bytes, exact selected raw splice or
unchanged rejection/STOP, child return/output and task-specific protected bytes.
Only after the child returns may the host read the teacher/task grade fields.
Flush the complete action row before invoking `teach` on the same pending object,
same model and pre-action tape/workspace. Bind PID/instance, attempt ID,
request/workspace/proposal hashes, model-before/teacher-before/model-after,
memory hashes, exact action-line hash and ordered completion/teacher timestamps.
Execution grade is recorded evidence; the numeric learning input remains the
declared supervised correction. STOP offers only operation; SPLICE offers three
action heads and its literal payload plus EOS. Never pad these scopes.

Full regular counts are 1024 acquisition proposals/verification children and
192 endpoint proposals/verification children, hence 1216 children before any
failure interruption. Separate STOP, accepted/rejected SPLICE, zero/nonzero
return, timeout, output decoding failure and task success. Count reader offers
from actual teacher operations/payload lengths and delays from actual traversals.
No motor demonstrations or accepted-`ab` motor probe remains.

## Source-free endpoint worker and export

Export one complete checkpoint exclusively, with exact combined weights and the
roster's required tapes per arm, plus a teacher-free request manifest. Requests
contain task ID, full raw request, raw workspace, state identity and declared
delay reads; no target operation/span/payload, expected output or grade fields.
Publish the complete request once, including checkpoint/source/runtime/bootstrap
bindings. Count every tape, including explicit duplicate clones if retained;
deduplication cannot be assumed from repeated document text.

The isolated current-source worker owns each endpoint's complete
`propose -> apply/verify -> finish_without_teaching` loop. This avoids exporting
a live pending ticket or fabricating feedback in the parent. It loads one current
combined model per arm from the exported state and keeps that instance for all
48 tasks. No teacher/optimizer/roster targets enter its package. Hash weights and
tapes before/after the whole arm and each action. Stream paired prediction/action
rows, consume pending, release the pair, then continue. The parent waits for
worker completion and independently grades the saved action rows against the
host teacher roster, without another action or neural query. Keep parent grading
streamed. Worker completion binds both logs, EOF/count/order, package, checkpoint,
request and runtime/bootstrap hashes. Source-free means no teacher-bearing task
files supplied to the worker; actual process execution remains conventional.

## Receipts, timing and admission

Use the companion compact receipt contract: full mixture/posterior and initial
belief, action logits/indices and corrected-prefix identity; validate repeated
priors/cursors before omitting them. Never journal feature banks or local-update
matrices. Apply prospective 1 MiB line/tree guards and fixed text/raw-output
bounds on writer and reader. Preserve overflow evidence and fail instead of
truncating a successful record. One authored maximum-width serializer fixture
must precede freeze. Bound each pair function's lifetime to two trees/two lines;
parent and worker combined allowance is common E=12 MiB, with 128 MiB run and
32 MiB audit reserves. Tensor/allocator memory is additional.

Use disjoint lesson M/A/J. M is the sum before and after the real action: begins
before validation, scheduled ingestion, hashes/delay reads and proposal; resumes
for teacher preparation/correction and invariant checks. A includes exact apply
or rejection, child verification, external grading and action journal conversion,
guards/write/flush/hash. J includes compact consistency checking/conversion,
lesson guards/write/flush/hash and scalar summary updates. Record clock metadata.

Propose four fixed regular cost task IDs per arm, only after the final roster
establishes category/geometry coverage. Hubble is specifying four disposable
maximum-geometry experiences and four shared action categories; they are separate
authored cost exposure. Cost also executes all 192 endpoint actions and uses the
same prior/export/tape scope as full. Do not derive a probe from failed random
early behavior. Let effective maxima M_a/A_a/J_a be the maximum regular sample
and the prospectively applicable cap/category measurements. With L=256, cost
outer C and regular sample sum S:

`P_run = 2*256*sum_a(M_a+A_a+J_a) + 2*(C-S) + 30`.

All actual cap/probe cost remains in the residual even when guarding a maximum.
Finished cost audit pair timers start before reading either line and end after
all pair checks. With its outer U, regular pair sum S_u and per-arm pair maximum
u_a, `P_audit = 2*(256*sum_a(u_a) + U-S_u)`. Full endpoint/prior/checkpoint checks
are common fixed scope, included in residual. Preliminary admission is pending
and false until a passed finished audit and authenticated monitor are supplied.
Require `P_run+P_audit <=900`; no identical retry or post-cost bound relaxation.

For equal cost/full checkpoint scope, F=0:
`P_run_mem=max(cost parent peak+worker peak,cost owned-tree peak)+12+128 MiB`;
`P_audit_mem=finished cost-audit peak+12+32 MiB`.
Otherwise explicitly price the full-only tensor delta before cost. Require their
maximum <=2048 MiB. Audit/setup same-payload requirements include the transferred
prior and complete final checkpoint, not merely four scalar model summaries.

The independent checker imports no project model and performs no forwards. It
authenticates initial prior equality/fresh action initialization, mode metadata,
control/fixed tensors, tape reconstruction, journal custody, all-boundary indices,
free/corrected prefixes, operation-specific teacher counts, local-work arithmetic,
actual application/output grade, full EOF and launch/command/PID creation binding.
It reconstructs softmax/NLL and prefix consistency from saved vectors; exact
unsaved local parameter arithmetic remains established by source and ordinary QA,
not claimed to be replayed from hashes. Full admission reauthenticates passed
cost, both auditor identities, completed audit launch and exact finalized
projection. Root freezes criterion and roster before runner implementation.

## Settled observation and reading-panel scope

Sartre's concrete map resolves the provisional tape discussion above: retain 32
TRAIN packages once for eight-round reuse; 20 fresh packages serve the 32 fresh
endpoints; the 16 retention endpoints reuse TRAIN IDs 00-03, 08-11, 16-19 and
24-27. Four fresh states belonging to two delayed value-swap pairs are observed
before acquisition; the other 16 fresh states are observed after acquisition.
There is no repeated endpoint source ingestion.

Root additionally selected 20 original v6 raw-question reader queries before and
after acquisition for every arm: 160 reader-only predictions, zero additional
actions or teachers, max two bytes (at most three decisions each). Freshly ingest
the 12 authenticated original v6 document packages once per arm and reuse these
immutable states for both panels; do not import populated checkpoint tapes. This
uses the same reader already inside each persistent combined learner, with no
action/pending ticket. Memory-disabled receives the same raw observations through
its zero-storage ingest path. Bind the old document/question mapping to the
authenticated v6 roster; do not execute old code to regenerate it.

The concrete total is thus 64 retained states per arm, 256 states across four
arms: 168,167,424 tape tensor bytes plus 2,537,728 combined-model tensor bytes =
170,705,152 bytes, excluding initial snapshots, the separately loaded prior,
raw workspace metadata and transient arrays. Cost must retain/export/load this
same full state scope and run both complete reading panels, keeping F=0 honest.
The before panel runs in the acquisition process; the after panel can run in the
endpoint worker on identical exported reader weights/tapes without teachers.
These fixed 160 reads belong in cost and full residual work, with their own
query/codec/byte counters and streamed receipts. Full verification children stay
1216. The protocol must explicitly keep any retention/query target data out of
the worker and grade these saved raw outputs in the parent.

Hubble's cap correction is required: memory-disabled cannot emit a changing
byte/EOS sequence from identical zero sensory rows with held codec weights.
Its authored maximum-length free cap is therefore 16 bytes plus LIMIT (17
decisions); enabled-sensory caps may end in EOS. All four corrected cap teachers
can still offer 16 bytes plus EOS. Never force a mid-call parameter change to
manufacture a missing control behavior.

## Teacher-free dependency boundary correction

Source review identifies no existing clean journal module. The exact small
definitions to extract, without importing their old owner, are
`host_tree_bytes` (local_joint_address_online_v1.py:143), `JsonlJournal` (:157)
and `read_row` (:469). A new `experiments/r4_integrated/receipt_io.py` needs only
`hashlib`, `json`, `math` and `sys`. Preserve those function/class ASTs initially:
the `spec` argument is only a supplied object with a limits dictionary, never an
import of the parent spec class. No ARMS, roster, target, teacher, model factory,
LessonJournal, default spec, ROOT/sys.path mutation, or legacy runner import
belongs in support. The parent supplies a plain limits value; the worker may
wrap authenticated limits in a stdlib SimpleNamespace to preserve the original
API. Both sides import this support directly. Exact extraction is ordinary
source reuse, not an alias referring back to old function globals.

The earlier worker-owned-executor recommendation needs one further correction:
the frozen temporary bridge imports the stdlib bridge, which imports the direct
bridge; that imports `neutral_document_python_session.Case` and
`real_input_action_bridge`. The neutral session defines historical TRAIN
constants at module scope. Therefore extracting journal code alone cannot make
the full executor import closure teacher-free. Do not copy that closure into the
neuronal worker or claim that unused teacher constants are absent.

Smallest recommended boundary: the existing qualified executor remains in the
parent, while a separate `neuronal_splice_worker_v1.py` imports only the current
learner/transport modules, teacher-free receipt support, torch and stdlib. The
worker does not import the parent runner even for configuration, grading or
roster functions. Endpoint request JSON carries exact allowed fields and primitive
limits, not a serialized parent object. Its source package is an explicit allowlist
of these teacher-free modules and their actual transitive imports; no whole
experiments directory, old bridge, parent runner, roster, or archive is included.
The direct humanbrain imports inspected here are library definitions rather than
experiment tables. A final static dependency walk and negative package-inventory
check must verify that claim at source freeze; do not load modules to discover it.

Use one synchronous request/result exchange per endpoint, not another interpreter
per neural proposal. Worker proposes on its persistent model/tape and publishes
the compact prediction with attempt/proposal/workspace/state hashes. Parent reads
it, validates/applies that exact proposal, and performs the selected one actual
verification. Parent flushes the action row and replies with only the exact
execution envelope plus its line hash. Worker validates the envelope through
`finish_without_teaching`, clears its own still-live pending object, and acknowledges
completion before the next request. Teachers, external task grades and corrected
spans never travel back to the worker. This preserves the same pending instance;
it never rebuilds a prediction object in another model. Timeout/EOF/mismatched
reply fails closed with partial records retained. One pair remains live per side;
the common 12 MiB allowance already accounts for both sides concurrently.

Bounded stdin/stdout JSON lines or equivalent exclusive per-case request/reply
files are sufficient; choose one before implementation. Prefer stdin/stdout pipes
with explicit flush and strict next-message identity, reserve stderr for bounded
diagnostics, and apply the same preparse line guard. Parent starts its complete
endpoint action timer before reading the prediction and ends after action-row
flush/hash and receipt reply; worker model/receipt timers remain explicitly
separate. All 192 endpoints occur in both cost and full so this IPC cost stays
inside the measured common residual. Independent audit checks both ordered
streams, exact envelope/line hashes, worker completion acknowledgements and
before/after unchanged weights/tapes. No extra child or teacher is introduced.

This amendment supersedes worker-owned verification above. Root selected this
parent-side qualified execution boundary after source review. It avoids modifying
or requalifying the frozen executor and avoids carrying historical teacher modules
into the neuronal package. The prior tensor adapter remains as specified: authenticate all
old metadata, permit exactly the selected mode difference, and count the entire
loaded historical checkpoint payload while live.

The exact inspected local worker dependency set, beyond the new worker and
receipt support, is `humanbrain/__init__.py`, `neuronal_splice_learner.py`,
`neuronal_splice_action.py`, `local_memory_learner.py`,
`neuronal_sequence_memory.py`, `byte_sequence_decoder.py`, `plastic_memory.py`
and `multiscale_opaque_event_tape.py`. The latter contains the existing raw file
value type. External imports are the selected torch/numpy dependencies and stdlib.
Some library files define unused historical classes; no historical instance or
backpropagation path is constructed or called. The package initializer's lazy
language exports are not accessed. No experiment runner is required by this
local closure. Capture this explicit file inventory and hashes, verify actual
import origins against the supplied isolated source copies, and reject accidental
imports of any old runner/bridge or parent roster. This is an authenticated
input/source boundary, not a claim that a Python process has filesystem sandboxing.

Choose the bounded pipe exchange concretely. Each endpoint has three messages:
`proposal` from worker, `executed` from parent, `acknowledged` from worker. All
carry schema, monotonically ordered arm/task index and attempt ID. The proposal
contains the actual compact prediction and its immutable observation identities.
The executed message binds its exact action-line hash and contains the learner's
four-key execution envelope; the inner `execution` mapping contains only
`returncode`, `stdout_hex` and `stderr_hex` copied from the actual qualified
receipt. Never return legacy `passed`, expected stdout, teacher, external grade,
task success or corrective file/span. The complete execution identity/command/
runtime/bootstrap receipt stays in the parent action journal for independent
audit. Worker calls `finish_without_teaching` with the exact envelope, then
acknowledges the same action-line hash and unchanged model/tape hashes. Parent
requires that acknowledgement before advancing. No pending token is reconstructed
or restored across processes.

Use binary pipes and the same finite line/tree limits, one flushed line per
message, exact schemas and fail-first EOF/order handling. Child stderr is a
separately bounded diagnostic stream and never a protocol channel. Parent and
worker retain only the current proposal/envelope/acknowledgement; append their
scalar/hash records and release before the next task. Count all serialization,
pipe reads/writes, process waiting and acknowledgement validation in measured
endpoint time, not an omitted setup term. No action grade returns to the worker;
all endpoint qualification and retention scoring remains in the parent auditor.
