# Parent session checkpoint and restart

Status: selected source implementation; [ordinary QA passes](persistent-session-checkpoint-qa-v1.md)
after one test assertion repair. The original failed run is preserved.
This adds persistence to future broader runners. It neither reconstructs V6 nor
authorizes a retry, workload, checkpoint load or learner execution. The
[roadmap](roadmap-reading-to-coding.md) owns integration and release.

`experiments/r4_integrated/persistent_session_checkpoint.py` implements one
explicit store. Existing learner, journal, executor and frozen experiment sources
are unchanged. Models remain caller-owned; the store accepts the strict combined
sequence-assembly API, including its recognition-derived cache preparation, and
does not select or construct a neural architecture.

## Boundary semantics

Create the store **before** the future initializer or initial ingestion. `create`
authenticates source and manifest and writes/fsyncs initialization intent zero.
It does not retrospectively protect initialization performed before that call.
The first commit records the initialized idle state with zero current-session
exposure counters. Existing learned experiences and reference-history counters
may be nonzero and are preserved separately.

For each subsequent append or attempt, call `begin` before any associated work.
For an attempt, retain the existing order: actual prediction/execution, durable
attempt row, scheduled correction or acknowledgement, actual file/feedback carry,
then checkpoint. All combined and reader pending states must be idle. An attempt
row awaiting its correction is not a safe boundary. An append gets its own commit
before the following prediction, preventing repeated ingestion after restart.

The store commits only the exact next episode/route/attempt cursor. A scheduled
append requires one ingest row; an attempt requires one attempt row and, when the
episode has a non-null teacher, one teacher row. The correction binds the actual
attempt's LF-inclusive hash and frozen offered teacher. Route controls are fixed:
online and simple receive enabled corrections; learning_disabled receives the
same offered correction with its learning flag false. Private correctness never
selects a boundary, changes the cursor or supplies a recovery decision.

## API and concrete host record

```python
store = PersistentSessionCheckpoint.create(
    checkpoint_directory,
    source_root=repository,
    evidence_root=run_directory,
    manifest_path=frozen_manifest,
    manifest_sha256=expected_manifest_sha256,
    source_revision=reviewed_commit,
    source_sha256=reviewed_source_inventory,
    session_id=public_session_identity,
)
# Only now construct/initialize the two actors and restore the reference history.

frontiers = store.capture_frontiers(
    {"attempts": ["segment-0/attempts.jsonl"],
     "teachers": ["segment-0/teachers.jsonl"],
     "ingests": ["segment-0/ingests.jsonl"]},
    execution_root="executions",
)
receipt = store.commit(models, states, baseline, host, frontiers)

# Before the next attempt; append intents instead use kind="append" and no hash.
store.begin("attempt", host["cursor"], observation_sha256=actual_observation_record_sha256)
# Predict/execute/correct or acknowledge once, flush evidence, update host, commit.

reopened = PersistentSessionCheckpoint.open(
    checkpoint_directory, config_sha256=store.config_sha256)
checked = reopened.inspect()  # No tensor decoding or neural import.
restored = reopened.restore_into(fresh_models)
```

`models`, `states` and `fresh_models` have exactly `online` and `learning_disabled`
keys. `baseline` is the existing SimpleReference. `restore_into` returns states,
baseline, host, frontiers, boundary SHA, elapsed seconds and cumulative helper work.
It never predicts, teaches, appends a document or invokes an initializer. The
caller constructs each selected fresh actor once and starts no task before restore
returns. Exact saved class identity, mode, seed, state keys and metadata must match.
Fresh action and reader experience counters must both be zero before loading.

Initial host shape:

```python
host = {
    "cursor": {"episode_index": 0, "route_index": 0,
               "attempt_index": 1, "append_done": False},
    "projects": {route: {} for route in ("online", "learning_disabled", "simple")},
    "current": None,
    "counters": {route: dict.fromkeys(COUNTERS, 0)
                 for route in ("online", "learning_disabled", "simple")},
    "elapsed_seconds": 0.0,
    "budget_seconds": declared_total_seconds,
    "budget_mib": declared_peak_limit,
    "initial_bindings": {"initialization": initialization_sha256,
                         "baseline_restoration": baseline_restoration_sha256,
                         "prior_outcomes": prior_outcomes_sha256},
}
```

The exported `COUNTERS` are attempts, executions, teacher_occasions, reader_offers,
action_labels, ingest_documents, ingest_raw_bytes and ingest_eos. Each is an exact
nonnegative integer for each of the three routes. The store checks their increments
against durable event rows. `initial_bindings` values are SHA-256s; the future runner
owns the scientific sufficiency and authentication of the selected prior evidence.

After every attempt, `current` is exactly `{route, episode_id, files, feedback}`
for its actual outcome. A project's entry is `{episode_id, files, feedback}` under
`projects[route][project_id]`; update it after each attempt, including an intermediate
retry. Other projects remain unchanged. Appends change neither map nor current
workspace. Complete executor snapshots allow sixteen files of 4,096 bytes, so an
unsupported next neuronal observation need not be silently truncated. Actual
feedback preserves both raw streams (512 bytes each), signed return code and five
outcome flags. Incomplete actual file snapshots cannot form a resumable boundary.

The manifest supplies ordered `episodes`, each with `episode_id`, `attempt_limit`,
`append_document_ids` and optional `teacher`. This is the current broader-runner
schedule contract, not an arbitrary callback scheduler. The final cursor is
`{episode_index: len(episodes), route_index: 0, attempt_index: 1, append_done: False}`.
The intent's observation hash uses the runner's canonical `encode(observation)`
bytes including LF; it is not RawObservation's distinct domain-separated neural
custody hash. Parent journal rows retain their actual route, episode, attempt,
observation hash, prediction presence, model/memory hashes and execution fields.

## Saved state and atomic publication

Each changed neural route uses its existing `snapshot(state)` to clone all model
tensors/metadata and every tape row. Immutable tensor blobs contain only model
and memory. The small boundary separately stores exact combined and reader
experience counters, allowing tensor reuse when only free-response custody advances.
Unchanged model/tape digests reuse a prior authenticated blob. No counter is inferred
from a hardcoded session length or reconstructed by replaying responses.

Both routes, simple chronological documents/demonstrations/offer counters, actual
project/retry state, exact cursor, resource totals, source and prior bindings are
referenced by one boundary. The simple history preserves raw feedback-framed queries,
anchors, offered replacements and source/query digests. New history must extend
the prior record only through that route's actual append or correction. Old lessons
are not delivered again through `observe_teacher`, `predict` or `append_documents`.

Payloads are written exclusively to temporary files, flushed/fsynced and renamed
to unique immutable names. A complete immutable boundary manifest references their
hashes and sizes. A separately flushed temporary HEAD is replaced atomically in the
same directory. That pointer is the only publication decision. Failed temporary
files and older generations remain; the first implementation performs no garbage
collection. A small private live intent token prevents a reopened process from
finishing an interrupted transaction merely because it finds its intent file.

Journal frontiers bind ordered segment paths, exact complete byte-prefix SHA,
byte count and LF count. This binds every preceding record; no redundant last-record
hash scan is claimed. Actual execution files **and directories**, including empty
directories, are inventoried. The caller must flush journals and reap action children
before capture/commit; this module does not itself wait on or terminate a child.
The scientific auditor still owns actual-execution qualification and private grading.

## Clean and dirty restart

`inspect()` authenticates config/source, manifest, chosen HEAD/boundary/intent,
payload bytes and journal/execution frontiers without deserializing tensors.
An intent beyond HEAD, journal tail or extra execution file/directory raises
DirtyCheckpoint. `inspect(require_clean=False)` returns the last authenticated
boundary with explicit dirty reasons; it does not make that state a clean restart.
An interrupted initialization with no HEAD has no committed prior to return.

`restore_into` requires a clean frontier, reads bounded trusted local payload bytes,
rehashes the exact byte strings supplied to `torch.load(weights_only=True)`, and
validates both incoming models/tapes/metadata and derived state before the first
model copy. Existing checked `load_state_dict` publishes each fresh actor. The
reader and action counters are assigned exactly, not via a monotonic `max` that
could conceal a mismatch. Rebuilt views/wiring and both tapes must have independent
storage. If any later load/validation fails, both fresh actors must be discarded;
they are not exposed as a partly restored live session.

The derived caches are validated once during all-route prevalidation and rebuilt
again by the existing strict public loader. Both actual builds are charged. The
store does not bypass the loader to avoid this cost. No pending response, borrowed
free-response token, prefix scratch or Python object identity is serialized. A
restart gives new physical instances: the future runner must record a new segment
and qualify response IDs by segment while preserving learned lineage and counters.

An interruption after an intent can have consumed prediction, action or correction
work even without an attempt row. It therefore forbids automatic same-stream retry.
An update published in memory but not in HEAD may be unavailable; at most the current
transaction is outside the prior committed boundary under this cadence. A separate
authorized continuation may select that older safe state, but must retain the extra
exposed attempts and costs and cannot call itself an uninterrupted scientific run.
Missing or changed committed bytes reject restoration rather than being repaired.

File fsync and same-filesystem replacement address process termination; this is
not a proven Windows power-loss/directory-flush guarantee. Validate all referenced
objects after interruption. New evidence segments preserve exclusive journal
creation and bind old prefixes instead of truncating or rewriting old evidence.

## Accounting and future runner responsibilities

The helper reports actual read/hash/write/fsync counts, payload writes/reuses,
snapshot tensor bytes, restricted load calls, reference snapshots and strict
reader/cache validation work. `_actors` forwards the shared work dictionary into
the current combined `_check(work)` path. Existing `snapshot()` also performs
internal validation without a work argument: the helper names/counts these
uninstrumented validation passes and clone bytes, rather than claiming every
legacy internal operation is enumerated.

Work in the immutable boundary is a snapshot before final serialization/HEAD
publication. The returned operation receipt includes the remaining helper work
and elapsed time. Complete-run accounting must include this tail, constructors,
restore, every source/receipt hash and all retained/temp disk bytes. Live models,
one snapshot's clones, tensor serialization and restore's two decoded route states
plus rebuilt cache overlap belong in peak memory. The store is not an RSS monitor.

The caller retains final monitor receipts for all segments and interrupted starts.
Its cumulative budget includes checkpoint and restart overhead and must not reset
on reopening. Persisted elapsed time is captured before the current commit finishes;
monitor evidence must cover that tail before admitting a resumed segment. Never
restore a previous process's perf_counter deadline. Boundary reserve/voluntary stop
and complete-session admission remain future-runner integration work, not an
automatic guarantee that a long next response can finish before the wall cutoff.

## Ordinary QA scope

Two additions in the existing byte-sequence test file have completed ordinary
verification; the [QA report](persistent-session-checkpoint-qa-v1.md) owns actual
launches, the assertion repair, cumulative exposure and evidence:

* Ten pure parent cases use opaque bytes that are never tensor-decoded: clean HEAD,
  outstanding intent, partial journal, complete attempt before correction, uncommitted
  payload, uncommitted manifest, published HEAD, changed source, corrupt payload and
  an unexplained empty execution directory. These synthetic publication fixtures
  make no claim of neuronal state validity.
* One actual continuation fixture constructs four actors (two original, two fresh
  restore), ingests `a` once per original route (two raw bytes/two EOS total), executes
  two actual STOP children, then offers `1` plus EOS once per route (two enabled and
  two disabled symbols, six action labels). Three commits produce three payload writes
  and three reuses. One restore makes two restricted fixture tensor loads; no external
  experimental checkpoint is loaded. Four subsequent paired reading predictions
  plus the two action predictions give six free calls. Exact numerical state, tape,
  counters, independent caches, next trajectories, actual files/feedback and reference
  history are compared. The fabricated prior reference history is transport data,
  not another teacher or reference prediction.

These are ordinary durability/continuation checks, not a skill gate or
authorization to rerun the exposed V6 stream.
