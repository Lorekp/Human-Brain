# Simple-SWE opening preregistration (v1)

Status: DESIGN — nothing opened, frozen, or run. VERIFY PASS #3 COMPLETE
(2026-08-23): every claim below cross-checked against sources; corrections
applied in place (see Verification log); the runtime-pin blocker is RESOLVED.
Provenance: researched by subagent through freeze/reveal/adapter/runner
sources and its own test-suite probing (transcript-mined after a final-answer
failure), authored by orchestrator.

## What the research established (verified facts)

- The intended freeze ceremony exists and matches this description:
  `create_freeze_manifest(*, protocol_path, checkpoint_path, source_root,
  output_path, python_executable=None)` (freeze.py:503-582) freezes protocol
  id + canonical SHA-256, checkpoint hashes (file/state/config/tokenizer),
  clean source tree (`git rev-parse HEAD` + SHA-256 of
  `git ls-tree -r -z --full-tree HEAD`, freeze.py:291-302), adapter/compiler
  identities, sandbox runtime (executable path + implementation + version +
  executable SHA-256 + flags `-I -S -B`), worker snapshot hash, simulation
  backend; written atomically as `freeze.json` OUTSIDE the source root
  (canonical name + location enforced at freeze.py:514-516).
- External anchoring of the evaluation seed is enforced by tests (all in
  tests/test_simple_swe_reveal.py): reveal refuses direct anchors without a
  public receipt (`test_reveal_rejects_a_direct_anchor_without_a_public_receipt`,
  :189 — an anchor event missing the public URI/receipt fields is rejected),
  requires nonempty https authority
  (`test_external_anchor_requires_a_nonempty_https_authority`, :210), and
  reveals nothing until externally anchored (:176). The seed is persisted to
  `run_root/private/seed-reservation.json` and appended to the hash-chained
  artifact ledger (`evaluation-seed-reserved`, `evaluation_tasks_generated`
  false) BEFORE any task generation
  (`test_seed_is_persisted_and_committed_before_any_task_generation`, :93;
  "committed" = tamper-evident ledger append, not a git operation), after
  revalidating a clean source tree before drawing entropy (:82).
  Crash-after-reservation resumes the same seed without reroll
  (`test_crash_after_seed_reservation_resumes_the_same_seed_without_reroll`,
  :127 — entropy drawn exactly once). NOTE: the earlier draft's
  "`perm_appendix.json`-style commitment discipline" cited a filename that
  exists nowhere in the repo; the real artifact is seed-reservation.json +
  ledger.
- Baselines are structurally first-class:
  `test_baseline_controls_cannot_precede_or_change_the_cell_summary`
  (tests/test_simple_swe_runner.py:349) asserts baseline controls CANNOT
  precede or change the cell summary, and the code enforces it independently:
  `record_baseline_control` (runner.py:642) requires exactly one finalized
  cell-summary event before any baseline control may be recorded.
- IMPORTANT environment finding from the agent's probing — RESOLVED as of
  this review; it was the only blocker. The dead agent hit Windows
  pymanager's `[ERROR] No runtime installed that matches ...` when resolving
  the pinned interpreter against the system-default Python 3.14
  (C:/Python314); the quoted string "AstralCPython3.11.15" was a garbled
  pymanager tag form. The sandbox contract never consults pymanager or uv:
  the protocol freezes language `python-3.11` (protocol.py:143-144),
  `resolve_python_executable()` (sandbox.py:306) uses the RUNNING
  interpreter's base executable, and `python_runtime_identity`
  (sandbox.py:349) refuses any major/minor mismatch loudly — reproduced in
  this review: probed with C:/Python314/python.exe against 'python-3.11' →
  `SandboxConfigurationError: sandbox Python must match protocol
  python-3.11, got 3.14`. On this host today CPython 3.11.15 IS installed
  three ways (uv-managed %APPDATA%/uv/python/cpython-3.11.15-windows-x86_64-none,
  the project .venv, and pymanager tag `-V:Astral\CPython3.11.15`),
  runtime resolution SUCCEEDS (resolves to the uv-managed 3.11.15 python.exe,
  identity probe returns cpython/3.11.15 + SHA-256), and the full simple-SWE
  test suite is green under `.venv/Scripts/python.exe -m pytest` (see
  Verification log). No interpreter install was needed. Remaining rule:
  invoke pytest/the ceremony via .venv/Scripts/python.exe — the bare
  `python` on PATH is 3.14.0 and fails BY DESIGN (red-test behavior).

## Freeze order (what gets frozen, in order)

1. DONE (verified 2026-08-23): host runtime pin resolves — protocol-pinned
   python-3.11 satisfied by the project .venv (CPython 3.11.15);
   test_simple_swe_freeze/reveal green locally (19/19) and the full
   simple-SWE suite green (117 tests, exit 0). Invoke everything through
   .venv/Scripts/python.exe.
2. `create_freeze_manifest` with: protocol path, grown checkpoint (from the
   a013 L1 bridge arm), clean source tree at the pinned HEAD, adapter +
   compiler identity hashes, sandbox runtime identity.
3. Evaluation seed: generate externally anchored commitment (https receipt),
   recorded to the hash-chained ledger BEFORE task generation; reveal then
   derives the cohort. (Ledger-enforced, not git-enforced; git-committing
   the receipt is recommended ceremony practice, not a code gate.)
4. Only then: task generation -> hidden-case sealing.

## Minimal capability question (v1)

After grow on M training tasks from visible families, pass@1 on K tasks
drawn from the SAME families (capacity gate) AND K' tasks from held-out
families (transfer gate), scored by hidden-case oracle execution in the
sandbox. All-or-nothing per family as everywhere else.

## Adapter path (verified end-to-end components)

checkpoint -> adapter (cell completion -> program text via compiler) ->
sandbox subprocess run with hidden cases -> oracle verdict -> artifacts.
Each module link re-verified against sources in this review (adapter.py:
CellCompletionAdapter/_load_cell_model accept both checkpoint formats;
compiler.py: RawPythonCompiler.compile is the identity pass-through with
emitted-tokens-only provenance; sandbox.py: worker subprocess launched as
`[executable, -I, -S, -B, worker]` under Windows Job Object limits with a
sanitized environment, decoding regression+property case results into an
ExecutionResult verdict; runner.py: run_cell_evaluation orchestrates manifest
re-verification -> bundle -> adapter -> executor=execute_completion ->
summary). No missing glue. Remaining gap is exactly the (now resolved)
runtime-pin discipline plus an end-to-end dry run that has never been
executed on this host.

## Preregistered baselines (before any cell run)

- B-random: uniform program sampler over the grammar space.
- B-template: heuristic template-filler keyed on task signature.
Both must be executed under the identical freeze/sandbox/oracle path; their
results are locked into the same evidence envelope BEFORE cell completions
are unsealed (enforced by the existing baseline-ordering test).

## Dependencies / sequencing

Hard prerequisites, in order: (1) representation choice frozen (codebook);
(2) ladder L1 bridge arm executed with the chosen readout — its checkpoint
is what gets frozen here; (3) host runtime pin fixed (DONE 2026-08-23, see
Freeze order step 1). Opening before (1)+(2)
would measure nothing but the current sealed failure mode again.

## Red tests

- Runtime-pin test: sandbox refuses wrong-interpreter hosts loudly (already
  exists; must pass locally before ceremony).
- Seed-commitment ordering tests (exist; must pass).
- Baseline-before-cell ordering test (exists; must pass).
- NEW: end-to-end smoke — one dummy frozen manifest + one synthetic
  checkpoint through adapter->compiler->sandbox->oracle on this host,
  asserted green, BEFORE the real opening.

## Verification log

Cross-review 2026-08-23 (verify pass #3), sources read at
C:/Users/laure/Desktop/DeepSim/Human Brain. All line numbers refer to that
tree.

1. create_freeze_manifest — CONFIRMED with one wording correction.
   Signature freeze.py:503-510 (keyword-only args). Freezes protocol_id +
   protocol_sha256 (NOT raw protocol bytes — corrected in the facts
   section), six checkpoint hashes/fields, source_head + source_tree_sha256
   from `git rev-parse HEAD` and SHA-256 over `git ls-tree -r -z --full-tree
   HEAD` output (_clean_source_snapshot, freeze.py:291-302), adapter id/hash,
   compiler identity, sandbox executor/worker hashes, python executable path
   + implementation + version + executable_sha256 + flags tuple
   SANDBOX_PYTHON_FLAGS = ("-I", "-S", "-B") (sandbox.py:23),
   simulation backend identity; `_atomic_create_json` (freeze.py:580) via
   artifacts.atomic_create_bytes; canonical name `freeze.json` and
   outside-source-root enforced (freeze.py:514-516).

2. Seed-anchoring tests — CONFIRMED, all quoted names exist and assert as
   claimed: test_simple_swe_reveal.py:189 (direct anchor without public
   receipt rejected, "anchor fields"), :210 (nonempty https authority,
   "public HTTPS URI"), :176 (reveal stops until externally anchored), :93
   (seed persisted + ledger-committed before task generation; bundle absent;
   evaluation_tasks_generated False), :82 (dirty source tree refuses entropy),
   :127 (crash resume, entropy drawn exactly once). CORRECTION: the draft's
   "`perm_appendix.json`" filename exists nowhere in the repository; the real
   commitment artifact is run_root/private/seed-reservation.json plus the
   hash-chained artifact ledger (reveal.py:76-150), and "commit" is a ledger
   append rather than a git commit.

3. Baseline-ordering test — CONFIRMED:
   test_baseline_controls_cannot_precede_or_change_the_cell_summary lives in
   tests/test_simple_swe_runner.py:349; code-level gate duplicated in
   record_baseline_control (runner.py:642+, requires exactly one cell-summary
   event first).

4. Runtime-pin blocker — REPRODUCED BOTH WAYS, THEN RESOLVED (no install
   needed).
   - Failure mode preserved by design: probed C:/Python314/python.exe
     against 'python-3.11' → SandboxConfigurationError("sandbox Python must
     match protocol python-3.11, got 3.14"); bare `python` on PATH is 3.14.0
     (C:/Python314).
   - The dead agent's exact error string could not be reproduced verbatim;
     its true source is Windows pymanager (`py` launcher): e.g.
     `py -V:AstralCPython/3.11.15 -V` → "[ERROR] No runtime installed that
     matches AstralCPython\3.11.15. Try running \"py install ...\"" (a
     malformed tag form). The correctly formed tag `-V:Astral\CPython3.11.15`
     lists as INSTALLED today, alongside the uv-managed
     cpython-3.11.15-windows-x86_64-none under %APPDATA%/uv/python and the
     project .venv (pyvenv.cfg: uv 0.12.3, version_info 3.11 → reports
     Python 3.11.15).
   - Empirical success: resolve_python_executable() under .venv resolves to
     C:/Users/laure/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe;
     python_runtime_identity(exe, 'python-3.11') returns {cpython, 3.11.15,
     sha256 6765c6b1685c86877fabe14d82240f3fab2913a617a85e8d09e61dc40798013b}.
   - Tests green on this host: tests/test_simple_swe_freeze.py +
     tests/test_simple_swe_reveal.py = 19 passed; full simple-SWE set
     (adapter, sandbox, runner, protocol, artifacts, public_evidence,
     reveal_bundle_schema, freeze, reveal) = 117 collected, exit 0 under
     `.venv/Scripts/python.exe -m pytest`. The earlier claim that "several
     freeze/reveal tests fail on this host for environment reasons" no
     longer holds.

5. Adapter->compiler->sandbox->oracle path — CONFIRMED, no missing glue.
   adapter.py CellCompletionAdapter/_load_cell_model (both checkpoint
   formats human-cell-brain-checkpoint-v1 and recurrent-human-cell-checkpoint-v3)
   -> compiler.py RawPythonCompiler.compile (identity pass-through,
   algorithm_source="emitted-tokens-only") -> sandbox.py _run_worker_process
   ([executable, *SANDBOX_PYTHON_FLAGS, worker], Windows Job Object limits,
   sanitized env, CREATE_NO_WINDOW) -> ExecutionResult outcome/passed from
   regression+property case decoding -> runner.py run_cell_evaluation
   (manifest re-verification at seal boundary, bundle validation, ledger,
   executor=execute_completion) -> CellEvaluationSummary. The end-to-end dry
   run remains unexecuted and is still required before opening (see Red
   tests).
