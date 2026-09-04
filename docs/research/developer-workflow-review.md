# Developer workflow review

Implementation follow-up: the [development workflow](../development.md) now owns
setup, worktree creation, explicit public QA tiers, debug receipts, and the
Windows CI configuration. The findings below describe the pre-change audit;
use the implementation guide for current commands and remaining coverage limits.

Status: workflow audit and proposals, 2026-09-04. The workflow audit changed no
setup, worktrees, services, or evaluation configuration. The accompanying code
cleanup separately passed 59 focused tests, including native-backed regressions;
it did not run protected evaluations. Configuration structure was inspected
without credential values. Findings describe the working tree at review time,
including unrelated ongoing edits.

## Ranked changes

### 1. Separate bootstrap, focused QA, and experiment execution

**Verified:** [README.md](../../README.md), lines 134–143, installs `.[test]`
and immediately runs the entire test directory. [pyproject.toml](../../pyproject.toml),
lines 43–45, supplies only `-q` and `testpaths`; it defines no tier markers.
Optional coverage depends on the machine: [test_cuda_pipeline.py](../../tests/test_cuda_pipeline.py),
lines 12–14, skips without CUDA, and [test_ladder_ridge_backend.py](../../tests/test_ladder_ridge_backend.py),
line 17, skips without SciPy. This is a discoverability/coverage gap, not evidence
that the suite currently fails or that every test accesses protected material.

**Proposal:** publish one small QA matrix before adding any runner framework:

| Tier | Trigger and scope | Evidence |
| --- | --- | --- |
| Bootstrap | New checkout: install, verify interpreter/import path, tiny public unit test, then explicit native preflight if needed | Resolved checkout and interpreter, versions, native identity or actionable prerequisite error |
| Focused | Each change: affected existing tests and relevant native tests | Exact selection, pass/fail/skip counts, time, revision/diff identity |
| Integration | Reader/action changes: reusable public development fixture through reading, retention, actual Python execution, fresh transfer, and no-update control | Same persistent lineage, real exit/output, decoy integrity, learning advantage and resource cost |
| Broad regression | Before merge or scheduled local maintenance; separate CPU/native, optional SciPy, and CUDA lanes | Full selected-suite report, explicit unavailable lanes, no protected evaluation execution |
| Adjudication | Only the separately frozen scientific protocol and its prerequisites | Independent evaluation evidence; never a side effect of setup, CI, or ordinary debugging |

Start with explicit existing test node/file lists; add registered markers only
after inspecting their contents and dependencies. A filename or `public` prefix
alone is insufficient to classify a safe test. Do not advertise the integration
row as implemented: the [balanced plan](balanced-evidence-to-action-session-v1.md)
still identifies that bridge as pending.

### 2. Give each worktree a reproducible bootstrap and ownership record

**Verified:** setup uses an editable install and floating minimum versions for
Python, NumPy, pytest, and setuptools ([pyproject.toml](../../pyproject.toml),
lines 1–13). No Python lock/constraints file or Rust toolchain pin was found in
the bounded tracked-config inventory. [conftest.py](../../tests/conftest.py),
lines 7–10, prepends this checkout's `src`, so passing pytest alone does not
establish that the installed package resolves to the intended checkout.

**Proposal:** select an explicit committed base, record worktree/branch/owner and
expected files, then create the worktree's own `.venv` and editable install.
Verify `git rev-parse --show-toplevel`, `git rev-parse HEAD`, interpreter path,
and `humanbrain.__file__` before tests. Never reuse another worktree's editable
environment. A fresh worktree from `HEAD` excludes current uncommitted changes;
decide the intended base explicitly rather than silently losing that context.
Keep a small known-good environment record or constraints snapshot for repeatable
development; retain compatibility ranges in package metadata.

The cleanup's runtime checks illustrate the gap: system `python` resolves to
`C:\Python314\python.exe` without pytest, while this checkout's `.venv` has
pytest and NumPy but no Ruff module. The `ruff` executable on PATH comes from a
separate Hermes environment. Declare lint tooling with the development setup and
invoke tools from the chosen environment. Default Ruff also attempted to write
the parent repository's cache and received access denied; `--no-cache` worked.
Use an explicit checkout-local cache in a future setup revision.

The existing [Hermes launcher](../../tools/hermes/README.md), lines 24–26 and
49–64, already requires a clean exact base. Reuse that discipline without making
Hermes or its external-source authorization flow a prerequisite for local work.
Before teardown, inspect commits, changes, and ownership; no automatic cleanup
of unknown worktrees or scratch data is proposed.

### 3. Make native preflight and cache provenance visible

**Verified:** [native.py](../../src/humanbrain/native.py), lines 606–683, builds
with a locked/offline release plan and binds toolchain/environment plus absolute
source root into the identity. Lines 750–754 default to the system temporary
cache; lines 863–896 attest the cached binary. Therefore identical source in a
different checkout can have a different cache key; copying a DLL between
worktrees is not a valid bootstrap shortcut. `loaded_build_info()` at line 1055
can build/load the kernel and is not a read-only diagnostic probe.

**Proposal:** make native compilation a named bootstrap step with its own error
output and elapsed time, using existing `build_native()`. Record source digest,
resolved DLL, toolchain identity, and CPU/device choice with every native test
report. Use the supported `HUMANBRAIN_NATIVE_CACHE` override for intentional
cache placement; preserve attestation and existing configuration rejection.
Keep debug/profiling outputs separate from the production cache and freeze.
Avoid simultaneous first builds against the same cache identity until build
serialization is verified; this audit did not establish a race failure.

### 4. Offer a public-development debug entry point

**Verified:** [Neural Observatory](../neural-observatory.md), lines 20–27, accepts
a smaller result tree as well as all `experiments`. [Hermes documentation](../../tools/hermes/README.md),
lines 124–125, explicitly says worktrees are not a confidentiality boundary.
The native [release profile](../../src/humanbrain/_native_src/Cargo.toml),
lines 33–36, uses `panic = "abort"`; Python exception debugging alone is not
an adequate native crash workflow.

**Proposal:** give developers a dedicated public-fixture input/output directory
and point the Observatory there. Keep debug receipts (commands, selected actions,
stdout/stderr, update/checkpoint lineage, elapsed time) separate from immutable
scientific artifacts. For protected adjudication, use an evaluator boundary that
actually controls read access; an isolated Git branch does not hide sibling files.
No new server, credential grant, or external debugger package is needed for the
first step. Reproduce native failures as focused Rust tests and preserve their
build identity; introduce symbol-rich builds only as a separately declared debug
profile if a concrete crash requires them.

### 5. Standardize scratch paths and automate only the proven QA tier

**Verified:** [.gitignore](../../.gitignore), line 15, ignores `.pytest-tmp/`;
both `.pytest-tmp` and `.pytest_tmp` directories exist, and the latter generated
access-denied noise during bounded file discovery. An internal `.worktrees`
directory also exists without a matching root ignore entry. No tracked GitHub,
GitLab, Azure, or Jenkins workflow configuration was found; external CI settings
were not inspected, so this is not a claim that no CI exists anywhere.

**Proposal:** use one ignored scratch convention, unique per run; inventory
existing data before any cleanup. Keep worktrees outside the checkout where
practical. Once the focused safe test list is established, reuse it in a small
Windows CI job with public fixtures only; add optional GPU/broad jobs separately.
Document which lint tool/version is required before making it mandatory:
Ruff is not currently declared in the project's extras.

## Concrete starting commands for a later implementation task

Run these only inside the deliberately selected worktree. The first two follow
the existing README; this is a proposed bootstrap sequence, not an implemented
setup command or a record of the cleanup's separate test invocation.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -c "import sys, humanbrain; print(sys.executable); print(humanbrain.__file__)"
.\.venv\Scripts\python.exe -m pytest tests/test_cell_program.py::test_scalar_program_grammar_exposes_only_prefixes_that_can_finish -ra
```

The selected [grammar test](../../tests/test_cell_program.py), line 27, checks
public prefix validity; it is a bootstrap check, not end-to-end learning evidence.
For interactive diagnosis, append `-x --pdb` to a reviewed failing test selection.
For timing and skip visibility, use `--durations=10 -ra`. These are built-in
[pytest debug](https://docs.pytest.org/en/stable/how-to/failures.html),
[selection](https://docs.pytest.org/en/stable/how-to/usage.html), and
[reporting](https://docs.pytest.org/en/stable/how-to/output.html) facilities.

For a reviewed native test selection, run from `src/humanbrain/_native_src` so
the existing vendored Cargo configuration is in scope:

```powershell
cargo test --locked --offline
```

Use a narrowed test filter during diagnosis and add release parity for relevant
native changes. Cargo documents the [locked/offline options](https://doc.rust-lang.org/cargo/commands/cargo-test.html).
Neither native tests nor the bootstrap checks replace Python FFI integration QA
or permit running a protected scientific evaluation.

## Validation and restart boundary

Workflow implementation verified on Windows x64/Python 3.11.15 on 2026-09-04:
setup installed the declared development tools, `pip check` passed, and the
explicit lanes passed **22 smoke + 10 native + 16 integration + 3 tooling tests**.
The tooling tests exercise actual pytest child success/failure, retained logs and
JUnit, preflight rejection, and isolation from inherited collection/plugin flags.
The wrong system interpreter was also rejected in an actual CLI invocation.
The worktree helper passed an isolated temporary Git-repository test covering
explicit base, dirty-source exclusion, ownership receipt, spaces, relative paths
from another working directory, existing destinations, and invalid bases.
The dedicated public debug catalog loads both demo arms. Tool lint and
`git diff --check` passed. Hosted CI and rendered GUI behavior remain unverified.

Before reconciliation, `cargo test --locked --offline temporal_runtime` returned
**194 passed, 7 failed**. The exact same seven failures occurred in:

- the current working source with the default feature set;
- the current working source with `--features temporal-profile`;
- a clean `git archive` export of committed native source at
  `3735a79147f5b7b3e4ec86f667aeb68fde00800f` with the default feature set.

This establishes that the failures predate the workflow and uncommitted profiler
changes. Source review identified these test/contract inconsistencies:

| Failing test (under `temporal_runtime`) | Diagnosis from committed source |
| --- | --- |
| `checkpoint::tests::checkpoint_state_validator_rejects_chronology_without_recount_scratch` | Older below-cap rejection conflicts with nonempty predecessor rule introduced by `d7b4346` |
| `checkpoint::tests::pass2_replays_global_cohort_chronology_without_live_csr` | Same chronology contract conflict |
| `tests::literal_python_matching_band_trace_is_bit_exact` | Ordinary matching-drive expectation predates its deliberate removal (`950f85d`) and later exact-source learning reservation (`d455613` / `62872f0`) |
| `tests::multiple_active_and_matching_owners_enter_the_deterministic_physical_race` | Same matching-drive mismatch |
| `tests::one_step_physically_classifies_active_matching_and_novel_columns` | Same mismatch; observed tick 229 versus expected 199 |
| `tests::ordinal_slot_bank_repeats_slot_one_and_emits_max_length_unseen_entry_point` | Three-slot fixture predates four-slot codebook requirement (`197930df`) |
| `tests::recognized_column_can_still_stop_on_the_physical_winners_exact_owner_cap` | Suppressed reserved owner now fails the physical race before the expected alternate-owner cap |

### Reconciliation decisions

All seven tests were retained and repaired against the committed contracts;
none required a runtime change. The repairs preserve their useful checks:

- Checkpoint chronology now explicitly accepts a used, below-cap predecessor
  and still rejects a truly unused predecessor with consistent row/load data.
- The literal Python matching-current tick arrays are unchanged. That test now
  drives exact-source LEARN reuse, the current path that uses matching current,
  and verifies reinforcement with zero new allocation.
- READ_ONLY race tests distinguish potential-match diagnostics from electrical
  drive, retain active-owner behavior, and check different segment/afferent
  rankings, exact currents, physical latency, replay, and unchanged learned rows.
- The copying fixture supplies the full four-slot vocabulary while still testing
  unseen text, repeated slot reuse, exact generated output, and ordinary EOS.
- The owner-cap fixture deliberately forces a zero-current, capped cell to spike
  through an extreme threshold. It reaches the intended capacity guard while
  retaining exact stop priority and all state-atomicity assertions.

The expanded full-crate run found five additional failures. All five also
reproduced in the clean committed export (266 passed, 12 failed before any
reconciliation). Four concerned obsolete eager-allocation or historical byte-cost
assumptions in `procedural.rs`; one required sorted active-segment output despite
the deliberate removal of that sort in `0857c35`. These are handled as test
maintenance, with deterministic behavior and current memory invariants preserved.
The eager-allocation changes were deliberate in `9aa4d10` and `d90ebbe`:

- Impossible allocation now targets the still-eager spike buffer, preserving
  failure at the actual creation boundary rather than treating a logical limit
  on lazy storage as a promise to allocate it immediately.
- Packed-ledger tests check initially empty backing, materialized word counts,
  real allocated bytes, allocation retention after clear, and byte-oracle parity.
- In-place merge retains its independent oracle and now checks backing reuse
  for a fully overlapping run at the exact logical capacity.
- Historical eager payload/savings arithmetic and wall-clock comparisons were
  removed from the staging/ledger tests. Deterministic dynamics and overflow
  parity, plus the existing current lazy-payload tests, remain covered.
- Active-segment introspection checks every ID exactly once, without requiring
  an obsolete traversal order; output-tail and small-buffer atomicity remain.

All twelve test cases remain enabled. No production behavior or frozen protocol
was changed during reconciliation; existing uncommitted profiling work was preserved.
The Windows CI configuration now runs the full Rust suite with default features
and `temporal-profile`, alongside the existing focused Python lanes.

Release validation also exposed two host-timing gates mixed into delivery
correctness tests. Their deterministic parity/overflow/memory checks remain in
ordinary tests. The original timing blocks and thresholds are preserved in
separate `#[ignore]` tests ending in `release_performance_gates`; these require
a release build and are run explicitly with one test thread. No correctness
test is ignored. See the [performance command](../development.md#qa-tiers-and-receipts).

Final reconciliation validation on Windows x64, Rust/Cargo 1.94.0:

- Full default-feature debug suite: **278 passed, 0 failed, 2 explicit benchmarks ignored**.
- Full `temporal-profile` debug suite: **278 passed, 0 failed, 2 benchmarks ignored**.
- Full default-feature release suite: **278 passed, 0 failed, 2 benchmarks ignored**.
- Python/native preflight, checkpoint/publication, and observer checks: **10 passed**.
- Rust formatting, diff whitespace, CI YAML, and documentation links checked.

The two explicit release benchmarks were then run serially, with the other
validation jobs complete. **Both failed their retained 1.15x recurrent-speedup
gate**: fused handoff measured 1.105x and hybrid delivery 1.137x. These are
measurements from this host/run, not a general performance-regression diagnosis.
Their command reports failure; neither thresholds nor production code were
changed to obtain a pass. Performance qualification remains open and requires
controlled measurement and, if warranted, optimization. Functional development
can resume against the repaired correctness suite without claiming those speed
gates or any new scientific capability have passed.
