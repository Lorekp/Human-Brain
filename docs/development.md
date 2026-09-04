# Development workflow

Use this workflow for setup, worktrees, ordinary QA, and debugging. The
[roadmap](research/roadmap-reading-to-coding.md) owns research priorities and
each frozen protocol owns its evaluation authorization.

## Bootstrap a checkout

Run from the selected checkout, using Python 3.11 for the tested development
baseline. Rust/Cargo and the MSVC C++ build tools/Windows SDK are needed for
native QA; CUDA and the visualizer GUI are optional.

```powershell
./tools/setup.ps1 -Python 'C:/path/to/python.exe'
./.venv/Scripts/python.exe tools/dev.py native
```

Omit `-Python` if `python` already selects the intended interpreter. An existing
`.venv` is reused. Setup installs `.[dev]` with the direct dependency baseline in
`tools/dev-constraints.txt`, verifies the editable import, and runs smoke QA.
It stops on installation or test failure. Package metadata retains compatibility
ranges; the constraints are a known-good baseline, not a complete dependency lock.
Installation requires package-index access or a populated pip cache.

Always invoke tooling through this checkout's `.venv`; activation is optional.
`tools/dev.py doctor` checks Git root, environment root, and the actual imported
package before pytest can prepend `src`. It records the revision, tracked-diff
hash, dirty/untracked status, versions, and device. Untracked file contents are
not covered by the diff hash. Commit the reviewed work before treating a revision
as a reproducible baseline.

## Create a worktree

Choose an explicit committed base and bounded ownership scope:

```powershell
./tools/new-worktree.ps1 -Base HEAD -Branch reader-fix `
  -Destination '../human-brain-reader-fix' -Owner 'reader-agent' `
  -Scope 'reader implementation and its focused tests'
```

The helper creates `codex/reader-fix` and records source, base, branch, owner,
and scope in the new checkout's `.runtime/worktree.json`. Existing destinations
and branches fail without replacement. Relative destinations resolve from the
source checkout. Prefer sibling worktrees to nested directories.

**The committed base excludes all uncommitted changes**, including new workflow
files. If a task needs current edits, first make a reviewed checkpoint commit or
continue in the current checkout. Run the new checkout's setup independently;
never share an editable `.venv` or copy native DLLs between worktrees. This helper
does not stash, commit, merge, or remove work. Before manual teardown, inspect
`git worktree list`, that checkout's status, and commits against its recorded base.

## QA tiers and receipts

```powershell
./.venv/Scripts/python.exe tools/dev.py qa smoke
./.venv/Scripts/python.exe tools/dev.py qa native
./.venv/Scripts/python.exe tools/dev.py qa integration
./.venv/Scripts/python.exe tools/dev.py qa tooling
```

| Tier | Coverage |
| --- | --- |
| smoke | Tokenizer, public grammar/compiler examples, synthetic windowed-probe math; no native build |
| native | Explicit attested build/load, small real FFI checkpoint/publication round trips, observation and restore invariants |
| integration | Public Observatory demo and temporary fixtures: circuit observation, flushed stream, replay, catalog, and CLI validation |
| tooling | Real pytest child success/failure and preflight rejection, including exit codes and receipts |

Selections live once in `tools/dev.py`. Pytest plugin autoload is disabled and
ambient `PYTEST_ADDOPTS`/`PYTEST_PLUGINS` are cleared for these lanes, so shell
settings cannot silently replace execution with collection or add test paths.
Every invocation gets a unique `.pytest-tmp/dev/<run>/receipt.json`;
pytest also writes `output.log`, `junit.xml`, and retained temporary fixture files.
Receipts include the exact pytest command and elapsed time. These files can be
shared for debugging after inspecting them. Existing scratch data is left intact.

Native commands default to CPU and checkout-local `.native-cache`, respecting
explicit `HUMANBRAIN_DEVICE` and `HUMANBRAIN_NATIVE_CACHE` overrides. Preflight
records the source and binary hashes, resolved library, target, toolchain and SDK
identities through the existing attested loader. It can compile code; `doctor`
does not. Complete the first build before launching parallel native jobs that
share a cache. Build identity includes source location and compiler details.

Those environment defaults apply to children launched by `tools/dev.py`; they
do not change the calling shell. Before invoking a native experiment directly,
set `$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'` and
`$env:HUMANBRAIN_DEVICE = 'cpu'` unless deliberately testing another setting.
Record the loaded library identity from the experiment itself. A doctor receipt
alone does not prove that a later direct invocation used the same native cache.
Set these variables **before Python starts**: some existing modules initialize
native cells during import, so assignments inside an experiment's `run()` can
occur after the process has already loaded and cached a different library.

For change-specific QA, run reviewed existing test nodes with this interpreter.
Use `-o addopts= -ra --durations=10` to expose counts, skips, and timing. Broader
CPU/native, SciPy, and CUDA regressions require explicit selections and available
dependencies; passing the small lanes is not a claim that every repository test
passes. Frozen freshness failures require protocol review, not weaker checks.
The [verified baseline and test reconciliation](research/developer-workflow-review.md#validation-and-restart-boundary)
record the repaired historical test/contract inconsistencies and validation scope.
The [Windows CI job](../.github/workflows/developer-qa.yml) reuses these same
commands and uploads only QA receipts, logs, and JUnit. It runs after publication
to GitHub; a local pass does not establish a hosted CI pass.

For changes to native code or its contracts, also run the full Rust unit suite
from `src/humanbrain/_native_src`: `cargo test --locked --offline` and the same
command with `--features temporal-profile`. CI runs both. Use `--release` for
relevant physical/numerical parity checks. Keep performance measurements in
explicit benchmark runs; unit tests should assert deterministic behavior and
resource invariants rather than elapsed-time speedups.

The two retained native delivery speed gates are explicit, ignored-by-default
performance tests. Run them on an otherwise idle machine, in release mode and
serially, from the native crate:

```powershell
cargo test --release --locked --offline release_performance_gates -- --ignored --test-threads=1 --nocapture
```

Their original thresholds still apply. The ordinary suite exercises the same
fixtures' deterministic correctness checks without timing samples. A passing
correctness suite does not certify the performance gates.

The integration lane tests existing developer plumbing. It does not implement
or establish the roadmap's reading -> retained knowledge -> executed edit ->
fresh transfer milestone, and does not run protected adjudication.

## Debug access

Start with the failing receipt's command and preserved `tmp` fixtures. For an
interactive Python failure, rerun the failing node directly with `-x --pdb -s`.
Do not reuse a previous `--basetemp`: pytest clears its contents on reuse. Choose
a fresh directory or omit it for interactive runs.

For the GUI, install `.[visualizer]` in the same environment and open only the
public demo under a dedicated catalog root:

```powershell
./.venv/Scripts/python.exe -m pip install -e '.[visualizer]'
New-Item -ItemType Directory -Force .runtime/public-debug/experiments | Out-Null
Copy-Item -LiteralPath experiments/visualizer_demo -Destination .runtime/public-debug/experiments -Recurse -Force
./.venv/Scripts/python.exe -m humanbrain.visualizer.cli .runtime/public-debug/experiments --experiment visualizer_demo
```

Use `--activity-stream` to attach a retained public fixture stream from QA.
GUI rendering is a separate manual check; headless QA does not exercise the GPU
window. Worktrees share filesystem access and are not confidentiality boundaries.

For native failures, reproduce in a focused Rust unit test from
`src/humanbrain/_native_src`, where the vendored configuration applies:

```powershell
cargo test --locked --offline temporal_runtime
cargo test --locked --offline --features temporal-profile temporal_runtime
```

Record `rustc -Vv` and `cargo -V` alongside native diagnostics. Release uses
`panic = "abort"`; a Python exception debugger cannot diagnose every native
crash. Keep Cargo debug builds separate from the attested release cache and
repeat relevant FFI QA for release parity.

For developer-tool lint, use the explicit config to avoid parent-project rules:

```powershell
./.venv/Scripts/python.exe -m ruff check --config tools/ruff.toml tools/dev.py tests/test_dev_workflow.py
```

References: [Python environments](https://docs.python.org/3/library/venv.html),
[pytest debugging](https://docs.pytest.org/en/stable/how-to/failures.html),
[Ruff configuration](https://docs.astral.sh/ruff/configuration/), and
[Cargo test options](https://doc.rust-lang.org/cargo/commands/cargo-test.html).
