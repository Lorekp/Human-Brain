# Qualified temporary-module execution without runpy imports

Selected by the orchestrator after the [one-child stdlib import diagnostic](local-stdlib-action-import-profile-v1.md)
found a 17.281-ms cumulative runpy subtree. This exceeds its prospective 5-ms
design-review cutoff. The nested importlib.util/contextlib timings are included
in that figure; removing them is not a measured wall-clock saving. V5 remains
closed at 956.052 projected seconds, with no full acquisition or identical retry.

## One execution change

Replace only the regular-source bootstrap's runpy dependency with a small
public-API implementation of the already qualified temporary-main lifecycle.
Keep the pinned base executable, separate parent venv identity, `-I -S -X utf8 -B`,
copied environment with the launcher removed, strict UTF-8 stdout/stderr and
explicit newline `\n`. Retain real files, a fresh process, captured raw output,
ten-second timeout and complete verification. Bare script invocation was reviewed
but is not selected: its normal Windows newline conversion and stdio error
policies differ from the current contract.

Continue reading with `io.open_code(os.path.abspath('main.py'))` and compiling
source bytes with filename `main.py`, preserving encoding-cookie/BOM and open-code
behavior. Compile before replacing the main module. A function-local helper
creates `type(sys)('__main__')`, temporarily publishes it in `sys.modules`, and
sets argv[0] to `main.py`. Populate the same seven special globals as the pinned
runpy helper: name, file, cached, doc, loader, package and spec, with the same
values. Execute in that module's dictionary. Nested finally blocks restore
argv[0] and then the original main module, including on script exceptions or exit.
Return a shallow globals copy as the old helper does and discard it at the caller.
Do not clear the original globals or retain the temporary module at bootstrap
global scope; callbacks must retain the same useful namespace lifetime.

This specializes the current constant-name, no-spec, no-initial-globals regular
source call. It does not recreate the general runpy API or support zip/directory/
sourceless-bytecode execution. Different wrapper stack frames and import/audit
events are explicit differences. Task-visible source exception location/type,
namespace, output bytes and cleanup behavior remain qualification requirements.
Process isolation is not a security sandbox.

Use a versioned bridge beside the frozen implementations. Reuse the current
stdlib launch record/qualification strictly as unchanged runtime identity; bind
the new bootstrap policy, source/hash and paired evidence separately. Never
describe the old manifest's bootstrap hash as the new bootstrap identity.
Keep execute/actuate bodies otherwise identical. No monkeypatching frozen module
globals or changes to the neuronal circuit, teacher, exposure or action ordering.

## Bounded qualification and measured return

After source review, lint and checkout doctor, run eight authored old/new pairs,
sixteen fresh children total, in alternating route order. Reuse the existing
seven regular-source fixtures unchanged: namespace/cleanup/sibling, UTF-8 BOM
and CRLF source, Latin-1 cookie, syntax error, runtime exception, nonzero exit with
invalid UTF-8 bytes, and pending action. Add one callback-lifetime fixture that
does not import or retain the temporary main module and reads its module globals
at exit. Store raw receipts before assertions. Compare meaningful namespace and
runtime fields, encoding/error/exit behavior and all materialized bytes, allowing
only declared wrapper/import events. No learner, teacher or motor runs. Reuse
unchanged process/environment qualification and check exact transport bodies by
source comparison; no broad historical test suite is selected.

If qualification passes, one fixed authored comparison uses the actual old and
new executors on three pending and three corrected ab pairs: twelve children,
alternating cases and route order by pair index, no warmup or adaptive repetition.
Require all semantic checks, both case medians favoring the new route and summed
new/old elapsed execution ratio at most 0.90. Record performance-counter identity,
all observations and limits on order/host variability. Use a 60-second/512-MiB
owned-process watchdog. No sample exclusion or repeat follows failure.

Only that pass earns one integrated version with the unchanged held-codec neuron
simulation and local learning. Review affected runner/auditor bootstrap custody,
then require fresh complete cost and finished independent audit under the same
900-second/2,048-MiB formula, accepted-action probe and 4,096 acquisition actions.
An authored speed pass does not admit full learning. The same neuronal learner
must generate each first attempt and receive subsequent local correction; the
supplied motor remains a separate limitation to be replaced after useful learning.
