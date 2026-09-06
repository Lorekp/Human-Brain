# Local online return: bounded runtime and held-activity optimization

Selected by the orchestrator, 2026-09-06, after the
[v4 cost rejection](../../experiments/r4_integrated/local_joint_address_online_v4_result.md).
One shared integrated return has two independent implementation tracks. Neither
changes the neuronal learning equation, rate, lesson roster, exposure or action
before feedback. Frozen v4 remains rejected; no identical cost retry follows.

## Dedicated standard-library action execution

Use the pinned base CPython 3.11 executable in a new explicit action environment:
`-I -S -X utf8 -B`, with the existing qualified regular-source bootstrap. Copy the
parent environment and remove `__PYVENV_LAUNCHER__` for this child. Keep site
disabled; do not reconstruct venv prefixes, site paths or development hooks.
The learner continues in the selected development `.venv`. Bind its identity
separately from the action interpreter, flags, source bootstrap and new runtime
qualification. Reusing the historical base executable hash authenticates that
file, not the new runtime behavior.

This is a declared standard-library-and-authored-sibling environment. It does not
implicitly provide project imports, NumPy or other development packages. Future
tasks needing dependencies must declare a separate provisioned action environment.
Every action executes actual regular Python source in a fresh OS process with
the same workspace materialization, UTF-8 byte capture, ten-second timeout,
exception/exit handling and file verification. It is process-state isolation,
not a security sandbox. No persistent worker, host answer cache or simulated
replacement for code execution is introduced.

Source review found that Windows multiprocessing uses fresh-interpreter spawn;
a supervisor cannot inherit warmed imports through the supported mechanism.
The earlier import profile recorded a 16.009-ms site subtree, but neither that
instrumented duration nor the v4 slowdown establishes the prospective savings.
See the saved source review `.pytest-tmp/windows-action-startup-options-review.md`,
SHA256 `9e29e1a4acab69b4ba55899938ad2fc60adbbf900439d89d6ce0db2915a80d75`.

After source review, lint and doctor, authored qualification has a fixed budget
of seventeen children: one new-runtime identity probe, seven old/new semantic
pairs reusing the existing fixture definitions, and two independent-process
contamination fixtures. Declare base prefixes, no-site flag, absent development
paths/hooks and direct executable identity prospectively. Semantic pairs retain
raw outputs/errors before assertions and compare the established meaningful
namespace, cleanup, encoding, sibling, error, exit and pending behavior, allowing
only the explicitly new environment fields and existing wrapper differences.
The contamination pair changes in-memory module/environment/cwd state in one
workspace and verifies fresh state in the next. Zero neuronal models or motor
calls are needed. No broad historical/backpropagating tests are selected.

Then run one fixed authored timing comparison, at most 60 seconds and 512 MiB,
using the old v4 and new actual executors: three pairs each for supplied pending
and corrected `ab` workspaces, twelve children total. Order pairs by alternating
case (pending, corrected) three times; alternate old/new and new/old order by
pair index. Record complete execution times and raw semantic receipts; no
adaptive repeats, warmups or deleted outliers. This engineering diagnostic earns
integration only if all semantics pass, both case medians favor the new runtime,
and the aggregate new/old paired-duration ratio is at most 0.90. Preserve all
durations and any failures. This is not neuronal accuracy or cost admission.
Capture contemporaneous host observations if available, but do not use them to
discard samples or retry. Run performance work serially after other QA finishes.

## Current-call held codec responses

The first branch cursors enumerate all occupied tape slots. During a prediction
or teacher call, codec weights/bias and sensory slots remain held. Evaluate that
memoryless codec once for every occupied slot; route its held responses through
the actual later EOS-absorbing cursors. Rebuild independently for every prediction
and teaching call; never retain the bank across an action, lesson or synaptic
publication. No teacher target, prefix selection or query-dependent answer is
stored in this bank. This changes the simulation schedule and temporary activity
storage, not the direct local learning rule.

Keep the actual float32 X-before-B bank, address scores, prefix mixtures and
beliefs, per-step local codec errors, ordered error accumulation, float64 X-times-
output-error products, casts and finite all-group publication unchanged. Do not
factor the circuit into an upstream reverse error route. Count actual codec
evaluations and gathered activities separately; account for the bank and gathered
row lifetimes rather than claiming zero added storage.

The source proposal is `.pytest-tmp/joint-model-held-codec-reuse-review.md`, SHA256
`a449d9feba0e1373bf5cb759006d23971851647c3e4d0414b7de9a881f2e6a11`.
Saved M timings do not isolate the codec, and removing all projected v4 M would
still fail its ceiling. This is a source-supported optimization opportunity,
not a measured speed claim or a sufficient repair by itself.

Use the existing test home for authored exact numerical parity against the
preserved current no-backprop source: all four modes, repeated bytes, zero
sensory activity, EOS absorption, a teacher longer than the tiny document, and
learning disabled. Compare per-step probabilities/beliefs/cursors, outputs/stops,
losses, local update flags and final parameter/state tensors. Work counters differ
only as declared. Check invalid input/parameter rejection, finite-proposal failure
atomicity and rebuilding after a weight update. Preserve the tested source and
all QA receipts; no old adjoint/backpropagating reference or trained endpoint.

## Return and evidence discipline

After both tracks qualify, integrate them once in a new version with fresh
source custody and narrowly affected independent-auditor checks. Preserve the
same four arms, seed 733, rate .01, 1,024 lessons per arm, 80 final actions,
accepted-action probe, complete M/A/J timings and 33,801,664-byte persistent
endpoint. Update transient/work accounting for held activities. Require one
fresh complete cost and finished independent audit under the unchanged
900-second/2,048-MiB admission formula. Only a passing final admission authorizes
the full acquisition. No exposure, ceiling or accepted-path relaxation is selected.

The diagnosing-bugs workflow uses the preserved failed complete cost as the
measured baseline and authored differential fixtures for the new boundaries.
Repeated reproduction/minimization of that frozen cost is intentionally omitted
because it would be an identical timing retry. This return does not claim to
have identified the cause of v4's transient slowdown. Its two changes test
explicit redundant work, and whole-workload success must still be measured.
