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

## Completed qualification before authored timing

Main source is SHA256 `2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8`;
new action bridge is `3bacf37d703af9c6685a4ece8b554cc6779597073fa94adae2e6004cc8dd50c1`.
Ruff and source review passed. The address feature construction, observation,
mixture and standalone codec equations remain unchanged. Held codec activity
now has explicit checkpoint metadata and actual evaluation/gather counters.

Seventeen initial model checks passed on their first run in 4.35 pytest seconds
(5.7256813 process seconds), covering exact old/new no-backpropagation parity,
local scalar equations, all controls, finite publication failure and full-capacity
accounting. JUnit `.pytest-tmp/held-codec-v1-qa.xml` is SHA256
`7db47696edeb1fc2b639d095f5b667b3c841ec6008f76595c36335f478ed8eb1`.
Five saved parity receipts record actual decision counts. The review is
`.pytest-tmp/held-codec-v1-qa-review.md`, SHA256
`43f2e36d9f5c51bc38947796018e77bac5ee4a7ab3be14e1a7325b04e3de646e`.

Saved counts exposed a coverage gap: the nonzero-sensory parity fixtures stopped
at their first EOS, while the longer free parity traversals used zero sensory
activity. One additional authored biased-output fixture therefore compares all
four held response tensors with actual fresh codec evaluations at five free
positions over repeated bytes and absorbing EOS. It passed on its first run in
2.99 pytest seconds (4.3874785 process seconds), with no teaching or actions.
Its JUnit is SHA256 `2b263b67e95eff5cf902e23a9741c8c1589d1c6f06f28e8b2ef32b7f0f74c685`;
JSON is `7722afd33c539b10f3bace8e889eb32b9fa7bb0e5058f953399398389e600f56`.
The aggregate model QA made 23 constructions plus four disposable clones,
23 ingestions, 26 valid free predictions with 54 decisions, one separate query,
and 17 actual teacher traversals with 71 offers, including three offers in the
injected publication rejection. Two invalid predictions and one invalid teacher
were rejected at their boundaries. The extra free fixture made five standalone
codec comparisons over twenty rows. No old adjoint reference or acquisition ran.

All nine action qualification cases passed first try: seventeen actual children,
zero failures/skips, 5.27 pytest seconds and 6.5050213 process seconds. JUnit
`.pytest-tmp/stdlib-python-v1-qualification-qa.xml` is SHA256
`4d5162c4bcaa1dcdd4e4966690160584a478818fd055e29f23712fb6c650355d`.
The new standard-library identity and process-contamination checks passed, as
did all seven semantic pairs. No neuronal model or motor was used by this suite.

The saved-only combiner passed source review and completed once in 1.8511146
process seconds, authenticating 21 evidence files with no new child executions.
Its manifest `.pytest-tmp/stdlib-python-action-v1-qualification.json` is SHA256
`4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c`.
The exact initially tested source is archived at
`.pytest-tmp/stdlib-python-action-v1-tested-source.py`, SHA256
`217296ba5fdff11addae60f10be147c173c0208ef583be7a730b04b82f68aa0c`.
It binds both initial suites. The later test extension and line-ending
normalization preserve every original AST node; source-only receipt SHA256
`973e85251122d9fb86884a544ddb95d3c7b613e9d6b1714fcbdde926544cef1a`.
Current tested source is `3b086c8bf7ef82e89c7ea2d002b39756c988eadc81eb3e343d2714057b07335f`.
Doctor receipts are `.pytest-tmp/dev/doctor-la9ze38f/receipt.json` and, for the
additional case, `.pytest-tmp/dev/doctor-5rntiha_/receipt.json`.

The reviewed authored timing driver is
`.pytest-tmp/stdlib-python-action-timing-v1.py`, SHA256
`7293675b4caea6689075320722beb3173ac6a3d48f958706b043e1dbb4ba8f8c`.
It records the performance-counter identity and optional contemporaneous
GetSystemTimes counters, preserves all twelve observations, and uses the fixed
criteria above. Qualification permits this diagnostic; its outcome still must
earn integration before a new scientific harness or cost is executed.

## Authored timing outcome and integrated return

The one fixed comparison passed all semantics and its preregistered timing
criteria. Across six executions per runtime, new total complete execution time
is 0.3197405000 seconds versus old 0.3572715000, ratio 0.8949510387. Pending
medians are 53.0101 versus 57.8297 ms; corrected medians are 53.5667 versus
58.0082 ms. This is a 10.5% aggregate reduction in this small authored sample,
not an end-to-end learner speedup or a claim that the entire site subtree was
removed from elapsed cost. One corrected pair was slightly slower on the new
runtime. All twelve observations are retained. Within each case the route order
is fixed by the prospective alternating schedule; order and environmental
variation remain limitations on causal interpretation.

The complete owned-process monitor passed in 2.7144727 seconds at 79.44921875
MiB. No neuronal model, teacher or motor ran. Source revision is
`587afa0f3d345324a360e12cb3f4d978743cf9e0`; clean doctor receipt is
`.pytest-tmp/dev/doctor-liqlf0fz/receipt.json`. Saved evidence:

- `.pytest-tmp/stdlib-python-action-timing-v1/report.json`, SHA256
  `d8ac9291bd61841f9fa0de73e05c8879e91a46f6949f204315851b9db1be9bf8`;
- its `executions.jsonl`, SHA256
  `78e182d3eef91e2787a24817a4c7c7cc4d5a7becd9a5e0dff566c2b706b36d35`;
- `.pytest-tmp/stdlib-python-action-timing-v1.launch.json`, SHA256
  `160e7065d591b665a1a0241896b916d58dfff3f92670dec6dedbc21de449ac61`.

This earns the [v5 integrated return](../../experiments/r4_integrated/local_joint_address_online_v5.md).
The dedicated runtime and held codec work must now satisfy the complete workload
and saved independent audit under the unchanged admission formula. No full fit
is authorized by authored timing alone.

V5 source/AST review passed ten checks, including exact export, worker,
same-instance lesson, probe, final-action and main bodies. Receipt
`.pytest-tmp/joint-online-v5-source-parity.json` is SHA256
`a00f184724f7018d34d470823bbb8072d9df3cf215258f15ab6290cd01fd5a03`.
Runner SHA256 is `616e5e9fca99498d0dd5d74bfea00bb11df82a343a7809b2093a69809b2d3e8c`;
recipe SHA256 is `b305cd53a6a9a1f9938ee449497b9ae5d6f62186684bfb3bab69775eb57e3594`.
Independent source reviews also passed the new runtime/held-work clauses and
the narrowly changed preparation and final admission drivers.

The independent auditors passed all seventy authored helper checks on their
first attempt, with zero failures, 0.1701763 internal seconds and 0.3365557
process seconds. No model, tensor, action, real qualification or checkpoint was
read/executed by that helper. Unchanged historical helper suites were not repeated.
Receipt `.pytest-tmp/local-joint-address-online-v5-helper-qa-v1.json` is SHA256
`4749f403921b9d234ebc47a113a7ea990b05b3bf3c4fccc15914a2c1983ae489`.
Exact tested checker source is preserved; only each DRAFT guard was then disabled.
Frozen cost checker SHA256 is
`b97d73eee4614a69c3cfaada8ff4c0fd0695c0ceaa5ef6edfdc0d4373045ddf8`;
run checker is `7d98ae6885d28776d94f5ba4ab0d7445fa4fdf541868f635460b24003cc05259`.
Readiness `.pytest-tmp/local-joint-address-online-v5-readiness.json` is SHA256
`85bab90708d8a1c27933bbe295181c73f4b50558b7cdcbf087baa774c56165ca`.
This closes source/QA preparation and permits the one fresh v5 cost after a clean
commit and admission preparation. It does not admit full learning.
