# Repeated teaching: incomplete acquisition with measurable local learning

The fixed-architecture continuation completed its predeclared **32 corrections
per skill**. Neither integrated reading nor editing became correct. Online
learning did acquire the correct edit boundaries by checkpoint 16 and preserve
them through checkpoint 64, but generated the wrong payload throughout. This
separates useful motor learning from complete software behavior.

All 35 target positions received direct literal updates, and their probabilities
improved slightly at identical diagnostic emission features. The distribution
remained near uniform; the free reading and edit payload stayed one NUL byte
followed by EOS. The limiting measured computation is intended-byte generation,
including weak literal emission and its mode weighting. Missing teacher contact
alone cannot explain this result. The evidence does not isolate normalization,
branch selection, feature competition or upstream credit as the sole cause.

## Fixed comparison and complete outcomes

Execution used source `f577413dd9bac32302c3c84d5e09b950f8c8473c`, following the
[released protocol](../../docs/research/repeated-teaching-acquisition-v1.md).
The neuronal architecture, local rule, learning rate, tasks and response limit
were unchanged. The same persistent system acted, observed its executions and
learned locally; no backpropagation or separate fitted actor was used.

Online and learning-disabled routes began as independent copies of the completed
causal ONLINE parent. The simple retrieval/editing reference continued its own
actual history. Each route received eight reading occasions, eight editing
occasions, then 24 interleaved reading/editing pairs. All actual free responses
preceded their corrections. Task workspaces reset to the original fixtures;
live learner state and RNG continued without reset. No factual source was added.

| Route | Live reading | Live editing | All checkpoint outcomes | Complete acquisition |
| --- | ---: | ---: | ---: | --- |
| Online local learning | 0/32 | 0/32 | 0/20 | Failed |
| Learning disabled | 0/32 | 0/32 | 0/20 | Failed |
| Simple retrieval/editing | 0/32 | 32/32 | 10/20 | Failed |

Checkpoint indices were 0, 8, 16, 32 and 64. Every panel contained A00-R, A00-E,
R02-R and R12-R in that order. Neural outcome vectors were always FFFF; simple
was always FTTF. The simple edit already succeeded at checkpoint 0, so its
success is retained reference behavior rather than new acquisition here. The
predeclared criterion required both taught skills correct at both checkpoints
32 and 64; none of the three routes met it.

At every neural checkpoint, A00-R and the A00-E payload were hex `00` plus EOS,
R02-R was empty plus EOS and R12-R emitted hex `63d1`. Online editing selected
SPLICE app.py 10:13 at checkpoints 0/8, then the intended 20:24 at 16/32/64.
Disabled editing remained at 10:13. Correct boundaries with payload `00` fail
the actual executed-behavior grade; they are not a successful integrated edit.

## Teaching coverage and probability changes

Each neural route received 1,120 live byte/EOS offers and 96 motor labels:
32 opportunities for each target position and each edit head. Direct literal
teaching was selected on 273/1,120 offers (24.375%): 95/384 reading and 178/736
editing. Online published those 273 updates; disabled proposed the same
inclusions and published zero. Every target position received 4–12 direct
updates. Raw bytes account for 260 publications and EOS for 13. Upstream local
credit observes all offered symbols, so literal coverage does not enumerate
every possible learning effect. The
[learning review](../../docs/research/repeated-teaching-acquisition-v1-learning-review.md)
preserves the full position vectors, motor-update counts and conditional scores.

| Intended-symbol diagnostic | Online CP0 | Online CP64 | Disabled change |
| --- | ---: | ---: | ---: |
| Reading mean literal probability, including EOS | 0.003897608 | 0.003910385 | 0 |
| Editing mean literal probability, including EOS | 0.003896533 | 0.003904750 | 0 |
| Reading conditional mixture path log score | -71.409602 | -71.373802 | 0 |
| Editing conditional mixture path log score | -51.807093 | -51.802872 | 0 |

All 35 literal target probabilities improved online from checkpoint 0 to 64;
all saved diagnostic activity hashes and sampled modes matched across the five
checkpoints and both neural arms. This supports a fixed-feature weight effect.
The probability increase is small: mean relative gains are 0.328% for reading
and 0.211% for editing. Ten reading and four editing positions have no matching
prior/register emission at the measured contexts. Their intended-symbol mixture
probability relies solely on the weighted literal branch, around 0.000975.

These scores condition on the correct target prefix and actual sampled
context/gate trajectory. They are not marginalized probabilities of complete
free answers. Mean mixture probability and its log-product need not move
together: editing's mean mixture probability declines while its log score
improves. Teacher contexts also differ from the free trajectory. The
[source analysis](../../docs/research/repeated-teaching-local-path-analysis-v1.md)
and learning review explain the small direct-update bounds without claiming
that those bounds cover all scans, consequences or shared-feature updates.

## Interference, retention, transfer and recovery

Mean literal log probability and mixture path log score improve after every
online teaching block. There are smaller opposing positional changes: seven
editing positions decline after the reading-only block; the reading slash
declines after the editing-only block. This is local probability interference
amid net gains, not evidence that an acquired complete behavior was forgotten.
The learned edit span persists through later interleaved work. Neither complete
neural skill was acquired, so complete-skill retention cannot be established.

The two historical sentinels already fail at the initial panel. The previous
comparison's one late R02-R hit had different stochastic context and cannot
serve as a guaranteed successful baseline here. Fresh transfer, expanded
changed-context evaluation and a new teacher-free recovery battery were not
run. Earlier unsuccessful recovery evidence remains unchanged; altered spans
under direct teaching do not demonstrate productive teacher-free feedback use.

Keep architecture and task difficulty stable. No new component, automatic
budget extension or broader evaluation is selected by this result. Any next
learning change must address the identified output-generation/update limitation
and preserve the measured motor behavior, with its contribution distinguished
from the existing rule rather than inferred from implementation QA.

## Integrity, resources and retained continuation

The first experiment launch completed **252 responses and 111 actual edit
executions** in 1,519.0128574 seconds, with 1,529.1484375 MiB sampled process-tree
peak. This includes all five panels, diagnostics, final publication, fresh
paired restoration and saved scalar review. Neural diagnostic teaching adds
350 symbol offers and 30 motor labels with publication disabled: combined
live-plus-diagnostic totals are 2,590 symbols and 222 motor-label offers. The
222 teacher journal records across all routes are a different denominator.

Three focused tests passed after two preserved fixture/observer QA failures;
Ruff passed. Run doctor is `.pytest-tmp/dev/doctor-4xbxrf47/receipt.json`;
focused QA doctor is `.pytest-tmp/dev/doctor-hg4j3c7w/receipt.json`. Fresh final
restoration passed with two payload decodes and zero task or teacher replay.
An independent small-report audit checked all main/panel outcomes, coverage,
counts, hashes and restore counters without loading models or journals.

Intermediate measurement panels are disposable RAM copies and nondurable
boundaries. One complete final block is durable; its fresh integrity checks
remain intact. The
[cost review](../../docs/research/repeated-teaching-acquisition-v1-cost.md)
separates nested phases, actual file traffic, instrumentation limits and reduced
checkpoint frequency. There is no matched whole-run speedup claim. Final run
artifacts occupy 742,750,281 bytes; the report's earlier disk snapshot is not
the final inventory.

Continue from `.pytest-tmp/repeated-teaching-v1-run/checkpoints`, final HEAD0,
boundary SHA256 `3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677`.
Online payload SHA256 is
`f77695cd9603e00e031cc867ce53573e2148eb14bb39d2e70b576b317ef66f0e`, model
`07855c61ca02e1dea8b1e8044df2576643689b8e55a28a1d50109a3580b54d7a`.
Both neural actors retain 401 actor and 406 reader experiences, with the same
1,982-slot factual memory hash
`cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e`.
Keep these states and the actual simple history; do not reset to the parent.

The immutable bank contains exact copies of the saved
[learning result](../../benchmarks/repeated_teaching_acquisition_v1.result.json)
(SHA256 `402aae6d061464aecccbdc73d12abd0683ed0ee7d4408d650550070a69efea61`)
and [run report](../../benchmarks/repeated_teaching_acquisition_v1.run.json)
(`3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262`).
The [measurement ledger](../../benchmarks/repeated_teaching_acquisition_v1.measurement.json)
records 1,571.3497749 seconds across the listed QA, preparation and run monitors;
banking adds 0.4782003 seconds separately, with zero model calls. Manual review,
doctor, lint and documentation time are outside that measured total.
