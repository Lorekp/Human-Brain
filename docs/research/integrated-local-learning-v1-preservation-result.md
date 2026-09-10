# Retained local-rule preservation result

Completed 2026-09-10 at execution source `356db7d`, comparing the selected
implementation `7e164e7` with the exact predecessor sources from `7271b9c`.
The [declared protocol](integrated-local-learning-v1-preservation.md) completed
without additional teaching. The [saved result bank](../../benchmarks/integrated_local_learning_v1.preservation.json)
authenticates both inventories, the saved comparator, source/initialization
receipts, doctor and all outer launches. This is preservation evidence, not
acquisition under the revised rule.

## Behavior and state

All four paired cases have identical forward traces and numerical/RNG state
progression, including actual logits, probabilities, random draws, prior choices,
context/register trajectories and free outputs. Both versions score **0/4 complete**.

| Existing case | Identical free answer or replacement, hex | Complete outcome |
| --- | --- | --- |
| A00-R | `0000ba74d42f77` | Incorrect |
| A00-E | `81206920646c` | Incorrect; actual execution returns 1 |
| R02-R | `8a` | Incorrect |
| R12-R | `586f00` | Incorrect |

The actual edit remains `SPLICE app.py 20:24`: the previously acquired motor
placement survives the revision, while the payload and complete behavior remain
wrong. Both proposals bind the same original input workspace, SHA256
`7d28f5867272a1fcd53d048f92eb7dfbe7481d9d8c629bf717dff038a366617b`.
Raw execution stdout is equal; stderr bytes and lengths differ and remain saved
exactly. Forward and task-outcome equality does not assert raw diagnostic equality.

Each worker starts with the actual retained RNG, then continues it through the
four cases without resetting. The older learning panels used a reset diagnostic
RNG; their different output bytes are therefore not a before/after rule contrast.
Only the actual paired workers establish the forward comparison here.

The authenticated parent remains at boundary
`3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677`,
actor/reader counters **401/406**, and **1,982 factual slots**. Each worker decodes
that parent once into two independent numerical copies. Its inactive copy is
unchanged. Its active disposable copy finishes at 402/410 with only the expected
RNG/counter progression; non-RNG numeric state, facts, views and self-wiring remain
unchanged. Neither disposable result is the next continuation checkpoint.

## Complete work and cost

Per worker: four free responses, one real edit execution, 718 public scan offers,
21 free ticks, 2,396 controller draws and four RNG publications. Across the pair:
eight responses, two executions, 1,436 scan offers, 42 free ticks, 4,792 draws,
eight RNG publications, four constructed actors and two parent payload decodes.
There are **zero teachers, source ingests or slow-learning publications**.
The restored simple reference is neither queried nor taught; this preservation
pair does not replace a matched enabled/disabled/reference acquisition comparison.

| Outer monitored operation | Seconds | Sampled process-tree peak, MiB |
| --- | ---: | ---: |
| Initial failed import preflight | 2.583667 | 211.01 |
| Corrected predecessor worker | 32.729132 | 844.13 |
| Current worker | 30.635397 | 839.90 |
| Saved-only comparison | 3.849378 | 110.79 |
| Sum of listed outer operations | 69.797573 | — |

The first launch failed on missing `psutil` before model construction, decoding
or case exposure. Its failure remains counted and saved. Reusing the existing
Windows memory reader corrected the launcher without a dependency addition or
weakened resource check. The corrected pair supplies the first actual case
exposure; no experienced case was retried.

The current worker's report samples 29.947378 seconds internally, including
7.971011 for initialization, 9.523318 for prediction and 5.010029 for disabled
resolution. Explicit worker numerical hashing takes 1.656863 seconds, trace
projection .134762 and JSON/hash/disk publication .595903. These scopes are
nested within the worker total, not extra costs; explicit hashing excludes
internal model hashes, and publication excludes final report/completion.
The predecessor's internal total is 31.955688 seconds. One fixed-order pair
does not identify a speedup or measure enabled teacher throughput.

The bank preserves exact phase definitions and source identities. Saved-only
bank preparation is timed separately with its exclusions; interactive inspection,
documentation and review are not included in the 69.797573-second outer sum.
The prior [three focused QA checks](integrated-local-learning-v1-qa.md) remain
separate implementation evidence and cost, not additional acquisition attempts.

## Acquisition decision remains separate

This result establishes preservation of existing forward behavior and learned
placement. It measures no new effective teacher coverage, intended-output
probability improvement or interference from learning. Complete acquisition under
the revised rule remains unmeasured; retention/changed-context expansion and
teacher-free recovery remain unestablished and were not expanded here.

The [budget review](integrated-local-learning-v1-budget-review.md) combines these
phase costs with the source-derived contact timescale. Compact logs address disk
pressure, but measured JSON time does not establish the main CPU limitation.
Before acquisition execution, explicitly justify the current update units and
one adequate finite teaching/resource budget. The large repetition projection is
unselected. No new component, gain sweep, difficulty increase or automatic short
teaching gate follows from preservation passing. Preserve local online learning,
the same learner/actor and the no-backpropagation constraint.
