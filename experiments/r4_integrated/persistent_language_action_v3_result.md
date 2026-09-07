# Persistent language/action continuation v3

The full continuation and independent saved-record audit passed on their first
launches. **The neuronal learner still solves none of these broader tasks.**
The actual-path correction is implemented and mechanically verified, but useful
acquisition, transfer, retained competence and autonomous recovery remain unmet.

Frozen source: `fc7f614cfbc7c01489245c6fec05520a9d1c586f`. The
[result bank](../../benchmarks/persistent_language_action_v3.result.json) binds
the source, manifest, saved state, measured costs and exposure. This is authored
public development, not a protected language test or a SWE leaderboard result.

## Four separate outcomes

| Outcome | Online neurons | Learning disabled | Retrieval/editing reference |
| --- | ---: | ---: | ---: |
| Supervised acquisition, first responses | 0/24 | 0/24 | 12/24 |
| Fresh transfer, first responses | 0/16 | 0/16 | 6/16 |
| Earlier-session retention | 0/8 | 0/8 | 8/8 |
| Teacher-free recovery among first failures | 0/8 | 0/8 | 0/8 |

Acquisition interleaves six requirements through reading and executable actions,
then revisits them in distinct instruction forms after twelve episodes. Both
neuronal routes score `[0,0]` across the two blocks of twelve; the reference
scores `[4,8]`. The reference solves six reading tasks and six edits. These are
actual first attempts before each declared correction, with no immediate
post-teacher probe or repeated fitting until success.

Fresh tasks introduce negation, reversed ordering, arithmetic, suffix composition
and new code structures. The reference solves two reading tasks and four edits.
Supported-subset scores are 0/12, 0/12 and 5/12. All four longer-instruction
challenges remain in overall totals; the neuronal 31-byte request bound rejects
them unchanged. These are new authored forms and combinations over already
observed facts, with disclosed lexical overlap. They are not a fully disjoint
or protected language-family holdout.

Retention repeats eight exact delayed tasks from the previous session after
further intervening work. The reference preserves its eight earlier successes.
The neuronal routes had already failed these tasks, so this does not demonstrate
forgetting eight acquired skills. All eight recovery tasks fail initially in
each route, receive three fixed actual attempts and remain unsolved. No tool-text
encoder or local consequence update is present; acknowledgement alone does not
provide a recovery mechanism.

## Persistent learning and actual execution

Both neuronal routes continue the completed v2 online numerical model, whole
505-slot tape and action experience count 92. They start from identical values
with independent storage. No numerical parameter is added, reset or selected
from task outcomes. The new policy changes output teaching; exact incremental
prefix activity reuse preserves the free equations. No prior model decoding is
replayed to initialize the continuation.

The same reader supplies language responses and action payloads. Each correction
uses the first mismatching event in that reader's actual recorded response,
including its own selected feature and currents. At most one local output clamp
is offered per response. B/C address teaching still uses the full offered target
with U held during its private stages, followed by atomic publication. There is
no backpropagation, separately fitted executor, target-address selection or host
repair. Teacher-free responses explicitly resolve their pending state.

Every neuronal route receives 24 corrections containing 148 raw bytes and 24 EOS
symbols, plus 36 action labels. All 172 symbols feed the address-teaching path;
24 active output clamps are offered. Online publishes 24 updates each to U, B
and C. Learning-disabled computes matched offers but publishes none. This is
**more full-target exposure than v2's 98 symbols**; fewer output updates do not
mean less teacher information.

The tape stays at 505 occupied slots throughout, with no append, eviction or
task-selected context. This extends retention age rather than memory capacity.
The conventional reference restores its actual 18 documents and 24 earlier edit
demonstrations without prediction replay, then receives twelve new edit
demonstrations. Its host retrieval and edit reuse remain separate reference
computation.

Across all routes the run records 216 responses, 144 real Python executions,
72 corrections and zero ingestion events. Workspaces reset at episode starts
and persist only within retries; model and shared memory persist across phases.
Both final neuronal model/tape states are saved for continuation.

## Verification and resources

The [ordinary QA](../../docs/research/active-path-output-qa-v1.md) verifies exact
incremental features, the local clamp arithmetic, unchanged disabled parameters,
late atomic failure and actual execution before teaching. It is mechanics
verification, not acquired-skill evidence. The full run used doctor receipt
`.pytest-tmp/dev/doctor-pxh9lmh_/receipt.json` and an empty tracked diff.

The monitored run took **56.9543509 seconds** at **510.5703125 MiB** sampled
process-tree peak. Independent saved review took **3.6428642 seconds** at
**512.5234375 MiB**, totaling **60.5972151 seconds** within the declared
900-second/2048-MiB envelope. Final model/tape tensors remain 4,275,920 bytes.
This is a different workload and teaching policy from v2; the elapsed ratio is
not an isolated measurement of the prefix optimization.

The checker independently verifies saved behavior and actual applied bytes,
causal prefixes, current normalization/selection, the recorded first-error
binding, teacher/acknowledgement counts, reference restoration and persistent
state chains. It performs no model replay or checkpoint deserialization. The
private full trace hash covers more arrays than the saved summary and is not
independently reconstructed. Process sampling every 500 ms can miss brief child
peaks; actual execution command/PID/output records remain retained.

Evidence: `.pytest-tmp/persistent-language-action-v3-run/`,
`.pytest-tmp/persistent-language-action-v3-review.json`, and their two monitored
`.launch.json` receipts. The unused direct-request branch remains unexecuted
source. Its presence in the source inventory is not model activity.

## What this result changes

The [learning review](../../docs/research/persistent-language-action-v3-learning-review.md)
finds changed outputs on seven of 22 supported readings and twelve of 30
supported first edit payloads versus the disabled control. Nevertheless, every
one of the 46 neuronal edit payloads per route reaches the byte limit without
EOS. No related acquisition pair gains a longer correct prefix. Twenty-two
output clamps target the first event and two the second; none targets EOS.
The whole supplied answer therefore does not become whole-sequence output
teaching under this rule.

The 24 recorded correction features have mean cosine overlap 0.80728. Contexts
requiring different bytes overlap about as much as those requiring the same byte.
Two corrections also have a negative projected effect on an earlier correct
event's margin at its held recorded activity. These are measured features and
scalar local-update effects, not regenerated responses or a complete causal
explanation of later failures. They motivate reviewing context-specific local
learning together with coverage of the already offered sequence.

The [evaluation review](../../docs/research/persistent-language-action-v3-evaluation-review.md)
attributes all 72 reference outputs to available history. Several successful
composition edits insert a literal into logic already supplied by the program;
the corresponding composition readings fail. Broader missing-logic software,
longer instructions and actual use of diagnostic observations remain important
next pressures, without requiring perfection on these exposed regressions.

The [resource review](../../docs/research/persistent-language-action-v3-cost-review.md)
verifies 90.0007% fewer prefix pair reevaluations on the same saved trajectories.
It also identifies 82.64 MB of retained attempt JSON and an auditor peak slightly
above the runner's. Efficient simulation and bounded receipt handling remain
integral to growth, but further speed tuning is not a prerequisite for the next
broader learning change. Gemma and Opus benchmark comparisons remain unattempted.
