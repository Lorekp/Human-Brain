# Stable language/action local-credit comparison

**Validated saved result, 2026-09-09.** Both workers and all eight checkpoint
batteries completed, and the corrected independent audit passes. All four neural
arms fail every main task. Causal-enabled produces one correct historical reading
at CP48 after failing it at CP0/16/32; there is no later retention probe of that
gain. No neuronal taught-skill acquisition, correct software edit, changed-context
success or productive recovery is established. Numerical preservation and equal
failures do not establish useful retained skill.

The [complete result bank](../../benchmarks/stable_language_action_credit_v1.result.json)
contains every per-case outcome, comparison, exposure counter and state/source
identity: 3,935,620 bytes, SHA256
`1bd3073fac22fdd54cc27e16ad7c8e75d106b15e9263e44bb1161b1a85c54919`.
The [frozen protocol](../../docs/research/stable-language-action-credit-comparison-v1.md)
binds runtime commit `5622f600921e6811d66254e25c8e956a31cd19ab` and manifest SHA256
`7b07b6984d6511a5f854d530775a4e7d9772d45ab9ce4330c2e3f4a74259c481`.
Old controller/learner source is authenticated V8 commit
`8e8df91d7e7c6889dff15f97d9fec379ebff71bb`.

There are five scientific arms: old-enabled, old-disabled, causal-enabled,
causal-disabled and a transparent retrieval/editing reference. Two serial workers
each execute three routes, giving six actual routes and two simple runs. The
second reference is a parity control, not an independent replication. All tasks,
source reoffers and repeated probes are exposed development material. No fresh
transfer or broad capability claim follows from this comparison.

## Main stream

These scores apply separately to each of the four neural arms and to each
simple run. Every planned case is supported and observed; none is omitted.

| Stratum | Each neural arm | Each simple run |
| --- | ---: | ---: |
| First taught-skill encounter | 0/2 | 1/2 |
| Changed wording encounter | 0/2 | 1/2 |
| Exact delayed encounter | 0/2 | 1/2 |
| Historical retention | 0/14 | 9/14 |
| Instruction construction | 0/10 | 0/10 |
| Structural combination | 0/10 | 0/10 |
| Recovery: first / any later / terminal correct | 0/8 each | 0/8 each |

Acquisition totals are 0/3 reading and 0/3 editing per neural arm; simple is
0/3 reading and 3/3 editing. Changed wording follows nine intervening attempts;
exact repetition follows 61. Simple's editing skill is already correct at the
first encounter, so its follow-ups show preservation, not new acquisition.
Historical simple scores split into 3/7 readings and 6/7 edits. Changed-context
scores are 0/20, including two dependent project edits; recovery includes all
24 attempts per route, not only terminal attempts.

Every neural route makes 44 accepted but incorrect SPLICE actions. Each simple
run makes nine accepted correct changed SPLICE actions, seven incorrect accepted
SPLICE actions and 28 incorrect STOP actions. Public checks pass on 1/44 old-enabled,
3/44 old-disabled, 2/44 causal-enabled, 2/44 causal-disabled and 10/44 per simple
run; a public-check pass alone does not meet the private task requirement.
All main correctness contrasts between credit rules or enabled/disabled routes
are zero. Enabled/disabled main comparisons have only 3/64 matching actual
input/RNG pairs in each worker; those agree on task behavior. Main forward parity
is not claimed for the remaining 61 unpaired attempts.

## Scheduled checkpoint outcomes

At **each of CP0, CP16 and CP32**, all four neural arms score zero in every group.
At CP48 only causal-enabled historical retention changes. First and final scores
agree throughout; all recovery attempts remain incorrect.

| Probe group | Each neural arm at CP0/16/32 | Old-enabled, both disabled at CP48 | Causal-enabled CP48 | Each simple run at every CP |
| --- | ---: | ---: | ---: | ---: |
| Both taught skills | 0/2 | 0/2 | 0/2 | 1/2 |
| Historical retention | 0/14 | 0/14 | 1/14 | 9/14 |
| Changed wording | 0/2 | 0/2 | 0/2 | 1/2 |
| Changed values/context | 0/2 | 0/2 | 0/2 | 0/2 |
| Structural combinations | 0/2 | 0/2 | 0/2 | 0/2 |
| Recovery: first / any later / final | 0/2 each | 0/2 each | 0/2 each | 0/2 each |

Taught reading A00-R and wording case A01-R fail everywhere. Taught edit A00-E
and wording case A01-E fail in every neural arm and succeed in both simple runs
at every checkpoint. Changed values/context also changes wording, so it does
not isolate value sensitivity.

The simple positive historical cases are R00-R, R01-E, R02-R, R03-E, R04-R,
R05-E, R09-E, R11-E and R13-E, preserved across the four checkpoints. R13-E
is already positive at CP0. R12-R, the saved counterpart of V7 ONLINE's correct
A04-R, already fails at old CP0; this is not a newly observed migration loss.
R10-R's earlier success belonged to a different, disabled lineage. No positive
neuronal anchor is demonstrated through CP32.

The sole per-case correctness transition is causal-enabled **R02-R at CP48**:
incorrect at CP0/16/32, correct at CP48. The request is “How many seconds on
delay?”, with sole accepted answer `2` (hex `32`), no files and no feedback.
All compared R02-R inputs have the same observation hash. The audited correct
outcome establishes that this one response meets that exact answer requirement;
the compact bank retains its raw-row identity, not a separate output-byte field.
Old-enabled and both disabled arms remain incorrect, while simple is correct
throughout. This is +1/14 against old-enabled and causal-disabled, with no other
gain or loss. Their incoming RNG states differ, so this is a descriptive late
hit, not a paired causal estimate. **No probe follows the hit:** stable retention
of the gain, reliable knowledge acquisition and generalization are unestablished.

## Fixed teaching and exposure ages

The same persistent acting learner continues through each main stream, with one
neuronal reader serving language and action payloads. The selected causal rule
teaches observed literal bytes and assigns local credit to preceding choices
without backpropagation or backward error transport. Existing motor corrections
and public consequences keep the frozen contract; private grades do not guide
actions or stop retries.

Each neural arm receives exactly two scheduled teacher offers: A00-R supplies
`cedar/maple`; A00-E supplies `return max(0,min(n,6))`. Actual totals per arm are
33 raw answer bytes, two EOS and three motor labels: **140 symbols and 12 labels**
across four arms. Disabled arms receive the same offers with learning disabled.
Each route receives 24 source offers containing 9,860 raw bytes plus 24 EOS.
Neural source learning is transient and factual memory stays at 1,982 slots;
simple retains source text in its history, a disclosed storage asymmetry.

These are actual exposure ages since each teacher offer. Attempts and source
symbols are the same across routes. Public-input symbols differ with actual
execution feedback. CP0 ages are null because neither correction has occurred;
disposable probe exposure is excluded from all ages.

| CP / skill | Main attempts | Source symbols | Old enabled public | Old disabled public | Causal enabled public | Causal disabled public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / A00-R | 21 | 2,913 | 7,699 | 8,461 | 8,039 | 8,537 |
| 16 / A00-E | 20 | 2,384 | 7,312 | 8,074 | 7,652 | 8,150 |
| 32 / A00-R | 43 | 5,892 | 17,109 | 18,187 | 17,651 | 18,136 |
| 32 / A00-E | 42 | 5,363 | 16,722 | 17,800 | 17,264 | 17,749 |
| 48 / A00-R | 63 | 9,795 | 24,649 | 25,780 | 24,972 | 25,722 |
| 48 / A00-E | 62 | 9,266 | 24,262 | 25,393 | 24,585 | 25,335 |

Main public-input totals are respectively 24,791 / 25,922 / 25,114 / 25,864
symbols; each neural arm additionally observes 9,884 source symbols and 35
teacher symbols. Full draw/category counters remain in the bank.

## Recovery and pairing limits

Each main route has eight three-attempt recovery episodes, hence 16 opportunities
to change a consecutive proposal after actual feedback. Each route's four probe
batteries supply another 16 such opportunities. These are proposal-record
changes, not necessarily correct or useful edits.

| Route | Main changed / identical proposals | Probe changed / identical proposals | Successful later repair |
| --- | ---: | ---: | ---: |
| Old enabled | 15 / 1 | 16 / 0 | 0 |
| Old disabled | 16 / 0 | 16 / 0 | 0 |
| Causal enabled | 16 / 0 | 16 / 0 | 0 |
| Causal disabled | 16 / 0 | 16 / 0 | 0 |
| Each simple run | 0 / 16 | 0 / 16 | 0 |

No later recovery attempt is correct in either main or probes. Changed proposals
therefore provide no positive evidence of productive feedback use.

At CP0 old/causal numerical initial states agree, and each neural comparison
passes all 23 required pairs: 22 independent cases plus the first C00-E attempt.
Forward signatures include currents, probabilities, choices, generated bytes,
context/cursors and task outcomes. There are zero paired mismatches, five later
unpaired recovery attempts and 11 raw-diagnostic differences per neural comparison,
including six within required pairs. Enabled/disabled CP0 has the same 23/5
pairing limit. Raw feedback is never normalized: differing execution diagnostics
remain actual subsequent inputs, and final RNG equality is not required after
those paths diverge. Equal failures establish the declared migration preservation,
not positive skill retention.

At CP16/32/48 all 28 attempts per old/new neural comparison and per
enabled/disabled comparison are unpaired by actual input/RNG. Their group scores
remain complete descriptive contrasts. The two simple probe runs match on all
28 attempts at every checkpoint with equal raw diagnostics. Their main runs
have 63 paired equal task outcomes and one unpaired F06-E outcome, with seven
raw-diagnostic differences overall; full main parity is not established.

## Completion, state custody and audit history

Each worker completes 192 main attempts, 132 executions, 72 source rows, six
teacher rows and 265 safe publications through HEAD sequence 264. Each of its
four batteries completes 84 attempts and 51 executions with no source/teacher
offers or live-state publication. Combined totals are **1,056 attempts and 672
actual executions**: 384/264 main plus 672/408 probe. All disposable actors and
references are released without cleanup errors; slow and factual state remain
held during every battery. Main actors finish at actor experience 369 and reader
experience 342; each CP48 disposable battery advances its own counters to 386/370.

The shared initial and both final disabled slow numerical hash is
`b47e11a330752cf5adc146db1edf43d969b246ce7e2d5e648d7aa8e6fa45be86`.
Final old-enabled slow hash is
`b051a3b7aa6bdd7c39b75c0de4abf57031c54ce10258dfcfa8bb0061d146ee43`;
final causal-enabled is
`9ae5e049f7fe7edd7eb28ac5c4293d91b1ae9abdae61d242a73fb7d8e902937f`.
All neural factual hashes remain
`cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e`.
Final boundary hashes are old
`9e862e97279e2d9c021172f3dedd25460fa02ad6b47a16cd4f6b505908a330a0`
and causal `88ddafe138abc25f29d878e4c5e6848fa9ecb6c66960ab105e603450d1ebc63a`.
The bank binds every final model/payload and the distinct RNG frontiers.

The original outer launch remains **failed** after both workers completed:
6,472.8048409 seconds and 1,453.3203125 MiB observed process-tree peak. Three
saved-reader defects were corrected separately: unconditional reading `grade`
access, phase deltas incorrectly equated to adjacent cumulative differences,
and a missing required `partial=False` probe-journal argument. The
[correction record](../../docs/research/stable-credit-v1-audit-correction.md)
preserves every failed review, regression and cost; no task, teacher, model or
checkpoint was replayed or rewritten.

The final saved-only audit uses auditor SHA256
`116e77e187749b10812fda30bfbc6f770492e55cc4ea46861cf283a8bd650423`.
It passes in 255.2342021 internal seconds, 256.7525935 outer seconds and
115.03125 MiB observed peak, with zero model constructions, tensor decodes,
predictor calls, teacher replays or execution children. These reporting costs
are separate from the original launch. Journal-path timing is documented in the
[matched cost result](../../docs/research/owned-journal-prefix-cost-v1-result.md);
it does not establish a complete-checkpoint or whole-session speedup. No new
mechanism or experiment is selected by this result.

The [learning review](../../docs/research/stable-credit-v1-learning-review.md)
separates the small observed literal-prediction improvement from task outcomes.
Its [diagnostic bank](../../benchmarks/stable_language_action_credit_v1.diagnostics.json)
is SHA256 `23f574a1a30e21c6bd18ef12219b52721fcfcaa3f23226a53f5d780987cca0a2`.
The [measurement ledger](../../benchmarks/stable_language_action_credit_v1.measurement.json)
records 7,504.8741267 seconds across 22 listed monitored processes, including
implementation QA, matched journal measurement, preparation and all reporting
failures. The separately linked banking monitor adds 1.2759124 seconds, giving
7,506.1500391 recorded process seconds (about 125.10 minutes). This is a sum of
those process durations, not total project or agent time; manual source review,
metadata commands, doctor/lint and documentation work are explicitly excluded.
Nested worker/probe clocks are not added again. The maximum observed process-tree
peak is 1,453.3203125 MiB.
