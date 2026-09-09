# Causal credit and owned journal implementation QA

2026-09-09. **Five selected cases pass** at source commit
`32542101f4698a6346e931acd93abaf1bcdde3bd`. This is ordinary implementation QA
for the [selected local-credit rule](causal-observation-credit-v1.md) and
[owned journal design](persistent-language-action-v8-cost-review.md#selected-next-source-design-owned-journal-prefixes).
It establishes arithmetic, transaction custody, actual fixture continuation and
byte-level journal checks. It establishes neither useful learning nor a measured
speedup. The [active plan](persistent-language-action-development-v1.md) still
requires the stable curriculum, fixed teaching budget, paired behavioral migration
comparison and complete resource plan before a new measured session.

## Execution and evidence

The checkout-local Python 3.11.15 environment passes
[doctor](../../.pytest-tmp/dev/doctor-7rrhfzl7/receipt.json) before pytest.
Doctor records the commit above, an empty tracked diff, CPU, NumPy 2.4.6 and
pytest 9.1.1. The existing untracked
`src/humanbrain/direct_request_splice_learner.py` is disclosed by doctor and is
outside the selected work. Untracked contents are not covered by its diff hash.

The [launch receipt](../../.pytest-tmp/causal-credit-owned-journal-qa-3254210.launch.json)
records the exact interpreter and command, with `-X utf8 -m pytest -o addopts=
-ra --durations=10`, a new retained basetemp and JUnit destination. Four selected
function names expand to five cases because actual checkpoint continuation has
ordinary-journal and owned-journal parameters. Exit code is zero; no cases skip.
The [stdout](../../.pytest-tmp/causal-credit-owned-journal-qa-3254210.stdout) and
[JUnit](../../.pytest-tmp/causal-credit-owned-journal-qa-3254210.xml) report
5 passed, 1 warning, 64.293 seconds inside pytest. The outer watchdog records
65.6903893 seconds and 1281.69921875 MiB against limits of 180 seconds and
4096 MiB. Memory is the largest sampled simultaneous sum of live owned process
peak working sets at 500 ms intervals; brief children can be missed. Individual
process peaks must not be added again. Doctor takes a separate 0.297 seconds.

The single warning is `PytestCacheWarning`: Windows denies creation of
`.pytest_cache/v/cache/nodeids`. It does not prevent case execution or retained
JSON/JUnit results. This report reads only these completed local QA receipts and
source identities; it launches no additional tests, actors or child programs.

## Actual cases and exposure

All nodes are in `tests/test_byte_sequence_decoder.py`.

| Node | Seconds | Actual bounded work |
|---|---:|---|
| `test_recurrent_content_local_arithmetic_and_visibility` | 2.871 | One standalone controller; six ticks, 23 draws; four scan offers, one teacher offer, one free tick; zero actors or children. |
| `test_recurrent_content_same_actor_online_lifecycle` | 33.016 | Two recurrent actors; eight successful and two rejected source offers; 105 public scan offers; three free calls/ticks; four teacher symbols; three motor labels; one actual child. |
| `test_persistent_checkpoint_actual_actor_continuation[ordinary-journals]` | 12.177 | Four actors including two fresh restore targets; two tiny ingests; six free calls; two teachers/four symbols/six motor labels; two actual children; three successful publications and one injected failed commit followed by the valid ordinary-path retry. |
| `test_persistent_checkpoint_actual_actor_continuation[owned-journals]` | 11.126 | The same actor/exposure counts; three successful publications, no failed commit; owned journals and independent fresh restore. |
| `test_persistent_owned_journal_refusal_and_interruption` | 0.063 | Authored bytes only; four successful row appends and one failed physical-tail append; one injected publication-wrapper interruption; zero models, source/teacher offers, children, tensor loads or successful checkpoint publications. |

The arithmetic oracle sums causal per-symbol prefix credit independently of the
implementation's `A*E-P` regrouping. It checks upstream expected-match credit,
literal direct cross-entropy, complete external denominators, public consequence,
held baseline publication and the declared norm bound. Four target offers include
literal teaching; one target offer deliberately takes a nonliteral branch. The
already-counted first literal offer has a finite subnormal target probability
and checks its stable log-sum-exp loss. Refusals cover a premature target, four
NumPy mutations of sealed pre-target probability/log-probability/index/activity
tensors, and two mutations of finished fork credit banks. One valid and one
invalid in-memory strict load occur; no checkpoint file is decoded. Three
additional frame-only receives are visibility checks, not learner offers.

The recurrent lifecycle has two admitted documents and three transient documents,
including the rejected transient transaction. Each document is one raw byte plus
EOS. Its 122 controller ticks and 366 draws include the rejected two-offer
transaction; 119 of those ticks have genuine targets and three are free output.
The seven held-prior calls are three free and four teacher calls. It verifies
held old-reader tensors, source admission versus transient visibility, the common
scan and correction fork, post-free RNG continuation, disabled publication,
actual action before correction and atomic late rejection. Mutating a saved
trace or either of two pending credit banks is rejected. It has one valid and
one invalid in-memory strict load, no checkpoint-file load, and 44 frame-only
receives outside learner exposure.

Each checkpoint parameter ingests two raw bytes plus two EOS across its two
original actors. Its four teacher symbols split two enabled/two disabled. It
restores once into two fresh independent actors, decoding exactly two payloads
created by that case, then compares actual continuation outputs and cached tensor
independence. Together the cases create ten actors plus one standalone controller,
execute five actual action children, offer 13 teacher symbols and 15 motor labels,
publish six checkpoint boundaries, and perform two restores/four local payload
decodes. No historical experimental checkpoint, corpus, saved lesson or model is
loaded, and no broad learning session or benchmark runs.

## Journal and restore evidence

Both actor cases actually use the Windows NTFS immutable payload-lease backend.
The owned-journal actor case also uses the native append-only writer: three writer
opens and four native write-through completions authenticate four rows. The byte
refusal case confirms the native backend and separately forces one unsupported
fallback writer, whose one append uses fsync and refuses cached-prefix access.
Native write-through is not described as a `FlushFileBuffers` or fsync call.

The following counters are each case's saved cumulative `commit_work` snapshot
at its third successful commit, not a sum of intermediate cumulative receipts:

| Counter | Ordinary journals | Owned journals |
|---|---:|---:|
| Final outcome-journal bytes | 8,222 | 8,219 |
| Parsed new event bytes | 8,222 | 8,219 |
| Owned actual tail bytes read | 0 | 8,219 |
| Avoided journal-prefix hash bytes | 0 | 49,215 |
| Journal cached receipt checks | 0 | 48 |
| Journal full hash checks | 60 | 3 |
| Included journal fresh hash checks | 6 | 3 |
| All checkpoint file-hash bytes | 869,879,154 | 695,857,411 |

Fresh reopened stores have no inherited owners or payload leases. Each saved
`restore_work` includes the explicit reopen inspection and restore inspections:
nine full journal checks, including six fresh checks; six full payload checks,
including four fresh checks; and two restricted local payload decodes. Its total
file-hash reads are 521,917,516 bytes for ordinary journals and 521,917,507 for
owned journals. These are software-accounted passes over files, not measured
physical-device traffic. The full-read counts demonstrate that live-owner reuse
does not carry across fresh open/restore.

The byte-only node records eleven refusal categories: existing-file creation,
append and close during freeze, external append, truncation, deletion,
replacement, writable mapping, unsupported cached-prefix access, mismatched
actual disk tail, and interrupted publication. Its standalone writers account
114 actual written/read tail bytes, including the rejected mismatched tail; the
separate interrupted store adds 32 written/read tail bytes and 32 avoided prefix
bytes. The mismatch preserves disk evidence and poisons continuation. The
publication fixture injects the real commit wrapper's failure path before actor
validation; it preserves intent, creates no HEAD and advances no committed
frontier. It is not a complete neuronal publication fault-injection test.

The 12.177 versus 11.126 second case durations are **not a speedup measurement**.
The ordinary case performs an additional failed commit/retry and fresh checks,
the serialized rows differ in size, and execution order/cache effects are not
controlled. Counter savings under proved ownership are supported; attributable
latency improvement on the complete stable workload remains unmeasured. No
cross-version v1-to-v2 action comparison or positive historical-skill retention
test occurs in this five-case QA.

## Exact source and receipt identities

SHA256 values below identify the checkout file bytes used for this execution.

| Source | SHA256 |
|---|---|
| `src/humanbrain/recurrent_content_controller.py` | `9b6568d315c2069db250f342c67dfe8d9946ec963f668c263cdcd8bdbbc0f870` |
| `src/humanbrain/recurrent_content_learner.py` | `3a7d85612a8313bc9da427525520e294af05d191e3d5b84c1e0b9330f94ded5a` |
| `experiments/r4_integrated/persistent_checkpoint_journal.py` | `b46f4f222634fae2d19ebb6e7c9dd6485b1a52dfd472c853e871bb6d70abcd1d` |
| `experiments/r4_integrated/persistent_session_checkpoint.py` | `19efd7f21d3387cdaec9d08618a8e2d1320696c1b1303a95d1ec8e2f42a7bc1c` |
| `tests/test_byte_sequence_decoder.py` | `00bae6f9de29d6f0e49d6ac9a38887596f246f4bb9887344ba7d64d47305993f` |

All case receipts below are under
`.pytest-tmp/causal-credit-owned-journal-qa-3254210/`.

| Receipt path | SHA256 |
|---|---|
| `test_recurrent_content_local_a0/recurrent-content-arithmetic-passed.json` | `df2bf28972e7b2151c80649e763823303b6d3bd0300564ddfc54737ba80534a5` |
| `test_recurrent_content_same_ac0/recurrent-content-same-actor-passed.json` | `70dda06a7bbe6bae66ddc6e26f26eff00463d473f756227c47f2bf646e66f95c` |
| `test_persistent_checkpoint_act0/persistent-checkpoint-continuation-passed.json` | `de55850ad46523d7089978ba8ddd52d253ba86c30577c9a96ce57349ef44cc84` |
| `test_persistent_checkpoint_act1/persistent-checkpoint-continuation-passed.json` | `94ec28fedd9182e7f72c18172d8abc8845612b3f2e7f22da6df681ba23c4b691` |
| `test_persistent_owned_journal_0/owned-journal-refusals-passed.json` | `c0b24ca12ccf1cbddea9badebc1c6687e887cc7b19ecfd5d0286ed559bd30486` |

The launch receipt SHA256 is
`445ca60c138e9160b42082f2918e2e5ee53d95fce6c1e4743399a1d43be5fa3a`;
JUnit is `ab31b79059a389b7e6fab2ffacef85923daf1308d0a3a06bf9dd8a4845c210e7`;
stdout is `1a721b6259cdd72bcef0aee3f8f096e57337bf6e5ec07c1823c1e1176b5e2d5e`;
doctor is `bd8e8742f5c18e7a35d9e18d382b13edefbc87a60d2213f3b8d4208e1494e94f`.
