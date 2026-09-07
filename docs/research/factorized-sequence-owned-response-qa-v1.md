# Factorized routing: owned-response ordinary QA

Status: two existing mechanics tests passed on their first monitored invocation,
2026-09-07, at source `f2adc63f1402051aaf192377aa4bba9901fca423`.
This is ordinary affected QA for an integral implementation optimization. It
does not rerun the completed V5 workload or establish a full-workload speedup.
The [banked V5 cost review](persistent-language-action-v5-cost-review.md) remains
unchanged.

## Change and ownership

Public `FactorizedSequenceRouter.evaluate` still validates the complete current
population and live theta or teacher stage. Its ordered float64 arithmetic now
lives in one private kernel shared with the free-response caller: every owner,
candidate activity check, gather, product, one-dimensional reduction, baseline,
threshold competition and selected-row reconstruction remains the same.

During one synchronous `predict_with_receipt`, the reader uses the canonical
population validation already performed by `build_context`. It borrows one
theta reference, version, owner identity and router-domain byte digest for that
call. Nothing is cached across calls or added to a checkpoint. Identity/version
checks run at every owned routing event and at the response boundary. Existing
end checks still bind model, tape, candidate views and their prototype sources
by bytes before a response becomes pending. Those byte checks remain essential:
NumPy writes do not advance a Torch tensor's version. Public evaluation, teacher
staging, pending validation and atomic publication retain their checked paths.

## Executed scope

The exact nodes were:

- `test_factorized_router_local_outcome_contract`
- `test_factorized_reader_action_persistence_and_atomicity`

The standalone router node used one router, two authored candidate views and one
current population. It executed two router evaluations / three activity calls,
three accepted direct competition cases, two rejected direct competition cases,
and one internal competition. Five outcome attempts yielded three accepted
offers, one rejected tampered decision and one rejected stale decision. There
was one staged update and commit, one snapshot and one same-instance restore;
zero reader models, ingests, reader teachers, children or checkpoint loads.

The reader/action node used one actor with one reader, one ingestion of 512 raw
bytes plus EOS, and a 513-slot tape. Its saved receipt records:

| Activity | Actual count |
|---|---:|
| Successful standalone reader predictions | 3 |
| Successful action/payload prediction | 1 |
| Completed free byte/EOS decisions | 8 |
| Additional checked public router evaluations | 6 |
| Teacher-free acknowledgement | 1 |
| Rejected free calls / completed events within them | 5 / 5 |
| Teacher calls, including the rejected call | 4 |
| Offered teacher symbols, including rejected-call exposure | 8 |
| Successful enabled / disabled / rejected teacher calls | 2 / 1 / 1 |
| Real qualified child executions | 1 |
| Successful same-instance state restore / rejected loads | 1 / 3 |
| Experimental checkpoint loads | 0 |

The six reference evaluations had owner counts `[0, 2, 2, 2, 2, 2]`. They
compared complete decisions, seals, float64 scores, selected activities and
competition traces with the unchanged public checked path. Every work-counter
difference was required to equal the explicitly removed validation/hash work.
A controlled zero-current output produced three successive raw events through
changing prefixes, then teacher-free acknowledgement and exact state restore.
No learned-accuracy assertion was needed for this reuse check.

The five rejected free calls cover the existing equal-value outgoing-buffer
replacement plus theta replacement, persistent NumPy mutation with unchanged
version, a same-value Torch write with changed version, and router-owner
replacement. Each failed before publishing pending state or incrementing the
reader experience counter. Restored model and tape values were checked.
Existing disabled-learning, strict-load rejection, pending-receipt tampering,
and late joint-publication failure assertions also passed. The one real STOP
execution returned code 0 and stdout `0\n` before the same actor received its
action correction; no extra child was introduced by this optimization.

Relative to the previous node, the extension added exactly one successful
three-event prediction, one acknowledgement, four one-event rejected calls and
the six checked router reference evaluations. Model/ingestion/teacher/child
counts were not increased.

## Timing, environment and limits

The outer watchdog passed in **12.7656899 seconds**, with observed owned-process
peak sum **701.8046875 MiB**, under 60 seconds / 2048 MiB. JUnit reports two
passes, zero errors/failures/skips and 11.604 seconds total (pytest displays
11.60 seconds). Recorded testcase times are 0.137 and 7.756 seconds. This
watchdog samples owned process peaks every 500 ms; brief unobserved children
can be missed. These are ordinary QA timings, not an isolated routing benchmark.

The selected-checkout doctor passed in 0.281 seconds, with the expected revision,
Python 3.11.15, pytest 9.1.1 and CPU device. Its tracked diff hash is the empty
SHA256. It disclosed the pre-existing untracked
`src/humanbrain/direct_request_splice_learner.py`; the checkout was clean for
tracked source, not empty of all untracked files. Root's explicit repository
configuration Ruff check passed for both changed modules and the complete
existing test file. Pytest's only warning concerned inability to write its
cache directory; both nodes passed and stderr was empty.

## Analytical work reduction

Let `D = 1033304`, lexical width `L = 1033290`, `E` be actual free decisions and
`F` successful free calls. The new path removes one redundant full context
validation and one theta finite scan per decision. Each removed context check
accounts for `D` validation values, `L` cast/norm products, `L - 45` norm
reductions and `16L` cumulative named temporary bytes. Theta hashing changes
from `E` full hashes to `F`, removing `(E - F) * 8D` counted bytes. Entry/end
validation, receipt sealing and all pending/teacher custody work remain.

For the already recorded V5 trajectory, `E = 2970` and `F = 160`: this arithmetic
corresponds to 3068912880 fewer context-validation values, the same number of
fewer theta-validation values, 49101940800 fewer cumulative named temporary
bytes and 23228673920 fewer theta hash bytes. These are source-operation
differences on the same recorded event counts. They are not measured seconds,
peak-memory reductions, or evidence that a rerun would follow an identical
trajectory. No V5 model replay, checkpoint decoding or speed experiment was run
for this QA or this documentation.

## Saved evidence identities

All paths below are relative to the repository. The three fixture JSONs were
read and hashed after completion; the access-restricted doctor and fixture
directory required an escalated read-only shell. No artifact was regenerated.

| Evidence | SHA256 |
|---|---|
| `src/humanbrain/factorized_sequence_router.py` | `3b9c071ccb5db49bdc3d780b3805255e0064682db7cd82bd6d106b79ba0888d3` |
| `src/humanbrain/factorized_sequence_memory_learner.py` | `4c98d4324697218b424b400b02aa3e88be281d786e24278d084679414b90ba2d` |
| `tests/test_byte_sequence_decoder.py` | `fee1638964802ece249f8cf1576c76eba8dde5c6f6b74fd3bdced08a6da7ec0b` |
| `.pytest-tmp/dev/doctor-bxyl1ltf/receipt.json` | `7d7bb39aa51b2c29f37434f6ec65076cfbff344d09249054eaedf42510729b83` |
| `.pytest-tmp/factorized-owned-free-qa-v1.launch.json` | `e7b34a7a829feafd5480701d18528d6667d2286e3a339152ae4f7d61422cad28` |
| `.pytest-tmp/factorized-owned-free-qa-v1.junit.xml` | `b01a01dc99120a8ae01066ebd863a5d205b1b771b28456a65e4710f6fb6666cd` |
| `.pytest-tmp/factorized-owned-free-qa-v1.stdout` | `8d37d43aa90d34e4306db593e87d478b3b68759b0edd346861c288ee84e02c5a` |
| `.pytest-tmp/factorized-owned-free-qa-v1.stderr` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `factorized-integration-passed.json` | `650b8b914a35c9f43cf323f1f370303d041e6637868156f41f19bc34d795b942` |
| `factorized-router-passed.json` | `de87ae2b514689fbb655f8553b469d4ebfa5d101d2645b93963bb9f91d766df8` |
| `factorized-router-raw.json` | `d76f81c8bddb69a61711ac061dbb68f5dc539f82f5e9452a19bd58fa4656a5a9` |

Fixture locations are under `.pytest-tmp/factorized-owned-free-qa-v1-tmp/`:
`test_factorized_reader_action_0/` contains the integration receipt;
`test_factorized_router_local_o0/` contains both router receipts.
