# Sixth persistent language/action session: partial evaluation review

Status: saved evidence checked after a wall-budget failure, 2026-09-07.
The run stopped at 840.4533474 seconds, peak 890.42578125 MiB, exit 1.
It did not produce a report, completion record, final checkpoint or failure JSON.
This is a partial evaluation, not a passed completion audit. The complete V5
checkpoint remains the last recoverable learner state.

The durable prefix contains every acquisition, fresh-transfer and prior-regression
case for all three routes. Online obtains one exact delayed reading success,
no fresh success and no correct software edit. Recovery is interrupted. These
results do not establish a useful broader advantage for recognition learning.
The [result bank](../../benchmarks/persistent_language_action_v6.result.json),
[learning review](persistent-language-action-v6-learning-review.md) and
[protocol](persistent-language-action-v6-protocol.md) provide complementary scope.

## Observed scores and missing outcomes

All denominators below are fully observed first responses. Acquisition combines
three distinct encounters and is separated further below.

| Route | Acquisition reading | Acquisition edit | Acquisition total | Fresh reading | Fresh edit | Fresh total | Prior reading | Prior edit | Prior total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Online | 1/9 | 0/9 | 1/18 | 0/12 | 0/12 | 0/24 | 0/9 | 0/9 | 0/18 |
| Correction-disabled | 0/9 | 0/9 | 0/18 | 0/12 | 0/12 | 0/24 | 1/9 | 0/9 | 1/18 |
| Simple reference | 3/9 | 3/9 | 6/18 | 0/12 | 0/12 | 0/24 | 4/9 | 7/9 | 11/18 |

| Route | Six first responses subsequently corrected | Six changed-requirement encounters | Six exact delayed repeats |
|---|---:|---:|---:|
| Online | 0/6 | 0/6 | 1/6 |
| Correction-disabled | 0/6 | 0/6 | 0/6 |
| Simple reference | 1/6 | 1/6 | 4/6 |

The four metric families are acquisition, fresh transfer, prior retention and
teacher-free recovery. Only the first three finish. There are 182 durable attempts
of 288 planned: online 62, disabled 60 and simple 60. Online `C00-E` has two
incorrect, rejected attempts; its third attempt is the next missing event. The
second consumes actual execution feedback. No recovery episode has a recorded
terminal attempt, and neither comparison route begins recovery. Thus recovery
has two observed failures in one incomplete episode; a full 0/12 score or a
matched recovery comparison would be misleading. There are 106 missing planned
attempt outcomes.

The online success is `N06-R`, exactly `hold/express`, on a delayed repeat of an
already corrected request. Forty-seven intervening attempts and changed factual
memory separate the encounters. This is evidence of one retained response under
that schedule, not of changed-wording or changed-requirement transfer.

## Task novelty, memory and exposure

The 24 fresh cases comprise twelve instruction-construction and twelve structural-
combination cases; each stratum is 0/12 for every route. They retain vocabulary,
facts and component operations from public development. Whole utterances and
combinations are new at their scheduled encounter, but this is not a whole-family
language holdout. The six changed-requirement acquisition cases also change what
the answer/program must do: their score is not a pure paraphrase test.

All twelve document append occasions complete for each route. The last durable
factual occupancy is 1,760, grown from 864 without eviction, within capacity 2,048.
The new observations contain 884 raw bytes plus twelve document EOS symbols per
route, or 2,652 raw bytes and 36 EOS symbols across the three routes. These are
ordinary memory writes, not source-next-byte prediction labels. Increased memory
and retention age are both present, so they cannot be isolated as causes here.

All six scheduled correction occasions finish for each route: three reading and
three editing. Each neural route receives 135 reader target symbols (129 raw
bytes plus six EOS) and nine action labels. Calling this merely six labels would
hide the output teaching exposure. The simple reference sees the same occasions
and stores three new edit demonstrations; reading corrections are not stored by
its fixed policy. Eighteen teacher rows and 36 ingest rows are durable. The
disabled route computes correction work while keeping its learned state frozen;
the interpretation of local publication and event recruitment belongs to the
separate learning review.

## Actual actions, continued projects and feedback

Every observed request/workspace is supported by the declared interface. Over the
first 60 episodes, the thirty edit outcomes per route are:

| Route | Correct accepted SPLICE | Accepted SPLICE, process failure | Accepted SPLICE, wrong behavior | Rejected proposal | Incorrect STOP |
|---|---:|---:|---:|---:|---:|
| Online | 0 | 21 | 3 | 6 | 0 |
| Correction-disabled | 0 | 27 | 3 | 0 | 0 |
| Simple reference | 10 | 9 | 1 | 0 | 10 |

Accepted incorrect patches remain useful distinctions from rejected syntax or
interface proposals. Success is independently recomputed from actual return code,
exact required stdout, protected `main.py` integrity and complete captured files;
canonical patch equality is not required. The two additional online recovery
actions are rejected. All 92 durable execution receipts have complete checked
workspace snapshots: 32 online and 30 each for disabled and simple.

Three projects continue actual files along `N00-E → N03-E → F06-E`,
`N01-E → N04-E → F07-E` and `N02-E → N05-E → F08-E`. All nine stages fail for
every route. Each of the six successors per route carries actual preceding files
and consumes actual feedback. The three dependent fresh edits consequently start
from failed earlier programs; they are not independent clean-workspace transfer
tests. The other nine fresh edits also all fail. Outside these explicit chains,
episodes reset their workspace, while retries continue the actual workspace.
Persistent parameters and factual memory do not imply one uninterrupted project.

Feedback is raw public stdout/stderr, return code and completeness/status fields.
Its inclusion in an observation proves access, not successful use or an effective
teacher-free learning rule. Private correctness neither supplies an answer nor
changes the fixed retry allowance. The missing third recovery response prevents
any terminal recovery conclusion.

## Reference attribution and retention

The reference starts from 24 observed documents and 46 actual historical edit
demonstrations, restored through authenticated journals without prediction replay.
By the last completed append/correction it has 36 documents and 49 demonstrations.
All sixty recorded reference calls preserve attribution: thirty use trigram
cosine followed by the first boundary-delimited quoted literal, and thirty use
the nearest demonstrated query plus a uniquely matching old literal anchor.
Actual feedback contributes through the declared generic raw query frame; there
is no error-message interpreter, task-ID lookup or correct-patch input at act time.

The successful delayed edits `N06-E`, `N07-E` and `N08-E` select current-session
demonstrations 46, 47 and 48. Each has identical query hashes, similarity 1 and one
anchor match. Their replacements are respectively the previously offered dispatch
conditional, guarded retry bound and normalized parcel-name expression. This
explains their three edit successes as demonstrated-patch reuse rather than
construction of new logic. Conversely `N00-R` extracts `hold` where the request
requires `hold/express`; `N00-E` retrieves an older `ping`/`mute` conditional for
the new dispatch task. The simple policy lacks the requested composition even
when it retrieves relevant words or syntactically plausible code.

This review retains attribution for every observed reference call and validates
all actual reference edit applications and behavioral grades. It does not claim
an independent recomputation of all cosine rankings or replay the baseline API.
Reference limitations are therefore explicit, and its failures are not evidence
that the workload has been solved by another general language learner.

All 54 current prior-regression observations exactly match their actual saved V5
counterparts. Both neural routes compare against prior online observations; the
simple route compares against its own. Raw intervening-attempt ages are 51 for
`R00`–`R11`, 57 for `R12`–`R15`, and 115 for `R16`–`R17`. Online loses the sole
previously correct selected neural reading (`idle`, old `A11-R`, current `R16-R`);
disabled retains it. The simple reference retains all eleven previously correct
selected cases, with no new successes or losses. These are exposed exact
regressions with measured age, not new instruction forms.

## Evidence custody and limits

The stdlib-only partial extractor passes once under its separate 60-second,
2,048-MiB watchdog: 3.172612 seconds, peak 96.46875 MiB, exit 0. It authenticates
the frozen source and manifest, the exact global durable event prefix, actual
observations, byte applications, protected files, execution snapshots, grades,
teacher ordering and append counts. All three journals end on complete lines.
There is no extra unjournaled execution directory. This does not exclude
interrupted neuronal computation before the next journal record, and supplies no
missing final tensor state or full completion-auditor pass.

| Saved input/result | SHA-256 |
|---|---|
| Release | `46839d7fcea50dc6e805db910a36616b0bd51589` (Git commit) |
| Frozen manifest | `9ac45b7754f9ac033cc85aba49d66bee1b471e9a6da7ee26ecd883c25d65051f` |
| Failed run monitor | `7d923b3edaaddcc39486b2255801c2aac17a57286a858448fc64943fd7316b56` |
| Post-stop custody JSON | `eca11dc219eacfd3bbabca40f320da227ec2b370af8e3dfd827ec7bf4cbcc972` |
| Partial extractor source | `dc9cb865a5df59aa01de1038da3f3e517ba3a8f606fb67c4049bc629a7802da5` |
| Partial extraction JSON | `1b693f997081e2dfbf6d013d0b940b544444afd8bf3763534b8932079d922c86` |
| Extraction monitor | `337d207572cbdf0d7f3b88dafa50f2fd0679e2f7ee9e21d3768317e7cad6bffe` |
| Prior V5 attempts, used for actual-observation/age comparison | `9139824c7897f15e3a3aed71a7692f7a15c28ad18fa4d82d48072dee5981e638` |

The partial extraction and custody files are retained under `.pytest-tmp/` with
the `persistent-language-action-v6-` prefix; the extraction filename ends
`partial-evaluation-extraction-v1.json`. The completed-run extractor remains held
and unexecuted. No model, checkpoint, feature, reference prediction or program
was replayed for this review.

The next broader pressure should retain these failed public regressions while
keeping interleaved demonstrations, dated changes, delayed reuse and new
combinations distinguishable. Runtime and durable state at safe boundaries must
permit a complete recovery comparison. More exact replay alone would not test
relevance or composition; conversely removing interpretable learning opportunities
would leave another uninformative all-zero cohort. No elementary perfection gate,
new workload execution or additional learning mechanism is selected by this review.
