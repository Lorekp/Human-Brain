# Third persistent session: evaluation and reference attribution

Status: completed saved-only review, 2026-09-07. The full session and independent
review passed on their first launch at source
`fc7f614cfbc7c01489245c6fec05520a9d1c586f`. Passing the review establishes valid
records, not useful neuronal competence. This review uses saved JSON and scalar
arithmetic only: no model, checkpoint load, baseline API, prediction, teacher,
program execution or alternate decoding.

## Separate outcomes

| Outcome | Online | Learning disabled | Simple reference |
| --- | ---: | ---: | ---: |
| Acquisition first responses | 0/24 | 0/24 | 12/24 |
| Fresh first responses | 0/16 | 0/16 | 6/16 |
| Fresh within the neural interface limits | 0/12 | 0/12 | 5/12 |
| Exact exposed delayed retention | 0/8 | 0/8 | 8/8 |
| Recovery after initial failure | 0/8 | 0/8 | 0/8 |

The reference acquisition score divides into 6/12 readings and 6/12 edits; its
fresh score divides into 2/8 readings and 4/8 edits. Both neuronal routes score
zero in each division. The two interleaved acquisition blocks score 4/12 then
8/12 for the reference, and zero then zero for both neuronal routes. These are
first responses, with twelve episode starts between related forms. The blocks
contain different requests and structures; their difference is not an isolated
learning-effect estimate or an immediate post-correction mastery test.

All eight recovery issues fail on every one of their three attempts in every
route. There is no transient success hidden by the final-attempt score. Four
fresh instructions deliberately exceed the neuronal 31-byte request interface.
The reference receives their whole raw text and solves one of four. This access
difference remains explicit; the supported comparison still favors the reference.

The run took 56.9543509 seconds and its independent review 3.6428642 seconds:
60.5972151 seconds together. Their observed peaks were 510.5703125 and
512.5234375 MiB respectively. These are this completed workload's measurements,
not a forecast for larger language or software tasks.

## All reference outputs have an available source

An independent scalar attribution pass checked all 72 saved reference attempts
with 2,809 assertions. It reconstructed the reference's **observed history**
from both previous sessions: 18 documents and 24 editing demonstrations. Every
demonstration is bound to its preceding actual acquisition action by the saved
attempt hash. This session adds no documents and twelve editing demonstrations,
ending with 18 documents and 36 demonstrations. It restores no reading-answer
table and calls neither the reference predictor nor its teaching API.

For every reading, raw-query trigram cosine scores agree with the recorded
document choice, including earliest-index ties. The answer is exactly that
document's first boundary-delimited quoted literal, with matching source hash
and literal span. For every edit, the same calculation confirms the selected
available demonstration and query/source hashes. Its recorded old literal must
occur exactly once in the actual current files; otherwise the reference emits
STOP. Every saved replacement or STOP agrees with that rule. Among the 48 edit
attempts, nine use session-one demonstrations, 27 use session-two demonstrations
and twelve use current-session demonstrations. These counts include failures
and recovery repetitions.

The following supported strata expose the difference between available facts,
literal reuse and the requested behavior. Both neuronal routes score zero in
every row.

| Fresh demand | Reference reading | Reference edit | Actual interpretation |
| --- | ---: | ---: | --- |
| Exclude testing colour, F00 | 0/1 | 1/1 | Reading returns `green`; the edit reuses an earlier `blue` demonstration. |
| Debug then normal logs, F01 | 0/1 | 1/1 | Reading returns only `debug`; the provided program already joins the inserted fragment with `/audit`. |
| Reverse retries/delay, F02 | 0/1 | 0/1 | Reading returns `3`; the reused `'3/2'` edit produces `2/3/2`, not `2/3`. |
| Twice the retry count, F03 | 0/1 | 1/1 | Reading returns `3`, not `6`; the supplied lambda doubles the inserted numeric `3`. |
| Backup unless cold, F04 | 1/1 | 0/1 | Reading retrieves `west`; the edit reuses `north` and violates the condition. |
| Append archive ending to logs, F05 | 0/1 | 1/1 | Reading returns `.bak`; the supplied program concatenates it to `audit`. |

These mixed results do not establish a generic negation or composition rule in
the reference. Correct program behavior can arise from reusable literal editing
combined with computation already present in the workspace. Conversely, an
available demonstrated compound answer can be inserted in the wrong context.
The scorer uses actual stdout, return code and protected-file integrity, rather
than requiring the canonical annotated patch.

## Freshness, retained history and limits

All sixteen fresh requests differ byte-for-byte from the previous two sessions
and current acquisition requests. New demands include exclusion, changed join
order, doubling and conditional selection, alongside f-string, join, lambda,
dictionary and class/loop contexts. Some structure labels are reused. The data
explicitly do not claim wholly withheld linguistic families or a protected split.

Independent token-set overlap against earlier requests and current acquisition
ranges from 0.2222 to 1.0, with median 0.4643. In particular, F02-E rearranges
familiar words into delay/retries order while retaining the same token set as a
prior request. Byte-level novelty is therefore insufficient evidence of broad
language generalization. Facts, vocabulary and many program primitives remain
exposed development material. The failed supported composition demands remain
part of the score rather than being replaced by easier copying tasks.

Both neuronal routes start from the previous **online** final model and the
same 505-slot shared tape; the disabled route suppresses this session's updates.
No documents are added, evicted or selected by task, and memory capacity does
not grow. The session instead adds retention age and further experience over
all accumulated memory. Each neuronal route receives 24 one-pass corrections,
172 offered reader symbols including EOS, and 36 action labels. Actual-path
output correction permits at most one output offer per response; it does not
turn all 172 address-teaching offers into separate output clamps.

The eight retention observations exactly repeat session-two retention items,
not similarly numbered current acquisition tasks. Each has 71 intervening
responses since its previous delayed observation, including the 24 current
acquisition corrections. The reference preserves all eight previously correct
outcomes. Those eight already failed in the previous online neuronal state and
fail again in both continuation routes; this is not evidence of eight newly
forgotten skills. Nor did the protocol establish immediate post-teacher mastery
of them in an earlier session.

Weights and shared memory persist. Files reset at episode starts and continue
only within each three-attempt recovery sequence. This is not one continuously
edited software project. Tool results are recorded but do not enter an
implemented feedback encoder; zero recovery therefore supplies no evidence
about the effectiveness of a consequence-conditioned update rule.

The unchanged zero neuronal task scores leave the shared question at useful
sequence learning and transfer under actual free generation. The attribution
review does not select another mechanism or fit schedule. Its evidence also
does not permit reference-produced answers to be inserted into the neuronal
route and credited as learned outputs.

## Bound evidence

- Frozen manifest: `.pytest-tmp/persistent-language-action-v3-run/manifest.json`,
  SHA256 `fc83513a6d6accee744c5d03188bc9cec75022db3cd00ff5c49f561745aa3056`.
- Independent completed review: `.pytest-tmp/persistent-language-action-v3-review.json`.
- All-attempt attribution and input hashes:
  `.pytest-tmp/persistent-language-action-v3-reference-attribution-review.json`,
  SHA256 `91d530b0f52c15ffd65c8995a20481a76bbaf4c06683b14323e9d66dde1242ae`.
- Attribution source: `.pytest-tmp/review-persistent-language-action-v3-reference.py`;
  its hash is recorded in that receipt. Only saved scalar calculations ran.
