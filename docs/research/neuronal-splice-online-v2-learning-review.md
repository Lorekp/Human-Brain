# Competitive raw editing: completed learning review

Status: saved-evidence review, 2026-09-06. The fixed v2 run is closed. Its fresh
cohort is now exposed development evidence. This review authorizes no additional
prediction, decoding policy, teacher, epoch, checkpoint probe or acquisition.

The [selected comparison](neuronal-address-competition-comparison-v1.md) establishes
a useful but incomplete step: the competitive learner makes nineteen correct
fresh raw edits, versus six for its contemporaneous marginal reference and zero
for each matched control. The same persistent learner selects and executes its
own proposals before receiving local correction. The strict selected criterion
still fails: fresh completion and action retention are incomplete.

## What the completed comparison shows

| Arm | Fresh edits / 28 | Fresh STOP / 4 | Retention / 16 | Old reading, before → after | Acquisition / 256 |
|---|---:|---:|---:|---:|---:|
| Learned competitive | 19 | 1 | 14 | 20 → 20 | 139 |
| Action frozen | 0 | 1 | 4 | 20 → 20 | 38 |
| Slow frozen | 0 | 1 | 2 | 20 → 20 | 24 |
| Memory disabled | 0 | 1 | 8 | 0 → 0 | 92 |
| Learned marginal reference | 6 | 1 | 12 | 18 → 18 | 101 |

Competitive fresh totals by factor are document order D=6/8, continuation C=8/8,
value exchange/delay/interference V=4/8, and operation/layout O=2/8. Jointly
successful pairs are 3/4, 4/4, 2/4 and 0/4 respectively. All two-document V cases
pass; all four five-document V cases fail. The competitive acquisition rounds
give `2,6,12,15,22,26,28,28` actual first-action successes. These are repeated
TRAIN experiences, not eight independent transfer measurements. The old panel's
20/20 already holds before acquisition, so its two extra successes over the
marginal reader are not newly acquired reading knowledge.

The audit accounts for 1,280 learning actions, 2,400 reader symbol offers, 3,200
action labels, 240 endpoint actions and 200 separate old-panel queries: 1,520
actual process verifications in total. Learned and marginal arms each report
192 changed start groups, 192 end groups, 256 operation groups, and 192 each
codec/address groups. Action-frozen changes no action group; slow-frozen changes
none; memory-disabled changes no address group. Its 92 acquisition successes
and eight retention successes therefore cannot be treated as evidence of
episodic reading. They coexist with zero fresh edited-task successes.

The saved audit verifies the proposal/action/correction identity and ordering,
unchanged episode hashes, source-free worker acknowledgements and frozen source
custody. Those records do not independently reconstruct unsaved weight updates.
Source and the previously completed ordinary QA establish the direct local
arithmetic; this review neither replays nor extends that QA.

## Remaining failures are not unresolved competition ties

The saved endpoint stream contains actual competition scores and committed
cursors. An external descriptive join labels the intended literal boundary
using the authored raw request/document identity. This analysis label is never
given to the model or teacher. All nine competitive fresh payload mismatches
commit to another record's literal start, while the intended boundary has
**unique rank two**. All learned payloads terminate with EOS; there are no
structurally rejected learned endpoint proposals. No learned SPLICE case with
a rank-one intended boundary has a payload mismatch in this saved cohort.

| Failed tasks | Actual saved behavior | Intended score minus winning score |
|---|---|---:|
| D2-0, D2-1 | Correct file/span; `db` from k98 instead of `bd` from k82 | −0.107700 |
| V2-0, V2-1 | Correct file/span; `dd` from k99 instead of `ac` / `ca` from k90 | −0.265252 |
| V3-0, V3-1 | Correct file/span; `dd` from k99 instead of `bd` / `db` from k91 | −0.225837 |
| O0-0 | Wrong offsets; `bc` instead of `ad`; process error | −0.010792 |
| O1-0 | Wrong file and offsets; `bc` instead of `da`; process error | −0.001479 |
| O3-0 | Wrong file; `bc` instead of empty deletion payload | −0.356987 |

For empty O3, the rank labels the authored empty-literal boundary; it is not a
claim that this is the sole address capable of producing EOS. The two D2
document orders choose the same wrong record despite its changed absolute slot.
The V failures choose the last distractor, but its identity and position are
confounded; they do not establish a recency mechanism. Key activity resets per
document in the frozen tape source, so historical lag leakage across EOS is not
the implemented explanation.

The remaining three fresh failures are KEEP tasks O1-1, O2-1 and O3-1 that
choose SPLICE. O2-1 retrieves the requested payload correctly but still edits:
retrieval correction alone cannot settle STOP behavior. O1-1 also produces a
process error. Thus the twelve fresh failures comprise six payload-only
failures, three combined payload/location failures, and three operation
failures; there are three process errors and zero structural rejections.
Retention T01 selects the wrong record (`cc` instead of `ba`, rank two,
margin −0.042863); T08 retrieves `aa` correctly but inserts at offset 12 rather
than the canonical offset 5.

The marginal reference has 21 fresh SPLICE payload mismatches and emits an empty
payload on 23/32 fresh tasks. The intended-boundary ranks agree between learned
competitive and marginal readers on all 48 action endpoints; target-minus-top
margins differ by at most `2.22e-16` after comparing raw scores with normalized
log beliefs. This is consistent with their inherited identical reader-teacher
path. It does not prove equality of every final tensor, which was not loaded.
The comparison includes online feedback interaction: STOP operation teaching
uses each arm's actual proposal, so this is not an isolated frozen-decoder-only
contrast. The older v1 cohort is different and supplies no matched improvement
denominator here.

## One next priority: discriminate raw identities before commitment

Recommend one bounded **source/math review of generic neuronal byte-coincidence
features for robust raw identity binding**, leading back to the same integrated
acting-and-learning task only if root selects a concrete design. The immediate
question is whether the present forward representation can reliably favor the
requested stored byte sequence under similar distractor keys and nuisance
request context, without supplying a parser, target address, role mask or
canonical span to the learner.

In the frozen [reader query](../../src/humanbrain/local_memory_learner.py),
`X_s[i,j] = Σ_d Z[i,d] K_s[j,d]` is formed before the sole learned temporal
score layer, `score_s = Σ_ij B[i,j] X_s[i,j]`. The 32-channel fixed random byte
codes make these activities graded correlations, not categorical byte equality.
The remaining wrong ranks justify examining that discrimination and temporal
alignment. They do **not** identify random-code overlap, insufficient training
coverage, the learned temporal weights or latent teacher ambiguity as the
proven cause. A generic categorical/nonlinear coincidence circuit is a hypothesis
for this review, not an adopted replacement. Its physical forward activity,
local teaching ownership, information preservation, parameter and execution
cost must be explicit before any implementation or measured run.

The current all-address teacher uses byte/EOS likelihood evidence and its own
forward activities for local `X*(p−posterior)` products. It supplies no correct
address. The [competitive free reader](../../src/humanbrain/competitive_local_memory_learner.py)
changes only acting; that surrogate is not a derivative through its threshold
gates. Rank-two failures on untaught fresh identities do not by themselves show
that this local credit rule malfunctioned. Sharpening the same scores or changing
tie resolution cannot reverse these observed strict rankings.

The shared priority remains useful complete raw edits, not reader accuracy in
isolation. Keep the independent location and KEEP failures visible during the
review. The [operation features](../../src/humanbrain/neuronal_splice_learner.py)
contain query×context/path interactions, span×payload activity and three scalars,
but no direct raw-query term. Their dependence on a freely selected provisional
span is a concrete representational limitation to assess; it is not proof that
SET/KEEP is unrepresentable. Do not combine an unreviewed operation fix with a
reader change merely to repair these examples. Any selected successor requires
new preregistered withheld identities/layouts, matched controls and immediate
reference, fresh complete cost admission, and the same persistent local learner.
No epoch extension, gain/seed sweep or endpoint-specific matching is selected.

## Evidence and limits

One bounded saved-only analysis streamed 480 endpoint rows once and retained 96
small learned/reference summaries. It imported no project module, loaded no
checkpoint, and generated no output. Its raw request-key extraction is solely
an external analysis label; STOP has no canonical reader teaching target. Source
hashes for the reader, competitor, action learner and tape match the audit.

| Saved artifact under `.pytest-tmp/` | SHA256 |
|---|---|
| `neuronal-splice-online-v2-run-independent-audit.json` | `d31238776795984c6ac58fa9c6b2549323e603b795a652044bb01dacd16322db` |
| `neuronal-splice-online-v2-run/report.json` | `52edd52de06776a253de4462c796ebe74f44fcaa1dd6659185f1c475ed644de8` |
| `neuronal-splice-roster-qa-generated-v2.json` | `1f76de378e238ddebd6ebcc8dd62f9dffc3187cada1923e794f86e51dfb4440e` |
| `neuronal-splice-online-v2-run/endpoint/worker/endpoints.jsonl` | `65d1f48d9cde609728909778f03c5e15de6f6e79acd4ffcf50c8a9fda1bd625d` |
| `neuronal-splice-v2-learning-analysis.py` | `b3d6bc274c20216ca094a0452f384da21f85fdd908c701b650d141028c65859d` |
| `neuronal-splice-v2-learning-analysis.json` | `c2cb0572ab97496afd6445561a240726570464c166452e0207b9ff0aaeb103fe` |

These results establish bounded raw-copy editing with useful learned selection,
not general repository repair, reliable abstention or general language mastery.
The fixed run remains closed regardless of the next source-review outcome.
