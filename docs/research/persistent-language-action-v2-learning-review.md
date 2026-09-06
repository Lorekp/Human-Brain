# Persistent language/action v2: saved learning review

The completed continuation establishes functioning local updates and per-output
memory access, but no useful neuronal task success. Both online and
learning-disabled routes score **0/24 acquisition, 0/16 transfer, 0/8 recovery and
0/8 retention**. The simple reference scores 13/24, 11/16, 0/8 and 8/8. These are
public development outcomes; this cohort is now exposed. The fixed run is closed.

This review uses the completed report, independent saved review and one streamed
extraction of 216 attempt, 72 teacher and six ingest rows. It loads no checkpoint,
constructs no model and produces no alternate decoding or rescored successes.

## What the actual outputs show

Each neuronal route makes 68 supported free calls: 22 reading and 46 action
payloads, including recovery retries. Both produce 1,108 actual byte/EOS
decisions. Four first attempts per route are unsupported long requests, split
equally between reading and action; they remain failures without truncation.

| Actual first-attempt diagnostic | Online | Learning disabled |
| --- | ---: | ---: |
| Supported reading outputs | 22 | 22 |
| Reading limit / EOS | 19 / 3 | 19 / 3 |
| Supported action payloads | 30 | 30 |
| Action payload limit / EOS | 30 / 0 | 30 / 0 |
| Exact reading or canonical payload bytes | 0 / 52 | 0 / 52 |
| Calls with within-output address switching | 2 / 52 | 0 / 52 |

All 22 reading outputs and their selected-address sequences are identical between
routes. Every nonempty reading output repeats one byte 16 times; the other three
emit EOS immediately. For example, A00-R asks for the badge colour and emits
`bbbbbbbbbbbbbbbb` instead of `blue`, repeatedly selecting address 14. T08-R
emits 16 `w` bytes instead of `west`, repeatedly selecting address 221. No complete
expected literal occurs in any of the 52 supported first neuronal outputs. This
is a raw substring observation, not a semantic scorer.

Only two first-attempt action payloads differ between online and disabled.
Online F05-E alternates addresses 342 and 48 on all 16 transitions, producing
`etetetetetetetet` for target `6`; X00 switches away from address 363 to 322 and
back, producing `tttgtttttttttttt` for target `0`. Across retries there are four
switching calls and 22 switches. These are real prefix-sensitive changes, but
switching alone has not become useful sequencing or composition.

All 46 online action proposals are SPLICE without payload EOS and are rejected.
The two unsupported edit requests also verify the unchanged workspace. Disabled
produces 45 rejected SPLICEs and one accepted STOP; none succeeds. Thus this
session's online actions never apply an edit. The eight first-attempt nonzero
process exits are executions of the unchanged authored recovery workspaces,
not newly broken programs. Other location errors coexist: among online's 30
supported first proposals, canonical path matches 20 times, start once and end
once. These diagnostics do not replace behavior grading or invalidate equivalent
correct patches. Missing payload termination already blocks every online splice.

## Learning occurred, but its useful effect is not established

Both neuronal routes receive exactly 24 correction calls and 98 reader offers
(74 raw bytes and 24 EOS), plus 36 action labels. Online publishes U, query B
and prefix C in all 24 reader corrections and each action head in all 12 edit
corrections. Disabled publishes none; legacy codec tensors remain fixed in both.
Actual responses/actions are durably recorded before target-bearing calls.

The actual teacher-prefix, pre-update mixture NLL is worse online:

| Offer-weighted statistic, nats | Online | Disabled |
| --- | ---: | ---: |
| Reading target NLL | 4.194 | 3.762 |
| Action-payload target NLL | 3.911 | 3.632 |
| Reading responsibility entropy | 3.764 | 4.246 |
| Action responsibility entropy | 3.767 | 4.097 |

All 24 matched call-mean NLLs are higher online. This is not an immediate
post-teacher regression test: online stages updates between earlier teacher
symbols even within the first call, while disabled retains its initial weights.
Later calls also include accumulated updates. Concentrated responsibility is
therefore not evidence of better answering. Mean maximum teacher responsibility
is only about 0.133 for reading and 0.116 for action online, while free selected
address prior mass averages roughly 0.006. These are different distributions
under different prefixes, not interchangeable estimates of address correctness.

The source's unit normalized output-current rule has a fixed-responsibility
quadratic interpretation. It guarantees neither mixture-NLL descent nor success
on the freely generated prefix. The current target is absent from forward
features; teacher prefixes enter only after the actual attempt. Saved teacher
rows omit conditional logits and feature banks, so this review cannot reproduce
responsibility formation or weight deltas, or decide whether feature overlap,
shared-output interference or latent-address credit dominates. No single cause
is established by these pooled observations.

## Persistence and the next shared question

Both routes continue the previous online numerical state and its entire 453-slot
tape, then append two notes totalling 50 raw bytes plus two EOS. Both finish with
the same 505-slot memory; only online's model hash changes. All eight retention
items already failed at the previous session's delayed measurement and fail
again after another 48–55 intervening episode positions. This is persistent
failure, not eight forgotten skills. Simple retains its eight previously correct
items. There is no immediate post-acquisition retention probe.

Recovery has three fixed attempts without teachers or execution-feedback
encoding. Rejected neuronal edits leave the same workspace and do not recover;
the result does not test learning from error messages. The old sequential-copy
policy was deliberately replaced by per-byte reselection, so preserved numerical
weights do not imply preserved prior output trajectories.

The next shared priority should be **effective local sequence learning under the
learner's own generated prefix**, for both language and action payloads. Start
with one bounded source/math examination of how the current all-address
responsibility-weighted current update changes the selected free output and its
termination, including cross-context interference. Preserve persistent state,
generic inputs and the prohibition on target-address labels or reverse-gradient
transport. The evidence does not justify an operation-only patch, more epochs,
an amplitude sweep, or another automatic toy gate. Any runtime change requires
explicit selection and a broader continuation contract.

## Evidence identity

The extraction is `.pytest-tmp/persistent-language-action-v2-learning-extract.json`,
SHA256 `de6945ce0da0dbf788decb0f270347986c46d90424b247ab781cc4864e2c73ec`.
Its input hash table binds:

| Saved input | SHA256 |
| --- | --- |
| report.json | `04f2d46ac72ff9c4bfbadcdfe6caf0944463e63903644fd9035109da9b5b8526` |
| manifest.json | `efac496ad0f5e8e6de982c5a6b9418621359996ba9064465e117a1e396d5e721` |
| attempts.jsonl | `7b21e2cacc6d39ff0a1b4ab7398e872195c5bed82c190e14a851dd2c85f93ccf` |
| teachers.jsonl | `10df7e75af48eb6ffc8b82b69640bce5534a601f25be5e16895a7bf4abdf7571` |
| ingests.jsonl | `040426d1390847bcb51b9035bc39212f00de14b96de835c8109e23f1e2f4eebc` |
| completion.json | `716159dae22bb90adf17caf70250c184b6039cfde314649d1c83ed05c31e4a3e` |
| independent review | `ea54da72c5b245f3c1019e84d66eac8129d55298cd436db6c8e75318225fed12` |

Journal, report and manifest hashes match the completed artifact manifest.
Independent saved review passed. No frozen Python source was edited for this
analysis.
