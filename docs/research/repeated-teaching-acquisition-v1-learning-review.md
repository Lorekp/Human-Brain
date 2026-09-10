# Repeated teaching: learning review

Completed saved-only review, 2026-09-10. The bounded repetition run completed
its execution and saved-audit contract, but neither neural arm acquired either
complete skill. Learning-enabled literal predictions improved slightly at
identical emission activities, and its edit span became correct. Every scheduled
taught free response still emitted the byte `00` followed by EOS.

This review reads only `learning-review.json` and `report.json` from
`.pytest-tmp/repeated-teaching-v1-run`; no original journals, model checkpoints,
teacher replay or new responses were used. Execution source was frozen at
`f577413`. The [protocol](repeated-teaching-acquisition-v1.md) owns the fixed
schedule; the earlier [source-only framework](repeated-teaching-local-path-analysis-v1.md)
contains bounds, not measurements.

## Complete behavior and motor learning

| Route | Main reading | Main editing | Checkpoint taught reading | Checkpoint taught editing | Checkpoint R02-R | Checkpoint R12-R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learning enabled | 0/32 | 0/32 | 0/5 | 0/5 | 0/5 | 0/5 |
| Learning disabled | 0/32 | 0/32 | 0/5 | 0/5 | 0/5 | 0/5 |
| Retrieval/edit reference | 0/32 | 32/32 | 0/5 | 5/5 | 5/5 | 0/5 |

The required final-two-checkpoint complete acquisition criterion fails for every
route. These are repeated exposed developmental cases, not fresh tests. There
is no expansion of retention or changed-context evaluation in this result.

For both neural routes, all ten taught checkpoint responses have one raw byte,
two decisions, EOS stopping and zero matching intended-prefix bytes. This is
wrong content, not a length-limit truncation or merely missing final EOS.
Their historical R02-R response is empty and R12-R is hex `63d1` at every
checkpoint. The reference answers `dim` for both taught reading and R12-R,
and `2` for R02-R.

| Checkpoint | Cumulative R/E teachers | Enabled free edit span | Disabled free edit span |
| --- | --- | --- | --- |
| 0 | 0/0 | app.py 10:13 | app.py 10:13 |
| 8, after reading block | 8/0 | app.py 10:13 | app.py 10:13 |
| 16, after editing block | 8/8 | app.py 20:24 | app.py 10:13 |
| 32, after eight pairs | 16/16 | app.py 20:24 | app.py 10:13 |
| 64, after 24 pairs | 32/32 | app.py 20:24 | app.py 10:13 |

All these proposals select SPLICE. The enabled span matches the teacher from
checkpoint 16 onward; the replacement remains `00`, so the executed edit fails.
Each of the three enabled motor heads reports a nonzero published parameter
change on all 32 edit corrections; disabled reports zero for each head. The
coordinate change is a useful partial action-selection improvement. Update
booleans alone do not establish improved motor target probability, and this
review artifact does not retain motor teacher losses/choices. Complete software
editing requires the replacement bytes as well as the operation and boundaries.

## Effective answer teaching

Every one of the 35 target positions, including the two EOS positions, received
32 live external offers. All positions selected literal at least four times;
none remained wholly without direct literal teaching. Enabled and disabled
actual inclusion counts are identical, while only enabled publishes them.

| Skill | External symbols | Literal inclusions | Raw inclusions / raw offers | EOS inclusions / EOS offers | Published enabled / disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| A00-R | 384 | 95 | 88/352 | 7/32 | 95 / 0 |
| A00-E | 736 | 178 | 172/704 | 6/32 | 178 / 0 |
| Both | 1,120 | 273 | 260/1,056 | 13/64 | 273 / 0 |

The literal branch therefore receives direct teaching on 24.375% of live
symbol offers. Per-position inclusion counts, in target-byte order followed by
EOS, are:

- Reading: `12, 10, 12, 4, 4, 5, 8, 4, 8, 11, 10, 7`.
- Editing: `5, 6, 12, 9, 8, 8, 4, 5, 5, 8, 12, 10, 11, 8, 6, 11, 8, 9, 7, 5, 8, 7, 6`.

Each neural arm also has 16,928 main public-input scan ticks and 243 actual
free-output ticks, separately from its 1,120 live teacher symbols. Thus enabled
weight movement cannot be attributed solely to the answer corrections. The
protocol has no source ingestion. Scheduled diagnostic teachers add 175 answer
symbols and 15 motor labels per neural arm, with zero slow publication; they
are counted exposures, not live teaching. All live motor offers add 96 labels
per arm. The two neural arms together therefore receive 2,590 live-plus-diagnostic
answer symbols and 222 motor labels.

## Intended-prefix probabilities at every checkpoint

Every table mean below includes the complete intended output, including EOS:
12 reading positions or 23 editing positions. `p_literal` is evaluated at every
position, even when the actual teacher did not select that branch. CE is mean
negative literal target log probability. The full emission mixture weights the
prior, two register emissions and literal distribution at the actual sampled
context/gate state. The summed mixture log score conditions on that sampled
target-prefix path; it is not a globally marginalized complete-answer probability.

| Skill | Enabled checkpoint | Mean p_literal | Literal CE | Mean emission mixture | Sum mixture log score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reading | 0 | 0.003897608 | 5.547393 | 0.063482666 | -71.409602 |
| Reading | 8 | 0.003900844 | 5.546563 | 0.063483486 | -71.400395 |
| Reading | 16 | 0.003901311 | 5.546444 | 0.063484526 | -71.399151 |
| Reading | 32 | 0.003904643 | 5.545590 | 0.063484746 | -71.389840 |
| Reading | 64 | 0.003910385 | 5.544122 | 0.063483628 | -71.373802 |
| Editing | 0 | 0.003896533 | 5.547669 | 0.251042950 | -51.807093 |
| Editing | 8 | 0.003897228 | 5.547491 | 0.251041716 | -51.806581 |
| Editing | 16 | 0.003898840 | 5.547077 | 0.251040467 | -51.805481 |
| Editing | 32 | 0.003900589 | 5.546629 | 0.251039169 | -51.804581 |
| Editing | 64 | 0.003904750 | 5.545564 | 0.251027651 | -51.802872 |

Disabled equals the corresponding checkpoint-0 row **exactly at all five
checkpoints** for these quantities. The reference has no corresponding neural
probability diagnostic. Enabled mean literal target probability increases by
0.3278% relative for reading and 0.2109% for editing. Both remain close to the
uniform 257-symbol probability `1/257 = 0.003891051`.

The conditional product score changes from `9.709656e-32` to `1.006356e-31`
for reading and `3.165668e-23` to `3.179058e-23` for editing. These small scores
describe the recorded latent paths, not estimates of overall free success rates.
At each checkpoint the actual diagnostic teacher samples zero correct reading
symbols out of 12 and seven correct editing symbols out of 23. Actual literal
inclusions are 2/12 and 8/23. Those sampled outcomes remain unchanged despite
the probability changes. Both selected-mode path scores contain zero-probability
target positions; they must not replace the full mixture calculation.

## Identical activity and conditional decomposition

All 12 reading and all 23 editing emission-activity hashes match their own
checkpoint-0 identity at **every checkpoint and across both neural routes**.
All sampled mode indices also match across the enabled checkpoint series.
The literal target probability increases at all 35 positions from enabled
checkpoint 0 to 64, while disabled remains fixed. This is stronger evidence
than equal raw observations: the literal population is evaluated at exactly
the same sparse emission activity. Its measured change reflects changed
literal weights, rather than a different emission input. It still combines
the effects of answer teaching, public scans and actual consequences.

| Skill | Mean prior contribution, CP0 to CP64 | Mean register-1 contribution | Mean register-2 contribution | Mean literal contribution |
| --- | --- | --- | --- | --- |
| Reading | 0.041677220 to 0.041673652 | 0.020831281 to 0.020832588 | 0 to 0 | 0.000974166 to 0.000977388 |
| Editing | 0.206593891 to 0.206575247 | 0.021736799 to 0.021737616 | 0.021738314 to 0.021738765 | 0.000973946 to 0.000976023 |

The reading mixture mean gains only `9.6193e-7`. The editing mixture mean
**falls** by `1.52991e-5`: its prior contribution decreases by `1.86441e-5`,
more than the combined register and literal gains. Its log score nonetheless
improves because logarithms give strong weight to rare intended symbols. A
better mean literal probability therefore does not imply that every full
emission probability improved: reading improves at 10/12 positions and editing
at only 4/23 from checkpoint 0 to 64.

Ten reading positions have no matching prior/register emission: all raw
positions except the slash. Editing has four such positions: 0 (`r`), 7 (`m`),
9 (`x`) and 19 (`6`). At those sampled contexts only the literal contribution
can produce the intended symbol, leaving probabilities around `0.000975`.
For example, the first intended reading byte rises from `0.000973571` to
`0.000977435`; the first editing byte from `0.000974544` to `0.000976277`.
The relatively high means in the table hide these weak positions. Intended-EOS
mixture probability at checkpoint 64 is 0.501012 for reading and 0.750975 for
editing; this is evaluated after the complete intended prefix and does not
measure the premature EOS seen in free generation.

## Interference and the limiting computation

Both mean literal log probability and summed mixture log score improve after
every enabled block, including for the skill not currently taught. All disabled
block deltas are zero. After eight reading corrections, editing gains 0.0001783
in mean literal log probability but seven editing positions decline slightly
(3, 9, 10, 16, 19, 20, 21). After the eight editing corrections, reading gains
0.0001198 on average, while its slash position declines by `1.64256e-5` in
literal log probability. This is measured local interference amid net gains,
not complete-skill forgetting. There was no acquired complete neural behavior
to lose; the fixed historical sentinels remain failures. Their zero scores
cannot establish preserved successful historical skill.

The limiting measured path is now more specific than absent teacher coverage.
Every intended position obtained multiple direct literal updates, and the
same-feature literal probabilities improved, but the output distribution remains
near uniform. Where the actual prior/register states do not supply the intended
byte, mode weighting reduces that already small literal probability to roughly
one in a thousand. The free response still diverges immediately. Correct edit
boundaries show that motor learning occurred without solving byte generation.

The fixed rule normalizes each selected reading target update by `0.01/24`
and each editing target by `0.01/69`. Given the observed 4–12 inclusions per
position, the upper bound on that position's own positive contribution to a
fixed target current is 0.001667–0.005 for reading and 0.000580–0.001739 for
editing. These bounds follow from unit-energy nonnegative activity and exclude
all other positions, scans and public consequences. Shared bytes/features can
add or compete. They make the small measured changes consistent with the
existing update scale; they do not isolate normalization as the sole cause.

The evidence rules out wholly missing direct coverage and changing diagnostic
emission activity as explanations for these particular fixed-feature comparisons.
It does not isolate the relative contribution of sparse branch selection,
normalization, shared-feature competition, or upstream causal credit. Teacher
and free trajectories use different conditioning and RNG timing, so these
target-prefix scores cannot identify the exact probability of the failed free
trajectory. No new mechanism or automatic teaching-budget increase is selected
by this result.

## Evidence and cost

The report contains 252 completed attempts, 222 completed teacher records and
111 actual executions across the three routes. Its internal elapsed time is
1,518.052709 seconds; the launch's separately captured outer duration is
1,519.0128574 seconds with 1,529.1484375 MiB peak RSS. The saved learning reviewer
took 7.1920214 seconds, with zero model and teacher calls. The report's disk
snapshot is 510,632,511 artifact bytes and 9,648,037,888 free bytes, sampled before
final persistence. It is not the final inventory; see the
[cost review](repeated-teaching-acquisition-v1-cost.md) for the final disk scope.

The two reviewed files total 551,096 bytes:

- `learning-review.json`: 526,062 bytes; SHA256
  `402aae6d061464aecccbdc73d12abd0683ed0ee7d4408d650550070a69efea61`.
- `report.json`: 25,034 bytes; SHA256
  `3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262`.

This interpretation recomputes scalar aggregates and compares saved hashes; it
does not independently recompute model probabilities from tensors. Execution
custody and observer implementation QA remain separate from the scientific
claim: partial local learning was observed, complete acquisition was not.
