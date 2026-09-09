# Stable credit v1: learning review

2026-09-09. The causal rule produces a small, measurable improvement in the
saved literal prediction diagnostic. It does not demonstrate useful acquisition
of either taught skill, changed-context success or successful repair. One
historical reading case succeeds at the final causal-enabled checkpoint; its
pre-attempt RNG differs from controls and no later retention measurement exists.

This review reads only the [supplemental diagnostics](../../benchmarks/stable_language_action_credit_v1.diagnostics.json)
and [final saved audit](../../benchmarks/stable_language_action_credit_v1.result.json).
These banks preserve the exact bytes of the run-local artifacts used for this review.
The [fixed protocol](stable-language-action-credit-comparison-v1.md) and
[interpretation template](stable-credit-v1-interpretation-template.md) own scope.
All material remains exposed development data. No source journal, model or
checkpoint tensor was read for this narrative; no lesson or action was replayed.

## Evidence and task outcomes

The frozen execution source is `5622f600921e6811d66254e25c8e956a31cd19ab`.
Both workers and all scheduled batteries supply the complete 1,056 attempts and
672 actual verifications. The four neural arms receive 140 answer symbols and
12 motor labels in total. The original launch remains failed because its saved
reader failed; the final corrected saved audit passes without re-executing the
learner. Its 255.2342021 seconds and preceding failed audits remain separate
post-run costs, not a retroactive successful original launch.

O+ and O- denote old-enabled and old-disabled; C+ and C- denote causal-enabled
and causal-disabled. All four start from the retained V8 ONLINE state.

* Each neural arm scores 0/64 main attempts: all acquisition encounters fail,
  all 20 changed-context cases fail, all 14 historical anchors fail and no main
  recovery succeeds. Each of its 44 edits is an accepted but incorrect SPLICE.
* At CP0/16/32/48, every neural arm scores 0/2 on the two taught skills, 0/2 on
  wording, 0/2 on values/context and 0/2 on structural combinations. Neither
  A00-R nor A00-E becomes correct, including their delayed main encounters.
  There is therefore no demonstrated newly taught skill to retain or forget.
* Historical checkpoint scores are 0/14 throughout except C+ at CP48: R02-R
  produces one correct byte, yielding 1/14. Its corresponding O+ and C- cases
  are incorrect. R02-R was incorrect at earlier checkpoints; the final hit has
  no subsequent retention test. Its raw observation matches the controls, but
  its pre-attempt RNG does not, so it is not a controlled rule-attribution result.
* Both simple copies score 12/64 main attempts. They already solve A00-E before
  its correction and solve A01-E/A02-E; they fail the reading skill. At every
  checkpoint they score 1/2 acquisition, 1/2 wording, 9/14 historical anchors,
  0/2 values/context, 0/2 structure and no recovery. This is a disclosed
  retrieval/editing reference with retained text and demonstrations, not evidence
  that the neural rule acquired those capabilities.

Migration passes on all 23 required paired neural encounters. The remaining five
recovery encounters per neural old/new comparison retain different real contexts
or RNG and remain unpaired. Full-battery aggregate CP0 diagnostics can consequently
differ even though the required independent forward behavior is preserved.
Reference copies agree on all 28 checkpoint attempts at every checkpoint; main
reference parity is established on 63 paired attempts, with one unpaired case.

## What the diagnostic numbers mean

For every actually offered symbol, q is the saved literal target probability
when LITERAL was selected, and the actual 0/1 match otherwise. The same formula
is applied to old and causal receipts. It integrates out only the final literal
sample; it is not a probability marginalized over every context, gate and mode.
M/N is sampled matches/offers; L/N is the actual literal inclusion denominator.
CE is mean pretarget literal cross entropy in nats on those L offers only.
Nonliteral distributions are never invented. Status includes its framing EOS;
raw content, other EOS and status are disjoint. No source/teacher status offers
exist. The extraction also publishes all-offer category means, without counting
these strata a second time.

Lower CE is a prediction diagnostic, unlike nonzero proposed update norms or
baseline activity. It still measures selected literal populations and inputs,
not useful answer generation. Disabled arms compute complete proposed updates;
those computations do not publish slow learning.

## Main source and public-input prediction

The following are complete stream aggregates, not an early-to-late learning
curve: source text, actual actions, diagnostics and sampled contexts vary.

| Category | Arm | Raw M/N | Raw q | Literal L/N | Literal CE |
| --- | --- | ---: | ---: | ---: | ---: |
| Source | O+ | 136/9860 | 0.013773 | 2520/9860 | 5.549076 |
| Source | O- | 120/9860 | 0.012063 | 2556/9860 | 5.549073 |
| Source | C+ | 133/9860 | 0.013324 | 2406/9860 | 5.547569 |
| Source | C- | 133/9860 | 0.013269 | 2527/9860 | 5.549073 |
| Public | O+ | 329/23665 | 0.013747 | 5994/23665 | 5.549077 |
| Public | O- | 372/24796 | 0.015337 | 6246/24796 | 5.549074 |
| Public | C+ | 360/23988 | 0.015053 | 6185/23988 | 5.547587 |
| Public | C- | 360/24738 | 0.014362 | 6239/24738 | 5.549074 |

C+ source raw accuracy is 133/9,860, equal to its disabled control and below O+'s
136/9,860. Its source q is also below O+. C+ public raw q is higher than O+ and
C-, but public inputs differ: their raw denominators are 23,988, 23,665 and
24,738. These are descriptive differences, not interchangeable trials.
Literal CE is slightly lower for C+ in both streams; this does not establish a
consistent improvement in actual raw-byte match rates.

| Category / symbol | Arm | M/N | q | Literal L/N | Literal CE |
| --- | --- | ---: | ---: | ---: | ---: |
| Source EOS | O+ | 9/24 | 0.376135 | 7/24 | 5.549066 |
| Source EOS | O- | 7/24 | 0.292153 | 3/24 | 5.549058 |
| Source EOS | C+ | 5/24 | 0.209144 | 5/24 | 5.548807 |
| Source EOS | C- | 11/24 | 0.458982 | 4/24 | 5.549054 |
| Public EOS | O+ | 71/294 | 0.242476 | 74/294 | 5.549059 |
| Public EOS | O- | 77/294 | 0.262924 | 77/294 | 5.549052 |
| Public EOS | C+ | 78/294 | 0.266379 | 81/294 | 5.548741 |
| Public EOS | C- | 77/294 | 0.262871 | 73/294 | 5.549053 |
| Public status | O+ | 101/832 | 0.121156 | 206/832 | 5.549099 |
| Public status | O- | 110/832 | 0.132020 | 216/832 | 5.549098 |
| Public status | C+ | 127/832 | 0.152407 | 206/832 | 5.547570 |
| Public status | C- | 111/832 | 0.133301 | 233/832 | 5.549097 |

The higher EOS/status rates must not be averaged into a claim of raw-content
understanding. C+ has lower source EOS accuracy than every other arm. Its public
status q exceeds both relevant controls, while later checkpoint status results
vary. The population-level C+ direct-literal publications are 2,411/9,884 source,
6,472/25,114 public and 7/35 teacher offers. All nonliteral offers remain in
those denominators. Disabled/probe direct proposals publish zero slow state.

## Scheduled checkpoint prediction

These tables include the full 28-attempt battery, including actual recovery
feedback. Thus raw denominators differ. Each population is probed with learning
disabled; changes across checkpoints concern retained state and actual sampling,
not learning inside the battery. Later checkpoint RNG frontiers are not reset.

### Raw content

| CP | Arm | M/N | q | Literal L/N | Literal CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | O+ | 85/7484 | 0.011461 | 1997/7484 | 5.549073 |
| 0 | O- | 110/7894 | 0.014080 | 2094/7894 | 5.549073 |
| 0 | C+ | 95/7768 | 0.012363 | 2065/7768 | 5.549074 |
| 0 | C- | 100/7780 | 0.013117 | 2068/7780 | 5.549074 |
| 16 | O+ | 114/7726 | 0.015226 | 1963/7726 | 5.549077 |
| 16 | O- | 110/7371 | 0.014426 | 1885/7371 | 5.549074 |
| 16 | C+ | 96/7844 | 0.012213 | 2002/7844 | 5.548059 |
| 16 | C- | 110/7528 | 0.014678 | 1926/7528 | 5.549074 |
| 32 | O+ | 118/7357 | 0.016470 | 1843/7357 | 5.549077 |
| 32 | O- | 126/7764 | 0.015920 | 1954/7764 | 5.549073 |
| 32 | C+ | 116/7962 | 0.014288 | 1991/7962 | 5.547080 |
| 32 | C- | 91/7505 | 0.011501 | 1880/7505 | 5.549074 |
| 48 | O+ | 94/7743 | 0.012475 | 1952/7743 | 5.549080 |
| 48 | O- | 80/7195 | 0.010990 | 1818/7195 | 5.549074 |
| 48 | C+ | 108/7334 | 0.013670 | 1859/7334 | 5.546137 |
| 48 | C- | 91/7364 | 0.012647 | 1833/7364 | 5.549073 |

### EOS outside status

| CP | Arm | M/N | q | Literal L/N | Literal CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | O+ | 26/108 | 0.241894 | 32/108 | 5.549054 |
| 0 | O- | 24/108 | 0.223555 | 37/108 | 5.549053 |
| 0 | C+ | 31/108 | 0.288262 | 34/108 | 5.549055 |
| 0 | C- | 27/108 | 0.251225 | 34/108 | 5.549054 |
| 16 | O+ | 37/108 | 0.343313 | 20/108 | 5.549062 |
| 16 | O- | 38/108 | 0.352500 | 18/108 | 5.549051 |
| 16 | C+ | 30/108 | 0.278462 | 19/108 | 5.548976 |
| 16 | C- | 30/108 | 0.278823 | 29/108 | 5.549054 |
| 32 | O+ | 25/108 | 0.232238 | 21/108 | 5.549059 |
| 32 | O- | 37/108 | 0.343421 | 23/108 | 5.549052 |
| 32 | C+ | 29/108 | 0.269528 | 28/108 | 5.548592 |
| 32 | C- | 27/108 | 0.250937 | 26/108 | 5.549050 |
| 48 | O+ | 31/108 | 0.287902 | 24/108 | 5.549057 |
| 48 | O- | 34/108 | 0.315499 | 19/108 | 5.549052 |
| 48 | C+ | 31/108 | 0.287938 | 25/108 | 5.548329 |
| 48 | C- | 38/108 | 0.352933 | 30/108 | 5.549052 |

### Status

| CP | Arm | M/N | q | Literal L/N | Literal CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | O+ | 62/364 | 0.168598 | 95/364 | 5.549104 |
| 0 | O- | 61/364 | 0.168577 | 93/364 | 5.549103 |
| 0 | C+ | 56/364 | 0.154840 | 93/364 | 5.549103 |
| 0 | C- | 59/364 | 0.157588 | 93/364 | 5.549103 |
| 16 | O+ | 50/364 | 0.138228 | 81/364 | 5.549103 |
| 16 | O- | 57/364 | 0.157641 | 98/364 | 5.549100 |
| 16 | C+ | 58/364 | 0.160208 | 81/364 | 5.547528 |
| 16 | C- | 68/364 | 0.187711 | 84/364 | 5.549103 |
| 32 | O+ | 60/364 | 0.165797 | 90/364 | 5.549105 |
| 32 | O- | 54/364 | 0.149228 | 82/364 | 5.549101 |
| 32 | C+ | 48/364 | 0.132789 | 86/364 | 5.546933 |
| 32 | C- | 57/364 | 0.157502 | 85/364 | 5.549105 |
| 48 | O+ | 41/364 | 0.113685 | 98/364 | 5.549102 |
| 48 | O- | 63/364 | 0.173921 | 79/364 | 5.549103 |
| 48 | C+ | 45/364 | 0.124666 | 97/364 | 5.546045 |
| 48 | C- | 60/364 | 0.165808 | 91/364 | 5.549096 |

### Fixed independent cases clarify the small local gain

Removing only the protocol's two recovery cases from the extraction leaves all
22 predeclared independent cases, with exactly 3,935 raw offers at every checkpoint
and arm. This is a disclosed subset analysis, not an additional probe or a selected
best checkpoint. All352 saved case records have identical observation hashes
for each named case across arms/checkpoints. Sampled context and literal inclusion
still vary.

| Arm | CP0 literal CE (L) | CP16 literal CE (L) | CP32 literal CE (L) | CP48 literal CE (L) |
| --- | ---: | ---: | ---: | ---: |
| O+ | 5.549073 (1046) | 5.549076 (981) | 5.549077 (1022) | 5.549080 (1002) |
| O- | 5.549073 (1046) | 5.549074 (1007) | 5.549072 (996) | 5.549074 (1025) |
| C+ | 5.549073 (1046) | 5.548040 (995) | 5.547070 (1001) | 5.546056 (1002) |
| C- | 5.549073 (1046) | 5.549074 (1002) | 5.549074 (990) | 5.549073 (1014) |

C+ CE falls by 0.003018 nats across these checkpoints. Its mean literal target
probability rises from 0.003891061 to 0.003902834; a uniform 257-symbol distribution
has probability 0.003891051 and CE 5.549076. The improvement is measurable in the
saved evaluated distributions but small. The old and disabled CE values remain
near that starting level. Since literal inclusion and sampled contexts differ,
this is not a statistical estimate over matched latent trajectories.

Even on this fixed raw-input subset, C+ sampled matches are 47→46→59→54 out of
3,935 and q is 0.012216→0.011404→0.014461→0.012684. Its raw-match diagnostic does
not improve monotonically. The slight final advantage over the controls does
not establish reliable answer skill; neither taught skill becomes correct.

## Each actual correction

A00-R offers 11 raw bytes plus EOS; A00-E offers 22 raw bytes plus EOS and three
motor labels. The table retains each arm/correction separately. No correction is
replayed. Teacher weights are held through the correction, so these are offered
symbol predictions before that correction's publication, not post-correction
answer tests.

| Arm / correction | Raw M/N | Raw q | Raw literal L/N | Raw literal CE | EOS M/N | EOS q | EOS literal L/N | EOS literal CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O+ / A00-R | 0/11 | 0.001061 | 3/11 | 5.549063 | 0/1 | 0.000000 | 0/1 | — |
| O+ / A00-E | 2/22 | 0.091440 | 3/22 | 5.549063 | 0/1 | 0.003891 | 1/1 | 5.549069 |
| O- / A00-R | 0/11 | 0.001061 | 3/11 | 5.549063 | 0/1 | 0.000000 | 0/1 | — |
| O- / A00-E | 2/22 | 0.091440 | 3/22 | 5.549063 | 0/1 | 0.003891 | 1/1 | 5.549067 |
| C+ / A00-R | 0/11 | 0.001061 | 3/11 | 5.549023 | 0/1 | 0.000000 | 0/1 | — |
| C+ / A00-E | 2/22 | 0.091440 | 3/22 | 5.549031 | 0/1 | 0.003891 | 1/1 | 5.549076 |
| C- / A00-R | 0/11 | 0.001061 | 3/11 | 5.549063 | 0/1 | 0.000000 | 0/1 | — |
| C- / A00-E | 2/22 | 0.091440 | 3/22 | 5.549063 | 0/1 | 0.003891 | 1/1 | 5.549067 |

Every arm matches 0/11 raw reading symbols and 2/22 raw edit symbols; every
teacher EOS is missed. Each arm evaluates literal prediction for only six raw
teacher symbols and one EOS, seven of 35 total external offers. The small C+
raw teacher CE differences are already present at each held prepublication
prediction. They cannot show that the just-issued correction improved its own
prediction, and the inherited `prior_nll` is a different diagnostic.

## Actions, recovery and unresolved explanations

All 176 free predictions per neural arm stop at EOS. Their total generated bytes
are O+641, O-601, C+588 and C-590; maximum lengths are 21,20,23 and20. This describes
short outputs, not an identified cause of failure. There is no successful neural
software edit in the main stream or checkpoint batteries.

All eight main recovery cases initially fail for every arm; all later and final
neural outcomes also fail. Every neural arm consumes actual feedback on 16 later
main attempts. C+ changes proposals across all 16 consecutive recovery pairs;
O+ changes them across 15 and repeats one. Neither produces a successful repair.
Both reference copies repeat all 16 pairs and also fail recovery. At each
checkpoint every arm again fails both recovery cases and performs zero successful
later repairs. Proposal variation, accepted patches and exit status alone are
not productive feedback use.

The diagnostics leave specific questions unresolved:

* Whether the small literal probability shift can support correct answer
  generation rather than observed-input regularities. Public scans measure
  arriving requests/workspaces/diagnostics; they do not test the teacher answer
  distribution after learning. The actual acquisition outputs remain incorrect.
* How much observed match variation comes from context/gate/mode sampling rather
  than literal prediction quality. q removes only the final literal sample's
  noise, and later RNG/context pairings differ. The traces do not isolate a
  context representation or prove it is absent.
* Whether limited direct teacher inclusion contributes to failed acquisition.
  Seven evaluated literal teacher offers are a measured fact, not proof that more
  teaching, a different rate or another mechanism would fix the tasks.
* Why the particular edits fail. The public consequence jointly reflects payload,
  operation and span choices; this comparison supplies no causal blame oracle.
  Short outputs and changed proposals do not isolate that failure among them.

The supported result is a small local literal-prediction gain with no demonstrated
acquisition, changed-context or repair advantage, plus one unpaired final reading
hit whose retention is unmeasured. Keep the fixed teaching budget and difficulty.
These observations do not automatically select another mechanism or authorize
additional exposure.

## Extraction and evidence costs

The supplemental extractor completes in 34.9028742 monitored seconds at
78.765625 MiB peak, outside the frozen comparison ceiling. Its internal extraction
clock is 33.5367323 seconds: journal indexing4.6649068 and parsing/aggregation
28.8148119 seconds. It performs two real passes over 2,125,909,485 journal bytes,
reading 4,251,818,970 journal bytes plus4,667,045 metadata and443,026 source bytes:
**4,256,929,041 confirmed bytes**, with exact accounting. It parses1,212 rows and
exports808 neural case records, including eight corrections. Output is2,780,209
bytes. These are analysis costs; it makes zero model calls, tensor decodes,
teacher replays or execution children. This narrative reads only that extraction
and the final audit, not those journals again.

Diagnostic SHA256:
`23f574a1a30e21c6bd18ef12219b52721fcfcaa3f23226a53f5d780987cca0a2`.
Final-audit SHA256:
`1bd3073fac22fdd54cc27e16ad7c8e75d106b15e9263e44bb1161b1a85c54919`.
The audit retains the hashes of all three preceding failed saved audits; the
initial failure is not replaced. Reading these artifacts creates no additional
scientific exposures or independent replication.
