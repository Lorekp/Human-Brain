# Frozen document-memory localization result

Status: completed and independently audited; diagnostic only. 2026-09-06.
Source: `d724e06183ea7232f59a506febe2f8f0b02b5b1b`.

The [frozen protocol](local_memory_localization_v1.md) completes within its
60-second aggregate budget. Assignment differences survive the write stream,
reach query reads and alter output potentials, but usually fail to change the
answer appropriately. The actual positive writer and address updates each reduce
the scored loss in all eight isolated cases; their negative counterparts increase
it. None changes a generated answer. This localizes limitations without earning
capability qualification, adopting a repair or authorizing another fit.

## Fixed checkpoint and input scope

The tagged and writer-frozen endpoints come from the authenticated failed
[two-byte return](local_memory_two_byte_v1_result.md), source
`a94b466f1b2c34e83b53d19e43f9430e52191648`. All diagnostic memories start empty.
Four two-record TRAIN assignments are A: k00=ab/k01=bc, B: k00=bc/k01=ab,
C: k02=cd/k03=da, and D: k02=da/k03=cd. Both document orders and both questions
are tested for both endpoints. These are exposed development values and a
narrow authored record grammar, not fresh adjudication.

Each of eight original-order tagged cases receives one actual three-position
teacher correction on an independent disposable copy. Its writer or address
block is tested separately after fresh ingestion, with the original output
weights and other block fixed. Positive conditions copy the actual teacher-after
arrays; negative conditions use represented float32 `before - (after - before)`.
No displacement, teacher-modified output, or diagnostic trace is retained in the
production learner. Neither a derivative nor exact sign symmetry is claimed.

## Observed behavior and localization

| Endpoint / condition | Exact executed answers | Order-sensitive answers |
| --- | ---: | ---: |
| Tagged, original order | 2/8 | 2/8 change on reversal |
| Tagged, reversed order | 1/8 | — |
| Writer-frozen, original order | 2/8 | 2/8 change on reversal |
| Writer-frozen, reversed order | 1/8 | — |
| Each of writer+, writer−, address+, address− | 2/8 | No change from its baseline |

The two endpoints generate exactly the same sixteen baseline answers. In case
order A/B/C/D, with both questions per memory, original outputs are
`ca, cb, ca, cb, cd, ca, cd, cd`; reversed outputs are
`ca, cb, ca, cb, cd, cd, cd, ca`. Six of eight same-query assignment contrasts
per endpoint produce identical answers. The remaining contrasts change an answer
without solving both opposing assignments. All 64 answers terminate with learned
EOS and execute losslessly; 14 are correct, comprising 6/32 baseline and 8/32
perturbed cases. The two separate motor demonstrations also pass.

The saved tagged prefix-matrix contrasts end at 71.0938–85.2301% of their maximum
prefix norms. Final matrix-difference norms are 1.697858–1.931472, and actual
read-difference norms are 0.170692–0.391358. First/second-byte output-potential
contrast norms remain nonzero at 0.019194–0.077878. Writer-frozen exhibits the
same qualitative pattern: final/max-prefix ratios 72.0366–84.1145%, matrix
contrasts 1.769049–1.922239 and read contrasts 0.178994–0.455226. Thus complete
erasure before reading or output is not the observed explanation. These norms
do not prove that the retained representation supports correct decoding.

The contribution decomposition shows order pressure. Across all 32 baseline
queries, the first document's transported contribution retains 29.6812–53.9215%
of its norm before the second document, and the last document's final contribution
norm is 1.949954–3.229772 times the first's. Contributions can cancel; these are
not additive percentages of useful information. Tagged same-assignment-swap read
directions across the two questions have cosine 0.911542–0.936529; their first/
second-byte potential changes have cosine 0.940380–0.993851 despite opposite
required answer changes. This describes weakly differentiated binding effects,
not an independently established recoverability or saturation claim.

| Isolated block | Mean change in three-phase loss | Cases with lower / higher loss | Answer changes |
| --- | ---: | ---: | ---: |
| Writer positive | −0.000088090665256748 | 8 / 0 | 0/8 |
| Writer negative | +0.00008831806812612322 | 0 / 8 | 0/8 |
| Address positive | −0.00035182798402100257 | 8 / 0 | 0/8 |
| Address negative | +0.0003524368835029733 | 0 / 8 | 0/8 |

Loss is the mean of three half-sum-squared potential errors. These finite effects
use the declared rate 0.01 and unchanged output weights. They argue against an
immediately reversed update direction on these eight cases; they do not establish
convergence, sufficient credit, useful learning, or a reason to repeat training.

## Numerical audit and custody

The independent saved-only audit passes all 1,139 source-file checks, 217 artifact
hashes and 66 execution receipts. It checks worker input separation, original
endpoint preservation, actual teacher/proposal accounting and the exact positive
after / negative subtraction custody. Query-only workers receive bound weights,
memories and question bytes, with no targets, documents or offline roster.
Source hashing still reads implementation files containing authoring literals:
this is data-routing custody, not filesystem secrecy.

Float64 reconstruction from represented float32 traces separates recurrence
rounding from the ideal write identity. Across baseline observations, maximum
absolute matrix closure error is 1.637689e−7 without rounding residuals and
4.996004e−16 after their propagation. Read closure is 8.253503e−8 without
residuals and 2.498002e−16 after recurrence and final-read residuals. The auditor
records its fixed tolerances per reconstruction. This is receipt algebra within
those tolerances, not an exact-rational proof of every floating operation.

Before this freeze, the old two-byte runner and recipe were normalized to their
already committed LF Git blobs. Their original physical bytes remain archived
under `.pytest-tmp/localization-source-line-endings-before`; the hash-bound
normalization record preserves both identities. No old evidence metadata or
numerical code was rewritten. This run uses the newly admitted source manifest.

## Exposure and resources

There are 56 ingestions (16 baseline, 8 proposal, 32 perturbed), totaling 2,800
byte/terminator steps; 64 free queries use 1,664 query-byte steps. The eight
actual teaching calls add eight query traversals/208 steps and 24 byte/EOS
offers, or 6,168 scalar neuronal residual offers. Their weights are discarded.
All free calls advance three phases: 192 free advances and zero supplemental
phases. Together with 24 teacher advances and 24 teacher score revalidations,
there are 240 actual output-score evaluations.

External diagnostics separately use 24 proposal residual vectors and 24 signal
projections, alongside the production teacher's 24 projections; parent scoring
uses 192 target residual vectors from saved phase potentials. Observations repeat
64 query-key projections and 800 writer-map observation steps (2,400 linear
maps). There are 64 real action attempts plus two motor demonstrations. None of
these external label uses is an additional teacher update. The decision JSON
retains the complete exact count dictionary.

Serialized tensor accounting records 14,348,800 baseline trace bytes,
62,056,088 baseline retained bytes, 6,709,920 proposal bytes, 146,786,688 perturbed
retained bytes and 907,264 observation bytes. These artifact/stage quantities
are distinct from simultaneous process memory.

| Monitored process | Outer seconds | Observed tree peak MiB | Limit |
| --- | ---: | ---: | --- |
| Diagnostic including workers/actions | 22.1348402 | 822.66015625 | 60 s / 2,048 MiB |
| Saved-only audit including startup | 4.1007099 | 510.05078125 | Remaining 37 s / 2,048 MiB |
| Aggregate computation | **26.2355501** | Separate sequential peaks | **60 s** |

The audit's internal duration is 3.4855971 seconds; the outer duration governs
the aggregate. The launcher's PowerShell `exit` prevented the intended following
audit command, so root invoked that audit once in a separate monitored process
with the remaining 37-second cap. There was no repeated model run or audit.
Orchestration gaps are not computation time. Memory monitoring samples live owned
process peak working sets every 500 ms and may miss brief children; the values
are observed process-tree bounds, not allocator-exact peaks.

The later saved-only descriptive summary performs no learner/native import,
new model call or probe. It is external result analysis, separate from the
protocol run and its monitored independent audit.

## Evidence and closure

The [machine-readable decision](../../benchmarks/local_memory_localization_v1.decision.json)
binds admission, report/completion, both watchdogs, auditor/script, summary/script,
normalization record and endpoint evidence by SHA256. The report/completion
remain under `.pytest-tmp/local-memory-localization-v1-run`. The summary is
`.pytest-tmp/local-memory-localization-v1-summary.json`, SHA256
`e1aae1aeacba7af3e0578c54b4f69bbcf9a87166beb82a56bc06538e664c7f58`.
The independent audit SHA256 is
`ce18f50ac0d7ffc256b6cbc4a2fb2e8791ad3566ffdac6534010eed11d2897fe`.

Close this diagnostic with the frozen endpoints unchanged. The host-authored
occurrence motor still supplies the action schema; real execution does not
establish neuronal action selection or learning from execution feedback. No
retention improvement, general reading, coding transfer, benchmark comparison,
architecture adoption or gate advancement is established. Any next repair needs
its own prospective review and contract.
