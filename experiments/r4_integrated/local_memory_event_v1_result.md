# Fixed neuronal byte-event return: failed qualification

The fixed event encoder passes engineering and cost checks but reduces useful
behavior in this integrated workload. Tagged solves only 1/8 fitted cases and
none of the novel, delayed or interference cases. Its twenty final answers are
all `cd`, exactly matching aggregate, writer/maps-frozen and writes-disabled.
The immediate byte-input comparator reproduces its prior result exactly and
retains 5/8 fitted, 2/4 novel and 2/4 delayed successes. Qualification and all
reported advantage flags are false. Close this fixed fit; further work requires
a bounded architecture review, not an automatic repair, sweep or extension.

The [frozen recipe](local_memory_event_v1.md) and source
`1935060d6c1c39ed9d0115218b2a635137f5c1e3` own this experiment. The
[decision](../../benchmarks/local_memory_event_v1.decision.json) binds 22 evidence
hashes, the full configuration, admission and measured results. No protected
benchmark or general language capability is established.

## Controlled comparison and final behavior

Five candidate arms use a fixed 32-neuron recurrent encoder over generic,
byte-preserving cap-16 events. Every raw byte and string EOS still advances that
encoder; each event's final activity advances the writer/query controller. Its
three immutable tensors add 8,320 bytes per candidate. All six arms retain the
same multiscale writer, local read/value credit and LMS output. The sixth arm is
the fresh a1ab v4 multiscale byte learner, with identical common initial tensors.
No fitted checkpoint is used as initialization.

| Arm | Fit /8 | Novel /4 | Delayed /4 | Interference /4 |
| --- | ---: | ---: | ---: | ---: |
| Tagged event | 1 | 0 | 0 | 0 |
| Aggregate event | 1 | 0 | 0 | 0 |
| Writer/maps-frozen event | 1 | 0 | 0 | 0 |
| Frozen event | 0 | 0 | 0 | 0 |
| Writes-disabled event | 1 | 0 | 0 | 0 |
| Current tagged byte reference | 5 | 2 | 2 | 0 |

All twenty outputs from each updating event arm are `cd` with genuine EOS; only
fitted case 5 has that target. Both same-question novel swap pairs, `ac/ca` and
`bd/db`, emit `cd/cd`, as do their delayed/interference checks. Frozen emits hex
`0000` at the cap on all twenty cases and is not accepted as an answer. Thus the
event candidate shows no final-answer benefit from writes or learned writing.
Constant free outputs do not prove identical memories, erased contexts, saturation
or a specific internal cause; those forward features were not measured here.

For completeness, the byte reference's twenty outputs in case order are:

| Cases | Reference outputs |
| --- | --- |
| Fit 0-7 | `aa ab bb bb cd cd dd dd` |
| Novel 8-11 | `ca ca bd dd` |
| Delayed 12-15 | `ca ca bd dd` |
| Interference 16-19 | `da da cd cd` |

Its initial and endpoint full states, all 1,024 scientific lesson receipts and
twenty output receipts exactly match the authenticated a1ab multiscale tagged
run, excluding only arm label, elapsed time and separately audited new input-work
metadata. Candidate writes-disabled is not required to reproduce the old control:
its question representation also changes with the event encoder.

## Acquisition and learning curves

The run completes exactly 6,144 first-attempt lessons, 18,432 teacher byte/EOS
offers and 15,360 enabled decoder opportunities. It records 15,360 actual changed
decoder steps, separately from 4,737,024 offered neuron residuals and 3,947,520
enabled residual opportunities. All 120 final Python actions and two motor
demonstrations have audited execution receipts. Teacher corrections follow first
attempts and are not autonomous successes or action-feedback learning.

Each row below summarizes its fixed 1,024 lessons. Loss is mean
half-sum-squared potential error across the three teacher positions, not generated
sequence error or cross-entropy.

| Arm | First exact /1024 | First 128 exact | Last 128 exact | First 128 loss | Last 128 loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tagged event | 104 | 12 | 16 | 0.277614 | 0.254183 |
| Aggregate event | 103 | 12 | 16 | 0.277628 | 0.254238 |
| Writer/maps-frozen event | 104 | 12 | 16 | 0.277616 | 0.254128 |
| Frozen event | 0 | 0 | 0 | 0.500000 | 0.500000 |
| Writes-disabled event | 98 | 12 | 12 | 0.277736 | 0.255976 |
| Current tagged byte | 159 | 13 | 34 | 0.274995 | 0.230540 |

Tagged's last three 128-lesson losses are 0.254221, 0.254761 and 0.254183. The
fixed endpoint is poor; these online windows do not establish convergence or
justify more exposure. Query updates occur in 1,023 lessons of every updating
arm. The `memory_updated` flag includes query-map changes, so its corresponding
1,023 writer-frozen changes are not evidence of learned writing. Writes-disabled
has zero memory updates and frozen has zero parameter updates.

## Work and resource accounting

| Execution | Seconds | Process-tree peak MiB |
| --- | ---: | ---: |
| Disposable cost | 25.9666326 | 1029.09765625 |
| Saved cost audit | 3.4607916 | 448.12109375 |
| Full acquisition/action run | 309.8872667 | 1104.93359375 |
| Saved full audit | 4.2491095 | 499.71484375 |

Full run plus audit totals 314.1363762s. Peaks are independently measured, not
summed across sequential runs. Fresh cost admission projects 773.893428341368s:
694.2271487414837s for lessons/cap at the original 2x margin,
49.66627959988427s for other cost at 2x, plus 30s allowance. Its conservative
parent-plus-worker peak is 1032.56640625 MiB. The original 900s/2048-MiB ceiling
is unchanged; no rejected launch or relaxed checker precedes this admission.
Actual full-run peak exceeds that estimate by 72.3671875 MiB while remaining
within the ceiling; the sampled estimate is not an exact allocation bound.

Cost supplies 24 disposable lessons, 72 teacher offers, sixty enabled/changed
decoder steps, six extra cap predictions, 120 actions and two demonstrations.
Those weights are discarded. The prospective
[cost decision](../../benchmarks/local_memory_event_v1.cost_decision.json) remains
unchanged, including its issuance-time `full_fit_ran: false` field.

The full run separately counts 628,344 raw input bytes, 25,632 string ends,
544,980 actual encoder steps, 166,080 writer-controller events and 199,836
query-controller events. It counts 143,244 memory-write opportunities, 120,408
writer-controller eligibility events and 97,572 credit-transport opportunities.
These include final ingestion and are not changed-matrix or slow-weight counts.
Frozen writer controls still write persistent memory; writes-disabled still
advances controller traces but skips writes and credit transport.

Production host segmentation covers 21,360 calls/523,620 bytes. Additional host
accounting covers 15,620 calls/380,120 bytes; the saved auditor independently
checks these with a regex and never reexecutes an encoder. Complete model and
retained-state tensor payload is 232,441,976 bytes. The encoder reduces expensive
controller events while adding byte-level encoder work; fewer events alone do
not establish a speedup or useful retained information.

Measured complete tagged lessons total 62.6021654s versus 100.7649337s for the
byte comparator (ratio 0.6212693553). All six arms' lessons total 284.3944962s;
final ingestion takes 2.2515463s, serialization 0.3110860s, the worker package
7.9868738s and the action package 9.5798121s. These are descriptive measurements
of this ordered run, not a randomized speed comparison. The faster tagged lesson
execution accompanies substantially worse answers.

Per-output worker seconds start after distractor reads although row input-work
counts include them. `worker.package_seconds` and the outer watchdog include all
reads, serialization, hashing and accounting. Compare costs with matching scopes.

## QA, custody and limits

The frozen recipe records 158 distinct passing affected cases, plus targeted
reruns of two strengthened production-reference fixtures. The new/amended QA
subset supplies 105 teacher offers; this is separate from cost/full exposure and
excludes other pre-existing disposable fixtures in the affected-file receipt.
Encoder-only QA makes 71 steps, including one deliberately rejected nonfinite
step; worker QA makes twelve untrained empty-memory queries without teachers or
actions. Source-only scalar accounting QA introduces no model calls. The worker's
pytest cache-permission warning is nonblocking; there are no failed model/process
attempts to reinterpret as passing evidence.

The full audit verifies 1,156 source files, all five transport/training artifacts,
the exact command/owned-process identities, fixed controls, fresh initialization,
complete work ledger and predecessor parity. Audit SHA256:
`ff392f2a3d9253175e529c53dce4b8451d26e2381d2f684d667c3641b01e1933`.
Report SHA256:
`1e241e7482bda857e31f207fca17583bf86872a32f02bb85aa3fc26c6d24d69c`.
The saved-only descriptive extraction is
`.pytest-tmp/local-memory-event-v1-description.json`, SHA256
`4f0274a5879298a0b85ca2188fe32085a322e7537300d2cd6bd7e61b02064601`.
It verifies all 22 decision evidence hashes and performs no model calls.
The independent saved timing extraction is
`.pytest-tmp/local-memory-event-v1-performance.json`, SHA256
`c967b8688b4dadac6869f79ca6d9b38a8f91558890606f9966e2edc57c02a383`;
it also uses saved JSON only.

The host-authored occurrence motor supplies matching boundaries and candidate
drives; generated neuronal bytes feed that narrow Python-action scaffold. This
does not establish a general neuron-learned action policy. Delayed checks measure
read-only distraction, not decay or retention through further teaching. Novel
combinations are previously exposed development material. Lossless preprocessing
qualification remains valid, but this fixed neuronal encoding has not earned
adoption as a useful replacement in the integrated learner.
