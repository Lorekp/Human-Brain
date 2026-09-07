# Recurrent content ordinary QA: observed cost

2026-09-07. The first bounded invocation at
`bf9a9f0451b3d7de1d17f9e9782f872dc69f112e` passed both ordinary checks in
18.527647 watchdog seconds, with 866.30859375 MiB observed owned-process peak.
Its limits were 120 seconds and 2,048 MiB. JUnit reports 17.239 seconds total;
the arithmetic and same-actor test cases report 1.698 and 11.202 seconds.
These intervals include different setup/teardown scopes and are not additive.
The [QA review](recurrent-content-qa-v1.md) owns the mechanics result; the
[resource plan](recurrent-content-resource-plan-v1.md) owns prospective bounds.

This review only read completed ordinary receipts and source, with no model,
checkpoint decoder, feature evaluation, action or test replay. The Windows
fixture ACL required elevated read-only access. A pytest node-ID cache write
warning (`WinError5`) did not affect either check; stderr was empty. Doctor
reported clean tracked source and the unrelated untracked
`src/humanbrain/direct_request_splice_learner.py`.

## Actual exposure

One standalone controller and two fresh combined actors performed 128 ticks:
124 externally offered symbols plus 4 actual free ticks. There were 390
categorical draws. The six-tick arithmetic fixture made 24 draws; the combined
fixture made 366. The latter used authored mode potentials, so its absence of
literal draws is fixture behavior, not an estimate for a zero-initialized
broader policy.

| Recorded scope | Ticks | Offers | Draws | Controller current products |
| --- | ---: | ---: | ---: | ---: |
| Standalone arithmetic | 6 | 5 | 24 | 3,612 |
| Four completed source observations | 8 | 8 | 24 | 2,432 |
| Three public scans and actual free responses | 108 | 105 | 324 | 51,460 |
| Two corrections | 4 | 4 | 12 | 2,304 |
| Disabled acknowledgement | 0 | 0 | 0 | 0 |
| Rejected late source publication | 2 | 2 | 6 | 608 |
| Total | 128 | 124 | 390 | 60,416 |

The combined fixture admitted two one-byte documents and observed three
transient one-byte documents, including the deliberately rejected transaction.
Ten source offers therefore occurred; eight completed publication and two
reached the injected failure. Public scanning contributed 105 offers, including
39 status/framing offers. Four combined correction offers and five standalone
offers complete the 124. Separate frame-only mechanics consumed 47 symbols
(3 arithmetic, 44 combined) without controller draws or learning; they are not
silently added to the controller offer total.

There were three combined free calls, seven actual frozen-prior evaluations
(three free and four teacher), three motor labels and one actual STOP child.
The child, PID 14836, returned 0 with stdout `610a`, no stderr and complete capture;
its own interval was 0.036619999998947605 seconds under a 3-second limit.
Both nodes each accepted one strict in-memory state load and rejected one
malformed load: two accepted and two rejected in total. No checkpoint file was
decoded. The combined test also rejected a modified pending trace and the late
source proposal. There was no test-until-success loop or acquired-skill threshold.

## Work and storage interpretation

The following are sums of the distinct returned work dictionaries, including
the rejected source's captured work. They are not a complete process profiler;
fixture construction, standalone assertions and some checks outside returned
ledgers are covered by the watchdog, not these counters.

| Scope | Controller bytes hashed | Eligibility fork bytes | Controller named allocations |
| --- | ---: | ---: | ---: |
| Standalone arithmetic | 369,719,177 | 73,870,848 | 665,025,648 |
| Completed sources | 492,872,686 | 0 | 788,047,808 |
| Public scans/free | 372,762,296 | 0 | 223,451,936 |
| Corrections | 591,221,280 | 147,741,696 | 394,058,720 |
| Acknowledgement | 295,548,778 | 73,870,848 | 196,988,928 |
| Rejected source | 123,218,145 | 0 | 197,011,952 |
| Total | 2,245,342,362 | 295,483,392 | 2,464,584,992 |

Additional returned custody counters report 2,993,610,662 tensor bytes hashed
outside the controller-specific hash counter. Controller validation visited
301,968,326 values, and inherited assembly validation visited 1,432,316,949
values. Eligibility initialization wrote 101,572,416 float64 zeros. The two
admissions allocated 6,602,768 tape tensor bytes; transient source calls allocated
no new tape. These cumulative operations can greatly exceed live memory and
must not be reported as peak allocations.

Thus the recorded arithmetic is sparse, while state validation, hashing and
transaction preparation remain substantial dense work. The receipts do not
time these stages independently, so they cannot establish an elapsed-time
bottleneck or justify a throughput extrapolation. The source envelope remains
196,988,928 controller bytes for weights, original/forked eligibility and a
proposal during resolution, before old actors and scratch. Disabled learning
still computes proposals and publishes actual RNG advancement.

The two saved JSON receipts occupy 86,762 and 22,555,357 bytes. Their detailed
fixture serialization is also real output cost, not a selected future journal
format. Before a broader continuation, retain separate timing for input scan,
free/teacher prior, dense resolution/custody, serialization and safe-boundary
persistence. These short fixtures do not measure 10,000-symbol learning,
large occupied-memory behavior, V7 restoration, or full-workload speedup.

The monitor samples the simultaneous sum of owned processes' peak working sets
every 500 ms; brief children can be missed. Its 866.31 MiB is that declared metric,
not isolated tensor residency or a GPU measurement.

## Exact bindings

Paths below are repository-relative. SHA-256 hashes bind the bytes read.

| Evidence/source | SHA-256 |
| --- | --- |
| `.pytest-tmp/recurrent-content-v1.launch.json` | `20667153e169ab070d4cc0f9ac2003013cb8036cf3aab4bcef54665665d905f6` |
| `.pytest-tmp/recurrent-content-v1.junit.xml` | `ddeef244a8ae732cb9e117e172f1c2542285db357881aefd34184d55eec521ec` |
| `.pytest-tmp/recurrent-content-v1.stdout` | `545a580048b8fcd3b140327bf1d6d2e263e37c4dce52bebadb883e2ab2a59c74` |
| `.pytest-tmp/recurrent-content-v1.stderr` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.pytest-tmp/dev/doctor-al7ngzac/receipt.json` | `df18c5a1a762dde5ff9d67f187c77f384dc99f071478de5de83f2c10c0fd1e24` |
| `.pytest-tmp/recurrent-content-v1-tests/test_recurrent_content_local_a0/recurrent-content-arithmetic-passed.json` | `a09d27d2cf79af2f050a04caa8ee652a3f07b64079ef230cd68f4be373befde8` |
| `.pytest-tmp/recurrent-content-v1-tests/test_recurrent_content_same_ac0/recurrent-content-same-actor-passed.json` | `5272f7f3e81af9c517596a4d8bc5b139cbd788a2b8eee88b0c80b417fe2408a6` |
| `src/humanbrain/recurrent_content_controller.py` | `e2fb738b86d820db85f3e8b6190a93a9c85fbc704bbfe897d8ac7c5a335f56b5` |
| `src/humanbrain/recurrent_content_frame.py` | `e7cd31a52a27137976b372b90549e2d05afc1cede6a680f42a82efff3f456007` |
| `src/humanbrain/recurrent_content_learner.py` | `e89eb5ccfa44eda3b9a8306bc8603b2aeab11d78630e47de4cc2d8f1e22c9ece` |
| `tests/test_byte_sequence_decoder.py` | `de760eeb039a0e02cef7bb47733d0f8ce157d40d94da6f95a9e98b902690c902` |

The monitor binds watchdog launcher
`4f26b669442a596cf3023c8d066ef49ff6231732baa575b8c503ad4ae3d198dd`.
