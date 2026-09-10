# Repeated teaching: measured cost and final restoration

The single frozen run completed successfully in **1,519.0128574 seconds** with
a sampled process-tree peak of **1,529.1484375 MiB**. Its final paired snapshot
passed fresh restoration. These are costs of the complete bounded acquisition
protocol, including its diagnostics and persistence. There is no matched timing
comparison and no end-to-end speedup claim. Successful custody does not establish
successful learning; the [protocol](repeated-teaching-acquisition-v1.md) defines
the separate behavioral measurements.

Execution used commit `f577413dd9bac32302c3c84d5e09b950f8c8473c` and the unchanged
causal learner. Evidence is under `.pytest-tmp/repeated-teaching-v1-run`; the
[outer launch receipt](../../.pytest-tmp/repeated-teaching-v1-run.launch.json)
records exit code 0, a 3,600-second limit and a 4,096-MiB monitored memory limit.
Memory is the maximum simultaneous sum of live owned processes' peak working
sets sampled every 500 ms. Brief unobserved children may be missed; this is not
an exhaustive allocator or instantaneous working-set measurement.

## Measured elapsed scopes

| Receipt or phase | Seconds | Scope |
| --- | ---: | --- |
| Outer launch | 1,519.0128574 | Complete monitored process lifetime |
| Final runner stdout | 1,518.2762336 | Through final report publication and final disk admission; before final cleanup |
| Report elapsed | 1,518.0527094 | Sampled before final source verification, report publication and disk admission |
| Retained-state initialization | 10.7084797 | Two parent authentications, one selected payload decode, two independent actor copies |
| Final block commit | 12.1104661 | Complete ledger validation, paired snapshot serialization and final publication |
| Fresh final audit | 18.4815484 | Full saved-file checks, two strict payload restores and complete block-ledger validation |
| Saved scalar learning review | 7.1920214 | Authenticated journal streams, task regrading and saved diagnostic aggregation |

The phase rows are components of the runner and outer totals, not additional
costs to add to them. Initialization, commit, final audit and scalar review are
sequential components, but do not cover all runner work. Commit and audit return
their clocks before context-exit checks or close operations. The final commit's
`work` dictionary is cumulative over the checkpoint store's lifetime, including
earlier journal appends; its elapsed time covers only the final commit call.

Attempt timers start after disk admission and include public observation setup,
state hashes, free prediction, actual edit execution where applicable, teaching,
literal-weight capture, output diagnostics and final state hashes. They exclude
journal publication and the subsequent disabled-state seal. Diagnostic capture
and analysis timers are nested within attempt time, as are execution timings.
Panel progress clocks are cumulative runner clocks, not isolated probe durations.
Cloning, panel seals and append work outside attempts have no separate saved
elapsed total. This review did not scan attempt or teacher journals to derive
additional aggregates.

## Completed work and durable state

The receipts record 252 complete responses, 222 teacher records and 111 actual
edit executions. Main work contains 192 responses and corrections; five panels
contain 60 responses and 30 diagnostic teacher records, including reference
records that publish no probe history. Each neural route receives 64 main
corrections with 1,120 byte/EOS offers and 96 motor labels, plus ten disabled
diagnostic corrections with 175 symbols and 15 motor labels. Across the two
neural routes this is 2,590 symbols and 222 motor-label offers. There is no source
ingestion or factual append.

Construction receipts confirm two initial actors, ten disposable panel actors
and two fresh final restore actors: **14 constructions**. Restricted decoder
counters record one parent payload decode and two final payload decodes:
**3 decodes**. The frozen source calls the idle full snapshot API ten times for
RAM panel copies and twice for final payloads: **12 full snapshots**, a
source-derived count rather than a dedicated snapshot counter. Probe snapshots
write no tensor payloads. Final persistence writes exactly two payload files,
each 115,712,640 bytes, for 231,425,280 bytes combined.

There is one final durable block boundary, HEAD sequence 0. The earlier scalar
panels are not resumable snapshots. Until final publication, the authenticated
parent remains the latest safe state; incomplete work has no automatic replay.
The final boundary SHA256 is
`3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677`.

| Route | Final model SHA256 | Final payload SHA256 |
| --- | --- | --- |
| online | `07855c61ca02e1dea8b1e8044df2576643689b8e55a28a1d50109a3580b54d7a` | `f77695cd9603e00e031cc867ce53573e2148eb14bb39d2e70b576b317ef66f0e` |
| learning_disabled | `d75683ec64583b5ae4ee6786d367555ee702fb2e57fe44848b25fd7151fc5007` | `159577980753f49e3e97ba235430671218ba93f2e539622e11573c120d58d6a1` |

Both strict restorations preserve actor experience 401 and reader experience
406. Both factual-memory hashes remain
`cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e`,
retaining the parent's 1,982 slots. The reference retains 84 documents and ends
with 82 demonstrations; its cumulative offers are reading 36, SPLICE 36 and
STOP 0. The final audit performs no task execution or teacher replay.

Initialization authenticates the bank, complete parent checkpoint metadata
inventory, source bindings and fresh parent frontiers twice. The sole historical
auditor exception authenticates the pinned original auditor bytes; all other
parent sources retain their exact checks. The final store binds the current
frozen source map. Fresh restoration authenticates both payloads before copying
either into caller actors, validates tensor geometry, finite state and fixed
wiring, restores independent storage, compares hashes and rechecks the saved
frontier after restoration.

## File traffic and ownership counters

The three final journals contain 510,318,993 bytes: attempts 275,870,643 bytes
in 192 rows; teachers 152,720,194 bytes in 192 rows; probes 81,728,156 bytes in
five rows. The native owned-journal counters record 389 write calls and
write-through completions, 510,318,993 bytes written, and exactly the same number
of actual new-tail bytes read back in 730 reads. Six owned current-prefix checks
avoid 1,020,637,986 bytes of repeated prefix hashing. Those avoided bytes are a
counterfactual read count for these selected checks, not measured elapsed savings.

| Saved counter scope | Recorded work |
| --- | --- |
| Initializer direct authenticated reads | 285,902,300 bytes; 1,082 passes |
| Initializer parent inspectors | 1,593,682,812 hash-read bytes; 1,384,232 metadata-read bytes; six fresh journal and four fresh payload checks |
| Final store general reads | 490,417,379 file-hash bytes and 428,751 metadata-read bytes |
| Final store ledger validation | 510,318,993 index-read bytes plus 510,318,993 parse-read bytes |
| Final store writes | 231,707,118 checkpoint bytes; seven checkpoint fsync calls; two payload writes and one HEAD publication |
| Final payload ownership | Two authenticated versions; two reused checks avoid 231,425,280 hash bytes |
| Fresh final audit | 2,261,996,193 file-hash bytes and 232,523,724 direct-read bytes; nine fresh journal and six fresh payload checks |
| Fresh audit ledger validation | 510,318,993 index-read bytes plus 510,318,993 parse-read bytes |

General file-hash counters include their relevant journal/payload reads; do not
add the same category bytes again as though they were separate traffic. Index
and parse reads are separate passes. Saved review makes one further full pass
over each new journal and authenticates its bytes, row count and SHA256 against
the final boundary. It does not emit a read-byte counter. From the successful
frozen call schedule, total new-journal reads are nine times the final journal
size, **4,592,870,937 bytes**: actual append tails once, commit ledger twice,
fresh final inspector three times, final audit ledger twice and scalar review
once. This is a source-derived accounting identity, not a device I/O measurement.

Host state-seal work records 80,229,681,776 tensor bytes hashed. The ten clone
receipts together record another 5,241,708,280 custody tensor bytes hashed and
2,528,598,800 named factorized tensor bytes. These are scoped work counters, not
peak memory or a complete count of all actor computation.

## Disk envelope and instrumentation limits

Final runner stdout records **742,750,281 artifact bytes** and **9,420,750,848
free bytes**. A subsequent file-stat inventory independently confirms that
artifact total without reading journal or payload contents. This is below the
805,306,368-byte artifact cap and above the 8,589,934,592-byte free-space floor.
The 268,435,456-byte pending reserve is inside the artifact cap. The paired
118-MiB payload bounds leave room for up to 16 MiB of boundary metadata and the
small HEAD. Admission covers this runner's known writes; it is not a filesystem
quota against unrelated writers. The outer launch/stdout/stderr receipts live
outside the run directory and are not included in its artifact total.

`report.json`'s disk values, 510,632,511 artifact bytes and 9,648,037,888 free
bytes, precede final persistence. They must not be relabeled as final inventory.
Free-space snapshots can change because of unrelated filesystem activity.

Full snapshot deepcopies, factual-state copies, constructor allocations and
some finite-state scans lack dedicated complete byte/operation counters.
Source/manifest hash reads performed by the runner, receipt encoding and
publication, directory traversal, disk-admission stats and simple-reference
copies are not exhaustively instrumented. The append object measures elapsed
time internally, but the frozen runner does not save that summary. Literal
diagnostics retain their own copy/work/time receipts; nested teacher/controller
work may overlap parent aggregates and must not be summed indiscriminately.
Returned store work snapshots precede final scope exit and handle closure, so
their live-handle values are not evidence of leaked handles. Complete monitored
elapsed time and sampled peak memory remain distinct from these partial counters.

## QA and preparation outside the acquisition run

These measured launches are preserved separately and are not included in the
1,519.0128574-second acquisition total. QA uses its declared small fixtures;
it does not represent additional historical acquisition trials.

| Launch receipt under `.pytest-tmp` | Status | Seconds | Sampled peak MiB |
| --- | --- | ---: | ---: |
| `repeated-teaching-v1-qa.launch.json` | failed | 27.3633660 | 787.1640625 |
| `repeated-teaching-v1-qa-corrected.launch.json` | failed | 10.8562925 | 1,006.2812500 |
| `repeated-teaching-v1-qa-hash-corrected.launch.json` | passed | 13.7885141 | 1,007.23828125 |
| `repeated-teaching-v1-manifest-preparation.launch.json` | passed | 0.3287449 | No process sample captured |

The preparation receipt's zero sampled peak does not mean zero memory use.
The [measurement ledger](../../benchmarks/repeated_teaching_acquisition_v1.measurement.json)
sums these four launches and the complete experiment to 1,571.3497749 seconds.
Subsequent banking passed in 0.4782003 monitored seconds at 24.96484375 MiB,
with zero model calls; its receipt is
`.pytest-tmp/repeated-teaching-v1-banking.launch.json`. Banking time is explicitly
outside the ledger's listed total. Documentation, manual review, doctor and lint
are outside these measured totals; no additional model or task execution was
used to prepare this cost document.

Receipt identities: initialization
`3fe30f77db52035df5370e10ec9e18de2d1a376555727352ade907d2fb75fd89`;
report `3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262`;
restore `b42c5fd25972b4e984901502a5bd50ecbecc565a6ee7aa4370174284b9da86f5`;
saved learning review
`402aae6d061464aecccbdc73d12abd0683ed0ee7d4408d650550070a69efea61`.
