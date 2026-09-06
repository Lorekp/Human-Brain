# Local memory/byte v5: affordable fit, no useful source conditioning

2026-09-06. The optimized workload passes cost admission and completes its one
full acquisition in **322.420 seconds**, with **847.141 MiB** conservative peak
memory. The independent saved-only audit passes. **Capability qualification
fails:** every updating arm emits `mavi lune` on all twenty final questions,
including when the retained document requires a different answer.

Frozen source: `cdb9c9171ab5bde24fff2877bfdb67432708b99e`.
The [recipe](local_memory_byte_v5.md),
[prospective admission](../../benchmarks/local_memory_byte_v5.cost_decision.json)
and [completed decision](../../benchmarks/local_memory_byte_v5.decision.json)
preserve distinct cost and acquisition evidence. The admission's `full_fit_ran`
field describes its creation before launch; it remains immutable.

## Acquisition and action result

Each arm receives 1,024 sequential lessons. First attempts precede teacher
byte/EOS correction; source ingestion precedes questions and targets. Cost
weights were discarded, verified by identical fresh initial tensor states.

| Arm | Acquisition first exact / 1,024 | Fit / 8 | New value / 4 | Delayed / 4 | Interference / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tagged | 72 | 1 | 0 | 0 | 0 |
| Aggregate | 70 | 1 | 0 | 0 | 0 |
| Writer/maps frozen | 65 | 1 | 0 | 0 | 0 |
| Fully frozen | 0 | 0 | 0 | 0 | 0 |
| Writes disabled | 68 | 1 | 0 | 0 | 0 |

All four updating arms generate valid UTF-8 and EOS on every final question;
the same trained phrase reaches the real edit/execution path. The frozen arm
produces invalid UTF-8 on all twenty checks (sixteen raw limits and four raw
EOSs). Exact generated answers plus lossless executed output own the score;
successful execution of an unchanged `pending` program cannot count as solving.

Tagged teacher-byte-weighted loss falls from 2.944 in the first 256 lessons to
0.651 in the last 256. Writes-disabled loss similarly falls from 2.941 to 0.649.
First-attempt exact counts in those windows are 0 to 23 for tagged and 0 to 24
for writes disabled. These improvements show teacher-forced sequence learning
without useful document conditioning. The single fit success is the one item
whose target happens to match the constant final phrase. There is no useful
learned-writing or tag-specific advantage.

The preselected saved-only contrast diagnostic finds different tagged synapse
matrices for the two fit/new-value memory pairs (relative Frobenius differences
0.354 and 0.491; cosine similarities 0.936 and 0.876), yet identical generated
answers for all four same-query contrasts. Aggregate and writer-frozen matrices
also differ while their answers collide. Source-dependent matrix differences
survive; useful fact coding, addressing and decoder
use remain unresolved. Metrics are descriptive, with no new success threshold.

The workload contains 5,120 first attempts, 33,665 teacher byte/EOS offers,
26,932 enabled decoder update opportunities, 292,155 document byte/terminator
steps, 266,240 acquisition query steps, 19,240 final query steps, 100 action
attempts and two motor demonstrations. Source-free tensor state is 124,109,512
bytes, including source-derived synapses and credit; absence of raw documents
does not mean absence of retained information or storage cost.

## Cost and evidence

The disposable phase finishes in 27.430 seconds. The unchanged maximum-sample,
capped-generation, factor-two formula projects 800.543 seconds of learning plus
52.106 seconds of other work and the original 30-second reserve: **882.649
seconds**, below 900. Conservative cost peak is 772.770 MiB. No rate, seed,
lesson count, clock floor, margin or budget was changed to admit the run.
The original pre-optimization projection was 2,013.103 seconds; this is a
complete-workload engineering improvement, not a learned-capability gain.

Both strict audits authenticate 1,132 source hashes, five artifact hashes,
clock/native/package identities, source-free tensor/control invariants,
creation-keyed owned processes, 102 execution receipts and 100 workspaces.
The full audit additionally checks all ordered lessons, exposure totals,
initial-state reset, weight hash chains, exact scores and qualification.
It does not independently recompute local updates or rerun models/children.

- Cost audit: `.pytest-tmp/local-memory-byte-v5-cost-independent-audit.json`,
  SHA-256 `2a5ed6231a77b0c2b1aa1cda7d0fbae0290585a8a23e4787e2a64766958787ce`.
- Full audit: `.pytest-tmp/local-memory-byte-v5-run-independent-audit.json`,
  SHA-256 `3e94c78c9be678f978b2f4cb6a09948d37f92b2fa73f7ccc4a5d35e6f5114deb`.
- Saved-only description: `.pytest-tmp/local-memory-byte-v5-run.description.json`,
  SHA-256 `59f6c167d559ef68defac0adb6e676133e029f499d9a2e05100be29e98055925`.
- Saved contrasts: `.pytest-tmp/local-memory-byte-v5-contrasts.result.json`,
  SHA-256 `c04c9cc43cfe4681c9ca435fa208c704649256dc80f0fcc294dc73fb5cd1387f`.
  Its original diagnostic assertion confused raw EOS/limit with the motor's
  UTF-8 validation status; the recorded correction changes that assertion only,
  retaining the before-result pair selection, windows and numeric metrics.

V5 is closed without another fit, seed/rate sweep or longer replay. Preserve
the reusable speed improvements. Inspect the already saved same-query,
different-memory contrasts and compare local credit/context-use explanations
before selecting one bounded successor. Constant outputs with differing
synapses cannot by themselves locate the failure in the writer, addressing or
decoder. The preflight's order-credit hypothesis remains a hypothesis.

This is synthetic development evidence. The action motor still supplies a
host-authored matching/edit schema; even a successful answer would not establish
entirely neuron-learned action selection. No execution-feedback learning,
autonomous recovery, retention across further slow learning, comprehension,
integrated gate or protected coding evaluation is established or opened.
