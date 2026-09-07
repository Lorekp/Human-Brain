# Sequence assembly integration verification

Status: component mechanics, reader/action integration and independent action
arithmetic verified. No broader v4 workload has run. These are ordinary affected QA
checks, not an acquired-skill gate or a benchmark result.

## Component checks

Source `e2b51fe6fd3b0c9573f75b2c496a478af04c2f6b` passes three explicit nodes in
`tests/test_byte_sequence_decoder.py` on their first monitored launch:

- `test_sequence_assembly_cells_local_staging_contract`: one ten-input standalone
  bank, eight evaluations, eight teacher calls/seven entered offers/six completed
  offers. Three staged updates recruit two units and publish once. Two completed
  offers are disabled, one skips a novel context at capacity, and one entered
  offer fails the authored late finite check. Conflicting exact contexts reuse
  their unit, nearest ties select the lower index, and rejected stage/load changes
  leave live values intact. An authored full 512-slot bank tests exhaustion; it
  is not built by 512 teaching calls.
- `test_sequence_assembly_rolling_prefix_exact_boundaries`: one 129-candidate
  cache, 128 raw advances and seven feature reads. Full declarative coincidence
  activity and weighted scores agree exactly at prefix lengths 0, 16, 31, 32,
  127 and 128, using twelve reference kernel blocks. BOS leaves the rolling
  window at byte 32; three invalid advances leave it unchanged.
- `test_sequence_assembly_raw_observation_and_execution`: full raw framing at
  the selected limits, portable paths, exact splice/STOP/stale-workspace checks
  and two actual Python children. PID 45008 produces exactly 512 stdout and 512
  stderr bytes, including non-UTF8 bytes, after a 128-byte edit. PID 41352 emits
  513 stdout bytes: capture retains 512 and explicitly reports overflow and
  incompleteness. The capped output cannot receive a successful grade.

No reader or combined learner is constructed in these three nodes. They load
no experiment checkpoint and offer no task lesson. Both actual children return
zero; overflow is deliberately preserved as incomplete evidence.

The tests report **3 passed in 5.67 seconds**. The monitored process takes
**6.921719 seconds**, with **298.02734375 MiB** sampled maximum simultaneous
owned-process peak working sets, under a 60-second/2048-MiB guard. Brief children
may be missed by its 500-ms sampling; child identities also appear in the actual
execution receipts. This is QA cost, not projected broader-session latency.

## Retained receipts

The doctor records the selected checkout/environment at the committed source.
Its tracked diff is the evolving design document; untracked reader/action source
and the paused direct-request branch are disclosed and not exercised here.

| Receipt | SHA256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-ii9tf57g/receipt.json` | `7c8cbd8e377aa9fd51100f5665be1e1a9f8f5fb2a0ef5db790386f6f027d156c` |
| `.pytest-tmp/sequence-assembly-mechanics-qa-v1.launch.json` | `67785d56ebfb97c3cd97e044f0f1090af21285e206e313f8cdd2118501cce8b9` |
| `.pytest-tmp/sequence-assembly-mechanics-qa-v1.junit.xml` | `7848fdada374711e427cf1760a05e574da5a9910f0ee8509a1be1bf36842fe6a` |
| `.../test_sequence_assembly_cells_l0/assembly-cells-passed.json` | `f1c55d2ca0817f7d068869b6a7f79ef0da215bb56be5050a00d80bed90a2901c` |
| `.../test_sequence_assembly_rolling0/assembly-prefix-passed.json` | `86feb3896e223767272cffa564ae4a0327f62da5cd8bb71c0ee4986b5587ff1e` |
| `.../test_sequence_assembly_raw_obs0/exact-byte-execution.json` | `4fc50f631fb617ba7bf3ea044027d1d7948aa0df752b0f75379ffc7f93885011` |
| `.../test_sequence_assembly_raw_obs0/overflow-execution.json` | `6b5b40b2f89db88f57aea38392050823f4fbd0c701ed1a1d99404e7895671968` |

The abbreviated paths share `.pytest-tmp/sequence-assembly-mechanics-qa-v1-tmp`.
Raw assertion inputs and per-stage counters are retained beside their summaries.

## Reader and combined action checks

Source `42c747a1b44e7372de41af09be4cb2397abc4c9b` passes two further explicit nodes
on their first monitored launch:

- `test_sequence_assembly_reader_causal_prefix_and_atomicity`: one reader, one
  four-slot tape from three raw bytes and EOS, three free responses/40 decisions.
  An authored fixed codec first emits 33 question marks and reaches its limit.
  One offered `ab`/EOS sequence recruits three assemblies; subsequent same-input
  responses emit `ab` and EOS from those exact contexts. A disabled `c`/EOS offer
  leaves even the private bank unadvanced. A final two-event lesson fails its
  second proposal and leaves live values and the pending response unchanged.
  Across all three teacher calls there are seven entered events, six completed,
  three published clamps and no child executions. Independent scalar rails
  verify every request and synthetic non-UTF8 diagnostic byte in the recruited
  receptive patterns. Prior reader tensors and the whole tape stay unchanged.
- `test_sequence_assembly_combined_custody_and_atomicity`: one combined model,
  one three-slot tape from two raw bytes and EOS, three free actions/five payload
  decisions and three actual STOP executions. PIDs 49384, 27060 and 33092 each
  return zero and print `0\n`. The first SPLICE teacher deliberately fails at
  final reader publication after two reader events and three action proposals;
  an explicit ordinary-fixture retry succeeds. A later STOP teacher affects
  only its operation head, and teacher-free acknowledgement resolves both
  pending states. Total offered exposure is four reader events and seven action
  labels, including failed staging. Invalid incoming assembly occupancy and
  changed fixed context wiring fail before any incoming action value is copied.
  The next actual observation includes the previous child's raw feedback.

These are authored lifecycle fixtures, not evidence of fresh language transfer,
coding ability or teacher-free repair. They deserialize no experiment checkpoint.
The failed lessons are intentional ordinary fault fixtures and remain counted;
there are no failed pytest invocations or hidden workload retries.

The tests report **2 passed in 11.43 seconds**. The monitored process takes
**12.5077729 seconds** at **639.24609375 MiB** sampled simultaneous owned-process
peak, under the same 60-second/2048-MiB guard. Reader and combined call times are
3.35 and 4.56 seconds; these are not isolated performance comparisons.

| Receipt | SHA256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-8lbz4lsq/receipt.json` | `572f5d38ad4dac5e4c6dfbe2e21b3bc31565f6afe97208e6c8f5eaa5a3fc30af` |
| `.pytest-tmp/sequence-assembly-integration-qa-v1.launch.json` | `40b67af0bdfcec53f188e123d75db4cdd9d5c0db4beb362630ceb40440d20541` |
| `.pytest-tmp/sequence-assembly-integration-qa-v1.junit.xml` | `5459fa6452bc8458e8af043c9214d242554ac83d4de1879960694a81c913346e` |
| `.pytest-tmp/sequence-assembly-integration-qa-v1-tmp/test_sequence_assembly_combine0/combined-qa-outcome.json` | `0ad3cad72c7a39e36488075e4ea51c0405e998322543eff1a87142ee360fb5a7` |

The doctor records an empty tracked diff at this revision. The unexecuted v4
initializer and paused direct-request source are disclosed as untracked. All
raw reader and combined records remain under that launch's retained `-tmp`
directory. Across both launches there are two reader constructions (one nested
in a combined learner), one additional standalone assembly bank, five actual
children, 45 free reader decisions, eleven integrated reader target events
(including one failed event proposal and two completed events in a rejected
lesson stage), and seven combined action labels. The standalone
ten-input bank's seven entered offers are separate from those reader events.

Broader acquisition, fresh transfer, retention and teacher-free recovery must
be measured separately on the selected continuation.

## Independent action arithmetic

Source `6e9ce89386a991709237a8eea77194b98083a794` passes
`test_sequence_assembly_action_blocks_embedding_and_atomic_update` on its first
launch. One standalone action circuit loads four authored legacy tensors; an
independent inverse mapping checks every old coordinate and the entire zero
complement. A sixteen-byte file supplies seventeen boundaries, crossing the
sixteen-candidate block size. Full declarative features, output probabilities
and whole-bank local-update arithmetic agree with all three streamed heads.
The declared tolerance accommodates changed floating reduction grouping.

There are three direct scores, three successful private update proposals and
one deliberately rejected operation update. The latter fails its second
parameter proposal without changing live values. Four supplied action labels,
fourteen module feature passes, twenty-two blocks and 142 candidate feature
vectors are counted separately from three independent full feature oracles.
No update publishes; no reader, tape, child or checkpoint file is involved.

Pytest reports **1 passed in 4.69 seconds**; the monitored process takes
**6.1030449 seconds** at **674.609375 MiB** sampled simultaneous owned-process
peak, within 60 seconds/2048 MiB. This full-oracle QA peak is not a production
working-set measurement.

| Receipt | SHA256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-wniokm03/receipt.json` | `10a1893fb15d52dc695c63676d19cfce3962fb8a780d9c5fee0bdae559272fe4` |
| `.pytest-tmp/sequence-assembly-action-qa-v1.launch.json` | `07c41d73df504a3064b2fc519f75cdc45cadfb5dd069e002a626c967afcc86b3` |
| `.pytest-tmp/sequence-assembly-action-qa-v1.junit.xml` | `538f626a36954fdb2cd21b340abbf19235da453c9f151b56de8cc6309750124c` |

The three launches verify six distinct ordinary nodes. Their aggregate elapsed
time is 25.5325368 seconds; no overall throughput comparison is implied. The
independent action circuit's four offered labels are additional to the seven
combined-fixture labels described above.
