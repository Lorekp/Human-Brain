# Sequence assembly integration verification

Status: component mechanics verified; reader/action integration is still under
source review. No broader v4 workload has run. These are ordinary affected QA
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

Reader/action integration will add its own source revision, ordinary QA receipt
and exposure accounting here. Broader acquisition, fresh transfer, retention and
teacher-free recovery must be measured separately on the selected continuation.
