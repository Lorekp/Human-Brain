# Competitive comparison roster data QA

Root selection, 2026-09-06, after source review. Commit source and this contract,
run the selected checkout's doctor, then run only
`tests/test_byte_sequence_decoder.py::test_competitive_splice_roster_data_contract`.
Use one fresh 60-second / 2,048-MiB watchdog with
`.pytest-tmp/neuronal-address-competition-data-qa-v1` basetemp/log prefix/JUnit.
This is pure authored data QA, separate from the
[single worker exchange](neuronal-address-competition-worker-qa-v1.md).

Call v2 `build_roster` exactly once. Read the authenticated saved v1 fixture
and original v6 roster; never invoke the v1 fresh builder. Source review proves
the original TRAIN loop is unchanged after namespace qualification. Compare
all 32 TRAIN records/packages, the fixed cyclic schedule, retention IDs, twenty
old-panel rows and twelve old packages exactly against the authenticated fixture.

The one constructor creates 64 task records and 72 memory packages, including
the selected 32 new structural tasks in sixteen pairs and 28 new packages.
Verify 48 before-learning and 24 after-learning packages; 168 documents,
4,022 raw bytes and 168 EOS symbols per arm. No ingestion or model occurs.
Keep all four document-order, continuation, value-role and operation/layout
pairs, including six-byte continuations, whitespace, four delayed states and
125-slot interference. Confirm fresh keys/module paths are disjoint from the
exposed cohort except the shared generic `main.py` entry point.

Check all 64 canonical actions with independent raw slicing/workspace hashes.
Parse all three before/after modules: 384 AST parses. Inspect 256 assignment
literals plus 64 separator literals as data, validate the print AST's exact
attribute arguments, and assemble the declared stdout from these literals.
No program or child is executed. Check 64 observation exports for exact raw
field exclusion and copy isolation. Independent checks confirm all sixteen
pair contrasts, memory schedule, geometry and full five-arm exposure totals.

Publish the generated JSON exclusively immediately after construction at
`.pytest-tmp/neuronal-splice-roster-qa-generated-v2.json`, followed by
`.pytest-tmp/neuronal-splice-roster-data-qa-v2-raw.json` binding its hash and
source/data inputs before semantic assertions. Publish the separate
`.pytest-tmp/neuronal-splice-roster-data-qa-v2.json` passed receipt only after
checks. Preserve any failure; no automatic regeneration or retry is selected.
Bank source/test hashes, doctor, monitor and JUnit after execution.

Zero models, checkpoint loads, neuronal traversals, teacher calls, workers,
actions or endpoint predictions are selected. A data pass freezes prospective
task bytes; it is not evidence of learner success and does not admit acquisition.

## Preserved assertion failure and reviewed correction

The first launch at `0a9bac538cac2d4c992b491a5149ba8a04976e56` fails at the
constructor's final count assertion. Actual counts are 64 tasks, 72 packages,
48/24 stage split, 168 documents, 4,022 raw bytes, maximum 125 slots and
480/640 teaching offers. The expected maximum incorrectly retained 128 from
excluded splice-v1 fresh interference packages. The saved-only review finds
retained TRAIN maximum 52, old-panel maximum 125, new document-order maximum
50, continuation maximum 54, value-role maximum 125 and operation maximum 50.
The 128-slot tape capacity and maximum-geometry engineering cap are unchanged.

Preserve `.pytest-tmp/neuronal-address-competition-data-qa-v1` monitor, stdout,
stderr and JUnit. This spends one builder call, zero models/actions/teachers;
the constructor did not return, so no generated roster or pending/passed JSON
was published. The launch takes 4.4467717 seconds and 267.125 MiB observed peak.
Root selects the scalar assertion correction only, with unchanged task bytes,
and one additional data-only builder call after a new commit and doctor. Reuse
the never-created output JSON names; use a fresh
`.pytest-tmp/neuronal-address-competition-data-qa-v2` watchdog/basetemp/JUnit.
All other exposure and resource limits above remain fixed. The independently
passed worker check is unaffected and must not be rerun.
