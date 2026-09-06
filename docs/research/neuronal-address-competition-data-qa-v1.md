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
