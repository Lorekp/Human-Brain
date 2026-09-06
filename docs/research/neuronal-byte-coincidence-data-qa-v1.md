# Coincidence comparison: one pure-data roster check

Completed ordinary QA scope, 2026-09-06. The node below passed once after source
commit and the selected checkout's clean doctor. The frozen selection was
`tests/test_byte_sequence_decoder.py::test_coincidence_splice_roster_data_contract`.
No collection, roster construction or execution occurred while authoring it.
Preserve all 196 existing top-level AST nodes and their exact source bytes.

Use a fresh 60-second / 2,048-MiB watchdog with
`.pytest-tmp/neuronal-byte-coincidence-data-qa-v1` basetemp/log/JUnit prefix.
The node calls v3 `build_roster` exactly once, reading the authenticated original
v6 old-panel JSON and the already saved v2 generated roster. It invokes no
historical builder. The original data helper and final v3 source hashes are
pinned before and after; the tested file hash is saved with the raw receipt.
The saved v2 roster hash is
`1f76de378e238ddebd6ebcc8dd62f9dffc3187cada1923e794f86e51dfb4440e`.

The one constructor returns 64 task records and 72 literal memory packages.
Compare all 32 TRAIN tasks/packages, 256-entry cyclic schedule, cost IDs,
retention IDs, twenty old-panel records and twelve old packages exactly to the
authenticated v2 fixture. Check all sixteen fresh D/C/V/O pairs, new fixed-width
z00..z19 identities, fresh module paths, selected literal values, whitespace,
insertion/deletion offsets and exact SET/KEEP request changes independently of
the new builder helpers. Check 48 before-learning and 24 after-learning packages,
four delayed fresh states, 28 fresh packages and maximum 125 slots.

Independent raw arithmetic is 168 documents, 4,022 raw bytes and 168 EOS markers
per arm: TRAIN 64/1,512, fresh D16/384, C16/400, V28/672, O8/190, plus unchanged
old-panel 36/864. No memory ingestion occurs. The five-arm scope remains 1,520
eventual full actions, 200 reader-panel calls, 2,400 reader offers and 3,200 action
labels; these are declared data counts, not activity performed by this check.

Check 64 canonical raw edits with independent byte slicing and workspace hashes.
Parse three modules before and after each task: **384 AST parses**. Read the
four resulting assignment literals per task with `ast.literal_eval`:
**256 literal reads**. Inspect the 64 separator AST constants directly, verify
exact print attribute order, and assemble expected stdout from literal values.
No compile/exec/eval of task programs and no child process occurs. Check all 64
raw observation exports for exact field allowlists and copy isolation.

Before semantic assertions, publish the returned roster exclusively at
`.pytest-tmp/neuronal-splice-roster-qa-generated-v3.json`, then its hash/input
bindings at `.pytest-tmp/neuronal-splice-roster-data-qa-v3-raw.json`.
Only a complete pass writes the separate
`.pytest-tmp/neuronal-splice-roster-data-qa-v3.json`.
All three paths must be absent before the sole constructor call. Preserve a
failed constructor or assertion receipt; no retry or regeneration is selected.

Exposure: one v3 roster construction, zero historical builders, zero models,
checkpoints/deserializations, neuronal calls, teacher offers, workers, actions,
program executions or endpoint predictions. Importing the existing test file
loads its existing library definitions but constructs no learner for this node.
Root banks source/doctor/JUnit/watchdog evidence after the separately admitted
ordinary check. A pass freezes task data; it does not establish capability or
admit the integrated comparison.

The node and independent saved review now pass at source `a925b97`, after clean
doctor `doctor-qesp0g8h`. Outer duration is 6.320887 seconds and observed owned
peak is 231.00390625 MiB. The [data QA bank](../../benchmarks/neuronal_byte_coincidence_v1.data_qa.json)
has SHA256 `7df5ed5c7b694566984927abcdaa81aa868f6805a0bd2ca7e45ba2b38532ebab`.
The sole generated roster has SHA256
`9b8f03657c1604b7e8e286e12b7534db750a5ad29ab9c872f3481d58bf2aabd5`.
The independent review preserves its corrected review-only O-layout assertion;
neither data construction nor QA was repeated. This node is closed.
