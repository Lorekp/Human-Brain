# Stable credit V1 saved-audit correction

2026-09-09. Both workers and all eight scheduled batteries completed at frozen
runtime `5622f600921e6811d66254e25c8e956a31cd19ab`. The first saved audit failed
with `KeyError: 'grade'` after 18.7546711 seconds. The original outer launch
remains **failed**, at 6472.8048409 seconds and 1453.3203125 MiB observed process
tree peak. No learner restart, checkpoint rewrite or additional teaching follows.

The main worker intentionally writes an execution `grade` object only for edits.
Reading correctness instead uses accepted answers and the saved `correct` field.
The auditor independently checked that reading outcome, then unconditionally
requested `grade` while constructing its compact behavior signature. This is
an auditor field-contract error, not missing reading evidence.

A saved-only direct worker review reproduced the same exception at the signature
comprehension in 18.6984207 seconds, with 55.921875 MiB observed peak. A focused
regression in the existing `test_byte_sequence_decoder.py` reproduces it with
the actual reading-row field contract. Its initial run failed in 7.0748090 outer
seconds; the corrected run passed in 2.4862426 seconds, at 231.37890625 MiB.
The test checks reading signatures, changed answer and edit-grade sensitivity,
and strict rejection of an edit without its required grade. It creates no model
and runs no task. Pytest's cache-directory ACL warning remains disclosed.

The correction gives readings an explicit null grade in the audit signature and
keeps strict edit-grade access. A keyword-only original-auditor source argument
allows a separately declared post-run audit to authenticate exactly that one
frozen source file. All other execution sources must still match their frozen
hashes. An auditor-local inspector subclass redirects only the canonical auditor
file read to those same authenticated original bytes; unchanged store binding,
fresh payload, journal and frontier checks still execute. Saved manifests,
configuration, inventories, source maps and runtime evidence are never edited.
The corrected result records original and current auditor hashes separately.

The original source SHA256 is
`2f53d1edea559fee230205d9e107fe4df54e9366d793a9781009d2d0db864c0c`.
Its Git blob has LF newlines; the measured checkout had a CRLF final newline
and LF elsewhere. Preserving that exact checkout representation reproduces the
frozen hash. Two preparation assertions rejected raw LF and all-CRLF candidates
before writing anything (0.3437375 and 0.3355008 seconds). Exact preparation
passed in 0.3563742 seconds. No memory sample was captured in these brief runs;
the monitor's zero is not a measured zero footprint.

All receipts remain under `.pytest-tmp/stable-credit-v1-audit-*` and
`.pytest-tmp/stable-credit-v1-original-auditor-preparation*`; the exact source
copy is `.pytest-tmp/stable-credit-v1-original-auditor.py`. Doctor receipts are
`.pytest-tmp/dev/doctor-rx82ekgt/receipt.json` and
`.pytest-tmp/dev/doctor-slhhx78v/receipt.json`.

The corrected saved-only review has a separate 600-second review ceiling under
a 650-second / 2048-MiB outer monitor, allowing publication and shutdown. It must
write a new result and retain the first failed audit. Its actual costs belong
outside the original frozen launch. No corrected-audit pass is claimed here
before that review completes. The prevention is to exercise both real reading
and edit row contracts when checking the auditor, rather than relying on source
review alone.

The first corrected saved review passed the reading-row point but found a
second audit assumption: it compared each phase's `work_delta` with the change
since the previous telemetry row. Actual deltas begin at a later phase-start
snapshot; owned-journal append and scope entry/exit work occurs between phases.
The failed corrected review remains `review-corrected.json` (86.3544813 review
seconds; 88.0430764 outer seconds, 101.90625 MiB). This does not alter the first
failure or any worker outcome.

An authenticated 5,573,740-byte checkpoint-journal diagnosis took 0.3661166 outer
seconds. Its intervening counters matched the source's native writer operations.
The next audit correction reconciles exact expected gap counters from validated
appended-row byte lengths: ownership entry/exit, append writes, real tail reads
and their 1-MiB buffer allocation. Unknown counters, non-journal gaps, missing
expected work, changed counter sets and negative/oversized deltas remain errors.
Phase-start snapshots were not saved, so inferred baselines do not independently
establish timing; final cleanup remains a separate final-store counter scope.

The phase regression failed before the fix (6.9494454 outer seconds) and both
reader/phase regressions then passed together (5.5423658 seconds, 273.70703125 MiB).
Doctor: `.pytest-tmp/dev/doctor-annogmcg/receipt.json`. The final source adds
coverage of two co-published rows crossing the 1-MiB read boundary. The final
saved-only result will use a new `review-corrected-phase.json`; earlier audit
files remain unchanged, and all additional reporting costs are separate.

Final focused QA passes both tests, including the large-row, begin-scope and
unknown-counter cases: 6.0334261 outer seconds, 265.734375 MiB. Doctor receipt:
`.pytest-tmp/dev/doctor-84xv1dpm/receipt.json`. Ruff passes for the corrected
auditor and test file. This is saved-reader QA, not a learner capability result.

That full review validates the old main worker, then exposes a missing required
`partial=False` argument at the probe journal reader. Its failed result remains
`review-corrected-phase.json` (95.5507661 internal seconds; 97.3054548 outer
seconds, 101.5390625 MiB). The corrected call explicitly rejects incomplete
probe journals. A call-signature regression fails before correction (8.0317569
outer seconds) and the final three audit regressions pass together (3.1305288
seconds, 235.921875 MiB; doctor `doctor-4ibwmwzr`).

A direct saved-probe check using previously validated old-worker metadata passes
all four batteries in 32.5895968 outer seconds, 80.08203125 MiB. It constructs no
model, decodes no tensors and executes no tasks. Only JSON integer keys are
restored to their in-process representation for this diagnostic call. The full
final audit still revalidates every worker and does not reuse this shortcut.
Its new output is `review-final.json`; all preceding audit attempts remain
separate, immutable evidence of reporting-tool failures and their actual costs.
