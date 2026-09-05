# Reporting-only recovery of the protected comparison

Recorded 2026-09-05 before replay. The original
[protocol](contextual_protected_contrast_protocol.md) and scientific choices
remain unchanged.

Source `2c44b4e6b58bf47c8ae290183f76ad16f4a6e115` completed model execution,
including teaching, transfer, retention and examination, but exited 1 when
serializing its final report: `TypeError: Object of type int64 is not JSON
serializable`. The report was not written. The final NPZ and raw log are
preserved byte-for-byte with an
[interruption record](../../benchmarks/contextual_protected_contrast_v1_interruption.json)
in commit `cdd0dbfa607bbd94a3bdc144b4e8c33c76ce2860`.

A reduced source-function reproduction identifies NumPy boolean values in
`projection_exhausted` and `projection_stalled`; summing them produces a NumPy
integer in the teaching summary. Both the event fields and the summary need
ordinary JSON scalar types. The reporting-boundary correction converts precisely
those two diagnostic fields to Python booleans for updating and frozen receipts.
It changes no projection arithmetic, learning, checkpoint identity, candidate,
feedback schedule, readout, action, exam rubric or numerical tolerance.

After the failing synthetic receipt/summary serialization check passes with this
correction, commit corrected source before one faithful replay. Reuse the same
cohort and contract solely to recover the missing trace; this is not a new
independent fresh result or an opportunity for tuning. Preserve the first attempt
and disclose both executions and the replay's measured cost. The original full
in-memory receipt trace cannot be recovered and must not be invented.

Require all 142 preserved teaching/evaluation/exam log events to match the replay
after excluding timing fields, and every final NPZ array to match in dtype,
shape and bytes, including protected basis, deltas and certification records.
Audit the recovered full receipt trace, source/native/encoder hashes, allocation,
frozen controls, numerical updates and executed outcomes independently. A mismatch
must be reported and investigated; it cannot be silently treated as equivalent
evidence. The original two-hour implementation/diagnosis budget still applies.
