# Affine comparison: exam harness correction

The attempt from `3c7c474` stopped with a verified checkpoint error after the
BINDING action evaluation. Preserve
[`contextual_affine_readout_v1.json`](../../benchmarks/contextual_affine_readout_v1.json)
unchanged (SHA-256 `23e57d431ca89f0e40496f5e637838ab5c36fbeff37f63bbbe2610917df6ef06`).
It matched all 416 archived decisions and completed 104 BINDING executions.
Affine selection agreed with all 26 raw occurrences and recovered 4/8 fresh and
2/16 old edits. GAIN16 action evaluation and the full exam comparison are incomplete.

The harness incorrectly assumed the shared exam helper could be called twice
without changing storage. That helper ingests its passage on every call. The
second invocation therefore changed the checkpoint, which the existing assertion
correctly rejected. This is a harness error, not a readout failure or a reason to
weaken the checkpoint assertion.

Correct the harness to evaluate old and new policies in a single helper call,
ingesting the original passage once per model. Compare the old result subset
with the archive and require the original post-exam checkpoints after all
inference. Preserve all questions, marking rules and pending human marks.

Repeat the fixed saved-weight comparison into `contextual_affine_readout_v2.json`
after committing this correction. The core map, weights, sources, motor prior,
requests and scientific criteria remain those of the
[original protocol](contextual_affine_readout_protocol.md). No tuning or training.
Verify repeated BINDING decisions/outcomes against the interrupted artifact;
this is completion/reproduction, not an independent replicate. Report both
attempts and their costs. Keep the interrupted artifact and original protocol
immutable, and retain the unchanged checkpoint checks.
