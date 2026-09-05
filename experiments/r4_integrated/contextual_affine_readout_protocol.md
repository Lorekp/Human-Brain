# Affine current map at saved learned weights

Preregistered 2026-09-05. One public-development readout comparison, with no new
evidence training or evaluation cases. Follow the
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md).

## Failure, hypothesis and fixed change

The exploration run from `6120335`, recorded at `6c64189`, gives BINDING unique
raw leaders on all 26 transfer/retention requests. Their drives 26.697–58.794
saturate `600+200*tanh(s)` to identical 800 pA currents. Its raw reference executes
4/8 fresh and 2/16 old edits, while mapped/discrete/interpolated policies STOP
throughout. Test whether preserving those distinctions before the existing
interpolated GLIF race recovers their actual neuronal executions.

Use exactly one new opt-in map, with all candidates participating:

```text
tau = atanh((600.1 - 600) / 200)
d = raw_drive - tau
R = max(1, max(abs(d)))
candidate_current = 600.1 + 200 * (d / R)
STOP_current = 600.1
```

For an empty candidate set, only STOP participates. Divide before multiplying
to avoid intermediate overflow. Reject nonfinite drives explicitly. This common
positive affine map preserves mathematical ordering against the existing raw
STOP boundary and bounds currents to 400.1–800.1 pA. It uses no labels, host winner
selection, clipping or score-dependent tie breaking. Floating-point aliases and
true ties remain STOPs; do not claim universal finite-precision injectivity.

Feed these currents to the existing linear endpoint-interpolated GLIF race with
its unchanged adjacent-float64 tie rule, step size and horizon. Preserve old
methods, defaults, checkpoint identities and decision digests. Bind the new
formula, normalizer and interpolation policy to the new decision's readout
identity. No feature, encoder, native learning, exploration or motor-rule change.

## Exact reconstruction and comparison

Use `benchmarks/contextual_exploration_python_development_v1.json` (SHA-256
`df87159c490975be010e2630ec1716d03e3d177d4ef8fefc5445cb1846d790cf`)
and its NPZ (`1464658db1c133af5089c03e2d5e8b16102bccdcab27e2f0fd5f416908952188`).
Reconstruct the original ordered retained sources, lineage/counters and saved
numerical synapses for GAIN16/BINDING and their frozen mates. Do not replay
evidence learning. Verify reconstructed checkpoint, storage and synapse hashes,
encoder identity, original artifact integrity and loaded checkout-native DLL.

First reproduce all 416 archived ordinary transfer/retention decisions, including
the original seven full policies and two BINDING ablations. Compare decision
custody and emitted occurrences, not just aggregate scores. Reuse the original
motor or reconstruct its two declared demonstrations and require the identical
motor checkpoint; this restores existing motor experience, not new learning
evidence. Count reconstruction costs separately within total runtime.

Then execute the new and existing interpolated neuronal policies on the same
sixteen old and ten transfer requests, for each updater and frozen mate, through
the same actual Python edit and execution path. No retries, teachers, new cases
or feedback. All inputs are
openly reused development. Preserve the original source-absent request condition
while disclosing retained raw documents and vectors. Keep original KS2 Q6
questions and exposure unchanged; reproduce old answers at their original
post-exam storage, then record the new readout without changing or automating
pending official-rubric human marks.

Record old-policy parity, raw-to-new neuronal occurrence agreement, solved edits,
both-property entities, unsupported actions/STOPs, decoy integrity, timing and
tie metadata, immutable checkpoint checks and total resource costs. Frozen
controls must remain frozen. References still share the learned motor; this
comparison isolates evidence selection/readout, not the entire neuronal core.
Freeze reviewed source hashes in a source commit before the run. New result
artifacts must refuse replacement of any existing evidence.

Retain the reconstruction helper's final certified-pair margins and original/
last post-feedback gaps as a retrospective diagnostic of the previous TRAIN
stream. These use only archived eligibilities and saved weights; they do not
select exposures, change the readout, or constitute new evaluation evidence.

## Success, limits and budget

Readout success requires reproducing original decisions, recovering every
representably distinct raw winner on this cohort through the neuronal race,
and matching its executed outcome while preserving ties and checkpoint state.
Expected recoverable BINDING raw successes are 4/8 fresh and 2/16 retained;
recovering them is not improved ranking or fresh generalization. Exposing raw
unsupported false positives is a measured consequence, not a new learned
abstention failure. The broader targets remain 6/8 fresh, 12/16 retained and
both properties for three fresh entities; this readout alone cannot meet them.

Focused QA covers empty/zero states, candidate and STOP ties, permutation,
large finite drives, nonfinite rejection, preservation of ordinary policies,
custody and read-only state. Budget one map, one saved-checkpoint comparison,
and two hours implementation/diagnosis. No scale search, extra training, feature
ablation selection or new protected evaluation. Afterward return to the measured
entity/property ranking and retention failures in the integrated workload.
