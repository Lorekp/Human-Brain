# The current TRAIN representation admits the required ranking

Completed 2026-09-05 from frozen source `cfd0950b3464743d0ca7a70e1051f1d4ac70d8c9`,
following the [protocol](contextual_representation_feasibility_protocol.md).
The [single-run artifact](../../benchmarks/contextual_representation_feasibility_v1.json)
and [feature/witness archive](../../benchmarks/contextual_representation_feasibility_v1.npz)
preserve a verified host-linear solution for all sixteen taught questions.

All 4,544 teacher-versus-wrong-candidate constraints and sixteen STOP-clearance
constraints pass. Every teacher is the unique raw-score leader among all 285
TRAIN candidates. The witness fixes the first 384 query coordinates and bias to
zero; the remaining 1,237 coordinates have unrestricted signs and magnitude.

| Verification measurement | Value |
| --- | ---: |
| Required ranking and STOP gap | 1, allowance 1e-7 |
| Smallest independently recomputed gap | 0.9999999999972142 |
| Largest difference from runner scores | 1.29e-13 |
| Witness weight-difference L2 norm | 604.51 |
| Largest absolute witness coordinate | 159.64 |
| Raw score range | -10.368 to 6.646 |

Root recomputed all scores with scalar products accumulated by `math.fsum`,
without calling the runner's verifier or solver. This checked every inequality,
teacher identity and source byte span, alongside source-commit bytes, encoder
files, native library and archive hashes. The runner separately verifies original
feature scores and LP-matrix residuals. Both methods pass the frozen tolerance.

The learner remains unchanged: zero feedback, zero native updates, zero plastic
weights and identical before/after checkpoints. The diagnostic opens only the
TRAIN projection: four documents and sixteen existing teaching query/value
records. It does not read the mixed original fixture, transfer or exam labels.
No witness weights enter the learner, no edit executes, and no capability gate
passes. Synthetic feasible/infeasible and invalid-witness checks pass, including
a negative free coordinate; CLI error preservation and nonzero exit are verified.

SciPy 1.17.1 HiGHS terminates successfully after 2,187 iterations. The complete
diagnostic takes 28.95 seconds: 3.12 for setup/feature extraction, 24.64 for the
solve and 0.008 for runner verification. The independent audit takes 1.31 seconds.
Parent peak working set is 1,124.15 MiB, final private memory 1,023.51 MiB and
traced Python peak 391.17 MiB. This is the cost of an external capacity reference,
not a learner speed or scaling result.

The archived feature tensor occupies 59,134,080 uncompressed bytes; the compressed
NPZ is 28,865,660 bytes, including the 12,968-byte witness and verification arrays.
The frozen learner retains four documents, 545 raw bytes, 285 candidates and
946,027 counted array/byte payload bytes, excluding Python overhead. Encoder,
feature tensor, solver memory and learner payload are separate costs.

Artifact SHA-256: `cbffd67f2989b1755773e133ad14a3fbd9868dd1fbfce18337ef495fb0f6a97c`.
NPZ SHA-256: `6da3c2257ea46ba29357523cbcc193c72311b1052ba9b8ac7b38735cc9eee4c0`.

This rules out impossibility of the declared joint host-linear ranking on this
finite TRAIN roster. It does not establish native-update reachability, convergence
under the available feedback budget, stable neuronal readout at arbitrary weight
scales, or generalization. The known feedback-coverage and interference failures
remain. Keep the features and fixed update rule, and return to the integrated
comparison that reserves feedback opportunities for later actual errors. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md) owns
that work; the oracle witness will not be deployed or evaluated further.
