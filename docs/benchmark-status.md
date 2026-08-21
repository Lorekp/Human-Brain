# Benchmark status

Status date: 2026-08-21

## What has passed

The human-cell model ran `sqlfluff__sqlfluff-1625` from the SWE-bench Lite dev
split through mini-swe-agent 2.4.6 and the official cached Docker image
`swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-1625:latest`.

- Growth: 7 public teacher interactions, 887 token cells, 335,713 dendritic
  branches, 1.37 seconds.
- Frozen recall gate: 7/7 exact shell actions.
- Agent run: submitted in 7 actions; the first and last cell response timestamps
  are 2.98 seconds apart (repository commands run between responses).
- Official SWE-bench 5.0.2 evaluation: 1 submitted, 1 completed, 1 resolved, no
  infrastructure or ambiguous failure.
- Project tests after native integration: 46 passed.

## Active-cell scaling and dynamics

The first registered scaling grid varied active cells (24, 48, 96, 192),
training characters (3k, 6k, 12k, 24k), and three fixed seeds. At 24k
characters, teacher-forced medians rose monotonically from 15.23% to 22.27%,
34.11%, and 38.28%. Dormant controls with 24 active cells and up to 192 total
cells exactly reproduced the compact 24-cell circuit while taking longer.

State diagnostics then exposed the confound: the original network had roughly
one effective state dimension irrespective of cell count. A train-only
calibration tested 11 dynamical regimes without using validation predictions.
The selected sparse/slow/short-trace regime raised median effective rank at 96
cells from 1.19 to 20.84 while lowering firing from 107.1 Hz to 12.4 Hz and
absolute cell correlation from 0.833 to 0.133.

With that regime frozen, proportional cell/data scaling produced:

| Active cells | Training chars | Teacher-forced | Free-running | Effective rank |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 3,000 | 43.10% | 6.25% | 7.57 |
| 48 | 6,000 | 47.53% | 6.77% | 11.05 |
| 96 | 12,000 | 47.27% | 10.16% | 18.03 |
| 192 | 24,000 | 48.83% | 12.76% | 21.15 |
| 384 | 48,000 | 49.61% | 12.76% | 23.85 |

This supports a useful-cell scaling effect over the tested diagnostic range,
not a universal power law. Rank growth is strongly sublinear and free-running
accuracy plateaus from 192 to 384 cells, identifying redundancy/readout
learning as the next bottleneck.

## Grouping and spatial topology

Topology v1 used normalized recurrent current but retained the synchronized
dynamics; every architecture remained near rank 1, so it is recorded only as a
diagnostic failure. Calibrated v2 degree-matched all variants and used the same
three seeds. At 96 cells, a six-layer sheet met the preregistered primary
criterion (47.27% teacher-forced, 10.68% free-running versus 47.27%/10.16% for
uniform). At 192 cells, none of the registered single-scale, nested, 2D-sheet,
or mixed-membership variants beat uniform. Nested groups had slightly higher
rank (22.02 versus 21.15) but materially lower teacher-forced accuracy (45.44%
versus 48.83%). Larger, locally learned networks are required before making a
scale-level grouping claim.

## Native simulation scaling

The complete recurrent token interval now executes in a Rust native kernel.
It is bit-exact against the former Python/NumPy loop over 64 tokens. The frozen
benchmark reports a 6.78x speedup at 1,536 cells and nearly linear scaling:

| Cells | Native tokens/s | Cell-steps/s | Persistent arrays |
| ---: | ---: | ---: | ---: |
| 1,536 | 1,045.45 | 160.6M | 3.34 MiB |
| 6,144 | 301.70 | 185.4M | 13.22 MiB |
| 24,576 | 82.49 | 202.7M | 52.73 MiB |
| 98,304 | 20.05 | 197.1M | 210.79 MiB |

The empirical time exponent is 0.949 and persistent-memory exponent is 0.997.
This removes Python simulation and inference allocation as the immediate scale
limit. Exact ridge learning remains quadratic in memory and cubic at solve.

## Scalable local learning

An experimental native competitive three-factor rule now provides online
synapse-local readout updates without ridge's normal matrix, target moments,
batch workspace, or dense solve. The learning rate was selected on a named
training-only calibration file and frozen before the named validation file was
opened; neither artifact defines or accesses a test split.

The predictive comparison reaches 6,144 active cells with 13.92 MiB traced
peak memory. Teacher-forced accuracy rises from 3.91% at 96 cells to 10.16% at
6,144, but local learning trails ridge substantially at 96 and 384 cells and
free-running accuracy remains weak. This establishes linear-memory learning
and a positive size trend, not ridge parity or useful generation.

A separate dataset-free 32-example smoke reaches 98,304 active cells at 9.34
updates/s with 211.70 MiB peak memory and no ridge arrays, avoiding a
theoretical 72.00-GiB normal matrix. At that size, topology construction takes
31.26 seconds while growth takes 3.43 seconds, making construction a measured
initialization bottleneck. See `docs/local-readout-learning.md` for the rule,
locality boundary, full comparison, and claim limits.

The accepted patch changes only L031's emitted description, matching the public
upstream resolution in [SQLFluff PR 1625](https://github.com/sqlfluff/sqlfluff/pull/1625/files).
The official evaluator applies the patch in Docker and grades FAIL_TO_PASS and
PASS_TO_PASS tests as described by the
[SWE-bench evaluation guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md).

## Held-out cell capability

The code-language v3 gate was registered before test access. It partitions 34
pre-existing `src/deepsim` Python modules by a SHA-256 hash of each relative
path: 25 train, 6 validation, and 3 test. Growth sees at most 4,096 characters
from each training file. The frozen model then predicts up to 1,500 bytes per
held-out file with teacher forcing and no test-time growth.

On validation, Witten-Bell interpolation of lower-order dendritic branches
improved the cell circuit from 7,034 to 7,045 correct predictions. Increasing
synaptic probability gain raised measured cell inference from 638 to 987
tokens/s. After those choices and their hashes were frozen, the test split was
opened once:

| Split | Predictions | Human cells | Bigram | Same-order Witten-Bell |
| --- | ---: | ---: | ---: | ---: |
| Validation (6 files) | 8,773 | 80.30% | 32.09% | 80.38% |
| Test (3 files) | 4,500 | 86.42% | 29.93% | 86.40% |

The test result passes on every file and is one prediction above the
conventional same-order Witten-Bell reference. The reference is evaluation-only;
token selection still comes from the first fitted human GLIF3 cell to spike.
The interpolation follows the recursive formulation in
[Chen and Goodman](https://u.cs.biu.ac.il/~yogo/courses/mt2014/papers/chen-goodman-99.pdf).

After the v3 test was consumed, v4 locked that test split and allowed only an
output-equivalent speed change on validation. The GLIF3 membrane and decaying
NMDA current are linear before the first spike, so their exact forward-Euler
unit response is now compiled once and scaled per active synaptic current. All
7,045 validation predictions and all six mean spike times are unchanged.
Measured cell inference increased from 987 to 5,358 tokens/s (5.4x); compared
with the original longest-branch circuit at 638 tokens/s, the cumulative gain
is 8.4x. The v3 test was not run again.

A separately pre-registered domain-shift cohort was then opened once, with no
additional model change: `argparse.py`, `email/parser.py`, `json/decoder.py`,
`pathlib.py`, and `urllib/parse.py` from Python 3.11.15. The model still grew
only on the 25 DeepSim training files. It predicted 2,848/7,500 bytes (37.97%),
versus 17.60% unigram, 21.37% bigram, 37.57% longest backoff, and 37.85%
Witten-Bell. Every external file beat unigram and bigram. The lower absolute
accuracy measures the expected cross-codebase domain shift rather than hiding
it with a selected in-repository fold.

## Free-running development gate

Teacher forcing is not autonomous generation. A validation-only gate therefore
uses 512 held-out prompt characters and asks the cells to emit the following
256 tokens without correction. Across six files, the current circuit predicts
269/1,536 aligned bytes (17.51%), exactly tied with the train-only unigram, and
keeps an exact prefix for 10.2 bytes on average. Several continuations collapse
to repeated spaces. This gate currently fails and is the next capability target.

One biologically literal attempt carried the fitted GLIF3 after-spike currents
and refractory cut across generated tokens. The specimen's short positive
after-current encouraged bursting; accuracy fell to 14.39% and whitespace
collapse remained. That implementation was rejected and removed rather than
retained as an ornamental biological feature.

## Recurrent developmental-growth gate

The obsolete v1 post-hoc readout-pruning benchmark has been removed. Its
replacement changes the recurrent circuit during training: dormant fitted
GLIF5 cells are recruited at 25%, 50%, and 100% of final size; recurrent edges
overgrow, collect local coactivity evidence, turn over, and mature to the same
final fanout. Layered versus random birth order, activity versus random
selection, homeostasis versus no homeostasis, and a full-size-from-start control
are matched for training bytes and attempted updates.

The v2 protocol freezes 384- and 1,536-cell runs over seeds 17, 29, and 43
before opening the previously unused `recurrent_artifacts.py` validation file;
it explicitly has no test access. Its numeric primary criterion requires the
staged condition to stay within one percentage point of full-start median
teacher-forced accuracy, not lose median free-running accuracy, and record at
least two of three strict paired-seed wins on either endpoint. It also records
state-rank/correlation/firing, stage structure, runtime, and memory.

The numeric primary failed without retuning. At 384 cells, staged versus
full-start medians were 4.69% versus 4.69% teacher forced and 4.69% versus
6.25% free running, with no strict paired-seed win. At 1,536 cells, staged was
4.69% versus 5.47% teacher forced and tied 4.69% free running; only one of three
teacher pairs won. The predeclared unchanged 6,144-cell diagnostic was 6.25%
versus 8.59% teacher forced and tied 4.69% free running, again with no win.

The dynamics tell a narrower positive story. Staged effective rank increased
from 2.425 to 4.055 to 5.414 over 384, 1,536, and 6,144 cells while sampled
correlation and firing generally fell. Random pruning collapsed rank near one
and caused 57-81 Hz firing; activity selection and homeostasis therefore help
circuit health, but not enough to improve prediction. Full-start teacher
accuracy reaching 8.59% at 6,144 cells also contradicts a blanket claim that
larger networks are worse. The present failure is staged learning exposure and
architecture, not a negative law of scale.

## Reproducibility artifacts

Hashes below identify the canonical LF-normalized file bytes stored in Git,
not a platform-specific checkout's newline representation.

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/swe_sqlfluff_1625_cells.npz` | `bf54b181ce24ff636d84a3e9a712eaf96caceb1ae343299169727f884f2dda91` |
| `artifacts/swe-dev-expert/sqlfluff__sqlfluff-1625/teacher.traj.json` | `cae6a4dbe8750a8bfe83db1e04103555a4f657ae6ac960cda1f814e761a34a4f` |
| `artifacts/swe-dev-cell/sqlfluff__sqlfluff-1625/dendritic-v4.traj.json` | `8d7b7ba5b2911573505110f61cc4a457c9b595abb895ac8b8d21027b225e5cd2` |
| `artifacts/swe-eval-cell/human-cell-dendritic-episodic-v2.human-cell-sqlfluff-1625-v3.json` | `6a8cfc794732220a26fa9f2f27b9575c36dbfe5ce1580b1000bbdeed430f53b8` |
| `artifacts/code_language_validation_v3_witten_bell_cells_gain16.json` | `a27f995af6d72f9f5c7c95f217013a156f09f80593004f62201cced6343796bc` |
| `artifacts/code_language_test_v3.json` | `adf25d727acfed0ca61754c372a5cdda39560daf009adc1b0524efdfa0f682bb` |
| `artifacts/code_language_freeze_v3.json` | `39f62eaac24695bdc52bc2682f75e8b598720102e8f8995a955c8b980c9b1834` |
| `benchmarks/code_language_protocol_v3.json` | `54f07ff3dd4abb141ed4ce6226e0f30ce27ea98faafe7fea7075df2791606d60` |
| `artifacts/code_language_validation_v4.json` | `e8222472ba1cb1f9855e409c0f7a15e77a047725f8c190992f3c757f5ca62e4d` |
| `benchmarks/code_language_protocol_v4.json` | `9b5543e8d9b201c744a3606c38de7bf32df7f191720c9b307e59186d6affa891` |
| `artifacts/external_code_test_v1.json` | `b758421f663c48bc214f633d943237f648ff5d173896ba37c9619d034e2da9d6` |
| `benchmarks/external_code_protocol_v1.json` | `63069db2fd8d6a3ea3b1cfd211785686cb87e60248fdf4ed95fadab5bf9d9f11` |
| `artifacts/generation_validation_v1.json` | `555d77f0f41ab874390aa26c8f0c1dba1b62b2fb6bc3eb54e9ac59e14d126654` |
| `artifacts/generation_validation_v1_persistent_asc.json` | `5c6828be868ad95d45c7a6be7e358f57433b819691431887f6f3ee6f579e20cf` |
| `benchmarks/generation_protocol_v1.json` | `254754c7464a3b2727aadd51de3f72c597958ff487ebd74aed06ee3ce20d8388` |
| `artifacts/cell_scaling_validation_v1.json` | `dd045ef19bc7069d8624ddf5bf56e9e0b4195f30d423f0984320ec0798ee489d` |
| `benchmarks/cell_scaling_protocol_v1.json` | `40a24b23334953209182727f108cceb0dd9a3b7c4c43639c228aeedbe6403706` |
| `artifacts/dynamics_calibration_train_v1.json` | `ca30de2816888cdf23bef612571ebf4369dab460f205353f874f1b6052e9db7f` |
| `benchmarks/dynamics_calibration_protocol_v1.json` | `0af9ee10a370323b2b6a94e757be0e17e8d8f2392d67e6492428fb89aa5bb06a` |
| `artifacts/dynamics_scaling_validation_v3.json` | `e3c71ff09627b1cb1b878da57e223ab3fe463528046f32a5f7498c502f11b5b3` |
| `benchmarks/dynamics_scaling_protocol_v3.json` | `456d04f1d947a37bad9984196ae9f7084f1492bf51ac5fa60ba8aa8865b9d7a9` |
| `artifacts/topology_validation_v2.json` | `a9abb62c8d9361fe3633de992274467dd55a6491cbd872abb46e5b0d2086f878` |
| `benchmarks/topology_protocol_v2.json` | `2c182ae8f275b17a91acaffc7831d9aa8ab08172bc5eb2507441fa29b41e5b21` |
| `artifacts/native_scaling_v1.json` | `591fb2ead4d064163e7023a77e24aede48aad0d65d873d764fc19d33ea9f4251` |
| `benchmarks/native_scaling_protocol_v1.json` | `fde5bdfd2817008ef105523c01b2c1a09e714ad8cd4664cb2fc6e8dd38497355` |
| `artifacts/local_learning_scaling_v1.json` | `0d80c1c209c947000c1d6a6aa823d68cf957ad418a6d0f1b5568f83ae88e0bb7` |
| `artifacts/local_learning_large_scale_v1.json` | `b93853ab9496694260eed3d0f3c62f1721509176543896b08703f451fdba5c5c` |
| `artifacts/development_validation_v2.json` | `8e0fbacd8ad1eee42fa481b9b7fcc59f89d9ea83b4f15d65f2b4e16506484d0f` |
| `artifacts/development_scale_6144_v2.json` | `656836961565f3d0259f8a48d630b66b7c674ad6967d9cc2e571a7a2c7a4c592` |
| `benchmarks/development_protocol_v2.json` | `5e3a5131d2ed60877aa19bc8e364f3b32322af74fbe689f061e20eab69f7885a` |
| `artifacts/swe-dev-gemma/sqlfluff__sqlfluff-1625/sqlfluff__sqlfluff-1625.traj.json` | `d8d1007bc618f5c28794c21327834018a7187f6822b40c775afe8375d7c9b0a3` |
| Gemma 4 12B QAT Q4_0 GGUF | `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` |

## Gemma baseline configuration

The comparison uses Google's official Gemma 4 12B instruction QAT Q4_0 GGUF
(11.95B parameters) through llama.cpp b10218 (`de699957b`), with the native
chat template and thinking parser enabled. The server has one 131,072-token
slot, GPU layer offload, Flash Attention, and Q4_0 key/value cache. Sampling is
temperature 1.0, top-p 0.95, top-k 64, seed 42, and a hard 4,096-token response
cap. Both models use the same mini-swe-agent system/instance templates, Docker
image, cost limit, and 100-action ceiling.

## What this result does not establish

This is an instance-specific episodic checkpoint grown after reading a public
solution trajectory. It demonstrates that simulated human cells alone can
store, emit, and execute a successful SWE trajectory, but it is not a fair
held-out comparison with Gemma and must not be reported as general SWE-bench
performance.

The byte-prediction gates are genuinely file-disjoint and now include a
cross-codebase cohort, but remain teacher-forced Python sequence tasks. They
establish transfer to unseen modules and guard circuit changes against
memorization; they do not establish autonomous repair ability, broad
natural-language modeling, or parity with a modern LLM.

The headline gate remains: freeze the cell checkpoint before selecting a
held-out post-checkpoint cohort; use the same mini-swe-agent templates, action
limit, Docker images, and compute limits for both models; then report paired
resolution rates with a 95% confidence interval. The completed one-instance
Gemma development run exhausted all 100 actions, returned `LimitsExceeded`,
and produced an empty submission, so it records 0/1 on that development
instance. That is an honest diagnostic result, not a statistically meaningful
comparison and not a reason to tune the cell system to this instance.
