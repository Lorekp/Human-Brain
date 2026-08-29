# Benchmark status

Status date: 2026-08-29

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
- Combined developmental/native integration: 69 forced-CPU tests and 4 real
  CUDA tests passed, plus Rust formatting, Clippy, and Cargo tests.

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
scale-level grouping claim. Direct protocol-v3 builders now construct grouped,
1D, 2D, layered, and 3D topologies at one million cells with linear target
memory, but that is a construction result rather than predictive evidence.

## Native simulation scaling

The complete recurrent token interval executes outside Python. Native ABI 3 is
bit-exact against the frozen serial kernel over 64 tokens and selects between
deterministic dense Rayon chunks and an exact quiescent frontier. The registered
natural sparse trajectories report:

| Configured cells | Frontier tokens/s | Maximum updated cells/token | Dense tokens/s | Persistent arrays |
| ---: | ---: | ---: | ---: | ---: |
| 98,304 | 843.58 | 2,466 (2.51%) | 79.15 | 209.76 MiB |
| 131,072 | 754.17 | 2,004 (1.53%) | 61.53 | 279.67 MiB |
| 393,216 | 449.83 | 937 (0.24%) | 25.92 | 838.92 MiB |
| 1,048,576 | 193.36 | 544 (0.052%) | 3.59 | 2,237.04 MiB |

Controlled-density exactness runs prevent these sparse trajectories from being
misread. At 131,072 cells the frontier runs at 568.02, 176.15, 51.50, and
11.71 tokens/s at 1%, 5%, 20%, and 100% updated cells; at 1,048,576 cells the
rates are 53.25, 5.27, 3.00, and 1.41. CPU therefore passes the >100-token/s
gate through roughly 5% activity at 131,072 cells, while the million-cell pass
is explicitly a 0.052%-updated result.

The persistent CUDA path is also exact and does not rely on frontier skipping.
It touches every enabled cell and all six recurrent edges per cell on every one
of 100 steps, reaching 263.26 end-to-end tokens/s at 100,000 cells and 144.03
at 200,000 cells on the RTX 4070 SUPER. Sparse spike counts are reported, so
this proves the hard simulation-throughput gate rather than useful recruitment.

At one million cells, direct structured builders create six million edges in
15.74-48.43 seconds with 22.89 MiB target arrays, no self/duplicate edges, and
measured near-linear time/memory exponents. Exact ridge learning remains
quadratic in memory and cubic at solve; the dense vocabulary-by-cell readout is
the next inference/storage bottleneck.

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
theoretical 72.00-GiB normal matrix. In that frozen v1 artifact, topology
construction took 31.26 seconds while growth took 3.43 seconds. The later
direct sampler removes that initialization bottleneck without changing this
capability result. See `docs/local-readout-learning.md` for the rule, locality
boundary, full comparison, and claim limits.

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

## Registered large-scale capability gate

Large-network construction and simulation are now fast enough to test
capability without treating speed as capability. Protocol v1 freezes 148
validation-only runs at 6,144, 24,576, 98,304, and 131,072 cells before any
result exists. It separates a fixed 1,024-character experience track from a
square-root-scaled data track, uses five paired seeds for the primary scale
comparison, and records activity/readout-source density alongside prediction.

The architecture matrix covers uniform and self-edge-permitting random
controls, single-scale/nested/mixed groups, 1D/2D/layered-2D/3D geometry, and an
abstract 4D functional lattice. The 4D condition is a computational control,
not an anatomical claim. No topology may be selected post hoc: only the named
scale, nested-group, and layered-sheet effects are confirmatory, and every
stage remains required regardless of earlier results.

The runner is plan-only unless both `--execute` and the exact protocol hash are
provided. It verifies canonical source hashes and the fresh validation-history
boundary, writes each run atomically, and resumes only in frozen order. The
full registered estimate is 12-20 hours. One frozen 6,144-cell canary is now
recorded: 17.06% teacher-forced accuracy, 1.04% free-running accuracy, 94.14%
responsive state cells, 14.97 Hz mean firing, and 1,740 applied local updates.
The 1/148 incomplete artifact supports no scale or topology claim; further
execution is paused while the simple software-task curriculum is built. See
`docs/large-scale-capability-v1.md`.

## Unopened simple software-task gate

The family-disjoint v1 protocol and fail-closed local harness are implemented.
They freeze the cell checkpoint/source/decoder, runtime, native artifact, and
sandbox worker; reserve and externally anchor one seed before task generation;
reload/regenerate the canonical bundle; seal all 30 cell completions before
hidden tests execute; and report pass@1 plus the full failure taxonomy. No
canonical freeze, evaluation seed, private bundle, completion set, or score has
been created. Local artifacts are trusted-operator evidence rather than remote
attestation. This is an internal development gate because evaluation-family
oracle logic is committed; it is not a contamination-clean external benchmark.
The breaking v3 task/bundle schema now carries explicit typed callable
signatures and was introduced before any canonical freeze, so no evaluation
artifact required migration. See
`docs/simple-swe-curriculum.md` for the task split and claim boundary.

## Bottom-up typed semantic public TRAIN bridge

The replacement for the retired H=2 prompt-projection planner has passed its
first broad public-TRAIN gate. A single frozen bottom-up typed semantic
synthesizer completed 30/30 canonical TRAIN tasks: 10/10 scalar-offset,
10/10 `List.map(add)`, and 10/10 `List.map(abs)`. All 960 property cases that
were excluded from model selection passed in the canonical sandbox. The exact
selected AST-cost histogram was 10 at cost 3, 10 at cost 5, and 10 at cost 6.

The model received only the typed signature and four visible regression input/
output rows per task. Prompt text, task/family identity, and all 32 property
rows per task were excluded. One immutable checkpoint produced 17,220
candidate evaluations, 1,750 observed semantic classes, 110,200 target-free
operator predictions, and 270,200 fitted-GLIF races. The run took 8,937.22
seconds. Independent verification matched all frozen source hashes, rebuilt
checkpoint identities, reconstructed all selected programs and source hashes,
and replayed all 30 final sources through the sandbox.

The banked verdict is
`PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1`; the artifact SHA-256
is `09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767`.
This is physical programming by examples on three TRAIN families. It is not
prompt-only completion, family-disjoint transfer, M0, DEV/eval, protected,
held-out, Path-2, or C1 evidence. The canonical Simple-SWE evaluation remains
unopened.

A subsequent performance-only implementation keeps the banked artifact
immutable while removing repeated checkpoint-array hashing inside one
authenticated semantic inference and replacing dense one-hot operator matrix
products with event-sparse column gathers plus bounded batched GLIF races.  On
a fresh replay of the first round-robin task in each banked TRAIN family, all
selection fields, program/source hashes, operator counts, ordered race digests,
checkpoint identities, and sandbox results matched the artifact exactly.  The
three task times changed from 263.93→2.33 seconds (scalar offset),
602.40→16.71 seconds (`List.map(add)`), and 23.87→2.01 seconds
(`List.map(abs)`), a 42.3x aggregate reduction.  This is an engineering replay
of three public-TRAIN tasks, not a second 30-task gate or broader capability
claim.

## Path-2 factorized successor replay readiness

The separately named public-TRAIN Path-2 successor replay is implemented and
preflight-ready, but no 49-book replay has been spent. It starts from empty
native and scorer state, consumes only the authenticated 49-book
`VerifiedTrainInput` stream, and records complete-tuple/per-column successor
identity while keeping matching and active telemetry non-gating. An exclusive
source-bound spend marker, one active book-boundary native checkpoint, compact
per-book scorer journals, and an atomic cursor manifest make it resumable.

Each scorer record is 62 bytes, for a maximum 131,282,288-byte journal. The
estimated final durable state is at most 1,621,955,424 bytes and checkpoint
publication peak is 3,120,208,302 bytes. Scorer restoration is O(authenticated
prefix records); its 3–8 GiB live Python-memory range is an unmeasured
engineering estimate, not a demonstrated envelope. The actual source/TRAIN
preflight returned `PREFLIGHT_READY_NOT_RUN`, with zero held-out/C1 input
materialization and no marker, manifest, native checkpoint, or simulation run.
The completed dense checkpoint is deliberately not imported. This is runner
readiness only, not a Path-2 PASS/STOP, cost-cap result, C1 opening, transfer, or
capability evidence.

## Typed-program public TRAIN curriculum

The preregistered Layer-1 recurrent fitted-cell TRAIN run completed once from
source commit `1ae5418b92e81dc48c6a2e36fb8822ceec7ce202` under raw protocol
SHA-256 `1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191`.
One growth attempt produced one 986,677-byte checkpoint with raw SHA-256
`0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d`.

All 3/3 TRAIN tasks and 9/9 public semantic cases passed on their sole
completion attempts. The model exactly replayed all 16 public teacher tokens,
with 16/16 valid fitted-cell races and every registered task/accounting gate
passed. The complete run took 1,102,609,000 ns wall time and 3,781,250,000 ns
process time; summary traced memory was 3,295,589 current and 7,506,071 peak
bytes. Development completion remained unopened during that TRAIN run.

This is **memorization only, not generalization**. See
`docs/typed-program-public-train-v1-results.md` for the exact evidence hashes,
metrics, gates, and claim boundary, and
`docs/typed-program-public-curriculum.md` for the preregistered protocol.

The separate teacher-free public DEV step was then executed exactly once.
The exact public DEV protocol SHA-256 is
`c7be743617c5fbc090891275f77a636bf1a7a6a89d27a1d1e7cf0e889d19ae56`.
It binds the committed TRAIN artifact at
`8291fe1a0039db7141817c438854a1817a0a8300`, opens only the two existing
teacher-free public definitions, and reports exact pass@1 from 0 through 2
with every failure. The frozen result was **0/2**, so promotion failed. Both
tasks compiled, but both emitted the exact one-token TRAIN identity stream
`variable-0`; five of six semantic cases failed. No retry, tuning, growth, or
replacement occurred. See `docs/typed-program-public-dev-v1-results.md`.

The live TRAIN-v1 and DEV-v1 loaders, runners, CLIs, and execution-only tests
were retired when incompatible Layer-1 v4 replaced v3. The byte-exact
protocols, checkpoint, evidence, and result reports above remain historical
records, and their bound commits remain the reproducibility anchors. DEV v1 is
retired permanently and is not a tuning set. The unexecuted v4 A0/A1/A2 stage
is frozen in `docs/typed-composition-a012-v1.md`.

## Native role/place scalar development

The newer native temporal lane now owns generation atomically: internal COPY
controls, physical role/place recall, fitted output-cell races, feedback, and
EOS all execute behind the no-argument `NativeTemporalCircuit.generate()`
surface. An identifier-only prototype was replaced rather than retained; one
slot-generic transient store now binds identifier and literal roles.

The sole public synthetic combined run trained add/1 and subtract/2 templates
on one circuit. Teacher-free DEV then generated exact compiling functions for
unseen `boost`/37 and `drop`/14, executed both over the fixed public inputs
`{-5, 0, 9}`, and produced the expected six results before ordinary EOS. The
multi-digit literal bytes held temporal state across place recall. No target,
oracle result, Python token selection, learned head, protected material, or C1
material entered generation. This is a two-route slot-recombination
micro-capability, not family-disjoint composition or SWE evidence. See
`experiments/r3_pilot/public_train_dev_role_place_scalar_routing_probe.json`.

## R3 full-TRAIN development replay

The native `(theta_match=16, theta_active=18)`, seed-17 cost runner completed
all 49 canonical TRAIN books: 2,117,412 token steps, 49 transient resets,
23,286,478 allocated/live segments, 2,853 punishments, 410 full predictions,
and no native scientific STOP. Held-out books materialized: zero. The final
logical-state SHA-256 is
`d4a3ae81c034c68c8f2f606dcbb2a6d0e1c2dc2337979383ed2e64033ac3305a`.

The runner's `PASS` means complete gross replay, native/tracked count parity,
and physical winner/basket/GABA causality under its transition-bounded store.
It is not an R3-F0 structural pass. The frozen distinct-key fixture expects
23,286,406 segments; the 72-segment excess is a seven-repeat net shortfall from
full 11-column reuse, including five partial one-column predictions. The
aggregate record cannot attribute those replacements between punishment-driven
deactivation and winner-key divergence. Recorded throughput is also excluded
from cap evidence because concurrent native work contaminated timing after
about book 12. See
`benchmarks/r3_native_train_full_two_tier_theta16_18_seed17.json`.

## Reproducibility artifacts

Hashes below identify the canonical LF-normalized file bytes stored in Git,
not a platform-specific checkout's newline representation.

| Artifact | SHA-256 |
| --- | --- |
| `benchmarks/typed_program_public_train_protocol_v1.json` | `1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191` |
| `artifacts/typed_program_public_train_v1/checkpoint.json` | `0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d` |
| `artifacts/typed_program_public_train_v1/evidence/005-summary.json` | `71133044d55e48a3bbe5cb147a33ce2f543e0a0dad855ae11e2e1b140b6b02d7` |
| `benchmarks/typed_program_public_dev_protocol_v1.json` | `c7be743617c5fbc090891275f77a636bf1a7a6a89d27a1d1e7cf0e889d19ae56` |
| `artifacts/typed_program_public_dev_v1/evidence/003-summary.json` | `9cf93dfa9412af9a2b7b347ef3bc4878b4fa96d76b8fe48aa7b50fea7bd47e09` |
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
| `artifacts/native_scaling_v2.json` | `4bdfbee1c53c57e3842671829a4f903bffa46cc46a0d73b0582a8eccfd2d897b` |
| `benchmarks/native_scaling_protocol_v2.json` | `ef54dc44a40eb14f23c50ddd85554fe0aec304340eea1513d0546ee2c9404545` |
| `artifacts/native_activity_density_v1.json` | `a1a95f66a382451d779abd53ed22bcc724c32fdcb7efba3f04026bad2dc15a7b` |
| `benchmarks/native_activity_density_protocol_v1.json` | `a9e8d8fceaad6b895c9b277342f903e3d709e33454d4fc623e64a6f69105b5a4` |
| `artifacts/structured_topology_construction_v3.json` | `8280d624662dc3e767517678882edc404ea75f42f772ebfc7d8dccc2d360ac0b` |
| `benchmarks/structured_topology_protocol_v3.json` | `2e56a4c30a64e2cf23178281c1ae1d052c148f505a49e19a143ad5b82edf702c` |
| `artifacts/cuda_scaling_v1.json` | `9a41c7241e6256ff128021259fb63c01756e73e87b5c3ff5d721c7bed5f2f09f` |
| `benchmarks/cuda_scaling_protocol_v1.json` | `42b2e469517c23a54ea74538e165bb62a3cb4590fe58958e5214a8dc07cd7d92` |
| `artifacts/local_learning_scaling_v1.json` | `0d80c1c209c947000c1d6a6aa823d68cf957ad418a6d0f1b5568f83ae88e0bb7` |
| `artifacts/local_learning_large_scale_v1.json` | `b93853ab9496694260eed3d0f3c62f1721509176543896b08703f451fdba5c5c` |
| `artifacts/development_validation_v2.json` | `8e0fbacd8ad1eee42fa481b9b7fcc59f89d9ea83b4f15d65f2b4e16506484d0f` |
| `artifacts/development_scale_6144_v2.json` | `656836961565f3d0259f8a48d630b66b7c674ad6967d9cc2e571a7a2c7a4c592` |
| `benchmarks/development_protocol_v2.json` | `5e3a5131d2ed60877aa19bc8e364f3b32322af74fbe689f061e20eab69f7885a` |
| `artifacts/large_scale_capability_validation_v1.json` | `633a3cf3d0ae46f7d7e8c8e5ba0a6cd617566aac66844069499c20a102f0c71d` |
| `benchmarks/large_scale_capability_protocol_v1.json` | `004caa095e796f8671a5989148b881058c0cdc16334101f31de42784fdd84535` |
| `benchmarks/simple_swe_protocol_v1.json` | `f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2` |
| `experiments/r3_pilot/public_typed_semantic_simple_swe_train_bridge_v1.json` | `09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767` |
| `experiments/r3_pilot/public_train_dev_role_place_scalar_routing_probe.json` | `3afac696077d7524444b9128c52e1c0655468955697626a0451eaaf539f23ec4` |
| `benchmarks/r3_native_train_full_two_tier_theta16_18_seed17.json` | `a972902e716e521590db61df0efc6292bd469017bd79ce93286aa51e8132d467` |
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
