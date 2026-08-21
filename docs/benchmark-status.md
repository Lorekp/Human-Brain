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
- Project tests: 35 passed.

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

## Reproducibility artifacts

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/swe_sqlfluff_1625_cells.npz` | `bf54b181ce24ff636d84a3e9a712eaf96caceb1ae343299169727f884f2dda91` |
| `artifacts/swe-dev-expert/sqlfluff__sqlfluff-1625/teacher.traj.json` | `cae6a4dbe8750a8bfe83db1e04103555a4f657ae6ac960cda1f814e761a34a4f` |
| `artifacts/swe-dev-cell/sqlfluff__sqlfluff-1625/dendritic-v4.traj.json` | `8d7b7ba5b2911573505110f61cc4a457c9b595abb895ac8b8d21027b225e5cd2` |
| `artifacts/swe-eval-cell/human-cell-dendritic-episodic-v2.human-cell-sqlfluff-1625-v3.json` | `6a8cfc794732220a26fa9f2f27b9575c36dbfe5ce1580b1000bbdeed430f53b8` |
| `artifacts/code_language_validation_v3_witten_bell_cells_gain16.json` | `a76ae9575c22f81278071508716fa03dac19b46c9d43a7acc406545740e17366` |
| `artifacts/code_language_test_v3.json` | `03eb18664a288377dbe240d5d31fc7aff092849be7bdf6ad6d66063b659f68d5` |
| `artifacts/code_language_freeze_v3.json` | `39f62eaac24695bdc52bc2682f75e8b598720102e8f8995a955c8b980c9b1834` |
| `benchmarks/code_language_protocol_v3.json` | `54f07ff3dd4abb141ed4ce6226e0f30ce27ea98faafe7fea7075df2791606d60` |
| `artifacts/code_language_validation_v4.json` | `9d5d3d148dd9949ea14bcfcf24fd91977948c114c5cd087815180305fd24a335` |
| `benchmarks/code_language_protocol_v4.json` | `9b5543e8d9b201c744a3606c38de7bf32df7f191720c9b307e59186d6affa891` |
| `artifacts/external_code_test_v1.json` | `6e76e18d4663d9f07c945e60b791dabd57e3137f87b02551fad9f7dca022294f` |
| `benchmarks/external_code_protocol_v1.json` | `63069db2fd8d6a3ea3b1cfd211785686cb87e60248fdf4ed95fadab5bf9d9f11` |
| `artifacts/generation_validation_v1.json` | `2391a947bbb4f66fef330df611ed55e2008c44e785e528dda90bfcd586e4822a` |
| `artifacts/generation_validation_v1_persistent_asc.json` | `347f168ef3cc367fedf85f4b7d18d2ee04db16d6f26c7dbadb78052c59384d2f` |
| `benchmarks/generation_protocol_v1.json` | `254754c7464a3b2727aadd51de3f72c597958ff487ebd74aed06ee3ce20d8388` |
| `artifacts/native_scaling_v1.json` | `54d9bae841901b37a2c72ae10eb458c56d7064ced1e22699f794a8ad5d2bcbeb` |
| `benchmarks/native_scaling_protocol_v1.json` | `fde5bdfd2817008ef105523c01b2c1a09e714ad8cd4664cb2fc6e8dd38497355` |
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
resolution rates with a 95% confidence interval. Gemma's current one-instance
development run is also bounded at 100 actions and must finish before even the
memorized-dev comparison is recorded.
