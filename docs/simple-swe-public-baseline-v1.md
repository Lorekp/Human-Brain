# Noncanonical public Simple-SWE cell baseline v1

## Result

This public-development run establishes an honest current capability baseline:
the prompt-only cell model solved **0/90 train tasks and 0/27 development
tasks**. Every emitted completion failed Python syntax. No output reached the
signature check, sandboxed runtime, regression assertions, or property
assertions.

This is a capability failure, not a negative result about cell-network scale.
The experiment used only 342 token cells and did not vary neuron count. It
instead identifies a missing program-learning/output mechanism: adding longer
context branches increased learned branches from 15,690 to 248,403, but both
models collapsed to the same invalid output.

## Boundary and preregistration

The experiment protocol was committed before the material run:

- Protocol: `benchmarks/simple_swe_public_baseline_protocol_v1.json`
- Protocol SHA-256:
  `f0218de25bf852286396c2bfd9adf13ee50d29f153df5b46a6d2fefc20ee1c59`
- Protocol commit: `9283d5e703eef9131902803438c4ee338b5c9223`
- Cell source commit: `28ac90871ea4b21bb6a0704ef6e534949127f352`
- Task protocol SHA-256:
  `f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2`
- Portable result artifact: `artifacts/simple_swe_public_baseline_v1.json`
- Portable result artifact SHA-256:
  `524c9df0971b1684e22398866fef0108f48fd40ab3fe8caeb221196c1e753f8b`
- Publication record:
  `artifacts/simple_swe_public_baseline_v1.publication.json`
- Publication record SHA-256:
  `37bca94bc6130c4887aaa24d489e932a4ab82e53b42f322d81251658400cb631`
- Publisher SHA-256:
  `7a094aeb87cc508337adc030bdba18692ecd99a7b6f42b6507b03848cbc7d345`
- Portable metrics/results payload SHA-256:
  `6cb971c5407bdaf4cca4ade574c135cea8531ca6538d5874ed6f3b685b6cf801`
- Frozen raw result SHA-256:
  `bbfa23cde2ce7d8bbf01119e7b253cd1d28282d9fada9b3c551edcb2c4e44900`

Only `generate_visible_partition(protocol, "train")` and
`generate_visible_partition(protocol, "dev")` were called. The model received
only each train task's public prompt, once, as an independent growth sequence.
It received no solution, expected value, regression case, property case, tool
result, external-model output, synthesized program, or template answer.
Generated grading cases stayed inside `execute_completion` and were used only
after the cell emission was complete.

The candidate was selected on train before the development partition was
generated. Development did not change the candidate, checkpoint, tokenizer,
compiler, token budget, or stopping rule. No canonical or held-out evaluation
artifact participated.

## Cohorts

The visible train partition contained 90 tasks, ten from each family:

- `t0_scalar_clamp`
- `t0_scalar_nonnegative`
- `t0_scalar_offset`
- `t1_list_map_abs`
- `t1_list_map_add`
- `t1_string_filter_class`
- `t2_clamp_then_sum`
- `t2_filter_then_reverse`
- `t2_map_then_count`

The development partition contained 27 tasks, three from each disjoint family:

- `t0_digit_sum`
- `t0_longest_character_run`
- `t0_stable_parity_select`
- `t1_longest_target_run`
- `t1_nonzero_digit_product`
- `t1_stable_range_select`
- `t2_digit_reduce_mod`
- `t2_run_length_signature`
- `t2_select_then_span`

## Cell checkpoints and selection

Both candidates used `CellLanguageModel`, seed 7, minimum phrase frequency 2,
and one structural-growth pass over 25,191 public-prompt characters. Learning
was additive growth/strengthening of context-to-token dendritic branches. Token
selection remained a simulated human-cell race with Witten-Bell interpolation
expressed as synaptic current. There was no gradient descent or external token
selector.

| Candidate | Max order | Max phrases | Token cells | Grown branches | Growth seconds | Train inference tok/s | Train result |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt-only-order-8` | 8 | 512 | 342 | 15,690 | 0.0391194 | 13,281.61 | 0/90 |
| `prompt-only-order-64` | 64 | 2,048 | 342 | 248,403 | 0.2221946 | 9,007.51 | 0/90 |

The preregistered ranking was pass count, semantic-test reach, exact-signature
reach, then parse count, with candidate ID as the deterministic final tie-break.
Both candidates ranked `[0, 0, 0, 0]`; therefore
`prompt-only-order-64` was selected by ID, not by a capability advantage.

The selected model was regrown from the same prompts, saved, and freshly
reloaded before development:

- Format: `human-cell-brain-checkpoint-v1`
- SHA-256:
  `14e9041537feb5f944d47508ecb0bc813cb35a9337a61c31d3ee561acf67dfef`
- Size: 1,320,766 bytes
- Training digest:
  `d5220bc935b4ec9d30a8318836af6b6ef7816256ecc65609ca8b0c8de6716210`
- Measured retraining time: 0.3015685 seconds

The checkpoint is a reproducible runtime byproduct under `.runtime`, not a
versioned capability checkpoint. Its hash and complete configuration are in
the result artifact. Its content hash and behavior are the reproducibility
claims; the measured duration is environment-dependent.

## Failure taxonomy

| Partition/candidate | Syntax | Format | Policy | Signature | Runtime | Assertion | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train, order 8 | 90 | 0 | 0 | 0 | 0 | 0 | 0 |
| Train, order 64 | 90 | 0 | 0 | 0 | 0 | 0 | 0 |
| Development, selected order 64 | 27 | 0 | 0 | 0 | 0 | 0 | 0 |

Both train candidates and the selected development checkpoint emitted one
unique source hash across every task:
`e0f6bdce99852540f5cf05b362c7652a4f2bdd2f3bc1ff143a0ba3039213d7ae`.
The free-running sequence repeatedly began with a memorized description suffix,
`Return n plus the fixed offset -5.\"\"\"...`, rather than a complete function.
This is direct evidence of prompt-suffix attractor collapse. The raw compiler
correctly added no algorithmic content, so the failure was not masked by a
heuristic decoder.

## Activity and speed

Every task emitted the fixed budget of 192 cell-selected tokens.

| Phase | Emitted tokens | Inference seconds | Token/s | Mean active cells/race | Active range | Unique winning cells | Simulated cell-steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train, order 8 | 17,280 | 1.3010473 | 13,281.61 | 33.5521 | 1–90 | 11 | 52,987,666 |
| Train, order 64 | 17,280 | 1.9183988 | 9,007.51 | 33.5521 | 1–90 | 11 | 35,919,049 |
| Development, order 64 | 5,184 | 0.7083915 | 7,317.99 | 33.5521 | 1–90 | 11 | 29,181,770 |

Sandbox classification took a measured 0.0742716 seconds for order-8 train,
0.0708099 seconds for order-64 train, and 0.0240294 seconds for development.
Because every completion failed preflight syntax, these times mostly measure
trusted classification rather than executing functions. All timings in the
artifact are measured and environment-dependent; they are preserved evidence,
not deterministic constants.

The `>100 token/s` engineering threshold is exceeded here, but only for a tiny
342-cell context-branch model. It is not evidence that a 100,000-cell network,
let alone a 100-billion-cell network, meets that rate.

## Reproduction

From the repository root on a compatible Python 3.11 runtime:

```text
python benchmarks/simple_swe_public_baseline.py
```

The runner now writes raw output to
`.runtime/simple-swe-public-baseline-v1/raw-result.json` by default and refuses
to overwrite the checked-in evidence. Publish a frozen raw result explicitly:

```text
python benchmarks/publish_simple_swe_public_baseline.py --raw .runtime/simple-swe-public-baseline-v1/raw-result.json --published artifacts/simple_swe_public_baseline_v1.json --record artifacts/simple_swe_public_baseline_v1.publication.json --archive-raw .runtime/simple-swe-public-baseline-v1/raw-result-archive.json --expected-raw-sha256 <frozen-raw-sha256>
```

For this recorded run, the publisher applied the deterministic
`portable-public-evidence-v1` transform to the already frozen raw bytes; it did
not rerun inference or regenerate tasks. It replaced only workstation-bearing
path/command fields with repository-relative paths, an executable basename,
and a portable command, while preserving all results and measured timings. The
publication record binds the raw hash, portable artifact hash, publisher hash,
and complete changed-field list.

The behavior reproducibility claim means the frozen protocol/configuration and public
inputs reproduce the checkpoint content, emitted-token/source hashes, outcome
counts, and activity semantics. It does not promise byte-identical timing
values on another run or machine. The publication transform itself is
byte-deterministic when its frozen raw input bytes and publisher source are
identical.

The portable artifact records the repository-relative protocol, worker, and
checkpoint locations; Python executable basename, version, implementation, and
SHA-256; sandbox-worker/compiler/task-generator/runner hashes; per-task
emitted-token/source hashes; per-task failures and measured timings; and
cell-activity counters.

## Interpretation

Checkpoint save/reload, deterministic task generation, direct cell emission,
cell-activity measurement, raw compilation, outcome separation, and trusted
sandbox classification all worked. Those are plumbing results only.

The capability result is zero. More context branches alone did not change one
output token or recruit another winning cell. The next justified experiment is
therefore a preregistered cell-only program-learning/readout mechanism trained
on legitimate public train solutions or trajectories, with raw-vs-constrained
decoding separated as an ablation. It should first demonstrate train-task
memorization, then open the same family-disjoint development cohort once. More
cells should be tested after that end-to-end learning path can emit at least one
syntactically valid exact-signature function; this run does not support a claim
that larger cell networks are worse.

The frozen benchmark runner accessed `CellLanguageModel._tokenizer` and
`CellLanguageModel._cortex` only to collect per-race activity alongside the raw
completion. That is benchmark-only instrumentation debt, not a supported model
interface. Changing the cell API after development had already been opened
would make this evidence describe code it did not run, so this publication does
not retrofit a new seam. Before the next preregistered capability run, expose a
public instrumented-completion result carrying text, token IDs, race activity,
and timing, and make the benchmark depend only on that seam.
