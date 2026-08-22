# Typed-program public DEV v1 results

Status date: 2026-08-22

The frozen teacher-free public DEV gate was executed exactly once under raw
protocol SHA-256
`c7be743617c5fbc090891275f77a636bf1a7a6a89d27a1d1e7cf0e889d19ae56`.
It restored the committed TRAIN checkpoint with raw SHA-256
`0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d`
freshly for each task. There was no growth, teacher program, retry, tuning,
search, repair, alternate seed, replacement checkpoint, or external model.

The primary result is **0/2 semantic pass@1; promotion failed**.

## Result

| Task | Emitted program | Syntax/compile | Public cases | Boundary | pass@1 |
| --- | --- | --- | ---: | --- | ---: |
| `dev_int_two_stage_offset` | `variable-0` | pass/pass | 0/3 | exact TRAIN-label replay | 0 |
| `dev_list_two_stage_map_offset` | `variable-0` | pass/pass | 1/3 | exact TRAIN-label replay | 0 |

The scalar program compiled to `return value`; expected behavior was signed
add `+7` followed by `-4`, or net `+3`. The list program compiled to
`return items`; expected behavior was elementwise signed add `-6` followed by
`+8`, or net `+2`. Only the empty-list case passed vacuously. The other five
registered semantic cases failed and remain in the evidence.

Both completions emitted the exact public TRAIN teacher stream from
`train_int_identity`. Therefore the preregistered
`teacher_free_label_boundary` assertion—that a returned stream is not an exact
TRAIN label—failed for both tasks. This is a memorization-control violation,
not evidence that a DEV teacher was accessed: the DEV records contain no
teacher tokens, and the comparison required zero additional model calls.

Every task used one fresh checkpoint restore and one completion attempt. Each
one-token program came from one fitted-cell race. Typed lowering, Python
syntax, compilation, prompt/signature identity, raw checkpoint immutability,
cell/synapse-state immutability, race audit, and timing/memory accounting all
passed for both tasks. The run-level recording and accounting gates passed;
that does not mean the capability gate passed.

## Measurements

| Field | Value |
| --- | ---: |
| Checkpoint restores | 2 |
| Completion attempts | 2 |
| Completion tokens / cell races | 2 / 2 |
| Wall time | 298,252,300 ns |
| Process time | 234,375,000 ns |
| Current / peak traced memory | 2,682,022 / 7,382,637 bytes |
| Evidence bytes | 10,755 |

The checkpoint remained 986,677 bytes. Its raw SHA-256 and cell/synapse
SHA-256 remained unchanged before and after both completions.

## Artifact hashes

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/typed_program_public_dev_v1/evidence/000-run-start.json` | 677 | `c5babcdb8b77c4cd0ca6207c2af5a1db1bbc1c190de00e028b9faf60607f0ff2` |
| `artifacts/typed_program_public_dev_v1/evidence/001-dev_int_two_stage_offset.json` | 3,586 | `d9a1f3a0a9b035a8cf700d1ffb4880a8997526f1c3d84766d16528d9a6d06030` |
| `artifacts/typed_program_public_dev_v1/evidence/002-dev_list_two_stage_map_offset.json` | 3,631 | `8ac5ca063fc17f2aec89b341053db96e7e4029d9a62db5622ed55e3732ba05b3` |
| `artifacts/typed_program_public_dev_v1/evidence/003-summary.json` | 2,861 | `9cf93dfa9412af9a2b7b347ef3bc4878b4fa96d76b8fe48aa7b50fea7bd47e09` |

An independent parser-only audit found four canonical JSON files in exact
ordinal order, no temporary or extra files, internally consistent task/race/
timing counters, and correct protocol, source, checkpoint, and TRAIN-artifact
linkage. It did not restore the checkpoint, complete a prompt, or rerun a DEV
case. The v1 records do not contain previous-record hashes; their durable
identity comes from these checked-in raw hashes and the frozen protocol/source/
checkpoint commitments.

## Consequence

This checkpoint does not demonstrate prompt-conditioned composition. The
typed grammar successfully constrained and compiled a valid program, but the
recurrent readout collapsed both unseen prompts to the strongest short TRAIN
label. The next experiment must improve cell-state discrimination and
compositional learning on new public TRAIN-only material, then use a fresh,
predeclared development cohort. This v1 DEV cohort will not be rerun or used
for tuning.
