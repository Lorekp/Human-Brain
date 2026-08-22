# Typed-program public TRAIN v1 results

Status date: 2026-08-22

The preregistered public TRAIN curriculum completed once from source commit
`1ae5418b92e81dc48c6a2e36fb8822ceec7ce202`. The raw protocol SHA-256 is
`1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191`,
and the recorded typed-stack base commit is
`600ef82bd2edddb00b5df237754bec62be93c0b1`.

The result is **public TRAIN label replay: memorization only, not
generalization**. Development completion remained unopened and its run count
remained zero. No Simple-SWE, canonical, private, or other evaluation was
opened or run.

## Result

- One growth attempt produced one common checkpoint.
- All 3/3 TRAIN tasks completed on their sole attempt.
- All 9/9 registered public semantic cases passed.
- The 16 emitted tokens exactly matched the 16 public teacher tokens.
- All 16 emitted tokens had one valid fitted-cell race.
- Every task passed label replay, typed lowering, syntax, compilation,
  semantics, prompt/signature identity, raw checkpoint immutability,
  cell/synapse-state immutability, race audit, and timing/memory accounting.
- The checkpoint parsed canonically from raw bytes without completion, and its
  raw bytes and cell/synapse-state hash remained unchanged across all tasks.

| Task | Attempts | Tokens/races | Cases | Wall time (ns) | Process time (ns) | Current/peak traced bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train_int_identity` | 1 | 1/1 | 3/3 | 113,220,800 | 109,375,000 | 3,111,365 / 7,506,071 |
| `train_int_add_negative_two` | 1 | 6/6 | 3/3 | 80,573,800 | 78,125,000 | 3,264,766 / 7,506,071 |
| `train_list_map_add_positive_three` | 1 | 9/9 | 3/3 | 105,083,400 | 109,375,000 | 3,301,797 / 7,506,071 |

Training used three sequences containing 651 total sequence tokens with
sequence lengths `[180, 190, 281]`. Training took 769,603,300 ns wall time and
3,468,750,000 ns process time. The complete run took 1,102,609,000 ns wall time
and 3,781,250,000 ns process time, with 3,295,589 current and 7,506,071 peak
traced bytes at summary time. The six evidence events occupy 17,867 bytes.

## Checkpoint identity

| Field | Value |
| --- | --- |
| Bytes | `986677` |
| Raw SHA-256 | `0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d` |
| Cell/synapse SHA-256 | `86b40631c49db55124dff3212645dc571555c5939fc4004a5d94c54b49e99779` |
| Layer-1 protocol SHA-256 | `7daf2ee18d20585b2bf533eb2eafaa7a0150bc2e7da122eb791d3e32759d2213` |
| Training digest | `788eedb727cd5c30144275f3708a110b0c962d47053c42ae4eb8c93ab89a152f` |

The ordered training-input SHA-256 values are:

1. `20e88a83e064a433de8bf2a887d135a57900e66a4aa48289513bd4c9d64f9c39`
2. `437e9ae4b9cdd1b628d4e5ae46c76b402c3c59181406623d065531604386e9fe`
3. `2a4e6ea9e968551939adb0c69e8a56570eac5f40c9dcde50c5da62bf5b2ffea6`

## Artifact hashes

These hashes identify the exact checked-in bytes. The evidence order is
run-start, training, the three TRAIN tasks in protocol order, then summary.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/typed_program_public_train_v1/checkpoint.json` | 986,677 | `0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d` |
| `artifacts/typed_program_public_train_v1/evidence/000-run-start.json` | 494 | `79dff01017e63072a3d32eb38d3a2435d1cfd9b008eae5501e55612a2be64b1d` |
| `artifacts/typed_program_public_train_v1/evidence/001-training.json` | 1,000 | `926daebc386eac7e13c77dcb1746d8e8e920f2c7716b6040d10faa9330195daf` |
| `artifacts/typed_program_public_train_v1/evidence/002-train_int_identity.json` | 3,082 | `6e223fed4fc66c7932ad4f3c1ac0ca590ac8e1ec0ce565c79eb08a8456f8cdeb` |
| `artifacts/typed_program_public_train_v1/evidence/003-train_int_add_negative_two.json` | 4,928 | `c10f903fc114e988ab6c3a78338bf04bf9531f0c20b8ce61f902664b8b29812d` |
| `artifacts/typed_program_public_train_v1/evidence/004-train_list_map_add_positive_three.json` | 6,027 | `9df04197196402cde5762c1222f8f6dd631682b2d30a89234ec69241e8b6cdce` |
| `artifacts/typed_program_public_train_v1/evidence/005-summary.json` | 2,336 | `71133044d55e48a3bbe5cb147a33ce2f543e0a0dad855ae11e2e1b140b6b02d7` |

The evidence was independently checked as strict, canonical JSON with no
duplicate keys. Its protocol, prompt, model-input, program, compiled-source,
checkpoint, task-order, race, timing, and accounting identities all recompute.
It contains no absolute path, URL, secret, development-task definition, or
private/canonical/Simple-SWE identifier.

The runner's incremental, nonoverwriting failure preservation remains intact:
a training failure is sealed before rethrow, and a completion or audit failure
is retained as terminal task evidence without retry or skipping later tasks.
This successful run contains no failure marker.

## Next step

The next capability step is a one-shot, teacher-free public development run
from the frozen checkpoint: no development teacher tokens, growth, retry,
tuning, or replacement. That future result must be reported separately as the
first development observation; none is included here.
