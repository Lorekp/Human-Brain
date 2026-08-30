# Public-TRAIN prompt-factor manifest V1 preflight

Status: implemented, authenticated, and stopped before learning.

This is the first manifest-only layer of
[`public-train-prompt-factor-bridge-design.md`](public-train-prompt-factor-bridge-design.md).
It does not allocate the 49,152-byte learner, update a factor head, construct a
checkpoint, or open public DEV or any later surface.

## Frozen public surface

The gate consumes the source-bound capsule
[`public_train_prompt_signature_capsule_v1.json`](../../experiments/r3_pilot/public_train_prompt_signature_capsule_v1.json).
Its 90 records have exactly two fields, `prompt` and `signature`.  The gate
does not import the task generator, task records, sandbox, results, or
artifacts.  After strict parsing, each signature is detached through the
banked typed-signature parser and each record is immediately reduced to the
checkpoint-free `PromptSpanInput` interface.

Frozen identities:

- capsule raw SHA-256:
  `0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289`;
- capsule canonical-payload SHA-256:
  `d64c61452d8a4fcd2e67a4deb4848c07cd93a757d36da2cfd2b90f81fa00a617`;
- 24-factor registry SHA-256:
  `f7df546c7c89c20d48c79c737dab8fcb101d9c861cb041c0c39654268ecb28c6`;
- input-projection SHA-256:
  `2108f4864cfbbcd3f44be96ad7b55d95a89d8d4899c824ec913204b5318320d6`;
- 90-appearance atomic lesson-manifest SHA-256:
  `139e678a6001f5264d8e6ba44609e2b97082cb8b7c443462326c370d4fc5e774`.
- canonical preflight-payload SHA-256:
  `c4dd1d14b57c9359691f8077742f41b2fe4277e6cca414b5b2f60f94871459cd`.

Every atomic target resolves to an actual ordered one-to-four-token span in
its authenticated physical prompt input.  Repeated semantic inputs have
identical target lists, and no repeated span identity changes positive versus
negative status for a factor.  The record parser rejects unknown and extra
fields recursively, including all forbidden identity, case, solution,
candidate, program, and protected-surface names from the design.

## Exact preflight result

The canonical invocation is:

```powershell
.venv\Scripts\python.exe experiments/r3_pilot/public_train_prompt_factor_manifest_v1.py
```

It returns
`STOP_PUBLIC_TRAIN_PROMPT_FACTOR_IDENTIFIABILITY_V1` before invoking the split
constructor.  The 90 public prompt/signature appearances collapse to 42
distinct `PromptSpanInput.sha256` values after the required declared-name
normalization; there are 48 duplicate appearances in 19 duplicate groups.
Multiplicity is 23 groups of one, eight of two, six of three, two of four,
one of five, and two of ten.

Factor coverage is shown as `appearances / distinct physical inputs`:

| factor | coverage |
| --- | ---: |
| add | 30 / 22 |
| subtract | 0 / 0 |
| minimum | 20 / 12 |
| maximum | 30 / 13 |
| absolute | 10 / 1 |
| less-than | 10 / 8 |
| map | 40 / 21 |
| sum | 10 / 7 |
| count | 10 / 8 |
| contains | 0 / 0 |
| filter | 20 / 6 |
| reverse | 10 / 3 |
| class:ascii-decimal-digits | 5 / 2 |
| class:ascii-lowercase | 6 / 2 |
| class:english-vowels | 9 / 2 |
| every other frozen class factor | 0 / 0 |

The 90-unique prerequisite fails.  Independently, `absolute`,
`class:ascii-decimal-digits`, `class:ascii-lowercase`, and
`class:english-vowels` occur in fewer than three distinct physical inputs.
The exact 72/18 split is therefore unidentifiable under its frozen rules.
Preflight records `split_manifest_sha256 = null`, zero learner-array bytes,
zero learning updates, and no checkpoint.

## Consequence

Do not restore uniqueness with entrypoint spelling, task/family/ordinal
identity, or the whole-prompt audit hash.  Those values are deliberately
excluded from semantic input identity, and using them would break the frozen
renaming and audit-hash invariants.  Do not reduce the 72/18 split or the
three-distinct-input minimum after observing this result.

This STOP falsifies the current capacity-split manifest, not physical prompt
spans and not local factor learning.  A successor must be preregistered on a
genuinely diverse public-TRAIN prompt corpus or on a target-independent
augmentation that preserves the same semantic input interface.  Public DEV,
M0, canonical evaluation, held-out data, Path-2 protected material, and C1
remain sealed.
