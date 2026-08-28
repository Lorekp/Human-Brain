# R3 calibration-v3 verdict — terminal reject at development R3-F0

**Date:** 2026-08-28
**Status:** **REJECT_R3_CALIBRATION_V3 — spent once; no retry**

The frozen source-bound calibration was executed once under protocol SHA-256
`5b916fe88cdba401c2a80e89b25bfc64447d189d216029d65bb40678c499dfa6`
and native source-bundle identity
`9e5844b51bb3843f8cd08be5247fe0e5ac58d8dcbae6452f527e35a7fa6acabf`.
Its immutable publication contains 41 JSON envelopes totaling 224,834 bytes.
Independent namespace validation reproduced the sorted
`path + NUL + raw-file-SHA bytes` digest
`534ae4a9d3a6cf19bd008942f72cfefae96ca00a6d9dea75d6580d9b575a1630`.
The final file's raw SHA-256 is
`08ce33684c283006aac3aa799c0e9da2acd2313162f3ac48101c40d156a32dd0`;
its envelope records payload SHA-256
`852c0235d890d4a5dbcfb05ece96c36fc4ee17d85f93ded852a3f1d1c6e68448`.

## Recorded gate results

| Gate | Observation | Verdict |
| --- | --- | --- |
| Electrical v3 | All development seeds `3, 7, 17, 29` | PASS |
| q=4 compressed reserve | All four arms at all four development seeds | PASS |
| q=255 production overlap | `(16,18)` passed all four seeds; `(10,13)`, `(18,20)`, and `(20,20)` stopped at all four seeds | One arm eligible for the disjoint control |
| q=255 target-disjoint control | `(16,18)` passed all four development seeds | PASS |
| TRAIN-order R3-F0 | Seed-17 `(16,18)` reached a scorer STOP before a terminal identity result | FAIL: `r3_f0_development` |

The final immutable envelope therefore records exactly:

```text
verdict = REJECT_R3_CALIBRATION_V3
first_failed_gate = r3_f0_development
selected_arm_slug = null
```

No threshold was selected. Per stop-on-failure, independent confirmation,
deterministic confirmation, held-out data, C1, capability scoring, and transfer
evaluation were not accessed. The absence of confirmation artifacts is the
required consequence of the development failure, not missing evidence.
Validation also confirmed the required absence of development selection,
R3-result, resume, and checkpoint artifacts; no nonterminal R3 identity was
fabricated into a result envelope.

## Execution disclosure

An initial direct-file launcher attempt stopped in tooling before artifact
creation; the scientific run remained unspent. The canonical package command
then executed the frozen plan once:

```powershell
python -m experiments.r3_pilot.run_calibration_v3_fixtures --run-frozen-plan
```

There was no scientific retry and there will be no threshold tuning against
this spent result. The next architectural intervention remains provisional
until the dedicated failure-localization lane reports; this verdict alone does
not choose that repair.

## Claim boundary

This is TRAIN-only structural calibration evidence. It is not held-out,
capability, transfer, C1, or software-engineering evidence. The terminal result
artifact is
[`experiments/r3_pilot/calibration_v3_artifacts/selection/final-verdict.json`](../../experiments/r3_pilot/calibration_v3_artifacts/selection/final-verdict.json).
