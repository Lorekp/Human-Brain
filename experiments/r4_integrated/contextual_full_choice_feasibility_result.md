# Joint TRAIN ranking feasibility remains inconclusive

Completed 2026-09-05 under the [frozen protocol](contextual_full_choice_feasibility_protocol.md),
source `d27c3418851072b6f9f4cd01a952b6bef72b681f`. One host-only run started at
08:48:34 UTC and exited normally at 09:03:35. No solver retry, parameter change,
learner update, encoder call, neuronal simulation or task execution followed.

## Outcome and decision

**Close this diagnostic as INCONCLUSIVE.** The fourth linear program reached
its time limit without a solution. The saved vector is the third program's
partial solution, not a witness for the entire ranking problem. Independent
verification rejects it; there is no feasibility or infeasibility certificate.
Fixed-feature capacity, conditioning and acquisition reachability remain unresolved.

The fixed problem uses all 64 actual TRAIN queries: 48 unique whole-span targets
and 16 explicit STOPs, each competing with 1,458 wrong choices. The 93,312
inequalities require target-minus-wrong margins of 1.000001 during fitting and
at least 1 during verification. All 1,622 score coordinates have free signs;
the auxiliary objective minimizes their maximum absolute value.

| LP call | Cumulative active constraints | Outcome | Objective t |
| --- | ---: | --- | ---: |
| 1 | 64 | Optimal for active subset | 0.02223605 |
| 2 | 128 | Optimal for active subset | 0.12393001 |
| 3 | 192 | Optimal for active subset | 0.16317060 |
| 4 | 256 | Time limit; no returned solution | — |

The saved vector satisfies its first 192 constraints with minimum margin
1.000000999916. The next 64, added for the unsolved fourth call, have minimum
margin -1.942507325682. Exporting 256 constraints does not mean that 256 were fit.
Its L-infinity norm is 0.163170604790 and L2 norm is 6.480461949111.

An independent scalar `math.fsum` check of every original comparison finds
**28,151 unit-margin violations**, minimum margin **-1.942507325682**, and no
query with its gold choice ranked first. This rejects this incomplete vector
only. Maximum conservative arithmetic error bound is 2.22e-13; ordinary matrix
scoring agrees with the reference within the frozen tolerance throughout.
Numerical rounding cannot explain these negative margins.

No witness is deployed. This is a numerical diagnostic over previously exposed,
supervised TRAIN data, not learning, neuronal contribution, fresh transfer or
an acquisition gate. The [failed acquisition result](contextual_full_choice_result.md)
and all earlier contracts remain unchanged. The integrated gate, G0 and protected
R5/R6 evaluations remain unpassed and sealed as applicable.

The diagnostic budget is exhausted. Do not extend the solve or infer that a new
representation is required. A subsequent [bounded review](../../docs/research/contextual-rate-cell-preflight-plan.md)
selects a rate-cell sequence learner with separate span/answerability credit for
preflight against learned-head controls. Its motivation is the observed whole-span
and STOP interference failures; it is not implemented or adopted by this result.
The [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns that decision and its representative preflight.

## Verification and cost

All 64 reconstructed feature hashes match the original acquisition offers.
Independent input checks verify actual-query membership, unique authorized
targets, candidate/source custody and the sealed source pool. The 256 saved
active row identities, order and directions match their original features
exactly. Audits verify source/artifact hashes, free bounds and fixed options.
Intermediate vectors were not saved, so auditors checked the source's cut rule
and exported order without claiming an independent replay of every intermediate
argmin. No audit reran the solver or called a model.

Three focused synthetic tests passed before freezing: joint span/STOP fitting,
contradictory targets remaining inconclusive, and omitted-competitor detection.
The earlier 68 tests and existing learner/native code are unchanged. Held-source
lint, input/exit preflight and source-custody checks passed; broader native QA
was not rerun for this standalone saved-data diagnostic.

The run used NumPy 2.4.6, SciPy 1.17.1 and HiGHS 1.12.0 with `highs-ds`,
presolve enabled and primal/dual tolerances 1e-9. Four of at most 32 LP calls
were attempted. Feature reconstruction took 53.336 seconds; the separation and
solver loop took 846.704 seconds, including about 837.7 seconds in the final call.
Recorded total runtime was **900.123 seconds**: the configured 900-second
allowance was exceeded by 0.123 seconds of return/bookkeeping overhead. The
independent full check took 23.32 seconds and the cut audit 2.58 seconds; both
finished before the overall preparation/solve/verification deadline of 09:05 UTC.

| Resource | Recorded cost |
| --- | ---: |
| Peak parent working set, including solver and mapped pages | 1,476.54 MiB |
| Final parent working set / private memory | 1,338.18 / 911.41 MiB |
| Disk-backed feature payload / complete local file | 1,210,070,016 / 1,210,070,144 bytes |
| Saved witness | 12,976 bytes |
| Compressed witness and active-constraint archive | 2,464,946 bytes |

The feature memmap stays in ignored local scratch; its disk bytes and resident
pages overlap in scope and should not be added as independent RAM allocations.
The approximately 1.21 GB feature payload is diagnostic work, not learner state
or evidence of useful scaling.

## Artifact custody

- [Run record](../../benchmarks/contextual_full_choice_feasibility_v1.json),
  23,162 bytes, SHA256 `56e8f8cd1bee7feed785f20d9f1aa5b103bcca3e50c07a24627c2835dc6c34cf`.
- [Witness and active constraints](../../benchmarks/contextual_full_choice_feasibility_v1.npz),
  SHA256 `a030bc7cf386ded8a58b95f8c71f6b1cb74ada93e35b164376a29b0f891d5ff1`.
- [Independent complete witness audit](../../benchmarks/contextual_full_choice_feasibility_v1.audit.json),
  76,253 bytes, SHA256 `72e77a61ca196eb7fd5b4561ee1d8c2013c1ad0c5726e9956bb3a3fb50591344`.

The run record binds unchanged acquisition JSON/NPZ inputs, TRAIN fixture,
feature hashes, protocol and executed source. The independent audit binds those
inputs, the exported vector/archive, checker source and every query's margin
summary. The unchanged local cut-audit receipt has SHA256
`f280a80d94a50ee980f56f7f09cce937d19beafb2ec3188cf0d5b253a3c312b7`;
its solved-prefix check has SHA256
`43fb8968fb6a9bfe0a47cfecaaf67052af72fb670645ae02fd65d9a7c9d321b4`.
