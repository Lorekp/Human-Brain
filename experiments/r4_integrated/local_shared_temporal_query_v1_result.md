# Shared temporal query v1: completed result

The shared temporal map solves **18/20** final checks: 8/8 taught, 4/4 new-value,
4/4 delayed and 2/4 interference. All three matched controls score zero. The
immediate dense-map reference exactly reproduces 17/20. The shared map corrects
one taught answer without losing any previously correct case, but the two
interference failures leave the preregistered strict criterion unmet.

This closes the fixed fit. No additional lessons, gain changes or automatic
repeat fit are authorized by this result.

## Comparison and criterion

Frozen source: `82db27e847f00fb065f03d5464fc519116e6a6db`.
The [recipe](local_shared_temporal_query_v1.md) fixes five fresh arms, seed 733,
1,024 lessons per arm and learning rate 0.01. A single 32-by-32 temporal map is
applied across 32 code channels, replacing the dense 1,024-by-1,024 projection.
The normalized raw-byte query lags, literal neuronal tape, shared byte codec,
causal prefix beliefs and teacher rule retain their declared contracts. Shared
channel correspondence is a supplied structural prior.

`address_frozen` keeps the address map fixed while the codec learns;
`slow_frozen` keeps both fixed with sensory storage active; `writes_disabled`
keeps sensory/key storage zero. `dense_reference` is the explicit immediate
predecessor, freshly initialized rather than warmed from its learned endpoint.
Its saved initialization, endpoint, all 1,024 scientific lesson records and all
20 outputs match the authenticated preceding run. The comparison omits arm,
time and changed work-accounting fields from scientific row equality.

| Arm | Taught /8 | New-value /4 | Delayed /4 | Interference /4 | Total /20 |
|---|---:|---:|---:|---:|---:|
| Learned shared map | 8 | 4 | 4 | 2 | 18 |
| Address frozen | 0 | 0 | 0 | 0 | 0 |
| All slow weights frozen | 0 | 0 | 0 | 0 | 0 |
| Writes disabled | 0 | 0 | 0 | 0 | 0 |
| Dense reference | 7 | 4 | 4 | 2 | 17 |

Novel superiority over the three matched controls, no stratum regression and
strict overall improvement over the dense reference all hold. The required
absolute 20/20 does not. `learned_retrieval_advantage=true` therefore has the
narrow matched-control meaning; `qualification=false` remains the final gate.

## All final outputs

Both learned and dense arms stop with EOS on all 20 checks. In every row below,
`address_frozen` and `writes_disabled` emit the empty byte string and stop at EOS;
`slow_frozen` emits hex `2068` (space followed by `h`) and reaches the byte limit.
This accounts for all 100 final free outputs, including the failed controls.

| ID | Stratum | Target | Learned | Dense reference |
|---:|---|---|---|---|
| 0 | fit | `aa` | `aa` | `aa` |
| 1 | fit | `ab` | `ab` | `ab` |
| 2 | fit | `bb` | `bb` | `bb` |
| 3 | fit | `bc` | `bc` | `bb` |
| 4 | fit | `cc` | `cc` | `cc` |
| 5 | fit | `cd` | `cd` | `cd` |
| 6 | fit | `dd` | `dd` | `dd` |
| 7 | fit | `da` | `da` | `da` |
| 8 | unseen_value | `ac` | `ac` | `ac` |
| 9 | unseen_value | `ca` | `ca` | `ca` |
| 10 | unseen_value | `bd` | `bd` | `bd` |
| 11 | unseen_value | `db` | `db` | `db` |
| 12 | delayed | `ac` | `ac` | `ac` |
| 13 | delayed | `ca` | `ca` | `ca` |
| 14 | delayed | `bd` | `bd` | `bd` |
| 15 | delayed | `db` | `db` | `db` |
| 16 | interference | `ac` | `cc` | `cc` |
| 17 | interference | `ca` | `ca` | `ca` |
| 18 | interference | `bd` | `cc` | `cc` |
| 19 | interference | `db` | `db` | `db` |

All four different-question pairs on a shared taught memory now return their
respective correct values. The corrected case is ID 3: `bc`, where the dense
reference emits `bb`. The two same-question novel assignment swaps, `ac/ca` and
`bd/db`, are jointly correct, and their delayed repeats remain correct. Those
values are withheld from this fresh TRAIN roster but were development-exposed
in preceding experiments.

Interference IDs 16 and 18 instead emit `cc`; IDs 17 and 19 remain correct.
External indexing of saved source positions and priors puts the correct target
start first in both failures, with probabilities 0.266172 and 0.237168. The `cc`
record's start has probabilities 0.185609 and 0.193007 respectively. These saved
facts distinguish start ranking from the actual marginal-byte output. They do
not establish what a different retrieval policy would emit; no counterfactual
output or extra query was generated for this report.

## Acquisition and actual work

Every lesson records its free first attempt before the three byte/EOS teacher
offers. Loss is `sequence_mixture_negative_log_likelihood`; the table reports
mean loss per teacher position, not free-answer error or squared potential loss.

| Arm | First 128 exact | Last 128 exact | First 128 mean NLL | Last 128 mean NLL |
|---|---:|---:|---:|---:|
| `learned` | 0 | 128 | 2.945856 | 0.308636 |
| `address_frozen` | 0 | 0 | 3.349201 | 1.486033 |
| `slow_frozen` | 0 | 0 | 3.778747 | 3.773979 |
| `writes_disabled` | 0 | 0 | 5.138699 | 2.102593 |
| `dense_reference` | 0 | 85 | 2.310500 | 0.338597 |

The learned arm's successive 128-lesson exact counts are
`0, 21, 128, 126, 128, 128, 128, 128`; the dense reference's are
`0, 13, 70, 69, 83, 71, 81, 85`. Each control remains at zero in every bin.
Perfect final acquisition first attempts do not erase the separate interference
failures or prove convergence under further experience.

The full run contains 5,120 first attempts, 12,090 first-generation decisions,
15,360 teacher offers, 100 executed final action receipts and two motor demos.
Of the teacher offers, 12,288 have learning enabled. Across all five arms there
are 5,120 codec and 5,120 address group offers, 4,096 enabled/changed codec
groups, 3,072 enabled address groups and 2,048 changed address groups. In the
writes-disabled arm the enabled address offers cause no actual address change.
These group publications are distinct from byte-level teacher exposure.

The audited full-run work ledger includes acquisition, final ingestion,
query-only delay reads and final free traversal:

| Work | Actual count |
|---|---:|
| Raw document bytes / document EOS | 249,120 / 10,380 |
| Tape allocation and adjacency assignments, each | 259,500 |
| Enabled sensory/key association opportunities, each | 207,600 |
| Raw query bytes / query EOS | 274,500 / 10,980 |
| Query lag shifts / population normalizations | 285,480 / 10,980 |
| Address score evaluations | 550,500 |
| Address projection synapse evaluations | 2,590,507,008 |
| Address local synapse contributions | 1,207,959,552 |
| Shared channel applications / local channel reductions | 281,088 / 130,023,424 |
| Codec slot evaluations / output potentials | 1,386,800 / 356,407,600 |
| Mixture output reductions | 7,111,190 |
| Prefix belief updates | 1,386,800 |
| Branch cursor initializations / advances | 518,500 / 868,300 |

The dense reference retains its larger actual projection work. Frozen controls
still incur the declared computations even when they do not publish updates.

## Admission, timing and storage

The fresh cost workload is separate from acquisition: 20 disposable lessons,
60 teacher offers, five 128-slot cap fixtures, 100 actions and two motor demos.
No cost weights become full-run initialization. The admitted projection was
322.733733 seconds under the fixed 900-second run-plus-audit ceiling. Complete
lesson and journal maxima, cap prediction, other measured work, the 2x margin
and 30-second finalization allowance are included.

| Measured process | Outer seconds | Process-tree peak MiB |
|---|---:|---:|
| Full run | 101.3854070 | 665.5390625 |
| Independent saved-only audit | 18.3599572 | 432.83203125 |
| Aggregate elapsed | 119.7453642 | Peaks are not summed |

The run estimate of 762.1328125 MiB and audit estimate of 579.6445236 MiB each
cover their observed peak; the admitted maximum is 762.1328125 MiB under 2,048.
The prospective allowance includes 3 MiB for bounded streamed rows/lines,
128 MiB run reserve, 32 MiB audit reserve and the declared historical checkpoint
payload. These are engineering estimates; the process watchdog is authoritative.

Complete lessons take 75.184738 seconds and journal work 8.116467 seconds.
The journal contains 5,120 complete scientific rows in 220,882,600 bytes;
maximum line size is 51,839 bytes and recursive row size 94,741 bytes, both
below the fixed 262,144-byte guards. Both audits stream the full historical
6,144-row journal and compare the selected dense-reference rows without
retaining a history list. Audit internal time is 17.669074 seconds; the outer
audit time above also includes startup.

Each shared model stores 567,568 tensor bytes versus 4,757,776 for the dense
reference. The five models plus 60 retained tape states total 46,442,288 bytes.
These payload counts exclude Python objects and transient allocations, which
remain inside measured process peaks. Final ingestion takes 0.270428 seconds,
serialization 0.034564, worker package 4.002493 and action package 7.909804;
stages overlap and should not be summed as independent costs. Per-output time
includes its preceding delay distractor reads, as do the worker package and
work ledger. Changing the reference roster makes cross-run aggregate
timing unsuitable as a clean circuit speed comparison.

Within this same five-arm run, the shared learned arm's 1,024 complete model
lessons take 13.624167 seconds versus 25.793236 for the dense reference (1.893x
lower elapsed time). Including each arm's journal work gives 15.438754 versus
27.699914 seconds (1.794x). Remaining acquisition work counts match, including
3,072 teacher offers and 307,200 codec-slot evaluations each; the declared
address-map products differ. These are descriptive interleaved-run totals,
not randomized timings or isolated kernel speedups. Initialization and learned
trajectories differ. The same-run resource review is
`.pytest-tmp/shared-temporal-resource-review.md`, SHA-256
`ff37c1f13e1325f302125b5cb9db50d227c599893bd5a285e58a93a12c31428d`.
The faster result still fails strict completion.

## QA, custody and interpretation

Seventeen distinct reviewed implementation cases pass with no failed or skipped
cases: one end-to-end case, 15 focused cases and one fresh-process worker case.
Their disposable exposure is 51 teacher offers and 28 free predictions, separate
from cost and acquisition. Receipts are `shared-temporal-e2e-v1.xml`,
`shared-temporal-focused-v1.xml` and `shared-temporal-worker-v1.xml` under
`.pytest-tmp`. The independent auditor has 53 passing authored helper checks
with no failures. Reporting/configuration corrections were completed before
cost admission; no post-result criterion or model change was made. Saved-result
recorders passed on their first default-context attempts, with no ACL-copy
recovery or repeated training/audit.

The full audit passes 1,146 source-file bindings, six completion artifact hashes
and 102 execution receipts. It checks saved tape contents, accounting, custody,
initialization and reference parity without rerunning the learner. Source-free
transport supplies checkpoint state and query-only cases; source routing is not
a filesystem sandbox. Literal source bytes remain explicitly represented in
the neuronal tape. The host-authored action motor executes the generated bytes;
it does not demonstrate learned general-purpose tool use.

Useful learned addressing and the shared temporal prior improve this fixed
small workload. The experiment does not establish unrestricted language
understanding, semantic compression, general software engineering or SWE
benchmark capability. Remaining interference failure motivates only a bounded
review of saved evidence before another policy or architecture is selected.

Evidence anchors (SHA-256):

- Full audit: `3b47336cd99fe868783528179f73ffff4e2df9d767047bed83bec8151d19ea91`.
- Saved description: `0f4d17fac90fb4c03f8317a5b164c0e9faeb615413624c559911e8a64b9e3c6c`.
- [Decision](../../benchmarks/local_shared_temporal_query_v1.decision.json), binding 27 evidence files: `72af1b5f702082c32bc4bb15fe93a677a9b724bee2b61f475297331e1db1a723`.
- Complete lesson journal: `b43de8774cb67337fc467c75820dd33390804c5fa2a25f82219b21903a1b102b`.
