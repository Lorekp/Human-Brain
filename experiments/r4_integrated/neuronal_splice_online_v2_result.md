# Neuronal splice online v2: completed competitive-reader result

Completed and independently audited on 2026-09-06. The competitive learner
solves 20/32 fresh tasks, including 19 real edits and one explicit STOP, versus
7/32 for its freshly trained marginal reference. It retains 14/16 actions.
The mechanism-superiority component passes; the strict all-fresh/all-retention
criterion fails. This closes the fixed five-arm comparison without additional
rounds, endpoint replay or selection of a better checkpoint.

The [decision bank](../../benchmarks/neuronal_splice_online_v2.decision.json)
binds the finalized report, independent audit, complete source and roster
manifests, launch custody and resource receipts. Its SHA256 is
`55407a4f621ce36235c48cdbeba6e6bb08bf4a1c3a3ecc48af0bf29636e01756`.
The [selected comparison](../../docs/research/neuronal-address-competition-comparison-v1.md)
preserves the original acquisition tasks but introduces a new structural
transfer cohort. The earlier splice-v1 endpoint score is not a matched baseline
for subtracting scores on this different cohort.

## Executed outcomes

| Arm | Fresh /32 | Fresh edits /28 | Fresh STOP /4 | Retained edits /12 | Retained STOP /4 | Old reader before → after /20 |
|---|---:|---:|---:|---:|---:|---:|
| Competitive learned | 20 | 19 | 1 | 10 | 4 | 20 → 20 |
| Action frozen | 1 | 0 | 1 | 0 | 4 | 20 → 20 |
| All slow weights frozen | 1 | 0 | 1 | 0 | 2 | 20 → 20 |
| Memory disabled | 1 | 0 | 1 | 4 | 4 | 0 → 0 |
| Marginal reference | 7 | 6 | 1 | 8 | 4 | 18 → 18 |

Competitive learning adds thirteen successful fresh edits over the matched
marginal arm and exceeds all three publication/memory controls. It preserves
all eighteen originally correct reader cases. Its 20/20 reader score was
already present before this acquisition, so the extra two reader answers are
not evidence of newly acquired knowledge during these rounds. The marginal
reference misses the same two old cases, 16 and 18, before and after learning.

The following cells give successful tasks out of eight, then complete contrast
pairs out of four. A pair passes only when both actual endpoint actions pass.

| Arm | Document order | Continuation | Value role | Operation/layout | Complete pairs /16 |
|---|---:|---:|---:|---:|---:|
| Competitive learned | 6; 3 | 8; 4 | 4; 2 | 2; 0 | 9 |
| Action frozen | 0; 0 | 0; 0 | 0; 0 | 1; 0 | 0 |
| All slow weights frozen | 0; 0 | 0; 0 | 0; 0 | 1; 0 | 0 |
| Memory disabled | 0; 0 | 0; 0 | 0; 0 | 1; 0 | 0 |
| Marginal reference | 0; 0 | 3; 1 | 3; 1 | 1; 0 | 2 |

The learned arm passes both members of D0, D1, D3, all four C pairs, and V0/V1.
All four delayed value-role tasks pass; all four five-document interference
tasks fail. All four short/long continuation pairs pass, including six-byte
repetition beyond the acquisition payload lengths. Neither task in document
order pair D2 passes. No operation/layout pair is complete. These are specific
transfer results over the frozen bytes, not arbitrary dependency understanding.

The learned fresh failures are D2-0, D2-1, V2-0, V2-1, V3-0, V3-1, O0-0,
O1-0, O1-1, O2-1, O3-0 and O3-1. Saved diagnostics flag nine wrong payloads,
three wrong operations, two wrong paths, two wrong starts and two wrong ends;
three processes fail. There are no rejected proposals or missing-EOS outcomes.
These categories overlap and do not establish causal attribution. Retention
fails T01 and T08, with one wrong payload and one wrong start/end. All learned
successful endpoints also match the canonical proposal; no noncanonical
equivalent edit is counted in its successes.

## Acquisition and actual publications

Every entry below counts correct actual first attempts before supervision,
out of 32 tasks in that round. Corrective outputs do not count as successes.

| Arm | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | Total /256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Competitive learned | 2 | 6 | 12 | 15 | 22 | 26 | 28 | 28 | 139 |
| Action frozen | 4 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 38 |
| All slow weights frozen | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 24 |
| Memory disabled | 5 | 10 | 10 | 10 | 12 | 14 | 16 | 15 | 92 |
| Marginal reference | 5 | 10 | 10 | 10 | 12 | 16 | 19 | 19 | 101 |

The auditor validates the following regular-acquisition changed-group receipt
counts against chronology, work and final control invariants. They are not a
numerical replay of every unsaved intermediate weight update.

| Arm | Start | End | Operation | Reader codec group | Reader address |
|---|---:|---:|---:|---:|---:|
| Competitive learned | 192 | 192 | 256 | 192 | 192 |
| Action frozen | 0 | 0 | 0 | 192 | 192 |
| All slow weights frozen | 0 | 0 | 0 | 0 | 0 |
| Memory disabled | 192 | 192 | 256 | 192 | 0 |
| Marginal reference | 192 | 192 | 256 | 192 | 192 |

Memory-disabled codec-group changes can be bias changes: its codec weight and
address matrix remain unchanged with zero episodic sensory/key activity.

## Same-instance learning and complete exposure

Each arm starts from the same authenticated original v6 reader numerical
tensors and fresh action weights at seed 733, with eta .01. No learned
splice-v1 endpoint weights are resumed. The same persistent combined learner
proposes, receives an actual qualified workspace verification, and then learns
from supervised local errors. Execution grades are not numerical rewards.
The neurons choose raw file, start, end, payload and operation; the host applies
only that byte splice. STOP and rejected proposals verify unchanged bytes.
There is no target matching, escaping, substituted edit or repair fallback.

Competitive free reading uses a fixed two-layer threshold circuit to commit
to one actual initial address, followed by one cursor. Correction reopens all
addresses on the same address and codec synapses using the inherited local
likelihood surrogate. It is not a derivative through the hard winner gate.
No backpropagation, reverse-Jacobian hidden credit, optimizer or separately
trained executor runs. Different free payloads can change later operation-head
learning; unchanged reader teaching does not imply identical combined histories.

Full exposure is 1,280 acquisition first attempts and 240 teacher-free endpoint
proposals, each followed by exactly one real Python verification: 1,520 total.
Acquisition makes 4,018 actual payload decisions and endpoints 868. There are
2,400 reader teacher offers and 3,200 action labels; fixed controls enable
1,920 of each. The separate before/after reader panel makes 200 calls and
supplies no actions or teachers. Across five arms, 360 tapes are ingested once
at their declared stages: 20,110 raw bytes plus 840 EOS symbols. Endpoint
workers restore those tapes rather than reingesting or teaching them.

The separate engineering cost exposure includes 25 learning pairs (twenty
ordinary lessons and five disposable caps), 120 reader offers, 65 action labels,
240 endpoint proposals and 269 actual verifications. Its free payload decisions
are 162 for acquisition/caps and 1,386 for endpoints, plus the same 200 panel
calls. Four supplied transport-category probes and hand-set caps establish
coverage, not learned skill. The preregistered cost phase traverses endpoints
before full execution; no endpoint-informed source or fit change follows it.
Earlier component, data, worker and synthetic QA remain separate exposure,
including the preserved first data-check failure and its bounded correction.

Each cost/full phase constructs five parent combined models and five worker
models; cost additionally makes five disposable cap copies. Each phase loads
the original prior once in the parent and the complete final payload once in
the worker. Its independent audit loads both payloads without constructing a
model or making a forward, teaching call or action replay. Saved source-free
worker custody verifies pending proposal → actual parent execution envelope →
same-instance acknowledgement, without transmitting grades or teachers.

## Resources and evidence limits

The cost run takes 34.3996267 seconds and its independent audit 11.1646145,
45.5642412 seconds combined. The completed
[cost decision](../../benchmarks/neuronal_splice_online_v2.cost_decision.json)
admits 583.707702804179 seconds and 1,879.48828125 MiB. This includes
500.22578740683775 projected run seconds and 83.48191539734131 audit seconds,
with the fixed 24-MiB transport allowance, 128-MiB run reserve and 32-MiB audit
reserve. Cost and full retain the same complete checkpoint payload scope.

The actual full run takes 120.4581989 monitored seconds and its audit
15.314734: **135.7729329 seconds combined**. Maximum observed owned-tree peak
is **1,522.5390625 MiB**; the audit peak is 551.79296875 MiB. Both admission and
completed execution pass the unchanged 900-second/2,048-MiB ceiling.
Acquisition's nested model, complete action and journal stages total 25.32,
54.75 and 9.18 seconds respectively; these are included in the outer duration.
The monitor samples every 500 ms and is not a continuous simultaneous-RSS trace.

The independent audit SHA256 is
`d31238776795984c6ac58fa9c6b2549323e603b795a652044bb01dacd16322db`.
The saved-only reporting summary is
`.pytest-tmp/competitive-full-result-summary-v1.json`, SHA256
`267b75ecbc71b7a1b8c666a52c07e28aac9d6bea027aa0a58178a09bfa017a0a`.
It derives these tables from finalized receipts without model or checkpoint
loading. The audit establishes saved arithmetic, actual files and outputs,
source/process custody and endpoint invariants. Exact local update arithmetic
also relies on source review and ordinary QA; unsaved intermediate tensors are
not reconstructed. Source custody is not a confidentiality sandbox.

This demonstrates a narrow improvement in learned, actually executed literal
byte edits under matched local-learning controls. All-fresh and all-retention
success remain unmet. It does not establish general code synthesis, learned
escaping, planning, autonomous recovery, broad language comprehension or a
SWE advantage. No protected benchmark or target-model comparison was attempted.
The exposed cohort is now closed development evidence.
