# Neuronal splice online v1: completed result

Completed and independently audited on 2026-09-06. The neuronal learner solves
12/32 fresh tasks, including nine real edits, and 12/16 retained tasks. The
strict selected criterion fails. Local action learning produces a useful
narrow gain, but complete paired transfer remains unmet. Close this fixed run;
do not add rounds, tune its seed/rate, or replay its endpoints as fresh evidence.

The [decision bank](../../benchmarks/neuronal_splice_online_v1.decision.json)
binds the complete saved report, independent audit, monitor receipts, admission,
source manifest and frozen roster. Source freeze is `7fac85d`; full execution
used its unchanged source after cost decision bank commit `0053c0c`.

| Arm | Fresh edits /28 | Fresh STOP /4 | Retained edits /12 | Retained STOP /4 | Old reader before → after /20 |
|---|---:|---:|---:|---:|---:|
| Learned | 9 | 3 | 8 | 4 | 18 → 18 |
| Action frozen | 0 | 1 | 0 | 4 | 18 → 18 |
| All slow weights frozen | 0 | 2 | 0 | 4 | 18 → 18 |
| Memory disabled | 1 | 3 | 4 | 4 | 0 → 0 |

Learned first attempts improve from 5/32 in round one to 19/32 in each of the
last two rounds. Its complete eight-round sequence is 5, 10, 10, 10, 12, 16,
19, 19: 101/256 correct before correction. Action-frozen and slow-frozen each
score 40/256; memory-disabled scores 92/256. Corrective teacher outputs never
count as successful first attempts.

The system exceeds each control on fresh edits and preserves all eighteen
originally correct reader cases. It does not pass all fresh pairs or all
retained edits. Its nine successful fresh edits occur in P0-0, P1-1, P3-0,
L0-0, L0-1, L3-0, L3-1, O0-0 and O2-0. No V assignment-swap endpoint succeeds.
Four of sixteen contrast pairs pass both members: L0, L3, O0 and O2.

The authenticated saved-action breakdown sharpens the gap: every one of the
28 fresh SPLICE tasks selected the canonical path, start and end. Nineteen
selected the wrong payload; one KEEP task incorrectly selected SPLICE. There
were zero endpoint rejections or failed processes in the learned arm. All four
retention failures had wrong payloads; one also had the wrong start/end.
Difference categories overlap and describe saved outputs rather than causal
attribution. The raw summary is
`.pytest-tmp/neuronal-splice-online-v1-outcome-summary.json`, SHA256
`8be8f588acd0db1428a636e1581dd28c5ea3a4a1efc594489489f3bbdbbef6e1`.

## What ran and what the evidence establishes

The same persistent learner chose each acquisition proposal, retained its own
synapses through actual execution, then received direct local correction.
No backpropagation, reverse-Jacobian error transport, optimizer or separately
trained executor ran. The completed v6 reader tensors initialized all four
modes identically; action synapses started fresh. Every free head and the
reader received the same raw request. The host applied only the chosen byte
splice, with no target matching, escaping, substituted edit or repair.

Full exposure is 1,024 learning first attempts plus 192 teacher-free endpoint
proposals, each with one real Python verification: 1,216 total. Acquisition
made 1,124 actual payload decisions and endpoints made 268. There were 1,920
reader teacher offers and 2,560 action-head labels, with 1,440 and 1,280 enabled
respectively by the fixed control modes. The separate before/after reader panel
made 160 calls and supplied no teachers or actions. All 256 episode tapes were
ingested once at their declared stages: 14,552 raw bytes plus 608 EOS symbols.

The fixed engineering cost probe is separate exposure: 216 verifications,
20 learning pairs including four disposable caps, 96 reader offers and 52
action labels. It made 84 regular/cap free payload decisions plus 192 endpoint
decisions and the same 160 reading-panel calls. Its hand-set caps and transport
category probes do not count as learned skill. Earlier QA exposure remains in
its individual banks, including the single STOP worker plumbing check.

The independent auditor streamed the saved chronology, actual applied files,
raw process outputs, neuron receipts, mode invariants, source-free worker
identity and final tensors. It performed no neuronal forward, teaching or
action replay. It does not independently reconstruct unsaved intermediate
synapse updates; source review and the focused arithmetic QA establish that
part of the evidence. Process/source custody is not a confidentiality sandbox.

## Resources and limits

The first cost probe completed in 23.8957297 seconds and its independent audit
in 7.2562322 seconds. Their completed projection admitted 418.09543180200296
seconds and 1,500.5 MiB for the full workload, including audit and reserves.
The [cost decision](../../benchmarks/neuronal_splice_online_v1.cost_decision.json)
preserves full arithmetic and authenticated inputs.

The actual full run took 84.74331 monitored seconds and its audit 11.3985175:
96.1418275 seconds combined. Maximum observed owned-tree peak was 1,234.234375
MiB; the audit peak was 481.8359375 MiB. These pass the unchanged 900-second,
2,048-MiB limits. Sampling every 500 ms can miss brief child processes, so the
reported monitor measure is not continuous simultaneous RSS.

The [saved resource review](../../docs/research/neuronal-splice-online-v1-cost-review.md)
attributes 18.64 seconds to acquisition model stages, 41.24 to complete action
stages, and 2.78 to complete learning journals. Duplicate endpoint receipt
encoding accounts for at most its entire 0.468-second stage, so no simulation
change is selected on that basis. Preserve the existing vectorized implementation.

This demonstrates local learning of some raw byte-edit choices in a small,
authored literal-copy domain. It does not establish general code synthesis,
learned escaping, planning, autonomous recovery, broad language comprehension,
or a SWE benchmark advantage. No protected benchmark or target-model comparison
was attempted. The next decision must use the saved failure distribution and
cost evidence before any new acquisition is selected.
