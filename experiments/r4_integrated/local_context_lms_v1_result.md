# Local potential learning qualifies for the document-memory return

2026-09-06. **Both informative arms pass the fixed component criterion:**
12/12 TRAIN and 4/4 withheld two-byte outputs, each followed by learned EOS,
with every final target margin above 0.96. Both controls score zero withheld.
This earns one separately reviewed, cost-admitted return to raw-document memory
and real Python actions. It does not establish reading or learned action selection.

Source: `19f767b0335265045c08b78e4179d99e6c491b78`. The
[recipe](local_context_lms_v1.md),
[composition review](../../docs/research/local-composition-return-review.md), and
[decision record](../../benchmarks/local_context_lms_v1.decision.json) bind the
fixed four arms, 1,920 lessons per arm, rate 0.01, CPU float64 local residual
updates and one final read-only endpoint. The combinations `ab,bc,cd,da` were
excluded from TRAIN but previously exposed in development; they are not a fresh
assessment. No older decoder or perceptron arm was rerun.

## Final endpoint

| Arm | TRAIN / 12 | Withheld / 4 | Minimum final margin | Useful criterion |
| --- | ---: | ---: | ---: | --- |
| Lexical | 12 | 4 | 0.9604450609261397 | Pass |
| Permuted | 12 | 4 | 0.9610096413450895 | Pass |
| Zero context | 1 | 0 | -0.022608117955676632 | Control |
| Frozen | 0 | 0 | 0 | Control |

Each informative arm satisfies all 48 target-position margins strictly above
0.10, including EOS, and strictly exceeds both controls on withheld exact
outputs. Margins come from the same free generation's saved potentials; missing
phases would fail. There were no extra target-conditioned forward calls.
Minimum first-byte, second-byte and EOS margins are respectively
`0.9604937438671692, 0.9604450609261397, 0.9997628640365755` for lexical and
`0.9610096413450895, 0.9610119966142713, 0.9999769981227904` for permuted.

Both informative arms produce all sixteen target pairs exactly. Zero context
produces `db` for every final query, with EOS, explaining its single TRAIN
success. Frozen produces bytes `00 00` and reaches the cap without EOS on every
query. The decision record preserves all 64 outputs and per-position margins.

## Acquisition and exposure

Every lesson records a genuine read-only first attempt before three explicit
byte/EOS teacher offers. All 257 output neurons receive their residual target;
there is no alphabet mask, winner-only update or model autocorrection. The loss
is `half_sum_squared_potential_error`, one half the sum of all 257 squared
potential residuals. It is neither cross-entropy nor a mistake count, and must
not be compared numerically with those earlier losses.

| Arm | First attempts exact / 1,920 | First 128 exact | Last 128 exact | First 128 mean loss | Last 128 mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | 1,906 | 114 | 128 | 0.2492473004 | 0.0001627543 |
| Permuted | 1,908 | 116 | 128 | 0.2492734065 | 0.0001630273 |
| Zero context | 0 | 0 | 0 | 0.3430794967 | 0.2524667198 |
| Frozen | 0 | 0 | 0 | 0.5 | 0.5 |

Lexical has 1/12 and 9/12 first successes in cycles one and two; permuted has
1/12 and 11/12. Both have 12/12 in every cycle from three through 160.
Zero and frozen have no first-attempt success in any cycle. Each updating arm
has one initial cap stop and 1,919 EOS stops; frozen has 1,920 cap stops.
All-cycle exact counts and each consecutive 128-lesson block are saved in the
decision record. These online curves use changing weights, not added endpoint
probes.

Last-128 mean losses for first byte / second byte / EOS are:

| Arm | First byte | Second byte | EOS |
| --- | ---: | ---: | ---: |
| Lexical | 0.0002441718357 | 0.0002440797049 | 1.122371066e-8 |
| Permuted | 0.0002444755173 | 0.0002446060636 | 1.773676181e-10 |
| Zero context | 0.3786357805 | 0.3787643790 | 4.120674604e-17 |
| Frozen | 0.5 | 0.5 | 0.5 |

The fixed run accounts for 7,680 first attempts, 64 final generations and
23,232 generation decisions. It offers 23,040 byte/EOS labels and 5,921,280
scalar neuronal residuals, including frozen offers. Of those, 17,280 teacher
steps and 4,440,960 scalar residual opportunities are enabled. All 17,280
enabled steps change weights. There are 57,530 nonzero offered residuals,
of which 51,770 are enabled. Each arm receives 5,760 label offers and
1,480,320 scalar residual offers; frozen receives no enabled updates.

## Independent represented-weight certificate

The saved-only auditor separately sums the final represented binary64 TRAIN
coefficients as exact rationals. It obtains these maximum absolute output
potential residuals across the TRAIN inputs and all output neurons:

| Arm | Exact maximum TRAIN residual | Decimal epsilon | Missing-pair margin lower bound |
| --- | --- | ---: | ---: |
| Lexical | 1569523886315125 / 144115188075855872 | 0.010890759726788803 | 0.9346554416391165 |
| Permuted | 759402409727681 / 72057594037927936 | 0.01053882550294373 | 0.9367670469821869 |

Lexical's witness is phase zero, pair `ac`, output neuron 97; permuted's is
phase one, pair `db`, output neuron 98. Both epsilon values are below 1/6.
The generic three-edge missing-row identity from the review, together with
binary64 dot-product roundoff envelopes of `7.526480272769184e-14` and
`7.526396559517379e-14`, certifies positive withheld margins from TRAIN
coefficients alone. This is an a posteriori certificate of these saved weights,
distinct from the measured free-generation margins. It is not a proof of every
floating training operation, nor a guarantee for arbitrary continuous memory
inputs replacing the authored factors.

## Resource and custody record

The run's external watchdog records **33.8972996 seconds** and a peak process-tree
working set of **368.9609375 MiB**. The independent saved-only audit has its own
watchdog: **4.1704137 seconds** and **303.546875 MiB**. Their aggregate elapsed
time is **38.0677133 seconds**, below the fixed 120-second computation ceiling;
each tree remains below 2,048 MiB. The audit's separate timeout was 86 seconds,
the floor of the remaining budget. Startup, imports and logging are included;
orchestration gaps are excluded. Sequential memory peaks are not summed.
The audit's internal 3.6171923-second timer excludes startup and is not the
aggregate-budget authority. Durations use the recorded performance counter.

Each model accounts for 644 neurons, 74,787 learned output synapses, 585
allocated fixed synapses (578 nonzero), three clock initialization values and
602,952 serialized tensor bytes. Its 4,396-byte step already includes the
12-byte next-clock state. The 2,056-byte residual and supplied 768-byte float64
context are additional. Python containers, copies and transient arithmetic are
covered by process monitoring.

The passing audit verifies twelve source hashes and five artifact hashes,
matched initial states, schedules and teacher exposures, first-before-teacher
hash chains, read-only final calls, sparse potential/residual semantics and
reconstruction of final weights from saved residual corrections. It performs
no project/model imports, new inference or teaching. Unlisted sparse potentials
and residuals mean exact zero; all 257 residual opportunities remain counted.

| Evidence | SHA-256 |
| --- | --- |
| `.pytest-tmp/local-context-lms-v1-run/report.json` | `cdf609014fc11a8c46d4a91abe1b77a783fe2fc646592ae5f3f5a52178069e07` |
| `.pytest-tmp/local-context-lms-v1-run/completion.json` | `6afca735005b4ae3240f95ff0087d04d50cc7f7943faead838babdc06e7681e7` |
| `.pytest-tmp/local-context-lms-v1-run.launch.json` | `2813984fc207871db456b9715b42e84c5a0cdf94409e305b0689087ea9e0dee9` |
| `.pytest-tmp/local-context-lms-v1-independent-audit.json` | `62820c8df486b79a43be9c161c24c374f2e9ae462b2dcb83cfc40b014ddcb5b1` |
| `.pytest-tmp/local-context-lms-v1-audit-watchdog.launch.json` | `515a84097781224609a98d0b4d6fba3531d6a5188890fb874c086ceb5370277d` |

The decision record binds every source and saved artifact hash, including
initial/final tensors, the complete lesson log and schedule. Frozen source,
recipe, admitted exposure and thresholds remain unchanged.

## Earned return and limits

Close this successful component comparison. Return directly to the narrowed
two-byte document-memory/action workload with fresh output weights, explicit
document/query input-code ownership and pre-update upstream neuronal teaching.
Remove the authored answer factors; require the same question to retrieve
different values from different retained memories. Requalify complete-workload
cost with the original margin and preserve the five learning controls, source
removal, swaps, delay/interference and actual execution custody.

The fixed neuronal clock and generic context gating remain supplied priors.
This result establishes local acquisition and composition over an authored
factor representation, not document comprehension, useful source-absent
writing, variable-length generation, an entirely learned action policy or a
software benchmark pass. The legacy motor's host matching remains disclosed;
teacher byte corrections are separate from execution-feedback learning. No
further output diagnostic or automatic fit is authorized by this result.
