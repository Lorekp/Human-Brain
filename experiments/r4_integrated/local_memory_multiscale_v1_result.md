# Fixed multiscale writer: modest prior gain, qualification failed

The fixed comparison and independent saved audit passed execution checks, but
the learner failed the [preregistered capability criterion](local_memory_multiscale_v1.md).
Multiscale tagged improves novel-value and delayed success from 1/4 to 2/4
relative to its scalar predecessor. Candidate writer-frozen and aggregate match
those scores, so this is a modest benefit of the changed fixed representation
prior, not demonstrated learned-writing or tagged-credit advantage. No arm solves
both members of an assignment-swap pair, and interference success remains zero.

Frozen source: `a1ab5ee0d82c5cc047392ec8e6b22be3a4386438`.
The one fixed fit is closed. No additional fit, rate/bank sweep or architecture
adoption follows automatically; a bounded binding-architecture review is pending
orchestrator selection. These are exposed development tasks, not protected SWE
evaluation or a general language/neuronal action-policy result.

## Final executed results

| Arm | Fit /8 | Novel /4 | Delayed /4 | Interference /4 |
| --- | ---: | ---: | ---: | ---: |
| Multiscale tagged | 5 | 2 | 2 | 0 |
| Multiscale aggregate | 5 | 2 | 2 | 0 |
| Multiscale writer/maps frozen | 5 | 2 | 2 | 0 |
| Multiscale fully frozen | 0 | 0 | 0 | 0 |
| Multiscale writes disabled | 1 | 0 | 0 | 0 |
| Immediate scalar tagged reference | 5 | 1 | 1 | 0 |

Tagged gains novel case 10 (`bd`) and its delayed copy 14 over the scalar
reference, losing no correct case. Those are one fresh case and its retention
check, not two independent transfer successes. For the same-question `ac/ca`
swap, tagged emits `ca/ca`; for `bd/db`, it emits `bd/dd`. Thus one assignment is
correct in each pair, but neither pair is jointly solved. Delayed outputs repeat
these results. Interference gives `da/da` and `cd/cd`, all wrong.

Writer-frozen matches tagged's first sixteen outputs exactly and differs only on
wrong interference outputs 16/17 (`dd/dd` instead of `da/da`). Aggregate differs
only at wrong novel/delayed cases 11/15 (`cd` instead of `dd`). Their correctness
matches tagged on all twenty cases. Changing wrong bytes is not useful binding.
The full saved [description](../../.pytest-tmp/local-memory-multiscale-v1-description.json)
contains every output, target, paired comparison and 128-lesson block.

The fixed bank changes both rate heterogeneity and average update fraction, so
this result does not isolate heterogeneity as the cause. Final outputs and
weights do not measure prefix identifier retention or address geometry; no new
temporal-contrast claim is made from them. Failure at this exposure does not
establish convergence or prove that useful information is absent.

## Learning, exposure and parity

Each arm received exactly 1,024 lessons at rate 0.01, with fresh seed-733
initialization. No cost-trained weights were reused. The five candidate arms
share 32 writer fractions of 0.5, 16 of 0.125 and 16 of 0.03125. The reference
uses scalar 0.5; query dynamics, raw bytes, direct read/value credit and local
LMS output remain unchanged. No optimizer or BPTT is used.

| Arm | First exact, all 1,024 | First/last 128 exact | First/last 128 mean teacher loss |
| --- | ---: | ---: | ---: |
| Tagged | 159 | 13 / 34 | 0.274995 / 0.230540 |
| Aggregate | 161 | 13 / 35 | 0.274996 / 0.229979 |
| Writer/maps frozen | 161 | 13 / 35 | 0.275010 / 0.229537 |
| Fully frozen | 0 | 0 / 0 | 0.500000 / 0.500000 |
| Writes disabled | 100 | 12 / 13 | 0.276725 / 0.257694 |
| Scalar reference | 141 | 14 / 32 | 0.275414 / 0.233903 |

Loss is the mean of the three pre-update half-sum squared potential errors,
not cross-entropy or free-answer accuracy. All updating arms' last 128 first
attempts end with EOS; fully frozen remains at the byte limit. These descriptive
windows do not authorize more training or establish a plateau.

The audited run totals 6,144 first attempts, 18,432 byte/EOS teacher offers,
15,360 enabled decoder opportunities and changed decoder steps, 307,200 document
byte/EOS steps, 319,488 acquisition query steps and 23,088 final query steps.
All 4,737,024 neuron residual offers are counted, including 3,947,520 enabled;
48,957 residuals are nonzero, 45,885 enabled. All lessons compute actual read and
direct value credit; 5,120 use multiscale writers and 1,024 scalar writers.
There are 120 final real actions and two motor demonstrations. Teacher answers
follow first attempts and are not autonomous recovery or execution feedback.

The audit confirms the scalar reference reproduces the immediate value-credit
tagged initial/final states, all 1,024 non-timing lesson receipts and all twenty
outputs exactly. Writes-disabled likewise reproduces its prior common parameters,
zero memories, numerical receipts and outputs, excluding only its fixed rate
buffer and declared metadata. Candidate writer-frozen/frozen have different
forward dynamics and are not required to reproduce old scalar trajectories.
All common initial tensors match; candidate rate buffers remain unchanged.

## Resources and evidence quality

| Process | Outer seconds | Observed tree peak MiB |
| --- | ---: | ---: |
| Full comparison | 370.1265055 | 1090.60546875 |
| Full saved audit | 4.4154719 | 688.96875 |
| Aggregate computation | 374.5419774 | Separate process peaks above |

The full audit's internal time is 3.7814005 seconds. Complete lessons consume
342.7057685 seconds. Final ingestion is 2.8756780 seconds, serialization 0.1614763,
worker packaging 8.7964553 and action packaging 9.6761755; these are included
stages, not additional time on top of the outer run. The conservative recorded
parent-plus-worker bound is 1101.359375 MiB. The watchdog samples simultaneous
owned process peak working sets and can miss brief children; peaks are not summed
across the separate run and audit. No randomized simulation speedup is claimed.

Each candidate model stores 760,908 tensor bytes versus the reference's 760,652;
the only extra array is the 256-byte fixed rate vector. Six models plus all
retained state contain 232,400,120 logical tensor bytes. No per-stream tag/activity
bank was added. Imports, copies, hashing, output and execution remain budgeted.

### Cost admission and preserved auditor failure

One disposable cost workload completed in 27.8968871 seconds at observed tree
peak 1041.61328125 MiB: 24 lessons, 72 teacher offers, 60 enabled opportunities,
six capped predictions, 120 actions and two demonstrations. Its weights were
discarded. The original saved auditor failed because its tensor-byte walker
rejected the new tuple metadata; it took 2.9908983 seconds at 206.13671875 MiB.
No model or cost workload was rerun. The original admission, checkers, failed
receipt and model artifacts remain preserved.

Versioned v2 auditors recursively count tuple/list containers while preserving
tensor/type/finiteness checks. Thirty-two authored counter cases passed without
models or endpoint reads; the tested helper AST is identical in both final
checkers. The explicit [correction receipt](../../.pytest-tmp/local-memory-multiscale-v1-auditor-correction-v2.json)
binds original/replacement checkers, failed audit, original admission and exact
saved cost hashes. It also records the added full-checker enforcement of declared
writes-disabled predecessor parity. No score or resource criterion was relaxed.

The corrected saved cost audit passed in 3.4484223 outer seconds at 447.19140625
MiB. Original 2x margins plus 30 seconds projected 862.5138055639769 seconds and
1052.5234375 MiB, below the unchanged 900-second/2,048-MiB ceilings. The separately
bound full admission retained original auditor identities and named the corrected
checkers. The cost decision's prospective `full_fit_ran:false` remains unchanged.

### QA and custody

Before freeze, 196 distinct affected tests passed: 139 decoder-file and 57 memory
cases, including exact scalar old-source trajectories, numerical local-credit
checks, fixed rates/controls and the new source-free worker. Preserve the initial
decoder fixture failure (tuple metadata), its one-case recovery, the memory OS
temporary-directory setup failure/recovery and worker cache-write warning.
Doctor, receipts and named disposable exposure are recorded in the frozen recipe;
the 32 later auditor-counter cases are separate coverage, not model training.

The full independent audit verifies 1,151 sources, five artifact hashes, 122
execution receipts, fresh initialization, process ownership, absolute UTF-8
worker routing, saved state and reference/control parity. It does not independently
recompute every local update. No project model or action child was invoked by the
saved audit/description. Source absence is an input-routing boundary, not a
filesystem sandbox. The host-authored occurrence motor remains an explicit narrow
action scaffold and does not establish neuron-only action selection.

Primary evidence, with full hash custody retained in the
[result decision](../../benchmarks/local_memory_multiscale_v1.decision.json) and
the unchanged [prospective cost decision](../../benchmarks/local_memory_multiscale_v1.cost_decision.json):

- Full audit: `.pytest-tmp/local-memory-multiscale-v1-run-independent-audit.json`,
  SHA256 `a9a1ebfec69f909124f8583834469739fdf13b959999ddac6b03aa0dea87dc48`.
- Saved description: `.pytest-tmp/local-memory-multiscale-v1-description.json`,
  SHA256 `45a83cc3b48772b81a6606873e2a83fdb008de619ba01fbd5b31fbc23fa0ee30`.
- Auditor correction: SHA256
  `3b7144dff6ec0de92fabdb744bc9742a67ad3033f5584e1697fecd6ef8991a90`.
- Full admission: SHA256
  `7aca122803eba0652aef6b196d96413a0e4d94f49c7ab30160e314ee99a07648`.

Strict qualification, learned-writing advantage and tagged-over-aggregate
advantage are false. The modest scalar-reference gain is retained as evidence;
the fixed experiment is closed without claiming useful general binding.
