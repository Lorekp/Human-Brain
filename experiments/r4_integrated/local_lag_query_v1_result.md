# Direct lag-query retrieval: improved binding, strict criterion failed

The fixed comparison completes with **7/8 taught, 4/4 novel, 4/4 delayed and
2/4 interference successes**. All three matched candidate controls score zero.
The recurrent-query sequence reference scores 4/2/2/1 and the multiscale byte
reference 5/2/2/0, exactly reproducing their prior scientific trajectories.
The new learner solves both novel assignment swaps and three of four taught
different-question pairs. This is useful bounded evidence of learned
query-to-record selection. It still fails the preregistered criterion because
one taught case and two interference cases fail. No gate advances or automatic
additional fit follows.

The [frozen recipe](local_lag_query_v1.md) and source revision
`8ef70d6f512caffc0cbc9eac2f19b772298447b2` bind this run. The
[decision ledger](../../benchmarks/local_lag_query_v1.decision.json) records the
source, admission, QA and execution evidence. This report uses authenticated
saved JSON only, adding no inference, teaching or action execution.

## Executed outcomes and controls

| Arm | Taught /8 | Novel /4 | Delayed /4 | Interference /4 |
| --- | ---: | ---: | ---: | ---: |
| Learned lag query | 7 | 4 | 4 | 2 |
| Address frozen, codec plastic | 0 | 0 | 0 | 0 |
| All slow parameters frozen, sensory storage active | 0 | 0 | 0 | 0 |
| Sensory/key writes disabled | 0 | 0 | 0 | 0 |
| Prior recurrent-query sequence reference | 4 | 2 | 2 | 1 |
| Strongest multiscale byte reference | 5 | 2 | 2 | 0 |

Both identical-question novel swaps, `ac`/`ca` and `bd`/`db`, are correct in
both memories. Delay preserves all four answers. Following unrelated writes,
the `ac` and `bd` cases return distractor value `cc`; `ca` and `db` remain
correct. The remaining taught error asks for `bc` and receives `bb`.

Every learned and reference output below ends in EOS. Address-frozen and
writes-disabled emit empty immediate EOS on all twenty cases each. Every
slow-frozen output hits the cap; quoted strings preserve its leading space.
These columns and the two stated constant controls cover all 120 free outputs
and paired action receipts.

| ID / group | Target | Learned | Sequence reference | Byte reference | Slow-frozen, capped |
| --- | --- | --- | --- | --- | --- |
| 0 taught | aa | aa | aa | aa | `" a"` |
| 1 taught | ab | ab | aa | ab | `" a"` |
| 2 taught | bb | bb | bc | bb | `" b"` |
| 3 taught | bc | bb | bc | bb | `" b"` |
| 4 taught | cc | cc | cc | cd | `" c"` |
| 5 taught | cd | cd | cc | cd | `" c"` |
| 6 taught | dd | dd | da | dd | `" d"` |
| 7 taught | da | da | da | dd | `" d"` |
| 8 novel | ac | ac | ac | ca | `" h"` |
| 9 novel | ca | ca | ca | ca | `" h"` |
| 10 novel | bd | bd | db | bd | `" h"` |
| 11 novel | db | db | bd | dd | `" h"` |
| 12 delayed | ac | ac | ac | ca | `" h"` |
| 13 delayed | ca | ca | ca | ca | `" h"` |
| 14 delayed | bd | bd | db | bd | `" h"` |
| 15 delayed | db | db | bd | dd | `" h"` |
| 16 interference | ac | cc | cc | da | `" h"` |
| 17 interference | ca | ca | ca | da | `" h"` |
| 18 interference | bd | cc | cc | cd | `" h"` |
| 19 interference | db | db | cc | cd | `" h"` |

For taught pairs sharing one memory, changing the question now changes the
answer correctly in pairs 0/1, 4/5 and 6/7. Pair 2/3 still answers `bb` twice.
Saved initial target-start log-odds also reverse sign in the three successful
pairs: approximately 0.110756 to -0.036814, 0.043264 to -0.016104, and 0.063144
to -0.007015. The failed pair moves from 0.171669 to 0.124734, retaining the
same preference. These are posthoc differences between saved log probabilities;
record keys and value-start positions are external analysis labels, never
additional learner inputs or new queries.

The address-frozen comparison supports a contribution from learned A under
the shared codec and literal-memory priors. The reference comparison changes
query representation, address geometry and query-credit structure together;
it does not isolate normalization alone or establish a general language
representation. Novel combinations were withheld from this fresh acquisition
but had already been exposed during development.

## Acquisition at the fixed exposure

Each arm receives 1,024 genuine first attempts followed by three teacher offers,
including EOS. Loss is mean loss per offered position.

| Arm | All first-exact /1024 | First128 exact | Last128 exact | First128 loss | Last128 loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned lag query | 472 | 0 | 85 | 2.310500 | 0.338597 |
| Address frozen | 32 | 0 | 0 | 3.212563 | 1.347196 |
| Slow frozen | 0 | 0 | 0 | 3.635073 | 3.624714 |
| Writes disabled | 0 | 0 | 0 | 5.138699 | 2.102593 |
| Sequence reference | 435 | 0 | 70 | 2.495589 | 0.344678 |
| Byte reference | 159 | 13 | 34 | 0.274995 | 0.230540 |

The five sequence-family arms use
`sequence_mixture_negative_log_likelihood`; the byte reference uses
`half_sum_squared_potential_error`. Their loss magnitudes are not comparable
across algorithms. Learned first-exact counts in successive 128-lesson blocks
are 0, 13, 70, 69, 83, 71, 81 and 85. Mean NLL is 2.310500, 1.262676,
0.710686, 0.512355, 0.430281, 0.387640, 0.359988 and 0.338597. Late loss is
still decreasing descriptively; this does not establish convergence, prove
that more exposure would repair failures, or authorize an extension.

## Cost, streaming receipts and actual work

One disposable cost run uses 24 lessons, 72 teacher offers, six 128-slot cap
fixtures, 120 actions and two demonstrations. Its fitted states are discarded.
Admission uses the original 2x model/journal/cap maxima, 2x other time and
30-second allowance: **644.1842905 projected seconds**. The prospective cost
decision remains unchanged after the full run.

| Measured phase | Outer seconds | Process-tree peak MiB |
| --- | ---: | ---: |
| Disposable cost run | 22.2554262 | 780.765625 |
| Cost saved audit | 7.4529197 | 987.562500 |
| Fixed full run | 254.9730419 | 907.88671875 |
| Full saved audit | 16.8325430 | 988.67578125 |

Full run plus audit totals **271.8055849 seconds**, within the 900-second ceiling.
Audit internal work is 16.149973 seconds; its outer measurement includes startup.
Run and audit peaks are separate maxima, not quantities to sum. The run-specific
estimate of 832.48046875 MiB underpredicts its actual peak by 75.40625 MiB.
The final admission estimate, 1,617.7407303 MiB, includes the larger historical
audit allowance and covers both actual peaks under the 2,048-MiB watchdog.
The engineering estimates are not mathematical allocator bounds.

All 6,144 complete scientific rows are preserved in a 221,473,955-byte JSONL
journal and streamed by the checker. Maximum line size is 52,489 bytes; maximum
writer row tree is 95,890 bytes and reader row tree 96,746 bytes, each below its
262,144-byte guard. No scientific fields are dropped. Model/receipt lesson time
sums to 223.767487 seconds; measured journaling adds 8.692597 seconds, leaving
22.512958 outer seconds outside those two stages. Worker package time is
5.183427 seconds and action package time 9.444854 seconds. These stages can
overlap other named totals; they are descriptive timings, not a randomized
speed comparison. Per-output timing includes preceding distractor queries.

The full audited ledger records:

- 6,144 first attempts and 15,016 first-generation decisions; 18,432 teacher
  byte/EOS offers, of which 15,360 are enabled; 120 final actions and two separate
  motor demonstrations.
- Across the four lag arms, 4,096 codec and 4,096 address group opportunities.
  Codec has 3,072 enabled and changed groups. Address has 2,048 enabled groups
  but 1,024 changed groups. The sequence reference separately changes each of
  its codec, address and query groups on all 1,024 lessons. The byte reference
  changes 3,072 decoder steps and its query/memory on 1,023 lessons each.
- 298,944 raw document bytes plus 12,456 EOS symbols; 259,500 tape allocation
  and adjacency steps, 207,600 enabled sensory/key association opportunities
  each, and 51,900 byte-reference writer steps.
- 329,400 raw query bytes plus 13,176 EOS symbols; 228,384 candidate lag shifts,
  8,784 query population normalizations, and 9,210,691,584 candidate address
  projection-synapse evaluations. Reference recurrent work remains separate:
  114,192 controller steps and 56,576 eligibility steps.
- 4,294,967,296 candidate local address-synapse contributions, including frozen
  group computations; 1,379,500 branch-codec evaluations and 354,531,500 output
  potentials; 518,500 cursor initializations and 861,000 cursor advances.
  Computed opportunities are distinct from enabled publication and actual
  numerical changes. The byte reference separately performs 6,204 output steps.

Each lag model stores 4,757,776 tensor bytes; sequence and byte references store
702,864 and 760,908 bytes. All six models and their twelve retained memories
total 105,236,476 tensor bytes. Literal sensory tapes, temporary arrays, local
contributions, snapshots, parser objects and receipts remain explicit costs;
tensor payload alone does not describe process memory.

## Evidence and limits

Nineteen distinct model/integration/transport QA cases pass across twenty case
executions. They contain 54 authored teacher offers and zero acquisition or
action execution, as recorded in the frozen recipe. The independent checker
has 36 passing stdlib-only helper cases. Its initial helper CONFIG failure
(missing `receipt_format`) and the 23 preceding checks are preserved; the
checker was corrected before admission, without reexecuting a model experiment
or relaxing its scientific criterion.

Two saved-only result-recorder attempts failed on doctor-directory access
permissions. Root copied three doctor receipts byte-identically to shared
ignored paths and bound original/copy hashes and the access-recovery record.
The third recorder attempt passed in 1.3189 seconds. This was tooling recovery
only: no training or independent audit was repeated.

The full audit verifies 1,143 source files, six artifact hashes and 122 execution
receipts. Both historical references match authenticated initial/final states,
all 1,024 scientific lesson receipts and twenty final outputs, excluding only
arm/time, newly declared work fields and empty byte-reference step arrays.
The source-free worker receives saved arrays and query-only requests; it does
not receive document or target fields. This routing claim is not a filesystem
sandbox: literal content is retained in neuronal sensory associations.

Key evidence hashes are:

- Final decision, binding 33 evidence hashes: `628fba2e2d0436c61d62314084727fe3ab45c904388185abed40723be7db7a64`.
- Full independent audit: `064da6dde81f3faee42f02b68737d74378843aa388df26c069fdfca5ee0dffcf`.
- Saved-only description: `5c5057a9771888c4fc9276fcef05fe06c55e257722e609b47db5a49d2004a0d1`.
- Complete lesson journal: `f703cba8aeefcda8a394fc7b0e5f914bf93dca62869c666e71536aa892b7b32a`.

The result establishes useful retrieval and question dependence on this small,
repeated development workload. Literal storage, a generic identity-copy prior,
two-byte answers and a supplied host edit motor do not establish semantic
compression, open-ended reading, independently learned action schemas or
general software engineering. The strict criterion remains false. Close this
fixed fit; any further work requires its own selected and bounded review.
