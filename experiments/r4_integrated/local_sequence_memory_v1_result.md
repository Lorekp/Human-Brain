# Sequence memory: learned retrieval benefit, qualification failed

The fixed five-arm run and independent audit pass their execution and custody
checks. The learned sequence model solves 4/8 taught cases, 2/4 novel-value cases,
2/4 delayed repeats and 1/4 interference case. All three matched sequence controls
solve none. This supports a useful contribution from the learned query/address
component within this system. The strict qualification nevertheless fails:
accuracy is incomplete and novel success ties the strongest byte baseline.
No gate advances or automatic additional fit follows.

The [frozen recipe](local_sequence_memory_v1.md) and source revision
`0b3d7d34f451fbbd315f472aa07732c0cd3eb0e0` bind the run. The
[decision ledger](../../benchmarks/local_sequence_memory_v1.decision.json) binds
the full source, admission, QA, audit and execution evidence. This report uses
authenticated saved JSON only; it adds no inference, teaching or execution.

## Final executed outcomes

| Arm | Taught /8 | Novel /4 | Delayed /4 | Interference /4 |
| --- | ---: | ---: | ---: | ---: |
| Learned sequence | 4 | 2 | 2 | 1 |
| Address/query frozen, codec plastic | 0 | 0 | 0 | 0 |
| All slow parameters frozen, sensory storage active | 0 | 0 | 0 | 0 |
| Sensory/key writes disabled | 0 | 0 | 0 | 0 |
| Strongest a1ab multiscale byte reference | 5 | 2 | 2 | 0 |

The learned model answers the identical-question `ac`/`ca` assignment swap
correctly in both memories. For the other swap, it emits `db` for target `bd`
and `bd` for target `db`: distinct, but both wrong. Delay preserves each answer.
After unrelated writes, only the `ca` case remains correct. The byte reference
has no jointly correct novel swap. Both models have nine correct final cases
overall, with six gains and six losses when comparing the learned model against
the reference; this is not a general accuracy improvement.

On all four taught pairs that share one memory but ask about different record
keys, the learned model returns the same answer to both questions. The correct
`k00` novel swap therefore does not establish generally query-specific binding:
a learned preference for a record or location could also contribute. The matched
frozen control establishes a contribution from the learned component, not which
query information it uses. The reversed `k02` swap makes this limitation concrete.

All learned and byte-reference outputs below terminate with learned EOS. The
address-frozen and writes-disabled controls emit an empty string with immediate
EOS on **every** row. All slow-frozen outputs hit the two-byte cap. Its JSON
strings below preserve the leading space. These columns and the two stated
constant controls account for all 100 free outputs and paired action receipts.

| ID / group | Target | Learned | Byte reference | Slow-frozen, capped |
| --- | --- | --- | --- | --- |
| 0 taught | aa | aa | aa | `" a"` |
| 1 taught | ab | aa | ab | `" h"` |
| 2 taught | bb | bc | bb | `" b"` |
| 3 taught | bc | bc | bb | `" k"` |
| 4 taught | cc | cc | cd | `" c"` |
| 5 taught | cd | cc | cd | `" h"` |
| 6 taught | dd | da | dd | `" k"` |
| 7 taught | da | da | dd | `" k"` |
| 8 novel | ac | ac | ca | `" h"` |
| 9 novel | ca | ca | ca | `" h"` |
| 10 novel | bd | db | bd | `" k"` |
| 11 novel | db | bd | dd | `" k"` |
| 12 delayed | ac | ac | ca | `" h"` |
| 13 delayed | ca | ca | ca | `" h"` |
| 14 delayed | bd | db | bd | `" k"` |
| 15 delayed | db | bd | dd | `" k"` |
| 16 interference | ac | cc | da | `" h"` |
| 17 interference | ca | ca | da | `" h"` |
| 18 interference | bd | cc | cd | `" k"` |
| 19 interference | db | cc | cd | `" k"` |

The matched address-frozen comparison leaves the codec plastic and sensory
associations active. Its failure supports the contribution of the jointly
learned query/address path, conditional on the declared codec and memory priors.
It does not isolate the query cell from the address projection, or prove that
codec learning alone could never work with another exposure schedule. The
writes-disabled control retains allocation and adjacency but zero sensory/key
activities; its enabled address/query groups make no numerical changes.

## Acquisition within the fixed exposure

Each arm receives 1,024 first attempts followed by its declared teacher offers.
The first and last windows contain 128 lessons each. Loss is the mean per offered
teacher position, including EOS.

| Arm | All first-exact /1024 | First128 exact | Last128 exact | First128 loss | Last128 loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned sequence | 435 | 0 | 70 | 2.495589 | 0.344678 |
| Address frozen | 20 | 0 | 0 | 3.208850 | 1.344807 |
| Slow frozen | 0 | 0 | 0 | 3.646847 | 3.640467 |
| Writes disabled | 0 | 0 | 0 | 5.138699 | 2.102593 |
| Byte reference | 159 | 13 | 34 | 0.274995 | 0.230540 |

Sequence losses are `sequence_mixture_negative_log_likelihood`; the byte
reference uses `half_sum_squared_potential_error`. Their magnitudes are not
comparable across algorithms. Learned first-exact counts across successive
128-lesson blocks are 0, 16, 68, 66, 72, 74, 69 and 70. Its block-mean NLL falls
from 2.495589 to 0.344678, including 0.365484 in the penultimate block. The
address-frozen control briefly obtains 20 online exact attempts in the middle
of training, then ends with premature EOS. Falling teacher loss does not by
itself establish useful free generation, convergence or a sufficient budget.

## Cost, exposure and retained state

One disposable cost workload contains 20 lessons, 60 teacher offers, five
128-slot cap fixtures, 100 actions and two motor demonstrations. Its weights
are discarded. The fresh original 2x projection plus 30-second allowance is
518.720497 seconds; the cost workload and audit take 23.870371 seconds combined.
The prospective cost decision remains unchanged after acquisition.

| Measured phase | Outer seconds | Process-tree peak MiB |
| --- | ---: | ---: |
| Disposable cost workload | 20.006048 | 697.316406 |
| Cost audit | 3.864323 | 356.097656 |
| Fixed full run | 204.730381 | 1094.074219 |
| Full saved-only audit | 8.790440 | 715.738281 |

The full run plus audit totals **213.5208209 seconds**. Audit internal work is
8.076967 seconds; its outer time includes startup. Run and audit peaks are
separate maxima, not quantities to sum. The cost-based memory estimate of
734.363281 MiB underpredicts the actual run peak by 359.710938 MiB. The run stays
within 2,048 MiB, but that estimate was not a valid upper bound for full retained
Python reports and other allocations.

Complete lesson time sums to 177.115639 seconds, leaving 27.614742 outer seconds
for other work. Named stages include final ingestion 1.272675 seconds,
serialization 0.053973, worker package 5.058671 and action package 7.986015.
These are descriptive serial measurements, not a randomized speed comparison.
Output timing includes its preceding distractor queries; worker and outer timing
also include loading, validation and serialization.

The audited full-run ledger records:

- 5,120 first attempts, 12,043 first-generation decisions, 15,360 teacher
  byte/EOS offers and 12,288 enabled teacher offers; 100 final actions plus two
  separate motor demonstrations.
- For each candidate update group, 4,096 lesson-level opportunities. Codec has
  3,072 enabled and changed groups. Address and query each have 2,048 enabled
  groups but only 1,024 changed groups. The byte baseline changes its decoder
  on 3,072 teacher positions and its query and memory on 1,023 lessons each.
- Across acquisition and final ingestion, 249,120 raw document bytes plus
  10,380 EOS symbols; 207,600 candidate tape allocation/adjacency steps,
  155,700 enabled sensory and key association opportunities each, and 51,900
  baseline writer steps. Opportunities do not imply every coordinate changed.
- 274,500 raw query bytes plus 10,980 EOS symbols, hence 285,480 query-controller
  steps and 136,448 query-eligibility steps, including the declared distractions.
- 1,073,350 actual candidate branch-codec passes and 275,850,950 output-neuron
  potentials; 414,800 initial cursor activations and 658,550 cursor advances.
  Current probabilities are reused for local errors, not recomputed by a hidden
  teacher pass. The byte baseline separately performs 6,204 output steps.

Each candidate model stores 702,864 tensor bytes and each tape 656,904 bytes.
The baseline model stores 760,908 bytes. All five models and their twelve
retained memories total 80,431,116 tensor bytes. The 128-branch conditional
cache alone can contain 921,088 bytes; errors, proposals, casts, snapshots and
external JSON receipts also consume memory. Neither tensor payload nor sparse
cursor storage is presented as the full logical-neuron or process-peak count.

## Evidence boundary and closure

The strict audit verifies 1,140 source files, five saved artifacts and 102
execution receipts. It reconstructs 3,600 saved tape slots from disclosed raw
documents, checking literal sensory/layout arrays exactly and normalized keys
against the declared float64 reference tolerance. It performs no learner,
codec or parameter-update replay. The byte reference exactly reproduces a1ab
initial and final tensors/state, all 1,024 scientific lesson receipts and all
20 final outputs, excluding only arm/time and newly declared work fields.

Implementation QA verifies 170 distinct affected cases plus 34 independent
auditor helper checks. The recipe preserves the accidental broad selector,
four failed fixture attempts, targeted recoveries, retired historical worker
contracts and explicit subset-exposure limits. Those exposures are separate
from the single cost workload and single fixed acquisition.

The full audit SHA256 is
`3b32eedbcb583d0ea3fd2e7a742c3b177f98a7fe4da3c4d5d7e6f42768a0e73c`.
The saved JSON description SHA256 is
`bb1ca936bbff535ef3883a73fc4155001ba377b975fa0351cd58d2082d5dadd2`.
The decision ledger SHA256 is
`eb6315b9ef4c78bb209bf9ed4f71b96e4959958d5d6959cb42c7445b7c560ca6`.

This is previously exposed synthetic development, not protected adjudication.
The tape retains literal bytes and preceding fixed-code keys; it is episodic
storage, not semantic compression. Fixed adjacency and a smooth identity codec
are declared generic priors. The occurrence motor supplies host-authored edit
matching and two demonstrations; execution does not establish learned general
action selection or recovery from execution feedback. Delayed cases repeat
existing memories and are not independent new tasks. The result earns a bounded
review of the remaining retrieval errors, not another automatic fit or an
unqualified reading, reasoning or software capability claim.
