# Context learnability: certified TRAIN fit, incomplete composition

2026-09-06. **Both local reference models pass their TRAIN mistake certificate,
but every useful-endpoint criterion fails.** The reference solves all twelve
taught pairs with 16 mistakes in lexical order and 15 in permuted order, below
the bound of 54. It solves only two or three of the four withheld combinations.
The existing decoder reaches 5/12 TRAIN and 0/4 withheld under either order
after 768 lessons. Neither the longer exposure nor the fixed permutation
produces the required compositional endpoint.

Source: `30fe85c2417ba87902cd4466f79ff9c7932fffd4`.
The [recipe](local_context_learnability_v1.md),
[mathematical review](../../docs/research/local-decoder-learnability-review.md)
and [decision record](../../benchmarks/local_context_learnability_v1.decision.json)
preserve the fixed six trajectories, exact integer reference rule, two
checkpoints and unchanged decoder. The four combinations `ab,bc,cd,da` remain
previously exposed development probes excluded from TRAIN, not a fresh assessment.

## Both predeclared checkpoints

Each score requires exact two-byte output followed by learned EOS. Each cell
below shows TRAIN out of twelve / withheld out of four.

| Arm | 384 lessons | 768 lessons |
| --- | ---: | ---: |
| Decoder, lexical | 4 / 0 | 5 / 0 |
| Decoder, permuted | 4 / 0 | 5 / 0 |
| Reference, lexical | 12 / 2 | 12 / 2 |
| Reference, permuted | 12 / 3 | 12 / 3 |
| Reference, zero context | 1 / 0 | 1 / 0 |
| Reference, frozen | 0 / 0 | 0 / 0 |

The lexical decoder at 384 exactly reproduces the authenticated context-v2
random-correct trajectory: first outputs, teacher losses, state hashes, complete
tensors and checkpoint outputs. Checkpoint probes are read-only; their outcomes
do not change teaching or stopping. All four candidate useful-endpoint flags
are false because none reaches 12/12 TRAIN and 4/4 withheld.

Reference errors are identical at both checkpoints. Lexical reference emits
`ac` for `ab` and `ca` for `da`; permuted reference emits `ad` for `ab`. Its other
withheld outputs are correct. The reference's completed TRAIN learning therefore
does not select a uniformly correct factor composition on these four probes.
Both negative controls score zero withheld; superiority to them is insufficient
without the fixed 4/4 requirement.

A separate saved-synapse decomposition identifies the three failed reference
byte decisions, in exact integer score units:

- Lexical `ab→ac`: at the second byte, `b` scores 1 while `c` scores 2 from
  the phase baseline and the coactive first-factor connection.
- Lexical `da→ca`: at the first byte, `c` and `d` both score 1; the declared
  lowest-ID tie rule chooses `c`. Phase and opposite-factor contributions
  support that competing score.
- Permuted `ab→ad`: at the second byte, `b` scores 0 while `d` scores 2 from
  phase and first-factor contributions.

Thus a TRAIN-correct separator can retain connections to the other factor that
mislead these withheld decisions. The saved reference states are identical at
384 and 768. Once every TRAIN input is correct, further replay or permutation
of those same inputs cannot change this mistake-only learner. This is a
description of the saved scoring mechanism, not a new intervention or a claim
that a particular modification would repair it. The decomposition is
`.pytest-tmp/local-context-learnability-v1-margins.json`, SHA-256
`333b2c0bd41c35532fca1c5b3eead75d2daf6fa5c809ee3267f470971b534c88`;
it uses authenticated saved integer synapses and probe identities, without model
imports, new forward evaluation, fitting or added evaluation exposure.

At the final checkpoint all updating arms emit two bytes and EOS. The decoder's
withheld outputs remain wrong:

| Target | Lexical decoder | Permuted decoder |
| --- | --- | --- |
| ab | ba | ba |
| bc | cc | cc |
| cd | dc | dc |
| da | db | ad |

The seven remaining TRAIN errors are lexical `ac→cc`, `ad→dd`, `ba→ad`, `bd→dd`,
`ca→ac`, `cb→bb`, `dc→cc`; permuted `ac→ca`, `ba→aa`, `bd→db`, `ca→aa`, `cb→bb`,
`db→bd`, `dc→cc`. Zero-context reference emits `dd` for all sixteen probes,
solving only the taught `dd`. Frozen reference emits raw `00 00` bytes and hits
the cap on every probe. Full failure mappings for both checkpoints are retained
in the decision record. These distinguish wrong byte choice from a missing EOS.

## Acquisition and the certificate

These are genuine free first attempts before each lesson's teacher corrections.
The six entries per curve count success in consecutive blocks of 128 lessons;
each block has its own denominator of 128.

| Arm | Total first exact / 768 | Six 128-lesson blocks |
| --- | ---: | --- |
| Decoder, lexical | 189 | 0, 0, 13, 57, 65, 54 |
| Decoder, permuted | 186 | 0, 0, 17, 54, 57, 58 |
| Reference, lexical | 754 | 114, 128, 128, 128, 128, 128 |
| Reference, permuted | 758 | 118, 128, 128, 128, 128, 128 |
| Reference, zero context | 0 | 0, 0, 0, 0, 0, 0 |
| Reference, frozen | 0 | 0, 0, 0, 0, 0, 0 |

The lexical reference makes 10, 4 and 2 teacher-position mistakes in its first
three cycles, then zero from cycle four onward. Permuted reference makes 15 in
cycle one and zero from cycle two onward. Both satisfy the predeclared bound
and find a complete no-mistake cycle well before cycle 55. The bound concerns
teacher-position mistakes; it is distinct from complete-answer first attempts.
After a no-mistake cycle there are no further reference corrections. Both saved
checkpoints consequently preserve the same reference answers, including their
withheld errors.

The lexical decoder's last eight cycles each produce 5/12 exact first attempts;
the permuted decoder produces 6, 7, 5, 5, 6, 5, 5 and 5. Complete per-cycle
counts are saved in the decision. This finite observation does not establish
convergence or guarantee improvement from another extension.

Decoder teacher cross-entropy decreases from 3.483 to 0.815 in the first versus
last 128 lexical lessons, and 3.485 to 0.811 under permutation. Last-window
first-byte / second-byte / EOS cross-entropies are 1.310 / 1.021 / 0.114 and
1.295 / 1.024 / 0.114, respectively. These are online teacher-forced losses,
not success rates or fixed-checkpoint counterfactuals.

Reference losses are **0/1 teacher mistakes**, not cross-entropy. Correct-context
references have zero such loss in the final 128 lessons, while zero-context
reference continues to make 1,025 mistakes across the full run and frozen
reference records 2,304 offered-label mistakes without updating. Numerical loss
magnitudes must not be compared across the decoder and reference architectures.

## Costs and saved-evidence audit

All 4,608 first attempts, 13,824 byte/EOS offers, 11,520 enabled teacher-step
opportunities, 192 checkpoint probes and 13,481 generation decisions are
accounted for. There are 5,664 actually changed steps: 2,304 in each decoder,
16 and 15 in the correct references, 1,025 in zero-context reference and zero
in frozen reference. Teacher corrections never count as first-attempt successes.

The monitored run takes **18.777891 seconds**, with **305.8046875 MiB** observed
process-tree peak. The saved audit's internal timer is 1.4617564 seconds, making
20.2396474 seconds when added to run time. That audit timer excludes its imports
and process startup; including separately observed audit-process startup gives
the conservative combined duration **20.6008779 seconds**, below 120 seconds.
No independent audit-process peak-memory measurement is claimed. The run's
observed tree peak is below its 2,048 MiB ceiling.

Each decoder stores 313,732 tensor bytes; each reference stores 302,652.
Reference accounting includes 644 neurons, 74,787 learned readout synapses,
585 allocated fixed synapses (578 nonzero), 288 thresholds and three initial
clock values. Its local step is 3,232 bytes **including** its 12-byte next-clock
state; a standalone clock state is 12 bytes. The supplied context is another
384 bytes. These payloads exclude Python
objects, copies and scratch, which are part of process memory.

The independent audit passes twelve source hashes and eight artifact hashes,
both fixed streams, state chains, frozen/read-only invariants, count/scoring
reconstruction, certificates and exact decoder-at-384 parity. It reconstructs
integer correction sums solely from saved teacher targets, winners and active
features and matches every reference checkpoint. It does not recompute neuronal
potentials or winners, import a model, or perform inference/teaching. Source
review supplies the locality and first-before-teacher ordering argument alongside
the saved hashes; it is not a claim of independent reexecution of all dynamics.

Audit: `.pytest-tmp/local-context-learnability-v1-independent-audit.json`, SHA-256
`18464f915fa37d0755bb548bc80f19c11c11eed7f569e242f541a8f03b57e0e9`.
The decision binds report, completion, launcher and audit hashes and retains all
source/artifact identities.

## Decision and limits

Close this bracket without an automatic extra fit, endpoint, seed, phase variant
or threshold change. The certified reference establishes an efficient local
TRAIN solution with an explicit supplied temporal representation. The decoder's
failure under both orders and the larger fixed budget is a practical limitation
of this computation/learning combination relative to that reference. It does
not prove RNN impossibility or separate expressivity from credit by itself.

Incomplete reference composition remains a separate failure. Neither reference
is adopted as a successful output solution, and certificate passage alone earns
no integration or gate advancement. Any next scoped return requires its own
review and preserves these closed results. The clock and answer-revealing context
are supplied priors; this experiment establishes no document reading,
source-absent knowledge acquisition, variable-length output, learned action
selection, real software success or protected evaluation progress.
