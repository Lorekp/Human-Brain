# Direct context v1: partial fit, no held-out composition

2026-09-06. The one fixed diagnostic completes in **8.951 seconds**, with
**289.215 MiB** observed process-tree peak. The independent saved-only audit
passes, but the positive-control criterion fails. Informative authored context
helps fit four taught pairs; it does not yield reliable ordered two-byte output
or success on any withheld combination within the fixed exposure.

Source: `7bed991a51296fed827fa227db84863e6402ba1b`.
The [recipe](local_context_use_v1.md) and
[decision](../../benchmarks/local_context_use_v1.decision.json) bind this
engineering result. The decoder's nine initial state entries exactly match
the v5 starting decoder: experiment seed 733, decoder seed 735. Every arm starts
with those weights; no v5 learned weights are reused.

| Arm | TRAIN / 12 | Held out / 4 | Acquisition first exact / 384 |
| --- | ---: | ---: | ---: |
| Correct context | 4 | 0 | 13 |
| Zero context | 0 | 0 | 0 |
| Shuffled context | 0 | 0 | 1 |
| Frozen | 0 | 0 | 0 |

The four correct outputs are `aa`, `cc`, `dc` and `dd`. Across all sixteen
final pairs the correct-context arm stops after the single byte `d` eight
times; its other outputs are `aa`, `cc`, `dc` and `dd`. Zero context emits `d`
everywhere, shuffled context emits only `d` or `dc`, and frozen generation hits
the two-byte limit on every pair. All updating final generations end in actual
EOS. The missing second bytes are model decisions, not supplied truncation.

Correct-context teacher loss falls from 3.483 in the first 128 lessons to 1.137
in the last 128, compared with 3.517 to 1.227 for zero context and 3.518 to 1.234
for shuffled context. Teacher correction loss and 13 successful acquisition
first attempts do not replace the failed endpoint criterion. Context improves
some taught behavior; the result does not show that the decoder entirely
ignores context or that local learning is incapable of composition.

Saved per-byte losses sharpen the failure: correct-context first-byte,
second-byte and EOS losses fall from 4.158 / 4.110 / 2.182 to
1.586 / 1.538 / 0.287. Final first bytes are correct on 6/16 pairs, and
second bytes on 4/6 of those. At first-symbol transitions within the last
128 lessons, outputs match the previous symbol 17/32 times and the current
symbol 0/32 times; this descriptive association does not establish causality.
V5's final teacher was `o'lin`, so its constant `mavi lune` output is not simple
last-answer copying. The saved-only description is
`.pytest-tmp/local-context-use-v1-description.json`, SHA-256
`bbc0292edefdd0a922626267210f4d81fe2fff5bfad5fd912ab190d10946f95d`.

A separate source audit finds that gain, self-path eligibility, signs and
pre-update feedback timing match their documented local rule. No arithmetic
repair is selected. The next proposed comparison changes only hidden neuronal
error feedback from random projection to the current readout's symmetric
instantaneous signal. It must retain the same local eligibility and exposure,
explicitly declare weight transport, and use its own reviewed source identity.
This is a learning-rule comparison, not a bug fix or another v1 fit.

All 1,536 first attempts, 4,608 byte/EOS offers, 3,456 enabled and actually
changed teacher steps, 64 read-only final generations and 3,449 generation
decisions are accounted for. The auditor independently reconstructs balanced
TRAIN factors, each fixed-seed shuffle, held-out exclusion and final-input
policy. It checks eight source hashes, five artifact hashes, state chains,
fixed tensors, frozen/read-only equality, exact v5 initialization, scoring,
precise clocks and the launcher's creation-keyed owned processes and resource
limits. It does not rerun the model or recompute its learning trajectory.

Audit: `.pytest-tmp/local-context-use-v1-independent-audit.json`, SHA-256
`2f17fa3dc759bd9f4a1f31581d0934d5a3f257661a348ba61d2f1904ef04f2cb`.

This diagnostic is closed without additional lessons, seeds, schedules or
threshold changes. Its explicit context rates reveal the target's factors,
so it supplies no reading, source-absent memory, software or gate evidence.
An upstream writer-feedback-only change cannot repair this fixed-input test.
The next bounded step is saved-error analysis and a source/local-credit review
of the decoder's first-byte choice, premature EOS and teacher-forcing behavior.
Select any corrective comparison from that diagnosis, then return a useful
decoder to the raw-document, retained-memory, executed-action workload.
