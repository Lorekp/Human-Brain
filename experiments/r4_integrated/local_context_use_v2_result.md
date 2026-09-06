# Direct context v2: symmetric feedback changes errors, not qualification

2026-09-06. The matched eight-arm comparison completes in **13.965 seconds**,
at **295.367 MiB** observed process-tree peak. Its independent saved-only audit
passes, including exact reproduction of v1's random-arm predictions, teacher
losses and final tensors. **The symmetric candidate fails qualification** and
does not earn integration into the document learner.

Source: `a879c4ea678e64a70b906702397048c96775d64f`.
The [recipe](local_context_use_v2.md) and
[decision](../../benchmarks/local_context_use_v2.decision.json) preserve the
fixed seed, 384 lessons per arm, local eligibility, rate, exposure and scoring.
Only the declared hidden neuronal feedback rule differs between matched pairs.
The four probe combinations were already exposed development items in v1,
while remaining absent from TRAIN; this is not a fresh assessment.

| Feedback | Context/control | TRAIN / 12 | Withheld / 4 |
| --- | --- | ---: | ---: |
| Random | Correct | 4 | 0 |
| Random | Zero | 0 | 0 |
| Random | Shuffled | 0 | 0 |
| Random | Frozen | 0 | 0 |
| Symmetric | Correct | 3 | 0 |
| Symmetric | Zero | 1 | 0 |
| Symmetric | Shuffled | 1 | 0 |
| Symmetric | Frozen | 0 | 0 |

Symmetric correct-context output is always two bytes followed by actual EOS:
`cc` four times, `db` three, `dc` three and `dd` six across all sixteen probes.
It solves `cc`, `dc` and `dd`; all four withheld combinations remain wrong.
The eight premature one-byte stops from random feedback disappear, but the
ordered character choices do not become reliable. Correct EOS alone is not
exact answer success.

Symmetric correct-context acquisition has 20/384 exact first attempts, versus
13/384 for random feedback. Mean teacher loss is 1.672 versus 2.041, and its
last-128 mean is 1.024 versus 1.137. These descriptive improvements do not
compensate for the failed endpoint or establish generalization. The fixed
budget does not establish convergence or impossibility under other exposure.

Saved curves still decrease late. Comparing the preceding and final 48 lessons,
random correct-context first successes increase 1 to 12; symmetric stays at 8
to 8. Symmetric first/second-byte/EOS losses decrease from 1.505 / 1.413 / 0.176
to 1.464 / 1.329 / 0.149. Controls also improve loss without useful accuracy.
These data justify reviewing the exposure and stability assumptions, not
claiming that extra experience will succeed. The saved-only curve report is
`.pytest-tmp/local-context-use-v2-curves.json`, SHA-256
`f9e36a1a69cd8c08e596ede21ae7d814d5c7a707dee32f54bc7767eac3ff5cd7`.

All 3,072 first attempts, 9,216 teacher byte/EOS offers, 6,912 enabled and
actually changed teaching steps, 128 final generations and 7,291 generation
decisions are accounted for. The audit authenticates ten source hashes, five
artifact hashes, schedules/shuffles, mode-aware weight chains, frozen and
read-only states, tensor equality with the v5 initialization and the external
creation-keyed process/time/memory evidence. Its random-arm parity check allows
only the explicitly revised checkpoint metadata; numerical behavior matches v1.
It does not rerun models or independently recompute update trajectories.

Audit: `.pytest-tmp/local-context-use-v2-independent-audit.json`, SHA-256
`ff2bb87f7f83042fcf49504ad09e907c99e853c33138a42410c3d3ddd6cebbf2`.

Close v2 without additional epochs, alternate seeds, thresholds or another
automatic feedback variant. Symmetric feedback remains an explicit comparison
option, not the adopted main learner. Step back to one bounded learnability
review: distinguish inadequate exposure/stability from temporal representation
or recurrent-credit limitations, using saved learning curves and a justified
local neuronal reference before selecting another experiment. Preserve both
failed comparisons and require useful exact behavior before returning a changed
decoder to raw-document, source-absent executed actions. No reading, coding or
protected gate is advanced by this answer-revealing engineering diagnostic.
