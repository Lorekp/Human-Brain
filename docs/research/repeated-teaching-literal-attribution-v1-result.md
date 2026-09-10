# Historical literal-update attribution

Completed saved-only analysis, 2026-09-10. The result advances the earlier
weak-update diagnosis: direct teacher contributions are small and mostly survive
opposing teacher contacts, but public-input scans preferentially strengthen the
competing space cell. At the independently checked editing sites, scans erase
94.81% of the mean teacher gain in target-versus-space margin. Raising a target
current therefore did not necessarily improve its discrimination from the
strongest initial competitor.

The attribution analyzer used source revision `812e726`, SHA256
`e18a85c47486009eac6b2730d36430b61a12023af1da7bad240d33df77f3df9e`.
This interpretation reads only `.pytest-tmp/literal-attribution-v1-result.json`;
it makes no model, original-journal, checkpoint-tensor or teacher calls.

## What closes against actual logits

The analyzer projects the actual enabled learner's historical literal updates
onto the 35 fixed target-prefix activities at checkpoints 0, 8, 16, 32 and 64.
Ten sites have saved full literal logits at all checkpoints: reading positions
2 and 9, and editing positions 0, 1, 4, 7, 12, 17, 18 and 21. At these sites,
the reconstructed changes close against all 257 actual logits with maximum
absolute residual `3.469447e-18`, below the declared `2e-11` tolerance.

The other 25 sites have projected current contributions only, without an
independent full-logit endpoint check. A hash identifying an emission activity
does not itself provide its full output distribution. This report preserves
that distinction and does not extrapolate the ten-site competitor result into
an independently verified margin result at all 35 sites.

All contributions below are cumulative changes from checkpoint 0 to 64 at
fixed activities, not initial currents, probabilities, or new training results.
Means include EOS: 12 reading sites and 23 editing sites.

## Target-current contributions

| Evaluated skill | Historical component | Mean current change | Minimum | Maximum |
| --- | --- | ---: | ---: | ---: |
| Reading | Reading teachers | 0.002521373 | 0.001013866 | 0.004155787 |
| Reading | Editing teachers | 0.000239957 | -0.000043026 | 0.000689319 |
| Reading | Public-input scans | 0.000511313 | 0.000048252 | 0.001274851 |
| Reading | Actual public consequence | -0.000001244 | -0.000013814 | 0.000010030 |
| Editing | Reading teachers | 0.000383409 | -0.000069377 | 0.003142074 |
| Editing | Editing teachers | 0.001184955 | 0.000398340 | 0.002606234 |
| Editing | Public-input scans | 0.000536973 | -0.000006116 | 0.002140159 |
| Editing | Actual public consequence | 0.000000440 | -0.000016257 | 0.000008570 |

Total target-current change is positive at every site: reading mean
0.003271400, range 0.001245482–0.005777470; editing mean 0.002105776,
range 0.000598554–0.004918268. Same-skill teaching accounts for 77.07% and
56.27% of those respective mean net changes. Cross-skill teaching is positive
on average but negative at 5/12 reading and 16/23 editing sites. Scan target
contributions are positive at all reading sites and 22/23 editing sites.
The actual public-consequence contribution is much smaller and mixed in sign.

| Evaluated skill | Mean positive teacher contribution | Mean negative teacher contribution | Negative/positive ratio of sums | Per-site cancellation range |
| --- | ---: | ---: | ---: | ---: |
| Reading | 0.002866854 | -0.000105523 | 3.68% | 2.28%–7.99% |
| Editing | 0.001676835 | -0.000108471 | 6.47% | 2.78%–25.24% |

These signs separate supportive and opposing teacher terms on the target row.
They do not measure competition from increases in other output rows. The small
ratios reject wholesale cancellation of the target's positive teacher updates
as the principal explanation for their small net current movement. There is
local competition, especially at editing position 6, but positive target teaching
is largely retained.

## Competition changes the interpretation

The strongest initial competing literal cell is space, byte 32, at all ten
fully anchored sites. The table gives each component's change in
`target_current - space_current`, in units of `10^-3`.

| Site | Reading teachers | Editing teachers | Scan | Public | Net margin change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reading 2 (`d`) | 3.382336 | -0.247222 | -1.689229 | -0.013325 | 1.432560 |
| Reading 9 (`l`) | 3.083707 | -0.247216 | -1.842958 | 0.007757 | 1.001290 |
| Editing 0 (`r`) | 0.553993 | 0.611725 | -1.662235 | 0.008083 | -0.488434 |
| Editing 1 (`e`) | 3.205861 | 0.365065 | -0.834515 | -0.001583 | 2.734828 |
| Editing 4 (`r`) | 0.698913 | 0.842101 | -1.819373 | 0.007233 | -0.271126 |
| Editing 7 (`m`) | 1.407334 | 0.806839 | -2.153995 | 0.007043 | 0.067221 |
| Editing 12 (`,`) | 0.000356 | 1.443552 | -1.719757 | 0.007575 | -0.268274 |
| Editing 17 (`n`) | 0.000261 | 2.046553 | -1.164681 | -0.001297 | 0.880836 |
| Editing 18 (`,`) | 0.000349 | 1.320796 | -1.975667 | -0.002809 | -0.657331 |
| Editing 21 (`)`) | 0.000392 | 1.019243 | -2.249517 | -0.000071 | -1.229953 |

For the two reading anchors, the mean net teacher margin gain is 0.002985803
and the scan contribution is -0.001766094: 59.15% cancellation by scans before
the small public term. The mean final margin gain is 0.001216925.
For the eight editing anchors, mean teacher gain is 0.001790417 and scan is
-0.001697468: 94.81% cancellation, leaving mean net gain 0.000095971.
These are ratios of aggregate anchored contributions, not mean per-site ratios.

All ten anchored target probabilities improve, yet the target-versus-initial-space
margin worsens at five of eight editing anchors. Scan updates raise space more
than the target at every anchor. This is measured interference in discrimination,
which the positive target-current totals and slight target-probability gains
alone concealed. It does not establish that space is the strongest competitor
after learning, or that this one margin determines all free outputs.

## What the result now establishes

The [repetition review](repeated-teaching-acquisition-v1-learning-review.md)
already showed multiple direct contacts at every position, improved literal
probabilities at identical activities, correct learned edit boundaries and failed
complete byte output. The attribution now separates two effects:

- The absolute teacher contribution is small under the existing normalized
  rule; opposing teacher contributions usually remove only a small fraction
  of its positive target-current contribution.
- Shared public-input learning changes output competition. At these fixed
  activities it substantially favors space relative to the intended target,
  often consuming the whole teacher margin improvement in editing.

This is a more specific explanation than either "no effective teaching" or
"all updates cancel." It is not a proof that the sparse emission representation
cannot distinguish the tasks, or that scan learning should be removed. Nor does
this literal-only decomposition identify how mode/context/gate updates affect
the complete emission mixture or the free trajectory.

## The normalization candidate remains a diagnostic candidate

The report supports a simple scalar check of the previously proposed constant
`.01` coefficient using only its component fields. Holding the entire recorded
history fixed, replace each reading-teacher margin component by 24 times itself
and each editing-teacher component by 69 times itself, leaving scan/public fixed.
This is an additive historical projection, not a re-trained outcome.

That substitution gives reading-anchor margin changes of 0.055116–0.062415
(mean 0.058766), and editing-anchor changes of 0.053851–0.140052 (mean 0.088840).
They are positive at all ten anchors, but still only modest log-odds shifts.
The factor 24 or 69 is neither a probability multiplier nor evidence that the
same repetition budget would acquire either complete skill. An initially
uniform 257-cell literal population needs a margin of roughly `log(256)=5.545`
over all other cells to give one target probability one half; that idealized
scale comparison is not a bound on the actual actor or future learning.

A full conditional projection would require applying the substitution to all
257 currents and then using the recorded mode weights/matches. The present
compact result does not retain those component vectors; scaling a target
probability or one margin cannot recover the candidate distribution. Such a
projection would still hold historical activities/probabilities fixed and leave
actual future learning, routing and complete behavior unmeasured.

Root does not select another pass merely to quantify this already modest scalar
effect, or promote the standalone coefficient change to an acquisition trial.
The next learning design must address weak teacher selectivity, competing scan
updates and the existing mode computation together, while retaining the current
forward architecture and measured motor behavior. This finding does not select
new populations, a fit sweep or expanded task difficulty. The result identifies
specific computations to repair rather than treating a positive update or a
passed implementation check as sufficient evidence to add a mechanism.

## Custody and cost

The attribution result is `passed` and binds retained boundary
`3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677`
and original run report
`3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262`.
Actual selected-literal counts per neural route are 4,200 scan, 64 free and
273 teacher contacts. The analysis introduces zero model calls, payload decodes,
teacher calls, RNG draws or learning publications.

The first launch took 9.5265008 seconds with 146.3828125 MiB peak RSS; its internal
analysis timer records 9.069303 seconds. It read 510,318,993 journal bytes across
389 rows, 306,454 metadata bytes and 9,195,194 source bytes: 519,820,641 bytes
total. These are separately measured post-run analysis costs, not new learner
experience. This prose review subsequently reads only the 87,807-byte result,
whose SHA256 is
`a05ad45eeaec57fe85696be3e7866436353c91a0e05e32270910d8c9c1b3b6ba`.

The exact [result bank](../../benchmarks/repeated_teaching_literal_attribution_v1.result.json)
and [measurement ledger](../../benchmarks/repeated_teaching_literal_attribution_v1.measurement.json)
preserve the report and separate scopes. Both focused numerical tests pass on
their first launch in 5.0337697 monitored seconds at 273.2265625 MiB sampled peak;
the two monitored launches sum to 14.5602705 seconds. Banking's separately timed
copy/hash operations take 0.0641594 seconds, without an independent process-tree
monitor. A prior schema inspection read 1,048,576 teacher-journal bytes outside
the analysis pass. Doctor, lint, source/prose review and measurement serialization
remain outside these measured elapsed totals.

Run doctor is `.pytest-tmp/dev/doctor-vv_79wbw/receipt.json`; QA doctor is
`.pytest-tmp/dev/doctor-xlkt5w1x/receipt.json`. The launch receipts are
`.pytest-tmp/literal-attribution-v1-run.launch.json` and
`.pytest-tmp/literal-attribution-v1-qa.launch.json`. Independent review verified
the compact report's 175 site/checkpoint rows, 50 anchored checks, scalar sums,
counts, hashes and competing-margin arithmetic. It did not rerun the journal
reconstruction. Original learner outcomes and final continued state are unchanged.
