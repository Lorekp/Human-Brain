# Local memory localization: repair source-dependent query credit

2026-09-06. Independent interpretation requested by the orchestrator. This
review reads authenticated saved receipts and the separately produced saved-array
summary; it performs no model call, new query, fit or protected evaluation.
The frozen [localization contract](local-document-memory-return-review.md) remains
unchanged.

**Recommend one repair: transport the instantaneous memory-read error through
the actual read synapses and query normalization into the query map and query
cell.** Keep the writer's local credit and the LMS output rule unchanged. The
observed gap is inadequate source-by-query binding; the measurements do not
support reversing the current updates, suppressing all late writes, or returning
to a supplied-answer decoder diagnostic. This recommendation selects a
falsifiable link to improve, not a claim that the proposed rule already works.

## Authenticated result and cost

Source is `d724e06183ea7232f59a506febe2f8f0b02b5b1b`. The independent audit at
`.pytest-tmp/local-memory-localization-v1-independent-audit.json` has SHA256
`ce18f50ac0d7ffc256b6cbc4a2fb2e8791ad3566ffdac6534010eed11d2897fe`.
Its input hashes and checker identity were verified again using standard-library
file reads. The saved-array summary from the parallel review is
`.pytest-tmp/local-memory-localization-v1-summary.json`, SHA256
`e1aae1aeacba7af3e0578c54b4f69bbcf9a87166beb82a56bc06538e664c7f58`.

The run takes 22.1348402 seconds at 822.66015625 MiB; its monitored audit takes
4.1007099 seconds at 510.05078125 MiB. The auditor's limit is correctly
`floor(60 - 22.1348402) = 37` seconds. Total monitored time is 26.2355501
seconds, within the original aggregate 60 seconds; both process trees remain
below 2,048 MiB. There are 56 ingestions, 64 free attempts and corresponding
actions, two motor demonstrations, eight isolated proposal corrections and
24 teacher offers. All free attempts consume three decisions, so supplemental
phase evaluations are zero. The declared 240 output-score evaluations include
the teacher's validation recomputations. The audit authenticates all 66 execution
receipts. Its pass is evidence custody and numerical accounting, not capability.

For both tagged and writer/maps-frozen endpoints, original document order gets
2/8 diagnostic actions right; reversed order gets 1/8. All 32 baseline first
bytes are `c`. Reversing whole documents changes the two `k03` outputs while
preserving their true assignments, and loses one correct action in each mode.
Tagged and writer-frozen agree on all sixteen corresponding baseline outputs.
The 32 perturbed conditions retain their respective baseline generated answers.
These are exposed TRAIN-valued diagnostic conditions, not fresh transfer.

## The existing updates have the useful sign on every tested condition

The entries below are changes in mean three-phase half-squared potential loss,
with the old LMS output weights fixed and documents reingested. Positive writer
and address conditions install the actual teacher-after parameter blocks;
negative conditions use the declared original-dtype subtraction. Values are
rounded for display; the report and audit retain full floating values.

| Baseline query ID | Writer positive | Writer negative | Address positive | Address negative |
| --- | ---: | ---: | ---: | ---: |
| 0 | -9.813188553e-5 | +9.906218225e-5 | -3.020690832e-4 | +3.026361059e-4 |
| 1 | -1.986536961e-7 | +7.960609952e-7 | -3.288000949e-4 | +3.299190514e-4 |
| 2 | -3.229566263e-5 | +3.238862718e-5 | -4.667149422e-4 | +4.682298873e-4 |
| 3 | -9.462926385e-6 | +9.711888065e-6 | -1.393077194e-4 | +1.396949204e-4 |
| 4 | -6.058142707e-5 | +6.051730146e-5 | -1.950421111e-4 | +1.962374627e-4 |
| 5 | -1.051708809e-4 | +1.059865786e-4 | -6.101962121e-4 | +6.090795495e-4 |
| 6 | -2.074612073e-4 | +2.068720545e-4 | -5.436419321e-4 | +5.437715527e-4 |
| 7 | -1.914226786e-4 | +1.912098519e-4 | -2.288517772e-4 | +2.299265381e-4 |
| Mean | -8.809066526e-5 | +8.831806813e-5 | -3.518279840e-4 | +3.524368835e-4 |

All eight positive directions improve loss for each block, and all eight
negative directions worsen it. The smallest writer improvement is small but
above the declared numerical comparison tolerance. This falsifies the proposed
wrong-direction signature on these conditions. It does not prove general
gradient alignment, adequate acquisition speed or convergence. An address block
contains both the query cell's direct output-context route and the memory query
map; these aggregate effects do not isolate which route supplies the gain.
Each correction is tested on its own question, so benefits to the other question
in the same memory are also unestablished. No discrete answer improves here.

## Distinctions survive, but their response is poorly conditioned on the query

Tagged assignment-matrix contrasts peak just after the first byte of the second
document and retain 71.1–85.2% of that peak at the final memory. Their final
Frobenius norms are 1.6979–1.9315. Actual read contrasts are 0.1707–0.3914;
the ratio to matrix contrast ranges from 0.0952 to 0.2305. These are descriptive
geometry measurements, not percentages of semantic information retained.
The per-query first-document contribution retains 30.1–53.9% of its norm before
the second document; the second source contribution is 1.95–3.23 times larger.
There is substantial interference/recency, but no evidence of complete erasure.

The more informative comparison asks whether swapping assignments changes the
two questions in opposite, answer-appropriate directions. For original-order
tagged A/B memories, the two read-change vectors have cosine 0.924128, norms
0.325726 and 0.279813, and difference norm 0.126247. For C/D they have cosine
0.912921, norms 0.170692 and 0.334281, and difference norm 0.191569. Reversed
order gives cosines 0.911542 and 0.936529, with difference norms 0.148131 and
0.221745. Thus much of the observed source change is common to both queries,
although the required target changes are opposite. Nonzero interaction remains.

The query keys themselves are clearly different: tagged within-pair distances
are 0.866898 and 0.865553. Six of eight memories per mode change output when
their question changes. The diagnosis is therefore a weak **assignment-dependent
query interaction**, not identical addresses or complete query blindness.

The common-direction pattern survives the output projection. Tagged first/second
byte potential-change cosines across the two queries are 0.985205/0.979303 for
original A/B and 0.988161/0.970693 for original C/D; reversed values are
0.940380/0.967807 and 0.993851/0.987038. Writer-frozen shows the same pattern.
Nonzero character-potential contrasts, approximately 0.019–0.078 in norm, often
coexist with larger wrong-target deficits. Across baseline conditions character
margins range from -0.169190 to +0.123386, while EOS margins stay positive.
This neither proves the read vectors are sufficient for a correct decoder nor
attributes all failure to output training. An affine readout can retain a source
change that does not express the needed key/value relationship.

Numerical rounding does not explain these effects. Maximum uncorrected matrix
and read reconstruction residuals are about 1.64e-7 and 8.25e-8; after the
explicit recorded recurrence/read residual terms, maximum closures are
5.00e-16 and 2.50e-16. The correction accounting is floating receipt algebra,
not an exact-rational certificate of historical forward arithmetic. The
instrumentation's separately reviewed forward parity remains necessary.

## One repaired link: actual instantaneous read-path credit

The current [learner](../../src/humanbrain/local_memory_learner.py) sends a fixed
random projection of context error to the query map and only the direct query
slice to the query cell. The proposed replacement uses the actual retained
read connections. With query activity `u`, pre-normalization drive `a=Q u+b`,
normalized key `q`, read `r=M q`, and mean pre-update LMS context signal
`[g_u, g_r]`, the instantaneous neuronal errors are:

```text
e_q = M^T g_r
e_a = (e_q - q * dot(q, e_q)) / ||a||     when ||a|| > epsilon
e_a = e_q / epsilon                      on the existing clamped branch
query-map synapse change = -eta * e_a_j * u_i
query-map bias change = -eta * e_a_j
query-cell learning signal = g_u + Q^T e_a
```

This is our direct chain-rule derivation for the existing instantaneous read.
Use pre-update `M`, `Q`, query activity and the same mean LMS signal. The query
map must use raw presynaptic eligibility with this normalization-adjusted neuron
signal: multiplying by the old diagonal normalization gain again would be an
error. Keep the query cell's existing local recurrent self-path eligibility;
this does not turn it into full recurrent differentiation. Preserve the existing
float64 sum/mean and single checked cast into the upstream dtype, finite proposal
checks and all freezing controls.

Every map update still uses its own presynaptic activity and a declared
postsynaptic error. The transpose read connections and divisive-normalization
error pool are explicit weight transport and neuronal broadcasts. No retained
network history, BPTT, optimizer or externally generated per-synapse correction
enters the main learner. E-prop's distinction between eligibility and learning
signals motivates checking this omitted route; it does not supply a theorem
that this particular memory learner will acquire binding.
[Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y).

This choice is warranted as a focused hypothesis because the missing route
depends on the actual source-conditioned matrix and current read error. It can
provide query learning with information absent from a fixed projection. The
saved data do not establish that it is sufficient, optimal, or the only possible
repair. In particular, writer interference and representational inadequacy can
still limit the result. Keep their mechanisms unchanged in this comparison so
the query-read link has an interpretable test.

## Bounded return to the integrated workload

Do ordinary local arithmetic/finite/frozen-control QA and verify unchanged
forward behavior before learning. Then preregister one direct integrated
comparison at the existing rate 0.01 and 1,024 lessons per arm. Use the repaired
five controls plus one matched current-rule tagged arm, all starting from the
same fresh slow weights and zero LMS output. This is six arms in one comparison,
not six architecture proposals. Do not reuse the disposable perturbations or
endpoint weights as initialization. The two-byte TRAIN schedule, raw grammar,
same-query swaps, source-free worker and actual motor/executor remain fixed.
Previously exposed checks must keep that label; no new adjudication claim arises.

Six-arm complete-workload cost must be admitted prospectively under the original
2x margin, 30-second allowance and 900-second/2,048-MiB limits, with all extra
read-feedback computation counted. Neither the old five-arm admission nor the
60-second diagnostic admission transfers. If the cost fails, stop before fitting.

The useful endpoint remains repaired tagged 8/8 fit and 4/4 in each new-value,
delay and interference stratum, with strict new-value advantage over repaired
writer-frozen, fully frozen and writes-disabled. Require an actual new-value
action gain over the matched current tagged rule; report aggregate/tag-specific
advantage separately. Measure the already declared same-query swaps and both
questions per memory, not merely an average loss reduction. No automatic rate,
seed, epoch or signal sweep follows a failure. A successful result returns to
learned action selection and broader persistent tasks under the roadmap; it
still uses the declared host motor and does not pass the general integrated gate.
