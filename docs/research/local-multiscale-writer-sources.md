# Fixed neuronal timescales: sources and implementation feasibility

Status: bounded research/design input, 2026-09-06. No model calls, fitting,
benchmarking or production edits. Root owns the selected design and next recipe;
the companion multiscale writer review owns the local-rule decision.

## Evidence and primary sources

The current binding audit, SHA256
`bd5c86fbf800519ada5053d1ead22c5ec78ba3062b058f58e9e2e628e6bf0b4c`,
confirms that tagged writer activity distinguishes identifiers strongly at byte9
but retains only 10.576%/9.864% of that difference before values at byte20.
At value-end22, identifier-induced activity distances are 0.193–0.259 while
value-induced distances are 2.325–2.644. Write-key distances show the same
imbalance: 0.0458–0.0907 versus 0.6869–0.7382. Historical d724 activity retention
was 10.852%/10.139%. These are saved descriptive contrasts, not proof that the
remaining identity information is unusable. They favor investigating temporal
representation before another routing-only credit modification.

Three primary papers were verified online for this review:

1. **Perez-Nieves et al. (2021), Neural heterogeneity promotes robust learning.**
   Their spiking networks assign individual membrane/synaptic time constants,
   either fixed after initialization or trainable. Heterogeneity helps especially
   on temporally structured inputs and under timescale mismatch; it adds neuronal
   parameters without adding neurons or synapses. Their main training uses
   surrogate-gradient BPTT, PyTorch automatic differentiation and Adam; a separate
   FORCE experiment also investigates fixed heterogeneity. Neither training
   protocol is our synapse-local writer rule. This supports a heterogeneous
   dynamics prior, not transfer of their performance or fitted time constants.
   [Publisher paper, Results and Methods](https://www.nature.com/articles/s41467-021-26022-3).

2. **Jaeger et al. (2007), Optimization and applications of echo state networks
   with leaky-integrator neurons.** Their discrete leaky recurrent equations
   separate state retention from the gain admitting a new nonlinear drive.
   They discuss stability and use online LMS for readout fitting, plus gradient
   adaptation of global controls such as leak, recurrent scaling and input gain.
   The latter propagates sensitivities through recurrent connectivity; it is
   not the main learner's retained diagonal self-path approximation. The paper
   is useful for interpreting leak and input amplitude together, rather than
   treating a long decay time as a guarantee of useful memory.
   [Author-hosted manuscript, equations 3, 6 and 14–21](https://www.ai.rug.nl/minds/uploads/leakyESN.pdf).

3. **Bellec et al. (2020), A solution to the learning dilemma for recurrent
   networks of spiking neurons.** E-prop factors learning into forward local
   eligibility and neuron-level learning signals. The ideal learning signal
   generally requires future/recurrent credit; its online approximation is
   explicitly distinct. Slow hidden neuronal variables support longer local
   traces. This motivates changing the trace recurrence consistently with the
   neuron equation. It does not make an arbitrary broadcast or our omission
   of off-diagonal recurrent paths exact. The main learner retains its declared
   fixed-rate local update, not an imported optimizer or BPTT comparison.
   [Publisher paper, learning-signal discussion and equation 27](https://www.nature.com/articles/s41467-020-17236-y).

These papers do not prescribe the bank sizes or byte-timescales below. Those are
prospective engineering choices. Publisher search text supplied the Nature
methods when direct page opens redirected; the Jaeger manuscript was read
directly. No secondary source supplies a scientific claim here.

## Smallest reusable mechanism

Current `StreamingRateCell` in `src/humanbrain/plastic_memory.py` computes
`a=tanh(W*x+U*h+b)` and `h'=h+0.5*(a-h)`, with initial `U=0.9I`.
Its `_step` is shared by differentiable and local paths. A fixed vector
`alpha[j]` within the same 64 writer units is the smallest change:

```text
h'_j = h_j + alpha_j * (a_j - h_j)
d_j = alpha_j * (1 - a_j^2)
c_j = (1 - alpha_j) + d_j * U_jj
e'_ji = c_j * e_ji + d_j * pre_i
```

The presynaptic term is the current input, previous cell activity, or one for
input, recurrent, or bias eligibility respectively. This remains a local
self-path approximation. All maps see all64 units; the clock advances on every
ordinary byte/EOS, with existing masking, cross-document carry and final purge.
No token identity, punctuation, key/value role or boundary selects a bank. Keep
query-cell dynamics unchanged. Keep writer/map feedback, overwrite tags and LMS
fixed. Do not add learned timescale parameters or auxiliary teaching targets.

Recommend a single fixed 32/16/16 bank with alpha=0.5/0.125/0.03125, subject to
root's review: preserve half the existing fast capacity, then cover progressively
longer event spans with two factor-four slower admissions. This broadens the
prior within the existing state dimension; it is not a chosen winner from a
sweep. A two-bank design has fewer timescales but identical implementation and
storage cost; no evidence establishes which allocation will learn better.

## Amplitude and credit limits

For the isolated leak-only recurrence, a unit one-step drive contributes
`alpha*(1-alpha)^d` after d subsequent steps. At d=11 this is 0.0002441,
0.0287739 and 0.0220383 for the three proposed alphas. At d=15 it is
0.00001526, 0.0168667 and 0.0194100. The slowest bank does not have the largest
absolute contribution at every delay. Its immediate drive and trace increment
are sixteen times smaller than the fast bank's.

These are analytical kernels, not simulated model observations. Initial
unsaturated `U=0.9I` instead gives self-carry `1-0.1*alpha`; corresponding
e-folding spans are about 19.50/79.50/319.50 steps, not the leak-only
1.44/7.49/31.50. Saturation and learned off-diagonal coupling alter the actual
response. Slow dynamics can also retain irrelevant preceding documents. Report
absolute activity, key and value differences alongside retention ratios; do not
rescale slow activity, normalize banks separately or modify map initialization
to rescue weak amplitude within this comparison. Those would change the question.

## Resource and exact-reference boundaries

A fixed64-element alpha buffer costs256 bytes at float32,512 at float64.
It adds no trainable weights, caller-owned activity, eligibility dimensions or
tag dimensions. If the reusable cell serializes an alpha vector for the unchanged
32-unit query cell too, count an additional128 float32 bytes:384 total over the
old joined model, rather than claiming only256. Store configuration metadata
and validate finite alpha in (0,1], matching shape/dtype/device and the frozen
configuration. Such metadata is not an additional numerical state tensor.

At D32/H64/K64/V64 there are14,593 writer/map parameters whose tagged credit has
933,952 floats (3,735,808 bytes). Live cell state/traces occupy25,088 bytes;
the complete tagged retained state remains3,777,280 bytes per stream. A fixed
within64 bank preserves these allocations. An added slow activation bank would
need its own carried state, map inputs and derivative bookkeeping; copying
post-activity alone would not preserve the present local eligibility semantics.
There is no need to incur that complexity for the selected question.

Dense matrix work and O(PK) tag transport keep their shapes. Scalar-to-vector
elementwise factors and vector validation add small operations; this is not a
measured runtime guarantee or speedup. Fresh complete-workload admission remains
necessary. Prior value-credit acquisition cost was850.126 projected seconds,
360.662 actual run seconds, with1080.559MiB observed tree peak. That margin is
evidence about the old workload only.

Preserve private RNG consumption and every shared initial weight/input code.
The homogeneous alpha=.5 path should use the same expression order as the old
cell, then demonstrate exact full forward/state/teacher trajectory parity against
the immediate frozen source86cc723, in both upstream dtypes. Exact old output
parity is a requirement for that reference, not for the heterogeneous candidate.
Archive only the minimal old cell arithmetic in an experiment-owned reference if
new serialization changes the constructor contract; do not add a production
legacy branch. Same initial tensors alone do not establish trajectory parity.
All frozen/writes-disabled controls must still honor their declared updates.

Graph discovery was attempted but transport was closed, so known source files
were read directly. A scoped Mem0 search returned unrelated development history;
none was used as scientific evidence. No tests or new endpoint observations were
performed for this report.
