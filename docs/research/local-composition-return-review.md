# Local composition: potential-error learning and the return to documents

2026-09-06. Bounded primary-source and mathematical review after the frozen
six-arm learnability comparison. This report is the only tracked edit; no new
model, fit, prediction or benchmark was executed. The research skill was read;
this delegated background review coordinates the root's algebra and the other
agents' saved-evidence analyses. The roadmap and active plan remain authoritative.

**Recommendation:** retain the successful acquisition finding, close the
mistake-only reference, and consider one fixed four-arm local LMS comparison.
Unlike merely demanding a larger classification margin, matching all output
neuron potentials supplies explicit equations that imply the missing compositions
for this particular additive representation. A conservative real-arithmetic
bound supports a prospective 160-cycle endpoint at the existing rate 0.01.
This is a recommendation for review, not authorization or a claim of measured
LMS performance. It does not repair the upstream document learner by itself.

## What the saved evidence establishes

The [independent audit](../../.pytest-tmp/local-context-learnability-v1-independent-audit.json)
passed once against source `30fe85c2417ba87902cd4466f79ff9c7932fffd4`.
Its SHA256 is
`18464f915fa37d0755bb548bc80f19c11c11eed7f569e242f541a8f03b57e0e9`.
Outer execution took 18.777891 seconds, peak process-tree memory was
305.8046875 MiB, and the saved-only audit took 1.4617564 seconds. It authenticates
12 source files, eight artifacts, 4,608 first attempts, 13,824 teacher offers,
192 probes, both checkpoint state chains, and exact old decoder reproduction
at 384 lessons. Integer corrections reconstructed exclusively from saved
teacher/winner/features match all reference checkpoints; this is receipt-algebra
verification, not an independent neural forward run.

At 768 lessons both decoder orders fit 5/12 TRAIN and 0/4 withheld pairs.
The reference fits 12/12 TRAIN under both orders, with 2/4 lexical and 3/4
permuted withheld success. The zero-context reference scores 1/12 and 0/4;
the frozen reference scores zero. Both reference certificates pass: 16 and
15 teacher mistakes, with the first zero-mistake cycles at four and two.
**No arm passes the frozen useful-composition criterion.** The four withheld
pairs `ab,bc,cd,da` remain previously exposed development probes.

The independent [saved margin analysis](../../.pytest-tmp/local-context-learnability-v1-margins.json)
finds residual phase/cross-factor connections in the three failed reference
bytes. For example, lexical `ab` emits `ac`: the second-phase target `b` has
score one while `c` has score two. Permuted `ab` emits `ad`, with target `b`
score zero versus `d` score two. Lexical `da` fails its first byte on a tie
between `c` and `d`; the fixed lower-index competition selects `c`.
These are observed weights and scores, not a hypothesis about hidden memory.

The reference states at 384 and 768 are identical. Every later TRAIN teacher
winner is already correct, so mistake-only plasticity is disabled by its own
rule. Further repetitions or reorderings of these same fixed inputs cannot
change this checkpoint. This is stronger than the RNN result: the latter's
losses still improve, so finite failure does not prove its convergence or
expressive impossibility. The comparison establishes practical acquisition
inefficiency of the current RNN against a locally learnable temporal reference.

## Two learning interpretations, one preferred candidate

**Margin or norm selection.** A multiclass passive-aggressive update can continue
when the correct class wins but its margin is inadequate; its step is a local
presynaptic vector times neuron-level teacher/rival signals with a declared
scalar gain. The [original PA paper](https://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf)
derives online margin updates and loss bounds. Such updates add an inductive
bias; they do not, merely by their name or a positive TRAIN margin, prove the
missing combinations. A minimum-change update at each example is also not a
certificate of the final global minimum-norm separator. Do not select this
as another parallel arm or silently implement phase-specific factor wiring.

**Potential-error learning.** Give each of the 257 output neurons a desired
activity: one for the teacher byte/EOS neuron and zero for every other output.
With pre-update potential `z_j = sum_k W_jk x_k`, use

```text
r_j = desired_j - z_j
W_jk <- W_jk + 0.01 * r_j * x_k
```

This is the LMS/Widrow-Hoff rule with the factor of two absorbed into the rate.
[Flores and Widrow's primary account, equation 1](https://isl.stanford.edu/~widrow/papers/c2004assessmentof.pdf)
states the input-times-instantaneous-error update and explains the role of
input spectra in convergence. Their statistical analysis is not the deterministic
certificate below. The [original adaptive-neuron paper](https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf)
provides the historical learning-neuron construction; it is not evidence of
modern language or coding capability.

Each synapse uses its own presynaptic activity and its output neuron's residual.
The dense 257-neuron desired-activity broadcast is explicit supervision. It
contains no externally calculated per-synapse update, historical reverse gradient,
optimizer, inverse covariance, fitted witness or target-dependent feature routing.
The residual is computed before any output update. Although this rule is the
instantaneous gradient of a linear squared error, its implementation is genuinely
local under the roadmap's information constraint. All outputs receive error
signals even when the greedy winner is correct; that difference directly addresses
the observed cessation of useful learning.

## Why exact potential equations imply these compositions

The following is our derivation, not a result claimed from the LMS papers.
The independent cell-arithmetic agent verified the TRAIN-only rank, spectrum,
lower-Gram norm and row-space/projection argument without a learner run.
Fix one neuronal phase. Its generic feature vector is the phase unit and both
four-way factor populations; the other coordinates are zero. For any output
neuron its potential has the additive form

```text
z(a,b) = p + A_a + B_b.
```

At the first phase the desired full output vector is `e_a`; at the second it
is `e_b`; at the third it is `e_EOS`. TRAIN's bipartite factor graph is
`K4,4` minus one perfect matching. Every pair of second-factor vertices shares
a first-factor neighbor. Subtracting two exact first-phase equations through
that shared neighbor forces the corresponding `B` vectors to agree. Hence all
`B` agree, and each first-factor equation extends to its missing partner.
Exchange the factors for phase two. The constant EOS equations extend likewise.
This constrains learned associations through TRAIN labels; no runtime component
selects which factor is relevant to a phase.

Equivalently, every missing feature row has a three-edge TRAIN path:
`x(a,b) = x(a,b') + x(a',b) - x(a',b')`. The desired vectors obey the same
identity. Thus every exact interpolating additive solution composes, including
the minimum-norm one; minimum norm is not the essential assumption. Winner
inequalities do not provide these equalities, which is why a zero-mistake
perceptron certificate can coexist with the actual failed compositions.

The identity also gives an a posteriori mathematical certificate: if every
TRAIN output potential has absolute residual at most `epsilon`, each missing
potential has residual at most `3*epsilon`. Strict `epsilon < 1/6` gives every
correct missing byte/EOS a strict advantage over all wrong outputs. This
certificate concerns represented weights and linear potentials, not just
rounded reported classification scores. It must not feed corrections or
selection back from the withheld cohort.

## A finite prospective budget, with its numerical boundary

For one phase the TRAIN design matrix has 12 rows and nine active columns.
Each row has squared norm three. Its positive Gram eigenvalues are
`18` once, `4` three times and `2` three times; rank is seven. The two null
directions redistribute constant offsets among phase and the two factors.
All three phases occupy disjoint feature blocks, so intervening teacher phases
do not interfere with this per-phase calculation.

Let `S` be the TRAIN row space and project the proof-only exact separator onto
`S`. The resulting `U` still fits TRAIN and every missing row, because those
rows lie in `S`; projection cannot increase its squared Frobenius norm above
nine. Zero initialization and local LMS increments remain in `S`. Therefore
the error `E=W-U` lies in `S`, where the minimum Gram eigenvalue is two.
No projected separator is computed or loaded into the learner.

For one complete phase cycle, let `r_t=E_(t-1)x_t` be the sequential residual.
The exact per-update energy identity is

```text
||E_t||_F^2 = ||E_(t-1)||_F^2 - eta*(2-3*eta)*||r_t||^2.
```

Write `L` for the strictly lower part of the 12-row Gram matrix in the actual
order. Each row has four off-diagonal overlaps of two and seven of one, so
`||L||_F^2 = 12*(4*4+7)/2 = 138`, in every permutation. The vector of residuals
evaluated at cycle start is `(I+eta*L)r` (equivalently its transposed matrix
form). Therefore its norm is at most `(1+eta*sqrt(138))*||r||`. Combining
this with the minimum positive eigenvalue yields, for every cycle order,

```text
||E_after||_F^2 <= q * ||E_before||_F^2
q = 1 - eta*(2-3*eta)*2/(1+eta*sqrt(138))^2
eta = 0.01: q = 0.9684483676583155.

After K cycles, any supported or missing row has output-vector error norm
    <= sqrt(27*q^K).
K=147: bound 0.4923824169 < 0.5.
K=160: bound 0.3997602429; ideal correct-vs-wrong margin > 0.2004795141.
```

The Frobenius bound sums over all 257 outputs and all three phase blocks;
it therefore covers both byte decisions and genuine EOS. Use **160 cycles,
1,920 lessons per arm**, as one prospective endpoint above the derived bound,
not a sequence of adaptive extensions. The calculation used only scalar
arithmetic and the TRAIN graph; no learner was fitted or withheld prediction
tested to choose it.

This is a **real-arithmetic proof**. Select an explicitly float64 new reference,
with the binary inputs cast exactly and all proposals checked for finiteness;
the integer perceptron's exact certificate cannot transfer. Float64 is still
rounded arithmetic. The 0.20 ideal margin leaves room but is not itself a
roundoff bound. Source/QA must validate pre-update residuals. A prospective
numerical acceptance check can require actual correct-vs-wrong potential margins
strictly above 0.10 on every final TRAIN/withheld byte and EOS, alongside exact
free generation. This is measured numerical qualification with allowance below
the analytical bound, not a machine-checked theorem about every floating operation.
If a numerical guarantee is claimed, also provide a conservative rounding
envelope or audit represented weights against the TRAIN residual certificate
with an explicit summation-error bound. Exact rational checks of saved floating
coefficients are permissible external verification, never model inputs or
updates. Do not assume that a noisy, inconsistent or differently scaled
document context inherits this proof.

## One comparison and one return to the shared workload

If selected after source review, use four fresh zero-initialized float64 LMS arms:
lexical correct context, the existing private permuted correct stream, lexical
zero context, and lexical frozen correct context. Keep all 257 outputs, the
same generic phase/context features, rate 0.01 and teacher labels. The closed
perceptron and RNN receipts already supply their comparisons; do not rerun them.
There are 7,680 first attempts, 23,040 teacher offers and 17,280 enabled
teacher-step opportunities. One 160-cycle endpoint, no seed/rate search, under
120 seconds and 2,048 MiB, with all recorded cost and read-only audit included.
Do not add intermediate evaluation unless fixed before launch. Require
12/12 TRAIN plus 4/4 exposed withheld exact bytes/EOS and a strict advantage
over both controls. Keep frozen state, all-output residual accounting and
source/receipt identities auditable. Failure closes this comparison; the
certificate's violated assumptions direct diagnosis, not more epochs.

Declare generic continuous gating `phase * context` in the new reference now,
so it accepts the actual signed neuronal read vector. Keep the fixed 96-input,
three-phase shape; no general constructor or variable-length expansion is needed.
The current binary threshold conjunctions and integer ticks cannot serve this
interface. This is an explicitly reviewed change to continuous rate dynamics,
equal on the binary development inputs, with all coordinates treated uniformly.
Binary inputs give the proof above; arbitrary continuous inputs do not.

The current reference is also not a drop-in replacement for the joined learner:
it lacks the decoder's upstream `context_signal` and embedding-related interface,
and its three phases cannot emit the old multi-byte phrases. A successful LMS
output diagnostic earns one integrated return with a declared, limited two-byte
action-value roster, or requires a separately reviewed timing architecture
before longer outputs. Do not silently relabel that narrower workload as the
old v5 task. Remove answer-revealing factor inputs; feed only document-derived
neuronal memory/read activity, retain fresh swapped values and source removal,
and preserve writer-frozen, writes-disabled, delay/interference and real
execution controls.

Any upstream teaching signal needs its own explicit local contract, for example
neuron-level residuals propagated through the **pre-update** phase-gated output
synapses, with the existing upstream local traces. That weight transport is
declared; it does not establish that the writer or addressing will learn the
needed representation. The legacy host motor's action matching also remains
supplied scaffolding. The shared destination is learned neuronal answer and
action selection from retained document knowledge, not an indefinite sequence
of answer-revealing output benchmarks.
