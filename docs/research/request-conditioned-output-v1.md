# Request-conditioned neuronal output

Status: root-selected source implementation, 2026-09-07. No model, checkpoint,
feature, learning or task execution is authorized by this document. It follows
the [broader development plan](persistent-language-action-development-v1.md)
and its [completed learning review](persistent-language-action-v1-learning-review.md).

## Choice and scope

Two small candidates were considered. An additive request/prefix residual with
the existing softmax-error update is easy to attach, but lacks nonlinear binding
between request and retrieved content. A zero residual also initially competes
with the old codec's approximately log(256) identity margin. With presynaptic
energy32 and rate.01, an individual direct CE update changes a matched current
by at most.32. This is not a sufficient one-pass correction guarantee.

Select instead fixed mixed threshold cells, a locally plastic residual byte/EOS
output layer, and **new all-memory retrieval at every output step**. Use a
normalized postsynaptic-potential clamp for that output layer. This is neither
an operation-only change nor a quote/span parser. The same reader produces
language answers and action replacement bytes; the existing neuronal action
heads and real executor remain in the same persistent learner.

Continue the completed broader online model and its actual453-slot tape. Retain
all old numerical tensors, including address B, codec and action heads. Add
prefix-address C and output U, initialized to zero, plus fixed source-bound
wiring. The old codec is a fixed numerical output prior in this version;
B continues learning, C/U learn, and action-head learning remains unchanged.
There is no claim of initial output parity: reselecting an address each step
removes old cursor advancement, so zero C/U can repeatedly choose the same byte.

## Exact forward population

Bounds remain N≤512 occupied slots, request≤31 raw bytes, output≤16 raw bytes
plus one EOS/limit decision. All memory addresses remain eligible, including
document EOS. No teacher address, answer-containing document selection or host
role extraction is permitted.

At step t=0..16, Q is the entire raw request followed by EOS, beginning-aligned
in32×18 dual-rail rows. P is BOS followed by the **actual previous output bytes**,
beginning-aligned in17×18 rows and empty thereafter. In correction only, those
previous bytes are the already offered teacher prefix. The current target is
absent from both input populations. A position population has17 cells with
only cell t active. Empty rows are allzero; byte0, EOS256 and BOS257 remain
distinct. Encoding uses the existing generic nine-bit/18-rail IO function.

Address inputs Xq_s and Xp_s are the existing occupied-count-normalized
coincidence features against each stored key, using newest-first32-lag versions
of Q and P respectively. There is no address-pool pruning.

```
a_s,t = sum(B * Xq_s) + sum(C * Xp_s,t)
p_t = softmax(float64(a_t))
w_t = existing two-threshold-layer competition(a_t)
```

B and C are float32[32,32]. Keep the two reductions separate. Exact score ties
use the established lower-index rule. C starts atzero. Recompute Xp and a for
every actual prefix. Xq is immutable within a call and may be held. Free output
uses the winner's conditional output, but the next step again considers every
address; it is not constrained to the winner's successor. This permits access
to different notes and noncontiguous content within one answer.

For each candidate s, define a64×18 Boolean context: its stored32 preceding
raw-byte rows, followed by current sensory and31 successor sensory positions.
The latter follow existing adjacency with absorbing EOS. There is no source
parser or target span. Past rows exclude EOS/BOS; future rows exclude BOS.
Current rails must equal context row32. Disabled content remains zero.

The mixed-cell input v_s,t concatenates Q576, P306, context1152 and position17:
2051 Boolean inputs. There are256 fixed threshold cells, each with32 distinct
within-block afferents:8 query,8 prefix,12 context and4 position. For each cell,
sample those indices without replacement using a local CPU generator seeded
`seed+104729`, then sample32 signs from{-1,+1}. This is fixed generic wiring,
not an initialization chosen from endpoint results. Use exact signed integer
currents and threshold atzero:

```
h_j = 1[sum_i sign_ji * v[index_ji] >= 0]
r = concat(h[256], current_rails[18], position[17], constant_one[1])
x = sqrt(32) * float64(r) / sqrt(sum(float64(r)**2))   # 292 inputs
d = sum(x*x)                                        # actual positive denominator
z_s,t = frozen_old_codec_current(sensory_s) + U @ x_s,t
c_s,t = softmax(z_s,t)
```

U is float64[257,292], initiallyzero. The constant input prevents zero energy.
Both byte cells0..255 and EOS256 are ordinary output cells. Free output chooses
the largest potential, with the existing lowest-symbol tie convention. EOS ends
the answer; a non-EOS decision at t=16 yields the existing visible limit result.
New bytes and punctuation need not equal the selected sensory byte. This admits
construction rather than enforced substring copying; it does not prove correct
composition, semantic binding or adequate capacity of256 fixed mixed cells.
Different contexts can collide in this feature map.

## Local correction and publication

One correction rebuilds inputs from the same raw request and authenticated
pre-action tape. It does not insert a target span or reuse an externally chosen
address. At every teacher position, evaluate allN conditional populations using
the earlier teacher prefix and the current **staged** B/C/U. Derive only a
current-position responsibility from actual forward likelihood:

```
b_s = p_s * c_s[target] / sum_j(p_j * c_j[target])
T_y = log(256) if y == target else 0
U_next = U + sum_s b_s * outer(T - z_s, x_s) / d_s
B_next = B - .01 * sum_s (p_s - b_s) * Xq_s
C_next = C - .01 * sum_s (p_s - b_s) * Xp_s
```

Compute b, z, x and both address deltas from the same pre-update staged values.
Then advance the staged parameters and earlier teacher prefix. No backward
history, optimizer, reverse-Jacobian or error through threshold/competition
cells exists. U receives output-cell potential error times that synapse's own
input, weighted by the explicit current teacher responsibility. B/C receive
their own score-cell activity error times their own coincidence input. This is
a declared local latent-address surrogate, not a derivative of hard selection
or a sequence-gradient claim.

The log(256) clamp reuses the old codec's identity-current scale. For one isolated
feature and responsibility1, a unit normalized update sets all output currents
to T in real arithmetic, giving the target probability1/2. It avoids a tuned
gain sweep. With multiple candidates, shared synapses and later teacher steps,
neither exact clamp satisfaction nor one-pass sequence retention is guaranteed.

For fixed x, old-codec currents and normalized b, the potential loss
`L(U)=.5*sum_s b_s*||z_s-T||²/d_s` has Hessian spectral norm at most1, because
each `x_s x_sᵀ/d_s` has norm1 and the weights sum to1. The U update is its unit
negative-gradient step, using only local output products, so this fixed-field
quadratic decreases in exact arithmetic. This mathematical identity does not
authorize reverse gradients. It does not prove decrease of mixture NLL or a
joint loss after b, B/C or the prefix changes. Address concatenated feature
energy is at most64; .01 is below the corresponding conservative fixed-target
softmax step bound. Floating arithmetic and cast/publication failures still
require finite validation, not an unconditional numerical descent claim.

Teacher steps are sequential in private staged copies; externally all model
groups remain held until the entire correction passes finite/shape validation.
Publish B/C/U together at the end. `learning_enabled=False` computes the same
offers and records work but leaves actual parameters unchanged; staging follows
the held disabled policy: candidate arithmetic is checked but staged parameters
do not advance when their publication group is disabled. Existing combined
action proposals must all validate before reader teaching; no fallible action
proposal remains after reader publication. A late failure publishes neither
reader nor action groups. New receipts distinguish `output_updated`, both address
groups and `legacy_codec_updated=False`. For the existing combined feedback
interface, `codec_updated` is explicitly an alias for new U output publication
under this new schema; it never means the legacy codec changed.

## Implementation and accounting seams

New fixed helper: `request_conditioned_output_cells.py`,
`RequestConditionedOutputCells.forward(Q,P,context,current,position,work=None)`
returns float64[N,292]. Fixed indices/signs must be validated before checkpoint
publication; the old combined loader only specially pins `reader.input_codes`,
so its new initialization/load boundary must also authenticate this wiring.

New reader: `request_conditioned_memory_learner.py`, class
`RequestConditionedMemoryLearner`, subclassing
`PersistentCoincidenceReader512`, with the same raw predict/teach entry points.
Override the complete generation/correction loop; the old cached final codec
probabilities and absorbing single-cursor receipt schema no longer describe it.
Immutable old codec currents and raw context may be cached within a call;
conditional potentials/softmax and prefix-dependent features must be recomputed.

Publish new per-step receipt fields for selected address, competition/work,
actual conditional potentials/probabilities and prefix; teacher fields contain
actual all-address p/b and group changes. Do not label an initial-only competition
as the whole trace. The parent must adapt its compact summary and metadata while
keeping real action-before-correction custody and raw execution unchanged.

C adds4096 bytes and U adds600352 bytes:604448 new slow tensor bytes. Fixed
8192 indices and8192 signs cost73728 bytes with int64/int8 storage, plus any
explicit buffers. At maximum N512 and17 positions, a full teacher evaluates8704
conditional populations:71,303,168 fixed afferent applications,2,228,224 mixed
threshold evaluations and653,182,976 output products. The exact product expression
is `512*17*257*292`; implementation accounting must use the integer expression,
not an approximate label. Free conditional output evaluates only17 winning
populations, but address construction and competition still run over allN at
every step. Count teacher staging, repeated prefix kernels, context construction,
normalization, all-address softmax and actual allocation separately; no measured
runtime or RSS follows from these arithmetic bounds.

Ordinary QA should cover authored request/prefix distinction, a single-feature
clamp and weighted multi-address oracle, actual free versus teacher prefix,
multi-address switching, EOS/limit, frozen and late atomic failure, strict
continuation state, and one same-instance actual action before correction.
These are mechanics checks, not another fitted byte-task gate. The next workload
must retain broader language/action/retention/recovery pressures and disclose
that the existing tape leaves59 slots. Public feedback can later enter through
an explicitly bounded raw observation channel or append event available to all
routes; this version does not silently consume stdout, private grades or answer
hints, and does not claim teacher-free recovery learning.
