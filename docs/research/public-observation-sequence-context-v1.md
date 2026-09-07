# Public observation, reusable context and content binding

2026-09-07. **Implemented and exercised in the completed broader V8 session.**
The [roadmap](roadmap-reading-to-coding.md) owns
selection. Use one locally learned recurrent controller with two content-address
populations inside the existing acting reader. Its choices learn through their
own stochastic eligibility and observed outcomes. Independent source/math review
and root's causal-input review are complete. The
[V8 result](../../experiments/r4_integrated/persistent_language_action_v8_result.md)
establishes no useful task advantage over disabled learning. The
[two ordinary checks](recurrent-content-qa-v1.md) verify mechanics separately.
The design below records V8's implemented rule; the
[learning review](persistent-language-action-v8-learning-review.md) owns the
measured diagnosis and next credit-refinement design direction.

## Actual source and rejected patch-only option

Graph discovery used the canonical checkout after verifying its root. Current
Dendritic/RawObservation symbols were absent, so exact-file reads supplied them.
Mem0 returned unrelated older experiments and was not scientific evidence. No
model, feature, raw journal, checkpoint decoder or experiment ran for this review.

[RawObservation.sensor_frame](../../src/humanbrain/sequence_assembly_observation.py)
preserves the whole request, four paths/files, diagnostics, signed returncode
and twelve public flags. Missing differs from present-empty. The
[inherited preparation](../../src/humanbrain/sequence_assembly_memory_learner.py)
builds this sensor and all factual contexts. The
[tournament reader](../../src/humanbrain/tournament_recognition_learner.py)
selects memory from request/prefix activity and includes its32 prior/32 forward
rows in recognition input. File/feedback bytes influence recognition, though
they do not directly enter that address score. The
[dendritic reader](../../src/humanbrain/dendritic_sequence_learner.py) evaluates
the inherited prior once per actual generated prefix, but its new branches
receive only prefix, encoded request and prior currents. Their categorical
support can outrank the context-sensitive prior.

Field-bound ordered patches could make this override more selective. **Reject
that as the next broader learner:** literal conjunctions alone add no reusable
operation or variable binding. They would not explain new wording or structural
composition after [V7's zero fresh/recovery scores](persistent-language-action-v7-learning-review.md).
Those scores do not identify a particular interference mechanism. The one
controller/content proposal below supersedes the earlier patch-overlay draft;
there is no second patch bank, API or experiment queue.

## What primary sources supply, and what they do not

Müller et al. demonstrate Hebbian assembly projections between structural and
content spaces. Their simulations assume content representations and schedule
disinhibition; their sentence example supplies agent/patient timing and ignores
other words. This supports a binding primitive, not a raw-request-trained
selector of the appropriate structural space.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7266140/).

Martin and Doumas preserve role/filler independence through time-separated
representations. Their model assumes objects/features; the paper explicitly
does not provide a complete parsing model. Its relational results cannot be
imported as evidence that our raw byte stream already supplies roles.
[Primary paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2000663).

Williams supplies a local control-learning route: stochastic units associate
reinforcement with their own score-function eligibility, subject to baseline
independence conditions. It requires no downstream Jacobian. This permits an
explicit local gate-learning proposal, not a claim of fast language learning.
[Primary paper](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf).
The following is our design inference, not reproduction of those experiments.

## Causal populations and fixed content relays

Use32 mutually exclusive recurrent context cells h, two typed one-hot address
populations a1/a2, eight gate cells, four emission-mode cells and257 literal
byte/EOS cells. Context cells have no assigned linguistic meanings. Addresses
refer to the whole current public frame or admitted factual memory, never a
teacher span, task ID or host-selected semantic role.

Present observations in existing fixed field order, one symbol at a time:
request, actual diagnostics/returncode, all paths/files, then explicit flags.
Use raw identities0..255, EOS256, BOS257, absent258, fixed field identities and
observed flag values encoded through byte0/1. These are declared IO boundaries,
not a parser. A source document uses the same stream with a source-field tag.
No pretrained encoder or host-produced phrase grouping is supplied.

At each new source document or public observation, reset h to fixed cell0,
both registers to invalid, and trajectory eligibility to zero; weights and
baselines persist. Preserve h/registers across fields, into the actual output,
and in the saved pre-output state used by a correction. Within a field, offer
each real byte once and then exactly one EOS, including present-empty fields.
BOS and an absent-field marker are public framing inputs, not prediction
targets or rewards. Do not train on fixed-size padding. The twelve Boolean
flags form one final observed0/1 field plus one EOS; charge those13 symbols
as framing/status supervision separately from raw content. Field transitions
set last-symbol BOS but do not reset h. After final field EOS, use output phase
and output BOS, preserving the common state and eligibility. This fixes the
causal boundary/offer count rather than treating a whole frame as one label.

A register captures the current visible input address or the actual inherited
factual winner, plus validity. Fixed sensory relays read its current symbol.
Within-field successor relays advance it; EOS remains a symbol and moving past
the field invalidates the register. No crossing into another file/diagnostic
is implicit. Factual successors use existing admitted tape links. Unseen source
suffixes are visibility-masked even if a private whole-document tape has been
allocated. A read cannot access future symbols to predict them.
Apply that visibility restriction to every public input field and register
read, not just standalone source documents. A full caller-supplied frame is
not fully visible to predictive scanning merely because its buffer exists.

Implement a read as the one-hot gated sum
`c_b=sum_a address_a*visible_a*sensory[a,b]`, with an explicit invalid output
when no visible address contributes. Advance is the fixed successor relation
applied to that one-hot address; bind and swap are gate-controlled copy relays.
Public positions include the3,471 sensor slots, one additional returncode-EOS
slot and13 flag/EOS positions:3,485 public plus2,048 factual slots, or5,533
total; invalidity is separate. Padding remains
invisible. A source document uses a separate, expiring input frame within the
same declared public-address capacity. No address is interpreted semantically.

Eight learned gate choices drive fixed relays: HOLD, BIND1_CURRENT,
BIND2_CURRENT, BIND1_RETRIEVED, BIND2_RETRIEVED, ADVANCE1, ADVANCE2, SWAP.
CURRENT means the just-received symbol during input and is invalid during
output. RETRIEVED is valid only when the inherited competition actually ran;
there is no extra retrieval during source scanning. Invalid binding remains
invalid, never silently repaired to an answer location. Count all gates/reads.

This separates content from operation: the same learned transition and
ADVANCE/read choices can emit a new literal from a newly bound address, without
storing that literal in their weights. Two registers can preserve independent
contents while the controller changes which it reads. This supplies capacity
for substitution and ordered recombination, not proof that learning discovers
it. Two cursors and32 states do not implement arbitrary nested programs,
arithmetic or a hidden program interpreter.

## Forward cells, stochasticity and byte output

Form x from the existing raw-request population Q, last received/generated symbol,
field identity, public source/input/output phase, previous h, both register
read symbols/validity, actual prior winner where available, and bias. No
current next-symbol target enters x. Normalize to unit Euclidean energy with
counted sums/square-root/division; the bias makes the denominator nonzero.

Q is absent during standalone source observation. While scanning the request,
it encodes only the already received request prefix: never encode a future
request byte or its EOS before arrival. Suppress the final request-to-EOS pair
until EOS arrives; retain the other existing feature identities. Once fully
observed, hold complete Q through later fields and generation. Public phase
distinguishes standalone-source absence from a completed empty request. Thus
next-request-symbol supervision cannot leak through whole-request features.

Context currents u=W_h x undergo stable softmax and one categorical draw h'.
Gate currents W_g x_g use the same inputs with h' replacing h; draw one gate
and apply its fixed relays. Normalized z contains h', last symbol, updated
register reads/validity, phase, prior winner and bias. Draw an emission mode
from softmax(W_m z): PRIOR emits the frozen V7 winner, REGISTER1/2 emits its
actual byte/EOS, LITERAL draws from softmax(W_o z) over all257 columns. Invalid
register emission produces EOS and records invalidity. It does not consult a
target. During source scanning PRIOR has no V7 evaluation and emits the declared
absent-prediction marker258, always incorrect for a real byte/EOS. This differs
from using an oracle prior or constructing old queries per source symbol.

Raw-observation predictions never replace arriving symbols. During free output,
only the actually emitted byte advances the prefix. EOS and the inherited128
byte limit remain explicit. The frozen V7 prior runs once per actual output
prefix. No complete alternative answer or separate executor is generated.
The later correction branch likewise evaluates the frozen prior and its
factual competition once per causal teacher prefix, counted as a teaching
forward, using the same original observation and only earlier offered teacher
bytes. Count the retrieved-cursor activity as well. It cannot reuse the actual
free prefix's diverged winner. Both source
and public-input scanning have absent prior and invalid RETRIEVED addresses;
neither invokes an old model prediction to obtain a next-symbol target.
Use a specified saved RNG and stable inverse-CDF sampling with no rerolls;
record probabilities, actual draws and RNG frontiers. Every categorical
alternative participates, including choices whose outcome proves unhelpful.
Pin a dedicated CPU torch.Generator seeded733, its implementation/runtime
identity and full state. Draw one float64 uniform per sampled population;
accumulate its float64 CDF in ascending cell order, require finite normalized
probabilities, close the final endpoint to1, and select the first cumulative
value strictly greater than the uniform. Record this rounding convention;
finite sampling is not exact real-arithmetic sampling.

A concrete weight layout has x width69,954: Q68,868, last-symbol259, field14,
phase3, state32, register symbols518, validity2, prior257 and bias1. The14 field
identities cover twelve byte fields, flags and source. z width1,072 omits Q
and field. Matrices32x69,954, 8x69,954, 4x1,072 and257x1,072 contain3,077,952
float64 values, **24,623,616 bytes**. Biases are columns. All weights start zero;
stochastic activity breaks symmetry without hand-assigned semantic states.
This is plastic recurrence with explicit content relays, not a fixed random
reservoir plus a trained readout. Zero initialization creates exploration,
not functional parity with V7.

## Own-synapse learning, without reverse transport

For stochastic population k, preactivity x, probabilities p and sampled
one-hot emission y, the direct local eligibility increment is

`e^k_ij(t) = (y_i(t)-p_i(t))*x_j(t)`.

This is the derivative of that population's own log sampling probability with
respect to its direct afferent. Stop differentiation at all preactivities,
previous states, address relays, earlier choices and prior currents. There is
no BPTT, RTRL, synthetic gradient or downstream output-error chain rule.
Only populations actually sampled accrue eligibility; unused LITERAL draws
are not fabricated.

A source document is one trajectory. A task interaction has a common observed
input scan, an actual free response, and (only when offered) a later teaching
branch from the saved pre-output state. **Hold controller weights throughout
all three.** Do not publish the scan update before acting or clear its
eligibility before a correction: that would prevent answer outcomes from
teaching which input content to bind. The teaching branch is a declared local
correction rollout, with no second scan or new answer exposure.

Accumulate separate own-synapse sums E_scan, E_free and E_teacher for these
actual sampled paths. At each externally arriving byte/EOS, r=+1 iff its
pre-arrival prediction matches, otherwise -1. R_scan and R_teacher are the
respective mean outcomes. Let R_public=-1 on public execution/capture failure
and0 otherwise; use0 for an actual STOP. No private grade enters any reward.
For each applicable category form

```
G_scan    = (R_scan-b_scan)*E_scan
G_teacher = (R_teacher-b_teacher)*(E_scan+E_teacher)
G_public  = R_public*(E_scan+E_free)
Delta W   = (0.01/L)*mean(applicable G categories)
```

The public baseline is fixed0, avoiding action-selection dependence. Other
baselines are prior completed category means, held before scanning. There are
three stored baseline slots: source scan, public-observation scan and teacher
output; the first two condition the scan mean on the externally scheduled kind.
Update them only after forming all proposals. L is the external upper bound on ticks:
source offer count for a source trajectory; scan offers + offered correction
symbols +129 for an interaction. Unused free ticks after EOS are absorbing,
with no draw or eligibility. L never uses policy-dependent generated length.
Absent categories contribute no fabricated reward. Category means are an
explicit equal weighting of different observational signals, not a hidden
claim that821 bytes equal821 independent tasks.
Category inclusion and its averaging denominator are fixed by the external
source/reading/edit/correction schedule. They never depend on correctness,
sampled modes or selected STOP. In a scheduled edit, STOP keeps the public
category present with R_public=0; it does not remove that category or rescale
the other rewards. Scheduled reading has no public execution category.

For fixed weights/external symbols, the joint probability factors into the
common scan draws and the two conditional response branches. The teacher
reward is causally downstream of scan and teacher choices; public execution
is downstream of scan and free choices. Their direct log-probability
eligibilities are exactly the sums above. Conditional outcomes independent
of another branch require no derivative through that branch. Thus a common
input-binding choice can receive answer credit without a correct-address
label or backward transport. Baselines fixed before the common scan and
exogenous normalization preserve this score-function argument. Intermediate
weight updates or a baseline depending on current choices would change it.
This is gradient-following in expectation, not per-trajectory descent.
The claim is conditional on the sealed initial interaction state, its public
input and fixed old prior. It is not a full-history gradient through previous
learning, earlier workspace actions, memory construction or initial state.

Disabled routes compute proposals but publish neither weights nor baseline
changes; their activations/registers still evolve. With unit-energy inputs,
each population eligibility increment has norm at most sqrt(2); at most four
populations sample per tick. Each category's causal eligibility sum has norm
at most L*sqrt(8). Since outcomes/baselines lie in[-1,1], the proposed update
norm is at most0.02*sqrt(8). This bounds a finite step, not unlimited weight
growth or useful learning. Strict finite checks remain mandatory. The0.01
is a declared starting convention, not a fitted gain or proposed sweep.
The credit is noisy: a wrong address can emit a common correct byte, and a
trajectory outcome does not identify the useful bind. Long delays increase
variance. But unlike a fixed-context recognizer, the weights controlling the
bind and recurrent state can themselves change from the offered outcome.
This is the precise new learning opportunity, not an assertion that it works.

## Same actor, observed feedback and actual ownership

Scan each public observation once, with every next-symbol offer counted.
The final scan symbol/EOS closes its input phase; seal the resulting state
and retain scan eligibility and unchanged weights before actual generation.
For a later correction, start a private trajectory from that pre-output state
and consume only its already-offered causal symbols. Do not scan/train the
observation twice or substitute teacher prefixes into the earlier free answer.
The correction is teacher-forced, not the actual free trajectory.
Restore h/register/prefix state, **not** the old RNG position. Teaching uses
fresh draws from the same monotonic generator after the actual free response.
Under the ideal independent-uniform model, the free stopping time does not
recycle randomness into the conditional teaching branch. Record both RNG
frontiers and all draws; do not choose seeds or rewind to obtain favorable
coupled responses. A deterministic finite PRNG approximates this stochastic
model; it does not turn its expectation statement into exact finite-run algebra.

The same existing actor consumes the payload, selects operation/span and
executes the exact splice. Action teachers retain their existing separate
local rule and ledger. Seal observation, tape/model/RNG identities, sampled
states and actual execution; teach must validate the same pending response.
Before source/reader/action publication, allocate receipts and validate all
proposals. Prepublication failures leave live state unchanged. Stop and retain
evidence on later journal failure; do not promise rollback after live copies.

The consequence category above consumes sealed actual free eligibility only
after execution. Zero public reward for STOP avoids assigning its unused
payload a failure penalty. Returncode0 is not semantic success: this can learn
avoidance of coarse process failures, not correct repair. Count consequence
updates separately and preserve a feedback-present/update-disabled control.
More informative repair requires actual public checks or observations, never
the private correctness bit. V7 had no such feedback-driven update.
## V7 continuation and state compatibility

Authenticate V7 safe HEAD/frontiers and load its original schema only in a
separately authorized continuation. Preserve original artifacts, all old tensor
values, tape rows, counters, projects and reference history. Freeze old reader
parameters as prior; add new controller, baseline and RNG state under a new
schema. Do not replay750 source/71 answer symbols or reconstruct the seven
missing attempts. Both matched neural routes start from independent copies of
V7 online, as selected in the active development plan, with the same seed733
RNG convention. The reference restores its own actual V7 history. Preserve the
different original V7 routes as historical evidence; do not relabel their old
outcomes as results from this new matched initialization.

The new stochastic policy changes initial behavior even with preserved old
weights: an enabled/disabled match is necessary, and numeric continuation must
not be advertised as V7 behavioral parity. Checkpoint all learned weights,
baselines, RNG state, cursor validity/addresses, live context where applicable
and unresolved eligibility boundaries. Idle transaction boundaries discard
completed eligibility, not learned recurrence or persistent knowledge. Public
observation addresses expire with their immutable backing frame; factual
addresses remain bound to the admitted tape identity. No stale pointer follows
an unrelated workspace reset.

## Exposure scale and bounded ordinary QA

V7's821 new source/answer symbols and64 branches per byte do not constitute a
learned language/code model. A prospective broader observation curriculum needs
many distinct sources, programs, requests and genuine before/action/after
observations, repeating structural regularities across changing content and
wording. An order of10^4–10^5 newly observed symbols is a planning scale, not a
proven sufficient amount or authorized run. Include intervening material and
revisions. Do not manufacture role labels or repeat exposed lessons until they
succeed. Raw prediction may learn syntax without instruction-following; genuine
meaning and useful control remain separate measured outcomes.

V7's1,614 occupied factual slots cannot retain10^4 new raw symbols within its
2,048-slot capacity. Such an exposure proposal therefore needs an explicitly
reviewed capacity increase or a declared distinction between transient observed
streams and retained factual admission. No silent eviction/truncation or claim
that transient input is permanent memory is allowed. That cost decision precedes
any curriculum release.

Count source symbols, request/file/diagnostic predictive offers, correction
symbols, action labels, actual free choices and public-consequence events
separately. Re-reading already stored bytes for learning is another exposure.
Fewer answer corrections and more observed-language experience are different
claims. No corpus fetching, curriculum generation or experiment occurs here.

Weights, three eligibility banks and one private proposal require about123.1 MB
before old models, active input/visibility buffers, RNG and journals. Sparse
active afferent products may reduce tick work, but all alternatives and actual
allocations must remain accounted. The old prior remains the output-time cost;
source scanning does not invoke it. Longer trajectories cost time and noisier
credit despite bounded state. No claim that32 cells are sufficient follows.

Two ordinary mechanics nodes in the existing test home can check scalar
softmax/inverse-CDF draws, own eligibility and baseline timing, update bound,
bind/advance/invalid/visibility relays and disabled publication, then one
same-actor response/correction/execution lifecycle with late atomic rejection
and strict RNG/state restoration. Use small authored in-memory fixtures, no
historical checkpoint or exposed-task fit. Count every symbol, prior call and
child. These checks are arithmetic/custody QA, not proof of learned generality.

Root selects this recurrent-control/content-binding design, superseding the
patch-only option. It can represent content-independent
bind/advance/read policies, while its own local eligibility can train their
selection. The remaining empirical barrier is whether observed-byte/public
outcome signals teach useful state partitions and gates at affordable exposure
rather than common-byte shortcuts. Ordinary QA now checks causal visibility,
sampling, local updates and actual RNG publication during disabled acknowledgement.
Weights stay fixed through the common scan and its actual/correction branches.
The next integration must authenticate historical initialization and distinguish
transient observations from factual admission in persistent transactions.
The resource and saved-cost ledgers account for sparse products, eligibility,
teacher-prefix prior calls and dense custody; complete broader costs remain
unmeasured before choosing the new exposure budget.
Root can then release one broader curriculum measuring acquisition,
new forms/combinations, retention and genuine recovery; no toy-perfection gate
or gain sweep is implied.
