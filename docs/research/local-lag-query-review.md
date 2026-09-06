# Sequence-memory follow-up: expose query evidence directly to local address learning

2026-09-06. Root-selected structural comparison after independent scalar/source
review. This design selection does not admit a cost workload or fit. No model construction,
checkpoint forward, fit, new query, teacher or action was performed. Research skill
applied; Hubble supplied the parallel saved-prefix analysis. Mem0 search was external
development context and supplied no scientific evidence. Implement this one
integrated change, finish ordinary QA and freeze complete resource/custody
bindings before fresh cost admission.

## What improved, and what did not

The completed five-arm run at `0b3d7d34f451fbbd315f472aa07732c0cd3eb0e0`
passes its saved audit but fails the preregistered useful-completion criterion.
Learned scores are 4/8 fit, 2/4 new values, 2/4 delayed and 1/4 interference.
Each of address-frozen, slow-frozen and writes-disabled scores zero. The exact
a1ab byte reference scores 5/8, 2/4, 2/4 and zero. Thus the learned addressing
path contributes behavior relative to matched controls; it does not beat the
strongest reference on novel count or establish the complete task.

The learned arm makes 435/1,024 successful first attempts versus 159 for the
byte reference. Its successive 128-lesson success counts are
`0,16,68,66,72,74,69,70`; mean per-offer NLL falls from 2.495589 to 0.344678.
That is acquisition under the fixed exposure, not evidence of convergence.
Byte half-SSE and sequence NLL must not be compared numerically as one loss.
The monitored run plus audit takes 213.520821 seconds; separate peaks are
1,094.074 MiB and 715.738 MiB. The full-run peak exceeds the disposable estimate,
so growing receipt storage remains a real admission consideration.

More importantly, all four fit memories return the same answer for their two
different questions: `aa/aa`, `bc/bc`, `cc/cc`, `da/da`. The k00 assignment swap
produces both `ac` and `ca` correctly. The k02 assignment swap produces `db`
and `bd`, both wrong. One correctly reversed assignment pair is informative
source sensitivity, but does not by itself establish query-specific binding:
the same record preference could explain it. Delayed repeats are duplicates of
these same four stored memories, not four more independent generalizations.

Hubble's saved priors sharpen the diagnosis. Across all twenty learned cases,
literal value-start branches collectively receive 0.991986–0.996018 of initial
address mass; address-frozen receives approximately 0.0455–0.0521. The learned
system therefore acquires a generic content-location preference. The four paired
fit questions change their address distributions by total variation
`0.0207145, 0.00335043, 0.00636535, 0.00829336`. Their correct-direction target
log-odds shifts are `0.0806565, 0.0119143, 0.0242758, 0.0253973`, but none
reverses the preferred value-start branch. The query effect is weak, not zero.

All twenty learned outputs are complete two-byte values found somewhere in their
source memories, with EOS. At the third decision the leading branch sees period,
and the observed EOS mixture probability is approximately 0.91533. This is not
a general codec qualification, but it argues against another output-only repair.
Interference case 16 also exposes mixture competition: the correct k00/ac start
has the largest individual prior (about 0.21383), yet c-starting branches from
ca/cc/cd together hold about 0.5989. The first generated c then favors cc. A hard
single-address switch might change that case; it would not repair the wrong
rankings in the paired fit questions and is not selected here.

Evidence, all saved JSON:

- `.pytest-tmp/local-sequence-memory-v1-description.json`, SHA256
  `bb1ca936bbff535ef3883a73fc4155001ba377b975fa0351cd58d2082d5dadd2`.
- `.pytest-tmp/local-sequence-memory-v1-run-independent-audit.json`, SHA256
  `3b32eedbcb583d0ea3fd2e7a742c3b177f98a7fe4da3c4d5d7e6f42768a0e73c`.
- `.pytest-tmp/sequence-saved-routing-review.json`, SHA256
  `913e99731fe75d01fff587e96e6d8a530a4aaf1b08876f0b8b364f89551a6e6f`.

## Selected direct query interface

**Test a direct generic lag-query address learner in the same integrated task.**
Replace the 32-neuron recurrent query compression and its self-path eligibility
with a fixed newest-first population holding the last 32 raw byte codes, followed
by one learned 1024-by-1024 query-to-address projection. Keep the literal tape,
stored key normalization, all-start mixture, forward adjacency, conditional
codec, current-prefix codec credit, held lesson publication and eta=0.01 fixed.
No parser, identifier detector, selected lag, answer factor or host-trained probe
is introduced. Do not stack a residual encoder or another routing policy onto it.

The competing explanations are (1) poor accessibility/conditioning of identity
evidence through the learned recurrent query and approximate temporal credit,
and (2) insufficient learned discrimination under this objective and fixed data,
including common-prefix pooling. Saved receipts contain no query activity u or
projected query q; they cannot distinguish representation contraction from a
poorly learned projection, and do not establish that the present 32 dimensions
lack representational capacity. This is a prospective structural comparison of
direct feature access and exact local projection credit, not a proven repair.
It changes width, representation and removal of recurrent credit together; a
positive result must not be attributed uniquely to one of these changes.

Another frozen query-geometry diagnostic would report distances or ranks but
would not settle whether more discriminative learning is achievable. Existing
paired priors already identify the actionable weak link. The direct comparison
therefore offers more return than another observation-only loop. More epochs,
temperature, eta, hard-address selection or codec-objective changes are deferred.

## Exact interface, locality and signal conditioning

For every raw query, initialize a 32-by-32 lag population to zero. Feed every byte
and one distinct EOS through the existing fixed code table, shifting the same
generic fixed adjacency after each symbol; flatten newest-first as a in R^1024.
EOS occupies the newest lag. Normalize the complete population generically as
`z = sqrt(32) * a / max(||a||, 1e-8)`. No lag or byte class receives preferential
gain. This is a fixed total-energy representation, not a fitted gain, learned
normalizer, lexical cache or question-specific reset. A longer query
retains its last 32 symbols, explicitly; this is finite sensory context, not
lossless retention of an arbitrary-length question. Reset only between queries.

Use one shared learned matrix A in R^(1024 by 1024):

```
q = A^T z
s_j = q^T k_j
p = softmax(s)
e_j = p_j - b_final,j
h_i = sum_j e_j k_j,i
A_ri <- A_ri - 0.01 * z_r * h_i
```

Here b_final is the existing causal teacher-prefix belief, with A and the codec
held throughout the lesson. Address neuron errors e travel through the actually
stored key synapses to projection neurons h. Each A synapse receives its own
presynaptic z and its postsynaptic h. Since z is fixed sensory activity, there is
no query-neuron feedback computation `A*h`, query-gradient vector, query update
group or query eligibility to retain. There is no externally supplied matrix
gradient, solved target address, backwards traversal, parameter tying across
independent slot neurons or retained query-learning history. The projection
update is the exact instantaneous sequence-NLL derivative conditional on the
held codec and fixed features. Codec filtering remains the existing declared
approximation; changing the query path does not make it an exact full-sequence
codec gradient. [Bellec et al.](https://www.nature.com/articles/s41467-020-17236-y)
distinguish local eligibility from neuronal learning signals; the present
projection derivation eliminates recurrent eligibility rather than claiming the
paper guarantees this task's convergence.

Use CPU float32 codes, lags, A and address forward values, and the existing
float64 belief/error reductions. Accumulate h and the local outer contribution
in float64, finite-check, then cast the contribution once to A's dtype before
the eta update, checking the candidate before publication. Preserve old-weight
codec timing. Query-cell parameters, traces and query-gradient receipt fields
are retired, with an honest new schema rather than dummy arrays.

Initialize the new A using the same private seed+4 Xavier convention for its
actual dimensions; preserve code-table seed+2 and codec identity initialization
exactly. Its initial predictions need not equal the current sequence reference:
the dimensions and query activation norms change. The frozen-address control
must share this new initialization, so gains cannot be credited merely to it.

The activity budget is derived before any trial. The previous 32 rate neurons
have absolute activity at most one, hence total squared activity at most 32.
The replacement holds the same maximum total energy while distributing it over
all 1,024 lag coordinates. It preserves the direction and relative coordinates
of the raw lag vector, not its absolute norm; this is neither an injectivity nor
a general language-representation theorem.

For a fixed query, the real-arithmetic local A update induces
`q_new = q - eta*||z||^2*h`. For held codec and a complete teacher sequence, write
the branch likelihood as ell_j. The address objective is
`L(q)=logsumexp_j(q·k_j)-logsumexp_j(q·k_j+log ell_j)`.
Its gradient is h and its Hessian is
`Cov_p(k) - Cov_b_final(k)`. Under the declared real-arithmetic normalization,
every stored key has norm at most one, so
each covariance is positive semidefinite and bounded above by the identity;
their difference has operator norm at most one. Thus L has a globally
1-Lipschitz gradient in q. The descent lemma gives
`L(q-alpha*h) <= L(q)-alpha*(1-alpha/2)*||h||^2` for `0<alpha<2`.
Here `alpha=eta*||z||^2<=0.32`, giving a sufficient conditional descent bound.
The two matrix publications and float32 forward rounding are not certified by
this real-arithmetic inequality; finite checks and ordinary scalar QA remain.
Publishing codec changes simultaneously also changes L, so this is not a
whole-learner monotonic-loss or convergence guarantee.

Reject the initial raw-amplitude draft prospectively. The fixed questions have
25 bytes plus EOS, or 26 active lags; independent unit-variance codes give expected
raw energy 832, hence expected alpha 8.32 at eta 0.01, outside the sufficient bound.
That does not prove overshoot, but cannot justify calling raw updates safely
conditioned. No endpoint activities or trial gains were measured to set this
normalization. A unit-norm query would give alpha 0.01; a conventional
presynaptic-norm-normalized local update could similarly bound the effective
step, but would instead change the plasticity rule while leaving forward
amplitudes length-dependent. Select fixed energy32 because it preserves the
prior query population's total activity bound with one explicit forward circuit
and retains the existing eta/local outer rule. Do not compare or sweep those
alternatives in the proposed run. It is unknown whether this bounded signal is
large enough to learn reliable discrimination within 1,024 lessons. Carson's
independent source/algebra review confirms the covariance and descent bounds;
the condition also does not guarantee retention of other questions after one
question's local update.

## Fixed comparison, resource boundary and return

Use six arms: direct-lag learned, direct-lag
address-frozen, direct-lag slow-frozen, direct-lag writes-disabled, exact current
0b3d7d3 sequence reference, and the a1ab byte reference. Address-frozen now freezes
only A because there is no learned query cell. Slow-frozen retains literal sensory
writes and freezes A/codec. Memory-off keeps the same allocation/adjacency and
zero sensory/key activity. No obsolete writer/tag or separate query-freeze arm.
Both references use static experiment archives and must reproduce their saved
initial/final common state, 1,024 scientific lesson receipts and twenty outputs.

Keep 1,024 ordered lessons per arm, seed733, eta0.01, first-before-three-teachers,
the same exposed 8/4/4/4 action cohort and two-byte cap. Record all six arms'
outcomes and retain strict 8/8 plus 4/4 in each other stratum, both novel assignment
pairs correct, and more novel successes than each matched control and reference.
Additionally report paired different-query answers and correct-direction target
odds from the ordinary saved priors, without adding forward calls or thresholds.
No useful-completion claim is inferred from merely larger query contrasts.

The new A contains 1,048,576 float32 synapses (4,194,304 bytes), compared with
32,768 previously. Retiring the 2,080 query parameters gives a new model payload
of 4,757,776 bytes, versus 702,864. The live lag population is 4,096 bytes, with
no eligibility bank. Tape allocation stays 656,904 bytes per episode. Float64
local A accumulation alone is 8,388,608 bytes; its parameter proposal, casts,
initial/final model copies and returned receipts are additional. Do not serialize
full A changes, query-feature matrices or h per lesson; ordinary scalar QA checks
those locally, while the fixed run retains hashes, required small beliefs and
existing work receipts. Actual watchdog peaks remain authoritative.

Root selects streamed per-lesson JSONL for this comparison's reporting. Preserve
every scientific row, ordering, reference field and count; include append/flush
work in the measured workload, bind the final log hash/count in completion, and
release each row after writing. Keep only fixed scalar/hash/count summaries in
the parent. The new audit streams these rows rather than materializing the full
learning history. This changes evidence storage, not learning or exposure.
Cost auditing must include the same historical-report parsing as full auditing,
with explicit accounting for any full-only historical checkpoint payload. Bind
the finished cost-audit launch as well as the model-cost launch when deriving
final admission. Freeze the exact memory formula and receipt schemas before
cost execution; the four-lesson working-set sample alone is not a full-run bound.

The prospective journal guards are 262,144 bytes per serialized line and per
recursive CPython row-tree estimate, with at most two new lesson rows live in
either runner or auditor. Their explicit transient allowance is
`E = 2*2*262144 + 2*262144 = 1.5 MiB`. Let C_run be the larger of the disposable
parent-plus-worker peak and its owned-tree peak. Estimate the full run as
`C_run + E + 32 MiB`. The fixed 32-MiB allocator/metadata reserve is an engineering
margin, not a formal allocator bound. For auditing, include the same historical
report parsing and lifetimes during cost and full audit, and load historical
checkpoint comparisons sequentially. Let F be the authenticated full-only
historical tensor payload: 232,400,120 bytes for a1ab plus 80,431,116 for 0b3d7d3.
Estimate full audit memory as the finished cost-audit owned-tree peak plus
`E + 2*F + 32 MiB`. Both estimates must fit 2,048 MiB. The cost checker explicitly
leaves audit-peak admission pending; only a separately bound completed audit
launcher can finalize admission. The full runner and independent auditor must
rederive the final projection from those bound receipts.

Project learning time as twice 1,024 times the sum across arms of the separate
maximum measured model-lesson, maximum journal and complete-cap durations.
This is at least as conservative as the maximum combined lesson/journal sample.
Other time is `2*(cost launch duration - sum(model durations) - sum(journal durations))`,
followed by the existing 30-second allowance. Row bounds,
serialization, writing, flushing and incremental hashing belong to measured
journal time. Freeze these constants before cost measurement; do not adjust
them retrospectively to admit the run.

Require fresh six-arm cost admission: four ordered samples per arm, 24 lessons,
72 teacher offers, all 120 final actions plus two demos, and six complete-cap
fixtures with 127 period bytes plus EOS. Count raw query/EOS lag transitions,
one whole-query population normalization per traversal,
1,048,576 projection synapse evaluations per query, actual address/codec/cursor
work and enabled local updates separately. Retain the existing 2x measured
projection plus 30 seconds and the 900-second/2,048-MiB ceiling; include the saved
audit in the aggregate execution ceiling. Stop on cost, custody or finite-check
failure; do not shorten the workload or adjust gain to force admission.

After ordinary mathematical/interface QA and a successful fresh cost audit, run
exactly one integrated fit. On failure, close this direct-feature address
hypothesis before another mechanism. On useful success, return to longer content
and varied document/query forms in the same real Python action loop. Literal
storage, generic copying and the supplied action motor remain explicit limits;
there is no protected advancement or claim of general language understanding.
