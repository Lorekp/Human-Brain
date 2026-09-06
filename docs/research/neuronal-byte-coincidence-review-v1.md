# Generic neuronal byte coincidence before address learning

Status: root-reviewed source and six ordinary QA nodes selected, 2026-09-06.
After a committed source freeze and clean doctor receipt, execute the six nodes
specified below once under the qualified 60-second/2,048-MiB watchdog. Preserve
failures and actual exposure. No model construction, feature evaluation, teacher,
prediction or action outside that authored scope is selected. No historical
checkpoint load or acquisition experiment is admitted. Root owns the launch,
qualification result and later complete cost admission.
The [completed learning review](neuronal-splice-online-v2-learning-review.md)
motivates discrimination before commitment; its exposed outcomes are not a
training set or a counterfactual test of this design. The companion
[primary-source review](neuronal-splice-discrimination-sources-v1.md) examines
representability and why supplied-correct-address learning theorems do not
justify an inferred address oracle here.

## One candidate and its physical forward calculation

Recommend a **fixed dual-rail coincidence frontend followed by the existing
shared temporal output layer**. Keep initial neuronal winner selection,
single-cursor free recall, codec, action heads and current local teacher rule.
The change concerns only the address representation. Retain the original
`input_codes[258,32]` buffer for action features; replacing that table globally
would silently change a second subsystem.

For symbol IDs `x=0..257`, let `b_r(x)` be bit `r=0..8`. Its fixed nonnegative
population has 18 rails:

```
u[r,0](x) = 1 - b_r(x)
u[r,1](x) = b_r(x)
```

Each occupied symbol has exactly nine active rails. Use a source-defined generic
bit fan-out function, with no new persistent codebook tensor. This is a
byte-to-population encoder, equivalent to fixed fan-out from a symbol input;
it has no key, path, word, task-role or correct-address knowledge. Defining these
fixed input rails by integer bits is not a host identity decision during retrieval.
The actual candidate computation must materialize the following neural currents
and thresholds, not replace them with a host `query_byte == stored_byte` selector.

Let `q[i,r,v]` and `k[s,j,r,v]` denote raw rails in query lag `i` and candidate
prefix lag `j`. All lags `i,j=0..31` and every occupied candidate `s` participate.
Using `H(t)=1` when `t>=0`, otherwise zero:

```
c[s,i,j,r,v] = H(q[i,r,v] + k[s,j,r,v] - 1.5)
m[s,i,j]     = H(sum_(r,v) c[s,i,j,r,v] - 8.5)
```

These are eighteen two-input coincidence cells and one match cell per lag pair.
Valid equal symbols give nine rail coincidences; unequal symbols give at most
eight. On these integer activities the comparisons are exactly `rail_sum>=2`
and `match_count>=9`. Store associations as Boolean rails; cast operands to
uint8 before adding, then materialize uint8 rail sums and Boolean AND activities.
Boolean tensor addition must not replace the numeric current sum, and unsigned
subtraction must not implement the negative threshold bias. The eighteen-cell
sum is at most nine on valid rails (at most eighteen before validation), so an
explicit uint8 count reduction does not overflow. A partially matched bit string contributes
zero at the completed match cell. The downstream learned neuron receives
completed match activity, not Hamming similarity. The circuit can be shared
through time across slots/lag pairs, but its actual evaluated cells, fixed
synapse applications, intermediate activity and clock schedule must be counted.

### Empty lags, byte zero and sequence boundaries

- Empty lag: all eighteen rails zero. Empty/empty yields zero coincidences and
  therefore no match; no externally supplied occupancy mask is necessary.
- Byte zero: nine active zero-bit rails. It differs from an empty lag and from
  every other byte.
- EOS is ID 256, with a distinct occupied pattern; reserved BOS is ID 257 and
  is also distinct. Neither is an ordinary byte. The codec output alphabet
  remains bytes plus EOS, without BOS output.
- Preserve the current protocol: append one query EOS; store each candidate's
  **preceding raw-byte** prefix; reset lag activity between documents. Do not
  insert a new BOS marker. Unfilled prefix lags are empty, not BOS.

Consequently, current candidate keys never contain EOS or BOS. The query's EOS
lag has no categorical match and its temporal-feature row is identically zero.
EOS still occupies a query lag and contributes to its population normalization;
an EOS sensory slot still has an ordinary preceding-prefix key and absorbing
cursor. This removes the old EOS code's graded correlations with raw bytes.
It is a real representation change, not an incidental optimization or a reason
to invent a matching key sentinel. Query truncation/caps and every candidate
boundary remain as currently declared. The stored first row of B therefore has
zero feature activity and zero local update under this query convention; its
retained values do not establish 1,024 independently active temporal synapses.

Rail validation must require either an entirely empty lag or exactly one active
rail in every bit pair. Invalid partial occupancy is a state error. Memory
disabled retains allocation/adjacency but stores zero address and sensory
activity, so all completed match features and address updates are zero.

## Normalization and the local teacher

Apply normalization **after** the raw threshold gates. Pre-normalizing rails
would invalidate the 1.5 and 8.5 thresholds. Let `Lq` and `Ls` count active
symbol lags. The selected kernel pools each lag's Boolean rails with `any`, then
sums the resulting 32 occupancy cells in float32, exactly representing counts
0..32. Summing all rails and dividing by nine is equivalent, but that alternative
would require a wide reduction: 288 active rails exceed uint8. This is
population activity, not a task annotation. Use

```
X_s = sqrt(32) * m_s / sqrt(Lq * Ls)   if Ls > 0
X_s = 0                               if Ls = 0
score_s = sum_(i,j) B[i,j] * X_s[i,j]
```

The zero-prefix candidate remains present with score zero. A denominator clamp
to at least one gives the same result for the all-zero feature bank. This is
exactly the inner product of categorical one-hot lag populations with query
squared energy 32 and key squared energy one, represented through compact
coincidence gates rather than 258 categorical channels. Since there are at most
`Lq*Ls` nonzero matches, `||X_s||F² <= 32`.

Prefer this occupied-count normalization over constant `m_s/sqrt(32)`. Both
are bounded and coincide when both lag populations have length 32, but constant
scaling reduces short-prefix activity by an additional factor and departs from
the existing population-normalization principle. The recommended scale is fixed
by that principle, not by saved score margins. It does not preserve actual old
feature norms or prove fast acquisition: old random-code lengths and overlaps
vary, whereas categorical symbol norms are fixed.

Both free and corrective calls build this same teacher-free `X` bank before
the sole trained address neuron. Free recall passes the actual scores into the
already selected fixed winner circuit. After the same instance executes its
action, corrective teaching explicitly reopens all addresses. With held codec
likelihoods, `p=softmax(scores)` and final forward prefix posterior `b`, retain

```
G[i,j] = sum_s X_s[i,j] * (p_s - b_s)
B_new  = B_old - 0.01 * G
```

Every synapse accumulates its own presynaptic match activity times its output
neuron's scalar phase error. No error returns through the match cells, bit
encoder, lag history or winner gates. The teacher receives payload bytes/EOS,
not a target address or extracted key. Keep held-weight float64 accumulation,
one float32 cast, finite proposal checks and explicit publication gates. The
shared codec and action transaction remain the same physical learner; there is
no separate trainer or distilled executor. All slow parameters remain outside
autograd; plasticity uses the existing explicit mode and learning-enabled gates.

For held codec and features, the conditional address loss has Hessian
`Cov_p(X)-Cov_b(X)` and largest eigenvalue at most 32. Thus the exact-real local
step satisfies `delta_loss <= -0.0084*||G||F²` at eta 0.01. This is the direct
single-output-layer conditional gradient, not reverse-Jacobian transport.
It is not a hard-policy gradient, a float32 monotonicity guarantee, or a theorem
about the simultaneous codec update, retained tasks or action success. Equal
features or ambiguous teacher likelihoods can still produce no useful credit.

## What exact equality does and does not represent

Current features sum 32-channel random-code inner products before the temporal
weights. Their symbol Gram matrix has rank at most 32, so it cannot equal an
identity kernel on all 257 input/output symbols. The candidate supplies that
identity kernel through fixed nonlinear gates. This is a disclosed categorical
prior, not a learned discovery, and the saved wrong ranks do not prove that
random overlap caused them.

The 32×32 matrix preserves where individual query/prefix matches occur. Learned
temporal weights can reward ordered alignments and penalize other matches. For
a known alignment, counting matching positions can distinguish an exact
multi-byte pattern from a mismatch. However, a linear weighted sum is not an
explicit conjunction over every possible sequence relation. It must still learn
which query positions and alignments matter; irrelevant path/operation/context
bytes also match. Different longer patterns may produce identical features or
equal scores, repeated records may be genuinely ambiguous, and information
outside the retained lag window is absent. No output learning rule can separate
two identical feature banks for different desired rankings.
The existing hard winner still commits once, uses the lower-index rule for
exact score ties, and advances one cursor. Duplicate evidence is not pooled and
later prefixes do not reopen free alternatives. Coincidence changes none of
those previously declared policy tradeoffs.

Equality features are invariant under a consistent relabeling of all symbols.
That is useful for matching unseen identities but removes absolute-symbol
meaning from this address frontend. It does not learn SET/KEEP semantics or
repair the independent operation/location failures. Keep those measured and
do not expand this proposal into an operation-head change. No repaired v2
output, increase in accuracy or elimination of interference is inferred here.

## Prior, implementation seam and controlled comparison

The existing random action code table, codec tensor shapes and temporal
`B[32,32]` shape can remain. Old normalized random-code tape keys cannot be
reinterpreted as raw dual rails: new keys need fresh ordinary ingestion under
a new typed state/schema. A matching B tensor shape is not functional parity.
Keeping its numerical values is allowed only as an **explicit transferred
parameter prior with changed input meaning**.

For the smallest interpretable comparison, recommend candidate and immediate
graded-competitive reference receive identical tensors from the prospectively
selected common reader prior (currently original v6), and identical fresh action
weights. This asks whether the changed frontend helps that continued learner;
it does not claim representation-neutral initialization. Resetting B in only
the candidate adds a loss-of-prior confound. A fresh-B study would require a
matched reset in both models and is not an extra selected arm here. Resuming a
newer v2 checkpoint instead is also a separate prior decision, not implicit
authorization in this note.

Keep the frozen graded reader as the exact reference. Prefer a separately typed
address/frontend and tape component with an explicit new reader schema, sharing
the verified codec, cursor and local-teacher routines where their interfaces
remain honest. A subclass that only overrides `_query` is insufficient: the
current tape fixes normalized `[128,32,32]` keys and validates 32-channel codes.
Do not override these semantics through a legacy loader, padded hidden encoding,
global policy mutation or a second teacher implementation without need. Actual
source design must expose the new ingest/state/checkpoint/accounting interfaces.

The selected implementation has three disjoint modules. The stateless encoder,
rail validator and actual threshold kernel live in `neuronal_byte_coincidence.py`.
`CoincidenceSequenceMemory` in `coincidence_sequence_memory.py` owns fresh Boolean
keys, validation, ordinary ingestion and accounting while sharing the existing
`TapeState` container and generic snapshot/cursor routines.
`CoincidenceLocalMemoryLearner` in `coincidence_local_memory_learner.py` subclasses
the competitive reader, replacing ingestion, query construction and metadata/
accounting. It inherits the exact local teacher and competitive free path.
Its query receipt has raw Boolean activity `[1,576]`, float32 features
`[N,32,32]`, actual float32 scores and float64 diagnostic log prior.

Construct the original parameters once through the base constructor, then replace
its stateless memory component. No new registered codebook or parameter is added;
the base constructor's dynamic parameter check therefore sees only its original
complete tensor contract. Strict inherited checkpoint loading validates the new
metadata and the unchanged fixed action-code table before publication. Prior
metadata adaptation belongs in a later explicitly authenticated experiment
adapter; old metadata or normalized tape tensors are not silently accepted.
New query work retains old counter names with retired dense products/reductions
and pre-gate query normalizations zero, and adds actual encoder, validation,
threshold, occupancy and post-gate scaling work. The inherited teacher/free
methods initialize and retain the subclass's full work dictionary.

The bounded candidate roster would keep learned, action-frozen, slow-frozen and
memory-disabled variants plus one freshly run immediate graded-competitive
reference. Preserve identical raw observations, actual execution-before-teacher,
local feedback and complete endpoint/panel accounting. Root must separately
freeze new withheld identities/layouts, the old-reading preservation criterion,
shared prior, initialization and cost admission before any run. Present failures
cannot choose the new cases or a gain/epoch/seed sweep.

## Resource implications and unresolved selection conditions

At capacity 128 there are `F=128*32*32=131072` lag pairs per query: `18F`
coincidence-cell and `F` match-cell evaluations, or **2,490,368 threshold
evaluations** and `54F=7,077,888` fixed input applications before learned scoring.
The 1,024 learned address synapses and their `N*1024` local products remain.
These are evaluated activities; time sharing changes physical population size
and clock, not the counted work. Whole-symbol encoder fan-out and lag shifting,
occupancy pools, divisive normalization and final winner work also belong in
the receipt/accounting contract.

Raw Boolean `[128,32,18]` keys occupy 73,728 bytes; unchanged sensory/adjacency
make one tape **206,344 bytes**. The stateless encoder adds no persistent tensor.
The reader remains 567,568 tensor bytes and the combined action learner 634,432.
For an illustrative four-candidate/one-graded-reference roster with 72 tapes per
arm, mixed tape storage is `4*72*206344 + 72*656904 = 106724160` bytes; five models
add 3,172,160, totaling **109,896,320 bytes**. This is not a selected roster,
checkpoint-read allowance, measured peak or runtime forecast.

The full uint8 rail-current bank and Boolean AND bank each occupy 2.25 MiB;
their explicit overlap is 4.5 MiB. The uint8 match-count and Boolean match banks
each occupy 131,072 bytes, and completed float32 `X` occupies 524,288 bytes.
An explicit uint8 cast of the AND bank also allocates 2.25 MiB, overlapping
the Boolean AND bank after the original rail-current bank has been released.
Operand casts use 73,728 key bytes and 576 query bytes. Occupancy uses Boolean
lag pools and exact float32 counts, and scaling has its own denominators and activity
temporaries. Existing teacher float64 feature cast and local-product banks each
occupy 1 MiB; the accumulator is 8 KiB, cast and proposal 4 KiB each. Query tokens,
encoded rails, validation scratch, lag shift copies, codec banks and returned
receipts also contribute. Source must release completed gate scratch before
teacher accumulation and declare actual overlap; no equality shortcut may erase
the simulated gate work. No millions-value gate journal is required by analogy:
the old full X bank was not serialized per lesson either. Authored gate truth,
source custody, scalar work and bounded scientific receipts have distinct roles.

## Six bounded ordinary QA nodes selected before execution

These are authored fixtures, not old endpoint replays or feature selection.
Root and the independent learning reviewer accepted the final test source;
the committed-source doctor receipt remains a launch prerequisite.

1. **Full symbol truth:** encode IDs 0..257 once. Include an empty population
   and pad 259 populations to nine 32-lag blocks. Nine calls to the actual kernel
   against nine key blocks exercise 82,944 lag pairs (67,081 distinct ordered
   symbol/empty pairs); independently assert exact equal/nonempty truth before
   normalization. No model, semantic query or action is constructed.
2. **Occupancy and collision arithmetic:** two valid kernel calls cover query
   `[A,B]`, key banks `[A]`, `[A,B]`, duplicate `[A,B]`, empty, then repeated
   `[A,A]` populations attaining energy 32. Assert scalar normalized values,
   duplicate equality and zero-prefix behavior. Two rejected calls cover a
   partial rail population and wrong dtype. No model or action.
3. **Raw tape publication:** two memory components ingest `(b'\x00A', b'', b'B')`
   once each, enabled and disabled. This is six raw bytes plus six EOS allocations
   across the two states. Check lag reset, no key EOS/BOS, exact binary endpoint
   copy, sensory/adjacency, snapshot identity and the 206,344-byte payload. One
   129-slot request must reject before allocation/encoding/publication; old
   float-key geometry must reject. No model, recall or action.
4. **Independent direct local delta:** one new reader ingests `b'A'`, receives
   query `b'A'` and teacher `b'A'` plus EOS once (two offers). Capture its actual
   query event during that call, without another query, and independently sum
   each actual float32 X feature times the returned output error. Verify the
   single cast/proposal, held codec calculation, work and unchanged tape. No free
   prediction or action.
5. **Controls and atomicity:** four new readers, one per existing reader mode,
   each ingest `b'A'` and receive one `b'A'` query/teacher (eight offers total).
   Reuse the learned instance for one injected late address-proposal validation
   failure (two more offers), proving neither parameter group nor tape publishes.
   Include snapshot restoration, metadata/fixed-code validation and outside-
   autograd checks without more queries. The injection is transaction evidence,
   not a natural overflow claim.
6. **Same-instance action then correction:** one combined learner containing one
   new reader ingests `b'ab'`. Authored zero address weights give the declared
   first-index winner; original identity codec emits `ab` then EOS from slot 0.
   Force only the operation head to STOP through authored parameters. One
   qualified fresh process verifies the unchanged fixture workspace, then the
   same instance receives its canonical raw SPLICE correction and `ab` plus EOS
   payload teacher. Use query `b'a'` for both calls, and preserve actual proposal,
   process receipt and before/after hashes. This proves mechanics, not acquired
   success; no target bytes enter the free proposal through a host edit.

The selected total is six reader constructions (one inside a combined learner
with one action circuit), two standalone memory components, eight successful
ingestions, thirteen raw ingested bytes and twelve EOS allocations. Seven teacher
calls present fifteen payload offers; eight total query traversals consume eight
raw query bytes plus eight EOS. There is one free prediction with three decisions
and one actual verification child. Pure kernel fixtures add eleven valid and two
rejected calls, separately counted from the eight reader query kernels. Any test
implementation change must reconcile this ledger before execution. No prior load,
old panel, benchmark task or fitting is included.

Root has selected this one source implementation. Its benefit to longer-pattern
binding and useful complete edits remains an empirical question; the new withheld
comparison, shared prior adapter, scientific schema and fresh complete cost
admission remain separate unexecuted work.
