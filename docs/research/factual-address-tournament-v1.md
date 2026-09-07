# Exact factual-address tournament — source implementation

Root selected the factual float32 path after V6's partial resource review. This
note records its implemented source contract. The source passed explicit Ruff
format/check; no kernel, model, import probe or QA execution has occurred here.
The float64 event router remains unchanged. All N occupied factual addresses,
1 <= N <= 2048, enter the same circuit, without a shortlist or target input.

## Narrow module and ABI

New [factual_address_tournament.py](../../src/humanbrain/factual_address_tournament.py):

`compete_factual(scores: Tensor) -> FactualAddressTournament`, no-grad, finite
detached CPU float32 vector `[N]`, N in 1..2048. Take an owned exact score clone.
No parameters, persistent cache, learned state or new checkpoint tensors.

Frozen receipt `FactualAddressTournament(scores, winner_index, winner_score,
levels, work)` retains the owned original float32 scores, final Python integer
identity, copied potential as Python float, and a tuple of frozen level records.
`TournamentLevel(input_count, pair_currents, left_wins, score_relays,
identity_relays)` retains actual float64 comparator currents, Boolean threshold
activity, and float64/int64 relay outputs. This exposes the actual small circuit
for ordinary QA without claiming all-pairs intermediate-current equivalence.
Existing caller seams need `winner_index` and `work`; other fields are explicit
new receipt semantics. Only factual callers switch after root integration review.

## Fixed computation and exactness

Validate dtype/device/rank/detachment and N before allocation. Clone original
scores, check finite input once using an explicit isfinite population and its
all reduction, then release that validation scratch. Widen to float64 and create
int64 leaf identities `arange(N)`. For each
level of width m, pair adjacent lanes `(0,1),(2,3),...`; P=floor(m/2), next width
ceil(m/2). Four strided score/identity slices are views, not copied banks.

Each pair has one signed-current threshold cell:
`d = left_score - right_score`, `g = (d >= 0)`. Float32 inputs widened to float64
make all differences finite, even opposite maximum float32 values. Every relay
potential is an unchanged source potential, so this bound is invariant at later
levels. There is no gain/temperature or arithmetic reconstruction of a winner.

One fixed two-input potential relay copies left if g, otherwise right; another
copies the corresponding identity. Tensor `where(g,left,right)` represents these
two binary-gated relays, not a free ranking operation. Logical wiring consists
of complementary controls and two input branches per relay. Count four gated
payload branches and their controls per pair even though the simulator performs
two fused tensor copy selections. Do not replace exact copying with
`g*left+(1-g)*right`, which can change signed-zero bits. Odd rightmost lanes pass
through one fixed potential and one identity relay. An odd level concatenates
the P computed relays and that last lane; an even level uses the P outputs
directly. Release temporary selected arrays after odd concatenation and initial
widened scores/identities after level one. Retain actual level outputs as above.

All indices in a left subtree precede all indices in its right subtree. Hence
g>=0 makes each subtree output its largest original potential with the lowest
original index on a tie. Induction over the fixed bracket gives exactly the old
all-pairs winner. Odd carry preserves the same invariant. Final index and score
are two scalar reads from the final relay population; there is no argmax, sort,
host winner search or extra full identity weighted sum. N=1 has no comparator
level and reads its single leaf directly. Original scores remain owned/unchanged.

This is a comparator plus analog-value/identity gate circuit, not a claim that
the old two-layer pure threshold construction has become linear. Its stage
semantics and additional relay primitives are explicit. Forward winners, held
scores, downstream outputs and learning behavior must match; intermediate
currents and work receipts intentionally describe different actual circuitry.

## Actual counters and named allocations

New work keys use `factual_tournament_` to distinguish old all-pairs counters:
calls, candidates, input_validation_values, finite_reductions,
validation_scalar_reads, score_clone_values,
score_widen_values, identity_initializations, comparator_levels, relay_levels,
comparators, signed_comparator_applications, comparator_subtractions, thresholds,
logical_control_complements, logical_gated_payload_branches,
logical_control_fanouts, potential_relay_outputs, identity_relay_outputs,
tensor_where_calls, tensor_where_outputs, odd_carries, concatenation_calls,
concatenation_copy_values, final_scalar_reads, named_allocated_bytes,
retained_tensor_bytes, peak_named_live_bytes.

For P_total=N-1 and L=ceil(log2(N)): comparator applications=2P_total,
subtractions=thresholds=P_total; comparator levels=relay levels=L. Logical
complements=P_total, gated payload branches=control fanouts=4P_total.
Potential/identity relay outputs each=P_total+C, where C counts odd carries.
Actual tensor-where outputs=2P_total, in 2L vector calls. Actual concatenations
are 2 per odd level, copying `2*ceil(m/2)` values at that level. Final scalar
reads=2; input validation has one scalar read, and finite Boolean reduction
combinations=N-1. Unused provisioned capacity adds no work.

Explicit named buffers (views/Python objects/allocator internals excluded):
owned float32 score clone4N, initial widened scores8N, initial identities8N,
finite activity N plus one Boolean reduction result. Each level allocates 8P
current bytes, P threshold bytes and 16P selected score/identity bytes. An odd
level additionally allocates16*ceil(m/2) concatenated output bytes. Total named
allocation is `21N+1+25(N-1)+16*O`, with O the sum of next widths at odd levels.
Retained receipt tensors are `4N+9(N-1)+16*S`, where S sums all next widths.

Peak named liveness is derived from actual level order: before a level, B is
4N plus retained previous level arrays. The current maximum is
`B + (16N if first level else 0) + 25P + (16*next_width if odd else 0)`.
Then replace B by `B+9P+16*next_width`; selected odd scratch and original initial
lane banks have been released. Also compare initial20N and finite-check5N+1.
This accounts for full receipt retention, not just an ideal two-buffer kernel.

| N | Comparisons | Odd carries | Levels | Cumulative named B | Retained B | Peak named B |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 22 | 4 | 20 |
| 3 | 2 | 1 | 2 | 146 | 78 | 117 |
| 864 | 863 | 2 | 10 | 40008 | 25063 | 28080 |
| 1760 | 1759 | 2 | 11 | 81448 | 51047 | 57200 |
| 2048 | 2047 | 0 | 11 | 94184 | 59367 | 66560 |

The 2048 row is not the worst allocation: odd carries add concatenation work.
Applying this integer recurrence to every allowed N=1..2048 gives maxima
**124086 cumulative named bytes at N=2017**, **59367 retained bytes at N=2048**
and **82899 peak named live bytes at N=2047**. This is finite source arithmetic,
not kernel execution. A looser analytic bound for N>=2 follows from
`O<=S<=N-1+ceil(log2(N))`: cumulative allocation is at most
`62N+16*ceil(log2(N))-40` bytes. No universal bound should use the power-of-two
point alone. Ordinary QA should include N=2017 and2047 when checking accounting.

These formulas assume the precise array order above; implementation must revise
them if it adds materialized complements, copies, concatenations or validation
buffers. They are not RSS estimates. No arbitrary 2K padding or masked dummy
candidate is needed. The V6 same-trajectory count is4151630 comparisons versus
3357755083 old factual pairs across2642 durable decisions; this is not a runtime
ratio. Keep event routing, address-score formation and learner updates unchanged.

## Ordinary QA home and release boundary

Use the existing `tests/test_byte_sequence_decoder.py`, one focused authored
mechanics node. Root selects its precise call ledger before execution. Compare
the new final index and copied source score against the existing `_compete`
network on widths1,2,3,127,128,129,511,512,513,1760,2048. Include lowest-index
ties crossing brackets/odd carries, all-equal values, signed zeros, positive and
negative finite float32 extremes and ordinary mixed values. For one small case,
independently inspect every saved comparator/relay. Assert original score
ownership, correct named work, strict dtype/shape/nonfinite rejection and no
model parameters. Downstream caller integration should reuse ordinary existing
same-instance response/teacher mechanics coverage, without a capability gate,
corpus fit, timing sweep or rerun of the failed V6 stream.
