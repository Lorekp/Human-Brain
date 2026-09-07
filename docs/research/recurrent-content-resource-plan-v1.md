# Recurrent content: source resource ledger

2026-09-07. Prospective source arithmetic for the selected [public-observation
sequence controller](public-observation-sequence-context-v1.md). No model,
feature, checkpoint decoder, workload builder, action or timing probe was run
to produce this ledger. Source implementation and ordinary QA remain under
root review. This document selects neither a curriculum nor an experiment.

The implementation uses [immutable causal frames](../../src/humanbrain/recurrent_content_frame.py)
and the [local controller](../../src/humanbrain/recurrent_content_controller.py).
Existing V7 costs remain historical measurements, not a latency projection for
this changed stochastic policy.

## Addresses, visibility and frame storage

There are 3,485 public addresses and 2,048 factual addresses: 5,533 in total,
with invalidity separate. Existing 3,471 sensor coordinates remain intact;
returncode EOS occupies3471 and the twelve status bytes plus EOS occupy
3472–3484. The twelve byte fields retain their original capacities. A standalone
source uses an expiring public frame of at most3,484 raw bytes plus EOS.

Only already received public rows are visible, across every field. The factual
copy contains only the old admitted tape, so a privately staged source suffix
cannot become a target-predicting read. Factual successors retain their saved
forward/absorbing wiring. Public successors stop after the same field's EOS;
they never cross into another field. A register read visits its sole active
address and sensory connection; it does not allocate a5,533×259 address/sensory
matrix. All alternatives remain available to the learned gate populations.

The frame retains three immutable tuples of5,533 integer references (symbols,
successors and reveal ranks), plus field/token/address tuples, seal and causal
positions. These are Python objects, **not** a claimed packed integer tensor
allocation. Even just the three tuples' reference payloads are132,792 bytes
on a64-bit interpreter, before headers, integer objects and field copies.
The present constructor converts N occupied sensory rows to Python lists,
examines257 columns per row, and reads N successors. Its transient list and
float objects must be included in observed process RSS; no exact portable
Python-object byte total is claimed. The ledger charges these actual scans and
copies, rather than pretending the sparse retained wiring avoided construction.
The caller separately authenticates the original complete tape/model custody.

## Sparse tick arithmetic

The raw request has68,868 logical Boolean coordinates. At most769 can be
active:256 raw-byte presence,257 BOS/byte/EOS pair identities and256 positional
identities. Repeated identities collapse by Boolean OR. The final request-to-EOS
pair appears only after EOS arrival. During source observation Q is absent.

The normalized context/gate input has69,954 logical coordinates and at most779
active entries. Its emission input has1,072 logical coordinates and at most9
active entries. Source scanning has at most9 context/gate entries because Q and
the old output prior are absent. Values use actual float64 square/sum/sqrt/divide
normalization; sorted sparse index/value arrays preserve fixed coordinates.

For actual active counts Kx and Kz, one tick computes40Kx+4Kz direct current
products for context, gate and mode. LITERAL adds257Kz only when actually sampled.
Thus the upper bound is33,509 current products per public/output tick, or2,709
per source tick. This excludes eligibility operations, softmax/CDF work,
validation, hashing, relay reads and the old prior. All32/8/4 alternatives and,
when selected, all257 literal alternatives are evaluated; there is no shortlist.
The old prior and its factual tournament run once per actual free or offered
teacher output prefix, and never during source/public scanning.

Each sparse activity with K entries retains16K tensor bytes (int64 indices plus
float64 values). The frame currently allocates32K+16 named tensor bytes during
its construction, including raw ones, squares, energy and norm. The controller's
own sparse validation and gathered weight/eligibility scratch are additional.
Frame work uses the `content_frame_` prefix; controller work uses `controller_`.
Named allocation is cumulative work, not a measured live peak.

## Persistent and private controller state

| Population | Float64 coefficients | Bytes |
| --- | ---: | ---: |
| Context32×69,954 | 2,238,528 | 17,908,224 |
| Gate8×69,954 | 559,632 | 4,477,056 |
| Mode4×1,072 | 4,288 | 34,304 |
| Literal257×1,072 | 275,504 | 2,204,032 |
| Total | 3,077,952 | 24,623,616 |

Three dense eligibility banks require73,870,848 bytes per trajectory. Weights,
three banks and one complete proposal total123,118,080 bytes, before the old
actor, baselines, RNG, receipts and scratch. This is the design's approximately
123 MB figure; it is **not** the actual correction peak.

The current `fork_stage` preserves the pending actual free stage and clones all
three banks for atomic resolution. While both remain alive, weights, six banks
and a proposal total196,988,928 bytes per active route. Dense finish arithmetic
has additional matrix-sized temporaries, with the context matrix alone
17,908,224 bytes. Disabled learning still constructs its prescribed proposals
and must not be charged as zero work. Controller baselines add48 bytes; the
actual CPU-generator state size is to be reported by runtime accounting rather
than guessed. Snapshot/load validation and RNG custody remain separate work.

The two previous neural routes' named initial state was190,922,008 bytes,
including old derived representations. Adding only the two new weight banks
gives240,169,240 bytes before new baselines/RNG and any pending trajectories.
This is a planning addition to the historical shape inventory, not evidence
that a future initialization has loaded or preserved those tensors. Actual
continuation must authenticate and report its own safe HEAD and shapes.

Sparse ticks do not remove whole-state work. Begin/finish/publication still
validate/hash learned state; three eligibility matrices per population are
allocated and cleared, forks copy them, and finish scans them and forms dense
local proposals. These costs need separate stage intervals and work counters
in the future complete run. No throughput is inferred from product counts.

## Persistence and complete-run decisions

At an idle safe boundary, one route's numerical checkpoint gains24,623,616
weight bytes plus baselines/RNG and serialization metadata. Pending eligibility
is not silently treated as idle checkpoint state; any in-flight continuation
would require its own explicitly implemented custody contract. The existing
safe-boundary publication mechanism remains the integration baseline.

If V fresh route payload versions are retained, the weight addition alone costs
V×24,623,616 disk bytes. For illustration,100 versions add2,462,361,600 bytes
(decimal2.46 GB), even when old payload reads are avoided by leases. This is
arithmetic, not a selected checkpoint schedule. Temporary serialization,
authenticated handoff hashes, retained versions, journal records, source
inventories, fresh restore and independent saved review all remain real work.

The [owned-payload change](persistent-checkpoint-owned-payload-qa-v1.md) removes
eligible repeated reads while retaining operating-system-enforced immutability
and explicit fresh checks. It does not remove writes, fsync, initial hashes,
live-state scans or lease metadata checks. Every owned store must close handles
in `finally`; unsupported platforms retain repeated full hashing. Ordinary QA
proved that mechanism, not a broader workload time reduction.

A broader release must separately declare transient observed symbols versus
retained factual admissions: the historical1,614/2,048 factual occupancy leaves
434 slots, which cannot retain10^4–10^5 new symbols. It must price the chosen
source lengths, public scans including all files/status, actual free/teacher
prefixes, action heads/children, stage forks, capture/commit and complete saved
audit. No source prefix truncation, silent eviction, answer-selected memory,
isolated speed gate or empirical admission is authorized by this ledger.
