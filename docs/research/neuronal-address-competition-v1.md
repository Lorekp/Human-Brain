# Selected initial-address competition v1

Implementation selection, 2026-09-06. Root accepts the
[learning contract](neuronal-address-competition-learning-v1.md) and
[primary-source review](neuronal-address-competition-sources-v1.md): an explicit
fixed two-layer threshold circuit commits free recall to one address. Teaching
reopens all candidates in the same persistent reader after actual execution.
Its existing direct local `X*(p-b_final)` and codec updates remain numerically
unchanged and are declared supervised surrogates, not derivatives of the hard
free decision. No backpropagation, separately trained executor or host-selected
correct address is introduced.

## Selected implementation

Add a small `neuronal_address_competition.py` module. For actual finite float32
scores and every pair i<j, evaluate float64 current `a_i-a_j` and threshold
`c_ij = 1[current >= 0]`. Winner currents are
`i + sum(j>i)c_ij - sum(j<i)c_ji - (N-1)`, followed by the same nonnegative
threshold. Fixed wiring chooses the lower index on exact ties. Validate the
one-hot winner population and obtain its sparse cursor by the fixed identity
readout `sum(i*w_i)`. Do not replace these cells with sort or argmax.
Retain actual input scores, comparator currents/activities and winner currents/
activities in a receipt, with explicit operation and tensor-byte accounting.
At 128 candidates the circuit has 8,256 threshold neurons and 32,512 signed
synaptic applications, excluding score production, biases and cursor readout.

Add `CompetitiveLocalMemoryLearner` as a small subclass of the held
`LocalMemoryLearner`; supply it through the existing splice reader-injection
interface. Inherit `_query`, `teach`, storage and atomic publication bodies.
Version metadata must identify committed free recall and the reopened
all-address corrective surrogate. No trained checkpoint is loaded during
implementation or ordinary QA.

Free prediction calls the inherited `_query` once, then the declared competitor
once. Preserve its full soft log prior as a named diagnostic only. The actual
free reader has one cursor and one normalized branch, so its step prior and
posterior are genuinely length one. A prediction subclass carries competition
and diagnostic fields separately. It must not pretend that suppressed branches
were traversed or observed. The same free policy applies to reader-only calls.

Use the existing full occupied codec bank once per call with held weights, then
gather only the actual cursor at every decision, including the first. This keeps
the existing checked codec boundary and avoids a whole-parameter scan at each
byte. Count full initial codec work honestly; do not claim visited-only codec
evaluation. Advance one cursor using the existing forward/EOS-absorbing adjacency.
Actual codec output alone determines EOS or the declared byte limit. No later
readdressing, repair, parsing, escaping or added stopping rule occurs.

The same reader's inherited teacher phase uses its full population and actual
soft evidence. There is no redundant competition pass during teaching. STOP
correction still teaches only the existing operation head. The actual free
commitment remains in the pending splice receipt through execution; ordinary
QA must show that inherited correction can change address weights after a wrong
commitment and that numerical teacher updates match the held implementation.

## Verification and integration boundary

Implement and source-review first. Root will fix a small exact selection in the
existing decoder QA file and its exposure before executing it. Cover scalar
competition parity, ties/finite input guards, actual single-cursor byte/EOS/limit
behavior, reopened teacher parity and same-instance action-before-correction.
Use the selected checkout, committed source, doctor and qualified executor.
Do not run old backpropagating tests or exposed v1 benchmark cases.

Root accepts the [six-node authored QA contract](neuronal-address-competition-qa-v1.md).
Run its first five exact function nodes together, then review their saved receipts
before the sixth, single real-child node. Each launch is bounded by the existing
owned-process watchdog at 60 seconds and 2,048 MiB. Use distinct retained
`.pytest-tmp/neuronal-address-competition-qa-v1-components` and
`.pytest-tmp/neuronal-address-competition-qa-v1-action` basetemps, log prefixes
and JUnit files; never reuse a basetemp. A failure is preserved before choosing
any correction or changed exposure. These are ordinary component tests, not an
acquisition trial. No separate user approval is required for this authorized QA.

Source review reconciles the competitor's named allocated tensor ledger at
N=128 to 404,048 bytes, including the owned score clone and winner-count scalar;
the retained trace is 74,816 tensor bytes. This is not allocator or peak-memory
measurement. The implemented codec still evaluates every occupied row once.

The existing compact serializer, source-free worker and independent auditor do
not understand this new free receipt yet. Ordinary component QA cannot admit an
integrated run. Their successor schemas must distinguish one-cursor free activity
from all-address teaching, bind the actual competition trace, and receive fresh
complete-workload cost admission. Any successor must preregister new withheld
cases and matched controls; v1 endpoints remain exposed development evidence.
No acquisition, checkpoint probe or alternate decoding of saved v1 outputs is
selected by this document.

## Completed component return

The [six-node component result](neuronal-address-competition-qa-v1-result.md)
passes at cbf9403, including a genuine wrong action before the same reader's
local correction and exact inherited-teacher parity. No acquisition or old
checkpoint ran. Return this mechanism to the shared integrated workload.
Root selects successor receipt serialization and saved independent competition
trace validation under the [integration contract](neuronal-address-competition-integration-v1.md)
as the next implementation step. Separate writer and saved scalar checker
implementations share the strict schema, with expected policy supplied by frozen
arm metadata. The two-mebibyte size guard remains prospective until authored
maximum-row QA. Preserve the completed v1
experiment and reuse suitable pure helpers. The subsequent source-free worker,
new withheld comparison and full measured cost are still separate prerequisites;
passing component QA does not admit a fit.
