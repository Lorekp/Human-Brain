# Generic byte events and fixed neuronal encoding

Status: selected implementation contract, 2026-09-06. Follows the
[binding review](local-token-event-binding-review.md) and the closed
[multiscale comparison](../../experiments/r4_integrated/local_memory_multiscale_v1_result.md).
No eventized acquisition or endpoint observation has run. This is one structural
input change, not a tokenization, encoder or learning-rate sweep.

## Failure and hypothesis

The changed writer dynamics gain one fresh case, also gained by a frozen writer,
but neither assignment pair is jointly correct and all interference fails.
Earlier scalar localization found attenuated identifier contrast before values.
Test a generic separation between within-event byte composition and across-event
association. Fewer writer updates through common text may help binding and reduce
expensive eligibility work. Neither outcome follows from segmentation alone.

The encoder is a fixed representation prior. Writer/query and output synapses
retain their current local online learning. Do not claim the new encoder learned
lexical meaning, or that its activity is lossless because the host spans are.

## One exact segmentation policy

Use maximal runs of ASCII letters, digits and underscore; maximal runs of ASCII
whitespace bytes `09,0a,0b,0c,0d,20`; and singleton events for every other byte.
Split long runs from their beginning into at most **16-byte** events. This fixed
cap bounds buffering and within-event recurrence; it is a generic engineering
choice, not a measured optimum. Do not adapt it to document content or targets.

No decoding, Unicode normalization, case folding, trimming, whitespace collapse,
learned vocabulary or task-specific merge is allowed. Concatenating event payloads
must recover every original byte exactly. Non-ASCII and malformed UTF-8 bytes are
preserved as singletons; this reduces compression for non-ASCII text and is not
a multilingual language claim. Long words/identifiers split without truncation.

The streaming segmenter holds only an unfinished bounded run, emits on a generic
class transition or the cap, and flushes at the string boundary. Its event payloads
and raw half-open offsets must be identical for every input chunking. Finishing
an empty string emits no raw event. The model separately appends its existing
document/query EOS symbol 256 once. No per-token EOS or new EOT feature is added.
No role is attached to an identifier, value, query key, selected answer or address.

Existing string/vocabulary tokenizers do not supply this bounded arbitrary-byte
stream contract. Reuse ordinary Python byte handling and add one small utility;
do not add a package, corpus-trained vocabulary or lexical embedding cache.

## Fixed neuronal event content

Use ordinary `StreamingRateCell(32, 32, seed=seed+4)` with scalar update fraction
0.5. Its private Xavier input weights, diagonal `0.9 I` recurrence and zero bias
are fixed; disable gradients and never update them. There are 32 additional
neurons, 2,048 weights and 32 biases: 8,320 float32 bytes, plus 128 bytes of live
activity per serial stream. No encoder eligibility or teaching feedback is
constructed. The separate seed preserves all existing initialization draws.

For each event, reset encoder activity to zero and process its ordered bytes
using the learner's existing fixed `[258,32]` input-code table. Emit only the final
32-dimensional neuronal activity as one writer/query input. Encode the final
symbol 256 as its own one-step event through the same cell. Apply this identically
to documents and questions. Do not add answer-position slots, whole-token hashes,
mean pooling or a hidden text parser. Keep the initial implementation serial and
simple; batching is a later measured optimization requiring numerical review.

Fixed recurrent composition is neither injective nor guaranteed linearly
decodable. Longer events may attenuate early content; even two different neuron
codes do not establish useful unseen-byte composition. The earlier supplied-factor
LMS certificate does not transfer to these codes. Report these limitations instead
of selecting a code against the four exposed novel answers.

Writer activity carries across events and document terminators exactly as before,
then retires after the entire document list. Query activity starts fresh for each
traversal. Purge encoder state and all raw segmentation buffers before ingestion
returns. Snapshots retain fixed encoder tensors/configuration and the existing
memory/credit, never event strings, offsets, dictionaries or memoized encodings.

## Layered implementation and qualification

First implement the pure byte segmenter and review its ordinary existing-test
coverage. Then run one deterministic **60-second, 2,048-MiB preprocessing
qualification** over the frozen TRAIN schedule's document and query strings,
plus authored byte-preservation fixtures. No neuronal models, teacher updates,
novel endpoint questions, action children or protected inputs enter this check.
The frozen 1,024-lesson schedule remains the only training text source; target
values are not supplied to the segmenter as labels or merge guidance.

Check full reconstruction, offsets, every two-part split of short fixtures,
one-byte chunking, empty strings, all 256 byte values, invalid UTF-8, mixed
whitespace/punctuation, and run lengths 15/16/17/31/32/33. Record actual invocations
and bytes, raw and EOS event counts, max buffering and deterministic source/input
hashes. Include document/query event-count distributions. Event boundaries must
depend only on current bytes, declared classes and the cap. All counts describe
host processing; there is no neural simulation speed claim.

Successful preservation and actual fewer document-writer events than byte+EOS
steps earn integration of the fixed encoder into the existing learner. This is
an engineering readiness condition, not a capability threshold. A preservation
failure or no event reduction closes this candidate for review; do not extend
the budget or try another cap. The ordinary encoder/integration QA must then
verify finite, read-only fixed dynamics, same document/query event identity,
fresh shared initialization, source-free transport and exact old-reference
trajectories before fresh complete-workload cost admission. No output-only fit
or new endpoint diagnostic is queued.

The existing byte tracer indexes `_encoded()` by raw byte position. It must not
silently consume an event sequence. Any new event tracer must use the same
production events and report raw spans separately; it may not rerun the encoder
to reconstruct observations. Closed byte diagnostics remain tied to their frozen
source. No traced endpoint execution is authorized by this interface contract.

## Integrated comparison boundary

Keep the five candidate controls on this same fixed event encoder and the
immediate a1ab5ee multiscale byte-stream learner as the sixth reference, starting
fresh. Preserve its explicit reference source and dependency parity; no checkpoint
migration or additional scalar arm. Only the genuine encoder tensors are new
candidate state. Writer rates, local credit, output phases, teacher rule, seed,
roster and fixed fit budget remain unchanged for the first return.

Count every raw byte plus EOS processed by the encoder, every emitted controller
event, every memory write/credit opportunity, both query traversals per lesson,
all teachers and all final executed actions. Input clock changes from bytes to
events; do not rescale leaks by span length. Total simulation, encoding, source
removal, retained state, serialization and execution require a new conservative
cost admission. Retain the 2x margins, 30-second allowance and 900-second ceiling.
No count reduction alone authorizes a fit or implies reduced total runtime.

The return still requires complete source-assignment swaps, new-value success,
retention and interference, with an advantage over the candidate frozen writer
and immediate reference. The host motor remains declared scaffolding. Fixed
eventization does not establish learned neuronal action selection or language
understanding, and no protected gate advances on preprocessing QA.
