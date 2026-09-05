# One frozen extractive-reader qualification

Preparation decision, 2026-09-05, within the 11:24–13:24 UTC window selected by
the [strategy review](../../docs/research/acquisition-strategy-review.md).
Freeze this protocol and its source before target outputs. This is one external
component check on exposed TRAIN, with no neuronal fitting or new exam access.

## Failure, hypothesis and decision

The [aligned reader](query_aligned_rate_cell_result.md) acquires 0/48 supported
executions. Before implementing episodic routing, test whether a declared frozen
pretrained reader can extract the required whole answers and abstain from the
actual language interface at acceptable cost. A weak reader cannot be repaired
merely by choosing its input source better.

Require at least 44/48 exact supported answers and 15/16 correct STOPs for both
actual and alternate wording when the evaluator supplies the known relevant
source. This is an explicitly oracle-source component ceiling, not permissible
learner routing or a neuronal success. Also report unrestricted full-corpus
reading with every source available and the identical decoder, including costs.
No answer normalization or substring credit substitutes for exact original bytes.
Passing only permits preparation of the reviewed persistent comparison; failure
closes this reader qualification without router fitting or automatic replacement.

## Fixed reader and inputs

Use the published `optimum/roberta-base-squad2` ONNX export at revision
`db07770d7fa024df076e60947b445b0b3290f8eb`, model SHA256
`204a84036a76ec7dafad3a4613f9425316b0308d77ad88326795fef6eb64b718`.
Its card identifies conversion of deepset version 2, under CC-BY-4.0. This does
not certify tensor equality to the current upstream model. Pin the tokenizer,
configs and card in a local manifest; use the existing ONNX Runtime/tokenizers
dependencies, CPU, one shared sequential session and one execution thread.
No silent download, alternate export, quantization or package substitution at run time.

Reuse `contextual_full_choice_train.json`, SHA256
`9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059`.
There are 16 documents, 64 actual queries and 64 alternate queries. Each wording
contains 48 supported and 16 unsupported items. Use all of them once in each
of two access conditions: known relevant source, then full TRAIN corpus.
Expected source IDs and answers stay in evaluator code; no answer-bearing source
selection or question rewriting enters the full-corpus reader. No growth or
fresh cohort, corrections, retries, model updates or official/protected content.
Unsupported known-source items receive their declared nonempty source document,
exactly like supported items; answerability must not select an empty input.

## Fixed decoding

Encode original query/context pairs with the pinned byte-level tokenizer. Cap
queries at 64 content tokens by rejecting overflow, never silently truncating.
Use a total paired length of 386, context-only windows with overlap 128, and
batch one. Retain context-local character offsets and map them to UTF-8 bytes.
Question, special and padding tokens cannot supply answer endpoints.
Encode each original query/context once with the tokenizer's postprocessor
temporarily detached; truncate only the context encoding at `386-query_tokens-4`
with overlap 128, then apply the pinned pair postprocessor exactly once to each
window. Verify original token-slice coverage. This avoids losing overflow tokens
or trimming byte-level offsets twice in paired postprocessing.

Enumerate every legal start/end pair within a window, start <= end and at most
30 context tokens; do not prune endpoints by top-k or impose a word-count cap.
The answer is the exact original source byte slice between those endpoints.
For each window, the null score is CLS start + end logits; a span's margin is
its start + end logits minus that same window's null score. Across all windows
and documents, use the maximum span margin. STOP when that margin is <= 0.
An exact maximum tie between different answer bytes also stops; identical-byte
ties use original document/window/endpoint order only for provenance.
Promote each finite model logit to float64 before arithmetic: calculate
`(start + end) - (CLS_start + CLS_end)` in that order before zero/tie comparisons.

This is disclosed maximum per-window null-margin aggregation, not the reference
postprocessor's cross-window minimum-null rule or calibrated confidence. It is
identical for both access conditions and must not be retuned from outcomes.
Keep duplicate occurrence custody. No span crosses a document/window boundary.
Empty evidence is a separately recorded STOP; malformed outputs, nonfinite
logits, signature errors, token overflow and resource failure are errors, not
correct abstentions.

## Qualification and reporting budget

Use at most 15 minutes for unscored synthetic loading/decoder/cost checks and
30 minutes for the one qualification, 45 minutes combined. Include loading
and checkpoint/file authentication in reported cold costs; separate steady
inference. The timer includes final source/model reauthentication and excludes
final JSON serialization/write. Enforce a 2 GiB peak parent working-set ceiling,
including loading.
Count every reader pair/window and token processed, wall time, model-file bytes,
retained source state and process memory. Record exact runtime and graph I/O
metadata. Download the six pinned files once into the ignored local model cache
during preparation and verify every actual byte hash before loading.

Before target outputs, use synthetic tests for paired context masking, Unicode
and punctuation byte slices, multiword/overlapping spans, null/ties, invalid
outputs and query overflow. Preserve one source-bound numerical/runtime receipt.
The final artifact records every answer/STOP/error, source/window/span custody,
exact marks by access/wording, resource observations and source/model identities.
Retain partial rows on failure and create result files without overwriting.
All 256 rows are mandatory, with fixed 48-supported/16-unsupported denominators
in each access/wording group. Incomplete or error/resource termination never
qualifies; do not rescale partial denominators or count errors as STOPs.

This component check does not execute Python actions, award learning credit,
adopt a hybrid, or pass any integration/reading/generalization gate. The later
comparison must separately bind source selection, extraction and actual motor
execution, with all teacher/proposal calls and matched controls counted.
