# Counterbalanced pairwise evidence learning

Repeatable public development. Declared before candidate outputs on these cases.
The [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns the two-hour budget and return to the integrated workload. Prior results
at commit `6b1f37f` and all protected protocols remain unchanged.

## Failure and hypothesis

The previous absolute GO/NO_GO rule learned a globally preferred old value:
both rates scored 0/5 fresh tasks even with conventional score selection.
An individually acceptable teacher candidate could still lose the global race.
Hypothesis: learning the eligibility difference between a failed answer and its
executed correction better trains query-dependent ranking. This does not assume
the current fixed features have sufficient capacity.

Keep the frozen MiniLM encoder, generic 1–3-word spans in 48-word windows,
feature blocks, shared two-row native synapses, GLIF dynamics, and learned
occurrence motor. Two executed `axel`/`bora` demonstrations establish the same
motor prior. All source bytes and vectors stay in disclosed external storage;
source-absent means absent from live input, not semantic compression.

## Curriculum and feedback

[`contextual_pairwise_cases.json`](contextual_pairwise_cases.json) fixes sixteen
teaching requests over eight documents. Each entity has two properties; a value
is correct for one query and incorrect for the other. First queries alternate
transport/audit by entity; the complementary queries follow in reverse entity
order. All answers are evaluator-only. Ingestion deduplicates by document ID
after verifying identical bytes; inference never selects documents by that ID.

For each teaching request, record a neuronal first attempt and execute its
selected edit. If evidence selection stops, allow one explicitly labeled
external-cosine exploration and execute it. Then execute the teacher correction.
Only an actually executed wrong candidate followed by a successful teacher
correction creates a learning pair. A correct actual candidate needs no pair;
if there is no executed wrong candidate, both methods skip the update. Do not
invent negative candidates or label unexecuted spans from an answer key.

Compare at learning rate **1**, with no extra epochs or tuning:

- **ABSOLUTE:** one native negative update on the failed candidate, followed by
  one positive update on the freshly rebound teacher candidate.
- **PAIRWISE:** one native update on accepted minus rejected eligibility,
  enforcing a same-query score margin of 1. Shared bias is excluded from the
  margin and remains unchanged; temporary native competition bias is discarded.
- Each updater has a **FROZEN** control with identical ingestion and replayed
  executed feedback. Frozen first attempts are reported separately from the
  replay exposure. No control success is credited from the updater's action.

The two updating policies can collect different executed failures as learning
diverges; report those ledgers and update counts. Their on-policy comparison
does not isolate the objective under an identical sequence of rejected spans.
Each paired frozen comparison does isolate updating under its own exposure.
Teachers never count as first-attempt success or autonomous recovery.

## Evaluation and decision

After teaching, ingest all six new documents and remove them from live input.
Evaluate eight supported requests (four entities, both properties) and two
unsupported requests, without feedback, tuning or retries. Replay all sixteen
teaching requests afterward. For each objective report updating neuronal,
frozen neuronal, conventional learned-score and external-only evidence selection
through the same motor. Preserve incidental query overlap and multiword outputs;
execute wrong or malformed edits and report their failures and unchanged decoys.

Keep the original KS2 Q6a/b as an untuned diagnostic with official evaluator-only
marking. Noncanonical emitted answers retain the existing pending human-scoring
status. These synthetic programming fixtures do not replace public-exam evidence.

Report first-attempt executed accuracy, evidence accuracy, both-property retention,
unsupported STOPs, concentration on one answer, score-versus-spike disagreement,
feedback/update counts, retained bytes, timing and process memory. Success requires
fresh query-dependent gain over the frozen control and retained earlier skills;
report whether pairwise improves on absolute feedback separately. A constant
answer, a collision-induced STOP, or a training-only improvement does not qualify.

Budget: one pairwise head, one learning rate, sixteen teaching requests per
updater, eight supported transfer requests, two unsupported probes, and two hours
for implementation/diagnosis. Do not change fixture answers, query forms, features
or dynamics after the first run. If ranking remains wrong, return to feature and
learning design; fix spike resolution only once useful scores justify it.

Use the pinned encoder setup in the [previous protocol](contextual_evidence_protocol.md)
and explicit CPU/checkout-cache environment in [development.md](../../docs/development.md).
Run the new runner with a fresh output path and preserve terminal results:

```powershell
$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'
$env:HUMANBRAIN_DEVICE = 'cpu'
./.venv/Scripts/python.exe -m experiments.r4_integrated.contextual_pairwise_python_session --output benchmarks/contextual_pairwise_python_development_v1.json
```
