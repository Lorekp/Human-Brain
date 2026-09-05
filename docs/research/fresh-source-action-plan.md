# Fresh questions in one cumulative-source action session

Status: active preparation plan, 2026-09-05; dataset selection, encoding and the
session have not run. Freeze the completed implementation and inputs before the
session. Follow the [roadmap](roadmap-reading-to-coding.md).
The [fixed-rehearsal qualification](../../experiments/r4_integrated/fixed_rehearsal_result.md)
earns this integration attempt, not architecture adoption. Keep that experiment
and its preparation/transport failures unchanged.

## Question and boundary

Can the qualified learner use new passages and original questions, without an
evaluator selecting the relevant paragraph, learn from ordinary post-attempt
feedback, and retain its sixteen acquired tasks? Compare fresh executable
behavior with matched frozen parents and a conventional learner.

This is **passage-visible public development**. Raw documents, word vectors and
candidate bytes remain retained. The external labeled rehearsal store does not
establish source-absent knowledge, autonomous consolidation or recovery. Teacher
repairs are supervision, not learner successes. No provenance revision or new
reasoning/action schema is introduced.

## Fixed parents and policies

Authenticate `benchmarks/fixed_rehearsal_v1.json`, its completion and launch
receipts, checkpoint archive and source revision beginning `2eb0fb1`. Before
freezing this successor, record their complete hashes in its input manifest.
Restore the final `cell_rehearsal` and `conventional_rehearsal` states, including
all Adam moments, per-parameter counters, offer/update counts and configuration.
Require exact export/checkpoint equality. Do not import an intermediate best
checkpoint or reset optimizer state. Retain the sixteen original paragraphs and
their original questions from `real_input_qualification_cases_v1.json`.

Six policies share identical evidence, candidate order, explicit action tasks,
encoder and two original motor demonstrations:

| Policy | Parent and new learning |
| --- | --- |
| `cell_update` | Final rehearsal cells; prescribed offers only |
| `cell_frozen` | Independent identical copy; no offers or state mutation |
| `conventional_update` | Final conventional rehearsal; same offers |
| `conventional_frozen` | Independent identical copy; no offers or mutation |
| `frozen_producer_update` | Final frozen-producer rehearsal; same offers, existing masks |
| `external` | Existing frozen-vector cosine over the identical entire inventory |

The frozen-producer parent has a different historical learning trajectory and
initial competence. It is a useful trained-head reference, not a matched causal
ablation of the final cell producer. The independently trained conventional
comparison also does not isolate a single equation. Keep those limits explicit.
No core, initialization, encoder, window, optimizer, clipping or STOP changes:
existing rate .001, clip 1 and all current optimizer masks remain fixed.

## Outcome-independent public data selection

Use the already pinned official SQuAD 2.0 **TRAIN** source and
`public_qa_selection_v1.json` article-disjoint `probe` partition. This is not a
download of official DEV/test. The pool was exposed in earlier project work;
"fresh" means absent from these restored parents' sixteen-question experience,
not unseen by the project, independent adjudication or contamination-free data.
Preserve original source/question bytes, every accepted annotation and attribution.

Select twelve questions from twelve distinct paragraphs and articles, none sharing
an article or context hash with the original sixteen. Use the existing eligibility
rules and ascending SHA256(original QA ID), then ID, within each label bin.
Walk the following slot labels in order, choosing the first eligible unused
article/paragraph in that bin; no score, model output or lexical preference enters:

| Slots | Labels and role |
| --- | --- |
| E0–E7 | supported, supported, supported, STOP, supported, supported, supported, STOP; learning episodes |
| L0–L3 | supported, STOP, supported, STOP; wholly untaught later tasks |

Thus twelve fresh tasks comprise eight supported and four STOP questions.
Eight episodes receive learning; four late tasks never do. Freeze their exact
IDs, article/context identities and order before any prediction or update.
Do not rewrite questions, choose easier replacement items, or select by a fitted
reader's confidence. If fixed selection cannot satisfy the contract, stop preparation.

### Cumulative-label review is required before freeze

Original SQuAD labels are paragraph-local. An independent annotation review must
inspect the selected questions against **all 28 retained paragraphs**, including
the sixteen originals. Check both old and new supported/unsupported questions.
An unsupported question must remain unsupported at every scheduled source state;
checking the final superset is conservative for the append-only schedule.
Supported questions must retain their original accepted answer semantics without
a contradictory answer introduced for the same referent and circumstances.

Record source-referenced review reasons and reviewer agreement, separately from
learner inputs. A matching answer string alone does not establish semantic support
or absence. If another source answers an originally unsupported question, a
reference becomes ambiguous, the source set contradicts an original answer, or
reviewers cannot resolve validity,
**abort this proposed cohort**. Do not relabel the official item, drop the awkward
question, or redraw replacements under this protocol. Any revision requires a
separate documented preparation decision before outputs.

## One exact session

The same cumulative inventory is used for every policy and primary prediction.
Its opaque document IDs/provenance are custody metadata, never a query-dependent
filter. All candidates from all documents remain eligible under the unchanged
48-word window rules. Run query inference and replay under the current inventory;
encode each new query/document once with the pinned unchanged encoder and reuse
the authenticated old representations. Count all encoding and retained arrays.

1. **Initial state:** pool all sixteen original paragraphs in original fixture
   order. Execute the sixteen original questions once per policy, before new
   feedback. These are old-task baseline probes, not fresh successes.
2. **Episodes E0–E7:** append that episode's complete paragraph once, then obtain
   and execute one first prediction per policy on its unchanged question.
   All six first attempts precede feedback for that episode. No earlier probe
   or offer may have included this question or its source.
3. **Feedback:** each updating arm receives exactly one supervised offer for Ei,
   followed by original question O(2i) and O(2i+1), where O0–O15 is the original
   fixture order. No other offers occur. Replay uses the current cumulative
   inventory and original accepted annotations from the external labeled store;
   it is not a new failed execution and is not re-executed. No unrecorded prior
   teacher execution is implied by an annotation.
4. **Later transfer:** after the last episode's offers, append L0–L3's paragraphs
   in that order before querying any of them. Execute L0–L3 once per policy,
   with no feedback, corrections, repeated probes or subsequent learning.
   None of these four questions/labels enters any earlier offer or model probe.
5. **Final retention:** at this final 28-paragraph inventory, execute O0–O15 and
   E0–E7 once per policy. The eight repeated episode questions measure acquired
   retention after supervision; they are never counted as fresh transfer.

This is **24 new offers per updating arm** (8 fresh +16 old), below the agreed
36-offer maximum; three updating arms receive 72 offers total. The unused allowance
is not available for padding, extension or retries. Frozen controls see the same
source growth and action opportunities while all learned state remains unchanged.

Every episode first attempt is genuine. Successful actions may provide the
declared positive receipt. When correction is required, execute one exact teacher
repair after the failed attempt; a STOP teacher restores the original workspace
and verifies it under the reviewed absent-answer annotation. Preserve the failed
workspace receipt before correction. Execution alone cannot establish
source answerability. Labels remain explicit supervision, never fabricated FAIL
labels for unexecuted competitors. Teacher actions do not count toward accuracy.

## Actuation, scoring and diagnostic costs

Reuse `utf8_action_bridge.py` and the unchanged two-demo motor. Number the 28
distinct tasks O0–O15, E0–E7, L0–L3 consistently; declare ordinal service/module
names and `Configure ...` instructions before outputs. Repeated probes reuse
the same instruction in separate pristine workspaces. Reading questions stay
unchanged and separate. Predictions alone select answers from the entire pool.
Preserve raw UTF-8 stdout/stderr, strict decoding, exact LF, literal roundtrips,
decoys, source/query/occurrence custody and unchanged core/motor checkpoints.

Preserve all official accepted-answer alternatives. Before freeze, bind an
evaluator-only multi-answer scoring seam: success requires an actual selected
accepted span, lossless execution of those same bytes and no collateral change.
Do not choose the payload from the accepted set. The old single-expected-answer
bridge must not silently reject another officially valid annotation or turn
target selection into actuation. Unsupported success requires learner STOP and
unchanged execution. Bind supervised gold indices to the original source's
accepted annotated byte intervals; matching bytes in another paragraph are not
automatically valid supervision. A supported success must use one of those
authenticated annotated occurrences. Audit alternative-answer marginalization
in the unchanged loss and document this stricter occurrence-based scoring.

At each scheduled primary prediction, separately compute an **oracle-paragraph
diagnostic** with the same checkpoint and the question's original paragraph only.
It is read-only, receives no feedback, performs no action and never supplies the
primary answer. It uses evaluator source identity deliberately and is reported
only as a source-known reference, not autonomous retrieval or learner success.

Per policy: 16 initial +8 episode +4 late +24 final =52 primary predictions/actions,
plus 52 oracle predictions. Across six policies: **312 primary actions and 312
oracle predictions**, plus two shared executed motor demos. At most 24 teacher
executions add a maximum of **338 child executions**; count actual corrections.
Also count offer forward/backward work, ingestion/encoding, all retained raw and
derived evidence, labeled replay records, optimizer state, artifacts and diagnostics.

## Decision and budget

Freeze exact implementation/input/parent identities, scoring, counts and resource
accounting after at most **two hours' preparation**, starting when input and
implementation preparation begins; record that UTC window before starting it.
One **20-minute total run**
includes setup, encoding, restoration, diagnostics, all actions and artifact writes;
parent peak RSS must remain at most **2 GiB**. Preserve partial evidence on any
failure. No automatic rerun, rate/threshold sweep, additional epochs or new model.

Proposed provisional engineering success requires cell_update to execute at least
6/8 supported fresh first attempts and 3/4 fresh STOPs across E+L; within the wholly
untaught L subset require both supported answers and both STOPs, with at least one
supported L gain over cell_frozen. Require all sixteen O tasks still correct at
the final pooled state and all decoys unchanged. Report the conventional paired
results and old-task initial pooling losses separately; do not mask these behind
aggregate accuracy. If frozen cells already solve both supported L tasks, report
zero available gain and an inconclusive incremental-transfer comparison rather
than selecting replacement tasks. These tiny counts are feasibility criteria,
not a confidence interval or evidence of broad generalization.

Stop after the session whether it passes or fails. If paragraph-oracle answers
succeed while pooled actions fail, source competition is an observed gap; if both
fail, transfer/comprehension remains unresolved. Neither proves model impossibility.
A pass supports this narrow persistent-source action behavior only. Source-absent
knowledge still needs a separately declared learned write/read/output interface;
deleting external files while retaining complete raw/span inventories does not
satisfy it. Official Amelia scoring remains pending genuine structured production
under the [reading plan](reading-comprehension-evaluation.md) and
[candidate contract](official-reading-next-candidate.md); no scalar-to-option,
table or ordering solver is authorized here.
