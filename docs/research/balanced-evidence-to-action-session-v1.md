# Balanced Evidence-to-Action Session V1

Status: public-TRAIN design, not frozen and not spent.

## Purpose

Make language, knowledge, memory, and software apply pressure to one persistent
learner. Coding is an exact-feedback laboratory inside this session, not the
learner's ontology. A result cannot pass by compensating for weak language with
strong edits, or vice versa.

The session uses fictional scientific records and small Python repositories.
Every coding transfer depends on relations acquired from prose, while later
language tasks require delayed retention, revision, provenance, conflict, and
one-to-three-hop composition.

## Public surface

Every episode presents the same modality-complete observation:

```python
respond(
    request: bytes,
    documents: tuple[OpaqueDocument, ...],
    files: tuple[OpaqueWorkspaceFile, ...],
    tool_events: tuple[bytes, ...] = (),
    ablation=NONE,
) -> PhysicalActionDecision
```

No task kind, relation label, answer span, target path, candidate action,
replacement value, desired output, or parsed structure is exposed. Documents
and files are present in every episode so modality presence cannot route the
answer.

One shared fitted-cell arbiter may select textual bytes, a file replacement, a
tool invocation, or tonic/collision `STOP`. Learning is a separate checkpoint-
bound operation over the complete observation, the learner's actual action, raw
tool result, scalar consequence, and evaluator receipt. Teacher success becomes
available only after the prequential response.

## Public-TRAIN stream

The evaluator-only descriptions below are never model inputs.

| Episodes | Required pressure |
| --- | --- |
| 01–04 | Two independent examples for each of two prose surfaces. |
| 05–06 | Neutral document ingestion whose facts are needed after the documents disappear. |
| 07 | First knowledge-conditioned code attempt, then one authenticated successful transition. |
| 08 | Fresh renamed two-hop language transfer. |
| 09 | Second independent knowledge-conditioned code transition, permitting action-schema promotion. |
| 10 | Fresh paraphrase transfer without shared content n-grams. |
| 11 | Fresh code transfer through two learned scientific relations. |
| 12 | Language transfer with reversed document order and matched distractors. |
| 13 | Counterfactual code transfer where one changed fact changes both file and value. |
| 14 | Delayed recall of a fact ingested in 05–06 with its original document absent. |
| 15 | Three-hop knowledge-conditioned code repair; request names neither file nor replacement. |
| 16 | Newer provenance-bearing evidence contradicts an old fact; emit the revision. |
| 17 | Apply the revised fact in code. |
| 18 | Recall the provenance supporting the revised fact. |
| 19 | Equally supported conflict makes a code action unsafe: physical `STOP`. |
| 20 | The corresponding language query also requires physical `STOP`. |
| 21 | Delayed code/action retention after revision and conflict. |
| 22 | Unsupported language relation: physical `STOP`. |
| 23 | Two-action recovery: run one diagnostic, consume raw output, repair, and pass execution. |
| 24 | Delayed cross-project three-hop language composition. |

This allocates fifteen language/knowledge episodes and nine coding episodes.
All coding episodes are causally dependent on the language/knowledge state.
Code mutation stays deliberately narrow—one source-relative literal
substitution—so the session measures integration rather than pretending to be
broad synthesis.

## Learner hypotheses under pressure

1. A persistent variable relational workspace over the opaque document tape and
   sparse role/filler substrate must learn recurring slot structure, associate
   paraphrases through coactivation and consequence, retain provenance-bound
   facts, inhibit superseded evidence, and retrieve through a shared recurrent
   one-to-three-hop loop.
2. A pointer-conditioned source motor must consume workspace winners, learn one
   relative path/value substitution program, emit only live sensory bytes or
   neuronally retained lexical fragments, and share the final action arbiter.
3. Tool feedback must re-enter the recurrent workspace as another observation;
   it must not select a host-side task-specific solver.

The near-term implementation order is paraphrase association, neutral
consolidation, provenance/revision, shared recurrence, then the knowledge-action
bridge. Each addition should return to this integrated session rather than grow
an indefinite microgate series.

## Implementation progress and corrected prerequisites

Commit `b5ccc73` implements the first narrow knowledge-action bridge beneath
episodes 07/09/13: a retained exact fact can determine both a fresh
occurrence-relative file edit and inserted lexeme without either being supplied
to inference. It is not a session result and does not make later episodes
eligible yet. The action program is fixed, and the upstream frame was still
learned through an observed-answer compiler.

The next language-first successor must remove that compiler. Neutral documents
should induce anonymous two-slot frames from repeated structure, store both
readout directions, and associate exact surfaces only after two independent
shared filler bindings physically establish a SAME or SWAPPED alignment. A
prequential third binding, a matched shuffled-binding control, relabeling, and
collision STOP are the minimum evidence; lexical recurrence alone is not broad
language understanding.

Episodes 16--20 also require a learned provenance asymmetry before scoring.
Contradictory facts plus opaque source identities are symmetric: append order,
document order, or checkpoint age cannot define truth. Earlier public-TRAIN
examples must teach a transferable raw supersession surface from independently
supported consequences. Its interneuron may inhibit only the superseded source
vote before one unified fact race. Without such learned evidence, equal conflict
must STOP; episode 16 cannot be treated as a first-example last-write-wins task.

## Scoring

Curriculum episodes record first attempts; teacher-provided corrections never
count as model successes. The scored transfer set contains nine language tasks
(seven emits and two STOPs) and seven coding tasks (five direct edits, one STOP,
and one tool-recovery success).

Advance only if:

- language scores at least 8/9;
- coding scores at least 6/7;
- the headline balanced score is the minimum of language and coding accuracy;
- revision, delayed recall, delayed code retention, conflict STOPs, unsupported
  STOP, tool recovery, and final three-hop composition all pass;
- treatment improves over an independently seeded `NO_UPDATE` arm by at least
  three language and three coding episodes;
- no decoy change, nonphysical decision, hidden retry, or inference-time
  checkpoint mutation occurs.

Record outcome and failure distributions, including hop, provenance arbitration,
cleanup ambiguity, knowledge-action bridging, motor materialization, tool
recurrence, domain, checkpoint age, paraphrase family, document order, and
action count.

## Causal arms

- `NO_UPDATE`: independently seeded full stream.
- `CONSOLIDATION_OFF`: delayed recall and dependent code must lose neutral facts.
- `WORKSPACE_RECURRENCE_OFF`: one-hop behavior may survive; multi-hop and tool
  recovery must physically stop.
- `PROVENANCE_INHIBITION_OFF`: ordinary facts survive; revision, provenance
  recall, and conflict handling fail.
- `KNOWLEDGE_ACTION_BRIDGE_OFF`: language stays intact while dependent code
  actions physically stop.

All arms use the same final cell race. An ablation may remove a signal but host
code may not select the resulting answer.

## Leakage and overfit controls

- Generic byte, token, sentence, document, and path framing only; no AST,
  ontology, embeddings, relation IDs, target spans, or candidate patches.
- Requests omit answers, target filenames, module identities, and replacement
  values.
- Private fact graphs and executable expectations are read only after the
  neuronal action.
- Whole-transition authentication may issue learning credit but may not choose
  an internal path.
- Counterfactual pairs hold lengths, vocabulary frequencies, code shape, and
  distractor counts fixed while changing individual facts.
- Names, document order, sentence voice, file position, and directory depth vary
  independently.
- Architecture and cohort hashes freeze before the one-shot run. Failure creates
  a newly named cohort, never a repaired rerun.
- Protected R5/R6 data remain sealed.

## V2 interface lesson

The spent coding-only V2 cohort accidentally placed `obsolete` and `bytes` in
both its query and multiple sources. The binder therefore observed six
query/source coincidences instead of one and correctly rejected the stream
before evaluation. A future cohort should use an opaque request such as
`adjust <stem> now` and freeze an inert exact coincidence-count preflight for
every training row. This corrects the experiment interface, not V2, which stays
immutable.
