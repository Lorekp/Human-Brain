# KS2 2024 public reading development diagnostic

This is an authentic-question, text-only development subset from **Streaky and
Squeaky**, administered by England's Standards and Testing Agency in May 2024.
It evaluates Q6a and Q6b (2 marks). The other 14 marks belonging to this passage
and the rest of the 50-mark paper are **unevaluated**. This is neither a full exam
score nor a protected or reserved evaluation. Full provenance, original PDF
hashes, extraction changes, license and whole-paper exposure are in manifest.json.

Contains material developed by the Standards and Testing Agency for 2024 national
curriculum assessments and licensed under
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
The [official source](https://www.gov.uk/government/publications/key-stage-2-tests-2024-english-reading-test-materials)
provides the passage, questions and official mark scheme. No photography, images
or logos are redistributed here. In particular, the tenrec photograph credited to
Arto Hakola / Alamy Stock Photo, ENBJ9N, is excluded.

## Declared diagnostic and budget

Observed mechanism limitation: the neutral reader expects an exact learned
sentence frame with one token missing. These original exam questions use natural
interrogatives, repeated entities, longer prose and a contextual pronoun.
Hypothesis: its currently demonstrated two-slot mechanism does not yet support
this language interface. Completion means recording original-question responses,
STOPs or budget failures, with matched acquisition controls, without question or
answer-driven tuning. A successful run need not have positive marks.

Two subprocesses start independently from the same empty `start()` checkpoint.
`NO_UPDATE` has no acquisition updates. `UPDATE` receives the full passage once
through `consolidate`. Each answers Q6a and Q6b once with the entire passage
visible. Question 6a receives its shared stem and part a; question 6b receives
the shared stem and parts a and b, so the antecedent of “they” remains available.
The target response field is the final part. Neither previous answers nor keys
are included. Inference must leave the checkpoint unchanged. There is no teacher
feedback, synthetic frame seeding, delayed-memory condition, or LLM component.

Default budget: **90 seconds total per subprocess**, including import,
acquisition and two query calls, one subprocess at a time. The parent forcibly
kills and waits for a timed-out worker, retaining flushed progress and marking
unanswered items as budget failures. This whole-arm limit is deliberately simple;
latencies for completed acquisition/query calls are also reported. Python
allocation tracing is enabled and its overhead is included. Native allocations
and total process peak RSS are not measured by this diagnostic.

## Run and score

Use the checkout's environment after `tools/dev.py doctor`:

```powershell
.venv/Scripts/python.exe experiments/reading_exam_development/run.py --check
.venv/Scripts/python.exe experiments/reading_exam_development/run.py --output experiments/reading_exam_development/initial.result.json
```

The output path must be fresh. The parent opens the evaluator-only rubric; the
worker reads only passage, questions and manifest. This process boundary is
auditable data routing, not an operating-system security sandbox. Workers have
no code path to load keys or score answers. Source text naturally includes facts
needed to answer, as in the examination.

The parent awards official canonical responses, records STOP/missing/invalid
responses as zero, and leaves every other answer pending blinded human scoring
under the official principles. It never converts a noncanonical paraphrase into
a wrong answer solely because exact matching fails. No LLM judge supplies marks.
Retain pending scores and omit an earned-total claim until adjudicated.

The artifact records fixture/source/checkpoint hashes, both arms, causal traces'
aggregate cell steps and synapses, acquisition summaries, per-query latency,
STOP reasons and Python traced peak allocation. Each question shares the same
passage: two answers are not independent evidence of general reading ability.
No advancement threshold is set. Before broader claims, evaluate the remaining
original question types and reserve a fresh paper/year before inspecting it.

## Initial measured development result

`source-stable.result.json` records both arms completing inside their 90-second
budgets, with the five recorded implementation files unchanged across the run.
Both start from checkpoint
`9a2be202d30e0517f920858dd159c6aebf222dce399d4b9a7f4ec0576f83144f`.
Both receive **0/2 marks**: each question returns `motor:tonic_stop`, no emitted
bytes and zero support paths. The other 14 passage marks remain unevaluated.

The updating arm acquires the 3,578-byte passage in 46.54 seconds and retains
5,146 evidence records, but promotes **zero frames and zero facts**. It reports
25,732,058,400 learning physical cell steps and 2,042,962 active synapses; these
are implementation counters, not independently measured hardware operations.
Its two queries take 4.00 and 4.22 seconds; total subprocess time is 61.92 seconds.
The frozen subprocess takes 7.26 seconds, with queries taking 1.80 and 1.67 seconds.
Python traced peak allocations are 130,347,390 and 38,248,338 bytes respectively.
Tracing overhead and possible concurrent CPU work are included; these are
diagnostic resource observations, not isolated performance benchmarks.

This establishes failure of an empty reader with one passage exposure under
original exam questions. It does **not isolate question understanding from the
upstream absence of promoted frames**, or test a generally pretrained reader.
No answer-driven tuning or extra exposure followed these failures.

`initial.result.json` preserves an earlier attempt with the same outcomes, but
the reader source changed during that run. Its initial source hashes are not a
reliable cross-arm code identity; use the source-stable artifact above.
`timeout-selfcheck.result.json` deliberately uses a 1 ms budget to verify child
termination and unanswered-timeout reporting. It is tooling evidence, not a
reading capability result. Fixture checks, canonical/pending/invalid/STOP scorer
selfchecks, and `ruff check --no-cache` passed. The source-stable rerun added no
learner exposure to an existing checkpoint: both fresh arms restarted empty.
