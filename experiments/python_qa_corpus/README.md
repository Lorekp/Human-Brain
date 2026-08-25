# python_qa_corpus — raw-Python-text QA corpus (coding-transfer prep)

Frozen input corpus for the coding-transfer experiment that follows the R2
Bible-QA verdict (Bible-pretrain → code-finetune vs code-only). The M0 audit
found the typed-grammar SWE channel has a 0.0 ceiling, so this corpus reuses
the Bible-QA machinery on raw Python text instead: QA triples mined
deterministically from this repository's own `src/humanbrain` modules.

- Builder   : `tools/build_python_qa_corpus.py`
- Library   : `src/humanbrain/python_qa_corpus.py` (mirrors `qa_corpus.py`;
              same `qa_prompt` framing ending in `answer:`)
- Tests     : `tests/test_python_qa_corpus.py`

## Contents

| file                     | what it is                                        |
|--------------------------|---------------------------------------------------|
| `python_qa_corpus_v1.json` | 100 validated triples: 75 TRAIN / 25 VAL, three families (`function_summary`, `module_containment`, `signature_recall`), mined from 32 pinned modules |
| `manifest.json`          | artifact sha256, corpus sha256, pinned git head, per-cohort/per-family counts |

## Discipline

- Deterministic: no wall-clock anywhere; rerunning the builder on an
  unchanged tree reproduces byte-identical artifacts (asserted in-process).
- TRAIN/VAL split is BY FILE: every triple's source module lies wholly in one
  cohort; no function from a VAL file can leak into TRAIN.
- Every answer occurs verbatim (case-folded) inside its context lines.
- Budgets pinned at build time: 75/25 with per-family quotas and
  deterministic backfill.

## Rebuild

```
python tools/build_python_qa_corpus.py
```

The artifact pins its git head; if `src/humanbrain` moves past that head,
rebuild and recommit so the corpus tracks the sources it describes
(`tests/test_python_qa_corpus.py` flags the drift).
