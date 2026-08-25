# Python-QA n-gram baselines (M1 comparators): FINDINGS

Date: 2026-08-24. Branch: `codex/a013-cognition-design` (worktree
human-brain-a013-cognition-design). Harness: `tools/python_qa_baselines.py`
(mirrors `tools/qa_baselines.py` arm-for-arm). Corpus:
`experiments/python_qa_corpus/python_qa_corpus_v1.json` —
`humanbrain/python-qa-corpus/v1`, artifact sha256
`dd4ba1c96c398da161d52703c4ca44a06f0dbdc71496f486183245b8e0b34212`,
corpus digest `789e3ac592fd8aa1d4f68af8704a4a9b19d64db7277efea59ebc50ff8dd446ec`;
TRAIN stream = the 24 TRAIN-role source files joined by `\n`, stream sha256
`06fe41f1a3aa3aed55d590598ddd21baffe4a55e2d2955eabae0c6d7364f47a0`
(53,750 words, per-file digests verified against the frozen corpus before
fitting). Records: `experiments/python_qa_baselines_v1/<arm>.json`,
schema `humanbrain/qa-baselines/v1` — field-comparable to
`experiments/qa_baselines_bible_v1/`.

## Headline numbers (VAL cohort, 25 items)

- **unigram_smoothed answer_span accuracy = 0.0** exactly (0/104 teacher-forced
  predictions).
- **trigram_smoothed answer_span accuracy = 0.057692307692307696**
  (6/104 predictions).

| arm | span_acc | free-run EM | degeneracy guard |
|---|---|---|---|
| unigram_smoothed | 0.0 (0/104) | 0/25 | PASS (6 distinct, modal share 0.48) |
| trigram_smoothed | 0.057692307692307696 (6/104) | 0/25 | PASS (7 distinct, modal share 0.32) |
| majority_floor | n/a (mode floor) | 0/25 | FAIL by construction (1 distinct, share 1.0) |

Slack rule identical to the Bible harness (`completion_token_slack = 8`);
tokenization identical except the word regex also admits `_` so snake_case
identifiers stay one token; prompts use the corpus's own prompt-last framing
(`qa_prompt`, ends `answer:`) and metrics condition on the prompt head only.
Determinism verified: an independent rerun reproduces all three records
byte-identically modulo wall-clock keys. Full run ≈ 4.7 s.

## Reading

1. These are the preregistered M1 comparators for the coding-transfer
   experiment on the raw-Python-text channel, locked BEFORE any cell run
   (ordering discipline in MILESTONES.md). The bar they set is low: a grown
   cell needs >5.77% span accuracy to clear the smoothed trigram.
2. Unigram sits at exactly zero for a structural reason: its argmax is always
   the modal training token, which over this code stream is `self`
   (1,748/53,750), and `self` never occurs inside any VAL answer span. The
   Bible unigram got 4.54% only because English prose answers share high
   frequency words with the prompt stream; code answers do not.
3. Trigram's 5.77% comes from local identifier continuation (signature and
   containment answers echo tokens adjacent in their own def lines), not from
   comprehension — free-run exact match is 0/25 everywhere and completions
   degrade into def-line token soup within a few steps.
4. Degeneracy guards behave identically to the Bible harness: both n-gram arms
   pass (distinct completions ≥3, modal share ≤0.9), majority_floor fails as
   designed.

## Deviations from tools/qa_baselines.py (all mechanical)

- Data load reads the frozen corpus JSON + manifest instead of the Bible
  freeze dir, and sha256-verifies both the artifact and every TRAIN source
  file before fitting; `train_manifest_sha256` / `val_manifest_sha256` are
  canonical-JSON digests of each cohort's entry list.
- One extra provenance key `corpus_artifact_sha256`; all other record fields,
  metric shapes, guards, lambdas (JM 0.6/0.3/0.1), slack, tie-breaking, and
  fail-closed write discipline are byte-matched to the Bible harness.
