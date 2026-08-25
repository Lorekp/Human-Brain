# FINDINGS: concept-chunk tokenizer prototype (study `concept-tokens-v1`)

**Question:** do recurring multi-token *concepts* mined from a corpus carry
compressible structure that plain BPE misses?

**Decisive answer: yes — on in-domain text a concept stack beats even a
larger-vocabulary BPE on one domain and matches-budget BPE everywhere, but
concepts transfer poorly across domains (as predicted).**

- Bible (in-domain): ConceptStack **2.465 B/tok** vs BPE-4096 2.299 (**+7.2%**) and
  vs matched-vocab BPE-1280 2.066 (**+19.3%**). The stack wins *despite* having
  1280 ids against BPE's 4096.
- Python code (in-domain): ConceptStack 3.270 B/tok vs BPE-4096 3.452 (**−5.3%**),
  but vs matched-vocab BPE-1280 2.762 (**+18.4%**).
- Cross-domain transfer collapses (expected): bible-fitted stack on code drops to
  1.196 B/tok (−21.6% vs same-stack in-domain); code-fitted stack on bible drops
  to 1.330 (−59.4%). Concepts are corpus-specific structure, not universal glue.

## Why this answers the question

The baseline is GPT-2-style BPE: merges are confined inside pretokenization
pieces, so BPE *cannot represent* multi-word units like `"the LORD "` or
`")\n        "` as one token at any vocabulary budget. The concept layer adds
exactly this axis: recurring spans across piece boundaries become single atomic
ids. On formulaic, high-register text (KJV) that axis dominates; on
heterogeneous code, 4096 intra-piece byte merges still buy more raw compression
than 299–512 concepts — but at equal vocab budget the structural axis wins both
domains (+18–19%). For the two-corpus experiment design this says: give each
domain its own concept bank; expect near-zero sharing; treat concepts as the
shared *feature* substrate (cell assemblies), not as a universal tokenizer.

## Method

### ID layout and collision guard (cited)

From `src/humanbrain/tokenizer.py`: `self._id_to_bytes = [bytes([i]) for i in
range(256)] + unique`, phrase ids assigned via `enumerate(unique, start=256)`,
capped by `TokenizerConfig.max_phrases = 512` → base vocabulary occupies
**[0, 768)** under defaults. `concept_tokens.py` therefore reserves
**[1024, 1536)** for concept ids (`CONCEPT_ID_BASE = 1024`,
`MAX_CONCEPTS = 512`; `[768, 1024)` unused headroom) and the constructor raises
`ValueError` if `base.vocab_size > CONCEPT_ID_BASE`. Tests assert band
disjointness, per-id payload round-trip (`token_bytes`), and rejection of an
oversized base tokenizer (`tests/test_concept_tokens.py`).

### Miner (stdlib-only, exact for its gates)

`mine_concepts_detailed(texts, base, ConceptConfig)`:

1. Encode each document with the fitted BytePhraseTokenizer (no cross-document
   n-grams); count unigrams over the id stream.
2. Level-wise Apriori n-gram counting (n = 2..12 tokens): an n-gram is counted
   only where its frequent (n−1)-prefix occurs — exact w.r.t. `min_frequency`.
3. Mutual-information gate `log2(p(ngram)/∏p(token_i)) ≥ 5.0 bits`.
4. Dedup by payload bytes, rank by (−count, −MI bits, bytes), cap 512.
   Defaults follow the brief: freq ≥ 50, MI ≥ 5 bits, ≤ 512 concepts, span
   2..12 base tokens.

Encoding: greedy longest-match merge of the base id stream through a trie keyed
on canonical concept id-sequences; decode expands concept ids back through the
base layer. Losslessness was asserted on every grid cell below
(`assert decode(encode(x)) == x` on full corpora, incl. non-ASCII probes).

### Corpora

| corpus | bytes | source |
|---|---|---|
| bible_train | 3,656,825 | 49 TRAIN books sliced from `%TEMP%/hb_bible_freeze/evidence_pg10.trimmed.bin` via `pg10_book_spans_v3.json` (freeze candidate `cell-lm-v1-freeze-candidate-bible-kjv-pg10-v3`, trimmed sha256 `d585e56…9ad3aa`) |
| code_sample | 215,694 | first 13 of 41 `.py` files under this repo's `src/humanbrain/` in sorted order (~200 KB target) |

Fits are strictly per-corpus and in-domain unless labeled otherwise. All
tokenizers verified lossless on both corpora.

## Compression grid (bytes/token; higher = better)

| fit ↓ eval → | bible_train | code_sample |
|---|---|---|
| **bible-fitted** | | |
| BPE-4096 | 2.299 | 1.525 |
| BPE-1280 (matched budget) | 2.066 | 1.397 |
| BytePhrase base only [0,768) | 1.801 | 1.115 |
| **ConceptStack** (768+512 ids) | **2.465** | 1.196 |
| **code-fitted** | | |
| BPE-4096 | 1.642 | 3.452 |
| BPE-1280 (matched budget) | 1.516 | 2.762 |
| BytePhrase base only [0,768) | 1.203 | 2.380 |
| **ConceptStack** (768+299 ids) | 1.330 | **3.270** |

Deltas of interest: stack vs BPE-4096 in-domain = **+7.2%** (bible), **−5.3%**
(code); stack vs matched BPE-1280 in-domain = **+19.3%** (bible), **+18.4%**
(code); cross-domain stack drops −21.6% / −59.4% respectively.

Raw JSON: `results_v1.json` (provenance, thresholds, timings, full grid,
top-50 lists with counts and MI bits).

## Top-50 discovered concepts

Full machine-readable lists in `results_v1.json`
(`top50_concepts_bible`, `top50_concepts_code`). Top of each, `count × MI bits`:

**Bible** (eyeball: verse-boundary scaffolding + KJV register; genuinely
domain-specific units surface, e.g. `the LORD`):

```
19705×9.15  '.\r\n\r\n'      9593×5.29  'of the'        5147×6.70  'the LORD'
19701×6.15  ', and'          8248×8.52  ' of the '      3108×5.29  ' and the'
18038×7.74  ', and '         3800×6.14  ' in the'       2974×7.57  'eth'      (-eth verb suffix)
                                              2773×9.25  ' the LORD '
                                              2083×6.31  ', that '
                                              1977×7.54  'shall be'
```

**Code** (eyeball: indentation scaffolds dominate by count; higher-MI entries
are true idioms — type-hint arrows reach MI 18.1 bits):

```
1104×5.33   ' = '            510×7.40   ')\n        '    276×7.78   'self._'
504×8.05    ',\n            '                224×12.97  ' -> '
462×7.31    ',\n        '                    222×18.14  ') -> '
450×6.78    '),\n'           205×12.26  '\n        self.'
215×8.06    ',\n                '            208×6.89   ' np.'
                                             190×6.52   ' ctypes.'
```

Quality read: concepts are (a) whitespace/indentation scaffolds — high count,
low surprise, real compression; (b) domain register units — `the LORD`,
`shall be`, `-eth`, `self._`, `->`, `np.`/`ctypes.` — exactly the "atomic
concept" targets; (c) some near-duplicate shifted variants (`'of the'`,
`' of the'`, `'of the '`…) spend slots; a substring-collapse pass in dedup is
the obvious next improvement.

Mining yield: bible hits the 512 cap; code yields only **299** concepts at
freq ≥ 50 on 215 KB (sensitivity: freq ≥ 20 caps out at 512 — smaller/heterogeneous
corpora need lower gates, which matters when sizing the reserved band).

## Threats to validity

- BPE baseline is pretokenized (GPT-2-style, same `_PIECE` shapes as
  BytePhraseTokenizer). A whitespace-spanning BPE could learn some multi-word
  units given enough budget; our claim is about the standard architecture plus
  the matched-budget comparison, where the stack wins regardless.
- One repo sample (13 files) is not a code universe; counts shift with file mix.
- Greedy longest-match encoding is deterministic but not globally optimal.
- Compression ≠ learnability: the point of concepts is feature starvation
  (one distributed code per recurring unit), not raw ratio; ratio here bounds
  how much structure exists to attach codes to.

## Repro

```bash
cd <worktree> && PYTHONPATH=src python experiments/concept_tokens/run_study.py
PYTHONPATH=src python -m pytest tests/test_concept_tokens.py tests/test_tokenizer.py
```

Artifacts: `experiments/concept_tokens/{run_study.py, bpe_baseline.py,
results_v1.json, FINDINGS.md}`; module `src/humanbrain/concept_tokens.py`;
tests `tests/test_concept_tokens.py`.
