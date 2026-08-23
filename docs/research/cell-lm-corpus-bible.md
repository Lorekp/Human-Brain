# Cell-LM v1 corpus pivot: Bible (KJV) primary

Status: DESIGN — corpus-pivot proposal per user directive (2026-08-23). No
freeze action taken; all fetches below are single-fetch dry runs recorded as
evidence, not freeze-time acquisitions. Exclusive-scope file: this document.
Provenance: researched + measured by subagent (live HF API probes, live
mirror fetch, tokenizer replay against the frozen BPE-4096 artifact); all
numbers in §5-§6 are from real fetches/scripts under
`%LOCALAPPDATA%/Temp/hb_cell_lm_freeze/` (scripts `hf_probe*.py`,
`pg10_*.py`; evidence `pg10_book_spans_dryrun.json`).

## 1. Directive and position

The v1 corpus changes: the Bible (KJV) becomes the PRIMARY training text,
replacing the 5-fiction-book Gutenberg set. The fiction freeze candidate
(`FREEZE-CANDIDATE.json`, 834,336 BPE-4096 tokens, books {11, 84, 1342,
1661, 2701}) is demoted to SECONDARY/reference — its hashes stay valid for
the reference arm but no longer gate the v1 claim. Everything else in
`a013-word-prediction-protocol.md` (one-shot discipline, prompt-last
framing, book-level author-disjoint gating split, B0/B1/B2 baselines,
margin+bootstrap gating, shared frozen token stream) carries over unchanged.
This document owns: source selection, split, QA framing, cognitive-property
mapping, red tests, and the freeze-checklist delta.

Why the pivot serves the program (not just LM metrics): the KJV is a single
hash-pinnable public-domain artifact with 66 canonical books (natural
book-level transfer axis), dense cross-referential entity structure (the
same ~1,300 proper names recur across books — an integrated-representation
workout fiction excerpts cannot match), and verse-addressable structure that
supports deterministic, provenance-clean QA generation (the
question-answering directive) without ingesting any third-party
machine-written text.

## 2. Source survey

Surveyed live on 2026-08-23 (HF REST API + datasets-server + direct mirror
fetches). The user-suggested collection
`sugatoray/llm-training-datasets` ("LLM Training Datasets", last updated
2026-08-16) was fetched via the collections API; its visible entries are
generic LLM corpora (HuggingFaceTB/cosmopedia, fka/prompts.chat,
HuggingFaceTB/cosmopedia-100k, CohereLabs/aya_dataset, m-a-p/Code-Feedback,
nvidia/OpenMathInstruct-1, …). None are Bible-specific; the collection's
relevance is format-level only (instruction/QA pairing conventions such as
aya's prompt/response schema), not as a corpus source. Bible-specific
candidates:

| Candidate | What it is | Size | License | Verdict |
|---|---|---|---|---|
| **PG10 — Gutenberg #10 "The King James Version of the Bible"** | Single raw TXT on the already-pinned mirror `https://gutenberg.pglaf.org/cache/epub/10/pg10.txt` (fallback `https://gutenberg.org/cache/epub/10/pg10.txt`) | 4,455,950 B raw (verified HTTP 200, Content-Length + full fetch); 4,436,168 B after the pinned trim rule | Text = KJV 1611, public domain; PG boilerplate stripped by trim rule | **PRIMARY (pure KJV, verse-structured).** Reuses the existing dual-fetch/SHA-256/trim pipeline unchanged |
| **`k-mktr/bible_king_james_version_en`** (HF) | Verse-structured KJV parquet: `data/data.parquet`, exactly 31,102 rows (canonical KJV verse count), fields `book_number/book/book_abbr/chapter/verse_number/verse_suffix/verse_id/text`; 1 config, 3 files, ~4.8 MB in memory | 31,102 verses | No license field on card; content is PD KJV 1611 (card self-describes as KJV 1611) | **SECONDARY — metadata/verse index only** (QA addressing, verse-count cross-check). NOT a training-bytes source: keeps the one-artifact hash discipline |
| **`bible-nlp/biblenlp-corpus`** (HF) | Multilingual verse-aligned corpus (`corpus.json` + `vref.txt`, loader script), ~1M<n<10M rows, ~1,300+ languages, eng included; task: translation | 1M<n<10M rows | CC-BY-4.0 + "other" (per-language licenses vary — must be checked per language) | **OPTIONAL parallel-translation arm** (deferred). Not needed for v1; if used later, per-language license audit is a freeze blocker |
| `davidstap/biblenlp-corpus-mmteb` (+ mirror `mteb/biblenlp-corpus-mmteb`) | The task's "biblenlp-corpus-mm" resolves here: 828 eng–X parallel configs, 2,504 files, 2,758 downloads | 1M<n<10M | CC-BY-4.0 + other | Same role as bible-nlp corpus (it is the MTEB packaging of it); optional arm only |
| `monetise/Structured-Scripture-for-AI` (HF) | Curated scripture QA pairs (`dataset.jsonl`), MIT, task_categories {text-generation, question-answering}, GitHub-synced; NOT parseable by datasets-server (custom loader) — manual inspection required | small | MIT | **SECONDARY QA candidate, quarantined**: usable only after a provenance review confirms answers quote KJV verbatim and no machine-written commentary enters training |
| `Bright1994/theology_qa_bible_filtered` | Theology QA parquet | small | none declared | REJECT (no license, no provenance, 8 downloads) |
| `TitleOS/scripture_1500_pairs_gemini_flash_lite` | 1,500 LLM-generated pairs (torah+quran mix) | 1K<n<10K | MPL-2.0 | REJECT (machine-generated content, non-KJV material — contamination by construction) |
| `nikita200/scripture`, `strijder/scripture` | Unrelated/mismatched repos (one ships diffusion-model weights; one gated+empty) | — | — | REJECT (not Bible corpora) |
| bible-api.com | Live per-verse KJV JSON API, response self-labels `"translation_note": "Public Domain"` (verified with John 3:16) | per-verse | PD | Spot-check tool only (QA red-test cross-check of a verse sample); impractical for 31k-verse bulk |
| openbible.info/text/ | Intended raw-text source | — | — | DEAD END: returns 404 (verified 2026-08-23); not usable |

Contamination notes: KJV 1611 text is public domain — zero license
contamination, and no modern translation (NIV/ESV/…) enters anywhere, so no
publisher-text leakage. The only machine-written text risk is third-party QA
sets; the primary QA plan (§7) generates pairs deterministically from the
frozen PG10 bytes themselves, so the entire training corpus remains one
public-domain artifact. The fiction set (secondary) carries its existing
PG boilerplate hygiene.

## 3. Primary corpus decision: PG10

- **Acquisition**: pinned mirror form
  `https://gutenberg.pglaf.org/cache/epub/{ID}/pg{ID}.txt` with ID=10 — the
  exact mirror, dual-fetch policy, and trim rule already recorded in
  `FREEZE-CANDIDATE.json` apply verbatim. Dry-run single fetch 2026-08-23:
  HTTP 200, 4,455,950 bytes,
  `sha256(raw) = 0204adaed1f25700aa854218cae63c7172228c41088f335e99167a071eed83c0`
  (freeze requires the two-fetch equality check against this value or a
  deliberate re-pin).
- **Trim**: START/END marker rule unchanged; trimmed = 4,436,168 bytes
  (dry-run).
- **Structure** (verified against the fetched bytes): front-matter TOC lists
  all 66 book titles; each title line then re-appears as the book heading;
  chapters/verses are marked inline (`1:1 In the beginning …`). This gives a
  deterministic book-span derivation (§4) without any external index.

## 4. Split design (gating)

**Derivation rule (pinned, deterministic, test-first):**
1. Trim raw PG10 via the pinned START/END rule.
2. TOC region = bytes from start until the SECOND occurrence of
   `The Old Testament of the King James Version of the Bible`.
3. The 66 canonical titles = non-empty TOC lines, excluding the two section
   headers (`The Old Testament…`, `The New Testament of the King James Bible`).
4. Book i's heading = FIRST occurrence of its exact title line after book
   i−1's heading (scan starts after the TOC region); span_i = [heading_i,
   heading_{i+1}), last book ends at EOF.
Dry-run verification: 66/66 titles parsed, 66/66 headings located in order,
span coverage = 99.95% of trimmed bytes (the 0.05% is TOC/front matter,
excluded from training by construction).

**Genre blocks** (canonical order; note Daniel groups with the major
prophets, Hebrews with the Pauline epistles per its KJV heading):
Pentateuch #0-4 · History #5-16 · Wisdom/Poetry #17-21 · Major prophets
#22-26 · Minor prophets #27-38 · Gospels #39-42 · Acts #43 · Pauline
epistles #44-56 · General epistles #57-64 · Apocalypse #65.

**PRIMARY gating split (proposed): whole-block holdout of TWO intact
blocks.**
- **Validation (17 books, 2 blocks):** Minor prophets #27-38 (Hosea→Malachi)
  ∪ Gospels+Acts #39-43 (Matthew→Acts).
- **Train (49 books):** everything else (Pentateuch, History, Wisdom, Major
  prophets, Pauline + General epistles, Revelation).
- Rationale: (a) block integrity — no genre family is split across sides;
  (b) author/attributed-author disjointness is structural (no held-out book
  shares attribution with any train book); (c) it avoids the synoptic trap:
  holding out ONE gospel while training on the other three would leak
  val content through near-verbatim synoptic parallels (Matt/Luke share
  ~200 verses with Mark) — rejected explicitly; (d) val spans two distant
  genres (narrative-prophecy + narrative-gospel), so the generalization
  measurement is cross-genre, not within-family interpolation.

**Measured dry-run accounting** (fiction-trained BPE-4096 replayed over the
real fetched bytes — an UPPER-BOUND token proxy; the freeze retrains the
tokenizer on the new train split, which will compress KJV English better):

| Set | Books | Bytes | Tokens (proxy) | Share |
|---|---|---|---|---|
| Train | 49 | 3,656,777 | 1,162,238 | 82.9% |
| Validation | 17 | 777,206 | 239,446 | 17.1% |
| Total (66) | 66 | 4,433,983 (+2,185 front matter) | 1,401,684 | 100% |

Largest books (tokens, proxy): Psalms 82,188 · Jeremiah 75,488 · Genesis
67,852 · Ezekiel 66,750 · Isaiah 65,353. Smallest: 2 John 515 · 3 John 552 ·
Philemon 815. Per-book table: `evidence/pg10_book_spans_dryrun.json`.

**Projected freeze-time tokens**: with a Bible-retrained BPE-4096 expect
~5-15% fewer tokens (archaic high-frequency function words merge well);
working estimate ≈1.2-1.35 M total. Either way the corpus lands INSIDE the
protocol's 1-10 M token target band, which RESOLVES deviation D1 of the
fiction candidate (834,336 tokens was below band).

**Parallel-passage same-side hygiene** (new, testable): the canon's known
parallel corpora must sit on ONE side of the split. Under this split they
do: Samuel/Kings ↔ Chronicles (both train), Isaiah 36-39 ↔ 2 Kings 18-20
(both train), Jude ↔ 2 Peter 2 (both train), synoptic triples + Acts (all
val). Red test 18 enforces this list.

**Rejected alternatives (recorded so freeze doesn't relitigate):**
- *Single-gospel holdout* (e.g., Mark out): synoptic leakage — rejected.
- *Minor-prophets-only holdout* (12 books, 52,594 tok val): cleanest
  style-disjoint block, but val QA diversity collapses to one genre;
  retained as the preregistered FALLBACK swap if freeze-time verification
  finds a leakage/imbalance problem with the primary — declared now, not
  chosen post hoc.
- *Gospels+Acts-only holdout* (186,852 tok val): richest QA material but
  removes the most entity-dense narrative from LM training; superseded by
  the two-block primary.
- *Positional last-10% cut*: stays demoted to non-gating diagnostic per the
  protocol's cross-review.

## 5. Tokenization delta

Unchanged mechanism (byte-level BPE, {4k, 8k} sweep on train-only data,
deterministic pretokenize `\S+\s*` per book, pinned tie-breaks). What
changes: the sweep and the final tokenizer train on the NEW 49-book train
split; the fiction-trained artifact (`tokenizer_bpe_4096.json`,
sha256 `6757495c…`) is demoted with the fiction corpus. All systems (B0/B1/
B2/cell) consume the one frozen Bible tokenizer, per red test 14. The 8k arm
matters more here: 31,102 verses of high-frequency archaic function words
("and", "unto", "it came to pass") may compress enough that 4k wastes
accuracy — the sweep decides, once, before any cell run.

## 6. QA-format design (question-answering directive)

**How QA enters a next-token-prediction circuit.** A QA pair is just a
sequence with a prompt/answer boundary — the harness already has exactly
this contract: `prepared = (input_ids, teacher_ids)`, and under
`--supervision program_only` the grow loop strengthens ONLY the transitions
where `index + 1 >= len(input_ids)` (benchmarks/ladder_experiments.py:573-580;
src/humanbrain/typed_program.py:1218-1227 — the A2 lane's
`program-token-targets-only` scope). So:

- **Framing (prompt-last, red test 4 preserved):** every QA item is
  serialized `…context… \nQ: {question}\nA: {answer}\n`; `input_ids` =
  all tokens up to and including the `A:` delimiter; `teacher_ids` = the
  answer tokens + terminator. The prediction target remains LAST; QA does
  not introduce a new framing order, it reuses the sealed one.
- **Delimiters:** plain byte strings `"\nQ: "` and `"\nA: "` encoded by the
  same frozen BPE (no special tokens, no vocab surgery — byte-level BPE
  represents them deterministically). Exact byte strings pinned in the
  freeze manifest; red test 17 asserts the boundary lands exactly there.
- **Supervision semantics:** plain-LM epochs use
  `--supervision all_transition` (every next-token target strengthened);
  QA epochs use `--supervision program_only` (only answer transitions
  strengthened — answer-only strengthening). This is precisely the T4
  "supervision-density mixing" axis (a013-teaching-processes.md row T4):
  the preregistered mixing arms {all,all,program}-style epoch schedules
  apply unchanged, with QA epochs substituting for program epochs —
  framing supervision builds shared geometry, answer-only pressure
  concentrates where the question is decided. T4's mechanism gap (per-epoch
  scope override hoisting the fixed check at typed_program.py:1224-1227) is
  shared plumbing with the QA pivot, not new QA-only work.
- **Where QA pairs come from (provenance-clean by construction):**
  deterministic template generators run ONCE at freeze over the frozen,
  trimmed PG10 bytes + the verse index (k-mktr parquet for addressing;
  bible-api.com spot-checks a sample):
  - G1 next-verse: given verse n of chapter c, complete verse n+1 (pure LM
    target in QA clothing — bridges the two task families).
  - G2 source attribution: "Who said, '…'?" with speaker strings extracted
    from canonical narrative patterns (`and X said`, `then answered X`) —
    proto-metacognition about who-knows-what (§8.5).
  - G3 address lookup: "What does the book of Jonah chapter 1 verse 3
    say?" (book/chapter/verse address → text binding).
  - G4 structural: first words of a book, book ordering, chapter counts
    (from the verse index only — never from generated commentary).
  Every generated manifest is hashed into the freeze artifact; generation
  is rerun-and-equal (red test 15). Train QA draws ONLY from train books,
  val QA ONLY from val books (red test 16).
- **Evaluation additions:** QA answer top-1 exact-match reported beside
  perplexity/top-1 on the LM stream, with its own B0-analogue floor guard
  (majority-answer predictor — red test 20). QA items are evaluation
  families, gated all-or-nothing like the existing families; the LM gate
  remains the capability bar (QA strengthens the cognitive-property story,
  it does not dilute the preregistered LM claim).
- **New plumbing QA needs (beyond existing knobs):** (1) a deterministic
  QA-manifest builder tool + hash pinning; (2) a sequence packer that
  interleaves LM-stream segments and QA items under a declared mixing ratio
  (the T4 epoch-schedule override covers the supervision side; the packer
  covers the data side); (3) QA metrics registration in the scorecard;
  (4) delimiter-boundary and manifest-disjointness red tests. No changes to
  the circuit, the learning rule, or the framing order.

## 7. Cognitive-property mapping (corpus → architecture goals)

One line per user property: how the Bible-QA task family exercises it, or
an honest "not exercised".

1. **Strong recurrence** — EXERCISED: cross-chapter narrative dependencies
   (pronouns, continuing arcs) inside W-token windows make multi-step trace
   carry load-bearing; QA-on-prior-chapter items (G1/G2) are direct
   long-range-binding probes.
2. **Integrated representations** — EXERCISED: ~1,300 recurring proper
   names across 66 books force entity-consistent whole-context codes;
   answering "who said X" for a figure last named chapters earlier cannot
   be solved by local n-gram lookup.
3. **Global workspace (3a/3b/3c)** — DEMANDED, NOT PROVIDED: a QA answer
   step must let scattered evidence (question constraints + passage facts)
   compete and broadcast ONE winning token; the task supplies the pressure,
   the workspace machinery itself remains ABSENT (scorecard GWT-1..3) —
   corpus choice serves the goal, architecture must still deliver it.
4. **Persistent internal state** — EXERCISED: book-length sequences with
   `reset_sequence` at document boundaries; cross-chapter vs within-chapter
   QA accuracy decay measures retention directly (and stress-tests the
   sealed framing-dependent-retention lesson).
5. **Metacognition** — PROTO: source-attribution (G2) plus preregistered
   "cannot be determined from this passage" abstention targets give a
   calibration signal; margins-as-input machinery remains future work (HOT).
6. **Attention model (state-dependent selection)** — PARTIAL: the same
   passage supports different answers under different questions, so
   selection must depend on state carrying the question, not template
   position; in-circuit gating itself stays ABSENT (GWT-4).
7. **Self-model** — NOT EXERCISED (honest none): nothing in Bible-QA
   represents the system's own selection process; nearest future hook is an
   abstention head over property 5.
8. **Autobiographical memory** — PARTIAL, plumbing-dependent: the corpus
   supplies re-encounter structure (formulaic recurrences, repeated
   formulae) that rewards replay-consolidation (T6 seam), but the property
   is exercised only if the replay pass is built; corpus alone provides no
   memory.
9. **Agency** — NOT EXERCISED (honest none, mirrors AE-1 ABSENT): questions
   and targets are externally fixed; no self-generated subgoals.
10. **Embodiment/closed loop** — NOT EXERCISED (honest none): text-in/
    text-out; the sandbox action→observation seam is unused by this corpus.
11. **Valence/homeostasis** — MARGINAL: item difficulty spreads margin
    distributions widely (genealogy vs narrative QA), giving the
    plasticity-rate homeostat a real regulated quantity; no affective
    valence claim.

Net: the pivot converts 4 properties from unexercised to exercised/proto
(1, 2, 4, 5) and 2 more to demanded-but-unbuilt (3, 6) relative to a plain
fiction-LM stream — while leaving 7, 9, 10 honestly untouched, consistent
with the roadmap's "out of scope for fixed-task prediction today" note.

## 8. Red tests (additions 15-20; 1-14 unchanged)

15. **QA-manifest determinism + hash pin:** rerunning the generators on the
    frozen bytes reproduces a byte-identical manifest; manifest sha256
    pinned in the freeze artifact.
16. **QA split disjointness:** every val-QA verse address (book, chapter,
    verse) is absent from train-QA AND from the train LM stream manifest;
    enforced at the verse-address level, not just book level.
17. **Delimiter-boundary integrity:** for every QA item, the input/teacher
    token split falls exactly on the `A:` delimiter boundary; no answer
    token appears at prompt positions (extends framing-order red test 4).
18. **Parallel-passage same-side assertion:** the pinned parallel list
    (Samuel/Kings ↔ Chronicles; Isaiah 36-39 ↔ 2 Kings 18-20; Jude ↔
    2 Peter 2; Matthew/Mark/Luke/John/Acts) is verified single-side.
19. **Verse-index round trip:** PG10-derived verse count reconciles with
    the canonical 31,102 (k-mktr parquet) within a preregistered documented
    delta (PG10's inline C:V markers vs the index); mismatch beyond delta
    blocks freeze.
20. **QA floor guard:** a majority-answer (mode) baseline for QA top-1; any
    system at or below it is flagged exactly like the B0 unigram guard.

## 9. Freeze-candidate checklist updates (delta for v2)

FREEZE-CANDIDATE.json → v2 (`bible-kjv-pg10` candidate) must:
1. Replace `corpus` with PG10: dual-fetch `pg10.txt`, enforce sha256
   equality with `0204adaed1f25700aa854218cae63c7172228c41088f335e99167a071eed83c0`
   (or deliberate re-pin); trim-rule recompute (dry-run: 4,436,168 B).
2. Record the 66-book derivation (§4 rule) + per-book span table
   (regenerate `pg10_book_spans_dryrun.json` at freeze from the freeze-time
   fetch, not this dry run).
3. Pin the split: train 49 / val 17 (minor-prophets ∪ gospels+acts blocks);
   run the author-disjointness and parallel-passage checks (red tests 2/8/18).
4. Retrain tokenizer {4k, 8k} on the new train split only; execute the
   single vocab-selection sweep; re-pin `tokenizer_sha256`; recompute all
   token counts (expect ≈1.2-1.35 M total — D1 resolved, band satisfied).
5. Build + hash the QA manifests (G1-G4) from the frozen bytes; pin
   delimiter byte strings; register QA metrics + majority-answer floor.
6. Demote the fiction set to `secondary_reference` (keep its hashes; no
   gating role), and amend the protocol doc's Corpus section to cite PG10 +
   this split (protocol edit belongs to the protocol owner).
7. Keep D4 discipline: dry-run stability here does NOT substitute for the
   freeze-time fetch; any fetch failure BLOCKS freeze.
8. Optional arms (deferred, each a freeze blocker if enabled without its
   audit): parallel-translation arm (bible-nlp corpus — per-language
   license audit), curated-QA arm (monetise set — provenance review).

## 10. Open items

- Protocol doc amendment (Corpus + Evaluation sections) — owner: protocol
  doc; this file is the source for the edit.
- T4 per-epoch scope override lands with or before QA epochs (shared
  mechanism; currently a proposed row in the teaching-processes matrix).
- QA abstention targets ("cannot be determined") need a preregistered
  generator spec before freeze if the metacognition proto-claim (§8.5) is
  to be measured rather than asserted.
- 8k-vs-4k vocab sweep on KJV text (expected to matter more than on
  fiction; decided once at freeze).
- Hebrew-Greek interlinear sources (openbible interlinear etc.): surveyed
  NOT-NEEDED for v1; openbible.info/text is 404; any future interlinear arm
  needs a fresh source survey.

## 11. Verification log (this design pass, 2026-08-23)

- Live fetches: PG10 mirror HTTP 200, 4,455,950 B, sha256 recorded (single
  fetch, dry-run only); bible-api.com KJV verse JSON OK ("Public Domain");
  openbible.info/text/ → 404.
- HF REST probes: k-mktr (31,102 rows verified via datasets-server size +
  first-rows), bible-nlp corpus (license cc-by-4.0+other, corpus.json +
  vref.txt), davidstap/biblenlp-corpus-mmteb (828 configs), monetise (MIT,
  QA categories, unparseable by datasets-server), Bright1994 (no license),
  TitleOS (MPL-2.0, LLM-generated), sugatoray collection (generic corpora,
  none Bible-specific).
- Token replay: fiction-trained BPE-4096 over fetched PG10 → 1,401,684
  tokens (3.16 B/token proxy); per-book spans via the pinned TOC-anchored
  derivation (66/66, 99.95% coverage) → split arithmetic in §4; evidence
  written to `%LOCALAPPDATA%/Temp/hb_cell_lm_freeze/evidence/`.
- Harness citation check: `--supervision {all_transition, program_only}`
  and the `index + 1 >= len(input_ids)` strengthen condition verified in
  benchmarks/ladder_experiments.py:573-580 and
  src/humanbrain/typed_program.py:1218-1227; T4 row verified in
  a013-teaching-processes.md.
