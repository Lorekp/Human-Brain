# T-ROUTE reserve plan — synthetic who-said episode synthesizer

**Status: BUILT & GREEN — ready for immediate use if E3 passes.**
One command regenerates everything byte-identically:
`PYTHONPATH=src python -m humanbrain.episode_synthesizer --seed 7`

## What was built

`src/humanbrain/episode_synthesizer.py` (stdlib-only) mines speech-event
episodes mechanically from the pinned PG10 KJV corpus using the corpus
builder's OWN narrative-pattern miner (`tools/build_bible_qa_corpus.py`
imported verbatim — same verse parser, split rule, speaker-shape gate,
delimiters), extended with additional attribution idioms. It emits
`(verse_ref, speaker, quote_span)` triples whose quote span is, by
construction, a contiguous verbatim substring of its source verse text.

## Headline numbers (seed 7)

| artifact | scope | episodes |
|---|---|---|
| `episodes.jsonl` | all 66 books, frozen-item addresses excluded (**the task brief's "all 31,102 verses"**) | **2,006** |
| `episodes_train.jsonl` | TRAIN-side 49 books only (builder split rule) | **1,656** |

- vs natural corpus: **200 → 2,006 = 10.0x** (33.4x on the who_said family
  specifically: 60 → 2,006).
- 219 unique speakers; 47/66 books contribute >=1 episode.
- All manifest checks PASS (verbatim, caps, frozen disjointness, pins).

## HYGIENE WARNING (read before E3 preregistration)

`episodes.jsonl` includes validation-book verses (gospels/acts/minor
prophets). That matches the task brief literally ("ALL 31,102 verses") and is
safe for any protocol that does NOT gate on the held-out VAL cohort — but it
CONTAMINATES unseen-book generalization claims. If E3's gates score VAL
who_said items (mirroring G-T1's two-split design), train from
`episodes_train.jsonl` instead and expect ~1,656 episodes. Both files are
committed so verdict-to-campaign latency stays zero either way; the choice is
one line in the E3 harness.

Note: scanning only TRAIN-side books cannot reach 2,000: the complete named-
attribution universe of the 25,266 TRAIN-side verses is ~1,356 verses
(census: proper-name + speech-verb adjacency), ~1,650 after prophetic-formula
tiers. The 2,000 floor requires the gospels' dialogue density.

## Extraction design (all mechanical, all guarded)

Pattern tiers (candidates, all-scope): `short_infix` dominant, `thus_saith`
527, `miner_exact` 93.

1. **miner_exact** — the builder's own `SPEAKER_VERB_RE` ("and X said,")
   used verbatim.
2. **short_infix** — the KJV idioms the builder regex cannot see because it
   demands a comma right after the verb:
   - `<Name> said/spake/answered unto|to <X>[, saying],` (dominant class)
   - `<Name> answered [unto X] [the LORD] and said,`
   - `<Name> answered <obj>,` (Job-cycle form)
   - `<Name> [<=5 lowercase clause tokens][, and] said,` ("Abraham drew
     near, and said") — lowercase-only fillers make sentence-boundary
     crossings structurally impossible.
3. **thus_saith** — closed formulas with whitelisted speakers: `Thus saith
   the LORD[,;]`, post-posed `saith the LORD[,;]`, backward `, ..., saith
   the LORD.` (sentence-bounded pre-comma span as the quote), and `the word
   of the LORD came unto X, saying,`.
4. **Verse-final anchors** — an attribution formula reaching verse end
   ("And the LORD spake unto Moses, saying,") takes its quote from the first
   span of the FOLLOWING verse; the anchor address stays the reference and
   the source verse is recorded in `meta.source_address`. This class alone is
   worth ~120 episodes (Exodus–Numbers law formula).

Guards (each event independently):
- nested-speech guard scoped to the CURRENT SENTENCE SEGMENT (any prior
  speech verb or question mark => skip);
- <=2 events per anchor verse;
- speaker purge: pronouns (builder blocklist) AND sentence-initial junk
  ("And", "The", "Thus" as lone captures — found leaking during development
  and fixed);
- quotes MIN..MAX words (2..20); two-word spans must carry a content word
  (`_TINY_STOPWORDS` guard kills `"and said"`-type fragments); cascade cut at
  `[.?!;]` -> first `:` -> hard truncate to 20 words keeps every emitted span
  a contiguous substring of its source verse.

Dedupe: anchor+source addresses excluded against ALL 260 frozen QA item
addresses (TRAIN 200 + VAL 60; digest pins verified before use — raw sha256
plus embedded/recomputed canonical); same-address pair collapse; identical
utterances at DIFFERENT addresses are kept deliberately (separate narrative
speech acts with different passages).

Determinism: extraction is a pure function of (pinned corpus bytes, pins);
ordering is a seed-keyed shuffle — same seed reproduces byte-identical files
(ids and order included), different seeds permute order only. Red tests:
`tests/test_episode_synthesizer.py` (13, all green).

## Distributions vs natural corpus

Quote length in words:

| bucket | natural (N=60) | synthetic (N=2,006) |
|---|---|---|
| 2–4 | 17% | 8% |
| 5–8 | 47% | 18% |
| 9–12 | 37% | 18% |
| 13–16 | 0% | 21% |
| 17–20 | 0% | 35% |

mean 7.58 / median 8 (natural) vs mean 13.17 / median 14 (synthetic). The
synthetic corpus skews LONGER than the eval-format ceiling (12 words) because
the 20-word cap plus colon/truncate cascade keeps long prophetic speeches.
If E3 wants length-matched training, filter `meta.quote_words <= 12`
(keeps 883 all-scope / 704 train-scope episodes — still >11x natural).

Speakers (top 15, surface forms kept verbatim): the LORD 427, LORD 335,
Jesus 119, the Lord GOD 101, David 88, Moses 67, God 59, Israel 43, Saul 35,
Samuel 26, Joseph 24, Peter 23, Lord 19, Jacob 18, Joshua 17. NOTE for
downstream: `the LORD`/`LORD` are the same referent with different surface
forms (builder convention never normalized them; we preserve that). Top book
contributors: Jeremiah 261, Genesis 177, 1 Samuel 162, Ezekiel 127, Exodus
127, Numbers 120.

Random-sample QA (n=18, unedited): every binding manually verified correct
(Nathan 2 Sam 7:3, Baruch Jer 36:18, Peter Acts 5:3, Jesus Matt 19:28/Luke
22:48, ...). Known residual risks, accepted: rare appositive-name bindings
("Phinehas, the son of Eleazar..." class rejected by the single-token shape
gate rather than risk it), and backward-saith sentences whose pre-comma span
is narration rather than direct speech (~handful; sentence-bounded by
construction).

## Files

- `src/humanbrain/episode_synthesizer.py` — synthesizer + CLI.
- `tests/test_episode_synthesizer.py` — red tests: determinism (same-seed
  byte-identity, seed-keyed permutation), verbatim guarantee (every quote a
  substring of its source verse), dedupe-vs-frozen (address + pair disjoint,
  pins verified), count>=2000 (all-scope) + train-scope hygiene subset.
- `experiments/t_route/synthetic_episodes/episodes.jsonl` — 2,006 all-scope
  episodes (sha256 in `manifest.json`).
- `experiments/t_route/synthetic_episodes/episodes_train.jsonl` — 1,656
  TRAIN-side hygiene subset.
- `experiments/t_route/synthetic_episodes/manifest.json` — schema
  `humanbrain/synthetic-episodes/v1`; pins, counts, checks, stats,
  `sha256_content`.

No gates were run, no probes touched, no frozen protocol executed — pure
corpus engineering per the reserve-plan mandate.
