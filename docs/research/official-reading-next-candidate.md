# Official reading candidate: KS2 2026 Amelia Earhart

Source custody and checked fixture preparation completed, 2026-09-05.
The [four-item fixture](../../experiments/reading_exam_development/ks2_2026_amelia/README.md)
contains corrected passage text, the Crown route map, original questions and
separate official keys. No learner ingestion, evaluation or reserve designation.
This follows the [reading plan](reading-comprehension-evaluation.md).

## Source and reuse scope

The Standards and Testing Agency released the May 2026 KS2 reading materials
on 22 May 2026. Candidate: **Amelia Earhart**, reading-booklet pages 8–9,
original questions 25–38, worth 18 marks. The [official landing page](https://www.gov.uk/government/publications/key-stage-2-tests-2026-english-reading-test-materials)
links the source documents:

- [Reading booklet, STA/26/9011/e](https://assets.publishing.service.gov.uk/media/6a0af4c9229555d0fe826722/2026_KS2_English_reading_booklet.pdf).
- [Answer booklet, STA/26/9010/e](https://assets.publishing.service.gov.uk/media/6a0af4ea229555d0fe826723/2026_KS2_English_reading_answer_booklet.pdf), printed pages 14–19.
- [Mark scheme, STA/26/9012/e](https://assets.publishing.service.gov.uk/media/6a0af5ae279ebb7d24f8f3ae/2026_KS2_English_reading_mark_schemes.pdf), printed pages 26–35.
- [Copyright report](https://assets.publishing.service.gov.uk/media/6a0c63d9f7c2e79c33db91fc/2026_copyright_ownership_key_stage_2_national_curriculum_tests.pdf).

The report identifies the passage text and final-route map as commissioned Crown
material. Its Crown default covers the questions/marking material without a
listed exception for this group. Exclude the Getty/Bettmann and Getty/Staff
photographs, cover photograph and logos. Preserve the Crown route map when
preparing the complete passage group. Include the required attribution and
[OGL v3.0 licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):

> Contains material developed by the Standards and Testing Agency for 2026 national curriculum assessments and licensed under Open Government Licence v3.0

## Prepared small diagnostic

Preserve the original passage and use a declared four-item, four-mark subset:

| Item | Original format | Official key |
| --- | --- | --- |
| Q25 | One option | Fourth option alone |
| Q27 | One option | Second option alone |
| Q30 | Three table entries | 1923, 1928, 1932 in printed row order; all for one mark |
| Q36 | Five-event ordering, first entry supplied | 5, 1, 2, 4, 3 in printed row order; one mark |

Keys were checked against official scheme text. Preserve wording, instructions,
option order, row labels and supplied entries; do not split Q30's mark. The
other fourteen marks remain unevaluated. This is a proposal, not a frozen roster.
The current scalar span interface cannot emit all these response
formats. The contract below makes the missing interface explicit;
the scorer is now implemented, while model binding remains pending before evaluation.

## Implemented machine response contract

The [pure scorer](../../experiments/reading_exam_development/ks2_2026_scoring.py)
provides `learner_request`, `validate_response` and evaluator-only `score_response`.
Its 45 mechanical tests pass, including strict serialization, indivisible marks,
source hashes, and a check that request construction/validation never read the
rubric. The source-bound receipt is `.pytest-tmp/ks2-2026-scoring-qa.json`
(`7e4686307983583c48f92a41addca43574aed6d67ad8dec6c7632c4cee6d7a30`).
This is scorer QA, not a learner result or an exam attempt. Request construction
includes the original map PNG; the future producer must actually consume it or
declare a text-only condition. Key separation is a data-routing convention, not
a filesystem sandbox.

Present the unchanged passage/structure, route map and one original item at a
time, in order 25, 27, 30, 36. Rubric bytes and previous marks stay outside the
learner. A model unable to consume the map requires a declared text-only condition,
not a claim of complete original passage presentation.

One final UTF-8 JSON response per item has `item` (the original string ID),
`status` (`answer` or `stop`),
and `response` (an object, or null for STOP). Generic format instructions contain
no worked answers from this paper. Printed-order positions address the original
options/rows without rewriting them:

| Item | Response object |
| --- | --- |
| Q25, Q27 | `selected_options`: exactly one integer index, 1–4. |
| Q30 | `years`: three integers or null blanks, in printed row order. |
| Q36 | `ranks`: four integers or null blanks for printed rows 1, 3, 4, 5. |

For Q36, the evaluator inserts the supplied `1` at row 2 from the **question**.
Complete writable ranks must permute 2–5. Score the reconstructed response against
the separate official rubric; Q30 and Q36 remain one mark each, with no partial
credit. Never infer a missing value, sort answers, or extract digits from prose.
Malformed JSON, duplicate fields, extra selections, wrong lengths/types/IDs and
invalid ranks receive zero with a format reason. STOP, blank and missing/late
responses also receive zero, recorded separately from valid wrong answers.

This requires genuine structured answer production. Host option matching,
year extraction or event ordering would add a solver, not merely format a span.
The existing 2024 scalar scorer and its learner ingestion cannot supply this seam;
preserve that experiment. The pure scorer now exists; a genuine structured
producer binding remains necessary. Until then, report the unsupported interface
before evaluation rather than fabricate four comprehension failures. No model
execution has been introduced by this implementation.

## Remaining preparation and exposure

The four exact official URLs above returned HTTP 200 (`application/pdf`) on
2026-09-05 at 08:55 UTC, without URL redirection. Local originals, raw extraction,
21 individual page renders and 11 contact sheets remain under ignored
`.pytest-tmp/official-reading-2026-source-custody/`; none of the photographs or
PDFs is published in the repository. `manifest.json` records final URLs, byte
counts, PDF and raw-extraction hashes, and all render hashes.

| Original PDF | Bytes | Pages | SHA256 |
| --- | ---: | ---: | --- |
| Reading booklet | 10,307,852 | 12 | `98fc6c64f67d08b0a17e333ba4a379f4acc281defd873a4576bdfdf2ae5c8871` |
| Answer booklet | 428,093 | 20 | `6c179aa107f4d9fca1c87109c4da02813d152cb97b80dd39dbedd44f1a43576c` |
| Mark scheme | 852,782 | 36 | `e309d94fa6e30c089ee4ba9462e0936b493993b4cf957dc0cd317aeb76c8cd8a` |
| Copyright report | 161,477 | 3 | `70c9edcc8177cad38c4aee94bdec3c754f8a1bd665a8daaeb9f6331a7d899309` |

Poppler 26.07.0 rendered at 110 DPI. Visual review covered reading pages 8-9,
answer pages 14-19, mark pages 26-35 and all three copyright pages; PDF page
indices (one-based) match the printed page numbers for the reading, answer and
mark pages. The passage headings, paragraphs and final-route map are legible.
The answer layouts and four selected keys above agree with the visible scheme:
Q25/Q27 on scheme pages 26/27, Q30 on page 28, and Q36 on page 32. Q36's supplied
`1` is in its second printed row. Copyright pages 2-3 confirm commissioned Crown
text/map and the two Getty photograph exceptions; the cover photograph and logos
remain excluded as stated above.

The raw pypdf layout extraction remains unchanged and is not the learner fixture:
it contains replacement characters, excessive spacing and hidden duplicated
question content. The [prepared manifest](../../experiments/reading_exam_development/ks2_2026_amelia/manifest.json)
binds the corrected transcription, source structure, original four questions,
separate rubric, unchanged-pixel map crop and source custody. Transcriber and
independent orchestrator comparisons cover all eight passage paragraphs and the
selected question/marking pages. Corrections and checks are in the linked fixture.

The optional HTML review aid now passes a
[desktop browser review](../../experiments/reading_exam_development/ks2_2026_visual_review.json)
at 1234×712: passage text, map labels, all four response layouts, Q36's supplied
rank and attribution are readable without observed clipping. No layout change
was needed. This review does not certify mobile or print layouts and does not
replace the original page comparisons. The original extraction files remain
unchanged. The machine contract above still requires an evaluation-specific
source/interface freeze and model binding before evaluation. No answer policy,
score or learner exposure was introduced by preparation.

The developer inspected full 2026 reading/answer text and relevant marking
sections. Treat the **whole KS2 2026 paper as developer-exposed**, including
sibling passages. Earlier 2025 rights/landing metadata and 2023 rights search
metadata were also inspected without learner ingestion. The 2025 nonfiction
candidate was declined because its passage is third-party Guardian material.
Public release does not establish absence from pretrained-model data. Any eventual
subset result is a partial public-development diagnostic, not independent broad
comprehension evidence or a reserved assessment.
