# KS2 2026 Amelia Earhart: four-item preparation draft

**Reviewable extraction only, 2026-09-05. No learner ingestion, scoring or model
integration has occurred.** This directory preserves the Crown passage text and
route map plus original Q25, Q27, Q30 and Q36 (four available marks). The other
fourteen marks in the passage group remain unevaluated. The whole 2026 paper is
already developer-exposed and is not reserved or independent transfer material.

Open `review.html` for the source-ordered visual review. `passage.txt` contains
the title, headings, eight paragraphs, two original photograph captions and map
caption, with line wrapping joined. `passage_structure.json` records page/block
boundaries and original emphasis. `route-map.png` is an unchanged-pixel crop of
the official page render, including its caption and printed page number.
The two Getty photographs, cover photograph, logos, barcodes and registration
marks are excluded. Their original text captions remain clearly identified;
they must not be mistaken for included photographs.

`questions.json` preserves original numbering, wording, instructions, option/row
order, marks and response formats. In Q36, the `1` in the second row is supplied
by the original paper and is learner-visible. Q30 remains one three-row question
worth one mark; Q36 remains one ordering question worth one mark. No official
question has been rewritten or split into new marked questions.

`rubric.json` contains the official keys separately. It is evaluator-only material;
the review page does not load or embed it. This separation is a preparation
convention, not an implemented access boundary. Before any run, freeze a learner
response interface and scorer that preserve the original closed-response formats,
including handling missing/invalid selections. The current scalar span-output
interface is insufficient. Do not insert keys into prompts or turn official
answers into new training examples under this preparation authorization.

`manifest.json` records official source URLs, PDF/raw-extraction/render hashes,
rights, exposure and final extraction hashes. `extraction-review.json` records
line-by-line visual verification, transcription corrections, map crop custody
and static checks. The original PDFs and full-page renders remain in ignored
scratch; they are not redistributed here. New layout is a review aid, not a
facsimile of the original visual exam administration.

The transcriber and an independent agent compared the passage, four questions
and keys against the official page renders and found no discrepancy. The map
crop was also visually checked and verified pixel-exact. The static HTML structure
is checked, but browser-rendered layout review remains pending because no browser
runtime was available. The one-off preparation generator remains in ignored
scratch; its path and hash are recorded in the manifest, not shipped as runtime.

Contains material developed by the Standards and Testing Agency for 2026 national curriculum assessments and licensed under Open Government Licence v3.0

[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
· [Official test materials](https://www.gov.uk/government/publications/key-stage-2-tests-2026-english-reading-test-materials)
· [Project reading plan](../../../docs/research/reading-comprehension-evaluation.md)

No result on this tiny subset would establish broad comprehension, intelligence,
coding ability, or absence from pretrained-model training data. Passage-visible
comprehension and a later delayed-retention adaptation must be reported separately.
