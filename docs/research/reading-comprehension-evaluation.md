# Public-exam reading comprehension evaluation

Status: public development diagnostic implemented, 2026-09-05. The whole KS2
2024 paper is now developer-exposed. Its original Q6a/b text-only passage-visible
diagnostic scores 0/2 in both empty-seed arms; the remaining questions are
unevaluated. [Source manifest, official rubric, runner and measured limits](../../experiments/reading_exam_development/README.md)
record the selected passage's OGL attribution and excluded photography.
Reserved material and adjudication thresholds remain unselected.

[KS2 2026 Amelia Earhart](official-reading-next-candidate.md) now has a checked
passage/map and four original closed-response items with separate official keys.
The whole 2026 paper is developer-exposed. Source custody and transcription review
are complete; its linked candidate plan now specifies a proposed structured response
contract. Scorer implementation and genuine model binding remain unfinished.
The optional HTML aid still needs rendered-layout review. No new exam
has run.

## Purpose and sequencing

Use existing public exams authored by assessment authorities to test reading
comprehension. LLM-authored exams or rewritten questions do not qualify for this
track. This gives the project an external task distribution alongside the small
mechanism tests needed to diagnose learning failures.

The immediate development priority remains the smallest persistent-session loop:
read a short passage, retain a useful fact, use it for an actual Python edit,
execute the edit, and transfer to a fresh task. Establish that loop before adding
revision or multi-hop reasoning. Prepare public-exam evaluation in parallel;
introduce development examples when the reader can accept passages and emit
answers, then reserve evaluation use for a separately frozen protocol.

## Candidate sources

KS2 2024 supplies the first development subset. The shortlist below remains
useful for subsequent material; no frozen reserved corpus has been selected.

| Candidate | Verified availability and proposed use | Limits to resolve |
| --- | --- | --- |
| England KS2 English reading | The [official 2024 landing page](https://www.gov.uk/government/publications/key-stage-2-tests-2024-english-reading-test-materials) lists reading and answer booklets, administration instructions, and official mark schemes. This makes a small whole-paper development pilot feasible, subject to rights and input-format checks. The [past-materials collection](https://www.gov.uk/government/collections/national-curriculum-assessments-past-test-materials) is the starting point for identifying separate years. | The [official copyright report](https://www.gov.uk/government/publications/key-stage-2-tests-2024-mathematics-test-modified-materials/2024-copyright-ownership-key-stage-2-national-curriculum-tests) distinguishes reusable Crown material from third-party text and images. Public availability does not grant this project permission to redistribute or reuse every component. |
| NAEP released reading materials | The [official educator guide](https://www.nationsreportcard.gov/educators_alt.aspx) describes released questions, answers, sample responses, and scoring guides, and lists reading at grades 4, 8, and 12. Grade 4 is a candidate second source; use complete released passage groups rather than assuming a complete annual exam is available. | The [official FAQ](https://www.nationsreportcard.gov/faq.asp) gives a public-domain default for its own site with exceptions, explicitly excluding other linked sites from that statement. Verify the actual material's terms and passage rights. A selected released-item set is not the calibrated national assessment. |

Before ingestion, record permission or license coverage separately for passages,
questions, illustrations, and scoring material, including required attribution.
If rights are unresolved, keep metadata only and choose eligible material or
obtain permission. Replacing a passage to avoid a rights problem would produce
an adapted task and cannot be reported as the original exam.

## Development and reserved evaluation

1. Maintain a source manifest from official metadata. Record authority,
   level, year, release/version, source URLs, rights status, whole-paper or
   passage-group identity, intended partition, and custody owner. Register
   downloaded artifact hashes and extraction/version hashes during authorized
   preparation; the implemented KS2 manifest supplies the first example.
2. Allocate reusable public development material and separately reserved
   evaluation material before model developers inspect the reserved content.
   Hold out whole papers/years where feasible. Keep every question sharing a
   passage, and repeated or adapted versions of that passage, in one partition.
   Never randomly divide questions from a shared passage across partitions.
3. A designated preparation/scoring process should verify reserved extraction
   and official rubric mapping without exposing content or keys to development.
   Preserve instructions, options, diagrams needed to answer, and answer format;
   record exclusions and reasons before scoring. Label any partial-paper result.
4. Before a reserved run, freeze material identities, model/checkpoint, permitted
   state updates, interface, scoring rules, sample size, budget, stop criteria,
   and decision thresholds. Set thresholds using development evidence, then
   keep them fixed for reserved evaluation. Final values are not specified here.
5. Log access and every evaluation attempt. Keep reserved keys and feedback out
   of learning and selection; treat content used for tuning as development and
   require a new reserve for the next independent claim.

“Reserved” means untouched by this project's development process. Released
public exams may already have appeared in pretrained LLM data. Consequently,
hybrid/LLM scores cannot be called contamination-free merely because a local
manifest is held out. Record known exposure and unknown training-data coverage.

## Tasks, scoring, and comparisons

- **Passage-visible comprehension:** keep the passage available while answering
  the official questions. Report raw earned/available marks, omissions, invalid
  answers, coverage, and results by paper/passage and response format.
- **Delayed retention:** separately specify exposure, passage removal, delay,
  intervening tasks, and allowed memory. Reusing official questions under this
  changed condition measures an adapted retention task; do not pool its score
  with passage-visible comprehension. Keep a shared passage in one partition
  across both conditions.
- **Multiple choice:** use official keys and option identities; record invalid
  or missing selections. **Extractive responses:** apply the official accepted
  forms with predeclared normalization. **Open responses:** preserve official
  partial-credit rubrics and use blinded human scoring/adjudication where
  semantic judgment is required. An LLM judge must not replace the official
  rubric or serve as the sole unvalidated scorer.
- Introduce comparison arms only when the component being tested exists: an
  updating cell learner versus the same initialized cell learner with updates
  disabled for the relevant acquisition/retention hypothesis; a hybrid versus
  the same LLM without the cell component when hybrid contribution is tested;
  and a suitable simple baseline. Match source items, exposure, initial state,
  access, response interface, and budgets; report remaining differences.
- Keep cell-only, hybrid, and external-memory results distinct. Mem0 remains
  development context outside the learner and evaluation evidence pipeline.
  Report paired effects and uncertainty at the paper/passage-group level;
  questions sharing a passage are not independent samples. Tiny pilots support
  feasibility and diagnosis, not precise broad performance estimates.

## Interpretation and next deliverable

Exam marks describe performance on the selected reading tasks under the stated
conditions. They do not establish general intelligence, coding ability, or a
human-equivalent reading age. In particular, do not convert raw released-item
accuracy into NAEP scale scores or achievement levels: the [official FAQ](https://www.nationsreportcard.gov/faq.asp)
describes a group assessment with separately reported scale scores, rather than
an individual score from an arbitrary released subset.

The next deliverable is a reviewable source-and-custody manifest plus a scoring
and comparison protocol, with rights checks and explicit development/reserve
allocation. This document authorizes no protected-content access, experiment,
new dependency, or change to an existing frozen evaluation surface.
