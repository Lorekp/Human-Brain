# Decision review after pooled-source reading failure

Completed: the prescribed primary-source review selected
[token-pointer acquisition](token-pointer-acquisition.md) for implementation and
cost/correctness preparation. The rationale below is historical; it does not
start another review or reopen the failed session.

2026-09-05. This decision review selects the next research activity; it adds no
model run. The [completed session](../../experiments/r4_integrated/fresh_source_result.md), frozen at
`6c2287afff09a1eb5b2265375ee584ea7c3e07a7`, closes without qualification.
Its saved source/lineage, semantic scoring and action audits pass. The original
cohort abort remains unchanged; E3/L3 are diagnostic-only under the separate
[successor contract](fresh-source-action-plan.md).

## Two failures, with different evidence

Before new feedback, cells answer none of the twelve old supported questions
from the pool: every emission comes from a wrong source, and annotated gold
ranks range from 89 to 3,182. The same checkpoints answer all twelve when given
their original paragraphs. This demonstrates a failure under source competition,
not loss of those twelve within-paragraph answers at the initial checkpoint.
It does not distinguish source-wide score offsets from other cross-source
ranking errors. A STOP threshold adjustment cannot repair those wrong leaders.

The external cosine policy selects an occurrence in the original source on every
supported initial, fresh and final row, but its answer spans are wrong. That is
useful source-selection evidence for this fixed pool, not a tested document
retriever or proof that a new document-level aggregation rule will behave alike.
Giving the cell reader the correct source could recover old-task opportunity;
the saved oracle condition already measures that opportunity directly.

Fresh language is a separate blocker. All policies answer 0/8 fresh supported
questions even in the oracle condition. Final taught-episode answers are 0/6,
also with the correct source. Each updating arm receives only seven fresh offers
and sixteen old replays; this does not establish that repeated teaching could
never acquire those cases. It establishes no useful rapid acquisition here.
Cell old-oracle answers fall from 12/12 to 4/12 after these updates, while the
unchanged parent stays at 12/12. Conventional pooled old answers improve 5 to 9,
with all twelve old oracle answers retained. Neither policy gains late transfer.

## Options and their limits

| Next direction | What it could resolve | Why it should not launch automatically |
| --- | --- | --- |
| More varied acquisition with the same cells | Whether substantial practice over varied language can produce reusable answers rather than fit individual questions. | The [1,024-lesson run](../../experiments/r4_integrated/public_qa_acquisition_result.md) gave only 2,047 offers and weak final TRAIN fit; the [sixteen-case qualification](../../experiments/r4_integrated/real_input_qualification_result.md) used 4,096 offers and achieved local acquisition. These differ in data, exposure and task competition. They neither prove adequate training nor justify blindly multiplying epochs. |
| Separate source addressing from conditional extraction | Removes one demonstrated competition burden and permits separately measured source and span errors. | Correct-source fresh answers are still zero. A router-only success on the old sixteen would conceal the primary missing acquisition capability. Any cosine reference must be evaluated at its actual aggregation rule and counted as an external component. |
| A better-supported language-learning representation, core or hybrid recipe | Could change the sample-efficiency and transfer regime instead of repeatedly fitting small exposed rosters. | No replacement is yet justified. The [previous frozen reader](../../experiments/r4_integrated/frozen_reader_qualification_result.md) also failed source-known qualification. An external solver plus incidental cell routing would not establish the intended neuronal contribution. |

## Recommendation: one acquisition-design review before another fit

Do not spend the next run on restoring the old sixteen with a router, or on an
epoch extension over the twelve failed fresh questions. Use one **45-minute
primary-source acquisition review**, with no downloads, model calls or fitting,
to choose one credible route from local fit to reusable reading. Its concrete
question is: what learning recipe can acquire varied original questions with
retention at an affordable exposure budget, and what computation must the cells
learn for that recipe to contribute beyond its conventional resources?

The review should compare the existing recipe with one established relevant
training approach and one permitted pretrained-language-resource alternative.
Read the actual task/objective, initialization, training exposure and reported
learning evidence, not only headline accuracy. A raw/base LLM is permitted as
a candidate knowledge resource; permission is not qualification or a decision
to install it. State whether it supplies features, answers or reasoning, and
what remains causally learned and controlled by simulated neurons.

Completion is one recommended, implementable acquisition comparison with an
explicit data/exposure estimate, measured-cost basis and a positive conventional
learning reference. It must keep original multiword/unsupported questions and
competing sources visible, pair producer-learning and no-learning controls,
measure both taught acquisition and untaught questions, and finish in the same
literal-edit/executed-action path with old retention. Specify the single run's
budget and useful behavior criteria before outputs; do not invent accuracy
floors from unrelated cohorts. If no feasible recipe can be supported in the
review budget, report that limit and the precise missing evidence without fitting.

Reject a proposal whose gain is confined to selecting the old known paragraph,
whose external component already chooses the answer/action and leaves no useful
learned cell contribution against strong matched controls, or whose only support
is reduced loss or more offers on the same few questions. A conventional learner
that also cannot acquire the chosen development workload is a failed positive
control, not evidence that the cell architecture is uniquely responsible.

## Why not another saved-state diagnostic first?

The recorded oracle and source outcomes already separate old source competition
from fresh source-known failure. Saved ranks/hashes do not contain full source
logit distributions; inspecting recomputed logits would be a new declared
diagnostic, not a reanalysis of retained scores. Even a demonstrated offset would
not distinguish inadequate language practice from inadequate representation.
That extra run is presently less decisive than specifying a credible acquisition
recipe. Preserve the option only if the review identifies a concrete decision
that depends on it; do not make it an automatic intermediate gate.

Authentic [Amelia reading](official-reading-next-candidate.md) remains useful
external pressure, but its structured response producer is unresolved. A span
router cannot supply tables, options and orderings through evaluator conversion.
Keep that gap visible without starting a second producer queue. Passage-visible
reading, source-absent knowledge and broad coding competence remain distinct
unmet goals; this review advances only the next shared reading-to-action decision.
