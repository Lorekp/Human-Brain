# Token-pointer acquisition and executed continuation

Completed 2026-09-05 at frozen source
`9d73bba3908583add036d88c0cc04a2d23b0f368`. The run completed mechanically;
**acquisition, integration and joint qualification all fail**. No architecture
is adopted and no roadmap gate passes. The
[contract](../../docs/research/token-pointer-acquisition.md),
[admission](../../benchmarks/token_pointer_acquisition_v1.admission.json),
[full report](../../benchmarks/token_pointer_acquisition_v1.json) and
[completion receipt](../../benchmarks/token_pointer_acquisition_v1.completion.json)
preserve the prospective criteria and complete evidence.

## Reading acquisition

Four fresh policies received the same 8,192 SQuAD2 TRAIN questions, eight epochs,
2,048 batch updates and 65,536 presentations each. MiniLM supplied frozen external
word features. The 256 untaught questions are article-disjoint and fresh to these
learners, but remain public development with unknown external pretraining exposure.
The fitted sixteen-case preflight did not supply learned state.

| Policy | Initial untaught supported / STOP | Final untaught supported / STOP | Final TRAIN supported / STOP | Final untaught balanced accuracy |
| --- | --- | --- | --- | --- |
| Rate cells | 0/192 · 3/64 | 18/192 · 26/64 | 24/96 · 25/32 | 25.00% |
| Frozen producer, trained heads | 0/192 · 3/64 | 0/192 · 64/64 | 0/96 · 32/32 | 50.00% |
| Trained recurrent-core bypass | 1/192 · 0/64 | 13/192 · 43/64 | 22/96 · 24/32 | 36.98% |
| Conventional BiLSTM | 0/192 · 1/64 | 23/192 · 28/64 | 53/96 · 28/32 | 27.86% |

Cells learn some untaught supported answers, but their gain is only 18 against
initial cells and the frozen producer, and five against the bypass; each required
at least 20. Their balanced accuracy is 25 percentage points below the always-STOP
frozen producer and 11.98 points below the bypass. The required STOP floor is
48/64; cells reach 26/64. This is not useful recurrent-core superiority.

BiLSTM meets its supported-gain and TRAIN-fit requirements, but reaches only
28/64 STOPs. The positive control is therefore unqualified too. Small-set fitting
worked; broad supported acquisition and answerability remain weak. Declining
training loss or isolated correct examples cannot replace the failed criteria.

TRAIN supported scores after epochs 1→4→8 are cells 3→14→24, bypass 8→17→22,
BiLSTM 4→28→53, and frozen producer 0→0→0, each out of 96. Cells' final untaught
supported errors comprise 129 emitted wrong spans and 45 abstentions. Changing
abstention alone cannot repair those 129 extraction errors.

## Executed continuation

The unchanged 28-question fixture is an **exposed regression**, with two ambiguous
items diagnostic-only. Each trained policy has an exact frozen continuation mate.
All nine policies make their first attempts before feedback; each updater then
receives seven episode lessons and sixteen old replays. Complete paragraphs remain
available, so this is neither answer-free acquisition nor source-absent recall.

| Policy | Initial old supported / STOP | Final old supported / STOP | Episode + late supported / STOP |
| --- | --- | --- | --- |
| Cells | 5/12 · 3/4 | 7/12 · 3/4 | 1/8 · 0/2 |
| Frozen cells | 5/12 · 3/4 | 5/12 · 3/4 | 1/8 · 1/2 |
| Frozen producer | 0/12 · 4/4 | 0/12 · 4/4 | 0/8 · 2/2 |
| Frozen producer mate | 0/12 · 4/4 | 0/12 · 4/4 | 0/8 · 2/2 |
| Bypass | 1/12 · 1/4 | 2/12 · 3/4 | 1/8 · 1/2 |
| Frozen bypass | 1/12 · 1/4 | 1/12 · 1/4 | 1/8 · 1/2 |
| BiLSTM | 6/12 · 0/4 | 9/12 · 4/4 | 1/8 · 1/2 |
| Frozen BiLSTM | 6/12 · 0/4 | 6/12 · 0/4 | 1/8 · 1/2 |
| External sensory cosine | 0/12 · 0/4 | 0/12 · 0/4 | 0/8 · 0/2 |

Cells lose no initially correct old item in either pooled or correct-paragraph
probes. Their two additional old supported successes are useful regression gains.
However, cells and their frozen mate both solve only one of eight episode/late
supported tasks, and the updating cells lose a correct fresh STOP. No policy
solves either of the two scorable late supported tasks. The continuation does not
demonstrate the required benefit on later fresh actions or autonomous recovery.

All eight supported fresh cell first attempts choose the correct original source;
seven choose a wrong span. Correct-paragraph restriction also yields only 1/8.
This run's remaining fresh failure is therefore within-source reading, not chiefly
document routing. Final cell old correct-paragraph support improves 5/12→9/12;
the six taught episode answers improve to 3/6 against the frozen mate's 1/6.
These are gains after explicit lessons with retained sources, separate from
first-attempt fresh performance.

All 468 primary actions, 468 correct-paragraph probes and 92 continuation offers
complete. Twenty-four external teachers and two motor demonstrations bring the
executed child count to 494. Teachers repair the same failed workspaces and count
as external supervision, never learner successes. Decoys remain unchanged.

## Cost and verification

The completion receipt records **3,637.303 seconds** and **2,234.484 MiB peak parent
working set**, within the four-hour / 4-GiB limits. Integration takes 157.446 seconds.
Timed broad updates total 1,165.677 seconds for cells, 306.482 for the frozen
producer, 323.477 for bypass and 1,298.527 for BiLSTM. These are measured update
costs; encoding, probes, actions, journals and serialization are additional work.

The indexed cache adds 1,079,773,324 disk bytes and reuses 2,058 authenticated old
inputs; 12,872 new inputs are encoded. The complete input dependency files retain
another 642,295,325 bytes, including unused parts of the old archive. Twelve saved
policy states occupy 23,478,987 bytes. Run artifacts before the report total
34,421,094 bytes; the report and completion add 5,508,940 and 443 bytes. Environment,
package, native-library and source-checkout costs are separate from these scopes.

The saved-only audit passes on its first authorized execution. It independently
checks original annotations and UTF-8 intervals, 8,192 update receipts and 262,144
presentations, all 3,584 local predictions, all action/oracle scores, teacher
repairs, 1,410 current workspace files, source/input/cache identities, twelve
checkpoint hashes, frozen lineage, qualification outcomes and every integration
flag. It does not
replay gradients, model forward passes or child executions, nor re-encode vectors.

| Artifact | SHA256 |
| --- | --- |
| Full report | `d4080dde3a8b7326d02d96a5255337ca99de972ad06594b37303cecd56d9a82d` |
| Completion | `686c15d6c984205d26a0b9f39396ed61ee087894d1bb09719e7b34237a194b1e` |
| Checkpoint archive | `ca76f5552528c30c40047ae75257d36b46a7acaeffc1a5ca04181acdc7e14ff2` |
| Saved-only audit, `.pytest-tmp/token-pointer-full-saved-audit-v1.json` | `d783769885b28f21f8a3a8445ed5ba0f0ce3f4eb6cf6021885bc503fa7f71867` |

This closes the prescribed comparison. Interpret acquisition, answerability and
continuation separately before selecting the next bounded change. Any further
fit needs a new evidence-based decision and declared budget. The persistent
source-absent loop and genuine official-exam response production remain the
integration targets.
