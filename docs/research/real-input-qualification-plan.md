# Actual-language acquisition, retention and execution qualification

Status: preparation contract; one run only after source freeze. Preparation is
2026-09-05 **16:01–16:46 UTC**. The frozen source revision and admission receipt
govern execution. This plan implements the [public-QA next decision](../../experiments/r4_integrated/public_qa_acquisition_result.md#next-decision);
the earlier comparison and its failed qualification remain closed.

## Question and scope

Can the unchanged cell learner and a conventional positive control jointly fit
real supported answers and STOP, retain block A after learning block B, and use
their selected answers in actual Python edits? Neither learner previously
demonstrated adequate TRAIN fit. This concentrates explicit supervised exposure
to establish engineering trainability; it is not a fresh cohort, a generalization
comparison, an adoption decision, or a pass of the general integrated gate.
Repeated labels and paragraph-local source access are disclosed. Raw paragraphs
and their frozen pretrained MiniLM word vectors remain retained throughout.
There is no absent-source fact-memory or full-corpus retrieval claim.

## Immutable inputs and custody

Use the [prepared fixture](../../experiments/r4_integrated/real_input_qualification_cases_v1.json)
without substitutions: the first 16 entries of the published public-QA TRAIN
order, A = first eight and B = next eight. Each block has six supported questions
and two STOP lessons. Preserve all original paragraphs, questions, annotations,
accepted answer bytes and offsets, including original spelling and punctuation.
The fixture's proposed rubric is superseded by the exact rules below.

- Original selection SHA256: `72fa7ac873643be36701c486a66c24ae67631fcd95c6fe18c2021539dd3cfe76`.
- Fixture SHA256: `aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b`.
- Saved-input NPZ SHA256: `5d60b2dad367a0a756c54fdce812b68865d51597b9b8ebfb92543e1968e12f9b`.
- Original acquisition JSON SHA256: `9e0f5a0a20ca1726777b5747e86bf69a151622e3911b608c043e24af6a850708`.
- Original acquisition NPZ SHA256: `ef1abd594613cebf8b5a138efd6c8ab49d9c44817c5e47a90c31a1692993a3d6`.

The original run is `4f363a3239a4be352a12043752d75cba12203d29`; its completion
receipt binds the JSON, which binds the archive and source manifest. The extracted
`.pytest-tmp/real-input-qualified-inputs.npz` contains only 32 original input arrays:
6,021,120 vector bytes, 5,778,426 compressed bytes. Authenticate every original
array name, vector hash, TokenInput hash, raw text, word offsets and paragraph
inventory identity against the fixture. No learned weights or optimizer states
are imported. There is no encoder inference, token rewriting or answer shortening.
Keep each complete paragraph's unchanged 48-word nonoverlapping span inventory.
All accepted annotations fit exact word boundaries; answers range from one to
seven words. Predictor arguments contain query/source inputs, never labels.

Original SQuAD 2.0 TRAIN attribution and CC BY-SA 4.0 notices remain in the
fixture; [authors](https://rajpurkar.github.io/SQuAD-explorer/) and
[license](https://creativecommons.org/licenses/by-sa/4.0/). Article partitioning
and prior exposure are unchanged. Unknown pretraining overlap prevents a
contamination-free claim. No DEV/test or new official-exam content is accessed.

## Fixed learners, teaching and probes

Use six policies: full trainable query-aligned cells; the same frozen producer
with trainable heads; fully frozen cells; trainable conventional attention reader;
matched fully frozen conventional reader; and external cosine. External cosine
aliases the frozen cell object's retained inputs/state but calls its external-only
readout. It is not a separately trained learner. Both conventional policies share
the existing conventional initialization. Cell controls share the existing cell
initialization; there are no model, core, representation or initialization changes.

Keep the existing seed 0, Adam rate .001, betas .9/.999, epsilon 1e-8, active
gradient norm clip 1, and supported gate-plus-span / STOP gate-only objective.
STOP skips span-head weights, moments and counters. Frozen producers retain their
existing masks; frozen policies receive no offers. Preserve hierarchical STOP
and exact span selection; do not tune thresholds or force below-gate answers.

Train **256 ordered full passes over A, then 256 over B**, one offer per item,
without additional replay. Each of the three updating learners receives 4,096
offers; total teaching is **12,288 offers**. This openly repeats each label 256
times. Do not stop early upon fit, select exposures from errors, or add passes.
Record original IDs, targets, source/query hashes and each update receipt.

Probe all 16 items for all six policies initially. Every 16 completed passes,
probe A during A teaching and A+B during B teaching. Probes are read-only and
fixed-order: 96 initial + 768 A-stage + 1,536 B-stage = **2,400 predictions**.
There are no per-offer predictions. Preserve checkpoint identity across every
probe; record complete denominators, supported exact-byte accuracy, STOP and
conditional span diagnostics. The final 96 prediction objects are reused for
action; there is no additional motor-stage reader inference.

## Numerical admission and budget

Before teaching, run the frozen [actual-input numerical checker](../../experiments/r4_integrated/real_input_numerical_check.py)
on disposable cell and conventional instances using the first supported and first
STOP item. Their weights and optimizer state are discarded. One direction RNG
seed 17 visits cell then conventional, supported then warmed STOP, parameter order.
Check eight parameter-block directions for each of four objectives: **64 perturbed
forward evaluations**, four baseline gradients, four predictions and **four
validation offers**. Each validation offer additionally computes its existing
internal gradient and post-update forward/objective; it is not a teaching offer.
Use central step `1e-5`, finite-difference absolute/relative tolerances `1e-7/1e-4`,
and independent Adam absolute/relative tolerances `1e-10/1e-10`. Require finite
results, exact source/input/receipt hashes and warmed STOP span-state preservation.
Finite differences check BPTT against the implemented scalar objective, not an
independent implementation of the forward model. Any failed admission stops fitting.

One **20-minute total run**, including admission, teaching, probes, motor execution,
authentication and artifacts; parent peak working set must remain **at most 2 GiB**.
No separate uncounted cost/training retry or parameter search is authorized. Count
raw retained inputs, shared vectors once, all unique learner/optimizer arrays,
candidate inventories and execution resources. Record actual timing/memory scopes.
Freeze all relevant source hashes, this plan, fixture and saved-input hashes before
actual-input scoring. Outputs are immutable; preserve partial rows and the original
failure on timeout, resource, numerical or serialization failure. A partial run
cannot qualify. Publication must account for final artifact writing costs.

## Qualification and immediate action

Both full cells **and** the trainable conventional control must satisfy:

- End A: 6/6 exact supported answers and 2/2 correct STOP on A.
- End B: A and B each 6/6 supported and 2/2 STOP, with no A regression.
- End B: at least four additional supported successes out of 12 compared with
  that learner's own matched initial frozen baseline.

Report frozen-producer performance as the producer-learning diagnostic; it is
not required to fail. A successful trained-head control limits attribution to
the producer. Report all paired outcomes; this tiny exposed set gives no estimate
of general language competence. Do not substitute falling loss for exact fit.

At the same final checkpoints, execute 16 tasks for each of six policies:
**96 actual attempts plus two shared learned-motor demonstrations**. The ordinal
workspace metadata are key `qa000`, module `qa_config000`, instruction
`Configure qa000`, stale value `pending`, incrementing the ordinal through 015.
This action address is independent of question, label and model output. The
[bridge](../../experiments/r4_integrated/real_input_action_bridge.py) gives the
unchanged motor the explicit instruction and losslessly escapes the selected
UTF-8 bytes inside the existing single-quoted Python literal. It supplies neither
answer selection nor label-based routing. Expected answers enter scoring only.

Each full learner must execute all 12 exact supported values and four correct
unchanged STOP cases, with every decoy unchanged. No teacher repair counts and
no learning occurs during action. A STOP execution PASS establishes only the
unchanged-workspace contract. A fit pass with execution failure remains an action
integration failure; failure of either learner's fit or retention closes this
qualification without automatic optimization or architecture variants.
Authentic official-reading structured responses remain a separate unbound
interface obligation under the [reading plan](reading-comprehension-evaluation.md).
