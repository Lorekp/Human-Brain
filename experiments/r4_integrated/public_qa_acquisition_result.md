# Broader public-QA acquisition: completed without adoption

2026-09-05. Frozen source: `4f363a3239a4be352a12043752d75cba12203d29`.
The [protocol](../../docs/research/public-qa-acquisition-plan.md) remains unchanged.
One scored acquisition completed; acquisition and motor-readiness criteria fail.
All saved roster, update-lineage, scoring, execution and resource audits pass.

## What was tested

The unchanged query-aligned rate cells received 1,024 human-written SQuAD 2.0
TRAIN lessons (768 supported, 256 unsupported), with paragraph-local candidate
competition and the pinned MiniLM encoder. Each updating policy received 2,047
current/previous offers. All five policies made 5,120 first attempts before the
corresponding updates and 1,760 fixed early/final TRAIN and article-disjoint
probe predictions. The same final cores then entered 640 real Python attempts,
with updates disabled, over all 16 original motor sources and both wordings.
Two shared executed demonstrations taught the existing literal-edit motor.

This is supervised, passage-visible hybrid development. It changes both the
teaching material and paragraph competition relative to the previous narrow
run; it does not isolate dataset size. It establishes neither answer-free
internalization nor source-absent semantic memory. The encoder's external prior
and unknown pretraining contamination remain part of every policy. No official
exam, protected evaluation, new motor teacher or additional learning run occurred.

## Outcomes

Each QA cell gives **exact supported answers / correct unsupported STOPs**.

| Policy | Early TRAIN /24, /8 | Final TRAIN /48, /16 | Article-disjoint probe /192, /64 |
| --- | ---: | ---: | ---: |
| Trained cells | 0 / 4 | 0 / 16 | 0 / 63 |
| Frozen producer, trained heads | 0 / 0 | 0 / 0 | 2 / 0 |
| Fully frozen cells/heads | 0 / 1 | 0 / 4 | 0 / 12 |
| Conventional attention/feedforward | 0 / 8 | 3 / 3 | 8 / 24 |
| Frozen sensory cosine | 0 / 0 | 0 / 0 | 0 / 0 |

The trained cells miss the public 96/192 supported floor, the required supported
advantage over both cell controls, final TRAIN fit and the conventional comparison.
The stability check passes only because early supported acquisition was already
zero; that is not evidence of retained useful learning. Paired public supported
outcomes contain no cell gains, two losses against the frozen producer and eight
against the conventional learner. Chosen margins are not significance estimates.

| Policy | Actual Python /48 supported, /16 STOP | Alternate Python /48, /16 |
| --- | ---: | ---: |
| Trained cells | 0 / 16 | 0 / 16 |
| Frozen producer, trained heads | 0 / 0 | 0 / 0 |
| Fully frozen cells/heads | 0 / 7 | 0 / 8 |
| Conventional attention/feedforward | 1 / 0 | 0 / 1 |
| Frozen sensory cosine | 0 / 0 | 0 / 0 |

Trained cells STOP on 127/128 motor requests. Their only emitted answer is a
wrong long span on alternate `urevi` transport; its execution fails. Every
decoy remains unchanged. Neither the 44/48 supported and 15/16 STOP readiness
criterion in both wordings nor the broader integrated goal passes.

## What the failure distinguishes

First-attempt cell supported successes by 256-encounter quartile are **1, 0, 0, 0**
(192 supported questions each); correct STOPs rise **11, 41, 59, 64** (64 each).
The final public cell policy stops on 189/192 supported questions. Only two
have a uniquely winning annotated answer before the gate; final TRAIN has zero
such winners. The result shows failed acquisition/ranking alongside increasing
abstention. It does not demonstrate earlier acquisition followed by forgetting.

Annotated-occurrence diagnostics are narrower than primary accepted-text scoring.
A saved-data check therefore allows every additional accepted-text occurrence
to be a hypothetical winner. Even this generous gate-only ceiling is **4/48**
on final TRAIN and **21/192** on public supported probes, below their 36/48 and
96/192 floors. No logits or thresholds were rerun. A gate adjustment alone
cannot meet these criteria while retaining the existing ranking/tie rule.

The conventional learner's 3/48 final TRAIN answers do not establish an adequate
positive control. These results do not identify representation insufficiency,
insufficient optimization, destructive interference, or architectural impossibility
as the unique cause. Declining per-offer loss and changing checkpoints are not
substitutes for useful reading. Stop this comparison without automatically adding
epochs, labels, a new encoder or another initialization variant.

## Resources and startup recovery

The completed synthetic preflight projected **570.531 seconds** for acquisition.
Actual acquisition took **202.043 seconds**, including primary JSON/NPZ output
and final authentication; the completion receipt's own small write and Python
bootstrap are excluded. Peak parent working set was **922.336 MiB**.

- Retained normalized word vectors: **472,108,032 bytes** over 2,665 inputs.
- Counted Python/NumPy core, cache and motor payload: **494,669,085 bytes**.
- Public source/label object graph, separately: **23,603,542 bytes**.
- Checkpoint/input archive: **460,360,052 bytes**, retained locally in the ignored
  `public_qa_acquisition_v1.npz`; its hash is in the primary report. This archive
  is not distributed through Git. Retained-object counts and process RSS measure
  different scopes; external encoder state, allocator and temporary work also cost memory.

The [first launch](../../benchmarks/public_qa_bootstrap_failure_v1.json) failed
during native imports before encoder construction, cost sampling or scored work.
Cargo reported Windows disk error 112 despite producing a DLL; a subsequent
observation found about 5.6 GB free. An isolated unchanged native build succeeded
in 11.667 seconds without cleanup. The initial cause remains unconfirmed.

One startup retry was explicitly declared before outputs. An outer controller
kept a conservative **13:49–15:49 UTC** wall-clock budget, beginning before the
failed launch; its **1,066.752 seconds** includes failure, diagnosis, waiting,
retry and acquisition. The original failure remains failed. There was one
completed cost sample and one scored acquisition. Both native completion receipts
and outer launch receipts must pass; their statuses concern execution/custody,
not capability. The original runner did not capture import-time failure in its
report, so the separate launch record preserves the observed failure excerpt.

The native cache key includes the complete build environment. Ephemeral PATH
entries can change it, but the previous environment was not retained sufficiently
to prove the cause of this cache miss. The successful build plan is preserved
in the audit. Do not manually reuse binaries under another attestation key.

## Evidence and validation limits

- [Preparation](../../benchmarks/public_qa_preparation_v1.json): exact data/license
  custody, full exclusions, inert integration checks and conventional qualification.
  The original failed initialization and separate diagnostic remain preserved;
  the corrected variant passed the unchanged toy and six focused checks.
- [Preflight](../../benchmarks/public_qa_preflight_v1.json) and its
  [completion](../../benchmarks/public_qa_preflight_v1.completion.json) and
  [outer launch](../../benchmarks/public_qa_preflight_v1.launch.json) receipts.
- [Acquisition](../../benchmarks/public_qa_acquisition_v1.json), SHA256
  `9e0f5a0a20ca1726777b5747e86bf69a151622e3911b608c043e24af6a850708`,
  with [completion](../../benchmarks/public_qa_acquisition_v1.completion.json)
  and [outer launch](../../benchmarks/public_qa_acquisition_v1.launch.json).
- [Saved audits](../../benchmarks/public_qa_acquisition_v1.audit.json): exact
  rosters, 6,141 offer/checkpoint/Adam-counter links and masks, original byte scoring,
  unchanged frozen states, 8,162 motor candidate spans, 642 unique workspaces and
  1,926 files, literal edits, stdout and decoy consistency. Checkers are embedded.
  These checks do not independently reconstruct actual-data gradients or logits.

## Next decision

Return to the persistent read-to-action objective through one real-input learning
qualification. Before another broad comparison, require both the cell learner
and a conventional positive control to learn supported answers and STOP jointly
on an ID-selected subset of already exposed public TRAIN, then measure retention
after later lessons. Preserve full paragraphs and legal candidate competition.
Concentrated/repeated teaching must report every label exposure and remains
engineering qualification rather than new generalization evidence.

Use at most **45 minutes of preparation and one 20-minute diagnostic run** to
freeze the subset, one teaching configuration, actual-input numerical checks,
fit/retention criteria and the return to actual Python execution. The same
checkpoint must supply the public answers used by the literal-edit motor, with
real execution and decoy checks; any instructed action routing or byte escaping
must be declared as actuation scaffolding. Paragraph-local qualification leaves
full-corpus retrieval and domain transfer unresolved. If fit remains absent,
stop broad comparisons and identify the failing computation or optimization
condition rather than extending the budget automatically.

The [official structured-response interface](../../docs/research/official-reading-next-candidate.md)
remains pending; it supplies complementary examination pressure once a genuine
producer can bind to it. This qualification does not open protected evaluations.
