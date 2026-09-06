# Direct neuronal-context use v1

Status: prepared 2026-09-06; full diagnostic unrun.

The completed local memory/byte v5 fit emits the same phrase across final
questions, including with writes disabled. Retained matrices differ, so that
result does not identify whether the encoded facts are inadequate or the decoder
fails to learn context use. This single diagnostic supplies explicitly informative
neuronal rates to the existing local decoder. It changes no learning rule.

## Fixed workload and comparisons

Use CPU float32, experiment seed 733, context size 96, embedding size 32, hidden size 64,
local learning enabled, rate 0.01. Keep fixed embeddings, fixed random neuronal
feedback, cell self-path eligibility and local readout updates as implemented.
The decoder seed is 735, preserving the joined v5 learner's `seed + 2` constructor
rule and exact initial decoder tensors. This is the same starting decoder, not a
new seed trial; include the joined learner source as seed-rule provenance.
No BPTT, optimizer, new feedback signal, pretrained model, document input, memory
writer or motor enters this diagnostic. Hidden activity/traces reset each lesson;
slow weights persist. Teacher forcing supplies each previous corrected byte.

Two four-neuron populations encode the two positions of a pair from `a,b,c,d`.
Each population has one authored rate 1 and three rates 0; the remaining 88
context rates are zero. The target is the two literal bytes followed by EOS.
This encoding directly supplies the factors needed for the answer. It is
supervised context-use QA, not learned representation discovery or reading.

Reserve `ab,bc,cd,da` as four held-out combinations. The twelve remaining pairs
occur in lexical order once in each of 32 cycles: 384 lessons per arm. Every
symbol occurs three times in each position per cycle. No held-out context-target
pair enters training. Four separately initialized decoders must have identical
complete initial state dictionaries, including feedback and fixed embeddings:

| Arm | Training context | Teaching | Final context |
| --- | --- | --- | --- |
| correct | Authored target pair | Local updates | Authored probe pair |
| zero | All zeros | Local updates | All zeros |
| shuffled | Permuted TRAIN context | Local updates | Authored probe pair |
| frozen | Authored target pair | Labels offered, updates disabled | Authored probe pair |

A private Python `random.Random(734)` draws one fresh permutation of the twelve
TRAIN contexts per cycle, independently of targets, decoder state and previous
scores. Chance correct assignments are allowed and recorded. The complete
context mapping is serialized before model allocation, first attempts and target
corrections. The shuffled arm receives the same context for its first attempt
and correction; it never receives held-out combinations during training.

Each lesson makes one greedy first attempt before offering two teacher bytes and
EOS, including to the frozen decoder. Generation has a two-byte cap and one
additional EOS decision; no EOS, byte or output repair is supplied. Interleave
arms in the fixed order correct, zero, shuffled, frozen for every roster index.
Run all 384 lessons regardless of accuracy, then one read-only final generation
for each of twelve TRAIN and four held-out pairs per arm. No intermediate final
probes, extra practice, retries, model-autocorrection or adaptive stopping.

Counts are 1,536 first attempts, 64 final generations, 4,608 teacher byte/EOS
offers including frozen, and 3,456 enabled local teacher steps. Actual changed
steps and generation decisions are separate measured counts. Record all losses
and first outputs; summarize all, first 128 and last 128 lessons per arm.

## Decision, budget and return

A positive diagnostic requires the correct-context arm to generate exact bytes
with EOS on 12/12 trained and 4/4 held-out pairs, and strictly exceed each other
arm's held-out exact count. Shuffled-trained decoders receive correct final
contexts to test any acquired conditional mapping. Report every arm's exact
outputs, scores, EOS/limit behavior and loss regardless of this criterion.

Allow one full diagnostic, at most 120 seconds total and 2,048 MiB through the
existing monitored launcher. All import, hashing, snapshot, label, generation and
output costs count. Use performance-counter durations with recorded clock
metadata; Unix deadlines remain separate. No timing extrapolation or cost phase
is required for this small fixed diagnostic. Failure or budget exhaustion closes
it without another seed, rate, exposure schedule or threshold change.

A pass establishes that this decoder can learn to exploit supplied informative
rates under this exposure, not prose learning, source-absent memory, software
ability or a gate pass. A failure bounds this particular local decoder/setup;
it does not establish that all local learning cannot use context. Changing only
upstream context feedback cannot affect this fixed-input control. Return to the
raw-document, source-absent actual-action workload after interpreting this result;
do not create an automatic feedback-variant queue.

## Frozen source and saved evidence

Before launch commit reviewed sources and bind `--revision` and `--sources-sha`
to `digest(sources())` from the runner. The manifest includes the runner, recipe,
decoder, shared cell/finite predicate, package initializer and monitored launcher.
No architecture import from the integrated/native runner is needed. The output
directory and launcher artifacts are exclusive; preserve failures and do not
resume or overwrite them. Source identities are verified before and after work.

`schedule.json` stores all context vectors, target pairs and per-cycle mappings.
`admission.json` binds its hash, config, source hashes, revision, process identity,
clock and Torch/NumPy versions. Memory is measured by the existing external
launcher; the decoder runner requires no native or additional monitoring library.
`initial.pt` and `final.pt` contain all four
complete decoder states. `lessons.jsonl` stores context, target, first bytes/EOS,
per-byte teacher losses/offers, enabled/changed steps and state hashes before
generation, after generation and after teaching. `report.json` binds those
artifacts and all 64 exact final outputs with read-only state hashes; completion
binds its hash. The runner checks initial equality, frozen equality and final
immutability. A saved-only independent audit must verify counts, roster/shuffles,
state-chain hashes, tensor equality, scoring and monitored launch provenance.

```powershell
$contextOutput = '.pytest-tmp/local-context-use-v1-run'
$contextRevision = (git rev-parse HEAD).Trim()
$contextSources = (& ./.venv/Scripts/python.exe -c 'from experiments.r4_integrated.local_context_use_v1 import digest,sources; print(digest(sources()))').Trim()
./experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1 -Seconds 120 -MaxMiB 2048 -LogBase $contextOutput -ChildArguments @('-X','utf8','-m','experiments.r4_integrated.local_context_use_v1','--output',$contextOutput,'--revision',$contextRevision,'--sources-sha',$contextSources)
```

The hash-only preparation imports the decoder but constructs no model and makes
no prediction. The one full launch remains conditional on review and source
freeze. Public interface tests use disposable minimal lessons, never the full
diagnostic fit; record their separate exposure.

## Pre-freeze implementation QA

Doctor receipt: `.pytest-tmp/dev/doctor-sqxjvtuq/receipt.json`. The existing
decoder suite passes 67 checks in 6.12 seconds, recorded in
`.pytest-tmp/local-context-use-v1-qa2.xml`; Ruff, syntax and diff checks pass.
The four new cases verify held-out exclusion and balanced factors, matched
initialization and offer accounting, first-attempt ordering/read-only generation,
and source rejection before model allocation. Their passing QA uses five
discarded lessons (five first attempts, fifteen offers, twelve enabled steps)
and one additional read-only generation, not a diagnostic fit.

The preserved initial receipt `.pytest-tmp/local-context-use-v1-qa.xml` records
an unavailable optional monitoring dependency and denied system-temp fixtures.
The runner now relies on the existing outer memory watchdog, and QA uses a
unique checkout-local temporary directory; no dependency was installed.
An independent initialization-only comparison matches all nine decoder state
entries exactly to the saved v5 initial tagged decoder, with no teaching or
generation. Experiment seed 733 therefore preserves decoder seed 735.
