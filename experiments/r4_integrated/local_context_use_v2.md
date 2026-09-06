# Direct-context v2: random versus symmetric hidden feedback

Status: prepared 2026-09-06; full comparison unrun.

V1's correct-context decoder reaches 4/12 trained and 0/4 withheld combinations;
all controls score zero. Eight of sixteen correct-arm outputs stop after one
byte. Saved teacher losses improve most for EOS, and lexical transitions show
previous-symbol correlations. Those observations do not establish a causal bug
or authorize a schedule/rate sweep. This one comparison asks whether an
instantaneous output-weight-aligned neuronal teaching signal improves the same
decoder's contextual byte learning. It changes no exposure or input representation.

## One declared rule change

For each active teacher byte, form the existing `delta = p - onehot(target)` from
the saved step logits. The random condition retains the existing fixed hidden
feedback projection. The symmetric condition uses `delta @ output.weight`, with
the current output weights **before** that byte's cell/readout updates. This
neuron-level signal multiplies the same local self-path eligibility; signs,
cell gains, rate and readout delta updates remain unchanged. The context feedback
projection is unchanged and unused by this fixed-context experiment.

No reverse pass, BPTT, retained autograd history, optimizer, feedback fitting or
externally computed per-synapse updates enters either condition. Weight transport
is explicitly required by the symmetric condition; this is not proof of
biological plausibility or full recurrent gradients. The existing hidden-feedback
buffer remains allocated, serialized and counted even when inactive.

The decoder's persistent `hidden_feedback_mode` metadata records random or
symmetric and exact checkpoint validation rejects mismatched modes. The local
learning schema is explicitly versioned. Initial **tensor values** must match
across all eight models and the authenticated v5 starting decoder, including
both fixed feedback buffers. Configuration metadata intentionally differs by
feedback mode and from the preceding local-learning schema.

## Frozen workload and exposed development probes

All numeric inputs, labels, order and output rules from [v1](local_context_use_v1.md)
remain fixed: experiment seed 733, decoder seed 735, shuffle seed 734; CPU
float32; context96/embedding32/hidden64; rate0.01; 384 lessons per arm (32 cycles
of twelve lexical-order TRAIN pairs); two output bytes plus an EOS decision.
The context has two authored four-neuron factor populations and 88 zeros.
`ab,bc,cd,da` remain excluded from training, including shuffled contexts, but
are **already exposed development probes from v1**, not a fresh assessment.

Eight arms execute in this exact order at every roster index:

| Arm | Feedback | Training/final context | Teaching |
| --- | --- | --- | --- |
| random_correct | Fixed random | Correct / correct | Enabled |
| random_zero | Fixed random | Zero / zero | Enabled |
| random_shuffled | Fixed random | Shuffled TRAIN / correct | Enabled |
| random_frozen | Fixed random | Correct / correct | Disabled |
| symmetric_correct | Preupdate readout weights | Correct / correct | Enabled |
| symmetric_zero | Preupdate readout weights | Zero / zero | Enabled |
| symmetric_shuffled | Preupdate readout weights | Shuffled TRAIN / correct | Enabled |
| symmetric_frozen | Preupdate readout weights | Correct / correct | Disabled |

Both feedback conditions receive the exact same saved v1 context schedule.
Serialize it before model construction, first attempts and corrections. Each
lesson makes one read-only first attempt, then offers the two teacher bytes and
EOS with the same previously assigned context. Frozen arms receive all labels.
Final probes run once on twelve trained and four withheld pairs per arm, without
teaching or output repair. No additional epochs, inference probes or adaptive
stopping are allowed. The repeated random baseline is explicitly counted.

Totals: 3,072 first attempts, 128 final generations, 9,216 byte/EOS label offers
and 6,912 enabled local teacher steps. Report actual changed steps, generation
decisions, all first attempts and per-byte losses, first/last128 summaries, final
raw bytes/EOS, complete snapshots and allocated/saved tensor costs.

## Decision, limits and custody

Positive requires symmetric_correct 12/12 trained plus 4/4 withheld exact outputs
with EOS, and a strict withheld-score advantage over symmetric_zero,
symmetric_shuffled, symmetric_frozen **and random_correct**. Report every score
and failure distribution even if this criterion fails. Any fit gain is scoped
to supplied informative rates on exposed development probes.

Allow one full comparison of at most 120 seconds and 2,048 MiB through the
existing monitored launcher, including all eight arms, imports, hashes,
snapshots and logging. Performance-counter durations and Unix deadline guards
remain separate. Failure or exhaustion closes v2 without an automatic extra
variant, seed, rate, threshold or schedule change. Return a useful decoder to
raw-document, source-absent actual-action work; this test supplies no reading,
memory, software or gate claim.

V1 source/recipe/receipts remain unchanged. The new runner imports its exact
numeric context/schedule/lesson/generation/hash helpers; the inherited helper
CONFIG is safe because all those numerical contracts are unchanged. V2 owns the
eight-arm factory, orchestration, identity and decision. No monkeypatch of frozen
globals or generic experiment interface is used. Its manifest includes the full
v1 closure and v2 runner/recipe. Preserve all v1 artifact schemas plus explicit
feedback mode/condition on lessons and final probes. Initial tensor equality
allows only the declared mode metadata difference. Source/revision end checks,
read-only hashes, frozen snapshots and monitored ownership remain required.

Before interpretation, an independent saved-only audit must reconstruct counts,
context permutations, first-before-teacher state chains and scoring; compare
all random-arm outputs, teacher losses and tensor trajectories available in
saved endpoints with authenticated v1 evidence. Metadata differences are explicit.

```powershell
$contextOutput = '.pytest-tmp/local-context-use-v2-run'
$contextRevision = (git rev-parse HEAD).Trim()
$contextSources = (& ./.venv/Scripts/python.exe -c 'from experiments.r4_integrated.local_context_use_v2 import digest,sources; print(digest(sources()))').Trim()
./experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1 -Seconds 120 -MaxMiB 2048 -LogBase $contextOutput -ChildArguments @('-X','utf8','-m','experiments.r4_integrated.local_context_use_v2','--output',$contextOutput,'--revision',$contextRevision,'--sources-sha',$contextSources)
```

Commit reviewed source before freezing that identity and launching. Hash-only
preparation constructs no model. Focused disposable interface QA is separate
from the single full comparison and must retain its own receipt/exposure count.

## Preparation QA

Doctor passes at revision `686ec7c` with receipt
`.pytest-tmp/dev/doctor-5o4j3xd1/receipt.json`. The affected existing decoder suite
passes **73 tests** in 5.53 seconds, receipt
`.pytest-tmp/local-context-use-v2-qa2.xml`; Ruff, syntax compilation and diff checks
pass. The first receipt `.pytest-tmp/local-context-use-v2-qa.xml` preserves
72 passes and one old metadata assertion failure, corrected to the explicitly
declared new schema before repeating QA. No model arithmetic changed to pass it.

Six added test instances cover both cell kinds' independent scalar update oracle,
float32/64 preupdate symmetric signal and read-only forward parity, matching-mode
serialization and mismatch rejection, and the exact eight-arm comparison setup.
Across the two suite invocations, those new instances offer 14 disposable teacher
labels and make 12 read-only generation calls; they do not run diagnostic lessons
or final cohorts. Existing regression fixtures also repeat, as recorded by the
complete receipts. The full eight-arm comparison remains unrun.
