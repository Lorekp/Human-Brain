# Fixed acquisition implementation QA

Four authored checks pass across two separately monitored runs, 2026-09-10. The
[selected comparison](integrated-local-learning-acquisition-v1.md) remains draft;
these are implementation checks, not acquisition, retention or recovery results.
No neuronal learning rule or forward architecture changed in this implementation.

## Verified observer and checkpoint behavior

The [QA bank](../../benchmarks/integrated_local_learning_acquisition_v1.qa.json)
binds eight exact source fingerprints, doctor, monitor, logs and the full retained
fixture inventory. The selected existing test file contains these three nodes:

- `test_integrated_local_learning_observers_preserve_contacts_and_fixed_anchors`:
  authored selected/unselected emitter receipts, disabled publication, conditional
  mixtures, fixed sparse-activity scoring, changed sampled activity, and private
  snapshot corruption refusal. Independent arithmetic checks p from 1/258 to
  3/260 and target-versus-SPACE margin from -log(2) to log(3/2) at the old activity,
  while the new activity remains at 1/258. Teacher/free first-position differences
  remain distinct from consequences after the first wrong emitted symbol.
- `test_integrated_local_learning_compact_teacher_refuses_lost_contacts_and_false_counts`:
  missing selected probabilities, false publication gates, probability/log-p
  inconsistency, fabricated unselected scores and incomplete/false denominators
  are rejected.
- `test_integrated_local_learning_checkpoint_authored_full_block_roundtrip`:
  real final payload publication and strict independent restoration, complete
  journal authentication, unchanged numerical tensors/RNG/reference history,
  distinct restored storage and rejection of an appended journal tail.

The round trip constructs **four authored CPU actors** with empty 2,048-slot
tapes and decodes **two own fixture payloads**. The fixture's 913/1430 counters,
3,072 attempt rows, 3,072 correction rows and seven panels are explicitly
synthetic validator inputs. They are not executed experiences. It performs zero
source offers, free responses, teacher calls, task executions or retained-parent
loads. Only fixture parent/tape identities are substituted; the built-in complete
ledger validator, codec, readback, source checks and restore integrity remain in
the exercised path. Its expected post-check journal tail is retained as a dirty
fixture, not a resumable learner checkpoint.

The monitored three-node run passes in **17.4214144 seconds**, with **1,414.78515625
MiB** sampled process-tree peak; pytest reports three passes in 16.51 seconds.
Its artifacts total **252,963,861 bytes**, below the declared 512-MiB fixture cap.
The 240-second/2,048-MiB outer bounds hold. All eight held source hashes match
afterward. `git diff --check` passes for the reviewed changes.

An earlier two-node authored observer check passed in 5.0092996 seconds at
280.47265625 MiB. The alignment observer was subsequently narrowed to retain
states only through the first divergent output, with omitted-tail counts, and
an actual teacher-draw counter was added. The current observer was then tested
in the three-node run above. The earlier receipt remains separate cost/history;
it has no separate pre-change source fingerprint and does not qualify current
bytes. Pytest's cache-directory warning did not prevent any selected check.

## Saved scientific reader and preparation

The separately monitored
`test_integrated_local_learning_saved_reader_stream_limits_and_interference`
passes in **5.4671022 seconds**, at **280.8984375 MiB** sampled process-tree peak;
pytest reports one pass in 4.49 seconds. The
[reader QA bank](../../benchmarks/integrated_local_learning_acquisition_v1.reader_qa.json)
binds the doctor, four exact source hashes, output and retained fixtures. It
checks authenticated streaming, count/tail/truncation refusal, the valid 129th
decision at a 128-byte generation limit, emitter contribution identity,
zero-contact EOS aggregation, and fixed-anchor interference. An isolated import
check confirms the saved reader imports neither `humanbrain` nor `torch`.
This check makes zero model, teacher, task-execution or tensor-decoding calls.

Independent source review caught two issues before this passing test: a valid
limit receipt was rejected, and equal-sum mixture components were not individually
bound to their actual emitters. Both are corrected. Root and independent review
confirm the complete-acquisition denominator, actual-input/RNG comparison limits
and fixed-anchor interpretation. Probability gains, correct spans and changed
proposals remain distinct from complete behavior or productive recovery.

Source-only preparation takes **0.3860045 seconds** under its 60-second/256-MiB
bound and creates a 66,509-byte manifest with SHA256
`de772f29ff34e5db1401b1a708f4e5349c4f850c0dad63d617349a87e421f293`.
The monitor did not observe the short process between samples; its zero field is
not evidence of zero memory. Preparation makes no model calls. Its receipts are
in the reader QA bank.

These checks do not execute the complete comparison runner. Both execution
guards remain draft. Reproducible source commitment and explicit release are
still pending; no new acquisition exposure or continuation checkpoint exists.

Final root lint review found five long lines in the observer. Line wrapping
alone corrects them; its before/after AST SHA256 is unchanged at
`4c8b6fc7c2bd603005ad75e1462b9423ade511d28a15835ee02eb50b4c73e3fb`.
The final observer source SHA256 is
`be68e380c3ef8dbd352987ac008f71b270725057565feeb368678985c520e8fc`;
the executed QA bank retains the actual pre-formatting source hash. All four
new experiment modules pass Ruff. Six new test line-length findings were also
wrapped without changing the whole-file AST, SHA256
`bcbc08cdd68b195b985069d0fdf2ca792d69be9e4a877016c1a3901d59d88db0`.
Final test-file SHA256 is
`bf20193c0c5dd18c9cecd206b6493d25fe40306097ad17df0f7b9dc94809ec74`.
The earlier Ruff checks used `tools/ruff.toml`, which excludes E501; that
configuration still passes. The unqualified shared-test check now has the same
57 findings as HEAD, with none in these four new nodes. Historical findings are
unchanged; do not infer a repository-wide lint pass. No test replay was needed
for these verified AST-equivalent formatting changes.

## Byte-preserving storage recovery

Available space fell from approximately 9.14 GB to 5.42 GB during implementation,
below the new comparison's 6-GiB floor. Root kept that floor and selected bounded
transparent NTFS compression of 32 completed, bank-listed historical checkpoint
payloads. No directory-wide compression, deletion, model load or numerical
conversion occurred. Each selected file's logical length and SHA256 matched the
existing completed bank before compression and again afterward.

The [storage bank](../../benchmarks/integrated_local_learning_v1.storage.json)
records **3,702,796,288 logical bytes**, with measured stored allocation reduced
from 3,702,796,288 to **196,329,472 bytes**: **3,506,466,816 bytes recovered**.
All 32 contents and paths remain unchanged. The operation took 13.8428197 seconds
in its outer monitor, peaking at 50.63671875 MiB. It used zero tensor decodes,
model calls or deletions. Per-file receipts, exact script and original bank
identities remain saved. System free space afterward was 8,683,339,776 bytes;
other activity can change free space independently of this allocation saving.

This is measured storage improvement, not a simulation or restore latency
speedup. Future source, payload and journal authentication remain unchanged.
Recheck actual available capacity before the long run; completed compression
does not reserve disk against other programs.
